"""solvi init / ask / calibrate / models, and part.save_calibration / load_calibration. Nothing is downloaded: models are
stand-ins (a keyword scorer, a local System One server, a fake Hugging Face cache); one test runs a real cached decider
when it and onnxruntime are there."""
import http.server
import io
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from solvi import calibfile, models
from solvi.cli import load_object, main
from solvi.decide import Facts
from solvi.scaffold import TEMPLATES

sys.path.insert(0, str(Path(__file__).parent))
from test_decide import TASK, TEAMS, model, texts  # noqa: E402


def run(capsys, *argv):
    """main(argv) → (exit status, stdout); a usage error (SystemExit 2) is returned as 2."""
    capsys.readouterr()
    try:
        code = main([str(a) for a in argv])
    except SystemExit as e:
        code = e.code
    return code, capsys.readouterr().out


def scaffold(tmp_path, template="support", with_model=False):
    d = tmp_path / f"{template}{'_m' if with_model else ''}"
    assert main(["init", str(d), "--template", template] + (["--with-model"] if with_model else [])) == 0
    return d


# --------------------------------------------------------------------------------------------------- init
@pytest.mark.parametrize("with_model", [False, True])
@pytest.mark.parametrize("template", TEMPLATES)
def test_init_scaffold_passes_test_and_check(tmp_path, capsys, template, with_model):
    d = scaffold(tmp_path, template, with_model)
    names = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
    assert {"catalog.py", "cases.json", "example.json", "README.md", ".gitignore",
            ".github/workflows/solvi.yml"} <= names
    assert ("labels.csv" in names) == with_model
    wf = (d / ".github/workflows/solvi.yml").read_text()
    assert "solvi test ." in wf and "solvi check catalog.py:system" in wf
    assert json.loads((d / "cases.json").read_text())["task"] == "catalog.py"
    capsys.readouterr()
    assert main(["test", str(d)]) == 0
    assert "cases passed" in capsys.readouterr().out
    assert main(["check", f"{d}/catalog.py:system", "--strict"]) == 0
    capsys.readouterr()
    code, out = run(capsys, "ask", f"{d}/catalog.py:system", d / "example.json", "--json")
    assert code == 0 and json.loads(out)["answers"]


def test_init_does_not_overwrite(tmp_path, capsys):
    d = scaffold(tmp_path, "minimal")
    (d / "catalog.py").write_text("# mine\n")
    capsys.readouterr()
    assert main(["init", str(d), "--template", "minimal"]) == 1
    assert (d / "catalog.py").read_text() == "# mine\n" and "already has catalog.py" in capsys.readouterr().err
    assert main(["init", str(d), "--template", "minimal", "--force"]) == 0
    assert "Catalog()" in (d / "catalog.py").read_text()
    code, _ = run(capsys, "init", d, "--template", "nope")
    assert code == 2


def test_init_into_a_path_that_is_a_file_says_so_and_writes_nothing(tmp_path, capsys):
    f = tmp_path / "notes.txt"
    f.write_text("mine\n")
    capsys.readouterr()
    assert main(["init", str(f)]) == 1                                   # not a TypeError
    err = capsys.readouterr().err
    assert "is a file, not a folder" in err and "nothing written" in err and f.read_text() == "mine\n"
    assert main(["init", str(f), "--force"]) == 1 and f.read_text() == "mine\n"
    with pytest.raises(NotADirectoryError):
        from solvi.scaffold import init
        init(f)


def test_init_workflow_points_to_the_project_in_a_repository(tmp_path):
    (tmp_path / ".git").mkdir()
    d = tmp_path / "services" / "triage"
    assert main(["init", str(d)]) == 0
    assert "working-directory: services/triage" in (d / ".github/workflows/solvi.yml").read_text()


# --------------------------------------------------------------------------------------------------- ask
def test_ask_state_inputs_outputs_and_exit_status(tmp_path, capsys, monkeypatch):
    d = scaffold(tmp_path, "support")
    sysspec = f"{d}/catalog.py:system"
    code, out = run(capsys, "ask", sysspec, d / "example.json")
    assert code == 0 and "priority" in out and "billing" in out
    legal = {"message": "My lawyer will see you in court.", "tier": "standard", "hours_open": 1}
    code, out = run(capsys, "ask", sysspec, "--state", json.dumps(legal), "--json")
    data = json.loads(out)
    assert code == 0 and data["answers"]["priority"] == {**data["answers"]["priority"], "answer": "urgent",
                                                         "status": "forced", "guard": "hard_check"}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(legal)))
    code, out = run(capsys, "ask", sysspec, "-", "--question", "route", "--json")
    assert code == 0 and list(json.loads(out)["answers"]) == ["route"]
    code, out = run(capsys, "ask", sysspec, d / "example.json", "--audit", "--json")
    assert "audit" in json.loads(out)
    code, out = run(capsys, "ask", sysspec, d / "example.json", "--audit", "--lang", "ru")
    assert code == 0 and "ответы" in out and "уверенность" in out
    code, out = run(capsys, "ask", sysspec, d / "example.json", "--report", "md")
    assert out.startswith("# Decision report")
    code, out = run(capsys, "ask", sysspec, d / "example.json", "--report", "html")
    assert out.lstrip().lower().startswith("<!doctype html")
    db = tmp_path / "decisions.db"
    code, out = run(capsys, "ask", sysspec, d / "example.json", "--store", db, "--json")
    assert json.loads(out)["stored_id"] and main(["verify", str(db)]) == 0
    # usage errors: no input, two inputs, an unknown question, not JSON, not an object
    for argv in ([], [d / "example.json", "--state", "{}"], [d / "example.json", "--question", "nope"],
                 ["--state", "{not json"], ["--state", "[1]"], [tmp_path / "missing.json"],
                 ["--state", "{}", "--lang", "xx"]):
        assert run(capsys, "ask", sysspec, *argv)[0] == 2, argv


def test_ask_abstention_exits_1(tmp_path, capsys):
    d = scaffold(tmp_path, "support", with_model=True)
    st = {"message": "Hello there.", "tier": "standard", "hours_open": 1}   # no keyword: the stand-in is unsure
    code, out = run(capsys, "ask", f"{d}/catalog.py:system", "--state", json.dumps(st), "--json")
    assert code == 1 and json.loads(out)["answers"]["route"]["status"] == "abstain"


TEXT_TASK = '''
from solvi import Answer, Catalog, Question, System
from solvi.decide import DecideModel
import numpy as np

cat = Catalog()


@cat.rule("refund")
def refund(amount: float) -> str:
    return "yes" if amount <= 100 else "no"


@cat.rule("cancel")
def cancel(order_id: str) -> str:
    return "yes"


one = System(cat, [Question("refund", "Refund the payment?", Answer.yes_no())])
two = System(cat, [Question("refund", "Refund the payment?", Answer.yes_no()),
                   Question("cancel", "Cancel the order?", Answer.yes_no())])


class Router:
    model_id = "test/router"

    def fingerprint(self):
        return "router-1"

    def logits(self, items):
        out = []
        for it in items:
            z = np.array([4.0 * (o in it.text.lower()) for o in it.options])
            out.append(np.stack([z, z], 1))
        return out


router = DecideModel(Router(), meta={"format": "test", "temperature": 1.0})


def prepare(state):
    state["prepared"] = True
    return state
'''


def test_ask_text(tmp_path, capsys, monkeypatch):
    f = tmp_path / "texttask.py"
    f.write_text(TEXT_TASK)
    code, out = run(capsys, "ask", f"{f}:one", "--text", "Please refund the 80 I paid.", "--json")
    data = json.loads(out)
    assert code == 0 and data["answers"]["refund"]["answer"] == "yes"
    assert data["textin"]["fields"]["amount"]["value"] == 80
    monkeypatch.setattr(sys, "stdin", io.StringIO("refund 250 please"))
    code, out = run(capsys, "ask", f"{f}:one", "--text", "-", "--json")
    assert json.loads(out)["answers"]["refund"]["answer"] == "no"
    # two entry points: a decider routes (module:attr), or --question skips routing
    code, out = run(capsys, "ask", f"{f}:two", "--text", "please refund 80", "--decider", f"{f}:router", "--json")
    assert code == 0 and json.loads(out)["textin"]["question"] == "refund"
    code, out = run(capsys, "ask", f"{f}:two", "--text", "80", "--question", "refund", "--json")
    assert json.loads(out)["textin"]["question"] == "refund"
    # a Hugging Face id that is not downloaded is never fetched
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "empty-cache"))
    assert run(capsys, "ask", f"{f}:two", "--text", "x", "--decider", "solvi-ai/not-there")[0] == 2
    assert run(capsys, "ask", f"{f}:one", "--state", "{}", "--decider", f"{f}:router")[0] == 2


# --------------------------------------------------------------------------------------------------- calibration files
def _examples():
    return [(t, k) for k in TEAMS for t in texts(k, 40)]


def test_save_and_load_calibration_restores_the_fingerprint(tmp_path):
    m = model(noise=3.0)
    part = m.decision("team", TASK, "email", TEAMS)
    info = part.act_guard(_examples(), risk=0.10)
    part.conformal(_examples(), coverage=0.9)
    f = part.save_calibration(tmp_path / "team.calib.json")
    rec = json.loads(Path(f).read_text())
    assert rec["format"] == calibfile.FORMAT and rec["escalate_below"] == info["threshold"]
    fresh = m.decision("team", TASK, "email", TEAMS)
    assert fresh.fingerprint() != part.fingerprint()
    assert fresh.load_calibration(f) is fresh
    assert fresh.fingerprint() == part.fingerprint() and fresh.guarantee == part.guarantee
    assert fresh.conformal_set == part.conformal_set
    d1, d2 = part.decide(texts("billing", 1)[0]), fresh.decide(texts("billing", 1)[0])
    assert (d1.value, d1.escalate, d1.extra.get("guarantee")) == (d2.value, d2.escalate, d2.extra.get("guarantee"))


def test_load_calibration_refuses_another_model_or_question(tmp_path):
    part = model().decision("team", TASK, "email", TEAMS)
    part.act_guard(_examples(), risk=0.10)
    f = part.save_calibration(tmp_path / "c.json")
    other = model(version="2").decision("team", TASK, "email", TEAMS)
    with pytest.raises(ValueError, match="another model"):
        other.load_calibration(f)
    with pytest.raises(ValueError, match="another question"):
        model().decision("team", "Which queue?", "email", TEAMS).load_calibration(f)
    adapted = model()
    p2 = adapted.decision("team", TASK, "email", TEAMS)
    p2.fit(_examples()[:30])
    with pytest.raises(ValueError, match="adaptation"):
        p2.load_calibration(f)
    other.load_calibration(f, strict=False)            # explicitly accepted
    assert other.guarantee == part.guarantee
    (tmp_path / "bad.json").write_text('{"format": "x"}')
    with pytest.raises(ValueError, match="not a solvi calibration"):
        other.load_calibration(tmp_path / "bad.json")


def test_a_calibration_file_does_not_load_onto_a_part_computing_another_signal(tmp_path):
    m = model(noise=3.0)
    for made, other in ((dict(option_order="average"), dict(option_order="given")),
                        (dict(option_order="average", permutations=3), dict(option_order="average", permutations=2)),
                        (dict(long="retrieve"), {}),
                        (dict(long="retrieve", top_k=2), dict(long="retrieve", top_k=3)),
                        (dict(long="retrieve", rerank=True), dict(long="retrieve"))):
        part = m.decision("team", TASK, "email", TEAMS, **made)
        part.act_guard(_examples(), risk=0.2)
        f = part.save_calibration(tmp_path / "s.json")
        assert m.decision("team", TASK, "email", TEAMS, **made).load_calibration(f).fingerprint() == part.fingerprint()
        with pytest.raises(ValueError, match="computes its signal"):
            m.decision("team", TASK, "email", TEAMS, **other).load_calibration(f)


def test_loading_a_calibration_keeps_both_thresholds_as_saved(tmp_path):
    m = model(noise=3.0)
    part = m.decision("team", TASK, "email", TEAMS)
    part.act_guard(_examples(), risk=0.2)
    part.act_threshold = 0.7                             # set by hand after calibrating: saved and restored as it is
    f = part.save_calibration(tmp_path / "t.json")
    fresh = m.decision("team", TASK, "email", TEAMS).load_calibration(f)
    assert (fresh.escalate_below, fresh.act_threshold) == (part.escalate_below, 0.7)
    assert fresh.fingerprint() == part.fingerprint()


def test_calibration_with_groups_and_everything_escalated(tmp_path):
    m = model(noise=3.0)
    part = m.decision("team", TASK, "email", TEAMS)
    ex = [(Facts(email=t, domain="a" if i % 3 else "b"), k) for i, (t, k) in enumerate(_examples())]
    part.act_guard(ex, risk=0.2, groups="domain", min_group=30)
    f = part.save_calibration(tmp_path / "g.json")
    fresh = m.decision("team", TASK, "email", TEAMS).load_calibration(f)
    assert fresh.fingerprint() == part.fingerprint()
    assert list(fresh.__signature__.parameters) == ["email", "domain"]
    # grouping by a function: pass it again; its code must match
    def dom(email):
        return "long" if len(email) > 45 else "short"
    part.act_guard(ex, risk=0.2, groups=dom, min_group=30)
    part.save_calibration(f)
    with pytest.raises(ValueError, match="pass it again"):
        m.decision("team", TASK, "email", TEAMS).load_calibration(f)
    assert m.decision("team", TASK, "email", TEAMS).load_calibration(f, groups=dom).fingerprint() == part.fingerprint()
    # nothing can be answered alone: the threshold is inf and it round-trips
    part.act_guard(_examples()[:3], risk=0.01)
    part.save_calibration(f)
    assert m.decision("team", TASK, "email", TEAMS).load_calibration(f).escalate_below == float("inf")


def test_combination_calibration_round_trip(tmp_path):
    from solvi.multi import Cascade
    a, b = model(noise=3.0), model(version="b")

    def cascade():
        return Cascade([a.decision("team", TASK, "email", TEAMS), b.decision("team", TASK, "email", TEAMS)])
    c = cascade()
    c.act_guard(_examples(), risk=0.1)
    f = c.save_calibration(tmp_path / "k.json")
    c2 = cascade().load_calibration(f)
    assert c2.fingerprint() == c.fingerprint() and c2.threshold == c.threshold
    with pytest.raises(ValueError):
        Cascade([b.decision("team", TASK, "email", TEAMS), a.decision("team", TASK, "email", TEAMS)]).load_calibration(f)
    with pytest.raises(ValueError, match="DecisionPart"):
        a.decision("team", TASK, "email", TEAMS).load_calibration(f)


# --------------------------------------------------------------------------------------------------- solvi calibrate
REAL_MODEL = '''import numpy as np
from solvi.decide import DecideModel


class Real:
    """What SOLVI_DECIDE_MODEL names in a scaffolded project: not the keyword stand-in."""
    model_id = "test/real"

    def fingerprint(self):
        return "real"

    def logits(self, items):
        return [np.stack([np.arange(len(it.options), 0, -1.0)] * 2, 1) for it in items]


model = DecideModel(Real(), meta={"format": "test", "temperature": 1.0})
'''


@pytest.mark.parametrize("template", TEMPLATES)
def test_init_project_calibrated_with_a_real_model_still_passes_its_ci_with_the_stand_in(tmp_path, capsys, monkeypatch,
                                                                                         template):
    from solvi.scaffold import SPECS
    monkeypatch.chdir(tmp_path)
    d = scaffold(tmp_path, template, with_model=True)
    q = SPECS[template]["model_question"]
    (d / "real.py").write_text(REAL_MODEL)
    monkeypatch.setenv("SOLVI_DECIDE_MODEL", f"{d}/real.py:model")       # the README: export ..., then calibrate
    code, _ = run(capsys, "calibrate", f"{d}/catalog.py:system", q, d / "labels.csv", "--risk", "0.5",
                  "--out", d / f"{q}.calib.json")
    assert (d / f"{q}.calib.json").exists(), code
    assert "test/real" in json.loads((d / f"{q}.calib.json").read_text())["models"]
    monkeypatch.delenv("SOLVI_DECIDE_MODEL")                             # CI, a new shell: the stand-in
    capsys.readouterr()
    assert main(["test", str(d)]) == 0
    io = capsys.readouterr()
    assert "cases passed" in io.out and f"{q}.calib.json" in io.err and "stand-in runs without it" in io.err
    assert main(["check", f"{d}/catalog.py:system"]) == 0
    code, out = run(capsys, "ask", f"{d}/catalog.py:system", d / "example.json", "--json")
    assert code == 0 and json.loads(out)["answers"]
    part = calibfile.find_part(load_object(f"{d}/catalog.py:system"), q)
    assert part.guarantee is None                                        # the real model's thresholds were not applied
    assert "stand-in" in (d / "README.md").read_text() and f"{q}.calib.json" in (d / "README.md").read_text()


def test_calibrate_command_writes_a_file_the_catalog_loads(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    d = scaffold(tmp_path, "support", with_model=True)
    out = d / "route.calib.json"
    code, text = run(capsys, "calibrate", f"{d}/catalog.py:system", "route", d / "labels.csv", "--risk", "0.1",
                     "--conformal", "0.9", "--out", out)
    assert code == 0 and "answered alone" in text and "must escalate at least" in text and out.exists()
    system = load_object(f"{d}/catalog.py:system")
    part = calibfile.find_part(system, "route")
    assert part.guarantee["method"] == "crc" and part.conformal_set is not None
    assert main(["test", str(d)]) == 0
    # a calibration made with another model stops the catalog from loading under a real model — but not `solvi calibrate`
    rec = json.loads(out.read_text())
    rec["fingerprint"], rec["models"] = "0" * 16, {"another/model": "1" * 16}
    out.write_text(json.dumps(rec))
    (d / "real.py").write_text(REAL_MODEL)
    monkeypatch.setenv("SOLVI_DECIDE_MODEL", f"{d}/real.py:model")
    capsys.readouterr()
    assert main(["test", str(d)]) == 1 and "calibrated on another model" in capsys.readouterr().out
    code, text = run(capsys, "calibrate", f"{d}/catalog.py:system", "route", d / "labels.csv", "--json")
    data = json.loads(text)
    assert code in (0, 1) and data["saved"] == "route.calib.json" and data["not_applied"]   # 1: it escalates everything
    Path("route.calib.json").replace(out)             # written in the working directory (the default name)
    assert main(["test", str(d)]) == 0


def test_calibrate_command_groups_ltt_and_usage_errors(tmp_path, capsys):
    d = scaffold(tmp_path, "support", with_model=True)
    rows = [{"text": t, "domain": "a" if i % 2 else "b", "label": k}
            for i, (t, k) in enumerate(_labels(d))] * 5
    jl = tmp_path / "labels.jsonl"
    jl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "g.json"
    code, text = run(capsys, "calibrate", f"{d}/catalog.py:system", "route", jl, "--groups", "domain", "--min-group",
                     "20", "--risk", "0.2", "--out", out)
    assert code in (0, 1) and "per group" in text
    assert json.loads(out.read_text())["groups"]["by"] == ["domain"]
    code, text = run(capsys, "calibrate", f"{d}/catalog.py:system", "route", jl, "--method", "ltt", "--risk", "0.2",
                     "--out", out, "--json")
    assert code in (0, 1) and "coverage" in json.loads(text)
    sysspec = f"{d}/catalog.py:system"
    for argv in (["priority", jl], ["nope", jl], ["route", tmp_path / "missing.csv"],
                 ["route", jl, "--method", "ltt", "--groups", "domain"]):
        assert run(capsys, "calibrate", sysspec, *argv, "--out", out)[0] == 2, argv
    (tmp_path / "nolabel.jsonl").write_text('{"text": "x"}\n')
    assert run(capsys, "calibrate", sysspec, "route", tmp_path / "nolabel.jsonl", "--out", out)[0] == 2


def _labels(d):
    import csv
    with open(d / "labels.csv") as fh:
        return [(r["text"], r["label"]) for r in csv.DictReader(fh)]


# --------------------------------------------------------------------------------------------------- solvi models
def _fake_cache(root):
    for repo, meta in (("solvi-ai/solvi-base", {"format": "solvi_decide v2", "modes": ["single", "multi"],
                                                "act_head": True}),
                       ("someone/my-decider", {"format": "l14b_decider v1"}), ("someone/not-a-decider", None)):
        d = root / ("models--" + repo.replace("/", "--"))
        snap = d / "snapshots" / "abc123"
        snap.mkdir(parents=True)
        (d / "refs").mkdir()
        (d / "refs" / "main").write_text("abc123")
        (snap / "config.json").write_text("{}")
        if meta is not None:
            (snap / "solvi_decide.json").write_text(json.dumps(meta))
    return root


@pytest.fixture
def no_hub(monkeypatch):
    """huggingface_hub replaced by a module that records calls (and fails the test if a download starts)."""
    calls = []
    fake = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, **kw):
        calls.append((repo_id, kw))
        return f"/cache/{repo_id}"
    fake.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake)
    return calls


def test_models_list_reads_the_cache(tmp_path, capsys, monkeypatch, no_hub):
    monkeypatch.setenv("HF_HUB_CACHE", str(_fake_cache(tmp_path / "hub")))
    code, out = run(capsys, "models", "list", "--json")
    data = json.loads(out)
    got = {m["id"]: m for m in data["models"]}
    assert code == 0 and got["solvi-ai/solvi-base"]["cached"] and not got["solvi-ai/solvi-large"]["cached"]
    assert "someone/my-decider" in got and "someone/not-a-decider" not in got
    code, out = run(capsys, "models")
    assert code == 0 and "solvi models pull solvi-ai/solvi-large" in out
    assert no_hub == []


def test_models_pull_only_when_asked(tmp_path, capsys, monkeypatch, no_hub):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    assert run(capsys, "models", "check", "solvi-ai/solvi-large")[0] == 2      # not downloaded: no download either
    assert no_hub == []
    code, out = run(capsys, "models", "pull", "solvi-ai/solvi-base")
    assert code == 0 and out.strip() == "/cache/solvi-ai/solvi-base"
    assert no_hub == [("solvi-ai/solvi-base", {"revision": None, "allow_patterns": ["*.json", "*.md", "onnx/*"]})]
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)                  # not installed
    assert run(capsys, "models", "pull", "solvi-ai/solvi-base")[0] == 2


def test_models_check_a_cached_folder_prints_the_declaration(tmp_path, capsys, monkeypatch, no_hub):
    monkeypatch.setenv("HF_HUB_CACHE", str(_fake_cache(tmp_path / "hub")))
    code, out = run(capsys, "models", "check", "solvi-ai/solvi-base", "--json")
    data = json.loads(out)
    assert code == 1 and "cannot load" in data["error"]                        # no weights in the stand-in folder
    assert data["declared"]["modes"] == ["single", "multi"] and data["declared"]["act"] is not None
    assert run(capsys, "models", "check", tmp_path / "hub")[0] == 2            # a folder without solvi_decide.json
    assert run(capsys, "models", "check", "not an id")[0] == 2
    assert run(capsys, "models", "check", "systemone:http://x")[0] == 2        # no #model
    assert no_hub == []


MODEL_FILE = '''
import sys
sys.path.insert(0, {tests!r})
from test_decide import model
decider = model()
'''


def test_models_check_measures_accuracy_and_latency(tmp_path, capsys):
    f = tmp_path / "mymodel.py"
    f.write_text(MODEL_FILE.format(tests=str(Path(__file__).parent)))
    ex = tmp_path / "ex.jsonl"
    ex.write_text("".join(json.dumps({"text": t, "label": k}) + "\n" for k in TEAMS for t in texts(k, 5)))
    code, out = run(capsys, "models", "check", f"{f}:decider", "--examples", ex, "--task", TASK, "--json")
    data = json.loads(out)
    assert code == 0 and data["examples"]["n"] == 15 and data["examples"]["accuracy"] == 1.0
    assert data["fingerprint"] and data["examples"]["ms_p50"] >= 0 and data["examples"]["options"] == sorted(TEAMS)
    code, out = run(capsys, "models", "check", f"{f}:decider", "--examples", ex, "--task", TASK,
                    "--options", "billing,technical,shipping", "--min-accuracy", "1.01")
    assert code == 1 and "accuracy     100.0%" in out and "below --min-accuracy" in out
    assert run(capsys, "models", "check", f"{f}:decider", "--examples", ex)[0] == 2      # no question


class _SystemOne(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        text = body["state"].lower()
        answers = {}
        for name, q in body["questions"].items():
            raw = {o: 1.0 + 8.0 * any(k in text for k in {"billing": ["charged"], "technical": ["crash"],
                                                          "shipping": ["parcel"]}.get(o, [])) for o in q["criteria"]}
            s = sum(raw.values())
            answers[name] = {"type": "choice", "probabilities": {o: v / s for o, v in raw.items()}}
        data = json.dumps({"answers": answers}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def test_models_check_a_system_one_service(tmp_path, capsys):
    srv = http.server.HTTPServer(("127.0.0.1", 0), _SystemOne)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        ex = tmp_path / "ex.csv"
        ex.write_text("text,label\n" + "".join(f"{t},{k}\n" for k in TEAMS for t in texts(k, 3)).replace(
            "I was charged twice, please refund", "I was charged twice").replace(", the tracking", " the tracking"))
        url = f"systemone:http://127.0.0.1:{srv.server_port}#kev-test"
        code, out = run(capsys, "models", "check", url, "--examples", ex, "--task", TASK, "--json")
        data = json.loads(out)
        assert code == 0 and data["id"] == "systemone:kev-test" and data["examples"]["accuracy"] == 1.0
    finally:
        srv.shutdown()


@pytest.mark.model
def test_models_check_a_real_cached_decider(capsys, tmp_path):
    pytest.importorskip("onnxruntime")
    pytest.importorskip("tokenizers")
    if models.cached_path("solvi-ai/solvi-base") is None:
        pytest.skip("solvi-ai/solvi-base is not downloaded")
    ex = tmp_path / "ex.jsonl"
    ex.write_text("".join(json.dumps({"text": t, "label": k}) + "\n" for k in TEAMS for t in texts(k, 2)))
    code, out = run(capsys, "models", "check", "solvi-ai/solvi-base", "--examples", ex, "--task", TASK, "--json")
    data = json.loads(out)
    assert code == 0 and data["id"] == "solvi-ai/solvi-base" and data["examples"]["n"] == 6


# --- fixes before 0.7
@pytest.mark.parametrize("cmd", [["verify"], ["replay"], ["report"], ["ask", "x:y"]])
def test_store_help_names_every_backend(cmd, capsys):
    try:
        main([*cmd, "-h"])
    except SystemExit:
        pass
    out = " ".join(capsys.readouterr().out.split())
    assert ".duckdb" in out and "postgresql://" in out


def test_a_postgres_url_is_not_a_missing_file(monkeypatch):
    from solvi import cli, storage
    seen = []
    monkeypatch.setattr(storage, "PostgresStorage", lambda url, catalog=None: seen.append(url) or "pg")
    assert cli._store("postgresql://u@h/db") == "pg" and seen == ["postgresql://u@h/db"]


def test_api_reference_has_the_07_modules_and_the_guide_is_precise():
    import re
    root = Path(__file__).resolve().parents[1]
    nav = (root / "mkdocs.yml").read_text()
    index = (root / "docs" / "api" / "index.md").read_text()
    for m in ("textin", "longdoc", "report", "otel", "counterfactual", "perturb"):
        page = root / "docs" / "api" / f"{m}.md"
        assert page.exists() and f"::: solvi.{m}" in page.read_text(), m
        assert f"solvi.{m}: api/{m}.md" in nav and f"]({m}.md)" in index, m
    guide = " ".join((root / "docs" / "guide.md").read_text().split())
    assert not re.search(r'extra\["long"\]`: in the trace record, hashed, printed by the audit \([^)]*\), and re-checked by '
                         r"replay", guide)
    assert "trusted replay (`trust_models=True`" in guide


def test_ask_text_with_an_llm_decider_takes_an_api_key_and_fails_cleanly(tmp_path, capsys):
    f = tmp_path / "texttask.py"
    f.write_text(TEXT_TASK)
    seen = []

    class Refuse(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(self.headers.get("Authorization"))
            self.rfile.read(int(self.headers.get("Content-Length") or 0))
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "bad key"}}')

        def log_message(self, *a):
            pass
    srv = http.server.HTTPServer(("127.0.0.1", 0), Refuse)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        capsys.readouterr()
        with pytest.raises(SystemExit) as e:
            main(["ask", f"{f}:two", "--text", "please refund 80", "--decider",
                  f"llm:http://127.0.0.1:{srv.server_port}/v1#tiny", "--api-key", "sk-test-1"])
    finally:
        srv.shutdown()
    err = capsys.readouterr().err
    assert e.value.code == 2 and seen and seen[0] == "Bearer sk-test-1"
    assert "HTTP 401" in err and "Traceback" not in err and "sk-test-1" not in err


def test_models_load_raises_model_error_not_system_exit_for_a_missing_file_or_attribute(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine.py").write_text("x = 1\n")
    for spec, why in (("nofile.py:model", "no such file"), ("mine.py:model", "has no attribute 'model'")):
        with pytest.raises(models.ModelError, match=why):          # library code: an ordinary exception
            models.load(spec)
    from solvi.loader import LoadError, load_object as load
    with pytest.raises(LoadError, match="expected module:attribute"):
        load("mine.py")
    assert load("mine.py:x") == 1
    with pytest.raises(SystemExit) as e:                               # the command still ends with status 2
        load_object("nofile.py:system")
    assert e.value.code == 2 and "no such file: nofile.py" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["models", "check", "nofile.py:model"])
    assert e.value.code == 2 and "no such file: nofile.py" in capsys.readouterr().err


def test_models_check_names_a_missing_folder_and_takes_a_decider_with_only_decision(tmp_path, capsys, monkeypatch, no_hub):
    monkeypatch.chdir(tmp_path)
    for spec in ("./my-decider", "../nope/model", str(tmp_path / "gone"), "~/no-such-solvi-model"):
        with pytest.raises(models.ModelError, match="no such folder"):
            models.resolve(spec)
    with pytest.raises(models.ModelError, match="is not downloaded: solvi models pull someone/thing"):
        models.resolve("someone/thing")
    (tmp_path / "dec.py").write_text("class OnlyDecision:\n    def decision(self, *a, **k):\n        raise NotImplementedError\n\n\nobj = OnlyDecision()\n")
    code, out = run(capsys, "models", "check", "dec.py:obj", "--json")
    data = json.loads(out)
    assert code == 0 and data["id"] == "OnlyDecision" and data["fingerprint"] is None
    assert "solvi-ai/solvi-large-long" in models.PUBLISHED
    assert "files" not in (models.cached.__doc__ or "")


def _stored_project(tmp_path, capsys):
    d = scaffold(tmp_path, "minimal")
    code, _ = run(capsys, "ask", f"{d}/catalog.py:system", d / "example.json", "--store", d / "decisions.db")
    assert code == 0
    return d, f"{d}/catalog.py:system", str(d / "decisions.db")


def test_usage_errors_exit_with_status_2_and_a_message_not_a_traceback(tmp_path, capsys, monkeypatch):
    d, system, store = _stored_project(tmp_path, capsys)
    monkeypatch.chdir(d)
    (d / "adir").mkdir()
    for argv, said in ((["ask", "no_such_module_zz:system", "example.json"], "no module named"),
                       (["verify", store, "--anchor", "abc"], "--anchor"),
                       (["replay", store, "--system", system, "--since", "garbage"], "--since"),
                       (["diff", store, "--system", system, "--until", "garbage"], "--until"),
                       (["report", store, "--html", "no/such/dir/out.html"], "out.html"),
                       (["verify", store, "--sign", "no/such/dir/sig.json"], "sig.json"),
                       (["verify", "adir"], "adir"),
                       (["serve", system, "--log-level", "bogus"], "--log-level")):
        capsys.readouterr()
        try:
            code = main(argv)
        except SystemExit as e:
            code = e.code
        err = capsys.readouterr().err
        assert code == 2 and said in err and "Traceback" not in err, (argv, code, err)
    (d / "broken.py").write_text("import nothing_like_this_zz\n")       # an error inside the user's module is theirs
    with pytest.raises(ModuleNotFoundError):
        main(["check", "broken.py:system"])


def test_replay_and_diff_refuse_a_mistyped_filter_and_say_how_many_decisions_they_covered(tmp_path, capsys):
    d, system, store = _stored_project(tmp_path, capsys)
    for cmd in ("replay", "diff"):
        code, out = run(capsys, cmd, store, "--system", system)
        assert code == 0 and ("every stored trace replays (1)" in out or "1 stored decision(s) re-run" in out), out
        for extra, said in ((["--question", "nope"], "no such question"), (["--status", "zzz"], "--status")):
            capsys.readouterr()
            with pytest.raises(SystemExit) as e:
                main([cmd, store, "--system", system, *extra])
            assert e.value.code == 2 and said in capsys.readouterr().err
        capsys.readouterr()
        assert main([cmd, store, "--system", system, "--until", "2000-01-01"]) == 0       # nothing matches: said aloud
        assert "no stored decision matches" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        main(["diff", store, "--system", system, "--limit", "-5"])
    assert e.value.code == 2


DATED_TASK = '''import datetime as dt
from typing import Literal

from solvi import Catalog, Question, System

cat = Catalog()


@cat.rule("late")
def late(paid_on: dt.date) -> Literal["yes", "no"]:
    return "yes" if paid_on < dt.date(2026, 9, 20) else "no"


system = System(cat, [Question("late", "Was it paid late?")])
'''


def test_ask_text_takes_today_and_asks_the_end_user_in_their_words(tmp_path, capsys):
    f = tmp_path / "dated.py"
    f.write_text(DATED_TASK)
    code, out = run(capsys, "ask", f"{f}:system", "--text", "paid on: 12 September", "--json")
    read = json.loads(out)["textin"]
    assert code == 1 and read["fields"]["paid_on"]["status"] == "unparsed"
    code, out = run(capsys, "ask", f"{f}:system", "--text", "paid on: 12 September")
    assert "today=" not in out and "the year is missing" in out                  # the clarifying question
    code, out = run(capsys, "ask", f"{f}:system", "--text", "paid on: 12 September", "--today", "2026-09-28", "--json")
    data = json.loads(out)
    assert code == 0 and data["textin"]["state"] == {"paid_on": "2026-09-12"} and data["answers"]["late"]["answer"] == "yes"
    assert run(capsys, "ask", f"{f}:system", "--text", "paid on: 12 September", "--today", "today", "--json")[0] == 0
    assert run(capsys, "ask", f"{f}:system", "--text", "x", "--today", "soon")[0] == 2
    assert run(capsys, "ask", f"{f}:system", "--state", "{}", "--today", "2026-09-28")[0] == 2


def test_help_and_the_unknown_command_error_list_every_command(capsys):
    assert main(["--help"]) == 0
    text = capsys.readouterr().out
    listed = text.split("positional arguments:")[1].split("option")[0]
    for cmd in ("verify", "replay", "diff", "report", "serve", "check", "ask", "calibrate", "models", "init", "test",
                "honesty", "hook"):
        assert f" {cmd} " in listed or f"{cmd}," in listed, cmd
    assert main(["bogus"]) == 2
    err = capsys.readouterr().err
    assert all(f"'{c}'" in err for c in ("test", "honesty", "hook", "init"))
    with pytest.raises(SystemExit):
        main(["replay", "x.db", "--system", "os:getcwd"])
    assert "os:getcwd: not a solvi System" in capsys.readouterr().err   # no "--system" for commands that have none
