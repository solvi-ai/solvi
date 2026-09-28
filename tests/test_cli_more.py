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
    # a calibration made with another model stops the catalog from loading — but not `solvi calibrate`
    rec = json.loads(out.read_text())
    rec["fingerprint"], rec["models"] = "0" * 16, {"another/model": "1" * 16}
    out.write_text(json.dumps(rec))
    capsys.readouterr()
    assert main(["test", str(d)]) == 1 and "calibrated on another model" in capsys.readouterr().out
    code, text = run(capsys, "calibrate", f"{d}/catalog.py:system", "route", d / "labels.csv", "--json")
    data = json.loads(text)
    assert code == 0 and data["saved"] == "route.calib.json" and data["not_applied"]
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
