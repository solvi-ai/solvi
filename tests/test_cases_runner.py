"""Decision regression tests from cases.json (solvi.testing): the gallery's cases pass, mismatches in answers, statuses and
safeguards are reported, `solvi test` exits with the right code, the pytest plugin collects cases as items, and fuzzing
reports exceptions that escape solvi and answers with a non-numeric confidence."""
import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from solvi import cli, testing

pytest_plugins = ["pytester"]

ROOT = Path(__file__).resolve().parents[1]

TASK = '''
from solvi import Answer, Catalog, Maybe, Question, Unknown

cat = Catalog()


@cat.fn
def words(text):
    return text.lower().split()


@cat.check(hard=True, then={"route": "security"})
def no_password(words):
    return "password" not in words


@cat.rule("route")
def route(words):
    return "billing" if "invoice" in words else "support" if "help" in words else None


@cat.rule("paid")
def paid(text) -> Maybe[bool]:
    return True if "paid" in text else Unknown


@cat.rule("labels")
def labels(words):
    return [w for w in ("invoice", "help") if w in words]


QUESTIONS = [Question("route", "Route?", Answer.choice(["billing", "support", "security"])), Question("paid", "Paid?"),
             Question("labels", "Labels?", Answer.multi(["invoice", "help"]))]
'''

CASES = [
    {"name": "invoice", "state": {"text": "invoice paid"},
     "expected": {"route": "billing", "paid": "yes", "labels": ["invoice"]}},
    {"name": "password", "state": {"text": "my password please help"},
     "expected": {"route": "security", "labels": ["help"]}, "status": {"route": "forced"},
     "safeguards": {"route": ["hard_check"]}},
    {"name": "nothing", "state": {"text": "hello"}, "expected": {"route": None, "paid": "<not stated>", "labels": []},
     "status": {"route": "abstain"}},
    {"name": "abstain word", "state": {"text": "hello"}, "expected": {"route": "abstain"}, "ask": ["route"]},
]


def _write(tmp_path, cases=CASES, name="cases.json", task=TASK):
    d = tmp_path / "entry"
    d.mkdir(exist_ok=True)
    (d / "task.py").write_text(task)
    (d / name).write_text(json.dumps(cases))
    return d / name


def test_gallery_cases_pass():
    files = testing.run_path(ROOT / "gallery")
    assert len(files) == 15 and all(f.error is None for f in files)
    bad = [(f.path.parent.name, c.name, c.problems) for f in files for c in f.cases if not c.ok]
    assert not bad
    assert sum(len(f.cases) for f in files) >= 163


def test_a_passing_file(tmp_path):
    (res,) = testing.run_path(_write(tmp_path))
    assert res.ok and res.passed == 4
    assert res.cases[0].answers == {"route": ["billing", "ok"], "paid": ["yes", "ok"], "labels": [["invoice"], "ok"]}


def test_mismatches_are_reported_per_question(tmp_path):
    wrong = [
        {"name": "wrong answer", "state": {"text": "invoice"}, "expected": {"route": "support", "labels": ["invoice", "help"]}},
        {"name": "should abstain", "state": {"text": "invoice"}, "expected": {"route": None}},
        {"name": "abstained", "state": {"text": "hello"}, "expected": {"route": "billing", "paid": "no"}},
        {"name": "status and safeguards", "state": {"text": "password"}, "expected": {"route": "security"},
         "status": {"route": "ok"}, "safeguards": {"route": []}},
        {"name": "unknown question", "state": {"text": "x"}, "expected": {"nope": "x"}},
        {"name": "response safeguards", "state": {"text": "password"}, "safeguards": ["grounding"]},
    ]
    (res,) = testing.run_path(_write(tmp_path, wrong))
    assert not res.ok and res.passed == 0
    p = {c.name: c.problems for c in res.cases}
    assert p["wrong answer"] == ["route: expected 'support', got 'billing' [ok]",
                                 "labels: expected ['invoice', 'help'], got ['invoice'] [ok]"]
    assert p["should abstain"] == ["route: expected an abstention, got 'billing' [ok]"]
    assert p["abstained"][0].startswith("route: expected 'billing', abstained (") and \
        p["abstained"][1] == "paid: expected 'no', got '<not stated>' [ok]"
    assert p["status and safeguards"][0].startswith("route: expected status ok, got forced")
    assert p["status and safeguards"][1] == "route: expected safeguards [], fired ['hard_check']"
    assert p["unknown question"][0].startswith("nope: not asked")
    assert p["response safeguards"] == ["safeguards: expected ['grounding'], fired ['hard_check']"]


def test_task_and_system_can_be_named_in_the_file(tmp_path):
    d = tmp_path / "named"
    d.mkdir()
    (d / "logic.py").write_text(TASK)
    (d / "build.py").write_text("from solvi import System\nimport sys\n\ndef make():\n    import logic\n"
                                "    return System(logic.cat, logic.QUESTIONS)\n")
    f = d / "routing.cases.json"
    f.write_text(json.dumps({"task": "logic.py", "cases": CASES[:1]}))
    assert testing.run_file(f).ok
    f.write_text(json.dumps({"task": "logic.py", "system": "build.py:make", "cases": CASES[:1]}))
    assert testing.load(f).factory == "build.py:make" and testing.run_file(f).ok
    (tmp_path / "lonely").mkdir()
    (tmp_path / "lonely" / "cases.json").write_text("[]")
    res = testing.run_file(tmp_path / "lonely" / "cases.json")
    assert res.error.startswith("FileNotFoundError") and not res.ok


def test_command_line(tmp_path, capsys):
    good = _write(tmp_path)
    assert cli.main(["test", str(good)]) == 0
    out = capsys.readouterr().out
    assert "4/4 cases passed in 1 file(s)" in out and "ok   invoice" in out
    assert cli.main(["test", str(good), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["ok"] and data["cases"] == 4 and data["files"][0]["cases"][1]["answers"]["route"] == ["security", "forced"]
    bad = _write(tmp_path, [{"name": "x", "state": {"text": "invoice"}, "expected": {"route": "support"}}], "bad.cases.json")
    assert cli.main(["test", str(bad), "-q"]) == 1
    assert "FAIL x" in capsys.readouterr().out
    assert cli.main(["test", str(tmp_path)]) == 1                 # the directory: both files, one fails
    assert cli.main(["test", str(tmp_path / "missing")]) == 2
    (tmp_path / "empty").mkdir()
    assert cli.main(["test", str(tmp_path / "empty")]) == 2       # no cases files
    assert cli.main(["--version"]) == 0 and "solvi 0." in capsys.readouterr().out
    assert cli.main(["nope"]) == 2 and cli.main(["--help"]) == 0


def test_console_script_and_plugin_are_registered():
    text = (ROOT / "pyproject.toml").read_text()
    assert '[project.scripts]\nsolvi = "solvi.cli:main"' in text
    assert '[project.entry-points.pytest11]\nsolvi = "solvi.pytest_plugin"' in text


def test_mutations_are_deterministic_and_varied():
    st = {"text": "invoice", "n": 3, "items": [1, 2], "meta": {"a": 1}}
    a, b = testing.mutations(st, 40, seed=1), testing.mutations(st, 40, seed=1)
    assert [w for w, _ in a] == [w for w, _ in b] and len(a) == 40
    kinds = " ".join(w for w, _ in a)
    for word in ("removed", "= None", "garbled", "extra", "["):
        assert word in kinds
    assert st == {"text": "invoice", "n": 3, "items": [1, 2], "meta": {"a": 1}}      # the original is not touched


def test_fuzz_reports_crashes_and_invalid_confidences(tmp_path):
    from solvi.runtime import Result

    class Fragile:
        questions = {}

        def ask(self, st):
            if not isinstance(st.get("n"), int):
                raise KeyError("n")
            return type("R", (), {"results": {"q": Result("yes", float("nan") if st.get("text") is None else 0.9, "")}})()

    out = testing.fuzz(Fragile(), {"n": 1, "text": "a"}, 60, seed=0)
    assert any(x.startswith("crash — ") and "KeyError" in x for x in out)
    assert any(x.startswith("invalid — ") and "confidence nan" in x for x in out)
    suite = testing.load(_write(tmp_path))
    assert testing.fuzz(suite.system(), {"text": "invoice paid"}, 60) == []      # a plain catalog: no crash on odd input


def test_pytest_plugin_collects_cases_as_items(pytester):
    d = pytester.mkdir("entry")
    (d / "task.py").write_text(TASK)
    cases = CASES[:2] + [{"name": "wrong", "state": {"text": "invoice"}, "expected": {"route": "support"}}]
    (d / "cases.json").write_text(json.dumps(cases))
    (pytester.path / "other.json").write_text("[]")                  # not a cases file: ignored
    registered = any(ep.name == "solvi" and ep.value == "solvi.pytest_plugin" for ep in entry_points(group="pytest11"))
    args = [] if registered else ["-p", "solvi.pytest_plugin"]
    r = pytester.runpytest(*args, "-q", "entry")
    r.assert_outcomes(passed=2, failed=1)
    r.stdout.fnmatch_lines(["*case 'wrong'*", "*route: expected 'support', got 'billing' ?ok?*"])
    r = pytester.runpytest(*args, "-q", "entry", "--solvi-fuzz", "10", "-k", "invoice")
    r.assert_outcomes(passed=1, deselected=2)


def test_pytest_plugin_does_not_run_or_collect_another_projects_files(pytester):
    """The plugin loads wherever solvi is installed: a cases.json next to an unrelated project's task.py used to get that
    task.py executed during collection, and a *.cases.json with a "task" key aborted the session."""
    jobs = pytester.mkdir("jobs")
    (jobs / "task.py").write_text("import pathlib\npathlib.Path(__file__).with_name('RAN.txt').write_text('x')\n")
    (jobs / "cases.json").write_text(json.dumps([{"id": 1, "input": "a", "output": "b"}]))
    fixtures = pytester.mkdir("fixtures")
    (fixtures / "eval.cases.json").write_text(json.dumps({"task": "sentiment", "cases": [{"text": "great", "label": "pos"}]}))
    lookalike = pytester.mkdir("lookalike")                           # solvi-shaped cases, a module that is not solvi's
    (lookalike / "task.py").write_text("import pathlib\npathlib.Path(__file__).with_name('RAN.txt').write_text('x')\n")
    (lookalike / "cases.json").write_text(json.dumps(CASES[:1]))
    pytester.makepyfile(test_own="def test_ok():\n    assert True\n")
    registered = any(ep.name == "solvi" and ep.value == "solvi.pytest_plugin" for ep in entry_points(group="pytest11"))
    r = pytester.runpytest(*([] if registered else ["-p", "solvi.pytest_plugin"]), "-q")
    r.assert_outcomes(passed=1, warnings=1)
    assert not (jobs / "RAN.txt").exists() and not (lookalike / "RAN.txt").exists()
    r.stdout.fnmatch_lines(["*lookalike/cases.json: the cases look like solvi cases, but task.py does not import solvi*"])


def test_a_case_that_cannot_fail_is_reported(tmp_path):
    """A misspelled key, a status or safeguards entry for a question that was not asked, a case with only a state: each
    used to pass (6 of 7 such cases), so a regression case that checked nothing looked like a passing one."""
    hollow = [
        {"name": "typo in the key", "state": {"text": "invoice"}, "expcted": {"route": "support"}, "expected": {"route": "billing"}},
        {"name": "nothing expected", "state": {"text": "invoice"}},
        {"name": "status of a question not asked", "state": {"text": "invoice"}, "status": {"rout": "ok"}},
        {"name": "safeguards of a question not asked", "state": {"text": "password"}, "safeguards": {"rout": ["hard_check"]}},
        {"name": "no safeguards at all is an expectation", "state": {"text": "invoice"}, "safeguards": []},
        {"name": "a note is allowed", "state": {"text": "invoice"}, "expected": {"route": "billing"}, "note": "why this case"},
    ]
    (res,) = testing.run_path(_write(tmp_path, hollow))
    p = {c.name: c.problems for c in res.cases}
    assert p["typo in the key"] == ["unknown key(s) 'expcted' (a case has: name, state, expected, gold, status, safeguards, ask, note)"]
    assert p["nothing expected"] == ['the case expects nothing: give "expected", "status" or "safeguards"']
    assert p["status of a question not asked"][0].startswith("status of rout: not asked")
    assert p["safeguards of a question not asked"][0].startswith("safeguards of rout: not asked")
    assert p["no safeguards at all is an expectation"] == [] and p["a note is allowed"] == []


STORING_TASK = '''
from pathlib import Path
from typing import Literal

from solvi import Catalog, Question, System

cat = Catalog()


@cat.rule("big")
def big(amount: float) -> Literal["yes", "no"]:
    return "yes" if amount > 100 else "no"


def system():
    return System(cat, [Question("big", "Is it big?")], storage=str(Path(__file__).parent / "decisions.jsonl"))
'''


def test_the_test_runners_do_not_write_their_inputs_into_the_systems_store(tmp_path, capsys):
    (tmp_path / "task.py").write_text(STORING_TASK)
    (tmp_path / "cases.json").write_text(json.dumps({"system": "task.py:system", "cases": [
        {"name": "big", "state": {"amount": 500}, "expected": {"big": "yes"}}]}))
    store = tmp_path / "decisions.jsonl"

    def stored():
        return len(store.read_text().splitlines()) if store.exists() else 0
    assert testing.main([str(tmp_path), "-q"]) == 0 and stored() == 0
    assert testing.main([str(tmp_path), "-q", "--fuzz", "20"]) == 0 and stored() == 0     # nor the fuzz mutations
    (res,) = testing.run_path(tmp_path)
    assert res.ok and stored() == 0
    assert testing.main([str(tmp_path), "-q", "--store"]) == 0 and stored() == 1          # only when asked
    (tmp_path / "honesty.json").write_text(json.dumps({"task": "task.py", "cases": [
        {"name": "big", "state": {"amount": 500}, "gold": {"big": "yes"}}]}))
    from solvi import honesty
    assert honesty.report(tmp_path / "honesty.json")["cases"] == 1 and stored() == 1


def test_one_case_format_for_solvi_test_and_solvi_honesty(tmp_path, capsys):
    from solvi import honesty
    gallery = Path(__file__).resolve().parents[1] / "gallery" / "01_support_triage" / "cases.json"
    rep = honesty.report(gallery)                                    # a cases.json ("expected") is a honesty set
    assert rep["cases"] == len(json.loads(gallery.read_text())) > 5 and rep["metrics"]
    (tmp_path / "task.py").write_text(STORING_TASK)
    (tmp_path / "cases.json").write_text(json.dumps({"system": "task.py:system", "cases": [
        {"name": "big", "state": {"amount": 500}, "gold": {"big": "yes"}},              # a honesty case ("gold")
        {"name": "small", "state": {"amount": 5}, "gold": {"big": "yes"}}]}))
    (res,) = testing.run_path(tmp_path)
    assert [c.ok for c in res.cases] == [True, False] and "big" in res.cases[1].problems[0]
    with pytest.raises(ValueError, match='both "gold" and "expected"'):
        honesty.load_set(_both(tmp_path))


def _both(tmp_path):
    p = tmp_path / "both.json"
    p.write_text(json.dumps({"task": "task.py", "cases": [{"state": {}, "gold": {"big": "yes"}, "expected": {"big": "no"}}]}))
    return p
