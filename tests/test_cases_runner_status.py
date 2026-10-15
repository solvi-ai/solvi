"""solvi test tells a regression from a nondeterministic decision and from a case that ran too long: each case has a
status, run_file(timeout=) stops a case that runs too long, run_file(reruns=) asks a failing case again."""
import json

from solvi import cli, testing

TASK = '''
import time
from solvi import Answer, Catalog, Question

cat = Catalog()
_n, _m = [0], [0]


@cat.fn
def words(text):
    if "slow" in text:
        time.sleep(5)
    return text.lower().split()


@cat.rule("route")
def route(words):
    if "flip" in words:                               # changes on every call: the replay does not recompute it
        _n[0] += 1
        return "billing" if _n[0] % 2 else "support"
    if "drift" in words:                              # changes between asks, the same within one ask and its replay
        _m[0] += 1
        return ["billing", "billing", "support", "support"][(_m[0] - 1) % 4]
    return "billing" if "invoice" in words else "support"


QUESTIONS = [Question("route", "Route?", Answer.choice(["billing", "support"]))]
'''

CASES = [
    {"name": "fine", "state": {"text": "invoice"}, "expected": {"route": "billing"}},
    {"name": "wrong", "state": {"text": "invoice"}, "expected": {"route": "support"}},
    {"name": "flip", "state": {"text": "flip"}, "expected": {"route": "billing"}},
    {"name": "drift", "state": {"text": "drift"}, "expected": {"route": "support"}},
    {"name": "slow", "state": {"text": "slow"}, "expected": {"route": "support"}},
    {"name": "boom", "state": {"text": "x"}, "expected": {"route": "support"}, "ask": ["nope"]},
    {"name": "after", "state": {"text": "help"}, "expected": {"route": "support"}},
]


def _write(tmp_path, cases=CASES):
    (tmp_path / "task.py").write_text(TASK)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    return tmp_path / "cases.json"


def test_each_case_has_a_status(tmp_path):
    res = testing.run_file(_write(tmp_path), timeout=0.5, reruns=3)
    st = {c.name: c.status for c in res.cases}
    assert st == {"fine": "ok", "wrong": "regression", "flip": "nondeterministic", "drift": "nondeterministic",
                  "slow": "timeout", "boom": "error", "after": "ok"}, st
    slow = next(c for c in res.cases if c.name == "slow")
    assert not slow.ok and slow.problems[0].startswith("timeout: ")
    assert res.counts() == {"ok": 2, "regression": 1, "nondeterministic": 2, "timeout": 1, "error": 1}


def test_without_reruns_only_a_step_that_does_not_recompute_is_nondeterministic(tmp_path):
    cases = [c for c in CASES if c["name"] not in ("slow",)]
    st = {c.name: c.status for c in testing.run_file(_write(tmp_path, cases)).cases}
    assert st["flip"] == "nondeterministic" and st["drift"] in ("regression", "ok")


def test_command_line_options(tmp_path, capsys):
    f = _write(tmp_path, [c for c in CASES if c["name"] in ("fine", "wrong", "slow")])
    assert cli.main(["test", str(f), "--timeout", "0.5", "--reruns", "1", "-q"]) == 1
    out = capsys.readouterr().out
    assert "1/3 cases passed" in out and "1 regression" in out and "1 timeout" in out
    assert cli.main(["test", str(f), "--timeout", "0.5", "--json"]) == 1
    data = json.loads(capsys.readouterr().out)
    assert [c["status"] for c in data["files"][0]["cases"]] == ["ok", "regression", "timeout"]
