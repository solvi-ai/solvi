"""run_file(fast=True): the answers and statuses of ask without its record keeping or the replay check — the same
verdicts as the full run on every gallery cases file, faster on many cases."""
import json
import time
from pathlib import Path

import pytest

from solvi import cli, testing

GALLERY = Path(__file__).resolve().parents[1] / "gallery"

TASK = '''
from solvi import Answer, Catalog, Question

cat = Catalog()


@cat.fn
def words(text):
    return text.lower().split()


@cat.check(hard=True, then={"route": "support"})
def not_empty(words):
    return bool(words)


@cat.rule("route")
def route(words):
    return "billing" if "invoice" in words else "support"


QUESTIONS = [Question("route", "Route?", Answer.choice(["billing", "support"]))]
'''


def _write(tmp_path, cases):
    (tmp_path / "task.py").write_text(TASK)
    (tmp_path / "cases.json").write_text(json.dumps(cases))
    return tmp_path / "cases.json"


def _verdicts(res):
    return [(c.name, c.status, c.answers, [p for p in c.problems if not p.startswith("trace replay")]) for c in res.cases]


def test_fast_gives_the_full_runs_verdicts_on_the_gallery():
    files = testing.find(GALLERY)
    assert len(files) >= 10
    for f in files:
        full, fast = testing.run_file(f), testing.run_file(f, fast=True)
        assert full.error == fast.error and _verdicts(full) == _verdicts(fast), f


def test_fast_is_faster_on_many_cases(tmp_path):
    cases = [{"name": f"c{i}", "state": {"text": ["pay the invoice", "help me", "where is my invoice now"][i % 3]},
              "expected": {"route": "billing" if i % 3 != 1 else "support"}} for i in range(300)]
    f = _write(tmp_path, cases)
    testing.run_file(f, fast=True)                                # warm up the imports
    testing.run_file(f)
    t_full, t_fast = [], []                                       # interleaved rounds, the best of each: one slow round
    for _ in range(5):                                            # on a loaded machine cannot decide the comparison
        t0 = time.perf_counter()
        full = testing.run_file(f)
        t1 = time.perf_counter()
        fast = testing.run_file(f, fast=True)
        t2 = time.perf_counter()
        t_full.append(t1 - t0)
        t_fast.append(t2 - t1)
        assert full.passed == fast.passed == 300
    assert min(t_fast) * 1.5 < min(t_full), (t_full, t_fast)


def test_safeguards_cases_are_asked_in_full_and_regressions_still_fail(tmp_path):
    cases = [{"name": "guarded", "state": {"text": ""}, "expected": {"route": "support"},
              "safeguards": {"route": ["hard_check"]}},
             {"name": "wrong", "state": {"text": "invoice"}, "expected": {"route": "support"}},
             {"name": "odd", "state": {"text": "invoice"}, "expected": {"route": "billing"}, "expcted": 1}]
    st = {c.name: c.status for c in testing.run_file(_write(tmp_path, cases), fast=True).cases}
    assert st == {"guarded": "ok", "wrong": "regression", "odd": "invalid"}
    with pytest.raises(ValueError, match="store"):
        testing.run_file(tmp_path / "cases.json", fast=True, store=True)


def test_command_line_fast(tmp_path, capsys):
    f = _write(tmp_path, [{"name": "fine", "state": {"text": "invoice"}, "expected": {"route": "billing"}}])
    assert cli.main(["test", str(f), "--fast", "-q"]) == 0
    assert "1/1 cases passed" in capsys.readouterr().out
