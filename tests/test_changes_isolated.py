"""solvi.experimental.changes.IsolatedTests: the candidate's code is run on test inputs in a fresh `python -I` process
with limits (solvi.experimental.compile.sandbox.run); the expected answers stay in the parent and are compared
strictly (type and value), so code that reads or prints the answers, crashes, loops or returns 1 for True is refused."""
from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import Change, IsolatedTests, Pipeline, StageVerdict

CASES = [{"id": "a", "input": [2, 3], "expected": 5}, {"id": "b", "input": [0, 0], "expected": 0},
         {"id": "c", "input": [-1, 1], "expected": 0}]


class Code:
    """A target that keeps the rule's code (an update with code)."""
    artifact = "update"

    def promote(self, change, measured):
        return {"code": change.code}

    def rollback(self, ref, by=None, why=None):
        pass


def code_change(src):
    return Change("outcome", artifact="update", payload={"fingerprint": src}, code=src, proposer="self")


def pipe_of(tmp_path, stage, name="c.jsonl"):
    return Pipeline([stage], str(tmp_path / name), targets=[Code()], approval={"update": "gate"})


def test_correct_code_passes_and_the_numbers_are_recorded(tmp_path):
    res = pipe_of(tmp_path, IsolatedTests(CASES, entry="add")).propose(code_change("def add(a, b):\n    return a + b\n"))
    assert res.promoted, str(res)
    m = res.verdicts[0].measured
    assert m["n"] == 3 and m["passed"] == 3 and m["failed"] == 0


@pytest.mark.parametrize("src, why", [
    ("def add(a, b):\n    return a - b\n", "wrong"),
    ("def add(a, b):\n    return float(a + b)\n", "wrong"),          # 5.0 is not 5: a strict compare
    ("def add(a, b):\n    return 1 // (a - a)\n", "raised"),
    ("def add(a, b):\n    while True:\n        pass\n", "raised"),    # the per-item alarm
    ("import os\ndef add(a, b):\n    return a + b\n", "load"),        # refused by the sandbox's check
])
def test_wrong_crashing_looping_or_forbidden_code_is_refused(tmp_path, src, why):
    st = IsolatedTests(CASES, entry="add", item_s=1, cpu_s=20, wall_s=40)
    res = pipe_of(tmp_path, st).propose(code_change(src))
    assert res.status == "rejected" and res.stage == "isolated_tests", str(res)
    if why == "load":
        assert "did not load" in res.reason
    else:
        assert res.verdicts[0].measured["failed"] >= 1


def test_the_expected_answers_never_reach_the_child(tmp_path):
    seen = {}

    def spy(source, driver, payload, **kw):
        seen.update(payload)
        return {"outputs": [{"value": 5}, {"value": 0}, {"value": 0}]}
    st = IsolatedTests(CASES, entry="add", runner=spy)
    st.run(code_change("def add(a, b):\n    return a + b\n"), None)
    assert "expected" not in repr(seen) and 5 not in [c for c in seen["inputs"]]


def test_no_code_is_refused_and_a_sandbox_failure_is_a_stage_error(tmp_path):
    res = pipe_of(tmp_path, IsolatedTests(CASES, entry="add")).propose(
        Change("outcome", artifact="update", payload={"fingerprint": "x"}))
    assert res.status == "rejected" and "no code" in res.reason
    st = IsolatedTests(CASES, entry="add", runner=lambda *a, **k: {"sandbox_error": "no interpreter"})
    res = pipe_of(tmp_path, st, name="c2.jsonl").propose(code_change("def add(a, b):\n    return a + b\n"))
    assert res.status == "error" and "no interpreter" in res.reason


def test_cases_from_a_function_and_an_own_compare(tmp_path):
    st = IsolatedTests(lambda change, ctx: [((1.0, 2.0), 3.0)], entry="add",
                       compare=lambda got, want: abs(got - want) < 1e-9)
    res = pipe_of(tmp_path, st).propose(code_change("def add(a, b):\n    return a + b + 1e-12\n"))
    assert res.promoted, str(res)
    assert StageVerdict  # the module's names are importable together
