"""Regression tests for the finite scalar DSL and its charged proposer."""

import pytest

from solvi.core.costs import BudgetStop
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.programs import Grammar, Program, synthesize


def test_composition_transfers_without_lookup():
    g = Grammar(("x",), operators=("add", "mul"))
    result = synthesize([({"x": x}, 2 * x + 1) for x in (-2, 0, 3)], g, WorkBudget(100000))
    assert result.reason == "found"
    assert result.candidate.nodes >= 5
    assert result.candidate.evaluate({"x": 17}, WorkBudget(100), g) == 35


def test_boolean_composition_and_strict_equality():
    g = Grammar(("x", "y"), operators=("lt", "eq"))
    r = synthesize(
        [({"x": 1, "y": 2}, True), ({"x": 2, "y": 1}, False), ({"x": 3, "y": 3}, False)], g, WorkBudget(10000)
    )
    assert r.candidate.evaluate({"x": 4, "y": 9}, WorkBudget(100), g) is True
    with pytest.raises(ValueError):
        Program(["eq", ["input", "x"], ["input", "y"]]).evaluate({"x": True, "y": 1}, WorkBudget(100), g)
    with pytest.raises(ValueError):
        Grammar(("x",), constants=(True,))


@pytest.mark.parametrize("ast", [[], ["bad"], ["const", float("nan")], ["add", ["const", 1]], ["input", 12]])
def test_malformed_ast(ast):
    with pytest.raises(ValueError):
        Program(ast)


def test_depth_overflow_and_operator_caps():
    ast = ["const", 1]
    for _ in range(70):
        ast = ["not", ast]
    with pytest.raises(ValueError):
        Program(ast)
    g = Grammar(("x",), max_abs=10)
    with pytest.raises(ValueError):
        Program(["mul", ["input", "x"], ["input", "x"]]).evaluate({"x": 4}, WorkBudget(100), g)
    with pytest.raises(ValueError):
        Program(["add", ["const", 1], ["const", 1]]).evaluate({}, WorkBudget(100), Grammar((), max_nodes=1))


def test_budget_precedes_execution_and_failure_is_charged():
    p = Program(["add", ["input", "x"], ["const", 1]])
    b = WorkBudget(0)
    with pytest.raises(BudgetStop):
        p.evaluate({"x": "invalid"}, b, Grammar(("x",)))
    assert b.spent == 0
    b = WorkBudget(20)
    with pytest.raises(ValueError):
        p.evaluate({"x": True}, b, Grammar(("x",)))
    assert b.spent > 0


def test_empty_examples_and_caps_have_explicit_status():
    assert synthesize([], Grammar(("x",)), WorkBudget(10)).reason == "no_examples"
    r = synthesize([({"x": 0}, 100)], Grammar(("x",), max_candidates=1), WorkBudget(100))
    assert r.reason == "candidate_limit" and r.tried == 1
    r = synthesize([({"x": 0}, 1)], Grammar(("x",)), WorkBudget(0))
    assert r.reason == "budget_exhausted" and r.candidate is None


def test_snapshot_identity_and_caller_mutation():
    tree = ["add", ["input", "x"], ["const", 1]]
    p = Program(tree)
    tree[2][1] = 99
    snapshot = p.to_dict()
    restored = Program.from_dict(snapshot)
    assert restored.fingerprint == p.fingerprint
    assert restored.input_names == ("x",) and restored.nodes == 3 and restored.depth == 2
    snapshot["ast"][2][1] = 7
    assert p.evaluate({"x": 2}, WorkBudget(50), Grammar(("x",))) == 3
    with pytest.raises(ValueError):
        Program.from_dict(snapshot)
