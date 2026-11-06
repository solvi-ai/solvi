"""Host task manifest checks, independent of selection, actor and split seeds."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import sys

import pytest

from solvi.experimental.experience._common import WorkBudget

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from refinement_tasks import tasks  # noqa: E402


@pytest.mark.parametrize(
    ("task_id", "values", "expected"),
    [
        ("net_balance", (1, 3), -2),
        ("capped_allocation", (5, 2), 2),
        ("unfilled_orders", (1, 6), 0),
        ("line_total", (3, 7), 21),
        ("below_alert", (2, 2), False),
        ("release_gate", (2, 2, False), True),
        ("priority_access", (3, 0, True), True),
        ("signed_match", (3, 3, False), False),
        ("arrival_slot", (1, 3, 1), 4),
        ("available_funds", (0, 2, 4), -2),
        ("backup_supply", (3, 1, False), 1),
        ("pipeline_yield", (2, 3, 2), 2),
    ],
)
def test_independent_reference_witness(task_id, values, expected):
    problem = next(p for p in tasks() if p.task_id == task_id)
    actual = problem.reference(dict(zip(problem.grammar.inputs, values, strict=True)))
    assert type(actual) is type(expected) and actual == expected


def test_manifest_is_balanced_and_mechanisms_have_distinct_structure():
    problems = tasks()
    assert len({p.task_id for p in problems}) == len(problems) == 12
    assert Counter(p.family for p in problems) == {
        "numeric_tabular": 4,
        "boolean_policy": 4,
        "multistep_dependency": 4,
    }
    assert len({p.formula.fingerprint for p in problems}) == 12

    def shape(ast):
        if ast[0] in ("input", "const"):
            return ast[0]
        return (ast[0], *(shape(child) for child in ast[1:]))

    assert len({shape(p.formula.ast) for p in problems}) == 12
    assert [(p.task_id, p.grammar, p.points, p.formula) for p in tasks()] == [
        (p.task_id, p.grammar, p.points, p.formula) for p in problems
    ]


@pytest.mark.parametrize("problem", tasks(), ids=lambda p: p.task_id)
def test_formulas_agree_with_direct_targets_on_full_typed_domain(problem):
    grammar = problem.grammar
    assert grammar.max_abs == 1000 and grammar.max_candidates <= 3000 and grammar.max_nodes <= 7
    assert all(tuple(row) == grammar.inputs for row in problem.points)
    assert len({tuple(row.items()) for row in problem.points}) == len(problem.points) == 32
    trusted = replace(grammar, operators=("add", "sub", "mul", "max", "min", "lt", "eq", "or", "and", "not", "if"))
    budget = WorkBudget(100000)
    for row in problem.points:
        expected = problem.reference(row)
        assert type(expected) is (bool if problem.family == "boolean_policy" else int)
        assert abs(expected) <= grammar.max_abs
        actual = problem.formula.evaluate(row, budget, trusted)
        assert type(actual) is type(expected) and actual == expected
        if problem.task_id != "line_total":
            permitted = problem.formula.evaluate(row, budget, grammar)
            assert type(permitted) is type(expected) and permitted == expected

    for name, value in problem.points[0].items():
        wrong_type = int(value) if type(value) is bool else bool(value)
        with pytest.raises(ValueError, match="typed domain"):
            problem.reference({**problem.points[0], name: wrong_type})


def test_additive_grammar_cannot_express_line_total_even_with_more_nodes():
    problem = next(p for p in tasks() if p.task_id == "line_total")
    assert problem.grammar.operators == ("add",)
    # Every integer expression from inputs, constants and addition is affine,
    # so its mixed difference on this input square must be zero.
    f = problem.reference
    assert (
        f({"quantity": 1, "unit_price": 1})
        - f({"quantity": 1, "unit_price": 0})
        - f({"quantity": 0, "unit_price": 1})
        + f({"quantity": 0, "unit_price": 0})
    ) == 1
    with pytest.raises(ValueError, match="operator outside grammar"):
        problem.formula.evaluate(problem.points[0], WorkBudget(100), problem.grammar)
