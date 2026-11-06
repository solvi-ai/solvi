from dataclasses import FrozenInstanceError

import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget
from solvi.experimental.experience.observations import ObservationOption, select_observation
from solvi.experimental.experience.programs import Grammar, Program


def test_frozen_copy_strict_inputs_and_cost():
    inputs = {"x": True}
    option = ObservationOption("read", inputs, 3)
    inputs["x"] = 7
    copy = option.inputs
    copy["x"] = 0
    assert option.inputs == {"x": True}
    with pytest.raises(FrozenInstanceError):
        option.cost = 8
    for bad in ({"x": 1.0}, {"x": None}, {"x": [1]}, {1: 1}):
        with pytest.raises(ValueError):
            ObservationOption("bad", bad, 1)
    for cost in (0, -1, True, 1.0):
        with pytest.raises(ValueError):
            ObservationOption("bad", {"x": 1}, cost)


def test_cost_effectiveness_ties_follow_host_order_not_name():
    grammar = Grammar(("x",), operators=("add",))
    programs = [Program(("input", "x")), Program(("const", 0)), Program(("const", 1))]
    options = [
        ObservationOption("expensive", {"x": 2}, 100),
        ObservationOption("z-cheap", {"x": 0}, 1),
        ObservationOption("a-cheap", {"x": 1}, 1),
    ]
    budget = WorkBudget(10000)
    choice = select_observation(programs, options, grammar, budget)
    assert (choice.name, choice.reason, choice.pairs_separated) == ("z-cheap", "selected", 2)
    assert budget.counts["program_node"] == 9
    assert budget.counts["observation_compare"] == 9
    assert budget.counts["observation_output"] == 9
    assert "domain_oracle" not in budget.counts
    assert "observation_read" not in budget.counts


def test_bool_int_are_distinct_and_invalid_is_not_gold():
    grammar = Grammar(("x",), operators=("add",), max_depth=1, max_nodes=3)
    pool = [ObservationOption("only", {"x": 0}, 1)]
    typed = select_observation([Program(("const", False)), Program(("const", 0))], pool, grammar, WorkBudget(1000))
    assert typed.pairs_separated == 1
    invalid = Program(("add", ("input", "x"), ("const", 0)))
    assert (
        select_observation(
            [invalid, Program(("const", False)), Program(("const", 0))], pool, grammar, WorkBudget(1000)
        ).pairs_separated
        == 3
    )
    # Different invalid errors share a marker; no oracle truth is manufactured.
    invalid_type = Program(("add", ("const", True), ("const", False)))
    assert select_observation([invalid, invalid_type], pool, grammar, WorkBudget(1000)).reason == "no_disagreement"


def test_no_disagreement_empty_and_explicit_caps():
    grammar = Grammar(("x",))
    option = ObservationOption("one", {"x": 2}, 1)
    same = [Program(("input", "x"))] * 2
    assert select_observation(same, [option], grammar, WorkBudget(1000)).name is None
    assert select_observation([], [option], grammar, WorkBudget(1000)).reason == "no_disagreement"
    assert select_observation(same, [], grammar, WorkBudget(1000)).reason == "no_disagreement"
    with pytest.raises(ValueError, match="32"):
        select_observation([same[0]] * 33, [option], grammar, WorkBudget(10000))
    with pytest.raises(ValueError, match="256"):
        select_observation(
            same, (ObservationOption(str(i), {"x": 1}, 1) for i in range(257)), grammar, WorkBudget(10000)
        )
    with pytest.raises(ValueError, match="uniquely"):
        select_observation(same, [option, option], grammar, WorkBudget(1000))
    with pytest.raises(ValueError, match="schema"):
        select_observation(same, [ObservationOption("wrong", {"y": 1}, 1)], grammar, WorkBudget(1000))


def test_budget_is_shared_and_read_price_is_not_purchased():
    grammar = Grammar(("x",))
    programs = [Program(("input", "x")), Program(("const", 0))]
    options = [ObservationOption("read", {"x": 1}, 1)]
    assert select_observation(programs, options, grammar, WorkBudget(0)).reason == "budget"
    measured = WorkBudget(1000)
    assert select_observation(programs, options, grammar, measured).reason == "selected"
    exact = WorkBudget(measured.spent)
    assert select_observation(programs, options, grammar, exact).reason == "budget"
    assert exact.spent == measured.spent
    shared = WorkBudget(measured.spent + 11)
    shared.charge("existing", 10)
    assert select_observation(programs, options, grammar, shared).name == "read"
    assert shared.remaining == 1


@pytest.mark.parametrize("exception", [ValueError, RuntimeError, BudgetStop])
def test_post_debit_sink_failure_propagates_inside_evaluator(exception):
    grammar = Grammar(("x",))

    def sink(snapshot):
        if snapshot["counts"].get("program_node"):
            raise exception("durable sink failed")

    budget = WorkBudget(1000, on_charge=sink)
    with pytest.raises(exception, match="durable sink failed"):
        select_observation(
            [Program(("input", "x")), Program(("const", 0))], [ObservationOption("read", {"x": 1}, 1)], grammar, budget
        )
    assert budget.counts["program_node"] == 1
