"""Backend-independent bounded selection, without oracle access."""

import itertools

import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget
from solvi.experimental.experience.observations import ObservationOption, select_observation_bounded
from solvi.experimental.experience.programs import Grammar, Program


def predict(model, inputs, budget):
    budget.charge("forecast")
    value = model[inputs["x"]]
    if isinstance(value, Exception):
        raise value
    return value


def options(costs):
    return [ObservationOption(str(i), {"x": i}, cost) for i, cost in enumerate(costs)]


def test_bool_bound_early_exit_never_calls_oracle_or_later_predictions():
    seen, oracle_calls = [], []

    def paid(model, inputs, budget):
        seen.append(inputs["x"])
        return predict(model, inputs, budget)

    # Bool's maximum is 4, attained immediately; all later rows tie at best.
    choice = select_observation_bounded(
        [[False, True], [False, True], [True, False], [True, False]],
        options([1, 1]), WorkBudget(1000), predict=paid, result_type=bool,
    )
    assert (choice.name, choice.reason, choice.pairs_separated) == ("0", "selected", 4)
    assert seen == [0] * 4
    assert oracle_calls == []


@pytest.mark.parametrize("result_type, values", [(bool, [False, True]), (int, [0, 1, 2]), (None, [False, 0, 1])])
def test_agrees_with_full_bruteforce_and_host_ties(result_type, values):
    for offset in range(6):
        rows = list(itertools.product(values, repeat=3))[offset:offset + 5]
        models = [[row[i] for row in rows] for i in range(3)]
        costs = [i % 3 + 1 for i in range(len(rows))]
        scores = [sum((type(a), a) != (type(b), b) for a, b in itertools.combinations(row, 2)) for row in rows]
        expected = None
        for i, pairs in enumerate(scores):
            if pairs and (expected is None or pairs * costs[expected] > scores[expected] * costs[i]):
                expected = i
        choice = select_observation_bounded(
            models, options(costs), WorkBudget(10000), predict=predict, result_type=result_type,
        )
        assert choice.name == (None if expected is None else str(expected))


def test_cap_discards_unfinished_row_and_preserves_reserve():
    models = [[False, False], [False, True], [True, True]]
    # Find a cap in row two: its incomplete, higher score must not replace row one.
    found = False
    for cap in range(1, 70):
        seen = []

        def paid(model, inputs, budget):
            value = predict(model, inputs, budget)
            seen.append(inputs["x"])
            return value

        budget = WorkBudget(200)
        choice = select_observation_bounded(models, options([2, 1]), budget, predict=paid, max_work=cap, reserve=170)
        assert budget.spent <= cap and budget.remaining >= 170
        if 0 < seen.count(1) < 3:
            found = True
            assert (choice.name, choice.reason, choice.pairs_separated) == ("0", "selected_partial", 2)
    assert found


def test_exact_parent_exhaustion_huge_price_and_zero_window():
    models = [[False], [True]]
    assert select_observation_bounded(models, options([1]), WorkBudget(100), predict=predict, max_work=0).reason == "budget"
    measured = WorkBudget(1000)
    select_observation_bounded(models, options([1]), measured, predict=predict)
    budget = WorkBudget(measured.spent)
    assert select_observation_bounded(models, options([1]), budget, predict=predict).reason == "budget"
    assert budget.remaining == 0
    assert select_observation_bounded(models, options([10**100]), WorkBudget(1000), predict=predict).reason == "budget"


def test_strict_bool_int_invalid_marker_and_contract_violations():
    assert select_observation_bounded([[False], [0]], options([1]), WorkBudget(1000), predict=predict).pairs_separated == 1
    assert select_observation_bounded(
        [[ValueError("a")], [ValueError("b")], [0]], options([1]), WorkBudget(1000), predict=predict,
    ).pairs_separated == 2
    for value in [0, ValueError("bad")]:
        with pytest.raises(ValueError):
            select_observation_bounded([[False], [value]], options([1]), WorkBudget(1000), predict=predict, result_type=bool)


@pytest.mark.parametrize("error_type", [ValueError, BudgetStop, RuntimeError])
@pytest.mark.parametrize("category", ["forecast", "observation_compare", "observation_cache"])
def test_post_debit_sink_failure_propagates_exactly(error_type, category):
    error = error_type("durable failure")

    def sink(snapshot):
        if snapshot["counts"].get(category):
            raise error

    budget = WorkBudget(1000, on_charge=sink)
    with pytest.raises(error_type) as raised:
        select_observation_bounded([[False], [True]], options([1]), budget, predict=predict)
    assert raised.value is error
    assert budget.counts[category] > 0


def test_one_model_skips_forecast_and_program_wrapper_is_supported():
    def forbidden(*args):
        pytest.fail("one model must not forecast")

    assert select_observation_bounded([object()], options([1]), WorkBudget(1000), predict=forbidden).reason == "no_disagreement"
    grammar = Grammar(("x",))
    choice = select_observation_bounded(
        [Program(("input", "x")), Program(("const", 1))], options([1]), WorkBudget(1000),
        predict=lambda program, inputs, paid: program.evaluate(inputs, paid, grammar), result_type=int,
    )
    assert choice.pairs_separated == 1


def test_caps_metadata_and_generator_are_bounded():
    with pytest.raises(ValueError, match="32"):
        select_observation_bounded([0] * 33, [], WorkBudget(1000), predict=predict)
    with pytest.raises(ValueError, match="256"):
        select_observation_bounded([0, 1], options([1] * 257), WorkBudget(1000), predict=predict)
    with pytest.raises(ValueError, match="uniquely"):
        select_observation_bounded([0, 1], [*options([1]), *options([1])], WorkBudget(1000), predict=predict)
    with pytest.raises(ValueError, match="32"):
        select_observation_bounded(itertools.repeat(0), [], WorkBudget(1000), predict=predict)
