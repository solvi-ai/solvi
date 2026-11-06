"""Known-answer checks for paid proposal-only stratification."""

from dataclasses import FrozenInstanceError
import pytest
from solvi.experimental.experience import (
    ExternalExample,
    ObservationOption,
    WorkBudget,
    BudgetStop,
    plan_stratified_observations,
)


def pilot(**changes):
    result = []
    for i in range(8):
        data = dict(
            event_id=f"e{i}",
            task_id="task",
            source="host",
            mechanism="observed",
            schema="bins",
            scope="scope",
            inputs={"f0": 0 if i < 4 else 7, "f1": 0 if i < 4 else 7},
            expected=i >= 4,
            split="proposal",
        )
        data.update(changes)
        result.append(ExternalExample(**data))
    return result


def options():
    return [
        ObservationOption(f"r{i}", {"f0": a, "f1": b}, 1)
        for i, (a, b) in enumerate([(7, 7), (0, 7), (7, 0), (0, 0), (0, 0), (7, 7), (7, 0), (0, 7)])
    ]


def plan(e=None, o=None, **kw):
    return plan_stratified_observations(
        pilot() if e is None else e, options() if o is None else o, WorkBudget(100000), **kw
    )


def test_upper_median_ties_round_robin_and_immutable():
    e, o = pilot(), options()
    before = ([x.to_dict() for x in e], [x.inputs for x in o])
    result = plan(e, o)
    assert result.features == ("f0", "f1")
    assert result.thresholds == (7, 7)
    assert result.names == ("r3", "r2", "r1", "r0", "r4", "r6", "r7", "r5")
    assert result == plan(e, o)
    assert before == ([x.to_dict() for x in e], [x.inputs for x in o])
    with pytest.raises(FrozenInstanceError):
        result.reason = "other"


def test_one_class_and_zero_score_fallback():
    assert plan(pilot(expected=True)).reason == "random_prefix"
    e = pilot()
    e = [ExternalExample(**{**x.to_dict(), "expected": i % 2 == 0}) for i, x in enumerate(e)]
    assert plan(e).names == tuple(x.name for x in options())
    assert plan(e).features == ()


@pytest.mark.parametrize(
    "changes",
    [{"expected": 1}, {"split": "evaluation"}, {"inputs": {"f0": True}}, {"inputs": {"f1": 1}}, {"inputs": {"f0": 8}}],
)
def test_strict_pilot(changes):
    with pytest.raises(ValueError):
        plan(pilot(**changes))


def test_caps_context_and_duplicate_ids():
    for e, o, kw in [
        (iter(pilot()), options(), {}),
        (pilot() * 9, options(), {}),
        (pilot(), options(), {"limit": True}),
        (pilot(), options(), {"candidate_limit": 65}),
        (pilot(), options(), {"limit": 9}),
        (pilot(), options() * 33, {}),
    ]:
        with pytest.raises(ValueError):
            plan(e, o, **kw)
    e = pilot()
    e[1] = e[0]
    with pytest.raises(ValueError):
        plan(e)
    e = pilot()
    e[1] = ExternalExample(**{**e[1].to_dict(), "scope": "different"})
    with pytest.raises(ValueError):
        plan(e)


def test_prefix_only_decode_and_shape_validation():
    o = options() + [ObservationOption("unvisited", {"wrong": True}, 1)]
    assert plan(o=o, candidate_limit=8) == plan()
    with pytest.raises(ValueError):
        plan(o=o, candidate_limit=9)


def test_budget_and_sink_failures_propagate_exactly():
    with pytest.raises(BudgetStop):
        plan_stratified_observations(pilot(), options(), WorkBudget(1))
    for error in (RuntimeError("sink"), BudgetStop("sink")):

        def fail(_):
            raise error

        b = WorkBudget(10000, on_charge=fail)
        with pytest.raises(type(error)) as caught:
            plan_stratified_observations(pilot(), options(), b)
        assert caught.value is error
        assert b.spent == 1


def test_each_cell_preserves_host_order_after_permutation():
    o = list(reversed(options()))
    r = plan(o=o)
    assert set(r.names) == {x.name for x in o}
    for bits in [(0, 0), (0, 7), (7, 0), (7, 7)]:
        expected = [x.name for x in o if tuple(x.inputs.values()) == bits]
        assert [name for name in r.names if name in expected] == expected


def test_exact_smoothed_scores_rank_before_numeric_ties():
    # f0: |3/8 - 5/8|=1/4; f1: |1/8 - 7/8|=3/4.
    # Both upper medians are 7; f1 must precede the lower-numbered f0.
    e = []
    low0 = {0, 1, 2, 3, 6, 7}
    for i in range(12):
        e.append(
            ExternalExample(
                f"e{i}",
                "task",
                "host",
                "observed",
                "bins",
                "scope",
                {"f0": 0 if i in low0 else 7, "f1": 0 if i < 6 else 7},
                i >= 6,
                "proposal",
            )
        )
    r = plan(e)
    assert r.features == ("f1", "f0")
    assert r.thresholds == (7, 7)
    assert r.names[:4] == ("r3", "r1", "r2", "r0")
