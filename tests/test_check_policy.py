import pytest

from solvi.core.plan.check_policy import CheckPolicy
from solvi.stat import NaiveCombiner, CorrelationAwareCombiner


def model(names=("a", "b"), correlated=False):
    cls = CorrelationAwareCombiner if correlated else NaiveCombiner
    args = (names, lambda a, b: 1.0 if a == b else 0.5) if correlated else (names,)
    return cls(*args).fit(
        [{s: outcome for s in names} for outcome in ("agree", "disagree") for _ in range(20)],
        [1] * 20 + [0] * 20)


def test_cheap_check_selected_and_expensive_check_declined():
    policy = CheckPolicy(model(), costs={"a": .01, "b": 2})
    action = policy.decide(.5, {}, 1)
    assert action.kind == "check" and action.check == "a" and action.value > 0
    assert CheckPolicy(model(), costs={"a": 2, "b": 2}).decide(.5, {}, 1).kind == "abstain"
    assert policy.decide(.99, {}, 0).kind == "answer"


def test_stopping_and_budgets_include_already_bought_checks():
    for budget in ({"max_checks": 0}, {"max_cost": 0}):
        assert CheckPolicy(model(), costs={"a": .01, "b": .01}, **budget).decide(.5, {}, 1).kind == "abstain"
    policy = CheckPolicy(model(), costs={"a": .01, "b": .01}, max_checks=1)
    assert policy.decide(.5, {"a": "agree"}, 1).kind == "answer"
    assert policy.decide(.5, {"a": "disagree"}, 1).kind == "abstain"


def test_seeded_risk_snapshot_and_callback_isolation():
    source = model(correlated=True)
    costs = {"a": .01, "b": .01}
    policy = CheckPolicy(source, costs=costs, risk_quantile=.1, n=64, seed=7)
    before = policy.fingerprint()
    expected = policy.decide(.5, {}, 1)
    costs["a"] = 999
    source.fit([], [])
    source.rho = lambda a, b: 1.0
    assert policy.fingerprint() == before
    assert policy.decide(.5, {}, 1) == expected
    assert policy.freeze().decide(.5, {}, 1) == expected


@pytest.mark.parametrize("kwargs", [
    {"costs": {}}, {"costs": {"a": -1, "b": 0}}, {"costs": {"a": True, "b": 0}},
    {"max_checks": True}, {"max_checks": -1}, {"max_cost": float("nan")},
    {"risk_quantile": 2}, {"n": 0}, {"seed": -1},
])
def test_invalid_config(kwargs):
    config = {"costs": {"a": .01, "b": .01}, **kwargs}
    with pytest.raises(ValueError):
        CheckPolicy(model(), **config)


def test_invalid_model_state_and_decision_inputs():
    with pytest.raises(ValueError, match="six"):
        CheckPolicy(model(tuple("abcdefg")), costs={s: 0 for s in "abcdefg"})
    with pytest.raises(ValueError, match="fit"):
        CheckPolicy(NaiveCombiner(["a"]), costs={"a": 0})
    policy = CheckPolicy(model(), costs={"a": 0, "b": 0})
    for prior, outcomes, loss in [(2, {}, 1), (.5, {"a": None}, 1), (.5, {"x": "agree"}, 1), (.5, {}, -1)]:
        with pytest.raises(ValueError):
            policy.decide(prior, outcomes, loss)


def test_one_check_bellman_value_matches_explicit_terminal_sum():
    native = model(("a",))
    prior, loss, price = .5, 1.0, .02
    expected = -price
    for outcome in native.outcomes:
        p = native.posterior(prior, {"a": outcome})
        predictive = prior * native.lik["a"][1][outcome] + (1 - prior) * native.lik["a"][0][outcome]
        expected += predictive * max(0, p - loss * (1 - p))
    action = CheckPolicy(native, costs={"a": price}).decide(prior, {}, loss)
    assert action.kind == "check"
    assert action.value == pytest.approx(expected)
    # The price of a completed check is sunk and must not be subtracted twice.
    terminal = CheckPolicy(native, costs={"a": price}).decide(prior, {"a": "agree"}, loss)
    p = native.posterior(prior, {"a": "agree"})
    assert terminal.value == pytest.approx(p - loss * (1 - p))


def test_availability_filters_future_acquisitions_without_imputing_outcomes():
    policy = CheckPolicy(model(), costs={"a": .01, "b": .02})
    assert policy.decide(.5, {}, 1, available=[]).kind == "abstain"
    assert policy.decide(.5, {}, 1, available=["b"]).check == "b"
    assert policy.decide(.5, {"a": "agree"}, 1, available=[]).kind == "answer"
    for available in (["missing"], ["a", "a"], "a", 42):
        with pytest.raises(ValueError):
            policy.decide(.5, {}, 1, available=available)
    unsupported = NaiveCombiner(["a"], outcomes=("yes", "no")).fit([], [])
    with pytest.raises(ValueError, match="exactly"):
        CheckPolicy(unsupported, costs={"a": 0})
