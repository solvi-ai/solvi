import math
import pytest
from solvi.experimental.laws import GaussianProbes


def model(means=(0.0, 2.0), **kw):
    return GaussianProbes([lambda s, a, m=m: m * s for m in means], sigma=kw.pop("sigma", 1.0), **kw)


def test_normal_likelihood_analytic_and_repeat_same_point():
    g = model()
    p = g.choose(lambda *_: True, [(1, "read")])
    assert 0.40 < p.info < 0.55  # binary Normal channel, not categorical flip entropy
    assert g.observe(p, 2.0)[1] == pytest.approx(1 / (1 + math.exp(-2)))
    assert g.observe(p, 2.0)[1] == pytest.approx(1 / (1 + math.exp(-4)))
    assert sum(g.weights) == pytest.approx(1)


def test_adaptive_loop_stops_confidence_and_preserves_repeat():
    g = model()
    seen = []
    rows = g.run(
        lambda s, a: seen.append((s, a)) or 2.0, lambda *_: True, [(1, "r")], max_observations=10, confidence=0.98
    )
    assert len(rows) == 2 and seen == [(1, "r")] * 2 and rows[-1]["spent"] == 2


def test_no_equal_mean_calls_and_budget_zero():
    def forbidden(*_):
        raise AssertionError("no measurement")

    assert model((1.0, 1.0)).run(forbidden, lambda *_: True, [(1, "r")], max_observations=5) == []
    assert model().run(forbidden, lambda *_: True, [(1, "r")], max_observations=0) == []
    assert model().run(forbidden, lambda *_: True, [(1, "r")], max_observations=5, max_cost=0) == []
    assert len(model().run(lambda *_: 2.0, lambda *_: True, [(1, "r")], max_observations=10, max_cost=2.5)) == 2


def test_safe_readonly_cost_and_order():
    g = model(sigma=lambda s, a: (_ for _ in ()).throw(AssertionError()) if a == "unsafe" else 1.0)
    points = [(2, "unsafe"), (2, "write"), (1, "expensive"), (1, "cheap"), (1, "tie")]
    cost = lambda s, a: 10 if a == "expensive" else 1
    p = g.choose(lambda s, a: a != "unsafe", points, read_only=lambda a: a != "write", cost=cost)
    assert p.action == "cheap"
    assert g.choose(lambda *_: True, [(1, "expensive"), (0.8, "cheap")], cost=cost, per_cost=True).action == "cheap"


@pytest.mark.parametrize("sigma", [0, -1, float("nan"), float("inf"), True])
def test_invalid_noise(sigma):
    with pytest.raises(ValueError):
        model(sigma=sigma)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_bad_observation_atomic(value):
    g = model()
    p = g.choose(lambda *_: True, [(1, "r")])
    before = g.weights
    with pytest.raises(ValueError):
        g.observe(p, value)
    assert g.weights == before and g.observations == 0


def test_logspace_extreme_and_nonuniform_prior():
    g = model(weights=[1e300, 1e300])
    p = g.choose(lambda *_: True, [(1, "r")])
    assert g.observe(p, 1000)[1] == pytest.approx(1)
    assert sum(g.weights) == pytest.approx(1)
    g = model(weights=[3, 1])
    p = g.choose(lambda *_: True, [(1, "r")])
    assert g.observe(p, 1) == pytest.approx((0.75, 0.25))


def test_invalid_budget_prediction_and_cost():
    for kw in [
        dict(max_observations=True),
        dict(max_observations=1, max_cost=float("nan")),
        dict(max_observations=1, confidence=0),
    ]:
        with pytest.raises(ValueError):
            model().run(lambda *_: 0, lambda *_: True, [(1, "r")], **kw)
    with pytest.raises(ValueError):
        model().choose(lambda *_: True, [(float("nan"), "r")])
    with pytest.raises(ValueError):
        model().choose(lambda *_: True, [(1, "r")], cost=lambda *_: -1)


@pytest.mark.parametrize("observation", [1e100, -1e100])
def test_astronomical_likelihood_rejected_atomically(observation):
    g = model()
    p = g.choose(lambda *_: True, [(1, "r")])
    before = (g.weights, g.observations)
    with pytest.raises(ValueError, match="supported finite range"):
        g.observe(p, observation)
    assert (g.weights, g.observations) == before


def test_centered_means_posterior_and_information_translation_invariant():
    g = model()
    p = g.choose(lambda *_: True, [(1, "r")])
    info = p.info
    shifted = model((1e12, 1e12 + 2))
    q = shifted.choose(lambda *_: True, [(1, "r")])
    assert q.info == pytest.approx(info, abs=1e-12)
    assert shifted.observe(q, 1e12 + 2) == pytest.approx(g.observe(p, 2.0))


def test_unsupported_relative_scale_is_atomic():
    g = model(sigma=1e-200)
    before = (g.weights, g.observations)
    with pytest.raises(ValueError, match="supported finite range"):
        g.choose(lambda *_: True, [(1, "r")])
    assert (g.weights, g.observations) == before


def test_repeated_in_range_opposites_preserve_known_logodds():
    g = model()
    p = g.choose(lambda *_: True, [(1, "r")])
    for _ in range(50):
        g.observe(p, 100.0)
        g.observe(p, -100.0)
    # Each pair contributes -4 log odds; 50 pairs => -200.
    assert math.log(g.weights[1] / g.weights[0]) == pytest.approx(-200, abs=1e-10)
    assert g.observations == 100 and sum(g.weights) == pytest.approx(1)


def test_foreign_noise_and_means_changed_probes_refuse_atomically():
    g = model()
    p = g.choose(lambda *_: True, [(1, "r")])
    other = model()
    before = (other.weights, other.observations)
    with pytest.raises(ValueError, match="this planner"):
        other.observe(p, 1.0)
    assert (other.weights, other.observations) == before
    state = {"sigma": 1.0, "mean": 2.0}
    g = GaussianProbes([lambda *_: 0.0, lambda *_: state["mean"]], sigma=lambda *_: state["sigma"])
    p = g.choose(lambda *_: True, [(1, "r")])
    before = (g.weights, g.observations)
    state["sigma"] = 2.0
    with pytest.raises(ValueError, match="noise changed"):
        g.observe(p, 1.0)
    state["sigma"] = 1.0
    state["mean"] = 3.0
    with pytest.raises(ValueError, match="predictions changed"):
        g.observe(p, 1.0)
    assert (g.weights, g.observations) == before
