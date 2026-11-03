"""S115 supplied domain correlations; descriptive weighting, no new calibration claim."""
import math

import pytest

from solvi.stat import CorrelationAwareCombiner


def fitted(**kw):
    return CorrelationAwareCombiner(["a", "b"], {("a", "b"): .4}, **kw).fit(
        [{"a": "agree", "b": "agree"}] * 8 + [{"a": "disagree", "b": "disagree"}] * 8,
        [1] * 8 + [0] * 8)


def test_same_outcomes_use_only_explicit_domain_correlation():
    model = fitted(rho_by_domain={"copies": {("a", "b"): .9}, "separate": {("a", "b"): 0}})
    obs = {"a": "agree", "b": "agree"}
    assert model.log_lr(obs, domain="separate") > model.log_lr(obs, domain="copies") > 0
    assert model.posterior(.5, obs, domain="separate") > model.posterior(.5, obs, domain="copies")
    assert model.n_effective(["a", "b"], domain="copies") < model.n_effective(["a", "b"], domain="separate")
    assert model.weights(["a"], domain="copies") == {"a": 1.0}
    assert model.counts == fitted().counts  # domain keys never enter likelihood training rows


def test_global_default_unchanged_and_input_matrix_copied():
    matrix = {("a", "b"): .9}
    model = fitted(rho_by_domain={"copies": matrix})
    before = model.weights(["a", "b"], domain="copies")
    matrix[("a", "b")] = 0
    assert model.weights(["a", "b"], domain="copies") == before
    for obs in ({}, {"a": "agree"}, {"a": "agree", "b": "disagree"}):
        assert model.log_lr(obs) == fitted().log_lr(obs)
        assert model.posterior(.25, obs) == fitted().posterior(.25, obs)


@pytest.mark.parametrize("domain", ["unseen", "", False, [], 0])
def test_unseen_or_unestimated_domain_refuses_even_without_observed_checks(domain):
    model = fitted()
    with pytest.raises(ValueError, match="unknown correlation domain"):
        model.log_lr({}, domain=domain)
    with pytest.raises(ValueError, match="unknown correlation domain"):
        model.n_effective([], domain=domain)


@pytest.mark.parametrize("matrix", [[], {"": {}}, {"x": lambda a, b: 0}, {"x": {("a", "b"): float("nan")}}])
def test_domain_configuration_validated(matrix):
    with pytest.raises(ValueError):
        fitted(rho_by_domain=matrix)


def test_domain_keeps_logspace_extreme_smoothing_finite():
    model = fitted(pool=1e300, alpha=1e-300, rho_by_domain={"x": {("a", "b"): .99}})
    assert math.isfinite(model.log_lr({"a": "agree", "b": "disagree"}, domain="x"))


def test_domain_shrinkage_and_small_counts_refusal():
    raw = fitted(rho_by_domain={"x": {("a", "b"): .9}})
    shrunk = fitted(rho_by_domain={"x": {("a", "b"): .9}}, domain_counts={"x": 2}, domain_shrinkage=2)
    assert shrunk._domain_rho("x")("a", "b") == pytest.approx(.65)
    huge = fitted(rho_by_domain={"x": {("a", "b"): .9}}, domain_counts={"x": 2}, domain_shrinkage=1e300)
    with pytest.warns(UserWarning, match="global rho"):
        assert huge.weights(["a", "b"], domain="x") == pytest.approx(huge.weights(["a", "b"]))
    assert raw.weights(["a", "b"], domain="x") != shrunk.weights(["a", "b"], domain="x")
    small = fitted(rho_by_domain={"x": {}}, domain_counts={"x": 1}, domain_shrinkage=1)
    with pytest.raises(ValueError, match="at least two"):
        small.log_lr({}, domain="x")


@pytest.mark.parametrize("counts,strength", [({}, 1), ({"x": True}, 1), ({"x": -1}, 1),
    ({"bad": 2}, 0), ({"x": 2}, float("inf")), ({"x": 2}, -1)])
def test_domain_count_and_shrinkage_validation(counts, strength):
    with pytest.raises(ValueError):
        fitted(rho_by_domain={"x": {}}, domain_counts=counts, domain_shrinkage=strength)
