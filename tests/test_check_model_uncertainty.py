"""Conditional uncertainty is a model diagnostic, not a coverage certificate."""
import numpy as np
import pytest

from solvi.stat import NaiveCombiner, CorrelationAwareCombiner


def fitted(cls=NaiveCombiner, repeats=1):
    kwargs = {"rho": {("a", "b"): 1.0}} if cls is CorrelationAwareCombiner else {}
    return cls(["a", "b"], outcomes=("yes", "no"), **kwargs).fit(
        [{"a": "yes", "b": "yes"}, {"a": "no", "b": "no"}] * repeats,
        [1, 0] * repeats,
    )


def test_legacy_fitted_model_retains_point_prediction_without_uncertainty_state():
    model = fitted(repeats=12)
    expected = model.posterior(.4, {"a": "yes"})
    del model._log_concentrations
    frozen = model.freeze()
    assert len(frozen.fingerprint()) == 64
    assert frozen.posterior(.4, {"a": "yes"}) == expected
    with pytest.raises(ValueError, match="fit before"):
        frozen.posterior_samples(.4, {"a": "yes"})


@pytest.mark.parametrize("cls", [NaiveCombiner, CorrelationAwareCombiner])
def test_replay_and_frozen_snapshot(cls):
    model = fitted(cls, 12)
    obs = {"a": "yes", "b": "no"}
    draws = model.posterior_samples(0.4, obs, n=128, seed=22)
    frozen = model.freeze()
    fingerprint = frozen.fingerprint()
    np.testing.assert_array_equal(draws, frozen.posterior_samples(0.4, obs, n=128, seed=22))
    model.fit([], [])
    np.testing.assert_array_equal(draws, frozen.posterior_samples(0.4, obs, n=128, seed=22))
    assert frozen.fingerprint() == fingerprint
    frozen._log_concentrations["a"][0]["yes"] += 0.1
    assert frozen.fingerprint() != fingerprint


def test_dirichlet_draws_match_explicit_conditional_model():
    model = fitted(repeats=12)
    # Pooled yes/no counts are (0,24) for class 0 and (24,0) for class 1.
    expected = np.array([2 * 0.5 / 25, 12 + 2 * 24.5 / 25])
    np.testing.assert_allclose(np.exp(list(model._log_concentrations["a"][0].values())), expected)
    rng = np.random.default_rng(17)
    p0 = rng.dirichlet(expected, size=80)[:, 0]
    p1 = rng.dirichlet(expected[::-1], size=80)[:, 0]
    score = np.log(p1) - np.log(p0)
    truth = 1 / (1 + np.exp(-np.clip(score, -40, 40)))
    np.testing.assert_allclose(model.posterior_samples(0.5, {"a": "yes"}, n=80, seed=17), truth)


def test_sparse_and_empty_calibration_warn_without_disabling_check():
    model = fitted()
    with pytest.warns(UserWarning, match="little calibration data"):
        lo, hi = model.posterior_interval(0.5, {"a": "yes"}, n=64)
    assert 0 <= lo <= hi <= 1
    model.fit([], [])
    with pytest.warns(UserWarning, match="little calibration data"):
        assert np.isfinite(model.posterior_samples(0.5, {"a": "yes"}, n=32)).all()
    np.testing.assert_allclose(model.posterior_samples(0.4, {}, n=3), [0.4] * 3)


def test_interval_matches_samples_and_validation():
    model = fitted(repeats=12)
    draws = model.posterior_samples(0.5, {"a": "yes"}, n=99, seed=3)
    np.testing.assert_allclose(model.posterior_interval(0.5, {"a": "yes"}, level=0.8, n=99, seed=3),
                               np.quantile(draws, [0.1, 0.9]))
    for kwargs in ({"n": 0}, {"n": True}, {"seed": -1}, {"seed": True}, {"domain": "x"}):
        with pytest.raises(ValueError):
            model.posterior_samples(0.5, {}, **kwargs)
    for level in (0, 1, float("nan")):
        with pytest.raises(ValueError):
            model.posterior_interval(0.5, {}, level=level)
    with pytest.raises(ValueError, match="fit before"):
        NaiveCombiner(["a"]).posterior_samples(0.5, {})


def test_correlation_domain_is_fixed_and_respected():
    model = CorrelationAwareCombiner(["a", "b"], {}, outcomes=("yes", "no"),
                                     rho_by_domain={"same": {("a", "b"): 1.0}})
    data = fitted(repeats=12)
    model.fit([{"a": "yes", "b": "yes"}, {"a": "no", "b": "no"}] * 12, [1, 0] * 12)
    obs = {"a": "yes", "b": "yes"}
    local = model.posterior_samples(0.5, obs, n=50, seed=9, domain="same")
    naive = data.posterior_samples(0.5, obs, n=50, seed=9)
    assert np.all(local <= naive)
    with pytest.raises(ValueError, match="unknown correlation domain"):
        model.posterior_samples(0.5, obs, domain="missing")
