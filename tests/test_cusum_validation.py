"""Calibration must not manufacture a threshold from invalid configuration or null draws."""
import numpy as np
import pytest

from solvi.core.guarantees.drift import Cusum, NumericDrift


@pytest.mark.parametrize("field,bad", [
    ("horizon", 0), ("horizon", -1), ("horizon", 1.5), ("horizon", True),
    ("sims", 0), ("sims", -1), ("sims", 2.5), ("sims", True),
])
def test_invalid_counts_refused_before_calling_the_null_factory(field, bad):
    calls = []

    def factory(rng, sims):
        calls.append(sims)
        return lambda: np.zeros((sims, 2))

    options = {"horizon": 1, "sims": 4, field: bad}
    with pytest.raises(ValueError, match=field):
        Cusum.calibrate(["up", "down"], factory, **options)
    assert not calls


@pytest.mark.parametrize("shape", [(), (2,), (4, 1), (3, 2)])
def test_null_draws_must_have_one_value_per_simulation_and_channel(shape):
    with pytest.raises(ValueError, match="shape"):
        Cusum.calibrate(["up", "down"], lambda rng, sims: lambda: np.zeros(shape), horizon=1, sims=4)


@pytest.mark.parametrize("value,horizon", [(np.inf, 1), (1e308, 2), (np.finfo(float).max, 1)])
def test_nonfinite_null_or_threshold_is_refused(value, horizon):
    with pytest.raises(ValueError, match="finite"):
        Cusum.calibrate(["up", "down"], lambda rng, sims: lambda: np.full((sims, 2), value),
                        horizon=horizon, sims=4)


def test_negative_infinity_is_a_valid_likelihood_ratio_reset():
    draws = iter([np.tile([2.0, 5.0], (4, 1)), np.full((4, 2), -np.inf)])
    monitor = Cusum.calibrate(["up", "down"], lambda rng, sims: lambda: next(draws),
                              alpha=0.25, horizon=2, sims=4)
    assert monitor.h == np.nextafter(5.0, np.inf) and monitor.rate == 0
    monitor.step([2.0, 5.0])
    monitor.step([-np.inf, -np.inf])
    np.testing.assert_array_equal(monitor.S, [0.0, 0.0])
    np.testing.assert_array_equal(monitor.since, [0, 0])


def test_nan_null_increments_are_still_missing_and_small_explicit_budget_is_kept():
    draws = np.array([[0, 1], [-1, np.nan], [2, 0], [3, 1]], dtype=float)
    monitor = Cusum.calibrate(["up", "down"], lambda rng, sims: lambda: draws,
                              alpha=0.25, horizon=np.int64(1), sims=np.int64(4))
    assert monitor.h == np.nextafter(2.0, np.inf)
    assert monitor.rate == 0.25 and monitor.sims == 4 and monitor.horizon == 1


@pytest.mark.parametrize("increments", [2.0, [2.0], [[2.0, 0.0]], [2.0, 0.0, -1.0]])
def test_bad_observation_shape_does_not_change_either_channel(increments):
    monitor = Cusum(["up", "down"], 5)
    monitor.step([2, 1])
    state, since = monitor.S.copy(), monitor.since.copy()
    with pytest.raises(ValueError, match="increment"):
        monitor.step(increments)
    np.testing.assert_array_equal(monitor.S, state)
    np.testing.assert_array_equal(monitor.since, since)


def test_empty_channel_set_is_refused():
    with pytest.raises(ValueError, match="channel"):
        Cusum([], 5)


def test_single_channel_scalar_observations_remain_supported():
    monitor = Cusum(["only"], 5)
    assert monitor.step(2.0) == 0
    assert monitor.top() == ("only", 2.0, 1)
    monitor.step(np.nan)
    assert monitor.top() == ("only", 2.0, 1)


@pytest.mark.parametrize("options", [
    {"horizon": 0}, {"horizon": 1.5}, {"ref_n": 2.5}, {"settle": -1},
    {"settle": 0.5}, {"allowance": np.nan}, {"allowance": np.inf}, {"alpha": np.nan},
])
def test_numeric_monitor_rejects_invalid_configuration_immediately(options):
    with pytest.raises(ValueError):
        NumericDrift(**options)


@pytest.mark.parametrize("options", [
    {"ref_n": 2.5}, {"ref_n": True}, {"allowance": np.nan}, {"allowance": np.inf},
])
def test_builtin_null_rejects_invalid_reference_size_and_allowance(options):
    with pytest.raises(ValueError):
        Cusum.calibrate(["up", "down"], horizon=1, sims=4, **options)
