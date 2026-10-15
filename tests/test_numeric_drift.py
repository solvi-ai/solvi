"""NumericDrift: a Monitor over any numeric stream (residuals, latencies), its threshold set by Cusum.calibrate with the
reference's own sampling error (ref_n=); the reference is given or the first ref_n values; reset() keeps it."""
import numpy as np
import pytest

from solvi.core.guarantees.drift import Cusum, NumericDrift
from solvi.core.guarantees.monitor import Monitor
from solvi.testing.conformance import check_monitor


def _stream(n, mean=0.0, seed=0):
    return list(np.random.default_rng(seed).normal(mean, 1.0, n))


def test_cusum_calibrate_with_ref_n_raises_h_for_a_noisy_reference():
    known = Cusum.calibrate(["up", "down"], alpha=0.05, horizon=200, seed=0)
    small = Cusum.calibrate(["up", "down"], alpha=0.05, horizon=200, ref_n=20, seed=0)
    assert small.h > known.h > 0
    assert small.rate <= 0.05 + 1e-9
    with pytest.raises(ValueError):
        Cusum.calibrate(["a", "b", "c"], alpha=0.05, horizon=10)
    with pytest.raises(ValueError):
        Cusum.calibrate(["up", "down"], lambda rng, sims: (lambda: np.zeros((sims, 2))), alpha=0.05, horizon=10, ref_n=20)


def test_numeric_drift_is_a_conforming_monitor():
    def make():
        return NumericDrift(alpha=0.01, horizon=500, ref_n=50).set_reference(_stream(200, seed=1))
    assert isinstance(make(), Monitor)
    out = check_monitor(make, _stream(300, seed=2), changed=_stream(100, mean=1.5, seed=3))
    assert out["changed"] is True


def test_a_shift_is_flagged_soon_with_a_reason_and_latched_until_reset():
    m = NumericDrift(alpha=0.01, horizon=500).set_reference(_stream(200, seed=1))
    reps = [m.observe(x) for x in _stream(100, seed=2) + _stream(60, mean=2.0, seed=3)]
    at = next(i for i, r in enumerate(reps) if r["drift"])
    assert 100 <= at < 120
    assert "up" in reps[at]["why"][0]
    assert all(r["drift"] for r in reps[at:]) and m.flagged
    m.reset()
    assert not m.flagged and m.mean is not None              # the reference is kept


def test_the_reference_can_come_from_the_stream_and_adapted_takes_a_new_one():
    m = NumericDrift(alpha=0.01, horizon=500, ref_n=30)
    for x in _stream(30, mean=5.0, seed=4):
        assert not m.observe(x)["drift"]
    assert m.mean == pytest.approx(5.0, abs=0.6)
    m.adapted()
    for x in _stream(30, mean=9.0, seed=5):
        m.observe(x)
    assert m.mean == pytest.approx(9.0, abs=0.6)


def test_a_short_reference_warns_and_values_come_through_value():
    with pytest.warns(UserWarning, match="false-alarm bound"):
        NumericDrift().set_reference([1.0, 2.0, 3.0])
    m = NumericDrift(value=lambda d: d["latency"]).set_reference(_stream(100, seed=6))
    assert m.observe({"latency": 0.1})["drift"] is False
    with pytest.raises(TypeError):
        NumericDrift().set_reference(_stream(100, seed=6)).observe({"latency": 0.1})
