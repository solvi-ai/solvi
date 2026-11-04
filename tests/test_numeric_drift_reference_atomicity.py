"""A refused replacement reference must leave the calibrated monitor usable."""
import pickle
import warnings

import pytest

from solvi.core.guarantees.drift import NumericDrift


def calibrated():
    return NumericDrift(horizon=10).set_reference(list(range(20)))


def test_warning_as_error_preserves_the_previous_reference_and_detector():
    monitor = calibrated()
    monitor.observe(15)
    before = pickle.dumps(monitor)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="degenerate reference"):
            monitor.set_reference([0] * 20)
    assert pickle.dumps(monitor) == before


def test_unrepresentable_reference_statistics_refused_without_changing_monitor():
    monitor = calibrated()
    before = pickle.dumps(monitor)
    with pytest.raises(ValueError, match="finite"):
        monitor.set_reference([1e308, -1e308] * 10)
    assert pickle.dumps(monitor) == before


def test_failed_calibration_preserves_the_previous_reference(monkeypatch):
    from solvi.core.guarantees.drift import Cusum

    monitor = calibrated()
    monitor.observe(15)
    before = pickle.dumps(monitor)

    def unavailable(*args, **kwargs):
        raise RuntimeError("calibration unavailable")

    monkeypatch.setattr(Cusum, "calibrate", unavailable)
    with pytest.raises(RuntimeError, match="calibration unavailable"):
        monitor.set_reference(list(range(100, 120)))
    assert pickle.dumps(monitor) == before


@pytest.mark.parametrize("minimum", [0, -1, float("nan"), float("inf")])
def test_invalid_standardization_floor_refused_at_construction(minimum):
    with pytest.raises(ValueError, match="min_sd"):
        NumericDrift(min_sd=minimum)
