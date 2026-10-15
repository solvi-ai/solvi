"""A guarantee says when it cannot be kept: conformal risk control whose only threshold is inf (everything escalates)
gives a why and a GuaranteeWarning, as does a calibration too small for the promise."""
import warnings

import numpy as np
import pytest

from solvi.core.guarantees.guarantee import GuaranteeWarning, calibrate


def _weak(n=30, seed=0):
    rng = np.random.default_rng(seed)
    return list(rng.random(n)), list(rng.random(n) < 0.5)


def test_crc_with_no_finite_threshold_says_why_and_warns():
    s, ok = _weak()
    with pytest.warns(GuaranteeWarning, match="everything escalates"):
        p = calibrate(s, ok, max_risk=0.05, weak="warn")
    assert p.threshold == float("inf")
    assert p.why and "no finite threshold" in p.why and "30 examples" in p.why
    assert p.report["why"] == p.why


def test_a_kept_promise_on_enough_examples_is_silent():
    rng = np.random.default_rng(1)
    s = rng.random(400)
    ok = s > 0.2
    with warnings.catch_warnings():
        warnings.simplefilter("error", GuaranteeWarning)
        p = calibrate(list(s), list(ok), max_risk=0.05)
    assert np.isfinite(p.threshold) and p.why is None


def test_crc_on_too_few_examples_warns():
    # 25 examples at risk 0.05: the promise allows no wrong answer among the answered at all
    s = list(np.linspace(0, 1, 25))
    ok = [True] * 25
    with pytest.warns(GuaranteeWarning, match="too few"):
        p = calibrate(s, ok, max_risk=0.05)
    assert np.isfinite(p.threshold)


def test_ltt_with_no_threshold_warns():
    s, ok = _weak(100, 2)
    with pytest.warns(GuaranteeWarning, match="everything escalates"):
        p = calibrate(s, ok, max_error=0.05, weak="warn")
    assert p.threshold == float("inf") and p.why
