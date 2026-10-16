"""Fallback-chain learn-then-test (calibrate(procedure="fallback"), ltt_certified)."""
import math

import numpy as np
import pytest

from solvi.core.calibration import ltt_certified, ltt_threshold
from solvi.core.guarantees.guarantee import calibrate


def _data(n, seed):
    rng = np.random.default_rng(seed)
    s = rng.uniform(size=n)
    wrong = rng.uniform(size=n) < 0.3 * (1 - s) ** 2
    return s, wrong


def test_fallback_zero_is_the_fixed_sequence():
    for seed in range(40):
        s, w = _data(500, seed)
        for e in (0.02, 0.05, 0.1):
            k = np.array([int((s >= t).sum()) for t in np.sort(s)[::8]])
            m = np.array([float(w[s >= t].sum()) for t in np.sort(s)[::8]])
            assert (ltt_certified(k, m, e, 0.1, "fallback", fallback=0.0)
                    == ltt_certified(k, m, e, 0.1, "fixed")).all()


def test_a_failure_does_not_end_the_descent():
    # the second from the top fails; the fixed sequence stops there, the fallback chain goes on at its own level
    k = np.array([3000, 2800, 2600, 2400, 500, 400])
    m = np.array([10, 9, 8, 7, 200, 0])
    fixed = ltt_certified(k, m, 0.05, 0.1, "fixed")
    fb = ltt_certified(k, m, 0.05, 0.1, "fallback")
    assert fixed.tolist() == [False, False, False, False, False, True]
    assert fb.tolist() == [True, True, True, True, False, True]


def test_levels_pass_forward_only_on_certification():
    # with 5 thresholds below the start each has its own level delta/2/5 = 0.01; the start fails, then a threshold
    # whose p-value is between 0.01 and 0.02 passes only after its neighbour passed and handed on its level
    from solvi.core.calibration import _binom_cdf
    e, d = 0.05, 0.1
    k = np.array([2000, 1900, 1800, 1700, 1600, 1000])
    m = np.array([0, 0, 0, 0, 0, 100])                    # the start (top) fails, the others have no error
    # make the 5th from the top (index 1) have p in (0.01, 0.02]
    for x in range(200):
        if 0.01 < _binom_cdf(float(x), 1900, e) <= 0.02:
            m[1] = x
            break
    good = ltt_certified(k, m, e, d, "fallback")
    assert not good[5] and good[2] and good[1]
    m2 = m.copy()
    m2[2] = 200                                            # its neighbour fails → it has only its own 0.01
    assert not ltt_certified(k, m2, e, d, "fallback")[1]


def test_start_uses_the_first_level():
    e, d = 0.05, 0.1
    k0 = math.ceil(math.log(d / 2) / math.log(1 - e))      # 59, not 45
    k = np.array([500, k0, k0 - 1, 45])
    m = np.zeros(4)
    assert ltt_certified(k, m, e, d, "fallback").tolist() == [True, True, False, False]


def test_fallback_keeps_the_promise_under_the_null():
    bad = 0
    for seed in range(300):
        rng = np.random.default_rng(2000 + seed)
        s = rng.uniform(size=400)
        w = rng.uniform(size=400) < 0.08
        bad += math.isfinite(ltt_threshold(s, w, 0.05, 0.1, procedure="fallback"))
    assert bad / 300 <= 0.1 + 0.05


def test_fallback_argument_checked():
    with pytest.raises(ValueError):
        ltt_certified([10], [0], 0.1, 0.1, "fallback", fallback=1.0)


def test_calibrate_fallback_threads_through():
    s, w = _data(2000, 3)
    p = calibrate(s, ~w, max_error=0.05, procedure="fallback")
    assert p.procedure == "fallback" and p.report["procedure"] == "fallback"
    assert "fallback-chain" in p.promise and p.record()["procedure"] == "fallback"
    q = calibrate(s, ~w, max_error=0.05)
    assert p.threshold <= q.threshold
    g = ["a" if x < 0.5 else "b" for x in np.random.default_rng(5).uniform(size=2000)]
    r = calibrate(s, ~w, max_error=0.05, groups=g, min_group=100, procedure="fallback")
    assert set(r.nodes) >= {(), ("a",), ("b",)}
