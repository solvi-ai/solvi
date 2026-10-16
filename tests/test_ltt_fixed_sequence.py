"""Fixed-sequence learn-then-test (calibrate(procedure="fixed"), ltt_certified)."""
import math

import numpy as np
import pytest

from solvi.core.calibration import _binom_cdf, ltt_certified, ltt_threshold
from solvi.core.guarantees.guarantee import calibrate


def _data(n, seed):
    rng = np.random.default_rng(seed)
    s = rng.uniform(size=n)
    wrong = rng.uniform(size=n) < 0.3 * (1 - s) ** 2      # the error falls with the score
    return s, wrong


def test_bonferroni_default_unchanged():
    s, w = _data(2000, 1)
    # the default is the same Bonferroni rule as before
    t = ltt_threshold(s, w, 0.05, 0.1)
    grid = np.unique(s)
    grid = np.unique(grid[np.round(np.linspace(0, len(grid) - 1, 64)).astype(int)])
    ref = math.inf
    for g in sorted(grid):
        a = s >= g
        if a.sum() and _binom_cdf(float(w[a].sum()), int(a.sum()), 0.05) <= 0.1 / len(grid):
            ref = float(g)
            break
    assert t == ref


def test_fixed_stops_at_first_failure():
    # thresholds ascending: k answered, m wrong; the 3rd from the top fails, so lower ones are not certified even if
    # they would pass on their own
    k = np.array([1000, 900, 800, 700, 600])
    m = np.array([10, 9, 200, 7, 6])
    good = ltt_certified(k, m, 0.05, 0.1, "fixed")
    assert good.tolist() == [False, False, False, True, True]
    assert ltt_certified(k, m, 0.05, 0.1, "bonferroni").tolist() == [True, True, False, True, True]


def test_fixed_starts_where_a_test_can_pass():
    # the top thresholds have too few cases to pass even with no error: they are skipped, not a stop
    k0 = math.ceil(math.log(0.1) / math.log(0.95))       # 45
    k = np.array([500, 200, k0, k0 - 1, 5])
    m = np.array([5, 2, 0, 0, 0])
    good = ltt_certified(k, m, 0.05, 0.1, "fixed")
    assert good.tolist() == [True, True, True, False, False]


def test_fixed_answers_at_least_as_much_when_error_is_monotone():
    s, w = _data(3000, 2)
    tb = ltt_threshold(s, w, 0.05, 0.1)
    tf = ltt_threshold(s, w, 0.05, 0.1, procedure="fixed")
    assert tf <= tb


def test_fixed_keeps_the_promise_under_the_null():
    # every threshold has error exactly 0.08 > 0.05: certifying anything is a false certificate; ≤ delta of the time
    bad = 0
    for seed in range(300):
        rng = np.random.default_rng(1000 + seed)
        s = rng.uniform(size=400)
        w = rng.uniform(size=400) < 0.08
        bad += math.isfinite(ltt_threshold(s, w, 0.05, 0.1, procedure="fixed"))
    assert bad / 300 <= 0.1 + 0.05


def test_calibrate_procedure_threads_through():
    s, w = _data(2000, 3)
    p = calibrate(s, ~w, max_error=0.05, procedure="fixed")
    assert p.procedure == "fixed" and p.report["procedure"] == "fixed"
    assert "fixed-sequence" in p.promise and p.record()["procedure"] == "fixed"
    q = calibrate(s, ~w, max_error=0.05)
    assert "procedure" not in q.record() and q.report["procedure"] == "bonferroni"
    assert p.threshold <= q.threshold
    with pytest.raises(ValueError):
        calibrate(s, ~w, max_error=0.05, procedure="holm")


def test_calibrate_fixed_with_groups():
    s, w = _data(3000, 4)
    g = ["a" if x < 0.5 else "b" for x in np.random.default_rng(5).uniform(size=3000)]
    p = calibrate(s, ~w, max_error=0.05, groups=g, min_group=100, procedure="fixed")
    assert set(p.nodes) >= {(), ("a",), ("b",)}
