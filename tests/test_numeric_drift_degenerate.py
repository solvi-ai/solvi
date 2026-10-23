"""NumericDrift on a degenerate reference (mostly one value, rare spikes) says so: the false-alarm bound assumes roughly
Gaussian values, and on such a stream an unchanged stream is flagged far more often than alpha."""
import random
import warnings

import pytest

from solvi.core.guarantees.drift import NumericDrift


def _spiky(r, n):
    return [r.choice([0.2, 0.35, 0.5]) if r.random() < 0.05 else 0.0 for _ in range(n)]


def test_a_mostly_zero_reference_warns_and_the_alarm_says_so():
    r = random.Random(0)
    with pytest.warns(UserWarning, match="degenerate reference"):
        m = NumericDrift(alpha=0.01, horizon=500).set_reference(_spiky(r, 200))
    assert m.degenerate and "values are 0" in m.degenerate
    for _ in range(20):
        if m.observe(5.0)["drift"]:
            break
    assert m.flagged and "degenerate reference" in m.observe(5.0)["why"][0]


def test_a_reference_collected_from_the_stream_warns_too():
    r = random.Random(1)
    m = NumericDrift(alpha=0.01, horizon=500, ref_n=20)
    with pytest.warns(UserWarning, match="degenerate reference"):
        for x in _spiky(r, 20):
            m.observe(x)
    assert m.degenerate


def test_a_gaussian_reference_does_not_warn():
    r = random.Random(2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = NumericDrift().set_reference([r.gauss(0, 1) for _ in range(200)])
    assert m.degenerate is None
    m.adapted()
    assert m.degenerate is None
