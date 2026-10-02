"""solvi.drift: a window of decisions against a reference — what moved (the share answered alone, the answers, the
confidence, with labels the accuracy and the calibration), flagged only when significant and large enough."""
import random

import pytest

from solvi import Decision
from solvi.drift import DriftMonitor, Observation, window_stats

CATS = ["billing", "shipping", "technical"]


def stream(n, seed, weights=(1, 1, 1), conf=0.9, wrong=0.05, escalate=0.05):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        y = rng.choices(CATS, weights)[0]
        v = y if rng.random() > wrong else rng.choice([c for c in CATS if c != y])
        d = Decision(v, {v: conf}, confidence=min(1.0, max(0.0, rng.gauss(conf, 0.05))), extra={"act": rng.gauss(conf, 0.05)})
        if rng.random() < escalate:
            d.escalate = "model escalated"
        out.append((d, y))
    return out


def test_a_stationary_stream_is_not_flagged_and_a_shift_is():
    mon = DriftMonitor(window=100)
    for d, _ in stream(100, 1):
        assert mon.observe(d)["phase"] == "reference"
    assert mon.calibrated
    reps = [mon.observe(d) for d, _ in stream(300, 2)]
    assert not any(r["drift"] for r in reps) and reps[-1]["phase"] == "monitor"
    assert "the window holds 10 decisions" in reps[9]["why"][0]                 # too few to say anything
    shifted = [mon.observe(d) for d, _ in stream(150, 3, weights=(1, 8, 1))]     # the mix of the answers changes
    first = next(i for i, r in enumerate(shifted) if r["drift"])
    assert first < 100 and shifted[first]["flags"] == ["answers"] and "'shipping' grew most" in shifted[first]["why"][0]
    assert shifted[-1]["tests"]["answers"]["tv"] > 0.15 and mon.seen == 550 and len(mon.history) == 550


def test_each_signal_needs_significance_and_a_minimum_effect():
    ref = stream(200, 1)
    mon = DriftMonitor(window=200).calibrate([d for d, _ in ref], [y for _, y in ref])
    for d, y in stream(200, 4, conf=0.6, wrong=0.3, escalate=0.4):                # less sure, wrong and escalating more
        rep = mon.observe(d, label=y)
    assert rep["drift"] and {"answered", "confidence", "act", "accuracy"} <= set(rep["flags"])
    assert rep["tests"]["accuracy"]["reference"] > rep["tests"]["accuracy"]["window"]
    small = DriftMonitor(window=200, min_shift=0.05).calibrate([d for d, _ in ref])
    for d, _ in stream(200, 5, conf=0.88):                                        # significant on 200, but 0.02: not drift
        rep = small.observe(d)
    assert not rep["drift"] and abs(rep["tests"]["confidence"]["window"] - rep["tests"]["confidence"]["reference"]) < 0.05
    both = DriftMonitor(window=100, min_signals=2).calibrate([d for d, _ in ref])
    for d, _ in stream(100, 6, weights=(1, 8, 1)):
        rep = both.observe(d)
    assert rep["flags"] == ["answers"] and not rep["drift"]                       # one signal where two are asked for


def test_observations_from_decisions_results_and_dicts():
    d = Decision("billing", {"billing": 0.8}, extra={"act": 0.7})
    o = Observation.of(d, "billing")
    assert (o.value, o.confidence, o.alone, o.act, o.correct) == ("billing", 0.8, True, 0.7, True)
    d.escalate = "act 0.3 < 0.5"
    assert Observation.of(d).alone is False and Observation.of(d).correct is None
    o = Observation.of({"answer": "x", "confidence": 0.5, "status": "abstain"}, "y")
    assert (o.value, o.alone, o.correct) == ("x", False, False)
    s = window_stats([Observation("a", 0.9, True, None, True), Observation("b", 0.6, False, None, False)])
    assert s["answered"] == 0.5 and s["answers"] == {"'a'": 1, "'b'": 1} and s["accuracy"] == 0.5 and "act" not in s
    with pytest.raises(ValueError):
        DriftMonitor(window=10).calibrate([d], [None, None])
    with pytest.raises(ValueError):
        DriftMonitor(window=1)
