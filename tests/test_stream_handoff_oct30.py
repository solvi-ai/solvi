"""The 2026-10-30 external handoff, stream items (S297–S299). S299: DriftMonitor.reset() re-simulated the CUSUM
threshold on the reference it keeps — the calibration is memoized on the reference's content and the configuration.
S298: a Dispatcher's own stream certificate trips its DriftMonitor on an unchanged stream (the certification starts
System 1 answering alone: "answered alone 0% → 18%, p = 3e-06") — the share answered alone is not tested against a
reference of 0 or 1, and the monitor's reference is re-taken at every threshold move. S297: the auto grid's first
grid_from labels are dropped, and under a Dispatcher each was paid for as an escalated slow-path call — the
Dispatcher warns."""
import math
import time
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import RiskDetector, StreamPromise

from solvi import Answer, Catalog, Question, System
from solvi.core.dispatch import AskPath, Dispatcher
from solvi.core.guarantees.drift import Cusum, DriftMonitor

ALPHA, DELTA = 0.05, 0.05


def _systems():
    """System 1 echoing its input's prediction; the slow path answers the gold label (the worker's wiring)."""
    cat = Catalog()

    @cat.rule("intent")
    def intent(pred: str) -> str:
        return pred

    gcat = Catalog()

    @gcat.rule("intent")
    def intent2(gold: str) -> str:
        return gold

    q = Question("intent", "Which intent?", Answer.choice(["a", "b"]))
    return System(cat, [q]), System(gcat, [q])


def _stream(seed, n):
    """The worker's stationary stream: scores Beta(8, 2), wrong-rate rising steeply as the score falls."""
    rng = np.random.default_rng(seed)
    s = rng.beta(8, 2, n)
    p_wrong = np.clip(0.95 * (1.0 - s), 0.0, 1.0) ** 2
    return s, rng.random(n) < p_wrong


# --- S299: reset() does not re-simulate the sequential threshold on the reference it keeps
def test_reset_does_not_resimulate_the_sequential_threshold(monkeypatch):
    import solvi.core.guarantees.drift as drift

    calls = []
    real = Cusum.calibrate.__func__

    def counting(cls, *a, **kw):
        calls.append(1)
        return real(cls, *a, **kw)

    monkeypatch.setattr(drift.Cusum, "calibrate", classmethod(counting))
    vals = np.random.default_rng(7).beta(8, 2, 200)
    m = DriftMonitor(window=30, alpha=ALPHA, horizon=150)
    for t in range(30):
        m.observe({"value": "a", "confidence": float(vals[t]), "alone": bool(vals[t] > 0.5)})
    assert len(calls) == 1                                  # the reference filled: one simulation
    h = m.cusum.h
    t0 = time.perf_counter()
    for _ in range(3):
        m.reset()
    assert len(calls) == 1                                  # reset keeps the reference: memoized, not resimulated
    assert m.cusum.h == h                                   # the same threshold, bit for bit
    assert time.perf_counter() - t0 < 0.3                   # and fast (was ~1 s per reset, a full simulation)
    m.set_reference([{"value": "a", "confidence": float(v), "alone": bool(v > 0.4)}
                     for v in np.random.default_rng(8).beta(4, 2, 30)])
    assert len(calls) == 2                                  # another reference's content: resimulated
    m.reset()
    assert len(calls) == 2                                  # ... and memoized again from there


# --- S298: the share answered alone against a degenerate reference, and the reference re-take
def test_a_degenerate_answered_reference_is_not_tested_and_says_why():
    """The worker's mechanism, standalone: a reference of decisions none of which was answered alone (the certificate
    not yet in force) has share 0 — every window that moved at all was 'significant' (0% → 100%, p ~ 1e-10)."""
    m = DriftMonitor(window=20, alpha=ALPHA, horizon=100)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        for _ in range(20):
            m.observe({"value": "a", "confidence": 0.8, "alone": False})     # the reference: nothing alone
        reps = [m.observe({"value": "a", "confidence": 0.8, "alone": True})  # the certificate certified: everything is
                for _ in range(60)]
    assert all("answered" not in r["flags"] and not r["drift"] for r in reps)
    assert "answered" in reps[-1]["not_tested"] and "0%" in reps[-1]["not_tested"]["answered"]
    assert sum("share answered alone" in str(w.message) for w in got) == 1   # warned once, not per decision


def test_adapted_re_takes_the_reference_from_the_stream():
    m = DriftMonitor(window=20, alpha=ALPHA, horizon=100)
    for t in range(20):
        m.observe({"value": "a", "confidence": 0.5 + 0.1 * (t % 3), "alone": t % 2 == 0})
    assert m.calibrated
    m.adapted()                                             # what the monitor reads changed on its own
    assert not m.calibrated
    assert m.observe({"value": "a", "confidence": 0.6, "alone": True})["phase"] == "reference"
    for t in range(19):
        m.observe({"value": "a", "confidence": 0.5 + 0.1 * (t % 3), "alone": t % 3 == 0})
    assert m.calibrated and m._ref["answered"] == pytest.approx(8 / 20)     # 7 + the one above the call
    for t in range(20):                                     # the window fills: min_n reached, the tests run again
        rep = m.observe({"value": "a", "confidence": 0.5 + 0.1 * (t % 3), "alone": t % 3 == 0})
    assert rep["phase"] == "monitor" and "answered" in rep["tests"]         # tested again, against the new reference


def test_the_certificates_own_certification_is_not_monitor_drift():
    """S298 part A: Dispatcher(monitor=DriftMonitor, promise=sp) on an UNCHANGED stream — first certification at ~284,
    then loosening; the alone flag moves with the certificate, not the stream, so the monitor's reference is re-taken
    at every threshold move and no false drift is flagged (was: 'answered alone 0% → 18% (p = 3.3e-06)' at 313)."""
    from solvi.core.guarantees.guarantee import GuaranteeWarning

    s, wrong = _stream(0, 700)
    s1, gold = _systems()
    sp = StreamPromise(max_risk=ALPHA, delta=DELTA)
    s1.guarantee("intent", promise=sp, signal="msp")
    mon = DriftMonitor(window=50, alpha=ALPHA, horizon=200)
    d = Dispatcher(s1, AskPath(gold), question="intent", audit="adaptive", promise=sp, detector=RiskDetector(ALPHA),
                   on_drift="recertify", probation=300, secret=b"S298", monitor=mon)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        for t in range(600):
            d.ask({"pred": "a" if not wrong[t] else "b", "msp": float(s[t]), "gold": "a"})
            assert d.drift is None                          # an unchanged stream flags nothing, certification included
    kinds = [e["event"] for e in sp.events]
    assert "certified" in kinds and "loosened" in kinds      # the threshold moved: the re-take path ran, and ran again
    assert not any(issubclass(w.category, GuaranteeWarning) for w in got)
    assert any(rep["phase"] == "reference" for rep in mon.history)          # the monitor really re-took its reference
    assert not math.isinf(sp.threshold)                     # and the certificate itself certified


def test_a_real_shift_is_still_flagged_with_the_promise_composed():
    """The re-take must not turn the monitor off: the scores fall at decision 1000 (the share answered alone falls with
    them, 56% → 17%), and with no threshold move after the shift to absorb it the monitor flags it soon after. (When
    the certificate does loosen right after a shift, the new reference starts inside the moved regime and the monitor
    says nothing of it — the label-side detector is the alarm then; that is the price S298's fix accepts and the
    Dispatcher docstring names.)"""
    rng = np.random.default_rng(11)
    n, shift_at = 1600, 1000
    s = np.concatenate([rng.beta(8, 2, shift_at), rng.beta(3, 2, n - shift_at)])
    p_wrong = np.clip(0.95 * (1.0 - s), 0.0, 1.0) ** 2
    wrong = rng.random(n) < p_wrong
    s1, gold = _systems()
    sp = StreamPromise(max_risk=ALPHA, delta=DELTA)
    s1.guarantee("intent", promise=sp, signal="msp")
    mon = DriftMonitor(window=100, alpha=0.01, horizon=500)
    d = Dispatcher(s1, AskPath(gold), question="intent", audit="adaptive", promise=sp, detector=RiskDetector(ALPHA),
                   on_drift="recertify", probation=300, secret=b"S298", monitor=mon)
    flagged = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for t in range(n):
            if d.ask({"pred": "a" if not wrong[t] else "b", "msp": float(s[t]), "gold": "a"}).drift:
                flagged = t + 1
                break
    assert flagged is not None and shift_at < flagged <= shift_at + 250


# --- S297: the auto grid's paid-and-dropped labels say their price
def test_the_auto_grid_warns_about_paid_and_dropped_labels_with_a_slow_path():
    s1, gold = _systems()
    sp = StreamPromise(max_risk=ALPHA, delta=DELTA)          # the default grid="auto"
    s1.guarantee("intent", promise=sp, signal="msp")
    with pytest.warns(UserWarning, match="grid_from"):
        Dispatcher(s1, AskPath(gold), question="intent", audit="adaptive", promise=sp, secret=b"S297")


def test_no_auto_grid_warning_for_a_fixed_grid_or_without_a_slow_path():
    s1, gold = _systems()
    sp = StreamPromise(max_risk=ALPHA, delta=DELTA, grid="fixed")
    s1.guarantee("intent", promise=sp, signal="msp")
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        Dispatcher(s1, AskPath(gold), question="intent", audit="adaptive", promise=sp, secret=b"S297")
    assert not any("grid_from" in str(w.message) for w in got)   # a fixed grid drops nothing
    s2, _ = _systems()
    sp2 = StreamPromise(max_risk=ALPHA, delta=DELTA)         # auto, but no slow path: the labels are not paid for
    s2.guarantee("intent", promise=sp2, signal="msp")
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        Dispatcher(s2, None, question="intent", audit="adaptive", promise=sp2, secret=b"S297")
    assert not any("grid_from" in str(w.message) for w in got)
