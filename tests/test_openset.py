"""The open-set gate (solvi.core.guarantees.openset): a threshold sized for a share of inputs from outside the calibration set, a
detector with a bounded false-flag rate, the share estimate after a change, and the gate on a System's question."""
import math

import numpy as np
import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.guarantees.openset import OpenSetGate, leave_out


def _known(rng, n):
    """Known inputs: the signal is informative — P(right) rises with it."""
    s = rng.beta(6, 2, n)
    right = rng.uniform(size=n) < 1 - 0.3 * (1 - s) ** 1.5
    return s, right


def _outside(rng, n):
    """Inputs whose answer is not among the options: any answer is wrong; their signal is lower, but overlaps."""
    return rng.beta(2, 5, n)


def _gate(seed=0, n=3000, **kw):
    rng = np.random.default_rng(seed)
    ks, kr = _known(rng, n)
    return OpenSetGate.calibrate(ks, kr, _outside(rng, n), max_error=0.05, **kw), rng


def test_a_larger_share_of_outside_inputs_needs_a_higher_threshold():
    g, _ = _gate()
    t = [g.thresholds[s] for s in g.shares]
    assert all(a <= b for a, b in zip(t, t[1:]))
    assert t[0] < g.thresholds[0.4] < math.inf


def test_the_threshold_for_a_share_keeps_the_error_on_a_stream_with_that_share():
    for seed in range(5):
        g, rng = _gate(seed)
        for share in (0.2, 0.4):
            t = g.thresholds[share]
            n = 20000
            ks, kr = _known(rng, int(n * (1 - share)))
            us = _outside(rng, int(n * share))
            alone = (ks >= t).sum() + (us >= t).sum()
            wrong = ((ks >= t) & ~kr).sum() + (us >= t).sum()
            assert wrong / alone <= 0.05
        assert (_outside(rng, 5000) >= g.thresholds[0.0]).mean() > 0.01   # the plain threshold lets outside ones in


def test_a_stream_like_the_calibration_examples_is_rarely_flagged():
    g, rng = _gate(1, alpha=0.01, horizon=1000)
    flags = 0
    for _ in range(100):
        s, _ = _known(rng, 1000)
        flags += g.run(s)[1] is not None
    assert flags <= 3


def test_a_shift_is_flagged_and_the_error_after_it_is_kept():
    g, rng = _gate(2, min_share=0.1)
    wrong = alone = 0
    delays = []
    for _ in range(20):
        ks0, kr0 = _known(rng, 1000)
        ks1, kr1 = _known(rng, 600)
        us = _outside(rng, 400)
        sig = np.concatenate([ks1, us])
        ok = np.concatenate([kr1, np.zeros(len(us), bool)])
        order = rng.permutation(len(sig))
        out, flag = g.run(np.concatenate([ks0, sig[order]]))
        assert flag is not None and flag > 1000
        delays.append(flag - 1000)
        after = [a for _, a, _ in out[1000:]]
        alone += sum(after)
        wrong += sum(a and not o for a, o in zip(after, ok[order]))
    assert np.median(delays) < 120
    assert wrong / alone <= 0.05


def test_the_state_estimates_the_share_after_a_flag_and_reset_forgets_it():
    g, rng = _gate(3)
    ks, _ = _known(rng, 300)
    for s in ks:
        g.observe(s)
    assert g.flag_at is None
    for s in _outside(rng, 300):
        g.observe(s)
    st = g.state()
    assert st["flag_at"] and st["share"] > 0.5 and st["level"] is None or st["level"] >= 0.5
    g.reset()
    assert g.state() == {"seen": 0, "level": g.min_share, "cusum": 0.0}


def test_too_few_examples_or_a_signal_that_does_not_tell_outside_from_known_is_refused():
    rng = np.random.default_rng(4)
    ks, kr = _known(rng, 20)
    with pytest.raises(ValueError, match="at least 30"):
        OpenSetGate.calibrate(ks, kr, _outside(rng, 20))
    ks, kr = _known(rng, 500)
    with pytest.raises(ValueError, match="does not tell"):
        OpenSetGate.calibrate(ks, kr, rng.beta(6, 2, 500))


def test_the_gate_on_a_decision_escalates_with_the_reason():
    g, _ = _gate(5)
    d = g.gate(Decision("billing", {"billing": 0.2, "cards": 0.8}, confidence=0.2))
    assert d.escalate.startswith("open-set gate") and d.extra["open_set"]["threshold"] > 0.2
    d = g.gate(Decision("billing", {"billing": 0.99, "cards": 0.01}, confidence=0.99))
    assert d.escalate is None and g.seen == 2


def test_the_gate_on_a_question_records_its_state_and_a_replay_does_not_move_it():
    g, rng = _gate(6)
    cat = Catalog()

    @cat.rule("intent")
    def intent(p: float):
        return Decision("billing" if p > 0.5 else "cards", {"billing": p, "cards": 1 - p}, confidence=max(p, 1 - p))
    s = System(cat, [Question("intent", "which?", Answer.choice(["billing", "cards"]))])
    s.guarantee("intent", promise=g)
    res = [s.ask({"p": float(x)}) for x in rng.uniform(0.5, 1.0, 50)]
    assert g.seen == 50
    rec = next(r for r in res[-1].trace.records if r.name == "guard:intent")
    assert rec.extra["method"] == "open-set" and rec.extra["state"]["seen"] == 49
    assert all(r.trace.replay(s)["ok"] for r in res[:10])
    assert g.seen == 50                                    # replay re-derives the verdict from the recorded threshold


def test_leave_out_gives_known_and_outside_signals():
    opts = ["a", "b", "c", "d", "e", "f"]

    class Stand:
        def __init__(self, kept):
            self.kept = kept

        def __call__(self, x):
            return Decision(x if x in self.kept else self.kept[0], {}, confidence=0.9 if x in self.kept else 0.3)
    ex = [(o, o) for o in opts for _ in range(5)]
    sim = leave_out(ex, Stand, folds=3)
    assert len(sim["novel"]) == 30 and len(sim["known"][0]) == 60
    assert set(sim["novel"]) == {0.3} and all(sim["known"][1])


def test_a_drift_monitor_passed_in_also_starts_the_estimate():
    from solvi.core.guarantees.drift import DriftMonitor
    rng = np.random.default_rng(7)
    ks, kr = _known(rng, 3000)
    g = OpenSetGate.calibrate(ks, kr, _outside(rng, 3000), alpha=1e-4, design=(0.01,), monitor=DriftMonitor(window=50))
    # its own detector tuned to a share of 1% and held to 1e-4 false flags: slow, so the monitor flags first
    for s in _known(rng, 200)[0]:
        g.observe(s)
    for s in _outside(rng, 200):
        g.observe(s)
        if g.flag_at:
            break
    assert g.flag_at and g.why.startswith("DriftMonitor") and g.share > 0.3


def test_a_sudden_large_shift_moves_the_threshold_within_a_few_dozen_decisions_before_any_flag():
    g, rng = _gate(8)
    for s in _known(rng, 500)[0]:
        g.observe(s)
    before = g.level()
    levels = []
    for s in _outside(rng, 40):
        g.observe(s)
        levels.append(g.level())
    assert before is not None and before <= 0.25
    first = next(i for i, lv in enumerate(levels) if lv is None or lv >= 0.4)
    assert first < 25                                       # the short window (25) sees it; the long one (200) would not


def test_the_detector_level_is_simulated_and_its_false_flag_rate_reported():
    g, _ = _gate(9, alpha=0.01, horizon=500)
    det = g.report["detector"]
    assert det["simulated_false_flags"] <= 0.01 and det["h"] == g.h
    assert g.h < math.log(500 / 0.01)                      # below the union bound the gate used before
    with pytest.raises(ValueError, match="below 1e-4"):
        _gate(9, alpha=1e-6)
