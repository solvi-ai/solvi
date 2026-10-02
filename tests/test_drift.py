"""solvi.drift: a window of decisions against a reference — what moved (the share answered alone, the answers, the
confidence, with labels the accuracy and the calibration), flagged only when significant and large enough."""
import random

import numpy as np
import pytest

from solvi import Catalog, Decision, System
from solvi.decide import DecideModel
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


def iid(rng, n, k):
    """Independent decisions over k answers; nothing changes."""
    out = []
    for _ in range(n):
        conf = float(rng.beta(5, 2))
        out.append({"value": int(rng.integers(k)), "confidence": conf, "alone": conf > 0.5, "act": float(rng.beta(8, 2))})
    return out


def test_stationary_streams_are_not_flagged_with_the_documented_defaults():
    """The promise of the module docstring: on an unchanged stream the chance of a flag within `horizon` (1,000)
    decisions is at most alpha (1%). Every test used to run at alpha itself at every decision: of 20 independent streams
    over 57 answers 8–10 were flagged within 1,000 decisions (most by "answers": a chi-square test on counts of 1–2),
    and about as many over 3 answers."""
    rng = np.random.default_rng(0)
    flagged = []
    for k, ref in ((57, 1500), (57, 0), (3, 1500), (3, 0)):
        for _ in range(6):
            mon = DriftMonitor(window=100)
            if ref:
                mon.calibrate(iid(rng, ref, k))
            reps = [mon.observe(d) for d in iid(rng, 1000 + (0 if ref else 100), k)]
            flagged += [(k, ref, r["flags"]) for r in reps if r["drift"]][:1]
    assert flagged == []
    t = reps[-1]["tests"]
    # the window tests share three quarters of alpha (the CUSUMs have the rest)
    assert t["answers"]["level"] == pytest.approx(0.0075 / (4 * 1000))
    assert set(t) == {"answered", "answers", "confidence", "act", "sequential"} and t["sequential"]["S"] < t["sequential"]["h"]


def test_answers_too_rare_for_the_window_are_pooled_or_the_signal_is_said_to_be_untested():
    rng = np.random.default_rng(1)
    mon = DriftMonitor(window=100).calibrate(iid(rng, 1500, 57))         # each answer is expected 1.75 times in a window
    for d in iid(rng, 100, 57):
        rep = mon.observe(d)
    assert "answers" not in rep["tests"] and "fewer than two answers are expected at least 5 times" in rep["not_tested"]["answers"]
    assert {"answered", "confidence", "act"} <= set(rep["tests"]) and rep["tests"]["act"]["level"] == pytest.approx(0.0075 / 3000)
    mon = DriftMonitor(window=100).calibrate(iid(rng, 1500, 12))         # 8.3 each: tested, nothing pooled
    for d in iid(rng, 100, 12):
        rep = mon.observe(d)
    assert rep["tests"]["answers"]["df"] == 11 and rep["tests"]["answers"]["pooled"] == 0 and not rep["not_tested"]
    skew = [{"value": v, "confidence": 0.9} for v in ["a"] * 60 + ["b"] * 30 + list("cdefghijkl")]   # ten rare answers
    mon = DriftMonitor(window=100).calibrate(skew * 3)
    for d in skew:
        rep = mon.observe(d)
    assert rep["tests"]["answers"]["df"] == 2 and rep["tests"]["answers"]["pooled"] == 10 and not rep["drift"]
    for d in [{"value": "z", "confidence": 0.9}] * 100:                  # a real change is still seen
        rep = mon.observe(d)
    assert rep["flags"] == ["answers"] and "'z' grew most" in rep["why"][0]
    with pytest.raises(ValueError):
        DriftMonitor(alpha=0)


class ActScorer:
    model_id = "stand-in"

    def logits(self, items):
        return [{"logits": np.array([2.0 if o in it.text else 0.0 for o in it.options]), "act": 1.5} for it in items]


def test_a_response_gives_the_act_probability_and_a_bare_result_says_that_it_does_not():
    m = DecideModel(ActScorer(), meta={"format": "stand-in", "temperature": 1.0, "act": {}})
    part = m.decision("team", "Which team?", "text", ["billing", "shipping"])
    cat = Catalog()
    res = System(cat, [part.question(cat)]).ask({"text": "a billing question"})
    act = Observation.of(part("a billing question")).act
    assert act is not None and Observation.of(res, question="team").act == pytest.approx(act)
    assert Observation.of(res).act == pytest.approx(act)                # the response's only question
    assert Observation.of(res["team"]).act is None                      # a Result does not carry it …
    mon = DriftMonitor(window=20)
    with pytest.warns(UserWarning, match="carries no act probability"):  # … and the monitor says so, once
        mon.observe(res["team"])
    two = DriftMonitor(window=20).calibrate([part("a billing question")] * 20)
    for _ in range(20):
        rep = two.observe(res, question="team")
    assert "act" in rep["tests"] and not rep["not_tested"]
    for _ in range(20):
        rep = two.observe({"value": "billing", "confidence": 0.9})
    assert "act" not in rep["tests"] and "the window holds no act probability" in rep["not_tested"]["act"]


def test_the_sequential_test_flags_a_fall_of_the_share_answered_alone_sooner_than_the_window_tests():
    """A CUSUM sums the evidence decision by decision; a window test repeated at every decision is held to a union bound
    over the looks. On Banking77 the window tests alone never flagged 20 unseen intents for two deciders; with the
    CUSUMs they flagged within 52-72 requests."""
    rng = np.random.default_rng(3)
    first = {}
    for seq in (True, False):
        mon = DriftMonitor(window=100, sequential=seq)
        pre = iid(rng, 600, 3)
        post = [dict(d, alone=d["confidence"] > 0.89) for d in iid(rng, 300, 3)]      # answered alone ~89% → ~13%
        reps = [mon.observe(d) for d in pre + post]
        assert not any(r["drift"] for r in reps[:600])
        first[seq] = next(i for i, r in enumerate(reps) if r["drift"]) - 600
    assert first[True] < 40 and first[True] < first[False]
    rep = DriftMonitor(window=100).calibrate(iid(rng, 200, 3)).observe(iid(rng, 1, 3)[0])
    assert rep["tests"]["sequential"]["h"] > 0
    with pytest.raises(ValueError, match="4e-4"):
        DriftMonitor(alpha=1e-5)


def test_a_cusum_level_is_set_by_simulation_and_its_simulated_rate_kept():
    from solvi.drift import Cusum

    def null(rng, sims):
        return lambda: rng.standard_normal((sims, 2)) - 0.5
    c = Cusum.calibrate(["a", "b"], null, alpha=0.01, horizon=300, seed=1)
    assert c.rate <= 0.01 and c.h > 0 and c.sims == 2000
    c.step([3.0, -1.0])
    assert c.top() == ("a", 3.0, 1) and c.S[1] == 0.0
    with pytest.raises(ValueError, match="1e-4"):
        Cusum.calibrate(["a"], null, alpha=1e-6)
