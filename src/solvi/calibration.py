"""Is a confidence honest? Reliability, expected calibration error and coverage at a target accuracy — for any model or
question (solvi answers, a decider, a head).

    from solvi.calibration import coverage_at, ece, reliability, threshold_for
    coverage_at(conf, correct, 0.9)     # share of cases you can answer automatically at ≥ 90% accuracy
    threshold_for(conf, correct, 0.9)   # the confidence threshold that gives it (e.g. for Question(min_confidence=...))

`conf` — confidences in [0, 1]; `correct` — whether each answer was right (booleans or 0/1). `evaluate(system, question,
examples)` collects both from a System on labelled examples (abstentions count as not covered)."""
from __future__ import annotations

import numpy as np


def _arrays(conf, correct):
    c, o = np.asarray(conf, float), np.asarray(correct, float)
    if c.shape != o.shape:
        raise ValueError("conf and correct must have the same length")
    return c, o


def reliability(conf, correct, bins=10):
    """Equal-width confidence bins → [{"lo", "hi", "n", "confidence", "accuracy"}] (bins without cases are left out)."""
    c, o = _arrays(conf, correct)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        m = (c >= lo) & (c <= hi) if j == 0 else (c > lo) & (c <= hi)
        if m.any():
            out.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": float(c[m].mean()),
                        "accuracy": float(o[m].mean())})
    return out


def ece(conf, correct, bins=15):
    """Expected calibration error: the case-weighted mean |confidence − accuracy| over equal-width bins (0 = honest)."""
    c, _ = _arrays(conf, correct)
    if len(c) == 0:
        return 0.0
    return float(sum(b["n"] / len(c) * abs(b["confidence"] - b["accuracy"]) for b in reliability(conf, correct, bins)))


def coverage_at(conf, correct, accuracy=0.9):
    """The largest share of cases, taken from the most confident down, whose accuracy is at least `accuracy` (0 if even
    the most confident case alone does not reach it)."""
    c, o = _arrays(conf, correct)
    if len(c) == 0:
        return 0.0
    o = o[np.argsort(-c, kind="stable")]
    cum = np.cumsum(o) / np.arange(1, len(o) + 1)
    good = np.where(cum >= accuracy - 1e-12)[0]
    return float((good.max() + 1) / len(o)) if len(good) else 0.0


def threshold_for(conf, correct, accuracy=0.9):
    """The confidence threshold that achieves coverage_at(conf, correct, accuracy): answer when confidence ≥ it.
    None if no threshold reaches the accuracy."""
    c, o = _arrays(conf, correct)
    cov = coverage_at(c, o, accuracy)
    if cov == 0.0:
        return None
    k = int(round(cov * len(c)))
    return float(np.sort(c)[::-1][k - 1])


def accuracy_at(conf, correct, threshold):
    """Answer only when confidence ≥ threshold → (accuracy of the answered part, coverage)."""
    c, o = _arrays(conf, correct)
    m = c >= threshold
    return (float(o[m].mean()) if m.any() else float("nan")), float(m.mean()) if len(c) else 0.0


def summary(conf, correct, accuracy=0.9, bins=15):
    """{"n", "accuracy", "ece", "coverage_at", "threshold"} in one call."""
    c, o = _arrays(conf, correct)
    return {"n": int(len(c)), "accuracy": float(o.mean()) if len(o) else float("nan"), "ece": ece(c, o, bins),
            "coverage_at": coverage_at(c, o, accuracy), "threshold": threshold_for(c, o, accuracy)}


def evaluate(system, question, examples, accuracy=0.9):
    """Ask a System on labelled examples [(init_state, correct answer)] → summary() over the answered ones plus
    "answered" (share not abstained) and "accuracy_all" (abstentions counted as wrong); also "conf" and "correct" lists."""
    at = system.questions[question].answer
    conf, ok, answered = [], [], 0
    for st, y in examples:
        r = system.ask(st, [question])[question]
        if r.status == "abstain" or r.answer is None:
            continue
        answered += 1
        conf.append(r.confidence)
        ok.append(float(r.answer == at.normalize(y)))
    out = summary(conf, ok, accuracy)
    out.update(answered=answered / max(1, len(examples)), accuracy_all=sum(ok) / max(1, len(examples)), conf=conf, correct=ok)
    return out
