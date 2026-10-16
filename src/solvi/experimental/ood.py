"""A shift watch next to System.calibrate's default Platt scaling (experimental): `shift(batch)` for a batch of answers
unlike the calibration answers, and "out of domain" in `why` for one such answer. A packaged known method, not a new
detector: the batch test is the black-box shift detection of Rabanser, Günnemann and Lipton ("Failing Loudly", NeurIPS
2019) on three features of the decider's outputs; the per-answer note is a max-softmax style score (Hendrycks and
Gimpel, ICLR 2017) as a Mahalanobis distance in that space.

    from solvi.experimental.ood import ShiftWatch

    system.calibrate("intent", examples, watch=ShiftWatch())    # Platt as before; the watch is fitted on the same answers
    r = system.ask(state)["intent"]
    r.why                              # "...; out of domain: features outside the calibration data (...)"
    r.extra["out_of_domain"]           # {"raw": the confidence before Platt, "notes": [...]}
    system.shift_watch["intent"].shift(last_200_answers)       # {"shifted": True, "why": "entropy distributions differ ..."}

What it looks at. Three features of each answer, all from the answer itself (no label): logit of the raw confidence (before
calibration), the margin log p1 − log p2 between the two best options and the normalized entropy of the options'
probabilities (for an answer without them: the two-option reading of the confidence). Fitted on the calibration
answers, it keeps their features as the reference.

What it says. One answer: "features outside the calibration data" when its Mahalanobis distance from the reference is
above the reference's own 1 − alpha quantile (about alpha of in-distribution answers are flagged; a single answer is weak
evidence). A batch (`shift`): a two-sample Kolmogorov–Smirnov test per feature against the reference, Bonferroni-corrected
at alpha; "shifted" when any feature differs. Neither changes an answer or a confidence: what to do (abstain, ask a
person, recalibrate on labelled answers of the new regime) is yours.

What it reports is "the answers' features differ from the reference", not "the inputs are out of domain": it sees the
decider's confidence profile, not the content. Intents the decider has never seen, phrased like the calibration ones,
mostly pass (for inputs from outside the calibration set: solvi.core.guarantees.openset.OpenSetGate; for the share
answered alone and the answer mix: DriftMonitor); a rephrasing of the same domain can be flagged like another domain.
Confidence can rise under a shift as well as fall, which a monitor for a drop of confidence misses. The per-answer note
catches few answers of a new regime; test batches of about 200 answers with `shift` (small batches miss more). alpha is
nominal (asymptotic KS p-values, three Bonferroni tests).

What it cannot see. A regime whose answers look like the calibration answers (the same spread of confidences, margins
and entropies) but whose errors differ passes both checks: only labels show it. A shift is a warning, not a guarantee.

Only numpy is needed. A record is {"confidence": c, "probs": {option: p} or None}, or an answer (solvi Result) — its raw
confidence is taken from `extra["out_of_domain"]["raw"]` when the System calibrated it.
"""
from __future__ import annotations

import hashlib
import json
import math

from . import warn_on_import

warn_on_import(__name__)

_LO, _HI = 1e-4, 1 - 1e-4
FEATURES = ("logit_confidence", "margin", "entropy")


def _np():
    import numpy as np
    return np


def _check(conf):
    c = float(conf)
    if not math.isfinite(c) or not 0.0 <= c <= 1.0:
        raise ValueError(f"a confidence must be a number in [0, 1], not {conf!r}")
    return c


def features(conf, probs=None):
    """(logit c, log p1 − log p2, normalized entropy) of one answer from its confidence and its options' probabilities."""
    c = min(max(_check(conf), _LO), _HI)
    ps = sorted((min(max(float(p), _LO), 1.0) for p in (probs or {}).values()), reverse=True)
    if len(ps) >= 2:
        margin = math.log(ps[0]) - math.log(ps[1])
        tot = sum(ps)
        q = [p / tot for p in ps]
        ent = -sum(p * math.log(p) for p in q) / math.log(len(q))
    else:                                             # no options' probabilities: the two-option reading of c
        margin = math.log(c) - math.log(1 - c)
        ent = -(c * math.log(c) + (1 - c) * math.log(1 - c)) / math.log(2)
    return math.log(c / (1 - c)), margin, ent


def _record(r):
    """A record dict or an answer (Result) → (raw confidence, probs)."""
    if isinstance(r, dict):
        return r["confidence"], r.get("probs")
    ex = getattr(r, "extra", None) or {}
    raw = (ex.get("out_of_domain") or {}).get("raw") if isinstance(ex, dict) else None
    return (raw if raw is not None else r.confidence), getattr(r, "probs", None)


def _ks_p(d, n, m):
    """Asymptotic p-value of the two-sample Kolmogorov–Smirnov statistic d (samples of n and m)."""
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 1e-3:
        return 1.0
    s = sum(2 * (-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(min(1.0, max(0.0, s)))


def ks_2samp(a, b):
    """Two-sample Kolmogorov–Smirnov distance and its asymptotic p-value."""
    np = _np()
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    allv = np.concatenate([a, b])
    d = float(np.abs(np.searchsorted(a, allv, side="right") / len(a)
                     - np.searchsorted(b, allv, side="right") / len(b)).max())
    return d, _ks_p(d, len(a), len(b))


class ShiftWatch:
    """Out-of-domain notes for single answers and a shift test for batches, against the calibration answers (see the
    module docs). alpha: the share of in-distribution answers the single-answer note flags, and the level of `shift`."""

    method = "shift_watch"

    def __init__(self, alpha=0.01):
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha is in (0, 1), not {alpha!r}")
        self.alpha = alpha
        self.fitted = False

    def _F(self, records):
        np = _np()
        rows = [features(*_record(r)) for r in records]
        return np.array(rows, float).reshape(len(rows), 3)

    def fit(self, records):
        """records: the calibration answers ({"confidence", "probs"} or Results, raw confidences). → self."""
        np = _np()
        records = list(records)
        if len(records) < 2:
            raise ValueError("ShiftWatch.fit: at least 2 answers are needed")
        F = self._F(records)
        self.center_ = F.mean(axis=0)
        cov = np.atleast_2d(np.cov(F, rowvar=False))
        self.icov_ = np.linalg.pinv(cov + 1e-6 * np.eye(len(cov)))
        self.d2_cut_ = float(np.quantile(self._d2(F), 1 - self.alpha))
        self.F_ = F
        self.n_ = len(records)
        self.fitted = True
        return self

    def _need(self):
        if not self.fitted:
            raise RuntimeError("ShiftWatch: fit it first (System.calibrate(..., watch=) does)")

    def _d2(self, F):
        np = _np()
        D = F - self.center_
        return np.einsum("ij,jk,ik->i", D, self.icov_, D)

    def notes_many(self, records):
        """→ [the out-of-domain notes of each answer] (an empty list: nothing to note)."""
        self._need()
        records = list(records)
        if not records:
            return []
        far = self._d2(self._F(records)) > self.d2_cut_ + 1e-9
        msg = ("features outside the calibration data (confidence or the options' spread unlike any "
               f"{1 - self.alpha:.0%} of the {self.n_} calibration answers)")
        return [[msg] if f else [] for f in far]

    def note(self, confidence, probs=None):
        """One answer (its raw confidence and options' probabilities) → [out-of-domain notes]."""
        return self.notes_many([{"confidence": confidence, "probs": probs}])[0]

    def shift(self, records):
        """A batch of answers against the calibration answers → {"shifted", "features": {name: {"ks", "p"}}, "n",
        "why"}: a per-feature two-sample Kolmogorov–Smirnov test, Bonferroni-corrected at alpha."""
        self._need()
        records = list(records)
        if not records:
            raise ValueError("ShiftWatch.shift: an empty batch")
        F = self._F(records)
        a = self.alpha / len(FEATURES)
        feats, why = {}, []
        for c, nm in enumerate(FEATURES):
            d, p = ks_2samp(self.F_[:, c], F[:, c])
            feats[nm] = {"ks": d, "p": p}
            if p < a:
                why.append(f"{nm} distributions differ (Kolmogorov-Smirnov distance {d:.3f}, p-value {p:.2g} < {a:.2g})")
        return {"shifted": bool(why), "features": feats, "n": len(records), "why": "; ".join(why) or "no shift seen"}

    def fingerprint(self):
        if not self.fitted:
            return "unfitted"
        np = _np()
        body = {"alpha": self.alpha, "n": self.n_, "d2": round(self.d2_cut_, 10),
                "center": np.round(self.center_, 10).tolist(),
                "ref": hashlib.sha256(np.round(self.F_, 10).tobytes()).hexdigest()}
        return "shift:" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]

    def __repr__(self):
        return f"ShiftWatch(alpha={self.alpha}; {f'fitted on {self.n_}' if self.fitted else 'unfitted'})"


__all__ = ["FEATURES", "ShiftWatch", "features", "ks_2samp"]
