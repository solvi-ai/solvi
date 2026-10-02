"""Drift: has the stream of decisions moved away from the one the thresholds were calibrated on?

A threshold from act_guard holds "for inputs like the calibration examples". When the inputs change, the promise is kept
by escalating more — or the model goes on answering alone and is wrong more often — and nothing says so. A DriftMonitor
compares the last `window` decisions of one question with a reference window and names what moved.

    from solvi.drift import DriftMonitor
    mon = DriftMonitor(window=100)                 # the first 100 decisions observed are the reference
    for text in stream:
        d = part(text)
        rep = mon.observe(d)                       # or mon.observe(d, label=truth) when the truth is known
        if rep["drift"]:
            ...                                    # rep["why"]: what moved; recalibrate (act_guard), ask for labels

    mon = DriftMonitor(window=100).calibrate(decisions, labels)      # or: an explicit reference

Without labels, four signals of any decision: the share answered alone (two-proportion z test, Fisher's exact test once
it is small), the distribution of the answers (chi-square test of homogeneity, with the total-variation distance as its
effect size; answers expected fewer than 5 times in a window are pooled into one bin, and when fewer than two bins are
left the signal is not tested — the report's "not_tested" says so: a question with K answers needs a window of about
5 × K), the mean confidence and the mean act probability (z tests). With labels (at least `min_labelled` in both
windows): accuracy with escalations counted as errors, and — among the labelled decisions answered alone — the
calibration error (ECE) and coverage_at(accuracy), by a permutation test run once per `window` decisions (both are
noisy on a hundred cases). A signal is flagged when its test is significant AND its effect is at least the minimum — on
a large window a tiny shift is significant and is not drift. `drift` is true when at least `min_signals` signals are
flagged.

Significant at what level: the tests are repeated at every decision and there are several of them, so each is held to
alpha / (signals tested × decisions in `horizon`) — a union bound: on a stream that has not changed, the chance of a
false flag within `horizon` decisions (default 1,000) is at most alpha (default 1%), as far as each test's p-value is
exact; a longer stream gets alpha per horizon.

It is a signal, not a verdict: what to do — recalibrate, escalate, ask for labels — is the caller's. Simulated
(independent decisions; tests/test_drift.py runs a small part of it): 0 of 1,200 stationary streams of 1,000 decisions flagged with the
defaults (3 to 57 answers, a reference of 1,500 or the stream's first 100, three shapes of confidence); a change of the
share answered alone from 66% to 12% is flagged about 60 decisions after it with window=100 (40 with window=50, 90 with
window=200), a change of the mix of three answers from 1:1:1 to 1:8:1 about 80 decisions after it. A smaller window
answers sooner and sees less: it needs a larger change, and fewer answers per question."""
from __future__ import annotations

import math
import warnings
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, fisher_exact, norm

from .calibration import coverage_at, ece, summary

SIGNALS = ("answered", "answers", "confidence", "act", "accuracy", "ece", "coverage_at")
MIN_EXPECTED = 5          # an answer expected fewer times in a window is pooled with the other rare ones (chi-square)


@dataclass
class Observation:
    """One decision as the monitor reads it: the answer, its confidence, whether it was given alone, the act probability
    (when the model has an act head) and, when the truth is known, whether the answer was right."""
    value: object
    confidence: float
    alone: bool
    act: float | None = None
    correct: bool | None = None

    @classmethod
    def of(cls, d, label=None, question=None):
        """From a solvi Decision, a Response with `question` (the question's Result, and the act probability from its
        decision's record in the trace), a Result (answer / confidence / status — it carries no act probability), a dict
        of the same fields, or an Observation."""
        if hasattr(d, "results") and hasattr(d, "trace"):               # a Response
            if question is None:
                if len(d.results) != 1:
                    raise ValueError(f"the response answers {sorted(d.results)}: say which with question=")
                question = next(iter(d.results))
            o = cls.of(d.results[question], label)
            rec = next((r for r in d.trace.records if r.name == f"answer:{question}"), None)
            act = (getattr(rec, "extra", None) or {}).get("act")
            o.act = float(act) if isinstance(act, (int, float)) and not isinstance(act, bool) else None
            return o
        if isinstance(d, Observation):
            if label is not None and d.correct is None:
                d.correct = d.value == label
            return d
        if isinstance(d, dict):
            value = d.get("value", d.get("answer"))
            conf = float(d.get("confidence", d.get("conf", 1.0)))
            alone = bool(d["alone"]) if "alone" in d else d.get("escalate") is None and d.get("status", "ok") == "ok"
            act = d.get("act")
        else:
            value = d.value if hasattr(d, "value") else getattr(d, "answer", None)
            conf = getattr(d, "conf", None)
            conf = float(getattr(d, "confidence", 1.0) or 0.0) if conf is None else float(conf)
            alone = (getattr(d, "escalate", None) is None) if hasattr(d, "escalate") else getattr(d, "status", "ok") == "ok"
            act = (getattr(d, "extra", None) or {}).get("act")
        act = float(act) if isinstance(act, (int, float)) and not isinstance(act, bool) else None
        return cls(value, conf, alone, act, None if label is None else value == label)


def window_stats(obs, accuracy=0.9):
    """What a window of observations looks like → {"n", "answered", "answers", "confidence" (mean, std), "act" (mean,
    std, n) when there is an act probability, and with labels "labelled", "accuracy" (escalations count as errors),
    "alone" (n, accuracy, ece, coverage_at)}."""
    n = len(obs)
    out = {"n": n}
    if not n:
        return out
    out["answered"] = sum(o.alone for o in obs) / n
    out["answers"] = dict(Counter(repr(o.value) for o in obs))
    c = np.array([o.confidence for o in obs], float)
    out["confidence"] = {"mean": float(c.mean()), "std": float(c.std()), "n": n}
    a = np.array([o.act for o in obs if o.act is not None], float)
    if len(a):
        out["act"] = {"mean": float(a.mean()), "std": float(a.std()), "n": len(a)}
    lab = [o for o in obs if o.correct is not None]
    if lab:
        out["labelled"] = len(lab)
        out["accuracy"] = sum(o.correct and o.alone for o in lab) / len(lab)
        alone = [o for o in lab if o.alone]
        if alone:
            s = summary([o.confidence for o in alone], [float(o.correct) for o in alone], accuracy=accuracy)
            out["alone"] = {"n": len(alone), "accuracy": s["accuracy"], "ece": s["ece"], "coverage_at": s["coverage_at"]}
    return out


def _p_two_sided(z):
    return float(2 * norm.sf(abs(z)))


def _p_shares(k1, n1, k2, n2):
    """Two shares (k1 of n1, k2 of n2) → the two-sided p-value that they are one: the z test, and Fisher's exact test
    once it is small (the normal tail is not exact where the levels of a long stream are)."""
    pool = (k1 + k2) / (n1 + n2)
    p = _p_two_sided((k2 / n2 - k1 / n1) / math.sqrt(max(pool * (1 - pool) * (1 / n1 + 1 / n2), 1e-12)))
    if p < 0.05:
        p = float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    return p


def _pooled(r, c, n1, n2):
    """The answers' counts in the reference and the window with the rare answers pooled into one bin: the chi-square
    test holds only where every bin is expected at least MIN_EXPECTED times in the smaller sample."""
    exp = (r + c) * min(n1, n2) / (n1 + n2)
    rare = exp < MIN_EXPECTED
    rb, cb = list(r[~rare]), list(c[~rare])
    if rare.any():
        pr, pc = float(r[rare].sum()), float(c[rare].sum())
        if (pr + pc) * min(n1, n2) / (n1 + n2) >= MIN_EXPECTED or not rb:
            rb.append(pr)
            cb.append(pc)
        else:                                        # the pooled bin is rare too: it joins the smallest other bin
            j = int(np.argmin(np.array(rb) + np.array(cb)))
            rb[j] += pr
            cb[j] += pc
    return np.array(rb, float), np.array(cb, float)


class DriftMonitor:
    """See the module docstring. window: the decisions compared with the reference (and the size of a reference taken
    from the stream). alpha, horizon: the chance of a false flag within `horizon` decisions of an unchanged stream is at
    most alpha. min_n: the window must hold this many decisions before anything is
    flagged (default: half the window, at least 20). The minimum effects: min_share (answered alone), min_tv (answers),
    min_shift (mean confidence / act probability), min_accuracy_drop, min_ece_rise, min_coverage_drop.
    keep: the reports of the last `keep` observations are kept in `history` (0: none)."""

    def __init__(self, window=100, *, alpha=0.01, horizon=1000, min_n=None, min_signals=1, min_share=0.10, min_tv=0.15,
                 min_shift=0.05, min_accuracy_drop=0.10, min_ece_rise=0.05, min_coverage_drop=0.15, min_labelled=30,
                 accuracy=0.9, keep=1000, seed=0):
        if int(window) < 2:
            raise ValueError("window must be at least 2")
        if not 0 < float(alpha) < 1 or int(horizon) < 1:
            raise ValueError("alpha must be between 0 and 1 and horizon at least 1")
        self.window, self.horizon = int(window), int(horizon)
        self._perm = None                        # (seen, p-values) of the last permutation tests
        self._warned = False                     # said once: a Result has no act probability
        self.alpha, self.min_signals = float(alpha), int(min_signals)
        self.min_n = int(min_n) if min_n is not None else max(20, self.window // 2)
        self.minimum = {"answered": min_share, "answers": min_tv, "confidence": min_shift, "act": min_shift,
                        "accuracy": min_accuracy_drop, "ece": min_ece_rise, "coverage_at": min_coverage_drop}
        self.min_labelled, self.accuracy, self.seed = int(min_labelled), accuracy, seed
        self.reference: list[Observation] = []
        self.recent: deque[Observation] = deque(maxlen=self.window)
        self.history: deque[dict] = deque(maxlen=max(0, int(keep)) or None) if keep else deque(maxlen=0)
        self.seen, self._ref = 0, None

    # --- the reference
    def calibrate(self, decisions, labels=None):
        """Set the reference explicitly: decisions (Decision / Result / dict) and, when known, their correct answers."""
        decisions = list(decisions)
        labels = [None] * len(decisions) if labels is None else list(labels)
        if len(labels) != len(decisions):
            raise ValueError(f"{len(decisions)} decisions and {len(labels)} labels")
        if len(decisions) < 2:
            raise ValueError("the reference needs at least two decisions")
        self.reference = [Observation.of(d, y) for d, y in zip(decisions, labels)]
        self._ref = window_stats(self.reference, self.accuracy)
        return self

    @property
    def calibrated(self):
        return self._ref is not None

    # --- observing
    def observe(self, decision, label=None, question=None):
        """Take one decision → its report. Until the reference is full (`window` decisions, unless calibrate() set it)
        the decision goes into the reference and the report's phase is "reference". A Response: with question= (its
        act probability is read from the trace); a bare Result of a model's decision has none, so the act signal is
        not tested for it — said once in a warning, and in every report's "not_tested" when the reference has it."""
        o = Observation.of(decision, label, question)
        if o.act is None and not self._warned and getattr(decision, "provenance", None) == "decided":
            self._warned = True
            warnings.warn("DriftMonitor: a Result carries no act probability, so the act signal is not tested: observe "
                          "the Response (mon.observe(res, question=...)) or the part's Decision", stacklevel=2)
        self.seen += 1
        if self._ref is None:
            self.reference.append(o)
            if len(self.reference) >= self.window:
                self._ref = window_stats(self.reference, self.accuracy)
            rep = {"phase": "reference", "drift": False, "flags": [], "why": [], "seen": self.seen}
        else:
            self.recent.append(o)
            rep = self.report()
            rep["seen"] = self.seen
        self.history.append(rep)
        return rep

    # --- the report
    def report(self):
        """The last window against the reference → {"phase": "monitor", "drift", "flags": [signal], "why": [text],
        "tests": {signal: {"reference", "window", "p", "level", ...}}, "not_tested": {signal: why}, "window": stats,
        "reference": stats}. "level": what p must be below — alpha shared among the signals tested and the looks of
        `horizon` decisions (see the module docstring)."""
        ref, cur = self._ref, window_stats(list(self.recent), self.accuracy)
        rep = {"phase": "monitor", "drift": False, "flags": [], "why": [], "tests": {}, "not_tested": {}, "window": cur,
               "reference": ref}
        if ref is None:
            rep["phase"] = "reference"
            return rep
        if cur["n"] < self.min_n:
            rep["why"].append(f"the window holds {cur['n']} decisions, fewer than {self.min_n}")
            return rep
        n1, n2 = ref["n"], cur["n"]
        found = []                                   # (signal, p, effect, why, looks per horizon)

        p1, p2 = ref["answered"], cur["answered"]                      # 1. the share answered alone
        p = _p_shares(round(p1 * n1), n1, round(p2 * n2), n2)
        rep["tests"]["answered"] = {"reference": p1, "window": p2, "p": p}
        found.append(("answered", p, abs(p2 - p1), f"answered alone {p1:.0%} → {p2:.0%} (p = {p:.1e})", self.horizon))

        keys = sorted(set(ref["answers"]) | set(cur["answers"]))        # 2. the distribution of the answers
        r = np.array([ref["answers"].get(k, 0) for k in keys], float)
        c = np.array([cur["answers"].get(k, 0) for k in keys], float)
        rb, cb = _pooled(r, c, n1, n2)
        if len(keys) < 2:                            # one answer everywhere: nothing differs
            rep["tests"]["answers"] = {"chi2": 0.0, "df": 0, "tv": 0.0, "p": 1.0, "pooled": 0}
            found.append(("answers", 1.0, 0.0, "", self.horizon))
        elif len(rb) < 2:
            rep["not_tested"]["answers"] = (
                f"fewer than two answers are expected at least {MIN_EXPECTED} times in {min(n1, n2)} decisions "
                f"({len(keys)} different answers seen): the distribution of the answers cannot be tested on a window "
                "this small — a larger window (and reference)")
        else:
            tot = rb + cb
            er, ec = tot * n1 / (n1 + n2), tot * n2 / (n1 + n2)
            stat = float(((rb - er) ** 2 / er).sum() + ((cb - ec) ** 2 / ec).sum())
            p = float(chi2.sf(stat, len(rb) - 1))
            tv = float(0.5 * np.abs(cb / n2 - rb / n1).sum())
            rep["tests"]["answers"] = {"chi2": stat, "df": len(rb) - 1, "tv": tv, "p": p, "pooled": len(keys) - len(rb) + 1
                                       if len(rb) < len(keys) else 0}
            grew = max(keys, key=lambda k: cur["answers"].get(k, 0) / n2 - ref["answers"].get(k, 0) / n1)
            found.append(("answers", p, tv, f"the answers are distributed differently (total variation {tv:.2f}, "
                                            f"p = {p:.1e}; {grew} grew most)", self.horizon))

        for name, label in (("confidence", "mean confidence"), ("act", "mean act probability")):     # 3. the means
            if name not in ref or name not in cur:
                if name in ref or name in cur:
                    rep["not_tested"][name] = ("the reference" if name in cur else "the window") + \
                        " holds no act probability (a Result does not carry it: observe the Response with question=, " \
                        "or the part's Decision)"
                continue
            a, b = ref[name], cur[name]
            se = math.sqrt(max(a["std"] ** 2 / a["n"] + b["std"] ** 2 / b["n"], 1e-12))
            p = _p_two_sided((b["mean"] - a["mean"]) / se)
            rep["tests"][name] = {"reference": a["mean"], "window": b["mean"], "p": p}
            found.append((name, p, abs(b["mean"] - a["mean"]),
                          f"{label} {a['mean']:.2f} → {b['mean']:.2f} (p = {p:.1e})", self.horizon))

        perm = None
        if min(ref.get("labelled", 0), cur.get("labelled", 0)) >= self.min_labelled:                 # 4. with labels
            a1, a2, l1, l2 = ref["accuracy"], cur["accuracy"], ref["labelled"], cur["labelled"]
            p = _p_shares(round(a1 * l1), l1, round(a2 * l2), l2)
            rep["tests"]["accuracy"] = {"reference": a1, "window": a2, "p": p}
            found.append(("accuracy", p, a1 - a2,
                          f"right and alone {a1:.0%} → {a2:.0%} of the labelled decisions (p = {p:.1e})", self.horizon))
            ra, ca = ref.get("alone"), cur.get("alone")
            if ra and ca and min(ra["n"], ca["n"]) >= self.min_labelled:
                perm = (ra, ca)
        m = len(found) + (2 if perm else 0)          # alpha is shared among the signals tested in this look …
        if perm:
            ra, ca = perm
            looks = max(1.0, self.horizon / self.window)               # … a permutation test runs once per window
            pe, pc = self._permutation_now(self.alpha / (m * looks))
            rep["tests"]["ece"] = {"reference": ra["ece"], "window": ca["ece"], "p": pe}
            found.append(("ece", pe, ca["ece"] - ra["ece"], f"calibration error among the answers given alone "
                                                           f"{ra['ece']:.3f} → {ca['ece']:.3f} (p = {pe:.1e})", looks))
            rep["tests"]["coverage_at"] = {"reference": ra["coverage_at"], "window": ca["coverage_at"], "p": pc}
            found.append(("coverage_at", pc, ra["coverage_at"] - ca["coverage_at"],
                          f"coverage at accuracy {self.accuracy:g}: {ra['coverage_at']:.0%} → {ca['coverage_at']:.0%} "
                          f"(p = {pc:.1e})", looks))
        for name, p, effect, why, looks in found:    # … and among the looks of a horizon (a union bound)
            level = self.alpha / (m * looks)
            rep["tests"][name]["level"] = level
            if p < level and effect >= self.minimum[name]:
                rep["flags"].append(name)
                rep["why"].append(why)
        rep["drift"] = len(rep["flags"]) >= self.min_signals
        return rep

    def _permutation_now(self, level):
        """The permutation p-values, computed once per `window` observed decisions (and kept in between)."""
        if self._perm is None or self.seen - self._perm[0] >= self.window:
            self._perm = (self.seen, self._permutation(level))
        return self._perm[1]

    def _permutation(self, level):
        """Permutation p-values of the rise of ECE and the drop of coverage_at between the reference and the window
        (labelled decisions answered alone, pooled and split at random; seeded: the same every time). As many rounds as
        `level` needs (4 / level at most), stopped as soon as neither value can fall below it."""
        def arr(obs):
            xs = [(o.confidence, float(o.correct)) for o in obs if o.alone and o.correct is not None]
            return np.array([x[0] for x in xs]), np.array([x[1] for x in xs])
        ca, oa = arr(self.reference)
        cb, ob = arr(self.recent)
        d_ece = ece(cb, ob) - ece(ca, oa)
        d_cov = coverage_at(ca, oa, self.accuracy) - coverage_at(cb, ob, self.accuracy)
        pc, po, na = np.concatenate([ca, cb]), np.concatenate([oa, ob]), len(ca)
        rng, ge, gc, rounds = np.random.default_rng(self.seed), 0, 0, 0
        while rounds < math.ceil(4 / level) and min(ge, gc) < 4:
            ix = rng.permutation(len(pc))
            c1, o1, c2, o2 = pc[ix[:na]], po[ix[:na]], pc[ix[na:]], po[ix[na:]]
            ge += (ece(c2, o2) - ece(c1, o1)) >= d_ece - 1e-12
            gc += (coverage_at(c1, o1, self.accuracy) - coverage_at(c2, o2, self.accuracy)) >= d_cov - 1e-12
            rounds += 1
        return (ge + 1) / (rounds + 1), (gc + 1) / (rounds + 1)
