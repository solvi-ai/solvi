"""Inputs from outside the calibration set: an open-set gate that keeps a promised error rate when some of the inputs
are of a kind the decider has no answer for (a new topic, a product it never saw), and a detector that notices when the
stream changes.

Every promise of act_guard / calibrate_for / System.guarantee holds "for inputs like the calibration examples". An
input whose right answer is not among the options is outside that: whatever the decider answers is wrong, and a
threshold calibrated without such inputs lets a share of them through, and the error among the answers given alone can
end up several times the promised rate. The gate sizes the threshold for a share of such inputs and follows the share
as the stream goes:

    from solvi.openset import OpenSetGate, leave_out
    sim = leave_out(calib, make)                 # make(kept options) → a decider without the others: the left-out
                                                 # inputs play "new kinds", any answer to them is wrong
    gate = OpenSetGate.calibrate(known_signals, known_right, sim["novel"], max_error=0.05)
    system.guarantee("intent", promise=gate, signal="act")          # or, outside a System: gate.gate(decision)

How the threshold is sized. For a share π of inputs from outside, the error among the answers given alone at
threshold t is (π·U + (1 − π)·W) / (π·U + (1 − π)·A), with U the share of outside inputs whose signal reaches t, A
the share of known inputs that do and W those of them that are wrong. It grows with π. For each π of a grid the gate
takes the lowest t at which that error passes learn-then-test on a mixture of the calibration examples in exactly that
share — known and left-out ones, stratified, as many as there are — a binomial test at delta / 16 over 16 thresholds
(quantiles of the known signals); a larger share never gets a lower threshold. So, for each share on the grid, with
probability ≥ 1 − delta over the calibration sets, its threshold keeps the promise on every stream in which at most
that share of the inputs are outside (the error grows with the share), the outside inputs being like the left-out
examples and the rest like the calibration ones.

What share to size for. The gate reads one indicator per decision: is the signal below the cut c — the signal at which
known and left-out inputs differ most? Over each of the windows in `track` (the last 25 and the last 200 decisions) it
bounds the share of such signals from above (Clopper–Pearson at delta) and turns the bound into a share of outside
inputs through the known and left-out shares below c; the threshold is the one for the largest of these shares, never
below the one for `min_share` (0.1). The short window catches a sudden change within a few dozen decisions, the long one
a small share. A share above the largest one any threshold can serve escalates everything.

The flag. One-sided Bernoulli CUSUMs on the same indicator, one for each share in `design` (0.1, 0.3, 0.6), against q0,
an upper bound of the share below c among known inputs; a flag when any of them reaches h. h is set by simulation at
calibration: on streams of `horizon` decisions in which the indicator is Bernoulli(q0), the chance of a flag is at most
`alpha` (solvi.drift.Cusum, the detector DriftMonitor uses too: at least 2,000 and 20 / alpha simulated streams, a
fixed seed — the report gives h and the simulated rate; alpha below 1e-4 is refused). After a flag the share is
also estimated from the decisions since that CUSUM last stood at zero (its estimate of the change point; at most the
last `window`), and the largest estimate is used. A DriftMonitor (solvi.drift) can be passed as a second detector: its
flag starts the same estimate (from its window); the flag and its reason are in every record's state. The flag tells;
the estimate keeps the promise — it moves the threshold before any flag.

What it does not do: it does not know what the new inputs are and does not learn them (labels, a new option and a
recalibration are the caller's). The left-out inputs stand in for the real outside ones; when those look more familiar
to the decider than the stand-ins did, the bound is optimistic — a group of new intents can be harder to tell from the
known ones than every left-out fold. Between a sudden change and the moment the short window sees it the threshold is
the one for the share seen before: a stream that jumps to a large share of outside inputs gets a few wrong answers in
that time, which matter when little is answered after it. How to use it: docs/guide.md, "Inputs from outside the
calibration set"."""
from __future__ import annotations

import math

import numpy as np

SHARES = tuple(round(x, 2) for x in np.arange(0.0, 0.951, 0.05))


def _cp_upper(k, n, a):
    """Clopper–Pearson upper bound of a share: P(share > bound) ≤ a."""
    if n == 0:
        return 1.0
    if k >= n:
        return 1.0
    from scipy.stats import beta
    return float(beta.ppf(1 - a, k + 1, n - k))


def _signal_of(d):
    """A decision's signal: its act probability when it has one, else its confidence; a number is taken as it is."""
    if isinstance(d, (int, float, np.floating)) and not isinstance(d, bool):
        return float(d)
    a = (getattr(d, "extra", None) or {}).get("act")
    if isinstance(a, (int, float)) and not isinstance(a, bool):
        return float(a)
    c = getattr(d, "conf", None)
    return float(getattr(d, "confidence", 0.0) if c is None else c)


def _sizes(nk, nu, pi):
    """The largest mixture with a share pi of outside examples: (known, outside) counts."""
    if pi <= 0:
        return nk, 0
    a = nk
    b = int(round(pi / (1 - pi) * a))
    if b > nu:
        b = nu
        a = int(round((1 - pi) / pi * b))
    return a, b


def _ltt(s, w, known_s, error, delta, min_support, size=16):
    """Learn-then-test on one mixture: the lowest threshold of the grid (`size` quantiles of the known signals) at
    which the wrong answers among those let through pass the binomial test at delta / size; min_support known
    examples above."""
    from .calibration import _binom_cdf, ltt_grid
    grid = ltt_grid(known_s, size)
    for t in sorted(grid):
        auto = s >= t
        k = int(auto.sum())
        if int((known_s >= t).sum()) < min_support or not k:
            continue
        if _binom_cdf(float(w[auto].sum()), k, error) <= delta / len(grid):
            return float(t)
    return math.inf


def leave_out(examples, make, folds=3, seed=0, label=None):
    """The leave-options-out simulation: the options are split into `folds` groups; for each, a decider is made without
    that group (make(kept options) → a callable decider: a DecisionPart, or anything with decide(list) / __call__) and
    asked about every example — the examples of the kept options give known signals (right or wrong), those of the
    left-out group give signals of inputs whose answer is not among the options. examples: [(input, correct option)];
    label: the option as the decider names it, from an example's correct answer (default: as given). → {"known":
    (signals, right), "novel": signals, "groups": [left-out options per fold]}. Signals: the act probability when the
    decider gives one, else the confidence."""
    import random
    examples = list(examples)
    lab = label or (lambda y: y)
    opts = sorted({lab(y) for _, y in examples}, key=str)
    if int(folds) < 2 or int(folds) > len(opts):
        raise ValueError(f"folds must be between 2 and the number of options ({len(opts)})")
    random.Random(seed).shuffle(opts)
    groups = [opts[i::int(folds)] for i in range(int(folds))]
    ks, kr, us = [], [], []
    for g in groups:
        gs = set(g)
        kept = [o for o in sorted({lab(y) for _, y in examples}, key=str) if o not in gs]
        part = make(kept)
        xs = [x for x, _ in examples]
        ds = part.decide(xs) if hasattr(part, "decide") else [part(x) for x in xs]
        for d, (_, y) in zip(ds, examples):
            if lab(y) in gs:
                us.append(_signal_of(d))
            else:
                ks.append(_signal_of(d))
                kr.append(getattr(d, "value", None) == lab(y))
    return {"known": (ks, kr), "novel": us, "groups": groups}


def _detector(q0, f_novel, design, alpha, horizon, seed):
    """The CUSUMs (solvi.drift.Cusum) on the indicator "signal below the cut", one per design share d: the
    log-likelihood ratio of a stream with a share d of outside inputs (below the cut with probability
    (1 − d)·q0 + d·f_novel) against known inputs only (q0); h simulated on streams whose indicator is Bernoulli(q0)."""
    from .drift import Cusum
    up = np.array([math.log(((1 - d) * q0 + d * f_novel) / q0) for d in design])
    down = np.array([math.log(((1 - d) * (1 - q0) + d * (1 - f_novel)) / (1 - q0)) for d in design])

    def make_null(rng, sims):
        return lambda: np.where(rng.uniform(size=(sims, 1)) < q0, up, down)
    c = Cusum.calibrate([f"share {d:g}" for d in design], make_null, alpha, horizon, seed=seed)
    c.up, c.down = up, down
    return c


class OpenSetGate:
    """See the module docstring. Made by OpenSetGate.calibrate(...); stateful: observe(signal) after every decision
    it gated (System.guarantee does it), threshold_of() → the threshold for the next one."""
    stateful = True

    def __init__(self, error, delta, shares, thresholds, cut, f_known, f_novel, q0, design, cusum, min_share,
                 window, report, monitor=None, track=(25, 200), min_track=50):
        self.error, self.delta, self.shares, self.thresholds = float(error), float(delta), list(shares), dict(thresholds)
        self.cut, self.f_known, self.f_novel, self.q0 = cut, f_known, f_novel, q0
        self.design, self.cusum, self.h = tuple(design), cusum, cusum.h
        self.min_share, self.window, self.report, self.monitor = float(min_share), int(window), report, monitor
        self.track = tuple(int(x) for x in (track or ()))
        self.min_track = int(min_track)
        self.by, self.min_group, self.nodes = None, None, None
        self.method = "open-set"
        self.promise = (f"error among the answers given alone ≤ {self.error:g} with probability ≥ {1 - self.delta:g}, for "
                        f"streams in which at most the stated share of the inputs are from outside the calibration set "
                        f"(like the left-out examples) — the share estimated from the recent signals, at least "
                        f"{self.min_share:g}")
        self.reset()

    # --- calibration
    @classmethod
    def calibrate(cls, known_scores, known_correct, novel_scores, *, max_error=0.05, delta=0.10, min_share=0.10,
                  shares=SHARES, track=(25, 200), min_track=50, alpha=0.01, horizon=1000, design=(0.1, 0.3, 0.6),
                  window=500, min_support=10, monitor=None, grid=16, seed=0):
        """known_scores / known_correct: the decider's signal and right / wrong on calibration examples like the
        stream's known inputs (the deployed decider on held-out labelled examples); novel_scores: its signal on inputs
        whose answer is not among its options (leave_out(...)["novel"], or real outside examples). max_error, delta: the
        promise. min_share: the share of outside inputs the threshold is always sized for (0: the plain learn-then-test
        threshold while the stream looks like the calibration examples). shares: the grid of shares. track: the windows
        (decisions) the share is estimated over, each once it has min(window, min_track) decisions; () or None: only
        after a flag. alpha, horizon: the detector's false flags (≤ alpha within horizon decisions of a stream like the
        calibration examples). design: the shares of outside inputs the detector's CUSUMs are tuned to. window: at most
        this many decisions since the change are used for the estimate after a flag. min_support: a threshold must let
        through this many known examples. monitor: a solvi.drift.DriftMonitor whose flag also starts the estimate.
        grid: thresholds tried per share (learn-then-test, Bonferroni). → OpenSetGate"""
        from .calibration import check_rate
        error = max_error
        check_rate("max_error", error)
        check_rate("delta", delta)
        check_rate("alpha", alpha)
        if alpha < 1e-4:
            raise ValueError(f"alpha {alpha:g} is below 1e-4: the detector's level is set by simulation, which cannot "
                             "certify so rare a false flag")
        if not 0 <= float(min_share) < 1:
            raise ValueError("min_share must be in [0, 1)")
        if not design or not all(0 < float(d) < 1 for d in design):
            raise ValueError("design must be shares strictly between 0 and 1")
        if track and any(int(w) < 1 for w in track):
            raise ValueError("every window in track must be at least 1 decision")
        ks, kr = np.asarray(known_scores, float), np.asarray(known_correct, bool)
        us = np.asarray(novel_scores, float)
        if len(ks) != len(kr):
            raise ValueError(f"{len(ks)} known scores and {len(kr)} judged")
        if len(ks) < 30 or len(us) < 30:
            raise ValueError(f"the gate needs at least 30 known and 30 outside examples ({len(ks)} and {len(us)} given): "
                             "the bounds on fewer certify nothing")
        if not (np.isfinite(ks).all() and np.isfinite(us).all()):
            raise ValueError("every calibration signal must be a finite number")
        nk, nu = len(ks), len(us)
        shares = sorted({float(x) for x in shares} | {float(min_share)})
        rng = np.random.default_rng(seed)
        ko, uo = rng.permutation(nk), rng.permutation(nu)        # label-free orders for the stratified mixtures
        thresholds, sizes = {}, {}
        for pi in shares:
            a, b = _sizes(nk, nu, pi)
            s_mix = np.concatenate([ks[ko[:a]], us[uo[:b]]])
            w_mix = np.concatenate([~kr[ko[:a]], np.ones(b, bool)]).astype(float)
            sizes[pi] = (a, b)
            thresholds[pi] = _ltt(s_mix, w_mix, ks[ko[:a]], error, delta, min_support, grid) if a else math.inf
        run = -math.inf                                # a larger share never gets a lower threshold
        for pi in shares:
            run = thresholds[pi] = max(run, thresholds[pi])
        # the indicator's cut: where the known and left-out signals differ most
        cand = np.unique(np.concatenate([ks, us]))
        fk = np.searchsorted(np.sort(ks), cand, side="left") / nk          # share below c
        fu = np.searchsorted(np.sort(us), cand, side="left") / nu
        j = int(np.argmax(fu - fk))
        cut, f_known, f_novel = float(cand[j]), float(fk[j]), float(fu[j])
        from scipy.stats import ks_2samp
        p_sep = float(ks_2samp(us, ks, alternative="greater").pvalue)      # outside signals lower than the known ones?
        if f_novel - f_known < 0.05 or p_sep >= 0.05:
            raise ValueError(f"the signal does not tell the outside inputs from the known ones (largest difference of the "
                             f"shares below a cut: {f_novel - f_known:.3f}, p = {p_sep:.2g}): no gate can size a threshold "
                             "for them on it")
        q0 = _cp_upper(int(round(f_known * nk)), nk, delta)
        design = tuple(float(d) for d in design)
        cusum = _detector(q0, max(f_novel, q0 + 1e-6), design, alpha, horizon, seed)
        known_err = float((~kr).mean())
        report = {"n_known": nk, "n_outside": nu, "known_error": known_err, "error": error, "delta": delta,
                  "mixtures": {f"{k:g}": v for k, v in sizes.items()},
                  "thresholds": {f"{k:g}": v for k, v in thresholds.items()},
                  "max_share": max([k for k, v in thresholds.items() if math.isfinite(v)], default=None),
                  "cut": cut, "below_cut": {"known": f_known, "outside": f_novel, "q0": q0}, "separation_p": p_sep,
                  "detector": {"design": list(design), "h": cusum.h, "alpha": alpha, "horizon": horizon,
                               "simulated_false_flags": cusum.rate, "simulated_streams": cusum.sims},
                  "track": list(track or ()), "min_share": float(min_share)}
        for pi in (0.0, float(min_share)):
            t = thresholds.get(pi, math.inf)
            if math.isfinite(t):
                report[f"at_share_{pi:g}"] = {"threshold": t, "answered_known": float((ks >= t).mean()),
                                              "error_known": float((~kr[ks >= t]).mean()) if (ks >= t).any() else 0.0,
                                              "answered_outside": float((us >= t).mean())}
        if not math.isfinite(thresholds[float(min_share)]):
            report["why"] = (f"no threshold keeps the error ≤ {error:g} with {min_share:g} of the inputs from outside: "
                             "everything escalates")
        return cls(error, delta, shares, thresholds, cut, f_known, f_novel, q0, design, cusum, min_share, window,
                   report, monitor, track, min_track)

    # --- the online state
    def reset(self):
        """Forget the stream (a new one starts): no flag, the threshold for min_share."""
        from collections import deque
        self.seen, self.flag_at = 0, None
        self.cusum.reset()
        self.S, self.zero_at = 0.0, 0
        self._recent = deque(maxlen=max([self.window, *self.track]))   # 1[signal < cut] of the last decisions
        self.since = []                  # ... and since the change point, after a flag
        self.share = None                # the estimated share of outside inputs (an upper bound)
        self.why = None

    def observe(self, signal, result=None):
        """One decision's signal (after it was gated) → the state after it."""
        x = float(signal) < self.cut
        self.seen += 1
        j = self.cusum.step(self.cusum.up if x else self.cusum.down)
        _, self.S, since = self.cusum.top()
        self._recent.append(x)
        if self.flag_at is None:
            self.zero_at = self.seen - since
            flagged = self.S >= self.h
            why = None
            if flagged:
                why = (f"signals below {self.cut:.3g}: CUSUM {self.S:.1f} ≥ {self.h:.1f} (tuned to a share of "
                       f"{self.design[j]:g}; since decision {self.zero_at + 1})")
            elif self.monitor is not None:
                rep = self.monitor.observe({"value": getattr(result, "answer", None),
                                            "confidence": float(getattr(result, "confidence", 1.0) or 0.0),
                                            "alone": getattr(result, "status", "ok") == "ok", "act": float(signal)})
                if rep.get("drift"):
                    flagged, why = True, "DriftMonitor: " + "; ".join(rep.get("why", []))
                    self.zero_at = max(0, self.seen - getattr(self.monitor, "window", 100))
            if flagged:
                self.flag_at, self.why = self.seen, why
                self.since = list(self._recent)[-(self.seen - self.zero_at):] if self.seen > self.zero_at else []
        else:
            self.since.append(x)
            if len(self.since) > self.window:
                self.since = self.since[-self.window:]
        self._estimate()
        return self.state()

    def _upper_share(self, xs):
        n, k = len(xs), sum(xs)
        if not n:
            return 1.0
        qu = _cp_upper(k, n, self.delta)
        return min(1.0, max(0.0, (qu - self.f_known) / max(self.f_novel - self.f_known, 1e-9)))

    def _estimate(self):
        """The share of outside inputs, bounded from above: over each window of `track` (once it has min(window,
        min_track) decisions) and, after a flag, over the decisions since the change point — the largest."""
        est = []
        rec = list(self._recent)
        for w in self.track:
            if len(rec) >= min(w, self.min_track):
                est.append(self._upper_share(rec[-w:]))
        if self.flag_at is not None:
            est.append(self._upper_share(self.since))
        self.share = max(est) if est else None

    def level(self):
        """The share the next threshold is sized for (on the grid, rounded up)."""
        want = self.min_share if self.share is None else max(self.min_share, self.share)
        for s in self.shares:
            if s >= want - 1e-12:
                return s
        return None                                   # above every share on the grid: escalate all

    def threshold_of(self, group=None):
        lv = self.level()
        thr = math.inf if lv is None else self.thresholds[lv]
        return thr, None, None

    def state(self):
        """What the next decision's threshold rests on (recorded with it)."""
        out = {"seen": self.seen, "level": self.level(), "cusum": round(self.S, 3)}
        if self.share is not None:
            out["share"] = round(self.share, 4)
        if self.flag_at is not None:
            out.update(flag_at=self.flag_at, change_at=self.zero_at + 1, why=self.why)
        return out

    def record(self):
        return {"method": "open-set", "promise": self.promise, "error": self.error, "delta": self.delta,
                "n": self.report["n_known"], "n_outside": self.report["n_outside"], "min_share": self.min_share}

    def fingerprint(self):
        from .provenance import digest
        return digest("OpenSetGate", self.error, self.delta, sorted(self.thresholds.items()), self.cut, self.f_known,
                      self.f_novel, self.q0, list(self.design), self.h, self.min_share, self.window, list(self.track),
                      self.min_track)

    # --- outside a System
    def gate(self, decision):
        """Gate a solvi Decision (or a number) as System.guarantee would: below the current threshold the decision
        escalates with the reason; its extra["open_set"] records the threshold and the state; the gate observes it.
        → the decision (a number: whether it may be answered alone)."""
        s = _signal_of(decision)
        thr = self.threshold_of()[0]
        st = self.state()
        ok = s >= thr
        if hasattr(decision, "extra"):
            decision.extra = dict(decision.extra or {})
            decision.extra["open_set"] = {"signal": s, "threshold": thr if math.isfinite(thr) else None, **st,
                                          "promise": self.promise}
            if not ok and getattr(decision, "escalate", None) is None:
                decision.escalate = (f"open-set gate: signal {s:.3g} < {thr:.4g} (sized for up to {st['level']} of the "
                                     f"inputs from outside the calibration set); would have answered {decision.value!r}"
                                     if st["level"] is not None else
                                     f"open-set gate: the estimated share of inputs from outside the calibration set "
                                     f"({self.share:.2f}) is above what any threshold can serve; would have answered "
                                     f"{decision.value!r}")
        self.observe(s, decision)
        return decision if hasattr(decision, "extra") else ok

    def run(self, signals):
        """The gate over a stream of signals, from a fresh state, without touching this gate's → [(threshold, alone,
        level)] and the flag (decision number or None). For backtests on labelled data."""
        import copy
        g = copy.copy(self)
        g.monitor, g.cusum = copy.deepcopy(self.monitor), copy.deepcopy(self.cusum)
        g.reset()
        out = []
        for s in signals:
            thr = g.threshold_of()[0]
            out.append((thr, float(s) >= thr, g.level()))
            g.observe(s)
        return out, g.flag_at

    def __repr__(self):
        return f"OpenSetGate(error={self.error:g}, min_share={self.min_share:g}, level={self.level()})"


__all__ = ["OpenSetGate", "leave_out"]
