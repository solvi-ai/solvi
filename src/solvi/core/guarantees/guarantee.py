"""A guarantee on any signal: a calibrated threshold with a stated promise on a question's answer — given by a rule, a
fitted head (fit), a model decision — on a fact the catalog computes (a trust score, an agreement share), or
on any scalar outside a System.

    report = system.guarantee("pair", examples, max_risk=0.01)          # conformal risk control on the answer's confidence
    report = system.guarantee("pair", examples, max_error=0.02)         # learn-then-test: error among the answered ≤ 2%
    report = system.guarantee("act", examples, max_risk=0.03, signal="trust", correct=judge)     # a computed fact
    report = system.guarantee("correct", examples, max_error=0.3, answer="yes")   # one-sided: "yes" alone when P(yes) ≥ t

    from solvi.core.guarantees.guarantee import calibrate
    p = calibrate(scores, correct, max_error=0.05)                      # any scalar: p.threshold, p.allows(s), p.report

examples: [(init_state, correct answer)] held out from whatever fitted the answer (or folds=k for a fast head fitted
on them). The question is asked on each (nothing stored, nothing counted), the signal and right / wrong are collected,
and the threshold is chosen by one of three methods — each makes a different promise, for inputs like the calibration
examples (exchangeable with them: the same stream, not a new domain):

    method       parameter  promise
    "crc"        max_risk=r     P(answered alone and wrong) ≤ r — a share of ALL inputs, answered or escalated; on average
                            over calibration sets (conformal risk control: (wrong answered + 1) / (n + 1) ≤ r)
    "ltt"        max_error=e    the error AMONG the answers given alone ≤ e, with probability ≥ 1 − delta over the
                            calibration set (learn-then-test: a binomial test per threshold, Bonferroni over ≤ 64;
                            procedure="fixed": fixed-sequence, from the highest threshold down;
                            procedure="fallback": the fallback chain over the same descent)
    "empirical"  max_error=e    none: the error among the answered was ≤ e on the calibration examples only

max_risk= selects "crc", max_error= selects "ltt" (method="empirical" to ask for the plain one). groups= gives a threshold per
group (a fact name, a hierarchy of fact names, a function of facts, or "answer": the answer the question would give —
"among the inputs answered 'match', ..."), with the promise inside every group (after HG-CRC; see solvi.core.calibration).

What is checked before a threshold is set, so that a promise is never made on a signal that cannot carry it:
- separation: the signal must rank the right answers above the wrong ones on the calibration examples (a one-sided
  Mann–Whitney test at 5%); one that does not — a model approving its own answers at chance — raises (weak="warn":
  a warning, and the warning is recorded in the guarantee);
- support: a threshold must have at least min_support calibration examples at or above it (default 10); none that
  does → everything escalates, and the report says why;
- feasibility: with n examples, conformal risk control cannot certify a risk below 1 / (n + 1) — the report says so.
A guarantee whose threshold is inf (everything escalates; the report's "why" names the best finite threshold's bound) or
one calibrated on so few examples that the promise allows no wrong answer among the answered emits a GuaranteeWarning.

The report gives, on the calibration examples, both "risk" (answered alone and wrong, a share of all) and "error"
(among the answered), the share answered, the threshold's support and the separation (AUROC, p). Every answer of a
guarded question records the promise, the signal's value and the threshold in its result (extra["guarantee"]) and in
the trace (a hashed record "guard:<question>"); below the threshold the question abstains with the reason. Replay with
the System re-derives the verdict; a guarantee changed since the decision is a mismatch."""
from __future__ import annotations

import math
import warnings

import numpy as np

from ..runtime import replay_guard        # defined there (the runtime replays guard records); re-exported

METHODS = ("crc", "ltt", "empirical")
SEPARATION_P = 0.05            # the signal must separate right from wrong at this level (one-sided Mann–Whitney)
MIN_CLASS = 5                  # separation is tested only with at least this many right and wrong examples


class GuaranteeWarning(UserWarning):
    """A guarantee was set on a signal that did not show it can carry it (weak="warn"), or on too few examples."""


# ------------------------------------------------------------------------------------------------ the promise texts
def promise_text(method, level, delta=None, groups=False, n_groups=1, procedure="bonferroni"):
    """What a method promises, in words (recorded with every decision it gates)."""
    if method == "empirical":
        return (f"none: the error among the answers given alone was ≤ {level:g} on the calibration examples only"
                + (" (per group)" if groups else ""))
    if method == "crc":
        if not groups:
            return f"P(answered alone and wrong) ≤ {level:g} — a share of all inputs — for inputs like the calibration examples"
        if delta is None:
            return (f"P(answered alone and wrong) ≤ {level:g} within each group, for inputs like the calibration examples "
                    "(conformal risk control per group)")
        return (f"P(answered alone and wrong) ≤ {level:g} within every group at once ({n_groups} groups), with probability "
                f"≥ {1 - delta:g}, for inputs like the calibration examples")
    where = f" within every group at once ({n_groups} groups)" if groups else ""
    how = {"fixed": " (fixed-sequence learn-then-test)", "fallback": " (fallback-chain learn-then-test)"}.get(procedure, "")
    return (f"error among the answers given alone ≤ {level:g}{where} with probability ≥ {1 - delta:g}, for inputs like "
            f"the calibration examples{how}")


# ------------------------------------------------------------------------------------------------ choosing a threshold
def _curve(s, w):
    """Candidate thresholds (the distinct finite signals and inf, ascending) → (grid, answered count, wrong answered).
    A signal of +inf is always answered (a forced answer), −inf never (an abstention before the threshold)."""
    grid = np.concatenate([np.unique(s[np.isfinite(s)]), [np.inf]])
    order = np.argsort(-s, kind="stable")
    ss, cw = s[order], np.concatenate([[0.0], np.cumsum(w[order])])
    k = np.searchsorted(-ss, -grid, side="right")             # cases with signal ≥ t
    return grid, k, cw[k]


def _search(s, w, method, level, delta, min_support, bound_delta=None, procedure="bonferroni"):
    """The lowest threshold that keeps the method's promise on these examples → (threshold, why or None). bound_delta:
    conformal risk control as a binomial bound at that level (groups with delta). procedure: learn-then-test's multiple
    testing over its grid ("bonferroni", "fixed" or "fallback": calibration.ltt_certified)."""
    from ..calibration import loss_budget, ltt_certified, ltt_grid
    n = len(s)
    if n == 0:
        return math.inf, "no calibration examples"
    grid, k, m = _curve(s, w)
    ok_support = (k >= min_support) | ~np.isfinite(grid)
    if method == "empirical":
        good = ok_support & (k > 0) & (m <= level * np.maximum(k, 1) + 1e-9) & np.isfinite(grid)
    elif method == "crc":
        if bound_delta is None:
            good = ok_support & ((m + 1) / (n + 1) <= level + 1e-12)
        else:
            b = loss_budget(n, level, bound_delta)
            good = ok_support & (m <= b) if b >= 0 else np.zeros(len(grid), bool)
    else:
        g = ltt_grid(s[np.isfinite(s)])
        good = np.zeros(len(grid), bool)
        if len(g):
            pos = {float(t): i for i, t in enumerate(grid)}
            ix = np.array([pos[float(t)] for t in g])
            good[ix] = ltt_certified(k[ix], m[ix], level, delta, procedure, min_support) & ok_support[ix]
    hit = np.flatnonzero(good)
    if len(hit) and np.isfinite(grid[hit[0]]):
        return float(grid[hit[0]]), None
    if len(hit):        # only inf keeps it (conformal risk control: escalating everything is never wrong) — say why
        return math.inf, _why_only_inf(n, k, m, grid, level, min_support, bound_delta is not None)
    if method == "crc" and bound_delta is None and (n + 1) * level < 1:
        return math.inf, f"{n} examples cannot certify a risk of {level:g}: conformal risk control needs at least {math.ceil(1 / level) - 1}"
    if method == "empirical" and ((m <= level * np.maximum(k, 1) + 1e-9) & (k > 0) & np.isfinite(grid)).any():
        return math.inf, (f"the thresholds that reach the target rest on fewer than {min_support} calibration examples "
                          "(min_support)")
    if not (k[np.isfinite(grid)] >= min_support).any():
        return math.inf, f"no threshold has {min_support} calibration examples at or above it (min_support)"
    return math.inf, "no threshold reaches the target on these examples"


def _why_only_inf(n, k, m, grid, level, min_support, bound):
    """Why conformal risk control kept its promise only by escalating everything: the best finite threshold's bound,
    and (unbounded form) about how many examples that threshold's risk would need."""
    fin = np.isfinite(grid) & (k >= min_support)
    head = f"no finite threshold keeps the promise on these {n} examples"
    if not fin.any():
        return head + f": none has {min_support} calibration examples at or above it (min_support)"
    if bound:
        return head + f" (risk {level:g} per group with its binomial bound)"
    b = (m[fin] + 1) / (n + 1)
    i = int(np.argmin(b))
    out = head + f": the best gives (wrong answered + 1) / (n + 1) = {b[i]:.3f} > {level:g}"
    r = float(m[fin][i]) / n
    if r < level:
        out += f"; at its risk ({r:.3f}) about {math.ceil((1 - level) / (level - r))} examples would be needed"
    return out


def separation(s, ok):
    """Does the signal rank right answers above wrong ones? → {"auroc", "p" (one-sided Mann–Whitney), "n_right",
    "n_wrong", "tested"}; untested (p None) with fewer than MIN_CLASS of either."""
    s, ok = np.asarray(s, float), np.asarray(ok, bool)
    fin = np.isfinite(s)
    a, b = s[fin & ok], s[fin & ~ok]
    out = {"n_right": int(len(a)), "n_wrong": int(len(b)), "auroc": None, "p": None, "tested": False}
    if len(a) and len(b):
        from scipy.stats import mannwhitneyu
        r = mannwhitneyu(a, b, alternative="greater")
        out["auroc"] = float(r.statistic / (len(a) * len(b)))
        if min(len(a), len(b)) >= MIN_CLASS:
            out["p"], out["tested"] = float(r.pvalue), True
    return out


def _stats(s, w, t):
    a = s >= t
    k, m, n = int(a.sum()), float(w[a].sum()), len(s)
    return {"threshold": float(t), "n": n, "answered": k / n if n else 0.0, "error": m / k if k else 0.0,
            "risk": m / n if n else 0.0, "support": k}


class Promise:
    """A calibrated threshold on one signal with the promise it keeps — the result of calibrate(). allows(signal,
    group=None) → answer alone? threshold_of(group) → (threshold, node, info). `report`: what it does on the calibration
    examples; `record()`: what every gated decision records."""
    stateful = False

    def __init__(self, method, level, delta, threshold, nodes, by, report, why, min_group, signal="signal",
                 procedure="bonferroni"):
        self.method, self.level, self.delta = method, float(level), delta
        self.threshold, self.nodes, self.by, self.min_group = float(threshold), nodes, by, min_group
        self.report, self.why, self.signal = report, why, signal
        self.procedure = procedure
        self.promise = promise_text(method, level, delta if (method == "ltt" or nodes) else None, nodes is not None,
                                    len(nodes or {()}), procedure)

    def threshold_of(self, group=None):
        """The threshold for an input of this group (a value or a path; the deepest calibrated group on its path, else
        the rest of the stream) → (threshold, node, info); without groups: (threshold, None, None)."""
        if self.nodes is None:
            return self.threshold, None, None
        from ..calibration import group_path, node_of
        node = node_of(group_path(group), self.nodes)
        return self.nodes[node]["threshold"], node, self.nodes[node]

    def allows(self, signal, group=None):
        """Answer alone at this signal? (False for a missing or non-finite signal.)"""
        if signal is None or not math.isfinite(float(signal)):
            return False
        return float(signal) >= self.threshold_of(group)[0]

    def record(self):
        out = {"method": self.method, "promise": self.promise, "n": self.report["n"], "signal": self.signal}
        out["risk" if self.method == "crc" else "error"] = self.level
        if self.method == "ltt" or (self.nodes is not None and self.delta is not None):
            out["delta"] = self.delta
        if getattr(self, "procedure", "bonferroni") != "bonferroni":
            out["procedure"] = self.procedure
        if self.nodes is not None:
            out["groups"], out["min_group"] = self.by, self.min_group
        if self.report.get("warnings"):
            out["warnings"] = list(self.report["warnings"])
        return out

    def fingerprint(self):
        from ..provenance import digest
        nodes = None if self.nodes is None else sorted((list(k), v["threshold"]) for k, v in self.nodes.items())
        return digest("Promise", self.method, self.level, self.delta, self.threshold, nodes, self.by, self.signal)

    def __repr__(self):
        return f"Promise({self.method}, threshold={self.threshold:.4g}: {self.promise})"


def calibrate(scores, correct, *, max_error=None, max_risk=None, method=None, delta=0.10, groups=None, min_group=100,
              min_support=10, weak="raise", signal="signal", procedure="bonferroni"):
    """A threshold with a promise on any scalar signal: answer alone when signal ≥ threshold. scores: the signal per
    calibration example (None / NaN / −inf: never answered alone; +inf: always — a forced answer); correct: was the answer
    right (booleans). max_risk= → conformal risk control ("crc"), max_error= → learn-then-test ("ltt"), or method="empirical"
    with max_error=; the promises are in the module docstring. delta: learn-then-test's confidence and, with groups, the
    binomial bound of conformal risk control per group (delta=None there: on average per group). groups: one group (a
    value or a path) per example; min_group: a group with fewer examples is pooled with its parent. min_support: a
    threshold must have this many examples at or above it. weak: "raise" (default) or "warn" when the signal does not
    separate right from wrong. procedure: learn-then-test's multiple testing over its grid — "bonferroni" (default: each
    of ≤ 64 thresholds at delta / their number) or "fixed" (fixed-sequence: from the highest threshold down at the full
    delta, stopping at the first that fails) or "fallback" (the same descent, half of delta on the first threshold
    and half spread over the rest, a certified threshold passing its level on; a failure does not end it) — all
    calibration.ltt_certified, the same promise; the sequential ones usually answer more.
    → Promise (its threshold is inf — everything escalates — when none keeps the promise; report["why"] says why)."""
    from ..calibration import check_rate, group_nodes
    error, risk = max_error, max_risk
    if (error is None) == (risk is None):
        raise ValueError("give the promise as max_risk= (P(answered alone and wrong), conformal risk control) or "
                         "max_error= (the error among the answers given alone, learn-then-test)")
    if method is None:
        method = "crc" if risk is not None else "ltt"
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if method == "crc" and risk is None or method != "crc" and error is None:
        raise ValueError(f'method="{method}" takes ' + ("max_risk=" if method == "crc" else "max_error="))
    level = check_rate("risk" if method == "crc" else "error", risk if method == "crc" else error,
                       zero=method == "empirical")
    if method == "ltt" or (groups is not None and delta is not None):
        check_rate("delta", delta)
    from ..calibration import LTT_PROCEDURES
    if procedure not in LTT_PROCEDURES:
        raise ValueError(f"procedure must be one of {LTT_PROCEDURES}, not {procedure!r}")
    if weak not in ("raise", "warn"):
        raise ValueError('weak must be "raise" or "warn"')
    s = np.array([-math.inf if v is None or (isinstance(v, float) and math.isnan(v)) else float(v) for v in scores])
    ok = np.asarray([bool(c) for c in correct], bool)
    if len(s) != len(ok):
        raise ValueError(f"{len(s)} scores and {len(ok)} answers judged")
    if not len(s):
        raise ValueError("calibration needs examples")
    w = (~ok).astype(float)
    notes = []
    sep = separation(s, ok)
    if sep["tested"] and sep["p"] >= SEPARATION_P:
        msg = (f"the signal does not separate right from wrong answers on the calibration examples (AUROC "
               f"{sep['auroc']:.2f}, p = {sep['p']:.2f} ≥ {SEPARATION_P}; {sep['n_right']} right, {sep['n_wrong']} wrong): "
               "a threshold on it would pass the wrong answers as often as the right ones")
        if weak == "raise":
            raise ValueError(msg + ' — use another signal, or weak="warn" to set it anyway')
        warnings.warn(msg, GuaranteeWarning, stacklevel=2)
        notes.append(msg)
    elif not sep["tested"] and sep["n_wrong"] and sep["n_right"]:
        notes.append(f"separation not tested: {sep['n_right']} right and {sep['n_wrong']} wrong among the gated "
                     f"examples (fewer than {MIN_CLASS} of one)")
    if groups is None:
        t, why = _search(s, w, method, level, delta, min_support, procedure=procedure)
        nodes, by = None, None
        rep = _stats(s, w, t)
    else:
        paths = list(groups)
        if len(paths) != len(s):
            raise ValueError(f"{len(s)} scores and {len(paths)} groups")
        node_ix, owner = group_nodes(paths, min_group)
        nodes, why, by = {}, None, "given"
        d = None if delta is None else delta / len(node_ix)
        for node, ix in node_ix.items():
            ix = np.asarray(ix, int)
            if method == "crc":
                tt, wy = _search(s[ix], w[ix], "crc", level, None, min_support, bound_delta=d)
            else:
                tt, wy = _search(s[ix], w[ix], method, level, d if method == "ltt" else None, min_support,
                                 procedure=procedure)
            st = _stats(s[ix], w[ix], tt) if len(ix) else {"threshold": tt, "n": 0, "answered": 0.0, "error": 0.0,
                                                         "risk": 0.0, "support": 0}
            st["pooled"] = sorted({"/".join(map(str, p)) for p, o in zip(map(_path, paths), owner) if o == node and p != node})
            if wy:
                st["why"] = wy
            nodes[node] = st
        t = nodes[()]["threshold"]
        auto = np.array([s[i] >= nodes[owner[i]]["threshold"] for i in range(len(s))])
        rep = {"threshold": t, "n": len(s), "answered": float(auto.mean()),
               "error": float(w[auto].sum() / auto.sum()) if auto.any() else 0.0, "risk": float(w[auto].sum() / len(s)),
               "support": int(auto.sum()), "groups": {k: dict(v) for k, v in nodes.items()}}
    if not math.isfinite(t):
        msg = f"everything escalates: {why or (nodes or {}).get((), {}).get('why') or 'no threshold keeps the promise'}"
        warnings.warn(msg, GuaranteeWarning, stacklevel=2)
        notes.append(msg)
    elif method == "crc" and groups is None and (len(s) + 1) * level < 2:
        msg = (f"too few calibration examples for a risk of {level:g}: with {len(s)} the promise allows no wrong answer "
               f"among the answered, so one more wrong example moves the threshold (at least {math.ceil(2 / level) - 1} "
               "for room of one)")
        warnings.warn(msg, GuaranteeWarning, stacklevel=2)
        notes.append(msg)
    base = float(w.mean())
    rep.update(method=method, separation=sep, base_error=base, warnings=notes,
               promise=promise_text(method, level, delta if (method == "ltt" or groups is not None) else None,
                                    groups is not None, len(nodes or {()}), procedure))
    if method == "ltt":
        rep["procedure"] = procedure
    if method == "crc":
        rep["must_escalate_at_least"] = max(0.0, (base - level) / (1 - level))
    if why:
        rep["why"] = why
    if sep["n_wrong"] + sep["n_right"] < len(s):
        rep["not_gated"] = int(len(s) - sep["n_wrong"] - sep["n_right"])    # forced (+inf) or never answered (−inf)
    return Promise(method, level, delta, t, nodes, by, rep, why, min_group, signal, procedure)


def _path(p):
    from ..calibration import group_path
    return group_path(p)


# ------------------------------------------------------------------------------------------------ on a System's question
class QuestionGuard:
    """A Promise (or another object with its interface) bound to a question of a System: which signal it reads, which answer it
    lets through (answer=: one-sided), how groups are read. Applied to every answer of the question by System.ask."""

    def __init__(self, question, promise, signal=None, answer=None, groups=None):
        self.question, self.promise, self.answer = question, promise, answer
        self.signal = signal if signal is not None else ("probability" if answer is not None else "confidence")
        self.groups = groups
        self._by = None
        if groups is not None and groups != "answer":
            from ..deciders import GroupBy
            self._by = GroupBy(groups)

    def describe_signal(self):
        if callable(self.signal):
            return getattr(self.signal, "__name__", "a function")
        if self.signal == "probability":
            return f"P({self.answer!r})"
        return self.signal

    def fingerprint(self):
        fp = getattr(self, "_fp", None)
        if fp is not None and fp[0] is self.promise:       # cached: a guard does not change once attached
            return fp[1]
        from ..provenance import code_fingerprint, digest
        sig = {"fn": code_fingerprint(self.signal)} if callable(self.signal) else self.signal
        grp = None if self.groups is None else ("answer" if self.groups == "answer" else self._by.describe())
        self._fp = (self.promise, digest("QuestionGuard", self.question, sig, repr(self.answer), grp,
                                         self.promise.fingerprint()))
        return self._fp[1]

    # --- reading the signal and the group
    def value(self, result, vals, trace):
        """The signal of one answer → float, or a reason it cannot be read (str)."""
        sig = self.signal
        if callable(sig):
            try:
                v = sig(result, vals)
            except Exception as e:  # noqa: BLE001 — a signal that fails is a reason to abstain, not a crash
                return f"the signal {self.describe_signal()} raised {type(e).__name__}: {str(e)[:100]}"
            return _number(v, self.describe_signal())
        if sig == "confidence":
            return float(result.confidence)
        if sig == "probability":
            p = result.probs or {}
            for k in (self.answer, str(self.answer)):
                if k in p:
                    return float(p[k])
            return f"the answer's probabilities give no P({self.answer!r})"
        if sig == "act":
            rec = next((r for r in trace.records if r.name == f"answer:{self.question}"), None) if trace is not None else None
            act = (getattr(rec, "extra", None) or {}).get("act") if rec is not None else None
            if act is None:
                return "no act probability: the question's answer is not a model decision with an act head"
            return float(act)
        if sig not in vals and sig.startswith("extra:"):   # a detail the answer's decision recorded (a dotted path)
            rec = next((r for r in trace.records if r.name == f"answer:{self.question}"), None) if trace is not None else None
            v = getattr(rec, "extra", None) if rec is not None else None
            for k in sig[len("extra:"):].split("."):
                v = v.get(k) if isinstance(v, dict) else None
            if v is None:
                return f"the answer's decision recorded no {sig[len('extra:'):]!r}"
            return _number(v, sig)
        if sig not in vals:
            return f"the fact {sig!r} was not computed"
        v = vals[sig]
        return _number(getattr(v, "value", v) if not isinstance(v, (int, float)) else v, sig)

    def group(self, result, vals, answer):
        if self.groups is None:
            return None
        if self.groups == "answer":
            return (str(answer),)
        return self._by.path(vals)

    def would_answer(self, result):
        return self.answer if self.answer is not None else result.answer


def _number(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float, np.floating, np.integer)):
        return f"the signal {name} is {type(v).__name__}, not a number"
    v = float(v)
    return v if math.isfinite(v) else f"the signal {name} is {v}"


def _calibration_rows(system, question, examples, guard, correct, folds=None, seed=0, probs_out=None):
    """Ask the question on labelled examples with its guard off → (signals, right, groups, read failures). probs_out: a
    list that receives ({answer: probability} or None, the label or None) per example, for the label check."""
    from ..runtime import execute, vhash
    q = system.questions[question]
    examples = list(examples)
    if not examples:
        raise ValueError("calibration needs labelled examples [(init_state, correct answer)]")
    heads = [None] * len(examples)
    if folds:
        heads = _fold_heads(system, question, examples, int(folds), seed)
    sig, ok, grp, lost = [], [], [], []
    saved = system.guards.pop(question, None)
    was_head = system.heads.get(question)
    try:
        for i, (st, y) in enumerate(examples):
            if heads[i] is not None:
                system.heads[question] = heads[i]
            p = system._prepare(st, [question], None)
            trace, vals = execute(system.catalog, p.flow, p.state, workers=system.workers, order=p.order,
                                  costs=system.cost_book, policy=p.policy, known=p.known)
            r = system._results(p.questions, p.flow, trace, vals)[0][question]
            if probs_out is not None:
                probs_out.append(_label_row(q, r, y))
            if r.status == "abstain":
                sig.append(-math.inf)
                ok.append(True)                   # never answered alone: right / wrong does not count
                grp.append(None)
                continue
            ans = r.answer if r.status == "forced" else guard.would_answer(r)
            from ..catalog import label_options
            if correct is not None:               # a label with several right answers (AnyOf): right if any is
                import dataclasses
                right = any(bool(correct(dataclasses.replace(r, answer=ans), o)) for o in label_options(y))
            else:
                right = any(vhash(ans) == vhash(q.answer.normalize(o)) for o in label_options(y))
            ok.append(right)
            grp.append(guard.group(r, vals, ans))
            if r.status == "forced":              # a hard check decides: answered whatever the signal
                sig.append(math.inf)
                continue
            v = guard.value(r, vals, trace)
            if isinstance(v, str):
                lost.append(v)
                sig.append(-math.inf)
            else:
                sig.append(v)
    finally:
        if was_head is not None:
            system.heads[question] = was_head
        if saved is not None:
            system.guards[question] = saved
    return sig, ok, grp, lost


def _fold_heads(system, question, examples, folds, seed):
    """Cross-fitting for a question answered by a fast head fitted on these examples: each example gets a head fitted
    on the other folds (the same features and settings) → [head per example]."""
    from ..deciders.heads import FastHead
    head = system.heads.get(question)
    if not isinstance(head, FastHead):
        raise ValueError(f"folds= refits the question's head on part of the examples: {question!r} needs a head fitted with fit "
                         f"(it has {type(head).__name__ if head is not None else 'none'})")
    if folds < 2 or folds > len(examples):
        raise ValueError(f"folds must be between 2 and the number of examples ({len(examples)})")
    q = system.questions[question]
    rows = [system.facts_for(st) for st, _ in examples]
    ans = [q.answer.normalize(y) for _, y in examples]
    order = np.random.default_rng(seed).permutation(len(examples))
    fold = np.empty(len(examples), int)
    fold[order] = np.arange(len(examples)) % folds
    feats = list(getattr(head, "_requested", None) or head.features)
    out = [None] * len(examples)
    for f in range(folds):
        tr = [i for i in range(len(examples)) if fold[i] != f]
        h = FastHead(head.options, lam=head._lam0, pairs=head._pairs0, refit=None).fit([rows[i] for i in tr],
                                                                                         [ans[i] for i in tr], feats)
        for i in range(len(examples)):
            if fold[i] == f:
                out[i] = h
    return out


def guard_question(system, question, examples=None, *, max_error=None, max_risk=None, method=None, delta=0.10, signal=None,
                   answer=None, groups=None, min_group=100, min_support=10, correct=None, folds=None, seed=0,
                   weak="raise", promise=None, corrections=None, sources=None, procedure="bonferroni", label_check=False):
    """Put a calibrated threshold with a stated promise on a question of `system` (System.guarantee): every later answer
    is let through alone only when its signal ≥ the threshold, else the question abstains with the reason; the promise,
    the signal and the threshold are recorded with every answer. examples: [(init_state, correct answer)], not used to fit
    the answer (or folds=k: the question's fitted head is refitted k times, each example scored by a head that did not
    see it — the head that answers is fitted on all of them, so the promise is then approximate). False removes the
    question's guarantee.
    signal: "confidence" (default — the answer's confidence), "act" (the act probability of the model decision that
    answers it), "extra:<path>" (a number the answer's decision recorded in its extra, a dotted path: e.g.
    "extra:systemone.confidence", a System One service's own confidence), a fact name (any number the catalog computes or the input gives), or a function (result, facts) →
    number. answer: one-sided — the question answers this alone when its signal (default: the answer's probability,
    P(answer)) ≥ the threshold, whatever the most probable answer is, and abstains otherwise ("return the query when
    P(right) ≥ 0.43"). correct: a function (result, label) → bool for answers judged otherwise than by equality (a quote
    that overlaps the gold one). groups: a fact name, a hierarchy of fact names, a function of facts, or "answer".
    promise: an already calibrated Promise to attach instead of calibrating here.
    corrections: a TraceStorage (True: the system's storage) whose stored corrections of this question are added to
    `examples` — those from `sources` (default TRUSTED_SOURCES: "human", "outcome", "rule"); sources may add "verified",
    System 2 answers that passed the checks and a guarantee (TraceStorage.save_correction checks that when it stores
    them). This is the one channel that takes verified labels: fed every System 2 answer it vouched for, it let System 1
    answer more within its promise on two tasks of three (docs/best_practices.md). The report's "labels" says
    how many came from where and their stored ids (calibrate again without them to undo it).
    The other arguments, the methods and their promises: calibrate() and the module docstring.
    → the calibration report ({"threshold", "n", "answered", "error", "risk", "support", "separation", "promise", ...};
    for an attached promise, its report).
    label_check (default False, opt-in; it reads the answers already computed, no extra call): the report's "label_issues" —
    the examples whose label the answer's probabilities confidently contradict (solvi.core.calibration.label_issues;
    ids are the examples' positions, the appended corrections= included) — and a line in "warnings" when there are any;
    every label is still used. It needs out-of-sample probabilities: a fast head fitted on these examples has seen the
    labels it is asked to contradict, so without folds= a FastHead question gets a warning to pass folds=k (each example
    is then scored by a head fitted without it).

    Every call is logged in `system.guard_history`, in order: {"question", "time" (seconds since the epoch), "outcome"
    ("installed", "failed" or "removed"), "replaced" (a guarantee was in force before the call), "method", "n",
    "threshold", "answered", "promise" (installed), "error" (failed: the exception's type and message), "kept" (failed: a
    guarantee was in force and stays in force)}. A failed call raises its error and leaves the question as it was:
    the guarantee in force before it (if any) stays in force, with a GuaranteeWarning saying so (since 1.2; before, a
    failed call removed it)."""
    import time
    log = getattr(system, "guard_history", None)
    if log is None:
        log = system.guard_history = []
    guards = getattr(system, "guards", {})
    entry = {"question": question, "time": time.time(), "replaced": question in guards}
    # what a failed attempt puts back: the guarantee in force, its added checkpoints and the question as it was
    before = (guards.get(question), list(getattr(guards.get(question), "added", None) or []),
              getattr(system, "questions", {}).get(question))
    try:
        rep = _guard_question(system, question, examples, max_error=max_error, max_risk=max_risk, method=method,
                              delta=delta, signal=signal, answer=answer, groups=groups, min_group=min_group,
                              min_support=min_support, correct=correct, folds=folds, seed=seed, weak=weak,
                              promise=promise, corrections=corrections, sources=sources, procedure=procedure,
                              label_check=label_check)
    except Exception as e:
        old, added, q = before
        if q is not None:
            system.questions[question] = q
        kept = old is not None
        if kept:
            old.added = added
            system.guards[question] = old
        log.append({**entry, "outcome": "failed", "kept": kept, "error": f"{type(e).__name__}: {e}"})
        if kept:
            warnings.warn(f"System.guarantee({question!r}) failed ({type(e).__name__}: {e}): the guarantee in force "
                          f"before the call stays in force", GuaranteeWarning, stacklevel=3)
        raise
    if examples is False:
        log.append({**entry, "outcome": "removed"})
    else:
        r = rep or {}
        log.append({**entry, "outcome": "installed", "method": r.get("method"), "n": r.get("n"),
                    "threshold": r.get("threshold"), "answered": r.get("answered"), "promise": r.get("promise")})
    return rep


def _label_row(q, r, y):
    """One calibration example for the label check: (the answer's probabilities keyed like normalized answers, the
    normalized label) — None where it cannot be checked (no probabilities, a forced answer, several right answers)."""
    from ..catalog import label_options
    opts = label_options(y)
    if r.status == "forced" or not getattr(r, "probs", None) or len(opts) != 1:
        return None, None
    try:
        return {q.answer.normalize(k): float(v) for k, v in r.probs.items()}, q.answer.normalize(opts[0])
    except (ValueError, TypeError):
        return None, None


def _guard_question(system, question, examples, *, max_error, max_risk, method, delta, signal, answer, groups,
                    min_group, min_support, correct, folds, seed, weak, promise, corrections, sources, procedure="bonferroni",
                    label_check=False):
    """guard_question without the log."""
    if examples is False:
        old = system.guards.pop(question, None)
        if old is not None:
            _drop_checkpoints(system, old)
        return None
    if question not in system.questions:
        raise ValueError(f"no question {question!r} in the system")
    if answer is not None:
        answer = system.questions[question].answer.normalize(answer)
    old = system.guards.pop(question, None)
    if old is not None:                            # a guarantee set before: its required parts go with it
        _drop_checkpoints(system, old)
    guard = QuestionGuard(question, None, signal, answer, groups)
    _add_checkpoints(system, guard)               # the signal's and the groups' facts are computed in every flow
    labels = None
    if corrections is not None:
        examples, labels = _with_corrections(system, question, examples, corrections, sources)
    elif sources is not None:
        raise ValueError("sources= says which stored corrections to read: give corrections= (a TraceStorage, or True)")
    if promise is None:
        if examples is None:
            raise ValueError("give labelled examples [(init_state, correct answer)] to calibrate on, or promise=")
        try:
            rows = [] if label_check else None
            sig, ok, grp, lost = _calibration_rows(system, question, examples, guard, correct, folds, seed, rows)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                promise = calibrate(sig, ok, max_error=max_error, max_risk=max_risk, method=method, delta=delta,
                                    groups=None if groups is None else grp, min_group=min_group, min_support=min_support,
                                    weak=weak, signal=guard.describe_signal(), procedure=procedure)
        except BaseException:
            _drop_checkpoints(system, guard)
            raise
        for c in caught:
            warnings.warn(c.message, c.category, stacklevel=2)
        if lost:
            msg = (f"the signal could not be read for {len(lost)} of {len(sig)} calibration examples (counted as escalated): "
                   f"{lost[0]}")
            if 2 * len(lost) > len(sig):
                _drop_checkpoints(system, guard)
                raise ValueError(msg + " — most of them: check signal=")
            promise.report["warnings"].append(msg)
            warnings.warn(msg, GuaranteeWarning, stacklevel=2)
        if rows is not None:                       # confident learning on the labels: reported, never applied
            from ..calibration import label_issues, label_warning
            issues = label_issues([p for p, _ in rows], [y for _, y in rows])
            promise.report["label_issues"] = issues
            lw = label_warning(issues, len(sig))
            if lw:
                promise.report["warnings"].append(lw)
            from ..deciders.heads import FastHead
            if not folds and isinstance(system.heads.get(question), FastHead):
                promise.report["warnings"].append(
                    "the label check ran on in-sample probabilities (a fast head fitted on these examples has seen their "
                    "labels, so it lists fewer issues): pass folds=k to score each example by a head fitted without it")
        if groups is not None:
            promise.by = "answer" if groups == "answer" else guard._by.label()
        if folds:
            promise.report["folds"] = int(folds)
            promise.report["warnings"].append(f"cross-fitted ({folds} folds): the thresholds were set on heads fitted "
                                              "without each example; the head that answers saw them all, so the promise "
                                              "is approximate")
    if labels is not None:
        promise.report["labels"] = labels
    guard.promise = promise
    system.guards[question] = guard
    return promise.report


def _with_corrections(system, question, examples, corrections, sources):
    """examples + the stored corrections of `question` from `sources` → (examples, {"examples", "corrections":
    {source: n}, "ids"}); a correction from another source is left out (counted in "skipped")."""
    from ..sources import TRUSTED_SOURCES, VERIFIED, UntrustedLabel, check_source
    store = system.storage if corrections is True else corrections
    if store is None or not hasattr(store, "corrections"):
        raise ValueError("corrections= is a TraceStorage (or True: the system's storage, which it does not have)")
    sources = TRUSTED_SOURCES if sources is None else tuple([sources] if isinstance(sources, str) else sources)
    bad = [x for x in sources if x not in TRUSTED_SOURCES + (VERIFIED,)]
    if bad:
        raise UntrustedLabel(f"sources= {bad} are not label sources ({', '.join(TRUSTED_SOURCES + (VERIFIED,))})")
    out = list(examples or [])
    n0, by, ids, skipped = len(out), {}, [], 0
    for c in store.corrections():
        if c["question"] != question:
            continue
        if c["source"] not in sources:
            skipped += 1
            continue
        check_source(c["source"], accept=sources)
        out.append((c["init"], c["answer"]))
        by[c["source"]] = by.get(c["source"], 0) + 1
        ids.append(c["id"])
    return out, {"examples": n0, "corrections": by, "skipped": skipped, "ids": ids}


def _add_checkpoints(system, guard):
    """The facts the guard reads (a fact signal, group facts) that the catalog computes become required parts of the
    question, so every flow computes them; the ones added are remembered and removed with the guarantee."""
    import dataclasses
    q = system.questions[guard.question]
    need = ([guard.signal] if isinstance(guard.signal, str) else []) + (guard._by.names if guard._by is not None else [])
    add = [f for f in need if f in system.catalog.parts and f not in q.requires]
    guard.added = add
    if add:
        system.questions[guard.question] = dataclasses.replace(q, requires=list(q.requires) + add)


def _drop_checkpoints(system, guard):
    import dataclasses
    add = getattr(guard, "added", None)
    q = system.questions.get(guard.question)
    if add and q is not None:
        system.questions[guard.question] = dataclasses.replace(q, requires=[c for c in q.requires if c not in add])
    guard.added = []


# ------------------------------------------------------------------------------------------------ at ask time
def apply_guards(system, results, trace, vals, flow, live=True):
    """Gate the answers of the guarded questions (called by System.ask and System.answers_of). live: record each verdict
    in the trace (and let a stateful gate observe the signal); False — re-derive the verdicts of a recorded trace (a
    stateful gate's recorded threshold is taken as it was)."""
    from ..runtime import Record
    for qn, guard in list(system.guards.items()):
        r = results.get(qn)
        if r is None:
            continue
        old = next((x for x in trace.records if x.kind == "guard" and x.name == f"guard:{qn}"), None)
        if r.status != "ok":                       # abstained already, or forced by a hard check: the guard does not apply
            continue
        v = guard.value(r, vals, trace)
        ans = guard.would_answer(r)
        grp = guard.group(r, vals, ans)
        p = guard.promise
        if p.stateful and not live and old is not None:
            thr, node, info, state = (old.extra or {}).get("threshold"), None, None, (old.extra or {}).get("state")
            thr = math.inf if thr is None else float(thr)
        else:
            state = p.state() if p.stateful else None
            thr, node, info = p.threshold_of(grp)
        reason = None
        if guard.groups is not None and grp is None:
            reason = f"group unknown: the thresholds are per group ({p.by}) and this input does not give it"
            ok, why, s = False, reason, None if isinstance(v, str) else v
        elif isinstance(v, str):
            reason = f"guarantee signal not read: {v}"
            ok, why, s = False, reason, None
        else:
            s = v
            ok = s >= thr
            why = None if ok else f"{guard.describe_signal()} {s:.4g} < {thr:.4g} (threshold of the guarantee: {p.promise})"
        g = p.record()
        g.update(signal=guard.describe_signal(), value=s, threshold=thr if math.isfinite(thr) else None, answered=bool(ok),
                 fingerprint=guard.fingerprint())
        if node is not None:
            from ..calibration import group_name
            g.update(group=list(grp), applied=list(node), n=info["n"])
            g["promise"] = (f"{g['promise']}; here: group {group_name(node)} (threshold {thr:.4g}, n = {info['n']})"
                            + (f", pooled: {group_name(grp)} had fewer than {p.min_group} examples" if tuple(node) != tuple(grp)
                               else ""))
        if state is not None:
            g["state"] = state
        if reason is not None:
            g["reason"] = reason
        if live:
            from ..chain import append as _append
            from ..runtime import vhash
            inputs = {guard.signal: vhash(vals[guard.signal])} if isinstance(guard.signal, str) and guard.signal in vals else {}
            _append(trace, Record(step=0, kind="guard", name=f"guard:{qn}", inputs=inputs, value=bool(ok), confidence=1.0,
                                  provenance="computed", extra=_plain(g)), len(flow.steps))
            if p.stateful and s is not None:
                from types import SimpleNamespace
                p.observe(s, SimpleNamespace(answer=ans, confidence=r.confidence, status="ok" if ok else "abstain"))
        extra = dict(r.extra or {})
        extra["guarantee"] = g
        if ok:
            if guard.answer is not None and r.answer != guard.answer:
                r.why = f"answered {guard.answer!r} at {guard.describe_signal()} {s:.4g} ≥ {thr:.4g} (most probable: {r.answer!r}); " + r.why
                r.answer, r.confidence = guard.answer, float(s)
            r.extra = extra
            continue
        from ..runtime import Result
        results[qn] = Result(None, r.confidence, f"{why}; would have answered {ans!r} ({r.why})", "abstain", r.probs,
                             r.provenance, r.source, "low_confidence", r.repaired, r.kind, r.evidence, extra)


def _plain(d):
    """A record's extra must hash and store as JSON: tuples → lists, inf → None."""
    if isinstance(d, dict):
        return {str(k): _plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_plain(x) for x in d]
    if isinstance(d, float) and not math.isfinite(d):
        return None
    if isinstance(d, (np.floating, np.integer)):
        return d.item()
    return d


# ------------------------------------------------------------------------------------------------ as a file
GUARANTEE_FORMAT = "solvi guarantee v1"


def _keyed(d):
    """{group path (a tuple): info} → [[path as a list, info]] (JSON has no tuple keys); other values as they are."""
    return [[list(k), v] for k, v in d.items()] if isinstance(d, dict) else d


def _unkeyed(d):
    return {tuple(k): dict(v) for k, v in d} if isinstance(d, list) else d


def _answer_data(q):
    from ..schema import dump
    return dump(q.answer, "json") if q.answer is not None else None


def guarantee_record(system, question):
    """The guarantee in force on a question as JSON-ready data: the Promise (method, level, delta, thresholds per group,
    report), how the guard reads its signal, which answer it lets through and its groups, and what it was calibrated
    for — the catalog's fingerprint, the question's answer type and the guard's own fingerprint (what every gated
    decision records)."""
    from ..calibfile import _norm
    guard = system.guards.get(question)
    if guard is None:
        raise KeyError(f"no guarantee on {question!r}: calibrate one first (System.guarantee)")
    p = guard.promise
    if type(p) is not Promise:
        raise ValueError(f"the guarantee on {question!r} is a {type(p).__name__}, not a calibrated Promise: only a "
                         "Promise is saved")
    rep = dict(p.report)
    if isinstance(rep.get("groups"), dict):
        rep["groups"] = _keyed(rep["groups"])
    fn_signal, fn_groups = callable(guard.signal), callable(guard.groups)
    from ... import __version__
    return _norm({"format": GUARANTEE_FORMAT, "question": question, "answer_type": _answer_data(system.questions[question]),
                  "catalog": system.fingerprint()["catalog"], "guard": guard.fingerprint(),
                  "signal": None if fn_signal else guard.signal, "signal_function": fn_signal,
                  "answer": guard.answer, "groups": None if fn_groups else guard.groups, "groups_function": fn_groups,
                  "promise": {"method": p.method, "level": p.level, "delta": p.delta, "threshold": p.threshold,
                              "nodes": _keyed(p.nodes), "by": p.by, "report": rep, "why": p.why,
                              "min_group": p.min_group, "signal": p.signal, "procedure": p.procedure},
                  "solvi": __version__})


def save_guarantee(system, question, path):
    """Write the guarantee in force on a question to a JSON file (System.save_guarantee) → path."""
    import json
    from ..calibfile import _enc
    rec = guarantee_record(system, question)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_enc(rec), fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")
    return path


def load_guarantee(system, path, *, signal=None, groups=None, strict=True):
    """Attach a guarantee saved by save_guarantee (System.load_guarantee) → its report. The guard it rebuilds must have the
    saved fingerprint — the same promise, signal, answer and groups — so stored decisions replay against it as against
    the one calibrated; a signal or grouping that was a function is passed again (`signal=`, `groups=`; its code is
    fingerprinted and must match; for a saved fact or keyword, `signal=` / `groups=` raise). The fingerprints catch a file
    that does not fit (another catalog, an edit that changes the promise inconsistently), not a deliberate forgery that
    recomputes them: keep the file like code. strict (default): refuse a file made for another catalog (the signal is computed by
    the catalog's code) or for another answer type of the question; strict=False attaches it anyway (the guard's
    fingerprint is still checked)."""
    import json
    from ..calibfile import _dec
    with open(path, encoding="utf-8") as fh:
        rec = _dec(json.load(fh))
    if rec.get("format") != GUARANTEE_FORMAT:
        raise ValueError(f"{path}: not a solvi guarantee file (format {rec.get('format')!r}, expected {GUARANTEE_FORMAT!r})")
    q = rec["question"]
    if q not in system.questions:
        raise ValueError(f"{path}: a guarantee on {q!r}, which this system does not ask")
    if strict:
        if rec.get("answer_type") != json.loads(json.dumps(_answer_data(system.questions[q]), default=str)):
            raise ValueError(f"{path}: made for another answer type of {q!r} ({rec.get('answer_type')})")
        fp = system.fingerprint()["catalog"]
        if rec.get("catalog") != fp:
            raise ValueError(f"{path}: calibrated on catalog #{rec.get('catalog')}, this system's catalog is #{fp}: the "
                             "parts that compute the answer or its signal changed — calibrate again (System.guarantee), "
                             "or load with strict=False")
    for given, saved, what in ((signal, rec.get("signal_function"), "signal"), (groups, rec.get("groups_function"), "groups")):
        if saved and given is None:
            raise ValueError(f"{path}: the guarantee's {what} was a function: pass it again ({what}=that_function)")
        if given is not None and not saved:
            raise ValueError(f"{path}: the guarantee's {what} is {rec.get(what)!r}, not a function: {what}= is only for a "
                             f"{what} that was a function")
    sig = signal if rec.get("signal_function") else rec.get("signal")
    grp = groups if rec.get("groups_function") else rec.get("groups")
    pr = rec["promise"]
    rep = dict(pr["report"])
    if isinstance(rep.get("groups"), list):
        rep["groups"] = _unkeyed(rep["groups"])
    promise = Promise(pr["method"], pr["level"], pr["delta"], pr["threshold"], _unkeyed(pr["nodes"]), pr["by"], rep,
                      pr["why"], pr["min_group"], pr["signal"], pr.get("procedure", "bonferroni"))
    if isinstance(grp, list):
        grp = list(grp)
    answer = rec.get("answer")
    if answer is not None:
        answer = system.questions[q].answer.normalize(answer)
    guard = QuestionGuard(q, promise, sig, answer, grp)
    if guard.fingerprint() != rec.get("guard"):
        raise ValueError(f"{path}: the rebuilt guarantee is #{guard.fingerprint()}, the saved one #{rec.get('guard')}: "
                         "its signal, answer or groups differ (a function passed again must be the same code)")
    return guard_question(system, q, None, promise=promise, signal=sig, answer=answer, groups=grp)


__all__ = ["GuaranteeWarning", "Promise", "QuestionGuard", "apply_guards", "calibrate", "guard_question", "guarantee_record",
           "load_guarantee", "promise_text", "replay_guard", "save_guarantee", "separation"]
