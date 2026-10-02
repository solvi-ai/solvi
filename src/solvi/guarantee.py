"""A guarantee on any signal: a calibrated threshold with a stated promise on a question's answer — given by a rule, a
fitted head (fit / fit_fast), a model decision — on a fact the catalog computes (a trust score, an agreement share), or
on any scalar outside a System.

    report = system.guarantee("pair", examples, risk=0.01)          # conformal risk control on the answer's confidence
    report = system.guarantee("pair", examples, error=0.02)         # learn-then-test: error among the answered ≤ 2%
    report = system.guarantee("act", examples, risk=0.03, signal="trust", correct=judge)     # a computed fact
    report = system.guarantee("correct", examples, error=0.3, answer="yes")   # one-sided: "yes" alone when P(yes) ≥ t

    from solvi.guarantee import calibrate
    p = calibrate(scores, correct, error=0.05)                      # any scalar: p.threshold, p.allows(s), p.report

examples: [(init_state, correct answer)] held out from whatever fitted the answer (or folds=k for a fast head fitted
on them). The question is asked on each (nothing stored, nothing counted), the signal and right / wrong are collected,
and the threshold is chosen by one of three methods — each makes a different promise, for inputs like the calibration
examples (exchangeable with them: the same stream, not a new domain; solvi.openset for inputs from outside it):

    method       parameter  promise
    "crc"        risk=r     P(answered alone and wrong) ≤ r — a share of ALL inputs, answered or escalated; on average
                            over calibration sets (conformal risk control: (wrong answered + 1) / (n + 1) ≤ r)
    "ltt"        error=e    the error AMONG the answers given alone ≤ e, with probability ≥ 1 − delta over the
                            calibration set (learn-then-test: a binomial test per threshold, Bonferroni over ≤ 64)
    "empirical"  error=e    none: the error among the answered was ≤ e on the calibration examples only

risk= selects "crc", error= selects "ltt" (method="empirical" to ask for the plain one). groups= gives a threshold per
group (a fact name, a hierarchy of fact names, a function of facts, or "answer": the answer the question would give —
"among the inputs answered 'match', ..."), with the promise inside every group (after HG-CRC; see solvi.calibration).

What is checked before a threshold is set, so that a promise is never made on a signal that cannot carry it:
- separation: the signal must rank the right answers above the wrong ones on the calibration examples (a one-sided
  Mann–Whitney test at 5%); one that does not — a model approving its own answers at chance — raises (weak="warn":
  a warning, and the warning is recorded in the guarantee);
- support: a threshold must have at least min_support calibration examples at or above it (default 10); none that
  does → everything escalates, and the report says why;
- feasibility: with n examples, conformal risk control cannot certify a risk below 1 / (n + 1) — the report says so.

The report gives, on the calibration examples, both "risk" (answered alone and wrong, a share of all) and "error"
(among the answered), the share answered, the threshold's support and the separation (AUROC, p). Every answer of a
guarded question records the promise, the signal's value and the threshold in its result (extra["guarantee"]) and in
the trace (a hashed record "guard:<question>"); below the threshold the question abstains with the reason. Replay with
the System re-derives the verdict; a guarantee changed since the decision is a mismatch."""
from __future__ import annotations

import math
import warnings

import numpy as np

METHODS = ("crc", "ltt", "empirical")
SEPARATION_P = 0.05            # the signal must separate right from wrong at this level (one-sided Mann–Whitney)
MIN_CLASS = 5                  # separation is tested only with at least this many right and wrong examples


class GuaranteeWarning(UserWarning):
    """A guarantee was set on a signal that did not show it can carry it (weak="warn"), or on too few examples."""


# ------------------------------------------------------------------------------------------------ the promise texts
def promise_text(method, level, delta=None, groups=False, n_groups=1):
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
    return (f"error among the answers given alone ≤ {level:g}{where} with probability ≥ {1 - delta:g}, for inputs like "
            "the calibration examples")


# ------------------------------------------------------------------------------------------------ choosing a threshold
def _curve(s, w):
    """Candidate thresholds (the distinct finite signals and inf, ascending) → (grid, answered count, wrong answered).
    A signal of +inf is always answered (a forced answer), −inf never (an abstention before the threshold)."""
    grid = np.concatenate([np.unique(s[np.isfinite(s)]), [np.inf]])
    order = np.argsort(-s, kind="stable")
    ss, cw = s[order], np.concatenate([[0.0], np.cumsum(w[order])])
    k = np.searchsorted(-ss, -grid, side="right")             # cases with signal ≥ t
    return grid, k, cw[k]


def _search(s, w, method, level, delta, min_support, bound_delta=None):
    """The lowest threshold that keeps the method's promise on these examples → (threshold, why or None). bound_delta:
    conformal risk control as a binomial bound at that level (groups with delta)."""
    from .calibration import _binom_cdf, loss_budget, ltt_grid
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
            for t in g:
                i = pos[float(t)]
                good[i] = ok_support[i] and k[i] > 0 and _binom_cdf(float(m[i]), int(k[i]), level) <= delta / len(g)
    hit = np.flatnonzero(good)
    if len(hit):
        return float(grid[hit[0]]), None
    if method == "crc" and bound_delta is None and (n + 1) * level < 1:
        return math.inf, f"{n} examples cannot certify a risk of {level:g}: conformal risk control needs at least {math.ceil(1 / level) - 1}"
    if method == "empirical" and ((m <= level * np.maximum(k, 1) + 1e-9) & (k > 0) & np.isfinite(grid)).any():
        return math.inf, (f"the thresholds that reach the target rest on fewer than {min_support} calibration examples "
                          "(min_support)")
    if not (k[np.isfinite(grid)] >= min_support).any():
        return math.inf, f"no threshold has {min_support} calibration examples at or above it (min_support)"
    return math.inf, "no threshold reaches the target on these examples"


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

    def __init__(self, method, level, delta, threshold, nodes, by, report, why, min_group, signal="signal"):
        self.method, self.level, self.delta = method, float(level), delta
        self.threshold, self.nodes, self.by, self.min_group = float(threshold), nodes, by, min_group
        self.report, self.why, self.signal = report, why, signal
        self.promise = promise_text(method, level, delta if (method == "ltt" or nodes) else None, nodes is not None,
                                    len(nodes or {()}))

    def threshold_of(self, group=None):
        """The threshold for an input of this group (a value or a path; the deepest calibrated group on its path, else
        the rest of the stream) → (threshold, node, info); without groups: (threshold, None, None)."""
        if self.nodes is None:
            return self.threshold, None, None
        from .calibration import group_path, node_of
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
        if self.nodes is not None:
            out["groups"], out["min_group"] = self.by, self.min_group
        if self.report.get("warnings"):
            out["warnings"] = list(self.report["warnings"])
        return out

    def fingerprint(self):
        from .provenance import digest
        nodes = None if self.nodes is None else sorted((list(k), v["threshold"]) for k, v in self.nodes.items())
        return digest("Promise", self.method, self.level, self.delta, self.threshold, nodes, self.by, self.signal)

    def __repr__(self):
        return f"Promise({self.method}, threshold={self.threshold:.4g}: {self.promise})"


def calibrate(scores, correct, *, error=None, risk=None, method=None, delta=0.10, groups=None, min_group=100,
              min_support=10, weak="raise", signal="signal"):
    """A threshold with a promise on any scalar signal: answer alone when signal ≥ threshold. scores: the signal per
    calibration example (None / NaN / −inf: never answered alone; +inf: always — a forced answer); correct: was the answer
    right (booleans). risk= → conformal risk control ("crc"), error= → learn-then-test ("ltt"), or method="empirical"
    with error=; the promises are in the module docstring. delta: learn-then-test's confidence and, with groups, the
    binomial bound of conformal risk control per group (delta=None there: on average per group). groups: one group (a
    value or a path) per example; min_group: a group with fewer examples is pooled with its parent. min_support: a
    threshold must have this many examples at or above it. weak: "raise" (default) or "warn" when the signal does not
    separate right from wrong. → Promise (its threshold is inf — everything escalates — when none keeps the promise;
    report["why"] says why)."""
    from .calibration import check_rate, group_nodes
    if (error is None) == (risk is None):
        raise ValueError("give the promise as risk= (P(answered alone and wrong), conformal risk control) or error= "
                         "(the error among the answers given alone, learn-then-test)")
    if method is None:
        method = "crc" if risk is not None else "ltt"
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, not {method!r}")
    if method == "crc" and risk is None or method != "crc" and error is None:
        raise ValueError(f'method="{method}" takes ' + ("risk=" if method == "crc" else "error="))
    level = check_rate("risk" if method == "crc" else "error", risk if method == "crc" else error,
                       zero=method == "empirical")
    if method == "ltt" or (groups is not None and delta is not None):
        check_rate("delta", delta)
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
        t, why = _search(s, w, method, level, delta, min_support)
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
                tt, wy = _search(s[ix], w[ix], method, level, d if method == "ltt" else None, min_support)
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
    base = float(w.mean())
    rep.update(method=method, separation=sep, base_error=base, warnings=notes,
               promise=promise_text(method, level, delta if (method == "ltt" or groups is not None) else None,
                                    groups is not None, len(nodes or {()})))
    if method == "crc":
        rep["must_escalate_at_least"] = max(0.0, (base - level) / (1 - level))
    if why:
        rep["why"] = why
    if sep["n_wrong"] + sep["n_right"] < len(s):
        rep["not_gated"] = int(len(s) - sep["n_wrong"] - sep["n_right"])    # forced (+inf) or never answered (−inf)
    return Promise(method, level, delta, t, nodes, by, rep, why, min_group, signal)


def _path(p):
    from .calibration import group_path
    return group_path(p)


# ------------------------------------------------------------------------------------------------ on a System's question
class QuestionGuard:
    """A Promise (or a solvi.openset.OpenSetGate) bound to a question of a System: which signal it reads, which answer it
    lets through (answer=: one-sided), how groups are read. Applied to every answer of the question by System.ask."""

    def __init__(self, question, promise, signal=None, answer=None, groups=None):
        self.question, self.promise, self.answer = question, promise, answer
        self.signal = signal if signal is not None else ("probability" if answer is not None else "confidence")
        self.groups = groups
        self._by = None
        if groups is not None and groups != "answer":
            from .decide import GroupBy
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
        from .provenance import code_fingerprint, digest
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


def _calibration_rows(system, question, examples, guard, correct, folds=None, seed=0):
    """Ask the question on labelled examples with its guard off → (signals, right, groups, read failures)."""
    from .runtime import execute, vhash
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
                                  costs=system.costs, policy=p.policy, known=p.known)
            r = system._results(p.questions, p.flow, trace, vals)[0][question]
            if r.status == "abstain":
                sig.append(-math.inf)
                ok.append(True)                   # never answered alone: right / wrong does not count
                grp.append(None)
                continue
            ans = r.answer if r.status == "forced" else guard.would_answer(r)
            if correct is not None:
                import dataclasses
                right = bool(correct(dataclasses.replace(r, answer=ans), y))
            else:
                right = vhash(ans) == vhash(q.answer.normalize(y))
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
    from .fast import FastHead
    head = system.heads.get(question)
    if not isinstance(head, FastHead):
        raise ValueError(f"folds= refits the question's head on part of the examples: {question!r} needs a fit_fast head "
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


def guard_question(system, question, examples=None, *, error=None, risk=None, method=None, delta=0.10, signal=None,
                   answer=None, groups=None, min_group=100, min_support=10, correct=None, folds=None, seed=0,
                   weak="raise", promise=None):
    """Put a calibrated threshold with a stated promise on a question of `system` (System.guarantee): every later answer
    is let through alone only when its signal ≥ the threshold, else the question abstains with the reason; the promise,
    the signal and the threshold are recorded with every answer. examples: [(init_state, correct answer)], not used to fit
    the answer (or folds=k: the question's fit_fast head is refitted k times, each example scored by a head that did not
    see it — the head that answers is fitted on all of them, so the promise is then approximate). False removes the
    question's guarantee.
    signal: "confidence" (default — the answer's confidence), "act" (the act probability of the model decision that
    answers it), a fact name (any number the catalog computes or the input gives), or a function (result, facts) →
    number. answer: one-sided — the question answers this alone when its signal (default: the answer's probability,
    P(answer)) ≥ the threshold, whatever the most probable answer is, and abstains otherwise ("return the query when
    P(right) ≥ 0.43"). correct: a function (result, label) → bool for answers judged otherwise than by equality (a quote
    that overlaps the gold one). groups: a fact name, a hierarchy of fact names, a function of facts, or "answer".
    promise: an already calibrated Promise or solvi.openset.OpenSetGate to attach instead of calibrating here.
    The other arguments, the methods and their promises: calibrate() and the module docstring.
    → the calibration report ({"threshold", "n", "answered", "error", "risk", "support", "separation", "promise", ...};
    for an attached promise, its report)."""
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
    if old is not None:                            # a guarantee set before: its checkpoints go with it
        _drop_checkpoints(system, old)
    guard = QuestionGuard(question, None, signal, answer, groups)
    _add_checkpoints(system, guard)               # the signal's and the groups' facts are computed in every flow
    if promise is None:
        if examples is None:
            raise ValueError("give labelled examples [(init_state, correct answer)] to calibrate on, or promise=")
        try:
            sig, ok, grp, lost = _calibration_rows(system, question, examples, guard, correct, folds, seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                promise = calibrate(sig, ok, error=error, risk=risk, method=method, delta=delta,
                                    groups=None if groups is None else grp, min_group=min_group, min_support=min_support,
                                    weak=weak, signal=guard.describe_signal())
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
        if groups is not None:
            promise.by = "answer" if groups == "answer" else guard._by.label()
        if folds:
            promise.report["folds"] = int(folds)
            promise.report["warnings"].append(f"cross-fitted ({folds} folds): the thresholds were set on heads fitted "
                                              "without each example; the head that answers saw them all, so the promise "
                                              "is approximate")
    guard.promise = promise
    system.guards[question] = guard
    return promise.report


def _add_checkpoints(system, guard):
    """The facts the guard reads (a fact signal, group facts) that the catalog computes become checkpoints of the
    question, so every flow computes them; the ones added are remembered and removed with the guarantee."""
    import dataclasses
    q = system.questions[guard.question]
    need = ([guard.signal] if isinstance(guard.signal, str) else []) + (guard._by.names if guard._by is not None else [])
    add = [f for f in need if f in system.catalog.parts and f not in q.checkpoints]
    guard.added = add
    if add:
        system.questions[guard.question] = dataclasses.replace(q, checkpoints=list(q.checkpoints) + add)


def _drop_checkpoints(system, guard):
    import dataclasses
    add = getattr(guard, "added", None)
    q = system.questions.get(guard.question)
    if add and q is not None:
        system.questions[guard.question] = dataclasses.replace(q, checkpoints=[c for c in q.checkpoints if c not in add])
    guard.added = []


# ------------------------------------------------------------------------------------------------ at ask time
def apply_guards(system, results, trace, vals, flow, live=True):
    """Gate the answers of the guarded questions (called by System.ask and System.answers_of). live: record each verdict
    in the trace (and let a stateful gate observe the signal); False — re-derive the verdicts of a recorded trace (a
    stateful gate's recorded threshold is taken as it was)."""
    from .runtime import Record
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
            from .decide import group_name
            g.update(group=list(grp), applied=list(node), n=info["n"])
            g["promise"] = (f"{g['promise']}; here: group {group_name(node)} (threshold {thr:.4g}, n = {info['n']})"
                            + (f", pooled: {group_name(grp)} had fewer than {p.min_group} examples" if tuple(node) != tuple(grp)
                               else ""))
        if state is not None:
            g["state"] = state
        if reason is not None:
            g["reason"] = reason
        if live:
            from .system import _append
            from .runtime import vhash
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
        from .runtime import Result
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


def replay_guard(r, system, vals):
    """A recorded guard verdict (kind "guard"): its inputs must be the recorded facts, the verdict must follow from the
    recorded signal and threshold, and with the System its guarantee must be the one the question has now."""
    from .runtime import Mismatch, vhash
    bad = []
    for f, h in r.inputs.items():
        if f not in vals or vhash(vals[f]) != h:
            bad.append(Mismatch(r.step, r.name, f"input {f} does not match the recorded one", "integrity"))
    e = r.extra or {}
    s, t = e.get("value"), e.get("threshold")
    follows = False if e.get("reason") or s is None else float(s) >= (math.inf if t is None else float(t))
    if bool(r.value) != follows:
        bad.append(Mismatch(r.step, r.name, f"verdict {r.value!r} does not follow from the recorded signal {s} and "
                                            f"threshold {t}" + (f" ({e['reason']})" if e.get("reason") else ""), "integrity"))
    if system is not None:
        g = getattr(system, "guards", {}).get(r.name[6:])
        if g is None:
            bad.append(Mismatch(r.step, r.name, "the question has no guarantee now (removed since this decision)", "recompute"))
        elif g.fingerprint() != e.get("fingerprint"):
            bad.append(Mismatch(r.step, r.name, "the question's guarantee changed since this decision (recalibrated or "
                                                "another signal)", "recompute"))
    return bad


__all__ = ["GuaranteeWarning", "Promise", "QuestionGuard", "apply_guards", "calibrate", "guard_question", "promise_text",
           "replay_guard", "separation"]
