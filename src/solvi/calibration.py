"""Is a confidence honest? Reliability, expected calibration error and coverage at a target accuracy — for any model or
question (solvi answers, a decider, a head).

    from solvi.calibration import coverage_at, ece, reliability, threshold_for
    coverage_at(conf, correct, 0.9)     # share of cases you can answer automatically at ≥ 90% accuracy
    threshold_for(conf, correct, 0.9)   # the confidence threshold that gives it (e.g. for Question(min_confidence=...))

`conf` — confidences in [0, 1]; `correct` — whether each answer was right (booleans or 0/1). `evaluate(system, question,
examples)` collects both from a System on labelled examples (abstentions count as not covered).

Thresholds with a guarantee (used by DecisionPart.act_guard, calibrate_for(method="ltt") and conformal):

    crc_threshold(score, wrong, risk=0.1)       # P(answered alone and wrong) ≤ 10% of all questions
    ltt_threshold(score, wrong, error=0.1)      # error among the answered ≤ 10%, with probability ≥ 90%

They hold for inputs like the calibration examples, not under a shift of domain: calibrate on your own labelled data."""
from __future__ import annotations

import math

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


# --- thresholds with a guarantee (conformal risk control, learn-then-test) and conformal answer sets
# Measured on the shipped deciders: an act threshold chosen for "10% error" on one data set gave 32–52%
# errors among the answers it let through on others; the guarantees below hold only on data like the calibration examples.

def crc_threshold(score, wrong, risk=0.10):
    """Conformal risk control: the lowest threshold t such that (Σ 1[score ≥ t and wrong] + 1) / (n + 1) ≤ risk; answer
    alone when score ≥ t. Guarantee (inputs exchangeable with the calibration examples): P(answered alone and wrong) ≤ risk
    — a share of ALL questions, not of the answered ones. inf when the examples are too few or too hard for the risk."""
    s, w = _arrays(score, wrong)
    n = len(s)
    order = np.argsort(-s, kind="stable")
    ss, cw = s[order], np.cumsum(w[order])
    for t in np.concatenate([np.unique(s), [np.inf]]):       # the risk does not grow with t: take the first that holds
        k = int(np.searchsorted(-ss, -t, side="right"))       # cases with score ≥ t
        if ((cw[k - 1] if k else 0.0) + 1) / (n + 1) <= risk + 1e-12:
            return float(t)
    return float("inf")


def _binom_cdf(k, n, p):
    """P(Binomial(n, p) ≤ k), without scipy."""
    if n == 0:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(int(k) + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


def check_rate(name, v, zero=False):
    """A target error rate / risk / delta must be strictly between 0 and 1 (0 cannot be certified from finitely many
    examples; 1 promises nothing) → ValueError with the name otherwise. zero=True also allows 0 (an empirical target:
    no error on the calibration examples)."""
    try:
        ok = (0 <= float(v) if zero else 0 < float(v)) and float(v) < 1
    except (TypeError, ValueError):
        ok = False
    if not ok:
        raise ValueError(f"{name} must be a number {'in [0, 1)' if zero else 'strictly between 0 and 1'}, not {v!r}")
    return float(v)


LTT_GRID = 64          # thresholds ltt_threshold tries by default (the Bonferroni correction grows with their number)


def ltt_grid(score, size=LTT_GRID):
    """The default learn-then-test grid: the distinct finite scores, or `size` of them at evenly spaced quantiles when
    there are more. It reads the scores only, never the labels, so fixing it on the calibration examples keeps the
    guarantee; it follows the scores' own scale (an LLM's confidence packed near 1 as well as an act probability)."""
    u = np.unique(np.asarray(score, float))
    u = u[np.isfinite(u)]
    if len(u) > size:
        u = np.unique(u[np.round(np.linspace(0, len(u) - 1, size)).astype(int)])
    return u


def ltt_threshold(score, wrong, error=0.10, delta=0.10, grid=None):
    """Learn-then-test: the lowest threshold t on a fixed grid such that the error AMONG the cases answered alone
    (score ≥ t) is ≤ error with probability ≥ 1 − delta over the calibration examples (binomial test, Bonferroni over the
    grid). A stronger promise than crc_threshold, so it often allows no automatic answers at all (inf). grid=None: at
    most 64 quantiles of the distinct calibration scores (ltt_grid; before 0.7 a fixed linspace(0.2, 0.995, 32), which
    let nothing through for a decider whose scores sit above 0.995)."""
    check_rate("error", error)
    check_rate("delta", delta)
    s, w = _arrays(score, wrong)
    grid = ltt_grid(s) if grid is None else np.asarray(grid, float)
    if not len(grid):
        return float("inf")
    for t in sorted(grid):
        auto = s >= t
        k = int(auto.sum())
        if k and _binom_cdf(float(w[auto].sum()), k, error) <= delta / len(grid):
            return float(t)
    return float("inf")


def conformal_quantile(scores, alpha):
    """The ⌈(n + 1)(1 − alpha)⌉-th smallest score; inf when there are too few (n < 1/alpha − 1)."""
    s = np.sort(np.asarray(scores, float))
    k = int(math.ceil((len(s) + 1) * (1 - alpha) - 1e-12))
    return float("inf") if k > len(s) or not len(s) else float(s[k - 1])


def set_scores(p, ordinal=False, has_unknown=False):
    """Non-conformity of every answer of one question, from its probabilities p (the last one "not stated" when
    has_unknown): 1 − p (LAC); ordinal=True — for scores and numbers — the mass added before an answer while an interval
    grows from the mode towards the more probable neighbour ("not stated" competes as one more candidate), so the answer
    set is always one contiguous interval."""
    p = np.asarray(p, float)
    if not ordinal:
        return 1.0 - p
    nb = len(p) - 1 if has_unknown else len(p)
    start = int(np.argmax(p))
    order, unk_in = [start], has_unknown and start == nb
    lo = hi = None if unk_in else start
    while len(order) < len(p):
        cand = []
        if lo is None:
            b = int(np.argmax(p[:nb]))
            cand.append((p[b], b))
        else:
            if lo > 0:
                cand.append((p[lo - 1], lo - 1))
            if hi < nb - 1:
                cand.append((p[hi + 1], hi + 1))
        if has_unknown and not unk_in:
            cand.append((p[nb], nb))
        _, c = max(cand, key=lambda t: (t[0], -t[1]))
        order.append(c)
        if has_unknown and c == nb:
            unk_in = True
        elif lo is None:
            lo = hi = c
        else:
            lo, hi = min(lo, c), max(hi, c)
    s, cum = np.empty(len(p)), 0.0
    for c in order:
        s[c] = cum
        cum += p[c]
    return s


# --- group-wise guarantees: a threshold per group of a hierarchy (domain → task), after HG-CRC (arXiv 2607.24562)
# A threshold calibrated on the whole stream holds on average over it; inside a hard group the share answered alone and
# wrong can be far above the risk (tests/test_guarantees.py simulates it). Here every group with enough examples gets its
# own threshold; a smaller group is pooled with the rest of its parent (the parent's threshold is calibrated on exactly
# those pooled examples); the whole stream takes what is left.

def group_path(g):
    """A group as a path from the top of the hierarchy: a tuple of strings. A scalar is a one-level group, a tuple or list
    a path (domain, task); the path stops at the first None; None alone is the whole stream ()."""
    if g is None:
        return ()
    if isinstance(g, (tuple, list)):
        out = []
        for x in g:
            if x is None:
                break
            out.append(str(x))
        return tuple(out)
    return (str(g),)


def group_nodes(paths, min_group=100):
    """Which groups get a threshold of their own. Deepest level first: a group whose examples not yet taken by a group
    under it number at least `min_group` becomes a node and takes them; the whole stream () takes the rest. → ({node:
    [example indices]}, the node of each example). Depends on the group sizes only, not on the labels."""
    paths = [group_path(p) for p in paths]
    owner = [None] * len(paths)
    nodes = {}
    for d in range(max((len(p) for p in paths), default=0), 0, -1):
        cand = {}
        for i, p in enumerate(paths):
            if owner[i] is None and len(p) >= d:
                cand.setdefault(p[:d], []).append(i)
        for node in sorted(cand):
            if len(cand[node]) >= min_group:
                nodes[node] = cand[node]
                for i in cand[node]:
                    owner[i] = node
    nodes[()] = [i for i, o in enumerate(owner) if o is None]
    return nodes, [() if o is None else o for o in owner]


def node_of(path, nodes):
    """The node whose threshold applies to an input of this group: the deepest node on its path (a new or small group
    falls back to its parent's, then to the whole stream's)."""
    p = group_path(path)
    for d in range(len(p), -1, -1):
        if p[:d] in nodes:
            return p[:d]
    return ()


def loss_budget(n, risk, delta=None):
    """The most examples of n that may be answered alone and wrong for a threshold to certify `risk`: conformal risk
    control (delta=None: (k + 1) / (n + 1) ≤ risk, a promise on average) or a binomial test at level delta (k with
    P(Binomial(n, risk) ≤ k) ≤ delta: the risk ≤ `risk` with probability ≥ 1 − delta). −1: no threshold can."""
    if not 0 < risk < 1:
        raise ValueError("risk must be between 0 and 1")
    if delta is None:
        return int(math.floor(risk * (n + 1) - 1 + 1e-9))
    if n == 0:
        return -1
    lp, lq, cdf, k = math.log(risk), math.log1p(-risk), 0.0, -1
    while k < n:
        j = k + 1
        term = math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * lp + (n - j) * lq)
        if cdf + term > delta + 1e-15:
            break
        cdf, k = cdf + term, j
    return k


def certify_groups(losses_of, paths, risk=0.10, min_group=100, delta=0.10):
    """Group-wise risk control over a hierarchy: `losses_of(indices)` → (candidate thresholds ascending, answered-alone-
    and-wrong count at each — non-increasing) for those examples. Each node (group_nodes) takes the lowest candidate whose
    count is within loss_budget(n, risk, delta / nodes) — Bonferroni over the nodes, so with probability ≥ 1 − delta the
    promise holds in every group at once; delta=None: conformal risk control per group (each group on average, no
    correction needed). → ({node: {"threshold", "n", "index"}}, node of each example); inf where no threshold certifies."""
    nodes, owner = group_nodes(paths, min_group)
    d = None if delta is None else delta / len(nodes)
    out = {}
    for node, ix in nodes.items():
        grid, k = losses_of(ix)
        budget = loss_budget(len(ix), risk, d)
        good = np.where(np.asarray(k, float) <= budget)[0] if budget >= 0 else []
        g = int(good[0]) if len(good) else None
        out[node] = {"threshold": float(grid[g]) if g is not None else math.inf, "n": len(ix), "index": g}
    return out, owner


def _signal_losses(score, wrong):
    s, w = _arrays(score, wrong)

    def losses_of(ix):
        ss, ww = s[ix], w[ix]
        grid = np.concatenate([np.unique(ss), [np.inf]])
        order = np.argsort(-ss, kind="stable")
        cw = np.concatenate([[0.0], np.cumsum(ww[order])])
        k = cw[np.searchsorted(-ss[order], -grid, side="right")]       # wrong among the cases with score ≥ t
        return grid, k
    return losses_of


def group_thresholds(score, wrong, groups, risk=0.10, min_group=100, delta=0.10):
    """certify_groups for one signal (answer alone when score ≥ the threshold of the input's node) → {node: threshold};
    apply with node_of(group, nodes). groups: one per example — a value, a path (domain, task) or None."""
    nodes, _ = certify_groups(_signal_losses(score, wrong), groups, risk, min_group, delta)
    return {k: v["threshold"] for k, v in nodes.items()}
