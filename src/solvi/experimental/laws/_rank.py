"""Ranking candidate laws where each check is expensive (see the package docstring): features from the Buffer's
bitsets, a bag-of-atoms scorer trained on synthetic tasks drawn from the grammar, the raw-feature control, the joint
statistic of the whole condition (no training), a prior order over the condition's own features for a buffer with no
rows yet and the hand-over from it to the joint statistic, and a proposer that only reorders another one."""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Callable, Sequence

import numpy as np

from ._core import Buffer, Candidate, Grammar, LawSet, Transition, _key

FEATURES = ("precision", "recall", "jaccard", "anti", "rate", "base_rate", "pair_precision", "pair_recall",
            "pair_jaccard")


def _effect_mask(lawset, buffer, action, effect):
    if lawset is not None:
        return lawset._effect_mask(buffer, action, effect)
    return buffer.where(action, ("effect", "observed"), _key(effect), lambda t: t.observed == effect)


def _stats(m, E, allowed):
    n_all = allowed.bit_count()
    nm, ne = m.bit_count(), E.bit_count()
    tp = (m & E).bit_count()
    union = (m | E).bit_count()
    prec = tp / nm if nm else 0.0
    rec = tp / ne if ne else 0.0
    jac = tp / union if union else 0.0
    rest = n_all - nm
    anti = ((allowed & ~m & ~E).bit_count() / rest) if rest else 0.0
    return prec, rec, jac, anti, (nm / n_all if n_all else 0.0)


def atom_features(lawset: LawSet | None, buffer: Buffer, action, effect, *, exclude=None,
                  top_pairs: int = 64) -> tuple[list[str], np.ndarray]:
    """Per atom of the grammar, against the rows of `action` where `effect` predicts right (the effect mask) on the
    logged rows (without the group `exclude`): precision, recall, Jaccard, anti-match (rows without the atom that are
    without the effect, among those without the atom), the atom's rate, the effect's base rate, and the best precision /
    recall / Jaccard of a pair containing it — pairs only among the atoms of highest Jaccard, at most `top_pairs` pairs,
    one atom per feature. → (atom names in grammar order, an (atoms, 9) array). A fixed length for any grammar."""
    g = buffer.grammar
    allowed = buffer.allowed(action, exclude)
    E = _effect_mask(lawset, buffer, action, effect) & allowed
    n_all = allowed.bit_count()
    base = E.bit_count() / n_all if n_all else 0.0
    names = [a.name for a in g.atoms]
    masks = {x: buffer.match(action, (x,)) & allowed for x in names}
    X = np.zeros((len(names), len(FEATURES)))
    for i, x in enumerate(names):
        X[i, :5] = _stats(masks[x], E, allowed)
        X[i, 5] = base
    t = 2
    while (t + 1) * t // 2 <= top_pairs and t < len(names):
        t += 1
    top = sorted(range(len(names)), key=lambda i: (-X[i, 2], i))[:t]
    for i, j in combinations(sorted(top), 2):
        if g.feature[names[i]] == g.feature[names[j]]:
            continue
        p, r, jac, _, _ = _stats(masks[names[i]] & masks[names[j]], E, allowed)
        for k in (i, j):
            X[k, 6:9] = np.maximum(X[k, 6:9], (p, r, jac))
    return names, X


def candidate_features(cand: Candidate, buffer: Buffer, lawset: LawSet | None = None, *, exclude=None,
                       top_pairs: int = 64) -> np.ndarray:
    """One candidate's features, a fixed length whatever its size: its number of atoms, the effect's base rate, and the
    sum, the minimum and the maximum over its atoms of `atom_features` (zeros for the empty condition)."""
    names, X = atom_features(lawset, buffer, cand.action, cand.effect, exclude=exclude, top_pairs=top_pairs)
    return _cand_vec(tuple(cand.condition), {x: i for i, x in enumerate(names)}, X)


def _cand_vec(cond, index, X):
    rows = X[[index[x] for x in cond]] if cond else np.zeros((0, X.shape[1]))
    base = X[0, 5] if len(X) else 0.0
    if not cond:
        z = np.zeros(X.shape[1])
        return np.concatenate([[0.0, base], z, z, z])
    return np.concatenate([[float(len(cond)), base], rows.sum(0), rows.min(0), rows.max(0)])


class _ByAtom:
    """A scorer whose score for a condition is the sum of a per-atom score over its atoms (bag of atoms: no positions)."""
    top_pairs = 64

    def atom_scores(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def score(self, candidates: Sequence[Candidate], lawset: LawSet | None, buffer: Buffer, exclude=None) -> list[float]:
        """A score per candidate (higher = check earlier)."""
        cache: dict = {}
        out = []
        for c in candidates:
            k = (_key(c.action), _key(c.effect))
            if k not in cache:
                names, X = atom_features(lawset, buffer, c.action, c.effect, exclude=exclude, top_pairs=self.top_pairs)
                cache[k] = (dict(zip(names, self.atom_scores(X))))
            s = cache[k]
            out.append(float(sum(s[x] for x in c.condition)))
        return out


class FeatureRanker(_ByAtom):
    """The control: the raw statistic, no training. score(condition) = Σ over its atoms of (stat − center), stat one of
    STATS ("jaccard" by default), so an atom below `center` lowers the score."""
    STATS = ("precision", "recall", "jaccard")

    def __init__(self, stat: str = "jaccard", center: float = 0.5, top_pairs: int = 64):
        if stat not in self.STATS:
            raise ValueError(f"stat must be one of {self.STATS}")
        self.stat, self.center, self.top_pairs = stat, float(center), int(top_pairs)

    def atom_scores(self, X):
        return X[:, FEATURES.index(self.stat)] - self.center


class BagOfAtomsRanker(_ByAtom):
    """A numpy logistic regression from an atom's features to "this atom is in the true condition"; score(condition) =
    Σ over its atoms of the atom's log-odds (under independence, the log of the odds that this is the condition, up to
    a constant). Trained by `fit` on synthetic tasks drawn from the grammar itself: a random condition labels logged
    states, the features are read off the buffer, the target is the condition. Seeded and deterministic (full-batch
    gradient descent from zero weights)."""

    def __init__(self, seed: int = 0, l2: float = 1e-3, steps: int = 400, lr: float = 1.0, top_pairs: int = 64):
        self.seed, self.l2, self.steps, self.lr, self.top_pairs = int(seed), float(l2), int(steps), float(lr), \
            int(top_pairs)
        self.w = np.zeros(len(FEATURES))
        self.b = 0.0
        self.trained_on = 0

    def fit(self, grammar: Grammar, sample: Callable[[random.Random], Any], *, tasks: int = 200, rows: int = 300,
            noise: float = 0.0) -> "BagOfAtomsRanker":
        """Train on `tasks` synthetic tasks: `sample(rng)` → a state drawn as the logged states are (seeded rng)."""
        rng = random.Random(self.seed)
        Xs, ys = [], []
        for _ in range(tasks):
            t = synthetic_task(grammar, sample, rng, rows=rows, noise=noise)
            names, X = atom_features(t.lawset, t.buffer, t.action, t.observed, top_pairs=self.top_pairs)
            true = set(grammar.true(t.state, t.action))
            for i, x in enumerate(names):
                if x in true:                       # the atoms a candidate for this mismatch can hold
                    Xs.append(X[i])
                    ys.append(1.0 if x in t.condition else 0.0)
        return self.fit_arrays(np.array(Xs), np.array(ys), tasks=tasks)

    def fit_arrays(self, X: np.ndarray, y: np.ndarray, tasks: int = 0) -> "BagOfAtomsRanker":
        """The regression itself, on atom feature rows and their labels."""
        w, b = np.zeros(X.shape[1]), 0.0
        n = max(1, len(y))
        for _ in range(self.steps):
            z = np.clip(X @ w + b, -30, 30)
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - y
            w = w - self.lr * (X.T @ g / n + self.l2 * w)
            b = b - self.lr * float(g.mean())
        self.w, self.b, self.trained_on = w, float(b), int(tasks)
        return self

    def atom_scores(self, X):
        return X @ self.w + self.b


class JointRanker:
    """The joint statistic of the whole condition, no training: a candidate's score is the Jaccard between the logged
    rows its whole condition matches and the rows where its effect predicts right (`buffer.match(action, condition)`
    against the effect mask, both over the rows without the group `exclude`); equal Jaccard → fewer atoms first, then
    the inner order. The bag-of-atoms scorers add per-atom scores and so cannot see that two atoms together pick out the
    effect's rows when neither does alone, or that a correlated twin of a true atom adds nothing; this one scores the
    conjunction itself. A statistic over the logged rows: with no rows every score is equal and the order is the inner
    one (the blind order). `effect_mask` is the rows a score compares against — override it for a control."""

    def effect_mask(self, lawset: LawSet | None, buffer: Buffer, action, effect, exclude=None) -> int:
        """The rows of `action` (without the group `exclude`) where `effect` predicts right."""
        return _effect_mask(lawset, buffer, action, effect) & buffer.allowed(action, exclude)

    def jaccard(self, candidates: Sequence[Candidate], lawset: LawSet | None, buffer: Buffer,
                exclude=None) -> list[float]:
        """The Jaccard per candidate (0 where neither its condition nor its effect has a row)."""
        cache: dict = {}
        out = []
        for c in candidates:
            k = (_key(c.action), _key(c.effect))
            if k not in cache:
                cache[k] = (buffer.allowed(c.action, exclude), self.effect_mask(lawset, buffer, c.action, c.effect,
                                                                                exclude))
            allowed, E = cache[k]
            m = buffer.match(c.action, tuple(c.condition)) & allowed
            union = (m | E).bit_count()
            out.append((m & E).bit_count() / union if union else 0.0)
        return out

    def score(self, candidates: Sequence[Candidate], lawset: LawSet | None, buffer: Buffer, exclude=None) -> list[float]:
        """A score per candidate (higher = check earlier): the Jaccard minus a tie-break by the number of atoms small
        enough never to reorder two different Jaccards (they differ by at least 1 / rows²)."""
        if not candidates:
            return []
        jac = self.jaccard(candidates, lawset, buffer, exclude)
        actions = {_key(c.action): c.action for c in candidates}
        rows = max(buffer.allowed(a, exclude).bit_count() for a in actions.values())
        longest = max(len(tuple(c.condition)) for c in candidates)
        eps = 1.0 / ((rows + 1) ** 2 * (longest + 1))
        return [j - eps * len(tuple(c.condition)) for j, c in zip(jac, candidates)]


class RankedProposer:
    """Another proposer's candidates, reordered by `scorer.score(candidates, lawset, buffer)` (highest first, ties in
    the inner order). Never adds, drops or changes a candidate: only the order, which is what a probe budget spends
    (the order in which split / discriminate receive hypotheses). The buffer: the logged rows the features are read
    from; the held-out test stays the judge. Any scorer with that `score` works (FeatureRanker, BagOfAtomsRanker,
    JointRanker, PriorRanker, HandoverRanker); `scored_by` after a call: the scorer's `last` when it has one (a
    HandoverRanker's "prior" / "joint"), else its class name."""

    def __init__(self, inner, scorer, buffer: Buffer, *, exclude=None):
        self.inner, self.scorer, self.buffer, self.exclude = inner, scorer, buffer, exclude
        self.name = getattr(inner, "name", "ranked")
        self.scored_by: str | None = None

    def propose(self, lawset, state, action, observed):
        cands = list(self.inner.propose(lawset, state, action, observed))
        if len(cands) < 2:
            return cands
        scores = self.scorer.score(cands, lawset, self.buffer, exclude=self.exclude)
        self.scored_by = getattr(self.scorer, "last", None) or type(self.scorer).__name__
        order = sorted(range(len(cands)), key=lambda i: (-scores[i], i))
        return [cands[i] for i in order]


# ───────────────────────────────────────────── synthetic tasks
@dataclass
class SyntheticTask:
    """A task drawn from the grammar: a buffer of logged rows labelled by a random condition ("on" where it holds, "off"
    elsewhere, flipped with probability `noise`), a law set predicting "off", and a mismatch — a state where the
    condition holds, observed "on"."""
    buffer: Buffer
    lawset: LawSet
    state: Any
    action: Any
    observed: Any
    condition: tuple


def _off(state, action):
    return "off"


def synthetic_task(grammar: Grammar, sample: Callable[[random.Random], Any], rng: random.Random, *, rows: int = 300,
                   noise: float = 0.0, size: int | None = None, action: Any = "act", groups: int = 4,
                   min_support: int = 3) -> SyntheticTask:
    """One synthetic task (see SyntheticTask): a condition of `size` atoms (default: uniform in 1..k) with at least
    `min_support` matching rows (redrawn otherwise, up to 50 times), rows from `sample(rng)` in `groups` groups."""
    for _ in range(50):
        k = size if size is not None else rng.randint(1, grammar.k)
        cond = grammar.random_condition(rng, k)
        states = [sample(rng) for _ in range(rows)]
        hold = [set(cond) <= set(grammar.true(s, action)) for s in states]
        if sum(hold) >= min_support:
            break
    buf = Buffer(grammar)
    for i, (s, h) in enumerate(zip(states, hold)):
        y = h if rng.random() >= noise else not h
        buf.add(Transition(s, action, "on" if y else "off", group=f"g{i % groups}"))
    state = next((s for s, h in zip(states, hold) if h), None)
    for _ in range(1000):
        s = sample(rng)
        if set(cond) <= set(grammar.true(s, action)):
            state = s
            break
    return SyntheticTask(buf, LawSet(grammar, base=_off), state, action, "on", tuple(cond))


# ───────────────────────────────────────────── a prior order (no rows) and the hand-over
_OPS = (("!=", "!="), (">=", ">="), ("<=", "<="), ("==", "="), ("=", "="), (">", ">"), ("<", "<"))


def atom_kind(name: str) -> str:
    """An atom's comparison type, read off its name: the first of != >= <= == = > < it contains ("==" reads as "="),
    else "bool"."""
    for op, kind in _OPS:
        if op in name:
            return kind
    return "bool"


class _Counts:
    """How often each atom / feature / kind / feature pair that is true in the mismatched state is in the condition."""

    def __init__(self):
        self.true: Counter = Counter()
        self.inside: Counter = Counter()

    def add(self, grammar: Grammar, true: Sequence[str], condition: Sequence[str]):
        feat = grammar.feature
        cond = set(condition)
        tf = sorted({feat[x] for x in true})
        cf = {feat[x] for x in cond}
        for x in true:
            self.true["a", x] += 1
            self.true["k", atom_kind(x)] += 1
            if x in cond:
                self.inside["a", x] += 1
                self.inside["k", atom_kind(x)] += 1
        for f in tf:
            self.true["f", f] += 1
            self.inside["f", f] += f in cf
        for f, h in combinations(tf, 2):
            self.true["p", f, h] += 1
            self.inside["p", f, h] += f in cf and h in cf


class PriorRanker:
    """A prior order for candidate conditions that reads no rows: a linear score over features of the condition itself,
    learned from synthetic tasks drawn from the grammar — for a buffer with no rows yet (then the row statistics give
    the blind order). The features of a condition: its number of atoms (one-hot 0..k) and the sums over its atoms of
    four log-rates learned from the tasks' true conditions — the atom's prior usefulness (how often the atom is in the
    true condition when it is true in the mismatched state), its feature's, its kind's (the comparison type read off the
    name: `atom_kind`), and over its pairs of features their co-occurrence in true conditions beyond the two features'
    own rates (a log ratio; 0 for independent features). The rates are smoothed towards the overall rate with
    `smoothing` pseudo-counts; an atom, feature or kind never seen scores the overall rate.

    `fit` draws `tasks` synthetic tasks (`synthetic_task`, the same signature as BagOfAtomsRanker.fit; `sizes=(lo, hi)`
    draws condition sizes uniform in lo..hi as a setting does) and fits the weights by a softmax over each task's
    candidates (`grammar.conditions` of the atoms true in the mismatched state) with the true condition as the target:
    Newton's method from zero weights on standardized features (L2 `l2`, at most `steps` steps), the rates of each half of the tasks read
    from the other half (so a task's own condition never feeds its features), the final rates from all tasks. Seeded
    and deterministic. `shuffle_labels=True` is the control: each task's target replaced by one of its candidates drawn
    at random (seeded) — a prior that learns nothing should do no better than the blind order. Unfitted: every score
    0 (the inner order).

    The fallback (`fallback=True`, the default): a prior that learned nothing must never be worse than the blind order.
    `fit` measures, on its own training tasks, the mean checks to the true condition under weights fitted on the other
    half of the tasks (`cv["prior"]`), under a fit on shuffled labels measured the same way (`cv["shuffled"]`) and in
    the blind order (`cv["blind"]`: the grammar's order, by length then name); unless the first is at least `margin`
    (5%) below both others, `learned` is False and every score is 0 — the inner order. `learned` is None with
    `fallback=False` (the raw control)."""

    def __init__(self, seed: int = 0, *, l2: float = 1e-3, steps: int = 50, smoothing: float = 2.0,
                 shuffle_labels: bool = False, fallback: bool = True, margin: float = 0.05):
        if not 0.0 <= float(margin) < 1.0:
            raise ValueError("margin must be in [0, 1)")
        self.seed, self.l2, self.steps, self.smoothing = int(seed), float(l2), int(steps), float(smoothing)
        self.shuffle_labels, self.fallback, self.margin = bool(shuffle_labels), bool(fallback), float(margin)
        self.learned: bool | None = None
        self.cv: dict[str, float] = {}
        self.grammar: Grammar | None = None
        self.k = 0
        self.feature_names: list[str] = []
        self.weights = np.zeros(0)
        self.mean = np.zeros(0)
        self.scale = np.ones(0)
        self.usefulness: dict[str, dict] = {"atom": {}, "feature": {}, "kind": {}, "pair": {}}
        self._rates: dict = {}
        self._base = 0.0
        self.trained_on = 0

    # fitting
    def fit(self, grammar: Grammar, sample: Callable[[random.Random], Any], *, tasks: int = 200, rows: int = 300,
            noise: float = 0.0, sizes: tuple[int, int] | None = None) -> "PriorRanker":
        """Train on `tasks` synthetic tasks: `sample(rng)` → a state drawn as the logged states are (seeded rng)."""
        rng = random.Random(self.seed)
        ts = []
        for _ in range(tasks):
            size = rng.randint(*sizes) if sizes else None
            ts.append(synthetic_task(grammar, sample, rng, rows=rows, noise=noise, size=size))
        return self.fit_tasks(grammar, ts)

    def fit_tasks(self, grammar: Grammar, tasks: Sequence[Any]) -> "PriorRanker":
        """Train on given tasks: anything with `state`, `action` and `condition` (a SyntheticTask, a logged mismatch
        whose law is known). ValueError for none."""
        examples, truths = [], []
        rng = random.Random(self.seed * 7919 + 1)
        for t in tasks:
            true = grammar.true(t.state, t.action)
            conds = grammar.conditions(true)
            truth = tuple(sorted(t.condition))
            target = rng.choice([c for c in conds if c] or conds) if self.shuffle_labels else truth
            examples.append((true, conds, target))
            truths.append(truth)
        if not examples:
            raise ValueError("PriorRanker needs at least one task")
        self.grammar, self.k = grammar, grammar.k
        self.feature_names = [f"atoms={i}" for i in range(self.k + 1)] + ["atom", "feature", "kind", "pair"]
        folds = self._design(grammar, examples, truths)
        X = np.concatenate([f[0] for f in folds])
        seg = np.concatenate([folds[0][1], folds[1][1] + len(folds[0][0])]) if len(folds[1][1]) else folds[0][1]
        y = np.concatenate([f[2] for f in folds])
        self.weights, self.mean, self.scale = self._fit_softmax(X, seg, y)
        self.learned = None
        self.cv = {}
        if self.fallback:
            ref = random.Random(self.seed * 7919 + 2)
            shuffled = [(tr, cs, ref.choice([c for c in cs if c] or cs)) for tr, cs, _ in examples]
            blind = [cs.index(tt) + 1 if tt in cs else len(cs) + 1 for (_, cs, _), tt in zip(examples, truths)]
            self.cv = {"prior": self._cv_checks(folds), "shuffled": self._cv_checks(self._design(grammar, shuffled, truths)),
                       "blind": float(np.mean(blind))}
            self.learned = bool(self.cv["prior"] <= (1 - self.margin) * min(self.cv["blind"], self.cv["shuffled"]))
        self._set_rates(grammar, examples)
        self.trained_on = len(examples)
        return self

    def _design(self, grammar, examples, truths):
        """Per fold (the tasks at even / odd positions): the candidates' features with the rates read from the other
        fold, segment starts, the target indicator and the index of the true condition per task."""
        parts = [list(zip(examples, truths))[0::2], list(zip(examples, truths))[1::2]]
        out = []
        for i, part in enumerate(parts):
            other = [e for e, _ in (parts[1 - i] or part)]
            self._set_rates(grammar, other)
            X, seg, y, idx = [], [], [], []
            for (true, conds, target), truth in part:
                seg.append(len(X))
                idx.append(conds.index(truth) if truth in conds else -1)
                for c in conds:
                    X.append(self._phi(c))
                    y.append(1.0 if c == target else 0.0)
            out.append((np.array(X).reshape(len(X), len(self.feature_names)), np.array(seg, dtype=int), np.array(y),
                        idx))
        return out

    def _cv_checks(self, folds):
        """Mean checks to the true condition on each fold under the weights fitted on the other fold (ties in the
        candidates' order)."""
        checks = []
        for i in (0, 1):
            Xt, st, yt, _ = folds[1 - i] if len(folds[1 - i][1]) else folds[i]
            Xe, se, _, idx = folds[i]
            if not len(se):
                continue
            w, mean, scale = self._fit_softmax(Xt, st, yt)
            z = ((Xe - mean) / scale) @ w
            ends = list(se[1:]) + [len(Xe)]
            for start, end, j in zip(se, ends, idx):
                if j < 0:
                    checks.append(end - start + 1)
                    continue
                s = z[start:end]
                checks.append(1 + int((s > s[j]).sum()) + int((s[:j] == s[j]).sum()))
        return float(np.mean(checks))

    def _set_rates(self, grammar, examples):
        n = _Counts()
        for true, _, target in examples:
            n.add(grammar, true, target)
        tot_true = sum(v for key, v in n.true.items() if key[0] == "a")
        tot_in = sum(v for key, v in n.inside.items() if key[0] == "a")
        self._base = math.log((tot_in + 1.0) / (tot_true + 2.0))
        a, base = self.smoothing, math.exp(self._base)
        rates = {}
        for key, v in n.true.items():
            if key[0] != "p":
                rates[key] = math.log((n.inside[key] + a * base) / (v + a))
        for key, v in n.true.items():
            if key[0] == "p":
                both = math.exp(rates["f", key[1]] + rates["f", key[2]])
                rates[key] = math.log((n.inside[key] + a * both) / (v + a)) - math.log(both)
        self._rates = rates
        self._feat = dict(grammar.feature)
        u = self.usefulness = {"atom": {}, "feature": {}, "kind": {}, "pair": {}}
        names = {"a": "atom", "f": "feature", "k": "kind", "p": "pair"}
        for key, v in rates.items():
            u[names[key[0]]][key[1] if len(key) == 2 else key[1:]] = v

    def _phi(self, cond) -> list[float]:
        r, base = self._rates, self._base
        size = [0.0] * (self.k + 1)
        size[min(len(cond), self.k)] = 1.0
        feats = sorted({self._feat.get(x, x) for x in cond})
        atom = sum(r.get(("a", x), base) for x in cond)
        feat = sum(r.get(("f", f), base) for f in feats)
        kind = sum(r.get(("k", atom_kind(x)), base) for x in cond)
        pair = sum(r.get(("p", f, h), 0.0) for f, h in combinations(feats, 2))
        return size + [atom, feat, kind, pair]

    def _fit_softmax(self, X, seg, y):
        mean = X.mean(0)
        sd = X.std(0)
        scale = np.where(sd > 1e-12, sd, 1.0)
        Z = (X - mean) / scale
        ids = np.repeat(np.arange(len(seg)), np.diff(np.append(seg, len(Z))))
        n = len(seg)

        def loss_grad_hess(w):
            z = Z @ w
            m = np.maximum.reduceat(z, seg)
            e = np.exp(z - m[ids])
            tot = np.add.reduceat(e, seg)
            p = e / tot[ids]
            nll = float((np.log(tot) + m).sum() - (z * y).sum()) / n + 0.5 * self.l2 * float(w @ w)
            g = Z.T @ (p - y) / n + self.l2 * w
            M = np.add.reduceat(p[:, None] * Z, seg)
            H = ((Z * p[:, None]).T @ Z - M.T @ M) / n + self.l2 * np.eye(len(w))
            return nll, g, H

        w = np.zeros(Z.shape[1])
        f, g, H = loss_grad_hess(w)
        for _ in range(self.steps):              # Newton's method with a backtracking line search
            step = np.linalg.solve(H, g)
            t = 1.0
            while True:
                w2 = w - t * step
                f2, g2, H2 = loss_grad_hess(w2)
                if f2 <= f - 1e-4 * t * float(g @ step) or t < 1e-6:
                    break
                t *= 0.5
            done = f - f2 < 1e-10
            w, f, g, H = w2, f2, g2, H2
            if done:
                break
        return w, mean, scale

    # scoring
    def score(self, candidates: Sequence[Candidate], lawset: LawSet | None, buffer: Buffer, exclude=None) -> list[float]:
        """A score per candidate (higher = check earlier); reads no rows of `buffer` (its grammar names the features of
        atoms added after the fit)."""
        if not candidates:
            return []
        if self.grammar is None or not len(self.weights) or self.learned is False:
            return [0.0] * len(candidates)
        for x, f in buffer.grammar.feature.items():
            self._feat.setdefault(x, f)
        X = np.array([self._phi(tuple(c.condition)) for c in candidates])
        return [float(v) for v in ((X - self.mean) / self.scale) @ self.weights]


def _rank01(scores: Sequence[float]) -> np.ndarray:
    """Scores → their rank among themselves mapped to [0, 1] (lowest 0, highest 1; equal scores share their mean rank;
    all equal → 0)."""
    x = np.asarray(scores, dtype=float)
    n = len(x)
    if n < 2:
        return np.zeros(n)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks / (n - 1)


class HandoverRanker:
    """From the prior to the joint statistic as the logged rows arrive. A prior is a prior, the row statistic is evidence,
    and evidence takes over gradually.

    mode="blend" (the default): score = the joint statistic's Jaccard (in [0, 1]) + w(rows) · the prior's score mapped
    to [0, 1] by its rank among the candidates (so the two scales do not fight), w(rows) = n0 / (n0 + rows) — 1 with no
    rows (the prior's order), ½ at n0 rows, → 0 as rows grow; equal blends → fewer atoms first, then the inner order (as
    JointRanker's ties). A prior with `learned` False (PriorRanker's fallback) leaves the joint statistic alone (the
    blind order with no rows). mode="switch": the prior while the action has fewer than `min_rows` logged rows, the
    joint statistic from then on. `rows`: the logged rows of the candidates' action without the group `exclude` (several
    actions → the fewest). `last`: "blend", "prior" or "joint" — what scored the last call; `scored`: a Counter."""

    MODES = ("blend", "switch")

    def __init__(self, prior, joint=None, *, mode: str = "blend", n0: float = 20.0, min_rows: int = 20):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if int(min_rows) < 0:
            raise ValueError("min_rows must be ≥ 0")
        if not float(n0) > 0:
            raise ValueError("n0 must be > 0")
        self.prior, self.joint = prior, joint if joint is not None else JointRanker()
        self.mode, self.n0, self.min_rows = mode, float(n0), int(min_rows)
        self.last: str | None = None
        self.scored: Counter = Counter()

    def rows(self, candidates: Sequence[Candidate], buffer: Buffer, exclude=None) -> int:
        """The logged rows the hand-over counts: the fewest among the candidates' actions."""
        actions = {_key(c.action): c.action for c in candidates}
        return min((buffer.allowed(a, exclude).bit_count() for a in actions.values()), default=0)

    def weight(self, rows: int) -> float:
        """The prior's weight in the blend: n0 / (n0 + rows)."""
        return self.n0 / (self.n0 + rows)

    def score(self, candidates: Sequence[Candidate], lawset: LawSet | None, buffer: Buffer, exclude=None) -> list[float]:
        """A score per candidate (see the class)."""
        if not candidates:
            return []
        n = self.rows(candidates, buffer, exclude)
        if self.mode == "switch":
            who = "joint" if n >= self.min_rows else "prior"
            scorer = self.joint if who == "joint" else self.prior
            out = scorer.score(candidates, lawset, buffer, exclude=exclude)
        elif getattr(self.prior, "learned", True) is False:
            who = "joint"
            out = self.joint.score(candidates, lawset, buffer, exclude=exclude)
        else:
            who = "blend"
            jac = np.asarray(self.joint.jaccard(candidates, lawset, buffer, exclude), dtype=float)
            pri = _rank01(self.prior.score(candidates, lawset, buffer, exclude=exclude))
            longest = max(len(tuple(c.condition)) for c in candidates)
            # the tie-break by fewer atoms stays below any gap between two different Jaccards (≥ 1 / rows²) and any
            # gap between two prior ranks (w / (candidates − 1))
            eps = 1.0 / ((n + 1) ** 2 * (longest + 1) * (len(candidates) + 1))
            size = np.array([len(tuple(c.condition)) for c in candidates], dtype=float)
            out = [float(v) for v in jac + self.weight(n) * pri - eps * size]
        self.last = who
        self.scored[who] += 1
        return out


def checks_to(order: Sequence[Candidate], condition) -> int | None:
    """Checks until `condition` in a candidate order: its position among the distinct conditions (1-based), or None."""
    seen: list = []
    cond = tuple(condition)
    for c in order:
        cc = tuple(c.condition)
        if cc not in seen:
            seen.append(cc)
        if cc == cond:
            return len(seen)
    return None


__all__ = ["BagOfAtomsRanker", "FEATURES", "FeatureRanker", "HandoverRanker", "JointRanker", "PriorRanker",
           "RankedProposer", "SyntheticTask", "atom_features", "atom_kind", "candidate_features", "checks_to",
           "synthetic_task"]
