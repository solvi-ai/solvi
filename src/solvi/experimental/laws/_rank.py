"""Ranking candidate laws where each check is expensive (see the package docstring): features from the Buffer's
bitsets, a bag-of-atoms scorer trained on synthetic tasks drawn from the grammar, the raw-feature control, the joint
statistic of the whole condition (no training), and a proposer that only reorders another one."""
from __future__ import annotations

import random
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
    from; the held-out test stays the judge."""

    def __init__(self, inner, scorer, buffer: Buffer, *, exclude=None):
        self.inner, self.scorer, self.buffer, self.exclude = inner, scorer, buffer, exclude
        self.name = getattr(inner, "name", "ranked")

    def propose(self, lawset, state, action, observed):
        cands = list(self.inner.propose(lawset, state, action, observed))
        if len(cands) < 2:
            return cands
        scores = self.scorer.score(cands, lawset, self.buffer, exclude=self.exclude)
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


__all__ = ["BagOfAtomsRanker", "FEATURES", "FeatureRanker", "JointRanker", "RankedProposer", "SyntheticTask",
           "atom_features", "candidate_features", "checks_to", "synthetic_task"]
