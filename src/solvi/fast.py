"""Fast answer head: closed-form ridge regression, learned in milliseconds and updated instantly from every correction.

Features come from computed facts (numbers with quantile steps, booleans, categories — as in Head) and from vector facts such
as a document embedding (see LongSpanExtractor.embedder). Answers are one-vs-rest ridge scores turned into probabilities.
The head keeps the inverse of (XᵀX + λI), so `update` is a rank-one Sherman–Morrison step: a new labeled example is absorbed in
about a millisecond without retraining and without touching the rest of the system. Leave-one-out accuracy is exact and free
(the ridge hat matrix).

A rank-one step keeps what the first fit chose: the ridge strength, the featurizer (number scales, the known values of each
category) and whether pairwise products are used. Chosen on a handful of rows they are often wrong for hundreds (on open
tabular sets a head started on 10 rows and taught up to 300 was 5 points less accurate than one fitted on all 300). So the
head keeps its examples and, each time their number doubles (`refit=2.0`), fits again on all of them — the same as a fresh
fit on those rows — and goes on with rank-one steps. The amortised cost stays a constant per update; the update that
triggers a refit is as slow as a fit. Past `refit_until` examples (2000) it stops refitting and drops the kept rows."""
from __future__ import annotations

import numpy as np

from .heads import Featurizer


def _numeric_vector(v):
    if isinstance(v, np.ndarray):
        return v.ndim == 1 and np.issubdtype(v.dtype, np.number)
    return isinstance(v, (list, tuple)) and len(v) > 0 and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)


def _why_dropped(rows, f):
    vs = [r.get(f) for r in rows]
    if all(v is None for v in vs):
        return "never computed in the examples"
    kinds = sorted({type(v).__name__ for v in vs if v is not None})
    missing = sum(v is None for v in vs)
    return f"mixed or unsupported values ({', '.join(kinds)}{f', {missing} missing' if missing else ''})"


class VecFeaturizer(Featurizer):
    """Featurizer plus fixed-length numeric vectors (lists / arrays), standardized per dimension."""

    def fit(self, rows, facts):
        super().fit(rows, facts)
        for f in facts:
            vs = [r.get(f) for r in rows if r.get(f) is not None]
            if vs and all(_numeric_vector(v) for v in vs) and len({len(v) for v in vs}) == 1:
                a = np.array([np.asarray(v, float) for v in vs])
                self.spec[f] = ("vec", a.mean(0), a.std(0) + 1e-6)
        return self

    def cols(self, facts):
        out = []
        for f in facts:
            s = self.spec.get(f)
            if s is not None and s[0] == "vec":
                out += [(f, i) for i in range(len(s[1]))]
            else:
                out += super().cols([f])
        return out

    def compact(self, r, facts):
        """One value per number (standardized), ±1 per boolean, one-hot categories; vectors left out — the base for pairs."""
        x = []
        for f in facts:
            s = self.spec.get(f)
            if s is None or s[0] == "vec":
                continue
            v = r.get(f)
            if s[0] == "num":
                x.append((float(v) - s[1]) / s[2] if isinstance(v, (int, float)) else 0.0)
            elif s[0] == "bool":
                x.append(0.0 if v is None else float(bool(v)) * 2 - 1)
            else:
                x += [1.0 if v == c else 0.0 for c in s[1]]
        return x

    def row(self, r, facts):
        x = []
        for f in facts:
            s = self.spec.get(f)
            if s is not None and s[0] == "vec":
                v = r.get(f)
                x += [0.0] * len(s[1]) if v is None else list((np.asarray(v, float) - s[1]) / s[2])
            else:
                x += super().row(r, [f])
        return x


class FastHead:
    def __init__(self, options, lam=None, pairs=None, refit=2.0, refit_until=2000):
        """lam: ridge strength (None — chosen by exact leave-one-out accuracy); pairs: add pairwise products of the base features
        (None — only when there are at most 40 of them, so interactions and middle classes can be expressed). refit: when
        `update` brings the number of examples to `refit` times the number of the last fit, fit again on all of them (the
        ridge strength, the featurizer and the pairs decision are chosen again, as given here); None — never, and no rows are
        kept. refit_until: no refit beyond this many examples; the kept rows are dropped once none is due."""
        if refit is not None and refit and refit <= 1:
            raise ValueError("refit must be a growth factor above 1 (2.0: refit when the examples double) or None")
        self.options = list(options)
        self.lam = lam
        self.pairs = pairs
        self._lam0, self._pairs0 = lam, pairs         # as given: what a refit chooses again
        self.refit = float(refit) if refit else None
        self.refit_until = int(refit_until)
        self._rows = None                             # the examples kept for the next refit (None: none due)
        self._answers = None
        self._requested = None
        self.fitted_on = 0                            # number of examples at the last fit (or refit)
        self.features = []
        self.fz = None
        self.Ainv = None                  # (XᵀX + λI)⁻¹ over [features, bias]
        self.B = None                     # XᵀY
        self.W = None
        self.n = 0
        self.loo_acc = None
        self._fp = None

    def fingerprint(self):
        """A stable hash of the head's parameters (recorded with every answer it gives; changes with fit and each update)."""
        if getattr(self, "_fp", None) is None:
            from .provenance import digest
            self._fp = digest("FastHead", self.options, self.features, self.W, self.lam, self.pairs,
                              None if self.fz is None else self.fz.spec)
        return self._fp

    @property
    def cv_acc(self):                     # same name as Head, for code that prints it
        return self.loo_acc

    def _x(self, row):
        b = np.array(self.fz.row(row, self.features))
        if self.pairs:
            c = np.array(self.fz.compact(row, self.features))
            b = np.concatenate([b, np.outer(c, c)[np.triu_indices(len(c), 1)]])
        return np.append(b, 1.0)

    def fit(self, rows, answers, features):
        self.lam, self.pairs = getattr(self, "_lam0", self.lam), getattr(self, "_pairs0", self.pairs)
        self.fz = VecFeaturizer().fit(rows, features)
        self.features = [f for f in features if f in self.fz.spec]
        self.dropped = {f: _why_dropped(rows, f) for f in features if f not in self.fz.spec}   # asked for, but unusable
        if self.pairs is None:
            self.pairs = len(self.fz.compact(rows[0], self.features)) <= 40
        X = np.array([self._x(r) for r in rows])
        Y = np.eye(len(self.options))[[self.options.index(a) for a in answers]]
        B = X.T @ Y
        e, V = np.linalg.eigh(X.T @ X)                               # one decomposition serves every λ
        XV, VB = X @ V, V.T @ B
        best = None
        for lam in ([self.lam] if self.lam is not None else [0.1, 0.3, 1, 3, 10, 30, 100]):
            inv = 1.0 / (e + lam)
            h = np.einsum("ij,j,ij->i", XV, inv, XV)                  # leverages → exact leave-one-out predictions
            fit = XV @ (inv[:, None] * VB)
            loo = Y - (Y - fit) / (1 - h)[:, None]
            acc = float((loo.argmax(1) == Y.argmax(1)).mean())
            if best is None or acc > best[0]:
                best = (acc, lam, inv)
        self.loo_acc, self.lam, inv = best
        self.Ainv = (V * inv) @ V.T
        self.W = self.Ainv @ B
        self.B = B
        self.n = self.fitted_on = len(rows)
        self._fp = None
        refit = getattr(self, "refit", None)
        if refit and self._next_refit() <= self.refit_until:
            self._rows, self._answers, self._requested = [dict(r) for r in rows], list(answers), list(features)
        else:
            self._rows = self._answers = self._requested = None
        return self

    def _next_refit(self):
        import math
        return max(self.fitted_on + 1, math.ceil(self.fitted_on * self.refit))

    def update(self, row, answer):
        """Absorb one labeled example (rank-one update of the inverse; a refit on all kept examples when their number reaches
        the refit schedule); returns the time it took in ms."""
        import time
        t0 = time.perf_counter()
        k = self.options.index(answer)                # an unknown answer fails here, before anything changes
        rows = getattr(self, "_rows", None)
        if rows is not None and len(rows) + 1 >= self._next_refit():
            self.fit(rows + [row], self._answers + [answer], self._requested)
            return (time.perf_counter() - t0) * 1000
        x = self._x(row)
        y = np.eye(len(self.options))[k]
        Ax = self.Ainv @ x
        self.Ainv -= np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.B += np.outer(x, y)
        self.W = self.Ainv @ self.B
        self.n += 1
        self._fp = None
        if rows is not None:
            rows.append(dict(row))
            self._answers.append(answer)
        return (time.perf_counter() - t0) * 1000

    def scores(self, row):
        return self._x(row) @ self.W

    def predict(self, row):
        s = np.clip(self.scores(row), 1e-3, None)
        p = s / s.sum()
        return {o: float(v) for o, v in zip(self.options, p)}

    def contributions(self, row):
        """Per fact: its own weight times value (pairwise terms are split equally between the two facts)."""
        x = self._x(row)
        a = int(np.argmax(self.scores(row)))
        owner = [f for f in self.features for _ in range(len(self.fz.cols([f])))]
        out = {f: 0.0 for f in self.features}
        for j, f in enumerate(owner):
            out[f] += float(x[j] * self.W[j, a])
        if self.pairs:
            k = len(owner)
            for (i, j), c in zip(zip(*np.triu_indices(k, 1)), x[k:-1] * self.W[k:-1, a]):
                out[owner[i]] += c / 2
                out[owner[j]] += c / 2
        return out
