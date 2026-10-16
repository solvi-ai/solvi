"""Per-law calibrated confidence and abstention on a law's tail.

A law right on 95 % of its rows passes min_precision 0.9 and stays above retire_below 0.8: the law set predicts with it
as if it were exact, and where a hidden cause flips the outcome (a state the grammar cannot see) it is wrong without
warning. `LawConfidence(lawset, buffer)` attaches to every active law a confidence — the probability that its
prediction is right — estimated on the held-out rows where it is the applied law (the rows of its birth group left
out), and splits it once where the misses concentrate:

- the law's confidence: the posterior mean (hits + a) / (n + a + b) (default a = b = 1);
- its tail: of the grammar's atoms outside the law's condition (other features), the one whose true / false rows
  differ most in the share of misses, kept only when Fisher's exact test on the 2×2 table passes at
  alpha / (atoms tested) (Bonferroni; default alpha 0.01) and both sides have ≥ min_rows rows; each side's confidence
  is its own posterior shrunk towards the law's: (hits_side + m · c) / (n_side + m) (default m = 10);
- exact: no miss on at least `exact_rows` rows (default 60) — a law right 95 % of the time is exact on 60 rows with
  probability 0.95⁶⁰ ≈ 0.05.

`of(state, action)` → (value, law, confidence); `predict(state, action, below=t)` → the value, or `Abstained` where a law
speaks with confidence below t (where no law speaks: the base predictor's value, confidence None, never abstained).
The tail is one atom deep and needs an atom that tracks the hidden cause; with none, the confidence is the law's own
and abstention is per law, not per state. Refit (`fit()`) after the buffer or the laws change; `of` refits when it sees
either change."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ._core import Buffer, Law, LawSet, _key


@dataclass(frozen=True, eq=False)
class Abstained:
    """The answer of `LawConfidence.predict` where the applied law's confidence is below the threshold. Never equal to
    an observation."""
    law: Law
    prediction: Any
    confidence: float


@dataclass
class LawStats:
    """One law's counts and confidence: hits / n on its held-out rows, confidence (the posterior mean), exact, and the
    tail split (atom, its p-value, confidence where the atom is true / false, rows on each side) or atom None."""
    law: Law
    hits: int
    n: int
    confidence: float
    exact: bool
    atom: str | None = None
    p: float | None = None
    when_true: float | None = None
    when_false: float | None = None
    n_true: int = 0
    n_false: int = 0


def _log_comb(n, k):
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def fisher_p(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact test p-value of the 2×2 table [[a, b], [c, d]] (sum of the tables at most as likely)."""
    r1, r2, c1 = a + b, c + d, a + c
    n = r1 + r2
    lo, hi = max(0, c1 - r2), min(r1, c1)
    base = _log_comb(n, c1)

    def lp(x):
        return _log_comb(r1, x) + _log_comb(r2, c1 - x) - base
    obs = lp(a)
    p = sum(math.exp(lp(x)) for x in range(lo, hi + 1) if lp(x) <= obs + 1e-9)
    return min(1.0, p)


class LawConfidence:
    """Calibrated per-law confidence and tail abstention over a law set and its held-out buffer (see the module
    docstring). prior: (a, b) of the Beta prior; shrink: m, the pull of a tail side towards the law's confidence;
    alpha: the family-wise level of the tail test; min_rows: least rows on each side of a split; exact_rows: least
    rows for `exact`; atoms: the atom names a split may use (default: every atom of the grammar)."""

    def __init__(self, lawset: LawSet, buffer: Buffer, *, prior: tuple = (1.0, 1.0), shrink: float = 10.0,
                 alpha: float = 0.01, min_rows: int = 20, exact_rows: int = 60, atoms=None):
        self.lawset, self.buffer = lawset, buffer
        self.a, self.b = float(prior[0]), float(prior[1])
        self.shrink, self.alpha, self.min_rows = float(shrink), float(alpha), int(min_rows)
        self.exact_rows = int(exact_rows)
        self.atoms = list(atoms) if atoms is not None else None
        self.stats: dict[str, LawStats] = {}
        self._at = None

    def _stale(self):
        return self._at != (id(self.buffer), self.buffer.version, self.lawset._laws_version)

    def fit(self) -> dict[str, LawStats]:
        """Count every active law's held-out rows and find its tail → {law id: LawStats}."""
        ls, buf = self.lawset, self.buffer
        names = self.atoms if self.atoms is not None else [a.name for a in ls.grammar.atoms]
        out = {}
        for a in list({_key(law.action): law.action for law in ls.laws}.values()):
            _, _, _, take = ls._view(buf, a)
            for law in ls.laws:
                if law.id not in take:
                    continue
                rows = take[law.id]
                if law.born is not None:
                    rows &= ~buf.group_mask(a, law.born)
                right = rows & ls._effect_mask(buf, a, law.effect)
                n, h = rows.bit_count(), right.bit_count()
                c = (h + self.a) / (n + self.a + self.b)
                st = LawStats(law, h, n, c, exact=(h == n and n >= self.exact_rows))
                miss = n - h
                if miss and n >= 2 * self.min_rows:
                    feats = {ls.grammar.feature[x] for x in law.condition}
                    cands = [x for x in names if ls.grammar.feature.get(x) not in feats]
                    best = None
                    for x in cands:
                        m = buf.match(a, (x,)) & rows
                        n1 = m.bit_count()
                        n0 = n - n1
                        if n1 < self.min_rows or n0 < self.min_rows:
                            continue
                        h1 = (m & right).bit_count()
                        p = fisher_p(n1 - h1, h1, (n0 - (h - h1)), h - h1)
                        if best is None or p < best[0]:
                            best = (p, x, h1, n1)
                    if best is not None and best[0] * max(1, len(cands)) <= self.alpha:
                        p, x, h1, n1 = best
                        h0, n0 = h - h1, n - n1
                        st.atom, st.p, st.n_true, st.n_false = x, p, n1, n0
                        st.when_true = (h1 + self.shrink * c) / (n1 + self.shrink)
                        st.when_false = (h0 + self.shrink * c) / (n0 + self.shrink)
                out[law.id] = st
        self.stats = out
        self._at = (id(buf), buf.version, ls._laws_version)
        return out

    def of(self, state, action) -> tuple:
        """(value, law, confidence): the applied law's prediction and its confidence in this state (its tail side
        where it has one); where no law applies, the set's prediction, None and None."""
        if self._stale():
            self.fit()
        law = self.lawset.applied(state, action)
        if law is None:
            return self.lawset.predict(state, action, disagree="first"), None, None
        value = self.lawset.apply(law.effect, state)
        st = self.stats.get(law.id)
        if st is None:
            return value, law, None
        if st.atom is None:
            return value, law, st.confidence
        on = self.lawset.grammar.by_name[st.atom](state, action)
        return value, law, (st.when_true if on else st.when_false)

    def predict(self, state, action, below: float = 0.95):
        """The value, or `Abstained` where a law speaks with confidence < below."""
        value, law, conf = self.of(state, action)
        if law is not None and conf is not None and conf < below:
            return Abstained(law, value, conf)
        return value
