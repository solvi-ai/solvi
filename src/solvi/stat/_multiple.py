"""Multiple testing of valid e-values; numerical inputs alone do not establish validity.

Base e-BH: Wang & Ramdas (2022), section 4.1,
https://sas.uwaterloo.ca/~wang/papers/2021Wang-Ramdas-JRSSB.pdf
e-LOND: Xu & Ramdas (2024),
https://proceedings.mlr.press/v238/xu24a/xu24a.pdf

The papers require nonnegative null e-variables with expectation at most one;
dependence between them may be arbitrary. These routines do not make arbitrary
scores valid e-values. An adaptively chosen hypothesis/evidence must retain that
null validity; allocating before its outcome does not itself establish it.
"""
from __future__ import annotations

from dataclasses import dataclass
import math


def _probability(q):
    q = float(q)
    if not 0 < q < 1:
        raise ValueError("q is strictly between 0 and 1")
    return q


def _e_value(value):
    value = float(value)
    # +infinity belongs to the extended e-value domain [0, infinity].
    if math.isnan(value) or value < 0:
        raise ValueError("an e-value is nonnegative and not NaN")
    return value


def e_bh(e_values, q: float = 0.05) -> list[int]:
    """Base e-BH: return rejected zero-based indices in original input order.

    For descending e_(k), choose the largest k with e_(k) >= m/(q*k),
    or reject nothing if no such k exists. Ties are deterministic. Empty input
    returns []; NaN and negative values fail, while +infinity is accepted.
    This is a fixed-family procedure, not permission to repeatedly peek at FDR.
    """
    q = _probability(q)
    values = [_e_value(e) for e in e_values]
    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    count = 0
    for k, index in enumerate(order, 1):
        if values[index] >= (len(values) / k) / q:
            count = k
    return sorted(order[:count])


@dataclass(frozen=True)
class _Allocation:
    index: int
    alpha: float

    @property
    def threshold(self) -> float:
        return 1 / self.alpha


class ELOND:
    """Sequential e-LOND, alpha_t = q * gamma_t * (R_(t-1) + 1).

    reserve() fixes a one-based index and alpha BEFORE observing its outcome.
    finish(index, e_value) resolves it once by e_value >= 1/alpha. Only one
    reservation may be outstanding: reserve/finish must alternate, so future
    discoveries cannot change an earlier allocation. There is no cancellation
    or refund; an abandoned reservation blocks further tests. Discarded evidence
    can be resolved conservatively with finish(index, 0).

    schedule='basel' uses gamma_t=6/(pi^2*t^2) indefinitely. schedule='uniform'
    requires a positive fixed horizon T, uses gamma_t=1/T, and refuses index T+1.
    Do not reset the object to recycle a budget over the same hypothesis stream.
    FDR results concern each fixed test prefix, not arbitrary stopping of FDR.
    """

    def __init__(self, q: float = 0.05, *, schedule: str = "basel", horizon: int | None = None):
        q = _probability(q)
        if schedule not in ("basel", "uniform"):
            raise ValueError("schedule is 'basel' or 'uniform'")
        if schedule == "uniform":
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
                raise ValueError("uniform requires a positive integer horizon")
        elif horizon is not None:
            raise ValueError("horizon is only used with schedule='uniform'")
        self._q, self._schedule, self._horizon = q, schedule, horizon
        self._index = self._discoveries = self._finished = 0
        self._pending = None

    @property
    def q(self):
        return self._q

    @property
    def schedule(self):
        return self._schedule

    @property
    def horizon(self):
        return self._horizon

    @property
    def index(self):
        """Last reserved index (zero before the first reservation)."""
        return self._index

    @property
    def discoveries(self):
        return self._discoveries

    @property
    def finished(self):
        return self._finished

    @property
    def pending(self):
        return self._pending

    def reserve(self):
        """Return an immutable allocation(index, alpha); refuses unresolved work."""
        if self._pending is not None:
            raise RuntimeError("finish the pending reservation before reserving another")
        index = self._index + 1
        if self._schedule == "uniform":
            if index > self._horizon:
                raise RuntimeError("the fixed horizon is exhausted")
            gamma = 1 / self._horizon
        else:
            gamma = (6 / math.pi ** 2) / index / index
        alpha = self._q * gamma * (self._discoveries + 1)
        if alpha == 0:
            raise OverflowError("allocation underflow; no reservation was made")
        allocation = _Allocation(index, alpha)
        self._index, self._pending = index, allocation
        return allocation

    def finish(self, index: int, e_value: float) -> bool:
        """Resolve exactly the pending index, returning whether it is rejected.

        Invalid inputs leave all state unchanged. +infinity is accepted.
        """
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("index is an integer returned by reserve")
        if self._pending is None or index != self._pending.index:
            raise ValueError("index must be the unresolved reservation")
        value = _e_value(e_value)
        rejected = value >= self._pending.threshold
        self._discoveries += int(rejected)
        self._finished += 1
        self._pending = None
        return rejected


__all__ = ["e_bh", "ELOND"]
