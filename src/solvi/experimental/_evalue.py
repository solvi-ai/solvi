"""A betting e-process for a bounded numeric outcome (experimental, shared by skills.Certificate and changes.ShipWatch).

MeanBound(m): a test martingale against H0 "the mean of x is at most m", x in [0, 1] (a loss, an error size, a cost
share — not reduced to {0, 1}). K ← K·(1 + λ_t (x_t − m)) with the predictable aGRAPA bet
λ_t = clip((μ̂ − m) / (σ̂² + (μ̂ − m)²), 0, c/m) from the earlier outcomes (Waudby-Smith & Ramdas 2023), c = 0.75 so a
factor never drops below 1 − c. Under H0 every factor has mean ≤ 1, so P(K ever ≥ 1/alpha) ≤ alpha (Ville), for any
stopping rule. The bet uses the size of x, so a mean 0.45 vs a bound 0.3 with a small spread is found in tens of
outcomes where a {0, 1} reduction would wait for exceedances. Nothing imported here warns: this module is private."""
from __future__ import annotations

import math


class MeanBound:
    def __init__(self, m: float, *, c: float = 0.75):
        if not 0.0 < float(m) < 1.0:
            raise ValueError("the mean bound m is strictly between 0 and 1")
        if not 0.0 < float(c) < 1.0:
            raise ValueError("c is in (0, 1)")
        self.m, self.c = float(m), float(c)
        self.evidence = 0.0                       # log K
        self.n, self._s, self._ss = 0, 0.0, 0.0   # count, sum, sum of squared deviations (predictable estimates)

    def bet(self) -> float:
        mu = (0.5 + self._s) / (1.0 + self.n)
        var = (0.25 + self._ss) / (1.0 + self.n)
        d = mu - self.m
        return min(max(d / (var + d * d), 0.0), self.c / self.m)

    def update(self, x: float) -> float:
        """One outcome x in [0, 1] → the log evidence after it."""
        x = float(x)
        if not 0.0 <= x <= 1.0:                   # NaN fails too
            raise ValueError(f"an outcome is in [0, 1], not {x!r}")
        lam = self.bet()
        self.evidence += math.log1p(lam * (x - self.m))
        mu = (0.5 + self._s) / (1.0 + self.n)
        self._ss += (x - mu) ** 2
        self._s += x
        self.n += 1
        return self.evidence

    @staticmethod
    def level(alpha: float) -> float:
        """The log level at which H0 is rejected at alpha: log(1/alpha)."""
        return math.log(1.0 / alpha)

    @property
    def mean(self) -> float | None:
        return self._s / self.n if self.n else None


__all__ = ["MeanBound"]
