"""A betting e-process for a bounded numeric outcome (experimental, shared by skills.Certificate and changes.ShipWatch).

MeanBound(m): a test martingale against H0 "the mean of x is at most m", x in [0, 1] (a loss, an error size, a cost
share — not reduced to {0, 1}). K ← K·(1 + λ_t (x_t − m)) with the predictable aGRAPA bet
λ_t = clip((μ̂ − m) / (σ̂² + (μ̂ − m)²), 0, c/m) from the earlier outcomes (Waudby-Smith & Ramdas 2023), c = 0.75 so a
factor never drops below 1 − c. Under H0 every factor has mean ≤ 1, so P(K ever ≥ 1/alpha) ≤ alpha (Ville), for any
stopping rule. The bet uses the size of x, so a mean 0.45 vs a bound 0.3 with a small spread is found in tens of
outcomes where a {0, 1} reduction would wait for exceedances. The null is a conditional mean bound given all earlier outcomes; a marginal mean alone is insufficient."""
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
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha is a probability strictly between 0 and 1")
        return math.log(1.0 / alpha)

    def dump(self) -> dict:
        """Data-only state; restore it on a process with the same bound and bet cap."""
        return {"m": self.m, "c": self.c, "n": self.n, "evidence": self.evidence,
                "s": self._s, "ss": self._ss}

    def restore(self, state: dict) -> MeanBound:
        """Restore validated state atomically. This does not authenticate its provenance."""
        if state["m"] != self.m or state["c"] != self.c:
            raise ValueError("state belongs to a different mean bound or bet cap")
        n = state["n"]
        evidence, total, ss = (float(state[k]) for k in ("evidence", "s", "ss"))
        if isinstance(n, bool) or not isinstance(n, int) or n < 0 or not all(
                math.isfinite(v) for v in (evidence, total, ss)) or not 0 <= total <= n or ss < 0:
            raise ValueError("invalid bounded-mean process state")
        if n == 0 and (evidence != 0 or total != 0 or ss != 0):
            raise ValueError("an empty process has zero state")
        self.n, self.evidence, self._s, self._ss = n, evidence, total, ss
        return self

    @property
    def mean(self) -> float | None:
        return self._s / self.n if self.n else None


__all__ = ["MeanBound"]
