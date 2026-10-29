"""The e-processes of the stream certificate and its δ budget (experimental, private).

Two bets against H0 "the mean loss is at least α" (a certificate: H0 refuted ⇒ the risk is below α), both of the family
solvi already has (`experimental.skills.Certificate` is one constant bet, `experimental._evalue.MeanBound` the aGRAPA
bet in the refuting direction):

- `Mixture`: a mixture of constant bets. Component j multiplies its wealth by 1 + b_j·(α − L) with b_j = f_j/(1 − α)
  (a clean sample: 1 + f_j·α/(1 − α); a loss: 1 − f_j); the wealth is Σ_j w_j·K_j. Each K_j is a test martingale under
  H0 (its factor has conditional mean ≤ 1 when E[L | past] ≥ α), so is the mixture, and P(ever ≥ 1/δ) ≤ δ (Ville).
  Default f = (0.9, 0.5, 0.25, 0.1), w = (0.6, 0.25, 0.1, 0.05) — constants from the design this was adapted from, kept for comparability.
- `MirrorBound`: MeanBound(m = 1 − α) fed x = 1 − L (the mirror: "mean of 1 − L ≤ 1 − α" refuted).

`DeltaBudget`: the δ spent over the certifications of one promise, δ_v = `Certificate.delta_at(v)` (Basel:
6δ/(π²v²), Σ_v δ_v = δ); `next()` → (v, δ_v), `spent` = Σ_{u ≤ v} δ_u.

Arithmetic: `math.log1p` per factor in a fixed order and `math.fsum` for the mixture, so a replay of the same samples
gives the same numbers."""
from __future__ import annotations

import math
import warnings

from .._evalue import MeanBound

FRACTIONS = (0.9, 0.5, 0.25, 0.1)
WEIGHTS = (0.6, 0.25, 0.1, 0.05)
SCHEDULES = ("basel",)


class Mixture:
    """A mixture of constant bets against "the mean loss is at least alpha". update(loss) with loss in [0, 1]."""

    def __init__(self, alpha: float, fractions=FRACTIONS, weights=WEIGHTS):
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha is a rate strictly between 0 and 1")
        fr, w = tuple(float(f) for f in fractions), tuple(float(x) for x in weights)
        if not fr or len(fr) != len(w):
            raise ValueError("fractions and weights: one weight per bet fraction, at least one")
        if any(not 0.0 < f < 1.0 for f in fr):
            raise ValueError("a bet fraction is in (0, 1): at 1 a single loss would end the process for good")
        if any(x <= 0.0 for x in w) or abs(math.fsum(w) - 1.0) > 1e-9:
            raise ValueError("weights are positive and sum to 1")
        self.alpha, self.fractions, self.weights = alpha, fr, w
        self.b = tuple(f / (1.0 - alpha) for f in fr)
        self.lw = [0.0] * len(fr)                  # log wealth per component
        self.n = 0

    def update(self, loss: float) -> float:
        """One loss in [0, 1] → the log wealth of the mixture after it."""
        x = float(loss)
        if not 0.0 <= x <= 1.0:                    # NaN fails too
            raise ValueError(f"a loss is in [0, 1], not {x!r}")
        a = self.alpha
        for j, b in enumerate(self.b):
            self.lw[j] += math.log1p(b * (a - x))
        self.n += 1
        return self.log_wealth

    @property
    def log_wealth(self) -> float:
        top = max(self.lw)
        return top + math.log(math.fsum(w * math.exp(lw - top) for w, lw in zip(self.weights, self.lw)))

    def vector(self) -> list:
        return list(self.lw)

    def dump(self) -> dict:
        """The process's state (its log wealths and count): restore() on a Mixture of the same alpha, fractions and
        weights gives it back exactly."""
        return {"bet": "mixture", "lw": list(self.lw), "n": self.n}

    def restore(self, d) -> "Mixture":
        if d.get("bet") != "mixture" or len(d["lw"]) != len(self.lw):
            raise ValueError("not a dump of a Mixture with these bet fractions")
        self.lw, self.n = [float(x) for x in d["lw"]], int(d["n"])
        return self


class MirrorBound:
    """MeanBound(m = 1 − alpha) fed x = 1 − loss: the aGRAPA bet, certifying "the mean loss is below alpha"."""

    def __init__(self, alpha: float, c: float = 0.75):
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha is a rate strictly between 0 and 1")
        self.alpha = alpha
        self.bound = MeanBound(1.0 - alpha, c=c)

    def update(self, loss: float) -> float:
        x = float(loss)
        if not 0.0 <= x <= 1.0:
            raise ValueError(f"a loss is in [0, 1], not {x!r}")
        return self.bound.update(1.0 - x)

    @property
    def log_wealth(self) -> float:
        return self.bound.evidence

    @property
    def n(self) -> int:
        return self.bound.n

    def vector(self) -> list:
        return [self.bound.evidence]

    def dump(self) -> dict:
        """The aGRAPA bet's state (log evidence, count, its predictable estimates): restore() gives it back exactly."""
        b = self.bound
        return {"bet": "agrapa", "evidence": b.evidence, "n": b.n, "s": b._s, "ss": b._ss}

    def restore(self, d) -> "MirrorBound":
        if d.get("bet") != "agrapa":
            raise ValueError("not a dump of a MirrorBound")
        b = self.bound
        b.evidence, b.n, b._s, b._ss = float(d["evidence"]), int(d["n"]), float(d["s"]), float(d["ss"])
        return self


def _delta_schedule(delta: float):
    """Certificate(delta=delta, delta_split=True): its delta_at(v) is the schedule (imported quietly: the skills
    package's own experimental warning is not this module's business)."""
    from .. import _WARNED
    had = "solvi.experimental.skills" in _WARNED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ..skills._certificate import Certificate
    if not had:
        _WARNED.discard("solvi.experimental.skills")   # a later import of skills by the user still warns
    return Certificate(delta=delta, delta_split=True)


class DeltaBudget:
    """δ spent over the certifications of one promise (one budget per promise; a detector and the versions of the same
    promise share it). schedule="basel": δ_v = 6δ/(π²v²) (`Certificate.delta_at`), Σ_v δ_v = δ, so every
    certification ever made holds together with probability ≥ 1 − δ. next() → (v, δ_v) and moves the index;
    `v` is the index last handed out (0 before the first)."""

    def __init__(self, delta: float = 0.05, schedule: str = "basel"):
        delta = float(delta)
        if not 0.0 < delta < 1.0:
            raise ValueError("delta is a probability strictly between 0 and 1")
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule is one of {SCHEDULES}, not {schedule!r}")
        self.delta, self.schedule = delta, schedule
        self._cert = _delta_schedule(delta)
        self.v = 0

    def delta_at(self, v: int) -> float:
        return self._cert.delta_at(v)

    def next(self) -> tuple:
        """The next certification's index and δ: (v, δ_v)."""
        self.v += 1
        return self.v, self.delta_at(self.v)

    @property
    def spent(self) -> float:
        """Σ δ_u over the indices handed out so far."""
        return math.fsum(self.delta_at(u) for u in range(1, self.v + 1))

    def state(self) -> dict:
        return {"delta": self.delta, "schedule": self.schedule, "v": self.v, "spent": self.spent}

    def __repr__(self):
        return f"DeltaBudget(delta={self.delta}, schedule={self.schedule!r}, v={self.v})"


def clean_need(alpha: float, delta: float, bets: str = "mixture", **kw) -> int:
    """Clean samples (loss 0) that take a fresh e-process to 1/delta."""
    proc = Mixture(alpha, **kw) if bets == "mixture" else MirrorBound(alpha, **kw)
    level = math.log(1.0 / delta)
    while proc.log_wealth < level - 1e-12:
        proc.update(0.0)
        if proc.n > 1_000_000:
            raise ValueError("no certification within a million clean samples")
    return proc.n


__all__ = ["DeltaBudget", "FRACTIONS", "MirrorBound", "Mixture", "SCHEDULES", "WEIGHTS", "clean_need"]
