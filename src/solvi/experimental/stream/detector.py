"""RiskDetector: a Shiryaev–Roberts e-detector on the deployed risk (experimental; stage 2 of the stream certificate).

Input: the deployed loss of one labelled sample, L_t = 1{answered alone and wrong} (an escalated sample contributes 0:
it was not answered alone; a missing label on an alone answer is a loss). Null: E[L_t | past] ≤ α0 at every step, with
α0 = α (the promise itself, the default) or min(alternatives)·α (`alternatives=(1.5, 2.0)`: a stream at α never
alarms, a break below 1.5α is not looked for). Statistic: for bets λ_j = c_j/α0, c ∈ `scales`,

    R_j ← (1 + R_j)·(1 + λ_j·(L_t − α0)),        alarm when mean_j R_j ≥ A (= `arl`).

Under the null each R_j − t is a supermartingale, so is their mean, and E[time to a false alarm] ≥ A (Shin, Ramdas &
Rinaldo, e-detectors). **This is an expected run length, not a per-horizon probability**: unlike
`solvi.core.guarantees.drift.DriftMonitor` ("false flags ≤ alpha within horizon decisions"), a stream running exactly
at α alarms about every A samples on average — by construction ("at the promise, prove it again").

Arithmetic in a fixed order (one product per bet, `math.fsum` for the mean); the recorded R is rounded to 1e-9, so the
same losses in the same order give the same record. The `Monitor` protocol (`observe`, `flagged`, `reset`) is there so
`Dispatcher(monitor=det)` accepts it, but it reads only labelled samples: `observe(decision)` without a label is
ignored. The Dispatcher feeds it with `detector=` (each sampled decision's deployed loss)."""
from __future__ import annotations

import math

SCALES = (0.01, 0.025, 0.05, 0.1, 0.2, 0.4)


class RiskDetector:
    """See the module docstring. alpha: the promised rate; arl: the alarm level A (E[run to a false alarm] ≥ A);
    scales: the bet fractions c_j (λ_j = c_j/α0, each in (0, 1)); alternatives: None (null at α) or rates ≥ 1 (null at
    min(alternatives)·α)."""

    def __init__(self, alpha: float = 0.05, arl: float = 1000.0, scales=SCALES, alternatives=None):
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha is a rate strictly between 0 and 1")
        arl = float(arl)
        if not arl > 1.0:
            raise ValueError("arl (the alarm level A) is above 1")
        sc = tuple(float(c) for c in scales)
        if not sc or any(not 0.0 < c < 1.0 for c in sc):
            raise ValueError("scales are bet fractions in (0, 1), at least one")
        if alternatives is not None:
            alternatives = tuple(float(r) for r in alternatives)
            if not alternatives or any(r < 1.0 for r in alternatives):
                raise ValueError("alternatives are rates ≥ 1 (the null is min(alternatives)·alpha)")
        null = alpha * (min(alternatives) if alternatives else 1.0)
        if not null < 1.0:
            raise ValueError("min(alternatives)·alpha must stay below 1")
        self.alpha, self.arl, self.scales, self.alternatives, self.null = alpha, arl, sc, alternatives, null
        self.lams = tuple(c / null for c in sc)
        self.alarms = 0
        self.reset()

    # --- the statistic
    def update(self, loss: float) -> bool:
        """One deployed loss in [0, 1] → True when this sample raises the alarm (the flag stays up until reset())."""
        x = float(loss)
        if not 0.0 <= x <= 1.0:                                # NaN fails too
            raise ValueError(f"a loss is in [0, 1], not {x!r}")
        a = self.null
        self.r = [(1.0 + r) * (1.0 + lam * (x - a)) for r, lam in zip(self.r, self.lams)]
        self.n += 1
        if not self._flagged and self.statistic >= self.arl:
            self._flagged, self.alarmed_at = True, self.n
            self.alarms += 1
            return True
        return False

    @property
    def statistic(self) -> float:
        """mean_j R_j."""
        return math.fsum(self.r) / len(self.r)

    # --- the Monitor protocol
    def observe(self, decision, label=None, question=None) -> dict:
        """A labelled decision → {"drift", "why", "R", "n"}. decision: a dict {"answer", "alone", "label"}, a Dispatched (alone:
        by "s1"), or anything with `.answer` and `.status` ("ok": answered alone). Without a label: ignored."""
        if label is None and isinstance(decision, dict):
            label = decision.get("label")
        if label is None:
            return self.report(ignored=True)
        if isinstance(decision, dict):
            ans, alone = decision.get("answer"), bool(decision.get("alone", True))
        elif hasattr(decision, "by"):
            ans, alone = decision.answer, decision.by == "s1"
        else:
            ans, alone = getattr(decision, "answer", None), getattr(decision, "status", "ok") == "ok"
        self.update(1.0 if (alone and ans != label) else 0.0)
        return self.report()

    def report(self, ignored=False) -> dict:
        why = ([f"RiskDetector: the deployed risk is above {self.null:g} (Shiryaev–Roberts {self.statistic:.4g} ≥ "
                f"{self.arl:g} after {self.alarmed_at} samples; E[run to a false alarm] ≥ {self.arl:g})"]
               if self._flagged else [])
        out = {"drift": self._flagged, "why": why, "R": round(self.statistic, 9), "n": self.n}
        if ignored:
            out["ignored"] = "no label: the detector reads labelled samples only"
        return out

    @property
    def flagged(self) -> bool:
        return self._flagged

    def reset(self) -> None:
        """Start again: R_j = 0, no flag (the alarm count is kept)."""
        self.r = [0.0] * len(self.scales)
        self.n, self.alarmed_at, self._flagged = 0, None, False

    # --- what is recorded
    def state(self) -> dict:
        return {"R": round(self.statistic, 9), "n": self.n, "alarmed_at": self.alarmed_at}

    def dump(self) -> dict:
        """The statistic's state (JSON data): the configuration (`config()`, its fingerprint) and R_j, n, the alarm
        flag and count. restore() on a detector of the same configuration gives it back exactly."""
        return {"format": "solvi risk detector v1", "config": self.fingerprint(), "r": list(self.r), "n": self.n,
                "alarmed_at": self.alarmed_at, "flagged": self._flagged, "alarms": self.alarms}

    def restore(self, d) -> "RiskDetector":
        """→ self, in the state of `dump()`; a dump of another configuration is refused (ValueError)."""
        if d.get("format") != "solvi risk detector v1":
            raise ValueError(f"not a RiskDetector dump ({d.get('format')!r})")
        if d.get("config") != self.fingerprint() or len(d["r"]) != len(self.scales):
            raise ValueError("the dump is of a RiskDetector with another configuration")
        self.r = [float(x) for x in d["r"]]
        self.n, self.alarmed_at, self._flagged = int(d["n"]), d["alarmed_at"], bool(d["flagged"])
        self.alarms = int(d["alarms"])
        return self

    def config(self) -> dict:
        """The constructor's arguments: RiskDetector(**det.config()) is a detector of the same configuration."""
        return {"alpha": self.alpha, "arl": self.arl, "scales": list(self.scales),
                "alternatives": None if self.alternatives is None else list(self.alternatives)}

    @property
    def promise(self) -> str:
        return (f"E[samples to a false alarm] ≥ {self.arl:g} while the deployed risk is ≤ {self.null:g} "
                "(an expected run length, not a per-horizon probability)")

    def fingerprint(self) -> str:
        from ...core.provenance import digest
        return digest("RiskDetector", self.alpha, self.arl, list(self.scales),
                      None if self.alternatives is None else list(self.alternatives))

    def __repr__(self):
        return f"RiskDetector(alpha={self.alpha:g}, arl={self.arl:g}, R={self.statistic:.4g}, n={self.n})"


__all__ = ["RiskDetector", "SCALES"]
