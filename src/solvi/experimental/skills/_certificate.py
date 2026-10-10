"""The outcome certificate of a skill: an anytime-valid e-process (see the package docstring)."""
from __future__ import annotations

import math

SCOPES = ("skill", "goal", "set")


class Certificate:
    """The promise a skill acting alone keeps, and how it is certified: at most `promise` of its uses fail (its goal not
    reached), with error probability at most `delta` at any time — a test-martingale bet against "the failure rate is
    at least the promise" (Ville's inequality; no fixed sample size, outcomes may be read after every use).

    Per outcome (did the skill reach its goal: ok / failed) the e-value is
        e = 1 + bet · (promise − failed) / (1 − promise)        (ok: 1 + bet·promise/(1−promise); failed: 1 − bet)
    whose mean is ≤ 1 whenever the failure probability is ≥ the promise, for any fixed bet in [0, 1). The wealth is
    the product of the e-values since the process (re)started; it is certified once the wealth reaches 1/delta, and
    stays certified ("once at the need, certified": the statement holds at any stopping time). `need`: the clean
    outcomes that take a fresh process there, ceil(log(1/delta) / log(1 + bet·promise/(1−promise))) — 50 at 5% and
    0.1, 25 at 10%, 12 at 20%.

    The alarm: an e-detector on the outcomes of decisions taken alone (CUSUM wealth D = max(D, 1)·(1 + μ·(failed −
    promise)), μ = ½ / promise: an ok halves it, a failure multiplies it by 1 + (1 − promise) / (2·promise)); at
    D ≥ `alarm` (default 1/delta) the process restarts — uncertified, wealth 1 — and must reach the need again. At 5%
    one failure alone is an alarm; at 30% it takes three in a row.

    scope: whose outcomes one process pools — "skill" (each skill its own, the default), "goal" (the skills of one
    goal) or "set" (every skill): a pooled process certifies the pool's failure rate, not each skill's; each skill keeps
    its own process too, whose alarm retires that skill alone."""

    def __init__(self, promise: float = 0.05, delta: float = 0.1, *, bet: float = 0.9, scope: str = "skill",
                 alarm: float | None = None):
        if not 0.0 < float(promise) < 1.0:
            raise ValueError("promise is a failure rate strictly between 0 and 1")
        if not 0.0 < float(delta) < 1.0:
            raise ValueError("delta is a probability strictly between 0 and 1")
        if not 0.0 < float(bet) < 1.0:
            raise ValueError("bet is a fraction in (0, 1): at 1 a single failure would end the process for good")
        if scope not in SCOPES:
            raise ValueError(f"scope is one of {SCOPES}, not {scope!r}")
        self.promise, self.delta, self.bet, self.scope = float(promise), float(delta), float(bet), scope
        self.alarm = float(alarm) if alarm is not None else 1.0 / self.delta
        if self.alarm <= 1.0:
            raise ValueError("alarm is a level above 1")
        self.gain = math.log1p(self.bet * self.promise / (1.0 - self.promise))
        self.loss = math.log1p(-self.bet)

    @property
    def need(self) -> int:
        """Clean outcomes that certify a fresh process."""
        return math.ceil(math.log(1.0 / self.delta) / self.gain - 1e-9)

    def process(self) -> EProcess:
        """A fresh process under this certificate."""
        return EProcess(self)

    def to_dict(self) -> dict:
        return {"promise": self.promise, "delta": self.delta, "bet": self.bet, "scope": self.scope,
                "alarm": self.alarm, "need": self.need}

    def __repr__(self):
        return f"Certificate(promise={self.promise}, delta={self.delta}, scope={self.scope!r}, need={self.need})"


class EProcess:
    """One certificate's running state: log wealth, certified (stays so until an alarm), the alarm's detector, counts.
    `update(ok, alone)` after every outcome; `remaining()` the clean outcomes still needed."""

    def __init__(self, cert: Certificate):
        self.cert = cert
        self.log_wealth = 0.0
        self.certified = False
        self.detector = 1.0
        self.n = 0
        self.failures = 0
        self.alarms = 0
        self.since = 0                     # outcomes since the last (re)start

    def update(self, ok: bool, alone: bool = True) -> dict:
        """One outcome → {"certified": certified by this outcome, "alarm": restarted by this outcome}."""
        c = self.cert
        self.n += 1
        self.since += 1
        self.failures += 0 if ok else 1
        self.log_wealth += c.gain if ok else c.loss
        newly = False
        if not self.certified and self.log_wealth >= math.log(1.0 / c.delta) - 1e-12:
            self.certified = newly = True
        alarm = False
        if alone:
            mu = 0.5 / c.promise
            self.detector = max(self.detector, 1.0) * (1.0 + mu * ((0.0 if ok else 1.0) - c.promise))
            if self.detector >= c.alarm:
                alarm = True
                self.restart()
                self.alarms += 1
        return {"certified": newly, "alarm": alarm}

    def restart(self):
        """Back to a fresh process: uncertified, wealth 1, the detector reset."""
        self.log_wealth, self.certified, self.detector, self.since = 0.0, False, 1.0, 0

    def remaining(self) -> int:
        """Clean outcomes still needed to certify (0 when certified)."""
        if self.certified:
            return 0
        left = math.log(1.0 / self.cert.delta) - self.log_wealth
        return max(0, math.ceil(left / self.cert.gain - 1e-9))

    def to_dict(self) -> dict:
        return {"certified": self.certified, "log_wealth": round(self.log_wealth, 6),
                "need_log_wealth": round(math.log(1.0 / self.cert.delta), 6), "remaining": self.remaining(),
                "outcomes": self.n, "failures": self.failures, "alarms": self.alarms, "since_restart": self.since}

    def __repr__(self):
        return f"EProcess(certified={self.certified}, outcomes={self.n}, remaining={self.remaining()})"


__all__ = ["Certificate", "EProcess", "SCOPES"]
