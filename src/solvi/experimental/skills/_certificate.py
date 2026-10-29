"""The outcome certificate of a skill: an anytime-valid e-process (see the package docstring)."""
from __future__ import annotations

import math

from .._evalue import MeanBound

SCOPES = ("skill", "goal", "set")
DEFAULT_ALTERNATIVES = (1.5, 2.0)
DEFAULT_ALARM = 50.0


class Certificate:
    """The promise a skill acting alone keeps, and how it is certified: at most `promise` of its uses fail (its goal not
    reached), with error probability at most `delta` at any time — a test-martingale bet against "the failure rate is
    at least the promise" (Ville's inequality; no fixed sample size, outcomes may be read after every use).

    Per outcome (did the skill reach its goal: ok / failed) the e-value is
        e = 1 + bet · (promise − failed) / (1 − promise)        (ok: 1 + bet·promise/(1−promise); failed: 1 − bet)
    whose mean is ≤ 1 whenever the failure probability is ≥ the promise, for any fixed bet in [0, 1). The wealth is
    the product of the e-values since the process (re)started; it is certified once the wealth reaches 1/delta, and
    stays certified ("once at the need, certified": the statement holds at any stopping time). `need`: the clean
    outcomes that take a fresh process there, ceil(log(1/delta_1) / log(1 + bet·promise/(1−promise))) — 61 at 5% and
    0.1, 30 at 10%, 14 at 20% (50, 25 and 12 with delta_split=False).

    delta_split (default True): delta is spent over the restarts of one process. Its v-th certification (v = 1 for a
    fresh process, v + 1 after each restart — an alarm, or a skill condition widened by the condition rule; a lattice
    generalization widens a skill without a restart, so its certificate keeps its wealth and v) bets against
    delta_v = 6·delta / (π²·v²), and Σ_v delta_v = delta, so every certificate the process ever issues holds together
    with probability ≥ 1 − delta (a union bound over the restarts; each delta_v by Ville's inequality). The price is
    a larger need after each restart: at 5% and 0.1, 61 / 91 / 108 clean outcomes for v = 1 / 2 / 3.
    delta_split=False: every certification bets against the full delta (the behaviour before 1.4) — each certificate
    holds alone with probability ≥ 1 − delta, not together: after k restarts the chance that some certificate issued
    was wrong is bounded only by k·delta. The process reports v, delta_v and the delta spent so far either way.

    The alarm (default): a change detector that bets on breaks of a stated size — `alternatives`, a tuple of
    failure-rate multiples r of the promise, default (1.5, 2) (a multiple with r·promise ≥ 1 is left out; at a promise
    of 2/3 or more none is left and the old alarm below is used). The detector is the mean over r of the Bernoulli
    likelihood-ratio CUSUMs against the rate r·promise (a failure multiplies one by r, an ok by
    (1 − r·promise)/(1 − promise); each restarts from 1 when it falls below 1) — a standard CUSUM against stated
    alternatives run in parallel (Lorden 1971), averaged (a mixture e-detector). At D ≥ `alarm` (default 50) the
    process restarts — uncertified, wealth 1 — and must reach the need again. At a failure rate at or below the
    promise each product has mean at most 1, so P(an alarm within L outcomes) ≤ L/alarm: a bound over a horizon, empty
    once L passes `alarm`. The alarm is a change detector, not a level-delta test: a unit running at its promise is
    retired within a few thousand outcomes and must re-certify; a unit well inside its promise rarely alarms. The
    level 50 (= 5/delta at the default delta 0.1; it does not follow delta) and the alternatives were chosen on
    simulated streams at promises 0.05–0.3 (docs/best_practices.md).

    alternatives=None: the old alarm — an e-detector with CUSUM wealth D = max(D, 1)·(1 + μ·(failed − promise)),
    μ = ½ / promise (an ok halves it, a failure multiplies it by 1 + (1 − promise) / (2·promise)), default level
    1/delta. It bets on a failure rate near one half, so it is a burst detector whose level must grow as the promise
    shrinks: at 5% one failure alone is an alarm; at 30% it takes three in a row.

    scope: whose outcomes one process pools — "skill" (each skill its own, the default), "goal" (the skills of one
    goal) or "set" (every skill): a pooled process certifies the pool's failure rate, not each skill's; each skill keeps
    its own process too, whose alarm retires that skill alone."""

    def __init__(self, promise: float = 0.05, delta: float = 0.1, *, bet: float = 0.9, scope: str = "skill",
                 alarm: float | None = None, alternatives: tuple | None | str = "default", delta_split: bool = True):
        if not 0.0 < float(promise) < 1.0:
            raise ValueError("promise is a failure rate strictly between 0 and 1")
        if not 0.0 < float(delta) < 1.0:
            raise ValueError("delta is a probability strictly between 0 and 1")
        if not 0.0 < float(bet) < 1.0:
            raise ValueError("bet is a fraction in (0, 1): at 1 a single failure would end the process for good")
        if scope not in SCOPES:
            raise ValueError(f"scope is one of {SCOPES}, not {scope!r}")
        self.promise, self.delta, self.bet, self.scope = float(promise), float(delta), float(bet), scope
        self.delta_split = bool(delta_split)
        if isinstance(alternatives, str):
            if alternatives != "default":
                raise ValueError('alternatives: a tuple of multiples, None (the old alarm) or "default"')
            # (1.5, 2) where r · promise stays a failure rate; none left (promise ≥ 2/3): the old alarm
            alternatives = tuple(r for r in DEFAULT_ALTERNATIVES if r * self.promise < 1.0) or None
        if alarm is None:
            alarm = DEFAULT_ALARM if alternatives is not None else 1.0 / self.delta
        self.alarm = float(alarm)
        if self.alarm <= 1.0:
            raise ValueError("alarm is a level above 1")
        self.alternatives = None
        if alternatives is not None:
            rs = tuple(float(r) for r in alternatives)
            if not rs or any(not r > 1.0 for r in rs):
                raise ValueError("alternatives are failure-rate multiples of the promise, each above 1")
            if any(r * self.promise >= 1.0 for r in rs):
                raise ValueError("alternatives: r · promise must stay below 1 (a failure rate)")
            self.alternatives = rs
        self.gain = math.log1p(self.bet * self.promise / (1.0 - self.promise))
        self.loss = math.log1p(-self.bet)

    # a Certificate pickled before 1.4 has no delta_split attribute: it keeps the full delta at every restart, as
    # it did when it was stored (the same numbers on replay)
    delta_split = False

    def delta_at(self, v: int) -> float:
        """The delta the v-th certification of a process bets against (v = 1, 2, …): 6·delta/(π²·v²) with
        delta_split, else delta."""
        v = int(v)
        if v < 1:
            raise ValueError("v counts certifications from 1")
        return 6.0 * self.delta / (math.pi ** 2 * v * v) if self.delta_split else self.delta

    def need_at(self, v: int) -> int:
        """Clean outcomes that take the v-th certification of a process from wealth 1 to certified."""
        return math.ceil(math.log(1.0 / self.delta_at(v)) / self.gain - 1e-9)

    @property
    def need(self) -> int:
        """Clean outcomes that certify a fresh process (v = 1)."""
        return self.need_at(1)

    def process(self) -> EProcess:
        """A fresh process under this certificate."""
        return EProcess(self)

    def to_dict(self) -> dict:
        d = {"promise": self.promise, "delta": self.delta, "bet": self.bet, "scope": self.scope,
             "alarm": self.alarm, "need": self.need, "delta_split": self.delta_split}
        d["alternatives"] = None if self.alternatives is None else list(self.alternatives)
        return d

    def __repr__(self):
        return f"Certificate(promise={self.promise}, delta={self.delta}, scope={self.scope!r}, need={self.need})"


class EProcess:
    """One certificate's running state: log wealth, certified (stays so until an alarm), the alarm's detector, counts.
    `update(ok, alone)` after every outcome; `remaining()` the clean outcomes still needed. `v`: which certification
    of this process is running (1 for a fresh one, one more after every restart); `delta_v`: the delta it bets
    against (`Certificate.delta_at(v)`); `delta_spent`: Σ delta_u over u ≤ v — at most delta with delta_split, v·delta
    without it."""

    v = 1                                  # an EProcess pickled before 1.4 is in its first certification

    def __init__(self, cert: Certificate):
        self.cert = cert
        self.log_wealth = 0.0
        self.certified = False
        self.detector = 1.0
        self._lr = None if cert.alternatives is None else [0.0] * len(cert.alternatives)   # log CUSUM per alternative
        self.v = 1
        self.n = 0
        self.failures = 0
        self.alarms = 0
        self.since = 0                     # outcomes since the last (re)start
        self.loss_test = MeanBound(cert.promise)   # update(..., loss=x): x in [0, 1] against "mean loss <= promise"

    @property
    def loss_refuted(self) -> bool:
        """A numeric loss given to update(loss=) has mean above the promise: its e-process >= 1/delta (lifetime)."""
        return self.loss_test.evidence >= math.log(1.0 / self.cert.delta) - 1e-12

    @property
    def delta_v(self) -> float:
        """The delta the running certification bets against."""
        return self.cert.delta_at(self.v)

    @property
    def delta_spent(self) -> float:
        """Σ delta_u over the certifications so far (u = 1 … v): the bound on P(some certificate this process issued is
        wrong)."""
        return math.fsum(self.cert.delta_at(u) for u in range(1, self.v + 1))

    def update(self, ok: bool, alone: bool = True, *, loss: float | None = None) -> dict:
        """One outcome → {"certified": certified by this outcome, "alarm": restarted by this outcome}. loss: an
        optional numeric outcome in [0, 1] fed to a bounded betting e-process against "the mean loss is at most the
        promise" (`loss_refuted`; a statement, it does not move certification or the alarm)."""
        c = self.cert
        if loss is not None:
            self.loss_test.update(loss)
        self.n += 1
        self.since += 1
        self.failures += 0 if ok else 1
        self.log_wealth += c.gain if ok else c.loss
        newly = False
        if not self.certified and self.log_wealth >= math.log(1.0 / self.delta_v) - 1e-12:
            self.certified = newly = True
        alarm = False
        if alone:
            if self._lr is None:
                mu = 0.5 / c.promise
                self.detector = max(self.detector, 1.0) * (1.0 + mu * ((0.0 if ok else 1.0) - c.promise))
            else:
                p = c.promise
                for j, r in enumerate(c.alternatives):
                    step = math.log((1.0 - r * p) / (1.0 - p)) if ok else math.log(r)
                    self._lr[j] = max(self._lr[j], 0.0) + step
                self.detector = sum(math.exp(min(x, 700.0)) for x in self._lr) / len(self._lr)
            if self.detector >= c.alarm:
                alarm = True
                self.restart()
                self.alarms += 1
        return {"certified": newly, "alarm": alarm}

    def restart(self):
        """Back to a fresh process: uncertified, wealth 1, the detector reset — the next certification (v + 1), which
        bets against delta_{v+1} (with delta_split)."""
        self.log_wealth, self.certified, self.detector, self.since = 0.0, False, 1.0, 0
        self.v += 1
        if self._lr is not None:
            self._lr = [0.0] * len(self._lr)

    def remaining(self) -> int:
        """Clean outcomes still needed to certify (0 when certified)."""
        if self.certified:
            return 0
        left = math.log(1.0 / self.delta_v) - self.log_wealth
        return max(0, math.ceil(left / self.cert.gain - 1e-9))

    def to_dict(self) -> dict:
        return {"certified": self.certified, "log_wealth": round(self.log_wealth, 6),
                "need_log_wealth": round(math.log(1.0 / self.delta_v), 6), "remaining": self.remaining(),
                "outcomes": self.n, "failures": self.failures, "alarms": self.alarms, "since_restart": self.since,
                "loss_against": round(self.loss_test.evidence, 6), "loss_outcomes": self.loss_test.n,
                "v": self.v, "delta_v": self.delta_v, "delta_spent": self.delta_spent}

    def __repr__(self):
        return f"EProcess(certified={self.certified}, outcomes={self.n}, remaining={self.remaining()})"


__all__ = ["Certificate", "EProcess", "SCOPES"]
