"""A watch on a check run on a random share of decisions (experimental): AuditWatch.

A check that is not run on every decision (an expensive second opinion, a person's look) can break
without anyone seeing it — the decisions it is bought for are the unsure ones, where it disagrees anyway. Run it on
a random share of decisions as well and follow its failure rate on that share with a Bernoulli CUSUM: the reference
failure rate from a random sample, the threshold h simulated with the audit share, the horizon and the reference's
own sampling error, so that an unchanged check is flagged within `horizon` decisions with chance at most alpha.

     watch = AuditWatch(check=lambda d: d["second"] == d["answer"], share=0.1, alpha=0.01, horizon=1000)
     watch.set_reference(passed_on_a_random_sample)           # booleans: did the check pass
     rep = watch.observe(decision)                            # → {"drift", "why", "audited", "failed", "S", "h"}

A decision with an `audit` attribute (Dispatched from Dispatcher(audit=)) is audited when it says so; otherwise the
watch draws itself (a hash of seed, key and the decision's number). What it reports is "the check fails more often
than it did", not which of the check and the system changed.

A known method, packaged: inspecting a random fraction of a running stream (Dodge's continuous sampling plan) with a
Bernoulli CUSUM on each inspected outcome (Reynolds and Stoumbos; the odds-ratio tuning of Steiner, Cook, Farewell and
Treasure)."""
from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np

from . import warn_on_import

warn_on_import(__name__)

__all__ = ["AuditWatch"]


def _outcome(value):
    """Normalize only explicit scalar audit labels; never invoke arbitrary truthiness."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, (float, np.floating)) and bool(np.isnan(value)):
            return None
        if value == 0 or value == 1:
            return bool(value)
    raise ValueError("audit outcome must be a boolean or numeric scalar 0/1; None/NaN is missing")


class AuditWatch:
    """See the module docstring. check: decision → True when the check passes (False / 0: it fails); share: the audited
    share of decisions (the dispatcher's audit= when decisions carry `.audit`); alpha, horizon: an unchanged check is
    flagged within `horizon` decisions with chance ≤ alpha; odds: the rise of the failure odds the CUSUM is tuned to
    (default 2: a doubling); sims: null streams for h (default max(2000, 20 / alpha)). A flag is latched until reset()."""

    def __init__(self, check=None, share=0.1, *, alpha=0.01, horizon=1000, odds=2.0, sims=None, seed=0):
        if check is not None and not callable(check):
            raise TypeError("check is a function decision → bool (passed)")
        if not 0.0 < float(share) <= 1.0:
            raise ValueError("share is the audited share, in (0, 1]")
        if not float(odds) > 1.0:
            raise ValueError("odds: the rise of the failure odds to detect, above 1")
        if not 1e-4 <= float(alpha) < 1:
            raise ValueError("alpha between 1e-4 and 1")
        self.check, self.share, self.alpha, self.horizon = check, float(share), float(alpha), int(horizon)
        self.odds, self.sims, self.seed = float(odds), sims, seed
        self.p0 = self.p1 = self.h = self.cusum = None
        self.n_ref = self.k_ref = None
        self.seen, self.audited, self.failed = 0, 0, 0
        self.audit_requests, self.missing, self.audit_errors = 0, 0, 0
        self._flagged, self._why, self.alarms = False, [], []

    @property
    def flagged(self):
        return self._flagged

    def _rates(self, k, n):
        p0 = (k + 0.5) / (n + 1.0)                    # Jeffreys-like: never 0 or 1
        o1 = self.odds * p0 / (1 - p0)
        return p0, o1 / (1 + o1)

    def set_reference(self, passed):
        """The check's outcomes (True / 1 = passed) on a random sample of decisions → self (h simulated here)."""
        v = [label for x in passed if (label := _outcome(x)) is not None]
        if len(v) < 20:
            raise ValueError(f"a reference of {len(v)} outcomes: at least 20 (a random sample of decisions)")
        n, k = len(v), sum(1 for x in v if not x)
        p0, p1 = self._rates(k, n)
        from ..core.guarantees.drift import Cusum
        share, odds_fn = self.share, self._rates

        def make_null(rng, sims):
            # the true failure rate from the reference's posterior, the reference redrawn from it (its own error)
            p = rng.beta(k + 0.5, n - k + 0.5, size=sims)
            kk = rng.binomial(n, p)
            q0, q1 = odds_fn(kk, n)
            up, dn = np.log(q1 / q0), np.log((1 - q1) / (1 - q0))

            def nxt():
                obs = rng.random(sims) < share
                fail = rng.random(sims) < p
                x = np.where(fail, up, dn)
                return np.where(obs, x, np.nan)[:, None]
            return nxt
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cusum = Cusum.calibrate(["failures up"], make_null, alpha=self.alpha, horizon=self.horizon,
                                    sims=self.sims, seed=self.seed)
        up, dn = math.log(p1 / p0), math.log((1 - p1) / (1 - p0))
        self.n_ref, self.k_ref = n, k
        self.p0, self.p1, self.cusum, self.h = p0, p1, cusum, cusum.h
        self._up, self._dn = up, dn
        self.reset()
        return self

    def reset(self):
        """Clear the CUSUM and flag; keep the reference and lifetime counters/alarms.

        report() counts span resets and reference replacements, not only this detector epoch.
        """
        if self.cusum is not None:
            self.cusum.reset()
        self._flagged, self._why = False, []

    def report(self):
        """Lifetime label availability, descriptive only (not a risk or missingness bound).

        Coverage is observed labels / selected audit requests; None before any request.
        Errors propagate from observe and are counted separately, never as missing labels.
        """
        return {"seen": self.seen, "audit_requests": self.audit_requests, "audited": self.audited,
                "missing": self.missing, "audit_errors": self.audit_errors, "failed": self.failed,
                "observed_coverage": self.audited / self.audit_requests if self.audit_requests else None}

    def draw(self, key, n):
        """The watch's own audit draw of decision n (a number in [0, 1) from a hash)."""
        h = hashlib.sha256(f"auditwatch|{self.seed}|{key}|{n}".encode()).hexdigest()
        return int(h[:13], 16) / 16 ** 13

    def observe(self, decision=None, *, audited=None, passed=None, key=None):
        """One decision → {"drift", "why", "audited", "failed", "S", "h"}. audited: None — the decision's `.audit` if it
        has one, else the watch's own draw; passed: the check's outcome when you ran it yourself (else check(decision)
        is called, only on an audited decision)."""
        if self.cusum is None:
            raise RuntimeError("set_reference() first: the check's outcomes on a random sample")
        self.seen += 1
        if audited is None:
            audited = bool(decision.audit) if hasattr(decision, "audit") else \
                self.draw(key, self.seen) < self.share
        rep = {"drift": self._flagged, "why": list(self._why), "audited": bool(audited), "failed": None,
               "S": None, "h": self.h, "missing": False}
        if not audited:
            return rep
        self.audit_requests += 1
        try:
            if passed is None:
                if self.check is None:
                    raise ValueError("an audited decision needs passed= or a check")
                passed = self.check(decision)
            label = _outcome(passed)
            if label is None:
                self.missing += 1
                rep["missing"] = True
                return rep
            fail = not label
        except Exception:
            self.audit_errors += 1
            raise
        self.audited += 1
        self.failed += fail
        rep["failed"] = fail
        if self._flagged:
            return rep
        self.cusum.step([self._up if fail else self._dn])
        _, S, since = self.cusum.top()
        rep["S"] = S
        if S >= self.h:
            self._flagged = True
            self._why = [f"the audited check fails more often: CUSUM {S:.3g} ≥ h {self.h:.3g} over the last {since} "
                         f"audited decisions (reference failure rate {self.p0:.3g} on {self.n_ref})"]
            self.alarms.append({"at": self.seen, "S": S, "since": since})
            rep.update(drift=True, why=list(self._why))
        return rep
