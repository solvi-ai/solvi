"""ShipWatch: the owner of a shipped version — a post-ship watch in units of the promise (experimental).

A promoted change makes a promise per slice: "where I act alone, this slice fails at most `promise` of the time". Per
declared slice the watch keeps a test martingale against that promise on the change's alone outcomes,

    K_s ← K_s · (1 + λ_t · (failed − promise_s)),   λ_t = clip((q̂ − promise_s) / (promise_s (1 − promise_s)), 0, 0.95 / promise_s)

with q̂ the Krichevsky–Trofimov estimate of the slice's failure rate from its earlier outcomes (a predictable bet, the
a test martingale of the betting family). While a slice keeps its promise, K_s is a nonnegative supermartingale,
so it ever reaches S/alpha (S declared slices, Bonferroni) with probability at most alpha/S (Ville): over the whole
watch, a change that keeps every slice's promise is reverted with probability at most alpha — at any stopping time, no
fixed sample size, outcomes read after every use. When a slice's K_s reaches S/alpha, the watched version — and only it
— is undone through Pipeline.revert (rollback(v) undoes every later version too) and the event is journaled
with Pipeline.observe. After `horizon` alone outcomes without that, the watch ends ("graduated").

Outcomes taken under supervision (alone=False: a person or another check decided) are counted, not bet on. A slice
that was not declared raises: alpha was split over the declared ones. Limits: one version per watch (a watch per
shipped version); the promise per slice is the caller's (e.g. the pre-ship failure rate plus a tolerance); outcomes
are taken as independent given the slice; a slow drift inside the promise is not a revert (it is not harm in the
promise's units)."""
from __future__ import annotations

import math

from .._evalue import MeanBound

OUTCOMES = ("failure", "loss")


class ShipWatch:
    """ShipWatch(pipe, version, promise, *, slices=None, alpha=0.05, horizon=None, by="ship_watch", act=True).

    promise: a failure rate in (0, 1) for every slice (then `slices=` names them) or {slice: rate}; alpha: the
    probability bound of reverting a change that keeps its promise in every slice; horizon: alone outcomes after which
    the watch graduates (None: never); act=False decides without reverting or journaling (a simulation);
    outcome="loss": observe(x, ...) takes a numeric loss x in [0, 1] and each slice runs a bounded betting e-process
    (experimental._evalue.MeanBound) against "the slice's mean loss is at most its promise" ("failures" then
    counts losses above the promise).
    `observe(ok, slice=None, *, alone=True)` → None | {"event": "reverted" | "graduated" | "ended", ...};
    `status()` → per slice {promise, outcomes, failures, supervised, evidence (log K), threshold (log S/alpha)}."""

    def __init__(self, pipe, version: int, promise, *, slices=None, alpha: float = 0.05, horizon: int | None = None,
                 by: str = "ship_watch", act: bool = True, outcome: str = "failure"):
        if outcome not in OUTCOMES:
            raise ValueError(f"outcome is one of {OUTCOMES}, not {outcome!r}")
        if isinstance(promise, dict):
            if slices is not None and set(slices) != set(promise):
                raise ValueError("slices= and the promise's keys differ")
            proms = {s: float(p) for s, p in promise.items()}
        else:
            if slices is None:
                raise ValueError("a single promise needs slices= (the slices alpha is split over)")
            proms = {s: float(promise) for s in slices}
        if not proms:
            raise ValueError("at least one slice")
        if not all(0.0 < p < 1.0 for p in proms.values()):
            raise ValueError("a promise is a failure rate strictly between 0 and 1")
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError("alpha is a probability strictly between 0 and 1")
        if horizon is not None and (isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1):
            raise ValueError("horizon is a whole number of alone outcomes >= 1, or None")
        rec = next((v for v in pipe.versions() if v["version"] == version), None)
        if rec is None or not rec["in_force"]:
            raise KeyError(f"version {version} is not a version in force")
        self.pipe, self.version, self.change = pipe, version, rec["change"]
        self.promise, self.alpha, self.horizon, self.by, self.act = proms, float(alpha), horizon, by, bool(act)
        self.threshold = math.log(len(proms) / self.alpha)
        self.state = {s: {"outcomes": 0, "failures": 0, "supervised": 0, "evidence": 0.0} for s in proms}
        self.outcome = outcome
        self.tests = {s: MeanBound(p) for s, p in proms.items()} if outcome == "loss" else None
        self.alone = 0
        self.active = True
        self.events: list = []

    @staticmethod
    def bet(failures: int, outcomes: int, promise: float) -> float:
        """The predictable bet on a failure: the KT estimate's excess over the promise, clipped to [0, 0.95/promise]."""
        q = (failures + 0.5) / (outcomes + 1.0)
        return min(max((q - promise) / (promise * (1.0 - promise)), 0.0), 0.95 / promise)

    def observe(self, ok: bool, slice=None, *, alone: bool = True):
        """One outcome of the watched change in `slice` → None, or the event that ended the watch."""
        if not self.active:
            return None
        if slice not in self.state:
            raise KeyError(f"slice {slice!r} was not declared (declared: {sorted(map(str, self.state))})")
        st = self.state[slice]
        if not alone:
            st["supervised"] += 1
            return None
        p = self.promise[slice]
        if self.tests is not None:                # outcome="loss": `ok` is the loss x in [0, 1], mean bound = promise
            st["evidence"] = self.tests[slice].update(ok)
            y = float(ok)
        else:
            y = 0.0 if ok else 1.0
            lam = self.bet(st["failures"], st["outcomes"], p)
            st["evidence"] += math.log1p(lam * (y - p))
        st["outcomes"] += 1
        st["failures"] += int(y > p) if self.tests is not None else int(y)
        self.alone += 1
        if st["evidence"] >= self.threshold - 1e-12:
            why = (f"slice {slice!r}: {st['failures']} of {st['outcomes']} alone outcomes failed; evidence "
                   f"{math.exp(st['evidence']):.3g} >= {len(self.promise)}/{self.alpha:g} against its promise {p:g}")
            return self._revert(slice, why)
        if self.horizon is not None and self.alone >= self.horizon:
            return self._end("graduated", f"{self.horizon} alone outcomes, no slice broke its promise")
        return None

    def status(self) -> dict:
        return {s: {"promise": self.promise[s], **{k: (round(v, 6) if k == "evidence" else v) for k, v in st.items()},
                    "threshold": round(self.threshold, 6)} for s, st in self.state.items()}

    def _revert(self, slice_, why):
        if not self.act:
            return self._end("reverted", why, slice=slice_, undone=None)
        in_force = {v["version"] for v in self.pipe.versions() if v["in_force"]}
        if self.version not in in_force:
            return self._end("ended", f"version {self.version} is no longer in force; {why}", slice=slice_)
        undone = self.pipe.revert(self.version, by=self.by, why=why)
        return self._end("reverted", why, slice=slice_, undone=undone)

    def _end(self, event, why, **extra):
        self.active = False
        ev = {"event": event, "version": self.version, "why": why, "observations": self.alone, **extra,
              "slices": self.status()}
        if self.act:
            self.pipe.observe(self.change, dict(ev), by=self.by)
        self.events.append(ev)
        return ev


__all__ = ["ShipWatch"]
