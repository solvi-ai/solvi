"""Failure memory: do not repeat an action or a plan that failed in the last N steps (or episodes) — a built-in hard
check, with an expiry and a bound.

    from solvi.core.knowledge import FailureMemory
    fm = FailureMemory(window=20, unit="steps", max_blocked=8, min_open=1)
    fm.install(cat, plan="plan", then={"plan": "ask_person"})   # a hard check over the given fact "recent_failures"

    for step in ...:
        res = system.ask({"situation": s, **fm.given(options=plans_on_offer)})   # the memory as a given fact: replays
        ok = run(res["plan"].answer)
        fm.failed(res["plan"].answer, why="the door stayed shut") if not ok else fm.succeeded(res["plan"].answer)
        fm.step()

Every agent that proposes plans with a model needs this, and writes it by hand (a field report on solvi: "the model
proposed the plan that had just failed"). In benchmarks/knowledge/toy_crafting.py a proposer without memory repeated a
failure in the same situation 29.3 times per 100 steps, 0.53 behind this check. Here it is data plus a pure check: `given()` puts the blocked plans into the
decision's input, the check reads them, the trace records them, and the decision replays.

Bounded on purpose. A memory that only tightens can block everything (a learned gate fed by its own failures lowers
itself episode after episode: benchmarks/knowledge/risk_dungeon.py, arm protect_unbounded). So:
- expiry: a failure blocks its plan for `window` steps (or episodes), then the plan may be tried again;
- max_blocked: at most this many plans are blocked at once (the most recent failures);
- min_open: given the plans on offer (`options`), at least this many stay open — the oldest failures among them are
  let through first;
- evidence reopens: `succeeded(plan)` removes its block at once.
Plans are compared by their JSON form (a string, a tuple of actions, a dict)."""
from __future__ import annotations

import json

from ..slow.refine import Fail

UNITS = ("steps", "episodes")
GIVEN = "recent_failures"


def plan_key(plan):
    """A plan as the memory compares it: a string as it is, anything else as its JSON form (sorted keys)."""
    return plan if isinstance(plan, str) else json.dumps(plan, sort_keys=True, ensure_ascii=False, default=str)


def _verdict(plan, recent):
    """The check: True, or Fail with when the plan failed and until when it is blocked."""
    b = (recent or {}).get("blocked") or {}
    hit = b.get(plan_key(plan))
    if hit is None:
        return True
    return Fail(f"{plan_key(plan)} failed at {recent.get('unit', 'step')} {hit['failed_at']}"
                + (f" ({hit['why']})" if hit.get("why") else "")
                + f"; not repeated before {hit['expires_at']}")


class FailureMemory:
    """See the module docstring. window: how long a failure blocks its plan, in `unit` ("steps" or "episodes");
    max_blocked: plans blocked at once at most; min_open: plans on offer that always stay open; store: a
    KnowledgeStore every failure and success is journaled into."""

    def __init__(self, window=10, *, unit="steps", max_blocked=8, min_open=1, store=None):
        if unit not in UNITS:
            raise ValueError(f"unit is one of {UNITS}, not {unit!r}")
        if window < 1 or max_blocked < 1 or min_open < 0:
            raise ValueError("window and max_blocked are at least 1 and min_open at least 0: a failure memory without "
                             "an expiry or a bound tightens without limit")
        self.window, self.unit, self.max_blocked, self.min_open = int(window), unit, int(max_blocked), int(min_open)
        self.store = store
        self.now = 0
        self.failures = {}                 # key → {"plan", "failed_at", "why", "n"}

    def step(self, k=1):
        """The clock moves (unit "steps")."""
        if self.unit == "steps":
            self.now += int(k)

    def new_episode(self):
        """A new episode (unit "episodes": the clock moves)."""
        if self.unit == "episodes":
            self.now += 1

    def _journal(self, op, **data):
        if self.store is not None:
            self.store.note(failure_memory=op, at=self.now, unit=self.unit, **data)

    def failed(self, plan, *, why=None):
        """The plan failed now: it is blocked for `window`."""
        k = plan_key(plan)
        prev = self.failures.get(k)
        self.failures[k] = {"plan": plan, "failed_at": self.now, "why": why, "n": (prev or {}).get("n", 0) + 1}
        self._journal("failed", plan=k, why=why)

    def succeeded(self, plan):
        """The plan worked: evidence reopens it at once."""
        if self.failures.pop(plan_key(plan), None) is not None:
            self._journal("reopened", plan=plan_key(plan))

    def blocked(self, options=None):
        """The plans blocked now → {key: {"failed_at", "expires_at", "why", "n"}}: failures within the window, at most
        max_blocked (the most recent), and — when `options` (the plans on offer) are given — at least min_open of them
        left open (the oldest failures among them are let through)."""
        live = {k: f for k, f in self.failures.items() if self.now - f["failed_at"] < self.window}
        for k in [k for k in self.failures if k not in live]:
            del self.failures[k]                          # expired: forgotten
        keep = sorted(live, key=lambda k: (-live[k]["failed_at"], k))[:self.max_blocked]
        if options is not None:
            offered = [plan_key(o) for o in options]
            open_ = [o for o in offered if o not in keep]
            need = self.min_open - len(set(open_))
            if need > 0:
                inside = sorted((k for k in keep if k in offered), key=lambda k: (live[k]["failed_at"], k))
                for k in inside[:need]:
                    keep.remove(k)
        return {k: {"failed_at": live[k]["failed_at"], "expires_at": live[k]["failed_at"] + self.window,
                    "why": live[k]["why"], "n": live[k]["n"]} for k in sorted(keep)}

    def given(self, options=None):
        """The memory as a given fact for a decision: {"recent_failures": {"unit", "at", "blocked"}}."""
        return {GIVEN: {"unit": self.unit[:-1], "at": self.now, "blocked": self.blocked(options)}}

    def check(self, plan, options=None):
        """A Prediction for trying `plan` now: "refuse" (hard) when it is blocked, else "accept"."""
        from .actions import Prediction
        b = self.blocked(options).get(plan_key(plan))
        if b is None:
            return Prediction("accept", 0.0, 0, "no recent failure", action=plan_key(plan))
        return Prediction("refuse", 1.0, b["n"], f"failed at {self.unit[:-1]} {b['failed_at']}; blocked until "
                          f"{b['expires_at']}", hard=True, action=plan_key(plan))

    @staticmethod
    def check_function(plan="plan", given=GIVEN, name="not_a_recent_failure"):
        """The hard check as a catalog function of the facts `plan` and `given` (names as you choose them)."""
        for x in (plan, given, name):
            if not x.isidentifier():
                raise ValueError(f"{x!r} is not a fact name")
        ns = {"_verdict": _verdict}
        exec(f"def {name}({plan}, {given}):\n    return _verdict({plan}, {given})\n", ns)   # noqa: S102
        f = ns[name]
        f.__module__ = __name__
        f.__doc__ = f"{plan} is not a plan that failed recently (solvi.core.knowledge.failures.FailureMemory)"
        return f

    def install(self, cat, *, plan="plan", given=GIVEN, then=None, name="not_a_recent_failure"):
        """Declare the hard check in a catalog: when `plan` is a recent failure it is False with the reason, and the
        questions in `then` get their answers (a constant or a function of facts) — see Catalog.check. → the function."""
        f = self.check_function(plan, given, name)
        cat.check(f, hard=True, then=then)
        return f


__all__ = ["FailureMemory", "GIVEN", "plan_key", "UNITS"]
