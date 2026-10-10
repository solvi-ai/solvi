"""The exact model (a fork of the environment), the planner on it, and verify (see the package docstring)."""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import pickle
from dataclasses import dataclass, field
from typing import Any, Callable


def state_key(state) -> str:
    """The default key of a state: its canonical JSON (as `solvi.Agent`'s)."""
    return json.dumps(state, sort_keys=True, ensure_ascii=False, default=str)


def _hash(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:16]


def _plain(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return json.loads(json.dumps(x, sort_keys=True, ensure_ascii=False, default=str))


def simulation_hash(start: str, trace) -> str:
    """The hash of one simulation: the start key and, per step, (action, accepted, the key reached)."""
    return _hash([start, [[_plain(a), bool(ok), k] for a, ok, k in trace]])


class ExactModel:
    """A world model that is the environment's own code. `fork(state)` → an independent copy to step: by default
    `copy.deepcopy` of the environment as it is now (the state argument is then only a label); `clone(env, state)` → an
    independent environment at `state`, for environments with their own snapshot / restore (it is also used to copy a
    fork during a search). The real environment is only read (copied), never stepped: `fingerprint()` hashes it, so a
    caller can check that a plan call left it as it was.

    Counters: `forks`, `steps` (simulated steps, every search and every verify); `last`: what the last search did."""

    def __init__(self, env, clone: Callable[[Any, Any], Any] | None = None, key: Callable[[Any], str] | None = None):
        if not callable(getattr(env, "step", None)):
            raise TypeError("env is an Environment: reset(seed), actions(state), step(action) → Outcome")
        self.env, self.clone, self.key = env, clone, key or state_key
        self.forks = 0
        self.steps = 0
        self.last: dict = {}

    def fork(self, state=None):
        """An independent copy of the environment (at `state` when a clone hook is given)."""
        return self.copy(self.env, state)

    def copy(self, world, state=None):
        """An independent copy of `world` (the real environment or a fork)."""
        self.forks += 1
        return self.clone(world, state) if self.clone is not None else copy.deepcopy(world)

    def step(self, world, action):
        """One simulated step on a fork (counted)."""
        self.steps += 1
        return world.step(action)

    def simulate(self, actions, state=None) -> list:
        """Run `actions` on a fresh fork → their Outcomes (it stops after a refused step or the end of the episode)."""
        w = self.fork(state)
        outs = []
        for a in actions:
            o = self.step(w, a)
            outs.append(o)
            if not o.accepted or o.done:
                break
        return outs

    def fingerprint(self) -> str:
        """A hash of the real environment's contents (pickle; its repr when it cannot be pickled)."""
        try:
            raw = pickle.dumps(self.env, protocol=4)
        except Exception:                              # noqa: BLE001 — an environment that cannot be pickled
            raw = repr(vars(self.env) if hasattr(self.env, "__dict__") else self.env).encode()
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass(frozen=True)
class Plan:
    """A plan found on the exact model. actions: the sequence; predicted: the states the code gave after each action;
    cost: its cost (the number of actions without `cost=`); searched: the simulated steps the search took; start / end:
    the keys of the start and end states; simulation: the hash of the simulation that found it; goal: the goal check it
    was found for (not part of its equality)."""
    actions: tuple
    predicted: tuple = ()
    cost: float = 0.0
    searched: int = 0
    start: str = ""
    end: str = ""
    simulation: str = ""
    goal: Any = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Check:
    """What verify found: ok, why, the end state of the re-simulation, its hash and the steps it ran. Truthy when ok."""
    ok: bool
    why: str
    end: Any = None
    simulation: str = ""
    steps: int = 0

    def __bool__(self):
        return self.ok


def _actions_at(actions, world, state):
    if actions is None:
        return list(world.actions(state))
    if callable(actions):
        return list(actions(state))
    return list(actions)


def plan(model: ExactModel, state, goal: Callable[[Any], bool], *, actions=None, horizon: int = 10,
         budget: int = 10_000, cost: Callable[[Any, Any], float] | None = None, beam: int | None = None) -> Plan | None:
    """Search the exact model from `state` for a sequence of actions after which `goal(state)` holds → a Plan, or None.

    Every candidate step is taken on a copy of a fork — the real environment is never stepped. A refused step is not a
    step of a plan; a state already reached (by its key) is not searched again; a state where the episode ends and the
    goal does not hold is a dead end. Search: breadth-first (the fewest actions; ties in the order the environment
    offers its actions) by default; with `cost(state, action)` uniform-cost (the cheapest within `horizon`); with
    `beam=b` breadth-first keeping the `b` cheapest nodes per depth (an approximation). `actions`: None (the
    environment's `actions(state)`), a list, or a function of the state. `budget` bounds the simulated steps; when it is
    used up, or nothing within `horizon` reaches the goal, the result is None and `model.last["why"]` says which
    ("budget", "horizon", "exhausted"). Deterministic given the model."""
    key = model.key
    start = key(state)
    if goal(state):
        model.last = {"why": "found", "searched": 0}
        return Plan((), (), 0.0, 0, start, start, simulation_hash(start, []), goal)
    used = 0
    visited = {start}
    root = model.fork(state)
    step_cost = cost or (lambda s, a: 1.0)

    def found(path, states, trace, c):
        model.last = {"why": "found", "searched": used}
        return Plan(tuple(path), tuple(states), float(c), used, start, key(states[-1]), simulation_hash(start, trace),
                    goal)

    if cost is not None and beam is None:                        # uniform cost
        heap = [(0.0, 0, root, state, (), (), ())]
        n = 1
        best = {start: 0.0}
        cut = False
        while heap:
            c, _, world, st, path, states, trace = heapq.heappop(heap)
            if path and goal(st):
                return found(path, states, trace, c)
            if len(path) >= horizon:
                cut = True
                continue
            for a in _actions_at(actions, world, st):
                if used >= budget:
                    model.last = {"why": "budget", "searched": used}
                    return None
                child = model.copy(world, st)
                out = model.step(child, a)
                used += 1
                if not out.accepted:
                    continue
                k = key(out.state)
                c2 = c + float(step_cost(st, a))
                if k in best and best[k] <= c2:
                    continue
                best[k] = c2
                tr = trace + ((a, True, k),)
                if out.done and not goal(out.state):
                    continue
                heapq.heappush(heap, (c2, n, child, out.state, path + (a,), states + (out.state,), tr))
                n += 1
        model.last = {"why": "horizon" if cut else "exhausted", "searched": used}
        return None

    frontier = [(0.0, root, state, (), (), ())]
    for _depth in range(int(horizon)):
        nxt = []
        for c, world, st, path, states, trace in frontier:
            for a in _actions_at(actions, world, st):
                if used >= budget:
                    model.last = {"why": "budget", "searched": used}
                    return None
                child = model.copy(world, st)
                out = model.step(child, a)
                used += 1
                if not out.accepted:
                    continue
                k = key(out.state)
                if k in visited:
                    continue
                visited.add(k)
                c2 = c + float(step_cost(st, a))
                p2, s2, t2 = path + (a,), states + (out.state,), trace + ((a, True, k),)
                if goal(out.state):
                    return found(p2, s2, t2, c2)
                if out.done:
                    continue
                nxt.append((c2, child, out.state, p2, s2, t2))
        if beam is not None:
            nxt = sorted(nxt, key=lambda x: x[0])[:int(beam)]          # stable: ties keep the search order
        frontier = nxt
        if not frontier:
            model.last = {"why": "exhausted", "searched": used}
            return None
    model.last = {"why": "horizon", "searched": used}
    return None


def verify(model: ExactModel, state, plan: Plan, goal: Callable[[Any], bool] | None = None) -> Check:
    """Re-simulate `plan` on a fresh fork from `state` and check it → a Check. Every step must be accepted and the
    episode must not end before the last one; the goal (`goal`, else the plan's own) must hold at the end; and when the
    plan was found from this very state (its start key), every state must be the one it predicted (so its simulation
    hash is the same). A plan counts as verified only by this re-simulation on the code world — never by what a model
    says about it."""
    goal = goal if goal is not None else plan.goal
    start = model.key(state)
    exact = bool(plan.start) and plan.start == start and len(plan.predicted) == len(plan.actions)
    w = model.fork(state)
    trace = []
    st = state
    n = len(plan.actions)
    for i, a in enumerate(plan.actions):
        out = model.step(w, a)
        k = model.key(out.state) if out.accepted else model.key(st)
        trace.append((a, bool(out.accepted), k))
        h = simulation_hash(start, trace)
        if not out.accepted:
            return Check(False, f"step {i + 1} ({a!r}) was refused", st, h, i + 1)
        st = out.state
        if exact and k != model.key(plan.predicted[i]):
            return Check(False, f"step {i + 1} ({a!r}) reached a state other than the predicted one", st, h, i + 1)
        if out.done and i < n - 1:
            return Check(False, f"the episode ended at step {i + 1} of {n}", st, h, i + 1)
    h = simulation_hash(start, trace)
    if goal is not None and not goal(st):
        return Check(False, "the goal does not hold at the end", st, h, n)
    if exact and plan.simulation and h != plan.simulation:
        return Check(False, "the simulation differs from the one that found the plan", st, h, n)
    return Check(True, "verified on the exact model", st, h, n)


__all__ = ["Check", "ExactModel", "Plan", "plan", "simulation_hash", "state_key", "verify"]
