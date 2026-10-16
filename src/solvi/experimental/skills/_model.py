"""The exact model (a fork of the environment), the planner on it, and verify (see the package docstring)."""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
import pickle
import warnings
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

    Counters: `forks`, `steps` (simulated steps, every search and every verify); `last`: what the last search did.

    Without `clone`, the first fork checks what the deep copy carries and warns once when it holds a torch module (a
    trained model copied at every fork: a planner that seems to hang) or pickles to more than FORK_WARN_BYTES — give
    `clone=` a snapshot / restore of the world's state alone (check it with
    `solvi.testing.conformance.check_environment(make, clone=)`)."""

    FORK_WARN_BYTES = 8_000_000

    def __init__(self, env, clone: Callable[[Any, Any], Any] | None = None, key: Callable[[Any], str] | None = None):
        if not callable(getattr(env, "step", None)):
            raise TypeError("env is an Environment: reset(seed), actions(state), step(action) → Outcome")
        self.env, self.clone, self.key = env, clone, key or state_key
        self.forks = 0
        self.steps = 0
        self.last: dict = {}
        self._fork_checked = clone is not None

    def fork(self, state=None):
        """An independent copy of the environment (at `state` when a clone hook is given)."""
        return self.copy(self.env, state)

    def copy(self, world, state=None):
        """An independent copy of `world` (the real environment or a fork)."""
        self.forks += 1
        if self.clone is not None:
            return self.clone(world, state)
        if not self._fork_checked:
            self._fork_checked = True
            self._warn_heavy(world)
        return copy.deepcopy(world)

    def _warn_heavy(self, world):
        """Once: warn when the deep copy of `world` holds a torch module or is large (see the class docstring)."""
        why = None
        torch_at = _torch_module_in(world)
        if torch_at is not None:
            why = f"it holds a torch module ({torch_at})"
        else:
            try:
                size = len(pickle.dumps(world, protocol=4))
            except Exception:                          # noqa: BLE001 — not picklable: no size to report
                size = 0
            if size > self.FORK_WARN_BYTES:
                why = f"it pickles to {size:,} bytes"
        if why:
            warnings.warn(f"ExactModel.fork deep-copies the whole environment at every fork and {why}; give "
                          f"ExactModel(clone=) a snapshot / restore of the world's state instead (check it with "
                          f"solvi.testing.conformance.check_environment(make, clone=))", UserWarning, stacklevel=3)

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


def _torch_module_in(obj, depth: int = 4, budget: int = 10_000):
    """The path to a torch.nn.Module inside `obj` (attributes, dict values, list / tuple / set items, a few levels
    down) or None; torch is not imported."""
    seen: set = set()
    todo = [(obj, "env", 0)]
    while todo and budget > 0:
        x, path, d = todo.pop()
        budget -= 1
        if id(x) in seen or isinstance(x, (str, bytes, int, float, bool, type(None))):
            continue
        seen.add(id(x))
        if any(c.__name__ == "Module" and c.__module__.startswith("torch.nn") for c in type(x).__mro__):
            return path
        if d >= depth:
            continue
        if isinstance(x, dict):
            todo += [(v, f"{path}[{k!r}]", d + 1) for k, v in list(x.items())[:1000]]
        elif isinstance(x, (list, tuple, set, frozenset)):
            todo += [(v, f"{path}[{i}]", d + 1) for i, v in enumerate(list(x)[:1000])]
        elif hasattr(x, "__dict__") and not isinstance(x, type):
            todo += [(v, f"{path}.{k}", d + 1) for k, v in list(vars(x).items())[:1000]]
    return None


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


def _goal_of(goal):
    """goal(state) or goal(state, path) → a function of (state, path): a goal with two required positional parameters
    reads the actions that led to the state too."""
    import inspect
    try:
        ps = [p for p in inspect.signature(goal).parameters.values()
              if p.kind == p.VAR_POSITIONAL
              or (p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)]
    except (TypeError, ValueError):                     # a builtin without a signature: one argument
        ps = []
    if len(ps) >= 2 or any(p.kind == p.VAR_POSITIONAL for p in ps):
        return goal
    return lambda st, path: goal(st)


def plan(model: ExactModel, state, goal: Callable[..., bool], *, actions=None, horizon: int = 10,
         budget: int = 10_000, cost: Callable[[Any, Any], float] | None = None, beam: int | None = None,
         collect: int | None = None, distinct: Callable[[Any], Any] | None = None):
    """Search the exact model from `state` for a sequence of actions after which `goal(state)` holds → a Plan, or None.

    Every candidate step is taken on a copy of a fork — the real environment is never stepped. A refused step is not a
    step of a plan; a state already reached (by its key) is not searched again; a state where the episode ends and the
    goal does not hold is a dead end. Search: breadth-first (the fewest actions; ties in the order the environment
    offers its actions) by default; with `cost(state, action)` uniform-cost (the cheapest within `horizon`); with
    `beam=b` breadth-first keeping the `b` cheapest nodes per depth (an approximation). `actions`: None (the
    environment's `actions(state)`), a list, or a function of the state. `budget` bounds the simulated steps; when it is
    used up, or nothing within `horizon` reaches the goal, the result is None and `model.last["why"]` says which
    ("budget", "horizon", "exhausted"). Deterministic given the model.

    goal: a function of the state, or of (state, path) — two positional parameters: the actions that led to it too.
    collect=k (opt-in): the version space — the search goes on past the first plan and returns a list of up to k plans
    whose ends differ by `distinct(end state)` (default: the end state's key), the first found for each (in search
    order: fewest actions, or cheapest with `cost=`); fewer than k (possibly none) when the search ends first, with
    `model.last["why"]` "found" (k collected), "budget", "horizon" or "exhausted". `model.last["complete_depth"]`
    (breadth-first without a beam): the last depth searched in full — "no other plan up to depth d" rather than "the
    budget ran out"."""
    key = model.key
    start = key(state)
    g = _goal_of(goal)
    dkey = distinct if distinct is not None else key
    plans, kinds = [], set()
    used = 0

    def last(why, depth=None):
        model.last = {"why": why, "searched": used, **({"complete_depth": depth} if depth is not None else {}),
                      **({"plans": len(plans)} if collect is not None else {})}

    def hit(path, states, trace, c, end_state):
        """A plan reaching the goal: True when the search stops here."""
        p = Plan(tuple(path), tuple(states), float(c), used, start, key(end_state), simulation_hash(start, trace), goal)
        if collect is None:
            plans.append(p)
            return True
        d = dkey(end_state)
        d = d if isinstance(d, (str, int, float, bool, type(None))) else _hash(_plain(d))
        if d not in kinds:
            kinds.add(d)
            plans.append(p)
        return len(plans) >= int(collect)

    def out(why, depth=None):
        last(why, depth)
        if collect is not None:
            return plans
        return plans[0] if plans else None

    if collect is not None and int(collect) < 1:
        raise ValueError("collect is a number of plans (≥ 1)")
    if g(state, ()) and hit((), (), (), 0.0, state):
        return out("found", 0)
    visited = {start}
    root = model.fork(state)
    step_cost = cost or (lambda s, a: 1.0)

    if cost is not None and beam is None:                        # uniform cost
        heap = [(0.0, 0, root, state, (), (), ())]
        n = 1
        best = {start: 0.0}
        cut = False
        while heap:
            c, _, world, st, path, states, trace = heapq.heappop(heap)
            if path and g(st, path) and hit(path, states, trace, c, st):
                return out("found")
            if len(path) >= horizon:
                cut = True
                continue
            for a in _actions_at(actions, world, st):
                if used >= budget:
                    return out("budget")
                child = model.copy(world, st)
                o = model.step(child, a)
                used += 1
                if not o.accepted:
                    continue
                k = key(o.state)
                c2 = c + float(step_cost(st, a))
                if k in best and best[k] <= c2:
                    continue
                best[k] = c2
                tr = trace + ((a, True, k),)
                if o.done and not g(o.state, path + (a,)):
                    continue
                heapq.heappush(heap, (c2, n, child, o.state, path + (a,), states + (o.state,), tr))
                n += 1
        return out("horizon" if cut else "exhausted")

    frontier = [(0.0, root, state, (), (), ())]
    full = 0 if beam is None else None                  # the last depth searched in full (a beam searches none in full)
    for _depth in range(int(horizon)):
        nxt = []
        for c, world, st, path, states, trace in frontier:
            for a in _actions_at(actions, world, st):
                if used >= budget:
                    return out("budget", full)
                child = model.copy(world, st)
                o = model.step(child, a)
                used += 1
                if not o.accepted:
                    continue
                k = key(o.state)
                if k in visited:
                    continue
                visited.add(k)
                c2 = c + float(step_cost(st, a))
                p2, s2, t2 = path + (a,), states + (o.state,), trace + ((a, True, k),)
                if g(o.state, p2) and hit(p2, s2, t2, c2, o.state):
                    return out("found", full)
                if o.done:
                    continue
                nxt.append((c2, child, o.state, p2, s2, t2))
        if beam is not None:
            nxt = sorted(nxt, key=lambda x: x[0])[:int(beam)]          # stable: ties keep the search order
        else:
            full = _depth + 1
        frontier = nxt
        if not frontier:
            return out("exhausted", full)
    return out("horizon", full)


def verify(model: ExactModel, state, plan, goal: Callable[..., bool] | None = None) -> Check:
    """Re-simulate `plan` on a fresh fork from `state` and check it → a Check. Every step must be accepted and the
    episode must not end before the last one; the goal (`goal`, else the plan's own) must hold at the end; and when the
    plan was found from this very state (its start key), every state must be the one it predicted (so its simulation
    hash is the same). A plan counts as verified only by this re-simulation on the code world — never by what a model
    says about it. plan: a Plan, or a plain sequence of actions (nothing predicted: the steps and the goal are checked).
    goal: of the state, or of (state, path) as for `plan`."""
    if not isinstance(plan, Plan):
        if isinstance(plan, (str, bytes)) or not hasattr(plan, "__iter__"):
            raise TypeError("verify takes a Plan or a sequence of actions")
        plan = Plan(tuple(plan))
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
    if goal is not None and not _goal_of(goal)(st, tuple(plan.actions)):
        return Check(False, "the goal does not hold at the end", st, h, n)
    if exact and plan.simulation and h != plan.simulation:
        return Check(False, "the simulation differs from the one that found the plan", st, h, n)
    return Check(True, "verified on the exact model", st, h, n)


__all__ = ["Check", "ExactModel", "Plan", "plan", "simulation_hash", "state_key", "verify"]
