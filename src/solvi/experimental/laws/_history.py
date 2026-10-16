"""History atoms: laws over the previous steps (the last action, the last observation, the state before).

A law set sees one state per transition; a law that depends on what happened before ("after a push the door is open",
"the counter shows what it showed before unless you press") cannot be written in a grammar over the present state.
`History(grammar, depth=d)` lifts a grammar to contexts — the present state plus the last d steps — and adds atoms
over those steps; a `Tracker` builds the contexts along a stream and starts afresh at each episode.

    h = History(g, depth=1, actions=["push", "wait"], observed=["open", "shut"], same=True)
    laws = LawSet(h.grammar, **h.kwargs(base=my_base))
    tr = h.tracker()
    for state, action, seen, new_episode in stream:
        if new_episode:
            tr.reset()
        laws.step(tr.transition(state, action, seen, group=...), buffer, remember=True)
    laws.predict(tr.context(state), action)            # predict on a context, as the laws were learned

Atoms (for each d in 1..depth): every atom of the grammar on the present state keeps its name; "t-d:<atom>" is the
atom on the state and action d steps back; "t-d:action=<a>" for each listed action; "t-d:observed=<v>" for each listed
observation; "t-d:start" when the context has fewer than d past steps (the episode has just begun). Each history atom
has its own feature ("t-d:<feature>", "t-d:action", "t-d:observed", "t-d:start"), so a condition may combine a present
atom with a past one. same=True adds the effect `Same(d)` — "the observation is what was observed d steps back" — to the
effect variants of every mismatch: one law for a value that persists, where the literal effects need one law per value.

The atoms are a larger grammar: the sized acceptance counts the extra candidates (the bound holds over all of them), so
a law over the present state alone needs a little more support than without history atoms. Atoms added to the original
grammar after the lift are not lifted."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Iterable

from ._core import Atom, Grammar, Transition


@dataclass(frozen=True)
class Past:
    """One step back: the state, the action taken in it, what was observed after."""
    state: Any
    action: Any
    observed: Any


@dataclass(frozen=True)
class Context:
    """What a history-lifted law set sees: the present state and the past steps, the most recent first (fewer than
    `depth` at the start of an episode)."""
    now: Any
    past: tuple = ()

    def back(self, d: int) -> Past | None:
        """The step d back (1 = the last one), or None before the episode had d steps."""
        return self.past[d - 1] if 0 < d <= len(self.past) else None


@dataclass(frozen=True)
class Same:
    """The effect «the observation is what was observed d steps back». Predicts `Same.MISSING` (never an
    observation) where the context has no such step."""
    d: int = 1

    def __repr__(self):
        return f"Same(t-{self.d})"


class _Missing:
    def __repr__(self):
        return "MISSING"

    def __eq__(self, other):
        return False

    def __hash__(self):
        return 0


Same.MISSING = _Missing()


def _lift_now(atom: Atom) -> Atom:
    return Atom(atom.name, (lambda c, a, _f=atom: _f(c.now, a)), feature=atom.feature)


def _lift_back(atom: Atom, d: int) -> Atom:
    def fn(c, a, _f=atom, _d=d):
        p = c.back(_d)
        return p is not None and _f(p.state, p.action)
    return Atom(f"t-{d}:{atom.name}", fn, feature=f"t-{d}:{atom.feature}")


class History:
    """Lift `grammar` to contexts with atoms over the last `depth` steps (see the module docstring).

    actions / observed: the values the "t-d:action=" / "t-d:observed=" atoms name (none listed: no such atoms);
    past_atoms: lift the grammar's own atoms to the past steps (default True); start: add "t-d:start"; same: add the
    effect variant `Same(d)`; k: the largest conjunction (default: the grammar's)."""

    def __init__(self, grammar: Grammar, depth: int = 1, *, actions: Iterable = (), observed: Iterable = (),
                 past_atoms: bool = True, start: bool = True, same: bool = False, k: int | None = None):
        if int(depth) < 1:
            raise ValueError("depth is a number of past steps ≥ 1")
        self.depth, self.same = int(depth), bool(same)
        self.source = grammar
        self.actions, self.observed = list(actions), list(observed)
        atoms = [_lift_now(a) for a in grammar.atoms]
        for d in range(1, self.depth + 1):
            if past_atoms:
                atoms += [_lift_back(a, d) for a in grammar.atoms]
            for v in self.actions:
                atoms.append(Atom(f"t-{d}:action={v}", (lambda c, a, _d=d, _v=v: (p := c.back(_d)) is not None
                                                        and p.action == _v), feature=f"t-{d}:action"))
            for v in self.observed:
                atoms.append(Atom(f"t-{d}:observed={v}", (lambda c, a, _d=d, _v=v: (p := c.back(_d)) is not None
                                                          and p.observed == _v), feature=f"t-{d}:observed"))
            if start:
                atoms.append(Atom(f"t-{d}:start", (lambda c, a, _d=d: c.back(_d) is None), feature=f"t-{d}:start"))
        self.grammar = Grammar(atoms, k=grammar.k if k is None else int(k))

    # ------------------------------------------------------------------ the law set's callables, lifted
    def base(self, fn: Callable | None) -> Callable | None:
        """`fn(state, action)` → the same on a context's present state."""
        if fn is None:
            return None
        return lambda c, a: fn(c.now, a)

    def apply(self, fn: Callable | None = None) -> Callable:
        """The law set's `apply`: `Same(d)` → the observation d steps back; any other effect → `fn(effect, state)`
        on the present state (default: the effect itself)."""
        def apply(effect, c):
            if isinstance(effect, Same):
                p = c.back(effect.d)
                return Same.MISSING if p is None else p.observed
            return effect if fn is None else fn(effect, c.now)
        return apply

    def effects(self, fn: Callable | None = None) -> Callable:
        """The law set's `effects`: `fn(state, observed)` on the present state (default: [observed]), plus `Same(d)`
        for each d whose past observation equals this one (same=True)."""
        def effects(c, observed):
            out = list(fn(c.now, observed)) if fn is not None else [observed]
            if self.same:
                for d in range(1, self.depth + 1):
                    p = c.back(d)
                    if p is not None and p.observed == observed:
                        out.append(Same(d))
            return out
        return effects

    def kwargs(self, *, base: Callable | None = None, apply: Callable | None = None,
               effects: Callable | None = None) -> dict:
        """base / apply / effects for `LawSet(h.grammar, **h.kwargs(...))`."""
        return {"base": self.base(base), "apply": self.apply(apply), "effects": self.effects(effects)}

    def tracker(self) -> "Tracker":
        return Tracker(self.depth)


class Tracker:
    """Builds contexts along a stream: `context(state)` → the present state with the last `depth` steps; `push`
    records a step; `reset` starts an episode (no past). `transition` does both for one step."""

    def __init__(self, depth: int = 1):
        self.depth = int(depth)
        self._past: deque = deque(maxlen=self.depth)

    def context(self, state) -> Context:
        return Context(state, tuple(reversed(self._past)))

    def push(self, state, action, observed):
        self._past.append(Past(state, action, observed))

    def reset(self):
        self._past.clear()

    def transition(self, state, action, observed, *, group=None, id: str = "") -> Transition:
        """The transition on this step's context (built before the step is recorded), then the step is recorded."""
        t = Transition(self.context(state), action, observed, group=group, id=id)
        self.push(state, action, observed)
        return t


__all__ = ["Context", "History", "Past", "Same", "Tracker"]
