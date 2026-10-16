"""Coarse outcomes and a base predictor that backs off from specific to general (see the package docstring)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class OutcomeClass:
    """A predicted outcome class (the observation read through a `coarse` key): what a law of a coarse set predicts,
    and what `BackoffBase(outcome=)` answers. value: the class."""
    value: Any

    def __repr__(self):
        return f"OutcomeClass({self.value!r})"


def coarse(key: Callable[[Any], Any]) -> dict:
    """Outcomes grouped coarser than the observation: `LawSet(grammar, **coarse(lambda seen: seen["kind"]))`. A law
    then predicts the observation's class `key(observation)` (an OutcomeClass), and a prediction is right when the
    observation falls in that class — so details a law cannot know (a random amount, a timestamp) no longer make every
    row a mismatch. → the keyword arguments `effects`, `apply` and `same` for LawSet (do not pass those three too);
    `LawSet.report` then counts outcomes by class."""
    def cls(x):                                      # None: no prediction (no law, no base) — in no class
        return (False, None) if x is None else (True, x.value if isinstance(x, OutcomeClass) else key(x))

    def effects(state, observed):
        return [OutcomeClass(key(observed))]
    effects.outcome_key = key                        # LawSet.report counts the outcomes by class

    def apply(effect, state):
        return effect

    def same(a, b):
        return cls(a) == cls(b)
    return {"effects": effects, "apply": apply, "same": same}


def _level(spec):
    """A level: a function (state, action) → key, or a tuple of state keys (with the action, always)."""
    if callable(spec):
        return spec
    names = (spec,) if isinstance(spec, str) else tuple(spec)
    return lambda state, action: tuple(state.get(n) if isinstance(state, dict) else getattr(state, n, None)
                                       for n in names)


class BackoffBase:
    """A base predictor that backs off from specific to general: `levels`, most specific first — each a tuple of state
    keys (`("front", "wood")`) or a function (state, action) → key; the action is always part of the key, and the
    action alone is the last level unless `()` is given. `base(state, action)` is the most frequent observation among
    the fitted transitions with the same key at the first level that has at least `min_rows` of them (ties: the first
    seen); None where no level has enough. outcome: a `coarse` key — the base then counts and answers classes
    (OutcomeClass), as the laws of a coarse set do.

        base = BackoffBase([("front", "wood"), ("front",), ()], min_rows=5).fit(transitions)
        laws = LawSet(grammar, base=base)        # what is predicted where no law applies

    Fit it once, on transitions the laws are not tested on, before the set uses it: a LawSet caches where its base is
    right per buffer row (a base refitted while in use gives stale counts — give the set a new BackoffBase).
    `explain(state, action)` → {"value", "level", "key", "rows", "share"} (level None: no level had enough rows)."""

    def __init__(self, levels: Sequence = ((),), *, min_rows: int = 5, outcome: Callable[[Any], Any] | None = None):
        if int(min_rows) < 1:
            raise ValueError("min_rows is a number of rows (≥ 1)")
        self.levels = [_level(x) for x in levels] or [_level(())]
        self.min_rows, self.outcome = int(min_rows), outcome
        self.counts: list[dict] = [{} for _ in self.levels]       # per level: (action, key) → Counter of outcomes
        self._first: dict = {}                                       # outcome key → its value, the first seen
        self.n = 0

    def _out(self, observed):
        return OutcomeClass(self.outcome(observed)) if self.outcome is not None else observed

    def fit(self, transitions: Iterable) -> "BackoffBase":
        """Count transitions (Transition objects, a Buffer, or (state, action, observed) triples) → self."""
        from ._core import Buffer, _key
        rows = [t for a in transitions._rows for t in transitions._rows[a]] if isinstance(transitions, Buffer) \
            else transitions
        for t in rows:
            s, a, o = (t.state, t.action, t.observed) if hasattr(t, "observed") else t
            v = self._out(o)
            k = _key(v)
            self._first.setdefault(k, v)
            for lv, c in zip(self.levels, self.counts):
                c.setdefault((_key(a), _key(lv(s, a))), Counter())[k] += 1
            self.n += 1
        return self

    def explain(self, state, action) -> dict:
        from ._core import _key
        for i, (lv, c) in enumerate(zip(self.levels, self.counts)):
            key = _key(lv(state, action))
            got = c.get((_key(action), key))
            if got is None:
                continue
            rows = sum(got.values())
            if rows < self.min_rows:
                continue
            k, m = max(got.items(), key=lambda kv: kv[1])        # Counter keeps insertion order: ties, the first seen
            return {"value": self._first[k], "level": i, "key": key, "rows": rows, "share": m / rows}
        return {"value": None, "level": None, "key": None, "rows": 0, "share": None}

    def __call__(self, state, action):
        return self.explain(state, action)["value"]


__all__ = ["BackoffBase", "OutcomeClass", "coarse"]
