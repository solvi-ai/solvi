"""The probe planner for live environments: where each test is a call into a real system, ask the question that tells
the hypotheses apart, safely and cheaply (see the package docstring)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class Probe:
    """A chosen probe: the state and action to try, each hypothesis's prediction there, and the split score (how many
    hypotheses the answer can rule out at least: those outside the largest group of equal predictions)."""
    state: Any
    action: Any
    predictions: tuple
    score: int


def _ro(read_only):
    """read_only(action) or read_only(state, action) → a function of (state, action)."""
    if read_only is None:
        return None
    return read_only if _required_args(read_only) >= 2 else (lambda state, action: read_only(action))


def _required_args(fn):
    """The required positional arguments of `fn` (1 when it cannot be read); _core keeps its own copy so the two
    modules do not import each other."""
    try:
        ps = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return 1
    return sum(1 for p in ps if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty)


def _predictor(h):
    return h.predict if hasattr(h, "predict") else h


def _k(x):
    try:
        hash(x)
        return x
    except TypeError:
        return repr(x)


def split(hypotheses: Sequence, safe: Callable[[Any, Any], bool], probes: Iterable[tuple[Any, Any]], *,
          read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None) -> Probe | None:
    """The probe that best splits the hypotheses (law sets, or predictors `fn(state, action)`), among `probes` —
    (state, action) pairs — that `safe(state, action)` allows. Order: read-only probes first (`read_only(action)`, or
    `read_only(state, action)` when the kind of call is in the state),
    then the larger split, then the lower `cost(state, action)`, then the order given. None when no safe probe tells
    any two hypotheses apart."""
    preds = [_predictor(h) for h in hypotheses]
    ro = _ro(read_only)
    best = None
    for i, (state, action) in enumerate(probes):
        ps = tuple(p(state, action) for p in preds)
        counts: dict = {}
        for x in ps:
            counts[_k(x)] = counts.get(_k(x), 0) + 1
        score = len(ps) - max(counts.values(), default=0)
        if score <= 0:
            continue
        key = (0 if ro is None or ro(state, action) else 1, -score, cost(state, action) if cost else 0, i)
        if best is not None and key >= best[0]:
            continue
        if not safe(state, action):
            continue
        best = (key, Probe(state, action, ps, score))
    return best[1] if best else None


def discriminate(laws_a, laws_b, safe: Callable[[Any, Any], bool], probes: Iterable[tuple[Any, Any]], *,
                 read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None
                 ) -> Probe | None:
    """The cheapest safe probe where two law sets (or predictors) predict different outcomes — read-only actions
    before writes, then the lower cost, then the order given; None when they agree on every safe probe."""
    return split([laws_a, laws_b], safe, probes, read_only=read_only, cost=cost)


__all__ = ["Probe", "discriminate", "split"]
