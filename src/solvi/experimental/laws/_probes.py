"""The probe planner for live environments: where each test is a call into a real system, ask the question that tells
the hypotheses apart, safely and cheaply (see the package docstring)."""
from __future__ import annotations

import inspect
import numbers
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


def sameness(tol: float | None = None, same: Callable[[Any, Any], bool] | None = None, *,
             rtol: float | None = None) -> Callable[[Any, Any], bool] | None:
    """The comparison of two predictions (or a prediction and an observation): None — exact equality (the default);
    `same(a, b)` → bool as given; `tol` / `rtol` — numbers equal within an absolute `tol` and a relative `rtol`
    (|a − b| ≤ tol + rtol·max(|a|, |b|); either alone, the other is 0), tuples and lists element by element, anything
    else by ==. The relative form is for a quantity measured over several orders of magnitude, where one absolute
    tolerance either merges the small values or splits the large ones. same= with neither."""
    if same is not None and (tol is not None or rtol is not None):
        raise ValueError("give tol= / rtol= or same=, not both")
    if same is not None:
        return same
    if tol is None and rtol is None:
        return None
    t = float(tol or 0.0)
    r = float(rtol or 0.0)
    if t < 0 or r < 0:
        raise ValueError("tol and rtol are non-negative numbers")

    def close(a, b):
        if isinstance(a, numbers.Number) and isinstance(b, numbers.Number):
            try:
                return a == b or abs(a - b) <= t + r * max(abs(a), abs(b))
            except TypeError:
                return a == b
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
        return a == b
    return close


def groups_of(values: Sequence, eq: Callable[[Any, Any], bool] | None) -> list[list[int]]:
    """The indices of `values` grouped by equal value: exact (hashing) when `eq` is None, else each value joins the
    first group whose first member it is `eq` to (a tolerance is not transitive: the first member decides)."""
    out: dict = {}
    if eq is None:
        for i, x in enumerate(values):
            out.setdefault(_k(x), []).append(i)
        return list(out.values())
    gs: list[list[int]] = []
    for i, x in enumerate(values):
        for g in gs:
            if eq(values[g[0]], x):
                g.append(i)
                break
        else:
            gs.append([i])
    return gs


def split(hypotheses: Sequence, safe: Callable[[Any, Any], bool], probes: Iterable[tuple[Any, Any]], *,
          read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None,
          tol: float | None = None, same: Callable[[Any, Any], bool] | None = None,
          rtol: float | None = None) -> Probe | None:
    """The probe that best splits the hypotheses (law sets, or predictors `fn(state, action)`), among `probes` —
    (state, action) pairs — that `safe(state, action)` allows. Order: read-only probes first (`read_only(action)`, or
    `read_only(state, action)` when the kind of call is in the state),
    then the larger split, then the lower `cost(state, action)`, then the order given. None when no safe probe tells
    any two hypotheses apart. Predictions are told apart by exact equality, or within `tol` (numbers: |a − b| ≤ tol,
    tuples element by element), within a relative `rtol` (|a − b| ≤ tol + rtol·max(|a|, |b|)) or by `same(a, b)` —
    two formulas that differ by float rounding (x·0.1·10 and x) are the same hypothesis under tol=1e-9; over
    predictions from 0.003 to 3,000, rtol=1e-6 keeps a law 0.1 % off apart where no single tol does both."""
    preds = [_predictor(h) for h in hypotheses]
    eq = sameness(tol, same, rtol=rtol)
    ro = _ro(read_only)
    best = None
    for i, (state, action) in enumerate(probes):
        ps = tuple(p(state, action) for p in preds)
        score = len(ps) - max((len(g) for g in groups_of(ps, eq)), default=0)
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
                 read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None,
                 tol: float | None = None, same: Callable[[Any, Any], bool] | None = None,
                 rtol: float | None = None) -> Probe | None:
    """The cheapest safe probe where two law sets (or predictors) predict different outcomes — read-only actions
    before writes, then the lower cost, then the order given; None when they agree on every safe probe. tol / rtol /
    same: as in `split` (different = not within the tolerance)."""
    return split([laws_a, laws_b], safe, probes, read_only=read_only, cost=cost, tol=tol, same=same, rtol=rtol)


__all__ = ["Probe", "discriminate", "split"]
