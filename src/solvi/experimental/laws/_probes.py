"""The probe planner for live environments: where each test is a call into a real system, ask the question that tells
the hypotheses apart, safely and cheaply (see the package docstring)."""
from __future__ import annotations

import inspect
import math
import numbers
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence


@dataclass(frozen=True)
class Probe:
    """A chosen probe: the state and action to try, each hypothesis's prediction there, and the split score (how many
    hypotheses the answer can rule out at least: those outside the largest group of equal predictions); info: the
    expected information of its answer in bits (the entropy of the groups of equal predictions, every hypothesis
    equally likely); cost: its cost when `split` was given one."""
    state: Any
    action: Any
    predictions: tuple
    score: int
    info: float = 0.0
    cost: Any = None


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


def eig(group_weights: Sequence[float], noise: float) -> float:
    """The expected information gain in bits of one answer at a probe whose hypotheses fall into groups of equal
    prediction with these total weights (summing to 1), when an answer is wrong with probability `noise` and a wrong
    answer is one of the other groups' values, uniformly: H(answer) − H(answer | hypothesis)."""
    ws = [x for x in group_weights if x > 0]
    G = len(group_weights)
    if G < 2 or len(ws) < 2 and noise == 0:
        return 0.0
    e = float(noise)
    other = e / (G - 1)
    py = [x * (1 - e) + (1 - x) * other for x in group_weights]
    h_y = -sum(p * math.log2(p) for p in py if p > 0)
    h_n = -sum(p * math.log2(p) for p in [1 - e] + [other] * (G - 1) if p > 0)
    return max(0.0, h_y - h_n)


def split(hypotheses: Sequence, safe: Callable[[Any, Any], bool], probes: Iterable[tuple[Any, Any]], *,
          read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None,
          tol: float | None = None, same: Callable[[Any, Any], bool] | None = None,
          rtol: float | None = None, per_cost: bool = False, score: str = "split", noise: float = 0.0,
          weights: Sequence[float] | None = None) -> Probe | None:
    """The probe that best splits the hypotheses (law sets, or predictors `fn(state, action)`), among `probes` —
    (state, action) pairs — that `safe(state, action)` allows. Order: read-only probes first (`read_only(action)`, or
    `read_only(state, action)` when the kind of call is in the state),
    then the larger split, then the lower `cost(state, action)`, then the order given. None when no safe probe tells
    any two hypotheses apart. Predictions are told apart by exact equality, or within `tol` (numbers: |a − b| ≤ tol,
    tuples element by element), within a relative `rtol` (|a − b| ≤ tol + rtol·max(|a|, |b|)) or by `same(a, b)` —
    two formulas that differ by float rounding (x·0.1·10 and x) are the same hypothesis under tol=1e-9; over
    predictions from 0.003 to 3,000, rtol=1e-6 keeps a law 0.1 % off apart where no single tol does both.

    per_cost=True (opt-in; needs `cost`): the probe with the most information per unit of cost — `info / cost`, info
    the expected bits of its answer (Probe.info) — after read-only first; so a cheap probe that halves the hypotheses
    is taken before one ten times dearer that splits them a little better. A cost ≤ 0 counts as 1e-12 (a free probe
    first).

    score="eig" (opt-in): rank by the expected information gain of the answer under measurement noise — `noise` the
    chance that an answer is not the true outcome (0 ≤ noise < 0.5; a wrong answer is taken to be one of the other
    predicted values at that probe, uniformly) and `weights` the hypotheses' current probabilities (default: equal;
    normalised). Probe.info is then that gain in bits: H(answer) − H(answer | hypothesis). With noise 0 and equal
    weights it is the split's entropy (Probe.info of the default). Under noise the same probe may be the best again
    after its answer (a repeat measurement); probes are not removed once asked. After read-only first, the larger gain,
    then the lower cost, then the order given."""
    if per_cost and cost is None:
        raise ValueError("per_cost=True ranks by information per unit of cost: give cost=")
    if score not in ("split", "eig"):
        raise ValueError(f"score is 'split' or 'eig', not {score!r}")
    if not 0.0 <= float(noise) < 0.5:
        raise ValueError("noise is the chance an answer is wrong: 0 <= noise < 0.5")
    if score == "split" and (noise or weights is not None):
        raise ValueError("noise= and weights= are for score='eig'")
    if score == "eig" and per_cost:
        raise ValueError("per_cost=True ranks by information per unit of cost; give score='split' with it")
    preds = [_predictor(h) for h in hypotheses]
    if weights is not None:
        w = [float(x) for x in weights]
        if len(w) != len(preds) or any(not x >= 0 for x in w) or sum(w) <= 0:
            raise ValueError("weights: one non-negative number per hypothesis, not all zero")
        w = [x / sum(w) for x in w]
    else:
        w = [1.0 / len(preds)] * len(preds) if preds else []
    eq = sameness(tol, same, rtol=rtol)
    ro = _ro(read_only)
    best = None
    for i, (state, action) in enumerate(probes):
        ps = tuple(p(state, action) for p in preds)
        gs = groups_of(ps, eq)
        split_n = len(ps) - max((len(g) for g in gs), default=0)
        if split_n <= 0:
            continue
        if score == "eig":
            info = eig([sum(w[j] for j in g) for g in gs], float(noise))
            if info <= 0:
                continue
        else:
            info = -sum(len(g) / len(ps) * math.log2(len(g) / len(ps)) for g in gs)
        c = cost(state, action) if cost else None
        if score == "eig":
            key = (0 if ro is None or ro(state, action) else 1, -info, c if cost else 0, i)
        elif per_cost:
            key = (0 if ro is None or ro(state, action) else 1, -info / max(float(c), 1e-12), i)
        else:
            key = (0 if ro is None or ro(state, action) else 1, -split_n, c if cost else 0, i)
        if best is not None and key >= best[0]:
            continue
        if not safe(state, action):
            continue
        best = (key, Probe(state, action, ps, split_n, info, c))
    return best[1] if best else None


def discriminate(laws_a, laws_b, safe: Callable[[Any, Any], bool], probes: Iterable[tuple[Any, Any]], *,
                 read_only: Callable[..., bool] | None = None, cost: Callable[[Any, Any], Any] | None = None,
                 tol: float | None = None, same: Callable[[Any, Any], bool] | None = None,
                 rtol: float | None = None) -> Probe | None:
    """The cheapest safe probe where two law sets (or predictors) predict different outcomes — read-only actions
    before writes, then the lower cost, then the order given; None when they agree on every safe probe. tol / rtol /
    same: as in `split` (different = not within the tolerance)."""
    return split([laws_a, laws_b], safe, probes, read_only=read_only, cost=cost, tol=tol, same=same, rtol=rtol)


__all__ = ["Probe", "discriminate", "eig", "split"]
