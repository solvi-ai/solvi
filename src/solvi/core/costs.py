"""Run-time costs: CostBook, the moving average of each part's run time (ms) that every System keeps (`system.cost_book`)
and updates after every ask; the model calls a trace record holds and their dollars (_usages, price_of, recorded_calls,
cost_of); Cost (what a decision or a part of one cost: dollars, model calls, milliseconds, tokens) and Budget (limits
on them) with BudgetStop — shared by solvi.core.dispatch, solvi.core.slow.generate and solvi.core.slow.refine (defined in solvi.core.dispatch until
1.0, which re-exports them). Moved out of solvi.learned in 0.8 (nothing here is learned). MeasuredCosts
(System(cost_policy="measured")) was removed in 1.0: it showed no measured benefit."""
from __future__ import annotations

from dataclasses import dataclass

class CostBook:
    """Moving average of each part's run time in ms (exponential, alpha)."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.ms = {}
        self.n = {}

    def observe(self, name, ms):
        if name in self.ms:
            self.ms[name] += self.alpha * (ms - self.ms[name])
        else:
            self.ms[name] = ms
        self.n[name] = self.n.get(name, 0) + 1

    def get(self, part, default=1.0):
        if part.name in self.ms:
            return self.ms[part.name]
        if part.cost is not None:
            return float(part.cost)
        return default

    def __repr__(self):
        return "CostBook(" + ", ".join(f"{k}={v:.2f}ms" for k, v in sorted(self.ms.items())) + ")"


# --------------------------------------------------------------------------------------------------- recorded calls
# The model calls a trace record's extra holds, and their dollars: plain functions over the record's dicts (moved here
# from solvi.core.dispatch in 1.0, which re-exports them), so the system report reads them without importing the dispatcher.
def _usages(extra):
    """The model calls recorded in one trace record's extra → [(model, usage)]."""
    out = []
    if not isinstance(extra, dict):
        return out
    lm = extra.get("llm")
    if isinstance(lm, dict) and isinstance(lm.get("usage"), dict):
        out.append((lm.get("model"), lm["usage"]))
    gen = extra.get("generated")
    for m in gen if isinstance(gen, list) else [gen] if isinstance(gen, dict) else []:
        if isinstance(m, dict) and isinstance(m.get("usage"), dict):
            out.append((m.get("model"), m["usage"]))
    return out


def price_of(price, calls, recorded=False):
    """Dollars of recorded calls: price (dollars per million input and output tokens), a function (model, usage) →
    dollars, or None (unknown → None). recorded=True: with no price, the dollars the calls recorded themselves (a
    generator built with price= writes them in each call's usage as "usd") when every call has them."""
    if price is None:
        if recorded and calls and all(isinstance(u.get("usd"), (int, float)) for _, u in calls):
            return float(sum(u["usd"] for _, u in calls))
        return None if calls else 0.0
    total = 0.0
    for model, u in calls:
        if callable(price):
            total += float(price(model, u))
        else:
            pin, pout = price
            total += (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1e6
    return total


# ------------------------------------------------------------------------------------------------ budget and cost
_LIMITS = ("usd", "calls", "ms", "tokens")


@dataclass(frozen=True)
class Budget:
    """Limits on model calls: dollars, calls, milliseconds and tokens (input + output) — None: no limit on that one.
    The same Budget is per decision or in total, wherever it is given (solvi.core.dispatch, solvi.core.slow.generate, solvi.core.slow.refine)."""
    usd: float | None = None
    calls: int | None = None
    ms: float | None = None
    tokens: int | None = None

    def __post_init__(self):
        for k in _LIMITS:
            v = getattr(self, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != v):
                raise ValueError(f"Budget {k} must be a number ≥ 0 or None, not {v!r}")

    def over(self, cost, plus=None):
        """The first limit that `cost` (+ `plus`, an expected further cost) goes over → its reason, or None. A cost
        without dollars (no price known) is never over a limit in dollars: give a price with a budget in dollars."""
        for k in _LIMITS:
            lim = getattr(self, k)
            if lim is None:
                continue
            got, more = getattr(cost, k), getattr(plus, k) if plus is not None else 0
            if got is None or more is None:
                continue
            if got + more > lim + 1e-12:
                extra = f" + {more:.4g} expected" if plus is not None else ""
                return f"{k} {got:.4g}{extra} > {lim:g}"
        return None

    def used_up(self, cost):
        """The first limit `cost` has reached → its reason, or None."""
        for k in _LIMITS:
            lim, got = getattr(self, k), getattr(cost, k)
            if lim is not None and got is not None and got >= lim - 1e-12:
                return f"{k} {got:.4g} of {lim:g} used"
        return None

    def to_dict(self):
        d = {"usd": self.usd, "calls": self.calls, "ms": self.ms}
        if self.tokens is not None:                   # a budget without one hashes as before 1.0 (Dispatcher.config)
            d["tokens"] = self.tokens
        return d

    @classmethod
    def from_dict(cls, d):
        return None if d is None else cls(d.get("usd"), d.get("calls"), d.get("ms"), d.get("tokens"))


@dataclass
class Cost:
    """What a decision (or a part of one) cost: dollars (None when no price is known), model calls, milliseconds and
    tokens."""
    usd: float | None = 0.0
    calls: int = 0
    ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tokens(self):
        return self.input_tokens + self.output_tokens

    def __add__(self, o):
        usd = None if self.usd is None or o.usd is None else self.usd + o.usd
        return Cost(usd, self.calls + o.calls, self.ms + o.ms, self.input_tokens + o.input_tokens,
                    self.output_tokens + o.output_tokens)

    def __mul__(self, k):
        return Cost(None if self.usd is None else self.usd * k, int(round(self.calls * k)), self.ms * k,
                    int(round(self.input_tokens * k)), int(round(self.output_tokens * k)))

    def to_dict(self):
        return {"usd": None if self.usd is None else round(self.usd, 9), "calls": self.calls, "ms": round(self.ms, 3),
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("usd"), int(d.get("calls", 0)), float(d.get("ms", 0.0)), int(d.get("input_tokens", 0)),
                   int(d.get("output_tokens", 0)))


class BudgetStop(RuntimeError):
    """Stopped between two steps: what is left of the budget would not cover the next one (a slow path's next round,
    a refinement's next round, a generator's next request)."""


def recorded_calls(responses, generated=()):
    """The model calls recorded in responses' traces (and in generator records outside a trace, as refine keeps the
    proposer's) → [(model, usage)]."""
    out = []
    for res in responses:
        if res is None:
            continue
        for r in res.trace.records:
            out += _usages(getattr(r, "extra", None))
    for g in generated:
        out += _usages({"generated": g})
    return out


def cost_of(responses, price, ms=0.0, generated=(), recorded=False):
    """The Cost of what responses (and generator records) recorded, with `ms` as the time. recorded=True: without a
    price, the dollars the calls recorded (see price_of)."""
    calls = recorded_calls(responses, generated)
    return Cost(price_of(price, calls, recorded), len(calls), float(ms), sum(u.get("input_tokens", 0) for _, u in calls),
                sum(u.get("output_tokens", 0) for _, u in calls))


__all__ = ["Budget", "BudgetStop", "Cost", "CostBook", "cost_of", "price_of", "recorded_calls"]


def __getattr__(name):
    if name == "MeasuredCosts":
        raise AttributeError('solvi.core.costs.MeasuredCosts (System(cost_policy="measured")) was removed in 1.0: it showed no '
                             "measured benefit — declare cost= on the parts")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
