"""Run-time costs: CostBook, the moving average of each part's run time (ms) that every System keeps (`system.cost_book`)
and updates after every ask; the model calls a trace record holds and their dollars (_usages, price_of). Moved out of
solvi.learned in 0.8 (nothing here is learned). MeasuredCosts (System(cost_policy="measured")) was removed in 1.0: it
showed no measured benefit."""
from __future__ import annotations

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
# from solvi.dispatch in 1.0, which re-exports them), so the system report reads them without importing the dispatcher.
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


def price_of(price, calls):
    """Dollars of recorded calls: price (dollars per million input and output tokens), a function (model, usage) →
    dollars, or None (unknown → None)."""
    if price is None:
        return None if calls else 0.0
    total = 0.0
    for model, u in calls:
        if callable(price):
            total += float(price(model, u))
        else:
            pin, pout = price
            total += (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1e6
    return total


__all__ = ["CostBook", "price_of"]


def __getattr__(name):
    if name == "MeasuredCosts":
        raise AttributeError('solvi.costs.MeasuredCosts (System(cost_policy="measured")) was removed in 1.0: it showed no '
                             "measured benefit — declare cost= on the parts")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
