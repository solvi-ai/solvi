"""Run-time costs: CostBook, the moving average of each part's run time (ms) that every System keeps (`system.cost_book`)
and updates after every ask; MeasuredCosts, the settings under which the cost-optimal planner plans with those
measurements (System(cost_policy="measured")). Moved out of solvi.learned in 0.8 (nothing here is learned)."""
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


class MeasuredCosts:
    """Costs from measurements for the cost-optimal planner (System(..., cost_policy="measured") or costs=MeasuredCosts(...)).

    The planner (solvi.strategy.CostStrategist with producers="equivalent") picks the cheapest plan by the cost of each
    producer: its measured run time (system.cost_book, a moving average in ms) once it has run `min_samples` times; before
    that its declared `cost=`, or — undeclared — 0 ms, so that it is tried and measured (warm-up). A producer not run for
    `recheck` asks counts 0 ms again for one plan (it may have got faster; None: never). `alpha` sets the smoothing of
    system.cost_book (the weight of the newest run; None: keep CostBook's). `System.freeze_costs()` fixes the costs the planner
    uses; the plan record of each trace says which cost decided each choice and where it came from."""

    def __init__(self, min_samples=3, recheck=50, alpha=None):
        if min_samples < 1:
            raise ValueError("min_samples must be at least 1")
        if recheck is not None and recheck < 1:
            raise ValueError("recheck must be a positive number of asks, or None")
        if alpha is not None and not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.min_samples, self.recheck, self.alpha = min_samples, recheck, alpha
        self.frozen = None                          # (costs, sources) fixed by System.freeze_costs
        self.asks = 0                               # asks observed
        self.seen = {}                              # part → the ask it last ran in

    def __repr__(self):
        return (f"MeasuredCosts(min_samples={self.min_samples}, recheck={self.recheck}, alpha={self.alpha}"
                + (", frozen" if self.frozen is not None else "") + ")")

    def observe(self, names):
        """An ask ran these parts (their timings went to the CostBook)."""
        self.asks += 1
        for n in names:
            self.seen[n] = self.asks

    def costs(self, book, producers):
        """→ ({producer: cost}, {producer: (source, runs)}) for the planner; sources: measured, declared, warm-up,
        recheck, or frozen (see freeze)."""
        if self.frozen is not None:
            return dict(self.frozen[0]), dict(self.frozen[1])
        out, src = {}, {}
        for a in producers:
            n = book.n.get(a.name, 0)
            if n >= self.min_samples:
                if self.recheck is not None and self.asks - self.seen.get(a.name, 0) >= self.recheck:
                    out[a.name], src[a.name] = 0.0, ("recheck", n)
                else:
                    out[a.name], src[a.name] = float(book.ms[a.name]), ("measured", n)
            elif a.cost is not None:
                out[a.name], src[a.name] = float(a.cost), ("declared", n)
            else:
                out[a.name], src[a.name] = 0.0, ("warm-up", n)
        return out, src

    def freeze(self, book, producers, unit=1.0):
        """Fix the planner's costs: measured where a producer has run at all, else declared, else `unit` (no warm-up,
        no recheck) → {producer: cost}."""
        out, src = {}, {}
        for a in producers:
            n = book.n.get(a.name, 0)
            if n:
                out[a.name], src[a.name] = float(book.ms[a.name]), ("frozen: measured", n)
            elif a.cost is not None:
                out[a.name], src[a.name] = float(a.cost), ("frozen: declared", 0)
            else:
                out[a.name], src[a.name] = float(unit), ("frozen: unit", 0)
        self.frozen = (out, src)
        return dict(out)


__all__ = ["CostBook", "MeasuredCosts"]
