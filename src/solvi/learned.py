"""The learned strategist: models propose an execution order, the deterministic layer verifies and keeps the guarantees.

  • CostBook — a moving average of each part's run time (ms), kept on the System and updated after every ask.
  • OrderModel — per hard check, P(this check fails on this input) from facts known before running it (init_state and
    optionally cheap computed facts). The executor then runs hard checks one at a time, most expected saving first
    (P(fail) × cost it can save ÷ cost to evaluate), and stops as soon as the failed ones settle every question.
    Answers do not depend on the order: a question is settled only when every hard check declared before the deciding one
    (among those governing it) has been evaluated — exactly the "first declared failed check decides" rule.
  • ProducerPolicy — for a fact with several producers, the order in which to try them on this input: predicted
    P(accepted) and P(agrees with the reference producer | accepted), and cost. Outputs are still accepted only by the
    producer's own validator; the policy only chooses who goes first. It learns from the outcomes of past runs, instantly.

Every probability is an online model on cheap features: Laplace counts until `min_rows` labelled rows, then a FastHead
(closed-form ridge, solvi.fast) refitted every `refit_every` rows and updated by a rank-one step in between."""
from __future__ import annotations

import random
from collections import deque

from .core import Quote


def scalar_row(values, keys=None):
    """Cheap features from a dict of facts: numbers, booleans and short strings as they are; a long string gives its length;
    a list / tuple / dict gives its length. Anything else is left out."""
    out = {}
    for k in (values if keys is None else keys):
        if k not in values:
            continue
        v = values[k]
        if isinstance(v, Quote):
            v = v.value
        if isinstance(v, bool) or (isinstance(v, (int, float)) and not isinstance(v, bool)):
            out[k] = v
        elif isinstance(v, str):
            if len(v) <= 40:
                out[k] = v
            else:
                out[k + "#len"] = float(len(v))
        elif isinstance(v, dict):                         # one level down: {"invoice": {"currency": "EUR"}} → invoice.currency
            out[k + "#len"] = float(len(v))
            for kk, x in list(v.items())[:50]:
                if isinstance(x, bool) or isinstance(x, (int, float)) or (isinstance(x, str) and len(x) <= 40):
                    out[f"{k}.{kk}"] = x
        elif isinstance(v, (list, tuple, set)):
            out[k + "#len"] = float(len(v))
    return out


class Binary:
    """P(y = 1 | row), learned online."""

    def __init__(self, min_rows=20, refit_every=50, ring=2000, prior=0.5):
        self.min_rows, self.refit_every = min_rows, refit_every
        self.rows = deque(maxlen=ring)
        self.n = self.pos = 0
        self.prior = prior
        self.head = None
        self._since = 0

    def observe(self, row, y):
        y = bool(y)
        self.rows.append((row, y))
        self.n += 1
        self.pos += y
        if len(self.rows) < self.min_rows or len({b for _, b in self.rows}) < 2:
            return
        if self.head is None or self._since >= self.refit_every:
            self.refit()
        else:
            try:
                self.head.update(row, "yes" if y else "no")
                self._since += 1
            except Exception:  # noqa: BLE001  (a category never seen at fit time → refit)
                self.refit()

    def refit(self):
        from .fast import FastHead
        rows = [r for r, _ in self.rows]
        feats = sorted({k for r in rows for k in r})
        try:
            self.head = FastHead(["yes", "no"], pairs=False).fit(rows, ["yes" if y else "no" for _, y in self.rows], feats)
            if not self.head.features:
                self.head = None
        except Exception:  # noqa: BLE001
            self.head = None
        self._since = 0

    def rate(self):
        return (self.pos + self.prior * 2) / (self.n + 2)             # Laplace-smoothed base rate

    def p(self, row):
        if self.head is None:
            return self.rate()
        s = self.head.scores(row)
        p = float(s[0] - s[1] + 1) / 2                                # ridge on ±: (score_yes − score_no + 1) / 2
        n = self.n
        w = n / (n + 10.0)                                            # shrink towards the base rate while data is scarce
        return min(1.0, max(0.0, w * p + (1 - w) * self.rate()))


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
    """Costs from measurements for the cost-optimal planner (System(..., costs="measured") or costs=MeasuredCosts(...)).

    The planner (solvi.strategy.ModelStrategist with producers="equivalent") picks the cheapest plan by the cost of each
    producer: its measured run time (system.costs, a moving average in ms) once it has run `min_samples` times; before
    that its declared `cost=`, or — undeclared — 0 ms, so that it is tried and measured (warm-up). A producer not run for
    `recheck` asks counts 0 ms again for one plan (it may have got faster; None: never). `alpha` sets the smoothing of
    system.costs (the weight of the newest run; None: keep CostBook's). `System.freeze_costs()` fixes the costs the planner
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


class OrderModel:
    """P(hard check fails | cheap facts), one online model per hard check."""

    def __init__(self, features=None, min_rows=20, refit_every=50):
        self.features = list(features or [])        # computed facts to use besides init_state (cheap ones, computed early)
        self.models = {}
        self.min_rows, self.refit_every = min_rows, refit_every

    def row(self, vals, init_keys):
        return scalar_row(vals, list(init_keys) + [f for f in self.features if f not in init_keys])

    def observe(self, check, row, failed):
        m = self.models.setdefault(check, Binary(self.min_rows, self.refit_every))
        m.observe(row, failed)

    def p_fail(self, check, row):
        m = self.models.get(check)
        return 0.5 if m is None else m.p(row)

    def n(self, check):
        m = self.models.get(check)
        return 0 if m is None else m.n


class ProducerPolicy:
    """Chooses the order of a fact's alternative producers per input and learns from outcomes.

    Alternatives whose predicted agreement with the reference (the last declared producer, normally the most trusted) is at
    least `quality` are tried first, cheapest expected cost first (cost ÷ P(accepted)); the rest follow in the same order.
    With probability `explore` the remaining producers also run after the accepted one, as a shadow: their outputs are never
    used, only compared, which gives agreement labels for producers the policy would otherwise stop trying."""

    def __init__(self, quality=0.9, explore=0.1, seed=0, min_rows=20, refit_every=50):
        self.quality, self.explore = quality, explore
        self.rng = random.Random(seed)
        self.accept, self.agree = {}, {}
        self.min_rows, self.refit_every = min_rows, refit_every

    def _m(self, table, key):
        return table.setdefault(key, Binary(self.min_rows, self.refit_every))

    def plan(self, group, row, costs):
        """→ (ordered alternatives, notes per alternative, shadow?)"""
        ref = group.alternatives[-1].name
        info = []
        for pos, a in enumerate(group.alternatives):
            pa = self._m(self.accept, (group.name, a.name)).p(row)
            qa = 1.0 if a.name == ref else self._m(self.agree, (group.name, a.name)).p(row)
            c = costs.get(a)
            info.append((qa < self.quality, c / max(pa, 1e-3), pos, a, pa, qa, c))
        info.sort(key=lambda t: t[:3])
        notes = {a.name: f"P(accepted) {pa:.2f}, P(agrees with {ref}) {qa:.2f}, cost {c:.1f} ms"
                 for _, _, _, a, pa, qa, c in info}
        return [t[3] for t in info], notes, self.rng.random() < self.explore

    def observe(self, group, row, outcomes):
        """outcomes: {producer: (accepted, value)} for the producers that ran on this input."""
        ref = group.alternatives[-1].name
        for name, (ok, _) in outcomes.items():
            self._m(self.accept, (group.name, name)).observe(row, ok)
        if ref in outcomes and outcomes[ref][0]:
            from .runtime import vhash
            want = vhash(_plain(outcomes[ref][1]))
            for name, (ok, v) in outcomes.items():
                if name != ref and ok:
                    self._m(self.agree, (group.name, name)).observe(row, vhash(_plain(v)) == want)


def _plain(v):
    return v.value if isinstance(v, Quote) else v

