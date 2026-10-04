"""Strategist: given the questions and init_state, picks functions AND checks from the catalog and assembles the flow.

Targets for each question:
  • it has a rule → the rule's arguments;
  • an answer head is fitted → the facts selected during fitting;
  • otherwise → the `uses` hint; otherwise everything computable from init_state (marked "flow not narrowed").
Then it walks back through signatures down to what init_state provides; the question's required parts (`requires`) are always added.
Checks: besides those the targets need, the strategist takes every catalog check whose inputs are all already in the flow and
that touches at least one COMPUTED (non-input) fact — a check on computed facts. Nothing else from the catalog is executed.
Typed facts: the flow records the type of each fact it uses (`flow.types`: a producer's return type, or for a given fact the
type its typed readers expect); producer / consumer types were already checked when the parts were registered.
Decisions with a model: decision parts that read the same facts with the same model, when the model can answer several
questions in one forward pass, are grouped (`flow.batches`); the executor scores each group in one pass."""
from __future__ import annotations

import random
from collections import deque
from typing import Any, Protocol, runtime_checkable

from ... import _deprecate
from ..catalog import Quote
from ..runtime import Flow, Step, scalar_row, then_inputs   # defined there: the runtime runs a flow; re-exported


class PlanError(Exception):
    pass


def computable(catalog, init_keys):
    """All facts computable from init_state (closure over signatures)."""
    have = set(init_keys)
    changed = True
    while changed:
        changed = False
        for p in catalog.parts.values():
            if p.name not in have and all(x in have for x in p.inputs):
                have.add(p.name)
                changed = True
    return have


def given_facts(catalog, questions=()):
    """The facts the catalog reads but no part produces — what init_state has to provide: arguments of parts, rules and
    hard checks' `then` functions, and the questions' `uses` hints. (A rule that reads a question's name reads a given fact: answers are not facts.)"""
    read = set()
    for p in list(catalog.parts.values()) + list(catalog.rules.values()):
        read.update(p.inputs)
        for qn in (p.then or {}) if p.kind == "check" else ():
            read.update(then_inputs(p, qn))          # a `then` function's arguments are facts too
    for q in questions:
        read.update(q.uses or ())
    return read - set(catalog.parts)


@runtime_checkable
class Strategist(Protocol):
    """The planner of a System: which parts and checks run for these questions on these given facts.

    You implement: `plan(catalog, questions, init_keys, heads=None)` → a `Flow` (solvi.core.runtime: the steps in an
    order their inputs allow, the parts per question, the unresolved questions with the reasons). Pass it as
    `System(..., strategist=yours)`. Give the flow a `strategy` dict (as `CostStrategist` does) to have the plan hashed
    into every trace (`solvi.core.plan.cost.plan_record`).

    You get for free: hard checks enforced whatever the plan says (a hard check whose `then=` names a question must be
    in that question's flow, or System(...) refuses the strategist), types checked on every step, the plan recorded and
    replayed, every place that plans for the System (ask, fit, serve's input schemas, `solvi check`) seeing the same
    flow.

    Stability: stable. `DefaultStrategist` and `solvi.core.plan.cost.CostStrategist` are stable to use, provisional
    to subclass."""

    def plan(self, catalog: Any, questions: Any, init_keys: Any, heads: Any = None) -> Flow: ...


class DefaultStrategist:
    """The deterministic planner every System uses when it is given none (`plan` in this module), as an object: the
    targets of each question (its rule's arguments, a fitted head's facts, its `uses` hint), walked back to the given
    facts, every check on computed facts that the flow can run, the required parts. `System(strategist=
    DefaultStrategist())` plans exactly as `System()` does. Subclass it to change one step and keep the rest (call
    `super().plan(...)` and edit the Flow)."""

    record = False                                   # the deterministic plan is not written into the trace

    def plan(self, catalog, questions, init_keys, heads=None):
        """→ Flow (see the module docs)."""
        return plan(catalog, questions, init_keys, heads)

    def computable(self, catalog, init_keys):
        """The facts this planner can compute from these given facts (every alternative producer's inputs needed)."""
        return computable(catalog, init_keys)

    def __repr__(self):
        return "DefaultStrategist()"


def plan(catalog, questions, init_keys, heads=None):
    heads = heads or {}
    init_keys = set(init_keys)
    reach = computable(catalog, init_keys)
    chosen: dict[str, Step] = {}
    done: dict = {}                 # (fact, question) → walked: resolvable?
    per_q, unresolved = {}, {}

    def need(fact, why, q, trail=()):
        if fact in init_keys:
            return True
        p = catalog.parts.get(fact)
        if p is None or fact not in reach:          # a fact on a cycle is never reachable: its question abstains
            unresolved.setdefault(q, set()).add(fact)
            return False
        if (fact, q) in done:                       # already walked for this question: same result (a fact reachable by
            st = chosen[fact]                       # several routes is walked once — without this, exponential time)
            if why not in st.reasons:
                st.reasons.append(why)
            return done[(fact, q)]
        ok = all(need(x, f"input for {fact}", q, trail + (fact,)) for x in p.inputs)
        st = chosen.setdefault(fact, Step(p))
        if why not in st.reasons:
            st.reasons.append(why)
        per_q.setdefault(q, set()).add(fact)
        done[(fact, q)] = ok
        return ok

    for q in questions:
        rule = catalog.rules.get(q.name)
        if rule is not None:
            targets, why = rule.inputs, f"rule {q.name}"
        elif q.name in heads:
            targets, why = heads[q.name].features, f"answer feature {q.name}"
        elif q.uses:
            targets, why = q.uses, f"uses hint {q.name}"
        else:
            targets = sorted(f for f in reach - init_keys if catalog.parts[f].kind != "rule")
            why = f"{q.name}: flow not narrowed (no rule, fit or uses)"
        for f in targets:
            need(f, why, q.name)
        for c in q.requires:
            if c not in catalog.parts:
                raise PlanError(f"{q.name!r} requires {c!r}, which is not in the catalog")
            need(c, f"checkpoint {q.name}", q.name)      # the 0.7 wording: stored flows compare
        if rule is not None:
            per_q.setdefault(q.name, set())
    # checks on computed facts — decided per question, so a question's flow does not depend on which other questions are asked
    def on_computed(qs):
        for p in catalog.parts.values():
            if p.kind != "check" or p.name not in reach:
                continue
            for q, fs in per_q.items():
                if p.name in fs or q not in qs:
                    continue
                have = fs | init_keys
                touched = [x for x in p.inputs if x in fs and x not in init_keys]
                if touched and all(x in have for x in p.inputs):
                    st = chosen.setdefault(p.name, Step(p))
                    why = "check on computed: " + ", ".join(touched)
                    if why not in st.reasons:
                        st.reasons.append(why)
                    fs.add(p.name)
    on_computed(per_q.keys())
    # hard checks wired by `then` (1.0): a hard check whose `then` names a question is in that question's flow, and so are
    # the inputs of a `then` function — added to that question alone (the other questions' flows stay as they were); a
    # question that gained steps gets its checks on computed facts again
    wired = set()
    for q in questions:
        for p in catalog.parts.values():
            if p.kind != "check" or not p.hard or q.name not in (p.then or {}):
                continue
            if p.name not in per_q.get(q.name, ()):
                need(p.name, f"hard check for {q.name} (then)", q.name)
                wired.add(q.name)
            fn = p.then[q.name]
            if callable(fn):
                for x in then_inputs(p, q.name):
                    if x not in per_q.get(q.name, ()) and x not in init_keys:
                        need(x, f"then of {p.name} for {q.name}", q.name)
                        wired.add(q.name)
    if wired:
        on_computed(wired)
    # execution order is topological
    order, seen = [], set()

    def visit(f):
        if f in seen or f in init_keys:
            return
        seen.add(f)
        for x in chosen[f].part.inputs:
            if x in chosen:
                visit(x)
        order.append(chosen[f])
    # hard checks and what they depend on go first: when one fails, the executor can skip the rest (early exit)
    for f in sorted(f for f in chosen if chosen[f].part.kind == "check" and chosen[f].part.hard) + sorted(chosen):
        visit(f)
    for q in questions:
        rule = catalog.rules.get(q.name)
        if rule is not None:
            order.append(Step(rule, [f"answer {q.name}"]))
    skipped = {}
    for name, p in catalog.parts.items():
        if name not in chosen:
            skipped[name] = "no inputs" if name not in reach else "not needed for questions"
    types = {}
    if catalog.readers or catalog.types:            # typed catalogs only: the type of each fact in the flow
        for st in order:
            f = st.part.name
            if f in catalog.types:
                types[f] = catalog.types[f]
            for x in st.part.inputs:
                if x in init_keys and x not in types and catalog.readers.get(x):
                    types[x] = next(iter(catalog.readers[x].values()))
    batches = []
    if getattr(catalog, "decisions", 0):             # catalogs without decision parts do no work here
        from ..runtime import plan_batches
        batches = plan_batches(order)
    return Flow(order, {q: sorted(v) for q, v in per_q.items()}, skipped, {q: sorted(v) for q, v in unresolved.items()},
                types, batches)


# --- learned order and producer policy (solvi.learned up to 0.7)
# OrderModel — per hard check, P(this check fails on this input) from facts known before running it (init_state and
# optionally cheap computed facts). The executor then runs hard checks one at a time, most expected saving first
# (P(fail) × cost it can save ÷ cost to evaluate), and stops as soon as the failed ones settle every question. Answers do
# not depend on the order: a question is settled only when every hard check declared before the deciding one (among those
# governing it) has been evaluated — exactly the "first declared failed check decides" rule.
# ProducerPolicy — for a fact with several producers, the order in which to try them on this input: predicted
# P(accepted) and P(agrees with the reference producer | accepted), and cost. Outputs are still accepted only by the
# producer's own validator; the policy only chooses who goes first. It learns from the outcomes of past runs, instantly.
# Every probability is an online model on cheap features (Binary): Laplace counts until `min_rows` labelled rows, then a
# FastHead (closed-form ridge, solvi.core.deciders.heads) refitted every `refit_every` rows and updated by a rank-one step in between.


class Binary:
    """P(y = 1 | row), learned online."""

    def __init__(self, min_rows=20, refit_every=50, ring=2000, prior=0.5):
        self.min_rows, self.refit_every = min_rows, refit_every
        self.rows = deque(maxlen=ring)
        self.n = self.pos = 0
        self.prior = prior
        self.head = None
        self._since = 0

    observe = _deprecate.removed_attr("observe()", "teach()", "Binary")

    def teach(self, row, y):
        """One labelled row: y is True or False."""
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
                self.head.teach(row, "yes" if y else "no")
                self._since += 1
            except Exception:  # noqa: BLE001  (a category never seen at fit time → refit)
                self.refit()

    def refit(self):
        from ..deciders.heads import FastHead
        rows = [r for r, _ in self.rows]
        feats = sorted({k for r in rows for k in r})
        try:
            self.head = FastHead(["yes", "no"], pairs=False, refit=None).fit(rows, ["yes" if y else "no" for _, y in self.rows], feats)
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
        m.teach(row, failed)

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
            self._m(self.accept, (group.name, name)).teach(row, ok)
        if ref in outcomes and outcomes[ref][0]:
            from ..runtime import vhash
            want = vhash(_plain(outcomes[ref][1]))
            for name, (ok, v) in outcomes.items():
                if name != ref and ok:
                    self._m(self.agree, (group.name, name)).teach(row, vhash(_plain(v)) == want)


def _plain(v):
    return v.value if isinstance(v, Quote) else v


__all__ = ["Binary", "computable", "DefaultStrategist", "Flow", "given_facts", "OrderModel", "plan", "PlanError",
           "ProducerPolicy", "scalar_row", "Step", "Strategist"]
