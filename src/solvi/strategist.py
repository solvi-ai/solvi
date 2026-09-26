"""Strategist: given the questions and init_state, picks functions AND checks from the catalog and assembles the flow.

Targets for each question:
  • it has a rule → the rule's arguments;
  • an answer head is fitted → the facts selected during fitting;
  • otherwise → the `uses` hint; otherwise everything computable from init_state (marked "flow not narrowed").
Then it walks back through signatures down to what init_state provides; the question's checkpoints are always added.
Checks: besides those the targets need, the strategist takes every catalog check whose inputs are all already in the flow and
that touches at least one COMPUTED (non-input) fact — a check on computed facts. Nothing else from the catalog is executed."""
from __future__ import annotations

from dataclasses import dataclass, field


class PlanError(Exception):
    pass


@dataclass
class Step:
    part: object
    reasons: list = field(default_factory=list)


@dataclass
class Flow:
    steps: list                     # in execution order
    per_question: dict              # question → names of the parts in its flow
    skipped: dict                   # catalog part → why it was not taken
    unresolved: dict                # question → facts nothing can compute

    def __str__(self):
        lines = []
        for i, s in enumerate(self.steps, 1):
            p = s.part
            lines.append(f"{i:2d}. {p.kind:7s} {p.name:24s} ← {', '.join(p.inputs) or '—'}   [{'; '.join(s.reasons)}]")
        if self.skipped:
            lines.append("not taken: " + ", ".join(f"{k} ({v})" for k, v in sorted(self.skipped.items())))
        return "\n".join(lines)


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


def plan(catalog, questions, init_keys, heads=None):
    heads = heads or {}
    init_keys = set(init_keys)
    reach = computable(catalog, init_keys)
    chosen: dict[str, Step] = {}
    per_q, unresolved = {}, {}

    def need(fact, why, q, trail=()):
        if fact in init_keys:
            return True
        p = catalog.producer(fact)
        if p is None or fact not in reach:
            unresolved.setdefault(q, set()).add(fact)
            return False
        if fact in trail:
            raise PlanError(f"cycle in catalog: {' → '.join(trail + (fact,))}")
        ok = all(need(x, f"input for {fact}", q, trail + (fact,)) for x in p.inputs)
        st = chosen.setdefault(fact, Step(p))
        if why not in st.reasons:
            st.reasons.append(why)
        per_q.setdefault(q, set()).add(fact)
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
        for c in q.checkpoints:
            if c not in catalog.parts:
                raise PlanError(f"checkpoint {c} not found in catalog")
            need(c, f"checkpoint {q.name}", q.name)
        if rule is not None:
            per_q.setdefault(q.name, set())
    # checks on computed facts
    computed = {f for f in chosen if f not in init_keys}
    for p in catalog.parts.values():
        if p.kind != "check" or p.name in chosen:
            continue
        if p.name in reach and all(x in chosen or x in init_keys for x in p.inputs) and any(x in computed for x in p.inputs):
            touched = [x for x in p.inputs if x in computed]
            st = chosen.setdefault(p.name, Step(p))
            st.reasons.append("check on computed: " + ", ".join(touched))
            for q, fs in per_q.items():
                if any(x in fs for x in touched):
                    fs.add(p.name)
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
    return Flow(order, {q: sorted(v) for q, v in per_q.items()}, skipped, {q: sorted(v) for q, v in unresolved.items()})
