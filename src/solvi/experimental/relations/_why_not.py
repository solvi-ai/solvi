"""Why a relation was NOT derived — the mirror of `explain` (which shows the cells an answer used).

    rep = why_not(table, facts, "don", "dot", "aunt")      # facts: (a, relation, b) — «b is a's relation»
    rep.status        # "derivable" | "disputed" | "other" | "missing_cell" | "identity" | "underivable" | "no_path"
    rep.gaps          # [Gap(left, right, unblocks, example)]: absent cells, most paths unblocked first
    rep.summary()     # one line: "don → dot: no cell father ∘ sister → ? (would unblock 1 path; e.g. father ∘ sister)"
    rep.agenda_items(); rep.to_agenda(agenda, table)      # the refusal as a learning target («learn cell father ∘ sister»)

The paths: every simple path x → y of at most `max_len` facts (a fact is followed from a to b; with `inverse=` a
relation → its converse, also from b to a under the converse's name), in a fixed order, at most `max_paths` (then
`truncated`). Each path's chain of relations is derived by the table (CYK over every bracketing, its agreement rule);
the pooled derivation over all paths is `derivation` (`CompositionTable.derive_paths`). Status, in order: no path →
"no_path" (a missing FACT, not a rule); the pooled derivation answers `relation` (or anything, when no relation is
asked) → "derivable"; paths read as more than one relation → "disputed" (a conflict, not a hole: `readings`); one
reading that is not the asked relation → "other"; no reading and absent cells → "missing_cell"; otherwise the pooled
reason ("identity" — the chain composes to the identity only — or "underivable"). The gaps are the absent cells that
block the paths deriving nothing (`Derivation.missing` of each chain: on the shortest span that composes to nothing,
the pairs of its two composed parts with no cell), ranked by how many paths one such cell would unblock.

As a learning target: `agenda_items()` gives plain records — a cell to learn, a fact to learn, a dispute to settle —
and `to_agenda(agenda, table)` declares a goal «learn cell r1 ∘ r2» per gap in an `Agenda` (solvi.core.knowledge),
done when the table has that cell (its done check reads the table, so refitting the table on new observations closes
the goal). A missing fact and a dispute are records only: a dispute is already a person question in the knowledge
store (`forward_chain`), and whether a fact exists is not a check in code over the table.

`first_missing_step(steps, derived)`: with a reference derivation (steps «head ⟸ body» in order), the first step whose
head was not derived, with its rule schema (entities abstracted: r(X,Y) ⟸ r1(X,Z), r2(Z,Y)) — counted over many
refusals, the schemas name the missing rules. Ported from the author's earlier research line (a stepwise diagnosis of a
rule learner that named its missing rules within minutes)."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, NamedTuple, Sequence

from ._table import CompositionTable, Derivation

STATUSES = ("derivable", "disputed", "other", "missing_cell", "identity", "underivable", "no_path", "hypothesis",
            "no_known_positive", "functional_conflict")


class Gap(NamedTuple):
    """An absent cell left ∘ right → ?, the number of paths x → y it blocks (one such cell would unblock them), and
    the first such path's chain."""
    left: str
    right: str
    unblocks: int
    example: tuple


@dataclass
class WhyNot:
    """See the module docstring. readings: relation → the number of paths that give it under some bracketing;
    derivation: the pooled Derivation over every path (None without a path); paths: how many were read; truncated:
    max_paths cut the list."""
    x: object
    y: object
    relation: str | None
    status: str
    derivation: Derivation | None = None
    paths: int = 0
    truncated: bool = False
    readings: dict = field(default_factory=dict)
    gaps: list = field(default_factory=list)
    positive: str | None = None

    def summary(self) -> str:
        head = f"{self.x} → {self.y}"
        if self.status == "no_path":
            return f"{head}: no path through the facts — a missing fact, not a missing rule"
        if self.status == "no_known_positive":
            return (f"{head}: no known {self.positive} for {self.x} — {self.relation} comes only from the closed world "
                    f"of the functional {self.positive}; a missing fact, not a missing rule")
        if self.status == "functional_conflict":
            return (f"{head}: {self.x} has several known {self.positive} ({self.readings.get(self.positive)}) — the "
                    f"functional declaration is contradicted here, so the closed world is not applied")
        if self.status == "hypothesis":
            return f"{head}: {self.relation} rests on a disputed fact {self.readings} — a hypothesis until it is settled"
        if self.derivation is None and self.status == "derivable":
            return f"{head}: {self.relation or self.readings} is derived — the rules do not refuse it"
        if self.status == "derivable":
            return f"{head}: {self.derivation.relation} is derivable ({self.paths} paths) — the table does not refuse it"
        if self.status == "disputed":
            return f"{head}: the paths disagree {self.readings} — a dispute, not a hole"
        if self.status == "other":
            return f"{head}: the table derives {next(iter(self.readings))}, not {self.relation}"
        if self.gaps:
            g = self.gaps[0]
            more = f"; {len(self.gaps) - 1} more absent cells" if len(self.gaps) > 1 else ""
            return (f"{head}: no cell {g.left} ∘ {g.right} → ? (would unblock {g.unblocks} of {self.paths} paths; "
                    f"e.g. {' ∘ '.join(g.example)}){more}")
        return f"{head}: {self.status} over {self.paths} paths"

    def agenda_items(self) -> list[dict]:
        """The refusal as learning targets (plain records): one per gap {"kind": "cell", "goal": "learn cell l ∘ r",
        "left", "right", "unblocks", "example", "between"} for "missing_cell"; {"kind": "fact", ...} for "no_path";
        {"kind": "dispute", "readings", ...} for "disputed"; {"kind": "other", "derived", "cells"} for "other"; []
        when derivable."""
        between = [self.x, self.y]
        if self.status == "no_known_positive":
            return [{"kind": "positive", "goal": f"learn a {self.positive} fact for {self.x}",
                     "relation": self.positive, "subject": self.x, "between": between}]
        if self.status == "functional_conflict":
            return [{"kind": "conflict", "goal": f"settle {self.positive} of {self.x}: one of "
                     f"{self.readings.get(self.positive)}", "relation": self.positive, "subject": self.x,
                     "objects": list(self.readings.get(self.positive) or ()), "between": between}]
        if self.status == "no_path":
            return [{"kind": "fact", "goal": f"learn a fact linking {self.x} and {self.y}", "between": between}]
        if self.status == "disputed":
            return [{"kind": "dispute", "goal": f"settle {self.x} → {self.y}: " + " or ".join(self.readings),
                     "readings": dict(self.readings), "between": between}]
        if self.status == "other":
            r = next(iter(self.readings))
            cells = [[c.left, c.right, c.relation] for c in (self.derivation.cells if self.derivation else ())]
            return [{"kind": "other", "goal": f"check the cells deriving {r} for {self.x} → {self.y}",
                     "derived": r, "asked": self.relation, "cells": cells, "between": between}]
        if self.status == "missing_cell":
            return [{"kind": "cell", "goal": _goal(g.left, g.right), "left": g.left, "right": g.right,
                     "unblocks": g.unblocks, "example": list(g.example), "between": between} for g in self.gaps]
        return []

    def to_agenda(self, agenda, table: CompositionTable) -> list[str]:
        """Declare a goal «learn cell l ∘ r» per gap in `agenda` (an Agenda), done when `table` has that cell; a goal
        already declared is kept. → the goal names."""
        names = []
        for g in (self.gaps if self.status == "missing_cell" else ()):
            name = _goal(g.left, g.right)
            if name not in agenda.goals:
                agenda.goal(name, done=_cell_learned(table, g.left, g.right))
            names.append(name)
        return names


def _goal(left, right) -> str:
    return f"learn cell {left} ∘ {right}"


def _cell_learned(table, left, right):
    def done(state, table=table, left=left, right=right):
        return table.cell(left, right) is not None
    return done


def _paths(facts, x, y, max_len, max_paths, inverse):
    adj = defaultdict(list)
    for a, r, b in facts:
        adj[a].append((b, r))
        if inverse and r in inverse:
            adj[b].append((a, inverse[r]))
    for k in adj:
        adj[k] = sorted(set(adj[k]), key=lambda e: (str(e[0]), str(e[1])))
    chains, truncated = [], False
    stack = [(x, (), (x,))]
    while stack:
        node, rels, seen = stack.pop()
        if node == y and rels:
            if len(chains) >= max_paths:
                truncated = True
                break
            chains.append(rels)
            continue
        if len(rels) >= max_len:
            continue
        for nxt, r in reversed(adj.get(node, ())):
            if nxt not in seen:
                stack.append((nxt, rels + (r,), seen + (nxt,)))
    return chains, truncated


def why_not(table: CompositionTable, facts: Iterable[Sequence], x, y, relation: str | None = None, *,
            max_len: int = 6, max_paths: int = 2000, inverse: Mapping[str, str] | None = None) -> WhyNot:
    """Why `relation` (or any relation) between x and y is not derived from the pair facts by the table — see the
    module docstring. facts: (a, relation, b) triples, read from a to b, as the chains the table was fitted on.

    With guarded rules (`why_not(rules, story, x, y, relation)`; a GuardedRules and a Story): "derivable" (active on
    the pair), "disputed" (its key holds two values), "no_known_positive" (the complement of a declared functional
    relation, for a subject of the type with no active positive of it — `positive` names the relation, the agenda item
    is the fact to learn), "functional_conflict" (the subject has several known positives: the declaration is
    contradicted and the closed world left open there), "hypothesis" (present, resting on a disputed fact) or
    "underivable"."""
    from ._rules import GuardedRules
    if isinstance(table, GuardedRules):
        return _why_not_rules(table, facts, x, y, relation)
    if not isinstance(table, CompositionTable):
        raise TypeError("why_not reads a CompositionTable (fit one on the labelled chains) or GuardedRules")
    if int(max_len) < 1 or int(max_paths) < 1:
        raise ValueError("max_len and max_paths are at least 1")
    chains, truncated = _paths(list(facts), x, y, int(max_len), int(max_paths), inverse)
    if not chains:
        return WhyNot(x, y, relation, "no_path")
    pooled = table.derive_paths(chains)
    readings, blocked, example = Counter(), Counter(), {}
    for c in chains:
        d = table.derive(c)
        for r, k in d.candidates.items():
            if k and r != table.identity:
                readings[r] += 1
        for cell in d.missing:
            blocked[cell] += 1
            example.setdefault(cell, c)
    gaps = [Gap(a, b, n, example[(a, b)]) for (a, b), n in sorted(blocked.items(), key=lambda kv: (-kv[1], kv[0]))]
    readings = dict(sorted(readings.items()))
    if pooled.answered and (relation is None or pooled.relation == relation):
        status = "derivable"
    elif len(readings) > 1:
        status = "disputed"
    elif readings:
        status = "other" if relation is not None and relation not in readings else "derivable"
    elif gaps:
        status = "missing_cell"
    else:
        status = pooled.reason if pooled.reason in STATUSES else "underivable"
    return WhyNot(x, y, relation, status, pooled, len(chains), truncated, readings, gaps)


def _why_not_rules(rules, story, x, y, relation) -> WhyNot:
    c = rules.closure(story)
    ans = c.answer(x, y)
    readings = {r: c.status(x, y, r) for r in sorted(c.relations(x, y))}
    if (relation is None and ans) or relation in ans:
        return WhyNot(x, y, relation, "derivable", readings=readings)
    if relation is not None and c.status(x, y, relation) == "disputed":
        return WhyNot(x, y, relation, "disputed", readings=readings)
    for r, (su, _, rbar) in sorted(rules._closed_spec.items()):
        if relation == rbar and su in c.unary.get(x, ()):
            conflict = [zs for x2, r2, zs in c.closed_conflicts if x2 == x and r2 == r]
            if conflict:
                return WhyNot(x, y, relation, "functional_conflict", readings={**readings, r: list(conflict[0])},
                              positive=r)
            known = any(r in v and v[r][0] in (0, 1) and c.statuses.get((x, z, r)) == "active"
                        for (x2, z), v in c.pairs.items() if x2 == x)
            if not known:
                return WhyNot(x, y, relation, "no_known_positive", readings=readings, positive=r)
    if relation is not None and c.status(x, y, relation) == "hypothesis":
        return WhyNot(x, y, relation, "hypothesis", readings=readings)
    return WhyNot(x, y, relation, "underivable", readings=readings)


# ───────────────────────────────────────────── with a reference derivation
def rule_schema(head: Sequence, body: Sequence[Sequence]) -> str:
    """('r', a, b) ⟸ [('r1', a, c), ('r2', c, b)] → 'r(X,Y) ⟸ r1(X,Z), r2(Z,Y)': entities abstracted in order."""
    names: dict = {}

    def v(e):
        return names.setdefault(e, "XYZWUVST"[len(names)] if len(names) < 8 else f"V{len(names)}")
    h = f"{head[0]}({v(head[1])},{v(head[2])})"
    return h + " ⟸ " + ", ".join(f"{r}({v(a)},{v(b)})" for r, a, b in body)


def first_missing_step(steps: Sequence, derived) -> tuple | None:
    """steps: [(head, [body facts]), ...] of a reference derivation in order, facts as (relation, a, b); derived: the
    facts the system derived. → (head, body, schema) of the first step whose head was not derived, or None when every
    head was (the refusal is not the rules')."""
    for head, body in steps:
        if tuple(head) not in derived:
            b = [tuple(x) for x in body]
            return tuple(head), b, rule_schema(tuple(head), b)
    return None


__all__ = ["Gap", "STATUSES", "WhyNot", "first_missing_step", "rule_schema", "why_not"]
