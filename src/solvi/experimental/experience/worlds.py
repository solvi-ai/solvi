"""Finite joint alternatives and retractable observations, without invented probabilities."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from ...core.costs import BudgetStop
from ._common import WorkBudget, canonical, digest, identifier, natural, portable


@dataclass(frozen=True)
class Decision:
    status: str
    value: object = None
    alternatives: tuple = ()
    worlds: int = 0
    reason: str = ""

    @property
    def resolved(self) -> bool:
        return self.status == "resolved"


class WorldSet:
    """Explicit joint states, not independent marginal distributions.

    The host supplies the alternatives and pure callbacks. An invariant answer
    is conditional on this model containing the real state. Empty alternatives
    mean contradiction. Constraints are materialized over the full finite base,
    so withdrawing one can restore states without rerunning arbitrary callbacks.
    Snapshots retain the base, evidence and dependencies, not Python callables.
    """

    def __init__(self, worlds: Iterable[Mapping], *, version: str = "1", max_worlds: int = 4096):
        self.version = identifier(version, "model version")
        self.max_worlds = natural(max_worlds, "max_worlds", positive=True)
        base = set()
        schema = None
        for n, world in enumerate(worlds):
            if n >= self.max_worlds:
                raise ValueError("joint-world input exceeds max_worlds; no silent truncation")
            if not isinstance(world, Mapping):
                raise ValueError("a joint world must be a mapping")
            item = dict(world)
            canonical(item)
            if schema is None:
                schema = set(item)
            elif set(item) != schema:
                raise ValueError("joint alternatives must have the same field schema")
            base.add(canonical(item))
        self._base = tuple(sorted(base))
        self._constraints: tuple[dict, ...] = ()
        self._active = self._base

    def __len__(self):
        return len(self._active)

    @property
    def worlds(self) -> tuple[dict, ...]:
        return tuple(json.loads(k) for k in self._active)

    @property
    def evidence(self) -> tuple[str, ...]:
        return tuple(c["id"] for c in self._constraints)

    @property
    def fingerprint(self) -> str:
        return digest(self._body())

    def _body(self):
        return {
            "v": 1,
            "version": self.version,
            "max_worlds": self.max_worlds,
            "base": [json.loads(k) for k in self._base],
            "constraints": portable(list(self._constraints)),
        }

    def snapshot(self) -> dict:
        body = self._body()
        return {**body, "digest": digest(body)}

    def _derive(self, constraints, budget: WorkBudget):
        active = []
        allowed = [set(c["allowed"]) for c in constraints]
        for key in self._base:
            budget.charge("world_filter", 1 + len(allowed))
            if all(key in keys for keys in allowed):
                active.append(key)
        result = object.__new__(type(self))
        result.version, result.max_worlds = self.version, self.max_worlds
        result._base = self._base
        result._constraints = tuple(portable(list(constraints)))
        result._active = tuple(active)
        return result

    def constrain(self, evidence_id: str, predicate: Callable, budget: WorkBudget, *, depends_on=()):
        """A pure finite constraint; repeated identical evidence is idempotent.

        Predicate evaluation is paid over the full base, preserving its meaning
        if another restriction is subsequently withdrawn. The evidence must be
        supplied by the trusted host, not an answer proposed by this module.
        """
        identifier(evidence_id, "evidence id")
        deps = tuple(depends_on)
        if len(set(deps)) != len(deps) or any(type(d) is not str or d not in self.evidence for d in deps):
            raise ValueError("constraint dependencies must be distinct existing evidence IDs")
        if evidence_id in deps:
            raise ValueError("evidence cannot depend on itself")
        allowed = []
        for key in self._base:
            budget.charge("constraint")
            value = predicate(json.loads(key))
            if type(value) is not bool:
                raise ValueError("constraint must return bool")
            if value:
                allowed.append(key)
        item = {"id": evidence_id, "depends_on": list(deps), "allowed": allowed}
        previous = next((c for c in self._constraints if c["id"] == evidence_id), None)
        if previous is not None:
            if canonical(previous) != canonical(item):
                raise ValueError("an evidence ID cannot be reused for another constraint")
            return self
        return self._derive((*self._constraints, item), budget)

    def observe(self, field: str, value, evidence_id: str, budget: WorkBudget, *, depends_on=()):
        identifier(field, "observed field")
        target = canonical(value)
        if self._base and field not in json.loads(self._base[0]):
            raise ValueError("observed field is outside the model schema")
        return self.constrain(
            evidence_id, lambda world: canonical(world[field]) == target, budget, depends_on=depends_on
        )

    def retract(self, evidence_id: str, budget: WorkBudget):
        identifier(evidence_id, "evidence id")
        removed = {evidence_id}
        for c in self._constraints:
            budget.charge("dependency")
            if any(d in removed for d in c["depends_on"]):
                removed.add(c["id"])
        return self._derive(tuple(c for c in self._constraints if c["id"] not in removed), budget)

    def decide(self, question: Callable, budget: WorkBudget) -> Decision:
        if not self._active:
            return Decision("contradiction", reason="no joint state satisfies the evidence")
        values = {}
        try:
            for key in self._active:
                budget.charge("decision")
                answer = portable(question(json.loads(key)))
                values[canonical(answer)] = answer
        except BudgetStop:
            return Decision("budget", worlds=len(self), reason="work budget exhausted before complete reasoning")
        ordered = tuple(values[k] for k in sorted(values))
        if len(ordered) == 1:
            return Decision("resolved", ordered[0], ordered, len(self), "same result in every modeled joint state")
        return Decision("unknown", alternatives=ordered, worlds=len(self), reason="modeled states disagree")

    @classmethod
    def restore(cls, snapshot: Mapping):
        s = portable(dict(snapshot))
        required = {"v", "version", "max_worlds", "base", "constraints", "digest"}
        if set(s) != required or type(s["v"]) is not int or s["v"] != 1:
            raise ValueError("unsupported joint-world snapshot")
        body = {k: v for k, v in s.items() if k != "digest"}
        if s["digest"] != digest(body) or type(s["constraints"]) is not list or type(s["base"]) is not list:
            raise ValueError("corrupt joint-world snapshot")
        model = cls(s["base"], version=s["version"], max_worlds=s["max_worlds"])
        if [json.loads(k) for k in model._base] != s["base"]:
            raise ValueError("noncanonical base alternatives")
        known = set()
        for c in s["constraints"]:
            if type(c) is not dict or set(c) != {"id", "depends_on", "allowed"}:
                raise ValueError("invalid finite constraint")
            identifier(c["id"], "evidence id")
            deps, allowed = c["depends_on"], c["allowed"]
            if (
                c["id"] in known
                or type(deps) is not list
                or type(allowed) is not list
                or any(type(d) is not str or d not in known for d in deps)
                or len(deps) != len(set(deps))
                or any(type(k) is not str or k not in model._base for k in allowed)
                or sorted(set(allowed)) != allowed
            ):
                raise ValueError("invalid or cyclic evidence dependencies/alternatives")
            known.add(c["id"])
        # Restore is a pure integrity operation; no external effects or paid
        # inference are executed. The caller restores the original ledger too.
        model._constraints = tuple(portable(s["constraints"]))
        sets = [set(c["allowed"]) for c in model._constraints]
        model._active = tuple(k for k in model._base if all(k in a for a in sets))
        return model
