"""Bounded, verified dependency computation on a declared integer work ledger.

Callbacks and their revision labels are host trusted, not authenticated. A
separate verifier is a protocol obligation, not proof of arbitrary Python code.
The ledger is a declared work bound, not a runtime quota. Journal writers must
be serialized on one host; recovery never repeats an unknown pending callback.
One operator revision binds both producer and verifier implementations.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from functools import wraps
from types import MappingProxyType

from ._common import BudgetStop, WorkBudget, canonical, digest, identifier, natural, portable
from .journal import ExperienceJournal


def _scalar(value):
    if type(value) not in (int, bool):
        raise ValueError("computation values must be strict int or bool")
    return value


def _transaction(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if self._busy:
            raise RuntimeError("computation transaction is already active")
        self._busy = True
        try:
            return method(self, *args, **kwargs)
        finally:
            self._busy = False
    return guarded


@dataclass(frozen=True)
class ComputeOperator:
    name: str
    revision: str
    execute: Callable
    verify: Callable
    cost: int = 1
    verify_cost: int = 1
    is_valid: Callable | None = None

    def __post_init__(self):
        identifier(self.name)
        identifier(self.revision, "operator/verifier revision")
        natural(self.cost, "cost", positive=True)
        natural(self.verify_cost, "verify_cost", positive=True)
        if not callable(self.execute) or not callable(self.verify):
            raise TypeError("execute and verify must be callable")
        if self.execute is self.verify:
            raise ValueError("verifier must be a separate callback")
        if self.is_valid is not None and not callable(self.is_valid):
            raise TypeError("is_valid must be callable")


@dataclass(frozen=True)
class ComputeNode:
    name: str
    dependencies: tuple[str, ...]
    operators: tuple[str, ...]

    def __post_init__(self):
        identifier(self.name)
        object.__setattr__(self, "dependencies", tuple(self.dependencies))
        object.__setattr__(self, "operators", tuple(self.operators))
        for value in self.dependencies + self.operators:
            identifier(value)
        if len(self.dependencies) > 160:
            raise ValueError("node dependency port bound exceeded")
        if not 1 <= len(self.operators) <= 8 or len(set(self.operators)) != len(self.operators):
            raise ValueError("nodes require 1..8 distinct operator alternatives")


@dataclass(frozen=True)
class ComputationPlan:
    input_names: tuple[str, ...]
    nodes: tuple[ComputeNode, ...]
    outputs: tuple[str, ...]

    def __post_init__(self):
        for name in ("input_names", "nodes", "outputs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if len(self.input_names) > 32 or not 1 <= len(self.nodes) <= 128:
            raise ValueError("plan bound exceeded")
        seen = set()
        for name in self.input_names:
            identifier(name)
            if name in seen:
                raise ValueError("duplicate input")
            seen.add(name)
        for node in self.nodes:
            if not isinstance(node, ComputeNode):
                raise TypeError("nodes must be ComputeNode")
            if node.name in seen or not set(node.dependencies) <= seen:
                raise ValueError("duplicate node or dependencies not in topological order")
            seen.add(node.name)
        node_names = {n.name for n in self.nodes}
        if not self.outputs or len(set(self.outputs)) != len(self.outputs) or not set(self.outputs) <= node_names:
            raise ValueError("outputs must name distinct computed nodes")


@dataclass(frozen=True)
class ComputationReport:
    status: str
    node: str | None = None
    operator: str | None = None
    value: int | bool | None = None
    missing: tuple[str, ...] = ()

    def to_dict(self):
        return {"status": self.status, "node": self.node, "operator": self.operator,
                "value": self.value, "missing": list(self.missing)}


class VerifiedComputation:
    """Finite DAG session; only independently accepted values enter ``values``.

    With N nodes, E dependency ports and G goals, setup costs 2+(N+E)*(1+G).
    Each unfinished step pays 1+G+sum(2+ports+alternatives) over relevant nodes.
    An attempt reserves producer + verifier + two record units atomically.
    Updates pay N+E+invalidated entries+2; observations add their reader price.
    Live guards pay guarded entries+N+E+2 on every value access or step.
    Recovery pays sum(cached verifier prices)+2 before exposing cached values.
    These declared work prices apply equally without a durable journal.
    Exhausted alternatives terminate as rejected; ordinary callback exceptions
    are recorded rejections, whereas BaseException leaves a stopped session.
    """

    def __init__(self, plan, operators, budget, *, journal=None, inputs=None):
        self._busy = True
        if not isinstance(plan, ComputationPlan) or not isinstance(budget, WorkBudget):
            raise TypeError("expected ComputationPlan and WorkBudget")
        if journal is not None and not isinstance(journal, ExperienceJournal):
            raise TypeError("journal must be ExperienceJournal")
        self._plan = plan
        self._operators = self._operator_map(operators)
        self._budget = budget
        self._journal = journal
        self._inputs = {}
        self._cache = {}
        self._failed = {}
        self._last_error = None
        self._pending = None
        self._stopped = False
        self._scan_cost = len(plan.nodes) + sum(len(n.dependencies) for n in plan.nodes)
        records = journal.records() if journal is not None else []
        previous = None
        for event in records:
            ledger = (event["payload"].get("ledger") if event["kind"] == "debit" else
                      event["payload"]["state"].get("ledger") if event["kind"] == "checkpoint" else None)
            if ledger is not None:
                checked = WorkBudget.restore(ledger)
                if previous is not None and (checked.limit != previous.limit or checked.spent < previous.spent
                        or any(checked.counts.get(k, 0) < v for k, v in previous.counts.items())):
                    raise ValueError("nonmonotonic computation ledger history")
                previous = checked
        checkpoints = [e for e in records if e["kind"] == "checkpoint"]
        if records and not checkpoints:
            raise ValueError("journal has no computation checkpoint")
        if checkpoints:
            state = checkpoints[-1]["payload"]["state"]
            if state.get("config") != self._config():
                raise ValueError("incompatible computation configuration")
            if state.get("pending") is not None or state.get("stopped"):
                raise RuntimeError("session stopped or has unknown pending callback; reconcile explicitly")
            self._restore_ledger(previous.snapshot())
        # Bound and reserve the graph preprocessing before traversing it, also
        # on cold reopen. Journal validation itself is a trusted IO boundary.
        self._charge("compute_setup", 2 + self._scan_cost * (1 + len(plan.outputs)))
        self._active = set(plan.outputs)
        for node in reversed(plan.nodes):
            if node.name in self._active:
                self._active.update(node.dependencies)
        self._nodes = tuple(n for n in plan.nodes if n.name in self._active)
        self._goals = {n.name: 0 for n in self._nodes}
        for output in plan.outputs:
            ancestors = {output}
            for node in reversed(plan.nodes):
                if node.name in ancestors:
                    ancestors.update(node.dependencies)
            for name in self._goals:
                self._goals[name] += name in ancestors
        if checkpoints:
            state = checkpoints[-1]["payload"]["state"]
            self._restore(state)
            if self._pending is not None or self._stopped:
                raise RuntimeError("session stopped or has unknown pending callback; reconcile explicitly")
            if inputs is not None and canonical(dict(inputs)) != canonical({k: v["value"] for k, v in self._inputs.items()}):
                raise ValueError("restore inputs differ; use update_inputs")
            self._reverify_restored()
        else:
            if records:
                raise ValueError("journal has no computation checkpoint")
            if inputs is not None:
                self._inputs = self._checked_inputs(inputs, "initial")
            self._save()
        self._busy = False

    @property
    def plan(self):
        return self._plan

    @property
    def operators(self):
        return MappingProxyType(dict(self._operators))

    @property
    def budget(self):
        return self._budget

    @property
    @_transaction
    def values(self):
        self._refresh_validity()
        return MappingProxyType({k: v["value"] for k, v in self._cache.items()})

    @property
    @_transaction
    def outputs(self):
        self._refresh_validity()
        return MappingProxyType({k: self._cache[k]["value"] for k in self._plan.outputs if k in self._cache})

    def _operator_map(self, operators):
        values = tuple(operators.values()) if isinstance(operators, Mapping) else tuple(operators)
        if any(not isinstance(op, ComputeOperator) for op in values):
            raise TypeError("operators must be ComputeOperator")
        result = {op.name: op for op in values}
        if len(result) != len(values) or not {op for n in self._plan.nodes for op in n.operators} <= result.keys():
            raise ValueError("duplicate or missing operators")
        return result

    def _config(self):
        return {"v": 1, "inputs": list(self._plan.input_names),
                "nodes": [{"name": n.name, "dependencies": list(n.dependencies), "operators": list(n.operators)} for n in self._plan.nodes],
                "outputs": list(self._plan.outputs),
                "operators": {k: {"revision": o.revision, "cost": o.cost, "verify_cost": o.verify_cost, "liveness": o.is_valid is not None} for k, o in self._operators.items()}}

    def snapshot(self):
        return portable({"computation": 1, "config": self._config(), "ledger": self._budget.snapshot(),
                         "inputs": self._inputs, "cache": self._cache, "failed": self._failed,
                         "pending": self._pending, "stopped": self._stopped, "last_error": self._last_error})

    def _restore_ledger(self, snapshot):
        restored = WorkBudget.restore(snapshot)
        current = WorkBudget.restore(self._budget.snapshot())
        if restored.limit != current.limit:
            raise ValueError("incompatible work ledger")
        # Catalog reconstruction may already have extended this same ledger.
        # Preserve that paid work; merge only comparable category vectors.
        if all(current.counts.get(k, 0) >= v for k, v in restored.counts.items()):
            return
        if any(restored.counts.get(k, 0) < v for k, v in current.counts.items()):
            raise ValueError("incompatible work ledger category vectors")
        self._budget._spent = restored.spent
        self._budget._counts = restored.counts

    def _restore(self, state):
        required = {"computation", "config", "ledger", "inputs", "cache", "failed", "pending", "stopped", "last_error"}
        if set(state) != required or state["computation"] != 1 or state["config"] != self._config():
            raise ValueError("incompatible computation configuration")
        if type(state["stopped"]) is not bool:
            raise ValueError("invalid stopped state")
        self._inputs = self._checked_inputs({k: v["value"] for k, v in state["inputs"].items()}, "restore")
        for k, item in state["inputs"].items():
            identifier(item["revision"], "source revision")
            self._inputs[k] = dict(item)
        self._cache = portable(state["cache"])
        self._failed = portable(state["failed"])
        self._pending = portable(state["pending"])
        self._stopped = state["stopped"]
        self._last_error = portable(state["last_error"])
        names = {n.name for n in self._nodes}
        if not self._cache.keys() <= names or not self._failed.keys() <= names:
            raise ValueError("invalid cached node")
        for node in self._nodes:
            if node.name in self._cache:
                entry = self._cache[node.name]
                _scalar(entry["value"])
                if entry["operator"] not in node.operators or entry["key"] != self._key(node, self._operators[entry["operator"]]):
                    raise ValueError("invalid derived version")
            if node.name in self._failed:
                failed = self._failed[node.name]
                if type(failed) is not list or len(set(failed)) != len(failed) or not set(failed) <= set(node.operators):
                    raise ValueError("invalid failed alternatives")

    def _reverify_restored(self):
        if not self._cache:
            return
        self._charge("compute_restore_verification", 2 + sum(self._operators[e["operator"]].verify_cost for e in self._cache.values()))
        self._pending = {"kind": "restore_verification"}
        self._save()
        try:
            for node in self._nodes:
                if node.name not in self._cache:
                    continue
                entry = self._cache[node.name]
                op = self._operators[entry["operator"]]
                accepted = op.verify(tuple(self._entry(k)["value"] for k in node.dependencies), entry["value"])
                if type(accepted) is not bool or not accepted:
                    raise ValueError("restored value failed independent verification")
        except BaseException:
            self._callback_failed()
            raise
        self._pending = None
        self._save()

    def _check_running(self):
        if self._stopped or self._pending is not None:
            raise RuntimeError("session stopped or pending")

    def _charge(self, kind, units):
        if units > self._budget.remaining:
            raise BudgetStop("computation budget exhausted before operation")
        try:
            self._budget.charge(kind, units)
            if self._journal is not None:
                self._journal.debit(self._budget.snapshot())
        except BudgetStop as error:
            self._stopped = True
            raise RuntimeError("budget sink failed after reservation") from error
        except BaseException:
            self._stopped = True
            raise

    def _save(self):
        if self._journal is not None:
            try:
                self._journal.checkpoint(self.snapshot())
            except BaseException:
                self._stopped = True
                raise

    def _checked_inputs(self, values, revision):
        identifier(revision, "source revision")
        if not isinstance(values, Mapping) or not set(values) <= set(self._plan.input_names):
            raise ValueError("unknown input port")
        return {k: {"value": _scalar(v), "revision": revision} for k, v in values.items()}

    def _entry(self, name):
        return self._inputs[name] if name in self._inputs else self._cache[name]

    def _key(self, node, op):
        return digest({"dependencies": [self._entry(k) for k in node.dependencies],
                       "operator": op.name, "revision": op.revision})

    def _invalidate(self, roots):
        dirty = set(roots)
        for node in self._plan.nodes:
            if node.name in dirty or dirty.intersection(node.dependencies):
                dirty.add(node.name)
        return dirty

    def _refresh_validity(self):
        self._check_running()
        guarded = [(name, self._operators[entry["operator"]]) for name, entry in self._cache.items()
                   if self._operators[entry["operator"]].is_valid is not None]
        if not guarded:
            return
        self._charge("compute_liveness", len(guarded) + self._scan_cost + 2)
        self._pending = {"kind": "liveness"}
        self._save()
        try:
            invalid = []
            for name, op in guarded:
                valid = op.is_valid()
                if type(valid) is not bool:
                    raise ValueError("is_valid must return strict bool")
                if not valid:
                    invalid.append(name)
        except BaseException:
            self._callback_failed()
            raise
        self._drop(self._invalidate(invalid))
        self._pending = None
        self._save()

    @_transaction
    def update_inputs(self, values, revision):
        self._check_running()
        updates = self._checked_inputs(values, revision)
        self._charge("compute_invalidation_scan", self._scan_cost)
        dirty = self._invalidate(k for k, v in updates.items() if canonical(self._inputs.get(k)) != canonical(v))
        self._charge("compute_invalidation", len(dirty) + 2)
        self._inputs.update(updates)
        self._drop(dirty)
        self._save()

    def _drop(self, dirty):
        for name in dirty:
            self._cache.pop(name, None)
            self._failed.pop(name, None)

    @_transaction
    def update_operators(self, operators):
        self._check_running()
        updated = self._operator_map(operators)
        changed = {k for k in set(updated) | set(self._operators) if updated.get(k) != self._operators.get(k)}
        for name in changed & self._operators.keys() & updated.keys():
            if updated[name].revision == self._operators[name].revision:
                raise ValueError("changed callbacks or prices require a new revision")
        self._charge("compute_invalidation_scan", self._scan_cost)
        dirty = self._invalidate(n.name for n in self._plan.nodes if changed.intersection(n.operators))
        self._charge("compute_invalidation", len(dirty) + 2)
        self._operators = updated
        self._drop(dirty)
        self._save()

    @_transaction
    def observe(self, name, reader, cost, revision):
        self._check_running()
        if name not in self._plan.input_names or not callable(reader):
            raise ValueError("unknown input or noncallable reader")
        natural(cost, "observation cost", positive=True)
        identifier(revision, "source revision")
        self._charge("compute_invalidation_scan", self._scan_cost)
        dirty = self._invalidate((name,))
        self._charge("compute_observation", cost + len(dirty) + 2)
        self._pending = {"kind": "observation", "name": name, "revision": revision}
        self._save()
        try:
            value = _scalar(reader())
        except BaseException:
            self._callback_failed()
            raise
        self._inputs[name] = {"value": value, "revision": revision}
        self._drop(dirty)
        self._pending = None
        self._save()
        return value

    def _callback_failed(self):
        self._stopped = True
        pending = self._pending if self._pending is not None else {}
        self._pending = {**pending, "exception": True}
        self._save()

    @_transaction
    def step(self, policy="fifo"):
        self._check_running()
        if policy not in ("fifo", "cheap", "critical"):
            raise ValueError("unknown computation policy")
        try:
            self._refresh_validity()
            if all(k in self._cache for k in self._plan.outputs):
                return ComputationReport("complete")
            self._charge("compute_controller_lookup", 1 + len(self._plan.outputs) + sum(2 + len(n.dependencies) + len(n.operators) for n in self._nodes))
        except BudgetStop:
            return ComputationReport("budget")
        if all(k in self._cache for k in self._plan.outputs):
            return ComputationReport("complete")
        ready = []
        for node in self._nodes:
            if node.name in self._cache or not all(k in self._inputs or k in self._cache for k in node.dependencies):
                continue
            candidates = [self._operators[k] for k in node.operators if k not in self._failed.get(node.name, [])]
            if candidates:
                candidates = [op for op in candidates if op.cost + op.verify_cost + 2 <= self._budget.remaining]
                if candidates:
                    op = candidates[0] if policy == "fifo" else min(candidates, key=lambda o: o.cost + o.verify_cost)
                    ready.append((node, op))
        if not ready:
            missing = tuple(k for k in self._plan.input_names if k in self._active and k not in self._inputs)
            self._save()
            feasible_exists = any(n.name not in self._cache and all(k in self._inputs or k in self._cache for k in n.dependencies)
                                  and any(k not in self._failed.get(n.name, []) for k in n.operators) for n in self._nodes)
            return ComputationReport("budget" if feasible_exists else "needs_input" if missing else "rejected", missing=missing)
        if policy == "cheap":
            ready = [min(ready, key=lambda pair: pair[1].cost + pair[1].verify_cost + 2)]
        elif policy == "critical":
            # Bounded goal contribution per complete declared attempt cost.
            def priority(pair):
                return -Fraction(self._goals[pair[0].name], pair[1].cost + pair[1].verify_cost + 2)
            ready = [min(ready, key=priority)]
        node, op = ready[0]
        try:
            self._charge("compute_attempt", op.cost + op.verify_cost + 2)
        except BudgetStop:
            self._save()
            return ComputationReport("budget", node.name, op.name)
        self._pending = {"kind": "compute", "node": node.name, "operator": op.name}
        self._save()
        args = tuple(self._entry(k)["value"] for k in node.dependencies)
        try:
            proposal = _scalar(op.execute(args))
            accepted = op.verify(args, proposal)
            if type(accepted) is not bool:
                raise ValueError("verifier must return strict bool")
        except Exception as error:  # noqa: BLE001 - reject a failed host callback, retain paid work
            self._failed.setdefault(node.name, []).append(op.name)
            self._last_error = {"node": node.name, "operator": op.name, "exception": type(error).__name__}
            self._pending = None
            self._save()
            return ComputationReport("rejected", node.name, op.name)
        except BaseException:
            self._callback_failed()
            raise
        if accepted:
            self._cache[node.name] = {"value": proposal, "operator": op.name, "key": self._key(node, op)}
        else:
            self._failed.setdefault(node.name, []).append(op.name)
        self._pending = None
        self._save()
        return ComputationReport("accepted" if accepted else "rejected", node.name, op.name, proposal if accepted else None)

    def run(self, policy="fifo"):
        while True:
            report = self.step(policy)
            if report.status in ("accepted", "rejected") and report.node is not None:
                continue
            return report


__all__ = ["ComputeOperator", "ComputeNode", "ComputationPlan", "ComputationReport", "VerifiedComputation"]
