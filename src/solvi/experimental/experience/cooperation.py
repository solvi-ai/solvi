"""Checked cooperation on a finite, host-defined constraint optimization task.

Callbacks are trusted host capabilities. Evidence and snapshots are untrusted
portable data; neither can supply a checker or executable code.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping

from ._common import BudgetStop, WorkBudget, digest, identifier, natural, portable


class FiniteProblem:
    """Immutable finite assignments, named feasibility predicates, integer cost."""

    def __init__(self, assignments: Mapping, constraints: Mapping[str, Callable],
                 objective: Callable, *, revision: str, budget: WorkBudget):
        if type(assignments) is not dict or len(assignments) > 4096:
            raise ValueError("bounded assignment dictionary required")
        if type(constraints) is not dict or len(constraints) > 32:
            raise ValueError("bounded constraint dictionary required")
        budget.charge("cooperation.construct", 1 + len(assignments))
        self.revision = identifier(revision)
        self._assignments = portable(assignments)
        for key, attrs in self._assignments.items():
            identifier(key)
            if type(attrs) is not dict or len(attrs) > 32:
                raise ValueError("assignment attributes must be mappings")
        self._constraints = dict(constraints)
        for key, predicate in self._constraints.items():
            identifier(key)
            if not callable(predicate):
                raise TypeError("constraint must be host callable")
        if not callable(objective):
            raise TypeError("objective must be host callable")
        # Materialize host semantics once. Subsequent operations never execute
        # callback closures, so mutable callback state cannot invalidate evidence.
        self._ids = tuple(sorted(self._assignments))
        self._truth = {}
        self._costs = {}
        for key in self.ids:
            self._truth[key] = {}
            for name, predicate in self._constraints.items():
                budget.charge("cooperation.construct_constraint")
                truth = predicate(self.attributes(key))
                if type(truth) is not bool:
                    raise ValueError("constraint callback must return bool")
                self._truth[key][name] = truth
            budget.charge("cooperation.construct_objective")
            cost = objective(self.attributes(key))
            if type(cost) is not int:
                raise ValueError("objective must return integer")
            self._costs[key] = cost
        self._constraints = tuple(sorted(self._constraints))
        self.fingerprint = digest({"revision": revision, "assignments": self._assignments,
                                   "truth": self._truth, "costs": self._costs})

    @property
    def ids(self):
        return self._ids

    @property
    def constraint_names(self):
        return tuple(sorted(self._constraints))

    def attributes(self, key):
        return portable(self._assignments[key])

    def satisfies(self, key, name, budget):
        budget.charge("cooperation.constraint")
        return self._truth[key][name]

    def feasible(self, key, budget):
        return all(self.satisfies(key, name, budget) for name in self.constraint_names)

    def cost(self, key, budget):
        budget.charge("cooperation.objective")
        return self._costs[key]


@dataclass(frozen=True)
class Evidence:
    """A claim; dependencies name previously accepted active claims."""
    id: str
    source: str
    revision: str
    kind: str
    targets: tuple[str, ...] = ()
    value: object = None
    dependencies: tuple[str, ...] = ()

    def data(self):
        if type(self.targets) not in (tuple, list) or len(self.targets) > 4096:
            raise ValueError("bounded target sequence required")
        if type(self.dependencies) not in (tuple, list) or len(self.dependencies) > 4096:
            raise ValueError("bounded dependency sequence required")
        for value in (self.id, self.source, self.revision, self.kind, *self.targets, *self.dependencies):
            identifier(value)
            if len(value) > 256:
                raise ValueError("identifier exceeds limit")
        if self.value is not None and type(self.value) not in (str, int):
            raise ValueError("scalar evidence value required")
        if type(self.value) is str and len(self.value) > 256:
            raise ValueError("constraint identifier exceeds limit")
        return portable({"id": self.id, "source": self.source, "revision": self.revision,
                         "kind": self.kind, "targets": list(self.targets), "value": self.value,
                         "dependencies": list(self.dependencies)})


@dataclass(frozen=True)
class Residual:
    revision: str
    candidates: tuple[str, ...]
    incumbent: str | None
    upper_bound: int | None
    lower_bound: int | None


class ConstraintSolver:
    """Check one surviving candidate against the host constraints per call."""
    def __init__(self, problem, source="constraints"):
        self.problem, self.source = problem, identifier(source)
        self.seen = set()

    def __call__(self, residual, budget):
        for key in residual.candidates:
            if key in self.seen:
                continue
            self.seen.add(key)
            for name in self.problem.constraint_names:
                if not self.problem.satisfies(key, name, budget):
                    return Evidence(f"{self.source}:{len(self.seen)}", self.source,
                                    residual.revision, "exclusion", (key,), name)
            return None
        return None


class ObjectiveSolver:
    """Order residual candidates by cost and propose a checked feasible incumbent."""
    def __init__(self, problem, source="objective"):
        self.problem, self.source = problem, identifier(source)
        self.sequence = 0

    def __call__(self, residual, budget):
        self.sequence += 1
        ordered = sorted((self.problem.cost(key, budget), key) for key in residual.candidates)
        for cost, key in ordered:
            if residual.upper_bound is not None and cost >= residual.upper_bound:
                continue
            if self.problem.feasible(key, budget):
                return Evidence(f"{self.source}:{self.sequence}", self.source,
                                residual.revision, "incumbent", (key,), cost)
        return None


class CooperativeAuthority:
    """Host-owned monotonic checkpoint and live ledger; never trust snapshot data."""

    def __init__(self):
        self._epoch = 0
        self._checkpoint = None
        self._claimed = False
        self._budget: WorkBudget | None = None
        self._watermark: dict | None = None

    def export_trusted_state(self):
        """Persist via a trusted host channel, separately from the snapshot."""
        ledger = self._budget.snapshot() if self._budget is not None else self._watermark
        return portable({"v": 1, "epoch": self._epoch, "checkpoint": self._checkpoint,
                         "claimed": self._claimed, "ledger": ledger})

    @classmethod
    def from_trusted_state(cls, state):
        if type(state) is not dict or set(state) != {"v", "epoch", "checkpoint", "claimed", "ledger"}:
            raise ValueError("invalid cooperation authority anchor")
        if type(state["v"]) is not int or state["v"] != 1 or type(state["claimed"]) is not bool:
            raise ValueError("invalid cooperation authority version")
        natural(state["epoch"], "authority epoch")
        if state["checkpoint"] is not None and (type(state["checkpoint"]) is not str or len(state["checkpoint"]) != 64):
            raise ValueError("invalid checkpoint identity")
        if state["ledger"] is not None:
            WorkBudget.restore(state["ledger"])
        result = cls()
        result._epoch, result._checkpoint = state["epoch"], state["checkpoint"]
        result._claimed, result._watermark = state["claimed"], portable(state["ledger"])
        return result


class CooperativeSearch:
    """Independent finite checker and versioned evidence journal.

    No solver result is trusted. Each accepted exclusion is checked against the
    original host predicate; bounds are recomputed across the original domain.
    """
    def __init__(self, problem: FiniteProblem, budget: WorkBudget, *, authority=None):
        self.authority = CooperativeAuthority() if authority is None else authority
        if type(self.authority) is not CooperativeAuthority or self.authority._claimed:
            raise ValueError("fresh CooperativeAuthority required; restore for transfer")
        if type(budget) is not WorkBudget:
            raise ValueError("WorkBudget required")
        self.authority._claimed = True
        self.authority._budget = budget
        self._epoch = self.authority._epoch
        self._problem, self.budget = problem, budget
        self._active = {}
        self._used = set()
        self._events = []

    @property
    def problem(self):
        return self._problem

    def _guard(self):
        if self._epoch != self.authority._epoch or self.budget is not self.authority._budget:
            raise ValueError("stale cooperative search")

    def _advance(self):
        self.authority._epoch += 1
        self._epoch = self.authority._epoch
        self.authority._checkpoint = None

    def residual(self):
        self._guard()
        self.budget.charge("cooperation.residual", 1 + len(self.problem.ids) + sum(1 + len(e.targets) for e in self._active.values()))
        excluded = {key for e in self._active.values() if e.kind == "exclusion" for key in e.targets}
        incumbents = [(e.value, e.targets[0]) for e in self._active.values() if e.kind == "incumbent"]
        best = min(incumbents) if incumbents else None
        bounds = [e.value for e in self._active.values() if e.kind == "lower_bound"]
        return Residual(self.problem.revision, tuple(k for k in self.problem.ids if k not in excluded),
                        best[1] if best else None, best[0] if best else None,
                        max(bounds) if bounds else None)

    def submit(self, evidence: Evidence):
        self._guard()
        entry_epoch = self._epoch
        if type(evidence) is not Evidence:
            raise ValueError("exact Evidence required")
        if type(evidence.targets) not in (list, tuple) or len(evidence.targets) > 4096 or type(evidence.dependencies) not in (list, tuple) or len(evidence.dependencies) > 4096:
            raise ValueError("bounded evidence sequences required")
        self.budget.charge("cooperation.submit", 1 + len(evidence.targets) + len(evidence.dependencies))
        # Normalize once; validate and store this same immutable claim.
        data = evidence.data()
        evidence = Evidence(**{**data, "targets": tuple(data["targets"]),
                               "dependencies": tuple(data["dependencies"])})
        if len(self._events) >= 4096:
            raise ValueError("evidence journal capacity reached")
        identifier(evidence.id)
        identifier(evidence.source)
        if evidence.id in self._used or evidence.revision != self.problem.revision:
            raise ValueError("duplicate or stale evidence")
        if len(set(evidence.dependencies)) != len(evidence.dependencies) or any(
                dep not in self._active for dep in evidence.dependencies):
            raise ValueError("missing, revoked or cyclic dependency")
        if evidence.id in evidence.dependencies:
            raise ValueError("cyclic dependency")
        if len(set(evidence.targets)) != len(evidence.targets) or any(
                key not in self.problem._assignments for key in evidence.targets):
            raise ValueError("unknown or duplicate assignment")
        p, b = self.problem, self.budget
        if evidence.kind == "exclusion":
            if not evidence.targets or evidence.value not in p.constraint_names:
                raise ValueError("exclusion needs targets and named host constraint")
            valid = all(not p.satisfies(k, evidence.value, b) for k in evidence.targets)
        elif evidence.kind == "incumbent":
            valid = (len(evidence.targets) == 1 and type(evidence.value) is int
                     and p.feasible(evidence.targets[0], b)
                     and p.cost(evidence.targets[0], b) == evidence.value)
        elif evidence.kind == "lower_bound":
            if evidence.targets or type(evidence.value) is not int:
                raise ValueError("invalid bound")
            valid = all(p.cost(k, b) >= evidence.value for k in p.ids if p.feasible(k, b))
        else:
            raise ValueError("unsupported evidence kind")
        if not valid:
            raise ValueError("host checker rejected certificate")
        self._guard()
        if self._epoch != entry_epoch:
            raise ValueError("cooperation changed during verification")
        self._active[evidence.id] = Evidence(**{**data, "targets": tuple(data["targets"]),
                                               "dependencies": tuple(data["dependencies"])})
        self._used.add(evidence.id)
        self._events.append({"accept": data})
        self._advance()
        return self.residual()

    def revoke(self, evidence_id):
        self._guard()
        entry_epoch = self._epoch
        self.budget.charge("cooperation.revoke", 1 + len(self._active) ** 2)
        if len(self._events) >= 4096:
            raise ValueError("evidence journal capacity reached")
        if evidence_id not in self._active:
            raise ValueError("unknown active evidence")
        removed = {evidence_id}
        while True:
            extra = {k for k, e in self._active.items() if set(e.dependencies) & removed}
            if extra <= removed:
                break
            removed |= extra
        self._guard()
        if self._epoch != entry_epoch:
            raise ValueError("cooperation changed during revocation")
        for key in removed:
            del self._active[key]
        self._events.append({"revoke": evidence_id})
        self._advance()
        return tuple(sorted(removed))

    def run(self, solvers, *, rounds=1):
        self._guard()
        natural(rounds, "rounds")
        if rounds > 4096:
            raise ValueError("bounded rounds required")
        try:
            for _ in range(rounds):
                self.budget.charge("cooperation.round")
                for solver in solvers:
                    self.budget.charge("cooperation.solver_step")
                    claim = solver(self.residual(), self.budget)
                    if claim is not None:
                        self.submit(claim)
        except BudgetStop:
            return {"status": "unknown", "reason": "budget", "residual": None}
        try:
            residual = self.residual()
        except BudgetStop:
            return {"status": "unknown", "reason": "budget", "residual": None}
        return {"status": "unknown", "reason": "partial", "residual": residual}

    def certify(self):
        """Pay for independent exhaustive optimality check; never infer proof from silence."""
        self._guard()
        try:
            optimum = finite_reference(self.problem, self.budget)
        except BudgetStop:
            return {"status": "unknown", "reason": "budget"}
        return {"status": "infeasible" if optimum is None else "optimal", "solution": optimum}

    def snapshot(self):
        self._guard()
        self.budget.charge("cooperation.snapshot", 1 + sum(1 + len(e.get("accept", {}).get("targets", ()))
                         + len(e.get("accept", {}).get("dependencies", ())) for e in self._events))
        body = {"version": 1, "epoch": self._epoch, "problem": self.problem.fingerprint,
                "events": portable(self._events), "budget": self.budget.snapshot()}
        result = {**body, "digest": digest(body)}
        self.authority._checkpoint = result["digest"]
        return result

    @classmethod
    def restore(cls, snapshot, problem, budget, *, authority):
        if type(authority) is not CooperativeAuthority or not authority._claimed:
            raise ValueError("current host authority required")
        # Check the LIVE watermark before charging; a new restore debit must not
        # conceal a one-unit refund. Shared-ledger unrelated work counts too.
        watermark = authority._budget.snapshot() if authority._budget is not None else authority._watermark
        if watermark is None or type(budget) is not WorkBudget:
            raise ValueError("host ledger authority required")
        recorded_live = WorkBudget.restore(watermark)
        if budget.limit != recorded_live.limit or budget.spent < recorded_live.spent or any(
                budget.counts.get(k, 0) < v for k, v in recorded_live.counts.items()):
            raise ValueError("refunded live ledger rejected")
        if authority._budget is not None and authority._budget is not budget:
            raise ValueError("live host ledger identity required")
        authority._budget = budget
        budget.charge("cooperation.restore")
        if type(snapshot) is not dict or type(snapshot.get("events")) is not list or len(snapshot["events"]) > 4096:
            raise ValueError("bounded snapshot required")
        if set(snapshot) != {"version", "epoch", "problem", "events", "budget", "digest"}:
            raise ValueError("invalid snapshot schema")
        if type(snapshot["version"]) is not int or snapshot["version"] != 1 or type(snapshot["epoch"]) is not int:
            raise ValueError("invalid snapshot version or epoch")
        if type(snapshot["problem"]) is not str or len(snapshot["problem"]) != 64 or type(snapshot["digest"]) is not str or len(snapshot["digest"]) != 64:
            raise ValueError("invalid snapshot identity")
        if snapshot.get("epoch") != authority._epoch or snapshot.get("digest") != authority._checkpoint:
            raise ValueError("stale or untrusted cooperative checkpoint")
        # Price and bound untrusted traversal before a portable deep copy.
        for event in snapshot["events"]:
            budget.charge("cooperation.replay_input")
            if type(event) is not dict or len(event) != 1:
                raise ValueError("invalid event")
            if "accept" in event:
                row = event["accept"]
                if type(row) is not dict or set(row) != {"id", "source", "revision", "kind", "targets", "value", "dependencies"}:
                    raise ValueError("invalid evidence schema")
                for name in ("targets", "dependencies"):
                    if type(row[name]) is not list or len(row[name]) > 4096:
                        raise ValueError("bounded evidence sequence required")
                budget.charge("cooperation.replay_input", 1 + len(row["targets"]) + len(row["dependencies"]))
                Evidence(**row).data()
            elif "revoke" not in event or type(event["revoke"]) is not str or len(event["revoke"]) > 256:
                raise ValueError("invalid event")
        ledger = snapshot.get("budget")
        if type(ledger) is not dict or set(ledger) != {"v", "limit", "spent", "counts", "digest"} or type(ledger.get("counts")) is not dict:
            raise ValueError("bounded ledger required")
        budget.charge("cooperation.replay_input", 1 + len(ledger["counts"]))
        if any(type(ledger[k]) is not int for k in ("v", "limit", "spent")) or type(ledger["digest"]) is not str or len(ledger["digest"]) != 64 or any(type(k) is not str or type(v) is not int for k, v in ledger["counts"].items()):
            raise ValueError("invalid ledger scalars")
        s = portable(snapshot)
        if set(s) != {"version", "epoch", "problem", "events", "budget", "digest"}:
            raise ValueError("invalid snapshot schema")
        if type(s["version"]) is not int or s["version"] != 1 or type(s["epoch"]) is not int or s["problem"] != problem.fingerprint:
            raise ValueError("snapshot problem mismatch")
        if digest({k: v for k, v in s.items() if k != "digest"}) != s["digest"]:
            raise ValueError("corrupt snapshot")
        recorded = WorkBudget.restore(s["budget"])
        if budget.limit != recorded.limit or budget.spent < recorded.spent or any(
                budget.counts.get(k, 0) < v for k, v in recorded.counts.items()):
            raise ValueError("existing host ledger required; refunded ledger rejected")
        result = cls(problem, budget)
        for event in s["events"]:
            result.budget.charge("cooperation.replay")
            if set(event) == {"accept"}:
                e = event["accept"]
                if set(e) != {"id", "source", "revision", "kind", "targets", "value", "dependencies"}:
                    raise ValueError("invalid evidence schema")
                result.submit(Evidence(**{**e, "targets": tuple(e["targets"]),
                                          "dependencies": tuple(e["dependencies"])}))
            elif set(event) == {"revoke"}:
                result.revoke(event["revoke"])
            else:
                raise ValueError("invalid replay event")
        # Host sinks can mutate the original search during replay. Never
        # overwrite a revocation or transfer performed by such a callback.
        if authority._epoch != s["epoch"] or authority._checkpoint != s["digest"]:
            raise ValueError("cooperation authority changed during replay")
        # Only a complete paid replay transfers state ownership.
        authority._epoch += 1
        authority._checkpoint = None
        result.authority = authority
        result._epoch = authority._epoch
        return result


def finite_reference(problem, budget):
    """Exhaustive host oracle: return (integer cost, assignment ID), or None."""
    budget.charge("cooperation.reference")
    feasible = []
    for key in problem.ids:
        budget.charge("cooperation.enumerate")
        if problem.feasible(key, budget):
            feasible.append((problem.cost(key, budget), key))
    return min(feasible) if feasible else None


__all__ = ["FiniteProblem", "Evidence", "Residual", "ConstraintSolver", "ObjectiveSolver",
           "CooperativeSearch", "CooperativeAuthority", "finite_reference"]
