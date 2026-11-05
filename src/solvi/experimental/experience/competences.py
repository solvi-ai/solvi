"""Empirically admitted finite scalar programs, never a probability guarantee.

Sources are host assertions, not authentication or proof of truth. Inputs and
intermediate integers obey Grammar.max_abs. Admission is finite evidence only.
The host must checkpoint this state together with its shared WorkBudget ledger.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ._common import canonical, digest, identifier, natural, portable
from .programs import Grammar, Program, synthesize


@dataclass(frozen=True, init=False)
class ExternalExample:
    event_id: str
    task_id: str
    source: str
    mechanism: str
    schema: str
    scope: str
    _inputs: str
    expected: int | bool
    split: str

    def __init__(self, event_id, task_id, source, mechanism, schema, scope, inputs, expected, split):
        for name, value in locals().copy().items():
            if name in ("self", "inputs", "expected"):
                continue
            identifier(value, name)
            object.__setattr__(self, name, value)
        if type(inputs) is not dict or type(expected) not in (int, bool):
            raise ValueError("strict scalar example required")
        if split not in ("proposal", "admission", "evaluation"):
            raise ValueError("invalid example split")
        object.__setattr__(self, "_inputs", canonical(inputs))
        object.__setattr__(self, "expected", expected)

    @property
    def inputs(self):
        return json.loads(self._inputs)

    def to_dict(self):
        return {
            **{
                k: getattr(self, k)
                for k in ("event_id", "task_id", "source", "mechanism", "schema", "scope", "expected", "split")
            },
            "inputs": self.inputs,
        }

    @classmethod
    def from_dict(cls, value):
        return cls(**portable(value))


@dataclass(frozen=True)
class Candidate:
    program: Program
    proposal_examples: tuple[ExternalExample, ...]
    scope: str
    mechanism: str
    schema: str
    grammar_fingerprint: str
    dependencies: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "proposal_examples", tuple(self.proposal_examples))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    @property
    def proposal_ids(self):
        return tuple(e.event_id for e in self.proposal_examples)

    def to_dict(self):
        return {
            "program": self.program.to_dict(),
            "proposal_examples": [e.to_dict() for e in self.proposal_examples],
            "scope": self.scope,
            "mechanism": self.mechanism,
            "schema": self.schema,
            "grammar_fingerprint": self.grammar_fingerprint,
            "dependencies": list(self.dependencies),
        }

    @property
    def fingerprint(self):
        return digest(self.to_dict())

    @property
    def candidate_id(self):
        return self.fingerprint

    @classmethod
    def from_dict(cls, data):
        d = portable(data)
        d["program"] = Program.from_dict(d["program"])
        d["proposal_examples"] = tuple(ExternalExample.from_dict(e) for e in d["proposal_examples"])
        return cls(**d)


@dataclass(frozen=True)
class AdmissionRecord:
    accepted: bool
    reason: str
    competence_id: str | None = None


@dataclass(frozen=True)
class Application:
    status: str
    value: int | bool | None = None
    competence_id: str | None = None
    application_id: str | None = None


class CompetenceRegistry:
    def __init__(self, grammar: Grammar, *, store=None, allowed_sources=("environment", "human", "audit")):
        self.grammar = grammar
        self.store = store
        self.allowed_sources = tuple(allowed_sources)
        for source in self.allowed_sources:
            identifier(source, "source")
        self.last_search_reason = None
        self._events = {}
        self._tasks = {}
        self._candidates = {}
        self._admissions = {}
        self._competences = {}
        self._applications = {}
        self._contexts = {}
        self._invalidations = {}

    def _validate(self, example, budget, split):
        budget.charge("competence_validate")
        if not isinstance(example, ExternalExample):
            raise ValueError("external example required")
        if example.source not in self.allowed_sources or example.split != split:
            raise ValueError("untrusted source or wrong split")
        if set(example.inputs) != set(self.grammar.inputs):
            raise ValueError("input schema mismatch")
        for value in [*example.inputs.values(), example.expected]:
            budget.charge("competence_validate")
            if type(value) not in (int, bool) or (type(value) is int and abs(value) > self.grammar.max_abs):
                raise ValueError("value outside finite grammar domain")
        old = self._events.get(example.event_id)
        if old is not None and old != example.to_dict():
            raise ValueError("event identity collision")
        task = self._tasks.get(example.task_id)
        if task is not None and task != [example.split, example.event_id]:
            raise ValueError("task identity collision or cross-split leakage")

    def _batch(self, examples, budget, split):
        rows = tuple(examples)
        events, tasks = set(), set()
        context = None
        for e in rows:
            self._validate(e, budget, split)
            if any(a["task_id"] == e.task_id for a in self._applications.values()):
                raise ValueError("application task cannot become learning data")
            if e.event_id in events or e.task_id in tasks:
                raise ValueError("duplicate example in batch")
            events.add(e.event_id)
            tasks.add(e.task_id)
            ctx = (e.scope, e.mechanism, e.schema)
            if context is not None and ctx != context:
                raise ValueError("mixed example contexts")
            context = ctx
        return rows

    def _remember(self, rows):
        for e in rows:
            self._events[e.event_id] = e.to_dict()
            self._tasks[e.task_id] = [e.split, e.event_id]

    def propose(self, examples, budget):
        budget.charge("competence_propose")
        rows = self._batch(examples, budget, "proposal")
        self._remember(rows)
        result = synthesize([(e.inputs, e.expected) for e in rows], self.grammar, budget)
        self.last_search_reason = result.reason
        if result.candidate is None:
            return None
        e = rows[0]
        candidate = Candidate(result.candidate, rows, e.scope, e.mechanism, e.schema, self.grammar.fingerprint)
        self._candidates[candidate.fingerprint] = candidate.to_dict()
        return candidate

    def derive(self, program: Program, proposal_examples, dependencies, budget):
        """Host-proposed immutable AST composition; not autonomous discovery.

        Every named dependency must occur as an exact expanded AST subtree.
        Execution charges all expanded nodes through ordinary Program.evaluate.
        """
        budget.charge("competence_derive")
        if not isinstance(program, Program):
            raise ValueError("Program required")
        rows = self._batch(proposal_examples, budget, "proposal")
        if not rows:
            raise ValueError("proposal examples required")
        deps = tuple(dependencies)
        if not deps or len(set(deps)) != len(deps):
            raise ValueError("unique dependencies required")
        context = (rows[0].scope, rows[0].mechanism, rows[0].schema)
        subtrees = set()

        def visit(node):
            budget.charge("competence_composition")
            subtrees.add(node)
            if node[0] not in ("input", "const"):
                for child in node[1:]:
                    visit(child)

        visit(program.ast)
        for dep in deps:
            budget.charge("competence_lookup")
            rec = self._competences.get(dep)
            if rec is None or rec["status"] != "active":
                raise ValueError("dependency is not active")
            if self.store is not None and not self.store.usable(rec["store_id"]):
                self.invalidate(dep, "store_unusable", "store:" + dep)
                raise ValueError("dependency store item is not usable")
            c = Candidate.from_dict(self._candidates[rec["candidate_id"]])
            if (c.scope, c.mechanism, c.schema) != context or c.program.ast not in subtrees:
                raise ValueError("dependency context or expanded AST mismatch")
        for e in rows:
            value = program.evaluate(e.inputs, budget, self.grammar)
            budget.charge("competence_compare")
            if type(value) is not type(e.expected) or value != e.expected:
                raise ValueError("derived program fails proposal")
        self._remember(rows)
        c = Candidate(program, rows, *context, self.grammar.fingerprint, deps)
        self._candidates[c.fingerprint] = c.to_dict()
        return c

    def admit(self, candidate, examples, budget, *, min_examples=4, on_reserved=None):
        budget.charge("competence_admit")
        natural(min_examples, "min_examples", positive=True)
        if not isinstance(candidate, Candidate) or candidate.grammar_fingerprint != self.grammar.fingerprint:
            raise ValueError("candidate grammar mismatch")
        cid = candidate.fingerprint
        if self._candidates.get(cid) != candidate.to_dict():
            raise ValueError("candidate must be frozen by this registry")
        rows = self._batch(examples, budget, "admission")
        batch_id = digest(sorted([e.to_dict() for e in rows], key=lambda e: e["event_id"]))
        previous = self._admissions.get(cid)
        if previous is not None:
            if previous["batch_id"] != batch_id or previous["min_examples"] != min_examples:
                raise ValueError("candidate already frozen for a different admission")
            return AdmissionRecord(**previous["result"])
        for e in rows:
            budget.charge("competence_lookup")
            if e.event_id in self._events or e.task_id in self._tasks:
                raise ValueError("admission data already consumed")
            if (e.scope, e.mechanism, e.schema) != (candidate.scope, candidate.mechanism, candidate.schema):
                raise ValueError("admission context mismatch")
        # Reserve the whole batch before inspecting any label or running any AST.
        self._remember(rows)
        record = {
            "batch_id": batch_id,
            "min_examples": min_examples,
            "event_ids": [e.event_id for e in rows],
            "result": {"accepted": False, "reason": "interrupted", "competence_id": None},
        }
        self._admissions[cid] = record
        if on_reserved is not None:
            on_reserved()

        def finish(accepted, reason, competence_id=None):
            record["result"] = {"accepted": accepted, "reason": reason, "competence_id": competence_id}
            return AdmissionRecord(**record["result"])

        for dep in candidate.dependencies:
            budget.charge("competence_lookup")
            if dep not in self._competences or self._competences[dep]["status"] != "active":
                return finish(False, "dependency_revoked")
            if self.store is not None and not self.store.usable(self._competences[dep]["store_id"]):
                self.invalidate(dep, "store_unusable", "store:" + dep)
                return finish(False, "dependency_revoked")
        if len(rows) < min_examples:
            return finish(False, "insufficient_examples")
        for e in rows:
            try:
                value = candidate.program.evaluate(e.inputs, budget, self.grammar)
            except ValueError:
                return finish(False, "invalid_program")
            budget.charge("competence_compare")
            if type(value) is not type(e.expected) or value != e.expected:
                return finish(False, "counterexample")
        key = cid
        store_id = None
        if self.store is not None:
            budget.charge("competence_store")
            store_id = self.store.add(
                "skill",
                {"experience_program": candidate.to_dict()},
                scope={"scope": candidate.scope, "mechanism": candidate.mechanism, "schema": candidate.schema},
                source="outcome",
                evidence=[e.event_id for e in rows],
                derived_from=tuple(self._competences[d]["store_id"] for d in candidate.dependencies),
            )
            if store_id is None or not self.store.usable(store_id):
                return finish(False, "store_refused")
        self._competences[key] = {"candidate_id": cid, "status": "active", "store_id": store_id, "reason": None}
        return finish(True, "admitted", key)

    def predict(self, inputs, *, task_id, scope, mechanism, schema, budget):
        budget.charge("competence_predict")
        for value in (task_id, scope, mechanism, schema):
            identifier(value)
        if task_id in self._tasks or any(a["task_id"] == task_id for a in self._applications.values()):
            raise ValueError("application task identity already used")
        context = [mechanism, schema]
        previous = self._contexts.get(scope)
        if previous != context:
            for key, rec in self._competences.items():
                budget.charge("competence_lookup")
                c = self._candidates[rec["candidate_id"]]
                if c["scope"] == scope and [c["mechanism"], c["schema"]] != context and rec["status"] == "active":
                    self.invalidate(key, "context_changed", "context:" + digest([scope, previous, context, key]))
        self._contexts[scope] = context
        for key, rec in self._competences.items():
            budget.charge("competence_lookup")
            if rec["status"] == "active" and self.store is not None:
                budget.charge("competence_store")
                if not self.store.usable(rec["store_id"]):
                    self.invalidate(key, "store_unusable", "store:" + key)
            c = Candidate.from_dict(self._candidates[rec["candidate_id"]])
            if rec["status"] != "active" or (c.scope, c.mechanism, c.schema) != (scope, mechanism, schema):
                continue
            try:
                value = c.program.evaluate(inputs, budget, self.grammar)
            except ValueError:
                continue
            app = {
                "task_id": task_id,
                "scope": scope,
                "mechanism": mechanism,
                "schema": schema,
                "inputs": portable(inputs),
                "prediction": value,
                "competence_id": key,
            }
            aid = digest(app)
            self._applications[aid] = app
            return Application("applied", value, key, aid)
        return Application("unknown")

    def observe(self, application_id, example, budget):
        budget.charge("competence_observe")
        if not isinstance(example, ExternalExample) or example.split not in ("proposal", "admission", "evaluation"):
            raise ValueError("external observation required")
        self._validate(example, budget, example.split)
        budget.charge("competence_lookup")
        app = self._applications.get(application_id)
        if app is None:
            raise ValueError("unknown application")
        if any(getattr(example, k) != app[k] for k in ("task_id", "scope", "mechanism", "schema")) or canonical(
            example.inputs
        ) != canonical(app["inputs"]):
            raise ValueError("observation does not match application")
        if example.event_id in self._events:
            return ()
        budget.charge("competence_compare")
        if type(app["prediction"]) is not type(example.expected) or app["prediction"] != example.expected:
            revoked = self.invalidate(app["competence_id"], "external_counterexample", example.event_id)
            self._remember((example,))
            return revoked
        self._remember((example,))
        return ()

    def invalidate(self, competence_id, reason, event_id):
        identifier(reason, "reason")
        identifier(event_id, "event_id")
        payload = [competence_id, reason]
        if event_id in self._invalidations:
            if self._invalidations[event_id] != payload:
                raise ValueError("invalidation event collision")
            return ()
        if competence_id not in self._competences:
            raise ValueError("unknown competence")
        self._invalidations[event_id] = payload
        revoked = []
        pending = {competence_id}
        while pending:
            key = sorted(pending)[0]
            pending.remove(key)
            rec = self._competences[key]
            if rec["status"] == "revoked":
                continue
            rec["status"], rec["reason"] = "revoked", reason
            revoked.append(key)
            if self.store is not None and rec["store_id"] is not None:
                self.store.retract(rec["store_id"], why=reason)
            for other, entry in self._competences.items():
                if key in self._candidates[entry["candidate_id"]]["dependencies"] and entry["status"] == "active":
                    pending.add(other)
        return tuple(revoked)

    def snapshot(self):
        body = {
            "v": 1,
            "grammar_fingerprint": self.grammar.fingerprint,
            "allowed_sources": list(self.allowed_sources),
            "last_search_reason": self.last_search_reason,
        }
        for name in (
            "events",
            "tasks",
            "candidates",
            "admissions",
            "competences",
            "applications",
            "contexts",
            "invalidations",
        ):
            body[name] = portable(getattr(self, "_" + name))
        return {**body, "digest": digest(body)}

    @classmethod
    def restore(cls, snapshot, *, grammar, store=None, allowed_sources=("environment", "human", "audit")):
        data = portable(snapshot)
        checksum = data.pop("digest", None)
        if (
            checksum != digest(data)
            or type(data.get("v")) is not int
            or data.get("v") != 1
            or data.get("grammar_fingerprint") != grammar.fingerprint
        ):
            raise ValueError("registry snapshot integrity or grammar mismatch")
        registry = cls(grammar, store=store, allowed_sources=allowed_sources)
        if data.get("allowed_sources") != list(registry.allowed_sources):
            raise ValueError("source policy mismatch")
        expected = set(registry.snapshot()) - {"digest"}
        if set(data) != expected:
            raise ValueError("invalid registry snapshot fields")
        try:
            registry._validate_snapshot(data)
        except (KeyError, TypeError, AttributeError, IndexError, RecursionError) as exc:
            raise ValueError("invalid registry snapshot structure") from exc
        for name in (
            "events",
            "tasks",
            "candidates",
            "admissions",
            "competences",
            "applications",
            "contexts",
            "invalidations",
        ):
            setattr(registry, "_" + name, data[name])
        registry.last_search_reason = data["last_search_reason"]
        return registry

    def _validate_snapshot(self, data):
        """Check cross references without executing programs or restoring credit."""
        names = (
            "events",
            "tasks",
            "candidates",
            "admissions",
            "competences",
            "applications",
            "contexts",
            "invalidations",
        )
        if any(type(data[name]) is not dict for name in names):
            raise ValueError("registry sections must be dictionaries")
        if data["last_search_reason"] is not None:
            identifier(data["last_search_reason"])
        events, candidates = {}, {}

        def scalar(value):
            if type(value) not in (int, bool) or (type(value) is int and abs(value) > self.grammar.max_abs):
                raise ValueError("scalar outside grammar")

        def inputs(value):
            if type(value) is not dict or set(value) != set(self.grammar.inputs):
                raise ValueError("invalid stored inputs")
            for v in value.values():
                scalar(v)

        def context(value):
            for name in ("scope", "mechanism", "schema"):
                identifier(value[name], name)
            return tuple(value[k] for k in ("scope", "mechanism", "schema"))

        tasks = {}
        for key, value in data["events"].items():
            e = ExternalExample.from_dict(value)
            if e.event_id != key or e.source not in self.allowed_sources:
                raise ValueError("invalid event identity or source")
            inputs(e.inputs)
            scalar(e.expected)
            if e.task_id in tasks:
                raise ValueError("duplicate task identity")
            tasks[e.task_id] = [e.split, key]
            events[key] = e
        if tasks != data["tasks"]:
            raise ValueError("event/task index mismatch")

        def ast_nodes(node):
            if node[0] == "input":
                if node[1] not in self.grammar.inputs:
                    raise ValueError("undeclared input")
            elif node[0] == "const":
                scalar(node[1])
                if type(node[1]) is int and node[1] not in self.grammar.constants:
                    raise ValueError("undeclared constant")
            else:
                if node[0] not in self.grammar.operators:
                    raise ValueError("undeclared operator")
                for child in node[1:]:
                    yield from ast_nodes(child)
            yield node

        for key, value in data["candidates"].items():
            c = Candidate.from_dict(value)
            if c.fingerprint != key or c.grammar_fingerprint != self.grammar.fingerprint:
                raise ValueError("invalid candidate fingerprint")
            context(value)
            if c.program.nodes > self.grammar.max_nodes or c.program.depth > self.grammar.max_depth:
                raise ValueError("candidate exceeds grammar caps")
            set(ast_nodes(c.program.ast))
            if not c.proposal_examples or len(set(c.proposal_ids)) != len(c.proposal_ids):
                raise ValueError("invalid proposal identities")
            rowtypes = set()
            for e in c.proposal_examples:
                if (
                    events.get(e.event_id) != e
                    or e.split != "proposal"
                    or (e.scope, e.mechanism, e.schema) != context(value)
                ):
                    raise ValueError("candidate proposal provenance mismatch")
                rowtypes.add(
                    tuple(type(e.inputs[n]).__name__ for n in self.grammar.inputs) + (type(e.expected).__name__,)
                )
            if len(rowtypes) != 1 or len(set(c.dependencies)) != len(c.dependencies):
                raise ValueError("invalid candidate types or dependencies")
            candidates[key] = c

        admissions_used = set()
        for key, rec in data["admissions"].items():
            if key not in candidates or set(rec) != {"batch_id", "min_examples", "event_ids", "result"}:
                raise ValueError("invalid admission candidate or fields")
            natural(rec["min_examples"], positive=True)
            ids = rec["event_ids"]
            if type(ids) is not list or len(set(ids)) != len(ids) or admissions_used.intersection(ids):
                raise ValueError("reused admission identities")
            admissions_used.update(ids)
            c = candidates[key]
            rows = []
            for eid in ids:
                e = events[eid]
                if e.split != "admission" or (e.scope, e.mechanism, e.schema) != (c.scope, c.mechanism, c.schema):
                    raise ValueError("admission split or context mismatch")
                rows.append(e.to_dict())
            if digest(sorted(rows, key=lambda e: e["event_id"])) != rec["batch_id"]:
                raise ValueError("admission batch fingerprint mismatch")
            result = rec["result"]
            if set(result) != {"accepted", "reason", "competence_id"} or type(result["accepted"]) is not bool:
                raise ValueError("invalid admission result")
            if result["reason"] not in (
                "admitted",
                "interrupted",
                "dependency_revoked",
                "insufficient_examples",
                "invalid_program",
                "counterexample",
                "store_refused",
            ):
                raise ValueError("invalid admission reason")
            if result["accepted"]:
                if (
                    result["reason"] != "admitted"
                    or result["competence_id"] != key
                    or len(ids) < rec["min_examples"]
                    or key not in data["competences"]
                ):
                    raise ValueError("invalid accepted admission")
            elif result["competence_id"] is not None or key in data["competences"] or result["reason"] == "admitted":
                raise ValueError("invalid rejected admission")

        for key, rec in data["competences"].items():
            if (
                set(rec) != {"candidate_id", "status", "store_id", "reason"}
                or key != rec["candidate_id"]
                or key not in candidates
            ):
                raise ValueError("invalid competence identity")
            if rec["status"] not in ("active", "revoked") or not data["admissions"][key]["result"]["accepted"]:
                raise ValueError("invalid competence status")
            if (rec["status"] == "active" and rec["reason"] is not None) or (
                rec["status"] == "revoked" and not rec["reason"]
            ):
                raise ValueError("invalid revocation reason")
            if rec["store_id"] is not None:
                identifier(rec["store_id"])
                if self.store is not None:
                    item = self.store.item(rec["store_id"])
                    if (
                        item is None
                        or item["kind"] != "skill"
                        or item["body"] != {"experience_program": candidates[key].to_dict()}
                    ):
                        raise ValueError("store reference mismatch")
                    required = [data["competences"][d]["store_id"] for d in candidates[key].dependencies]
                    if set(item["derived_from"]) != set(required):
                        raise ValueError("store dependency mismatch")
            elif self.store is not None:
                raise ValueError("missing store reference")
        visited, visiting = set(), set()

        def visit(key):
            if key in visiting:
                raise ValueError("dependency cycle")
            if key in visited:
                return
            visiting.add(key)
            c = candidates[key]
            subtrees = set(ast_nodes(c.program.ast))
            for dep in c.dependencies:
                if dep not in data["competences"]:
                    raise ValueError("missing dependency")
                d = candidates[dep]
                if (c.scope, c.mechanism, c.schema) != (
                    d.scope,
                    d.mechanism,
                    d.schema,
                ) or d.program.ast not in subtrees:
                    raise ValueError("invalid expanded dependency")
                if (
                    key in data["competences"]
                    and data["competences"][key]["status"] == "active"
                    and data["competences"][dep]["status"] != "active"
                ):
                    raise ValueError("active dependent of revoked competence")
                visit(dep)
            visiting.remove(key)
            visited.add(key)

        for key in candidates:
            visit(key)
        application_tasks = set()
        for key, value in data["applications"].items():
            if (
                set(value) != {"task_id", "scope", "mechanism", "schema", "inputs", "prediction", "competence_id"}
                or digest(value) != key
            ):
                raise ValueError("invalid application identity")
            identifier(value["task_id"])
            c = candidates[value["competence_id"]]
            if value["competence_id"] not in data["competences"] or context(value) != (c.scope, c.mechanism, c.schema):
                raise ValueError("invalid application competence")
            inputs(value["inputs"])
            scalar(value["prediction"])
            if value["task_id"] in application_tasks:
                raise ValueError("duplicate application task")
            application_tasks.add(value["task_id"])
            if value["task_id"] in tasks:
                e = events[tasks[value["task_id"]][1]]
                if (e.scope, e.mechanism, e.schema) != context(value) or canonical(e.inputs) != canonical(
                    value["inputs"]
                ):
                    raise ValueError("observation/application mismatch")
                if e.event_id in admissions_used or any(e.event_id in cc.proposal_ids for cc in candidates.values()):
                    raise ValueError("application outcome leaked into learning")
        for key, value in data["contexts"].items():
            identifier(key)
            if type(value) is not list or len(value) != 2:
                raise ValueError("invalid context token")
            for token in value:
                identifier(token)
        for key, value in data["invalidations"].items():
            identifier(key)
            if (
                type(value) is not list
                or len(value) != 2
                or value[0] not in data["competences"]
                or data["competences"][value[0]]["status"] != "revoked"
            ):
                raise ValueError("invalid invalidation reference")
            identifier(value[1])


__all__ = ["AdmissionRecord", "Application", "Candidate", "CompetenceRegistry", "ExternalExample"]
