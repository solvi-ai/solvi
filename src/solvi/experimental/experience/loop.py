"""Finite host-driven experience loop with durable intent and fail-closed recovery.

Replay is pure. Recovery never repeats an ambiguous external effect. Interrupted learning is retained and blocked; admission reservations are
checkpointed before candidate execution. Host recovery must settle that transaction. If execute has an unknown outcome
and reconcile cannot settle it, automatic execution remains blocked: there is
no exactly-once claim for a host without reconciliation.
"""

from __future__ import annotations

from typing import TypedDict

from ...core.costs import BudgetStop
from ._common import WorkBudget, digest, identifier, natural, portable
from .acquisition import Probe, select_probe
from .interfaces import Receipt, TaskSpec
from .worlds import WorldSet


class _Configuration(TypedDict):
    environment: str
    mode: str
    seed: int
    horizon: int


class _Pending(TypedDict):
    attempt_id: str
    kind: str
    payload: dict
    receipt: dict | None


class ExperienceRunner:
    MODES = ("fixed", "random", "active", "full", "no_call")

    def __init__(self, environment, journal, budget, registry=None, *, mode="active", seed=0, horizon=2):
        if mode not in self.MODES:
            raise ValueError("unsupported experience mode")
        natural(seed, "seed")
        natural(horizon, "horizon", positive=True)
        if horizon > 16:
            raise ValueError("horizon exceeds planning cap")
        if journal.records():
            raise ValueError("existing cycle requires restore, never a fresh budget")
        self.environment, self.journal, self.budget, self.registry = environment, journal, budget, registry
        self.config: _Configuration = {
            "environment": identifier(environment.identity),
            "mode": mode,
            "seed": seed,
            "horizon": horizon,
        }
        self.task: TaskSpec | None = None
        self.worlds: WorldSet | None = None
        self.pending: _Pending | None = None
        self.phase, self.status = "ready", "inprogress"
        self.counter = 0
        self.queue, self.cursor = [], 0
        self.events, self.results, self.observed = {}, [], []
        self.application_id = self.competence_id = self.learning = None
        self._bind_budget()
        self._checkpoint()

    def _bind_budget(self):
        self.budget.on_charge = self.journal.debit

    def _state(self):
        return dict(
            version=1,
            config=self.config,
            ledger=self.budget.snapshot(),
            task=self.task.to_dict() if self.task else None,
            worlds=self.worlds.snapshot() if self.worlds is not None else None,
            registry=self.registry.snapshot() if self.registry else None,
            pending=self.pending,
            phase=self.phase,
            status=self.status,
            counter=self.counter,
            events=self.events,
            results=self.results,
            observed=self.observed,
            application_id=self.application_id,
            competence_id=self.competence_id,
            learning=self.learning,
            queue=self.queue,
            cursor=self.cursor,
        )

    def _record(self, kind):
        self.journal.append(kind, {"state": self._state()})

    def _checkpoint(self):
        self.journal.checkpoint(self._state())

    def snapshot(self):
        self._checkpoint()
        return self.journal.latest()

    def replay(self):
        """Pure fold of durable state; no environment calls or new writes."""
        state = None
        for event in self.journal.records():
            payload = event["payload"]
            if event["kind"] == "checkpoint":
                state = portable(payload["state"])
            elif "state" in payload:
                state = portable(payload["state"])
            elif event["kind"] == "debit" and state is not None:
                state["ledger"] = portable(payload["ledger"])
        return state

    @classmethod
    def restore(cls, snapshot, *, environment, journal, registry=None):
        snap = portable(snapshot)
        records = journal.records()
        if not any(e["kind"] == "checkpoint" and e["payload"] == snap for e in records):
            raise ValueError("snapshot has no matching durable journal anchor")
        obj = object.__new__(cls)
        obj.environment, obj.journal, obj.registry = environment, journal, registry
        state = obj.replay()
        if state["version"] != 1 or state["config"] != snap["state"]["config"]:
            raise ValueError("incompatible runner configuration")
        if state["config"]["environment"] != environment.identity:
            raise ValueError("foreign environment identity")
        obj.config = state["config"]
        obj.budget = WorkBudget.restore(state["ledger"])
        obj.task = TaskSpec.from_dict(state["task"]) if state["task"] else None
        obj.worlds = WorldSet.restore(state["worlds"]) if state["worlds"] is not None else None
        for name in (
            "pending",
            "phase",
            "status",
            "counter",
            "events",
            "results",
            "observed",
            "application_id",
            "competence_id",
            "learning",
            "queue",
            "cursor",
        ):
            setattr(obj, name, state[name])
        natural(obj.cursor, "queue cursor")
        if (
            type(obj.queue) is not list
            or obj.cursor > len(obj.queue)
            or any(type(t) is not str or not t for t in obj.queue)
            or len(set(obj.queue)) != len(obj.queue)
        ):
            raise ValueError("invalid durable task queue")
        if state["registry"] is not None:
            if registry is None:
                raise ValueError("registry required for recovery")
            obj.registry = type(registry).restore(
                state["registry"],
                grammar=registry.grammar,
                store=registry.store,
                allowed_sources=registry.allowed_sources,
            )
        obj._bind_budget()
        if obj.learning is not None:
            obj.status = "unresolved"
            obj._checkpoint()
        elif obj.phase == "intent":
            reconcile = getattr(environment, "reconcile", None)
            receipt = reconcile(obj.pending["attempt_id"]) if callable(reconcile) else None
            if receipt is None and getattr(environment, "idempotent_attempts", False) is True:
                # An explicit host capability, implemented by SQLiteEnvironment,
                # permits retrying this exact durable attempt, never a new ID.
                receipt = environment.execute(
                    obj.task.task_id, obj.pending["kind"], obj.pending["payload"], obj.pending["attempt_id"]
                )
            if receipt is None:
                obj.status = "unresolved"
                obj._checkpoint()
            else:
                obj._accept(receipt)
        return obj

    def start(self, task_id):
        identifier(task_id, "task_id")
        if any(result["task_id"] == task_id for result in self.results):
            raise ValueError("task already has a terminal result")
        if self.cursor < len(self.queue) and self.queue[self.cursor] != task_id:
            raise ValueError("task differs from durable queue cursor")
        if self.learning is not None or self.pending is not None or (self.task is not None and self.phase != "done"):
            raise ValueError("finish the current task before starting another")
        task = self.environment.task(task_id)
        if not isinstance(task, TaskSpec) or task.task_id != task_id:
            raise ValueError("environment returned another task")
        self.task, self.worlds = task, WorldSet(task.worlds)
        if any(task.target not in w for w in task.worlds):
            raise ValueError("target absent from joint worlds")
        self.observed, self.application_id, self.competence_id = [], None, None
        self.phase, self.status = "ready", "inprogress"
        self._checkpoint()
        return self.status

    def _finish(self, status, answer=None, *, receipt=None, proposed=None):
        self.status, self.phase = status, "done"
        self.results.append(
            dict(
                task_id=self.task.task_id,
                status=status,
                answer=answer,
                proposed_answer=proposed,
                expected=receipt["value"]["answer"] if receipt else None,
                ok=receipt["value"]["ok"] if receipt else None,
                competence_id=self.competence_id,
                observed=list(self.observed),
                receipt_event=receipt["event_id"] if receipt else None,
            )
        )
        if self.cursor < len(self.queue):
            if self.queue[self.cursor] != self.task.task_id:
                raise ValueError("terminal task differs from durable queue cursor")
            self.cursor += 1
        self._checkpoint()
        return status

    def _issue(self, kind: str, payload: dict, cost: int):
        self.counter += 1
        attempt = digest(dict(cycle=self.journal.cycle_id, task=self.task.task_id, counter=self.counter))
        # Debit is durable before effect; intent is durable before host execution.
        self.budget.charge(kind, cost)
        self.pending = {"attempt_id": attempt, "kind": kind, "payload": payload, "receipt": None}
        self.phase = "intent"
        self._record("intent")
        receipt = self.environment.execute(self.task.task_id, kind, payload, attempt)
        if receipt is None:
            self.status = "unresolved"
            self._checkpoint()
            return self.status
        self._accept(receipt)
        return self.status

    def _accept(self, receipt):
        if not isinstance(receipt, Receipt):
            raise ValueError("host must return a Receipt")
        p, t = self.pending, self.task
        if p is None or t is None:
            raise ValueError("receipt has no pending host attempt")
        if (
            receipt.source != self.environment.identity
            or receipt.task_id != t.task_id
            or receipt.mechanism != t.mechanism
            or receipt.attempt_id != p["attempt_id"]
            or receipt.kind != p["kind"]
        ):
            raise ValueError("foreign or mismatched external receipt")
        value = receipt.value
        if receipt.kind == "observe":
            if set(value) != {"field", "value"} or value["field"] != p["payload"]["field"]:
                raise ValueError("observation receipt does not match requested field")
        elif "answer" not in value or type(value.get("ok")) is not bool:
            raise ValueError("action receipt requires external answer and boolean ok")
        encoded = receipt.to_dict()
        old = self.events.get(receipt.event_id)
        if old is not None and old != encoded:
            raise ValueError("conflicting duplicate external event ID")
        self.events[receipt.event_id] = encoded
        p["receipt"] = encoded
        self.phase, self.status = "receipt", "inprogress"
        self._record("receipt")

    def _revise(self):
        r = Receipt.from_dict(self.pending["receipt"])
        if r.kind == "observe":
            self.worlds = self.worlds.observe(r.value["field"], r.value["value"], r.event_id, self.budget)
            if self.pending["payload"]["probe"] not in self.observed:
                self.observed.append(self.pending["payload"]["probe"])
        else:
            evidence = dict(event_id=r.event_id, attempt=r.attempt_id)
            # Survives a crash after add but before revision_applied.
            exists = any(
                x.get("op") == "assert"
                and x.get("source") == "outcome"
                and x.get("by") == r.source
                and x.get("evidence") == evidence
                for x in self.journal.store.journal
            )
            if not exists:
                item = self.journal.store.add(
                    "fact",
                    {"s": r.task_id, "r": "observed_answer", "o": r.value["answer"]},
                    scope={"mechanism": r.mechanism, "scope": self.task.scope},
                    source="outcome",
                    by=r.source,
                    evidence=evidence,
                )
                if item is None:
                    raise ValueError("knowledge policy refused external outcome")
            if self.registry is not None and self.application_id is not None:
                from .competences import ExternalExample

                example = ExternalExample(
                    r.event_id,
                    r.task_id,
                    r.source,
                    r.mechanism,
                    self.task.schema,
                    self.task.scope,
                    self.task.inputs,
                    r.value["answer"],
                    "evaluation",
                )
                self.registry.observe(self.application_id, example, self.budget)
        self.phase = "revision"
        self._record("revision")

    def step(self):
        if self.learning is not None:
            return "unresolved"
        task, worlds = self.task, self.worlds
        if task is None or worlds is None:
            raise ValueError("start a task first")
        if self.phase == "done" or self.status == "unresolved":
            return self.status
        try:
            if self.phase == "receipt":
                self._revise()
            if self.phase == "revision":
                receipt = self.pending["receipt"]
                proposed = self.pending["payload"].get("answer")
                self.pending = None
                if receipt["kind"] == "act":
                    return self._finish(
                        "done" if receipt["value"]["ok"] else "wrong", proposed, receipt=receipt, proposed=proposed
                    )
                self.phase = "ready"
                self._checkpoint()
                return self.status
            if self.phase == "intent":
                self.status = "unresolved"
                self._checkpoint()
                return self.status
            question = lambda world: world[task.target]
            if self.registry is not None and self.config["mode"] != "no_call":
                app = self.registry.predict(
                    self.task.inputs,
                    task_id=self.task.task_id,
                    scope=self.task.scope,
                    mechanism=self.task.mechanism,
                    schema=self.task.schema,
                    budget=self.budget,
                )
                if app.status == "applied":
                    self.application_id, self.competence_id = app.application_id, app.competence_id
                    return self._issue("act", {"answer": app.value}, self.task.action_cost)
            decision = worlds.decide(question, self.budget)
            if decision.status == "budget":
                return self._finish("budget")
            probes = [p for p in self.task.probes if p.name not in self.observed]
            mode = self.config["mode"]
            if decision.resolved:
                return self._issue("act", {"answer": decision.value}, self.task.action_cost)
            if not probes or decision.status == "contradiction":
                return self._finish("abstain")
            if mode in ("active", "full", "no_call"):
                choice = select_probe(
                    worlds,
                    question,
                    [Probe(p.name, lambda s, f=p.field: s[f], p.cost) for p in probes],
                    self.budget,
                    horizon=self.config["horizon"],
                )
                if choice.name is None:
                    return self._finish("budget" if choice.status == "budget" else "abstain")
                probe = next(p for p in probes if p.name == choice.name)
            elif mode == "random":
                key = digest(dict(seed=self.config["seed"], task=self.task.task_id, counter=self.counter))
                probe = probes[int(key, 16) % len(probes)]
            else:
                probe = probes[0]
            return self._issue("observe", {"probe": probe.name, "field": probe.field}, probe.cost)
        except BudgetStop:
            # A durable receipt remains recoverable even if revision is unaffordable.
            self.status = "budget"
            self._checkpoint()
            return self.status

    def run(self, task_ids=None, max_steps=1000):
        """Run or resume a durable finite queue; max_steps yields without reset.

        Supply the same queue to resume, no argument after restore, or a fresh
        disjoint queue after completion. Each terminal result and cursor advance
        share one checkpoint. An unresolved host outcome blocks automatic resume.
        """
        natural(max_steps, "max_steps")
        if task_ids is not None:
            proposed = list(task_ids)
            for task_id in proposed:
                identifier(task_id, "task_id")
            if len(set(proposed)) != len(proposed):
                raise ValueError("task queue requires unique task IDs")
            if proposed != self.queue:
                if self.cursor < len(self.queue):
                    raise ValueError("cannot replace an unfinished durable queue")
                completed = {r["task_id"] for r in self.results}
                if completed.intersection(proposed):
                    raise ValueError("task queue reuses terminal task IDs")
                if self.task is not None and self.phase != "done":
                    if not proposed or proposed[0] != self.task.task_id:
                        raise ValueError("queue must begin with the current task")
                self.queue, self.cursor = proposed, 0
                self._checkpoint()
        elif not self.queue and self.task is not None and self.phase != "done":
            self.queue, self.cursor = [self.task.task_id], 0
            self._checkpoint()
        steps = 0
        while self.cursor < len(self.queue) and steps < max_steps:
            if self.learning is not None or self.status in ("budget", "unresolved"):
                break
            if self.task is None or self.phase == "done":
                self.start(self.queue[self.cursor])
            elif self.task.task_id != self.queue[self.cursor]:
                raise ValueError("current task differs from durable queue cursor")
            self.step()
            steps += 1
        return portable(self.results)

    def learn(self, proposal, admission, min_examples=4):
        """Durably reserve an externally supplied learning transaction.

        An interrupted transaction blocks the runner rather than silently
        reusing its candidate or admission set. The registry's on_reserved
        callback persists all reservations before any candidate execution.
        """
        if (
            self.registry is None
            or self.pending is not None
            or self.learning is not None
            or self.phase not in ("ready", "done")
        ):
            raise ValueError("learning requires an idle runner with a registry")
        natural(min_examples, "min_examples", positive=True)
        proposal, admission = tuple(proposal), tuple(admission)
        previous_phase = self.phase
        self.learning = dict(
            proposal=[e.to_dict() for e in proposal],
            admission=[e.to_dict() for e in admission],
            min_examples=min_examples,
            candidate=None,
            previous_phase=previous_phase,
        )
        self.phase = "learning_intent"
        self._record("learning_intent")
        candidate = self.registry.propose(proposal, self.budget)
        if candidate is None:
            self.learning, self.phase = None, previous_phase
            self._checkpoint()
            return None
        self.learning["candidate"] = candidate.to_dict()
        self.phase = "learning_prepared"
        self._record("learning_prepared")

        def reserved():
            self.phase = "admission_pending"
            self._record("learning_reserved")

        result = self.registry.admit(candidate, admission, self.budget, min_examples=min_examples, on_reserved=reserved)
        self.learning, self.phase = None, previous_phase
        self._checkpoint()
        return result

    def discard_interrupted_learning(self):
        """Discard interrupted learning, retaining spent work and durable data.

        Before reservation no admission program has executed, so those inputs
        remain unused. After reservation registry IDs stay consumed forever.
        Proposal data remain in the learning/discard journal records in both cases.
        """
        if self.learning is None or self.phase not in ("learning_intent", "learning_prepared", "admission_pending"):
            raise ValueError("no interrupted learning transaction to discard")
        self.journal.append("learning_discarded", {"transaction": self.learning})
        previous_phase = self.learning["previous_phase"]
        self.learning, self.phase = None, previous_phase
        self.status = "inprogress" if previous_phase == "ready" else self.results[-1]["status"]
        self._checkpoint()


__all__ = ["ExperienceRunner"]
