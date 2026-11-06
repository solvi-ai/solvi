"""Finite host curriculum: future roles, external labels and bounded learning.

Controller records share ExperienceJournal but never use runner checkpoints or
its reserved ``state`` key. An interrupted learning packet is burned, not retried.
An ambiguous environment effect still requires the host's reconcile protocol.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal, NotRequired, TypedDict, cast

from ._common import BudgetStop, digest, identifier, natural, portable
from .competences import ExternalExample
from .interfaces import Receipt, TaskSpec
from .loop import ExperienceRunner


class _Configuration(TypedDict):
    tasks: list[str]
    proposal_quota: int
    admission_quota: int
    max_attempts: int


class _Packet(TypedDict):
    proposal: list[dict[str, Any]]
    admission: list[dict[str, Any]]
    transfer: bool
    generation: int


class _Assignment(TypedDict):
    task: dict[str, Any]
    context: str
    generation: int
    role: str
    used_as: NotRequired[str]


class _Learning(TypedDict):
    context: str
    boundary: int
    generation: NotRequired[int]


class _State(TypedDict):
    version: int
    config: _Configuration
    config_hash: str
    cursor: int
    assignments: dict[str, _Assignment]
    packets: dict[str, _Packet]
    attempts: int
    learning: _Learning | None
    status: str
    seen: list[str]
    contexts: dict[str, str]


class AutoCurriculum:
    def __init__(
        self,
        runner: ExperienceRunner,
        task_ids: Iterable[str],
        *,
        proposal_quota: int = 4,
        admission_quota: int = 4,
        max_attempts: int = 4,
    ):
        if runner.registry is None or runner.task is not None or runner.queue:
            raise ValueError("curriculum requires an idle fresh runner with registry")
        tasks = list(task_ids)
        for task in tasks:
            identifier(task, "task ID")
        if len(tasks) != len(set(tasks)):
            raise ValueError("curriculum task IDs must be unique")
        for quota in (proposal_quota, admission_quota):
            natural(quota, "quota", positive=True)
            if quota < 4:
                raise ValueError("proposal and admission quotas must each be at least four")
        natural(max_attempts, "max_attempts")
        if any(e["kind"] == "curriculum" for e in runner.journal.records()):
            raise ValueError("existing curriculum requires restore")
        self.runner: ExperienceRunner = runner
        self.config: _Configuration = _Configuration(
            tasks=tasks, proposal_quota=proposal_quota, admission_quota=admission_quota, max_attempts=max_attempts
        )
        self.state: _State = _State(
            version=1,
            config=self.config,
            config_hash=digest(self.config),
            cursor=0,
            assignments={},
            packets={},
            attempts=0,
            learning=None,
            status="inprogress",
            seen=[],
            contexts={},
        )
        self._save()

    def _save(self) -> None:
        self.runner.journal.append("curriculum", {"controller_state": self.state})

    def snapshot(self) -> dict[str, Any]:
        return dict(runner=self.runner.snapshot(), controller=portable(self.state))

    @classmethod
    def restore(cls, snapshot, *, environment, journal, registry, checked_domains=None,
                checked_store_policy="deny"):
        snapshot = portable(snapshot)
        records = journal.records()
        states = [e["payload"]["controller_state"] for e in records if e["kind"] == "curriculum"]
        if snapshot["controller"] not in states:
            raise ValueError("controller snapshot has no durable anchor")
        state = states[-1]
        if (
            state["version"] != 1
            or digest(state["config"]) != state["config_hash"]
            or state["config"] != snapshot["controller"]["config"]
        ):
            raise ValueError("controller config or queue mismatch")
        obj = object.__new__(cls)
        obj.runner = ExperienceRunner.restore(
            snapshot["runner"], environment=environment, journal=journal, registry=registry,
            checked_domains=checked_domains, checked_store_policy=checked_store_policy,
        )
        obj.state, obj.config = portable(state), portable(state["config"])
        return obj

    def _packet(self, context: str) -> _Packet:
        return self.state["packets"].setdefault(
            context, _Packet(proposal=[], admission=[], transfer=False, generation=0)
        )

    def _burn(self, context: str) -> None:
        packet = self._packet(context)
        self.state["packets"][context] = _Packet(
            proposal=[], admission=[], transfer=False, generation=packet["generation"] + 1
        )

    def _assign(self, task_id: str) -> None:
        self.runner.budget.charge("curriculum_select")
        task = self.runner.environment.task(task_id)
        if not isinstance(task, TaskSpec) or task.task_id != task_id:
            raise ValueError("host returned another curriculum task")
        context = digest([task.scope, task.mechanism, task.schema])
        previous = self.state["contexts"].get(task.scope)
        if previous is not None and previous != context:
            self._burn(context)
        self.state["contexts"][task.scope] = context
        packet = self._packet(context)
        role = "evaluation"
        if not packet["transfer"] and self.state["attempts"] < self.config["max_attempts"]:
            role = "proposal" if len(packet["proposal"]) < self.config["proposal_quota"] else "admission"
        self.state["assignments"][task_id] = _Assignment(
            task=task.to_dict(), context=context, generation=packet["generation"], role=role
        )
        # This write precedes start, predict, observations and actions.
        self._save()

    def _collect(self, task_id: str, result: dict[str, Any]) -> None:
        self.runner.budget.charge("curriculum_collect")
        assignment = self.state["assignments"][task_id]
        task = TaskSpec.from_dict(assignment["task"])
        event_id = result["receipt_event"]
        packet = self._packet(assignment["context"])
        if event_id is not None and event_id not in self.state["seen"]:
            receipt = Receipt.from_dict(self.runner.events[event_id])
            if (
                receipt.kind != "act"
                or receipt.source != self.runner.environment.identity
                or receipt.task_id != task_id
                or receipt.mechanism != task.mechanism
            ):
                raise ValueError("foreign curriculum evidence")
            role = assignment["role"]
            # Registry calls already consume these identities as evaluation.
            registry_state = self.runner.registry.snapshot()
            consumed = event_id in registry_state["events"] or task_id in registry_state["tasks"]
            for application in registry_state["applications"].values():
                self.runner.budget.charge("curriculum_application_scan")
                consumed = consumed or application["task_id"] == task_id
            if consumed or result["competence_id"] is not None:
                role = "evaluation"
            assignment["used_as"] = role
            if role in ("proposal", "admission"):
                example = ExternalExample(
                    event_id,
                    task_id,
                    receipt.source,
                    task.mechanism,
                    task.schema,
                    task.scope,
                    task.inputs,
                    receipt.value["answer"],
                    role,
                )
                packet[cast(Literal["proposal", "admission"], role)].append(example.to_dict())
            self.state["seen"].append(event_id)
            lost_transfer = packet["transfer"] and result["competence_id"] is None
            if role == "evaluation" and (
                lost_transfer
                or not receipt.value["ok"]
                or type(result["proposed_answer"]) is not type(receipt.value["answer"])
                or result["proposed_answer"] != receipt.value["answer"]
            ):
                self._burn(assignment["context"])
        self.state["cursor"] += 1
        self._save()

    def _learn(self, context: str) -> None:
        packet = self._packet(context)
        if self.state["learning"] is None:
            self.runner.budget.charge("curriculum_learning")
            self.state["attempts"] += 1
            boundary = len(self.runner.journal.records())
            self.state["learning"] = _Learning(context=context, boundary=boundary, generation=packet["generation"])
            self._save()
        result = self.runner.learn(
            [ExternalExample.from_dict(e) for e in packet["proposal"]],
            [ExternalExample.from_dict(e) for e in packet["admission"]],
            min_examples=self.config["admission_quota"],
        )
        if result is not None and result.accepted:
            packet["transfer"] = True
        else:
            self._burn(context)
        self.state["learning"] = None
        self._save()

    def _recover_learning(self) -> None:
        pending = self.state["learning"]
        if pending is None:
            return
        records = self.runner.journal.records()
        started = False
        committed = None
        for event in records[pending["boundary"] :]:
            self.runner.budget.charge("curriculum_scan")
            started = started or event["kind"] == "learning_intent"
            if event["kind"] == "learning_committed":
                committed = event["payload"]["state"]["learning"]
        if committed is not None:
            from .competences import Candidate

            packet = self._packet(pending["context"])
            if (pending.get("generation") != packet["generation"]
                    or committed["proposal"] != packet["proposal"]
                    or committed["admission"] != packet["admission"]):
                raise ValueError("committed learning belongs to another curriculum packet")
            cid = Candidate.from_dict(committed["candidate"]).fingerprint
            result = self.runner.registry.snapshot()["admissions"][cid]["result"]
            if result["accepted"] is not True or result["competence_id"] != cid:
                raise ValueError("durable checked admission is missing")
            if self.runner.learning is not None:
                self.runner.finish_committed_learning()
            packet["transfer"] = True
            self.state["learning"] = None
            self._save()
            return
        if not started:
            # The controller intent preceded all paid Runner.learn operations.
            self._learn(pending["context"])
            return
        if self.runner.learning is not None:
            self.runner.discard_interrupted_learning()
        # Completed or interrupted: never repeat the paid packet. A completed
        # admission remains in the registry; fresh tasks may use it immediately.
        self._burn(pending["context"])
        self.runner.journal.append("curriculum_burn", {"context": pending["context"], "reason": "recovery_no_retry"})
        self.state["learning"] = None
        self._save()

    def step(self) -> str:
        if self.runner._debit_failed:
            raise ValueError("durable debit failed; restore before continuing")
        if self.state["status"] != "inprogress":
            return self.state["status"]
        try:
            self.runner.budget.charge("curriculum_step")
            self._recover_learning()
            for context, packet in list(self.state["packets"].items()):
                self.runner.budget.charge("curriculum_packet_scan")
                if (
                    not packet["transfer"]
                    and self.state["attempts"] < self.config["max_attempts"]
                    and len(packet["proposal"]) >= self.config["proposal_quota"]
                    and len(packet["admission"]) >= self.config["admission_quota"]
                ):
                    self._learn(context)
                    break
            if self.state["cursor"] == len(self.config["tasks"]):
                self.state["status"] = "done"
                self._save()
                return "done"
            task_id = self.config["tasks"][self.state["cursor"]]
            if task_id not in self.state["assignments"]:
                self._assign(task_id)
            if self.runner.task is None or self.runner.task.task_id != task_id:
                self.runner.start(task_id)
            self.runner.budget.charge("curriculum_task_validate")
            if cast(TaskSpec, self.runner.task).to_dict() != self.state["assignments"][task_id]["task"]:
                raise ValueError("host task changed after curriculum role assignment")
            if self.runner.phase != "done":
                self.runner.step()
            if self.runner.phase == "done":
                self._collect(task_id, self.runner.results[-1])
                context = self.state["assignments"][task_id]["context"]
                packet = self._packet(context)
                if (
                    not packet["transfer"]
                    and self.state["attempts"] < self.config["max_attempts"]
                    and len(packet["proposal"]) >= self.config["proposal_quota"]
                    and len(packet["admission"]) >= self.config["admission_quota"]
                ):
                    self._learn(context)
            elif self.runner.status in ("budget", "unresolved"):
                return self.runner.status
            return self.state["status"]
        except BudgetStop:
            if self.runner._debit_failed:
                raise
            self.state["status"] = "budget"
            self._save()
            return "budget"

    def run(self, max_steps: int = 1000) -> list[dict[str, Any]]:
        natural(max_steps, "max_steps")
        for _ in range(max_steps):
            if self.step() != "inprogress":
                break
        return portable(self.runner.results)


__all__ = ["AutoCurriculum"]
