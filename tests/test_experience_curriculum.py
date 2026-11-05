import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.competences import CompetenceRegistry
from solvi.experimental.experience.curriculum import AutoCurriculum
from solvi.experimental.experience.interfaces import Receipt, TaskSpec
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.loop import ExperienceRunner
from solvi.experimental.experience.programs import Grammar


class Host:
    identity = "environment"

    def __init__(self):
        self.receipts = {}
        self.calls = []

    def task(self, task_id):
        x = int(task_id)
        return TaskSpec(task_id, "domain", "m1", "s1", {"x": x}, ({"answer": x},), ())

    def execute(self, task_id, kind, payload, attempt_id):
        assert kind == "act"
        assert attempt_id not in self.receipts
        self.calls.append(task_id)
        receipt = Receipt(
            "e-" + task_id,
            attempt_id,
            task_id,
            kind,
            self.identity,
            "m1",
            {"answer": int(task_id), "ok": payload["answer"] == int(task_id)},
        )
        self.receipts[attempt_id] = receipt
        return receipt

    def reconcile(self, attempt_id):
        return self.receipts.get(attempt_id)


def make(n=10):
    registry = CompetenceRegistry(Grammar(("x",), operators=("add",), max_candidates=10))
    runner = ExperienceRunner(Host(), ExperienceJournal(KnowledgeStore(), "cycle", {}), WorkBudget(100000), registry)
    return AutoCurriculum(runner, [str(i) for i in range(n)])


def test_collect_external_train_admit_and_transfer_automatically():
    controller = make()
    results = controller.run()
    assert len(results) == 10 and controller.state["status"] == "done"
    assert controller.state["attempts"] == 1
    assignments = controller.state["assignments"]
    assert [assignments[str(i)]["role"] for i in range(10)] == ["proposal"] * 4 + ["admission"] * 4 + ["evaluation"] * 2
    assert all(result["competence_id"] is not None for result in results[8:])
    registry = controller.runner.registry.snapshot()
    assert all(registry["events"]["e-" + str(i)]["split"] == "evaluation" for i in (8, 9))


def test_restore_mid_task_does_not_duplicate_effect_or_example():
    controller = make()
    controller.run(max_steps=5)
    snapshot = controller.snapshot()
    resumed = AutoCurriculum.restore(
        snapshot,
        environment=controller.runner.environment,
        journal=controller.runner.journal,
        registry=controller.runner.registry,
    )
    assert len(resumed.run()) == 10
    assert len(resumed.runner.environment.calls) == 10
    assert len(resumed.state["seen"]) == 10
    assert resumed.state["attempts"] == 1


def test_interrupted_learning_is_burned_without_paid_retry(monkeypatch):
    controller = make(16)
    snapshot = controller.snapshot()
    append = controller.runner.journal.append

    def crash(kind, payload):
        result = append(kind, payload)
        if kind == "learning_prepared":
            raise RuntimeError("learning interrupted")
        return result

    monkeypatch.setattr(controller.runner.journal, "append", crash)
    with pytest.raises(RuntimeError):
        controller.run()
    monkeypatch.setattr(controller.runner.journal, "append", append)
    spent = controller.runner.budget.spent
    resumed = AutoCurriculum.restore(
        snapshot,
        environment=controller.runner.environment,
        journal=controller.runner.journal,
        registry=controller.runner.registry,
    )
    resumed.run()
    assert resumed.state["attempts"] == 2
    assert resumed.runner.budget.spent >= spent
    assert len(resumed.runner.environment.calls) == 16
    proposals = [e for e in resumed.runner.journal.records() if e["kind"] == "learning_intent"]
    assert len(proposals) == 2
    first = {e["task_id"] for e in proposals[0]["payload"]["state"]["learning"]["proposal"]}
    second = {e["task_id"] for e in proposals[1]["payload"]["state"]["learning"]["proposal"]}
    assert first.isdisjoint(second)


def test_final_training_receipt_saved_before_learn_recovers_ready_packet(monkeypatch):
    controller = make(8)
    snap = controller.snapshot()
    save = controller._save

    def crash_after_collect():
        save()
        if controller.state["cursor"] == 8 and controller.state["attempts"] == 0:
            raise RuntimeError("receipt collected before learning")

    monkeypatch.setattr(controller, "_save", crash_after_collect)
    with pytest.raises(RuntimeError):
        controller.run()
    resumed = AutoCurriculum.restore(
        snap,
        environment=controller.runner.environment,
        journal=controller.runner.journal,
        registry=controller.runner.registry,
    )
    resumed.run()
    assert resumed.state["status"] == "done" and resumed.state["attempts"] == 1
    assert len(resumed.runner.registry.snapshot()["competences"]) == 1
    assert len(resumed.runner.environment.calls) == 8


def test_completed_learning_before_controller_save_is_never_repeated(monkeypatch):
    controller = make()
    snap = controller.snapshot()
    append = controller.runner.journal.append

    def crash_before_controller_commit(kind, payload):
        state = payload.get("controller_state")
        if (
            state
            and state["attempts"] == 1
            and state["learning"] is None
            and any(p["transfer"] for p in state["packets"].values())
        ):
            raise RuntimeError("learn completed before controller commit")
        return append(kind, payload)

    monkeypatch.setattr(controller.runner.journal, "append", crash_before_controller_commit)
    with pytest.raises(RuntimeError):
        controller.run()
    monkeypatch.setattr(controller.runner.journal, "append", append)
    resumed = AutoCurriculum.restore(
        snap,
        environment=controller.runner.environment,
        journal=controller.runner.journal,
        registry=controller.runner.registry,
    )
    resumed.run()
    assert resumed.state["attempts"] == 1
    assert len([e for e in resumed.runner.journal.records() if e["kind"] == "learning_intent"]) == 1
    assert len(resumed.runner.environment.calls) == 10


def test_interleaved_scopes_keep_independent_future_roles():
    controller = make(20)
    host = controller.runner.environment
    original = host.task

    def task(task_id):
        task = original(task_id)
        return TaskSpec(
            task_id,
            "odd" if int(task_id) % 2 else "even",
            task.mechanism,
            task.schema,
            task.inputs,
            task.worlds,
            task.probes,
        )

    host.task = task
    controller.run()
    assert controller.state["attempts"] == 2
    for parity in (0, 1):
        roles = [controller.state["assignments"][str(i)]["role"] for i in range(parity, 20, 2)]
        assert roles == ["proposal"] * 4 + ["admission"] * 4 + ["evaluation"] * 2
    assert len(controller.runner.environment.calls) == 20


def test_bookkeeping_budget_stop_has_no_external_effect():
    controller = make()
    controller.runner.budget.charge("host_reserve", controller.runner.budget.remaining)
    assert controller.run() == []
    assert controller.state["status"] == "budget"
    assert controller.runner.environment.calls == []


def test_counterexample_burns_old_packet_and_uses_only_future_tasks():
    controller = make(17)
    host = controller.runner.environment
    original = host.task

    def task(task_id):
        task = original(task_id)
        x = int(task_id)
        return TaskSpec(
            task_id, task.scope, task.mechanism, task.schema, task.inputs, ({"answer": x if x < 8 else x + 1},), ()
        )

    def execute(task_id, kind, payload, attempt_id):
        x = int(task_id)
        expected = x if x < 8 else x + 1
        host.calls.append(task_id)
        receipt = Receipt(
            "e-" + task_id,
            attempt_id,
            task_id,
            kind,
            host.identity,
            "m1",
            {"answer": expected, "ok": payload["answer"] == expected},
        )
        host.receipts[attempt_id] = receipt
        return receipt

    host.task, host.execute = task, execute
    controller.run()
    assert controller.runner.results[8]["status"] == "wrong"
    assert controller.state["assignments"]["8"]["used_as"] == "evaluation"
    assert [controller.state["assignments"][str(i)]["role"] for i in range(9, 17)] == ["proposal"] * 4 + [
        "admission"
    ] * 4
    assert controller.state["attempts"] == 2
    for packet in controller.state["packets"].values():
        assert all(int(e["task_id"]) > 8 for e in packet["proposal"] + packet["admission"])


def test_store_retraction_restarts_learning_only_from_future_tasks():
    controller = make(17)
    registry = controller.runner.registry
    registry.store = controller.runner.journal.store
    controller.run(max_steps=16)
    assert controller.state["cursor"] == 8 and controller.state["attempts"] == 1
    competence = next(iter(registry.snapshot()["competences"].values()))
    assert competence["store_id"] is not None
    registry.store.retract(competence["store_id"], why="external withdrawal")
    controller.run()
    assert controller.state["assignments"]["8"]["used_as"] == "evaluation"
    assert controller.runner.results[8]["competence_id"] is None
    assert controller.runner.results[8]["ok"] is True
    assert [controller.state["assignments"][str(i)]["role"] for i in range(9, 17)] == ["proposal"] * 4 + [
        "admission"
    ] * 4
    assert controller.state["attempts"] == 2
