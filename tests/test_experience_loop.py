import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.interfaces import TaskSpec, ProbeSpec, Receipt
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.loop import ExperienceRunner


class Host:
    identity = "environment"

    def __init__(self):
        self.calls, self.receipts = [], {}
        self.crash = None

    def task(self, task_id):
        return TaskSpec(
            task_id,
            "scope",
            "mechanism",
            "schema",
            {},
            ({"sensor": 0, "answer": 0}, {"sensor": 1, "answer": 1}),
            (ProbeSpec("sensor", "sensor", 2),),
            action_cost=2,
        )

    def execute(self, task_id, kind, payload, attempt_id):
        if self.crash == "before":
            raise RuntimeError("before effect")
        self.calls.append((kind, attempt_id))
        value = {"field": payload["field"], "value": 1} if kind == "observe" else {"answer": 1, "ok": True}
        r = Receipt("event-" + attempt_id, attempt_id, task_id, kind, self.identity, "mechanism", value)
        self.receipts[attempt_id] = r
        if self.crash == "after":
            raise RuntimeError("after effect")
        return r

    def reconcile(self, attempt_id):
        return self.receipts.get(attempt_id)


def runner(limit=10000):
    host = Host()
    journal = ExperienceJournal(KnowledgeStore(), "cycle", {})
    return ExperienceRunner(host, journal, WorkBudget(limit), mode="fixed")


def test_observe_act_and_next_task():
    r = runner()
    results = r.run(["one", "two"])
    assert [x["status"] for x in results] == ["done", "done"]
    assert len(r.environment.calls) == 4
    assert r.journal.verify()


@pytest.mark.parametrize("crash", ["before", "after"])
def test_restore_never_blindly_repeats_effect(crash):
    r = runner()
    r.start("one")
    snap = r.snapshot()
    r.environment.crash = crash
    with pytest.raises(RuntimeError):
        r.step()
    calls = len(r.environment.calls)
    restored = ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal)
    assert len(r.environment.calls) == calls
    assert restored.budget.spent >= 2
    assert restored.status == ("unresolved" if crash == "before" else "inprogress")
    if crash == "after":
        restored.step()
        assert restored.worlds.decide(lambda w: w["answer"], WorkBudget(10)).value == 1


def test_replay_receipt_is_pure_and_fact_not_duplicated():
    r = runner()
    r.start("one")
    r.step()
    r.step()
    r.step()
    assert r.phase == "receipt"
    snap = r.snapshot()
    calls = len(r.environment.calls)
    state = r.replay()
    assert state["phase"] == "receipt" and len(r.environment.calls) == calls
    r._revise()
    # Model a durable fact with a missing revision record by restoring its receipt
    # is covered separately by the evidence guard; ordinary repeated delivery is pure.
    restored = ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal)
    restored.step()
    assert len(r.environment.calls) == calls
    assert len([x for x in r.journal.store.journal if x.get("op") == "assert"]) == 1


def test_budget_stops_before_external_effect():
    r = runner(1)
    r.start("one")
    assert r.step() == "budget"
    assert r.environment.calls == []


def test_foreign_source_and_modified_snapshot_rejected():
    r = runner()
    r.start("one")
    r.environment.crash = "after"
    snap = r.snapshot()
    with pytest.raises(RuntimeError):
        r.step()
    receipt = next(iter(r.environment.receipts.values()))
    bad = Receipt(
        receipt.event_id, receipt.attempt_id, receipt.task_id, receipt.kind, "forged", receipt.mechanism, receipt.value
    )
    with pytest.raises(ValueError, match="foreign"):
        r._accept(bad)
    snap["state"]["ledger"]["limit"] = 999999
    with pytest.raises(ValueError, match="anchor"):
        ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal)


def test_crash_after_fact_before_revision_deduplicates_assertion(monkeypatch):
    r = runner()
    r.start("one")
    r.step()
    r.step()
    r.step()
    snap = r.snapshot()
    append = r.journal.append

    def fail_revision(kind, payload):
        if kind == "revision":
            raise RuntimeError("lost revision marker")
        return append(kind, payload)

    monkeypatch.setattr(r.journal, "append", fail_revision)
    with pytest.raises(RuntimeError):
        r.step()
    monkeypatch.setattr(r.journal, "append", append)
    restored = ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal)
    assert restored.step() == "done"
    assertions = [x for x in r.journal.store.journal if x.get("op") == "assert"]
    assert len(assertions) == 1
    assert assertions[0]["source"] == "outcome"


@pytest.mark.parametrize(
    "field,value",
    [
        ("source", "forged"),
        ("task_id", "foreign"),
        ("mechanism", "foreign"),
        ("kind", "act"),
        ("attempt_id", "foreign"),
    ],
)
def test_receipt_identity_checks(field, value):
    r = runner()
    r.start("one")
    r.step()
    receipt = dict(r.pending["receipt"])
    receipt[field] = value
    with pytest.raises(ValueError):
        r._accept(Receipt.from_dict(receipt))


def test_conflicting_event_id_rejected():
    r = runner()
    r.start("one")
    r.step()
    receipt = dict(r.pending["receipt"])
    receipt["value"] = {"field": "sensor", "value": 0}
    with pytest.raises(ValueError, match="conflicting"):
        r._accept(Receipt.from_dict(receipt))


@pytest.mark.parametrize("mode", ExperienceRunner.MODES)
def test_resolved_worlds_never_buy_extra_probes(mode):
    r = runner()
    r.config["mode"] = mode
    original = r.environment.task

    def resolved(task_id):
        t = original(task_id)
        return TaskSpec(t.task_id, t.scope, t.mechanism, t.schema, {}, ({"sensor": 1, "answer": 1},), t.probes)

    r.environment.task = resolved
    assert r.run(["resolved"])[0]["ok"] is True
    assert [x[0] for x in r.environment.calls] == ["act"]


def test_false_outcome_preserves_proposal_and_external_answer():
    r = runner()
    original = r.environment.execute

    def wrong(task_id, kind, payload, attempt_id):
        receipt = original(task_id, kind, payload, attempt_id)
        if kind == "act":
            receipt = Receipt(
                receipt.event_id, attempt_id, task_id, kind, "environment", "mechanism", {"answer": 0, "ok": False}
            )
        return receipt

    r.environment.execute = wrong
    result = r.run(["wrong"])[0]
    assert result["status"] == "wrong" and result["ok"] is False
    assert result["answer"] == result["proposed_answer"] == 1
    assert result["expected"] == 0 and result["observed"] == ["sensor"]
    assert result["receipt_event"]


def identity_learning():
    from solvi.experimental.experience.competences import CompetenceRegistry, ExternalExample
    from solvi.experimental.experience.programs import Grammar

    registry = CompetenceRegistry(Grammar(("x",), operators=("add",), max_candidates=10))

    def rows(split, values):
        return [
            ExternalExample(
                f"{split}-{x}", f"t-{split}-{x}", "environment", "mechanism", "schema", "scope", {"x": x}, x, split
            )
            for x in values
        ]

    return registry, rows("proposal", [2, 3]), rows("admission", [4, 5, 6, 7])


def test_interrupted_admission_reservation_survives_restore(monkeypatch):
    registry, proposal, admission = identity_learning()
    r = ExperienceRunner(Host(), ExperienceJournal(KnowledgeStore(), "learn", {}), WorkBudget(10000), registry)
    snap = r.snapshot()
    append = r.journal.append

    def crash(kind, payload):
        result = append(kind, payload)
        if kind == "learning_reserved":
            raise RuntimeError("after reservation")
        return result

    monkeypatch.setattr(r.journal, "append", crash)
    with pytest.raises(RuntimeError):
        r.learn(proposal, admission)
    monkeypatch.setattr(r.journal, "append", append)
    restored = ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal, registry=registry)
    assert restored.status == "unresolved"
    assert restored.phase == "admission_pending"
    assert all(e.event_id in restored.registry.snapshot()["events"] for e in admission)
    restored.discard_interrupted_learning()
    assert all(e.event_id in restored.registry.snapshot()["events"] for e in admission)
    restored.start("independent")


def test_crash_inside_registry_observe_replays_revocation_without_duplicate_fact(monkeypatch):
    registry, proposal, admission = identity_learning()
    b = WorkBudget(10000)
    candidate = registry.propose(proposal, b)
    assert registry.admit(candidate, admission, b).accepted
    host = Host()
    host.task = lambda task_id: TaskSpec(task_id, "scope", "mechanism", "schema", {"x": 9}, ({"answer": 9},), ())
    # Host independently reports answer=1, contradicting the learned identity.
    r = ExperienceRunner(host, ExperienceJournal(KnowledgeStore(), "observe", {}), WorkBudget(10000), registry)
    r.start("novel")
    r.step()
    snap = r.snapshot()
    observe = registry.observe

    def crash(*args, **kwargs):
        observe(*args, **kwargs)
        raise RuntimeError("after registry invalidation")

    monkeypatch.setattr(registry, "observe", crash)
    with pytest.raises(RuntimeError):
        r.step()
    restored = ExperienceRunner.restore(snap, environment=host, journal=r.journal, registry=registry)
    restored.step()
    assert all(v["status"] == "revoked" for v in restored.registry.snapshot()["competences"].values())
    assert len([x for x in r.journal.store.journal if x.get("op") == "assert"]) == 1
    assert len(host.calls) == 1


def test_crash_after_observe_debit_keeps_charge_before_registry_mutation():
    registry, proposal, admission = identity_learning()
    budget = WorkBudget(10000)
    candidate = registry.propose(proposal, budget)
    assert registry.admit(candidate, admission, budget).accepted
    host = Host()
    host.task = lambda task_id: TaskSpec(task_id, "scope", "mechanism", "schema", {"x": 9}, ({"answer": 9},), ())
    r = ExperienceRunner(host, ExperienceJournal(KnowledgeStore(), "debit", {}), WorkBudget(10000), registry)
    r.start("novel")
    r.step()
    snap = r.snapshot()
    sink = r.budget.on_charge

    def crash(ledger):
        sink(ledger)
        if ledger["counts"].get("competence_compare", 0):
            raise RuntimeError("durable debit before mutation")

    r.budget.on_charge = crash
    with pytest.raises(RuntimeError):
        r.step()
    spent = r.budget.spent
    restored = ExperienceRunner.restore(snap, environment=host, journal=r.journal, registry=registry)
    assert restored.budget.spent == spent
    restored.step()
    assert restored.budget.spent > spent
    assert all(v["status"] == "revoked" for v in restored.registry.snapshot()["competences"].values())


@pytest.mark.parametrize("cut", [1, 4, 5, 7])
def test_durable_queue_restore_matches_continuous_run(cut):
    uninterrupted = runner()
    expected = uninterrupted.run(["one", "two", "three"])
    interrupted = runner()
    interrupted.run(["one", "two", "three"], max_steps=cut)
    snap = interrupted.snapshot()
    restored = ExperienceRunner.restore(snap, environment=interrupted.environment, journal=interrupted.journal)
    assert restored.run() == expected
    assert restored.cursor == 3
    assert restored.queue == ["one", "two", "three"]
    assert restored.budget.snapshot() == uninterrupted.budget.snapshot()
    assert restored.environment.calls == uninterrupted.environment.calls
    assert restored.events == uninterrupted.events
    facts = lambda r: [(x["body"], x["evidence"]) for x in r.journal.store.journal if x.get("op") == "assert"]
    assert facts(restored) == facts(uninterrupted)
    assert restored.run() == expected
    assert len(restored.environment.calls) == 6


def test_queue_is_persisted_before_first_step_and_cannot_be_replaced():
    r = runner()
    assert r.run(["one", "two"], max_steps=0) == []
    restored = ExperienceRunner.restore(r.snapshot(), environment=r.environment, journal=r.journal)
    with pytest.raises(ValueError, match="unfinished"):
        restored.run(["other"])
    with pytest.raises(ValueError, match="unique"):
        restored.run(["one", "one"])
    assert len(restored.run()) == 2
    assert len(restored.run(["three"])) == 3
    with pytest.raises(ValueError, match="terminal"):
        restored.run(["one"])


def test_cursor_and_terminal_result_commit_together_on_crash(monkeypatch):
    r = runner()
    snap = r.snapshot()
    checkpoint = r.journal.checkpoint

    def crash_after_commit(state):
        result = checkpoint(state)
        if state["cursor"] == 1:
            raise RuntimeError("crash after terminal checkpoint")
        return result

    monkeypatch.setattr(r.journal, "checkpoint", crash_after_commit)
    with pytest.raises(RuntimeError):
        r.run(["one", "two"])
    monkeypatch.setattr(r.journal, "checkpoint", checkpoint)
    restored = ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal)
    assert restored.cursor == 1 and len(restored.results) == 1
    assert [x["task_id"] for x in restored.run()] == ["one", "two"]
    assert len(restored.environment.calls) == 4


@pytest.mark.parametrize("stage", ["learning_intent", "learning_prepared"])
def test_discard_interrupted_pure_learning_keeps_cost_and_data(stage, monkeypatch):
    registry, proposal, admission = identity_learning()
    r = ExperienceRunner(Host(), ExperienceJournal(KnowledgeStore(), "pure", {}), WorkBudget(10000), registry)
    snap = r.snapshot()
    append = r.journal.append

    def crash(kind, payload):
        result = append(kind, payload)
        if kind == stage:
            raise RuntimeError("pure learning interrupted")
        return result

    monkeypatch.setattr(r.journal, "append", crash)
    with pytest.raises(RuntimeError):
        r.learn(proposal, admission)
    monkeypatch.setattr(r.journal, "append", append)
    restored = ExperienceRunner.restore(snap, environment=r.environment, journal=r.journal, registry=registry)
    spent = restored.budget.spent
    restored.discard_interrupted_learning()
    assert restored.budget.spent == spent
    record = next(e for e in restored.journal.records() if e["kind"] == "learning_discarded")
    assert record["payload"]["transaction"]["proposal"] == [e.to_dict() for e in proposal]
    assert all(e.event_id not in restored.registry.snapshot()["events"] for e in admission)
    restored.start("independent")


def test_unknown_effect_without_reconcile_blocks_automatic_queue_resume():
    r = runner()
    execute = r.environment.execute

    def unknown(*args):
        execute(*args)
        return None

    r.environment.execute = unknown
    r.environment.reconcile = None
    assert r.run(["one", "two"]) == []
    assert r.status == "unresolved"
    restored = ExperienceRunner.restore(r.snapshot(), environment=r.environment, journal=r.journal)
    assert restored.run() == []
    assert restored.status == "unresolved" and restored.cursor == 0
    assert len(restored.environment.calls) == 1


def test_restore_after_real_store_retraction_before_registry_checkpoint(monkeypatch):
    registry, proposal, admission = identity_learning()
    store = KnowledgeStore()
    store.gates = []  # Explicit host policy for this bridge/recovery test.
    original_add = store.add

    def admit_item(*args, **kwargs):
        item = original_add(*args, **kwargs)
        if item is not None:
            store.promote(item)
        return item

    monkeypatch.setattr(store, "add", admit_item)
    registry.store = store
    preparation = WorkBudget(10000)
    candidate = registry.propose(proposal, preparation)
    admitted = registry.admit(candidate, admission, preparation)
    assert admitted.accepted
    host = Host()
    host.task = lambda tid: TaskSpec(tid, "scope", "mechanism", "schema", {"x": 9}, ({"answer": 9},), ())
    execute = host.execute

    def truthful_failure(task_id, kind, payload, attempt_id):
        receipt = execute(task_id, kind, payload, attempt_id)
        if kind == "act":
            receipt = Receipt(
                receipt.event_id, attempt_id, task_id, kind, host.identity, "mechanism", {"answer": 1, "ok": False}
            )
            host.receipts[attempt_id] = receipt
        return receipt

    host.execute = truthful_failure
    r = ExperienceRunner(host, ExperienceJournal(store, "real-retraction", {}), WorkBudget(10000), registry)
    r.start("new-external-task")
    r.step()
    snap = r.snapshot()
    retract = store.retract

    def committed_then_crashed(*args, **kwargs):
        retract(*args, **kwargs)
        raise RuntimeError("after durable KnowledgeStore retraction")

    monkeypatch.setattr(store, "retract", committed_then_crashed)
    with pytest.raises(RuntimeError, match="durable KnowledgeStore"):
        r.step()
    monkeypatch.setattr(store, "retract", retract)
    calls = list(host.calls)
    restored = ExperienceRunner.restore(snap, environment=host, journal=r.journal, registry=registry)
    assert restored.step() == "wrong"
    assert restored.registry.snapshot()["competences"][admitted.competence_id]["status"] == "revoked"
    assert host.calls == calls and len(restored.results) == 1
    facts = [item for item in store.items.values() if item.kind == "fact"]
    assert len(facts) == 1 and len(facts[0].asserts) == 1
    assert store.verify()
