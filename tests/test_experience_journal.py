"""Experience notes: persistence, namespaces and fail-closed replay."""

import copy

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.journal import ExperienceJournal


@pytest.mark.parametrize("persistent", [False, True])
def test_roundtrip_and_detached_values(tmp_path, persistent):
    where = str(tmp_path / "experience.db") if persistent else None
    store = KnowledgeStore(where)
    config = {"world": "v1", "limit": 12}
    journal = ExperienceJournal(store, "a", config)
    payload = {"items": [1, True]}
    event = journal.append("observe", payload)
    payload["items"].append(8)
    event["payload"]["items"].append(9)
    state = {"ledger": {"spent": 1}, "worlds": [1], "registry": {}, "phase": "idle"}
    journal.checkpoint(state)
    state["worlds"].append(2)
    checkpoint = journal.latest()
    checkpoint["state"]["worlds"].append(3)
    records = journal.records()
    records[0]["payload"]["items"].append(4)
    reopened_store = KnowledgeStore(where) if persistent else store
    reopened = ExperienceJournal(reopened_store, "a", config)
    assert reopened.records()[0]["payload"] == {"items": [1, True]}
    assert reopened.latest()["state"]["worlds"] == [1]
    assert reopened.verify()
    with pytest.raises(ValueError, match="configuration"):
        ExperienceJournal(reopened_store, "a", {"world": "v2"})


def test_cycles_and_facts_interleave():
    store = KnowledgeStore()
    a = ExperienceJournal(store, "a", {})
    b = ExperienceJournal(store, "b", {"other": True})
    assert a.latest() is None
    a.append("observation", {"kind": "nested", "seq": 100})
    store.add("fact", {"s": "x", "r": "value", "o": 1}, source="spec")
    b.append("observation", {})
    a.checkpoint({"phase": "idle"})
    b.append("done", {})
    assert [e["seq"] for e in a.records()] == [0, 1]
    assert [e["seq"] for e in b.records()] == [0, 1]
    assert a.latest()["cursor"] == 3
    assert a.latest()["anchor"] == store.journal[2]["hash"]


@pytest.mark.parametrize("bad", [{1: "bad"}, {"bad": 1.5}, {"bad": (1, 2)}, {"bad": object()}])
def test_unsupported_payload_is_atomic(bad):
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    with pytest.raises(ValueError):
        journal.append("x", bad)
    with pytest.raises(ValueError):
        journal.checkpoint(bad)
    assert len(store.journal) == 0


@pytest.mark.parametrize(
    "change",
    [
        lambda e: e.update(version=2),
        lambda e: e.update(version=True),
        lambda e: e.update(seq=2),
        lambda e: e.update(seq=True),
        lambda e: e.update(unexpected=1),
        lambda e: e.update(payload=[]),
        lambda e: e.update(config="bad"),
    ],
)
def test_valid_chain_malformed_experience_rejected(change):
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    event = journal.append("x", {})
    event["seq"] = 1
    change(event)
    store.note(experience=event)
    assert store.verify()
    with pytest.raises(ValueError):
        journal.append("y", {})
    with pytest.raises(ValueError):
        ExperienceJournal(store, "a", {})


def test_duplicate_event_rejected():
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    event = journal.append("x", {})
    store.note(experience=event)
    with pytest.raises(ValueError):
        journal.records()


@pytest.mark.parametrize("field,value", [("version", 2), ("anchor", "bad"), ("cursor", 99), ("digest", "bad")])
def test_checkpoint_validation_even_on_valid_chain(field, value):
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    checkpoint = journal.checkpoint({"phase": "idle"})
    forged = KnowledgeStore()
    checkpoint["payload"][field] = value
    forged.note(experience=checkpoint)
    assert forged.verify()
    with pytest.raises(ValueError):
        ExperienceJournal(forged, "a", {})


def test_in_memory_tamper_and_truncation_rejected_before_append():
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    journal.append("x", {"value": 1})
    saved = copy.deepcopy(store.journal[0])
    store.journal[0]["experience"]["payload"]["value"] = 2
    with pytest.raises(ValueError):
        journal.append("y", {})
    store.journal[0] = saved
    store.journal.pop()
    with pytest.raises(ValueError, match="truncated"):
        journal.append("y", {})


def test_checkpoint_kind_reserved():
    journal = ExperienceJournal(KnowledgeStore(), "a", {})
    with pytest.raises(ValueError):
        journal.append("checkpoint", {})
    with pytest.raises(ValueError):
        journal.checkpoint({"callback": lambda: None})


@pytest.mark.parametrize("persistent", [False, True])
def test_fast_debits_checkpoint_reopen_and_detached_values(tmp_path, persistent, monkeypatch):
    where = str(tmp_path / "debits.db") if persistent else None
    store = KnowledgeStore(where)
    journal = ExperienceJournal(store, "a", {})
    ledger = WorkBudget(100)
    # A regression guard for the actual performance property: no replay on
    # each pure debit, but checkpoint must replay the previously unchecked tail.
    original_verify = store.verify
    calls = []

    def counted_verify():
        calls.append(True)
        return original_verify()

    monkeypatch.setattr(store, "verify", counted_verify)
    for _ in range(100):
        ledger.charge("node")
        snapshot = ledger.snapshot()
        event = journal.debit(snapshot)
        snapshot["counts"]["node"] = -1
        event["payload"]["ledger"]["counts"]["node"] = -2
    assert calls == []
    journal.checkpoint({"ledger": ledger.snapshot(), "phase": "idle"})
    assert len(calls) == 1
    reopened = ExperienceJournal(KnowledgeStore(where) if persistent else store, "a", {})
    records = reopened.records()
    assert [e["seq"] for e in records] == list(range(101))
    assert [e["payload"]["ledger"]["spent"] for e in records[:-1]] == list(range(1, 101))
    assert reopened.latest()["state"]["ledger"] == ledger.snapshot()


def test_fast_suffix_tamper_blocks_next_intent():
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    ledger = WorkBudget(10)
    for _ in range(3):
        ledger.charge("node")
        journal.debit(ledger.snapshot())
    store.journal[0]["experience"]["payload"]["ledger"]["spent"] = 8
    # Fast debit deliberately cannot see old payload edits with unchanged head.
    ledger.charge("node")
    journal.debit(ledger.snapshot())
    count = len(store.journal)
    with pytest.raises(ValueError):
        journal.append("action_intent", {"attempt": "x"})
    assert len(store.journal) == count


def test_old_verified_prefix_tamper_blocks_checkpoint_after_fast_debits():
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    journal.append("initial", {"value": 1})
    ledger = WorkBudget(10)
    ledger.charge("node")
    journal.debit(ledger.snapshot())
    store.journal[0]["experience"]["payload"]["value"] = 2
    with pytest.raises(ValueError):
        journal.checkpoint({"ledger": ledger.snapshot()})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda s: s.update(digest="bad"),
        lambda s: s.update(v=2),
        lambda s: s.update(spent=True),
        lambda s: s.update(extra=1),
    ],
)
def test_invalid_fast_ledger_is_atomic(mutation):
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    snapshot = WorkBudget(10).snapshot()
    mutation(snapshot)
    with pytest.raises(ValueError):
        journal.debit(snapshot)
    assert store.journal == []


def test_external_fact_growth_forces_full_validation_before_fast_debit(monkeypatch):
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    ledger = WorkBudget(10)
    ledger.charge("node")
    journal.debit(ledger.snapshot())
    store.add("fact", {"s": "x", "r": "value", "o": 1}, source="spec")
    original_verify = store.verify
    calls = []

    def counted_verify():
        calls.append(True)
        return original_verify()

    monkeypatch.setattr(store, "verify", counted_verify)
    ledger.charge("node")
    journal.debit(ledger.snapshot())
    assert calls == [True]
    assert [e["seq"] for e in journal.records()] == [0, 1]


def test_fast_suffix_truncation_is_rejected():
    store = KnowledgeStore()
    journal = ExperienceJournal(store, "a", {})
    journal.debit(WorkBudget(10).snapshot())
    store.journal.pop()
    with pytest.raises(ValueError, match="truncated"):
        journal.debit(WorkBudget(10).snapshot())
    with pytest.raises(ValueError, match="truncated"):
        journal.append("action_intent", {})
