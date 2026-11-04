import copy

import pytest

from solvi.core.knowledge import KnowledgeStore


def fact(value):
    return {"s": "place", "r": "leader", "o": value}


def test_rebuild_is_independent_writable_chain_after_prefix_and_skip(tmp_path):
    source = KnowledgeStore(tmp_path / "source.jsonl")
    a = source.add("fact", fact("a"), source="outcome", by="world")
    source.add("fact", fact("b"), source="outcome", by="world")
    original = copy.deepcopy(source.journal), source.fingerprint(), source.head()
    for rebuilt in (source.rebuild(), source.rebuild(upto=1), source.rebuild(skip={a})):
        prior_n = rebuilt.journal[-1]["n"]
        c = rebuilt.add("fact", fact("c"), source="outcome", by="world")
        assert rebuilt.journal[-1]["n"] == prior_n + 1
        assert rebuilt.usable(c)
        assert rebuilt.supporters(c)
        assert any(row["id"] == c for row in rebuilt.snapshot()["facts"])
        assert rebuilt.verify()
    assert (source.journal, source.fingerprint(), source.head()) == original
    rebuilt.journal[0]["body"]["o"] = "mutated"
    assert source.journal == original[0]


def test_rebuild_preserves_clock_writer_gate_and_reassert_capability():
    source = KnowledgeStore(writers={"world": "outcome"}, reassert_writers={"world"}, now=15)
    iid = source.add("fact", fact("a"), source="outcome", by="world", valid=(10, 20))
    source.retract(iid, by="world", assertion=source.journal[-1]["n"])
    rebuilt = source.rebuild()
    assert rebuilt.now == 15 and rebuilt.reassert_writers == frozenset({"world"})
    assert rebuilt.add("fact", fact("a"), source="outcome", by="world", valid=(10, 20), reassert=True) == iid
    assert rebuilt.usable(iid)
    rebuilt.now = 21
    assert not rebuilt.usable(iid) and source.now == 15
    assert rebuilt.add("fact", fact("intruder"), source="outcome", by="intruder") is None
    gate = next(g for g in rebuilt.gates if g.name == "writers")
    gate.writers.clear()
    assert next(g for g in source.gates if g.name == "writers").writers
    assert source.status(iid) == "retracted"
    assert rebuilt.verify()


def test_rebuild_refuses_uncopyable_gate_before_source_changes():
    class Gate:
        def admit(self, store, item, shadow):
            raise AssertionError("not called")

        def __deepcopy__(self, memo):
            return self

    source = KnowledgeStore(gates=[Gate()])
    before = copy.deepcopy(source.journal), source.head()
    with pytest.raises(TypeError, match="independently copyable"):
        source.rebuild()
    assert (source.journal, source.head()) == before


def test_snapshot_prefix_count_after_skip_excludes_future_writes():
    source = KnowledgeStore()
    ids = [source.add("fact", {"s": str(i), "r": "v", "o": i}, source="outcome") for i in range(10)]
    rebuilt = source.rebuild(skip=set(ids[:8]))
    snapshot = rebuilt.snapshot()
    assert snapshot["at"] == len(rebuilt.journal) == 2
    assert rebuilt.journal[-1]["n"] == 9
    for i in range(10, 14):
        rebuilt.add("fact", {"s": str(i), "r": "v", "o": i}, source="outcome")
    historical = rebuilt.rebuild(upto=snapshot["at"])
    assert historical.snapshot()["facts"] == snapshot["facts"]
    assert historical.fingerprint() == snapshot["fp"]
    assert len(historical.items) == 2


@pytest.mark.parametrize("persistent", [False, True])
def test_verify_does_not_require_cloneable_write_policy(tmp_path, persistent):
    import threading
    from solvi.core.knowledge.gates import Verdict

    class LockedGate:
        def __init__(self):
            self.lock = threading.RLock()
            self.calls = 0

        def admit(self, store, item, shadow):
            with self.lock:
                self.calls += 1
                return Verdict(True)

    gate = LockedGate()
    source = KnowledgeStore(tmp_path / "parent.jsonl" if persistent else None, gates=[gate],
                            writers={"world": "outcome"}, reassert_writers={"world"})
    iid = source.add("fact", fact("a"), source="outcome", by="world", valid=(10, 20))
    source.retract(iid, by="world", assertion=source.journal[-1]["n"])
    source.add("fact", fact("a"), source="outcome", by="world", valid=(10, 20), reassert=True)
    before, calls = copy.deepcopy(source.journal), gate.calls
    assert source.verify()
    assert source.journal == before and gate.calls == calls
    destination = tmp_path / "branch.jsonl"
    for operation in (lambda: source.rebuild(), lambda: source.branch(destination)):
        with pytest.raises(TypeError, match="independently copyable"):
            operation()
    assert not destination.exists() and source.journal == before and gate.calls == calls
    source.journal = copy.deepcopy(source.journal)
    source.journal[0]["body"]["o"] = "tampered"
    assert not source.verify()
    assert gate.calls == calls
