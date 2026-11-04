"""A validated independent memory prefix keeps its valid chain without hashing it twice."""

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.core.knowledge.store import _MemoryJournal


def populated(storage=None):
    st = KnowledgeStore(storage)
    for n in range(12):
        st.add("fact", {"s": str(n), "r": "v", "o": n}, source="outcome")
    return st


def test_validated_memory_prefix_never_appends_its_records_again(monkeypatch):
    st = populated()
    original = _MemoryJournal.append

    def forbidden(self, body):
        raise AssertionError("verified copied memory prefix need not be rehashed")

    monkeypatch.setattr(_MemoryJournal, "append", forbidden)
    child = st.rebuild(upto=7)
    assert child.journal == st.journal[:7] and child._mem.verify()
    monkeypatch.setattr(_MemoryJournal, "append", original)
    child.note(own="child")
    assert len(st.journal) == 12 and child.verify() and st.verify()


def test_copied_chain_and_policy_are_independent():
    st = populated()
    child = st.rebuild()
    assert child.journal is not st.journal
    assert child.journal[0] is not st.journal[0]
    assert child.journal[0]["body"] is not st.journal[0]["body"]
    child.journal[0]["body"]["o"] = 999
    assert not child.verify() and st.verify()
    assert child.gates[0] is not st.gates[0]


def test_corrupt_parent_still_refuses_before_copy():
    st = populated()
    st.journal[-1]["body"]["o"] = 999
    with pytest.raises(ValueError, match="corrupt"):
        st.rebuild(upto=1)


def test_skip_and_persistent_records_keep_rechaining(tmp_path, monkeypatch):
    original = _MemoryJournal.append
    calls = []

    def counted(self, body):
        calls.append(body)
        return original(self, body)

    st = populated()
    disk = populated(tmp_path / "knowledge.jsonl")
    monkeypatch.setattr(_MemoryJournal, "append", counted)
    skip = st.rebuild(skip={next(iter(st.items))})
    assert len(calls) == 11 and skip._mem.verify()
    calls.clear()
    copy = disk.rebuild()
    assert len(calls) == 12 and copy._mem.verify()
    assert copy.fingerprint() == disk.fingerprint()
