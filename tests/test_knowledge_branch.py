import copy

import pytest

from solvi import Knowledge
from solvi.core.knowledge import KnowledgeStore


def test_two_branches_prefix_lineage_and_persistent_reopen(tmp_path):
    parent = KnowledgeStore(tmp_path / "parent.jsonl", writers={"world": "outcome"})
    a = parent.add("fact", {"s": "x", "r": "v", "o": "a"}, source="outcome", by="world")
    prefix = len(parent)
    parent.add("fact", {"s": "y", "r": "v", "o": "later"}, source="outcome", by="world")
    before = copy.deepcopy(parent.journal), parent.head(), parent.fingerprint()
    left = parent.branch(tmp_path / "left.jsonl", upto=prefix)
    right = parent.branch()
    left.retract(a, by="world")
    assert right.usable(a) and parent.usable(a) and not left.usable(a)
    assert len(left.items) == 1 and len(right.items) == 2
    assert left.journal[prefix]["branch"]["prefix_hash"] == parent.journal[prefix - 1]["hash"]
    assert left.journal[prefix]["branch"]["parent"] == before[1]
    assert (parent.journal, parent.head(), parent.fingerprint()) == before
    reopened = KnowledgeStore(tmp_path / "left.jsonl", writers={"world": "outcome"})
    assert reopened.verify() and reopened.fingerprint() == left.fingerprint()
    assert reopened.add("fact", {"s": "z", "r": "v", "o": 1}, source="outcome", by="other") is None
    assert right.verify() and left.verify() and parent.verify()
    with pytest.raises(ValueError, match="empty"):
        parent.branch(tmp_path / "parent.jsonl")
    with pytest.raises(ValueError, match="empty"):
        parent.branch(tmp_path / "left.jsonl")


def test_facade_preserves_native_runtime_and_blocks_custom_before_writes(tmp_path):
    km = Knowledge(vocabulary={"ready": lambda s, a: s["ready"]}, failures=True)
    km.goal("done", done=lambda s: s["done"])
    km.agenda.gate("ready", lambda s: s["ready"], blocks=["work"])
    km.actions.observe({"ready": True}, "work", {}, True, None)
    child = km.branch(tmp_path / "child.jsonl")
    assert not child.agenda.allows({"ready": False}, "work")[0]
    child.agenda.update({"done": True, "ready": True})
    assert "done" in child.agenda.done() and "done" not in km.agenda.done()
    child.actions.observe({"ready": False}, "work", {}, False, None)
    assert child.actions.transitions != km.actions.transitions
    child.failures.step()
    assert child.failures.now != km.failures.now
    assert all(x.store is child.store for x in (child.agenda, child.actions, child.failures))
    assert child.store.verify() and km.store.verify()

    class Custom:
        observe = predict = fingerprint = lambda *args: None

    custom = Knowledge(actions=Custom())
    target = tmp_path / "unsupported.jsonl"
    with pytest.raises(TypeError, match="custom"):
        custom.branch(target)
    assert not target.exists()


def test_verified_history_refused_before_persistent_destination(tmp_path):
    parent = KnowledgeStore()
    parent.note(of="original-decision")
    target = tmp_path / "proof.jsonl"
    with pytest.raises(ValueError, match="verified"):
        parent.branch(target)
    assert not target.exists()
    assert parent.branch().verify()


@pytest.mark.parametrize("upto", [-1, 1, True, 0.5])
def test_bad_prefix_is_atomic(upto):
    parent = KnowledgeStore()
    with pytest.raises(ValueError, match="prefix"):
        parent.branch(upto=upto)
    assert len(parent) == 0


def test_empty_parent_alias_refused_without_lineage_write(tmp_path):
    path = tmp_path / "empty.jsonl"
    parent = KnowledgeStore(path)
    with pytest.raises(ValueError, match="independent"):
        parent.branch(path)
    assert not parent.journal and parent.storage.head()["count"] == 0


def test_empty_parent_hardlink_is_not_an_independent_destination(tmp_path):
    import os
    path, alias = tmp_path / "parent.jsonl", tmp_path / "alias.jsonl"
    path.touch()
    parent = KnowledgeStore(path)
    os.link(path, alias)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="independent"):
        parent.branch(alias)
    assert path.read_bytes() == alias.read_bytes() == before
    assert not parent.journal


@pytest.mark.parametrize("persistent", [False, True])
def test_corrupt_parent_cannot_be_laundered_by_reconstruction(tmp_path, persistent):
    path = tmp_path / "parent.jsonl" if persistent else None
    parent = KnowledgeStore(path)
    parent.add("fact", {"s": "x", "r": "v", "o": "original"}, source="outcome")
    before_disk = path.read_bytes() if path else None
    if persistent:
        # Tamper only the reconstruction's cached journal, not the backend.
        parent.journal = copy.deepcopy(parent.journal)
    parent.journal[0]["body"]["o"] = "tampered"
    if persistent:
        assert parent.storage.verify()["ok"]
    assert not parent.verify()
    destination = tmp_path / "child.jsonl"
    for operation in (lambda: parent.rebuild(), lambda: parent.branch(), lambda: parent.branch(destination)):
        with pytest.raises(ValueError, match="corrupt"):
            operation()
    assert not destination.exists()
    assert parent.journal[0]["body"]["o"] == "tampered"
    if path:
        assert path.read_bytes() == before_disk
