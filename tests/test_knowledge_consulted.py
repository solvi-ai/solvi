"""`consulted=`: the items a conclusion looked at besides its premises — the evidence for the alternatives it did not
choose. Retracting one of them does not retract the conclusion (its premises stand) but marks it for review: a
hypothesis pending re-test, as for an item that rested on a retracted premise."""
import pytest

from solvi.core.knowledge import KnowledgeStore


def store():
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "x", "r": "vote", "o": "A"}, source="person", by="ann")
    b = ks.add("fact", {"s": "y", "r": "vote", "o": "B"}, source="person", by="bob")
    c = ks.add("fact", {"s": "z", "r": "winner", "o": "A"}, source="person", by="cal", derived_from=[a], consulted=[b])
    return ks, a, b, c


def test_retracting_a_losing_alternative_marks_the_conclusion_for_review():
    ks, a, b, c = store()
    assert ks.status(c) == "active"
    assert ks.item(c)["consulted"] == [b] and ks.item(c)["derived_from"] == [a]
    changed = ks.retract(b, by="admin", why="bob's vote was forged")
    assert changed[c] == ("active", "hypothesis")
    assert ks.pending_retest(c)
    ks.retest(c, ok=True, by="admin")
    assert ks.status(c) == "active" and not ks.pending_retest(c)
    ks.verify()


def test_retracting_the_premise_still_cascades_and_without_consulted_nothing_changes():
    ks, a, b, c = store()
    ks.retract(a)
    assert ks.status(c) != "active"
    plain = KnowledgeStore()
    a2 = plain.add("fact", {"s": "x", "r": "vote", "o": "A"}, source="person", by="ann")
    b2 = plain.add("fact", {"s": "y", "r": "vote", "o": "B"}, source="person", by="bob")
    c2 = plain.add("fact", {"s": "z", "r": "winner", "o": "A"}, source="person", derived_from=[a2])
    plain.retract(b2)
    assert plain.status(c2) == "active" and "consulted" not in plain.item(c2)


def test_the_journal_replays_the_consulted_edges(tmp_path):
    ks = KnowledgeStore(str(tmp_path / "k.jsonl"))
    a = ks.add("fact", {"s": "x", "r": "vote", "o": "A"}, source="person")
    b = ks.add("fact", {"s": "y", "r": "vote", "o": "B"}, source="person")
    c = ks.add("fact", {"s": "z", "r": "winner", "o": "A"}, source="person", derived_from=[a], consulted=[b])
    ks.retract(b)
    again = KnowledgeStore(str(tmp_path / "k.jsonl"))
    assert again.pending_retest(c) and again.fingerprint() == ks.fingerprint()


def test_consulted_items_must_exist():
    ks = KnowledgeStore()
    with pytest.raises(ValueError):
        ks.add("fact", {"s": "z", "r": "winner", "o": "A"}, source="person", consulted=["nope"])
