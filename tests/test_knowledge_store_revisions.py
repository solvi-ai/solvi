"""solvi.core.knowledge.store: reads inside a batch, validity intervals (`valid=`), re-asserted retracted content
waiting for a re-test, retraction of one writer's support, re-deciding on the corrected journal, and the cost of a
load."""
import time

from solvi.core.knowledge import KnowledgeStore


def fact(s, o, r="r"):
    return {"s": s, "r": r, "o": o}


# --- reads inside a batch
def test_an_item_added_in_a_batch_is_read_as_present_inside_the_batch():
    ks = KnowledgeStore()
    with ks.batch():
        a = ks.add("fact", fact("a", 1), source="spec")
        assert ks.status(a) == "active" and ks.usable(a)
        d = ks.add("fact", fact("d", 2), source="spec", derived_from=[a])
        assert d is not None and ks.status(d) == "active"
    assert ks.status(a) == ks.status(d) == "active"
    assert ks.verify()


# --- validity intervals
def test_values_on_disjoint_validity_intervals_do_not_dispute():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("drugX", "40mg", "max_dose"), source="spec", by="label-2023",
               valid=("2023-01-01", "2024-06-01"))
    b = ks.add("fact", fact("drugX", "20mg", "max_dose"), source="spec", by="label-2024", valid=("2024-06-01", None))
    assert ks.status(a) == ks.status(b) == "active" and ks.disputes() == []
    assert ks.usable(a, valid_at="2023-05-01") and not ks.usable(a, valid_at="2025-01-01")
    assert ks.usable(b, valid_at="2025-01-01") and not ks.usable(b, valid_at="2023-05-01")
    snap = ks.snapshot(about="drugX", valid_at="2025-01-01")
    assert [f["body"]["o"] for f in snap["facts"]] == ["20mg"] and snap["query"]["valid_at"] == "2025-01-01"
    assert [h["why"] for h in snap["hints"]] == ["not valid at 2025-01-01"]
    assert ks.verify()


def test_overlapping_validity_still_disputes():
    ks = KnowledgeStore()
    a = ks.add("fact", fact("x", 1), source="spec", by="p", valid=(None, 10))
    b = ks.add("fact", fact("x", 2), source="spec", by="q", valid=(5, None))
    assert ks.status(a) == ks.status(b) == "disputed" and len(ks.disputes()) == 1


def test_a_later_valid_since_supersedes_whatever_order_the_notes_arrive_in():
    ks = KnowledgeStore()
    new = ks.add("fact", fact("anna", "manager", "role"), source="outcome", by="hr", valid=(50, None))
    old = ks.add("fact", fact("anna", "engineer", "role"), source="outcome", by="wiki", valid=(0, None))
    assert ks.status(new) == "active" and ks.status(old) == "refuted"
    assert ks.item(old)["confidence"]["n_refute"] == 1


def test_the_same_valid_since_from_two_writers_is_a_dispute():
    ks = KnowledgeStore()
    x = ks.add("fact", fact("bob", "Oslo", "city"), source="outcome", by="A", valid=(10, None))
    y = ks.add("fact", fact("bob", "Rome", "city"), source="outcome", by="B", valid=(10, None))
    assert ks.status(x) == ks.status(y) == "disputed" and len(ks.disputes()) == 1


def test_an_expired_item_is_not_usable_at_a_later_time_and_items_without_valid_are_unchanged():
    ks = KnowledgeStore()
    z = ks.add("fact", fact("c", 1), source="spec", valid=(2020, 2021))
    assert ks.usable(z) and ks.usable(z, valid_at=2020) and not ks.usable(z, valid_at=2021)
    w = ks.add("fact", fact("w", 1), source="spec")
    assert ks.usable(w, valid_at=1999)
    p = ks.add("fact", fact("k", 1), source="spec")
    q = ks.add("fact", fact("k", 2), source="outcome")            # no valid: an observation refutes, as before
    assert ks.status(p) == "refuted" and ks.status(q) == "active"
