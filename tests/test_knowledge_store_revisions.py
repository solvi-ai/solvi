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


# --- re-asserting retracted content
def test_re_asserting_retracted_content_waits_for_a_retest_and_so_do_its_derivations():
    ks = KnowledgeStore()
    f = ks.add("fact", fact("w", "kg", "unit"), source="spec", by="v1")
    d = ks.add("rule", fact("auto", "kg", "uses"), source="person", derived_from=[f])
    assert ks.status(d) == "active"
    ks.retract(f, why="wrong")
    assert ks.status(f) == ks.status(d) == "retracted"
    assert ks.add("fact", fact("w", "kg", "unit"), source="spec", by="v6") == f
    assert ks.status(f) == ks.status(d) == "hypothesis" and ks.pending_retest(f) and ks.pending_retest(d)
    assert not ks.usable(f) and [h["why"] for h in ks.snapshot(about="w")["hints"]] == ["pending retest"]
    ks.retest(f, ok=True, by="qa")
    assert ks.status(f) == "active" and ks.status(d) == "hypothesis"       # a derivation waits for its own re-test
    ks.retest(d, ok=True, by="qa")
    assert ks.status(d) == "active" and not ks.pending_retest(d)
    assert ks.verify()


def test_a_failed_retest_retracts_and_a_rule_re_added_after_its_retraction_is_held():
    ks = KnowledgeStore()
    f = ks.add("fact", fact("a", 1), source="spec")
    ks.retract(f)
    ks.add("fact", fact("a", 1), source="spec")
    ks.retest(f, ok=False, why="still wrong")
    assert ks.status(f) == "retracted"
    r = ks.add("rule", fact("rule", 1), source="person")
    ks.retract(r)
    assert ks.add("rule", fact("rule", 1), source="person") == r
    assert ks.status(r) == "hypothesis" and ks.journal[-1]["op"] == "held"
    assert "re-test" in ks.journal[-1]["verdicts"][-1]["reason"]


def test_content_written_after_a_retraction_of_its_premise_is_not_pending():
    ks = KnowledgeStore()
    f = ks.add("fact", fact("p", 1), source="spec")
    ks.retract(f)
    ks.add("fact", fact("p", 1), source="spec")
    ks.retest(f, ok=True)
    d = ks.add("fact", fact("q", 2), source="spec", derived_from=[f])     # derived after the re-test: fresh
    assert ks.status(d) == "active" and not ks.pending_retest(d)
    g = ks.add("fact", fact("g", 1), source="person")
    ks.add("fact", fact("g", 1), source="outcome")                        # a confirmation, never retracted
    assert ks.status(g) == "active"


def test_an_item_a_rule_re_infers_after_its_retraction_follows_its_premises():
    ks = KnowledgeStore()
    p = ks.add("fact", fact("p", 1), source="spec")
    q = ks.add("fact", fact("q", 1), source="spec", derived_from=[p], inferred=True)
    ks.retract(q, why="the negative condition no longer holds")
    assert ks.add("fact", fact("q", 1), source="spec", derived_from=[p], inferred=True) == q
    assert ks.status(q) == "active" and not ks.pending_retest(q)
