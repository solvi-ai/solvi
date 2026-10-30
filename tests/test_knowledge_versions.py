"""Versions in time on a key with `valid`: a later since closes the earlier version at that since instead of refuting
it (a bounded later version gives the earlier value back after it ends; a retracted one re-opens what it closed),
validity per assertion (a value told for two periods holds in both), a dispute scoped to its overlap, rollback and a
person's answer closing only the overlap. Each test names what the store did before the change."""
import datetime

import solvi
from solvi.core.knowledge import KnowledgeStore


def vals(km, s, r, at):
    return [f["body"]["o"] for f in km.store.snapshot(about=s, relation=r, valid_at=at)["facts"]]


def test_an_as_of_read_before_a_later_version_gives_the_earlier_value():
    # before: the engineer row was refuted by the later since and the read at 10 was empty
    km = solvi.Knowledge()
    new = km.tell({"s": "anna", "r": "role", "o": "manager"}, source="outcome", by="hr", valid=(50, None))
    old = km.tell({"s": "anna", "r": "role", "o": "engineer"}, source="outcome", by="wiki", valid=(0, None))
    st = km.store
    assert st.status(new) == st.status(old) == "active"
    assert vals(km, "anna", "role", 10) == ["engineer"] and vals(km, "anna", "role", 60) == ["manager"]
    snap = km.snapshot()
    assert solvi.Knowledge.value(snap, "anna", "role", valid_at=10) == "engineer"
    assert solvi.Knowledge.value(snap, "anna", "role", valid_at=60) == "manager"
    assert st.verify()


def test_a_bounded_later_version_gives_the_earlier_value_back_after_it_ends():
    # before: the base price was refuted for good and the key had no value at 90 or at 10
    km = solvi.Knowledge()
    base = km.tell({"s": "A", "r": "price", "o": 100}, source="outcome", by="pricing", valid=(0, None))
    promo = km.tell({"s": "A", "r": "price", "o": 80}, source="outcome", by="pricing", valid=(50, 80))
    assert km.store.status(base) == "active"
    assert [vals(km, "A", "price", t) for t in (10, 60, 90)] == [[100], [80], [100]]
    assert km.store.periods(base) == [[0, 50], [80, None]] and km.store.periods(promo) == [[50, 80]]
    assert km.store.usable(base, valid_at=90) and not km.store.usable(base, valid_at=60)
    assert km.store.verify()


def test_a_value_announced_ahead_does_not_end_the_current_one_before_its_date():
    # before: Ann was refuted on the day Bob was written; the read at 95 was empty
    km = solvi.Knowledge()
    ann = km.tell({"s": "T", "r": "lead", "o": "Ann"}, source="outcome", by="hr", valid=(0, None))
    km.tell({"s": "T", "r": "lead", "o": "Bob"}, source="outcome", by="hr", valid=(108, None))
    assert km.store.status(ann) == "active"
    assert vals(km, "T", "lead", 95) == ["Ann"] and vals(km, "T", "lead", 108) == ["Bob"]


def test_a_retracted_later_version_re_opens_what_it_closed():
    km = solvi.Knowledge()
    base = km.tell({"s": "A", "r": "price", "o": 100}, source="outcome", by="pricing", valid=(0, None))
    later = km.tell({"s": "A", "r": "price", "o": 120}, source="outcome", by="pricing", valid=(50, None))
    assert vals(km, "A", "price", 60) == [120]
    km.store.retract(later, why="typo")
    assert vals(km, "A", "price", 60) == [100] and km.store.periods(base) == [[0, None]]
    assert km.store.verify()


def test_a_dispute_is_scoped_to_the_overlap():
    # before: both sides were disputed on their whole intervals; the read at 150 was empty
    km = solvi.Knowledge()
    a = km.tell({"s": "X", "r": "price", "o": 163}, source="outcome", by="pricing", valid=(73, 99))
    b = km.tell({"s": "X", "r": "price", "o": 244}, source="outcome", by="vendor", valid=(73, None))
    st = km.store
    assert st.status(a) == st.status(b) == "disputed" and len(st.disputes()) == 1
    assert vals(km, "X", "price", 80) == [] and vals(km, "X", "price", 150) == [244]
    assert st.usable(b, valid_at=150) and not st.usable(b, valid_at=80) and not st.usable(b)
    assert solvi.Knowledge.value(km.snapshot(), "X", "price", valid_at=150) == 244
    assert solvi.Knowledge.value(km.snapshot(), "X", "price", valid_at=80) is None


def test_a_persons_answer_closes_the_loser_only_on_the_overlap():
    km = solvi.Knowledge()
    a = km.tell({"s": "X", "r": "price", "o": 163}, source="outcome", by="pricing", valid=(73, 99))
    b = km.tell({"s": "X", "r": "price", "o": 244}, source="outcome", by="vendor", valid=(73, None))
    st = km.store
    st.resolve(st.disputes()[0]["key"], 163, by="ann")
    assert st.status(a) == st.status(b) == "active" and st.disputes() == []
    assert vals(km, "X", "price", 80) == [163] and vals(km, "X", "price", 150) == [244]
    assert st.periods(b) == [[99, None]]
    st.resolve(tuple(st.items[a].key), 244, by="ann")       # an answer with no open dispute changes nothing
    assert vals(km, "X", "price", 80) == [163]
    # the loser whose whole interval is the overlap is refuted, as before
    km2 = solvi.Knowledge()
    c = km2.tell({"s": "X", "r": "price", "o": 163}, source="outcome", by="pricing", valid=(73, 99))
    km2.tell({"s": "X", "r": "price", "o": 244}, source="outcome", by="vendor", valid=(73, None))
    km2.store.resolve(km2.store.disputes()[0]["key"], 244, by="ann")
    assert km2.store.status(c) == "refuted" and vals(km2, "X", "price", 80) == [244]


def test_a_higher_rank_on_the_same_since_takes_only_its_overlap():
    ks = KnowledgeStore()
    low = ks.add("fact", {"s": "X", "r": "p", "o": 1}, source="spec", by="a", valid=(10, None))
    high = ks.add("fact", {"s": "X", "r": "p", "o": 2}, source="person", by="b", valid=(10, 20))
    assert ks.status(low) == ks.status(high) == "active"
    assert ks.periods(high) == [[10, 20]] and ks.periods(low) == [[20, None]]
    ks.add("fact", {"s": "X", "r": "p", "o": 3}, source="person", by="c", valid=(10, None))
    assert ks.status(low) == "refuted"                       # covered whole: refuted, as before


def test_a_value_told_for_two_periods_holds_in_both():
    # before: one validity per item (the latest), so the first period of 12 was lost
    km = solvi.Knowledge()
    x = km.tell({"s": "A", "r": "warranty", "o": 12}, source="outcome", by="product", valid=(0, None))
    km.tell({"s": "A", "r": "warranty", "o": 24}, source="outcome", by="product", valid=(22, None))
    assert km.tell({"s": "A", "r": "warranty", "o": 12}, source="outcome", by="product", valid=(174, None)) == x
    assert [vals(km, "A", "warranty", t) for t in (10, 100, 200)] == [[12], [24], [12]]
    assert km.store.valid(x) == [174, None]                  # the latest assertion's, as before
    rec = km.store.item(x)["valid"]
    assert [c["valid"] for c in rec["claims"]] == [[0, None], [174, None]]
    assert rec["effective"] == [[0, 22], [174, None]]


def test_iso_strings_and_dates_work_as_times():
    km = solvi.Knowledge()
    km.tell({"s": "ECB", "r": "dfr", "o": 4.0}, source="outcome", by="ecb", valid=("2023-09-20", None))
    km.tell({"s": "ECB", "r": "dfr", "o": 3.75}, source="outcome", by="ecb", valid=(datetime.date(2024, 6, 12), None))
    assert vals(km, "ECB", "dfr", "2024-01-01") == [4.0] and vals(km, "ECB", "dfr", datetime.date(2024, 7, 1)) == [3.75]
    snap = km.snapshot()
    assert solvi.Knowledge.value(snap, "ECB", "dfr", valid_at=datetime.date(2024, 1, 1)) == 4.0
    assert km.store.snapshot(valid_at=datetime.date(2024, 1, 1))["query"]["valid_at"] == "2024-01-01"
    assert km.store.verify()


def test_versions_written_in_a_batch_and_read_from_a_second_store_on_the_same_journal(tmp_path):
    path = str(tmp_path / "k.jsonl")
    ks = KnowledgeStore(path)
    with ks.batch():
        base = ks.add("fact", {"s": "A", "r": "p", "o": 100}, source="outcome", by="w", valid=(0, None))
        ks.add("fact", {"s": "A", "r": "p", "o": 80}, source="outcome", by="w", valid=(50, 80))
        assert ks.usable(base, valid_at=90) and not ks.usable(base, valid_at=60)
    other = KnowledgeStore(path)
    assert other.fingerprint() == ks.fingerprint() and other.periods(base) == [[0, 50], [80, None]]
    assert other.verify()


def test_a_rollback_re_opens_the_version_the_rolled_back_one_closed():
    ks = KnowledgeStore()
    base = ks.add("fact", {"s": "A", "r": "p", "o": 100}, source="outcome", by="w", valid=(0, None))
    later = ks.add("fact", {"s": "A", "r": "p", "o": 120}, source="outcome", by="w", valid=(50, None))
    f = ks.flag(ids=[later], why="drift")
    ks.rollback(later, by="sup")
    assert ks.status(later) == "retracted" and ks.periods(base) == [[0, None]]
    assert ks.status(base) == "hypothesis"                    # restored during a pending flag: a hint
    ks.end_flag(f)
    assert ks.status(base) == "active" and ks.usable(base, valid_at=60)
    assert ks.verify()


# --- derived validity
def test_a_derived_fact_holds_only_where_its_premises_hold():
    # before: the derived item had no validity and stayed usable at 90 after its premise's interval ended
    km = solvi.Knowledge()
    st = km.store
    p = km.tell({"s": "A", "r": "price", "o": 80}, source="outcome", by="pricing", valid=(50, 80))
    d = km.tell({"s": "A", "r": "max_refund", "o": 64}, source="outcome", by="rule", derived_from=[p], inferred=True)
    assert st.valid(d) == [50, 80] and st.items[d].asserts[-1]["inferred"]
    assert st.usable(d, valid_at=60) and not st.usable(d, valid_at=90)
    assert [f["body"]["o"] for f in st.snapshot(about="A", relation="max_refund", valid_at=90)["facts"]] == []


def test_a_derived_fact_follows_its_premise_when_a_later_version_closes_it():
    ks = KnowledgeStore()
    base = ks.add("fact", {"s": "A", "r": "price", "o": 100}, source="outcome", by="w", valid=(0, None))
    ks.add("fact", {"s": "A", "r": "price", "o": 80}, source="outcome", by="w", valid=(50, 80))
    d = ks.add("fact", {"s": "A", "r": "refund", "o": 80}, source="spec", by="rule", derived_from=[base], inferred=True)
    assert ks.valid(d) == [0, None] and ks.periods(d) == [[0, 50], [80, None]]
    assert ks.usable(d, valid_at=10) and not ks.usable(d, valid_at=60) and ks.usable(d, valid_at=90)
    snap = ks.snapshot()
    assert solvi.Knowledge.value(snap, "A", "refund", valid_at=60) is None
    assert solvi.Knowledge.value(snap, "A", "refund", valid_at=90) == 80


def test_premises_that_never_hold_together_refuse_the_derivation_with_the_reason():
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "a", "r": "x", "o": 1}, source="outcome", by="w", valid=(0, 10))
    b = ks.add("fact", {"s": "b", "r": "x", "o": 1}, source="outcome", by="w", valid=(10, 20))
    c = ks.add("fact", {"s": "c", "r": "x", "o": 1}, source="outcome", by="w", valid=(5, 15))
    assert ks.add("fact", {"s": "d", "r": "x", "o": 2}, source="spec", by="rule", derived_from=[a, b]) is None
    assert ks.journal[-1]["op"] == "refused" and ks.journal[-1]["why"].startswith("validity: ")
    e = ks.add("fact", {"s": "e", "r": "x", "o": 2}, source="spec", by="rule", derived_from=[a, c])
    assert ks.valid(e) == [5, 10]
    f = ks.add("fact", {"s": "f", "r": "x", "o": 2}, source="spec", by="rule", derived_from=[a], valid=(0, 3))
    assert ks.valid(f) == [0, 3]                             # its own valid is kept
    g = ks.add("fact", {"s": "g", "r": "x", "o": 1}, source="spec", by="w")
    h = ks.add("fact", {"s": "h", "r": "x", "o": 2}, source="spec", by="rule", derived_from=[g])
    assert ks.valid(h) is None and ks.journal[-1].get("valid") is None   # premises without valid: as before
    assert ks.verify()


def test_an_inferred_item_follows_its_premise_after_the_premise_passes_its_retest():
    # before: after retest(a, ok=True) the inferred item stayed a hypothesis pending its own re-test
    km = solvi.Knowledge()
    st = km.store
    a = km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="w")
    d = st.add("fact", {"s": "D", "r": "x", "o": 1}, None, source="spec", by="sup", derived_from=[a], inferred=True)
    st.retract(a, by="w")
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="sync")
    assert st.pending_retest(a) and st.pending_retest(d)
    st.retest(a, ok=True)
    assert st.status(a) == st.status(d) == "active" and not st.pending_retest(d)
    # a derivation that is not inferred still waits for its own re-test (unchanged)
    km2 = solvi.Knowledge()
    s2 = km2.store
    a2 = km2.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="w")
    d2 = s2.add("fact", {"s": "D", "r": "x", "o": 1}, None, source="spec", by="sup", derived_from=[a2])
    s2.retract(a2, by="w")
    km2.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="sync")
    s2.retest(a2, ok=True)
    assert s2.status(d2) == "hypothesis" and s2.pending_retest(d2)


# --- the items waiting for a re-test
def test_pending_lists_the_items_waiting_for_a_retest_with_the_reason():
    # before: there was no list; a re-told retracted fact was a hypothesis with no reason given
    km = solvi.Knowledge()
    st = km.store
    f = km.tell({"s": "R", "r": "rule", "o": "x"}, source="spec", by="reg")
    d = st.add("fact", {"s": "R2", "r": "uses", "o": "x"}, source="spec", by="reg", derived_from=[f])
    assert st.pending() == []
    st.retract(f, why="forged")
    g = km.tell({"s": "R", "r": "rule", "o": "x"}, source="spec", by="reg")
    assert g == f and st.pending() == sorted([f, d])
    assert st.item(f)["pending"].startswith("written again after its retraction")
    assert st.item(d)["pending"].startswith(f"rests on {f}")
    st.retest(f, ok=True)
    assert st.pending() == [d] and "pending" not in st.item(f)
    st.retest(d, ok=True)
    assert st.pending() == []


# --- taking back one assertion
def test_one_of_two_identical_publications_can_be_taken_back():
    # before: retract(only_from="ed") took back both of ed's publications; the read at 20 was empty
    km = solvi.Knowledge()
    st = km.store
    a = km.tell({"s": "PM", "r": "incumbent", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev1")
    km.tell({"s": "PM", "r": "incumbent", "o": "V"}, source="outcome", by="vandal", valid=(30, None), evidence="rev2")
    b = km.tell({"s": "PM", "r": "incumbent", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev3")
    assert a == b and len(st.items[a].asserts) == 2
    second = st.item(a)["valid"]["claims"][1]["at"]
    st.retract(a, assertion=second, why="duplicate")
    assert st.status(a) == "active" and len(st.item(a)["valid"]["claims"]) == 1
    assert vals(km, "PM", "incumbent", 20) == ["P"]
    km2 = solvi.Knowledge()
    a2 = km2.tell({"s": "PM", "r": "incumbent", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev1")
    km2.tell({"s": "PM", "r": "incumbent", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev3")
    km2.retract(a2, only_from=("ed", "rev3"))
    assert km2.store.status(a2) == "active" and len(km2.store.item(a2)["valid"]["claims"]) == 1
    km2.retract(a2, only_from=("ed", "rev1"))
    assert km2.store.status(a2) == "retracted"
    assert st.verify() and km2.store.verify()


def test_only_from_a_writer_keeps_its_meaning_and_wrong_addresses_are_refused():
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "x", "r": "r", "o": 1}, source="spec", by="w", evidence="e1")
    ks.add("fact", {"s": "x", "r": "r", "o": 1}, source="spec", by="w", evidence="e2")
    for bad in ({"assertion": 99}, {"only_from": ("w", "nope")}, {"only_from": ("w",)},
                {"only_from": "w", "assertion": 0}):
        try:
            ks.retract(a, **bad)
            raise AssertionError(f"expected ValueError for {bad}")
        except ValueError:
            pass
    ks.retract(a, only_from="w")
    assert ks.status(a) == "retracted"
    assert ks.add("fact", {"s": "x", "r": "r", "o": 1}, source="spec", by="w") == a and ks.pending_retest(a)


# --- the revival guard: withdrawn content told again
def test_withdrawn_content_told_again_by_another_writer_waits_for_a_retest_and_the_honest_period_stays():
    # before: the partner's copy revived 12 from 174 without a re-test and the read at 10 was empty
    km = solvi.Knowledge()
    st = km.store
    old = km.tell({"s": "A", "r": "warranty", "o": 12}, source="outcome", by="product", valid=(0, None))
    km.tell({"s": "A", "r": "warranty", "o": 24}, source="outcome", by="product", valid=(22, None))
    km.tell({"s": "A", "r": "warranty", "o": 12}, source="outcome", by="vendor", valid=(174, None))
    st.retract_source("vendor")
    km.tell({"s": "A", "r": "warranty", "o": 12}, source="outcome", by="partner", valid=(174, None))
    assert st.pending_retest(old) and st.pending() == [old] and "withdrawn from vendor" in st.item(old)["pending"]
    assert st.status(old) == "active"                        # the product's own period still holds
    assert [vals(km, "A", "warranty", t) for t in (10, 100, 200)] == [[12], [24], [24]]
    assert [c.get("pending", False) for c in st.item(old)["valid"]["claims"]] == [False, True]
    st.retest(old, ok=False, by="qa")                        # only the copy is taken back
    assert st.status(old) == "active" and st.pending() == [] and vals(km, "A", "warranty", 10) == [12]
    assert st.verify()


def test_a_passed_retest_lets_the_copy_hold():
    km = solvi.Knowledge()
    st = km.store
    x = km.tell({"s": "A", "r": "p", "o": 12}, source="outcome", by="vendor", valid=(174, None))
    st.retract_source("vendor")
    assert km.tell({"s": "A", "r": "p", "o": 12}, source="outcome", by="partner", valid=(180, None)) == x
    assert st.pending() == [x] and vals(km, "A", "p", 200) == []          # overlapping, not the same interval
    st.retest(x, ok=True, by="qa")
    assert st.pending() == [] and vals(km, "A", "p", 200) == [12]
    km.tell({"s": "A", "r": "p", "o": 12}, source="outcome", by="mirror", valid=(190, None))
    assert st.pending() == []                                # the re-test cleared the withdrawal before it


def test_a_copy_admitted_while_another_writer_held_it_goes_pending_when_that_holder_is_cut():
    # before: the item stayed usable on the mirror's copy after the vendor was exposed
    km = solvi.Knowledge()
    st = km.store
    f = km.tell({"s": "A", "r": "price", "o": 50}, source="outcome", by="vendor", valid=(99, None))
    km.tell({"s": "A", "r": "price", "o": 50}, source="outcome", by="pricing", valid=(99, None))
    st.retract(f, only_from="pricing")
    km.tell({"s": "A", "r": "price", "o": 50}, source="outcome", by="mirror", valid=(99, None))
    assert not st.pending_retest(f) and st.usable(f)
    assert [c.get("unchecked", False) for c in st.item(f)["valid"]["claims"]] == [False, True]
    st.retract_source("vendor")
    assert st.pending_retest(f) and not st.usable(f) and st.pending() == [f]


def test_with_an_erratum_the_exposed_copy_no_longer_blocks_the_right_value():
    # before: the mirror's 50 and the erratum 84 stayed a dispute after the vendor's exposure; the read at 120 was empty
    km = solvi.Knowledge()
    st = km.store
    f = km.tell({"s": "A", "r": "price", "o": 50}, source="outcome", by="vendor", valid=(99, None))
    km.tell({"s": "A", "r": "price", "o": 50}, source="outcome", by="pricing", valid=(99, None))
    st.retract(f, only_from="pricing")
    e = km.tell({"s": "A", "r": "price", "o": 84}, source="outcome", by="pricing", valid=(99, None))
    km.tell({"s": "A", "r": "price", "o": 50}, source="outcome", by="mirror", valid=(99, None))
    st.retract_source("vendor")
    assert st.pending() == [f] and st.status(e) == "active" and vals(km, "A", "price", 120) == [84]


def test_a_retold_retracted_item_is_pending_and_takes_no_side_in_a_dispute():
    km = solvi.Knowledge()
    st = km.store
    km.tell({"s": "A", "r": "p", "o": 50}, source="outcome", by="vendor", valid=(99, None))
    km.tell({"s": "A", "r": "p", "o": 84}, source="outcome", by="pricing", valid=(99, None))
    st.retract_source("vendor")
    g = km.tell({"s": "A", "r": "p", "o": 50}, source="outcome", by="partner", valid=(99, None))
    assert (st.status(g), st.pending_retest(g), len(st.disputes())) == ("hypothesis", True, 0)


def test_the_guard_matches_content_across_ids_and_an_honest_re_dating_goes_to_a_retest():
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "A", "r": "p", "o": 7, "note": "feed"}, source="outcome", by="feed", valid=(10, 20))
    ks.retract(a, why="wrong feed")
    b = ks.add("fact", {"s": "A", "r": "p", "o": 7}, source="outcome", by="other", valid=(15, 30))
    assert b != a and ks.pending() == [b]                    # another id, same content, overlapping validity
    c = ks.add("fact", {"s": "A", "r": "p", "o": 7}, source="outcome", by="other", valid=(20, 30))
    assert ks.pending() == [b]                               # c is b (same content, same key): one more claim on b
    d = ks.add("fact", {"s": "A", "r": "q", "o": 7}, source="outcome", by="other", valid=(15, 30))
    assert not ks.pending_retest(d)                          # another key is other content
    e = ks.add("fact", {"s": "A", "r": "p", "o": 8}, source="outcome", by="feed", valid=(40, 50))
    assert not ks.pending_retest(e) and c == b
