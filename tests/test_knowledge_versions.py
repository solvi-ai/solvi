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
    same = ks.add("fact", {"s": "X", "r": "p", "o": 3}, source="person", by="c", valid=(10, None))
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
