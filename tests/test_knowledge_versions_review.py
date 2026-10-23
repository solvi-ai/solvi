"""Versions in time on a key with `valid`: the cases an implementation review found (a copy held by a version that does
not hold, mixed time types under a derivation, a deep derivation DAG, a duplicate taken back, the until of one version
told twice, the journal entry of `retract(id, assertion=)`). Each test names what the store did before the fix."""
import time

import solvi
from solvi.core.knowledge import KnowledgeStore


def vals(st, s, r, at):
    return [f["body"]["o"] for f in st.snapshot(about=s, relation=r, valid_at=at)["facts"]]


def test_an_exposed_value_retold_at_a_later_since_over_a_refuted_holder_waits_for_a_retest():
    # before: p's refuted 1 counted as a holder, so the mirror's copy was admitted "unchecked" and the key read 1 at 60
    km = solvi.Knowledge()
    st = km.store
    x = km.tell({"s": "A", "r": "p", "o": 1}, source="spec", by="p", valid=(0, None))
    q = km.tell({"s": "A", "r": "p", "o": 2}, source="outcome", by="q", valid=(0, None))
    assert st.status(x) == "refuted" and vals(st, "A", "p", 60) == [2]
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="vendor", valid=(50, None))
    st.retract_source("vendor")
    assert vals(st, "A", "p", 60) == [2]
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="mirror", valid=(50, None))
    assert st.pending() == [x] and "withdrawn from vendor" in st.item(x)["pending"]
    assert st.status(q) == "active" and vals(st, "A", "p", 60) == [2] and vals(st, "A", "p", 10) == [2]
    assert not st.usable(x, valid_at=60)
    st.retest(x, ok=True, by="qa")
    assert st.pending() == [] and vals(st, "A", "p", 60) == [1]
    assert st.verify() and st.rebuild().fingerprint() == st.fingerprint()


def test_an_exposed_value_retold_over_a_disputed_holder_does_not_settle_the_dispute():
    # before: the copy was admitted while p's 1 was disputed by q's 2: usable(x, 60) True, the key read [1]
    km = solvi.Knowledge()
    st = km.store
    x = km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="p", valid=(0, None))
    km.tell({"s": "A", "r": "p", "o": 2}, source="outcome", by="q", valid=(0, None))
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="vendor", valid=(50, None))
    st.retract_source("vendor")
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="mirror", valid=(50, None))
    assert st.pending() == [x]
    assert not st.usable(x, valid_at=60) and vals(st, "A", "p", 60) == []


def test_a_copy_held_by_a_version_that_holds_is_admitted_unchecked():
    # the strict re-check keeps the old answer where the holder really holds (it won by rank)
    km = solvi.Knowledge()
    st = km.store
    q = km.tell({"s": "A", "r": "p", "o": 2}, source="spec", by="q", valid=(0, None))
    x = km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="p", valid=(0, None))
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="vendor", valid=(50, None))
    st.retract_source("vendor")
    km.tell({"s": "A", "r": "p", "o": 1}, source="outcome", by="mirror", valid=(50, None))
    assert st.status(q) == "refuted" and st.status(x) == "active" and st.pending() == []
    assert st.state[x].get("unchecked_claims")
    st.retract(x, only_from="p")
    assert st.pending() == [x]


def test_a_derivation_over_premises_of_different_time_types_does_not_break_the_store():
    # before: snapshot() and periods(d) raised TypeError for the whole store; add() without own valid raised ValueError
    st = KnowledgeStore()
    a = st.add("fact", {"s": "a", "r": "x", "o": 1}, source="outcome", by="w", valid=(0, 10))
    b = st.add("fact", {"s": "b", "r": "x", "o": 1}, source="outcome", by="w", valid=("2024-01-01", None))
    d = st.add("fact", {"s": "d", "r": "x", "o": 2}, source="spec", by="rule", derived_from=[a, b], valid=(0, 5))
    assert st.periods(d) == [[0, 5]] and st.usable(d, valid_at=3) and not st.usable(d, valid_at=7)
    assert sorted(f["body"]["s"] for f in st.snapshot()["facts"]) == ["a", "b", "d"]
    n = len(st.journal)
    e = st.add("fact", {"s": "e", "r": "x", "o": 2}, source="spec", by="rule", derived_from=[a, b])
    assert e is None and len(st.journal) == n + 1
    assert st.journal[-1]["op"] == "refused" and "do not compare" in st.journal[-1]["why"]
    assert st.verify()


def test_a_deep_derivation_dag_reads_its_periods_in_linear_time():
    # before: usable(valid_at=) and periods() walked every path: 283 ms at depth 16, doubling per level
    st = KnowledgeStore()
    layer = [st.add("fact", {"s": s, "r": "p", "o": 1}, source="outcome", by="w", valid=(0, 100)) for s in "AB"]
    for d in range(16):
        layer = [st.add("fact", {"s": f"L{d}", "r": f"n{j}", "o": 1}, source="spec", by="rule",
                        derived_from=list(layer), inferred=True) for j in range(2)]
    top = layer[0]
    st.usable(top, valid_at=50)
    t0 = time.perf_counter()
    assert st.usable(top, valid_at=50) and not st.usable(top, valid_at=150)
    assert st.periods(top) == [[0, 100]]
    assert time.perf_counter() - t0 < 0.1


def test_taking_back_a_duplicate_does_not_make_the_writers_next_publication_pend():
    # before: the cut duplicate counted as a withdrawal from ed, and ed's third publication waited for a re-test
    km = solvi.Knowledge()
    st = km.store
    a = km.tell({"s": "PM", "r": "inc", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev1")
    km.tell({"s": "PM", "r": "inc", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev3")
    st.retract(a, assertion=st.item(a)["valid"]["claims"][1]["at"], why="duplicate")
    km.tell({"s": "PM", "r": "inc", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev5")
    assert st.pending() == [] and st.status(a) == "active"
    assert not any(c.get("pending") for c in st.item(a)["valid"]["claims"])


def test_a_lone_assertion_taken_back_is_still_a_withdrawal():
    # the duplicate rule needs a standing claim of the same writer: without one the copy still waits
    km = solvi.Knowledge()
    st = km.store
    a = km.tell({"s": "PM", "r": "inc", "o": "P"}, source="outcome", by="ed", valid=(16, None), evidence="rev1")
    km.tell({"s": "PM", "r": "inc", "o": "P"}, source="outcome", by="ed", valid=(0, 10), evidence="old")
    st.retract(a, assertion=st.item(a)["valid"]["claims"][0]["at"], why="wrong")
    km.tell({"s": "PM", "r": "inc", "o": "P"}, source="outcome", by="mirror", valid=(16, None))
    assert st.pending() == [a]


def test_one_version_told_with_two_untils_reads_the_latest_claims_until():
    # the latest claim of a version (one item, one since) decides its until: a correction, as one validity per item was
    st = KnowledgeStore()
    a = st.add("fact", {"s": "A", "r": "p", "o": 1}, source="outcome", by="w", valid=(0, 10))
    st.add("fact", {"s": "A", "r": "p", "o": 1}, source="outcome", by="w", valid=(0, 20))
    assert [c["valid"] for c in st.item(a)["valid"]["claims"]] == [[0, 10], [0, 20]]
    assert st.periods(a) == [[0, 20]] and vals(st, "A", "p", 15) == [1]
    st2 = KnowledgeStore()
    b = st2.add("fact", {"s": "A", "r": "p", "o": 1}, source="outcome", by="w", valid=(0, 20))
    st2.add("fact", {"s": "A", "r": "p", "o": 1}, source="outcome", by="w", valid=(0, 10))
    assert st2.periods(b) == [[0, 10]] and vals(st2, "A", "p", 15) == []


def test_taking_back_one_assertion_is_its_own_journal_op():
    # before: a "retract" entry with "assertions", which a 1.3 store read as a whole retraction; now an op it refuses
    st = KnowledgeStore()
    a = st.add("fact", {"s": "A", "r": "p", "o": 1}, source="outcome", by="w", valid=(0, 10))
    st.add("fact", {"s": "A", "r": "p", "o": 1}, source="outcome", by="v", valid=(0, 10))
    st.retract(a, assertion=st.item(a)["valid"]["claims"][1]["at"])
    r = st.journal[-1]
    assert r["op"] == "retract_assertions" and r["assertions"] == [1] and st.status(a) == "active"
    assert st.verify() and st.rebuild().fingerprint() == st.fingerprint()
    assert len(st.rebuild(upto=len(st.journal) - 1).item(a)["valid"]["claims"]) == 2
