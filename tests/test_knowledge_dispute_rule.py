"""Three narrow fixes to the two-writer dispute on a key with `valid`: a value another writer still holds is disputed,
not refuted, when its last writer changes its mind; two unbounded sinces are the same since; an item pending re-test
takes no side on its key."""
import solvi

SC = {"valid_from": None, "valid_to": None}


def test_another_writers_support_keeps_the_dispute():
    km = solvi.Knowledge()
    st = km.store
    reg = km.tell({"s": "vat@EU", "r": "vat", "o": 23}, source="outcome", by="regulator", valid=(0, None))
    km.tell({"s": "vat@EU", "r": "vat", "o": 23}, source="outcome", by="wiki", valid=(0, None))
    new = km.tell({"s": "vat@EU", "r": "vat", "o": 21}, source="outcome", by="wiki", valid=(0, None))
    assert (st.status(reg), st.status(new), len(st.disputes())) == ("disputed", "disputed", 1)


def test_one_writer_changing_its_own_value_still_refutes():
    km = solvi.Knowledge()
    st = km.store
    old = km.tell({"s": "vat@EU", "r": "vat", "o": 23}, source="outcome", by="wiki", valid=(0, None))
    new = km.tell({"s": "vat@EU", "r": "vat", "o": 21}, source="outcome", by="wiki", valid=(0, None))
    assert (st.status(old), st.status(new), len(st.disputes())) == ("refuted", "active", 0)


def test_two_unbounded_sinces_from_two_writers_dispute():
    st = solvi.Knowledge(writers={"a": "outcome", "b": "outcome"}).store
    x = st.add("fact", {"s": "x", "r": "f", "o": "A"}, SC, source="outcome", by="a", valid=[None, None])
    y = st.add("fact", {"s": "x", "r": "f", "o": "B"}, SC, source="outcome", by="b", valid=[None, None])
    assert (st.status(x), st.status(y), len(st.disputes())) == ("disputed", "disputed", 1)


def test_keys_without_valid_resolve_as_before():
    st = solvi.Knowledge(writers={"a": "outcome", "b": "outcome"}).store
    x = st.add("fact", {"s": "x", "r": "f", "o": "A"}, SC, source="outcome", by="a")
    y = st.add("fact", {"s": "x", "r": "f", "o": "B"}, SC, source="outcome", by="b")
    assert (st.status(x), st.status(y), len(st.disputes())) == ("refuted", "active", 0)


def test_an_item_pending_retest_takes_no_side():
    km = solvi.Knowledge()
    st = km.store
    km.tell({"s": "A", "r": "p", "o": 50}, source="outcome", by="vendor", valid=(99, None))
    other = km.tell({"s": "A", "r": "p", "o": 84}, source="outcome", by="pricing", valid=(99, None))
    st.retract_source("vendor")
    g = km.tell({"s": "A", "r": "p", "o": 50}, source="outcome", by="partner", valid=(99, None))
    assert (st.status(g), st.pending_retest(g), len(st.disputes())) == ("hypothesis", True, 0)
    assert st.status(other) == "active"
    st.retest(g, ok=True)                               # re-tested: it takes its side again
    assert (st.status(g), st.status(other), len(st.disputes())) == ("disputed", "disputed", 1)
