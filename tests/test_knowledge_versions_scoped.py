"""Two shapes on keys with `valid` that the first release of versions in time got wrong. Each test names what the store
did before the change.

1. A derived item whose premise is disputed only on an overlap (the premise told on two periods, disputed on the
   earlier one) was blocked at every time, because the check read the premise's item-level status; it now holds where
   every premise of one of its derivations holds at the asked time.
2. Two writers who each told one value and then another at one since left both values disputed; a writer's later value
   at one since now replaces its own earlier one before the writers are compared."""
import solvi
from solvi.core.knowledge import KnowledgeStore


def _premise_told_twice_disputed_once():
    km = solvi.Knowledge()
    p = km.tell({"s": "t0", "r": "role", "o": "researcher"}, source="outcome", by="hr", valid=(66, None))
    km.tell({"s": "t0", "r": "role", "o": "manager"}, source="outcome", by="crm", valid=(66, 100))
    km.tell({"s": "t0", "r": "role", "o": "engineer"}, source="outcome", by="hr", valid=(120, None))
    assert km.tell({"s": "t0", "r": "role", "o": "researcher"}, source="outcome", by="hr", valid=(187, None)) == p
    d = km.tell({"s": "t0", "r": "access", "o": "lab"}, source="outcome", by="rule", derived_from=[p], inferred=True)
    return km, p, d


def test_a_derived_item_holds_where_its_premise_holds_though_the_premise_is_disputed_on_another_period():
    # before: usable(d, valid_at=240) was False and no read gave "lab" at 240 (the premise's item-level status is
    # "disputed", from the overlap [66, 100))
    km, p, d = _premise_told_twice_disputed_once()
    st = km.store
    assert st.status(p) == "disputed" and st.periods(p) == [[100, 120], [187, None]]
    for t in (110, 240):
        assert st.usable(p, valid_at=t) and st.usable(d, valid_at=t)
    assert [f["body"]["o"] for f in st.snapshot(about="t0", relation="access", valid_at=240)["facts"]] == ["lab"]
    assert solvi.Knowledge.value(km.snapshot(), "t0", "access", valid_at=240) == "lab"


def test_the_derived_item_is_still_blocked_where_its_premise_does_not_hold():
    km, p, d = _premise_told_twice_disputed_once()
    st = km.store
    for t in (80, 150):                                  # 80: the contested overlap; 150: engineer closes researcher
        assert not st.usable(p, valid_at=t) and not st.usable(d, valid_at=t)
        assert st.snapshot(about="t0", relation="access", valid_at=t)["facts"] == []
        assert solvi.Knowledge.value(km.snapshot(), "t0", "access", valid_at=t) is None
    assert st.status(d) == "hypothesis" and not st.usable(d)    # without a time: as the disputed premise itself


def test_a_derived_item_on_a_premise_disputed_everywhere_stays_blocked():
    km = solvi.Knowledge()
    st = km.store
    p = km.tell({"s": "X", "r": "price", "o": 263}, source="outcome", by="pricing", valid=(61, None))
    km.tell({"s": "X", "r": "price", "o": 300}, source="outcome", by="vendor", valid=(61, None))
    d = km.tell({"s": "X", "r": "refund", "o": 200}, source="outcome", by="rule", derived_from=[p], inferred=True)
    assert st.status(p) == "disputed" and st.periods(p) == []
    assert not st.usable(d, valid_at=100) and st.snapshot(about="X", relation="refund", valid_at=100)["facts"] == []


def _four_claims(order):
    ks = KnowledgeStore()
    ids = {}
    for by, o in order:
        ids[o] = ks.add("fact", {"s": "p0", "r": "price", "o": o}, source="outcome", by=by, valid=(143, None))
    return ks, ids


def test_two_writers_who_both_move_to_one_value_at_one_since_leave_that_value():
    # before: both items "disputed" and no fact at 150
    for order in ([("vendor", 301), ("pricing", 301), ("vendor", 666), ("pricing", 666)],
                  [("vendor", 301), ("vendor", 666), ("pricing", 301), ("pricing", 666)],
                  [("pricing", 301), ("vendor", 301), ("pricing", 666), ("vendor", 666)]):
        ks, ids = _four_claims(order)
        assert ks.status(ids[666]) == "active" and ks.status(ids[301]) == "refuted", order
        assert [r["body"]["o"] for r in ks.snapshot(valid_at=150)["facts"]] == [666]


def test_a_writer_still_holding_the_old_value_keeps_the_dispute():
    ks, ids = _four_claims([("vendor", 301), ("pricing", 301), ("vendor", 666)])
    assert ks.status(ids[301]) == ks.status(ids[666]) == "disputed"
    ks, ids = _four_claims([("pricing", 301), ("vendor", 666), ("vendor", 301)])    # the vendor comes back to 301
    assert ks.status(ids[301]) == "active" and ks.status(ids[666]) == "refuted"
