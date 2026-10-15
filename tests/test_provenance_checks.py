"""A passed (or run) hard check is part of what an answer rests on: quarantine / where_is and diff's scope follow it."""
from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage
from solvi.core.store.diff import _deps


def _build():
    cat = Catalog()

    @cat.check(hard=True, then={"approve": False})
    def known_customer(customer):
        return customer != "blocked"

    @cat.rule("approve")
    def approve(amount) -> bool:
        return amount < 100

    return cat, [Question("approve", "Approve?", Answer.yes_no(), requires=["known_customer"])]


def test_quarantine_finds_decisions_where_a_required_hard_check_passed(tmp_path):
    st = JSONLStorage(str(tmp_path / "q.jsonl"))
    cat, qs = _build()
    s = System(cat, qs, storage=st)
    ids = [s.ask({"amount": 50, "customer": c}).stored_id for c in ("ann", "bob", "blocked")]
    hits = st.quarantine("customer")
    assert sorted(ids.index(h["id"]) for h in hits) == [0, 1, 2]
    assert hits[0]["questions"]["approve"]["path"] == ["customer", "known_customer"]
    assert [ids.index(h["id"]) for h in st.quarantine("customer", "bob")] == [1]
    assert len(st.where_is("customer")["dependent"]) == 3


def test_a_soft_check_is_not_a_root():
    cat = Catalog()

    @cat.check(then={"approve": False})
    def nice(mood):
        return mood != "bad"

    @cat.rule("approve")
    def approve(amount) -> bool:
        return amount < 100

    s = System(cat, [Question("approve", "Approve?", Answer.yes_no(), requires=["nice"])])
    r = s.ask({"amount": 5, "mood": "ok"}, store=False)
    assert "mood" not in _deps(r, "approve")


def test_diff_scope_includes_a_passed_hard_check():
    cat, qs = _build()
    r = System(cat, qs).ask({"amount": 50, "customer": "ann"}, store=False)
    assert {"known_customer", "customer"} <= _deps(r, "approve")
