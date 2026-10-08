"""Statuses as they are (docs/trace_spec.md §6): "no data for a hard check" and "the hard check could not be evaluated"
both abstain; the response tells them apart (guard, checks), the store's answer triple does not — a proposed change."""
import pytest
from semantics_helpers import MemoryStorage

from solvi import Answer, Catalog, Question, System


def _system():
    cat = Catalog()

    @cat.check(hard=True, then={"pay": "no"})
    def cap(amount, limit) -> bool:
        if limit < 0:
            raise RuntimeError("cannot check")
        return amount <= limit

    @cat.rule("pay")
    def pay(amount):
        return "yes"

    return System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]), requires=["cap"])])


@pytest.mark.parametrize("state,guard,check", [
    ({"amount": 5}, None, None),                                   # no data: the check cannot be planned
    ({"amount": 5, "limit": -1}, "hard_check", "error"),           # cannot be checked: the check raised
    ({"amount": 5, "limit": None}, "hard_check", "error"),
    ({"amount": 500, "limit": 100}, "hard_check", "failed"),
    ({"amount": 5, "limit": 100}, None, "passed"),
])
def test_the_response_tells_no_data_from_cannot_be_checked(state, guard, check):
    s = _system()
    res = s.ask(state)
    r = res["pay"]
    assert r.guard == guard
    assert [c.status for c in res.checks] == ([] if check is None else [check])
    if check in (None, "error"):
        assert (r.answer, r.status) == (None, "abstain")
    st = MemoryStorage(s)
    st.save(res)
    assert st.rows[0]["answers"]["pay"][2] == r.status        # the store's triple keeps the status only
    assert res.trace.replay(s, res.flow)["ok"]
