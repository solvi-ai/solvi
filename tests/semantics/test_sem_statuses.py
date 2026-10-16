"""Statuses (docs/trace_spec.md §6): "no data for a hard check" and "the hard check could not be evaluated" both
abstain, and both the response and the store tell them apart — guard "no_data" with the check listed as "skipped" and
its missing inputs named, against guard "hard_check" with the check "error"."""
import json

import pytest
from semantics_helpers import MemoryStorage, load_schema, read_lines, validate

from solvi import Answer, Catalog, Question, Response, System
from solvi.core.store import JSONLStorage


def _system():
    cat = Catalog()

    @cat.fn
    def ceiling(limit, factor):
        return limit * factor

    @cat.check(hard=True, then={"pay": "no"})
    def cap(amount, limit) -> bool:
        if limit < 0:
            raise RuntimeError("cannot check")
        return amount <= limit

    @cat.check(hard=True, then={"big": "no"})
    def under_ceiling(amount, ceiling) -> bool:
        return amount <= ceiling

    @cat.rule("pay")
    def pay(amount):
        return "yes"

    @cat.rule("big")
    def big(amount):
        return "yes"

    return System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]), requires=["cap"]),
                        Question("big", "Big?", Answer.choice(["yes", "no"]))])


@pytest.mark.parametrize("state,guard,check", [
    ({"amount": 5, "factor": 2}, "no_data", "skipped"),            # no data: the check cannot be planned
    ({"amount": 5, "limit": -1, "factor": 2}, "hard_check", "error"),   # cannot be checked: the check raised
    ({"amount": 5, "limit": None, "factor": 2}, "hard_check", "error"),
    ({"amount": 500, "limit": 100, "factor": 2}, "hard_check", "failed"),
    ({"amount": 5, "limit": 100, "factor": 2}, None, "passed"),
])
def test_the_response_and_the_store_tell_no_data_from_cannot_be_checked(state, guard, check):
    s = _system()
    res = s.ask(state, "pay")
    r = res["pay"]
    assert r.guard == guard
    assert [c.status for c in res.checks if c.name == "cap"] == [check]
    if check in ("skipped", "error"):
        assert (r.answer, r.status) == (None, "abstain")
    if guard == "no_data":
        (c,) = res.checks
        assert (c.name, c.questions, c.hard, c.missing) == ("cap", ["pay"], True, ["limit"])
        assert "missing limit" in c.reason and r.source == "cap" and r.extra == {"no_data": {"cap": ["limit"]}}
    else:
        assert all(c.missing is None and "missing" not in c.to_dict() for c in res.checks)
    st = MemoryStorage(s)
    st.save(res)
    row = st.rows[0]
    assert row["answers"]["pay"][2] == r.status
    assert row["guards"].get("pay") == guard                   # the store keeps the difference in its guards index
    assert st.verify()["ok"] and st.replay_all(s) == []
    assert res.trace.replay(s, res.flow)["ok"]


def test_no_data_names_the_given_inputs_behind_a_computed_fact():
    """A hard check reading a fact that cannot be computed: the missing inputs are the given ones behind it."""
    res = _system().ask({"amount": 5}, "big")
    r = res["big"]
    assert (r.answer, r.status, r.guard) == (None, "abstain", "no_data")
    (c,) = res.checks
    assert (c.name, c.status, c.missing, c.questions) == ("under_ceiling", "skipped", ["factor", "limit"], ["big"])


@pytest.mark.parametrize("record", ["full", "compact"])
def test_no_data_survives_storage_and_json(tmp_path, record):
    s = _system()
    res = s.ask({"amount": 5, "factor": 2})
    back = Response.from_json(res.to_json())
    assert back.checks == res.checks and back["pay"].guard == "no_data"
    assert json.loads(res.to_json())["checks"] == [c.to_dict() for c in res.checks]
    st = JSONLStorage(tmp_path / "s.jsonl", system=s, record=record)
    st.save(res)
    assert st.verify()["ok"] and st.replay_all(s) == []
    assert {q for q, g in st.record(res.stored_id)["guards"].items() if g == "no_data"} == {"pay", "big"}
    if record == "full":
        assert st.get(res.stored_id, s).checks == res.checks
    schema = load_schema()                                     # record schema 2.1.0 describes it (2.2.0 adds "run")
    assert schema["x-schema-version"] in ("2.1.0", "2.2.0")
    for rec in read_lines(st.path):
        assert validate(rec, schema) == []


def test_a_soft_check_without_data_is_not_no_data():
    """Only a hard check that governs the question gives no_data; anything else that cannot be computed abstains with
    guard None ("cannot compute"), as before."""
    cat = Catalog()

    @cat.check
    def known(customer) -> bool:
        return True

    @cat.rule("ok")
    def ok(amount):
        return "yes"

    s = System(cat, [Question("ok", "Ok?", Answer.choice(["yes", "no"]), requires=["known"])])
    r = s.ask({"amount": 1})["ok"]
    assert (r.status, r.guard) == ("abstain", None) and r.why.startswith("cannot compute")
    assert s.ask({"amount": 1}).checks == []


def test_a_no_data_abstention_stored_before_the_guard_still_replays():
    """Stored by solvi 1.0.1 or earlier: guard None, source None, why "cannot compute: cap" — the same abstention, not
    an edit. Another reason in its place is still a mismatch."""
    s = _system()
    res = s.ask({"amount": 5, "factor": 2}, "pay")
    d = json.loads(res.to_json())
    d["results"]["pay"].update(guard=None, source=None, why="cannot compute: cap", extra=None)
    old = Response.from_json(json.dumps(d))
    assert old.trace.replay(s, old.flow)["ok"]
    d["results"]["pay"]["why"] = "cannot compute: something else"
    forged = Response.from_json(json.dumps(d))
    assert forged.trace.replay(s, forged.flow)["kinds"] == {"answer": 1}
