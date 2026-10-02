"""Values JSON cannot carry: the dump records the stdlib type of each (date, Decimal, set, tuple, ...) and a load gives
it back, declared or not; what cannot be restored is a replay verdict of its own (not_restored), never "data damaged"."""
import copy
import datetime
import decimal
import enum
import json
import uuid

import pytest

from solvi import Answer, Catalog, JSONLStorage, Question, SQLiteStorage, System
from solvi.diff import diff
from solvi.runtime import Trace, srepr, vhash
from solvi.schema import from_tree, jsonable, restorable, type_tree
from solvi.system import Response


def quickstart():
    """The README quickstart: parts that read dates without declaring a type."""
    cat = Catalog()

    @cat.fn
    def days_requested(start, end):
        return (end - start).days + 1

    @cat.fn
    def remaining_after(balance, days_requested):
        return balance - days_requested

    @cat.check(hard=True, then={"approve": "reject"})
    def enough_balance(remaining_after):
        return remaining_after >= 0

    @cat.check
    def enough_notice(start, today, days_requested):
        return days_requested < 5 or (start - today).days >= 14

    @cat.rule("approve")
    def approve(enough_notice):
        return "approve" if enough_notice else "needs_manager"
    return cat, [Question("approve", "Approve the leave?", Answer.choice(["approve", "needs_manager", "reject"]),
                          requires=["enough_balance"])]


STATE = {"start": datetime.date(2026, 10, 19), "end": datetime.date(2026, 10, 23), "today": datetime.date(2026, 9, 25),
         "balance": 14}


@pytest.mark.parametrize("kind", ["jsonl", "sqlite"])
def test_the_quickstart_with_untyped_dates_replays_diffs_and_explains_itself_from_a_store(kind, tmp_path):
    """Its stored records used to be reported "data damaged" by replay_all and "2 changed" by diff against the very same
    system, and counterfactual on a loaded response concluded that no change of balance changes the answer."""
    store = JSONLStorage(tmp_path / "q.jsonl") if kind == "jsonl" else SQLiteStorage(tmp_path / "q.db")
    cat, qs = quickstart()
    s = System(cat, qs, storage=store)
    live = [s.ask(dict(STATE)), s.ask(dict(STATE, balance=3))]
    assert [r["approve"].answer for r in live] == ["approve", "reject"]
    assert store.verify()["ok"] and store.replay_all(s) == []
    got = store.get(live[0].stored_id)
    assert got.trace.init == STATE and type(got.trace.init["start"]) is datetime.date
    assert not hasattr(got.trace, "unrestored")
    d = diff(store, s)
    assert d.ok and d.total == 2
    assert got.report(format="data")["replay"]["status"] == "ok"
    cf = got.counterfactual("approve", system=s)
    assert any(c.changes[0].fact == "balance" and c.answer == "reject" for c in cf) and cf.inconclusive is None


def test_a_trace_with_untyped_dates_replays_after_json_without_any_declared_type():
    cat, qs = quickstart()
    res = System(cat, qs).ask(dict(STATE))
    text = res.to_json()
    assert json.loads(text)["trace"]["init_types"] == {"start": "date", "end": "date", "today": "date"}
    assert Trace.from_json(res.trace.to_json()).replay(cat)["ok"]                   # no catalog given to the load
    back = Response.from_json(text, catalog=cat)
    assert back.trace.replay(cat, back.flow)["ok"] and back.values["start"] == STATE["start"]
    assert Response.model_validate(res.model_dump(), catalog=cat).trace.init == STATE   # python mode: nothing to restore


VALUES = [datetime.date(2026, 1, 2), datetime.datetime(2026, 1, 2, 3, 4, 5, 600), datetime.time(3, 4),
          datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.timezone.utc),
          datetime.datetime(2026, 1, 2, 3, 4, tzinfo=datetime.timezone(datetime.timedelta(hours=2))),
          decimal.Decimal("1.50"), uuid.UUID("12345678123456781234567812345678"), {"b", "a"}, frozenset({1, 2}), (1, "a"),
          (), set(), [datetime.date(2026, 1, 2), None], {"when": datetime.date(2026, 1, 2), "n": 1, "tags": {"x"}},
          {"rows": [(1, decimal.Decimal("2")), (3, decimal.Decimal("4"))]}, {(1, 2), (3, 4)},
          [1, "a", {"k": [2.5, None]}], {datetime.date(2026, 1, 2), "a", 3}]


@pytest.mark.parametrize("v", VALUES, ids=[str(i) for i in range(len(VALUES))])
def test_a_stdlib_value_comes_back_from_json_as_it_was(v):
    tree = type_tree(v)
    assert restorable(tree)
    back = from_tree(json.loads(json.dumps(tree)), json.loads(json.dumps(jsonable(v))))
    assert back == v and type(back) is type(v) and vhash(back) == vhash(v) and srepr(back) == srepr(v)
    assert from_tree(tree, v) == v                                                  # already restored: left as it is


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


def test_what_the_dump_cannot_restore_is_marked_so():
    named = datetime.timezone(datetime.timedelta(hours=2), "CEST")                  # the text keeps the offset, not the name
    for v in (Color.RED, object(), {1: "a"}, datetime.datetime(2026, 1, 2, tzinfo=named), [Color.RED],
              {"c": Color.RED, "d": datetime.date(2026, 1, 2)}):
        assert not restorable(type_tree(v))
    assert type_tree([1, "a", {"k": None}]) is None and type_tree("2026-01-02") is None


def _colors():
    cat = Catalog()

    @cat.fn
    def warm(color):
        return color is Color.RED

    @cat.rule("go")
    def go(warm, n) -> bool:
        return warm and n > 0
    return cat, [Question("go", "", None)]


def test_an_untyped_enum_from_a_store_is_not_restored_and_that_is_not_called_damage(tmp_path):
    cat, qs = _colors()
    store = JSONLStorage(tmp_path / "c.jsonl")
    s = System(cat, qs, storage=store)
    res = s.ask({"color": Color.RED, "n": 1})
    assert res["go"].answer == "yes" and res.trace.replay(s)["ok"]
    got = store.get(res.stored_id)
    assert list(got.trace.unrestored) == ["color"] and got.trace.init["color"] == "red"
    (bad,) = store.replay_all(s)
    assert set(bad["kinds"]) == {"not_restored"} and bad["summary"].startswith("not verified: values stored without")
    assert all("color came back from storage as JSON gave it" in m[2] for m in bad["mismatches"])
    d = diff(store, s)
    assert not d.changed and len(d.errors) == 1 and "the stored input was not restored (color" in d.errors[0]["error"]
    assert "1 could not be re-run" in str(d)
    cf = got.counterfactual("go", system=s)                # re-run on "red" the decision says no: nothing is concluded
    assert not cf.found and "color" in cf.inconclusive and "no conclusion" in str(cf) and "color" in cf.not_searched
    assert cf.to_dict()["inconclusive"] == cf.inconclusive
    cold = store.get(s.ask({"color": Color.BLUE, "n": 1}).stored_id)   # on "blue" it still says no, as recorded: the
    cf = cold.counterfactual("go", system=s)                           # other inputs are searched, the lost one is not
    assert cf.inconclusive is None and cf.searched == ["n"] and "color" in cf.not_searched


def test_a_record_stored_without_the_types_is_not_restored_and_an_edited_one_is_still_damage(tmp_path):
    cat, qs = quickstart()
    store = JSONLStorage(tmp_path / "q.jsonl")
    s = System(cat, qs, storage=store)
    d = s.ask(dict(STATE)).to_dict()
    old = copy.deepcopy(d)                                                          # as solvi ≤ 0.7.1 wrote it
    del old["trace"]["init_types"]
    rep = Response.model_validate(old, catalog=s).trace.replay(s)
    assert not rep["ok"] and set(rep["kinds"]) == {"not_restored"} and "no verdict on the data" in rep["summary"]
    assert "its type was not recorded" in rep["mismatches"][0][2]
    edited = copy.deepcopy(d)
    edited["trace"]["init"]["balance"] = 1                                          # every value restored: an edit is damage
    rep = Response.model_validate(edited, catalog=s).trace.replay(s)
    assert "integrity" in rep["kinds"] and rep["summary"].startswith("data damaged")
    edited = copy.deepcopy(d)
    edited["trace"]["init"]["start"] = "2026-10-01"
    rep = Response.model_validate(edited, catalog=s).trace.replay(s)
    assert "integrity" in rep["kinds"] and rep["summary"].startswith("data damaged")
    broken = copy.deepcopy(old)                                                     # a broken link is damage even then
    broken["trace"]["records"][1]["prev"] = "0" * 16
    rep = Response.model_validate(broken, catalog=s).trace.replay(s)
    assert "integrity" in rep["kinds"] and rep["summary"].startswith("data damaged")
