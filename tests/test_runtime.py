"""Execution and independent replay: a clean trace passes; tampering (naive, or with recomputed hashes) is caught and located."""
import copy
import random

import pytest

from solvi import Answer, Catalog, Question, System
from examples_loader import load
from solvi.runtime import vhash

I = load("03_invoices")


@pytest.fixture(scope="module")
def trace():
    init, _ = I.make(random.Random(11), 11)
    return System(I.cat, [q for q in I.QUESTIONS if q.name != "risk"]).ask(init).trace


def _victim(t):
    return next(r for r in t.records if isinstance(r.value, (int, float)) and not isinstance(r.value, bool) and r.error is None)


def test_clean_replay(trace):
    rep = trace.replay(I.cat)
    assert rep["ok"] and rep["mismatches"] == []


def test_naive_tamper_found_and_located(trace):
    t = copy.deepcopy(trace)
    r = _victim(t)
    r.value = r.value + 1
    rep = t.replay(I.cat)
    assert not rep["ok"]
    assert rep["mismatches"][0][0] == r.step


def test_consistent_tamper_found_and_located(trace):
    t = copy.deepcopy(trace)
    r = _victim(t)
    r.value = r.value + 1
    prev = r.prev
    for x in t.records[r.step - 1:]:          # the attacker recomputed the hashes of the record and of the whole chain after it
        x.prev = prev
        x.hash = vhash(x.body())
        prev = x.hash
    rep = t.replay(I.cat)
    assert not rep["ok"]
    assert rep["mismatches"][0][0] == r.step


def test_quotes_point_into_document():
    init, _ = I.make(random.Random(12), 12)
    r = System(I.cat, [q for q in I.QUESTIONS if q.name == "approve"]).ask(init)
    for rec in r.trace.records:
        if rec.kind == "extract" and rec.error is None:
            s, e, src = rec.quote
            assert 0 <= s <= e <= len(init[src])


def test_replay_catches_deleted_step_changed_input_and_fake_error():
    import copy as _copy
    s = System(I.cat, I.QUESTIONS)
    init, _ = I.make(random.Random(3), 0)
    r = s.ask(init)
    assert r.trace.replay(I.cat, r.flow)["ok"]
    t = _copy.deepcopy(r.trace)                               # delete a middle step and rebuild the chain
    del t.records[2]
    prev = t.init_hash
    for rec in t.records:
        rec.prev = prev
        rec.hash = vhash(rec.body())
        prev = rec.hash
    assert not t.replay(I.cat, r.flow)["ok"]
    t = _copy.deepcopy(r.trace)                               # swap the recorded input
    t.init = {**t.init, "today": t.init["today"].replace(year=2030)}
    assert not t.replay(I.cat)["ok"]
    t = _copy.deepcopy(r.trace)                               # claim a step failed
    t.records[0].error, t.records[0].value = "ValueError: fake", None
    assert not t.replay(I.cat)["ok"]


def test_rehashed_value_on_a_failed_step_is_caught():
    from solvi import Answer, Catalog, Question

    cat = Catalog()

    @cat.fn
    def ratio(a, b):
        return a / b                          # fails on b = 0

    @cat.rule("ok")
    def ok(ratio):
        return ratio < 1

    t = System(cat, [Question("ok", "OK?", Answer.yes_no())]).ask({"a": 1, "b": 0}).trace
    r = next(x for x in t.records if x.name == "ratio")
    assert r.error is not None
    r.value = 0.5                             # the attacker fills in the failed step and re-hashes the chain
    prev = r.prev
    for x in t.records[r.step - 1:]:
        x.prev = prev
        x.hash = vhash(x.body())
        prev = x.hash
    rep = t.replay(cat)
    assert not rep["ok"] and rep["mismatches"][0][0] == r.step


# ---------------------------------------------------------------- early_exit=False: the whole flow, recorded
def _proposal():
    cat = Catalog()

    @cat.fn
    def draft(text):
        return text.upper()

    @cat.check(hard=True, then={"ok": "no"})
    def not_empty(text):
        return bool(text.strip())

    @cat.fn
    def length(draft):
        return len(draft)

    @cat.rule("ok")
    def ok(length):
        return "yes" if length > 3 else "no"
    return cat, [Question("ok", "ok?", Answer.yes_no(), requires=["not_empty", "draft"])]


def test_a_failed_hard_check_skips_the_rest_by_default_and_early_exit_false_computes_it_anyway():
    """A decision forced by a hard check had no rule values in its record, and a part listed in `checkpoints` was missing
    from res.values; System.ask had no way to ask for the whole flow."""
    import asyncio
    from solvi.runtime import Trace
    cat, qs = _proposal()
    s = System(cat, qs)
    res = s.ask({"text": " "})
    assert res["ok"].status == "forced" and "draft" not in res.values and res.trace.early_exit is True
    assert [n for n, _ in res.trace.skipped] == ["draft", "length", "answer:ok"]
    full = s.ask({"text": " "}, early_exit=False)
    assert (full["ok"].answer, full["ok"].status, full["ok"].guard) == ("no", "forced", "hard_check")
    assert full.values["draft"] == " " and full.values["length"] == 1 and full.trace.skipped == []
    assert [r.name for r in full.trace.records] == ["not_empty", "draft", "length", "answer:ok"]
    assert full.trace.early_exit is False and full.trace.replay(s, full.flow)["ok"]
    for other in (System(cat, qs, early_exit=False).ask({"text": " "}), asyncio.run(s.aask({"text": " "}, early_exit=False))):
        assert [r.hash for r in other.trace.records] == [r.hash for r in full.trace.records]
    assert System(cat, qs, early_exit=False).ask({"text": " "}, early_exit=True).trace.skipped != []
    d = full.to_dict()
    assert d["trace"]["early_exit"] is False and "early_exit" not in res.to_dict()["trace"]
    back = Trace.model_validate(d["trace"], catalog=cat)
    assert back.early_exit is False and back.replay(s, full.flow)["ok"]
    cut = Trace.model_validate({**d["trace"], "records": d["trace"]["records"][:2], "skipped": [["length", "x"], ["answer:ok", "x"]]},
                               catalog=cat)                      # a trace of a whole flow cannot have skipped steps
    rep = cut.replay(cat, full.flow)
    assert [m.kind for m in rep["mismatches"]] == ["flow", "flow"]


def test_a_stored_decision_made_with_early_exit_false_holds_the_rule_values_and_replays(tmp_path):
    from solvi import JSONLStorage
    cat, qs = _proposal()
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = System(cat, qs, storage=store, early_exit=False)
    res = s.ask({"text": " "})
    got = store.get(res.stored_id)
    assert got.values["length"] == 1 and got.trace.early_exit is False and got["ok"].status == "forced"
    assert store.replay_all(s) == [] and store.verify()["ok"]


# ---------------------------------------------------------------- vhash of arrays and plain objects
def test_vhash_tells_long_numpy_arrays_apart_and_hashes_an_array_as_the_list_it_is_stored_as():
    """repr elides a long array ("..."): two 5000-vectors that differed in the middle hashed the same."""
    import numpy as np
    a = np.arange(5000, dtype=float)
    b = a.copy()
    b[2500] = -1.0
    assert vhash(a) != vhash(b) and vhash(a) == vhash(a.copy()) == vhash(a.tolist())
    assert vhash(np.int64(5)) == vhash(5) and vhash(np.float64(0.5)) == vhash(0.5)
    assert vhash({"m": np.eye(2)}) == vhash({"m": [[1.0, 0.0], [0.0, 1.0]]})


def test_a_part_that_returns_a_plain_object_replays():
    """An object with the default repr was hashed by its memory address, so its step never recomputed on a live trace."""
    class Thing:
        def __init__(self, v):
            self.v = v
    cat = Catalog()

    @cat.fn
    def thing(x):
        return Thing(x)

    @cat.rule("q")
    def q(thing):
        return thing.v > 1
    s = System(cat, [Question("q", "?", Answer.yes_no())])
    res = s.ask({"x": 2})
    assert res["q"].answer == "yes" and res.trace.replay(s, res.flow)["ok"]
    assert vhash(Thing(1)) == vhash(Thing(1)) != vhash(Thing(2))
    assert vhash(test_a_part_that_returns_a_plain_object_replays) == vhash(test_a_part_that_returns_a_plain_object_replays)
    loop = Thing(0)
    loop.v = loop
    assert isinstance(vhash(loop), str)                              # attributes that point back: no crash


def test_trace_value_reads_a_computed_fact_then_a_given_one_then_missing():
    from solvi.runtime import MISSING
    cat = Catalog()

    @cat.fn
    def double(x):
        return x * 2

    @cat.fn
    def broken(x):
        raise ValueError("no")

    @cat.rule("big")
    def big(double, broken):
        return "yes"
    t = System(cat, [Question("big", "Big?", Answer.yes_no())]).ask({"x": 21}).trace
    with pytest.warns(DeprecationWarning, match=r"res.values\[name\]"):
        assert t.value("double") == 42 and t.value("x") == 21
        assert t.value("broken") is MISSING and t.value("nothing") is MISSING


def test_a_question_changed_in_place_changes_the_questions_fingerprint():
    """The fingerprint was cached by the question objects' ids: `q.min_confidence = 0.99` changed the answers but not
    the fingerprint the traces record."""
    cat = Catalog()

    @cat.rule("big")
    def big(x):
        return "yes" if x > 1 else "no"
    q = Question("big", "Big?", Answer.yes_no())
    s = System(cat, [q])
    before = s.fingerprint()["questions"]
    assert s.ask({"x": 2}).trace.fingerprint["questions"] == before
    s.questions["big"].min_confidence = 0.99
    assert s.fingerprint()["questions"] != before and s.ask({"x": 2}).trace.fingerprint["questions"] != before
    s.questions["big"].answer.options.append("maybe")
    assert len({before, s.fingerprint()["questions"]}) == 2 and s.fingerprint()["questions"] != \
        System(cat, [Question("big", "Big?", Answer.yes_no(), min_confidence=0.99)]).fingerprint()["questions"]
