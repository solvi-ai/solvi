"""The stored answers are bound to the trace: replay with the System derives the answers the trace gives and compares.
An answer edited after the run — in a response's JSON, or in a store whose every hash was recomputed — used to replay ok."""
import json

from solvi import Answer, Catalog, Question, Response, System
from solvi.storage import JSONLStorage, record_hash

OVER = {"amount": 9000, "limit": 100}


def paying():
    cat = Catalog()

    @cat.check(hard=True, then={"pay": "no"})
    def cap(amount, limit) -> bool:
        return amount <= limit

    @cat.rule("pay")
    def pay(amount):
        return "yes"

    return cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]), requires=["cap"])]


def flip(o):
    """Every `pay` result in a dumped response: no [forced] → yes [ok]."""
    if isinstance(o, dict):
        if isinstance(o.get("pay"), dict) and "answer" in o["pay"]:
            o["pay"].update(answer="yes", status="ok")
        for v in o.values():
            flip(v)
    elif isinstance(o, list):
        for v in o:
            flip(v)


def test_an_answer_edited_in_the_json_does_not_replay():
    cat, qs = paying()
    s = System(cat, qs)
    res = s.ask(OVER)
    assert (res["pay"].answer, res["pay"].status) == ("no", "forced")
    rep = res.trace.replay(s)
    assert rep["ok"] and rep["answers"] == "same"
    d = json.loads(res.to_json())
    flip(d)
    forged = Response.from_json(json.dumps(d))
    assert (forged["pay"].answer, forged["pay"].status) == ("yes", "ok")
    rep = forged.trace.replay(s)
    assert not rep["ok"] and rep["answers"] == "differ" and rep["kinds"] == {"answer": 1}
    assert rep["summary"].startswith("data damaged: a stored answer")
    (m,) = rep["mismatches"]
    assert m[1] == "answer:pay" and "'yes' [ok]" in m[2] and "'no' [forced]" in m[2]


def test_a_bare_catalog_replays_the_trace_and_says_the_answers_are_unchecked():
    cat, qs = paying()
    res = System(cat, qs).ask(OVER)
    rep = res.trace.replay(cat)
    assert rep["ok"] and rep["answers"] == "unchecked"


def test_a_changed_question_is_not_damaged_data():
    """The same catalog under a question with another checkpoint list: the stored answer is not the one the trace gives
    now, and the verdict says the system changed, not the data."""
    cat, qs = paying()
    res = System(cat, qs).ask(OVER)
    loose = System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "no"]))])
    cat.parts["cap"].then = {}                         # the check no longer governs `pay`
    rep = res.trace.replay(loose, res.flow)
    assert all(m.kind != "answer" for m in rep["mismatches"])


def test_the_store_replay_catches_an_answer_edited_with_every_hash_recomputed(tmp_path):
    cat, qs = paying()
    p = tmp_path / "d.jsonl"
    s = System(cat, qs, storage=JSONLStorage(p))
    for amount in (5, 9000, 50):
        s.ask({"amount": amount, "limit": 100})
    assert s.storage.replay_all(s) == []
    recs = [json.loads(x) for x in p.read_text().splitlines()]
    flip(recs[1])
    recs[1]["answers"]["pay"][0], recs[1]["answers"]["pay"][2] = "yes", "ok"      # the record's summary of the answers too
    prev = recs[0]["hash"]
    for r in recs[1:]:                                 # the whole chain after the edit, rebuilt
        r["prev"] = prev
        r["hash"] = record_hash(r)
        r["id"] = r["hash"][:16]
        prev = r["hash"]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    head = tmp_path / "d.jsonl.head"
    if head.exists():
        head.write_text(json.dumps({"count": len(recs), "hash": recs[-1]["hash"]}))
    st = JSONLStorage(p)
    assert st.verify()["ok"]                           # the chain is whole: only an anchor kept elsewhere would tell
    (bad,) = st.replay_all(s)
    assert bad["seq"] == 1 and bad["kinds"] == {"answer": 1} and bad["summary"].startswith("data damaged")
