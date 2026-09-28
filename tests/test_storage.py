"""TraceStorage: stored responses load back and replay, queries on both backends, a hash chain across stored records that
catches edits, deletions, reordering and a cut-off tail, journal= as a JSONL store, and provenance questions over the store."""
import datetime
import json
import os
import sqlite3
import subprocess
import sys

import pytest

from solvi import Answer, Catalog, Decision, JSONLStorage, Question, SQLiteStorage, System
from solvi.storage import record_hash


class Scorer:
    model_id = "scorer"

    def __init__(self, version="1"):
        self.version = version


def build(threshold=100, version="1"):
    cat = Catalog()

    @cat.fn
    def words(text):
        return set(text.lower().split())

    @cat.check(hard=True, then={"approve": False})
    def known_customer(customer):
        return customer != "blocked"

    @cat.fn(model=Scorer(version), options=["low", "high"])
    def risk(words):
        return Decision("high", {"high": 0.9, "low": 0.1}) if "urgent" in words else Decision("low", {"high": 0.2, "low": 0.8})

    @cat.rule("approve")
    def approve(amount, risk) -> bool:
        return amount < threshold and risk == "low"

    @cat.rule("urgent")
    def urgent(words) -> bool:
        return "urgent" in words

    return cat, [Question("approve", "Approve?", Answer.yes_no(), checkpoints=["known_customer"]),
                 Question("urgent", "Urgent?", Answer.yes_no())]


STATES = [{"text": "please pay", "amount": 50, "customer": "ann"},
          {"text": "urgent pay now", "amount": 50, "customer": "bob"},
          {"text": "pay", "amount": 500, "customer": "ann"},
          {"text": "pay", "amount": 10, "customer": "blocked"}]


class Clock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        self.t += 10.0
        return self.t


def make_store(kind, tmp_path, clock=None):
    if kind == "jsonl":
        return JSONLStorage(tmp_path / "decisions.jsonl", clock=clock)
    return SQLiteStorage(tmp_path / "decisions.db", clock=clock)


@pytest.fixture(params=["jsonl", "sqlite"])
def filled(request, tmp_path):
    store = make_store(request.param, tmp_path, Clock())
    cat, qs = build()
    s = System(cat, qs, storage=store)
    resps = [s.ask(st) for st in STATES]
    return request.param, store, s, resps


def test_save_get_and_replay(filled):
    _, store, s, resps = filled
    assert len(store) == 4 and all(r.stored_id for r in resps)
    for r in resps:
        back = store.get(r.stored_id)
        assert {q: x.answer for q, x in back.results.items()} == {q: x.answer for q, x in r.results.items()}
        assert [x.hash for x in back.trace.records] == [x.hash for x in r.trace.records]
        assert back.trace.replay(s, back.flow)["ok"]
    assert store.verify()["ok"]
    assert store.replay_all(s) == []
    with pytest.raises(KeyError):
        store.get("nope")


def test_query(filled):
    _, store, s, resps = filled
    ids = [r.stored_id for r in resps]

    def q(**kw):
        return [x.id for x in store.query(**kw)]
    assert q() == ids
    assert q(question="approve", answer="yes") == [ids[0]]
    assert q(question="urgent", answer="yes") == [ids[1]]
    assert q(answer="yes") == [ids[0], ids[1]]
    assert q(question="approve", status="forced") == [ids[3]]
    assert q(safeguard="hard_check") == [ids[3]]
    assert q(safeguard="hard_check", question="urgent") == []
    assert q(model="scorer") == ids[:3]                       # the blocked customer never reached the model
    fp = store.record(ids[0])["models"][0]["fp"]
    assert q(model=fp) == ids[:3] and q(model="other") == []
    t = [store.record(i)["time"] for i in ids]
    assert q(since=t[1], until=t[3]) == ids[1:3]
    assert q(since=datetime.datetime.fromtimestamp(t[2])) == ids[2:]
    assert [x.answers["approve"] for x in store.query(question="approve")] == ["yes", "no", "no", "no"]


def test_query_same_on_both_backends(tmp_path):
    got = {}
    for kind in ("jsonl", "sqlite"):
        store = make_store(kind, tmp_path, Clock())
        cat, qs = build()
        s = System(cat, qs, storage=store)
        for st in STATES * 2:
            s.ask(st)
        got[kind] = [[x.seq for x in store.query(**kw)] for kw in
                     ({}, {"answer": "no"}, {"question": "approve", "answer": None}, {"status": "ok"},
                      {"safeguard": "hard_check"}, {"model": "Scorer"}, {"until": 1045.0})]
    assert got["jsonl"] == got["sqlite"]


def test_teach_is_stored_and_chained(filled):
    _, store, s, _ = filled
    s.teach("approve", STATES[2], True)
    c = store.corrections()
    assert len(c) == 1 and c[0]["question"] == "approve" and c[0]["answer"] is True and c[0]["init"]["amount"] == 500
    assert len(list(store.iter())) == 4 and len(store) == 5
    assert store.verify()["ok"]


def test_corrections_carry_their_source_and_untrusted_sources_are_refused(filled):
    from solvi.storage import UntrustedLabel
    _, store, s, _ = filled
    rid = s.ask(STATES[0]).stored_id
    s.teach("approve", STATES[0], False, source="outcome", by="ledger", of=rid)
    s.teach("approve", STATES[2], True, by="ann")
    c = store.corrections()
    assert [(x["source"], x["by"], x["of"]) for x in c] == [("outcome", "ledger", rid), ("human", "ann", None)]
    assert "source" not in store.record(c[1]["id"])                  # a human correction is stored as in 0.6
    for bad in ("model", "system", "self", None):
        with pytest.raises(UntrustedLabel):
            s.teach("approve", STATES[0], True, source=bad)
        with pytest.raises(UntrustedLabel):
            store.save_correction("approve", STATES[0], True, source=bad)
    assert len(store.corrections()) == 2 and store.verify()["ok"]


# --- tampering: a helper per backend that edits the stored records directly
def _jsonl_lines(store):
    with open(store.path) as fh:
        return [json.loads(x) for x in fh if x.strip()]


def _jsonl_write(store, recs):
    with open(store.path, "w") as fh:
        for d in recs:
            fh.write(json.dumps(d, ensure_ascii=False, sort_keys=True) + "\n")


def _sql_bodies(store):
    return [json.loads(b) for (b,) in store.db.execute("SELECT body FROM records ORDER BY seq")]


def _rehash(recs, start=0):
    """What an attacker with write access does: recompute the hashes (and links) from `start` on."""
    prev = recs[start - 1]["hash"] if start else ""
    for d in recs[start:]:
        d["prev"] = prev
        d["hash"] = record_hash(d)
        d["id"] = d["hash"][:16]
        prev = d["hash"]
    return recs


def _edit_answer(d):
    d["answers"]["approve"][0] = not d["answers"]["approve"][0]
    d["response"]["results"]["approve"]["answer"] = d["answers"]["approve"][0]


def test_edit_is_detected(filled):
    kind, store, _, _ = filled
    if kind == "jsonl":
        recs = _jsonl_lines(store)
        _edit_answer(recs[1])
        _jsonl_write(store, recs)
    else:
        d = _sql_bodies(store)[1]
        _edit_answer(d)
        store.db.execute("UPDATE records SET body = ? WHERE seq = 1", (json.dumps(d),))
    v = store.verify()
    assert not v["ok"] and v["problems"][0][0] == 1 and "edited" in v["problems"][0][2]


def test_edit_with_rehash_breaks_the_next_link(filled):
    kind, store, _, _ = filled
    recs = _jsonl_lines(store) if kind == "jsonl" else _sql_bodies(store)
    _edit_answer(recs[1])
    _rehash(recs[:2], 1)                                   # only this record re-hashed: the next one still names the old hash
    if kind == "jsonl":
        _jsonl_write(store, recs)
    else:
        store.db.execute("UPDATE records SET body = ?, hash = ?, id = ? WHERE seq = 1",
                         (json.dumps(recs[1]), recs[1]["hash"], recs[1]["id"]))
    v = store.verify()
    assert not v["ok"] and any(p[0] == 2 and "chain broken" in p[2] for p in v["problems"])


def test_full_rewrite_is_caught_by_head_and_anchor(filled):
    kind, store, _, _ = filled
    anchor = store.head()
    recs = _jsonl_lines(store) if kind == "jsonl" else _sql_bodies(store)
    _edit_answer(recs[1])
    _rehash(recs, 1)                                       # the whole chain after the edit re-hashed
    if kind == "jsonl":
        _jsonl_write(store, recs)
        v = store.verify()
        assert not v["ok"] and "replaced" in v["problems"][-1][2]
        with open(store.head_path, "w") as fh:            # ... and the stored head rewritten too
            json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
    else:
        store.db.execute("DELETE FROM records")
        store.db.execute("DELETE FROM answers")
        store.db.execute("DELETE FROM safeguards")
        store.db.execute("DELETE FROM models")
        for d in recs:
            store._insert(d)
        store.db.execute("UPDATE meta SET value = ? WHERE key = 'head'",
                         (json.dumps({"count": len(recs), "hash": recs[-1]["hash"]}),))
    assert store.verify()["ok"]                            # consistent inside: only a head kept elsewhere tells
    v = store.verify(anchor=anchor)
    assert not v["ok"] and "rewritten" in v["problems"][-1][2]


def test_delete_is_detected(filled):
    kind, store, _, _ = filled
    if kind == "jsonl":
        recs = _jsonl_lines(store)
        del recs[1]
        _jsonl_write(store, recs)
    else:
        store.db.execute("DELETE FROM records WHERE seq = 1")
    v = store.verify()
    assert not v["ok"] and any("chain broken" in p[2] for p in v["problems"])


def test_reorder_is_detected(filled):
    kind, store, _, _ = filled
    if kind == "jsonl":
        recs = _jsonl_lines(store)
        recs[1], recs[2] = recs[2], recs[1]
        _jsonl_write(store, recs)
    else:
        b = {s: body for s, body in store.db.execute("SELECT seq, body FROM records")}
        store.db.execute("UPDATE records SET body = ? WHERE seq = 1", (b[2],))
        store.db.execute("UPDATE records SET body = ? WHERE seq = 2", (b[1],))
    v = store.verify()
    assert not v["ok"] and any("chain broken" in p[2] for p in v["problems"])


def test_truncated_tail_is_detected(filled):
    kind, store, _, _ = filled
    anchor = store.head()
    if kind == "jsonl":
        _jsonl_write(store, _jsonl_lines(store)[:-1])
    else:
        store.db.execute("DELETE FROM records WHERE seq = 3")
        for t in ("answers", "safeguards", "models"):
            store.db.execute(f"DELETE FROM {t} WHERE seq = 3")
    v = store.verify()
    assert not v["ok"] and "removed from the end" in v["problems"][-1][2]
    if kind == "jsonl":                                    # the head rewritten as well: the anchor still catches it
        recs = _jsonl_lines(store)
        with open(store.head_path, "w") as fh:
            json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
    else:
        h = _sql_bodies(store)[-1]["hash"]
        store.db.execute("UPDATE meta SET value = ? WHERE key = 'head'", (json.dumps({"count": 3, "hash": h}),))
    assert store.verify()["ok"]
    v = store.verify(anchor=anchor)
    assert not v["ok"] and "removed from the end" in v["problems"][-1][2]


def test_sqlite_index_edit_is_detected(tmp_path):
    store = make_store("sqlite", tmp_path, Clock())
    cat, qs = build()
    s = System(cat, qs, storage=store)
    for st in STATES:
        s.ask(st)
    store.db.execute("UPDATE answers SET answer = '\"yes\"' WHERE seq = 2 AND question = 'approve'")
    v = store.verify()
    assert not v["ok"] and "answers index" in v["problems"][0][2]


def test_rehashed_value_edit_is_caught_by_replay_all(filled):
    """A value inside a stored trace changed and every hash re-computed (trace and store): only re-computing the step tells."""
    from solvi.runtime import vhash
    kind, store, s, _ = filled
    recs = _jsonl_lines(store) if kind == "jsonl" else _sql_bodies(store)
    tr = recs[2]["response"]["trace"]
    r = next(x for x in tr["records"] if x["name"] == "answer:approve")
    r["value"] = True
    recs_t = tr["records"]
    for i in range(recs_t.index(r), len(recs_t)):          # the trace's own chain re-hashed from the edited step on
        x = recs_t[i]
        x["prev"] = recs_t[i - 1]["hash"] if i else tr["init_hash"]
        x["hash"] = vhash({**_body(x), "value": vhash(x["value"])})
    recs[2]["records"] = [[x["step"], x["name"], x["hash"]] for x in tr["records"]]
    recs[2]["answers"]["approve"][0] = True
    recs[2]["response"]["results"]["approve"]["answer"] = True
    _rehash(recs, 2)
    if kind == "jsonl":
        _jsonl_write(store, recs)
        with open(store.head_path, "w") as fh:
            json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
        store = JSONLStorage(store.path)
    else:
        store.db.execute("DELETE FROM records")
        for t in ("answers", "safeguards", "models"):
            store.db.execute(f"DELETE FROM {t}")
        for d in recs:
            store._insert(d)
        store.db.execute("UPDATE meta SET value = ? WHERE key = 'head'",
                         (json.dumps({"count": len(recs), "hash": recs[-1]["hash"]}),))
    assert store.verify()["ok"]
    bad = store.replay_all(s)
    assert [b["seq"] for b in bad] == [2]
    assert any(m[1] == "answer:approve" for m in bad[0]["mismatches"])


def _body(r):
    b = {k: r[k] for k in ("step", "kind", "name", "inputs", "quote", "error", "prev")}
    for k in ("model", "probs", "extra"):
        if r.get(k) is not None:
            b[k] = r[k]
    if r.get("provenance") not in (None, "computed", "quoted"):
        b["provenance"] = r["provenance"]
    return b


def test_replay_all_after_a_rule_change(filled):
    _, store, _, _ = filled
    cat, qs = build(threshold=40)                          # the rule changed since these decisions
    bad = store.replay_all(System(cat, qs))
    assert [b["seq"] for b in bad] == [0]                  # the only decision where amount < threshold mattered
    assert bad[0]["mismatches"][0][1] == "answer:approve"
    cat, qs = build(version="2")                           # the model changed
    bad = store.replay_all(System(cat, qs))
    assert [b["seq"] for b in bad] == [0, 1, 2]
    assert all("model changed" in b["mismatches"][0][2] for b in bad)


def test_quarantine_and_forget(filled):
    _, store, _, resps = filled
    ids = [r.stored_id for r in resps]
    hit = store.quarantine("risk")                          # the risk model found to be wrong
    assert [h["id"] for h in hit] == ids[:3]
    assert set(hit[0]["questions"]) == {"approve"}
    assert hit[0]["questions"]["approve"]["path"] == ["risk", "answer:approve"]
    assert [h["id"] for h in store.quarantine("risk", "high")] == [ids[1]]
    assert [h["id"] for h in store.quarantine("words")] == ids   # urgent reads words everywhere; approve via risk
    w = store.quarantine("words")[0]["questions"]
    assert w["approve"]["path"] == ["words", "risk", "answer:approve"] and w["urgent"]["path"] == ["words", "answer:urgent"]
    hc = store.quarantine("customer", "blocked")            # a hard check decided that answer
    assert [h["id"] for h in hc] == [ids[3]] and hc[0]["questions"]["approve"]["path"] == ["customer", "known_customer"]
    assert store.quarantine("nothing") == []
    f = store.forget("amount", 500)
    assert [d["id"] for d in f["dependent"]] == [ids[2]] and f["stored"] == [] and f["deleted"] == 0
    f = store.forget("amount", 10)                          # the blocked customer: stored, but no answer rests on it
    assert f["dependent"] == [] and f["stored"] == [ids[3]]
    assert len(store) == 4 and store.verify()["ok"]         # a report only


def test_journal_is_a_jsonl_store(tmp_path):
    path = tmp_path / "journal.jsonl"
    with open(path, "w") as fh:                           # a line written by solvi 0.5 before the chain
        fh.write(json.dumps({"init_hash": "x", "answers": {"approve": [True, 1.0, "ok"]}, "flow": [], "records": []}) + "\n")
    cat, qs = build()
    s = System(cat, qs, journal=str(path))
    assert isinstance(s.storage, JSONLStorage) and s.journal == path
    r = s.ask(STATES[0])
    s.teach("approve", STATES[0], False)
    lines = [json.loads(x) for x in open(path)]
    assert len(lines) == 3
    assert {"init_hash", "answers", "flow", "records"} <= set(lines[1])           # the 0.5 keys, same meaning
    assert lines[1]["answers"]["approve"] == ["yes", 0.8, "ok"] and lines[1]["init_hash"] == r.trace.init_hash
    assert lines[1]["records"] == [[x.step, x.name, x.hash] for x in r.trace.records]
    assert {"teach", "init", "answer"} <= set(lines[2]) and lines[2]["teach"] == "approve"
    again = JSONLStorage(path)
    v = again.verify()
    assert v["ok"] and v["legacy"] == 1 and v["count"] == 2
    assert [x.id for x in again.iter()] == [r.stored_id]
    with pytest.raises(ValueError):
        System(cat, qs, journal=str(path), storage=again)


def test_storage_from_a_path(tmp_path):
    cat, qs = build()
    assert isinstance(System(cat, qs, storage=str(tmp_path / "a.db")).storage, SQLiteStorage)
    assert isinstance(System(cat, qs, storage=tmp_path / "a.jsonl").storage, JSONLStorage)
    r = System(cat, qs, storage=str(tmp_path / "a.db")).ask(STATES[0], store=False)
    assert r.stored_id is None


def test_two_sqlite_writers_share_one_chain(tmp_path):
    cat, qs = build()
    a = System(cat, qs, storage=SQLiteStorage(tmp_path / "shared.db"))
    b = System(cat, qs, storage=SQLiteStorage(tmp_path / "shared.db"))
    for st in STATES:
        a.ask(st)
        b.ask(st)
    v = a.storage.verify()
    assert v["ok"] and v["count"] == 8


def test_typed_values_come_back(tmp_path):
    cat = Catalog()

    @cat.fn
    def age_days(opened: datetime.date, today: datetime.date) -> int:
        return (today - opened).days

    @cat.rule("old")
    def old(age_days: int) -> bool:
        return age_days > 30

    store = JSONLStorage(tmp_path / "t.jsonl")
    s = System(cat, [Question("old", "Old?")], storage=store)
    r = s.ask({"opened": datetime.date(2024, 1, 1), "today": datetime.date(2024, 3, 1)})
    back = JSONLStorage(tmp_path / "t.jsonl", catalog=s).get(r.stored_id)
    assert back.trace.init["opened"] == datetime.date(2024, 1, 1)
    assert store.replay_all(s) == []
    assert [h["id"] for h in store.quarantine("opened", datetime.date(2024, 1, 1))] == [r.stored_id]


CODE = """
import sys
sys.path.insert(0, {tests!r})
from test_storage import build, STATES
from solvi import JSONLStorage, System
store = JSONLStorage({path!r}, clock=lambda: 1000.0)
cat, qs = build()
s = System(cat, qs)
for st in STATES:
    r = s.ask(st)
    r.ms, r.trace.timings = 0.0, {{}}                # run times differ between runs; everything else must not
    store.save(r)
print(store.head()["hash"])
"""


def test_stored_hashes_do_not_depend_on_the_process(tmp_path):
    heads = set()
    for seed in ("0", "1", "4242"):
        path = str(tmp_path / f"p{seed}.jsonl")
        env = dict(os.environ, PYTHONHASHSEED=seed)
        out = subprocess.run([sys.executable, "-c", CODE.format(tests=os.path.dirname(__file__), path=path)],
                             capture_output=True, text=True, env=env, check=True)
        heads.add(out.stdout.strip())
    assert len(heads) == 1


def test_sqlite_file_is_plain_sqlite(filled):
    kind, store, _, _ = filled
    if kind != "sqlite":
        return
    con = sqlite3.connect(store.path)
    n = con.execute("SELECT count(*) FROM answers WHERE question = 'approve' AND answer = ?", ('"no"',)).fetchone()[0]
    assert n == 3


# --- fixes before 0.7
@pytest.mark.parametrize("kind", ["jsonl", "sqlite"])
def test_corrections_and_meta_read_back_non_finite_floats(tmp_path, kind):
    import math
    st = JSONLStorage(tmp_path / "c.jsonl") if kind == "jsonl" else SQLiteStorage(tmp_path / "c.db")
    st.save_correction("q", {"limit": math.inf, "x": 1.0}, -math.inf, meta={"t": math.inf})
    c = st.corrections()[0]
    assert c["init"] == {"limit": math.inf, "x": 1.0} and c["answer"] == -math.inf
    assert next(st.iter("teach")).meta == {"t": math.inf}
    assert st.forget("limit", math.inf)["stored"] == [c["id"]]
    assert st.verify()["ok"]

