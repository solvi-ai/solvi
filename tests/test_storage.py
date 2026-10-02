"""TraceStorage: stored responses load back and replay, queries on both backends, a hash chain across stored records that
catches edits, deletions, reordering and a cut-off tail, the deprecated journal= as a JSONL store, and provenance questions over the store."""
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
    assert len(c) == 1 and c[0]["question"] == "approve" and c[0]["answer"] == "yes" and c[0]["init"]["amount"] == 500
    #                                                     True is stored as the answer it means (yes / no question)
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


def renamed_build():
    """The catalog of build() after a refactoring: `words` was renamed to `tokens`."""
    cat = Catalog()

    @cat.fn
    def tokens(text):
        return set(text.lower().split())

    @cat.check(hard=True, then={"approve": False})
    def known_customer(customer):
        return customer != "blocked"

    @cat.fn(model=Scorer(), options=["low", "high"])
    def risk(tokens):
        return Decision("high", {"high": 0.9, "low": 0.1}) if "urgent" in tokens else Decision("low", {"high": 0.2, "low": 0.8})

    @cat.rule("approve")
    def approve(amount, risk) -> bool:
        return amount < 100 and risk == "low"

    @cat.rule("urgent")
    def urgent(tokens) -> bool:
        return "urgent" in tokens

    return cat, [Question("approve", "Approve?", Answer.yes_no(), checkpoints=["known_customer"]),
                 Question("urgent", "Urgent?", Answer.yes_no())]


def test_a_renamed_part_is_a_replay_verdict_not_a_key_error(filled):
    """A trace replayed against a catalog where a part was renamed: mismatches of kind missing_part / missing_input and a
    summary that says the data is intact — not a KeyError, and not what a damaged record looks like."""
    from solvi.runtime import Mismatch
    _, store, _, resps = filled
    cat, qs = renamed_build()
    rep = resps[0].trace.replay(cat)                       # used to raise KeyError: 'words'
    assert not rep["ok"] and rep["catalog"] == "changed"
    kinds = {m.kind for m in rep["mismatches"]}
    assert all(isinstance(m, Mismatch) for m in rep["mismatches"]) and kinds == {"missing_part", "missing_input"}
    assert ("words", "missing_part") in {(m[1], m.kind) for m in rep["mismatches"]}
    assert rep["kinds"]["missing_part"] == 1 and rep["summary"] == "data intact, catalog changed (parts missing)"
    step, name, why = rep["mismatches"][0]                 # still the plain triple
    assert rep["mismatches"][0] == (step, name, why) and "not in the catalog" in rep["mismatches"][0][2]
    bad = store.replay_all(System(cat, qs))
    assert len(bad) == len(STATES) and all(b["summary"] == "data intact, catalog changed (parts missing)" for b in bad)
    assert all(m[1] not in ("load", "replay") for b in bad for m in b["mismatches"])


def test_replay_summary_tells_damaged_data_from_a_changed_catalog_or_model(filled):
    import copy
    import pickle
    from solvi.runtime import Mismatch
    _, store, s, resps = filled
    ok = resps[0].trace.replay(s)
    assert ok["ok"] and "summary" not in ok and "kinds" not in ok          # nothing to summarize
    cat, qs = build(threshold=40)                          # a rule changed: the data is intact, a step does not recompute
    rep = resps[0].trace.replay(cat)
    assert {m.kind for m in rep["mismatches"]} == {"recompute"} and rep["summary"] == "data intact, catalog changed"
    cat, qs = build(version="2")                           # the model changed
    rep = resps[0].trace.replay(cat)
    assert {m.kind for m in rep["mismatches"]} == {"model_changed"} and rep["kinds"] == {"model_changed": 1}
    assert rep["summary"] == "data intact, model changed"
    bad = store.replay_all(System(cat, qs))
    assert [b["kinds"] for b in bad] == [{"model_changed": 1}] * 3 and bad[0]["catalog"] == "same"   # the code is the same
    tr = copy.deepcopy(resps[0].trace)                     # a value edited after the run, hashes left as they were
    r = next(x for x in tr.records if x.name == "answer:approve")
    r.value = not r.value
    rep = tr.replay(s)
    assert "integrity" in rep["kinds"] and rep["summary"].startswith("data damaged")
    m = rep["mismatches"][0]
    assert m.to_dict() == {"step": m[0], "name": m[1], "reason": m[2], "kind": m.kind}
    assert json.loads(json.dumps(m)) == list(m)            # in JSON: the triple, as before
    m2 = pickle.loads(pickle.dumps(m))
    assert m2 == m and m2.kind == m.kind and copy.deepcopy(m).kind == m.kind
    assert Mismatch(1, "x", "why").kind == "recompute"


def test_replay_all_reports_a_record_it_cannot_load_as_an_error(filled):
    kind, store, s, _ = filled
    recs = _jsonl_lines(store) if kind == "jsonl" else None
    if kind != "jsonl":
        pytest.skip("the stored body is edited through the JSONL file")
    del recs[1]["response"]["trace"]                       # a stored response without its trace
    _rehash(recs, 1)
    _jsonl_write(store, recs)
    with open(store.head_path, "w") as fh:
        json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
    bad = JSONLStorage(store.path).replay_all(s)
    assert [b["seq"] for b in bad] == [1] and bad[0]["mismatches"][0][1] == "load"
    assert bad[0]["kinds"] == {"error": 1} and bad[0]["summary"] == "replay failed (no verdict on the data)"


class FirstLine:
    """A decider stand-in that votes for the option named on the first line it reads: sensitive to the order of the keys
    of a dict input, as a real decider is (the text it reads lists them in their order)."""
    model_id = "test/first-line"

    def fingerprint(self):
        return "first-line-1"

    def logits(self, items):
        import numpy as np
        out = []
        for it in items:
            z = np.array([4.0 * (o == it.text.split("\n")[0].split(":")[0].strip()) for o in it.options])
            out.append(np.stack([z, z - 1.0], 1))
        return out


@pytest.mark.parametrize("kind", ["jsonl", "sqlite"])
def test_a_stored_trace_keeps_the_order_of_dict_keys_so_model_steps_replay(kind, tmp_path):
    """The store wrote its JSON with sorted keys: a dict a decider had read came back in another order, the decider read
    another text on replay and a sound trace did not replay ("value 'status' ≠ recomputed 'amount'")."""
    from solvi.decide import DecideModel
    store = make_store(kind, tmp_path)
    m = DecideModel(FirstLine(), meta={"format": "test", "temperature": 1.0})
    cat = Catalog()

    @cat.fn
    def situation(amount, status):
        return {"status": status, "amount": amount, "nested": {"z": 1, "a": 2}}      # a computed dict, keys not sorted

    opts = ["amount", "status", "zeta"]
    qs = [m.decision("computed", "Which field comes first?", "situation", opts).question(cat),
          m.decision("given", "Which field comes first?", "order", opts).question(cat)]     # reads a dict of the input
    s = System(cat, qs, storage=store)
    res = s.ask({"amount": 5, "status": "new", "order": {"zeta": 1, "amount": 2}})
    assert res["computed"].answer == "status" and res["given"].answer == "zeta"
    assert store.verify()["ok"] and store.replay_all(s) == []
    back = store.get(res.stored_id)
    assert list(back.trace.init["order"]) == ["zeta", "amount"]
    sit = next(r.value for r in back.trace.records if r.name == "situation")
    assert list(sit) == ["status", "amount", "nested"] and list(sit["nested"]) == ["z", "a"]
    reopened = make_store(kind, tmp_path)                                           # and after a restart
    assert reopened.verify()["ok"] and reopened.replay_all(s) == []
    assert reopened.head() == store.head()
    assert store.record(res.stored_id)["v"] == 2
    if kind == "jsonl":                             # the same record as solvi ≤ 0.7.1 stored it: format 1, keys sorted
        recs = [dict(d, v=1) for d in _jsonl_lines(store)]
        _jsonl_write(store, _rehash(recs))                                          # (_jsonl_write sorts the keys)
        with open(store.head_path, "w") as fh:
            json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
        old = JSONLStorage(store.path)
        assert old.verify()["ok"]
        (bad,) = old.replay_all(s)
        assert bad["kinds"] == {"recompute": 2} and "wrote dict keys sorted" in bad["note"]
        assert "trust_models=True" in bad["note"] and old.replay_all(s, trust_models=True) == []


def test_verify_on_a_store_that_is_being_written_to(filled):
    """A record appended between verify's reading of the head and of the records is not damage (it used to be: "the
    stored head says 5 records, the log has 4" / "records were appended outside the store")."""
    kind, store, s, _ = filled
    snapshot = store._snapshot

    def snapshot_then_a_writer_appends():
        snap = snapshot()
        s.ask(STATES[0])
        s.ask(STATES[1])
        return snap

    store._snapshot = snapshot_then_a_writer_appends
    v = store.verify()
    assert v["ok"], v["problems"]
    del store._snapshot
    assert store.verify()["ok"] and len(store) == len(STATES) + 2


def test_verify_still_reports_records_appended_outside_the_store(filled):
    kind, store, s, _ = filled
    last = (_jsonl_lines(store) if kind == "jsonl" else _sql_bodies(store))[-1]
    extra = dict(last, seq=last["seq"] + 1, prev=last["hash"])              # a well-formed record the head does not know
    extra["hash"] = record_hash(extra)
    extra["id"] = extra["hash"][:16]
    if kind == "jsonl":
        with open(store.path, "a") as fh:
            fh.write(json.dumps(extra) + "\n")
    else:
        store._insert(extra)
    v = store.verify()
    assert not v["ok"] and any("appended outside the store" in why for *_, why in v["problems"])


def test_several_writers_of_one_jsonl_file(tmp_path):
    """Two store objects on one path (two services, or a second System opened on the journal): each append continues the
    chain from what the other wrote. It used to fork the chain without an error — both kept their own count and last hash."""
    path = tmp_path / "shared.jsonl"
    a, b = JSONLStorage(path), JSONLStorage(path)
    ids = []
    for i in range(5):
        ids.append(a._append({"v": 1, "kind": "teach", "i": i})["id"])
        ids.append(b._append({"v": 1, "kind": "teach", "i": i})["id"])
    v = a.verify()
    assert v["ok"] and v["count"] == 10 and len(a) == len(b) == 10 and a.head() == b.head()
    assert [d["seq"] for d in _jsonl_lines(a)] == list(range(10)) and all(b.record(i) is not None for i in ids)
    os.replace(path, tmp_path / "moved.jsonl")                              # the file replaced by a shorter one: reread
    c = JSONLStorage(path)
    c._append({"v": 1, "kind": "teach", "i": 0})
    assert len(a) == 1 and a._append({"v": 1, "kind": "teach", "i": 1})["seq"] == 1


def test_jsonl_opens_from_its_head_without_reading_every_record(tmp_path, monkeypatch):
    path = tmp_path / "long.jsonl"
    with open(path, "w") as fh:
        fh.write(json.dumps({"old": "a 0.5 journal line"}) + "\n")            # before the chain: legacy
    full = JSONLStorage(path)
    ids = [full._append({"v": 1, "kind": "teach", "i": i})["id"] for i in range(20)]
    read = []
    monkeypatch.setattr(JSONLStorage, "_sync", lambda self, fh, _f=JSONLStorage._sync: (read.append(1), _f(self, fh))[1])
    fast = JSONLStorage(path, index=False)
    assert read == [] and fast.head() == full.head() and len(fast) == 20     # nothing read at open
    assert fast._append({"v": 1, "kind": "teach", "i": 20})["seq"] == 20
    assert fast.record(ids[3])["i"] == 3 and fast.verify() == full.verify() and fast.verify()["legacy"] == 1
    assert full._append({"v": 1, "kind": "teach", "i": 21})["seq"] == 21     # the other object caught up with it
    os.remove(str(path) + ".head")                                           # no head to trust: the file is read
    again = JSONLStorage(path, index=False)
    assert len(again) == 22 and again._append({"v": 1, "kind": "teach", "i": 22})["seq"] == 22
    assert again.verify()["ok"] and again.verify()["legacy"] == 1


def test_processes_writing_one_jsonl_file_at_once(tmp_path):
    from solvi import storage
    if storage.fcntl is None:
        pytest.skip("no advisory file locks on this system: one writing process at a time")
    path = tmp_path / "shared.jsonl"
    code = ("import sys, time\nfrom solvi import JSONLStorage\nst = JSONLStorage(sys.argv[1])\n"
            "while time.time() < float(sys.argv[3]):\n    time.sleep(0.001)\n"
            "for i in range(150):\n    st._append({'v': 1, 'kind': 'teach', 'who': sys.argv[2], 'i': i})\n")
    start = str(__import__("time").time() + 1.5)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    procs = [subprocess.Popen([sys.executable, "-c", code, str(path), w, start], env=env, stderr=subprocess.PIPE)
             for w in "abc"]
    errs = [p.communicate()[1] for p in procs]
    assert all(p.returncode == 0 for p in procs), errs
    store = JSONLStorage(path)
    v = store.verify()
    assert v["ok"] and v["count"] == 450, v["problems"][:3]
    who = [d["who"] for d in _jsonl_lines(store)]
    assert {w: who.count(w) for w in "abc"} == {"a": 150, "b": 150, "c": 150}
    assert len(set(who[:150])) > 1                                          # they did write at the same time


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
    with pytest.warns(DeprecationWarning, match=r"System\(journal=\) is deprecated: use storage=JSONLStorage\(path\)"):
        s = System(cat, qs, journal=str(path))
    assert isinstance(s.storage, JSONLStorage) and s.storage.path == str(path)
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
    with pytest.raises(TypeError, match="got both journal= and storage="):
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


def test_jsonl_append_after_a_cut_short_last_line_does_not_glue(tmp_path):
    cat, qs = build()
    p = tmp_path / "j.jsonl"
    s = System(cat, qs, storage=JSONLStorage(p))
    s.ask(STATES[0])
    s.ask(STATES[1])
    raw = p.read_bytes()
    last = raw.rstrip(b"\n").rsplit(b"\n", 1)[1]
    p.write_bytes(raw[:len(raw) - len(last) // 2 - 1])            # a crash in the middle of writing the last record
    st = JSONLStorage(p)
    assert len(st) == 1
    s2 = System(cat, qs, storage=st)
    rid = s2.ask(STATES[2]).stored_id
    assert st.get(rid) is not None and len(st) == 2
    again = JSONLStorage(p)                                         # a fresh reader sees the new record, not a glued line
    assert len(again) == 2 and again.get(rid).stored_id == rid
    assert [x.seq for x in again.iter()] == [0, 1]
    v = again.verify()
    assert not v["ok"] and any("not readable JSON" in why for *_, why in v["problems"])
    assert v["count"] == 2


def test_redact_erases_a_records_content_and_the_chain_still_verifies(filled):
    """A person's data must go. Deleting or editing a stored record breaks the chain; redact() removes the content, keeps
    the record's place and hash, and writes the erasure into the chain."""
    kind, store, s, resps = filled
    sig, head = store.signature(), store.head()
    target = resps[1].stored_id
    before = store.record(target)
    assert "urgent pay now" in json.dumps(before)
    rid = store.redact(target, by="dpo", note="erasure request 17")
    left = store.record(target)
    assert "urgent pay now" not in json.dumps(left) and "response" not in left and "meta" not in left
    assert (left["id"], left["hash"], left["seq"], left["prev"]) == (before["id"], before["hash"], before["seq"], before["prev"])
    assert left["redacted"]["by"] == "dpo" and left["answers"] == before["answers"]
    red = store.record(rid)
    assert red["kind"] == "redaction" and red["of"] == target and red["of_hash"] == before["hash"] and red["note"] == "erasure request 17"
    v = store.verify(anchor=head, signature=sig)                              # the head and signature taken before
    assert v["ok"], v["problems"]
    assert v["count"] == len(STATES) + 1 and len(store) == len(STATES) + 1
    assert [x.id for x in store.iter()] == [r.stored_id for i, r in enumerate(resps) if i != 1]      # passed over
    assert [x.id for x in store.iter(redacted=True)][1] == target and store.replay_all(s) == []
    assert target not in [x.id for x in store.query(question="approve")]
    with pytest.raises(ValueError, match="was redacted"):
        store.get(target)
    with pytest.raises(ValueError, match="already redacted"):
        store.redact(target)
    with pytest.raises(ValueError, match="cannot be redacted"):
        store.redact(rid)
    reopened = make_store(kind, store.path.parent if hasattr(store.path, "parent") else __import__("pathlib").Path(store.path).parent)
    assert reopened.verify()["ok"] and len(reopened) == len(STATES) + 1
    s.ask(STATES[0])                                                          # the store goes on
    assert store.verify()["ok"] and len(store) == len(STATES) + 2
    rid2 = store.redact(resps[2].stored_id, keep_answers=False)
    left2 = store.record(resps[2].stored_id)
    assert "answers" not in left2 and set(left2["redacted"]) == {"by", "note", "content", "answers"}     # their digests stay
    assert store.verify(anchor=head, signature=sig)["ok"] and rid2


def test_content_removed_outside_redact_is_reported(filled):
    kind, store, s, resps = filled
    if kind != "jsonl":
        pytest.skip("the stored body is edited through the JSONL file")
    recs = _jsonl_lines(store)
    recs[1] = {k: v for k, v in recs[1].items() if k not in ("response", "meta")}
    recs[1]["redacted"] = {"by": "nobody", "note": None, "content": "00" * 32}     # no redaction record
    _jsonl_write(store, recs)
    v = JSONLStorage(store.path).verify()
    assert not v["ok"] and any("without a redaction record" in why for *_, why in v["problems"])


def _forge(store, recs):
    _jsonl_write(store, recs)
    return JSONLStorage(store.path)


def test_what_is_left_of_a_redacted_record_is_still_verified(filled):
    """The hash of a record covers its lasting fields, the digest of its answers and the digest of its content, so an
    erased record verifies like any other. Its kept answer edited by hand, its time changed, another "who" in the mark,
    a record passed off as redacted with other answers: each is reported — with nothing kept elsewhere."""
    from solvi.storage import record_body
    kind, store, s, resps = filled
    if kind != "jsonl":
        pytest.skip("the stored body is edited through the JSONL file")
    head, sig = store.head(), store.signature()
    target = resps[1].stored_id
    store.redact(target, by="dpo", note="request 17")
    assert store.verify(anchor=head, signature=sig)["ok"]
    good = _jsonl_lines(store)
    q = next(iter(good[1]["answers"]))

    recs = json.loads(json.dumps(good))
    recs[1]["answers"][q][0] = "forged"
    v = _forge(store, recs).verify()
    assert not v["ok"] and any("redacted record edited" in why for *_, why in v["problems"])
    assert not _forge(store, recs).verify(anchor=head, signature=sig)["ok"]

    recs = json.loads(json.dumps(good))
    recs[1]["time"] = 5.0
    assert not _forge(store, recs).verify()["ok"]

    recs = json.loads(json.dumps(good))
    recs[1]["redacted"]["by"] = "nobody"
    v = _forge(store, recs).verify()
    assert not v["ok"] and any("differs from its redaction record" in why for *_, why in v["problems"])

    # another record made to look redacted: content dropped, other answers written, a redaction record appended by hand
    recs = json.loads(json.dumps(good))
    victim = recs[0]
    digests = record_body(victim)
    recs[0] = {k: v for k, v in victim.items() if k in ("v", "kind", "seq", "time", "prev", "hash", "id", "init_hash", "catalog",
                                                         "records", "flow", "producers", "models", "guards", "safeguards")}
    recs[0]["answers"] = {q: ["forged", 1.0, "ok"]}
    recs[0]["redacted"] = {"by": None, "note": None, "content": digests["#content"]}
    fake = {"v": 2, "kind": "redaction", "of": victim["id"], "of_hash": victim["hash"], "by": None, "note": None,
            "seq": len(recs), "time": 9.0, "prev": recs[-1]["hash"]}
    fake["hash"] = record_hash(fake)
    fake["id"] = fake["hash"][:16]
    recs.append(fake)
    _jsonl_write(store, recs)
    json.dump({"count": len(recs), "hash": fake["hash"]}, open(str(store.path) + ".head", "w"))
    v = JSONLStorage(store.path).verify()
    assert not v["ok"] and any(seq == 0 and "redacted record edited" in why for seq, _, why in v["problems"])

    recs = json.loads(json.dumps(good))               # ... or by claiming the old format, whose hash cannot be recomputed
    recs[1]["v"] = 1
    v = _forge(store, recs).verify()
    assert not v["ok"] and any("format 1 after a record of format 2" in why for *_, why in v["problems"])


def test_a_format_1_record_redacted_is_listed_as_unverified(tmp_path):
    """A record written by solvi 0.7.1 has one flat hash over content that redact removes: what is left cannot be
    recomputed, and verify says so instead of calling it verified."""
    cat, qs = build()
    p = tmp_path / "old.jsonl"
    st = JSONLStorage(p, clock=Clock())
    s = System(cat, qs, storage=st)
    rs = [s.ask(dict(x)) for x in STATES]
    recs, prev = [json.loads(x) for x in p.read_text().splitlines()], ""
    for r in recs:                                     # the same records as 0.7.1 wrote them: format 1, one flat hash
        r["v"], r["prev"] = 1, prev
        r["hash"] = record_hash(r)
        r["id"] = r["hash"][:16]
        prev = r["hash"]
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    json.dump({"count": len(recs), "hash": prev}, open(str(p) + ".head", "w"))
    old = JSONLStorage(p, clock=Clock())
    assert old.verify()["ok"] and "unverified" not in old.verify()
    old.redact(recs[1]["id"], by="dpo")
    v = JSONLStorage(p).verify()
    assert v["ok"] and [seq for seq, *_ in v["unverified"]] == [1] and len(rs) == len(STATES)


def test_a_json_line_without_a_hash_inside_a_jsonl_chain_is_passed_over_and_reported_once(tmp_path):
    """A line another tool appended to the file (or a hand edit) used to make iter / query / report / replay_all raise a
    bare KeyError: 'id'; verify listed it as five problems with an empty id and every record after it as "chain broken"."""
    store = make_store("jsonl", tmp_path, Clock())
    cat, qs = build()
    s = System(cat, qs, storage=store)
    for st in STATES[:2]:
        s.ask(st)
    sig = store.signature()
    for line in ({"note": "x"}, {"hash": 5, "id": "abc"}):
        with open(store.path, "a") as fh:
            fh.write(json.dumps(line) + "\n")
    again = JSONLStorage(store.path, catalog=s, clock=Clock(2_000.0))
    assert [x.seq for x in again.iter(None)] == [0, 1] and len(again.query()) == 2 and len(again) == 2
    assert again.replay_all(s) == [] and "2" in again.report()
    v = again.verify(signature=sig)
    assert len(v["problems"]) == 2 and all(p[2].startswith(f"record at position {i} is not a record of this store:")
                                           for i, p in zip((2, 3), v["problems"]))
    assert v["count"] == 2 and not v["ok"] and v["signature"]["ok"]
    System(cat, qs, storage=again).ask(STATES[2])                       # the chain goes on after the foreign lines
    v = JSONLStorage(store.path).verify()
    assert v["count"] == 3 and len(v["problems"]) == 2 and all("not a record of this store" in p[2] for p in v["problems"])
    assert [x.seq for x in JSONLStorage(store.path, catalog=s).iter()] == [0, 1, 2]


def test_the_audit_of_a_stored_decision_without_its_catalog_knows_which_checks_are_hard(tmp_path):
    """Opened without the catalog, a stored decision's audit printed its hard check as "(soft)"."""
    cat = Catalog()

    @cat.check(hard=True, then={"alert": "no"})
    def enough_history(n):
        return n >= 3

    @cat.check
    def calm(n):
        return n < 100

    @cat.rule("alert")
    def alert(n, calm):
        return "yes" if n > 10 and calm else "no"
    qs = [Question("alert", "Alert?", Answer.yes_no(), checkpoints=["enough_history"])]
    store = JSONLStorage(tmp_path / "a.jsonl")
    s = System(cat, qs, storage=store)
    for n in (20, 1):
        s.ask({"n": n})
    bare = JSONLStorage(tmp_path / "a.jsonl")                           # no catalog: the flow's parts are stand-ins
    passed, failed = [x.response() for x in bare.iter()]
    for res in (passed, failed, store.get(passed.stored_id)):
        hard = {c["name"]: c["hard"] for c in res.audit("alert").checks}
        assert hard["enough_history"] is True and hard.get("calm", False) is False
    text = str(passed.audit("alert"))
    assert "enough_history = True (hard)" in text and "calm = True (soft)" in text
    assert passed.report(format="data")["answers"][0] is not None
    old = json.loads(json.dumps(store.record(passed.stored_id)["response"]))   # as solvi ≤ 0.7.1 stored it: no hardness
    for st in old["flow"]["steps"]:
        st.pop("hard", None)
    from solvi.system import Response
    text = str(Response.model_validate(old).audit("alert"))
    assert "enough_history = True (hard or soft: not recorded)" in text and "(soft)" not in text
    old_failed = json.loads(json.dumps(store.record(failed.stored_id)["response"]))
    for st in old_failed["flow"]["steps"]:
        st.pop("hard", None)
    assert "enough_history = False (hard, decides the answer)" in str(Response.model_validate(old_failed).audit("alert"))


def test_query_finds_a_yes_no_answer_given_as_a_bool_on_every_backend(filled):
    """query(question="approve", answer=True) found 0 records where answer="yes" found them."""
    _, store, s, _ = filled
    by_text = [x.seq for x in store.query(question="approve", answer="yes")]
    assert by_text and [x.seq for x in store.query(question="approve", answer=True)] == by_text
    assert [x.seq for x in store.query(answer=False)] == [x.seq for x in store.query(answer="no")]
    store.catalog = None                                            # without the System: a bool is "yes" / "no" too
    assert [x.seq for x in store.query(question="approve", answer=True)] == by_text


def test_every_store_closes_and_is_a_context_manager_and_the_extension_is_read_in_any_case(tmp_path):
    import sqlite3

    from solvi.storage import open_storage
    with open_storage(tmp_path / "a.DB") as store:
        assert isinstance(store, SQLiteStorage)
        System(*build(), storage=store).ask(STATES[0])
    with pytest.raises(sqlite3.ProgrammingError):                   # closed: the connection is gone
        store.head()
    with JSONLStorage(tmp_path / "a.jsonl") as js:
        System(*build(), storage=js).ask(STATES[0])
    js.close()                                                      # nothing held open: closing twice is fine
    assert len(JSONLStorage(tmp_path / "a.jsonl")) == 1
