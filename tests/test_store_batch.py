"""A batch of writes: `with store.batch():` / `store.save_many(...)` — one lock, one head write (JSONL) or one commit
(SQL) per batch; the chain, the head and verify are as with single saves."""
import json
import threading

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage, SQLiteStorage


def system(storage=None):
    cat = Catalog()

    @cat.rule("approve")
    def approve(amount) -> bool:
        return amount < 100
    return System(cat, [Question("approve", "Approve?", Answer.yes_no())], storage=storage)


def stores(tmp_path):
    return [JSONLStorage(tmp_path / "d.jsonl"), JSONLStorage(tmp_path / "f.jsonl", fsync=True),
            JSONLStorage(tmp_path / "r.jsonl", rotate_records=7), SQLiteStorage(tmp_path / "d.db")]


def test_a_batch_writes_the_same_chain_and_head(tmp_path):
    for st in stores(tmp_path):
        s = system(st)
        s.ask({"amount": 1})
        with st.batch():
            ids = [s.ask({"amount": i}).stored_id for i in range(20)]
            assert st.get(ids[-1])["approve"].answer == "yes"
            assert getattr(st, "rotates", False) or st.head()["count"] == 21
        more = st.save_many([s.ask({"amount": 500}, store=False) for _ in range(3)], metas=[{"k": i} for i in range(3)])
        assert len(more) == 3 and st.record(more[2])["meta"] == {"k": 2}
        v = st.verify_segments() if getattr(st, "rotates", False) else st.verify()
        assert v["ok"], v
        assert len(set(ids + more)) == 23


def test_the_jsonl_head_is_written_once_per_batch(tmp_path, monkeypatch):
    st = JSONLStorage(tmp_path / "d.jsonl")
    s = system(st)
    writes = []
    real = st._write_head
    monkeypatch.setattr(st, "_write_head", lambda: (writes.append(1), real())[1])
    with st.batch():
        for i in range(50):
            s.ask({"amount": i})
    assert len(writes) == 1
    assert json.loads((tmp_path / "d.jsonl.head").read_text()) == st.head() and st.verify()["ok"]
    s.ask({"amount": 3})                                        # outside a batch: one head write per record, as before
    assert len(writes) == 2


def test_an_exception_rolls_a_sql_batch_back_and_keeps_jsonl_records_with_their_head(tmp_path):
    db, js = SQLiteStorage(tmp_path / "d.db"), JSONLStorage(tmp_path / "d.jsonl")
    for st in (db, js):
        s = system(st)
        with pytest.raises(KeyError):
            with st.batch():
                s.ask({"amount": 1})
                s.ask({"amount": 2})
                raise KeyError("stop")
        assert st.verify()["ok"]
    assert db.head()["count"] == 0 and js.head()["count"] == 2
    assert json.loads((tmp_path / "d.jsonl.head").read_text())["count"] == 2


def test_inside_a_batch_verify_and_redact_raise_and_other_threads_wait(tmp_path):
    st = JSONLStorage(tmp_path / "d.jsonl")
    s = system(st)
    done = []
    with st.batch():
        s.ask({"amount": 1})
        with pytest.raises(RuntimeError, match="batch"):
            st.verify()
        t = threading.Thread(target=lambda: done.append(s.ask({"amount": 2}).stored_id))
        t.start()
        t.join(0.3)
        assert not done                                         # blocked until the batch ends
        with st.batch():                                        # nested: part of the outer one
            s.ask({"amount": 3})
    t.join(5)
    assert done and st.head()["count"] == 3 and st.verify()["ok"]


def test_save_many_checks_the_metas_and_a_plain_backend_falls_back_to_single_saves(tmp_path):
    st = JSONLStorage(tmp_path / "d.jsonl")
    with pytest.raises(ValueError, match="metas"):
        st.save_many([system().ask({"amount": 1}, store=False)], metas=[None, None])
    from solvi.core.store import TraceStorage
    assert "batch" not in TraceStorage.__abstractmethods__


def test_verify_deep_false_skips_the_answer_re_check_and_keeps_the_chain_checks(tmp_path, monkeypatch):
    import solvi.core.store as store_mod
    from solvi.core.store import record_hash
    p = tmp_path / "d.jsonl"
    st = JSONLStorage(p)
    s = system(st)
    for i in range(5):
        s.ask({"amount": i * 50})
    assert st.verify() == st.verify(deep=True)
    lines = [json.loads(x) for x in p.read_text().splitlines()]
    d = lines[2]                                       # the response edited, its record re-hashed and re-linked
    d["response"]["results"]["approve"]["answer"] = not d["response"]["results"]["approve"]["answer"]
    prev = lines[1]["hash"]
    for r in lines[2:]:
        r["prev"] = prev
        r["hash"] = record_hash(r)
        r["id"] = r["hash"][:16]
        prev = r["hash"]
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))
    deep = JSONLStorage(p).verify()
    assert not deep["ok"] and any("stored answers differ" in pr[2] for pr in deep["problems"])
    calls = []
    real = store_mod._summary_problems
    monkeypatch.setattr(store_mod, "_summary_problems", lambda r: calls.append(1) or real(r))
    shallow = JSONLStorage(p).verify(deep=False)
    assert calls == [] and not any("stored answers differ" in pr[2] for pr in shallow["problems"])
    lines[3]["meta"] = {"edited": True}                # an edit without re-hashing is still caught
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in lines))
    assert any("hash does not match" in pr[2] for pr in JSONLStorage(p).verify(deep=False)["problems"])
