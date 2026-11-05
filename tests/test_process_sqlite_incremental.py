"""Operation counts for the existing indexed backend; no timing or trust relaxation."""

import json
import pytest
from solvi.core.store import JSONLStorage, SQLiteStorage
from solvi.experimental.process import MemoryQueue, Process


def test_sqlite_process_index_reads_only_new_rows(tmp_path, monkeypatch):
    store = SQLiteStorage(tmp_path / "process.db")
    with store.batch():
        for i in range(96):
            store._append({"v": 2, "kind": "human", "process": "other", "ticket": str(i)})
    p = Process("p", storage=store)
    assert len(p._index()["tickets"]) == 96
    seen = []
    original = store._rows

    def counted(*args, **kwargs):
        for seq, record in original(*args, **kwargs):
            seen.append(seq)
            yield seq, record

    monkeypatch.setattr(store, "_rows", counted)
    for _ in range(5):
        assert p._index()["seq"] == 95
    assert seen == []
    store._append({"v": 2, "kind": "human", "process": "p", "ticket": "new"})
    assert p._ticket_seen("new") and seen == [96]
    assert p._human_record("new").seq == 96 and seen == [96]
    plan = store.db.execute(
        "EXPLAIN QUERY PLAN SELECT seq,body FROM records WHERE seq>? ORDER BY seq", (95,)
    ).fetchall()
    assert any("INTEGER PRIMARY KEY" in row[-1] for row in plan)


@pytest.mark.parametrize("backend", [JSONLStorage, SQLiteStorage])
def test_short_handoff_reopen_and_explicit_tamper_verification(tmp_path, backend):
    path = tmp_path / ("journal.jsonl" if backend is JSONLStorage else "journal.db")
    store = backend(path)
    queue = MemoryQueue()
    p = Process("human", storage=store, human=queue)
    p.node("h", kind="human", question="q", reads=["x"], writes={"out": "answer"}, end=True)
    pending = p.run({"x": 1})
    assert pending.status == "pending"
    queue.answer(pending.ticket, "ok", reviewer="r")
    assert p.resume(pending.run).status == "decided"
    reopened = backend(path)
    assert reopened.verify()["ok"]
    other = Process("human", storage=reopened, human=queue)
    other.node("h", kind="human", question="q", reads=["x"], writes={"out": "answer"}, end=True)
    assert other._ticket_seen(pending.ticket) and len(other.records()) == len(p.records())
    if backend is SQLiteStorage:
        seq, body = reopened.db.execute("SELECT seq,body FROM records ORDER BY seq LIMIT 1").fetchone()
        record = json.loads(body)
        record["process"] = "tampered"
        reopened.db.execute("UPDATE records SET body=? WHERE seq=?", (json.dumps(record), seq))
    else:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[0]["process"] = "tampered"
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
    assert not reopened.verify()["ok"]
