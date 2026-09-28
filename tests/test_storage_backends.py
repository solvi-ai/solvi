"""PostgresStorage and DuckDBStorage: the TraceStorage interface, the hash chain and verify on the SQL backends. DuckDB runs
for real when installed; PostgreSQL runs against a server when SOLVI_TEST_POSTGRES names one (a connection string), and
always against a stand-in: a psycopg-like connection over sqlite3 that takes %s placeholders and records the statements,
which checks the store's own logic (placeholders, transactions, the head lock) without a server."""
import json
import os
import re
import sqlite3
import uuid

import pytest

from solvi import System
from solvi.storage import DuckDBStorage, PostgresStorage, SQLiteStorage, open_storage
from test_storage import STATES, Clock, _edit_answer, _rehash, build


class FakePostgres:
    """A psycopg-like connection in autocommit mode over sqlite3: %s placeholders, LOCK TABLE accepted (recorded)."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:", isolation_level=None, check_same_thread=False)
        self.log = []

    def execute(self, sql, args=()):
        self.log.append(sql.split()[0].upper() + (" " + sql.split()[1].upper() if sql.startswith("LOCK") else ""))
        if "?" in sql:
            raise AssertionError(f"a sqlite placeholder reached PostgreSQL: {sql}")
        if sql.startswith("LOCK TABLE"):
            assert re.fullmatch(r"LOCK TABLE \w+ IN SHARE ROW EXCLUSIVE MODE", sql)
            return self.db.execute("SELECT 1")
        return self.db.execute(sql.replace("%s", "?"), args)

    def close(self):
        self.db.close()


def make(kind, tmp_path, clock=None):
    if kind == "duckdb":
        pytest.importorskip("duckdb")
        return DuckDBStorage(tmp_path / "d.duckdb", clock=clock)
    if kind == "postgres-fake":
        return PostgresStorage(FakePostgres(), clock=clock)
    dsn = os.environ.get("SOLVI_TEST_POSTGRES")
    if not dsn:
        pytest.skip("set SOLVI_TEST_POSTGRES to a PostgreSQL connection string to run against a server")
    pytest.importorskip("psycopg")
    return PostgresStorage(dsn, clock=clock, prefix=f"solvi_t{uuid.uuid4().hex[:8]}_")


@pytest.fixture(params=["duckdb", "postgres-fake", "postgres"])
def store(request, tmp_path):
    st = make(request.param, tmp_path, Clock())
    yield st
    if request.param == "postgres":
        for t in ("records", "answers", "safeguards", "models", "meta"):
            st._x(f"DROP TABLE IF EXISTS {{p}}{t}")
    st.close()


def filled(store):
    cat, qs = build()
    s = System(cat, qs, storage=store)
    return s, [s.ask(st) for st in STATES]


def bodies(store):
    return [json.loads(b) for (b,) in store._x("SELECT body FROM {p}records ORDER BY seq").fetchall()]


def test_save_get_query_replay_and_verify(store, tmp_path):
    s, resps = filled(store)
    ids = [r.stored_id for r in resps]
    assert len(store) == 4 and store.verify()["ok"] and store.replay_all(s) == []
    back = store.get(ids[1])
    assert back.results["urgent"].answer == resps[1].results["urgent"].answer
    ref = SQLiteStorage(tmp_path / "ref.db", clock=Clock())
    filled(ref)
    for kw in ({}, {"answer": "no"}, {"question": "approve", "answer": None}, {"status": "ok"}, {"safeguard": "hard_check"},
               {"model": "Scorer"}, {"until": 1025.0}, {"question": "approve", "status": "forced"}):
        assert [x.seq for x in store.query(**kw)] == [x.seq for x in ref.query(**kw)], kw
    s.teach("approve", STATES[2], True, source="outcome", by="ledger", of=ids[2])
    (c,) = store.corrections()
    assert (c["source"], c["by"], c["of"]) == ("outcome", "ledger", ids[2]) and store.verify()["ok"]
    assert store.head() == {"count": 5, "hash": store.record(c["id"])["hash"]}


def test_edits_deletions_and_a_cut_tail_are_detected(store):
    filled(store)
    anchor = store.head()
    d = bodies(store)[1]
    _edit_answer(d)
    store._x("UPDATE {p}records SET body = ? WHERE seq = 1", (json.dumps(d),))
    v = store.verify()
    assert not v["ok"] and v["problems"][0][0] == 1 and "edited" in v["problems"][0][2]
    store._x("DELETE FROM {p}records WHERE seq = 1")
    assert any("chain broken" in p[2] for p in store.verify()["problems"])
    store._x("DELETE FROM {p}records WHERE seq = 3")
    v = store.verify(anchor=anchor)
    assert any("removed from the end" in p[2] for p in v["problems"])


def test_a_full_rewrite_is_caught_by_the_anchor_and_an_index_edit_by_verify(store):
    filled(store)
    anchor = store.head()
    recs = bodies(store)
    _edit_answer(recs[1])
    _rehash(recs, 1)
    for t in ("records", "answers", "safeguards", "models"):
        store._x(f"DELETE FROM {{p}}{t}")
    for d in recs:
        store._insert(d)
    store._x("UPDATE {p}meta SET \"value\" = ? WHERE \"key\" = 'head'",
             (json.dumps({"count": len(recs), "hash": recs[-1]["hash"]}),))
    assert store.verify()["ok"]
    v = store.verify(anchor=anchor)
    assert not v["ok"] and "rewritten" in v["problems"][-1][2]
    store._x("UPDATE {p}answers SET answer = '\"yes\"' WHERE seq = 2 AND question = 'approve'")
    assert any("answers index" in p[2] for p in store.verify()["problems"])


def test_a_failed_append_leaves_the_chain_as_it_was(store):
    filled(store)
    head = store.head()
    with pytest.raises((TypeError, ValueError)):
        store._append({"v": 1, "kind": "teach", "teach": "q", "init": {}, "answer": object()})   # not JSON
    assert store.head() == head and store.verify()["ok"]


def test_postgres_writes_under_the_head_lock():
    fake = FakePostgres()
    store = PostgresStorage(fake, clock=Clock())
    filled(store)
    tx = fake.log[fake.log.index("BEGIN"):]
    assert tx[:3] == ["BEGIN", "LOCK TABLE", "SELECT"] and "COMMIT" in tx
    assert fake.db.execute("SELECT count(*) FROM solvi_records").fetchone() == (4,)       # the default prefix
    with pytest.raises(ValueError):
        PostgresStorage(FakePostgres(), prefix="x; DROP TABLE y")


def test_open_storage_by_path(tmp_path):
    pytest.importorskip("duckdb")
    st = open_storage(tmp_path / "decisions.duckdb")
    assert isinstance(st, DuckDBStorage)
    filled(st)
    st.close()
    again = DuckDBStorage(tmp_path / "decisions.duckdb")       # reopened: the chain goes on
    assert len(again) == 4 and again.verify()["ok"]


def test_the_learning_loop_runs_on_duckdb(tmp_path):
    pytest.importorskip("duckdb")
    import warnings

    from test_learning import TEAMS, build as build_learning, stream
    part, s, _ = build_learning(tmp_path)
    store = DuckDBStorage(tmp_path / "l.duckdb", catalog=s)
    s.storage = store
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loop = s.learning(gates={"max_change": 0.9})
    stream(s, 10)
    rep = loop.run()
    assert rep.promoted and [h["action"] for h in loop.history()] == ["baseline", "update"]
    loop.rollback(0)
    assert part.adaptation is None and store.verify()["ok"] and len(TEAMS) == 3
