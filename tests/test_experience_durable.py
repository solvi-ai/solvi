from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from solvi.experimental.experience.durable import SQLiteEnvironment


CASES = {"r": {"family": "resource", "x": 7, "y": 2}}


def test_reopen_before_and_after_execute(tmp_path):
    path = tmp_path / "environment.sqlite"
    first = SQLiteEnvironment(path, CASES, identity="host")
    assert first.idempotent_attempts is True
    assert first.reconcile("attempt") is None
    reopened = SQLiteEnvironment(str(path), identity="host")
    assert reopened.task("r") == first.task("r")
    receipt = reopened.execute("r", "act", {"answer": 5}, "attempt")
    third = SQLiteEnvironment(path, CASES, identity="host")
    assert third.reconcile("attempt") == receipt
    assert third.execute("r", "act", {"answer": 5}, "attempt") == receipt
    assert third.state["r"] == {"available": 4, "reserved": 1, "effects": 1}
    assert first.state == third.state  # Existing instances read through.
    assert first.snapshot() == third.snapshot()
    assert receipt.source == "host"
    with pytest.raises(ValueError, match="conflicts"):
        third.execute("r", "act", {"answer": 4}, "attempt")
    assert first.state["r"]["effects"] == 1


def test_configuration_and_identity_are_pinned(tmp_path):
    path = tmp_path / "environment.sqlite"
    env = SQLiteEnvironment(path, CASES, identity="host")
    before = env.snapshot()
    with pytest.raises(ValueError, match="identity"):
        SQLiteEnvironment(path, CASES, identity="different")
    with pytest.raises(ValueError, match="configuration"):
        SQLiteEnvironment(path, {"r": {"family": "resource", "x": 8, "y": 2}}, identity="host")
    assert env.snapshot() == before
    normalized = before["cases"]
    assert SQLiteEnvironment(path, normalized, identity="host").snapshot() == before


@pytest.mark.parametrize(
    "column,value",
    [
        ("schema_version", 2),
        ("identity", "forged"),
        ("config_digest", "invalid"),
        ("snapshot_json", "{truncated"),
    ],
)
def test_corrupt_database_rejected_without_overwrite(tmp_path, column, value):
    path = tmp_path / "environment.sqlite"
    env = SQLiteEnvironment(path, CASES, identity="host")
    # Column names are fixed test constants, not user input.
    with sqlite3.connect(path) as conn:
        conn.execute(f"UPDATE experience_environment SET {column} = ?", (value,))
        corrupted = conn.execute("SELECT * FROM experience_environment").fetchall()
    with pytest.raises(ValueError):
        _ = env.state
    with pytest.raises(ValueError):
        SQLiteEnvironment(path, CASES, identity="host")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT * FROM experience_environment").fetchall() == corrupted


def test_parallel_duplicate_attempt_has_one_effect(tmp_path):
    path = tmp_path / "environment.sqlite"
    first = SQLiteEnvironment(path, CASES)
    second = SQLiteEnvironment(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(env.execute, "r", "act", {"answer": 5}, "shared") for env in (first, second)]
        receipts = [future.result() for future in futures]
    assert receipts[0] == receipts[1]
    assert first.state["r"]["effects"] == 1
    assert len(first.snapshot()["records"]) == 1


def test_failed_write_rolls_back_receipt_and_effect(tmp_path):
    path = tmp_path / "environment.sqlite"
    env = SQLiteEnvironment(path, CASES)
    before = env.snapshot()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "CREATE TRIGGER fail_write BEFORE UPDATE ON experience_environment "
            "BEGIN SELECT RAISE(ABORT, 'test write failure'); END"
        )
    with pytest.raises(sqlite3.IntegrityError, match="test write failure"):
        env.execute("r", "act", {"answer": 5}, "attempt")
    assert env.reconcile("attempt") is None
    assert env.snapshot() == before
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TRIGGER fail_write")
    assert env.execute("r", "act", {"answer": 5}, "attempt").value["ok"] is True
    assert env.state["r"]["effects"] == 1


def test_foreign_schema_is_preserved_and_namespace_collision_rejected(tmp_path):
    path = tmp_path / "foreign.sqlite"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE unrelated (value TEXT)")
        conn.execute("INSERT INTO unrelated VALUES ('keep')")
    SQLiteEnvironment(path, CASES)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT value FROM unrelated").fetchall() == [("keep",)]
    collision = tmp_path / "collision.sqlite"
    with sqlite3.connect(collision) as conn:
        conn.execute("CREATE TABLE experience_environment (foreign_data TEXT)")
        conn.execute("INSERT INTO experience_environment VALUES ('keep')")
    with pytest.raises(ValueError, match="table"):
        SQLiteEnvironment(collision, CASES)
    with sqlite3.connect(collision) as conn:
        assert conn.execute("SELECT * FROM experience_environment").fetchall() == [("keep",)]


def test_new_environment_requires_cases_and_existing_parent(tmp_path):
    path = tmp_path / "new.sqlite"
    with pytest.raises(ValueError, match="cases are required"):
        SQLiteEnvironment(path)
    with sqlite3.connect(path) as conn:
        assert not conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    with pytest.raises(sqlite3.OperationalError):
        SQLiteEnvironment(tmp_path / "missing" / "environment.sqlite", CASES)
