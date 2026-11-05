"""SQLite persistence for the finite host example, not learner-accessible IO.

Effects here are finite state transitions stored in SQLite. Their receipts and
state commit together. This does not make unrelated remote effects atomic.
Private snapshots expose ground truth to the trusted host; digests and source
names are integrity checks, not cryptographic owner authentication.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from ._common import canonical, digest, identifier
from .environments import FiniteEnvironment


class SQLiteEnvironment:
    """Durable Environment protocol adapter with process-safe attempt replay.

    ``cases`` is required only when creating a new environment. On reopen,
    supplied cases must match the normalized original configuration. The parent
    directory must already exist. Each operation uses its own connection.
    """

    idempotent_attempts = True

    def __init__(self, path, cases=None, *, identity="environment"):
        self.identity = identifier(identity, "environment identity")
        self.path = Path(path).expanduser().resolve()
        initial = None if cases is None else FiniteEnvironment(cases, identity=self.identity)
        self._config_digest = None if initial is None else digest(initial.snapshot()["cases"])
        with self._transaction(write=True) as conn:
            objects = conn.execute(
                "SELECT name, type FROM sqlite_master WHERE name = ? COLLATE NOCASE",
                ("experience_environment",),
            ).fetchall()
            if not objects:
                if initial is None:
                    raise ValueError("cases are required for a new durable environment")
                conn.execute(
                    "CREATE TABLE experience_environment ("
                    "id INTEGER PRIMARY KEY CHECK (id = 1), "
                    "schema_version INTEGER NOT NULL, identity TEXT NOT NULL, "
                    "config_digest TEXT NOT NULL, snapshot_json TEXT NOT NULL)"
                )
                conn.execute(
                    "INSERT INTO experience_environment "
                    "(id, schema_version, identity, config_digest, snapshot_json) VALUES (1, 1, ?, ?, ?)",
                    (self.identity, self._config_digest, canonical(initial.snapshot())),
                )
            elif objects != [("experience_environment", "table")]:
                raise ValueError("unsupported durable environment namespace")
            env = self._load(conn)
            self._config_digest = digest(env.snapshot()["cases"])

    @contextmanager
    def _transaction(self, *, write=False):
        conn = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA synchronous = FULL")
            conn.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _load(self, conn):
        columns = conn.execute("PRAGMA table_info(experience_environment)").fetchall()
        if [(row[1], row[2].upper()) for row in columns] != [
            ("id", "INTEGER"),
            ("schema_version", "INTEGER"),
            ("identity", "TEXT"),
            ("config_digest", "TEXT"),
            ("snapshot_json", "TEXT"),
        ]:
            raise ValueError("unsupported durable environment table")
        rows = conn.execute(
            "SELECT id, schema_version, identity, config_digest, snapshot_json FROM experience_environment"
        ).fetchall()
        if len(rows) != 1:
            raise ValueError("durable environment requires exactly one state row")
        key, version, identity, config_digest, encoded = rows[0]
        if type(key) is not int or key != 1 or type(version) is not int or version != 1:
            raise ValueError("unsupported durable environment version")
        if identity != self.identity:
            raise ValueError("durable environment identity mismatch")
        if self._config_digest is not None and config_digest != self._config_digest:
            raise ValueError("durable environment configuration mismatch")
        if type(encoded) is not str:
            raise ValueError("invalid durable environment snapshot")
        try:
            snapshot = json.loads(encoded)
        except (ValueError, TypeError) as exc:
            raise ValueError("corrupt durable environment JSON") from exc
        env = FiniteEnvironment.restore(snapshot, identity=self.identity)
        if config_digest != digest(env.snapshot()["cases"]):
            raise ValueError("durable environment case digest mismatch")
        return env

    def task(self, task_id):
        with self._transaction() as conn:
            return self._load(conn).task(task_id)

    def execute(self, task_id, kind, payload, attempt_id):
        with self._transaction(write=True) as conn:
            env = self._load(conn)
            receipt = env.execute(task_id, kind, payload, attempt_id)
            conn.execute(
                "UPDATE experience_environment SET snapshot_json = ? WHERE id = 1",
                (canonical(env.snapshot()),),
            )
        # The receipt is returned only after the transaction commits. If the
        # process dies first, reconcile/replay on a fresh instance settles it.
        return receipt

    def reconcile(self, attempt_id):
        with self._transaction() as conn:
            return self._load(conn).reconcile(attempt_id)

    @property
    def state(self):
        with self._transaction() as conn:
            return self._load(conn).state

    def snapshot(self):
        with self._transaction() as conn:
            return self._load(conn).snapshot()
