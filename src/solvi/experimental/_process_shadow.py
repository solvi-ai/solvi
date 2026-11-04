"""Compare complete sequential processes while serving only the current process.

Bindings and journals must be disjoint. This is not a sandbox for Python callbacks,
models or external effects; supply candidate systems whose effects are acceptable.
"""
import copy
import os

from ..core.provenance import digest
from ..core.store import FORMAT, DuckDBStorage, JSONLStorage, SQLiteStorage, open_storage, plain

__all__ = []


def _stores(process):
    stores = [process.storage, process._store]
    stores.extend(system.storage for system in process.systems.values())
    stores.extend(ns.get("store") for ns in process.nodes.values())
    return [s for s in stores if s is not None]


def _path(store):
    path = store if isinstance(store, (str, os.PathLike)) else getattr(store, "path", None)
    if path is None or (str(path) == ":memory:" and type(store) in (SQLiteStorage, DuckDBStorage)):
        return None
    return os.path.realpath(os.path.abspath(os.fspath(path)))


def _shared_store(a, b):
    if a is b and not isinstance(a, (str, os.PathLike)):
        return True
    for pa in _store_paths(a):
        for pb in _store_paths(b):
            if pa == pb:
                return True
            try:
                if os.path.samefile(pa, pb):
                    return True
            except FileNotFoundError:
                pass
    return False


def _store_paths(store):
    path = _path(store)
    if path is None:
        return []
    paths = [path]
    jsonl = type(store) is JSONLStorage or (isinstance(store, (str, os.PathLike)) and
                                          not str(store).lower().endswith((".db", ".sqlite", ".sqlite3", ".duckdb")))
    if jsonl:
        paths.extend((path + ".head", path + ".lock"))
        if type(store) is JSONLStorage:
            paths.extend(os.path.realpath(os.path.abspath(p)) for p in store.segments())
    else:
        paths.extend((path + "-wal", path + "-shm", path + "-journal"))
    return paths


class ProcessShadow:
    """Run current and candidate on separate copies of one input; return current's result.

    Supports fully bound native System nodes, with candidate handoff disabled and no
    human nodes. Bindings, guards and all journals must be disjoint. Candidate
    errors are recorded in ``last`` and counted, without replacing a served result.
    Optional ``storage`` holds hash-chained comparison indexes, not raw fact values.
    Each process retains its own normal decision records and replay API. There is
    no automatic promotion or statistical quality claim. ``keep`` bounds in-memory
    history; callbacks/model backends can still have shared external effects.
    """

    def __init__(self, current, candidate, *, storage=None, keep=1000):
        if type(keep) is not int or keep < 1:
            raise ValueError("keep must be a positive integer")
        self.current, self.candidate = current, candidate
        self.storage, self.keep = storage, keep
        self._store = None
        self.stats = {"runs": 0, "agree": 0, "differ": 0, "errors": 0}
        self.last, self.changed, self.errors = None, [], []
        self._validate()

    def _validate(self):
        from ..core.system import System
        if self.current is self.candidate:
            raise ValueError("current and candidate processes must be distinct")
        for process in (self.current, self.candidate):
            spec = process.spec()
            process._check_bound()
            if any(ns["kind"] != "system" for ns in spec["nodes"].values()):
                raise ValueError("ProcessShadow supports native System nodes only")
            if any(not isinstance(s, System) for s in process.systems.values()):
                raise ValueError("ProcessShadow supports native System bindings only")
        if (self.candidate.human is not None or self.candidate.on_abstain != "abstain"
                or self.candidate.on_human != "abstain"):
            raise ValueError("candidate must disable human channels and use on_abstain/on_human='abstain'")
        current_objects = set()
        for system in self.current.systems.values():
            current_objects.add(id(system))
            current_objects.update(id(g) for g in system.guards.values())
        for system in self.candidate.systems.values():
            if id(system) in current_objects or any(id(g) in current_objects for g in system.guards.values()):
                raise ValueError("current and candidate bindings and guards must be disjoint")
        current_stores, candidate_stores = _stores(self.current), _stores(self.candidate)
        for store in current_stores + candidate_stores + [self.storage, self._store]:
            if isinstance(store, str) and store.startswith(("postgres://", "postgresql://")):
                raise ValueError("ProcessShadow journals must be native file stores, not PostgreSQL URLs")
            if store is not None and not isinstance(store, (str, os.PathLike)) and type(store) not in (
                    JSONLStorage, SQLiteStorage, DuckDBStorage):
                raise ValueError("ProcessShadow journals must be native JSONL, SQLite or DuckDB stores or file paths")
        if any(_shared_store(a, b) for a in current_stores for b in candidate_stores):
            raise ValueError("current and candidate journals must be disjoint")
        for store in (self.storage, self._store):
            if store is not None and any(_shared_store(store, other) for other in current_stores + candidate_stores):
                raise ValueError("comparison journal must be separate from process and decision journals")

    def run(self, facts, run=None):
        """Serve current.run; evaluate candidate once on the original input in its own run.

        Setup/alias failures happen before either process runs. If current fails,
        its exception propagates and candidate is not invoked. Comparison write
        failures cannot change a successfully served current result.
        """
        self._validate()
        current_facts, candidate_facts = copy.deepcopy(facts), copy.deepcopy(facts)
        input_hash = digest(plain(facts))
        current = self.current.run(current_facts, run=run)
        self.stats["runs"] += 1
        candidate = None
        index = {"input_hash": input_hash, "current": {"run": current.run, "stored_id": current.stored_id,
                 "spec_hash": current.spec_hash, "status": current.status, "answer_hash": digest(plain(current.answer))}}
        try:
            candidate = self.candidate.run(candidate_facts, run=current.run + ":shadow")
            different = (current.status != candidate.status or
                         digest(plain(current.answer)) != digest(plain(candidate.answer)))
            path_changed = ([s["node"] for s in current.path] != [s["node"] for s in candidate.path])
            index.update(candidate={"run": candidate.run, "stored_id": candidate.stored_id,
                                    "spec_hash": candidate.spec_hash, "status": candidate.status,
                                    "answer_hash": digest(plain(candidate.answer))},
                         different=different, path_changed=path_changed)
            self.stats["differ" if different else "agree"] += 1
            if different or path_changed:
                self.changed = (self.changed + [copy.deepcopy(index)])[-self.keep:]
        except Exception as exc:  # noqa: BLE001 — a candidate must not replace the served result
            index["error"] = type(exc).__name__
            try:
                index["error_hash"] = digest(str(exc))
            except Exception:  # noqa: BLE001 — even a broken exception renderer cannot replace the served result
                index["error_hash"] = None
            self.stats["errors"] += 1
            self.errors = (self.errors + [copy.deepcopy(index)])[-self.keep:]
        self.last = {"current": current, "candidate": candidate, "comparison": copy.deepcopy(index)}
        if self.storage is not None:
            try:
                if self._store is None:
                    self._store = open_storage(self.storage)
                self._store._append({"v": FORMAT, "kind": "process_shadow", "shadow": index})
            except Exception as exc:  # noqa: BLE001 — index I/O cannot change the served answer
                error = {"run": current.run, "error": f"comparison journal: {type(exc).__name__}"}
                self.errors = (self.errors + [error])[-self.keep:]
                self.stats["errors"] += 1
        return current
