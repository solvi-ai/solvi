"""The change journal (records of kind "change" in a TraceStorage) and the evaluator lock.

    j = ChangeJournal(storage)                   # a path or a TraceStorage; the decisions' store keeps one hash chain
                                                 # None or ":memory:": a MemoryStorage (this process only)
    j.write("propose", change=..., ...)          # every record: {"v", "kind": "change", "experimental": True, "action", ...}
    j.records(), j.verify()                      # the records in order; the storage's chain check

    lock = EvaluatorLock("eval_seeds.json", pinned="<sha256>", code={"metric": score}, pins={"metric": "<sha256>"})
    lock.verify(); lock.lock_out(); lock.seeds("final evaluation")   # logged; refused in a locked-out process

A journal record follows the envelope of docs/trace_spec.md (v, kind, seq, time, prev, hash, id); a new kind is checked
for the envelope only, as "update" (the update target's records) and "knowledge" are. The writer is the update
records' pattern: the body made plain JSON, appended to the storage's chain. The journal keeps the records it read in
memory and adds each one it writes, so a proposal does not re-read the storage; it reads again only when the storage's
head moved without it (another writer, another process, the decisions saved into the same store). `fold(key, init,
step)` keeps a reader's state (the versions in force, the failure memory) up to date with the new records only.
MemoryStorage is a hash-chained TraceStorage in a Python list (no persistence; verify() works as on disk).

EvaluatorLock is a generic port of a research harness's lock (held-out seeds pinned by SHA-256, the evaluator's code
pinned by hash, a hash-chained access log, `lock_out()` for candidate processes, `leak_scan`). It guards against
leakage by accident and by candidate code that has no file access — not against the harness author."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path

from ...core.store import FORMAT, GENESIS, TraceStorage, chained, open_storage, plain, record_hash

KIND = "change"
MEMORY = ":memory:"


class MemoryStorage(TraceStorage):
    """A hash-chained TraceStorage kept in a Python list: the change journal of a run that needs no file (many short
    pipelines, tests). Nothing persists past the process; verify(), redact and the rest work as on disk (it passes
    solvi.testing.conformance.check_storage)."""

    def __init__(self, system=None, clock=None, *, record="full"):
        super().__init__(system, clock, record)
        self._recs, self._by_id = [], {}
        self._lock = threading.Lock()

    def _append(self, body):
        with self._lock:
            prev = self._recs[-1]["hash"] if self._recs else GENESIS
            rec = dict(json.loads(json.dumps(body, default=str)), seq=len(self._recs), time=float(self.clock()),
                       prev=prev)
            rec["hash"] = record_hash(rec)
            rec["id"] = rec["hash"][:16]
            self._recs.append(rec)
            self._by_id[rec["id"]] = rec
            return rec

    def _raw(self, snap=None):
        n = len(self._recs) if snap is None else snap
        return ((i, self._recs[i]) for i in range(n))

    def _snapshot(self):
        return len(self._recs)

    def _find(self, id):
        return self._by_id.get(id)

    def head(self):
        return {"count": len(self._recs), "hash": self._recs[-1]["hash"] if self._recs else GENESIS}

    def _rewrite(self, rec):
        with self._lock:
            self._recs[rec["seq"]] = rec
            self._by_id[rec["id"]] = rec


class ChangeJournal:
    """Records of kind "change" in a TraceStorage (a path, a TraceStorage, or None / ":memory:" for a MemoryStorage;
    see the module docstring)."""

    def __init__(self, storage):
        self.storage = MemoryStorage() if storage is None or storage == MEMORY else open_storage(storage)
        self._recs, self._head, self._gen, self._folds = [], None, 0, {}

    def _sync(self):
        """The cached records, read again only when the storage's head is not the one this journal last saw."""
        h = self.storage.head()
        if (h["count"], h["hash"]) != self._head:
            self._recs = [d for _, d in self.storage._raw() if chained(d) and d.get("kind") == KIND
                          and not d.get("redacted")]
            self._head, self._gen = (h["count"], h["hash"]), self._gen + 1
        return self._recs

    def write(self, action, **data):
        """Append one record → the stored record (with its id, seq and hash)."""
        body = {"v": FORMAT, "kind": KIND, "experimental": True, "action": action, **data}
        rec = self.storage._append(json.loads(json.dumps(plain(body), default=str)))
        if self._head is not None and rec.get("seq") == self._head[0] and rec.get("prev") == self._head[1]:
            self._recs.append(rec)                    # nobody wrote in between: the cache stays whole
            self._head = (self._head[0] + 1, rec["hash"])
        return rec

    def records(self, action=None):
        """The journal's records in order (dicts), optionally only those of one action."""
        out = self._sync()
        return list(out) if action is None else [d for d in out if d.get("action") == action]

    def fold(self, key, init, step):
        """A reader's state over the records, kept up to date: `init()` → the state, `step(state, record)` for each
        record in order; later calls apply only the records written since (all again when the storage was re-read).
        → the state."""
        recs = self._sync()
        st = self._folds.get(key)
        if st is None or st[0] != self._gen:
            st = [self._gen, 0, init()]
        for d in recs[st[1]:]:
            step(st[2], d)
        st[1] = len(recs)
        self._folds[key] = st
        return st[2]

    def verify(self):
        """The storage's chain check (TraceStorage.verify) → {"ok", "problems", ...}."""
        return self.storage.verify()


# ───────────────────────────────────────────────────────────── the evaluator lock (ported)
LOCK_ENV = "SOLVI_EVAL_LOCKED"


class EvalLocked(Exception):
    """The evaluator refuses: a pinned file or code changed, or this process is locked out."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _code_hash(x):
    if callable(x):
        return sha256(inspect.getsource(x).encode())
    p = Path(x)
    return sha256(p.read_bytes()) if p.exists() else None


class EvaluatorLock:
    """Held-out seeds pinned by hash, the evaluator's code pinned by hash, an access log, a lock-out for candidates.

    seeds_file: a JSON file {"seeds": [...], ...} written once (EvaluatorLock.create); pinned: its SHA-256 (fixed at
    pre-registration); code: {name: a path or a function} that make up the evaluator; pins: {name: SHA-256} for them;
    access_log: a TraceStorage or path for the access records (kind "eval_access", hash-chained); env: the variable
    lock_out() sets (inherited by child processes)."""

    def __init__(self, seeds_file, *, pinned=None, code=None, pins=None, access_log=None, env=LOCK_ENV):
        self.seeds_file = Path(seeds_file)
        self.pinned, self.code, self.pins, self.env = pinned, dict(code or {}), dict(pins or {}), env
        self.access_log = open_storage(access_log) if access_log is not None else None

    @staticmethod
    def create(path, n=None, *, seeds=None, low=1_000_000_000, high=2 ** 31 - 1, purpose="held-out evaluation"):
        """Write n random seeds (os.urandom, not a range: knowing the rule does not reveal them; default 500) once,
        read-only → the file's SHA-256 (the pin). Refuses when the file exists.

        seeds: your own held-out seeds instead (distinct ints; also accepted in place of n:
        `EvaluatorLock.create("eval_seeds.json", [11, 12, 13])`) — then keep them out of every file a proposer reads."""
        path = Path(path)
        if n is not None and not isinstance(n, int) and isinstance(n, (list, tuple, range, set, frozenset)):
            if seeds is not None:
                raise TypeError("pass the seeds as n or as seeds=, not both")
            n, seeds = None, n
        if seeds is not None:
            if n is not None:
                raise TypeError("pass n (how many random seeds) or seeds= (your own), not both")
            if isinstance(seeds, (str, bytes)) or not hasattr(seeds, "__iter__"):
                raise TypeError(f"seeds is a list of ints, not {type(seeds).__name__}")
            seeds = list(seeds)
            bad = [s for s in seeds if isinstance(s, bool) or not isinstance(s, int)]
            if bad:
                raise TypeError(f"seeds are ints, not {bad[:3]}")
            if not seeds or len(set(seeds)) != len(seeds):
                raise ValueError("seeds are a non-empty list of distinct ints")
            seeds = set(seeds)
            n = len(seeds)
        else:
            n = 500 if n is None else n
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError(f"n is the number of seeds to draw (an int ≥ 1), not {n!r}; your own seeds go in "
                                f"seeds=[...]")
            if n < 1:
                raise ValueError(f"n is the number of seeds to draw (an int ≥ 1), not {n}")
        if path.exists():
            raise FileExistsError(f"{path} exists: the evaluation seeds are written once")
        seeds = set(seeds or ())
        while len(seeds) < n:
            seeds.add(int.from_bytes(os.urandom(4), "big") % (high - low) + low)
        path.write_text(json.dumps({"purpose": purpose, "n": n, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "seeds": sorted(seeds)}, indent=0) + "\n")
        os.chmod(path, 0o444)
        return sha256(path.read_bytes())

    def manifest(self):
        """The current hashes: {"seeds": sha or None, "code": {name: sha or None}} — pin these at pre-registration."""
        return {"seeds": sha256(self.seeds_file.read_bytes()) if self.seeds_file.exists() else None,
                "code": {k: _code_hash(v) for k, v in self.code.items()}}

    def verify(self):
        """Raise EvalLocked unless the seeds file and every pinned piece of code match their pins → True."""
        if not self.pinned:
            raise EvalLocked("the seeds' hash is not pinned")
        m = self.manifest()
        if m["seeds"] is None:
            raise EvalLocked(f"{self.seeds_file} is missing")
        if m["seeds"] != self.pinned:
            raise EvalLocked(f"the seeds file changed: {m['seeds']} != pinned {self.pinned}")
        unpinned = sorted(set(self.code) - set(self.pins))
        if unpinned:
            raise EvalLocked(f"evaluator code not pinned: {unpinned}")
        bad = sorted(k for k in self.code if m["code"][k] != self.pins[k])
        if bad:
            raise EvalLocked(f"evaluator code changed: {bad}")
        return True

    def lock_out(self):
        """Call at the start of every proposer / search process: it and its children cannot read the seeds."""
        os.environ[self.env] = "1"

    def locked_out(self):
        return os.environ.get(self.env) == "1"

    def seeds(self, purpose, *, by=None):
        """The held-out seeds — for the pre-registered evaluation only. Refused in a locked-out process; verified;
        every call is an access record."""
        if self.locked_out():
            raise EvalLocked("this process is locked out (a proposer or search process): the held-out seeds are not "
                             "available")
        self.verify()
        seeds = json.loads(self.seeds_file.read_text())["seeds"]
        if self.access_log is not None:
            self.access_log._append({"v": FORMAT, "kind": "eval_access", "experimental": True, "purpose": str(purpose),
                                     "by": by, "argv": list(sys.argv), "pid": os.getpid(), "seeds_sha256": self.pinned})
        return seeds

    def access_count(self):
        """How many times the seeds were handed out (the access log's records; "evaluate once" = 1)."""
        return 0 if self.access_log is None else sum(1 for _ in self.access_log.iter("eval_access"))

    def leak_scan(self, paths, max_bytes=200 * 2 ** 20):
        """Files under `paths` that mention a held-out seed (journals, prompt caches, candidate sources) →
        [(path, [seeds...])]."""
        seeds = {str(s) for s in json.loads(self.seeds_file.read_text())["seeds"]}
        hits = []
        for root in map(Path, paths):
            for p in ([root] if root.is_file() else sorted(root.rglob("*"))):
                if not p.is_file() or p.resolve() == self.seeds_file.resolve() or p.stat().st_size > max_bytes:
                    continue
                text = p.read_text(errors="ignore")
                found = sorted(s for s in seeds if s in text)
                if found:
                    hits.append((str(p), found[:3]))
        return hits


__all__ = ["ChangeJournal", "EvalLocked", "EvaluatorLock", "KIND", "LOCK_ENV", "MEMORY", "MemoryStorage"]
