"""The change journal (records of kind "change" in a TraceStorage) and the evaluator lock.

    j = ChangeJournal(storage)                   # a path or a TraceStorage; the decisions' store keeps one hash chain
    j.write("propose", change=..., ...)          # every record: {"v", "kind": "change", "experimental": True, "action", ...}
    j.records(), j.verify()                      # the records in order; the storage's chain check

    lock = EvaluatorLock("eval_seeds.json", pinned="<sha256>", code={"metric": score}, pins={"metric": "<sha256>"})
    lock.verify(); lock.lock_out(); lock.seeds("final evaluation")   # logged; refused in a locked-out process

A journal record follows the envelope of docs/trace_spec.md (v, kind, seq, time, prev, hash, id); a new kind is checked
for the envelope only, as "update" (the update target's records) and "knowledge" are. The writer is the update
records' pattern: the body made plain JSON, appended to the storage's chain.

EvaluatorLock is a generic port of a research harness's lock (held-out seeds pinned by SHA-256, the evaluator's code
pinned by hash, a hash-chained access log, `lock_out()` for candidate processes, `leak_scan`). It guards against
leakage by accident and by candidate code that has no file access — not against the harness author."""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from pathlib import Path

from ...core.store import FORMAT, open_storage, plain

KIND = "change"


class ChangeJournal:
    """Records of kind "change" in a TraceStorage (a path or a TraceStorage; see the module docstring)."""

    def __init__(self, storage):
        if storage is None:
            raise ValueError("the change journal is kept in a TraceStorage: pass a path or a TraceStorage")
        self.storage = open_storage(storage)

    def write(self, action, **data):
        """Append one record → the stored record (with its id, seq and hash)."""
        body = {"v": FORMAT, "kind": KIND, "experimental": True, "action": action, **data}
        return self.storage._append(json.loads(json.dumps(plain(body), default=str)))

    def records(self, action=None):
        """The journal's records in order (dicts), optionally only those of one action."""
        out = [s.data for s in self.storage.iter(KIND)]
        return out if action is None else [d for d in out if d.get("action") == action]

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
    def create(path, n=500, *, low=1_000_000_000, high=2 ** 31 - 1, purpose="held-out evaluation"):
        """Write n random seeds (os.urandom, not a range: knowing the rule does not reveal them) once, read-only → the
        file's SHA-256 (the pin). Refuses when the file exists."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(f"{path} exists: the evaluation seeds are written once")
        seeds = set()
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


__all__ = ["ChangeJournal", "EvalLocked", "EvaluatorLock", "KIND", "LOCK_ENV"]
