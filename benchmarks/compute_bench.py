"""Crash-closed, persistent runner for the frozen finite computation protocol.

Run as ``python -m benchmarks.compute_bench --phase smoke --out RUN
--protocol-dir PROTOCOL``. A STOP directory is evidence and cannot be resumed.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SEEDS = {"smoke": (11,), "dev": (101, 102, 103), "final": tuple(range(21001, 21013))}
FAMILIES = ("reconcile", "invoice", "inventory")
LIMITS = (400, 1200, 4000)
ARMS = ("DIRECT", "DIRTY", "FIFO", "CHEAP", "DISPATCH", "NO_CACHE", "NO_LEARN")
PROTOCOL = ("PLAN.md", "PREREGISTRATION.md", "COST_CONTRACT.md")
ZERO = "0" * 64


class Stop(RuntimeError):
    pass


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_dir(path.parent)


def exclusive(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def read(path):
    return json.loads(path.read_bytes())


def freeze(root, protocol_dir):
    source = sorted(set(root.glob("src/solvi/**/*.py")) | set(root.glob("benchmarks/compute*.py")))
    if not source or not (root / "src/solvi").is_dir():
        raise Stop("source tree missing")
    return {"source_hashes": {str(p.relative_to(root)): file_hash(p) for p in source},
            "protocol_hashes": {name: file_hash(protocol_dir / name) for name in PROTOCOL}}


def matrix(phase):
    return [{"seed": seed, "family": family, "limit": limit, "arm": arm}
            for seed in SEEDS[phase] for family in FAMILIES for limit in LIMITS for arm in ARMS]


def trial(**key):
    from benchmarks.compute_trial import trial as actor
    return actor(**key)


def audit(row):
    from benchmarks.compute_score import audit as scorer
    return scorer(row)


def inspect(out, manifest, pending=None):
    """Validate every durable claim and chain entry, and recompute all scores."""
    expected = {digest(key): key for key in manifest["matrix"]}
    claims = {}
    for path in (out / "claims").iterdir():
        if path.suffix != ".json" or path.stem not in expected:
            raise Stop("unknown claim file")
        key = read(path)
        if key != expected[path.stem]:
            raise Stop("claim content mismatch")
        claims[path.stem] = key
    entries, seen, previous = [], set(), ZERO
    raw = out / "raw.jsonl"
    if raw.exists():
        with raw.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    raise Stop("truncated raw record")
                entry = json.loads(line)
                if set(entry) != {"prev", "hash", "payload"}:
                    raise Stop("invalid raw envelope")
                payload = entry["payload"]
                if entry["prev"] != previous or entry["hash"] != digest({"prev": previous, "payload": payload}):
                    raise Stop("raw hash chain mismatch")
                key = payload["key"]
                ident = digest(key)
                if ident not in claims or key != expected.get(ident) or ident in seen:
                    raise Stop("duplicate/unclaimed/unknown trial")
                if "error" in payload:
                    raise Stop("recorded actor error")
                row = payload["row"]
                if any(row.get(k) != v for k, v in key.items()):
                    raise Stop("actor identity mismatch")
                if audit(row) != payload["metrics"]:
                    raise Stop("independent audit mismatch")
                seen.add(ident)
                entries.append(entry)
                previous = entry["hash"]
    outstanding = set(claims) - seen
    allowed = {digest(pending)} if pending is not None else set()
    if outstanding != allowed:
        raise Stop("claimed trial without durable result; replay forbidden")
    return entries, seen, previous


def check_sources(root, protocol_dir, manifest):
    if freeze(root, protocol_dir) != {k: manifest[k] for k in ("source_hashes", "protocol_hashes")}:
        raise Stop("source/protocol changed")


def approve(path, manifest):
    if path is None:
        raise Stop("final requires --approval")
    approval = read(path)
    dev_out = Path(approval["dev_out"])
    dev_manifest = read(dev_out / "manifest.json")
    if dev_manifest["phase"] != "dev" or dev_manifest["matrix"] != matrix("dev"):
        raise Stop("approval requires complete dev matrix")
    if (dev_out / "STOP.json").exists():
        raise Stop("dev is stopped")
    for key in ("source_hashes", "protocol_hashes"):
        if approval[key] != manifest[key] or dev_manifest[key] != manifest[key]:
            raise Stop("approval freeze mismatch")
    entries, _, _ = inspect(dev_out, dev_manifest)
    if len(entries) != len(dev_manifest["matrix"]) or file_hash(dev_out / "raw.jsonl") != approval["dev_raw_sha256"]:
        raise Stop("approval dev raw mismatch/incomplete")
    return file_hash(path)


@contextmanager
def lease(out):
    with (out / ".lease").open("a+b") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise Stop("another parent owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def append(out, previous, payload):
    entry = {"prev": previous, "payload": payload}
    entry["hash"] = digest(entry)
    with (out / "raw.jsonl").open("ab") as stream:
        stream.write(encoded(entry) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(out)
    return entry["hash"]


def run(phase, out, protocol_dir, summarize=False, approval=None, root=ROOT):
    out, protocol_dir, root = Path(out).resolve(), Path(protocol_dir).resolve(), Path(root).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A rejected concurrent caller must never mutate the active parent's evidence.
    with lease(out):
        try:
            if (out / "STOP.json").exists():
                raise Stop("persistent STOP exists")
            current = {"schema": 1, "phase": phase, "matrix": matrix(phase),
                       "root": str(root), "protocol_dir": str(protocol_dir),
                       "interpreter": sys.executable, "python": platform.python_version(),
                       "sys_path": list(sys.path), **freeze(root, protocol_dir)}
            manifest_path = out / "manifest.json"
            if manifest_path.exists():
                manifest = read(manifest_path)
                if manifest != current:
                    raise Stop("manifest/environment/source mismatch")
            elif summarize:
                raise Stop("manifest missing")
            else:
                # Refuse pre-existing unmanifested evidence.
                if (out / "raw.jsonl").exists() or (out / "claims").exists():
                    raise Stop("unmanifested evidence")
                exclusive(manifest_path, current)
                manifest = current
                (out / "claims").mkdir()
                sync_dir(out)
            if phase == "final" and not summarize:
                approval_hash = approve(approval, manifest)
                approval_record = out / "approval.json"
                record = {"sha256": approval_hash}
                if approval_record.exists():
                    if read(approval_record) != record:
                        raise Stop("approval changed")
                else:
                    exclusive(approval_record, record)
            entries, seen, previous = inspect(out, manifest)
            check_sources(root, protocol_dir, manifest)
            if not summarize:
                for key in manifest["matrix"]:
                    if digest(key) in seen:
                        continue
                    # Re-audit all durable evidence before every irreversible claim.
                    entries, seen, previous = inspect(out, manifest)
                    check_sources(root, protocol_dir, manifest)
                    exclusive(out / "claims" / (digest(key) + ".json"), key)
                    try:
                        row = trial(**key)
                        metrics = audit(row)
                        if any(row.get(k) != v for k, v in key.items()):
                            raise Stop("actor identity mismatch")
                        # Validate serializability before appending, never replay a bad actor.
                        payload = {"key": key, "row": row, "metrics": metrics}
                        encoded(payload)
                        old_head = previous
                        _, _, previous = inspect(out, manifest, pending=key)
                        if previous != old_head:
                            raise Stop("raw changed during actor")
                        check_sources(root, protocol_dir, manifest)
                    except BaseException as exc:
                        # Save unexpected failures as evidence before permanently stopping.
                        append(out, previous, {"key": key, "error": {"type": type(exc).__name__, "message": str(exc)}})
                        raise
                    previous = append(out, previous, payload)
                    print(json.dumps({"heartbeat": time.time(), "complete": len(entries) + 1,
                                      "total": len(manifest["matrix"]), "key": key}), flush=True)
                entries, seen, previous = inspect(out, manifest)
                check_sources(root, protocol_dir, manifest)
            summary = {"phase": phase, "complete": len(entries), "total": len(manifest["matrix"]),
                       "chain_head": previous, "raw_sha256": file_hash(out / "raw.jsonl") if entries else None,
                       "metrics": [{"key": e["payload"]["key"], **e["payload"]["metrics"]} for e in entries]}
            atomic(out / "summary.json", summary)
            print(json.dumps(summary, sort_keys=True, allow_nan=False), flush=True)
            return summary
        except BaseException as exc:
            if not (out / "STOP.json").exists():
                exclusive(out / "STOP.json", {"type": type(exc).__name__, "reason": str(exc), "time": time.time()})
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=tuple(SEEDS))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--protocol-dir", required=True, type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(args.phase, args.out, args.protocol_dir, args.summarize, args.approval)
    except BaseException as exc:  # noqa: BLE001 - preserve durable STOP and a nonzero CLI exit
        print(f"STOP: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
