"""Frozen single-fit budget experiment supervisor. Raw rows are append-only."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import fcntl
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

from active_real_data import SOURCES
from packed_spam_data import prepare
from solvi.experimental.experience._common import digest

ROOT = Path(__file__).resolve().parents[1]
IMPL = Path("src/solvi/experimental/experience/packed.py")
ARMS = ("RANDOM48", "RANDOM40", "FULL_LABEL", "LEGACY40", "MEDIAN40", "CUT40", "ADAPT40")
METHODS = ("CUT40", "ADAPT40")
PHASES = ("smoke", "dev", "final")


def settings(phase):
    if phase not in PHASES:
        raise ValueError("unknown phase")
    for name in SOURCES:
        if phase == "smoke":
            if name in ("banknote", "spambase"):
                yield name, "f0", 1001
        elif phase == "dev":
            for seed in range(11001, 11007 if name == "spambase" else 11004):
                yield name, "f0", seed
        else:
            for f in range(3 if name == "spambase" else 1):
                first = 12001 + 1000 * f
                for seed in range(first, first + (10 if name == "spambase" else 5)):
                    yield name, f"f{f}", seed


def jobs(phase, selected=None):
    if phase == "final" and selected not in METHODS:
        raise ValueError("one frozen acquisition method required")
    result = []
    for name, fold, seed in settings(phase):
        for price in (128, 8192):
            for arm in ARMS:
                if phase == "final" and arm in METHODS and arm != selected:
                    continue
                if phase == "smoke" and ((name == "banknote" and price != 128) or
                                         (name == "spambase" and arm in ("RANDOM40", "FULL_LABEL"))):
                    continue
                result.append(dict(tier="packed_real", phase=phase, split="final" if phase == "final" else "dev",
                                   dataset=name, fold=fold, seed=seed, price=price, arm=arm))
    return result


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write(path, value, mode="x"):
    with path.open(mode) as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def load(path):
    return json.loads(path.read_text())


@contextmanager
def lease(output, *, readonly=False):
    with (output / "runner.lock").open("r" if readonly else "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pins(a):
    files = list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "benchmarks").glob("spam_budget*.py"))
    files += list((ROOT / "benchmarks").glob("packed_spam*.py"))
    for name in ("data", "models", "trial", "score"):
        files.append(ROOT / "benchmarks" / f"active_real_{name}.py")
    files += [ROOT / name for name in ("pyproject.toml", "uv.lock")]
    sources = {str(f.relative_to(ROOT)): sha(f) for f in sorted(files)}
    archives = {name: sha(a.data / (name + ".zip")) for name in SOURCES}
    if any(archives[k] != s.sha256 for k, s in SOURCES.items()):
        raise ValueError("source archive changed")
    return dict(sources=sources, archives=archives, roles=sha(a.roles), protocol=sha(a.protocol),
                protocol_annexes={p.name: sha(p) for p in sorted(a.protocol.parent.glob("COST_CONTRACT*.md"))},
                data_manifest=sha(a.data / "DATA_MANIFEST.json"), wrapper=sha(a.audit_wrapper),
                python=str(a.python), python_sha256=sha(a.python), source=str(ROOT),
                scorer=sources["benchmarks/packed_spam_score.py"])


def partition(data):
    return dict(groups=digest(data["group_ids"]), thresholds=digest(data["thresholds"]),
                pool_X=digest([[r["id"], r["inputs"]] for r in data["pool"]]),
                test_X=digest([[r["id"], r["inputs"]] for r in data["test"]]))


@lru_cache(maxsize=1)
def prepared(name, data, seed, split, fold, roles, manifest):
    result = prepare(name, data, seed, split, fold, roles)
    return result, partition(result)


def freeze(a):
    partitions = {}
    for phase in PHASES:
        for name, fold, seed in settings(phase):
            split = "final" if phase == "final" else "dev"
            data = prepare(name, a.data, seed, split, fold, a.roles)
            partitions[f"{name}:{fold}:{split}:{seed}"] = partition(data)
    return dict(v=1, pins=pins(a), partitions=partitions)


def immutable(a):
    if pins(a) != a.frozen["pins"]:
        raise ValueError("frozen source, data or protocol changed")


def audit(a, row):
    from packed_spam_score import audit_packed
    job, p = row["job"], row["payload"]
    for k in ("arm", "price", "dataset", "split", "seed", "tier", "fold"):
        if digest(job[k]) != digest(p[k]):
            raise ValueError("actor job mismatch")
    part = a.frozen["partitions"][f"{job['dataset']}:{job['fold']}:{job['split']}:{job['seed']}"]
    if (p["record"]["partition_digest"] != part["groups"] or digest(p["record"]["thresholds"]) != part["thresholds"]
            or row["origin"] != str(ROOT / IMPL) or row["actor_sha256"] != a.frozen["pins"]["sources"][str(IMPL)]):
        raise ValueError("foreign origin or partition")
    data, fingerprint = prepared(job["dataset"], a.data, job["seed"], job["split"], job["fold"], a.roles, a.manifest_hash)
    if part != fingerprint:
        raise ValueError("frozen X or group IDs differ from replay")
    metrics = audit_packed(p, data, a.roles)
    if metrics["assigned"] != 512 or row["peak_rss_kib"] <= 0:
        raise ValueError("incomplete assignments or missing RSS")
    return metrics


def select(rows):
    by = defaultdict(list)
    for row in rows:
        job = row["job"]
        if job["dataset"] == "spambase":
            by[job["arm"], job["price"]].append(row["metrics"])
    costs, means, gains = {}, {}, {}
    for arm in METHODS:
        for price in (128, 8192):
            method, baseline = by[arm, price], by["RANDOM48", price]
            if not method or len(method) != len(baseline):
                raise ValueError("incomplete dev selection")
            costs[arm, price] = Fraction(sum(m["units"] for m in method), sum(m["units"] for m in baseline))
            key = arm, price
            means[key] = gains[key] = None
            if all(m["ba"] is not None for m in method + baseline):
                means[key] = sum(Fraction(*m["ba"]) for m in method) / len(method)
                gains[key] = means[key] - sum(Fraction(*m["ba"]) for m in baseline) / len(baseline)

    valid = [arm for arm in METHODS if all(means[arm, p] is not None for p in (128, 8192))]
    cost = [arm for arm in valid if max(costs[arm, p] for p in (128, 8192)) <= 1]
    joint = [arm for arm in cost if min(gains[arm, p] for p in (128, 8192)) >= Fraction(1, 50)]
    if joint:
        tier, chosen = "joint", min(joint, key=lambda arm: costs[arm, 128])
    elif cost:
        tier = "cost"
        chosen = min(cost, key=lambda arm: (-(means[arm, 128] + means[arm, 8192]), costs[arm, 128]))
    else:
        tier = "fallback"
        chosen = min(valid or METHODS, key=lambda arm: max(costs[arm, p] for p in (128, 8192)))
    def report(value):
        return None if value is None else [value.numerator, value.denominator]
    stats = {arm: {str(p): dict(cost_ratio=report(costs[arm, p]), ba=report(means[arm, p]),
                                gain=report(gains[arm, p])) for p in (128, 8192)} for arm in METHODS}
    return dict(selected=chosen, tier=tier, eligible=joint, cost_eligible=cost, stats=stats,
                negative_dev=not joint, undefined_ba=[a for a in METHODS if a not in valid])


def check_pair(pairs, row):
    job = row["job"]
    if job["arm"] not in ("LEGACY40", "MEDIAN40"):
        return
    key = tuple(job[k] for k in ("dataset", "split", "fold", "seed", "price"))
    rec = row["payload"]["record"]
    signature = digest([[r["id"] for r in rec["receipts"]], rec["frozen"]["model"], rec["predictions"]])
    pair = pairs.setdefault(key, {})
    pair[job["arm"]] = signature
    if len(set(pair.values())) != 1:
        raise ValueError("paired MEDIAN40/LEGACY40 purchases, model or predictions differ")


def read_rows(a, phase, selected=None, *, complete=True, pairs=None):
    expected = {digest(j): j for j in jobs(phase, selected)}
    claims = a.output / (phase + "_claims")
    raw = a.output / (phase + ".jsonl")
    done, rows, previous = set(), [], None
    if pairs is None:
        pairs = {}
    if raw.exists():
        with raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                key = digest(row["job"])
                seal = row.pop("seal")
                if key not in expected or key in done or seal != dict(manifest=a.manifest_hash, previous=previous, sha256=digest(row)):
                    raise ValueError("foreign, repeated or unsealed row")
                if load(claims / (key + ".json")) != dict(manifest=a.manifest_hash, job=row["job"]):
                    raise ValueError("foreign claim")
                if digest(audit(a, row)) != digest(row["metrics"]):
                    raise ValueError("metrics differ from replay")
                check_pair(pairs, row)
                row["seal"] = seal
                previous = digest(row)
                done.add(key)
                rows.append(dict(job=row["job"], metrics=row["metrics"]))
    if {k + ".json" for k in done} != ({p.name for p in claims.iterdir()} if claims.exists() else set()):
        raise ValueError("unresolved claim; actor must never be repeated")
    if complete and done != expected.keys():
        raise ValueError("incomplete matrix")
    if complete and any(len(pair) != 2 for pair in pairs.values()):
        raise ValueError("incomplete MEDIAN40/LEGACY40 pair")
    return rows, previous


def approval(a):
    rows, _ = read_rows(a, "dev")
    return dict(manifest=a.manifest_hash, source=str(ROOT), output=str(a.output), claim=str(a.claim),
                dev_sha256=sha(a.output / "dev.jsonl"), **select(rows))


def global_claim(a, binding):
    if a.claim.exists():
        if load(a.claim) != binding:
            raise ValueError("foreign global claim")
    elif a.summarize or any(a.output.glob("final*")):
        raise ValueError("missing global claim")
    else:
        a.claim.parent.mkdir(parents=True, exist_ok=True)
        write(a.claim, binding)


def run_actor(a, job, key, work):
    job_path = work / (key + ".job.json")
    write(job_path, job)
    command = [a.audit_wrapper, a.python, "-u", ROOT / "benchmarks/packed_spam_worker.py", "--source", ROOT,
               "--job", job_path, "--roles", a.roles, "--data", a.data, "--journal", work / (key + ".journal.jsonl")]
    started = time.monotonic()
    with (work / (key + ".stderr")).open("x") as err, (work / (key + ".stdout")).open("x+") as out:
        with subprocess.Popen(command, stdout=out, stderr=err, start_new_session=True) as child:
            try:
                while child.poll() is None:
                    remaining = 180 - (time.monotonic() - started)
                    if remaining <= 0:
                        raise TimeoutError("180s actor timeout")
                    try:
                        child.wait(timeout=min(45, remaining))
                    except subprocess.TimeoutExpired:
                        print(f"actor alive {key} {time.monotonic()-started:.0f}s", flush=True)
                if child.returncode:
                    raise subprocess.CalledProcessError(child.returncode, command)
            finally:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        out.seek(0)
        row = json.load(out)
    row["worker_wall_seconds"] = f"{time.monotonic()-started:.9f}"
    if digest(row["job"]) != digest(job):
        raise ValueError("unassigned job")
    return row


def run_locked(a):
    stop = a.output / "STOP.json"
    if stop.exists():
        raise ValueError("persistent STOP")
    manifest = a.output / "SOURCE_RUN_MANIFEST.json"
    if a.prepare:
        write(manifest, freeze(a))
        return
    a.frozen = load(manifest)
    a.manifest_hash = digest(a.frozen)
    try:
        immutable(a)
        approved = a.output / "FINAL_APPROVAL.json"
        if a.approve:
            write(approved, approval(a))
            return
        selected = None
        if a.phase == "final":
            binding = approval(a)
            if binding != load(approved):
                raise ValueError("approval binding changed")
            global_claim(a, binding)
            selected = binding["selected"]
        assigned = jobs(a.phase, selected)
        pairs = {}
        rows, previous = read_rows(a, a.phase, selected, complete=a.summarize, pairs=pairs)
        if not a.summarize:
            done = {digest(row["job"]) for row in rows}
            work, claims = [a.output / (a.phase + suffix) for suffix in ("_work", "_claims")]
            for d in (work, claims):
                d.mkdir(exist_ok=True)
            for job in assigned:
                key = digest(job)
                if key in done:
                    continue
                if stop.exists():
                    raise ValueError("persistent STOP")
                immutable(a)
                write(claims / (key + ".json"), dict(manifest=a.manifest_hash, job=job))
                row = run_actor(a, job, key, work)
                immutable(a)
                started = time.monotonic()
                row["metrics"] = audit(a, row)
                row["audit_wall_seconds"] = f"{time.monotonic()-started:.9f}"
                row["seal"] = dict(manifest=a.manifest_hash, previous=previous, sha256=digest(row))
                write(a.output / (a.phase + ".jsonl"), row, "a")
                previous = digest(row)
                done.add(key)
                check_pair(pairs, row)
                print(f"{len(done)}/{len(assigned)} {job['dataset']} {job['arm']}", flush=True)
            rows, _ = read_rows(a, a.phase, selected)
        print(json.dumps(dict(phase=a.phase, trials=len(rows), selected=selected,
                              assigned=sum(r["metrics"]["assigned"] for r in rows),
                              units=sum(r["metrics"]["units"] for r in rows))), flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        if not a.summarize and not stop.exists():
            write(stop, dict(error=repr(exc), phase=a.phase, manifest=a.manifest_hash))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "data", "roles", "protocol", "claim", "audit-wrapper"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--phase", choices=PHASES, default="smoke")
    modes = parser.add_mutually_exclusive_group()
    for name in ("prepare", "approve", "summarize"):
        modes.add_argument("--" + name, action="store_true")
    a = parser.parse_args()
    for k in ("output", "data", "roles", "protocol", "claim", "audit_wrapper"):
        setattr(a, k, getattr(a, k).resolve())
    a.python = a.python.absolute()
    if a.summarize:
        if not a.output.is_dir():
            raise ValueError("missing output")
    else:
        a.output.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, _frame):
        raise InterruptedError(f"runner interrupted by signal {signum}")

    with lease(a.output, readonly=a.summarize):
        previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGTERM, signal.SIGHUP)}
        try:
            run_locked(a)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
