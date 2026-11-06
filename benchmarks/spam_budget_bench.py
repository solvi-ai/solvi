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
from spam_budget_data import prepare
from solvi.experimental.experience._common import digest

ROOT = Path(__file__).resolve().parents[1]
IMPL = Path("src/solvi/experimental/experience/stratified.py")
ARMS = ("RANDOM48", "RANDOM40", "RANDOM32", "MIXED40", "MIXED32", "FULL_LABEL")
SEEDS = {"smoke": (901,), "dev": range(9501, 9507), "final": range(9901, 9911)}


def jobs(phase, selected=None):
    if phase == "final" and selected not in ("MIXED40", "MIXED32"):
        raise ValueError("frozen MIXED arm required")
    result = []
    for name in SOURCES:
        for seed in SEEDS[phase]:
            for price in (128, 8192):
                for arm in ARMS:
                    if phase == "final" and arm.startswith("MIXED") and arm != selected:
                        continue
                    if phase == "smoke" and not (
                        (name == "banknote" and price == 128)
                        or (name == "spambase" and arm in ("MIXED40", "MIXED32", "RANDOM48"))
                    ):
                        continue
                    result.append(dict(tier="budget_real", phase=phase, split="final" if phase == "final" else "dev",
                                       dataset=name, seed=seed, price=price, arm=arm))
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
def lease(output):
    with (output / "runner.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def pins(a):
    files = list((ROOT / "src").rglob("*.py"))
    files += list((ROOT / "benchmarks").glob("spam_budget*.py"))
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
                scorer=sources["benchmarks/spam_budget_score.py"])


def partition(data):
    return dict(groups=digest(data["group_ids"]), thresholds=digest(data["thresholds"]),
                pool_X=digest([[r["id"], r["inputs"]] for r in data["pool"]]),
                test_X=digest([[r["id"], r["inputs"]] for r in data["test"]]))


@lru_cache(maxsize=1)
def prepared(name, data, seed, split, roles, manifest):
    result = prepare(name, data, seed, split, roles)
    return result, partition(result)


def freeze(a):
    partitions = {}
    for phase in SEEDS:
        for job in jobs(phase, "MIXED40"):
            name, split, seed = (job[k] for k in ("dataset", "split", "seed"))
            key = f"{name}:{split}:{seed}"
            if key in partitions:
                continue
            data = prepare(name, a.data, seed, split, a.roles)
            partitions[key] = partition(data)
    return dict(v=1, pins=pins(a), partitions=partitions)


def immutable(a):
    if pins(a) != a.frozen["pins"]:
        raise ValueError("frozen source, data or protocol changed")


def audit(a, row):
    from spam_budget_score import audit_budget
    job, p = row["job"], row["payload"]
    for k in ("arm", "price", "dataset", "split", "seed", "tier"):
        if digest(job[k]) != digest(p[k]):
            raise ValueError("actor job mismatch")
    part = a.frozen["partitions"][f"{job['dataset']}:{job['split']}:{job['seed']}"]
    if (p["partition_digest"] != part["groups"] or digest(p["thresholds"]) != part["thresholds"]
            or row["origin"] != str(ROOT / IMPL) or row["actor_sha256"] != a.frozen["pins"]["sources"][str(IMPL)]):
        raise ValueError("foreign origin or partition")
    data, fingerprint = prepared(job["dataset"], a.data, job["seed"], job["split"], a.roles, a.manifest_hash)
    if part != fingerprint:
        raise ValueError("frozen X or group IDs differ from replay")
    metrics = audit_budget(p, a.data, a.roles, prepared=data)
    if metrics["assigned"] != 512 or row["peak_rss_kib"] <= 0:
        raise ValueError("incomplete assignments or missing RSS")
    return metrics


def select(rows):
    by = defaultdict(list)
    for row in rows:
        j = row["job"]
        if j["dataset"] == "spambase":
            by[j["arm"], j["price"]].append(row["metrics"])
    eligible, ratios, undefined = [], {}, []
    for arm in ("MIXED40", "MIXED32"):
        ok = True
        for price in (128, 8192):
            method, random = by[arm, price], by["RANDOM48", price]
            if not method or len(method) != len(random):
                raise ValueError("incomplete dev selection")
            cost = Fraction(sum(m["units"] for m in method), sum(m["units"] for m in random))
            if any(m["ba"] is None for m in method + random):
                undefined.append(dict(arm=arm, price=price))
                ok = False
                continue
            gain = (sum(Fraction(*m["ba"]) for m in method) - sum(Fraction(*m["ba"]) for m in random)) / len(random)
            ok = ok and cost <= 1 and gain >= Fraction(2, 100)
            if price == 128:
                ratios[arm] = cost
        if ok:
            eligible.append(arm)
    return dict(eligible=eligible, undefined_ba=undefined, selected=min(eligible, key=ratios.get) if eligible else "MIXED40")


def read_rows(a, phase, selected=None, *, complete=True):
    expected = {digest(j): j for j in jobs(phase, selected)}
    claims = a.output / (phase + "_claims")
    raw = a.output / (phase + ".jsonl")
    done, rows, previous = set(), [], None
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
                row["seal"] = seal
                previous = digest(row)
                done.add(key)
                rows.append(dict(job=row["job"], metrics=row["metrics"]))
    if done != {f.stem for f in claims.glob("*.json")}:
        raise ValueError("unresolved claim; actor must never be repeated")
    if complete and done != expected.keys():
        raise ValueError("incomplete matrix")
    return rows, previous


def approval(a):
    rows, _ = read_rows(a, "dev")
    return dict(manifest=a.manifest_hash, output=str(a.output), claim=str(a.claim),
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
    command = [a.audit_wrapper, a.python, "-u", ROOT / "benchmarks/spam_budget_worker.py", "--source", ROOT,
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
        rows, previous = read_rows(a, a.phase, selected, complete=a.summarize)
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
                print(f"{len(done)}/{len(assigned)} {job['dataset']} {job['arm']}", flush=True)
            rows, _ = read_rows(a, a.phase, selected)
        print(json.dumps(dict(phase=a.phase, trials=len(rows), selected=selected,
                              assigned=sum(r["metrics"]["assigned"] for r in rows),
                              units=sum(r["metrics"]["units"] for r in rows))), flush=True)
    except Exception as exc:
        if not a.summarize and not stop.exists():
            write(stop, dict(error=repr(exc), phase=a.phase, manifest=a.manifest_hash))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "data", "roles", "protocol", "claim", "audit-wrapper"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--phase", choices=SEEDS, default="smoke")
    modes = parser.add_mutually_exclusive_group()
    for name in ("prepare", "approve", "summarize"):
        modes.add_argument("--" + name, action="store_true")
    a = parser.parse_args()
    for k in ("output", "data", "roles", "protocol", "claim", "audit_wrapper"):
        setattr(a, k, getattr(a, k).resolve())
    a.python = a.python.absolute()
    a.output.mkdir(parents=True, exist_ok=True)
    with lease(a.output):
        run_locked(a)


if __name__ == "__main__":
    main()
