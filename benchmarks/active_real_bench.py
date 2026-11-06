"""Frozen paired cost study and external-data acquisition with fail-closed trials.

Host supplies all paths. --prepare precedes smoke/dev; --approve freezes dev
selection; final runs once. --summarize calls only the independent auditors.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import fcntl
import io
import tarfile
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from active_real_data import SOURCES, prepare_dataset
from active_real_score import audit_real
from active_real_trial import ARMS, PRICES
from refinement_score import audit_row as audit_program
from refinement_tasks import tasks
from solvi.experimental.experience._common import digest

SEEDS = dict(dev_real=(5901, 5902, 5903), final_real=(6901, 6902, 6903, 6904, 6905),
             dev_program=(7901, 7902, 7903), final_program=(8901, 8902, 8903, 8904, 8905))


def atomic(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def jobs(phase, cap=1024):
    if phase == "smoke":
        real_seeds, program_seeds = (501,), (7501,)
        names, prices, problem_names = ("banknote",), (128,), (tasks()[0].task_id,)
    else:
        cohort = "final" if phase == "final" else "dev"
        real_seeds, program_seeds = SEEDS[cohort + "_real"], SEEDS[cohort + "_program"]
        names, prices, problem_names = tuple(SOURCES), PRICES, tuple(p.task_id for p in tasks())
    arms = ("QBC_CAPPED",) if phase == "dev_alt" else ARMS
    rows = []
    for dataset in names:
        for seed in real_seeds:
            for price in prices:
                for arm in arms:
                    rows.append(dict(tier="real", dataset=dataset, seed=seed, price=price, arm=arm,
                                     cap=cap if arm == "QBC_CAPPED" else 1024,
                                     split="final" if phase == "final" else "dev", phase=phase))
    if phase == "smoke":
        # Include the largest feature frame in the engineering resource estimate.
        for price in PRICES:
            for arm in ("QBC_FULL", "QBC_CAPPED"):
                rows.append(dict(tier="real", dataset="spambase", seed=501, price=price, arm=arm,
                                 cap=1024, split="dev", phase=phase))
    if phase != "dev_alt":
        for task in problem_names:
            for seed in program_seeds:
                for arm in ("OLD_ACTIVE", "NEW_ACTIVE"):
                    rows.append(dict(tier="program", task=task, seed=seed, arm=arm, phase=phase))
    return rows


def manifest(args):
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src/solvi").rglob("*.py"))
    paths += sorted((root / "benchmarks").glob("active_real_*.py"))
    paths += [root / "benchmarks" / name for name in ("refinement_tasks.py", "refinement_bench.py", "refinement_score.py")]
    paths += [root / "pyproject.toml", root / "uv.lock"]
    current = {str(path.relative_to(root)): sha(path) for path in paths}
    expected_revision = subprocess.check_output(
        ["git", "-C", str(args.baseline), "rev-parse", "--verify", args.baseline_revision + "^{commit}"],
        text=True).strip()
    actual_revision = subprocess.check_output(["git", "-C", str(args.baseline), "rev-parse", "HEAD"], text=True).strip()
    if actual_revision != expected_revision:
        raise ValueError("baseline HEAD differs from declared revision")
    archive = subprocess.check_output(["git", "-C", str(args.baseline), "archive", expected_revision,
                                       "src/solvi", "benchmarks/refinement_tasks.py", "benchmarks/refinement_bench.py",
                                       "benchmarks/refinement_score.py"])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        for item in tree:
            if item.isfile():
                if hashlib.sha256(tree.extractfile(item).read()).hexdigest() != sha(args.baseline / item.name):
                    raise ValueError("baseline file differs from committed tree: " + item.name)
    baseline = {str(path.relative_to(args.baseline)): sha(path)
                for path in sorted((args.baseline / "src/solvi").rglob("*.py"))}
    baseline.update({"benchmarks/" + name: sha(args.baseline / "benchmarks" / name)
                     for name in ("refinement_tasks.py", "refinement_bench.py", "refinement_score.py")})
    for name in ("programs", "observations", "refinement"):
        module = importlib.import_module("solvi.experimental.experience." + name)
        if Path(module.__file__).resolve() != root / "src/solvi/experimental/experience" / (name + ".py"):
            raise ValueError("foreign host module origin")
    partitions = {}
    for dataset in SOURCES:
        for split in ("dev", "final"):
            for seed in SEEDS[split + "_real"]:
                prepared = prepare_dataset(dataset, args.data, seed, split)
                partitions[f"{dataset}:{split}:{seed}"] = dict(
                    group_ids=prepared["group_ids"], thresholds=prepared["thresholds"],
                    pool_ids=[r["id"] for r in prepared["pool"]], test_ids=[r["id"] for r in prepared["test"]],
                    pool_X_sha256=digest([r["inputs"] for r in prepared["pool"]]),
                    test_X_sha256=digest([r["inputs"] for r in prepared["test"]]),
                )
    return dict(v=1, sources=current, baseline=baseline, protocol_sha256=sha(args.protocol), partitions=partitions,
                data_manifest=json.loads((args.data / "DATA_MANIFEST.json").read_text()),
                baseline_revision=actual_revision,
                seeds={k: list(v) for k, v in SEEDS.items()}, allowed_caps=[1024, 4096],
                real_budget=4000000, program_budget=100000, python=platform.python_version(),
                real_label_prices=list(PRICES), fixed_final_holdout_groups=512)


def audit(result, data):
    job, row = result["job"], result["payload"]
    if job["tier"] == "real":
        if any(row[name] != job[name] for name in ("dataset", "seed", "price", "arm", "cap", "split")):
            raise ValueError("actor job mismatch")
        return audit_real(row, data)
    problem = next(p for p in tasks() if p.task_id == job["task"])
    config = row.get("learning", {}).get("config", {})
    if job["arm"] == "OLD_ACTIVE":
        if "active_policy" in config or "active_work_limit" in config:
            raise ValueError("old actor used the new policy")
    elif job["arm"] == "NEW_ACTIVE":
        if config.get("active_policy") != "onepass-bounded-v1" or digest(config.get("active_work_limit")) != digest(1024):
            raise ValueError("new actor policy or cap changed")
    else:
        raise ValueError("unknown program arm")
    counts, work = audit_program(row, problem, job["seed"], "ACTIVE")
    return dict(assigned=sum(counts.values()), correct=counts["correct"], wrong=counts["wrong"],
                units=row["budget"]["spent"], auditorwork=work)


def expected_source(args, job):
    return args.baseline if job["arm"] == "OLD_ACTIVE" else Path(__file__).resolve().parents[1]


def validate_origin(args, row):
    source = expected_source(args, row["job"]) / "src/solvi/experimental/experience/refinement.py"
    if row.get("origin") != str(source) or row.get("actor_sha256") != sha(source):
        raise ValueError("foreign actor origin or source hash")


def immutable(args):
    current = Path(__file__).resolve().parents[1]
    for prefix, root in (("sources", current), ("baseline", args.baseline)):
        for relative, expected in args.frozen[prefix].items():
            if sha(root / relative) != expected:
                raise ValueError("source changed during run: " + relative)
    if sha(args.protocol) != args.frozen["protocol_sha256"]:
        raise ValueError("protocol changed during run")


def read_rows(args, phase, cap=1024, *, require_complete=True):
    raw = args.output / (phase + ".jsonl")
    expected = {digest(job): job for job in jobs(phase, cap)}
    result, previous = [], "0" * 64
    if raw.exists():
        with raw.open() as stream:
            for line in stream:
                row = json.loads(line)
                body = {k: v for k, v in row.items() if k != "seal"}
                if row.get("seal") != dict(manifest=args.manifest_hash, previous=previous, sha256=digest(body)):
                    raise ValueError("raw manifest/checksum chain")
                key = digest(row["job"])
                if key not in expected or any(digest(old["job"]) == key for old in result):
                    raise ValueError("duplicate or unassigned terminal trial")
                claim_path = args.output / (phase + "_claims") / (key + ".json")
                claim = json.loads(claim_path.read_text())
                if digest(claim) != digest(dict(job=row["job"], manifest=args.manifest_hash)):
                    raise ValueError("foreign trial claim")
                validate_origin(args, row)
                metrics = audit(row, args.data)
                if digest(row["metrics"]) != digest(metrics):
                    raise ValueError("derived metrics differ from independent audit")
                previous = digest(row)
                result.append(dict(job=row["job"], metrics=metrics))
    if require_complete and {digest(row["job"]) for row in result} != set(expected):
        raise ValueError("incomplete assigned trial matrix")
    claims = args.output / (phase + "_claims")
    if claims.exists() and {p.stem for p in claims.glob("*.json")} != {digest(r["job"]) for r in result}:
        raise ValueError("unsettled or foreign trial claim; no automatic actor retry")
    return result, previous


def summarize(rows):
    grouped = {}
    for row in rows:
        job, metrics = row["job"], row["metrics"]
        key = (f"real:{job['dataset']}:{job['price']}:{job['arm']}:{job['cap']}"
               if job["tier"] == "real" else "program:" + job["arm"])
        group = grouped.setdefault(key, dict(trials=0, units=0, assigned=0, correct=0, wrong=0,
                                             labels=0, ba_fractions=[], fallbacks=0))
        for name in ("units", "assigned", "correct", "wrong", "labels", "fallbacks"):
            group[name] += metrics.get(name, 0)
        group["trials"] += 1
        if job["tier"] == "real":
            group["ba_fractions"].append(metrics["ba"])
    return dict(trials=len(rows), groups=grouped)


def choose_cap(args, *, include_alternate=True):
    dev, _ = read_rows(args, "dev")
    candidates = {1024: dev}
    alternate_exists = (args.output / "dev_alt.jsonl").exists() or any((args.output / "dev_alt_claims").glob("*.json"))
    if include_alternate and alternate_exists:
        alternate, _ = read_rows(args, "dev_alt", 4096)
        candidates[4096] = alternate

    def statistics(rows, arm):
        selected = [row["metrics"] for row in rows if row["job"]["tier"] == "real"
                    and row["job"]["price"] == 128 and row["job"]["arm"] == arm]
        if len(selected) != 9 or any(m["ba"] is None for m in selected):
            return None, sum(m["units"] for m in selected)
        return sum((Fraction(*m["ba"]) for m in selected), Fraction()) / 9, sum(m["units"] for m in selected)

    base_ba, base_units = statistics(dev, "QBC_FULL")
    scores = {cap: statistics(rows, "QBC_CAPPED") for cap, rows in candidates.items()}
    first_ba, first_units = scores[1024]
    trigger = (first_ba is None or base_ba is None or first_ba < base_ba - Fraction(1, 100)
               or first_units * 100 > base_units * 85)
    if include_alternate and trigger and 4096 not in candidates:
        raise ValueError("predefined diagnostic cap4096 required before freeze")
    if not trigger and 4096 in candidates:
        raise ValueError("untriggered diagnostic variant")
    eligible = [cap for cap, (ba, _) in scores.items()
                if ba is not None and base_ba is not None and ba >= base_ba - Fraction(1, 100)]
    selected = min(eligible, key=lambda cap: (scores[cap][1], cap != 1024)) if eligible else 1024
    return selected, dict(trigger_alt=trigger, eligible=eligible,
                          baseline_units=base_units, candidate_units={str(k): v[1] for k, v in scores.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "data", "baseline", "protocol", "claim", "audit-wrapper"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True, help="expected immutable baseline commit")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--phase", choices=("smoke", "dev", "dev_alt", "final"), default="smoke")
    parser.add_argument("--cap", type=int, choices=(1024, 4096), default=1024)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    for name in ("output", "data", "baseline", "protocol", "claim", "audit_wrapper"):
        setattr(args, name, getattr(args, name).resolve())
    args.python = Path(os.path.abspath(args.python))  # preserve the virtualenv symlink path
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "runner.lock").open("a+") as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("another runner owns this output") from error
        return run_locked(args)


def run_locked(args):
    if args.phase in ("dev", "smoke") and args.cap != 1024:
        raise ValueError("unregistered primary-phase cap")
    frozen = manifest(args)
    args.frozen = frozen
    args.manifest_hash = digest(frozen)
    stored = args.output / "SOURCE_RUN_MANIFEST.json"
    if args.prepare:
        if stored.exists():
            raise ValueError("manifest already exists")
        atomic(stored, frozen)
        print("prepared", args.manifest_hash, flush=True)
        return
    if digest(json.loads(stored.read_text())) != args.manifest_hash:
        raise ValueError("source/data/protocol changed since freeze")
    stop = args.output / "STOP.json"
    if stop.exists():
        raise ValueError("persistent STOP requires investigation")
    if args.approve:
        cap, selection = choose_cap(args)
        approval = dict(manifest=args.manifest_hash, output=str(args.output), claim=str(args.claim), cap=cap,
                        dev_sha256=sha(args.output / "dev.jsonl"),
                        alternate_sha256=sha(args.output / "dev_alt.jsonl") if selection["trigger_alt"] else None,
                        selection=selection)
        path = args.output / "FINAL_APPROVAL.json"
        if path.exists():
            raise ValueError("approval already frozen")
        atomic(path, approval)
        print(json.dumps(approval, sort_keys=True), flush=True)
        return
    if args.phase == "dev_alt":
        try:
            _, selection = choose_cap(args, include_alternate=False)
            if not selection["trigger_alt"]:
                raise ValueError("diagnostic variant not triggered")
        except ValueError as error:
            if str(error) != "predefined diagnostic cap4096 required before freeze":
                raise
        if args.cap != 4096:
            raise ValueError("diagnostic cap must be4096")
    if args.phase == "final":
        approval = json.loads((args.output / "FINAL_APPROVAL.json").read_text())
        cap, selection = choose_cap(args)
        binding = dict(manifest=args.manifest_hash, output=str(args.output), claim=str(args.claim), cap=cap,
                       dev_sha256=sha(args.output / "dev.jsonl"),
                       alternate_sha256=sha(args.output / "dev_alt.jsonl") if selection["trigger_alt"] else None,
                       selection=selection)
        if digest(approval) != digest(binding) or args.cap != cap:
            raise ValueError("final approval binding changed")
        if args.summarize and (not args.claim.exists() or digest(json.loads(args.claim.read_text())) != digest(binding)):
            raise ValueError("missing or foreign global final claim")
        if not args.summarize:
            if args.claim.exists():
                if digest(json.loads(args.claim.read_text())) != digest(binding):
                    raise ValueError("final already claimed elsewhere")
            else:
                if ((args.output / "final.jsonl").exists()
                        or any((args.output / "final_claims").glob("*.json"))
                        or any((args.output / "final_work").glob("*"))):
                    raise ValueError("missing global claim for existing final artifacts")
                args.claim.parent.mkdir(parents=True, exist_ok=True)
                with args.claim.open("x") as stream:
                    json.dump(binding, stream, sort_keys=True)
                    stream.flush()
                    os.fsync(stream.fileno())
    try:
        rows, previous = read_rows(args, args.phase, args.cap, require_complete=args.summarize)
        if args.summarize:
            atomic(args.output / (args.phase + "_summary.json"), summarize(rows))
            return
        done = {digest(row["job"]) for row in rows}
        assigned = jobs(args.phase, args.cap)
        claim_dir = args.output / (args.phase + "_claims")
        claim_dir.mkdir(exist_ok=True)
        work = args.output / (args.phase + "_work")
        work.mkdir(exist_ok=True)
        for job in assigned:
            key = digest(job)
            if key in done:
                continue
            if stop.exists():
                raise ValueError("persistent STOP before next actor")
            immutable(args)
            with (claim_dir / (key + ".json")).open("x") as stream:
                json.dump(dict(job=job, manifest=args.manifest_hash), stream)
                stream.flush()
                os.fsync(stream.fileno())
            job_path = work / (key + ".job.json")
            atomic(job_path, job)
            source = expected_source(args, job)
            command = [str(args.audit_wrapper), str(args.python), "-u", str(Path(__file__).with_name("active_real_worker.py")),
                       "--source", str(source), "--job", str(job_path), "--data", str(args.data),
                       "--journal", str(work / (key + ".journal.jsonl"))]
            started = time.monotonic()
            child = subprocess.run(command, text=True, capture_output=True, check=False)
            worker_seconds = time.monotonic() - started
            (work / (key + ".stdout")).write_text(child.stdout)
            (work / (key + ".stderr")).write_text(child.stderr)
            if child.returncode:
                raise RuntimeError(f"actor failed code{child.returncode}: {key}")
            result = json.loads(child.stdout)
            immutable(args)
            if digest(result["job"]) != digest(job):
                raise ValueError("assigned job differs")
            validate_origin(args, result)
            audit_start = time.monotonic()
            metrics = audit(result, args.data)
            audit_seconds = time.monotonic() - audit_start
            result["metrics"] = metrics
            result["worker_wall_seconds"] = f"{worker_seconds:.9f}"
            result["audit_wall_seconds"] = f"{audit_seconds:.9f}"
            result["seal"] = dict(manifest=args.manifest_hash, previous=previous, sha256=digest(result))
            with (args.output / (args.phase + ".jsonl")).open("a") as stream:
                stream.write(json.dumps(result, sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            previous = digest(result)
            rows.append(dict(job=job, metrics=metrics))
            done.add(key)
            print(f"{len(done)}/{len(assigned)} {job} units={metrics['units']}", flush=True)
        read_rows(args, args.phase, args.cap)
        atomic(args.output / (args.phase + "_summary.json"), summarize(rows))
    except Exception as error:
        atomic(stop, dict(error=repr(error), phase=args.phase, manifest=args.manifest_hash))
        raise


if __name__ == "__main__":
    main()
