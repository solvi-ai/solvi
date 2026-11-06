"""Review-gated serial subprocess launcher; importing never schedules work."""

from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

# Pin local source before importing any solvi module, regardless of PYTHONPATH.
LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE / "src"))
sys.path.insert(0, str(LANE / "benchmarks"))
from experience_v2 import TrialLog, WorkBudget, compare_s1, compare_s2, load_baseline, digest  # noqa: E402

BASE_REVISION = "3850ef2eece111c5b82de6647867f2977f00495c"
ARMS = ("OLD_EMPIRICAL", "NEW_EMPIRICAL", "OLD_CHECKED", "NEW_CHECKED", "FIXED_MEASURE", "FORMULA", "FORMULA_CHECKED")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(lane, baseline_root, protocol):
    lane, base = Path(lane).resolve(), Path(baseline_root).resolve()
    revision = subprocess.check_output(["git", "-C", str(base), "rev-parse", "HEAD"], text=True).strip()
    if revision != BASE_REVISION:
        raise ValueError("baseline revision must be exact 3850ef2e")
    tracked = subprocess.check_output(
        ["git", "-C", str(base), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if tracked.strip():
        raise ValueError("baseline has tracked modifications")
    sources = {str(p.relative_to(lane)): sha(p) for p in sorted((lane / "src").rglob("*.py"))}
    sources.update({str(p.relative_to(lane)): sha(p) for p in sorted((lane / "benchmarks").glob("experience_v2*.py"))})
    for name in ("pyproject.toml", "uv.lock"):
        if (lane / name).exists():
            sources[name] = sha(lane / name)
    return dict(
        version=1,
        source=sources,
        protocol=sha(protocol),
        protocol_path=str(Path(protocol).resolve()),
        baseline_revision=revision,
        baseline={
            name: sha(base / "src/solvi/experimental/experience" / (name + ".py"))
            for name in ("programs", "acquisition")
        },
        python=sys.version,
        implementation=platform.python_implementation(),
        platform=platform.platform(),
    )


def verify_loaded_modules(manifest):
    for name, module in list(sys.modules.items()):
        if name != "solvi" and not name.startswith("solvi."):
            continue
        filename = getattr(module, "__file__", None)
        if filename is None:
            continue
        path = Path(filename).resolve()
        try:
            relative = str(path.relative_to(LANE))
        except ValueError as error:
            raise ValueError("foreign loaded solvi module: " + name) from error
        if relative not in manifest["source"] or sha(path) != manifest["source"][relative]:
            raise ValueError("loaded solvi source hash mismatch: " + name)


def verify_worker_claim(directory, spec, phase, manifest):
    directory = Path(directory).resolve()
    expected = dict(spec=spec, config=dict(phase=phase, manifest=digest(manifest)))
    if directory.name != trial_id(spec) or json.loads((directory / "running.json").read_text()) != expected:
        raise ValueError("worker has no matching durable running claim")
    expected_run = json.loads((directory.parent / "run.json").read_text())
    if expected_run != dict(out=str(directory.parent), phase=phase, manifest=digest(manifest)):
        raise ValueError("worker output run binding mismatch")
    if (directory / "result.json").exists():
        raise ValueError("worker trial already has terminal output")


def verify_approval(manifest_path, manifest, phase):
    if phase == "smoke":
        return
    approval = Path(manifest_path).with_name("HARNESS_REVIEW_APPROVAL.json")
    data = json.loads(approval.read_text())
    if (
        set(data) != {"manifest_hash", "verdict", "phases", "reviewer"}
        or data["manifest_hash"] != digest(manifest)
        or data["verdict"] != "PASS"
        or phase not in data["phases"]
        or not data["reviewer"]
    ):
        raise ValueError("independent harness review approval mismatch")


def claim_final(manifest_path, manifest, out, *, resume=False):
    claim = Path(manifest_path).with_name("FINAL_RUN_CLAIM.json")
    expected = dict(manifest_hash=digest(manifest), out=str(Path(out).resolve()))
    if resume:
        if json.loads(claim.read_text()) != expected:
            raise ValueError("foreign final resume claim")
        return
    with claim.open("x") as stream:
        json.dump(expected, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(claim.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def trial_plan(phase):
    if phase not in ("smoke", "development", "final"):
        raise ValueError("invalid phase")
    seeds = [0] if phase == "smoke" else range(5001, 5013) if phase == "development" else range(8101, 8113)
    for seed in seeds:
        for wire in (False, True):
            for profile in ("threshold", "conditional", "hidden"):
                yield dict(kind="s1", seed=seed, wire=wire, profile=profile)
            for family in ("inventory", "eligibility", "schedule"):
                yield dict(kind="s2", seed=seed, wire=wire, family=family)
                for arm in ARMS:
                    yield dict(kind="operational", seed=seed, wire=wire, family=family, arm=arm)


def trial_id(spec):
    return spec["kind"] + "-" + digest(spec)


def execute_trial(spec, baseline_root, manifest, phase, directory):
    old = load_baseline(baseline_root, manifest["baseline"])
    approved = phase if phase != "smoke" else None
    if spec["kind"] == "s1":
        parents = {name: WorkBudget(200000) for name in ("old", "new")}
        cells = []
        for label, prices in [("A", (5, 1, 2, 1, 1, 5)), ("B", (9, 2, 1, 1, 2, 9))]:
            for horizon in (1, 2, 3):
                cells.append(
                    dict(
                        cell=label + str(horizon),
                        pair=compare_s1(
                            spec["seed"],
                            spec["profile"],
                            prices,
                            horizon,
                            old["acquisition"],
                            parents,
                            spec["wire"],
                            approved_phase=approved,
                            include_reference=False,
                        ),
                    )
                )
        return dict(
            status="completed",
            cells=cells,
            ledgers={k: dict(spent=v.spent, counts=v.counts) for k, v in parents.items()},
        )
    if spec["kind"] == "s2":
        return dict(
            status="completed",
            diagnostic_only=True,
            pair=compare_s2(spec["seed"], spec["family"], old["programs"], spec["wire"], approved_phase=approved),
        )
    from experience_v2_strategies import run_trial

    return run_trial(
        Path(directory) / "operational",
        spec["seed"],
        spec["family"],
        spec["arm"],
        old=old["programs"],
        wire=spec["wire"],
        approved_phase=approved,
    )


def run_isolated(spec, *, baseline_root, manifest_path, phase, directory, heartbeat=30):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        json.dumps(spec),
        "--phase",
        phase,
        "--baseline-root",
        str(baseline_root),
        "--manifest",
        str(manifest_path),
        "--out",
        str(directory),
    ]
    started = time.monotonic()
    with (directory / "worker.stdout").open("w") as stdout, (directory / "worker.stderr").open("w") as stderr:
        child = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            env={**os.environ, "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1", "PYTHONPATH": str(LANE / "src")},
        )
        try:
            while True:
                try:
                    code = child.wait(timeout=heartbeat)
                    break
                except subprocess.TimeoutExpired:
                    print(
                        json.dumps(
                            dict(heartbeat=trial_id(spec), pid=child.pid, elapsed_seconds=time.monotonic() - started)
                        ),
                        flush=True,
                    )
        except BaseException:
            child.terminate()
            child.wait(timeout=10)
            raise
    result_path = directory / "result.json"
    if code != 0 or not result_path.exists():
        return dict(status="censored", returncode=code, wall_seconds=time.monotonic() - started)
    result = json.loads(result_path.read_text())
    result["parent_wall_seconds"] = time.monotonic() - started
    if spec["kind"] == "s1":
        evaluator_command = command + ["--reference"]
        with (directory / "reference.stdout").open("w") as stdout, (directory / "reference.stderr").open("w") as stderr:
            evaluator = subprocess.Popen(
                evaluator_command,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, "PYTHONPATH": str(LANE / "src"), "OMP_NUM_THREADS": "1"},
            )
            while True:
                try:
                    evaluator_code = evaluator.wait(timeout=heartbeat)
                    break
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(heartbeat="reference:" + trial_id(spec), pid=evaluator.pid)), flush=True)
        if evaluator_code != 0 or not (directory / "reference.json").exists():
            return dict(status="censored", reason="independent reference process failed", actor=result)
        reference = json.loads((directory / "reference.json").read_text())
        for cell, evaluated in zip(result["cells"], reference["cells"]):
            for version in ("old", "new"):
                cell["pair"][version]["reference"] = evaluated[version]
        result["evaluator_measurements"] = {k: v for k, v in reference.items() if k != "cells"}
    return result


def portable_measurements(value):
    # experience canonical values deliberately exclude floats. Timing remains
    # decimal text in evaluator output, never a work-ledger unit.
    if isinstance(value, float):
        import math

        if not math.isfinite(value):
            raise ValueError("nonfinite evaluator measurement")
        return format(value, ".12g")
    if type(value) is dict:
        return {k: portable_measurements(v) for k, v in value.items()}
    if type(value) in (tuple, list):
        return [portable_measurements(v) for v in value]
    return value


def run_plan(out, phase, manifest_path, manifest, baseline_root, *, resume=False, plan=None):
    out = Path(out)
    if out.exists() and not resume:
        raise ValueError("fresh output directory required")
    out.mkdir(parents=True, exist_ok=True)
    run_binding = dict(out=str(out.resolve()), phase=phase, manifest=digest(manifest))
    binding_path = out / "run.json"
    if binding_path.exists():
        if json.loads(binding_path.read_text()) != run_binding:
            raise ValueError("run binding mismatch")
    else:
        with binding_path.open("x") as stream:
            json.dump(run_binding, stream)
            stream.flush()
            os.fsync(stream.fileno())
    config = dict(phase=phase, manifest=digest(manifest))
    log = TrialLog(out / "trials.jsonl", config, resume=resume)
    for spec in trial_plan(phase) if plan is None else plan:
        identity = trial_id(spec)
        if identity in log.finished:
            continue
        directory = out / identity
        # Existence is an at-most-once durable claim, including ambiguous crashes.
        if directory.exists():
            log.append(identity, "censored", dict(spec=spec, reason="ambiguous prior running trial; never rerun"))
            continue
        directory.mkdir()
        claim = directory / "running.json"
        with claim.open("x") as stream:
            json.dump(dict(spec=spec, config=config), stream)
            stream.flush()
            os.fsync(stream.fileno())
        parent_fd = os.open(out, os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        result = run_isolated(
            spec, baseline_root=baseline_root, manifest_path=manifest_path, phase=phase, directory=directory
        )
        result = portable_measurements(result)
        status = result.get("status", "error")
        if status not in TrialLog.TERMINAL:
            result = dict(status="censored", reason="malformed worker terminal status", raw=result)
            status = "censored"
        log.append(identity, status, dict(spec=spec, result=result))
        print(json.dumps(dict(terminal=identity, status=status, done=len(log.finished))), flush=True)
    return log


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "development", "final"), default="smoke")
    parser.add_argument("--out", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--freeze-manifest", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    if args.freeze_manifest:
        if not args.protocol:
            parser.error("--protocol required for freeze")
        manifest = make_manifest(Path(__file__).parents[1], args.baseline_root, args.protocol)
        with path.open("x") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        return 0
    manifest = json.loads(path.read_text())
    verify_approval(path, manifest, args.phase)
    verify_loaded_modules(manifest)
    if args.worker:
        if make_manifest(Path(__file__).parents[1], args.baseline_root, manifest["protocol_path"]) != manifest:
            raise ValueError("worker frozen source/runtime/protocol changed")
        spec = json.loads(args.worker)
        valid_seeds = {"smoke": range(0, 1), "development": range(5001, 5013), "final": range(8101, 8113)}
        if spec["seed"] not in valid_seeds[args.phase]:
            raise ValueError("worker phase/seed mismatch")
        if args.reference:
            actor_path = Path(args.out) / "result.json"
            actor = json.loads(actor_path.read_text())
            # The actor already consumed this exact claim; evaluator reads only.
            expected = dict(spec=spec, config=dict(phase=args.phase, manifest=digest(manifest)))
            if json.loads((Path(args.out) / "running.json").read_text()) != expected:
                raise ValueError("reference claim mismatch")
            from experience_v2 import s1_fixture, s1_reference

            started = time.monotonic()
            cells = []
            for cell in actor["cells"]:
                prices = (5, 1, 2, 1, 1, 5) if cell["cell"][0] == "A" else (9, 2, 1, 1, 2, 9)
                worlds, question, probes = s1_fixture(spec["seed"], spec["profile"], prices, spec["wire"])
                cells.append(
                    {
                        version: s1_reference(
                            worlds, question, probes, int(cell["cell"][1]), cell["pair"][version]["remaining"]
                        )
                        for version in ("old", "new")
                    }
                )
            result = dict(
                cells=cells,
                evaluator_wall_seconds=time.monotonic() - started,
                evaluator_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            )
            with (Path(args.out) / "reference.json").open("x") as stream:
                json.dump(result, stream)
                stream.flush()
                os.fsync(stream.fileno())
            return 0
        verify_worker_claim(args.out, spec, args.phase, manifest)
        with (Path(args.out) / "execution.claim").open("x") as stream:
            stream.write(digest(spec))
            stream.flush()
            os.fsync(stream.fileno())
        started = time.monotonic()
        try:
            result = execute_trial(spec, args.baseline_root, manifest, args.phase, args.out)
        except Exception as error:  # noqa: BLE001 — each worker failure is a terminal audit result
            result = dict(status="error", error=type(error).__name__, message=str(error))
        result["worker_wall_seconds"] = time.monotonic() - started
        result["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_path = Path(args.out) / "result.json"
        with result_path.open("x") as stream:
            json.dump(result, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        return 0
    if not args.protocol:
        parser.error("--protocol required for source revalidation")
    if make_manifest(Path(__file__).parents[1], args.baseline_root, args.protocol) != manifest:
        raise ValueError("frozen source/runtime/protocol changed")
    if args.phase == "final":
        claim_final(path, manifest, args.out, resume=args.resume)
    run_plan(args.out, args.phase, path, manifest, args.baseline_root, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
