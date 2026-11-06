"""Engineering seed-0 SIGKILL recovery through the production CLI coordinator."""

import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

BENCH = Path(__file__).resolve().parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))
import experience_v2_run as run  # noqa: E402


def children(pid):
    try:
        return [int(value) for value in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()]
    except FileNotFoundError:
        return []


def wait_process(process, timeout=120):
    try:
        code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        for child in children(process.pid):
            os.kill(child, signal.SIGKILL)
        process.kill()
        process.wait()
        raise
    assert code == 0


def test_sigkill_running_trial_censors_once_then_resumes_other_trial(tmp_path):
    baseline = os.environ.get("SOLVI_EXPERIENCE_V2_BASELINE")
    if not baseline:
        pytest.skip("set SOLVI_EXPERIENCE_V2_BASELINE to exact clean 3850ef2e checkout")
    protocol = tmp_path / "engineering-protocol.txt"
    protocol.write_text("Engineering process recovery smoke only; seed=0; no comparative evaluation.\n")
    manifest = run.make_manifest(BENCH.parent, baseline, protocol)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True))
    output = tmp_path / "output"
    specs = [
        dict(kind="operational", seed=0, wire=False, family="inventory", arm="OLD_EMPIRICAL"),
        dict(kind="operational", seed=0, wire=False, family="eligibility", arm="FORMULA"),
    ]
    driver = tmp_path / "driver.py"
    driver.write_text(
        "import json,sys\n"
        f"sys.path.insert(0,{str(BENCH)!r})\n"
        "import experience_v2_run as run\n"
        f"manifest=json.loads(open({str(manifest_path)!r}).read())\n"
        f'run.run_plan({str(output)!r},"smoke",{str(manifest_path)!r},manifest,{baseline!r},'
        f"resume=len(sys.argv)>1,plan={specs!r})\n"
    )
    env = {**os.environ, "PYTHONPATH": str(BENCH.parent / "src"), "OMP_NUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    first_directory = output / run.trial_id(specs[0])
    owned_workers = set()
    parent = None
    report = dict(specs=specs, baseline=str(baseline), engineering_seed=0)
    try:
        with (tmp_path / "initial.stdout").open("w") as stdout, (tmp_path / "initial.stderr").open("w") as stderr:
            parent = subprocess.Popen([sys.executable, "-u", str(driver)], stdout=stdout, stderr=stderr, env=env)
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                owned_workers.update(children(parent.pid))
                if (first_directory / "execution.claim").exists():
                    os.kill(parent.pid, signal.SIGSTOP)
                    owned_workers.update(children(parent.pid))
                    break
                if parent.poll() is not None:
                    pytest.fail("coordinator finished before crash boundary; inspect initial logs")
                time.sleep(0.005)
            else:
                pytest.fail("execution.claim boundary not reached within 120 seconds")
            assert owned_workers
            terminal_path = output / "trials.jsonl"
            assert not terminal_path.exists() or not terminal_path.read_text().strip()
            assert not (first_directory / "result.json").exists()
            report["killed_coordinator_pid"] = parent.pid
            report["killed_worker_pids"] = sorted(owned_workers)
            for pid in owned_workers:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            os.kill(parent.pid, signal.SIGKILL)
            parent.wait(timeout=10)
        # The ambiguous first directory is retained; production resume must not execute it.
        first_files = {
            str(path.relative_to(first_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first_directory.rglob("*")
            if path.is_file()
        }
        with (tmp_path / "resume.stdout").open("w") as stdout, (tmp_path / "resume.stderr").open("w") as stderr:
            resumed = subprocess.Popen(
                [sys.executable, "-u", str(driver), "--resume"], stdout=stdout, stderr=stderr, env=env
            )
            wait_process(resumed)
        records = [json.loads(line) for line in (output / "trials.jsonl").read_text().splitlines()]
        assert len(records) == 2
        assert records[0]["trial"] == run.trial_id(specs[0]) and records[0]["status"] == "censored"
        assert records[1]["trial"] == run.trial_id(specs[1]) and records[1]["status"] == "completed"
        assert first_files == {
            str(path.relative_to(first_directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in first_directory.rglob("*")
            if path.is_file()
        }
        second = records[1]["payload"]["result"]
        assert second["external_calls"]["act"] == 32
        assert sum(task["effects"] for task in second["effects"].values()) == 32
        assert all(task["effects"] <= 1 for task in second["effects"].values())
        assert second["oracle_calls"] == 0
        before = hashlib.sha256((output / "trials.jsonl").read_bytes()).hexdigest()
        effect_before = (output / run.trial_id(specs[1]) / "operational" / "result.json").read_bytes()
        with (tmp_path / "noop.stdout").open("w") as stdout, (tmp_path / "noop.stderr").open("w") as stderr:
            noop = subprocess.Popen(
                [sys.executable, "-u", str(driver), "--resume"], stdout=stdout, stderr=stderr, env=env
            )
            wait_process(noop)
        assert hashlib.sha256((output / "trials.jsonl").read_bytes()).hexdigest() == before
        assert (output / run.trial_id(specs[1]) / "operational" / "result.json").read_bytes() == effect_before
        report.update(
            verdict="PASS",
            terminal_statuses=[r["status"] for r in records],
            final_log_sha256=before,
            successful_effects=sum(task["effects"] for task in second["effects"].values()),
            actual_calls=second["external_calls"],
        )
    finally:
        if parent is not None and parent.poll() is None:
            for pid in children(parent.pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            parent.kill()
            parent.wait(timeout=10)
        (tmp_path / "process-recovery-report.json").write_text(json.dumps(report, sort_keys=True, indent=2))
