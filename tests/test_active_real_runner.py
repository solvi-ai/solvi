"""Synthetic supervisor and read-only replay gates. No external data or model fits."""

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LANE / "benchmarks"))
import active_real_bench as bench  # noqa: E402

IMPL = Path("src/solvi/experimental/experience/refinement.py")


@pytest.fixture
def harness(tmp_path, monkeypatch):
    baseline = tmp_path / "baseline"
    (baseline / IMPL).parent.mkdir(parents=True)
    (baseline / IMPL).write_text("# synthetic OLD implementation")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("synthetic protocol")
    manifest = dict(v=1, protocol_sha256=bench.sha(protocol), sources={str(IMPL): bench.sha(LANE / IMPL)}, baseline={str(IMPL): bench.sha(baseline / IMPL)})
    state = SimpleNamespace(output=tmp_path / "out", claim=tmp_path / "global.json", launches=[],
                            fail=False, origin_override=None, raw_hash_override=None, scores={},
                            frozen=manifest, wrapper=tmp_path / "wrapper", baseline=baseline)

    def jobs(phase, cap=1024):
        if phase == "smoke":
            return [dict(tier="program", task="dummy", seed=1, arm=arm, phase=phase)
                    for arm in ("OLD_ACTIVE", "NEW_ACTIVE")]
        split = "final" if phase == "final" else "dev"
        arms = ["QBC_CAPPED"] if phase == "dev_alt" else ["QBC_FULL", "QBC_CAPPED"]
        return [dict(tier="real", dataset=str(d), seed=s, price=128, arm=a,
                     cap=cap if a == "QBC_CAPPED" else 1024, split=split, phase=phase)
                for d in range(3) for s in range(3) for a in arms]

    def spawn(cmd, **kwargs):
        assert Path(cmd[0]) == state.wrapper
        assert cmd[2] == "-u" and Path(cmd[3]).name == "active_real_worker.py"
        job = json.loads(Path(cmd[cmd.index("--job") + 1]).read_text())
        source = Path(cmd[cmd.index("--source") + 1]).resolve()
        assert source == (baseline if job["arm"] == "OLD_ACTIVE" else LANE)
        assert Path(cmd[cmd.index("--journal") + 1]).parent.parent == state.output
        state.launches.append(job)
        if state.fail:
            return SimpleNamespace(stdout="", stderr="artificial child failure", returncode=17)
        cap, arm = job.get("cap", 1024), job["arm"]
        units, ba = state.scores.get((arm, cap), (80 if arm == "QBC_CAPPED" else 100, [4, 5]))
        pay = dict(assigned=10, correct=8, wrong=2, units=units, ba=ba)
        row = dict(job=job, payload=pay, origin=state.origin_override or str(source / IMPL),
                   actor_sha256=state.raw_hash_override or bench.sha(source / IMPL), peak_rss_kib=123)
        return SimpleNamespace(stdout=json.dumps(row), stderr="", returncode=0)

    monkeypatch.setattr(bench, "manifest", lambda args: state.frozen)
    monkeypatch.setattr(bench, "jobs", jobs)
    monkeypatch.setattr(bench, "audit", lambda result, data: result["payload"].copy())
    monkeypatch.setattr(bench.subprocess, "run", spawn)

    def cli(*extra):
        monkeypatch.setattr(sys, "argv", ["bench", "--output", str(state.output), "--data", str(tmp_path / "data"),
                                         "--protocol", str(protocol), "--baseline", str(baseline),
                                         "--baseline-revision", "4bbb9033",
                                         "--claim", str(state.claim), "--audit-wrapper", str(state.wrapper), *extra])
        bench.main()

    state.cli = cli
    state.plan = jobs
    cli("--prepare")
    return state


def raw_rows(h, phase):
    path = h.output / (phase + ".jsonl")
    return [json.loads(s) for s in path.read_text().splitlines()]


def rewrite(h, phase, rows):
    previous = "0" * 64
    with (h.output / (phase + ".jsonl")).open("w") as stream:
        for row in rows:
            body = {k: v for k, v in row.items() if k != "seal"}
            row["seal"] = dict(manifest=bench.digest(h.frozen), previous=previous, sha256=bench.digest(body))
            previous = bench.digest(row)
            stream.write(json.dumps(row) + "\n")


def test_smoke_resume_replays_metadata_without_launch_and_preserves_raw(harness):
    h = harness
    h.cli()
    initial = (h.output / "smoke.jsonl").read_bytes()
    assert {j["arm"] for j in h.launches} == {"OLD_ACTIVE", "NEW_ACTIVE"}
    h.cli()
    h.cli("--summarize")
    assert (h.output / "smoke.jsonl").read_bytes() == initial
    assert len(h.launches) == 2


@pytest.mark.parametrize("damage", ["missing", "duplicate", "unassigned", "metrics_type", "claim_absent",
                                    "foreign_origin", "foreign_sha", "broken_chain"])
def test_terminal_replay_rejects_bad_record_even_when_resealed(harness, damage):
    h = harness
    h.cli()
    rows = raw_rows(h, "smoke")
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0].copy())
    elif damage == "unassigned":
        rows[0]["job"]["seed"] = 999
    elif damage == "metrics_type":
        rows[0]["metrics"]["correct"] = True
        rows[0]["payload"]["correct"] = 1
    elif damage == "claim_absent":
        (h.output / "smoke_claims" / (bench.digest(rows[0]["job"]) + ".json")).unlink()
    elif damage == "foreign_origin":
        # OLD and NEW origins are independently bound even when both contain a valid implementation.
        rows[0]["origin"] = rows[1]["origin"]
    elif damage == "foreign_sha":
        rows[1]["actor_sha256"] = "0" * 64
    rewrite(h, "smoke", rows)
    if damage == "broken_chain":
        raw = h.output / "smoke.jsonl"
        raw.write_text(raw.read_text().replace('"previous": "00', '"previous": "ff', 1))
    with pytest.raises((ValueError, FileNotFoundError)):
        h.cli("--summarize")
    assert len(h.launches) == 2
    assert (h.output / "STOP.json").exists()


@pytest.mark.parametrize("damage", ["orphan", "child_failure", "origin", "sha"])
def test_unresolved_or_wrong_child_creates_persistent_stop(harness, damage):
    h = harness
    if damage == "orphan":
        job = h.plan("smoke")[0]
        claims = h.output / "smoke_claims"
        claims.mkdir()
        bench.atomic(claims / (bench.digest(job) + ".json"), dict(job=job, manifest=bench.digest(h.frozen)))
    elif damage == "child_failure":
        h.fail = True
    elif damage == "origin":
        h.origin_override = "/outside/refinement.py"
    else:
        h.raw_hash_override = "unknown"
    with pytest.raises((ValueError, RuntimeError)):
        h.cli()
    assert not (h.output / "smoke.jsonl").exists()
    assert (h.output / "STOP.json").exists()
    assert len(h.launches) == (0 if damage == "orphan" else 1)
    count = len(h.launches)
    with pytest.raises(ValueError, match="STOP"):
        h.cli()
    assert len(h.launches) == count


def test_ineligible_alt_and_nondefault_primary_cap_cannot_start(harness):
    h = harness
    with pytest.raises(ValueError):
        h.cli("--phase", "dev", "--cap", "4096")
    assert not h.launches
    h.cli("--phase", "dev")
    count = len(h.launches)
    with pytest.raises(ValueError):
        h.cli("--phase", "dev_alt", "--cap", "4096")
    assert len(h.launches) == count


def test_conditional_alt_can_resume_clean_terminal_prefix_without_replaying_it(harness):
    h = harness
    h.scores["QBC_CAPPED", 1024] = (91, [4, 5])
    h.scores["QBC_CAPPED", 4096] = (69, [4, 5])
    h.cli("--phase", "dev")
    with pytest.raises(ValueError):
        h.cli("--approve")
    h.cli("--phase", "dev_alt", "--cap", "4096")
    rows = raw_rows(h, "dev_alt")
    # Simulate a clean shutdown after three fully appended trials (no orphan).
    rewrite(h, "dev_alt", rows[:3])
    for row in rows[3:]:
        (h.output / "dev_alt_claims" / (bench.digest(row["job"]) + ".json")).unlink()
    before = len(h.launches)
    h.cli("--phase", "dev_alt", "--cap", "4096")
    assert h.launches[before:] == [r["job"] for r in rows[3:]]
    h.cli("--approve")
    approval = json.loads((h.output / "FINAL_APPROVAL.json").read_text())
    assert approval["cap"] == 4096 and approval["selection"]["trigger_alt"] is True
    assert approval["alternate_sha256"] == bench.sha(h.output / "dev_alt.jsonl")
    assert approval["dev_sha256"] == bench.sha(h.output / "dev.jsonl")


@pytest.mark.parametrize("change", ["missing", "cap", "output", "claim", "dev_hash", "selection", "manifest"])
def test_no_final_worker_before_binding_to_dev_source_and_global_paths(harness, change):
    h = harness
    if change == "missing":
        with pytest.raises((ValueError, FileNotFoundError)):
            h.cli("--phase", "final")
        assert not h.launches
        return
    h.cli("--phase", "dev")
    h.cli("--approve")
    count = len(h.launches)
    phase = ["--phase", "final"]
    if change == "cap":
        phase += ["--cap", "4096"]
    elif change == "output":
        second = h.output.parent / "copy"
        shutil.copytree(h.output, second)
        h.output = second
    elif change == "claim":
        h.claim = h.output.parent / "otherclaim"
    elif change == "manifest":
        h.frozen["v"] += 1
    else:
        path = h.output / "FINAL_APPROVAL.json"
        value = json.loads(path.read_text())
        if change == "dev_hash":
            value["dev_sha256"] = "incorrect"
        else:
            value["selection"]["trigger_alt"] = not value["selection"]["trigger_alt"]
        bench.atomic(path, value)
    with pytest.raises(ValueError):
        h.cli(*phase)
    assert len(h.launches) == count


def test_final_analysis_requires_original_global_claim(harness):
    h = harness
    h.cli("--phase", "dev")
    h.cli("--approve")
    h.cli("--phase", "final")
    raw = (h.output / "final.jsonl").read_bytes()
    done = len(h.launches)
    h.cli("--phase", "final")
    h.cli("--phase", "final", "--summarize")
    assert len(h.launches) == done and (h.output / "final.jsonl").read_bytes() == raw
    h.claim.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        h.cli("--phase", "final", "--summarize")
    assert not h.claim.exists() and len(h.launches) == done


def test_registered_matrix_has_only_assigned_combinations():
    final = bench.jobs("final", 4096)
    assert sum(j["tier"] == "real" for j in final) == 150
    assert sum(j["tier"] == "program" for j in final) == 120
    assert len({bench.digest(j) for j in final}) == 270
    diagnostic = bench.jobs("dev_alt", 4096)
    assert len(diagnostic) == 18 and all(j["arm"] == "QBC_CAPPED" and j["cap"] == 4096 for j in diagnostic)


@pytest.mark.parametrize(("primary", "alternate", "expected"), [
    ((85, [79, 100]), None, 1024),       # Both registered inclusive boundaries pass.
    ((86, [80, 100]), (86, [80, 100]), 1024),  # Eligible equal units prefer default cap.
    ((86, [80, 100]), (85, [79, 100]), 4096),
    ((85, [78, 100]), (1, [78, 100]), 1024),  # Neither quality-eligible.
    ((85, None), (90, [80, 100]), 4096),
])
def test_cap_selection_uses_only_registered_low_price_and_all_nine_ba(primary, alternate, expected, tmp_path, monkeypatch):
    args = SimpleNamespace(output=tmp_path)
    if alternate is not None:
        (tmp_path / "dev_alt.jsonl").touch()

    def read(args, phase, *a, **kw):
        result = []
        for i in range(9):
            for arm in (["QBC_CAPPED"] if phase == "dev_alt" else ["QBC_FULL", "QBC_CAPPED"]):
                value = (100, [80, 100]) if arm == "QBC_FULL" else (alternate if phase == "dev_alt" else primary)
                for price, (units, ba) in [(128, value), (8192, (100000 if phase == "dev_alt" else 1, [0, 1]))]:
                    result.append(dict(job=dict(tier="real", dataset=str(i // 3), seed=i % 3, price=price, arm=arm),
                                       metrics=dict(units=units, ba=ba)))
        return result, "end"

    monkeypatch.setattr(bench, "read_rows", read)
    assert bench.choose_cap(args)[0] == expected


@pytest.mark.parametrize("evidence", ["raw", "orphan_claim"])
def test_existing_final_evidence_cannot_recreate_missing_global_claim(harness, evidence):
    h = harness
    h.cli("--phase", "dev")
    h.cli("--approve")
    if evidence == "raw":
        h.cli("--phase", "final")
        h.claim.unlink()
    else:
        job = h.plan("final")[0]
        claims = h.output / "final_claims"
        claims.mkdir()
        bench.atomic(claims / (bench.digest(job) + ".json"), dict(job=job, manifest=bench.digest(h.frozen)))
    launches = len(h.launches)
    with pytest.raises(ValueError):
        h.cli("--phase", "final")
    assert not h.claim.exists()
    assert len(h.launches) == launches


def test_no_approval_with_unsettled_alternative_claim(harness):
    h = harness
    h.cli("--phase", "dev")
    launches = len(h.launches)
    job = h.plan("dev_alt", 4096)[0]
    claims = h.output / "dev_alt_claims"
    claims.mkdir()
    bench.atomic(claims / (bench.digest(job) + ".json"), dict(job=job, manifest=bench.digest(h.frozen)))
    with pytest.raises(ValueError):
        h.cli("--approve")
    assert not (h.output / "FINAL_APPROVAL.json").exists()
    assert len(h.launches) == launches


def test_other_runner_holds_exclusive_output_without_poisoning_stop(harness):
    import fcntl

    h = harness
    with (h.output / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another runner"):
            h.cli()
    assert not h.launches and not (h.output / "STOP.json").exists()
    h.cli()
    assert len(h.launches) == 2


def test_immediate_source_drift_stops_new_worker(harness):
    h = harness
    (h.baseline / IMPL).write_text("changed since prepare")
    with pytest.raises(ValueError, match="source changed"):
        h.cli()
    assert not h.launches and (h.output / "STOP.json").exists()


def test_worker_rejects_foreign_import_before_reaching_trial(tmp_path, monkeypatch):
    import active_real_worker as worker

    job = tmp_path / "job.json"
    job.write_text(json.dumps(dict(tier="program", arm="OLD_ACTIVE")))
    journal = tmp_path / "journal.jsonl"
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", ["worker", "--source", str(tmp_path / "foreign"), "--job", str(job),
                                     "--data", str(tmp_path / "data"), "--journal", str(journal)])
    with pytest.raises(ValueError, match="foreign actor module origin"):
        worker.main()
    assert not journal.exists()


@pytest.mark.parametrize(("arm", "config"), [
    ("NEW_ACTIVE", {}), ("NEW_ACTIVE", {"active_policy": "onepass-bounded-v1", "active_work_limit": True}),
    ("OLD_ACTIVE", {"active_policy": "onepass-bounded-v1", "active_work_limit": 1024}),
])
def test_program_payload_must_carry_assigned_runtime_policy(arm, config, monkeypatch):
    job = dict(tier="program", task="finite", arm=arm, seed=1)
    monkeypatch.setattr(bench, "tasks", lambda: [SimpleNamespace(task_id="finite")])
    monkeypatch.setattr(bench, "audit_program", lambda *args: pytest.fail("incorrect method reached downstream audit"))
    with pytest.raises(ValueError, match="policy"):
        bench.audit(dict(job=job, payload=dict(learning=dict(config=config))), None)
