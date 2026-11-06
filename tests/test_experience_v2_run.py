import json
from pathlib import Path
import sys
import pytest

BENCH = Path(__file__).parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))
import experience_v2_run as run  # noqa: E402


def test_approval_pinned_to_exact_manifest_and_phase(tmp_path):
    manifest = {"source": "frozen"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    run.verify_approval(path, manifest, "smoke")
    with pytest.raises(FileNotFoundError):
        run.verify_approval(path, manifest, "development")
    approval = {
        "manifest_hash": run.digest(manifest),
        "verdict": "PASS",
        "phases": ["development"],
        "reviewer": "independent",
    }
    (tmp_path / "HARNESS_REVIEW_APPROVAL.json").write_text(json.dumps(approval))
    run.verify_approval(path, manifest, "development")
    with pytest.raises(ValueError):
        run.verify_approval(path, manifest, "final")
    with pytest.raises(ValueError):
        run.verify_approval(path, {"source": "changed"}, "development")


def test_smoke_plan_and_resume_never_repeat_ambiguous_trial(tmp_path, monkeypatch):
    specs = list(run.trial_plan("smoke"))
    assert len(specs) == 54
    assert {s["seed"] for s in specs} == {0}
    assert len([s for s in specs if s["kind"] == "operational"]) == 42
    out = tmp_path / "run"
    out.mkdir()
    spec = specs[0]
    (out / run.trial_id(spec)).mkdir()
    calls = []
    monkeypatch.setattr(run, "run_isolated", lambda *a, **kw: calls.append(1))
    run.run_plan(out, "smoke", tmp_path / "manifest.json", {}, tmp_path, resume=True, plan=[spec])
    assert not calls
    record = json.loads((out / "trials.jsonl").read_text())
    assert record["status"] == "censored"
    run.run_plan(out, "smoke", tmp_path / "manifest.json", {}, tmp_path, resume=True, plan=[spec])
    assert len((out / "trials.jsonl").read_text().splitlines()) == 1
    with pytest.raises(ValueError):
        run.run_plan(out, "smoke", tmp_path / "manifest.json", {}, tmp_path, plan=[])


def test_new_trial_claim_and_terminal_are_incremental(tmp_path, monkeypatch):
    spec = next(run.trial_plan("smoke"))
    calls = []

    def fake(*args, **kwargs):
        calls.append(1)
        assert (kwargs["directory"] / "running.json").exists()
        return {"status": "completed", "spent": 3, "wall_seconds": 0.125}

    monkeypatch.setattr(run, "run_isolated", fake)
    out = tmp_path / "run"
    run.run_plan(out, "smoke", tmp_path / "manifest.json", {}, tmp_path, plan=[spec])
    assert calls == [1]
    run.run_plan(out, "smoke", tmp_path / "manifest.json", {}, tmp_path, resume=True, plan=[spec])
    assert calls == [1]


def test_final_single_claim_rejects_new_output_and_permits_exact_resume(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = {"hash": "frozen"}
    run.claim_final(path, manifest, tmp_path / "final")
    with pytest.raises(FileExistsError):
        run.claim_final(path, manifest, tmp_path / "other")
    run.claim_final(path, manifest, tmp_path / "final", resume=True)
    with pytest.raises(ValueError):
        run.claim_final(path, manifest, tmp_path / "other", resume=True)


def test_worker_rejects_foreign_claim_and_malformed_status(tmp_path, monkeypatch):
    spec = next(run.trial_plan("smoke"))
    directory = tmp_path / run.trial_id(spec)
    directory.mkdir()
    with pytest.raises(FileNotFoundError):
        run.verify_worker_claim(directory, spec, "smoke", {})
    (directory / "running.json").write_text(
        json.dumps(dict(spec=spec, config=dict(phase="smoke", manifest=run.digest({}))))
    )
    (tmp_path / "run.json").write_text(json.dumps(dict(out=str(tmp_path), phase="smoke", manifest=run.digest({}))))
    run.verify_worker_claim(directory, spec, "smoke", {})
    changed = {**spec, "profile": "hidden"}
    with pytest.raises(ValueError):
        run.verify_worker_claim(directory, changed, "smoke", {})
    out = tmp_path / "malformed"
    monkeypatch.setattr(run, "run_isolated", lambda *args, **kwargs: {"status": "invented"})
    run.run_plan(out, "smoke", tmp_path / "manifest.json", {}, tmp_path, plan=[spec])
    record = json.loads((out / "trials.jsonl").read_text())
    assert record["status"] == "censored"
