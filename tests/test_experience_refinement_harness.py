"""Gate and transaction tests on one engineering-only task in temporary directories."""

import json
from pathlib import Path
import shutil
import sys

import pytest
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience._common import digest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import refinement_bench as bench  # noqa: E402
import refinement_score as scorer  # noqa: E402
from refinement_tasks import Problem  # noqa: E402


@pytest.fixture
def gate(tmp_path, monkeypatch):
    prob = Problem(
        "fixture",
        "numeric_tabular",
        Grammar(("x",)),
        tuple({"x": i} for i in range(32)),
        lambda p: p["x"],
        Program(("input", "x")),
        "One synthetic identity, no learner.",
    )
    for module in (bench, scorer):
        monkeypatch.setattr(module, "tasks", lambda: (prob,))
        monkeypatch.setattr(module, "ARMS", ("FORMULA",))
        monkeypatch.setattr(module, "SEEDS", {"smoke": [0], "dev": [1], "final": [2]})
    fake_root = tmp_path / "work"
    (fake_root / "research_growth/solvi_refinement_20261106").mkdir(parents=True)
    monkeypatch.setattr(bench, "__file__", str(fake_root / "lanes/fixture/benchmarks/refinement_bench.py"))
    monkeypatch.setattr(bench, "source_manifest", lambda: {"synthetic": "engineering"})
    folder = tmp_path / "output"

    def run(*args, output=folder):
        monkeypatch.setattr(sys, "argv", ["bench", "--output", str(output), *args])
        bench.main()

    run("--prepare")
    return folder, run


def test_completed_gate_noop_resume_and_standalone_scorer(gate, monkeypatch):
    folder, run = gate
    run()
    raw = folder / "smoke.jsonl"
    before = raw.read_bytes()
    monkeypatch.setattr(bench, "trial", lambda *_: pytest.fail("retried actor"))
    run()
    run("--summarize")
    assert raw.read_bytes() == before
    assert json.loads((folder / "smoke_summary.json").read_text())["arms"]["FORMULA"]["correct"] == 14


@pytest.mark.parametrize("damage", ["seal", "claim", "orphan", "stop"])
def test_damaged_or_incomplete_run_blocks_actors(gate, monkeypatch, damage):
    folder, run = gate
    run()
    raw = folder / "smoke.jsonl"
    claim = next((folder / "smoke_claims").glob("*.json"))
    if damage == "seal":
        row = json.loads(raw.read_text())
        row["seal"]["previous"] = "tampered"
        raw.write_text(json.dumps(row) + "\n")
    elif damage == "claim":
        claim.write_text(json.dumps(dict(task=["fixture", 0, "FORMULA"], manifest="wrong")))
    elif damage == "orphan":
        raw.unlink()
    else:
        (folder / "smoke_STOP.json").write_text("{}")
    monkeypatch.setattr(bench, "trial", lambda *_: pytest.fail("retried actor"))
    with pytest.raises(ValueError):
        run()
    if damage == "orphan":
        assert (folder / "smoke_STOP.json").is_file()
    elif damage in ("seal", "claim", "stop"):
        with pytest.raises(ValueError):
            run("--summarize")


def test_final_claim_forbids_copying_approval_to_second_output(gate, tmp_path, monkeypatch):
    folder, run = gate
    # Gate unit only: its comparator input is fixed and comes from no actual dev trials.
    real = bench.summarize
    dev = dict(arms={"FIXED": {"correct": 7}, "RANDOM": {"correct": 7}}, raw_sha256="abc")
    monkeypatch.setattr(bench, "summarize", lambda path, *, split: dev if split == "dev" else real(path, split=split))
    manifest = digest(json.loads((folder / "manifest.json").read_text()))
    approval = dict(approved=True, manifest=manifest, comparator="FIXED", dev_raw_sha256="abc", dev_sha256="abc")
    (folder / "FINAL_APPROVAL.json").write_text(json.dumps(approval))
    second = tmp_path / "other"
    shutil.copytree(folder, second)
    run("--split", "final")
    monkeypatch.setattr(bench, "trial", lambda *_: pytest.fail("reran final actor"))
    run("--split", "final")
    with pytest.raises(ValueError, match="claimed"):
        run("--split", "final", output=second)


def test_pertrial_audit_fails_before_append_and_blocks_retry(gate, monkeypatch):
    folder, run = gate
    original = bench.trial

    def corrupted(*args):
        row = original(*args)
        row["predictions"][0]["value"] = False
        return row

    monkeypatch.setattr(bench, "trial", corrupted)
    with pytest.raises(ValueError, match="point/answer/outcome"):
        run()
    assert (folder / "smoke_STOP.json").exists() and not (folder / "smoke.jsonl").exists()
    monkeypatch.setattr(bench, "trial", lambda *_: pytest.fail("retried failed actor"))
    with pytest.raises(ValueError, match="persistent STOP"):
        run()


def test_resume_rejects_unplanned_but_well_sealed_trial_before_next_actor(gate, monkeypatch):
    folder, run = gate
    run()
    problem = bench.tasks()[0]
    foreign = bench.trial(problem, 987, "FORMULA")
    manifest = digest(json.loads((folder / "manifest.json").read_text()))
    foreign["seal"] = dict(manifest=manifest, previous="0" * 64, sha256=digest(foreign))
    (folder / "smoke.jsonl").write_text(json.dumps(foreign) + "\n")
    claim = next((folder / "smoke_claims").glob("*.json"))
    claim.unlink()
    key = [problem.task_id, 987, "FORMULA"]
    (claim.parent / (digest(key) + ".json")).write_text(json.dumps(dict(task=key, manifest=manifest)))
    monkeypatch.setattr(bench, "trial", lambda *_: pytest.fail("executed actor after foreign trial"))
    with pytest.raises(ValueError, match="unplanned"):
        run()


def test_real_lane_manifest_pins_protocol_and_modules():
    library = Path(bench.__file__).resolve().parents[1]
    project = library.parent.parent if library.parent.name == "lanes" else library.parent
    if not (project / "research_growth/solvi_refinement_20261106/PREREGISTRATION.md").exists():
        pytest.skip("source-manifest integration requires the host research protocol")
    hashes = bench.source_manifest()
    for source in (
        "host/PREREGISTRATION.md",
        "pyproject.toml",
        "uv.lock",
        "benchmarks/refinement_score.py",
        "src/solvi/experimental/experience/refinement.py",
    ):
        assert len(hashes[source]) == 64
