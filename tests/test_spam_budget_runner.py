"""Supervisor checks use synthetic actors only."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import spam_budget_bench as runner  # noqa: E402


def test_registration():
    for phase, count in [("smoke", 12), ("dev", 216), ("final", 300)]:
        assigned = runner.jobs(phase, "MIXED32")
        assert count == len(assigned) == len(set(map(runner.digest, assigned)))
        if phase == "final":
            assert "MIXED40" not in {r["arm"] for r in assigned}


def test_exact_selection():
    rows = [dict(job=j, metrics=dict(units=10, ba=[52 if j["arm"].startswith("MIXED") else 50, 100]))
            for j in runner.jobs("dev")]
    assert runner.select(rows) == dict(undefined_ba=[], eligible=["MIXED40", "MIXED32"], selected="MIXED40")
    for r in rows:
        if r["job"]["arm"] == "MIXED40" and r["job"]["price"] == 8192:
            r["metrics"]["units"] += 1
    assert runner.select(rows)["selected"] == "MIXED32"
    for r in rows:
        r["metrics"]["ba"] = [1, 2]
    assert runner.select(rows) == dict(undefined_ba=[], eligible=[], selected="MIXED40")
    for r in rows:
        if r["job"]["arm"] == "RANDOM48" and r["job"]["dataset"] == "spambase":
            r["metrics"]["ba"] = None
    selection = runner.select(rows)
    assert len(selection["undefined_ba"]) == 4 and not selection["eligible"]


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    a = SimpleNamespace(output=tmp_path, claim=tmp_path / "custody.json", phase="smoke",
                        prepare=False, approve=False, summarize=False)
    frozen = dict(pins={"source": "fixed"})
    runner.write(a.output / "SOURCE_RUN_MANIFEST.json", frozen)
    monkeypatch.setattr(runner, "pins", lambda _: frozen["pins"])
    assigned = runner.jobs("smoke")[:1]
    monkeypatch.setattr(runner, "jobs", lambda *args: assigned)
    monkeypatch.setattr(runner, "audit", lambda *_: dict(assigned=512, units=33, ba=[1, 2]))
    return a, assigned[0]


def test_claim_resume_seal(fixture, monkeypatch):
    a, job = fixture
    launches = []
    def actor(_, j, key, work):
        launches.append(j)
        return dict(job=j, payload={})
    monkeypatch.setattr(runner, "run_actor", actor)
    runner.run_locked(a)
    raw = a.output / "smoke.jsonl"
    original = raw.read_bytes()
    runner.run_locked(a)
    a.summarize = True
    runner.run_locked(a)
    assert launches == [job] and raw.read_bytes() == original
    raw.write_bytes(original.replace(b'"units": 33', b'"units": 34'))
    with pytest.raises(ValueError, match="unsealed"):
        runner.run_locked(a)
    assert not (a.output / "STOP.json").exists()


def test_unknown_claim_stop(fixture, monkeypatch):
    a, job = fixture
    def crash(*_):
        raise RuntimeError("worker crashed")
    monkeypatch.setattr(runner, "run_actor", crash)
    with pytest.raises(RuntimeError, match="crashed"):
        runner.run_locked(a)
    assert (a.output / "smoke_claims" / (runner.digest(job) + ".json")).exists()
    with pytest.raises(ValueError, match="persistent STOP"):
        runner.run_locked(a)
    (a.output / "STOP.json").unlink()  # simulate manual deletion, claim still blocks
    with pytest.raises(ValueError, match="unresolved claim"):
        runner.run_locked(a)


def test_missing_global_and_lease(fixture):
    a, _ = fixture
    with runner.lease(a.output), pytest.raises(BlockingIOError), runner.lease(a.output):
        pass
    a.summarize = True
    with pytest.raises(ValueError, match="missing global"):
        runner.global_claim(a, {})
    a.summarize = False
    (a.output / "final_claims").mkdir()
    with pytest.raises(ValueError, match="missing global"):
        runner.global_claim(a, {})


def test_source_drift(fixture, monkeypatch):
    a, _ = fixture
    monkeypatch.setattr(runner, "pins", lambda _: dict(source="changed"))
    with pytest.raises(ValueError, match="changed"):
        runner.run_locked(a)
    assert not (a.output / "smoke_claims").exists()
