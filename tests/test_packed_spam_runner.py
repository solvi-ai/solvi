"""Supervisor checks use synthetic actors only."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import packed_spam_bench as runner  # noqa: E402


def test_registration():
    for phase, count in [("smoke", 17), ("dev", 168), ("final", 480)]:
        assigned = runner.jobs(phase, "ADAPT40")
        assert count == len(assigned) == len(set(map(runner.digest, assigned)))
        if phase == "final":
            assert "CUT40" not in {r["arm"] for r in assigned}


def test_exact_selection():
    rows = [dict(job=j, metrics=dict(units=10, ba=[52 if j["arm"] in runner.METHODS else 50, 100]))
            for j in runner.jobs("dev")]
    def result(tier, selected):
        selection = runner.select(rows)
        assert (selection["tier"], selection["selected"]) == (tier, selected)
    result("joint", "CUT40")
    for row in rows:
        if row["job"]["arm"] == "CUT40" and row["job"]["price"] == 8192:
            row["metrics"]["units"] = 11
    result("joint", "ADAPT40")
    for row in rows:
        row["metrics"]["ba"] = [1, 2]
    result("cost", "ADAPT40")
    for row in rows:
        if row["job"]["arm"] == "ADAPT40":
            row["metrics"]["units"] = 12
    result("fallback", "CUT40")
    for row in rows:
        if row["job"]["arm"] == "CUT40":
            row["metrics"]["ba"] = None
    result("fallback", "ADAPT40")
    for row in rows:
        if row["job"]["arm"] == "RANDOM48":
            row["metrics"]["ba"] = None
    assert runner.select(rows)["undefined_ba"] == list(runner.METHODS)


def test_cost_quality_order():
    rows = [dict(job=j, metrics=dict(units=100, ba=[51 if j["arm"] == "ADAPT40" else 50, 100]))
            for j in runner.jobs("dev")]
    assert runner.select(rows)["selected"] == "ADAPT40"
    for row in rows:
        row["metrics"]["ba"] = [1, 2]
        if row["job"]["arm"] == "ADAPT40":
            row["metrics"]["units"] = 90
    assert runner.select(rows)["selected"] == "ADAPT40"
    for row in rows:
        row["metrics"]["units"] = 100
    assert runner.select(rows)["selected"] == "CUT40"


@pytest.mark.parametrize("field", ["receipts", "model", "predictions"])
def test_median_legacy_pair(field):
    job = next(j for j in runner.jobs("smoke") if j["arm"] == "LEGACY40")
    record = dict(receipts=[dict(id="a"), dict(id="b")], frozen=dict(model=[1]), predictions=[True, False])
    row = dict(job=job, payload=dict(record=record))
    pairs = {}
    runner.check_pair(pairs, row)
    job["arm"] = "MEDIAN40"
    runner.check_pair(pairs, row)
    if field == "model":
        record["frozen"]["model"] = [2]
    else:
        record[field].reverse()
    with pytest.raises(ValueError, match="paired"):
        runner.check_pair(pairs, row)


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
