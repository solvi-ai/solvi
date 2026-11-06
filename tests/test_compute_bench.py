"""Persistence tests use tiny fake actors, never comparative experiment runs."""
import json

import pytest

from benchmarks import compute_bench as bench


@pytest.fixture
def rig(tmp_path, monkeypatch):
    root = tmp_path / "source"
    (root / "src/solvi").mkdir(parents=True)
    (root / "src/solvi/a.py").write_text("a = 1\n")
    (root / "benchmarks").mkdir()
    (root / "benchmarks/compute_fake.py").write_text("frozen = 1\n")
    protocol = tmp_path / "protocol"
    protocol.mkdir()
    for name in bench.PROTOCOL:
        (protocol / name).write_text(name)
    monkeypatch.setattr(bench, "SEEDS", {"smoke": (11,), "dev": (101,), "final": (21001,)})
    monkeypatch.setattr(bench, "FAMILIES", ("tiny",))
    monkeypatch.setattr(bench, "LIMITS", (1,))
    monkeypatch.setattr(bench, "ARMS", ("a", "b"))
    calls = []
    def actor(**key):
        calls.append(key)
        return {**key, "answer": 42}
    monkeypatch.setattr(bench, "trial", actor)
    monkeypatch.setattr(bench, "audit", lambda row: {"correct": int(row["answer"] == 42)})
    out = tmp_path / "run"
    def run(**kwargs):
        return bench.run("smoke", out, protocol, root=root, **kwargs)
    return root, protocol, out, calls, run


def test_resume_and_summary_never_replay(rig, monkeypatch):
    _, _, out, calls, run = rig
    assert run()["complete"] == 2
    assert len(calls) == 2
    monkeypatch.setattr(bench, "trial", lambda **key: pytest.fail("actor replay"))
    assert run()["complete"] == 2
    assert run(summarize=True)["complete"] == 2
    assert len(list((out / "claims").iterdir())) == 2


def test_unknown_claim_stops_without_actor(rig):
    _, _, out, calls, run = rig
    run()
    (out / "raw.jsonl").write_bytes(b"")
    with pytest.raises(bench.Stop, match="without durable"):
        run()
    assert len(calls) == 2
    assert (out / "STOP.json").exists()
    with pytest.raises(bench.Stop, match="persistent STOP"):
        run()


@pytest.mark.parametrize("damage", ["duplicate", "truncate", "hash", "claim"])
def test_corruption_stops(rig, damage):
    _, _, out, calls, run = rig
    run()
    raw = out / "raw.jsonl"
    lines = raw.read_bytes().splitlines(keepends=True)
    if damage == "duplicate":
        first = json.loads(lines[0])
        second = {"prev": first["hash"], "payload": first["payload"]}
        second["hash"] = bench.digest(second)
        raw.write_bytes(lines[0] + bench.encoded(second) + b"\n")
    elif damage == "truncate":
        raw.write_bytes(raw.read_bytes()[:-1])
    elif damage == "hash":
        raw.write_bytes(raw.read_bytes().replace(b'"answer":42', b'"answer":43', 1))
    else:
        next((out / "claims").iterdir()).write_text("{}")
    with pytest.raises((bench.Stop, KeyError)):
        run(summarize=True)
    assert len(calls) == 2
    assert (out / "STOP.json").exists()


def test_source_change_stops(rig):
    root, _, out, calls, run = rig
    run()
    (root / "src/solvi/a.py").write_text("a = 2\n")
    with pytest.raises(bench.Stop, match="mismatch"):
        run()
    assert len(calls) == 2
    assert (out / "STOP.json").exists()


def test_baseexception_is_durable_and_not_replayed(rig, monkeypatch):
    _, _, out, calls, run = rig
    def interrupted(**key):
        calls.append(key)
        raise KeyboardInterrupt("test interrupt")
    monkeypatch.setattr(bench, "trial", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run()
    assert json.loads((out / "raw.jsonl").read_text())["payload"]["error"]["type"] == "KeyboardInterrupt"
    with pytest.raises(bench.Stop):
        run()
    assert len(calls) == 1


def test_audit_mismatch_stops(rig, monkeypatch):
    _, _, out, calls, run = rig
    run()
    monkeypatch.setattr(bench, "audit", lambda row: {"correct": 0})
    with pytest.raises(bench.Stop, match="audit mismatch"):
        run(summarize=True)
    assert len(calls) == 2


def test_parent_lease_excludes_concurrency(rig):
    _, _, out, calls, run = rig
    out.mkdir()
    with bench.lease(out):
        with pytest.raises(bench.Stop, match="another parent"):
            run()
    assert not calls
    assert not (out / "STOP.json").exists()


def test_final_needs_approval(rig):
    root, protocol, out, calls, _ = rig
    with pytest.raises(bench.Stop, match="approval"):
        bench.run("final", out, protocol, root=root)
    assert not calls
    assert (out / "STOP.json").exists()
