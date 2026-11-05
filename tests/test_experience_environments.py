import pytest

from solvi.experimental.experience._common import digest
from solvi.experimental.experience.environments import FiniteEnvironment


def make(**extra):
    return FiniteEnvironment(
        {"r": {"family": "resource", "x": 7, "y": 2, **extra}, "w": {"family": "workflow", "x": 2, "y": 5, **extra}},
        identity="host",
    )


@pytest.mark.parametrize("task,expected", [("r", 5), ("w", 5)])
def test_manual_id_anchors(task, expected):
    env = make()
    r = env.execute(task, "observe", {"probe": "measure", "field": "answer"}, "measure")
    assert r.value == {"field": "answer", "value": expected}
    assert r.source == "host"
    assert env.execute(task, "act", {"answer": expected}, "act").value["ok"] is True
    assert env.state[task]["effects"] == 1


@pytest.mark.parametrize("announced", [False, True])
def test_manual_ood_anchors(announced):
    env = make(mechanism="v2", announced_version=announced)
    for task, answer in [("r", 4), ("w", 7)]:
        r = env.execute(task, "act", {"answer": answer}, task)
        assert r.value["answer"] == answer
        assert r.value["ok"] is True
        assert r.mechanism == env.task(task).mechanism == ("v2" if announced else "v1")


def test_idempotency_and_detachment():
    env = make()
    receipt = env.execute("r", "act", {"answer": 5}, "once")
    assert env.execute("r", "act", {"answer": 5}, "once") == receipt
    assert env.reconcile("once") == receipt
    receipt.value["state"]["effects"] = 99
    assert env.state["r"] == {"available": 4, "reserved": 1, "effects": 1}
    assert env.reconcile("once").value["state"]["effects"] == 1
    assert env.reconcile("absent") is None
    with pytest.raises(ValueError):
        env.execute("w", "act", {"answer": 5}, "once")
    task = env.task("r")
    task.inputs["x"] = 99
    task.worlds[0]["answer"] = 99
    assert env.task("r").inputs == {"x": 7, "y": 2}
    assert 99 not in [w["answer"] for w in env.task("r").worlds]


def test_wrong_answer_refuses_without_effect():
    env = make()
    before = env.state
    for task in ("r", "w"):
        r = env.execute(task, "act", {"answer": 100}, task)
        assert r.value["ok"] is False and r.value["refused"] is True
        assert r.value["answer"] == 5
    assert env.state == before


def test_probe_validation_and_no_truth_marker():
    env = make()
    task = env.task("r")
    assert task.scope == "resource" and task.schema == "xy-int-v1"
    assert set(task.inputs) == {"x", "y"}
    assert all(set(w) == {"answer", "noise", "parity"} for w in task.worlds)
    assert make(mechanism="v2").task("r").worlds == task.worlds
    for payload in ({"probe": "noise", "field": "answer"}, {"probe": "bogus", "field": "noise"}):
        with pytest.raises(ValueError):
            env.execute("r", "observe", payload, "invalid")
    with pytest.raises(ValueError):
        env.execute("r", "act", {"answer": True}, "bool")
    assert env.reconcile("invalid") is None


def test_snapshot_reopen_does_not_execute(monkeypatch):
    env = make()
    r = env.execute("w", "act", {"answer": 5}, "done")
    monkeypatch.setattr(FiniteEnvironment, "execute", lambda *a, **kw: pytest.fail("restore executed"))
    restored = FiniteEnvironment.restore(env.snapshot(), identity="host")
    assert restored.state == env.state
    assert restored.reconcile("done") == r
    assert restored.snapshot() == env.snapshot()


def test_snapshot_rejects_corruption_duplicates_and_case_conflicts():
    env = make()
    env.execute("r", "act", {"answer": 5}, "done")
    snap = env.snapshot()
    snap["state"]["r"]["reserved"] = 77
    with pytest.raises(ValueError):
        FiniteEnvironment.restore(snap)
    snap = env.snapshot()
    snap["records"].append(snap["records"][0])
    snap["digest"] = digest({k: v for k, v in snap.items() if k != "digest"})
    with pytest.raises(ValueError):
        FiniteEnvironment.restore(snap)
    with pytest.raises(ValueError):
        FiniteEnvironment.restore(env.snapshot(), identity="other")
    changed = env.snapshot()["cases"]
    changed["r"]["mechanism"] = "v2"
    with pytest.raises(ValueError):
        FiniteEnvironment.restore(env.snapshot(), cases=changed)
