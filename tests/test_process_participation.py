import copy

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.experimental.process import Process, ProcessRunError, ProcessSpecError


def system(tmp_path, name, calls):
    catalog = Catalog()

    @catalog.rule("q")
    def answer():
        calls.append(name)
        return "yes"

    return System(catalog, [Question("q", "q", Answer.choice(["yes"]))],
                  storage=tmp_path / f"{name}.jsonl")


def test_declaration_execution_snapshot_hash_and_replay(tmp_path):
    calls = []
    a = system(tmp_path, "a", calls).participate(writes={"value": "answer"})
    predicate = {"answer": "value", "eq": "yes"}
    b = system(tmp_path, "b", calls).participate(after="a", when=predicate, end=True)
    p = Process("declared", storage=tmp_path / "process.jsonl").participants({"a": a, "b": b})
    spec = p.spec()
    predicate["eq"] = "changed"
    exported = p.spec()
    exported["nodes"]["b"]["participation"]["when"]["eq"] = "changed"
    assert p.spec() == spec
    loaded = Process.from_spec(spec, systems={"a": a, "b": b}, storage=tmp_path / "process.jsonl")
    result = loaded.run({})
    assert result.status == "decided"
    assert [step["node"] for step in result.path] == ["a", "b"]
    assert calls == ["a", "b"]
    assert loaded.replay(result.run)["ok"]
    calls_after_replay = list(calls)  # Trace replay can recompute catalog parts.
    b.participate(after="a", when={"status": "abstain"}, end=True)
    assert p.spec() == spec
    with pytest.raises(ProcessSpecError, match="participation"):
        Process.from_spec(spec, systems={"a": a, "b": b})
    with pytest.raises(ProcessSpecError, match="participation"):
        Process.from_spec(spec, systems={"a": a, "b": b}, strict=False)
    with pytest.raises(ProcessRunError, match="participation"):
        p.run({})
    assert calls == calls_after_replay
    changed = Process("declared").participants({"a": a, "b": b}).spec()
    assert changed["hash"] != spec["hash"]
    rerouted = Process("rerouted", on_abstain="abstain").participants({"a": a, "b": b})
    stopped = rerouted.run({})
    assert stopped.status == "abstain"
    assert [step["node"] for step in stopped.path] == ["a"]


@pytest.mark.parametrize("kwargs", [{"when": {"status": "ok"}}, {"max_iter": 1},
    {"after": "a", "max_iter": True}, {"after": "a", "max_iter": 1.0},
    {"after": "a", "max_iter": 0}, {"after": "a", "when": {"unknown": 1}}])
def test_bad_declarations_fail(tmp_path, kwargs):
    with pytest.raises((ProcessSpecError, ValueError)):
        system(tmp_path, "a", []).participate(**kwargs)


def test_registration_atomic_and_graph_failclosed(tmp_path):
    a = system(tmp_path, "a", []).participate()
    b = system(tmp_path, "b", []).participate(after="missing", end=True)
    p = Process("bad")
    with pytest.raises(ProcessSpecError):
        p.participants({"a": a, "b": b})
    assert not p.nodes and not p.edges and not p.systems
    a.participate(after="b")
    b.participate(after="a", end=True)
    with pytest.raises(ProcessSpecError):
        p.participants({"a": a, "b": b})
    assert not p.nodes


def test_late_invalid_node_restores_existing_process(tmp_path):
    a = system(tmp_path, "a", []).participate(end=True)
    p = Process("existing").participants({"a": a})
    before = p.spec()
    b = system(tmp_path, "b", []).participate(after="a", end=True)
    with pytest.raises(ProcessSpecError):
        p.participants({"b": b, "invalid name": b})
    assert p.spec() == before
    assert set(p.systems) == {"a"}


def test_bounded_loop_executes_and_replays(tmp_path):
    calls = []
    a = system(tmp_path, "a", calls).participate(after="a", max_iter=2)
    b = system(tmp_path, "b", calls).participate(after="a", end=True)
    p = Process("loop", start="a", on_abstain="abstain", storage=tmp_path / "process.jsonl")
    p.participants({"a": a, "b": b})
    result = p.run({})
    assert calls == ["a"] * 3 + ["b"]  # entry plus two loop-edge traversals
    assert p.replay(result.run)["ok"]
    spec = copy.deepcopy(p.spec())
    spec.pop("hash")
    spec["edges"][0]["max_iter"] = 20
    with pytest.raises(ProcessSpecError, match="participation"):
        Process.from_spec(spec)
