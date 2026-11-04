import copy

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.experimental.process import Process, ProcessRunError, ProcessSpecError


def contract(version="quote/1"):
    return {"version": version, "schema": {"type": "object", "properties": {"price": {"type": "number"}},
            "required": ["price"], "additionalProperties": False}, "invariants": [{"answer": "price", "gt": 0}]}


def chain(tmp_path, value, incoming=None):
    calls = []
    producer, consumer = Catalog(), Catalog()

    @producer.rule("q")
    def quote():
        return value

    @consumer.rule("q")
    def accept(payload):
        calls.append(payload)
        return "ok"

    a = System(producer, [Question("q", "q", Answer.structured())], storage=tmp_path / "a.jsonl")
    b = System(consumer, [Question("q", "q", Answer.choice(["ok"]))], storage=tmp_path / "b.jsonl")
    p = Process("quotes", storage=tmp_path / "process.jsonl")
    declaration = contract()
    p.node("a", a, writes={"payload": "answer"}, output_contracts={"payload": declaration})
    p.node("b", b, reads=["payload"], end=True, input_contracts={"payload": incoming or declaration})
    p.edge("a", "b")
    return p, calls, declaration, {"a": a, "b": b}


def test_valid_boundaries_snapshot_spec_and_replay(tmp_path):
    p, calls, declaration, systems = chain(tmp_path, {"price": 2})
    spec = p.spec()
    declaration["version"] = "caller-mutated"
    declaration["schema"]["properties"]["price"]["type"] = "string"
    declaration["invariants"][0]["gt"] = 100
    assert p.spec() == spec
    loaded = Process.from_spec(spec, systems=systems, storage=tmp_path / "process.jsonl")
    assert loaded.spec()["hash"] == spec["hash"]
    result = loaded.run({})
    assert calls == [{"price": 2}]
    assert result.provenance["payload"]["interface_version"] == "quote/1"
    assert loaded.replay(result.run)["ok"]


@pytest.mark.parametrize("value,incoming", [({"price": -1}, None), ({"price": True}, None),
                                           ({"price": 2, "extra": 1}, None),
                                           ({"price": 2}, contract("quote/2"))])
def test_bad_payload_or_version_never_invokes_downstream(tmp_path, value, incoming):
    p, calls, _, _ = chain(tmp_path, value, incoming)
    with pytest.raises(ProcessRunError, match="boundary"):
        p.run({})
    assert calls == []


def test_unknown_schema_and_contract_fields_fail_before_node_install(tmp_path):
    p, _, _, systems = chain(tmp_path, {"price": 1})
    for malformed in ({**contract(), "typo": True}, {**contract(), "schema": {"type": "object", "$ref": "x"}}):
        q = Process("bad")
        with pytest.raises(ProcessSpecError):
            q.node("a", systems["a"], writes={"payload": "answer"}, output_contracts={"payload": malformed})
        assert not q.nodes
        spec = copy.deepcopy(p.spec())
        spec.pop("hash")
        spec["nodes"]["a"]["output_contracts"]["payload"] = malformed
        with pytest.raises(ProcessSpecError):
            Process.from_spec(spec, systems=systems)


def test_unversioned_initial_fact_is_not_a_contract_producer(tmp_path):
    p, calls, _, systems = chain(tmp_path, {"price": 1})
    q = Process("initial")
    q.node("b", systems["b"], reads=["payload"], end=True, input_contracts={"payload": contract()})
    with pytest.raises(ProcessRunError, match="version"):
        q.run({"payload": {"price": 1}})
    assert not calls


@pytest.mark.parametrize("value", [(1, 2), {1: "x"}, {False: "x"}, {None: "x"}, [float("inf")]])
def test_original_non_json_payload_is_rejected_without_coercion(value):
    from solvi.experimental.process import _boundary_payload
    schema = {"type": "object"} if isinstance(value, dict) else {"type": "array"}
    with pytest.raises(ProcessRunError, match="finite JSON"):
        _boundary_payload({"version": "1", "schema": schema}, value, "boundary")


@pytest.mark.parametrize("root", ["object", "array"])
def test_unconstrained_payload_children_still_obey_nesting_limit(root):
    from solvi.experimental.process import _boundary_payload
    value = "leaf"
    for _ in range(34):
        value = {"extra": value} if root == "object" else [value]
    with pytest.raises(ProcessRunError, match="nesting"):
        _boundary_payload({"version": "1", "schema": {"type": root}}, value, "boundary")


def test_from_spec_rejects_python_key_coercion(tmp_path):
    p, _, _, systems = chain(tmp_path, {"price": 2})
    spec = p.spec()
    spec.pop("hash")
    spec["nodes"]["a"]["output_contracts"]["payload"]["schema"]["properties"] = {1: {"type": "number"}}
    with pytest.raises(ProcessSpecError, match="finite JSON"):
        Process.from_spec(spec, systems=systems)


def test_changed_contract_is_bound_by_recorded_spec_hash(tmp_path):
    p, _, _, systems = chain(tmp_path, {"price": 2})
    result = p.run({})
    spec = p.spec()
    spec.pop("hash")
    spec["nodes"]["b"]["input_contracts"]["payload"]["version"] = "quote/2"
    changed = Process.from_spec(spec, systems=systems, storage=tmp_path / "process.jsonl")
    report = changed.replay(result.run)
    assert not report["ok"] and report["at"]["step"] == "spec"


@pytest.mark.parametrize("value", [{"price": 3}, {"price": -1}])
def test_human_pending_resume_validates_output_before_consumer(tmp_path, value):
    from solvi.experimental.process import MemoryQueue
    _, calls, _, systems = chain(tmp_path, {"price": 1})
    queue = MemoryQueue()
    p = Process("review", storage=tmp_path / "human.jsonl", human=queue)
    p.node("review", kind="human", question="quote", writes={"payload": "answer"},
           output_contracts={"payload": contract()})
    p.node("consumer", systems["b"], reads=["payload"], end=True, input_contracts={"payload": contract()})
    p.edge("review", "consumer")
    pending = p.run({})
    assert pending.status == "pending" and not calls
    queue.answer(pending.ticket, value, reviewer="reviewer")
    if value["price"] < 0:
        with pytest.raises(ProcessRunError, match="invariant"):
            p.resume(pending.run)
        assert not calls
    else:
        result = p.resume(pending.run)
        assert result.status == "decided" and calls == [value]
        assert p.replay(result.run)["ok"]


def test_calibration_walk_uses_output_boundary_before_downstream(tmp_path):
    p, calls, _, _ = chain(tmp_path, {"price": -1})
    with pytest.raises(ProcessRunError, match="invariant"):
        p._walk(p.spec(), {}, "fixture-calibration")
    assert not calls


@pytest.mark.parametrize("kind,value,allowed", [("integer", True, [True]), ("integer", 1, [True]),
                                               ("number", 1.0, [1]), ("integer", 1, [1])])
def test_enum_uses_exact_python_json_type_equality(kind, value, allowed):
    from solvi.experimental.process import _boundary_payload
    c = {"version": "1", "schema": {"type": kind, "enum": allowed}}
    if type(value) is int and allowed == [1] and type(allowed[0]) is int:
        _boundary_payload(c, value, "boundary")
    else:
        with pytest.raises(ProcessRunError):
            _boundary_payload(c, value, "boundary")


def test_missing_declared_version_rejected_in_spec(tmp_path):
    p, _, _, systems = chain(tmp_path, {"price": 2})
    spec = p.spec()
    spec.pop("hash")
    del spec["nodes"]["b"]["input_contracts"]["payload"]["version"]
    with pytest.raises(ProcessSpecError, match="version"):
        Process.from_spec(spec, systems=systems)


def test_enum_recursive_typed_values_and_object_order():
    from solvi.experimental.process import _boundary_payload
    c = {"version": "1", "schema": {"type": "object", "enum": [{"a": 1, "b": 2}]}}
    _boundary_payload(c, {"b": 2, "a": 1}, "boundary")
    for value in ({"a": True, "b": 2}, {"a": 1.0, "b": 2}):
        with pytest.raises(ProcessRunError, match="enum"):
            _boundary_payload(c, value, "boundary")


def test_large_json_integer_is_not_limited_to_float_range():
    from solvi.experimental._process_boundary import schema_check
    from solvi.experimental.process import _boundary_payload
    large = 10 ** 400
    schema = {"type": "number", "minimum": large - 1, "maximum": large + 1}
    schema_check(schema)
    _boundary_payload({"version": "1", "schema": schema}, large, "boundary")


@pytest.mark.parametrize("bound", [True, "large", float("inf"), float("nan")])
def test_bad_numeric_bound_has_spec_error_before_node_install(tmp_path, bound):
    _, _, _, systems = chain(tmp_path, {"price": 1})
    p = Process("invalid")
    c = {"version": "1", "schema": {"type": "number", "minimum": bound}}
    with pytest.raises(ProcessSpecError):
        p.node("a", systems["a"], writes={"payload": "answer"}, output_contracts={"payload": c})
    assert not p.nodes


def test_invalid_human_value_cannot_be_laundered_by_repeated_or_fresh_resume(tmp_path):
    from solvi.experimental.process import MemoryQueue
    _, calls, _, systems = chain(tmp_path, {"price": 1})
    queue = MemoryQueue()
    path = tmp_path / "human-array.jsonl"
    c = {"version": "array/1", "schema": {"type": "array"}}
    p = Process("review", storage=path, human=queue)
    p.node("review", kind="human", question="quote", writes={"payload": "answer"},
           output_contracts={"payload": c})
    p.node("consumer", systems["b"], reads=["payload"], end=True, input_contracts={"payload": c})
    p.edge("review", "consumer")
    pending = p.run({})
    queue.answer(pending.ticket, (1, 2), reviewer="reviewer")
    for process in (p, p, Process.from_spec(p.spec(), systems={"consumer": systems["b"]}, human=queue, storage=path)):
        with pytest.raises(ProcessRunError, match="finite JSON"):
            process.resume(pending.run)
        assert not calls
        assert list(process._open().iter("human")) == []
    assert p.replay(pending.run)["ok"]
