import json
import pytest
from solvi.core.store import JSONLStorage, SQLiteStorage
from solvi.core.store.calls import trace, replay_call, diff_calls, cases_from_calls


@pytest.mark.parametrize("backend", [JSONLStorage, SQLiteStorage])
def test_inputs_snapshotted_before_mutation_output_and_chain(tmp_path, backend):
    store = backend(str(tmp_path / "calls"))

    @trace(storage=store)
    def mutate(values):
        values.append(2)
        return {"values": values}

    arg = [1]
    result = mutate(arg)
    assert result == {"values": [1, 2]} and arg == [1, 2]
    result["values"].append(3)
    rows = list(store.iter(kind="call"))
    assert len(rows) == 1 and list(store.iter()) == []
    row = rows[0].data
    assert row["args"] == [[1]] and row["output"] == {"values": [1, 2]}
    assert store.verify()["ok"] and replay_call(row, mutate)["ok"]
    assert len(list(store.iter(kind="call"))) == 1
    json.dumps(row, allow_nan=False)
    assert cases_from_calls(store)[0]["format"] == "solvi-native-call-v1"


def test_nested_siblings_and_capture_suppressed_replay(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store)
    def child(x):
        return x + 1

    @trace(storage=store)
    def parent(x):
        return child(x) + child(x + 1)

    assert parent(2) == 7
    rows = [x.data for x in store.iter(kind="call")]
    a, b, p = rows
    assert a["parent_call_id"] == b["parent_call_id"] == p["call_id"] and p["parent_call_id"] is None
    assert replay_call(p, parent)["ok"] and len(list(store.iter(kind="call"))) == 3
    child(8)
    assert list(store.iter(kind="call"))[-1].data["parent_call_id"] is None


def test_exceptions_reraise_original_and_cleanup(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))
    error = ValueError("bad")

    @trace(storage=store)
    def fail(x):
        raise error

    with pytest.raises(ValueError) as raised:
        fail(1)
    assert raised.value is error
    row = list(store.iter(kind="call"))[0].data
    assert row["status"] == "error" and row["error"]["message"] == "bad"
    assert replay_call(row, fail)["ok"]

    @trace(storage=store)
    def ok(x):
        return None

    ok(1)
    assert list(store.iter(kind="call"))[-1].data["parent_call_id"] is None


@pytest.mark.parametrize("bad", [(1,), float("nan"), {1: "bad"}, object()])
def test_non_json_inputs_rejected_before_invoke(tmp_path, bad):
    calls = []

    @trace(storage=JSONLStorage(str(tmp_path / "calls")))
    def fn(x):
        calls.append(x)
        return x

    with pytest.raises(ValueError):
        fn(bad)
    assert calls == []


def test_generator_async_rejected_and_storage_failure_explicit(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))

    def gen():
        yield 1

    async def asyncfn():
        return 1

    with pytest.raises(TypeError):
        trace(gen, storage=store)
    with pytest.raises(TypeError):
        trace(asyncfn, storage=store)

    @trace(storage=store)
    def fn(x):
        return x

    store._append = lambda row: (_ for _ in ()).throw(OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        fn(1)


def test_explicit_diff_sees_value_change_and_never_imports_record_name(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store, name="not-an-import:local")
    def fn(x):
        return x + 1

    fn(1)
    row = list(store.iter(kind="call"))[0].data

    def changed(x):
        return x + 2

    assert not replay_call(row, changed)["ok"]
    changes = diff_calls(store, changed)
    assert len(changes) == 1 and changes[0]["facts"]["return"]["new"]["output"] == 3


def test_replay_pins_current_code_even_if_wrapper_fingerprint_was_created_earlier(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store)
    def fn(x):
        return x + 1

    fn(1)
    row = list(store.iter(kind="call"))[0].data

    def equivalent(x):
        return 1 + x

    fn.__solvi_trace_target__.__code__ = equivalent.__code__
    assert replay_call(row, fn)["reason"] == "function fingerprint changed"


def test_mixed_function_diff_requires_name_before_invocation(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store)
    def first(x):
        return x + 1

    @trace(storage=store)
    def second(x):
        return x + 2

    first(1)
    second(1)
    called = []

    def replacement(x):
        called.append(x)
        return x + 3

    with pytest.raises(ValueError, match="mixed-function"):
        diff_calls(store, replacement)
    assert called == []
    rows = diff_calls(store, replacement, name=first.__solvi_trace_name__)
    assert len(rows) == 1 and called == [1]


@pytest.mark.parametrize("original,changed", [(True, 1), (False, 0)])
def test_replay_and_diff_do_not_conflate_json_booleans_numbers(tmp_path, original, changed):
    from copy import deepcopy
    from solvi.core.store import record_hash

    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store)
    def fn():
        return original

    fn()
    row = deepcopy(list(store.iter(kind="call"))[0].data)
    row["output"] = changed
    row["hash"] = record_hash(row)
    assert not replay_call(row, fn)["ok"]
    assert diff_calls(store, lambda: changed)


def test_integrity_redaction_and_metadata_are_checked_before_replay(tmp_path):
    from copy import deepcopy
    from solvi.core.store import record_hash

    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store)
    def fn(x):
        return x

    fn(1)
    original = list(store.iter(kind="call"))[0].data
    for key, value in [("output", 99), ("redacted", True)]:
        row = deepcopy(original)
        row[key] = value
        with pytest.raises(ValueError):
            replay_call(row, fn)
    for key, value in [("function", {}), ("call_id", "invalid"), ("parent_call_id", original["call_id"])]:
        row = deepcopy(original)
        row[key] = value
        row["hash"] = record_hash(row)
        with pytest.raises(ValueError):
            replay_call(row, fn)


def test_json_global_and_closure_rebinding_changes_current_fingerprint(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))
    setting = {"delta": 1}

    @trace(storage=store)
    def fn(x):
        return x + setting["delta"]

    fn(1)
    row = list(store.iter(kind="call"))[0].data
    setting["delta"] = 2
    assert replay_call(row, fn)["reason"] == "function fingerprint changed"


def test_parent_uuid_crosses_store_boundary_without_local_parent_lookup(tmp_path):
    outer = JSONLStorage(str(tmp_path / "outer.jsonl"))
    inner = SQLiteStorage(str(tmp_path / "inner.db"))

    @trace(storage=inner)
    def child(x):
        return x + 1

    @trace(storage=outer)
    def parent(x):
        return child(x)

    parent(1)
    prow = list(outer.iter(kind="call"))[0].data
    crows = [r.data for r in inner.iter(kind="call")]
    assert crows[0]["parent_call_id"] == prow["call_id"]
    assert prow["call_id"] not in {r["call_id"] for r in crows}
    assert outer.verify()["ok"] and inner.verify()["ok"]


def test_native_fixture_json_roundtrip_replays_explicit_callable(tmp_path):
    store = JSONLStorage(str(tmp_path / "calls"))

    @trace(storage=store)
    def fn(x, *, flag=True):
        return [x, flag, None]

    fn(3)
    case = json.loads(json.dumps(cases_from_calls(store)))[0]
    assert replay_call(case, fn)["ok"]
    assert len(list(store.iter(kind="call"))) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "not-uuid"),
        ("function", {}),
        ("args", {}),
        ("expected", None),
        ("expected", {"status": "ok"}),
        ("expected", {"status": "unknown", "error": {"type": "ValueError", "message": "x"}}),
    ],
)
def test_malformed_native_fixtures_refused_before_callable(tmp_path, field, value):
    store = JSONLStorage(str(tmp_path / "calls"))
    called = []

    @trace(storage=store)
    def fn(x):
        called.append(x)
        return x

    fn(1)
    case = cases_from_calls(store)[0]
    before = list(called)
    case[field] = value
    with pytest.raises(ValueError):
        replay_call(case, fn)
    assert called == before


@pytest.mark.parametrize("keyword,mutable", [(False, False), (True, False), (False, True), (True, True)])
def test_current_defaults_change_refused_before_side_effect(tmp_path, keyword, mutable):
    store = JSONLStorage(str(tmp_path / "calls"))
    effect = tmp_path / "effect"
    default = [1] if mutable else 1
    if keyword:

        def fn(*, amount=default):
            effect.write_text(effect.read_text() + "x" if effect.exists() else "x")
            return amount
    else:

        def fn(amount=default):
            effect.write_text(effect.read_text() + "x" if effect.exists() else "x")
            return amount

    wrapped = trace(fn, storage=store)
    wrapped()
    row = list(store.iter(kind="call"))[0].data
    if mutable:
        default.append(2)
    elif keyword:
        fn.__kwdefaults__["amount"] = 2
    else:
        fn.__defaults__ = (2,)
    assert replay_call(row, wrapped) == {"ok": False, "reason": "function fingerprint changed"}
    assert effect.read_text() == "x"


@pytest.mark.parametrize("keyword", [False, True])
def test_nonfinite_current_defaults_rejected_before_side_effect(tmp_path, keyword):
    store = JSONLStorage(str(tmp_path / "calls"))
    effect = tmp_path / "effect"
    if keyword:

        def fn(*, amount=1):
            effect.write_text("called")
            return amount
    else:

        def fn(amount=1):
            effect.write_text("called")
            return amount

    wrapped = trace(fn, storage=store)
    wrapped()
    row = list(store.iter(kind="call"))[0].data
    effect.unlink()
    if keyword:
        fn.__kwdefaults__["amount"] = float("nan")
    else:
        fn.__defaults__ = (float("inf"),)
    with pytest.raises(ValueError, match="finite JSON"):
        replay_call(row, wrapped)
    assert not effect.exists()
