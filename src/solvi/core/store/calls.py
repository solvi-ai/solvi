"""Explicit eager JSON function-call capture. Stored calls are not decision/training records."""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import math
import uuid
from copy import deepcopy

from ..provenance import code_fingerprint, digest, _Seen
from . import FORMAT, open_storage, record_hash

_PARENT = contextvars.ContextVar("solvi_native_call_parent", default=None)
_CAPTURE = contextvars.ContextVar("solvi_native_call_capture", default=True)


def _json(value, active=None, depth=0):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if depth > 128 or type(value) not in (dict, list):
        raise ValueError("native trace requires finite JSON values; nested tuples/objects are unsupported")
    if type(value) is dict and any(type(key) is not str for key in value):
        raise ValueError("native trace requires string object keys")
    active = set() if active is None else active
    if id(value) in active:
        raise ValueError("native trace cannot store cyclic JSON")
    active.add(id(value))
    try:
        for item in value.values() if type(value) is dict else value:
            _json(item, active, depth + 1)
    finally:
        active.remove(id(value))


def _target(fn):
    if not callable(fn):
        raise TypeError("trace/replay requires an explicit callable")
    target = getattr(fn, "__solvi_trace_target__", fn)
    if not inspect.isfunction(target):
        raise TypeError("native trace requires a Python function")
    if inspect.isgeneratorfunction(target) or inspect.iscoroutinefunction(target) or inspect.isasyncgenfunction(target):
        raise TypeError("native trace supports only synchronous eager functions, not generators/async")
    return target


def _canonical(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(fn):
    target = _target(fn)
    # Existing code_fingerprint caches a function and mutable globals on first
    # sight. Explicit native replay must also notice current JSON environment.
    environment, pending, seen = [], [target], set()
    while pending:
        owner = pending.pop()
        if id(owner) in seen:
            continue
        seen.add(id(owner))
        # Defaults participate in omitted arguments and must be read afresh,
        # including mutable JSON containers. Only the outer positional tuple is
        # Python calling convention; nested tuples remain unsupported JSON.
        defaults = {"positional": list(owner.__defaults__ or ()), "keyword": owner.__kwdefaults__ or {}}
        _json(defaults)
        environment.append((owner.__qualname__, "__native_defaults__", _canonical(defaults)))
        values = [(name, owner.__globals__[name]) for name in owner.__code__.co_names if name in owner.__globals__]
        for name, cell in zip(owner.__code__.co_freevars, owner.__closure__ or ()):
            try:
                values.append((name, cell.cell_contents))
            except ValueError:
                pass
        for name, value in values:
            if inspect.isfunction(value) and value.__module__ == target.__module__:
                pending.append(getattr(value, "__solvi_trace_target__", value))
            try:
                _json(value)
            except ValueError:
                continue
            environment.append((owner.__qualname__, name, _canonical(value)))
    return digest("native-call-v1", code_fingerprint(target, _seen=_Seen()), sorted(environment))


def trace(fn=None, *, storage, name=None):
    """Wrap an eager JSON function, returning its actual result and storing kind='call'.

    Inputs are snapshotted before invocation; nested traced calls carry UUID
    parent links. Exceptions are recorded and re-raised. Capture/storage failures
    are explicit, and external side effects are never rolled back. Policies for
    generators, async, Python objects and nested tuples are intentionally absent.
    """
    if fn is None:
        return lambda target: trace(target, storage=storage, name=name)
    original = _target(fn)
    store = open_storage(storage)
    if store is None:
        raise ValueError("native trace requires storage")
    label = name if name is not None else f"{original.__module__}:{original.__qualname__}"
    if not isinstance(label, str) or not label:
        raise ValueError("native trace name must be a nonempty string")
    pinned = _fingerprint(original)

    @functools.wraps(original)
    def captured(*args, **kwargs):
        if not _CAPTURE.get():
            return original(*args, **kwargs)
        _json(list(args))
        _json(kwargs)
        record = {
            "v": FORMAT,
            "kind": "call",
            "call_id": uuid.uuid4().hex,
            "parent_call_id": _PARENT.get(),
            "function": {"name": label, "fp": _fingerprint(original)},
            "args": deepcopy(list(args)),
            "kwargs": deepcopy(kwargs),
        }
        token = _PARENT.set(record["call_id"])
        try:
            try:
                output = original(*args, **kwargs)
            except Exception as error:
                record.update(
                    status="error",
                    error={"type": f"{type(error).__module__}:{type(error).__qualname__}", "message": str(error)},
                )
                store._append(record)
                raise
            _json(output)
            record.update(status="ok", output=deepcopy(output))
            store._append(record)
            return output
        finally:
            _PARENT.reset(token)

    captured.__solvi_trace_target__ = original
    captured.__solvi_trace_fp__ = pinned
    captured.__solvi_trace_name__ = label
    captured.storage = store
    return captured


def _record(record):
    data = record.data if hasattr(record, "data") else record
    if isinstance(data, dict) and data.get("format") == "solvi-native-call-v1":
        fields = {"format", "name", "function", "args", "kwargs", "expected"}
        if set(data) != fields or type(data.get("expected")) is not dict:
            raise ValueError("invalid native call fixture fields")
        expected = data["expected"]
        required = {"status", "output"} if expected.get("status") == "ok" else {"status", "error"}
        if set(expected) != required:
            raise ValueError("invalid native call fixture outcome")
        data = {
            "kind": "call",
            "call_id": data["name"],
            "parent_call_id": None,
            "function": data["function"],
            "args": data["args"],
            "kwargs": data["kwargs"],
            **expected,
        }
    if not isinstance(data, dict) or data.get("kind") != "call":
        raise ValueError("expected an explicit native kind=call record")
    if data.get("redacted"):
        raise ValueError("native call record was redacted")
    if any(key in data for key in ("v", "prev", "hash")):
        if not all(key in data for key in ("v", "prev", "hash")) or record_hash(data) != data["hash"]:
            raise ValueError("native call record hash does not verify")
    function = data.get("function")
    if (
        type(function) is not dict
        or not isinstance(function.get("name"), str)
        or not function["name"]
        or not isinstance(function.get("fp"), str)
        or len(function["fp"]) != 16
        or any(c not in "0123456789abcdef" for c in function["fp"])
    ):
        raise ValueError("invalid native function metadata")
    for key in ("call_id", "parent_call_id"):
        value = data.get(key)
        if key == "parent_call_id" and value is None:
            continue
        if not isinstance(value, str) or len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid native call UUID")
    if data["call_id"] == data.get("parent_call_id"):
        raise ValueError("native call cannot parent itself")
    args, kwargs = data.get("args"), data.get("kwargs")
    if type(args) is not list or type(kwargs) is not dict:
        raise ValueError("invalid native call arguments")
    _json(args)
    _json(kwargs)
    if data.get("status") not in ("ok", "error"):
        raise ValueError("invalid native call status")
    if data["status"] == "ok":
        if "output" not in data:
            raise ValueError("native call has no output")
        _json(data["output"])
    elif not isinstance(data.get("error"), dict) or not all(
        isinstance(data["error"].get(k), str) for k in ("type", "message")
    ):
        raise ValueError("invalid native exception record")
    return data


def _outcome(data):
    return {
        "status": data["status"],
        **({"output": data["output"]} if data["status"] == "ok" else {"error": data["error"]}),
    }


def _invoke(data, fn):
    target = _target(fn)
    token = _CAPTURE.set(False)
    try:
        try:
            value = target(*deepcopy(data["args"]), **deepcopy(data["kwargs"]))
        except Exception as error:  # noqa: BLE001 — an exception is the recorded callable's outcome
            return {
                "status": "error",
                "error": {"type": f"{type(error).__module__}:{type(error).__qualname__}", "message": str(error)},
            }
        _json(value)
        return {"status": "ok", "output": deepcopy(value)}
    finally:
        _CAPTURE.reset(token)


def replay_call(record, fn):
    """Explicit replay with a trusted callable; may repeat its side effects.

    Accepts sealed native records or explicit solvi-native-call-v1 fixtures.
    Never imports code by a stored name. Capture is suppressed for the whole
    replay subtree. Pinned function identity must match before invocation.
    """
    data = _record(record)
    if _fingerprint(fn) != data.get("function", {}).get("fp"):
        return {"ok": False, "reason": "function fingerprint changed"}
    actual = _invoke(data, fn)
    return {"ok": _canonical(actual) == _canonical(_outcome(data)), "expected": _outcome(data), "actual": actual}


def _calls(records):
    return records.iter(kind="call") if hasattr(records, "iter") else records


def diff_calls(records, fn, *, name=None):
    """Explicitly re-run native records against a supplied replacement callable.

    Unlike replay, changed code is allowed and reported. No implicit execution
    by generic storage replay_all; this is not a decision-answer quality metric.
    """
    rows = [_record(item) for item in _calls(records)]
    name = name if name is not None else getattr(fn, "__solvi_trace_name__", None)
    if name is None and len({r["function"]["name"] for r in rows}) > 1:
        raise ValueError("a mixed-function store requires an explicit native call name")
    out = []
    for data in rows:
        if name is not None and data["function"]["name"] != name:
            continue
        actual = _invoke(data, fn)
        if _canonical(actual) != _canonical(_outcome(data)):
            out.append(
                {
                    "call_id": data["call_id"],
                    "function": data["function"],
                    "old": _outcome(data),
                    "new": actual,
                    "facts": {"return": {"old": _outcome(data), "new": actual}},
                }
            )
    return out


def cases_from_calls(records):
    """Export explicit native-call fixtures, not solvi.testing decision cases."""
    return [
        {
            "name": data["call_id"],
            "function": deepcopy(data["function"]),
            "args": deepcopy(data["args"]),
            "kwargs": deepcopy(data["kwargs"]),
            "expected": deepcopy(_outcome(data)),
            "format": "solvi-native-call-v1",
        }
        for data in (_record(item) for item in _calls(records))
    ]


__all__ = ["trace", "replay_call", "diff_calls", "cases_from_calls"]
