"""Strict bounded payload schema subset for Process declarations; no runner logic."""
import json
import math

from ..core._process_declaration import frozen

__all__ = []

_KEYS = {
    "object": {"properties", "required", "additionalProperties"},
    "array": {"items", "minItems", "maxItems"},
    "string": {"minLength", "maxLength"},
    "number": {"minimum", "maximum"},
    "integer": {"minimum", "maximum"}, "boolean": set(), "null": set(),
}


def schema_check(s, depth=0):
    if depth > 32 or not isinstance(s, dict) or s.get("type") not in _KEYS:
        raise ValueError("schema requires a supported type; nesting limit is 32")
    t = s["type"]
    if set(s) - ({"type", "enum"} | _KEYS[t]):
        raise ValueError("unsupported schema keywords")
    if "enum" in s and (not isinstance(s["enum"], list) or not s["enum"]):
        raise ValueError("enum must be a nonempty list")
    if t == "object":
        props = s.get("properties", {})
        if not isinstance(props, dict) or any(not isinstance(k, str) for k in props):
            raise ValueError("properties must map string keys to schemas")
        req = s.get("required", [])
        if not isinstance(req, list) or any(not isinstance(k, str) or k not in props for k in req) or len(set(req)) != len(req):
            raise ValueError("required must contain distinct declared property names")
        if type(s.get("additionalProperties", True)) is not bool:
            raise ValueError("additionalProperties must be boolean")
        for sub in props.values():
            schema_check(sub, depth + 1)
    if "items" in s:
        schema_check(s["items"], depth + 1)
    for lo, hi in (("minItems", "maxItems"), ("minLength", "maxLength"), ("minimum", "maximum")):
        for key in (lo, hi):
            if key in s:
                v = s[key]
                if type(v) not in (int, float) or (type(v) is float and not math.isfinite(v)) or (key not in ("minimum", "maximum") and (type(v) is not int or v < 0)):
                    raise ValueError("invalid schema bound")
        if lo in s and hi in s and s[lo] > s[hi]:
            raise ValueError("schema lower bound exceeds upper bound")


def payload_check(s, value, depth=0):
    if depth > 32:
        raise ValueError("payload nesting exceeds 32")
    t = s["type"]
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool, "null": type(None)}
    if t == "number":
        good = type(value) is int or type(value) is float and math.isfinite(value)
    else:
        good = type(value) is types[t]
    if not good:
        raise ValueError("payload type differs from " + t)
    if "enum" in s and not any(json.dumps(value, sort_keys=True) == json.dumps(v, sort_keys=True) for v in s["enum"]):
        raise ValueError("payload is outside enum")
    if t == "object":
        if any(k not in value for k in s.get("required", [])):
            raise ValueError("payload lacks required property")
        props = s.get("properties", {})
        if not s.get("additionalProperties", True) and set(value) - set(props):
            raise ValueError("payload contains extra properties")
        for k, v in value.items():
            if k in props:
                payload_check(props[k], v, depth + 1)
    if t == "array" and "items" in s:
        for v in value:
            payload_check(s["items"], v, depth + 1)
    for lo, hi, measured in (("minimum", "maximum", value), ("minLength", "maxLength", len(value) if t == "string" else 0),
                             ("minItems", "maxItems", len(value) if t == "array" else 0)):
        if lo in s and measured < s[lo] or hi in s and measured > s[hi]:
            raise ValueError("payload violates schema bound")


def invariant_check(p, depth=0):
    if depth > 32 or not isinstance(p, dict) or not p:
        raise ValueError("invariant must be a payload predicate")
    if "answer" in p:
        return
    if set(p) == {"not"}:
        invariant_check(p["not"], depth + 1)
    elif len(p) == 1 and next(iter(p)) in ("and", "or"):
        for sub in next(iter(p.values())):
            invariant_check(sub, depth + 1)
    else:
        raise ValueError("invariant supports only answer/and/or/not predicates")
