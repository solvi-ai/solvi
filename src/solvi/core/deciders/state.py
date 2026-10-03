"""The input a decider reads: a text, or a state serialized by state_text (paths / tree / json). (Part of solvi.core.deciders,
which re-exports every name.)"""
from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Mapping
from enum import Enum

import numpy as np

from ..catalog import Quote



SERIALIZATIONS = ("paths", "tree", "json")      # the state serializations this solvi writes (state_text)


_KEY_OK = re.compile(r"^[A-Za-z0-9_\-]+$")


def state_text(obj, fmt="paths"):
    """The input a decider reads: a text as it is; any other value (a dict, a list, a pydantic model, a dataclass) first as
    JSON data (see `jsonable`), then serialized — by default "paths", one line per leaf with its full key path:

        customer.name: Anna
        items[0].sku: A-1
        items[0].qty: 2
        note: two lines of text

    Keys in their order (a pydantic model: field order); a key that is not [A-Za-z0-9_-]+ is written ["key"] (a JSON
    string); strings without quotes (a new line becomes a space); null / true / false; floats rounded to 6 decimals without
    trailing zeros; empty {} and [] kept; a scalar at the top is ".: value". "tree" is the YAML-like indented form, "json"
    is json.dumps with ", " / ": " separators. These are exactly the typed decider's training serializations
    (`serialize` of its training code); docs/decide_format.md has the rules."""
    if isinstance(obj, Quote):
        obj = obj.value
    if isinstance(obj, str):
        return obj
    data = jsonable(obj)
    if fmt == "json":
        return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))
    out = []
    if fmt == "paths":
        _paths(data, "", out)
    elif fmt == "tree":
        _tree(data, 0, out)
    else:
        raise ValueError(f"unknown state serialization {fmt!r} (this solvi writes {SERIALIZATIONS})")
    return "\n".join(out)


def jsonable(v):
    """A Python value → JSON data, deterministically: a pydantic model → its model_dump(), a dataclass → its fields, dates and
    times → ISO 8601, an Enum → its value, Decimal / UUID / other objects → str, bytes → UTF-8 text, tuples → lists, sets →
    lists sorted by their JSON text, numpy → Python numbers."""
    import datetime
    if v is None or isinstance(v, (str, bool, int, float)):
        return v.value if isinstance(v, Enum) else v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Enum):
        return jsonable(v.value)
    if isinstance(v, (datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", "replace")
    from ..types import is_model
    if is_model(v):
        return jsonable(v.model_dump())
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: jsonable(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, Mapping):
        return {_jkey(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((jsonable(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _jkey(k):
    if isinstance(k, Enum):
        k = k.value
    return k if isinstance(k, (str, int, float, bool)) or k is None else str(k)


def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(round(v, 6)).rstrip("0").rstrip(".") if abs(v) < 1e15 else repr(v)
    return str(v).replace("\n", " ")


def _pkey(prefix, k):
    k = str(k)
    part = k if _KEY_OK.match(k) else json.dumps(k, ensure_ascii=False)
    if not _KEY_OK.match(k):
        return f"{prefix}[{part}]"
    return f"{prefix}.{part}" if prefix else part


def _paths(v, prefix, out):
    if isinstance(v, dict):
        if not v:
            out.append(f"{prefix or '.'}: {{}}")
        for k, x in v.items():
            _paths(x, _pkey(prefix, k), out)
    elif isinstance(v, (list, tuple)):
        if not v:
            out.append(f"{prefix or '.'}: []")
        for i, x in enumerate(v):
            _paths(x, f"{prefix}[{i}]", out)
    else:
        out.append(f"{prefix or '.'}: {_scalar(v)}")


def _tree(v, ind, out, key=None):
    pad = "  " * ind
    if isinstance(v, dict):
        if key is not None:
            out.append(f"{pad}{key}:" + (" {}" if not v else ""))
            ind += 1
        for k, x in v.items():
            _tree(x, ind, out, str(k))
    elif isinstance(v, (list, tuple)):
        if key is not None:
            out.append(f"{pad}{key}:" + (" []" if not v else ""))
        p2 = "  " * (ind + (1 if key is not None else 0))
        for x in v:
            if isinstance(x, dict) and x:
                sub = []
                _tree(x, 0, sub)
                out.append(f"{p2}- {sub[0]}")
                out.extend(f"{p2}  {s}" for s in sub[1:])
            elif isinstance(x, (list, tuple)):
                sub = []
                _tree(x, 0, sub)
                out.append(f"{p2}-")
                out.extend(f"{p2}  {s}" for s in sub)
            else:
                out.append(f"{p2}- {_scalar(x)}")
    else:
        out.append(f"{pad}{key}: {_scalar(v)}" if key is not None else f"{pad}{_scalar(v)}")


def _is_text(v):
    return v is None or isinstance(v, str) or (isinstance(v, Quote) and (v.value is None or isinstance(v.value, str)))


def _text(v, fmt="paths"):
    """A fact's value → the decider's input: a text as it is (a Quote: its value), a scalar (number, date, Enum) as its
    text, a container (dict, list, model) by state_text."""
    if isinstance(v, Quote):
        v = v.value
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    data = jsonable(v)
    return state_text(data, fmt) if isinstance(data, (dict, list)) else _scalar(data)


def _single(x):
    """Is x one input (a text, a Quote, a state) rather than a list of inputs?"""
    if isinstance(x, (str, Quote, Mapping)) or (dataclasses.is_dataclass(x) and not isinstance(x, type)):
        return True
    from ..types import is_model
    return is_model(x)


__all__ = ["jsonable", "state_text"]
