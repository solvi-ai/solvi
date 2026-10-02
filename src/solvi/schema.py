"""pydantic models of solvi's public data: answer types, questions, results, trace records, traces, flows and responses.

solvi's own classes stay plain dataclasses (fast to build inside ask); these models are their serialized form:
`res.model_dump()` / `res.model_dump("json")` / `res.to_json()`, `Response.from_json(s, catalog=cat)`,
`Response.model_json_schema()`, and `system.response_schema()` — the same schema with each answer as its closed set.

JSON has no dates, sets, enums or models. Values of the stdlib types the dump flattens — date, datetime, time, Decimal,
UUID, set, frozenset, tuple, also inside lists and dicts — are written with their type next to them (a trace's
`init_types`, a record's `type`) and come back as they were, declared or not. For the rest (an enum, a pydantic model, a
dataclass) `catalog=` (or a System) restores typed values from the facts' types (a producer's return type, the type a given
fact's readers expect, or System(inputs=...)), so a trace restored from JSON replays with the same hashes. A value that is
neither (an untyped enum; an untyped date in a record stored by solvi ≤ 0.7.1) comes back as JSON gave it and is listed
in the loaded trace's `unrestored`: replay then reports its steps as "not_restored" — no verdict on the data — rather
than as damaged."""
from __future__ import annotations

import math
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, create_model
from pydantic_core import to_jsonable_python


class AnswerSpec(BaseModel):
    """An answer type: yes/no, one of options, ordered levels, any subset, a span of a text, a ranking or an estimate;
    `unknown`: "not stated" is a valid answer."""
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    kind: Literal["yes_no", "choice", "ordinal", "multi", "span", "rank", "estimate"]
    options: list[Any]
    descriptions: dict[Any, str] = {}                # option → what it means (keyed by the option's text in JSON)
    unknown: bool = False
    k: Optional[int] = None
    bins: Optional[list[float]] = None
    coverage: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    type: Optional[str] = None                      # a span's value type, by name


class QuestionSpec(BaseModel):
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    name: str
    text: str
    answer: Optional[AnswerSpec] = None
    checkpoints: list[str] = []
    uses: Optional[list[str]] = None
    min_confidence: Optional[float] = None
    require_evidence: bool = False


class QuoteModel(BaseModel):
    """A supporting quote: text[start:end] of a given text fact."""
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    text: Any = None
    start: int
    end: int
    source: str = "doc"
    confidence: float = 1.0


class ResultModel(BaseModel):
    """The answer to one question."""
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    answer: Any = None
    confidence: float
    why: str
    status: Literal["ok", "forced", "abstain"] = "ok"
    probs: dict[str, float] = {}
    provenance: Optional[str] = None
    source: Optional[str] = None
    guard: Optional[str] = None
    repaired: Optional[tuple[Any, list[str]]] = None
    kind: Optional[str] = None
    not_stated: bool = False            # the answer is solvi.Unknown ("the text does not state it"; `answer` is null)
    evidence: list[QuoteModel] = []
    extra: Optional[dict[str, Any]] = None


class RecordModel(BaseModel):
    """One step of the hash-chained trace. `missing`: the step produced no value (failed or rejected; see `error`)."""
    model_config = ConfigDict(protected_namespaces=(), defer_build=True)
    step: int
    kind: str
    name: str
    inputs: dict[str, str]
    value: Any = None
    missing: bool = False
    not_stated: bool = False            # the value is solvi.Unknown
    native: bool = True                 # False: the value is not plain JSON (a date, an enum, a model): restored on load
    type: Any = None                    # its stdlib type as recorded by the dump (see type_tree), restored without a catalog
    quote: Optional[tuple[int, int, str]] = None
    confidence: float = 1.0
    error: Optional[str] = None
    prev: str = ""
    hash: str = ""
    producer: Optional[str] = None
    tried: Optional[list[tuple[str, str]]] = None
    provenance: Optional[str] = None
    model: Optional[dict[str, str]] = None
    probs: Optional[dict[str, float]] = None
    extra: Optional[dict[str, Any]] = None   # a model decision's details: act probability, expected level, shared pass
    tried_models: Optional[dict[str, dict[str, Any]]] = None   # the model-backed producers that ran and were not used


class TraceModel(BaseModel):
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    init_hash: str
    init: dict[str, Any]
    records: list[RecordModel]
    skipped: list[tuple[str, str]] = []
    schedule: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    rejected: list[tuple[str, str]] = []
    typed_init: list[str] = []          # given facts whose values are not plain JSON: restored on load
    init_types: dict[str, Any] = {}     # ... and the stdlib type of each as recorded by the dump (see type_tree)
    fingerprint: dict[str, Any] = {}    # the catalog's, questions' and models' fingerprints (System.fingerprint)
    early_exit: bool = True             # False: the whole flow was computed although a hard check failed


class StepModel(BaseModel):
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    name: str
    kind: str
    inputs: list[str]
    reasons: list[str] = []
    hard: Optional[bool] = None         # a check: is it a hard check? (None: not recorded — stored by solvi ≤ 0.7.1)


class FlowModel(BaseModel):
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    steps: list[StepModel]
    per_question: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    unresolved: dict[str, list[str]] = {}
    types: dict[str, str] = {}          # fact → type name
    batches: list[list[str]] = []       # decision parts scored together in one forward pass


class SafeguardEvent(BaseModel):
    model_config = ConfigDict(defer_build=True)       # validators built on first use: importing stays light
    kind: str
    fact: str
    detail: str
    questions: list[str] = []


class ResponseModel(BaseModel):
    """A response: answers, the flow, the trace, every fact's value, safeguards."""
    model_config = ConfigDict(protected_namespaces=(), defer_build=True)
    results: dict[str, ResultModel]
    flow: FlowModel
    trace: TraceModel
    values: dict[str, Any] = {}
    ms: float = 0.0
    feasible: bool = True
    violations: list[str] = []
    safeguards: list[SafeguardEvent] = []
    model_outputs: int = 0
    overall: dict[str, Any] = {}                      # derived on dump (Response.overall); ignored on load


MODELS = {"AnswerType": AnswerSpec, "Question": QuestionSpec, "Result": ResultModel, "Record": RecordModel,
          "Trace": TraceModel, "Flow": FlowModel, "Response": ResponseModel}


# --- non-finite floats: JSON has no inf / nan, so they are written as a tagged value that reads back as the float
FLOAT_TAG = "$float"                                  # {"$float": "inf"} / "-inf" / "nan" (also in calibration files)


def tag_floats(v):
    """JSON data with every non-finite float as {"$float": "inf" | "-inf" | "nan"} (strict JSON: no Infinity / NaN).
    Returns v itself when nothing changes."""
    try:                                              # fast path: serialized in C, no Infinity / NaN token → nothing to tag
        from pydantic_core import to_json
        raw = to_json(v)
        if b"Infinity" not in raw and b"NaN" not in raw:
            return v
    except Exception:  # noqa: BLE001, S110 — not plain JSON data (numpy scalars, ...): walk it
        pass
    return _tag(v)


def _tag(v):
    t = type(v)
    if t is float or isinstance(v, float):
        return v if math.isfinite(v) else {FLOAT_TAG: repr(float(v))}
    if t is dict:
        out = None
        for k, x in v.items():
            y = _tag(x)
            if y is not x and out is None:
                out = dict(v)
            if out is not None:
                out[k] = y
        return v if out is None else out
    if t is list or t is tuple:
        ys = [_tag(x) for x in v]
        return v if all(y is x for x, y in zip(ys, v)) else ys
    if t in (str, int, bool) or v is None:
        return v
    if hasattr(v, "item") and not isinstance(v, (str, bytes)) and getattr(v, "ndim", 1) == 0:   # numpy scalars
        return _tag(v.item())
    return v


def untag_floats(v):
    """The inverse of tag_floats: {"$float": "inf"} → inf. Returns v itself when nothing changes."""
    try:                                              # fast path: no tag anywhere → nothing to do
        from pydantic_core import to_json
        if b'"$float"' not in to_json(v):
            return v
    except Exception:  # noqa: BLE001, S110 — not plain JSON data: walk it
        pass
    return _untag(v)


def _untag(v):
    t = type(v)
    if t is dict:
        if len(v) == 1 and FLOAT_TAG in v and isinstance(v[FLOAT_TAG], str):
            try:
                return float(v[FLOAT_TAG])
            except ValueError:
                return v
        out = None
        for k, x in v.items():
            y = _untag(x)
            if y is not x and out is None:
                out = dict(v)
            if out is not None:
                out[k] = y
        return v if out is None else out
    if t is list:
        ys = [_untag(x) for x in v]
        return v if all(y is x for x, y in zip(ys, v)) else ys
    return v


def dumps(obj, **kw):
    """Strict JSON text (json.dumps with allow_nan=False) with non-finite floats tagged (tag_floats)."""
    import json
    return json.dumps(tag_floats(obj), allow_nan=False, **kw)


# --- dataclass → data
def _fallback(v):
    if hasattr(v, "tolist"):                          # numpy arrays and scalars (a non-finite one tagged)
        return _tag(v.tolist())
    return repr(v)


_LEAVES = (str, int, bool, type(None))


def _sorted_sets(v):
    """Sets as lists in the order vhash uses, so a set fact loaded back from JSON hashes (and replays) as before; a
    non-finite float as {"$float": ...} (tag_floats), in the same walk."""
    t = type(v)
    if t in _LEAVES:
        return v
    if t is float:
        return v if math.isfinite(v) else {FLOAT_TAG: repr(v)}
    if t is dict:
        return {k: _sorted_sets(x) for k, x in v.items()}
    if t is list:
        return [_sorted_sets(x) for x in v]
    if t is tuple:
        return tuple(_sorted_sets(x) for x in v)
    if isinstance(v, (set, frozenset)):
        from .runtime import _canon, _ckey
        return [_sorted_sets(x) for x in sorted(v, key=lambda x: _ckey(_canon(x)))]
    if isinstance(v, dict):
        return {k: _sorted_sets(x) for k, x in v.items()}
    if isinstance(v, float):                          # numpy float64 and other float subclasses
        return v if math.isfinite(v) else {FLOAT_TAG: repr(float(v))}
    return v


def jsonable(v):
    """JSON data: pydantic's to_jsonable_python, with sets sorted and non-finite floats tagged (strict JSON)."""
    return to_jsonable_python(_sorted_sets(v), fallback=_fallback)


def native(v):
    """Is v plain JSON data (tuples count: they hash like lists), so it comes back from JSON as it was?"""
    if v is None or type(v) in (str, int, float, bool):
        return True
    if type(v) in (list, tuple):
        return all(native(x) for x in v)
    if type(v) is dict:
        return all(type(k) is str and native(x) for k, x in v.items())
    return False


# --- stdlib values JSON flattens: the dump records their type (a "type tree"), the load restores them from it
UNRESTORABLE = "?"                                    # in a type tree: a value only a declared type can restore
_LEAF = None


def _leaves():
    """name → (type, restore from the JSON form) of the leaf types a dump records; type → name."""
    global _LEAF
    if _LEAF is None:
        import datetime
        import decimal
        import uuid

        def iso(cls):
            return lambda t: cls.fromisoformat(t[:-1] + "+00:00" if t.endswith("Z") else t)
        by_name = {"date": (datetime.date, datetime.date.fromisoformat), "datetime": (datetime.datetime, iso(datetime.datetime)),
                   "time": (datetime.time, iso(datetime.time)), "Decimal": (decimal.Decimal, decimal.Decimal),
                   "UUID": (uuid.UUID, uuid.UUID)}
        _LEAF = (by_name, {t: n for n, (t, _) in by_name.items()})
    return _LEAF


_SEQS = {"list": list, "tuple": tuple, "set": set, "frozenset": frozenset}


def type_tree(v):
    """What a load needs to give `v` back from its JSON form, or None when JSON gives it back as it is:
    "date" / "datetime" / "time" / "Decimal" / "UUID" for a value of exactly that type that its JSON form restores;
    {"k": "list" | "tuple" | "set" | "frozenset", "of": tree} (every item alike) or {..., "items": [tree]} (a set's
    items in the order the dump writes them); {"k": "dict", "items": {key: tree}} for the keys that need one;
    "?" (UNRESTORABLE) for anything else — an enum, a model, an aware datetime whose zone its text does not carry."""
    t = type(v)
    if v is None or t in (str, int, float, bool):
        return None
    if t in (list, tuple, set, frozenset):
        trees = [type_tree(x) for x in v]
        if t is list and all(x is None for x in trees):
            return None
        if all(x == trees[0] for x in trees):
            return {"k": t.__name__} if not trees or trees[0] is None else {"k": t.__name__, "of": trees[0]}
        if t in (set, frozenset):
            from .runtime import _canon, _ckey
            trees = [type_tree(x) for x in sorted(v, key=lambda x: _ckey(_canon(x)))]
        return {"k": t.__name__, "items": trees}
    if t is dict:
        if not all(type(k) is str for k in v):
            return UNRESTORABLE
        trees = {k: tr for k, x in v.items() for tr in [type_tree(x)] if tr is not None}
        return {"k": "dict", "items": trees} if trees else None
    by_name, by_type = _leaves()
    name = by_type.get(t)
    if name is None:
        return UNRESTORABLE
    try:
        back = by_name[name][1](to_jsonable_python(v))
    except Exception:  # noqa: BLE001 — its JSON form does not parse back
        return UNRESTORABLE
    return name if type(back) is t and repr(back) == repr(v) else UNRESTORABLE


def restorable(tree):
    """Does a type tree restore the whole value (no part of it needs a declared type)?"""
    if tree is None or isinstance(tree, str):
        return tree != UNRESTORABLE
    if not isinstance(tree, dict):
        return False
    if "of" in tree:
        return restorable(tree["of"])
    items = tree.get("items")
    return all(restorable(x) for x in (items.values() if isinstance(items, dict) else items or ()))


def from_tree(tree, v):
    """A value's JSON form → the value its type tree describes; what does not fit the tree (already restored, edited)
    is returned as it is."""
    if tree is None or tree == UNRESTORABLE or v is None:
        return v
    if isinstance(tree, str):
        leaf = _leaves()[0].get(tree)
        if leaf is None or not isinstance(v, str):
            return v
        try:
            return leaf[1](v)
        except Exception:  # noqa: BLE001
            return v
    if not isinstance(tree, dict):
        return v
    if tree.get("k") == "dict":
        items = tree.get("items")
        if not isinstance(v, dict) or not isinstance(items, dict):
            return v
        return {k: from_tree(items.get(k), x) for k, x in v.items()}
    cls = _SEQS.get(tree.get("k"))
    if cls is None or not isinstance(v, (list, tuple)):
        return v
    items = tree.get("items")
    if isinstance(items, list) and len(items) == len(v):
        out = [from_tree(t, x) for t, x in zip(items, v)]
    else:
        out = [from_tree(tree.get("of"), x) for x in v]
    try:
        return cls(out)
    except TypeError:                                 # an item that cannot be in a set (it was not restored)
        return v


_TYPES = None


def _type_names():
    global _TYPES
    if _TYPES is None:
        import datetime
        import decimal
        _TYPES = {t.__name__: t for t in (str, int, float, bool, datetime.date, datetime.datetime, decimal.Decimal)}
    return _TYPES


def _answer(at):
    if at is None:
        return None
    # descriptions are keyed by the option's text (JSON keys are strings; an int option keeps its own type in `options`)
    d = {"kind": at.kind, "options": list(at.options), "descriptions": {str(k): v for k, v in at.descriptions.items()}}
    for k in ("unknown", "k", "bins", "coverage", "unit", "source"):
        v = getattr(at, k)
        if v not in (None, False):                    # only what is set: answer types of 0.4 dump as before
            d[k] = list(v) if k == "bins" else v
    if at.type is not None:
        d["type"] = _span_type_name(at.type)
    return d


def _span_type_name(t):
    """A span's value type → its name in the dump: a built-in one by its name (str, int, float, bool, date, datetime,
    Decimal), any other class by "module:qualname" (an Enum, a model), so it can be found again on load."""
    from .typed import type_name
    if t in _type_names().values():
        return type_name(t)
    if isinstance(t, type) and "<locals>" not in t.__qualname__:
        return f"{t.__module__}:{t.__qualname__}"
    return type_name(t)


def _span_type(name):
    """The inverse of _span_type_name; a name that cannot be resolved raises ValueError (it used to load as None: a span
    of any text)."""
    t = _type_names().get(name)
    if t is not None:
        return t
    if ":" in name:
        import importlib
        mod, _, qual = name.partition(":")
        try:
            t = importlib.import_module(mod)
            for part in qual.split("."):
                t = getattr(t, part)
            if isinstance(t, type):
                return t
        except (ImportError, AttributeError):
            pass
    raise ValueError(f"span type {name!r} cannot be restored from its name: a built-in type (str, int, float, bool, "
                     "date, datetime, Decimal) or a class importable as module:qualname is")


def _unknown(v):
    from .core import Unknown
    return v is Unknown


def _record(r):
    from .runtime import MISSING
    missing = r.value is MISSING
    ns = _unknown(r.value)
    d = {"step": r.step, "kind": r.kind, "name": r.name, "inputs": dict(r.inputs),
         "value": None if (missing or ns) else r.value,
            "missing": missing, "native": missing or ns or native(r.value), "quote": None if r.quote is None else list(r.quote), "confidence": r.confidence,
            "error": r.error, "prev": r.prev, "hash": r.hash, "producer": r.producer,
            "tried": None if r.tried is None else [list(t) for t in r.tried], "provenance": r.provenance,
            "model": r.model, "probs": None if r.probs is None else {str(k): v for k, v in r.probs.items()},
            "extra": r.extra}
    if r.tried_models is not None:
        d["tried_models"] = r.tried_models
    if ns:
        d["not_stated"] = True
    if not (missing or ns):
        tree = type_tree(r.value)
        if tree is not None:
            d["type"] = tree
    return d


def _trace(t):
    d = {"init_hash": t.init_hash, "init": dict(t.init), "records": [_record(r) for r in t.records],
         "skipped": [list(s) for s in t.skipped], "schedule": list(t.schedule), "timings": dict(t.timings),
         "rejected": [list(x) for x in getattr(t, "rejected", None) or ()],
         "typed_init": [k for k, v in t.init.items() if not native(v)],
         "fingerprint": dict(getattr(t, "fingerprint", None) or {})}
    types = {k: tree for k, v in t.init.items() for tree in [type_tree(v)] if tree is not None}
    if types:
        d["init_types"] = types
    if not getattr(t, "early_exit", True):            # only when it was switched off: other traces dump as before
        d["early_exit"] = False
    return d


def _result(r):
    ns = _unknown(r.answer)
    rep = None if r.repaired is None else [None if _unknown(r.repaired[0]) else r.repaired[0], list(r.repaired[1])]
    return {"answer": None if ns else r.answer, "confidence": r.confidence, "why": r.why, "status": r.status,
            "probs": {str(k): v for k, v in (r.probs or {}).items()}, "provenance": r.provenance, "source": r.source,
            "guard": r.guard, "repaired": rep, "kind": r.kind, "not_stated": ns,
            "evidence": [{"text": e.value, "start": e.start, "end": e.end, "source": e.source, "confidence": e.confidence}
                         for e in r.evidence or ()],
            "extra": r.extra}


def _flow(f):
    from .typed import type_name
    return {"steps": [{"name": s.part.name, "kind": s.part.kind, "inputs": list(s.part.inputs), "reasons": list(s.reasons),
                       **({"hard": None if getattr(s.part, "hard_unknown", False) else bool(s.part.hard)}
                          if s.part.kind == "check" else {})} for s in f.steps],
            "per_question": {q: list(v) for q, v in f.per_question.items()}, "skipped": dict(f.skipped),
            "unresolved": {q: list(v) for q, v in f.unresolved.items()},
            "types": {k: type_name(t) if not isinstance(t, str) else t for k, t in (getattr(f, "types", None) or {}).items()},
            "batches": [list(b) for b in getattr(f, "batches", None) or ()]}


def _to_dict(obj):
    n = type(obj).__name__
    if n == "AnswerType":
        return _answer(obj)
    if n == "Question":
        d = {"name": obj.name, "text": obj.text, "answer": _answer(obj.answer), "checkpoints": list(obj.checkpoints),
             "uses": None if obj.uses is None else list(obj.uses), "min_confidence": obj.min_confidence}
        if obj.require_evidence:
            d["require_evidence"] = True
        return d
    if n == "Result":
        return _result(obj)
    if n == "Record":
        return _record(obj)
    if n == "Trace":
        return _trace(obj)
    if n == "Flow":
        return _flow(obj)
    if n == "Response":
        return {"results": {q: _result(r) for q, r in obj.results.items()}, "flow": _flow(obj.flow),
                "trace": _trace(obj.trace), "values": dict(obj.values), "ms": obj.ms, "feasible": obj.feasible,
                "violations": list(obj.violations or []), "safeguards": [{k: e.get(k) for k in ("kind", "fact", "detail", "questions")} for e in obj.safeguards or []],
                "model_outputs": obj.model_outputs, "overall": obj.overall}
    raise TypeError(f"no schema for {n}")


def dump(obj, mode="python"):
    """mode="json": JSON-ready data — non-finite floats tagged ({"$float": "inf"}, see tag_floats), so it is strict JSON."""
    d = _to_dict(obj)
    return jsonable(d) if mode == "json" else d        # jsonable tags non-finite floats


def json_schema(cls):
    return MODELS[cls.__name__].model_json_schema()


# --- data → dataclass
def _types(catalog):
    """→ (catalog, System or None)."""
    if catalog is not None and hasattr(catalog, "catalog") and hasattr(catalog, "questions"):
        return catalog.catalog, catalog
    return catalog, None


def _restore(t, v):
    from .typed import check, spec
    if t is None or v is None:
        return v
    s = spec(t, "restore")
    if s is None:
        return v
    nv, err = check(s, v)
    return v if err else nv


def _given_type(catalog, system, k):
    m = getattr(system, "inputs", None) if system is not None else None
    if m is not None and k in m.model_fields:
        return m.model_fields[k].annotation
    rs = catalog.readers.get(k) if catalog is not None else None
    return next(iter(rs.values())) if rs else None


def _fact_type(catalog, name, producer=None):
    if catalog is None:
        return None
    part = catalog.rules.get(name[7:]) if name.startswith("answer:") else catalog.parts.get(name)
    if part is None:
        return None
    if part.alternatives is not None and producer:
        try:
            part = catalog.alternative(name, producer)
        except KeyError:
            return None
    return part.returns


def _load_answer(d):
    from .core import AnswerType
    if d is None:
        return None
    by_text = {str(o): o for o in d.options}          # descriptions back on the options themselves (1, not "1")
    return AnswerType(d.kind, list(d.options), {by_text.get(str(k), k): v for k, v in d.descriptions.items()}, d.unknown,
                      d.k, None if d.bins is None else [int(b) if float(b).is_integer() else b for b in d.bins], d.coverage,
                      d.unit, d.source, _span_type(d.type) if d.type else None)


def _probs(p):
    from .core import unknown_key
    return {unknown_key(k): v for k, v in (p or {}).items()}


def _load_result(m):
    from .core import Quote, Unknown
    from .runtime import Result
    a = Unknown if m.not_stated else tuple(m.answer) if isinstance(m.answer, list) else m.answer   # multi-label, rank: tuples
    rep = None if m.repaired is None else (tuple(m.repaired[0]) if isinstance(m.repaired[0], list) else m.repaired[0],
                                           list(m.repaired[1]))
    ev = [Quote(e.text, e.start, e.end, e.source, e.confidence) for e in m.evidence]
    return Result(a, m.confidence, m.why, m.status, _probs(m.probs), m.provenance, m.source, m.guard, rep, m.kind, ev,
                  m.extra)


def _back(tree, declared, v, flattened):
    """A value from its JSON form → (the value, why it could not be restored or None): by the type the dump recorded
    when that restores all of it, else by the declared type (`declared`: a function → the type or None), else as far
    as the recorded type goes. flattened: the dump says the value is not plain JSON."""
    if tree is not None and restorable(tree):
        return from_tree(tree, v), None
    if tree is None and not flattened:
        return v, None
    t = declared()
    if t is not None:
        return _restore(t, v), None
    if tree is None:
        return v, "its type was not recorded (stored by solvi ≤ 0.7.1) and none is declared"
    return from_tree(tree, v), "its type is not one the dump restores (an enum, a model, a class) and none is declared"


def _load_trace(m, catalog, system):
    from .runtime import MISSING, Record, Trace
    typed = set(m.typed_init)
    init, lost = {}, {}                               # lost: fact → why its value is not the one that was hashed
    for k, v in m.init.items():
        init[k], why = _back(m.init_types.get(k), lambda k=k: _given_type(catalog, system, k), v, k in typed)
        if why:
            lost[k] = why
    recs = []
    from .core import Unknown
    for r in m.records:
        if r.missing or r.not_stated:
            v = MISSING if r.missing else Unknown
        else:
            v, why = _back(r.type, lambda r=r: _given_type(catalog, system, r.name[7:]) if r.kind == "textin" else
                           _fact_type(catalog, r.name, r.producer), r.value, not r.native)
            if why:
                lost[r.name] = why
        recs.append(Record(r.step, r.kind, r.name, dict(r.inputs), v, None if r.quote is None else tuple(r.quote),
                           r.confidence, r.error, r.prev, r.hash, r.producer,
                           None if r.tried is None else [list(t) for t in r.tried], r.provenance, r.model,
                           None if r.probs is None else _probs(r.probs), r.extra, r.tried_models))
    tr = Trace(m.init_hash, recs, init, [tuple(s) for s in m.skipped], list(m.schedule), dict(m.timings),
               [tuple(x) for x in m.rejected], dict(m.fingerprint), m.early_exit)
    if lost:
        tr.unrestored = lost                          # replay: mismatches these explain are "not_restored", not damage
    return tr


def _stub(name, kind, inputs, hard=None):
    """A part of a flow loaded from data without its catalog: its name, kind, inputs and — for a check — whether it is
    a hard one, as the flow recorded it (`hard_unknown`: it did not, so nothing may call it soft)."""
    from .core import Part

    def f(**_):
        raise RuntimeError("a part of a flow loaded from data: its function is not available")
    f.__name__ = name[7:] if name.startswith("answer:") else name
    p = Part(kind=kind, name=name, inputs=list(inputs), func=f, question=name[7:] if kind == "rule" else None,
             hard=bool(hard))
    if kind == "check" and hard is None:
        p.hard_unknown = True
    return p


def _load_flow(m, catalog):
    from .strategist import Flow, Step
    steps = []
    for s in m.steps:
        p = None
        if catalog is not None:
            p = catalog.rules.get(s.name[7:]) if s.name.startswith("answer:") else catalog.parts.get(s.name)
        steps.append(Step(p if p is not None else _stub(s.name, s.kind, s.inputs, s.hard), list(s.reasons)))
    return Flow(steps, {q: list(v) for q, v in m.per_question.items()}, dict(m.skipped),
                {q: list(v) for q, v in m.unresolved.items()}, dict(m.types), [list(b) for b in m.batches])


def load(cls, data, catalog=None):
    n = cls.__name__
    m = MODELS[n].model_validate(untag_floats(data) if isinstance(data, dict) else data)
    catalog, system = _types(catalog)
    if n == "AnswerType":
        return _load_answer(m)
    if n == "Question":
        from .core import Question
        return Question(m.name, m.text, _load_answer(m.answer), list(m.checkpoints), m.uses, m.min_confidence,
                        m.require_evidence)
    if n == "Result":
        return _load_result(m)
    if n == "Record":
        return _load_trace(TraceModel(init_hash="", init={}, records=[m]), catalog, system).records[0]
    if n == "Trace":
        return _load_trace(m, catalog, system)
    if n == "Flow":
        return _load_flow(m, catalog)
    if n == "Response":
        from .runtime import MISSING
        from .system import Response
        tr = _load_trace(m.trace, catalog, system)
        known = dict(tr.init)
        known.update({r.name: r.value for r in tr.records if r.value is not MISSING})
        values = {k: known.get(k, v) for k, v in m.values.items()}
        resp = Response({q: _load_result(r) for q, r in m.results.items()}, _load_flow(m.flow, catalog), tr, values, m.ms,
                        m.feasible, list(m.violations), catalog, [e.model_dump() for e in m.safeguards], m.model_outputs)
        if system is not None:
            resp._system = system                     # loaded with a System: reports and counterfactuals use it
            resp._heads = system.heads
        return resp
    raise TypeError(f"no schema for {n}")


# --- a system's own response schema
def response_model(system, names=None):
    """ResponseModel with `results` as an object of the system's questions (or of `names`), each answer its closed set: a
    pydantic class (solvi serve documents its endpoints with it)."""
    fields = {}
    for q in system.questions.values():
        if names is not None and q.name not in names:
            continue
        at = q.answer
        lit = Literal[tuple(at.options)] if at.options else Any
        ans = list[lit] if at.kind in ("multi", "rank") else float if at.kind == "estimate" else Any if at.kind == "span" \
            else lit
        R = create_model(f"Result_{q.name}", __base__=ResultModel, __doc__=q.text or None,
                         answer=(Optional[ans], None))
        fields[q.name] = (R, ...)
    Results = create_model("Results" if names is None else "Results_" + "_".join(fields), **fields)
    return create_model("Response", __base__=ResponseModel, results=(Results, ...))


def response_schema(system):
    """ResponseModel's JSON schema with `results` as an object of the system's questions, each answer its closed set."""
    return response_model(system).model_json_schema()
