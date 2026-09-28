"""pydantic models of solvi's public data: answer types, questions, results, trace records, traces, flows and responses.

solvi's own classes stay plain dataclasses (fast to build inside ask); these models are their serialized form:
`res.model_dump()` / `res.model_dump("json")` / `res.to_json()`, `Response.from_json(s, catalog=cat)`,
`Response.model_json_schema()`, and `system.response_schema()` — the same schema with each answer as its closed set.

JSON has no dates, enums or models: on load, `catalog=` (or a System) restores typed values from the facts' types (a
producer's return type, the type a given fact's readers expect, or System(inputs=...)), so a trace restored from JSON replays
with the same hashes. Untyped values that JSON cannot carry (a date in an untyped fact) come back as strings, and their steps
no longer replay."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, create_model
from pydantic_core import to_jsonable_python


class AnswerSpec(BaseModel):
    """An answer type: yes/no, one of options, ordered levels, any subset, a span of a text, a ranking or an estimate;
    `unknown`: "not stated" is a valid answer."""
    kind: Literal["yes_no", "choice", "ordinal", "multi", "span", "rank", "estimate"]
    options: list[Any]
    descriptions: dict[str, str] = {}
    unknown: bool = False
    k: Optional[int] = None
    bins: Optional[list[float]] = None
    coverage: Optional[float] = None
    unit: Optional[str] = None
    source: Optional[str] = None
    type: Optional[str] = None                      # a span's value type, by name


class QuestionSpec(BaseModel):
    name: str
    text: str
    answer: Optional[AnswerSpec] = None
    checkpoints: list[str] = []
    uses: Optional[list[str]] = None
    min_confidence: Optional[float] = None
    require_evidence: bool = False


class QuoteModel(BaseModel):
    """A supporting quote: text[start:end] of a given text fact."""
    text: Any = None
    start: int
    end: int
    source: str = "doc"
    confidence: float = 1.0


class ResultModel(BaseModel):
    """The answer to one question."""
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
    model_config = ConfigDict(protected_namespaces=())
    step: int
    kind: str
    name: str
    inputs: dict[str, str]
    value: Any = None
    missing: bool = False
    not_stated: bool = False            # the value is solvi.Unknown
    native: bool = True                 # False: the value is not plain JSON (a date, an enum, a model): restored on load
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


class TraceModel(BaseModel):
    init_hash: str
    init: dict[str, Any]
    records: list[RecordModel]
    skipped: list[tuple[str, str]] = []
    schedule: list[dict[str, Any]] = []
    timings: dict[str, float] = {}
    rejected: list[tuple[str, str]] = []
    typed_init: list[str] = []          # given facts whose values are not plain JSON: restored on load
    fingerprint: dict[str, Any] = {}    # the catalog's, questions' and models' fingerprints (System.fingerprint)


class StepModel(BaseModel):
    name: str
    kind: str
    inputs: list[str]
    reasons: list[str] = []


class FlowModel(BaseModel):
    steps: list[StepModel]
    per_question: dict[str, list[str]] = {}
    skipped: dict[str, str] = {}
    unresolved: dict[str, list[str]] = {}
    types: dict[str, str] = {}          # fact → type name
    batches: list[list[str]] = []       # decision parts scored together in one forward pass


class SafeguardEvent(BaseModel):
    kind: str
    fact: str
    detail: str
    questions: list[str] = []


class ResponseModel(BaseModel):
    """A response: answers, the flow, the trace, every fact's value, safeguards."""
    model_config = ConfigDict(protected_namespaces=())
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


# --- dataclass → data
def _fallback(v):
    if hasattr(v, "tolist"):                          # numpy arrays and scalars
        return v.tolist()
    return repr(v)


def _sorted_sets(v):
    """Sets as lists in the order vhash uses, so a set fact loaded back from JSON hashes (and replays) as before."""
    if isinstance(v, (set, frozenset)):
        from .runtime import _canon, _ckey
        return [_sorted_sets(x) for x in sorted(v, key=lambda x: _ckey(_canon(x)))]
    if isinstance(v, dict):
        return {k: _sorted_sets(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return type(v)(_sorted_sets(x) for x in v) if type(v) in (list, tuple) else v
    return v


def jsonable(v):
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
    d = {"kind": at.kind, "options": list(at.options), "descriptions": dict(at.descriptions)}
    for k in ("unknown", "k", "bins", "coverage", "unit", "source"):
        v = getattr(at, k)
        if v not in (None, False):                    # only what is set: answer types of 0.4 dump as before
            d[k] = list(v) if k == "bins" else v
    if at.type is not None:
        from .typed import type_name
        d["type"] = type_name(at.type)
    return d


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
    if ns:
        d["not_stated"] = True
    return d


def _trace(t):
    return {"init_hash": t.init_hash, "init": dict(t.init), "records": [_record(r) for r in t.records],
            "skipped": [list(s) for s in t.skipped], "schedule": list(t.schedule), "timings": dict(t.timings),
            "rejected": [list(x) for x in getattr(t, "rejected", None) or ()],
            "typed_init": [k for k, v in t.init.items() if not native(v)],
            "fingerprint": dict(getattr(t, "fingerprint", None) or {})}


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
    return {"steps": [{"name": s.part.name, "kind": s.part.kind, "inputs": list(s.part.inputs), "reasons": list(s.reasons)}
                      for s in f.steps],
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
    d = _to_dict(obj)
    return jsonable(d) if mode == "json" else d


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
    return AnswerType(d.kind, list(d.options), dict(d.descriptions), d.unknown, d.k,
                      None if d.bins is None else [int(b) if float(b).is_integer() else b for b in d.bins], d.coverage,
                      d.unit, d.source, _type_names().get(d.type) if d.type else None)


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


def _load_trace(m, catalog, system):
    from .runtime import MISSING, Record, Trace
    typed = set(m.typed_init)
    init = {k: _restore(_given_type(catalog, system, k), v) if k in typed else v for k, v in m.init.items()}
    recs = []
    from .core import Unknown
    for r in m.records:
        v = MISSING if r.missing else Unknown if r.not_stated else r.value if r.native else \
            _restore(_fact_type(catalog, r.name, r.producer), r.value)
        recs.append(Record(r.step, r.kind, r.name, dict(r.inputs), v, None if r.quote is None else tuple(r.quote),
                           r.confidence, r.error, r.prev, r.hash, r.producer,
                           None if r.tried is None else [list(t) for t in r.tried], r.provenance, r.model,
                           None if r.probs is None else _probs(r.probs), r.extra))
    return Trace(m.init_hash, recs, init, [tuple(s) for s in m.skipped], list(m.schedule), dict(m.timings),
                 [tuple(x) for x in m.rejected], dict(m.fingerprint))


def _stub(name, kind, inputs):
    from .core import Part

    def f(**_):
        raise RuntimeError("a part of a flow loaded from data: its function is not available")
    f.__name__ = name[7:] if name.startswith("answer:") else name
    return Part(kind=kind, name=name, inputs=list(inputs), func=f, question=name[7:] if kind == "rule" else None)


def _load_flow(m, catalog):
    from .strategist import Flow, Step
    steps = []
    for s in m.steps:
        p = None
        if catalog is not None:
            p = catalog.rules.get(s.name[7:]) if s.name.startswith("answer:") else catalog.parts.get(s.name)
        steps.append(Step(p if p is not None else _stub(s.name, s.kind, s.inputs), list(s.reasons)))
    return Flow(steps, {q: list(v) for q, v in m.per_question.items()}, dict(m.skipped),
                {q: list(v) for q, v in m.unresolved.items()}, dict(m.types), [list(b) for b in m.batches])


def load(cls, data, catalog=None):
    n = cls.__name__
    m = MODELS[n].model_validate(data)
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
        return Response({q: _load_result(r) for q, r in m.results.items()}, _load_flow(m.flow, catalog), tr, values, m.ms,
                        m.feasible, list(m.violations), catalog, [e.model_dump() for e in m.safeguards], m.model_outputs)
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
