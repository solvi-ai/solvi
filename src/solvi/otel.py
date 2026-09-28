"""Traces as OpenTelemetry spans: one decision → a root span "solvi.decision" with one child span per step of the trace
(and one per answer), so solvi decisions show up next to the rest of a service's traces (Jaeger, Tempo, Honeycomb, …).

    export(res_or_store, tracer=None, **filters)   # through the OpenTelemetry API (the `otel` extra: opentelemetry-api/sdk)
    to_otlp_json(res_or_store, **filters)          # OTLP/JSON (ExportTraceServiceRequest) as a dict, without OpenTelemetry

Attributes of a step span (only those that apply): solvi.step, solvi.kind, solvi.fact, solvi.provenance, solvi.value (a
short repr), solvi.confidence, solvi.error, solvi.producer, solvi.tried, solvi.quote.source / .start / .end,
solvi.model.type / .id / .fingerprint, solvi.probs (JSON), solvi.safeguard (the kinds that fired on this fact),
solvi.inputs (the facts it read), solvi.hash and solvi.prev (the hash chain). An answer span: solvi.question,
solvi.answer, solvi.status, solvi.confidence, solvi.why, solvi.guard, solvi.provenance, solvi.source, solvi.safeguard.
The root: solvi.questions, solvi.trace.init_hash / .head, solvi.catalog.fingerprint, solvi.stored_id, solvi.ms, and an
event per step skipped at run time.

A step that failed or whose output was rejected has status ERROR with the reason; an abstention is not an error (its span
says status abstain and the guard). Times: a trace records each step's run time, not its start, so the step spans are laid
end to end from the decision's start — their durations are measured, their start times are not (and parallel steps are
shown one after another). A stored decision starts at its stored time minus its run time; a fresh response ends when it is
exported, unless `end_ns=` is given. In the OTLP JSON the ids are derived from the trace's hashes (the same decision
exports the same ids); through the API the SDK assigns them."""
from __future__ import annotations

import hashlib
import json
import time

SCOPE = "solvi"


def _short(v, n=200):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _responses(obj, filters):
    """A Response, a TraceStorage (every stored decision matching `filters`, as (response, stored time)) or a list."""
    if hasattr(obj, "trace") and hasattr(obj, "results"):
        return [(obj, None)]
    if hasattr(obj, "query") and hasattr(obj, "iter"):
        return [(s.response(), s.time) for s in (obj.query(**filters) if filters else obj.iter())]
    return [x for o in obj for x in _responses(o, filters)]


def _hid(*parts, n=16):
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()[:n]


def spans(res, end_ns=None, stored_time=None):
    """One decision as neutral span dicts (the root first): {"name", "span_id", "parent", "trace_id", "start", "end",
    "attributes", "status": None or ("ERROR", message), "events": [(name, time, attributes)]}."""
    from .runtime import MISSING
    tr = res.trace
    head = tr.records[-1].hash if tr.records else tr.init_hash
    trace_id = _hid(tr.init_hash, head, getattr(res, "stored_id", None), n=32)
    total = max(int(round(float(res.ms) * 1e6)), 1)
    if end_ns is None:
        end_ns = int(stored_time * 1e9) if stored_time is not None else time.time_ns()
    start = end_ns - total
    root_id = _hid(trace_id, "root")
    fp = getattr(tr, "fingerprint", None) or {}
    events = getattr(res, "safeguards", None) or []
    by_fact, by_q = {}, {}
    for e in events:
        by_fact.setdefault(e["fact"], []).append(e["kind"])
        for q in e.get("questions") or ():
            by_q.setdefault(q, []).append(e["kind"])
    root = {"name": "solvi.decision", "span_id": root_id, "parent": None, "trace_id": trace_id, "start": start,
            "end": end_ns, "status": None,
            "attributes": {"solvi.questions": list(res.results), "solvi.trace.init_hash": tr.init_hash,
                           "solvi.trace.head": head, "solvi.trace.steps": len(tr.records),
                           "solvi.catalog.fingerprint": fp.get("catalog"), "solvi.questions.fingerprint": fp.get("questions"),
                           "solvi.stored_id": getattr(res, "stored_id", None), "solvi.ms": float(res.ms),
                           "solvi.confidence": float(res.confidence), "solvi.complete": bool(res.complete),
                           "solvi.model_outputs": int(getattr(res, "model_outputs", 0) or 0)},
            "events": [("solvi.skipped", start, {"solvi.fact": n, "solvi.why": w}) for n, w in tr.skipped]}
    out = [root]
    t = start
    for r in tr.records:
        dur = max(int(round(float(tr.timings.get(r.name, 0.0)) * 1e6)), 0)
        a = {"solvi.step": r.step, "solvi.kind": r.kind, "solvi.fact": r.name, "solvi.provenance": r.origin,
             "solvi.value": None if r.value is MISSING else _short(r.value), "solvi.confidence": float(r.confidence),
             "solvi.error": r.error, "solvi.producer": r.producer, "solvi.inputs": sorted(r.inputs),
             "solvi.hash": r.hash, "solvi.prev": r.prev,
             "solvi.safeguard": sorted(set(by_fact.get(r.name, [])))}
        if r.tried:
            a["solvi.tried"] = [f"{n}: {w}" for n, w in r.tried]
        if r.quote:
            a["solvi.quote.start"], a["solvi.quote.end"], a["solvi.quote.source"] = int(r.quote[0]), int(r.quote[1]), str(r.quote[2])
        if r.model:
            a["solvi.model.type"], a["solvi.model.id"], a["solvi.model.fingerprint"] = (str(r.model.get("type")),
                                                                                       str(r.model.get("id")), str(r.model.get("fp")))
        if r.probs:
            a["solvi.probs"] = json.dumps({str(k): round(float(v), 6) for k, v in r.probs.items()}, ensure_ascii=False)
        out.append({"name": f"solvi.{r.kind} {r.name}", "span_id": _hid(trace_id, "step", r.hash), "parent": root_id,
                    "trace_id": trace_id, "start": t, "end": min(t + dur, end_ns), "attributes": a,
                    "status": ("ERROR", r.error) if r.error else None, "events": []})
        t = min(t + dur, end_ns)
    for q, x in res.results.items():
        from .primitives import fmt
        a = {"solvi.question": q, "solvi.answer": None if x.answer is None else fmt(x.answer, x.kind, x.extra),
             "solvi.status": x.status, "solvi.confidence": float(x.confidence), "solvi.why": x.why, "solvi.guard": x.guard,
             "solvi.provenance": x.provenance, "solvi.source": x.source,
             "solvi.safeguard": sorted(set(by_q.get(q, [])))}
        out.append({"name": f"solvi.answer {q}", "span_id": _hid(trace_id, "answer", q), "parent": root_id,
                    "trace_id": trace_id, "start": t, "end": end_ns, "attributes": a, "status": None, "events": []})
    for s in out:                                     # attributes: primitives or lists of strings, never None / empty
        s["attributes"] = {k: v for k, v in s["attributes"].items() if v is not None and v != []}
    return out


# ---------------------------------------------------------------- through the OpenTelemetry API
def export(res_or_store, tracer=None, end_ns=None, **filters):
    """Export decisions as OpenTelemetry spans: a Response, a TraceStorage (every stored decision, or those matching the
    query `filters`) or a list of them. tracer: an opentelemetry Tracer (default: trace.get_tracer("solvi") of the global
    provider). Each root span is a child of the span current in the caller's context, if any. → the root spans.
    Needs the `otel` extra (pip install "solvi[otel]"); without it, use to_otlp_json."""
    try:
        from opentelemetry import trace
        from opentelemetry.trace import Status, StatusCode
    except ImportError as e:                          # pragma: no cover - the extra is installed in the tests
        raise ImportError('solvi.otel.export needs OpenTelemetry: pip install "solvi[otel]" (or use to_otlp_json)') from e
    from . import __version__
    tracer = tracer if tracer is not None else trace.get_tracer(SCOPE, __version__)
    roots = []
    for res, stored in _responses(res_or_store, filters):
        made = {}
        ss = spans(res, end_ns, stored)
        for s in ss:
            ctx = trace.set_span_in_context(made[s["parent"]]) if s["parent"] else None
            sp = tracer.start_span(s["name"], context=ctx, attributes=s["attributes"], start_time=s["start"])
            for name, t, attrs in s["events"]:
                sp.add_event(name, attrs, timestamp=t)
            if s["status"]:
                sp.set_status(Status(StatusCode.ERROR, s["status"][1]))
            made[s["span_id"]] = sp
        for s in ss[1:]:
            made[s["span_id"]].end(end_time=s["end"])
        made[ss[0]["span_id"]].end(end_time=ss[0]["end"])
        roots.append(made[ss[0]["span_id"]])
    return roots


# ---------------------------------------------------------------- OTLP JSON without OpenTelemetry
def _any(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}                   # OTLP/JSON: 64-bit integers as strings
    if isinstance(v, float):
        return {"doubleValue": v}
    if isinstance(v, (list, tuple)):
        return {"arrayValue": {"values": [_any(x) for x in v]}}
    return {"stringValue": str(v)}


def _attrs(d):
    return [{"key": k, "value": _any(v)} for k, v in d.items()]


def to_otlp_json(res_or_store, service_name="solvi", end_ns=None, **filters):
    """Decisions as OTLP/JSON — the body of an ExportTraceServiceRequest (POST it to a collector's /v1/traces with
    Content-Type application/json) — as a dict; no OpenTelemetry package needed. The same spans as export(); trace and
    span ids are derived from the traces' hashes."""
    from . import __version__
    out = []
    for res, stored in _responses(res_or_store, filters):
        for s in spans(res, end_ns, stored):
            span = {"traceId": s["trace_id"], "spanId": s["span_id"], "name": s["name"], "kind": 1,   # SPAN_KIND_INTERNAL
                    "startTimeUnixNano": str(s["start"]), "endTimeUnixNano": str(s["end"]),
                    "attributes": _attrs(s["attributes"]),
                    "status": {"code": 2, "message": s["status"][1]} if s["status"] else {}}
            if s["parent"]:
                span["parentSpanId"] = s["parent"]
            if s["events"]:
                span["events"] = [{"name": n, "timeUnixNano": str(t), "attributes": _attrs(a)} for n, t, a in s["events"]]
            out.append(span)
    return {"resourceSpans": [{"resource": {"attributes": _attrs({"service.name": service_name})},
                               "scopeSpans": [{"scope": {"name": SCOPE, "version": __version__}, "spans": out}]}]}
