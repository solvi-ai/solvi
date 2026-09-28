"""OpenTelemetry export: one span per step and per answer under a root span per decision, with the trace's facts,
provenance, models, safeguards and hashes as attributes — through the SDK (in-memory exporter) and as OTLP/JSON."""
import json

import pytest

from solvi import JSONLStorage, System
from solvi.otel import export, spans, to_otlp_json

from test_report import STATE, build



@pytest.fixture
def sdk():                                             # the JSON tests below run without OpenTelemetry installed
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    exp = InMemorySpanExporter()
    prov = TracerProvider()
    prov.add_span_processor(SimpleSpanProcessor(exp))
    return prov.get_tracer("test"), exp


def test_sdk_export(sdk):
    tracer, exp = sdk
    res = build(hallucinate=True).ask(STATE)
    roots = export(res, tracer)
    got = exp.get_finished_spans()
    assert len(roots) == 1 and len(got) == 1 + len(res.trace.records) + len(res.results)
    root = next(s for s in got if s.name == "solvi.decision")
    assert root.parent is None and root.attributes["solvi.trace.init_hash"] == res.trace.init_hash
    assert root.attributes["solvi.catalog.fingerprint"] == res.trace.fingerprint["catalog"]
    kids = [s for s in got if s is not root]
    assert all(s.parent.span_id == root.context.span_id and s.context.trace_id == root.context.trace_id for s in kids)
    by = {s.attributes.get("solvi.fact") or s.attributes.get("solvi.question"): s for s in kids}
    v = by["vendor"].attributes
    assert v["solvi.provenance"] == "quoted" and v["solvi.producer"] == "vendor_regex"
    assert set(v["solvi.safeguard"]) == {"grounding", "fallback"} and v["solvi.hash"] == res.trace.records[0].hash
    assert any("not grounded" in t for t in v["solvi.tried"])
    r = by["risk"].attributes
    assert r["solvi.model.id"] == "test/risk" and r["solvi.model.type"] == "Classifier"
    rec = next(x for x in res.trace.records if x.name == "risk")
    assert r["solvi.model.fingerprint"] == rec.model["fp"] and json.loads(r["solvi.probs"]) == {"high": 0.8, "low": 0.2}
    assert r["solvi.provenance"] == "decided" and r["solvi.confidence"] == pytest.approx(0.8)
    a = by["pay"].attributes
    assert a["solvi.answer"] == "'review'" and a["solvi.status"] == "ok"
    for s in kids:                                     # inside the root, in order
        assert root.start_time <= s.start_time <= s.end_time <= root.end_time


def test_error_status_and_skipped(sdk):
    tracer, exp = sdk
    res = build().ask({**STATE, "blocked": ['Acme <Tools> & "Sons"'], "limit": "x"})
    export(res, tracer)
    got = exp.get_finished_spans()
    root = next(s for s in got if s.name == "solvi.decision")
    assert {e.attributes["solvi.fact"] for e in root.events} == {"total", "risk", "answer:pay"}
    ans = next(s for s in got if s.name == "solvi.answer pay")
    assert ans.attributes["solvi.status"] == "forced" and ans.attributes["solvi.safeguard"] == ("hard_check",)

    cat_res = build().ask({"doc": "no vendor here", "blocked": [], "limit": 1})
    exp.clear()
    export(cat_res, tracer)
    from opentelemetry.trace import StatusCode
    errs = [s for s in exp.get_finished_spans() if s.status.status_code == StatusCode.ERROR]
    assert errs and all(s.attributes.get("solvi.error") for s in errs)


def test_store_export_and_parent_context(sdk, tmp_path):
    tracer, exp = sdk
    store = JSONLStorage(tmp_path / "d.jsonl", clock=iter(range(100, 200, 10)).__next__)
    s = build()
    sys2 = System(s.catalog, s.questions.values(), storage=store)
    sys2.ask(STATE)
    sys2.ask({**STATE, "limit": 10})
    with tracer.start_as_current_span("request") as outer:
        roots = export(store, tracer)
    assert len(roots) == 2
    got = exp.get_finished_spans()
    decisions = [x for x in got if x.name == "solvi.decision"]
    assert all(d.parent.span_id == outer.context.span_id for d in decisions)
    assert sorted(d.end_time for d in decisions) == [100 * 10**9, 110 * 10**9]
    exp.clear()
    assert len(export(store, tracer, question="pay", answer="yes")) == 0


def test_otlp_json_shape():
    res = build(hallucinate=True).ask(STATE)
    d = to_otlp_json(res, service_name="billing", end_ns=2_000_000_000_000)
    json.dumps(d)
    rs = d["resourceSpans"][0]
    assert rs["resource"]["attributes"] == [{"key": "service.name", "value": {"stringValue": "billing"}}]
    sc = rs["scopeSpans"][0]
    assert sc["scope"]["name"] == "solvi"
    ss = sc["spans"]
    assert len(ss) == 1 + len(res.trace.records) + len(res.results)
    root = ss[0]
    assert root["name"] == "solvi.decision" and "parentSpanId" not in root
    assert len(root["traceId"]) == 32 and len(root["spanId"]) == 16 and root["kind"] == 1
    assert all(s["traceId"] == root["traceId"] and s["parentSpanId"] == root["spanId"] for s in ss[1:])
    assert len({s["spanId"] for s in ss}) == len(ss)
    assert root["endTimeUnixNano"] == "2000000000000" and all(isinstance(s["startTimeUnixNano"], str) for s in ss)
    step = next(s for s in ss if s["name"] == "solvi.fn risk")
    attrs = {a["key"]: a["value"] for a in step["attributes"]}
    assert attrs["solvi.model.id"] == {"stringValue": "test/risk"}
    assert attrs["solvi.step"]["intValue"].isdigit() and "doubleValue" in attrs["solvi.confidence"]
    assert attrs["solvi.inputs"] == {"arrayValue": {"values": [{"stringValue": "total"}]}}
    # the same decision exports the same ids
    assert to_otlp_json(res, end_ns=2_000_000_000_000)["resourceSpans"][0]["scopeSpans"][0]["spans"][3]["spanId"] == ss[3]["spanId"]


def test_otlp_json_error_status():
    res = build().ask({"doc": "no vendor here", "blocked": [], "limit": 1})
    ss = to_otlp_json(res)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    bad = [s for s in ss if s["status"]]
    assert bad and all(s["status"]["code"] == 2 and s["status"]["message"] for s in bad)
    assert spans(res)[0]["attributes"]["solvi.complete"] is False


def test_identical_unstored_decisions_get_distinct_ids():
    s = build()
    a, b = s.ask(STATE, store=False), s.ask(STATE, store=False)
    assert a.stored_id is None and a.trace.records[-1].hash == b.trace.records[-1].hash
    ia = to_otlp_json(a, end_ns=1)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    ib = to_otlp_json(b, end_ns=1)["resourceSpans"][0]["scopeSpans"][0]["spans"]
    assert ia[0]["traceId"] != ib[0]["traceId"] and not {x["spanId"] for x in ia} & {x["spanId"] for x in ib}
    # the same response keeps its ids; a stored decision's ids come from its stored id alone
    assert to_otlp_json(a, end_ns=1)["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"] == ia[0]["traceId"]
    a.stored_id = b.stored_id = "0123456789abcdef"
    del a._otel_nonce
    assert spans(a)[0]["trace_id"] == spans(b)[0]["trace_id"]
