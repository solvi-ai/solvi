"""Fixes from the adversarial re-check before 0.7 outside the guard: exception texts in solvi serve's answers, the specs
of a hand-built TextRead, negations after a yes / no cue, and small edges (async max_inflight, MCP string arguments,
a failed forward)."""
import json

import pytest

from solvi import Answer, Catalog, Question, System

SECRET = "/srv/secret/models/risk.bin"


# ---------------------------------------------------------------------------------------------------- B7
def leaky(tmp_path=None):
    cat = Catalog()

    @cat.fn(provides="rate")
    def rate_remote(amount: float) -> float:
        raise ConnectionError(f"cannot open {SECRET}")

    @cat.fn(provides="rate")
    def rate_local(amount: float) -> float:
        raise OSError(f"{SECRET}: permission denied")

    @cat.fn
    def risk(amount: float) -> float:
        raise FileNotFoundError(f"{SECRET} not found")

    @cat.check(hard=True, then={"approve": "no"})
    def low_risk(risk) -> bool:
        return risk < 1

    @cat.rule("approve")
    def approve(amount) -> str:
        return "yes"

    @cat.rule("price")
    def price(amount, rate) -> str:
        return "ok"

    @cat.rule("limit")
    def limit(amount) -> str:
        raise RuntimeError(f"config at {SECRET} is broken")
    qs = [Question("approve", "Approve?", Answer.choice(["yes", "no"]), requires=["low_risk"]),
          Question("price", "Priced?", Answer.choice(["ok", "no"])),
          Question("limit", "Within limit?", Answer.choice(["ok", "no"]))]
    s = System(cat, qs)
    if tmp_path is not None:
        from solvi.storage import open_storage
        s.storage = open_storage(tmp_path / "t.jsonl", s)
    return s


def test_serve_answers_never_carry_exception_texts(tmp_path, caplog):
    from solvi.serve import Service
    svc = Service(leaky(tmp_path))
    with caplog.at_level("ERROR", logger="solvi.serve"):
        d = svc.ask({"amount": 5})
    body = json.dumps(d)
    assert SECRET not in body and "permission denied" not in body
    errs = {r["name"]: r["error"] for r in d["trace"]["records"] if r.get("error")}
    assert errs["risk"].startswith("FileNotFoundError (incident ")
    assert "RuntimeError (incident " in d["results"]["limit"]["why"]
    tried = [t for r in d["trace"]["records"] if r.get("tried") for t in r["tried"]]
    assert tried and all(SECRET not in t[1] for t in tried) and any("ConnectionError (incident" in t[1] for t in tried)
    assert SECRET in caplog.text                                   # the server log keeps the text
    stored = svc.system.storage.record(d["stored_id"])
    assert SECRET in json.dumps(stored)                            # and so does the stored trace
    for name in ("approve", "price", "limit"):
        assert SECRET not in json.dumps(svc.tool(name, {"amount": 5})), name


def test_serve_http_ask_hides_exception_texts():
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    c = TestClient(create_app(system=leaky()), raise_server_exceptions=False)
    r = c.post("/ask", json={"state": {"amount": 5}})
    assert r.status_code == 200 and SECRET not in r.text and "(incident " in r.text


def test_redact_leaves_answers_without_exceptions_alone():
    from solvi.serve import redact
    d = {"results": {"q": {"why": "hard check x is false"}},
         "trace": {"records": [{"name": "a", "error": "missing inputs: b"}, {"name": "c", "error": "timed out after 1s"}]}}
    assert redact(json.loads(json.dumps(d))) == d


# ---------------------------------------------------------------------------------------------------- B8
def orders():
    import datetime as dt
    cat = Catalog()

    @cat.rule("cancel_order")
    def cancel(urgent: bool, placed: dt.date) -> str:
        return "now" if urgent else "later"
    return System(cat, [Question("cancel_order", "Cancel the order?", Answer.choice(["now", "later"]))])


def forge_urgent(read, word, spec):
    import dataclasses

    from solvi.core import Quote
    from solvi.textin import SOURCE, FieldRead
    i = read.text.index(word)
    f = FieldRead("urgent", "read", True, Quote(word, i, i + len(word), SOURCE, 1.0), 1.0, True, "bool", spec)
    return dataclasses.replace(read, fields=dict(read.fields, urgent=f))


def test_ask_text_rebuilds_parser_specs_instead_of_trusting_the_read():
    from solvi.textin import TextIn, TextRead
    s = orders()
    text = "cancel A-12 placed 2026-09-01, banana"
    read = TextIn(s).read(text, question="cancel_order")
    assert read.fields["urgent"].status == "not_stated"
    forged = forge_urgent(read, "banana", {"cues": ["banana"]})
    res = s.ask_text(forged)
    f = res.read.fields["urgent"]
    assert f.status == "unparsed" and "not the field's own" in f.why and "urgent" not in res.trace.init
    assert res["cancel_order"].status == "abstain"
    bare = TextRead(forged.text, forged.question, forged.fields, forged.route)        # built by hand: no reader
    assert s.ask_text(bare).read.fields["urgent"].status == "unparsed"
    # a read made with the app's own cues goes through, with them, and only with them
    tin = TextIn(s, cues={"urgent": ["asap"]})
    ok = tin.read("cancel A-12 placed 2026-09-01, asap", question="cancel_order")
    assert ok.fields["urgent"].value is True
    assert s.ask_text(ok)["cancel_order"].answer == "now"
    assert s.ask_text(ok, textin=TextIn(s)).read.fields["urgent"].status == "unparsed"


def test_only_today_may_come_from_the_read():
    import dataclasses
    import datetime as dt

    from solvi.textin import TextIn
    s = orders()
    read = TextIn(s, today=dt.date(2026, 9, 28)).read("cancel A-12 placed 12 September, it is urgent",
                                                      question="cancel_order")
    assert read.fields["placed"].value == dt.date(2026, 9, 12)
    other = dataclasses.replace(read, fields=dict(read.fields, placed=dataclasses.replace(
        read.fields["placed"], spec={**read.fields["placed"].spec, "dayfirst": False})))
    assert s.ask_text(other).read.fields["placed"].status == "unparsed"
    bare = dataclasses.replace(read, reader=None)                                  # a TextIn without today= agrees
    assert s.ask_text(bare)["cancel_order"].answer == "now"


# ---------------------------------------------------------------------------------------------------- B9
@pytest.mark.parametrize("said, want", [
    ("Urgent: no", False), ("Is it urgent? No.", False), ("urgent = false", False), ("Urgent: yes", True),
    ("urgent? not at all", None), ("far from urgent", None), ("anything but urgent", None), ("less than urgent", None),
    ("it was urgent yesterday, not anymore", None), ("urgent-ish, not really", None),
    ("the refund is urgent but cancelling isn't", None), ("it is urgent", True)])
def test_a_negation_or_a_no_after_the_cue_is_read(said, want):
    from solvi.textin import TextIn
    s = orders()
    f = TextIn(s).read(f"Cancel the order placed 2026-09-01. {said}", question="cancel_order").fields["urgent"]
    if want is None:
        assert f.status == "unparsed" and f.value is None, (said, f)
    else:
        assert f.status == "read" and f.value is want, (said, f)
    res = s.ask_text(f"Cancel the order placed 2026-09-01. {said}", question="cancel_order")
    assert res.trace.replay(s)["ok"]
    if want is not True:
        assert res["cancel_order"].answer != "now", said


# ---------------------------------------------------------------------------------------------------- minor edges
def test_async_systems_respect_max_inflight():
    import asyncio

    from solvi.serve import Busy, Limits, Service
    cat = Catalog()
    gate = {}

    @cat.fn
    async def slow(text: str) -> str:
        await gate["go"].wait()
        return text

    @cat.rule("q")
    def q(slow) -> str:
        return "yes"
    svc = Service(System(cat, [Question("q", "?", Answer.choice(["yes", "no"]))]), limits=Limits(max_inflight=1))
    assert svc.is_async

    async def main():
        gate["go"] = asyncio.Event()
        first = asyncio.ensure_future(svc.aask({"text": "a"}))
        await asyncio.sleep(0.01)
        with pytest.raises(Busy):
            await svc.aask({"text": "b"})
        with pytest.raises(Busy):
            await svc.atool("q", {"text": "b"})
        gate["go"].set()
        assert (await first)["results"]["q"]["answer"] == "yes"
        assert (await svc.aask({"text": "c"}))["results"]["q"]["answer"] == "yes"     # the slot is free again
    asyncio.run(main())


def test_mcp_proxy_denies_arguments_given_as_a_json_string():
    from test_agents_hardening import SCHEMA, make_proxy

    from solvi.agents import Guard
    g = Guard()
    g.declare("delete")
    px, up = make_proxy(g, [{"name": "delete", "inputSchema": SCHEMA}])
    r = px.call({"name": "delete", "arguments": json.dumps({"n": 2})})
    assert r["isError"] and r["structuredContent"]["solvi"]["outcome"] == "deny" and up.calls == []
    assert "not a JSON object" in r["content"][0]["text"]
    assert not px.call({"name": "delete", "arguments": {"n": 2}}).get("isError") and len(up.calls) == 1
    assert px.call({"name": ["delete"], "arguments": {}})["isError"]


def test_mcp_proxy_records_a_forward_that_failed(tmp_path):
    from test_agents_hardening import SCHEMA, make_proxy

    from solvi.agents import Guard

    def boom(name, args):
        raise RuntimeError("the pipe broke")
    g = Guard(storage=tmp_path / "calls.jsonl")
    g.declare("delete")
    px, _ = make_proxy(g, [{"name": "delete", "inputSchema": SCHEMA}], reply=boom)
    with pytest.raises(RuntimeError):
        px.call({"name": "delete", "arguments": {"n": 2}})
    d = px.session.decisions[-1]
    assert d.outcome == "allow" and d.error == "forwarding failed: RuntimeError" and d.stored_id is not None
    [st] = list(g.storage.iter())
    assert st.meta["guard"]["error"] == "forwarding failed: RuntimeError"
    assert "forwarding failed" in px.session.context[-1][1]
