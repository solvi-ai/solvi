"""solvi serve: the questions over HTTP (FastAPI TestClient), the input schemas and the OpenAPI document from the same
types, stored answers, POST /v1/systemone backed by a decider (round trip with the solvi.systemone client), and the MCP
server over stdio (the built-in JSON-RPC subset and the official SDK) in a subprocess."""
import io
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.cli import main
from solvi.decide import DecideModel
from solvi.serve import Service, question_inputs, sdk_available
from solvi.systemone import systemone

CATALOG = '''
from typing import Literal
from solvi import Answer, Catalog, Question, System

cat = Catalog()


@cat.fn
def words(text: str) -> set:
    return set(text.lower().split())


@cat.fn(provides="customer_id")
def customer_from_email(email: str) -> str:
    return email.split("@")[0]


@cat.fn(provides="customer_id")
def customer_from_phone(phone: str) -> str:
    return "c" + phone[-4:]


@cat.check(hard=True, then={"approve": False})
def known_customer(customer_id: str) -> bool:
    return customer_id != "blocked"


@cat.rule("approve")
def approve(amount: int, words: set, customer_id: str) -> bool:
    return amount < 100 and "urgent" not in words


@cat.rule("team")
def team(words: set) -> Literal["billing", "shipping"]:
    return "billing" if "charged" in words else "shipping"


QUESTIONS = [Question("approve", "Approve the refund?", checkpoints=["known_customer"]),
             Question("team", "Which team handles it?")]


def build():
    return System(cat, QUESTIONS)
'''


@pytest.fixture
def catalog_file(tmp_path):
    f = tmp_path / "shop.py"
    f.write_text(CATALOG)
    return f


@pytest.fixture
def system(catalog_file):
    ns = {}
    exec(compile(CATALOG, str(catalog_file), "exec"), ns)
    return ns["build"]()


def client(**kw):
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    return TestClient(create_app(**kw))


def test_a_question_reads_the_given_facts_of_its_flow(system):
    info = question_inputs(system, "approve")
    assert info["properties"] == ["amount", "email", "phone", "text"]
    assert info["required"] == ["amount", "email", "phone", "text"]   # the producers of customer_id read email and phone
    assert question_inputs(system, "team") == {"properties": ["text"], "required": ["text"]}
    system.questions["free"] = Question("free", "Anything?", Answer.yes_no(), uses=["words", "customer_id"])
    assert question_inputs(system, "free") == {"properties": ["email", "phone", "text"],
                                               "required": ["email", "phone", "text"]}
    system.questions["open"] = Question("open", "Anything?", Answer.yes_no())      # no rule, fit or uses: all optional
    assert question_inputs(system, "open") == {"properties": ["email", "phone", "text"], "required": []}


def test_ask_over_http_stores_the_answer_with_its_trace(system, tmp_path):
    store = tmp_path / "decisions.db"
    c = client(system=system, storage=str(store))
    r = c.post("/ask", json={"state": {"text": "I was charged twice", "amount": 50, "email": "ann@x.org", "phone": "1"}})
    assert r.status_code == 200
    d = r.json()
    assert d["results"]["approve"]["answer"] == "yes" and d["results"]["team"]["answer"] == "billing"
    assert d["stored_id"] and d["trace_hash"] == d["trace"]["records"][-1]["hash"]
    stored = system.storage.get(d["stored_id"])
    assert stored.trace.records[-1].hash == d["trace_hash"]
    assert stored.trace.replay(system.catalog, stored.flow)["ok"]
    one = c.post("/ask/team", json={"text": "my parcel is lost"}).json()
    assert list(one["results"]) == ["team"] and one["results"]["team"]["answer"] == "shipping"
    blocked = c.post("/ask", json={"state": {"text": "hi", "amount": 5, "email": "blocked@x", "phone": "1"}, "questions": ["approve"],
                                   "store": False}).json()
    assert blocked["results"]["approve"]["status"] == "forced" and blocked["stored_id"]    # storing: the server's policy
    assert system.storage.verify()["ok"] and system.storage.verify()["count"] == 3


def test_wrong_inputs_are_decided_by_solvi_and_wrong_requests_are_http_errors(system):
    c = client(system=system)
    d = c.post("/ask/approve", json={"text": "hello", "amount": "a lot", "email": "ann@x", "phone": "1"}).json()
    r = d["results"]["approve"]
    assert r["status"] == "abstain" and any(s["kind"] == "type_rejected" for s in d["safeguards"])
    assert c.post("/ask", json={"state": {"text": "x"}, "questions": ["nope"]}).status_code == 404
    assert c.post("/ask/team", json=["not", "a", "state"]).status_code == 422
    assert c.post("/v1/systemone", json={"state": "x", "questions": {}}).status_code == 404   # no decider on this server
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["questions"] == ["approve", "team"] and h["catalog"] == system.fingerprint()["catalog"]


def test_questions_and_the_openapi_document_come_from_the_same_types(system):
    c = client(system=system)
    qs = {q["name"]: q for q in c.get("/questions").json()}
    sch = qs["approve"]["input_schema"]
    assert sch["properties"]["amount"]["type"] == "integer" and sorted(sch["required"]) == ["amount", "email", "phone", "text"]
    assert qs["team"]["answer"]["options"] == ["billing", "shipping"]
    doc = c.get("/openapi.json").json()
    comps = doc["components"]["schemas"]
    body = doc["paths"]["/ask/approve"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert body == {"$ref": "#/components/schemas/ApproveInput"}
    assert comps["ApproveInput"]["properties"]["email"]["type"] == "string"
    out = doc["paths"]["/ask/team"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    res = comps[out.rsplit("/", 1)[1]]
    assert "stored_id" in res["properties"] and "trace_hash" in res["properties"]
    assert set(comps[res["properties"]["results"]["$ref"].rsplit("/", 1)[1]]["properties"]) == {"team"}
    all_results = comps["AskResponse"]["properties"]["results"]["$ref"].rsplit("/", 1)[1]
    assert set(comps[all_results]["properties"]) == {"approve", "team"}
    team = comps["Result_team"]["properties"]["answer"]
    assert {"billing", "shipping"} <= set(json.dumps(team).replace('"', " ").split())


def test_inputs_model_types_the_schema():
    from pydantic import BaseModel

    class Claim(BaseModel):
        amount: float
        note: str = ""
    cat = Catalog()

    @cat.rule("ok")
    def ok(amount, note) -> bool:
        return amount < 10 and "fraud" not in note
    s = System(cat, [Question("ok", "OK?")], inputs=Claim)
    c = client(system=s)
    sch = c.get("/questions").json()[0]["input_schema"]
    assert sch["properties"]["amount"]["type"] == "number"
    assert c.post("/ask/ok", json={"amount": "3", "note": "hi"}).json()["results"]["ok"]["answer"] == "yes"


# --- System One
KW = {"billing": ["charged", "refund"], "shipping": ["parcel", "delivery"], "high": ["urgent", "asap"],
      "yes": ["urgent"]}


class KeywordScorer:
    model_id = "test/keywords"

    def fingerprint(self):
        return "keywords-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = np.array([2.0 * sum(low.count(k) for k in KW.get(o, [])) + 0.1 * i for i, o in enumerate(it.options)])
            out.append(np.stack([z, z - 1.0], 1))
        return out


def decider():
    return DecideModel(KeywordScorer(), meta={"format": "test", "temperature": 1.0})


def opener_for(c):
    """urllib's urlopen, answered by a FastAPI TestClient."""
    def open_(req, timeout=None):
        from urllib.parse import urlsplit
        r = c.post(urlsplit(req.full_url).path, content=req.data, headers=dict(req.header_items()))
        assert r.status_code == 200, r.text
        return io.BytesIO(r.content)
    return open_


def test_system_one_round_trip_through_the_solvi_client():
    server_model = decider()
    c = client(decider=server_model, model_name="kev-latest")
    remote = systemone("http://solvi.local:8000", "kev-latest", opener=opener_for(c))
    teams = {"billing": "Charges, invoices, refunds", "shipping": "Delivery, parcels"}
    team = remote.decision("team", "Which team?", "email", teams)
    urgent = remote.decision("urgent", "Urgent?", "email", type=bool)
    for text in ("I was charged twice, refund please", "where is my parcel", "hello"):
        want = server_model.score(text, "Which team?", list(teams), teams, other=False, kind="choice")
        got = team.decide(text)
        assert got.value == max(want, key=want.get)
        for o in teams:
            assert got.probs[o] == pytest.approx(want[o], abs=1e-6)
    u = urgent.decide("urgent: the parcel is lost")
    assert u.value is True and u.probs["yes"] == pytest.approx(
        server_model.score("urgent: the parcel is lost", "Urgent?", ["yes", "no"], kind="noul")["yes"], abs=1e-6)
    both = remote.decide_pass("urgent, I was charged twice", [team, urgent])
    assert [x.value for x in both] == ["billing", True]


def test_system_one_answers_choice_noul_and_score_in_one_request():
    c = client(decider=decider(), model_name="kev-latest")
    r = c.post("/v1/systemone", json={"state": {"email": "urgent: charged twice"}, "model": "anything", "questions": {
        "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": None, "shipping": "Delivery"}},
        "urgent": {"type": "noul", "instructions": "Urgent?"},
        "priority": {"type": "score", "instructions": "Priority?", "criteria": {"low": None, "medium": None, "high": None}}}})
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["model"] == "kev-latest" and d["usage"]["questions"] == 3 and d["latency_ms"] >= 0
    a = d["answers"]
    assert a["team"]["type"] == "choice" and a["team"]["choice"] == "billing"
    assert sum(a["team"]["probabilities"].values()) == pytest.approx(1.0)
    assert a["urgent"] == {"type": "noul", "noul": pytest.approx(a["urgent"]["noul"])} and a["urgent"]["noul"] > 0.5
    p = a["priority"]
    assert p["legend"] == ["low", "medium", "high"] and list(p["probabilities"]) == p["legend"]
    assert p["score"] == pytest.approx(sum(i * p["probabilities"][o] for i, o in enumerate(p["legend"])))
    assert p["score"] > 1.0                                    # "urgent" pulls towards "high"
    bad = c.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "choice", "instructions": "?"}}})
    assert bad.status_code == 422
    assert c.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "span", "instructions": "?"}}}) \
        .status_code == 422
    assert c.get("/health").json()["decider"]["model"] == "kev-latest"
    assert c.post("/ask", json={"state": {}}).status_code == 404   # a decider-only server has no questions


def test_service_without_fastapi_answers_system_one_and_questions(system):
    svc = Service(system, decider())
    d = svc.systemone({"state": "parcel", "questions": {"t": {"type": "choice", "instructions": "Team?",
                                                               "criteria": {"billing": None, "shipping": None}}}})
    assert d["answers"]["t"]["choice"] == "shipping"
    out = svc.tool("team", {"text": "charged twice"})
    assert out["answer"] == "billing" and out["status"] == "ok" and out["trace_hash"]


# --- MCP over stdio, in a subprocess
def _mcp_session(catalog_file, impl, store=None):
    args = [sys.executable, "-m", "solvi", "serve", f"{catalog_file}:build", "--mcp", "--mcp-impl", impl]
    if store:
        args += ["--store", str(store)]
    msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                        "clientInfo": {"name": "test", "version": "0"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "approve", "arguments": {"text": "please refund", "amount": 20, "email": "ann@x", "phone": "1"}}},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
             "params": {"name": "approve", "arguments": {"text": "please refund", "amount": 20, "email": "blocked@x",
                                                     "phone": "1"}}}]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                         cwd=os.path.dirname(catalog_file))
    out = {}
    try:
        for m in msgs:
            p.stdin.write(json.dumps(m) + "\n")
            p.stdin.flush()
            if "id" in m:
                while True:
                    line = p.stdout.readline()
                    assert line, p.stderr.read()
                    r = json.loads(line)
                    if r.get("id") == m["id"]:
                        out[m["id"]] = r
                        break
    finally:
        p.stdin.close()
        p.wait(timeout=30)
    return out


@pytest.mark.parametrize("impl", ["builtin", "sdk"])
def test_mcp_server_each_question_is_a_tool(catalog_file, tmp_path, impl):
    if impl == "sdk" and not sdk_available():
        pytest.skip("the MCP SDK (mcp>=2) is not installed")
    out = _mcp_session(catalog_file, impl, tmp_path / "mcp.jsonl")
    assert out[1]["result"]["serverInfo"]["name"] == "solvi" and "tools" in out[1]["result"]["capabilities"]
    tools = {t["name"]: t for t in out[2]["result"]["tools"]}
    assert set(tools) == {"approve", "team", "ask_text"}
    assert tools["approve"]["inputSchema"]["properties"]["amount"]["type"] == "integer"
    ok, blocked = out[3]["result"], out[4]["result"]
    assert not ok.get("isError") and ok["structuredContent"]["answer"] == "yes"
    assert json.loads(ok["content"][0]["text"])["stored_id"] == ok["structuredContent"]["stored_id"]
    assert blocked["structuredContent"]["status"] == "forced" and blocked["structuredContent"]["answer"] == "no"
    lines = (tmp_path / "mcp.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_builtin_mcp_errors(system):
    svc = Service(system)
    from solvi.serve import run_builtin
    inp = io.StringIO("\n".join(json.dumps(m) for m in [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
        {"jsonrpc": "2.0", "id": 2, "method": "resources/list"},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "team", "arguments": {"text": 5}}},
        {"jsonrpc": "2.0", "id": 4, "method": "ping"}]) + "\nnot json\n")
    out = io.StringIO()
    run_builtin(svc, inp, out)
    rs = [json.loads(x) for x in out.getvalue().splitlines()]
    assert rs[0]["error"]["code"] == -32602 and rs[1]["error"]["code"] == -32601
    assert rs[2]["result"]["structuredContent"]["status"] == "abstain"         # a wrong-typed input: solvi abstains
    assert rs[3]["result"] == {} and rs[4]["error"]["code"] == -32700


def test_serve_usage_errors():
    with pytest.raises(SystemExit) as e:
        main(["serve"])
    assert e.value.code == 2


# --- a free text: POST /ask_text and the ask_text tool
def _shop():
    from test_textin import decider, shop
    from solvi.textin import TextIn
    _, s = shop()
    tin = TextIn(s, decider(), patterns={"order_id": r"[A-Z]-\d+"},
                 synonyms={"currency": {"EUR": ["euro", "euros", "€"], "RUB": ["rubles", "руб", "₽"]}})
    return s, tin


def test_ask_text_over_http_routes_reads_with_quotes_and_answers(tmp_path):
    s, tin = _shop()
    c = client(system=s, textin=tin, storage=str(tmp_path / "d.db"))
    text = "Hi, please refund order A-10457: I paid 1.5 million rubles on 12 September and it arrived broken."
    d = c.post("/ask_text", json={"text": text, "today": "2026-09-28"}).json()
    rd = d["read"]
    assert rd["question"] == "request_refund" and rd["missing"] == [] and rd["clarify"] is None
    amount = rd["fields"]["amount"]
    assert amount["status"] == "read" and amount["value"] == 1500000.0
    q, a, b = amount["quote"]
    assert text[a:b] == q == "1.5 million"
    assert d["results"]["request_refund"]["answer"] == "review" and d["stored_id"]
    assert d["trace_hash"] == d["trace"]["records"][-1]["hash"]
    stored = s.storage.get(d["stored_id"])
    assert any(r.kind == "textin" for r in stored.trace.records)
    missing = c.post("/ask_text", json={"text": "Refund order A-17 please, I paid 40 euros."}).json()
    assert missing["read"]["missing"] == ["purchase_date"] and "purchase date" in missing["read"]["clarify"]
    assert missing["results"]["request_refund"]["status"] == "abstain"
    unsure = c.post("/ask_text", json={"text": "hello there", "store": False}).json()
    assert unsure["read"]["question"] is None and unsure["read"]["escalated"] and unsure["stored_id"]   # stored anyway
    given = c.post("/ask_text", json={"text": "cancel A-5, it is urgent", "question": "cancel_order"}).json()
    assert given["read"]["question"] == "cancel_order" and given["results"]["cancel_order"]["answer"] == "cancelled"
    assert c.post("/ask_text", json={"text": "x", "question": "nope"}).status_code == 404
    assert c.post("/ask_text", json={"text": ""}).status_code == 422
    assert "/ask_text" in c.get("/openapi.json").json()["paths"]


def test_ask_text_returns_dates_it_read_as_iso_strings_like_the_values():
    import datetime as dt
    s, tin = _shop()
    text = "Hi, please refund order A-10457: I paid 1.5 million rubles on 12 September and it arrived broken."
    d = client(system=s, textin=tin).post("/ask_text", json={"text": text, "today": "2026-09-28"}).json()
    assert d["values"]["purchase_date"] == "2026-09-12"
    assert d["read"]["state"]["purchase_date"] == "2026-09-12"                 # not "datetime.date(2026, 9, 12)"
    assert d["read"]["fields"]["purchase_date"]["value"] == "2026-09-12"
    tin.today = dt.date(2026, 9, 28)
    read = tin.read(text).to_dict()
    assert read["state"] == {"order_id": "A-10457", "amount": 1500000.0, "currency": "RUB", "purchase_date": "2026-09-12"}
    assert json.loads(json.dumps(read)) == read


def test_ask_text_needs_a_decider_to_route_and_is_an_mcp_tool():
    s, _ = _shop()
    c = client(system=s)                                    # no decider, three entry points
    r = c.post("/ask_text", json={"text": "refund A-1"})
    assert r.status_code == 422 and "decider" in r.json()["detail"]
    ok = c.post("/ask_text", json={"text": "cancel A-5 urgent", "question": "cancel_order", "store": False}).json()
    assert ok["read"]["missing"] == ["order_id"] and ok["read"]["clarify"] == "Please tell me the order id."
    assert ok["results"]["cancel_order"]["status"] == "abstain"      # no pattern for the order id: not read, not guessed
    from solvi.serve import mcp_tools, run_builtin
    s2, tin = _shop()
    svc = Service(s2, textin=tin)
    tools = {t["name"]: t for t in mcp_tools(svc)}
    assert tools["ask_text"]["inputSchema"]["required"] == ["text"]
    inp = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "ask_text", "arguments": {"text": "Please cancel order A-12, urgent!"}}}) + "\n")
    out = io.StringIO()
    run_builtin(svc, inp, out)
    res = json.loads(out.getvalue())["result"]
    assert not res["isError"] and res["structuredContent"]["read"]["question"] == "cancel_order"
    assert res["structuredContent"]["results"]["cancel_order"]["answer"] == "cancelled"
