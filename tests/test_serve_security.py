"""solvi serve, hardened: the bearer token, request size and JSON depth limits, the request timeout (async Systems through
System.aask's timeout), errors that never leak tracebacks or paths, CORS off by default, nothing loaded from request
data, and a decider that is never downloaded without --pull."""
import asyncio
import io
import json
import logging
import time

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.cli import main
from solvi.serve import Limits, Service, run_builtin, too_deep

SECRET = "/home/someone/secret/catalog.py"


def shop(slow=0.0, boom=False):
    cat = Catalog()

    @cat.rule("team")
    def team(text: str) -> bool:
        if slow:
            time.sleep(slow)
        if boom:
            raise RuntimeError(f"cannot open {SECRET}")
        return "charged" in text
    return System(cat, [Question("team", "Billing?", Answer.yes_no())])


def async_shop(delay):
    cat = Catalog()

    @cat.fn
    async def words(text: str) -> list:
        await asyncio.sleep(delay)
        return text.split()

    @cat.rule("team")
    def team(words: list) -> bool:
        return "charged" in words
    return System(cat, [Question("team", "Billing?", Answer.yes_no())])


def client(**kw):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    return TestClient(create_app(**kw), raise_server_exceptions=False)


def test_a_token_is_required_when_set_and_compared_in_constant_time():
    c = client(system=shop(), token="s3cret")
    assert c.get("/health").status_code == 401
    r = c.post("/ask/team", json={"text": "charged"})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    assert c.post("/ask/team", json={"text": "x"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.post("/ask/team", json={"text": "x"}, headers={"Authorization": "Basic s3cret"}).status_code == 401
    ok = c.post("/ask/team", json={"text": "charged"}, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200 and ok.json()["results"]["team"]["answer"] == "yes"
    assert client(system=shop()).post("/ask/team", json={"text": "x"}).status_code == 200     # no token: open


def test_request_size_and_depth_are_limited():
    c = client(system=shop(), limits=Limits(max_body=2000, max_depth=8))
    assert c.post("/ask/team", json={"text": "charged " * 10}).status_code == 200
    big = c.post("/ask/team", json={"text": "x" * 5000})
    assert big.status_code == 413 and "larger than 2000 bytes" in big.json()["detail"]
    deep = {"text": "x"}
    for _ in range(10):
        deep = {"a": deep}
    r = c.post("/ask/team", json=deep)
    assert r.status_code == 400 and "deeper than 8" in r.json()["detail"]
    hostile = c.post("/ask/team", content=b"[" * 900 + b"]" * 900, headers={"content-type": "application/json"})
    assert hostile.status_code == 400                                             # no RecursionError, no 500
    # unterminated: refused as deep (400) or as invalid JSON (422, the framework's own check) — never a 500
    cut = c.post("/ask/team", content=b"[" * 1500, headers={"content-type": "application/json"})
    assert cut.status_code in (400, 422)
    assert c.post("/ask/team", content=b"{not json", headers={"content-type": "application/json"}).status_code == 422
    assert too_deep([[[1]]], 2) and not too_deep([[[1]]], 3) and not too_deep({"a": 1, "b": [1, 2]}, 2)


def test_a_slow_sync_request_times_out_with_504():
    c = client(system=shop(slow=1.0), limits=Limits(timeout=0.2))
    r = c.post("/ask/team", json={"text": "charged"})
    assert r.status_code == 504 and "0.2 s" in r.json()["detail"]


def test_an_async_system_gets_the_timeout_as_aask_part_timeout_and_still_answers():
    s = async_shop(delay=5.0)
    svc = Service(s, limits=Limits(timeout=0.5))
    assert svc.is_async and svc.part_timeout == pytest.approx(0.4)
    c = client(system=s, limits=Limits(timeout=0.5))
    t0 = time.perf_counter()
    d = c.post("/ask/team", json={"text": "charged twice"}).json()
    assert time.perf_counter() - t0 < 3
    assert d["results"]["team"]["status"] == "abstain" and d["results"]["team"]["guard"] == "timeout"
    fast = client(system=async_shop(delay=0.0), limits=Limits(timeout=5))
    assert fast.post("/ask/team", json={"text": "charged twice"}).json()["results"]["team"]["answer"] == "yes"


def test_errors_never_leak_tracebacks_or_paths_and_are_logged(caplog, monkeypatch):
    s = shop()
    c = client(system=s)

    def broken(*a, **k):
        raise RuntimeError(f"cannot open {SECRET}")
    monkeypatch.setattr(s, "ask", broken)
    with caplog.at_level(logging.ERROR, logger="solvi.serve"):
        r = c.post("/ask/team", json={"text": "x"})
    assert r.status_code == 500
    body = r.text
    assert SECRET not in body and "Traceback" not in body and "RuntimeError" not in body
    incident = r.json()["detail"].split("incident ")[1].split(";")[0]
    assert any(incident in rec.getMessage() and rec.exc_info for rec in caplog.records)   # the server keeps the details
    rule_error = client(system=shop(boom=True)).post("/ask/team", json={"text": "x"}).json()
    assert rule_error["results"]["team"]["status"] == "abstain"   # a part's error is part of the decision (its trace)


def test_health_names_the_store_without_its_path(tmp_path):
    h = client(system=shop(), storage=str(tmp_path / "decisions.db")).get("/health").json()
    assert h["store"] == "decisions.db"


def test_cors_is_off_by_default():
    pre = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    r = client(system=shop()).options("/ask/team", headers=pre)
    assert "access-control-allow-origin" not in r.headers
    r = client(system=shop()).post("/ask/team", json={"text": "x"}, headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
    c = client(system=shop(), cors=["https://app.example"], token="t")
    ok = c.options("/ask/team", headers={**pre, "Origin": "https://app.example"})
    assert ok.headers["access-control-allow-origin"] == "https://app.example"      # a preflight needs no token
    assert "access-control-allow-origin" not in c.options("/ask/team", headers=pre).headers


def test_nothing_is_loaded_from_request_data(monkeypatch):
    import solvi.cli
    import solvi.models

    def refuse(*a, **k):
        raise AssertionError("loaded something named by a request")
    monkeypatch.setattr(solvi.cli, "load_object", refuse)
    monkeypatch.setattr(solvi.models, "load", refuse)
    from test_serve import decider
    c = client(system=shop(), decider=decider())
    r = c.post("/v1/systemone", json={"state": "os:system", "model": "os:system", "questions": {
        "q": {"type": "choice", "instructions": "solvi.cli:main", "criteria": {"a": None, "b": None}}}})
    assert r.status_code == 200
    assert c.post("/ask/team", json={"text": "subprocess:run"}).status_code == 200


def _builtin(svc, *lines):
    out = io.StringIO()
    run_builtin(svc, io.StringIO("".join(x + "\n" for x in lines)), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_builtin_mcp_limits_and_errors(caplog):
    svc = Service(shop(), limits=Limits(max_body=300, max_depth=6, timeout=5))
    deep = {"text": "x"}
    for _ in range(8):
        deep = {"a": deep}
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "team", "arguments": {"text": "charged"}}}
    rs = _builtin(svc, json.dumps({**call, "params": {"name": "team", "arguments": {"text": "x" * 400}}}),
                  json.dumps({**call, "id": 2, "params": {"name": "team", "arguments": deep}}),
                  "[" * 100 + "]" * 100,
                  json.dumps({**call, "id": 3, "params": {"name": "team", "arguments": ["not", "a", "state"]}}),
                  json.dumps({**call, "id": 4}))
    assert rs[0]["error"]["code"] == -32600 and "at most 300" in rs[0]["error"]["message"]
    assert rs[1]["error"]["code"] == -32600 and "deeper than 6" in rs[1]["error"]["message"]
    assert rs[2]["error"]["code"] == -32600
    assert rs[3]["result"]["isError"] and "JSON object" in rs[3]["result"]["structuredContent"]["error"]
    assert rs[4]["result"]["structuredContent"]["answer"] == "yes"                 # the server kept going
    boom = Service(shop(boom=True))
    boom.system.ask = lambda *a, **k: (_ for _ in ()).throw(RuntimeError(f"cannot open {SECRET}"))
    with caplog.at_level(logging.ERROR, logger="solvi.serve"):
        r = _builtin(boom, json.dumps(call))[0]["result"]
    assert r["isError"] and SECRET not in json.dumps(r) and "incident" in r["structuredContent"]["error"]
    slow = _builtin(Service(shop(slow=1.0), limits=Limits(timeout=0.2)), json.dumps(call))[0]["result"]
    assert slow["isError"] and "did not finish within 0.2 s" in slow["structuredContent"]["error"]


def test_serve_never_downloads_a_decider_without_pull(tmp_path, monkeypatch, capsys):
    import solvi.models
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    pulled = []

    def fake_pull(repo, backend="onnx", revision=None):
        pulled.append((repo, backend))
        raise solvi.models.ModelError("offline in this test")
    monkeypatch.setattr(solvi.models, "pull", fake_pull)
    with pytest.raises(SystemExit) as e:
        main(["serve", "--decider", "solvi-ai/solvi-base"])
    assert e.value.code == 2 and not pulled
    err = capsys.readouterr().err
    assert "is not downloaded" in err and "--pull" in err
    with pytest.raises(SystemExit) as e:
        main(["serve", "--decider", "solvi-ai/solvi-base", "--pull", "--backend", "torch"])
    assert e.value.code == 2 and pulled == [("solvi-ai/solvi-base", "torch")]


def test_the_system_one_client_speaks_http_only():
    from solvi.core.deciders.systemone import systemone
    for bad in ("file:///etc/passwd", "ftp://x", "/etc/passwd"):
        with pytest.raises(ValueError, match="http"):
            systemone(bad, "m")
    assert systemone("https://solvi.example", "m").model_id == "systemone:m"
    from solvi.core.deciders.llm import llm
    for bad in ("file:///etc/passwd", "ftp://x/v1"):
        with pytest.raises(ValueError, match="http"):
            llm(bad, "m")


# --- POST /ask_text and the ask_text tool go through the same guard
def test_ask_text_goes_through_the_token_limits_and_error_hiding(monkeypatch, caplog):
    from test_serve import _shop
    s, tin = _shop()
    c = client(system=s, textin=tin, token="t", limits=Limits(max_body=500))
    auth = {"Authorization": "Bearer t"}
    body = {"text": "cancel A-5, it is urgent", "question": "cancel_order", "store": False}
    assert c.post("/ask_text", json=body).status_code == 401
    assert c.post("/ask_text", json=body, headers=auth).json()["results"]["cancel_order"]["answer"] == "cancelled"
    assert c.post("/ask_text", json={**body, "text": "x" * 600}, headers=auth).status_code == 413
    bad_day = c.post("/ask_text", json={**body, "today": "yesterday"}, headers=auth)
    assert bad_day.status_code == 422 and "ISO date" in bad_day.json()["detail"]

    def broken(*a, **k):
        raise RuntimeError(f"cannot open {SECRET}")
    monkeypatch.setattr(s, "ask_text", broken)
    with caplog.at_level(logging.ERROR, logger="solvi.serve"):
        r = c.post("/ask_text", json=body, headers=auth)
        tool = _builtin(Service(s, textin=tin), json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "ask_text", "arguments": {"text": "cancel A-5", "question": "cancel_order"}}}))[0]["result"]
    assert r.status_code == 500 and SECRET not in r.text and "incident" in r.json()["detail"]
    assert tool["isError"] and SECRET not in json.dumps(tool) and "incident" in tool["structuredContent"]["error"]
    empty = _builtin(Service(s, textin=tin), json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "ask_text", "arguments": {"text": ""}}}))[0]["result"]
    assert empty["isError"] and "non-empty" in empty["structuredContent"]["error"]


def test_the_mcp_proxy_bounds_messages_and_hides_its_own_errors(tmp_path, monkeypatch):
    from test_agents_mcp import guard, rpc, run

    from solvi.agents.mcp import Proxy
    g = guard(tmp_path)
    init = rpc(1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
    big = rpc(2, "tools/call", {"name": "read_file", "arguments": {"path": "/work/" + "x" * 500}})

    def broken(self, params, ask=None):
        raise RuntimeError(f"cannot open {SECRET}")
    monkeypatch.setattr(Proxy, "call", broken)
    out, raw = run(g, [init, big, rpc(3, "tools/call", {"name": "read_file", "arguments": {"path": "/work/a"}}),
                       rpc(4, "ping")], limits=Limits(max_body=300))
    assert "at most 300" in raw
    assert out[3]["error"]["code"] == -32603 and "incident" in out[3]["error"]["message"] and SECRET not in raw
    assert out[4]["result"] == {}                                           # the proxy kept serving
