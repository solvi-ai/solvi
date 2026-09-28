"""Fixes before the 0.7 release in solvi serve: System One request limits, a busy server says so instead of queuing
threads, storing is the server's policy, a malformed MCP message never stops the built-in server, an empty token is a
configuration error."""
import io
import json
import threading
import time

import pytest

from solvi.cli import main
from solvi.serve import Busy, Limits, Service, run_builtin
from test_serve import decider
from test_serve_security import client, shop

CALL = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "team", "arguments": {"text": "charged"}}}


def one_q(i, n=2):
    return {"type": "choice", "instructions": f"Q{i}?", "criteria": {f"o{j}": None for j in range(n)}}


def test_system_one_requests_are_bounded():
    c = client(decider=decider(), limits=Limits(max_questions=3, max_options=4))
    ok = c.post("/v1/systemone", json={"state": "x", "questions": {f"q{i}": one_q(i) for i in range(3)}})
    assert ok.status_code == 200, ok.text
    many = c.post("/v1/systemone", json={"state": "x", "questions": {f"q{i}": one_q(i) for i in range(4)}})
    assert many.status_code == 422 and "at most 3 questions" in many.json()["detail"]
    wide = c.post("/v1/systemone", json={"state": "x", "questions": {"q": one_q(0, 5)}})
    assert wide.status_code == 422 and "at most 4 options" in wide.json()["detail"]


def test_a_busy_system_refuses_instead_of_queuing():
    svc = Service(shop(slow=0.6), limits=Limits(queue_timeout=0.05, max_inflight=2))
    t = threading.Thread(target=svc.ask, args=({"text": "charged"},))
    t.start()
    time.sleep(0.1)
    t0 = time.perf_counter()
    with pytest.raises(Busy, match="busy") as e:
        svc.ask({"text": "charged"})
    assert e.value.status == 503 and time.perf_counter() - t0 < 0.4
    t.join()
    assert svc.ask({"text": "charged"})["results"]["team"]["answer"] == "yes"     # free again
    one = Service(shop(slow=0.6), limits=Limits(queue_timeout=None, max_inflight=1))
    t = threading.Thread(target=one.ask, args=({"text": "charged"},))
    t.start()
    time.sleep(0.1)
    with pytest.raises(Busy, match="in flight"):
        one.ask({"text": "x"})
    t.join()
    c = client(system=shop(slow=0.6), limits=Limits(queue_timeout=0.05, timeout=5))
    res = {}
    th = threading.Thread(target=lambda: res.update(first=c.post("/ask/team", json={"text": "charged"})))
    th.start()
    time.sleep(0.15)
    assert c.post("/ask/team", json={"text": "charged"}).status_code == 503
    th.join()
    assert res["first"].status_code == 200


def test_storing_is_the_servers_policy(tmp_path):
    c = client(system=shop(), storage=str(tmp_path / "a.db"))
    r = c.post("/ask", json={"state": {"text": "charged"}, "store": False}).json()
    assert r["stored_id"] is not None                          # the client cannot opt out
    t = c.post("/ask_text", json={"text": "I was charged", "question": "team", "store": False}).json()
    assert t["stored_id"] is not None
    c2 = client(system=shop(), storage=str(tmp_path / "b.db"), allow_client_no_store=True)
    assert c2.post("/ask", json={"state": {"text": "charged"}, "store": False}).json()["stored_id"] is None
    assert c2.post("/ask", json={"state": {"text": "charged"}}).json()["stored_id"] is not None


def test_a_malformed_tools_call_is_an_error_and_the_server_keeps_going():
    svc = Service(shop())
    lines = [json.dumps({**CALL, "id": 1, "params": {"name": ["team"], "arguments": {}}}),
             json.dumps({**CALL, "id": 2, "params": {"name": {"x": 1}}}),
             json.dumps({"jsonrpc": "2.0", "id": 3, "method": "ping"}),
             json.dumps({**CALL, "id": 4})]
    out = io.StringIO()
    run_builtin(svc, io.StringIO("\n".join(lines) + "\n"), out)
    rs = {m["id"]: m for m in map(json.loads, out.getvalue().splitlines())}
    assert rs[1]["error"]["code"] == -32602 and rs[2]["error"]["code"] == -32602
    assert rs[3]["result"] == {} and rs[4]["result"]["structuredContent"]["answer"] == "yes"


SHOP = """
from solvi import Answer, Catalog, Question, System
cat = Catalog()

@cat.rule("team")
def team(text: str) -> bool:
    return "charged" in text
system = System(cat, [Question("team", "Billing?", Answer.yes_no())])
"""


def test_an_empty_token_is_a_configuration_error(tmp_path, capsys):
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    from solvi.serve import create_app
    for bad in ("", "  "):
        with pytest.raises(ValueError, match="empty"):
            create_app(system=shop(), token=bad)
    (tmp_path / "shop.py").write_text(SHOP)
    with pytest.raises(SystemExit) as e:
        main(["serve", f"{tmp_path / 'shop.py'}:system", "--token", ""])
    assert e.value.code == 2 and "empty token" in capsys.readouterr().err
