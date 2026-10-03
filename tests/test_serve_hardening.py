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


def slow_reader(seconds=0.3):
    """The text shop of test_serve with a TextIn whose read (the decider's passes) is slow and counts how many run at once."""
    from solvi.core.textin import TextIn
    from test_serve import _shop
    s, tin = _shop()
    seen = {"now": 0, "peak": 0, "reads": 0}
    count = threading.Lock()

    class Slow(TextIn):
        def read(self, text, question=None):
            with count:
                seen["now"] += 1
                seen["reads"] += 1
                seen["peak"] = max(seen["peak"], seen["now"])
            time.sleep(seconds)
            try:
                return super().read(text, question=question)
            finally:
                with count:
                    seen["now"] -= 1
    tin.__class__ = Slow
    return s, tin, seen


def test_ask_text_reads_the_text_inside_the_inflight_slot_and_the_lock():
    s, tin, seen = slow_reader()
    svc = Service(s, textin=tin, limits=Limits(max_inflight=1))
    out = []

    def go():
        try:
            out.append(svc.ask_text("cancel A-5, it is urgent")["read"]["question"])
        except Busy:
            out.append("busy")
    ts = [threading.Thread(target=go) for _ in range(6)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen["peak"] == 1 and seen["reads"] == 1                   # the five over the limit never reached the decider
    assert sorted(out) == ["busy"] * 5 + ["cancel_order"]
    with pytest.raises(Exception, match="non-empty"):                 # a refused request takes no slot
        svc.ask_text("")
    two = Service(s, textin=tin, limits=Limits(max_inflight=6))      # slots for all: still one pass at a time
    seen.update(peak=0, reads=0)
    ts = [threading.Thread(target=lambda: two.ask_text("cancel A-5, it is urgent")) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert seen["peak"] == 1 and seen["reads"] == 3


def test_ask_text_of_an_async_system_reads_off_the_event_loop():
    import asyncio
    s, tin, seen = slow_reader(0.4)
    svc = Service(s, textin=tin, limits=Limits(max_inflight=1))

    async def main():
        first = asyncio.ensure_future(svc.aask_text("cancel A-5, it is urgent"))
        t0 = time.perf_counter()
        await asyncio.sleep(0.05)                                     # the loop is free while the text is read
        waited = time.perf_counter() - t0
        with pytest.raises(Busy):
            await svc.aask_text("cancel A-5, it is urgent")
        return waited, (await first)["read"]["question"]
    waited, question = asyncio.run(main())
    assert waited < 0.3 and question == "cancel_order" and seen["reads"] == 1
