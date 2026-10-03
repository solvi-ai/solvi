"""System.aask: `async def` parts are awaited, independent steps run concurrently, a failed hard check stops the rest,
timeouts become abstentions (safeguard "timeout"), sync parts run inline or in a thread, and the trace is the one `ask`
writes — the same hashes on every gallery case and on the examples, with storage, audit, batched decisions and
Cascade / Vote / Route."""
import asyncio
import copy
import inspect
import random
import threading
import time
from datetime import date
from pathlib import Path

import pytest

from solvi import Answer, Catalog, Question, System, testing
from solvi.runtime import MISSING, async_parts
from examples_loader import load

ROOT = Path(__file__).resolve().parents[1]


def hashes(res):
    return [r.hash for r in res.trace.records]


def answers(res):
    return {q: (r.answer, r.status, r.guard) for q, r in res.results.items()}


def same(system, state, names=None, speculate=(False, True)):
    """ask and aask (phased and speculative) on the same input: the same answers, records and hashes."""
    a = system.ask(copy.deepcopy(state), names, store=False)
    for sp in speculate:
        b = asyncio.run(system.aask(copy.deepcopy(state), names, store=False, speculate=sp))
        assert hashes(b) == hashes(a) and answers(b) == answers(a), (sp, answers(a), answers(b))
        assert b.trace.skipped == a.trace.skipped and set(b.values) == set(a.values)
    return a


def shop(delay=0.1, calls=None, slow=None):
    """A refund desk reading three services: independent lookups, a hard check on one of them."""
    cat = Catalog()
    calls = [] if calls is None else calls
    slow = slow or {}

    async def wait(name):
        try:
            await asyncio.sleep(slow.get(name, delay))
        except asyncio.CancelledError:
            calls.append(name + " cancelled")
            raise
        calls.append(name)

    @cat.fn
    async def customer(customer_id):
        await wait("customer")
        return {"id": customer_id, "tier": "gold" if customer_id.startswith("g") else "basic"}

    @cat.fn
    async def orders(customer_id):
        await wait("orders")
        return [120.0, 80.0]

    @cat.fn
    async def fraud_score(customer_id):
        await wait("fraud_score")
        return 0.9 if "bad" in customer_id else 0.1

    @cat.check(hard=True, then={"refund": "no"})
    def not_fraud(fraud_score):
        return fraud_score < 0.5

    @cat.fn
    def spent(orders):
        return sum(orders)

    @cat.rule("refund")
    def refund(customer, spent, not_fraud):
        return "yes" if customer["tier"] == "gold" or spent > 150 else "no"
    return cat, [Question("refund", "Refund?", Answer.yes_no())]


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


# --------------------------------------------------------------------------------------------------- async parts
def test_async_parts_are_awaited_concurrently_with_the_trace_of_ask():
    cat, qs = shop(0.2)
    s = System(cat, qs)
    assert sorted(async_parts(cat)) == ["customer", "fraud_score", "orders"] and s.is_async
    a, t_ask = timed(lambda: s.ask({"customer_id": "g-1"}))                 # sync: one call after another
    b, t_phased = timed(lambda: asyncio.run(s.aask({"customer_id": "g-1"})))
    c, t_spec = timed(lambda: asyncio.run(s.aask({"customer_id": "g-1"}, speculate=True)))
    assert a["refund"].answer == b["refund"].answer == c["refund"].answer == "yes"
    assert hashes(a) == hashes(b) == hashes(c)
    assert t_ask >= 0.6 and t_phased < 0.55 and t_spec < 0.38               # 3 calls; 2 phases; all at once
    for r in (a, b, c):
        assert r.trace.replay(s)["ok"] and r.trace.replay(s, flow=r.flow)["ok"]
    assert not System(*shop(0)).ask({"customer_id": "x"}).trace.replay(shop(0)[0])["mismatches"]


def test_a_failed_hard_check_stops_the_rest():
    calls = []
    cat, qs = shop(0.05, calls, slow={"customer": 1.0, "orders": 1.0})
    s = System(cat, qs)
    r, t = timed(lambda: asyncio.run(s.aask({"customer_id": "bad-7"})))
    assert calls == ["fraud_score"] and t < 0.5                            # phased: the lookups never start
    assert r["refund"].answer == "no" and r["refund"].status == "forced"
    calls.clear()
    r2, t = timed(lambda: asyncio.run(s.aask({"customer_id": "bad-7"}, speculate=True)))
    assert calls[0] == "fraud_score" and sorted(calls[1:]) == ["customer cancelled", "orders cancelled"] and t < 0.5
    calls.clear()
    a = s.ask({"customer_id": "bad-7"})
    assert calls == ["fraud_score"] and hashes(a) == hashes(r) == hashes(r2)
    skipped = ["customer", "orders", "spent", "answer:refund"]
    assert [n for n, _ in r2.trace.skipped] == [n for n, _ in a.trace.skipped] == skipped
    assert "not needed: hard check not_fraud failed" in r2.trace.skipped[0][1]


def test_speculative_steps_that_finished_anyway_are_not_recorded():
    calls = []
    cat, qs = shop(0.0, calls, slow={"fraud_score": 0.2, "customer": 0.0, "orders": 0.0})
    s = System(cat, qs)
    r = asyncio.run(s.aask({"customer_id": "bad-1"}, speculate=True))
    assert {"customer", "orders"} <= set(calls)                             # ran before the check failed ...
    assert "customer" not in r.values and "spent" not in r.values          # ... but ask would not have run them
    assert hashes(r) == hashes(s.ask({"customer_id": "bad-1"}))


def test_cancelling_aask_cancels_pending_calls():
    calls = []
    cat, qs = shop(1.0, calls)
    s = System(cat, qs)

    async def main():
        t = asyncio.ensure_future(s.aask({"customer_id": "g-1"}, speculate=True))
        await asyncio.sleep(0.05)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
    asyncio.run(main())
    assert sorted(calls) == ["customer cancelled", "fraud_score cancelled", "orders cancelled"]


def test_ask_runs_async_parts_even_inside_a_running_loop():
    cat, qs = shop(0.0)
    s = System(cat, qs)

    async def main():
        return s.ask({"customer_id": "g-1"})                              # a sync call from async code: a worker loop
    assert asyncio.run(main())["refund"].answer == "yes"
    assert s.ask({"customer_id": "b-1"})["refund"].answer == "yes"         # 200 spent


# --------------------------------------------------------------------------------------------------- timeouts
def test_a_timeout_is_an_abstention_with_its_safeguard():
    cat, qs = shop(0.0, slow={"orders": 1.0})
    s = System(cat, qs, timeout=0.1)
    r, t = timed(lambda: asyncio.run(s.aask({"customer_id": "b-1"})))
    assert t < 0.6
    res = r["refund"]
    assert res.status == "abstain" and res.guard == "timeout" and "orders" in res.why
    rec = next(x for x in r.trace.records if x.name == "orders")
    assert rec.value is MISSING and rec.error == "timed out after 0.1 s"
    assert {(e["kind"], e["fact"]) for e in r.safeguards} >= {("timeout", "orders")}
    assert s.stats["timeouts"] == 1 and "timed out" in s.safeguard_summary()
    assert "timed out" in str(r.audit("refund"))
    assert r.trace.replay(s)["ok"]                                         # a timeout is not re-run on replay
    ok = asyncio.run(s.aask({"customer_id": "b-1"}, timeout=5))            # per call: a longer limit
    assert ok["refund"].answer == "yes"


def test_a_parts_own_timeout_and_a_hard_check_that_times_out():
    cat = Catalog()

    @cat.fn(timeout=0.05)
    async def sanctions(name):
        await asyncio.sleep(1)
        return False

    @cat.check(hard=True, then={"pay": "no"})
    def clear(sanctions):
        return not sanctions

    @cat.rule("pay")
    def pay(clear):
        return "yes"
    s = System(cat, [Question("pay", "Pay?", Answer.yes_no())], timeout=10)
    r = asyncio.run(s.aask({"name": "Ann"}))
    assert r["pay"].status == "abstain" and r["pay"].guard == "hard_check" and "could not be evaluated" in r["pay"].why
    assert r.trace.replay(s)["ok"]


def test_a_producer_that_times_out_falls_back_to_the_next():
    cat = Catalog()

    @cat.fn(provides="rate", timeout=0.05)
    async def rate_live(currency):
        await asyncio.sleep(1)
        return 1.1

    @cat.fn(provides="rate")
    def rate_table(currency):
        return {"EUR": 1.08}[currency]

    @cat.rule("convert")
    def convert(rate):
        return "yes" if rate > 1 else "no"
    s = System(cat, [Question("convert", "Convert?", Answer.yes_no())])
    r = asyncio.run(s.aask({"currency": "EUR"}))
    rec = next(x for x in r.trace.records if x.name == "rate")
    assert r["convert"].answer == "yes" and rec.producer == "rate_table"
    assert rec.tried == [["rate_live", "timed out after 0.05 s"], ["rate_table", "accepted"]]
    assert {e["kind"] for e in r.safeguards} == {"timeout", "fallback"}
    assert r.trace.replay(s)["ok"]


# --------------------------------------------------------------------------------------------------- sync parts
def test_blocking_sync_parts_run_in_threads_plain_ones_inline():
    def build(blocking):
        cat = Catalog()
        seen = {}

        @cat.fn(blocking=blocking)
        def a(x):
            seen["a"] = threading.current_thread() is threading.main_thread()
            time.sleep(0.2)
            return 1

        @cat.fn(blocking=blocking)
        def b(x):
            time.sleep(0.2)
            return 2

        @cat.rule("q")
        def q(a, b):
            return "yes" if a + b == 3 else "no"
        return System(cat, [Question("q", "q", Answer.yes_no())]), seen
    s, seen = build(True)
    r, t = timed(lambda: asyncio.run(s.aask({"x": 0})))
    assert r["q"].answer == "yes" and t < 0.38 and seen["a"] is False and s.is_async
    s2, seen2 = build(None)
    r2, t2 = timed(lambda: asyncio.run(s2.aask({"x": 0})))
    assert t2 >= 0.4 and seen2["a"] is True and not s2.is_async
    assert hashes(r) == hashes(r2)


def test_a_blocking_part_times_out_too():
    cat = Catalog()

    @cat.fn(blocking=True, timeout=0.05)
    def slow(x):
        time.sleep(0.3)
        return 1

    @cat.rule("q")
    def q(slow):
        return "yes"
    r = asyncio.run(System(cat, [Question("q", "q", Answer.yes_no())]).aask({"x": 1}))
    assert r["q"].guard == "timeout"


def test_an_async_part_marked_blocking_is_awaited_and_its_timeout_holds():
    """async def + blocking=True + timeout=0.05 ran 0.4 s and answered: the worker thread only built the coroutine."""
    c = Catalog()

    @c.fn(timeout=0.05, blocking=True)
    async def slow(x):
        await asyncio.sleep(0.4)
        return x

    @c.rule("q")
    def q(slow):
        return "yes"
    t = time.perf_counter()
    r = asyncio.run(System(c, [Question("q", "?", Answer.yes_no())]).aask({"x": 1}))
    assert (r["q"].status, r["q"].guard) == ("abstain", "timeout") and time.perf_counter() - t < 0.3


def test_speculate_under_a_learned_order_is_not_silently_ignored():
    s = System(*shop(0), order="learned")
    with pytest.warns(UserWarning, match=r"speculate=True\) is ignored under a learned order"):
        assert asyncio.run(s.aask({"customer_id": "g-1"}, speculate=True))["refund"].answer == "yes"


def test_async_extract_with_a_source_keeps_its_quote():
    from solvi import Quote
    cat = Catalog()

    @cat.extract(source="notes")
    async def name(notes):
        await asyncio.sleep(0)
        i = notes.index("Ann")
        return Quote("Ann", i, i + 3)

    @cat.rule("known")
    def known(name):
        return "yes" if name == "Ann" else "no"
    s = System(cat, [Question("known", "Known?", Answer.yes_no())])
    r = same(s, {"notes": "call Ann today"})
    rec = next(x for x in r.trace.records if x.name == "name")
    assert rec.quote == (5, 8, "notes") and r["known"].answer == "yes"


# --------------------------------------------------------------------------------------------------- storage, models
def test_storage_audit_and_concurrent_asks(tmp_path):
    cat, qs = shop(0.05)
    s = System(cat, qs, storage=tmp_path / "d.db")

    async def main():
        return await asyncio.gather(*(s.aask({"customer_id": f"g-{i}"}) for i in range(8)))
    rs, t = timed(lambda: asyncio.run(main()))
    assert t < 0.8 and len({r.stored_id for r in rs}) == 8                 # 8 × 2 phases × 50 ms, concurrently
    assert s.storage.verify()["ok"] and s.stats["asks"] == 8
    back = s.storage.get(rs[3].stored_id)
    assert hashes(back) == hashes(rs[3]) and back.trace.replay(s)["ok"]


def test_cascade_vote_route_and_one_pass_decisions_under_aask():
    from solvi.multi import Cascade, Route, Vote
    from test_multi import CLEAR, HARD, _parts, _system
    for make in (lambda s, l_: Cascade([s, l_]), lambda s, l_: Vote([s, l_]), lambda s, l_: Route({"vip": l_}, default=s)):
        s, l_, _, _ = _parts()
        cat, system = _system(make(s, l_))
        for email in (CLEAR, HARD):
            r = same(system, {"email": email, "vip": True})
            assert r.trace.replay(cat)["ok"]
        cat.parts["team"].blocking = True                                   # a model call in a worker thread
        same(system, {"email": HARD, "vip": False})
    from test_decide_typed import ticket_system, v2
    cat, system = ticket_system(v2(act=False))
    r = same(system, {"email": "I am furious! The parcel is late, refund!"})
    assert r.flow.batches and next(x for x in r.trace.records if x.name == "answer:urgency").extra["pass"]["shared"]


# --------------------------------------------------------------------------------------------------- same hashes as ask
def test_gallery_cases_give_the_hashes_of_ask():
    n = 0
    for f in testing.find(ROOT / "gallery"):
        suite = testing.load(f)
        system = suite.system()
        for case in suite.cases:
            same(system, suite.state(case), case.get("ask"))
            n += 1
    assert n >= 163


def test_examples_give_the_hashes_of_ask():
    L = load("01_leave_request")
    s = System(L.cat, L.QUESTIONS)
    for st in (L.REQUEST, {**L.REQUEST, "balance": 5}, {**L.REQUEST, "start": date(2026, 12, 21),
                                                        "end": date(2026, 12, 23)}):
        same(s, st)
    for stem in ("03_invoices", "04_refunds"):
        E = load(stem)
        s = System(E.cat, E.QUESTIONS)
        rng = random.Random(3)
        for i in range(12):
            x = E.make(rng, i)
            same(s, x[0] if isinstance(x, tuple) else x, [q.name for q in E.QUESTIONS if q.name != "risk"])
    C = load("09_strategy_at_scale")
    C.SLOW.clear()
    s = System(C.cat, C.QUESTIONS)
    for st in (C.CLAIM, {**C.CLAIM, "incident_date": date(2024, 1, 1)}, {**C.CLAIM, "claimant_name": "Ivan Petrov"}):
        same(s, st)
        same(s, st, ["fast_track"])
    G = load("12_grounded_audit")
    same(System(G.build(), G.QUESTIONS), G.REQUEST)
    P = load("16_primitives")
    cat, qs = P.rule_catalog()
    for doc in P.CLAIMS.values():
        same(System(cat, qs), {"doc": doc})


# --------------------------------------------------------------------------------------------------- solvi serve
def test_serve_answers_an_async_system_with_aask(tmp_path):
    import io
    import json

    from solvi.serve import Service, run_builtin
    cat, qs = shop(0.05)
    s = System(cat, qs, storage=tmp_path / "d.db")
    svc = Service(s)
    assert svc.is_async
    d = svc.ask({"customer_id": "g-1"})                                    # sync caller: aask in a loop of its own
    assert d["results"]["refund"]["answer"] == "yes" and d["stored_id"]
    out = asyncio.run(svc.atool("refund", {"customer_id": "bad-1"}))
    assert out["status"] == "forced" and out["answer"] == "no"
    inp = io.StringIO(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                  "params": {"name": "refund", "arguments": {"customer_id": "g-2"}}}) + "\n")
    buf = io.StringIO()
    run_builtin(svc, inp, buf)
    assert json.loads(buf.getvalue())["result"]["structuredContent"]["answer"] == "yes"
    pytest.importorskip("fastapi")
    pytest.importorskip("fastapi.testclient")
    from fastapi.testclient import TestClient

    from solvi.serve import create_app
    app = create_app(system=s)
    assert all(inspect.iscoroutinefunction(r.endpoint) for r in app.routes if getattr(r, "path", "").startswith("/ask"))
    c = TestClient(app)
    r = c.post("/ask", json={"state": {"customer_id": "g-3"}}).json()
    assert r["results"]["refund"]["answer"] == "yes" and r["trace_hash"] == r["trace"]["records"][-1]["hash"]
    assert c.post("/ask/refund", json={"customer_id": "bad-3"}).json()["results"]["refund"]["status"] == "forced"
    assert c.post("/ask", json={"state": {"customer_id": "x"}, "questions": ["nope"]}).status_code == 404
    assert s.storage.verify()["ok"] and s.storage.verify()["count"] == 5
