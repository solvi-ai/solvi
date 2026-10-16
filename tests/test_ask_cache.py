"""System(cache=N): a pure, model-free catalog's decision served from a cache keyed by (fingerprints, facts); a stored
hit is a compact record naming the reference decision, and replay still re-runs and compares it."""
import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.store import JSONLStorage


def leave(calls, threshold=5, cache=8, storage=None):
    cat = Catalog()

    @cat.fn
    def days(start, end):
        calls.append((start, end))
        return end - start + 1

    @cat.check(hard=True, then={"approve": "reject"})
    def enough_balance(balance, days):
        return balance >= days

    @cat.rule("approve")
    def approve(days):
        return "approve" if days < threshold else "needs_manager"

    return System(cat, [Question("approve", "Approve?", Answer.choice(["approve", "needs_manager", "reject"]),
                                 requires=["enough_balance"])], cache=cache, storage=storage)


S1 = {"start": 1, "end": 3, "balance": 10}
S2 = {"start": 1, "end": 9, "balance": 10}


def test_a_repeated_input_is_served_from_the_cache():
    calls = []
    s = leave(calls)
    a, b = s.ask(dict(S1)), s.ask(dict(S1))
    assert len(calls) == 1 and a.cached is None and b.cached == {"of": None}
    assert b["approve"].answer == a["approve"].answer == "approve"
    assert b.trace.init_hash == a.trace.init_hash and [r.hash for r in b.trace.records] == [r.hash for r in a.trace.records]
    b["approve"].answer = "changed"                     # a served copy: changing it does not change the cache
    assert s.ask(dict(S1))["approve"].answer == "approve" and len(calls) == 1
    assert s.stats["asks"] == 3
    s.ask(dict(S2))
    assert len(calls) == 2


def test_off_by_default():
    calls = []
    s = leave(calls, cache=None)
    s.ask(S1), s.ask(S1)
    assert len(calls) == 2


def test_a_stored_hit_is_a_light_record_that_still_replays(tmp_path):
    calls = []
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = leave(calls, storage=store)
    a, b = s.ask(dict(S1)), s.ask(dict(S1))
    assert len(calls) == 1 and b.cached == {"of": a.stored_id}
    rec = store.record(b.stored_id)
    assert rec.get("record") == "compact" and rec["meta"]["cache_of"] == a.stored_id
    assert store.record(a.stored_id).get("record") != "compact"
    assert store.verify()["ok"]
    assert store.replay_all(s) == []
    assert store.rederive(b.stored_id, s)["approve"].answer == "approve"
    changed = leave([], threshold=2)                    # the catalog changed since: the light record no longer re-derives
    assert any(f["id"] == b.stored_id for f in store.replay_all(changed))


def test_an_unstored_reference_is_replaced_by_the_first_stored_hit(tmp_path):
    calls = []
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = leave(calls, storage=store)
    s.ask(dict(S1), store=False)
    b = s.ask(dict(S1))
    c = s.ask(dict(S1))
    assert len(calls) == 1
    assert store.record(b.stored_id).get("record") != "compact"      # nothing to point at: kept in full
    assert c.cached == {"of": b.stored_id}


def test_a_change_of_the_questions_or_the_catalog_misses():
    calls = []
    s = leave(calls)
    s.ask(S1)
    s.questions["approve"].min_confidence = 0.99         # the questions' fingerprint changes
    s.ask(S1)
    assert len(calls) == 2

    @s.catalog.fn
    def unrelated(balance):
        return balance
    s.ask(S1)
    assert len(calls) == 3


def test_the_cache_is_bounded():
    calls = []
    s = leave(calls, cache=2)
    for st in (S1, S2, {"start": 2, "end": 3, "balance": 10}, S1):
        s.ask(st)
    assert len(calls) == 4                               # S1 was the oldest: evicted


def test_a_catalog_with_a_model_refuses_the_cache():
    class M:
        model_id = "m"
    cat = Catalog()

    @cat.fn(model=M(), options=["a", "b"])
    def pick(x):
        return Decision("a", {"a": 0.9, "b": 0.1})

    @cat.rule("q")
    def q(pick) -> bool:
        return pick == "a"
    with pytest.raises(ValueError, match="model"):
        System(cat, [Question("q", "?", Answer.yes_no())], cache=8)
    with pytest.raises(ValueError, match="cache"):
        leave([], cache=0)


def test_other_orders_and_a_learning_system_bypass_it():
    calls = []
    s = leave(calls)
    s.ask(S1, early_exit=False), s.ask(S1, early_exit=False)
    assert len(calls) == 1                               # early_exit is part of the key
    s.ask(S1)
    assert len(calls) == 2
    s.learn = True                                       # a learning system observes every ask: no cache
    s.ask(S1)
    assert len(calls) == 3


# --- aask and ask_text share the cache, with the same rules
def run(coro):
    import asyncio
    return asyncio.run(coro)


def test_aask_is_served_from_the_cache_and_shares_it_with_ask():
    calls = []
    s = leave(calls)
    a, b = run(s.aask(dict(S1))), run(s.aask(dict(S1)))
    assert len(calls) == 1 and a.cached is None and b.cached == {"of": None}
    assert [r.hash for r in b.trace.records] == [r.hash for r in a.trace.records]
    c = s.ask(dict(S1))                                  # aask and ask give the same decision: one cache
    assert len(calls) == 1 and c.cached == {"of": None} and c["approve"].answer == "approve"
    run(s.aask(dict(S2)))                                # another input: a miss
    assert len(calls) == 2
    s.questions["approve"].min_confidence = 0.99         # the questions changed: a miss
    run(s.aask(dict(S2)))
    assert len(calls) == 3


def test_aask_off_by_default_and_a_stored_hit_replays(tmp_path):
    calls = []
    s = leave(calls, cache=None)
    run(s.aask(S1)), run(s.aask(S1))
    assert len(calls) == 2
    store = JSONLStorage(tmp_path / "d.jsonl")
    s = leave(calls := [], storage=store)
    a, b = run(s.aask(dict(S1))), run(s.aask(dict(S1)))
    assert len(calls) == 1 and b.cached == {"of": a.stored_id}
    assert store.record(b.stored_id)["meta"]["cache_of"] == a.stored_id
    assert store.verify()["ok"] and store.replay_all(s) == []


def test_abstentions_are_cached_as_ask_caches_them():
    calls = []
    s = leave(calls)
    st = {"start": 1, "end": 3}                          # no balance: the required check cannot run, the question abstains
    a, b = s.ask(dict(st)), s.ask(dict(st))
    assert a["approve"].answer is None and b["approve"].answer is None
    assert len(calls) == 1 and b.cached == {"of": None}
    c = run(s.aask(dict(st)))
    assert c["approve"].answer is None and len(calls) == 1 and c.cached == {"of": None}


def test_a_timed_out_aask_is_not_cached():
    import asyncio
    calls = []
    cat = Catalog()

    @cat.fn
    async def slow(x):
        calls.append(x)
        await asyncio.sleep(0.2 if len(calls) == 1 else 0)
        return x

    @cat.rule("q")
    def q(slow) -> bool:
        return slow > 0
    s = System(cat, [Question("q", "?", Answer.yes_no())], cache=8)
    a = run(s.aask({"x": 1}, timeout=0.05))
    assert a["q"].answer is None                          # timed out: abstains, and is not kept
    b = run(s.aask({"x": 1}, timeout=0.05))
    assert len(calls) == 2 and b.cached is None and b["q"].answer == "yes"
    c = run(s.aask({"x": 1}, timeout=0.05))
    assert len(calls) == 2 and c.cached == {"of": None}


def shop(cache=8, storage=None):
    from solvi.core.textin import TextIn
    calls = []
    cat = Catalog()

    @cat.fn
    def known(order_id: str) -> bool:
        calls.append(order_id)
        return order_id.startswith("A-")

    @cat.rule("cancel")
    def cancel(known: bool) -> str:
        return "cancelled" if known else "keep"
    s = System(cat, [Question("cancel", "Cancel an order", Answer.choice(["cancelled", "keep"]))], cache=cache,
               storage=storage)
    return s, TextIn(s, None, patterns={"order_id": r"[A-Z]-\d+"}), calls


def test_ask_text_is_served_from_the_cache(tmp_path):
    store = JSONLStorage(tmp_path / "d.jsonl")
    s, tin, calls = shop(storage=store)
    a = s.ask_text("cancel A-12 please", textin=tin, question="cancel")
    b = s.ask_text("cancel A-12 please", textin=tin, question="cancel")
    assert len(calls) == 1 and a.cached is None and b.cached == {"of": a.stored_id}
    assert b["cancel"].answer == a["cancel"].answer == "cancelled" and b.read.state == {"order_id": "A-12"}
    assert [r.hash for r in b.trace.records] == [r.hash for r in a.trace.records]
    s.ask_text("cancel A-12 now", textin=tin, question="cancel")      # another text (its hash is a given fact): a miss
    assert len(calls) == 2
    c = run(s.aask_text("cancel A-12 please", textin=tin, question="cancel"))
    assert len(calls) == 2 and c.cached == {"of": a.stored_id}
    assert store.verify()["ok"] and store.replay_all(s) == []          # replay re-runs the light records too
    s2, tin2, calls2 = shop(cache=None)
    s2.ask_text("cancel A-12 please", textin=tin2, question="cancel")
    s2.ask_text("cancel A-12 please", textin=tin2, question="cancel")
    assert len(calls2) == 2


def test_ask_text_misses_when_the_read_differs():
    from solvi.core.textin import TextIn
    s, tin, calls = shop()
    s.ask_text("cancel A-12 please", textin=tin, question="cancel")
    other = TextIn(s, None, patterns={"order_id": r"[A-Z]-\d"})       # same text, another reading of its field
    r = s.ask_text("cancel A-12 please", textin=other, question="cancel")
    assert len(calls) == 2 and r.cached is None and r.read.state == {"order_id": "A-1"}
