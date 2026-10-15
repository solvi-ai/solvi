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
