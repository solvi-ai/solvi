"""S334: an unresolved identical input cannot be used to re-probe the audit coin."""
import concurrent.futures
import copy

import pytest

from solvi.core.store import JSONLStorage
from test_stream_tickets import Clock, _desk, _gold, _records, _run


def test_open_input_returns_identical_ticket_under_fresh_request_ids_without_a_draw(tmp_path, monkeypatch):
    clock = Clock()
    desk, _, _ = _desk(JSONLStorage(tmp_path / "same.jsonl"), clock=clock)
    first = desk.decide({"case": 7}, request_id="first")
    n = desk.dispatcher.n
    monkeypatch.setattr(desk.dispatcher, "start", lambda *_: pytest.fail("a duplicate must not run System 1"))
    clock.t += 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        repeated = list(pool.map(lambda i: desk.decide({"case": 7}, request_id=f"retry-{i}"), range(8)))
    assert repeated == [first] * 8
    assert desk.dispatcher.n == n
    assert len(_records(desk.dispatcher.storage, "pending")) == 1
    assert len(desk.open()) == 1
    desk.close()


def test_input_order_and_nested_data_roundtrip_recover_retry_alias(tmp_path):
    path = tmp_path / "restart.jsonl"
    desk, _, _ = _desk(JSONLStorage(path))
    first = desk.decide({"case": 3, "extra": {"a": [1, True], "b": "x"}})
    alias = desk.decide({"extra": {"b": "x", "a": [1, True]}, "case": 3}, request_id="alias")
    assert alias == first
    alias["input"]["extra"]["a"].append("caller mutation")
    desk.close()
    desk, _, _ = _desk(JSONLStorage(path))
    assert desk.decide(first["input"], request_id="alias") == first
    assert desk.decide(copy.deepcopy(first["input"])) == first
    assert desk.dispatcher.n == 1
    desk.resolve(first["ticket"], _gold(3))
    assert desk.decide(first["input"], request_id="alias") == first
    assert desk.decide(first["input"])["ticket"] != first["ticket"]
    assert desk.dispatcher.n == 2
    desk.close()


def test_legacy_duplicate_tickets_resume_oldest_then_next(tmp_path):
    path = tmp_path / "legacy.jsonl"
    desk, _, _ = _desk(JSONLStorage(path), deduplicate_open=False)
    one, two = desk.decide({"case": 9}), desk.decide({"case": 9})
    assert one["ticket"] != two["ticket"]
    desk.close()
    desk, _, _ = _desk(JSONLStorage(path))
    assert desk.decide({"case": 9}) == one
    desk.resolve(one["ticket"], _gold(9))
    assert desk.decide({"case": 9}) == two
    assert desk.dispatcher.n == 2
    desk.close()


def test_reused_ticket_alias_survives_interruption_after_durable_intent(tmp_path, monkeypatch):
    path = tmp_path / "alias-crash.jsonl"
    store = JSONLStorage(path)
    desk, _, _ = _desk(store)
    first = desk.decide({"case": 6})
    append = store._append

    def interrupted(record):
        result = append(record)
        if record.get("kind") == "ticket_request":
            raise OSError("after alias intent was durably committed")
        return result

    monkeypatch.setattr(store, "_append", interrupted)
    with pytest.raises(OSError):
        desk.decide({"case": 6}, request_id="alias")
    assert desk._closed
    desk, _, _ = _desk(JSONLStorage(path))
    assert desk.decide({"case": 6}, request_id="alias") == first
    assert desk.dispatcher.n == 1
    with pytest.raises(ValueError, match="different state"):
        desk.decide({"case": 8}, request_id="alias")
    desk.close()


def test_pending_input_is_detached_from_callers_mutable_state(tmp_path):
    desk, _, _ = _desk(JSONLStorage(tmp_path / "mutation.jsonl"))
    state = {"case": 3, "extra": {"tags": ["original"]}}
    first = desk.decide(state)
    state["extra"]["tags"].append("mutated")
    repeated = desk.decide(first["input"])
    assert repeated == first
    desk.resolve(first["ticket"], _gold(3))
    assert _records(desk.dispatcher.storage, "dispatch")[-1][1]["s1"]["trace"]["init"] == first["input"]
    desk.close()


def test_expiry_removes_input_index_and_different_inputs_are_not_deduplicated(tmp_path):
    clock = Clock()
    desk, _, _ = _desk(JSONLStorage(tmp_path / "expiry.jsonl"), clock=clock, ttl=5)
    one = desk.decide({"case": 1})
    two = desk.decide({"case": 2})
    assert one["ticket"] != two["ticket"]
    clock.t += 5
    later = desk.decide({"case": 1})
    assert later["ticket"] not in {one["ticket"], two["ticket"]}
    assert desk.dispatcher.n == 3 and len(desk.open()) == 1
    assert len(_records(desk.dispatcher.storage, "pending_expired")) == 2
    desk.close()


def test_post_certification_audit_cannot_be_distinguished_by_same_input_reprobe(tmp_path, monkeypatch):
    desk, sp, _ = _desk(JSONLStorage(tmp_path / "audit.jsonl"))
    _run(desk, range(400))
    assert sp.threshold < 1.0
    audit = None
    for case in range(400, 800):
        result = desk.decide({"case": case})
        if result["status"] == "needs_verdict":
            if desk._open[result["ticket"]]["kind"] == "audit":
                audit = result
                break
            desk.resolve(result["ticket"], _gold(case))
    assert audit is not None, "the fixed coin must exercise a real post-certification audit"
    n, seen = desk.dispatcher.n, sp.n
    monkeypatch.setattr(desk.dispatcher, "start", lambda *_: pytest.fail("audit re-probe must not draw"))
    assert desk.call_tool("decide", {"state": audit["input"], "request_id": "probe"}) == audit
    assert (desk.dispatcher.n, sp.n) == (n, seen)
    desk.close()
