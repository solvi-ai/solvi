"""Guard.pause / resume — the stop button: while paused every call is denied before any other check and no tool runs;
pause and resume are events in the store's chain; a call denied while paused replays to the same denial."""
import asyncio
import threading

import pytest

from solvi.solutions.guard import Guard


def guard(storage=None):
    g = Guard(storage=storage)
    made = []

    @g.tool
    def send(to: str) -> str:
        made.append(to)
        return "sent"

    @g.policy("send", on_fail="escalate")
    def internal(to: str) -> bool:
        """Mail outside corp.example goes to a person."""
        return to.endswith("@corp.example")
    return g, made


CALL = {"name": "send", "arguments": {"to": "bob@corp.example"}}


def test_a_paused_guard_denies_every_call_and_runs_no_tool():
    g, made = guard()
    assert g.paused is None and g.call(CALL).outcome == "allow" and made == ["bob@corp.example"]
    assert g.pause("incident 42: odd payouts", by="alice") is None        # no storage: no event id
    assert g.paused["by"] == "alice" and g.paused["reason"] == "incident 42: odd payouts"
    for c in (CALL, {"name": "send", "arguments": {"to": "eve@x.example"}}, {"name": "nope", "arguments": {}}):
        d = g.call(c)
        assert d.outcome == "deny" and not d.executed                     # deny, not escalate — even for an unknown tool
        assert d.failed == ["guard_not_paused"]
        assert "paused by alice" in d.reasons[0] and "incident 42" in d.reasons[0]
        assert "stopped by a person" in d.advice()
        assert d.replay()["ok"]
    assert made == ["bob@corp.example"]
    g.resume(by="alice")
    assert g.paused is None and g.call(CALL).outcome == "allow" and len(made) == 2


def test_pause_needs_a_reason_and_resume_needs_a_pause():
    g, _ = guard()
    with pytest.raises(ValueError):
        g.pause("  ")
    with pytest.raises(ValueError):
        g.resume()


def test_async_and_session_calls_are_paused_too():
    g, made = guard()
    g.pause("maintenance")
    d = asyncio.run(g.acall(CALL))
    assert d.outcome == "deny" and "the guard is paused since" in d.reasons[0]
    assert g.session("mail bob@corp.example").call(CALL).outcome == "deny"
    assert g.check(CALL, allowed_tools=["send"]).outcome == "deny"
    assert made == []


def test_an_escalation_is_not_approved_while_paused(tmp_path):
    g, made = guard(tmp_path / "g.jsonl")
    e = g.call({"name": "send", "arguments": {"to": "eve@x.example"}})
    assert e.outcome == "escalate"
    g.pause("stop", by="ops")
    with pytest.raises(PermissionError, match="paused by ops"):
        g.resolve(e, approve=True, reviewer="alice")
    assert made == []
    g.resume(by="ops")
    assert g.resolve(e, approve=True, reviewer="alice").executed and made == ["eve@x.example"]


def test_a_resume_racing_a_pause_is_written_after_it(tmp_path, monkeypatch):
    """pause() writes its event under the pause lock, as resume() does: a resume that starts while the pause's event is
    still being written waits for it, so the chain reads pause → resume and the guard ends running."""
    g, _ = guard(tmp_path / "g.jsonl")
    save, entered, gate = g.storage.save, threading.Event(), threading.Event()

    def slow_save(res, meta=None, **kw):
        if (meta or {}).get("guard", {}).get("event") == "pause":
            entered.set()
            gate.wait(5)                                   # the pause's write is in flight
        return save(res, meta=meta, **kw)
    monkeypatch.setattr(g.storage, "save", slow_save)
    pauser = threading.Thread(target=g.pause, args=("incident 7",), kwargs={"by": "alice"})
    pauser.start()
    assert entered.wait(5)
    resumer = threading.Thread(target=g.resume, kwargs={"by": "bob"})
    resumer.start()
    resumer.join(0.3)
    assert resumer.is_alive()                              # blocked on the lock until the pause is written
    gate.set()
    pauser.join(5)
    resumer.join(5)
    assert not pauser.is_alive() and not resumer.is_alive()
    ev = [(s.meta or {}).get("guard", {}).get("event") for s in g.storage.iter()]
    assert ev == ["pause", "resume"] and g.paused is None and g.storage.verify()["ok"]


def test_pause_and_resume_are_events_in_the_chain_and_paused_calls_replay(tmp_path):
    g, _ = guard(tmp_path / "g.jsonl")
    g.call(CALL)
    pid = g.pause("incident 42", by="alice")
    d = g.call(CALL)
    rid = g.resume(by="bob")
    ev = [(s.meta or {}).get("guard") for s in g.storage.iter()]
    assert [e.get("event") for e in ev] == [None, "pause", None, "resume"]
    assert ev[1]["by"] == "alice" and ev[1]["reason"] == "incident 42" and ev[1]["at"].endswith("Z")
    assert ev[3]["by"] == "bob" and ev[3]["reason"] == "incident 42"
    assert g.storage.record(pid)["seq"] < g.storage.record(d.stored_id)["seq"] < g.storage.record(rid)["seq"]
    assert ev[2]["outcome"] == "deny" and "paused by alice" in ev[2]["reasons"][0]
    assert g.replay(d.stored_id)["ok"] and g.replay(pid)["ok"] and g.replay(rid)["ok"]
    assert g.replay_all() == []
    assert g.storage.verify()["ok"]
    # a new Guard over the same store (a restart) runs again: the pause lives in memory, the chain keeps it
    g2, _ = guard(tmp_path / "g.jsonl")
    assert g2.paused is None and g2.replay(d.stored_id)["ok"]


def test_a_scope_denial_replays_from_the_store(tmp_path):
    g, _ = guard(tmp_path / "g.jsonl")

    @g.tool
    def wipe() -> str:
        return "gone"
    d = g.call({"name": "wipe", "arguments": {}}, allowed_tools=["send"])
    assert d.outcome == "deny" and g.replay(d.stored_id)["ok"]


def test_a_pause_takes_effect_for_every_call_that_starts_after_it():
    g, made = guard()
    out, lock = [], threading.Lock()
    paused = threading.Event()

    def worker():
        for _ in range(30):
            after = paused.is_set()                   # read before the call starts
            d = g.call(CALL)
            with lock:
                out.append((after, d.outcome))
    ts = [threading.Thread(target=worker) for _ in range(4)]
    for t in ts:
        t.start()
    g.pause("now")
    paused.set()
    for t in ts:
        t.join()
    assert out and all(o == "deny" for after, o in out if after)
    assert len(made) == sum(1 for _, o in out if o == "allow")
