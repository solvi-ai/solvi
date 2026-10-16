"""Guard candidate (opt-in, the default is unchanged): `guard.session(allowed_tools=)` and
`check` / `call(allowed_tools=)` — a catalog tool outside the task's allowed tools is denied (`in_session_tools`)
by a small replayable system before the tool's own checks."""
import pytest

from solvi.solutions.guard import Guard

SYS = {"role": "system", "content": "You are a helpful assistant."}


def ctx(user, *tools):
    out = [SYS, {"role": "user", "content": user}]
    for i, t in enumerate(tools):
        out.append({"role": "tool", "tool_call_id": f"c{i}", "content": t})
    return out


def tools_guard():
    g = Guard()
    made = []

    @g.tool
    def read_inbox() -> str:
        """Read the inbox."""
        return "Bob wrote: lunch at noon?"

    @g.tool(ground=["to"])
    def send_email(to: str, body: str) -> str:
        made.append(to)
        return "sent"

    @g.tool
    def delete_all() -> str:
        made.append("deleted")
        return "gone"
    return g, made


def test_session_allowed_tools_denies_others_and_does_not_run_them():
    g, made = tools_guard()
    s = g.session(ctx("Read my inbox and mail bob@corp.example back."), allowed_tools=["read_inbox", "send_email"])
    d = s.call({"name": "delete_all", "arguments": {}})
    assert d.outcome == "deny" and not d.executed and made == []
    assert "not among this session's allowed tools" in d.reasons[0]
    assert d.failed == ["in_session_tools"]
    assert d.replay()["ok"]
    assert s.call({"name": "read_inbox", "arguments": {}}).outcome == "allow"
    assert s.call({"name": "send_email", "arguments": {"to": "bob@corp.example", "body": "ok"}}).outcome == "allow"
    assert made == ["bob@corp.example"]


def test_session_allowed_tools_keeps_the_other_checks():
    g, _ = tools_guard()
    s = g.session(ctx("Mail Bob back."), allowed_tools=["send_email"])
    assert s.check({"name": "send_email", "arguments": {"to": "eve@x.example", "body": "ok"}}).outcome == "deny"


def test_session_allowed_tools_names_must_be_in_the_catalog():
    g, _ = tools_guard()
    with pytest.raises(ValueError):
        g.session(allowed_tools=["send_emial"])


def test_session_without_allowed_tools_is_as_before():
    g, _ = tools_guard()
    s = g.session(ctx("Clean up."))
    assert s.check({"name": "delete_all", "arguments": {}}).outcome == "allow"


def test_session_allowed_tools_unknown_tool_is_still_unknown():
    g, _ = tools_guard()
    s = g.session(ctx("hi"), allowed_tools=["read_inbox"])
    d = s.check({"name": "nope", "arguments": {}})
    assert d.outcome == "deny" and "unknown tool" in d.reasons[0]


def test_check_and_call_take_allowed_tools():
    g, made = tools_guard()
    c = ctx("Clean up.")
    assert g.check({"name": "delete_all", "arguments": {}}, c, allowed_tools="read_inbox").outcome == "deny"
    d = g.call({"name": "delete_all", "arguments": {}}, c, allowed_tools=["read_inbox"])
    assert d.outcome == "deny" and made == []
    assert g.call({"name": "delete_all", "arguments": {}}, c, allowed_tools=["delete_all"]).outcome == "allow"
    assert made == ["deleted"]


def test_allowed_tools_async():
    import asyncio
    g, made = tools_guard()
    d = asyncio.run(g.acall({"name": "delete_all", "arguments": {}}, ctx("x"), allowed_tools=["read_inbox"]))
    assert d.outcome == "deny" and made == []
    s = g.session(ctx("x"), allowed_tools=["read_inbox"])
    assert asyncio.run(s.acall({"name": "delete_all", "arguments": {}})).outcome == "deny"
