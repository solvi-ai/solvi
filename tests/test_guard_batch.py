import copy
import pytest
from pydantic import BaseModel
from solvi.solutions.guard import Guard


def book(*, once=False, fail=None):
    made = []
    g = Guard(fact_names={"spent": float})

    @g.tool(once=once)
    def expense(amount: float) -> str:
        made.append(amount)
        if amount == fail:
            raise RuntimeError("bank failed")
        return "created"

    @g.policy("expense")
    def budget(amount: float, spent: float) -> bool:
        return amount + spent <= 100

    @g.policy("expense")
    def invoice(amount: float) -> bool:
        return amount != 60

    return g, g.session(facts={"spent": 0.0}), made


def calls(*values):
    return [{"name": "expense", "arguments": {"amount": v}} for v in values]


def advance(facts, d):
    return {**facts, "spent": facts["spent"] + d.arguments["amount"]}


def test_invalid_last_executes_nothing_and_check_many_is_session_pure():
    g, s, made = book()
    before = (copy.deepcopy(s.facts), list(s.context), list(s.made), list(s.decisions))
    out = s.check_many(calls(40, 40, 60))
    assert not out["allowed"] and len(out["decisions"]) == 3 and out["executed"] == []
    assert (s.facts, s.context, s.made, s.decisions) == before and made == []
    out = s.call_many(calls(40, 40, 60))
    assert not out["allowed"] and made == [] and s.made == []


def test_cumulative_advance_and_exception_do_not_mutate_live_session():
    g, s, made = book()
    out = s.call_many(calls(40, 40, 40), advance=advance)
    assert not out["allowed"] and made == [] and s.facts == {"spent": 0.0}

    def bad(facts, d):
        facts["spent"] = 999
        raise ValueError("invalid advance")

    with pytest.raises(ValueError, match="invalid advance"):
        s.call_many(calls(10), advance=bad)
    with pytest.raises(TypeError):
        s.call_many(calls(10), advance=lambda f, d: None)
    assert made == [] and s.facts == {"spent": 0.0} and s.decisions == []


def test_once_and_framework_duplicates_are_preflighted_without_made_sideeffects():
    g, s, made = book(once=True)
    out = s.call_many(calls(10, 10))
    assert not out["allowed"] and made == [] and s.made == []

    class Args(BaseModel):
        amount: float

    g = Guard()
    g.tool(name="external", schema=Args, once=True)
    s = g.session()
    c = [{"name": "external", "arguments": {"amount": 10}}] * 2
    out = s.call_many(c)
    assert not out["allowed"] and out["executed"] == [] and s.made == []


def test_all_valid_run_once_bind_plan_and_commit_predicted_facts():
    g, s, made = book()
    out = s.call_many(calls("10", 20), advance=advance)
    assert out["allowed"] and out["executed"] == [0, 1] and made == [10.0, 20.0]
    assert s.facts["spent"] == 30 and len(s.decisions) == 2 and len(s.made) == 2
    for i, d in enumerate(out["decisions"]):
        assert d.response.trace.init["_guard_batch_index"] == i
        assert len(d.response.trace.init["_guard_batch_plan"]) == 2
    assert not s.facts.keys() & {"_guard_batch_plan", "_guard_batch_index"}


def test_runtime_error_stops_later_and_does_not_charge_failed_or_unexecuted():
    g, s, made = book(fail=20)
    out = s.call_many(calls(10, 20, 30), advance=advance)
    assert out["allowed"] and out["executed"] == [0, 1] and out["runtime_error"]["index"] == 1
    assert made == [10.0, 20.0] and s.facts["spent"] == 10 and len(s.made) == 1


def test_framework_success_handed_off_once_and_pause_scope_deny_plan():
    class Args(BaseModel):
        amount: float

    g = Guard()
    g.tool(name="external", schema=Args)
    s = g.session()
    out = s.call_many([{"name": "external", "arguments": {"amount": 10}}])
    assert out["handed_off"] == [0] and out["executed"] == [] and len(s.made) == 1
    g, s, made = book()
    g.pause("stop")
    out = s.call_many(calls(10))
    assert not out["allowed"] and made == []
    assert out["decisions"][0].response.trace.init["_guard_batch_index"] == 0


def test_every_member_checked_once_and_saved_once_only_when_attempted():
    g, s, made = book()
    checked = []
    saved = []
    check, save = g.check, g._save

    def count_check(*args, **kwargs):
        checked.append(args[0])
        assert kwargs["store"] is False
        return check(*args, **kwargs)

    def count_save(d):
        saved.append(d)
        return save(d)

    g.check = count_check
    g._save = count_save
    out = s.call_many(calls(60, 10, 20))
    assert len(checked) == 3 and not out["allowed"] and saved == [] and made == []
    checked.clear()
    out = s.call_many(calls(10, 20))
    assert len(checked) == len(saved) == len(s.decisions) == 2 and made == [10, 20]


def test_advance_cannot_mutate_validated_decision_and_scope_has_plan_identity():
    g, s, made = book()
    s.facts["_guard_batch_index"] = "forged"
    s.facts["_guard_batch_plan"] = "forged"

    def mutate(f, d):
        d.arguments["amount"] = 999
        return {"spent": f["spent"] + 10}

    out = s.call_many(calls(10), advance=mutate)
    assert made == [10] and out["decisions"][0].response.trace.init["_guard_batch_index"] == 0
    g.tool(name="other", schema={"type": "object", "properties": {}, "additionalProperties": False})
    s = g.session(facts={"spent": 0}, allowed_tools=["other"])
    out = s.call_many(calls(10))
    assert not out["allowed"] and out["decisions"][0].response.trace.init["_guard_batch_index"] == 0


def test_async_batch_awaits_actual_policies_tools_and_advance():
    import asyncio

    async def scenario():
        events = []
        g = Guard(fact_names={"spent": float})

        @g.tool
        async def expense(amount: float) -> str:
            await asyncio.sleep(0)
            events.append(("tool", amount))
            return "ok"

        @g.policy("expense")
        async def budget(amount: float, spent: float) -> bool:
            await asyncio.sleep(0)
            events.append(("check", amount))
            return spent + amount <= 100

        async def predict(f, d):
            await asyncio.sleep(0)
            return {**f, "spent": f["spent"] + d.arguments["amount"]}

        s = g.session(facts={"spent": 0.0})
        out = await s.acheck_many(calls(40, 40, 40), advance=predict)
        assert not out["allowed"] and len(out["decisions"]) == 3 and s.facts["spent"] == 0
        assert all(e[0] == "check" for e in events) and s.decisions == []
        events.clear()
        out = await s.acall_many(calls(10, 20), advance=predict)
        assert out["allowed"] and out["executed"] == [0, 1] and s.facts["spent"] == 30
        assert events == [("check", 10.0), ("check", 20.0), ("tool", 10.0), ("tool", 20.0)]
        assert len(s.decisions) == 2 and len(s.made) == 2
        assert out["decisions"][1].response.trace.init["_guard_batch_index"] == 1

    asyncio.run(scenario())


def test_async_runtime_failure_stops_and_preflight_exception_has_no_effect():
    import asyncio

    async def scenario():
        made = []
        g = Guard(fact_names={"spent": float})

        @g.tool
        async def expense(amount: float) -> str:
            made.append(amount)
            if amount == 20:
                raise RuntimeError("bank error")
            return "ok"

        s = g.session(facts={"spent": 0.0})

        async def bad(f, d):
            f["spent"] = 999
            raise ValueError("bad advance")

        with pytest.raises(ValueError, match="bad advance"):
            await s.acall_many(calls(10), advance=bad)
        assert made == [] and s.facts["spent"] == 0 and s.decisions == []
        out = await s.acall_many(calls(10, 20, 30), advance=advance)
        assert out["runtime_error"]["index"] == 1 and made == [10.0, 20.0] and s.facts["spent"] == 10
        assert len(s.made) == 1

    asyncio.run(scenario())


def test_async_once_scope_pause_unknown_and_reserved_advance_facts():
    import asyncio

    async def scenario():
        g, s, made = book(once=True)

        def malicious(f, d):
            return {"spent": 0.0, "calls_made": [], "_guard_batch_plan": [], "_guard_batch_index": 100}

        out = await s.acall_many(calls(10, 10), advance=malicious)
        assert not out["allowed"] and made == [] and s.made == []
        out = await s.acall_many([{"name": "unknown", "arguments": {}}, *calls(10)])
        assert not out["allowed"] and made == [] and len(out["decisions"]) == 2
        assert out["decisions"][0].response.trace.init["_guard_batch_index"] == 0
        g.pause("stop")
        out = await s.acall_many(calls(10))
        assert not out["allowed"] and out["decisions"][0].response.trace.init["_guard_batch_index"] == 0

    asyncio.run(scenario())


def test_sync_unknown_keeps_full_batch_trace_and_never_executes_registered_members():
    g, s, made = book()
    out = s.call_many([{"name": "missing", "arguments": {}}, *calls(10)])
    assert not out["allowed"] and made == [] and len(out["decisions"]) == 2
    assert out["decisions"][0].response.trace.init["_guard_batch_index"] == 0
    assert len(out["decisions"][0].response.trace.init["_guard_batch_plan"]) == 2


def test_async_framework_handoff_and_scope_are_distinct_from_executed_tools():
    import asyncio

    async def scenario():
        class Args(BaseModel):
            amount: float

        g = Guard()
        g.tool(name="external", schema=Args)
        g.tool(name="other", schema=Args)
        s = g.session(facts={"spent": 0.0}, allowed_tools=["external"])
        out = await s.acall_many([{"name": "external", "arguments": {"amount": 10}}], advance=advance)
        assert out["allowed"] and out["handed_off"] == [0] and out["executed"] == []
        assert len(s.made) == 1 and s.facts["spent"] == 10
        out = await s.acall_many([{"name": "other", "arguments": {"amount": 20}}])
        assert not out["allowed"] and len(s.made) == 1
        assert out["decisions"][0].response.trace.init["_guard_batch_index"] == 0

    asyncio.run(scenario())
