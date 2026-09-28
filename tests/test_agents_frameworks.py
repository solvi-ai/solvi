"""The Guard inside real agent frameworks, with scripted models (no API keys): PydanticAI (a toolset wrapper), LangGraph
(a guarded ToolNode) and the OpenAI Agents SDK (a tool input guardrail + needs_approval). Each test is skipped when its
framework is not installed (`uv sync --group agents` installs all three)."""
import asyncio

import pytest

from solvi.agents import Guard

IBAN = "DE89370400440532013000"
PROMPT = f"Please pay invoice 7 to {IBAN}: 250 EUR now, and 5000 EUR for invoice 8."
CALLS = [{"iban": "DE00000000000000000000", "amount": 10.0},     # an IBAN the user never wrote → deny
         {"iban": IBAN, "amount": 250.0},                         # allow: the framework runs the tool
         {"iban": IBAN, "amount": 5000.0}]                        # escalate → approved by a person → made


def guard_and_tool(storage=None):
    paid = []

    def send_payment(iban: str, amount: float) -> str:
        """Pay an invoice."""
        paid.append((iban, amount))
        return f"paid {amount} to {iban}"
    g = Guard(storage=storage)
    g.tool(send_payment, ground=["iban"])

    @g.policy("send_payment", on_fail="escalate")
    def within_auto_limit(amount: float) -> bool:
        """Payments above 1 000 need a person."""
        return amount <= 1000
    return g, send_payment, paid


def test_pydantic_ai_toolset(tmp_path):
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent, DeferredToolRequests, DeferredToolResults, FunctionToolset, RunContext
    from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    from solvi.agents.pydantic_ai import GuardedToolset, context_of
    g, fn, paid = guard_and_tool(tmp_path / "calls.db")

    def send_payment(ctx: RunContext[None], iban: str, amount: float) -> str:    # a context argument is not an argument
        return fn(iban, amount)
    g.tool(send_payment, ground=["iban"])
    script = iter(CALLS)

    def model(messages, info):
        last = messages[-1].parts[-1]
        if last.part_kind in ("user-prompt", "retry-prompt", "tool-return"):
            try:
                return ModelResponse(parts=[ToolCallPart("send_payment", next(script))])
            except StopIteration:
                pass
        return ModelResponse(parts=[TextPart("done")])
    tools = GuardedToolset(FunctionToolset([send_payment]), g)
    agent = Agent(FunctionModel(model), toolsets=[tools], output_type=[str, DeferredToolRequests])
    r = agent.run_sync(PROMPT)
    assert isinstance(r.output, DeferredToolRequests) and paid == [(IBAN, 250.0)]
    assert [d.outcome for d in tools.decisions] == ["deny", "allow", "escalate"]
    [call] = r.output.approvals
    meta = r.output.metadata[call.tool_call_id]["solvi"]
    assert meta["reasons"] == ["within_auto_limit: Payments above 1 000 need a person. [escalate]"] and meta["stored_id"]
    retry = [p for m in r.all_messages() for p in getattr(m, "parts", []) if p.part_kind == "retry-prompt"]
    assert "not in the conversation" in str(retry[0].content)
    assert ("user", PROMPT) in context_of(r.all_messages())
    r2 = agent.run_sync(message_history=r.all_messages(),
                        deferred_tool_results=DeferredToolResults(approvals={call.tool_call_id: True}))
    assert r2.output == "done" and paid == [(IBAN, 250.0), (IBAN, 5000.0)]
    assert tools.decisions[-1].approved_by == "pydantic-ai approval"
    assert g.storage.corrections()[0]["answer"] == "allow" and g.storage.verify()["ok"]
    assert all(g.replay(s.id)["ok"] for s in g.storage.iter())


def test_langgraph_tool_node():
    pytest.importorskip("langgraph")
    from typing import Annotated, TypedDict

    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from langgraph.types import Command

    from solvi.agents.langgraph import approved, guarded_tool_node
    g, send_payment, paid = guard_and_tool()

    class State(TypedDict):
        messages: Annotated[list, add_messages]
    script = iter(CALLS)

    def agent(state):
        if isinstance(state["messages"][-1], HumanMessage):
            args = next(script)
            return {"messages": [AIMessage("", tool_calls=[{"name": "send_payment", "args": args,
                                                            "id": f"call-{int(args['amount'])}"}])]}
        return {"messages": [AIMessage("done")]}
    node = guarded_tool_node([tool(send_payment)], g)
    b = StateGraph(State)
    b.add_node("agent", agent)
    b.add_node("tools", node)
    b.add_edge(START, "agent")
    b.add_conditional_edges("agent", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    b.add_edge("tools", END)
    app = b.compile(checkpointer=InMemorySaver())
    out = []
    for i in range(3):
        cfg = {"configurable": {"thread_id": str(i)}}
        r = app.invoke({"messages": [HumanMessage(PROMPT)]}, cfg)
        if "__interrupt__" in r:
            [intr] = r["__interrupt__"]
            assert intr.value["solvi"]["outcome"] == "escalate" and paid == [(IBAN, 250.0)]
            r = app.invoke(Command(resume=True), cfg)
        out.append(r["messages"][-1])
    assert out[0].status == "error" and out[0].content.startswith("send_payment denied: not in the conversation")
    assert out[0].artifact["solvi"]["outcome"] == "deny"
    assert out[1].content == f"paid 250.0 to {IBAN}" and out[2].content == f"paid 5000.0 to {IBAN}"
    assert paid == [(IBAN, 250.0), (IBAN, 5000.0)] and node.solvi_guard.decisions[-1].approved_by == "langgraph interrupt"
    assert approved({"approved": True}) and not approved("no")
    cfg = {"configurable": {"thread_id": "rejected"}}                  # a person says no: not made, the model is told
    script = iter([CALLS[2]])
    r = app.invoke({"messages": [HumanMessage(PROMPT)]}, cfg)
    r = app.invoke(Command(resume="no"), cfg)
    assert r["messages"][-1].status == "error" and "rejected by langgraph interrupt" in r["messages"][-1].content
    assert len(paid) == 2


def test_openai_agents_guardrail_and_approval():
    pytest.importorskip("agents")
    from agents import Agent, Runner, function_tool, set_tracing_disabled
    from agents.testing import ScriptedModel, assistant_message, function_call

    from solvi.agents.openai_agents import guard_tools
    g, send_payment, paid = guard_and_tool()
    set_tracing_disabled(True)

    async def run():
        tools = guard_tools([function_tool(send_payment)], g)
        model = ScriptedModel([[function_call("send_payment", a, call_id=f"c{i}")] for i, a in enumerate(CALLS)]
                              + [[assistant_message("done")]])
        agent = Agent(name="payer", model=model, tools=tools)
        r = await Runner.run(agent, PROMPT)
        assert [i.raw_item.call_id for i in r.interruptions] == ["c2"] and paid == [(IBAN, 250.0)]
        state = r.to_state()
        for item in r.interruptions:
            state.approve(item)
        r = await Runner.run(agent, state)
        outs = {x.raw_item["call_id"]: x.raw_item["output"] for x in r.new_items
                if isinstance(x.raw_item, dict) and x.raw_item.get("type") == "function_call_output"}
        return r, tools[0].solvi_guard, outs
    r, sg, outs = asyncio.run(run())
    assert r.final_output == "done" and paid == [(IBAN, 250.0), (IBAN, 5000.0)]
    assert outs["c0"].startswith("send_payment denied: not in the conversation")
    assert [d.outcome for d in sg.decisions] == ["deny", "allow", "allow"] and sg.decisions[-1].approved_by
