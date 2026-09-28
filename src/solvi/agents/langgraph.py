"""LangGraph: a ToolNode whose every tool call passes a solvi Guard.

    from langgraph.checkpoint.memory import InMemorySaver
    from solvi.agents import Guard
    from solvi.agents.langgraph import guarded_tool_node

    guard = Guard(storage="calls.db")
    guard.tool(send_payment, ground=["iban"])              # the plain typed function behind the @tool
    ...policies...
    tools = guarded_tool_node([send_payment_tool], guard, facts=lambda state: {"spent_today": state["spent"]})
    graph = builder.add_node("tools", tools) ... .compile(checkpointer=InMemorySaver())

The guard wraps the node's tool execution (`ToolNode(wrap_tool_call=..., awrap_tool_call=...)`, langgraph ≥ 1.0): for
each call in the model's message it checks the proposal (`request.tool_call`: name, args, id) against the graph's
messages (`state["messages"]`: human, ai, tool and system messages) and:

  allow     the ToolNode runs the tool (LangGraph runs it — with its injected state / runtime — not solvi)
  deny      a ToolMessage with status "error" and the reasons: the model sees why and may fix the call
  escalate  on_escalate="interrupt" (default): `interrupt({"solvi": decision})` pauses the graph (it needs a
            checkpointer); resume with Command(resume=True) (or "approve", {"approved": True}) and the call is made,
            recorded as approved by a person (guard.resolve); anything else rejects it. "message": a ToolMessage saying
            the call was escalated, not made.

A tool the guard does not know is denied (declare=True: declared from the tool's args_schema on first use). On resume
LangGraph runs the node again, so the call is checked (and stored) again before the approval is recorded."""
from __future__ import annotations

from typing import Callable

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from .guard import Guard, messages

APPROVE = (True, "approve", "approved", "allow", "yes")


def _messages_of(state):
    if isinstance(state, dict):
        return state.get("messages") or []
    if isinstance(state, list):
        return state
    return getattr(state, "messages", None) or []


def approved(answer) -> bool:
    """A resume value that approves an escalated call: True, "approve" / "allow" / "yes", or {"approved": True}."""
    if isinstance(answer, dict):
        return bool(answer.get("approved") or answer.get("approve"))
    return answer in APPROVE


class _Wrap:
    def __init__(self, guard, facts, on_escalate, declare):
        if not isinstance(guard, Guard):
            raise TypeError("guard is a solvi.agents.Guard")
        if on_escalate not in ("interrupt", "message"):
            raise ValueError('on_escalate: "interrupt" | "message"')
        self.guard, self.facts, self.on_escalate, self.declare = guard, facts, on_escalate, declare
        self.decisions = []

    def _pre(self, request):
        tc = request.tool_call
        if self.declare and tc["name"] not in self.guard.tools and request.tool is not None:
            schema = request.tool.get_input_schema().model_json_schema()
            self.guard.declare(tc["name"], schema=schema, description=request.tool.description or "")
        facts = self.facts(request.state) if callable(self.facts) else self.facts
        return {"name": tc["name"], "arguments": tc.get("args") or {}, "id": tc.get("id")}, \
            messages(_messages_of(request.state)), facts

    def _post(self, request, d):
        """→ ("run", decision) or ("reply", ToolMessage)."""
        g, tc = self.guard, request.tool_call
        if d.outcome == "escalate" and self.on_escalate == "interrupt":
            from langgraph.types import interrupt
            answer = interrupt({"solvi": d.to_dict()})
            d = g.resolve(d, approve=approved(answer), reviewer="langgraph interrupt", execute=False,
                          note=None if approved(answer) else f"resume value {answer!r}")
        self.decisions.append(d)
        if d.outcome == "allow":
            return "run", d
        return "reply", ToolMessage(content=d.message(), tool_call_id=tc.get("id"), name=tc["name"], status="error",
                                    artifact={"solvi": d.to_dict()})

    def __call__(self, request, execute):
        call, ctx, facts = self._pre(request)
        what, out = self._post(request, self.guard.check(call, ctx, facts))
        return execute(request) if what == "run" else out

    async def acall(self, request, execute):
        call, ctx, facts = self._pre(request)
        what, out = self._post(request, await self.guard.acheck(call, ctx, facts))
        return await execute(request) if what == "run" else out


def guard_wrappers(guard, facts: Callable | dict | None = None, on_escalate="interrupt", declare=False):
    """(wrap_tool_call, awrap_tool_call) for your own ToolNode (or a create_agent middleware); the first one's
    `.decisions` lists every GuardDecision."""
    w = _Wrap(guard, facts, on_escalate, declare)
    return w, w.acall


def guarded_tool_node(tools, guard, facts: Callable | dict | None = None, on_escalate="interrupt", declare=False,
                      **kw) -> ToolNode:
    """A ToolNode over `tools` whose every call passes `guard` (see the module docs). facts: a dict, or a function of
    the graph state returning one. Other keyword arguments go to ToolNode. The node's `solvi_guard.decisions` lists every
    GuardDecision."""
    w = _Wrap(guard, facts, on_escalate, declare)
    node = ToolNode(tools, wrap_tool_call=w, awrap_tool_call=w.acall, **kw)
    node.solvi_guard = w
    return node
