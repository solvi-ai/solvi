"""OpenAI Agents SDK: every call of a function tool passes a solvi Guard (a tool input guardrail, plus needs_approval for
escalations).

    from agents import Agent, Runner, function_tool
    from solvi.agents import Guard
    from solvi.agents.openai_agents import guard_tools

    guard = Guard(storage="calls.db")
    guard.tool(send_payment, ground=["iban"])             # the plain typed function behind the @function_tool
    ...policies...
    agent = Agent(name="payer", tools=guard_tools([function_tool(send_payment)], guard,
                                                  facts=lambda ctx: {"spent_today": ctx.context.spent}))
    result = await Runner.run(agent, "Pay invoice 7 ...")
    for item in result.interruptions:                     # escalated calls wait for a person
        state = result.to_state(); state.approve(item); result = await Runner.run(agent, state)

For each call the guard checks the proposal (the tool's name and its JSON arguments) against the turn's input items
(`RunContextWrapper.turn_input`: user, assistant, system / developer messages and function call outputs):

  allow     the SDK runs the tool (with its ToolContext — not solvi)
  deny      the tool input guardrail rejects the call with the reasons as the tool's output (reject_content): the model
            sees why and may fix the call
  escalate  on_escalate="approval" (default): the tool's needs_approval says yes, so the run stops with an interruption
            (`result.interruptions`); `state.approve(item)` and a new run make the call, recorded as approved by a person
            (guard.resolve); `state.reject(item)` does not. "reject": the guardrail rejects it like a deny.

A tool the guard does not know is denied (declare=True: declared from the tool's params_json_schema on first use). A
tool's own needs_approval (True, or a function) still applies. The decision made for needs_approval is reused by the
guardrail of the same call id with the same arguments in the same process (a call that reaches the guardrail with other
arguments is checked again); after a resume in another process it is checked again."""
from __future__ import annotations

import dataclasses
import json
from typing import Callable

from agents import FunctionTool, ToolGuardrailFunctionOutput, ToolInputGuardrail

from .guard import Guard, messages


def context_of(wrapper) -> list:
    """The turn's input items (RunContextWrapper.turn_input) → [(role, text)] for the guard."""
    return messages(list(getattr(wrapper, "turn_input", None) or []))


class _Guarded:
    def __init__(self, tool, guard, facts, on_escalate, declare):
        if not isinstance(guard, Guard):
            raise TypeError("guard is a solvi.agents.Guard")
        if on_escalate not in ("approval", "reject"):
            raise ValueError('on_escalate: "approval" | "reject"')
        self.tool, self.guard, self.facts, self.on_escalate, self.declare = tool, guard, facts, on_escalate, declare
        self.own = tool.needs_approval
        self.pending = {}                              # (call id, the arguments) → the decision made for needs_approval
        self.decisions = []

    async def check(self, wrapper, params, call_id):
        g, t = self.guard, self.tool
        if self.declare and t.name not in g.tools:
            g.declare(t.name, schema=t.params_json_schema, description=t.description or "")
        facts = self.facts(wrapper) if callable(self.facts) else self.facts
        return await g.acheck({"name": t.name, "arguments": params, "id": call_id}, context_of(wrapper), facts)

    async def needs_approval(self, wrapper, params, call_id):
        own = self.own
        if callable(own):
            own = own(wrapper, params, call_id)
            if hasattr(own, "__await__"):
                own = await own
        d = await self.check(wrapper, params, call_id)
        self.pending[(call_id, _canonical(params))] = d
        return bool(own) or (d.outcome == "escalate" and self.on_escalate == "approval")

    async def guardrail(self, data):
        c = data.context
        try:
            params = json.loads(c.tool_arguments or "{}")
        except ValueError:
            params = c.tool_arguments
        d = self.pending.pop((c.tool_call_id, _canonical(params)), None)
        for k in [k for k in self.pending if k[0] == c.tool_call_id]:     # made for other arguments: stale
            del self.pending[k]
        if d is None:
            d = await self.check(c, params, c.tool_call_id)
        if d.outcome == "escalate" and self.on_escalate == "approval":
            status = _approval(c, self.tool.name, c.tool_call_id)
            if status is None:                         # a pre-approval guardrail run: the approval gate decides next
                return ToolGuardrailFunctionOutput.allow(output_info={"solvi": d.to_dict(), "awaiting_approval": True})
            if status is True:                         # a person approved it (state.approve): record who decided
                d = self.guard.resolve(d, approve=True, reviewer="openai-agents approval", execute=False)
        self.decisions.append(d)
        if d.outcome == "allow":
            return ToolGuardrailFunctionOutput.allow(output_info={"solvi": d.to_dict()})
        return ToolGuardrailFunctionOutput.reject_content(d.message(), output_info={"solvi": d.to_dict()})


def _canonical(params):
    """The arguments as a key: their JSON with sorted keys (a string that is not JSON as it is)."""
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(params)


def _approval(ctx, name, call_id):
    """The SDK's approval status of a call: True (approved), False (rejected), None (not decided); False when unknown."""
    try:
        return ctx.get_approval_status(name, call_id)
    except Exception:  # noqa: BLE001 — no approval record we can read: not approved
        return False


def guard_tool(tool: FunctionTool, guard: Guard, facts: Callable | dict | None = None, on_escalate="approval",
               declare=False) -> FunctionTool:
    """A copy of a FunctionTool whose every call passes `guard` (see the module docs). facts: a dict, or a function of the
    RunContextWrapper returning one. The copy's `solvi_guard.decisions` lists every GuardDecision."""
    if not isinstance(tool, FunctionTool):
        raise TypeError(f"guard_tool guards a FunctionTool (function_tool(...)), not {type(tool).__name__}")
    g = _Guarded(tool, guard, facts, on_escalate, declare)
    rail = ToolInputGuardrail(guardrail_function=g.guardrail, name="solvi_guard")
    out = dataclasses.replace(tool, tool_input_guardrails=list(tool.tool_input_guardrails or []) + [rail],
                              needs_approval=g.needs_approval)
    out.solvi_guard = g
    return out


def guard_tools(tools, guard: Guard, **kw) -> list:
    """guard_tool for each FunctionTool of a list (other tools — hosted, MCP — are returned unchanged)."""
    return [guard_tool(t, guard, **kw) if isinstance(t, FunctionTool) else t for t in tools]
