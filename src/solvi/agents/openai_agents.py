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
    result = await Runner.run(agent, "Pay invoice 7 ...", run_config=guard_run_config())
    for item in result.interruptions:                     # escalated calls wait for a person
        state = result.to_state(); state.approve(item); result = await Runner.run(agent, state,
                                                                                  run_config=guard_run_config())

For each call the guard checks the proposal (the tool's name and its JSON arguments) against the conversation the
model saw when it proposed the call. The SDK gives a tool's context only the run's input (`turn_input`: the user's
messages and the history passed in), not the items the run generated since — the outputs of the tools it already
called. `guard_run_config()` (a RunConfig whose `call_model_input_filter` records each model input; an existing filter
is kept and runs first) lets the guard read the model's whole input — so an in-run tool output grounds values
(default `ground_from`) and taints them when it carries instruction-like text, and `injections="any"` sees it. Without
it the guard reads `turn_input` alone: a value found only in an in-run tool output is not grounded (deny), and an
instruction in one is not seen. The record is kept in the run's task (a context variable, so concurrent runs are apart)
and used only when its first items are the run's input items.

  allow     the SDK runs the tool (with its ToolContext — not solvi)
  deny      the tool input guardrail rejects the call with the reasons as the tool's output (reject_content): the model
            sees why and may fix the call
  escalate  on_escalate="approval" (default): the tool's needs_approval says yes, so the run stops with an interruption
            (`result.interruptions`); `state.approve(item)` and a new run make the call, recorded as approved by a person
            (guard.resolve); `state.reject(item)` does not. "reject": the guardrail rejects it like a deny.

An approval covers the call it was given for and the reasons it escalated for (in the process that asked: a resumed
call that escalates for other reasons is rejected, with the new reasons, and has to be proposed again). A standing
approval — `state.approve(item, always_approve=True)` — covers later calls of the tool only when your policies alone
escalated them (`GuardDecision.policy_only`); a call escalated by provenance or instruction-like text
(`no_injected_arguments`, `no_instructions_in_tool_outputs`), an unreadable schema, the authorizer or a check that could
not be evaluated needs a person's approval of that very call, and is rejected otherwise.

A tool the guard does not know is denied (declare=True: declared from the tool's params_json_schema on first use). A
tool's own needs_approval (True, or a function) still applies. The decision made for needs_approval is reused by the
guardrail of the same call id with the same arguments in the same process (a call that reaches the guardrail with other
arguments, or an escalation a person has answered since, is checked again); after a resume in another process it is
checked again.

`once=True` tools: the guarded tools of one `guard_tools(...)` call (or one `guard_tool`) keep the calls they allowed
(`solvi_guard.made`) for as long as they live — across runs, in this process — and give them as the fact `calls_made`,
together with any `calls_made` your `facts` give. The guardrail sees a call before the SDK runs it, not its result: an
allowed call counts as made even when the tool then fails (a repeat asks a person).

Handoffs: the SDK may rewrite the history a handed-off agent gets (nested into one summary message by
`nest_handoff_history`, or filtered by a handoff's `input_filter`); the guard reads what the run gives it and fails
closed — a value the user wrote before the handoff may no longer be found as the user's, and a user-grounded call is
then denied."""
from __future__ import annotations

import contextvars
import dataclasses
import inspect
import json
from typing import Callable

try:
    from agents import FunctionTool, RunConfig, ToolGuardrailFunctionOutput, ToolInputGuardrail
except ImportError as e:                               # "No module named 'agents'" reads like a broken solvi install
    raise ImportError('solvi.agents.openai_agents needs the OpenAI Agents SDK: pip install "solvi[openai-agents]"') from e

from .guard import Guard, messages, proposal, with_calls_made

_MODEL_INPUT = contextvars.ContextVar("solvi_model_input", default=None)   # the run's latest model input (its task)


def guard_run_config(run_config: RunConfig | None = None) -> RunConfig:
    """A RunConfig (a copy of `run_config`) whose call_model_input_filter records each model input, so the guard reads
    the tool outputs of the run itself (see the module docs). An existing filter runs first; what it returns is what
    the model sees and what is recorded. The record lives in the run's task (a context variable): concurrent runs do
    not see each other's."""
    rc = run_config or RunConfig()
    inner = rc.call_model_input_filter

    async def record(data):
        out = data.model_data
        if inner is not None:
            out = inner(data)
            if inspect.isawaitable(out):
                out = await out
        _MODEL_INPUT.set(list(out.input))
        return out
    record.solvi_records = True
    return dataclasses.replace(rc, call_model_input_filter=record)


def context_of(wrapper) -> list:
    """The conversation for the guard → [(role, text)]: the latest model input `guard_run_config` recorded in this run
    (when its first items read as the run's input items), else the run's input items (RunContextWrapper.turn_input)
    alone."""
    items = list(getattr(wrapper, "turn_input", None) or [])
    head = messages(items)
    recorded = _MODEL_INPUT.get()
    if recorded is not None and len(recorded) >= len(items) and messages(recorded[:len(items)]) == head:
        return messages(recorded)
    return head


class _Guarded:
    def __init__(self, tool, guard, facts, on_escalate, declare, made=None):
        if not isinstance(guard, Guard):
            raise TypeError("guard is a solvi.agents.Guard")
        if on_escalate not in ("approval", "reject"):
            raise ValueError('on_escalate: "approval" | "reject"')
        self.tool, self.guard, self.facts, self.on_escalate, self.declare = tool, guard, facts, on_escalate, declare
        self.own = tool.needs_approval
        self.pending = {}                              # (call id, the arguments) → the decision made for needs_approval
        self.asked = {}                                # call id → the approval key of the escalation shown to a person
        self.decisions = []
        self.made = [] if made is None else made       # the calls allowed to run (for once=True tools)

    async def check(self, wrapper, params, call_id):
        g, t = self.guard, self.tool
        if self.declare and t.name not in g.tools:
            g.declare(t.name, schema=t.params_json_schema, description=t.description or "")
        facts = with_calls_made(self.facts(wrapper) if callable(self.facts) else self.facts, self.made)
        return await g.acheck({"name": t.name, "arguments": params, "id": call_id}, context_of(wrapper), facts)

    async def needs_approval(self, wrapper, params, call_id):
        own = self.own
        if callable(own):
            own = own(wrapper, params, call_id)
            if hasattr(own, "__await__"):
                own = await own
        d = await self.check(wrapper, params, call_id)
        self.pending[(call_id, _canonical(params))] = d
        escalate = d.outcome == "escalate" and self.on_escalate == "approval"
        if escalate:
            self.asked.setdefault(call_id, d.approval_key())       # the first escalation shown is what an approval covers
        return bool(own) or escalate

    async def guardrail(self, data):
        c = data.context
        try:
            params = json.loads(c.tool_arguments or "{}")
        except ValueError:
            params = c.tool_arguments
        d = self.pending.pop((c.tool_call_id, _canonical(params)), None)
        for k in [k for k in self.pending if k[0] == c.tool_call_id]:     # made for other arguments: stale
            del self.pending[k]
        if d is not None and d.outcome == "escalate" and _approval(c, self.tool.name, c.tool_call_id) is not None:
            d = None                                   # decided by a person since: check the call as it is now
        if d is None:
            d = await self.check(c, params, c.tool_call_id)
        if d.outcome == "escalate" and self.on_escalate == "approval":
            status = _approval(c, self.tool.name, c.tool_call_id)
            if status is None:                         # a pre-approval guardrail run: the approval gate decides next
                return ToolGuardrailFunctionOutput.allow(output_info={"solvi": d.to_dict(), "awaiting_approval": True})
            asked = self.asked.pop(c.tool_call_id, None)
            if status is True and not d.policy_only and _approval(c, self.tool.name, c.tool_call_id, per_call=True) \
                    is not True:
                self.guard._save(d)
                return ToolGuardrailFunctionOutput.reject_content(
                    f"{self.tool.name} escalated ({'; '.join(d.reasons)}): a standing approval of the tool "
                    "(always_approve) covers only escalations by policies; this call needs a person's approval of its "
                    "own", output_info={"solvi": d.to_dict(), "standing_approval": False})
            if status is True and asked is not None and asked != d.approval_key():
                self.guard._save(d)
                return ToolGuardrailFunctionOutput.reject_content(
                    f"{self.tool.name} escalated again for other reasons ({'; '.join(d.reasons)}): the approval given "
                    "does not cover them; propose the call again to ask a person",
                    output_info={"solvi": d.to_dict(), "approval_stale": True})
            if status is True:                         # a person approved it (state.approve): record who decided
                d = self.guard.resolve(d, approve=True, reviewer="openai-agents approval", execute=False)
        self.decisions.append(d)
        if d.outcome == "allow":
            self.made.append(proposal(d.tool, d.arguments))        # the SDK runs it next
            return ToolGuardrailFunctionOutput.allow(output_info={"solvi": d.to_dict()})
        return ToolGuardrailFunctionOutput.reject_content(d.message(), output_info={"solvi": d.to_dict()})


def _canonical(params):
    """The arguments as a key: their JSON with sorted keys (a string that is not JSON as it is)."""
    try:
        return json.dumps(params, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return repr(params)


def _approval(ctx, name, call_id, per_call=False):
    """The SDK's approval status of a call: True (approved), False (rejected), None (not decided); False when unknown.
    per_call: only a decision on this very call (not a standing always_approve / always_reject of the tool)."""
    try:
        if per_call:
            return ctx._get_per_call_approval_status_for_key(name, call_id)
        return ctx.get_approval_status(name, call_id)
    except Exception:  # noqa: BLE001 — no approval record we can read: not approved
        return False


def guard_tool(tool: FunctionTool, guard: Guard, facts: Callable | dict | None = None, on_escalate="approval",
               declare=False, made: list | None = None) -> FunctionTool:
    """A copy of a FunctionTool whose every call passes `guard` (see the module docs). facts: a dict, or a function of the
    RunContextWrapper returning one. The copy's `solvi_guard.decisions` lists every GuardDecision. made: a list the
    allowed calls are appended to (for once=True tools; `guard_tools` shares one among its tools)."""
    if not isinstance(tool, FunctionTool):
        raise TypeError(f"guard_tool guards a FunctionTool (function_tool(...)), not {type(tool).__name__}")
    g = _Guarded(tool, guard, facts, on_escalate, declare, made)
    rail = ToolInputGuardrail(guardrail_function=g.guardrail, name="solvi_guard")
    out = dataclasses.replace(tool, tool_input_guardrails=list(tool.tool_input_guardrails or []) + [rail],
                              needs_approval=g.needs_approval)
    out.solvi_guard = g
    return out


def guard_tools(tools, guard: Guard, **kw) -> list:
    """guard_tool for each FunctionTool of a list (other tools — hosted, MCP — are returned unchanged)."""
    kw.setdefault("made", [])
    return [guard_tool(t, guard, **kw) if isinstance(t, FunctionTool) else t for t in tools]
