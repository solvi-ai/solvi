"""PydanticAI: every tool call of an agent passes a solvi Guard (a toolset wrapper).

    from pydantic_ai import Agent, DeferredToolRequests, FunctionToolset
    from solvi.agents import Guard
    from solvi.agents.pydantic_ai import GuardedToolset

    guard = Guard(storage="calls.db")
    guard.tool(send_payment, ground=["iban"])             # the same typed functions the agent gets
    ...policies...
    tools = GuardedToolset(FunctionToolset([send_payment]), guard, facts=lambda ctx: {"spent_today": ctx.deps.spent})
    agent = Agent(model, toolsets=[tools], output_type=[str, DeferredToolRequests])

For each call PydanticAI makes (its arguments already validated by the tool's schema), the guard checks the proposal
against the conversation so far (`ctx.messages`: system prompts, user prompts, tool returns, the model's text) and:

  allow     the wrapped toolset runs the tool (PydanticAI runs it — with its RunContext — not solvi)
  deny      on_deny="retry" (default): ModelRetry with the reasons, so the model can fix the call; "fail": ToolFailed
  escalate  on_escalate="approval" (default): ApprovalRequired — the run ends with DeferredToolRequests (the output type
            must allow it); resume with DeferredToolResults(approvals={id: True}) and the call is made, recorded as
            approved by a person (guard.resolve); "fail": ToolFailed with the reasons. An approval covers the call
            with the arguments and the reasons it was asked for (metadata["approval_key"]): a resumed call that
            escalates for other reasons (other arguments — ToolApproved(override_args=...) —, a changed fact) is asked
            again. This binding holds in the process that asked; after a restart the resumed call is approved as is.

The conversation is `ctx.messages`; a user prompt sent in the same request as a tool return is a tool output (that is
how PydanticAI sends `ToolReturn(content=...)` and MCP tool content), see `context_of`.

show_policies=True: the description of each tool the model is offered lists the reasons of the policies that check it
(`Guard.described`), so the model can follow them instead of learning each one from a refusal; off by default.

A tool the guard does not know is denied; with auto_declare=True (`declare=` in 0.7) it is declared from its JSON schema on first use (the
guard then checks its types, and any policies that name it). The first parameter of a tool function typed as RunContext
is not an argument (Guard.tool skips it).

`once=True` tools: the toolset keeps the calls it made per conversation — PydanticAI's `RunContext.conversation_id`,
which runs continuing one `message_history` (and a resumed deferred call) share, and which a run without history gets
fresh — and gives a call the ones made in its own conversation as the fact `calls_made`, together with any `calls_made`
your `facts` give (calls made earlier or elsewhere). `made` maps a conversation id to its calls (a call counts when
the tool returned without raising); a PydanticAI without conversation ids keys every call None, one memory for the
process. The memory is this process's: after a restart, give the calls made before through `facts`."""
from __future__ import annotations

import dataclasses
from typing import Any, Callable

try:
    from pydantic_ai import ApprovalRequired, ModelRetry, ToolFailed
    from pydantic_ai.toolsets import WrapperToolset
except ImportError as e:
    raise ImportError('solvi.agents.pydantic_ai needs PydanticAI: pip install "solvi[pydantic-ai]"') from e

from .. import _deprecate
from .guard import Guard, _text, proposal, with_calls_made


_FROM_TOOLS = ("tool-return", "retry-prompt", "builtin-tool-return")


def context_of(messages) -> list:
    """PydanticAI's message history → [(role, text)] for the guard (by each part's part_kind): system prompts, the user's
    prompts, tool returns (and a model's built-in tool returns: web search results ...), the model's text.

    A user-prompt part in a request that also holds tool returns or retry prompts is read as a tool output: PydanticAI
    sends a tool's `ToolReturn(content=...)` and an MCP tool's text or files that way — as a user prompt next to the
    return — so it is not the user's words (fail closed: a prompt the user wrote in the same request as a tool return is
    read as a tool output too). A retry prompt's own text (a validation error, the guard's own reasons) is not read:
    it repeats the rejected values. A compaction (a summary of the history) is the model's."""
    out = []
    for m in messages or ():
        parts = list(getattr(m, "parts", ()) or ())
        from_tools = any(getattr(p, "part_kind", None) in _FROM_TOOLS for p in parts)
        for p in parts:
            kind = getattr(p, "part_kind", None)
            if kind == "system-prompt":
                out.append(("system", _text(p.content)))
            elif kind == "user-prompt":
                out.append(("tool" if from_tools else "user", _text(p.content)))
            elif kind in ("tool-return", "builtin-tool-return"):
                out.append(("tool", p.model_response_str() if hasattr(p, "model_response_str") else _text(p.content)))
            elif kind in ("text", "compaction"):
                out.append(("assistant", _text(p.content)))
    return [(r, t) for r, t in out if t]


@_deprecate.removed_kwargs(declare="auto_declare")      # 0.7 name, removed in 0.9
@dataclasses.dataclass
class GuardedToolset(WrapperToolset):
    """A toolset whose every call passes `guard` (see the module docs). facts: a dict, or a function of the RunContext
    returning one — the facts your policies read (a user's role, a budget left)."""
    guard: Guard = None
    facts: Callable | dict | None = None
    on_deny: str = "retry"
    on_escalate: str = "approval"
    auto_declare: bool = False
    show_policies: bool = False                                              # list each tool's policies in its description
    decisions: list = dataclasses.field(default_factory=list, repr=False)   # every GuardDecision, in order
    made: dict = dataclasses.field(default_factory=dict, repr=False)        # conversation id → its calls made (once=True)
    _asked: dict = dataclasses.field(default_factory=dict, repr=False)      # call id → the approval key asked for

    def __post_init__(self):
        if not isinstance(self.guard, Guard):
            raise TypeError("GuardedToolset(toolset, guard): guard is a solvi.agents.Guard")
        if self.on_deny not in ("retry", "fail") or self.on_escalate not in ("approval", "fail"):
            raise ValueError('on_deny: "retry" | "fail"; on_escalate: "approval" | "fail"')

    async def get_tools(self, ctx) -> dict:
        """The wrapped toolset's tools; with show_policies=True each one the guard knows gets the reasons of its
        policies appended to its description (Guard.described), so the model reads them before it calls."""
        tools = await super().get_tools(ctx)
        if not self.show_policies:
            return tools
        out = {}
        for name, t in tools.items():
            if name in self.guard.tools and self.guard.tools[name].model is not None:
                td = dataclasses.replace(t.tool_def, description=self.guard.described(name, t.tool_def.description or ""))
                t = dataclasses.replace(t, tool_def=td)
            out[name] = t
        return out

    async def call_tool(self, name: str, tool_args: dict[str, Any], ctx, tool) -> Any:
        g = self.guard
        if self.auto_declare and name not in g.tools:
            td = tool.tool_def
            g.declare(name, schema=td.parameters_json_schema, description=td.description or "")
        conv = getattr(ctx, "conversation_id", None)
        facts = with_calls_made(self.facts(ctx) if callable(self.facts) else self.facts, self.made.get(conv, ()))
        d = await g.acheck({"name": name, "arguments": tool_args, "id": ctx.tool_call_id}, context_of(ctx.messages), facts)
        asked = self._asked.get(ctx.tool_call_id)
        if d.outcome == "escalate" and getattr(ctx, "tool_call_approved", False) and asked in (None, d.approval_key()):
            self._asked.pop(ctx.tool_call_id, None)
            d = g.resolve(d, approve=True, reviewer="pydantic-ai approval", execute=False)
        self.decisions.append(d)
        if d.outcome == "allow":
            out = await super().call_tool(name, tool_args, ctx, tool)
            self.made.setdefault(conv, []).append(proposal(d.tool, d.arguments))   # it returned: made (an error is not)
            return out
        if d.outcome == "escalate" and self.on_escalate == "approval":
            # the approval covers this call with these arguments and these reasons: a resumed call that escalates for
            # other ones (other arguments, a changed fact, a new tool output) is asked again
            self._asked[ctx.tool_call_id] = d.approval_key()
            raise ApprovalRequired(metadata={"solvi": d.to_dict(), "approval_key": d.approval_key()})
        if d.outcome == "deny" and self.on_deny == "retry":
            raise ModelRetry(d.message())
        raise ToolFailed(d.message())



__all__ = ["context_of", "GuardedToolset"]
