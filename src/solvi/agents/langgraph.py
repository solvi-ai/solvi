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
  escalate  on_escalate="interrupt" (default): `interrupt({"solvi": decision, "id": tool_call_id, "args_hash": ...,
            "key": ...})` pauses the graph (it needs a checkpointer); resume with {"approved": True, "id": tool_call_id}
            (or a map {tool_call_id: True}) and the call is made, recorded as approved by a person (guard.resolve);
            anything else rejects it. "message": a ToolMessage saying the call was escalated, not made.

An approval is bound to the call it answers. The parallel calls of one model message run in one node task, whose
interrupts share one sequence of resume values: a bare `Command(resume=True)` could be read by another call than the one
the person saw. So when the model's message holds several tool calls, only an answer naming the call approves it —
{"approved": True, "id": "<tool_call_id>"} or a map {"<tool_call_id>": True, ...} (one resume value may answer them
all); a bare True / "approve" is a rejection there, and an answer naming another call is not this call's (it waits
for its own). With a single call a bare True still approves. The answer may carry the payload's "args_hash" and "key"
(the decision's approval_key: the call, its arguments and the reasons it escalated for); when the call is checked again
on resume and escalates for other reasons (a changed fact, other arguments), an answer given for the old key does not
cover it: the node interrupts again with the new reasons. Without "key" in the answer this binding holds in the process
that asked (it remembers the keys it asked with); after a restart, an answer naming the call approves it as it now is.

A tool the guard does not know is denied (declare=True: declared from the tool's args_schema on first use). On resume
LangGraph runs the node again, so the call is checked (and stored) again before the approval is recorded. A call of the
same message that was already made — allowed at once, or approved while another call of the message still waited —
is not made again on that re-run: the wrapper returns the result it had (per thread, node task, tool call id and
arguments, in the process that made it; after a restart LangGraph's own re-run rule applies — keep such tools idempotent)."""
from __future__ import annotations

import collections
from typing import Callable

from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode

from .guard import Guard, messages

APPROVE = ("approve", "approved", "allow", "yes")      # the strings that approve (any case); else only True itself
DONE_KEPT = 4096                                       # the results of made calls kept, so a re-run node does not repeat them


def _messages_of(state):
    if isinstance(state, dict):
        return state.get("messages") or []
    if isinstance(state, list):
        return state
    return getattr(state, "messages", None) or []


def approved(answer) -> bool:
    """A resume value that approves an escalated call: True, "approve" / "approved" / "allow" / "yes", or {"approved": one
    of them} ({"approve": ...} too). Anything else rejects — "false", 1, "no", {"approved": "false"}."""
    if isinstance(answer, dict):
        return any(_yes(answer.get(k)) for k in ("approved", "approve"))
    return _yes(answer)


def _yes(v):
    return v is True or (isinstance(v, str) and v.strip().lower() in APPROVE)


def _args_hash(args):
    import hashlib
    import json
    from ..decide import jsonable
    return hashlib.sha256(json.dumps(jsonable(args), sort_keys=True, ensure_ascii=False, default=str).encode()
                          ).hexdigest()[:16]


def _calls_in_message(state, call_id):
    """How many tool calls the model's message that proposed `call_id` holds (1 when it cannot be found)."""
    for m in reversed(_messages_of(state)):
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if tcs and any((tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)) == call_id for tc in tcs):
            return len(tcs)
    return 1


def _for_call(answer, call_id):
    """A resume value → (this call's answer, bound: it names this call) — or (None, False) when it names another call."""
    if isinstance(answer, dict) and call_id is not None and call_id in answer and not ({"approved", "approve", "id"}
                                                                                       & set(answer)):
        return answer[call_id], True                   # a map {tool_call_id: answer}
    if isinstance(answer, dict) and "id" in answer:
        return (answer, True) if answer["id"] == call_id else (None, False)
    if isinstance(answer, dict) and call_id is not None and any(isinstance(k, str) for k in answer) and not (
            {"approved", "approve"} & set(answer)):
        return None, False                             # a map for other calls
    return answer, False


class _Wrap:
    def __init__(self, guard, facts, on_escalate, declare):
        if not isinstance(guard, Guard):
            raise TypeError("guard is a solvi.agents.Guard")
        if on_escalate not in ("interrupt", "message"):
            raise ValueError('on_escalate: "interrupt" | "message"')
        self.guard, self.facts, self.on_escalate, self.declare = guard, facts, on_escalate, declare
        self.decisions = []
        self._asked = {}                                 # tool call id → the approval keys interrupts showed, in order
        self._done = collections.OrderedDict()           # (thread, the node's task, tool call id, arguments hash) → its result

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
            ok, note = self._ask(request, d)
            d = g.resolve(d, approve=ok, reviewer="langgraph interrupt", execute=False, note=note)
        self.decisions.append(d)
        if d.outcome == "allow":
            return "run", d
        return "reply", ToolMessage(content=d.message(), tool_call_id=tc.get("id"), name=tc["name"], status="error",
                                    artifact={"solvi": d.to_dict()})

    def _ask(self, request, d):
        """Interrupt for a person's answer to this call → (approved, note). Answers that name another call are skipped;
        an answer given for another approval key (the call escalated for other reasons since) interrupts again."""
        from langgraph.types import interrupt
        cid = request.tool_call.get("id")
        key, ah = d.approval_key(), _args_hash(d.arguments)
        several = _calls_in_message(request.state, cid) > 1
        payload = {"solvi": d.to_dict(), "id": cid, "args_hash": ah, "key": key, "reasons": list(d.reasons)}
        asked = self._asked.setdefault(cid, [])
        n = 0
        while True:
            if n >= len(asked):
                asked.append(key)                        # the key of the payload an interrupt at this step shows
            raw = interrupt(payload)
            answer, bound = _for_call(raw, cid)
            if answer is None and not bound:
                continue                                 # another call's answer: wait for this call's own
            said = raw.get("key") if isinstance(raw, dict) and "key" in raw else \
                answer.get("key") if isinstance(answer, dict) and "key" in answer else asked[n]
            said_args = answer.get("args_hash", ah) if isinstance(answer, dict) else ah
            n += 1
            if said != key or said_args != ah:
                continue                                 # given for other reasons or arguments: ask again
            self._asked.pop(cid, None)
            if several and not bound:
                return False, (f"resume value {raw!r} does not name the call: with several calls pending, approve "
                               f'with {{"approved": True, "id": "{cid}"}}')
            ok = approved(answer)
            return ok, None if ok else f"resume value {raw!r}"

    def _once(self, request):
        """(the key of this call, the result it already had) — on resume LangGraph runs the whole node again, so a call
        of the same message that already ran (allowed, or approved while another call waited) is not made twice."""
        tc = request.tool_call
        if not tc.get("id"):
            return None, None
        try:                                             # the node's task: the same across the re-runs of one step
            from langgraph.config import get_config
            conf = get_config().get("configurable") or {}
            task = (conf.get("thread_id"), conf.get("checkpoint_ns"))
        except Exception:  # noqa: BLE001 — not inside a graph run: nothing to re-run
            return None, None
        if task == (None, None):
            return None, None
        k = (*task, tc.get("id"), _args_hash(tc.get("args") or {}))
        return k, self._done.get(k)

    def _keep(self, k, out):
        if k is not None:
            self._done[k] = out
            while len(self._done) > DONE_KEPT:
                self._done.popitem(last=False)
        return out

    def __call__(self, request, execute):
        k, done = self._once(request)
        if done is not None:
            return done
        call, ctx, facts = self._pre(request)
        what, out = self._post(request, self.guard.check(call, ctx, facts))
        return self._keep(k, execute(request)) if what == "run" else out

    async def acall(self, request, execute):
        k, done = self._once(request)
        if done is not None:
            return done
        call, ctx, facts = self._pre(request)
        what, out = self._post(request, await self.guard.acheck(call, ctx, facts))
        return self._keep(k, await execute(request)) if what == "run" else out


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
