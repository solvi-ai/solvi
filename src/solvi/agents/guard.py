"""Guarding an agent's tool calls: the agent proposes a call, solvi checks it and makes it.

    from solvi.agents import Guard

    guard = Guard(storage="calls.db")

    @guard.tool(ground=["iban"])                     # the IBAN must be quoted from the conversation
    def send_payment(iban: str, amount: float, currency: Literal["EUR", "USD"] = "EUR") -> str:
        '''Pay an invoice.'''
        return bank.pay(iban, amount, currency)

    @guard.policy("send_payment")                    # a hard check: False → deny
    def under_hard_cap(amount: float) -> bool:
        return amount <= 10_000

    @guard.policy("send_payment", on_fail="escalate")
    def within_daily_budget(amount: float, spent_today: float) -> bool:   # spent_today: a fact the app gives
        return amount + spent_today <= 2_000

    d = guard.call({"name": "send_payment", "arguments": {"iban": "DE89…", "amount": 250}},
                   context=messages, facts={"spent_today": 400.0})
    d.outcome        # "allow" (solvi ran send_payment and d.result is its return value), "deny" or "escalate"
    d.reasons        # why, in words; d.message() is the text for the model
    d.audit()        # the solvi audit of the decision; d.response is the whole Response (trace, replay)

A proposed call is data — `{"name": tool, "arguments": {...}}` (OpenAI, Anthropic and MCP shapes are read too, see
ToolCall.parse) — never code: solvi looks the tool up in its catalog, and only a registered function ever runs.

Each tool is a small solvi System with one question, `verdict` ∈ {allow, deny, escalate}; the checks of a call are its
catalog, in this order (a failed hard check decides; when several fail, the first in this order):

  arguments_valid              the arguments validate against the tool's types (pydantic; unknown arguments are errors) → deny
  arguments_grounded           every `ground=` argument is in the conversation as a whole word or number token (a quote
                               with offsets; an empty string never is) → deny
  no_injected_arguments        ... and not only in a tool output that carries instruction-like text (solvi.perturb) → escalate
  no_instructions_in_tool_outputs   tools with injections="any": no tool output in the conversation carries such text → escalate
  your policies                ordinary solvi hard checks over the arguments and the facts your app gives (deny first,
                               then escalate); `guard.fn` adds computations they read
  request_authorizes           with an authorizer (a decider's yes / no, act_guard, perturb): "does the conversation
                               authorize this call?" — no → escalate; an escalated or unsure decider → escalate

The rule `verdict` answers "allow" with the grounded arguments as its evidence (each quote is checked again by solvi's
grounding: literally at its offsets). A question that abstains (a check could not be evaluated, a fact the policies
need was not given, the decider escalated) is an escalation. Every decision is a full solvi response: stored with its
trace and the outcome in a TraceStorage (hash-chained), replayable (`guard.replay(id)`), with the audit.

Given facts of a call (the input of its trace): tool_name, tool_arguments (as proposed), conversation (the context as one
text), conversation_roles ([[start, end, role]] of each message in it), user_request (the user's messages) and the facts
your app passes (`facts=`). Computed facts: argument_errors, call_arguments (the validated arguments), one fact per
argument (its validated value, named after the argument), grounding (tools with ground=), proposal (the call as text)."""
from __future__ import annotations

import dataclasses
import inspect
import json
import re
from typing import Any, Callable

from ..core import Answer, Catalog, Claim, Question, Quote

VERDICTS = ("allow", "deny", "escalate")
GIVEN = ("tool_name", "tool_arguments", "conversation", "conversation_roles", "user_request")
BUILTIN = ("argument_errors", "call_arguments", "grounding", "proposal", "arguments_valid", "arguments_grounded",
           "schema_error", "schema_readable",
           "no_injected_arguments", "no_instructions_in_tool_outputs", "request_authorizes", "verdict", "tools_known",
           "known_tool")
ROLES = {"user": "user", "human": "user", "assistant": "assistant", "ai": "assistant", "model": "assistant",
         "tool": "tool", "function": "tool", "function_call_output": "tool", "tool_result": "tool", "system": "system",
         "developer": "system"}
AUTHORIZE_TASK = "Does the conversation authorize this tool call — did the user ask for this action, with these values?"


# ------------------------------------------------------------------------------------------------ the proposal
@dataclasses.dataclass
class ToolCall:
    """A proposed tool call: the tool's name and its arguments (a dict; a JSON string is parsed), and an id if the agent
    gave one."""
    name: str
    arguments: Any
    id: str | None = None

    @classmethod
    def parse(cls, obj) -> "ToolCall":
        """{"name", "arguments"} (MCP, a plain dict), {"type": "function", "function": {"name", "arguments": "<json>"}}
        and {"id", "name", "args"} (OpenAI-style, LangChain), {"type": "tool_use", "name", "input"} (Anthropic), or
        a ToolCall. Arguments given as a JSON string are parsed; a string that is not JSON stays a string (and fails
        validation)."""
        if isinstance(obj, ToolCall):
            return obj
        if not isinstance(obj, dict):
            name, args = getattr(obj, "name", None), getattr(obj, "arguments", getattr(obj, "args", None))
            if name is None:
                raise TypeError(f"a tool call is a dict {{'name', 'arguments'}}, not {type(obj).__name__}")
            obj = {"name": name, "arguments": args, "id": getattr(obj, "id", None)}
        f = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = f.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"a tool call needs a tool name: {obj!r}")
        args = next((f[k] for k in ("arguments", "args", "input", "parameters") if k in f), {})
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except ValueError:
                pass
        return cls(name, {} if args is None else args, obj.get("id") or obj.get("call_id") or obj.get("tool_call_id"))

    def to_dict(self):
        return {"name": self.name, "arguments": self.arguments, **({"id": self.id} if self.id else {})}


def messages(context) -> list:
    """A conversation → [(role, text)]: a string (one user message); a list of {"role", "content"} dicts (OpenAI,
    Anthropic, MCP-style; content a string or a list of parts with "text"; {"type": "function_call_output", "output"}
    items are tool outputs), (role, text) pairs, or message objects with .type / .role and .content (LangChain). Roles
    are normalized to user, assistant, tool and system; anything unreadable is skipped.

    A content list is read block by block: an Anthropic {"type": "tool_result"} block (or any "*_tool_result") is a tool
    output even inside a "user" message, and a {"type": "tool_use"} block (or "function_call") is the assistant's —
    never the user's words, so neither grounds a `ground_from=("user",)` argument, and a tool result gets the injection
    checks. Consecutive blocks of the same role make one message."""
    if context is None:
        return []
    if isinstance(context, str):
        return [("user", context)]
    out = []
    for m in context:
        if isinstance(m, (tuple, list)) and len(m) == 2:
            role, content = m
        elif isinstance(m, dict):
            if m.get("type") == "function_call_output":
                role, content = "tool", m.get("output")
            else:
                role, content = m.get("role") or m.get("type"), m.get("content")
        else:
            role = getattr(m, "role", None) or getattr(m, "type", None)
            content = getattr(m, "content", None)
        role = ROLES.get(str(role).lower()) if role is not None else None
        for r, text in _blocks(role, content):
            if r is not None and text:
                out.append((r, text))
    return out


def _block_role(kind):
    """The role of a content block by its type: a tool result is the tool's, a tool use the assistant's, else None (the
    message's own role)."""
    if not isinstance(kind, str):
        return None
    if kind.endswith("tool_result") or kind == "function_call_output":
        return "tool"
    if kind.endswith("tool_use") or kind == "function_call":
        return "assistant"
    return None


def _blocks(role, content):
    """A message's content → [(role, text)]: one part for a string; a list of blocks split where a block's type gives it
    another role (tool_result → tool, tool_use → assistant), consecutive blocks of one role joined."""
    if not isinstance(content, (list, tuple)):
        return [(role, _text(content))]
    out = []
    for c in content:
        kind = c.get("type") if isinstance(c, dict) else getattr(c, "type", None)
        r = _block_role(kind) or role
        if r == "assistant" and _block_role(kind) == "assistant":
            get = c.get if isinstance(c, dict) else (lambda k, _c=c: getattr(_c, k, None))
            args = next((get(k) for k in ("input", "arguments", "args") if get(k) is not None), {})
            t = f"{get('name')}({args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)})"
        elif isinstance(c, dict) and _block_role(kind) == "tool":
            t = _text(c.get("content") if "content" in c else c.get("output"))
        elif _block_role(kind) == "tool":
            t = _text(getattr(c, "content", None) if getattr(c, "content", None) is not None
                      else getattr(c, "output", None))
        else:
            t = _text([c])
        if not t:
            continue
        if out and out[-1][0] == r:
            out[-1] = (r, out[-1][1] + "\n" + t)
        else:
            out.append((r, t))
    return out


def _text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for c in content:
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, dict):
                t = c.get("text") if c.get("text") is not None else c.get("content")
                if isinstance(t, (str, list)):
                    parts.append(_text(t))
            elif getattr(c, "text", None) is not None:
                parts.append(str(c.text))
        return "\n".join(p for p in parts if p)
    if isinstance(content, dict):
        return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)
    return str(content)


def conversation(context):
    """A conversation → (text, [[start, end, role]], the user's messages as one text). A plain string is the text itself
    (one user message); messages are written one per line as "[role] text"."""
    if isinstance(context, str):
        return context, [[0, len(context), "user"]], context
    parts, roles, at = [], [], 0
    for role, text in messages(context):
        head = f"[{role}] "
        s = at + len(head)
        roles.append([s, s + len(text), role])
        parts.append(head + text)
        at = s + len(text) + 1
    text = "\n".join(parts)
    return text, roles, "\n".join(text[s:e] for s, e, r in roles if r == "user")


# ------------------------------------------------------------------------------------------------ tools
def _is_context_param(p):
    """A framework's context argument (RunContext, ToolContext, RunContextWrapper, ...): not an argument of the call."""
    a = p.annotation
    name = getattr(a, "__name__", None) or (a if isinstance(a, str) else str(a))
    return "Context" in str(name) and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)


def arguments_model(name, func):
    """A function's parameters → a pydantic model of its arguments (unknown arguments forbidden). A first parameter typed
    as a framework context (RunContext, ToolContext, ...) is not an argument; *args / **kwargs are not allowed."""
    from pydantic import ConfigDict, create_model
    fields, sig = {}, inspect.signature(func)
    params = list(sig.parameters.values())
    if params and _is_context_param(params[0]):
        params = params[1:]
    try:
        import typing
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:  # noqa: BLE001 — unresolvable forward references: the raw annotations
        hints = {}
    for p in params:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise ValueError(f"tool {name}: *{p.name} / **{p.name} cannot be checked; declare every argument")
        t = hints.get(p.name, p.annotation if p.annotation is not p.empty else Any)
        fields[p.name] = (t, ... if p.default is p.empty else p.default)
    return create_model(_camel(name) + "Arguments", __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True,
                                                                          allow_inf_nan=False), **fields)


def model_from_json_schema(name, schema):
    """A JSON schema of an object (an MCP tool's inputSchema, an OpenAI function's parameters) → a pydantic model: string,
    integer, number (finite: NaN and infinities are refused), boolean, null, array (items), object (properties → a
    nested model, else a dict), enum / const (Literal), anyOf / oneOf / type lists (a Union), $ref to $defs /
    definitions, defaults and required. A recursive $ref is followed once: inside itself it is any object (a dict) — the
    model stays finite. Anything else is Any. Unknown arguments are forbidden."""
    import typing

    from pydantic import ConfigDict, Field, create_model
    defs = dict(schema.get("$defs") or schema.get("definitions") or {})

    def ref_of(s):
        ref = s.get("$ref") if isinstance(s, dict) else None
        return ref.rsplit("/", 1)[-1] if isinstance(ref, str) and ref.rsplit("/", 1)[-1] in defs else None

    def resolve(s, seen):
        """→ (the schema a $ref points to, the refs followed so far) — None when the ref is already being followed."""
        hops = 0
        while (r := ref_of(s)) is not None:
            if r in seen or hops > len(defs):
                return None, seen
            seen, s, hops = seen | {r}, defs[r], hops + 1
        return (s if isinstance(s, dict) else {}), seen

    def typ(s, path, seen):
        s, seen = resolve(s, seen)
        if s is None:                                     # a recursive reference: any JSON object from here on
            return dict
        if "const" in s:
            return typing.Literal[s["const"]]
        if isinstance(s.get("enum"), list) and s["enum"]:
            return typing.Literal[tuple(s["enum"])]
        alts = s.get("anyOf") or s.get("oneOf")
        if isinstance(alts, list) and alts:
            ts = tuple(typ(a, path, seen) for a in alts)
            return ts[0] if len(ts) == 1 else typing.Union[ts]
        t = s.get("type")
        if isinstance(t, list):
            ts = tuple(typ({**s, "type": x}, path, seen) for x in t)
            return ts[0] if len(ts) == 1 else typing.Union[ts]
        if t == "string":
            return str
        if t == "integer":
            return int
        if t == "number":
            return float
        if t == "boolean":
            return bool
        if t == "null":
            return type(None)
        if t == "array":
            return list[typ(s.get("items") or {}, path, seen)]
        if t == "object" or "properties" in s:
            if s.get("properties"):
                return obj(s, path, seen)
            return dict
        return Any

    def obj(s, path, seen):
        req = set(s.get("required") or ())
        fields = {}
        for k, p in (s.get("properties") or {}).items():
            r, _ = resolve(p, seen)
            default = ... if k in req else (r or {}).get("default", None)
            fields[k] = (typ(p, path + [k], seen), Field(default, description=(r or {}).get("description")))
        return create_model(_camel("_".join([name] + path)) + ("Arguments" if not path else ""),
                            __config__=ConfigDict(extra="forbid", allow_inf_nan=False), **fields)
    top, seen = resolve(schema, frozenset())
    return obj(top or {}, [], seen)


def permissive_model(name):
    """The arguments' model of a tool whose schema could not be read: any JSON object (the guard escalates its calls)."""
    from pydantic import ConfigDict, create_model
    return create_model(_camel(name) + "Arguments", __config__=ConfigDict(extra="allow", allow_inf_nan=False))


def _camel(name):
    s = "".join(w[:1].upper() + w[1:] for w in str(name).replace("-", "_").replace(".", "_").split("_") if w)
    return s if s.isidentifier() else "Tool"


@dataclasses.dataclass
class Tool:
    """A tool in the guard's catalog: its name, the function solvi runs when a call is allowed (None: the framework runs
    it — adapters, the MCP proxy), the pydantic model of its arguments (None until a schema is known — MCP), which
    arguments must be quoted from the conversation, and from which roles."""
    name: str
    func: Callable | None
    model: Any
    description: str = ""
    ground: dict = dataclasses.field(default_factory=dict)     # argument → roles it may be quoted from
    injections: str = "grounded"                               # "grounded" | "any" | "off"
    authorize: bool | None = None                              # ask the guard's authorizer (None: when it has one)
    match: dict = dataclasses.field(default_factory=dict)      # argument → how its value is found ("token" when absent)
    schema_error: str | None = None                            # adopt: the schema could not be read (its calls escalate)

    @property
    def arguments(self):
        return list(self.model.model_fields) if self.model is not None else []

    def json_schema(self):
        """The arguments' JSON schema (the function-calling definition's "parameters")."""
        return self.model.model_json_schema() if self.model is not None else {"type": "object"}

    def definition(self):
        """{"name", "description", "parameters"}: the tool as a function-calling definition."""
        return {"name": self.name, "description": self.description, "parameters": self.json_schema()}


# ------------------------------------------------------------------------------------------------ the parts of a call
def _argument_errors(model, schema):
    def argument_errors(tool_arguments) -> list:
        """What is wrong with the proposed arguments: [] when they validate against the tool's types."""
        _ = schema                                        # the schema's text: part of this function's fingerprint
        if not isinstance(tool_arguments, dict):
            return [f"the arguments are not a JSON object: {tool_arguments!r:.120}"]
        from pydantic import ValidationError
        try:
            model.model_validate(tool_arguments)
        except ValidationError as e:
            return [f"{'.'.join(str(x) for x in err['loc']) or '(arguments)'}: {err['msg']}"   # solvi: ok
                    + (f" (got {err['input']!r:.80})" if "input" in err and err["type"] != "missing" else "")
                    for err in e.errors()]
        return []
    return argument_errors


def _call_arguments(model, schema):
    def call_arguments(tool_arguments) -> dict:
        """The proposed arguments, validated and coerced to the tool's types (defaults filled in)."""
        _ = schema
        m = model.model_validate(tool_arguments)
        return {**{k: getattr(m, k) for k in type(m).model_fields}, **(m.model_extra or {})}   # solvi: ok
    return call_arguments


def _argument(name, annotation):
    def f(call_arguments):
        return call_arguments[name]
    f.__name__ = f.__qualname__ = name
    f.__doc__ = f"The validated argument {name!r} of the call."
    if annotation is not None and annotation is not Any:
        f.__annotations__ = {"return": annotation}
    return f


def _grounding(spec, matchers=None):
    matchers = dict(matchers or {})                       # argument → a callable matcher (its code is in `spec`)

    def grounding(call_arguments, conversation, conversation_roles) -> dict:
        """Where each argument that must come from the conversation is quoted: {"found": {argument: [[text, start, end,
        role]]}, "missing": [...], "injected": [...]}. A string is found as a whole word (not inside a longer word; see
        MATCHERS), a number as a number token (thousands separators allowed; not a group of a spaced identifier), a list
        item by item; an empty or whitespace-only string is never grounded. The first occurrence in a message of an
        allowed role wins, one in a tool output with instruction-like text (solvi.perturb) only when there is no other."""
        from ..perturb import instruction_spans
        rules = json.loads(spec)
        roles = [(s, e, r) for s, e, r in conversation_roles]
        tainted = {}

        def taint(i):
            if i not in tainted:
                s, e, r = roles[i]
                sp = instruction_spans(conversation[s:e], actions=True) if r == "tool" else []
                tainted[i] = [conversation[s + a:s + b] for a, b in sp]
            return tainted[i]

        def where(a, b):
            for i, (s, e, _) in enumerate(roles):
                if s <= a and b <= e:
                    return i
            return None
        found, missing, injected = {}, [], []
        for arg, allowed in rules["roles"].items():
            v = call_arguments.get(arg)
            if v is None or v == [] or v == ():
                continue
            items = list(v) if isinstance(v, (list, tuple, set, frozenset)) and not isinstance(v, str) else [v]
            match = matchers[arg] if arg in matchers else rules["match"][arg]
            quotes = []
            for item in items:
                if isinstance(item, str) and not item.strip():
                    missing.append(f"{arg}={_short(item)} (empty)")
                    continue
                best = None
                for a, b in _occurrences(item, conversation, match):
                    i = where(a, b)
                    if i is None or roles[i][2] not in allowed:
                        continue
                    cand = [conversation[a:b], a, b, roles[i][2], taint(i)]
                    if not cand[4]:
                        best = cand
                        break
                    best = best or cand
                if best is None:
                    missing.append(f"{arg}={_short(item)}")
                    continue
                if best[4]:
                    injected.append(f"{arg}={_short(item)} appears only in a tool output that says "
                                    + "; ".join(_short(x, 80) for x in best[4]))
                quotes.append(best[:4])
            if quotes:
                found[arg] = quotes
        return {"found": found, "missing": missing, "injected": injected}
    return grounding


_NUMBER = re.compile(r"(?<![\w.,])-?(?:\d{1,3}(?:[,\u00a0\u202f' ]\d{3})+(?![\d])|\d+)(?:\.\d+)?(?![\w]|[.,]\d)")
_WHOLE_EDGE = set(" \t\r\n\"'`()[]{}<>,;:!?")
MATCHERS = ("token", "whole", "substring")


def _glued(text, a, b):
    """Is the number at text[a:b] one group of a longer identifier — next to a token with a digit across a single " ",
    "-" or "/" ("DE89 3704 0044", "555-1234", "2024-03-15")?"""
    if a >= 2 and text[a - 1] in " -/":
        k = a - 1
        while k > 0 and text[k - 1].isalnum():
            k -= 1
        if any(c.isdigit() for c in text[k:a - 1]):
            return True
    if b + 1 < len(text) and text[b] in " -/":
        k = b + 1
        while k < len(text) and text[k].isalnum():
            k += 1
        if any(c.isdigit() for c in text[b + 1:k]):
            return True
    return False


def _bounded(text, a, b, s, match):
    """Does the occurrence text[a:b] of the string s stand on its own under the matcher?"""
    if match == "substring":
        return True
    before, after = text[a - 1] if a > 0 else "", text[b] if b < len(text) else ""
    if match == "whole":                 # delimited by whitespace, quotes, brackets or punctuation (a "." ends a sentence)
        ok_before = before == "" or before in _WHOLE_EDGE
        ok_after = after == "" or after in _WHOLE_EDGE or (after == "." and (b + 1 >= len(text) or text[b + 1].isspace()))
        return ok_before and ok_after
    word = re.compile(r"\w")                                 # "token": not inside a longer word
    return not (word.match(s[0]) and before and word.match(before)) and not (word.match(s[-1]) and after
                                                                              and word.match(after))


def _occurrences(v, text, match="token"):
    """Where a value is written in a text → [(start, end)]: a string as a whole word under `match` ("token": not inside a
    longer word — "DE8937" is not found in "DE89370400…"; "whole": delimited by whitespace, quotes, brackets or
    punctuation — "x.org" is not found in "alice@x.org"; "substring": anywhere; a callable(value, text) → [(start,
    end)] decides itself); a number as a number token (not inside a word; thousands separators "1,250.50", "1 250",
    "1'250" allowed; 250 matches "250.00"; not a group of a spaced or dashed identifier — 3704 is not found in "DE89 3704
    0044"); an Enum by its value; anything else by str(). An empty or whitespace-only string is found nowhere."""
    import enum
    if isinstance(v, enum.Enum):
        v = v.value
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        out = []
        for m in _NUMBER.finditer(text):
            try:
                x = float(re.sub(r"[,\u00a0\u202f' ]", "", m.group(0)))
            except ValueError:
                continue
            if abs(x - float(v)) <= 1e-9 * max(1.0, abs(float(v))) and not _glued(text, m.start(), m.end()):
                out.append((m.start(), m.end()))
        return out
    s = str(v)
    if not s.strip():
        return []
    if callable(match):
        return [(int(a), int(b)) for a, b in (match(v, text) or ())
                if 0 <= int(a) < int(b) <= len(text)]
    out, i = [], text.find(s)
    while i >= 0:
        if _bounded(text, i, i + len(s), s, match):
            out.append((i, i + len(s)))
        i = text.find(s, i + 1)
    return out


def _short(v, n=60):
    """A value for a reason: a string quoted (cut at n characters), anything else as its repr."""
    if not isinstance(v, str):
        s = repr(v)
        return s if len(s) <= n else s[: n - 1] + "…"
    return repr(v) if len(v) <= n else repr(v[: n - 1] + "…")


def _schema_error(text):
    def schema_error(tool_name) -> str:
        """Why the tool's input schema could not be read (adopt): its arguments are not checked against types."""
        return text
    return schema_error


def schema_readable(schema_error) -> bool:
    """The tool's input schema could be read; a tool whose schema could not be read needs a person for every call."""
    return not schema_error


def arguments_valid(argument_errors) -> bool:
    """The arguments validate against the tool's types."""
    return not argument_errors


def arguments_grounded(grounding) -> bool:
    """Every argument that must come from the conversation is quoted there."""
    return not grounding["missing"]


def no_injected_arguments(grounding) -> bool:
    """No argument comes only from a tool output that carries instruction-like text."""
    return not grounding["injected"]


def no_instructions_in_tool_outputs(conversation, conversation_roles) -> bool:
    """No tool output in the conversation carries instruction-like text (solvi.perturb)."""
    from ..perturb import instruction_spans
    return not any(r == "tool" and instruction_spans(conversation[s:e], actions=True) for s, e, r in conversation_roles)


def request_authorizes(authorized) -> bool:
    """The authorizer (a decider) says the conversation authorizes this call."""
    return authorized is True


def proposal(tool_name, call_arguments) -> str:
    """The call as the authorizer reads it: the tool's name and its arguments as JSON."""
    from ..decide import jsonable
    return f"{tool_name}({json.dumps(jsonable(call_arguments), ensure_ascii=False, sort_keys=True, default=str)})"


def verdict_of(call_arguments):
    return "allow"


def grounded_verdict(call_arguments, grounding):
    """allow, with the quotes of the grounded arguments as evidence (solvi checks each is literally at its offsets)."""
    ev = [Quote(t, s, e, "conversation") for qs in grounding["found"].values() for t, s, e, _ in qs]
    return Claim("allow", evidence=ev, source="conversation") if ev else "allow"


def known_tool(tool_name, tools_known) -> bool:
    """The tool is in the guard's catalog."""
    return tool_name in tools_known


def unknown_verdict(tool_name):
    return "allow"


# ------------------------------------------------------------------------------------------------ the decision
@dataclasses.dataclass
class GuardDecision:
    """The guard's decision on one proposed call. outcome: "allow" | "deny" | "escalate"; reasons: why, in words (empty
    for an allowed call); call: the candidate call (the validated arguments when they validate); response: the solvi
    Response (answers, flow, trace — `response.trace.replay(...)`, `audit()`). For an allowed call made by solvi:
    executed, result (the tool's return value) or error. stored_id / trace_hash: where it is in the guard's store."""
    outcome: str
    tool: str
    arguments: Any
    reasons: list
    response: Any
    evidence: list = dataclasses.field(default_factory=list)   # [(argument, quoted text, start, end, role)]
    executed: bool = False
    result: Any = None
    error: str | None = None
    stored_id: str | None = None
    approved_by: str | None = None                             # guard.resolve: the person who approved an escalation
    id: str | None = None                                      # the agent's id of the call, if it gave one
    catalog: Any = dataclasses.field(default=None, repr=False)

    @property
    def allowed(self):
        return self.outcome == "allow"

    @property
    def trace_hash(self):
        from ..serve import trace_hash
        return trace_hash(self.response)

    @property
    def call(self):
        """The candidate call: {"name", "arguments"}."""
        return {"name": self.tool, "arguments": self.arguments}

    def message(self):
        """The text for the model (or a person): what happened and why."""
        if self.outcome == "allow":
            if self.error:
                return f"{self.tool} failed: {self.error}"
            return f"{self.tool} allowed" + (f" (approved by {self.approved_by})" if self.approved_by else "")
        head = "denied" if self.outcome == "deny" else "escalated to a person for approval (not executed)"
        return f"{self.tool} {head}: " + "; ".join(self.reasons)

    def audit(self, lang=None):
        """The solvi audit of the verdict (what it rests on, the checks, the safeguards that fired)."""
        return self.response.audit("verdict", lang=lang)

    def replay(self):
        """Re-compute the decision's trace from its recorded inputs → solvi's replay report ({"ok", "mismatches", ...})."""
        return self.response.trace.replay(self.catalog)

    def to_dict(self):
        """The decision as JSON data (without the trace: `response.to_dict()` has it)."""
        from ..decide import jsonable
        d = {"outcome": self.outcome, "tool": self.tool, "arguments": jsonable(self.arguments), "reasons": self.reasons,
             "evidence": [list(e) for e in self.evidence], "executed": self.executed, "error": self.error,
             "stored_id": self.stored_id, "trace_hash": self.trace_hash}
        if self.approved_by:
            d["approved_by"] = self.approved_by
        if self.id:
            d["id"] = self.id
        return d


# ------------------------------------------------------------------------------------------------ the guard
class Guard:
    """The catalog of tools an agent may call, the policies over their calls, and the store of every decision.

    storage: a TraceStorage or a path (.db / .sqlite: SQLite, else JSON lines) — every decision is saved with its trace
    and the outcome (meta "guard": tool, outcome, reasons, executed, the result's hash or the error).
    authorizer: an optional decision part answering "does the conversation authorize this call?" (bool) over the facts
    "conversation" (or "user_request") and "proposal" — see `make_authorizer()`; its act_guard threshold and perturb=k
    apply. facts: names (or {name: type}) of facts your app gives with every call (a user's role, a budget left): policies
    that read them apply to every tool without naming it (the types are for readers: a policy's own annotations are what
    solvi validates)."""

    def __init__(self, storage=None, authorizer=None, facts=None, lang="en"):
        from ..storage import open_storage
        self.storage = open_storage(storage)
        self.tools: dict[str, Tool] = {}
        self.facts = dict(facts) if isinstance(facts, dict) else {f: Any for f in (facts or ())}
        self._policies = []                  # [(func, tools or None, on_fail)]
        self._fns = []                       # [(func, tools or None)]
        self._authorizer = authorizer
        self._systems = {}
        self._unknown = None
        self.lang = lang

    # --- the catalog
    def tool(self, func=None, *, name=None, schema=None, description=None, ground=(), ground_from=("user", "tool", "system"),
             injections="grounded", authorize=None):
        """Declare a tool the agent may call. As a decorator on a typed function (`@guard.tool`, `@guard.tool(ground=[...])`),
        or `guard.tool(name="refund", schema=RefundArgs)` (a pydantic model or a JSON schema) for a tool the framework or
        an MCP server runs. The function is returned unchanged.

        ground: arguments that must be quoted from the conversation (a string as a whole word; numbers as number tokens;
        a list item by item; an empty string never) — a list of names, or {name: matcher}: "token" (the default: not
        inside a longer word), "whole" (delimited by whitespace, quotes, brackets or punctuation: for IBANs, e-mails,
        paths), "substring" (anywhere), or a callable(value, text) → [(start, end)] of the value's occurrences;
        ground_from: the roles of the messages they may be quoted from (default: the user's, tool outputs and system
        messages — never the assistant's own words; ("user",) for values only the user may give, like a payee).
        injections: "grounded" (default: a grounded argument found only in a tool output with instruction-like text
        escalates), "any" (also: any instruction-like text in a tool output escalates the call — for high-impact
        tools), "off". authorize: ask the guard's authorizer about this tool (default: when the guard has one)."""
        def add(f):
            n = name or (f.__name__ if f is not None else None)
            if not n:
                raise ValueError("a tool needs a name")
            if schema is None and f is not None:
                model = arguments_model(n, f)
            elif schema is None:
                model = None
            elif isinstance(schema, dict):
                model = model_from_json_schema(n, schema)
            else:
                model = schema
            desc = description if description is not None else ((inspect.getdoc(f) or "") if f is not None else "")
            roles = tuple(ROLES.get(r, r) for r in ((ground_from,) if isinstance(ground_from, str) else ground_from))
            spec = {ground: "token"} if isinstance(ground, str) else dict(ground) if isinstance(ground, dict) \
                else {a: "token" for a in ground}
            bad = {a: m for a, m in spec.items() if not callable(m) and m not in MATCHERS}
            if bad:
                raise ValueError(f"tool {n}: ground= matchers are {', '.join(MATCHERS)} or a callable, not {bad}")
            t = Tool(n, f, model, desc.strip(), {a: roles for a in spec}, injections, authorize,
                     {a: m for a, m in spec.items() if m != "token"})
            if injections not in ("grounded", "any", "off"):
                raise ValueError('injections must be "grounded", "any" or "off"')
            self._check_tool(t)
            self.tools[n] = t
            self._systems.pop(n, None)
            self._unknown = None
            return f
        return add if func is None else add(func)

    def declare(self, name, schema=None, description="", **kw):
        """Declare a tool that solvi does not run (the framework or an MCP server does): its name, the arguments' schema
        (a pydantic model or a JSON schema; None — given later by `adopt`, as the MCP proxy does from tools/list) and
        the options of `tool` (ground, ground_from, injections, authorize). → the Tool."""
        self.tool(name=name, schema=schema, description=description, **kw)(None)  # pyright: ignore[reportOptionalCall]
        return self.tools[name]

    def adopt(self, name, json_schema, description=""):
        """Give a declared tool without a schema (`guard.tool(name=...)`) its arguments' JSON schema — the MCP proxy
        does this from the server's tools/list. A tool that has a schema keeps it."""
        t = self.tools[name]
        if t.model is None:
            try:
                t.model = model_from_json_schema(name, json_schema if isinstance(json_schema, dict) else {"type": "object"})
                t.schema_error = None
            except Exception as e:  # noqa: BLE001 — one bad schema must not break the others: this tool's calls escalate
                import logging
                t.model, t.schema_error = permissive_model(name), f"{type(e).__name__}: {str(e)[:200]}"
                logging.getLogger("solvi.agents").warning("tool %s: its input schema could not be read (%s); its calls "
                                                          "escalate", name, t.schema_error)
            t.description = t.description or description or ""
            self._check_tool(t)
            self._systems.pop(name, None)
        return t

    def _check_tool(self, t):
        if t.model is None:
            return
        bad = [a for a in t.arguments if a in GIVEN or a in BUILTIN or a in self.facts]
        if bad:
            raise ValueError(f"tool {t.name}: argument(s) {bad} collide with the facts the guard gives or computes "
                             f"({', '.join(GIVEN + tuple(self.facts))}); rename them")
        unknown = [a for a in t.ground if a not in t.arguments]
        if unknown:
            raise ValueError(f"tool {t.name}: ground= names no argument: {unknown} (its arguments: {t.arguments})")

    def policy(self, tools=None, *, on_fail="deny"):
        """A policy over calls: an ordinary solvi hard check — argument names are the facts it reads (the call's
        validated arguments, the facts your app gives, conversation, user_request, tool_name), it returns True when
        the call may go ahead. on_fail: "deny" or "escalate". tools: a tool name or a list of them; None — every tool
        whose arguments and the guard's declared facts provide what it reads. Its docstring's first line is the reason
        given when it fails."""
        if on_fail not in ("deny", "escalate"):
            raise ValueError('on_fail must be "deny" or "escalate"')

        def add(f):
            self._policies.append((f, _names(tools), on_fail))
            self._systems.clear()
            return f
        if callable(tools) and not isinstance(tools, str):       # @guard.policy without arguments
            f, tools = tools, None
            return add(f)
        return add

    def fn(self, f=None, *, tools=None):
        """A computation the policies read (an ordinary solvi fn: `amount_eur(amount, currency)`), for the given tools
        (None: every tool that provides its inputs)."""
        def add(g):
            self._fns.append((g, _names(tools)))
            self._systems.clear()
            return g
        return add if f is None else add(f)

    @property
    def authorizer(self):
        return self._authorizer

    @authorizer.setter
    def authorizer(self, part):
        self._authorizer = part
        self._systems.clear()

    def make_authorizer(self, decider, task=AUTHORIZE_TASK, reads="conversation", perturb=2, **kw):
        """A decider's yes / no question "does the conversation authorize this call?" over `reads` ("conversation": the
        whole context, tool outputs included; "user_request": only the user's messages) and the proposed call as text,
        with perturb=k (re-asked without instruction-like sentences; a changed answer escalates) — set as the guard's
        authorizer and returned. Calibrate it with `guard.calibrate_authorizer(examples, risk=0.10)`."""
        if reads not in ("conversation", "user_request"):
            raise ValueError('reads must be "conversation" or "user_request"')
        part = decider.decision("authorized", task, [reads, "proposal"], type=bool, perturb=perturb, **kw)
        self.authorizer = part
        return part

    def authorizer_input(self, call, context=None):
        """The input the authorizer reads for a call (decide.Facts) — for fit / act_guard / conformal examples."""
        from ..decide import Facts
        c = ToolCall.parse(call)
        text, _, request = conversation(context)
        t = self.tools.get(c.name)
        args = c.arguments
        if t is not None and t.model is not None and isinstance(args, dict):
            try:
                args = _call_arguments(t.model, "")(args)
            except Exception:  # noqa: BLE001, S110 — an invalid call: the authorizer reads it as proposed
                pass
        return Facts(conversation=text, user_request=request, proposal=proposal(c.name, args))

    def calibrate_authorizer(self, examples, risk=0.10, **kw):
        """act_guard on the authorizer from labelled calls [(call, context, authorized: bool)]: P(allowed by the
        authorizer alone and wrong) ≤ risk for calls like these (solvi.decide.DecisionPart.act_guard). → its report."""
        if self._authorizer is None:
            raise ValueError("the guard has no authorizer: make_authorizer(decider) first")
        ex = [(self.authorizer_input(c, ctx), bool(y)) for c, ctx, y in examples]
        out = self._authorizer.act_guard(ex, risk=risk, **kw)
        self._systems.clear()
        return out

    # --- the systems
    def system(self, name):
        """The solvi System that checks calls of one tool (built on first use; rebuilt after a declaration changes)."""
        if name not in self.tools:
            return self._unknown_system()
        s = self._systems.get(name)
        if s is None:
            s = self._systems[name] = self._build(self.tools[name])
        return s

    def catalog(self, name):
        """The solvi Catalog of one tool's checks."""
        return self.system(name).catalog

    def _build(self, t):
        from ..provenance import code_fingerprint
        from ..system import System
        if t.model is None:
            raise ValueError(f"tool {t.name} has no argument schema yet (guard.tool(..., schema=...) or guard.adopt)")
        cat = Catalog()
        schema = json.dumps(t.model.model_json_schema(), sort_keys=True, default=str)
        have = set(GIVEN) | set(self.facts) | set(t.arguments) | {"argument_errors", "call_arguments", "grounding",
                                                                  "proposal"}
        fns = []                                  # the helper computations and policies this tool gets
        for f, tools in self._fns:
            ins = set(inspect.signature(f).parameters)
            if (tools is None and ins <= have) or (tools is not None and t.name in tools):
                fns.append(f)
                have.add(f.__name__)
        policies = [(f, of) for of in ("deny", "escalate") for f, tools, o in self._policies if o == of
                    and ((tools is None and set(inspect.signature(f).parameters) <= have)
                         or (tools is not None and t.name in tools))]
        read = {x for f in fns + [f for f, _ in policies] for x in inspect.signature(f).parameters}
        cat.fn(_argument_errors(t.model, schema))
        cat.fn(_call_arguments(t.model, schema))
        for a, fi in t.model.model_fields.items():       # an argument is a fact of its own when something reads it
            if a in read:
                cat.fn(_argument(a, fi.annotation))
        checks = []

        def check(f, on_fail):
            cat.check(hard=True, then={"verdict": on_fail})(f)
            checks.append(f.__name__)
        if t.schema_error is not None:
            cat.fn(_schema_error(t.schema_error))
            check(schema_readable, "escalate")
        check(arguments_valid, "deny")
        if t.ground:
            match = {a: t.match.get(a, "token") for a in t.ground}
            match = {a: m if isinstance(m, str) else f"callable {getattr(m, '__qualname__', type(m).__name__)} "
                     f"{code_fingerprint(m)}" for a, m in match.items()}        # a callable matcher: by its code
            cat.fn(_grounding(json.dumps({"roles": {a: list(r) for a, r in t.ground.items()}, "match": match},
                                         sort_keys=True), {a: m for a, m in t.match.items() if callable(m)}))
            check(arguments_grounded, "deny")
            if t.injections != "off":
                check(no_injected_arguments, "escalate")
        if t.injections == "any":
            check(no_instructions_in_tool_outputs, "escalate")
        for f in fns:
            cat.fn(f)
        for f, on_fail in policies:
            check(f, on_fail)
        auth = self._authorizer is not None and t.authorize is not False
        if auth:
            cat.fn(proposal)
            cat.fn(self._authorizer)
            check(request_authorizes, "escalate")
        if t.ground:
            cat.rule("verdict")(grounded_verdict)
        else:
            cat.rule("verdict")(verdict_of)
        q = Question("verdict", f"May the agent call {t.name} with these arguments?", Answer.choice(list(VERDICTS)),
                     checkpoints=checks)
        return System(cat, [q], lang=self.lang)

    def _unknown_system(self):
        if self._unknown is None:
            from ..system import System
            cat = Catalog()
            cat.check(hard=True, then={"verdict": "deny"})(known_tool)
            cat.rule("verdict")(unknown_verdict)
            self._unknown = System(cat, [Question("verdict", "Is the tool in the guard's catalog?",
                                                  Answer.choice(list(VERDICTS)), checkpoints=["known_tool"])],
                                   lang=self.lang)
        return self._unknown

    # --- checking and calling
    def _state(self, c, context, facts):
        facts = dict(facts or {})
        text, roles, request = conversation(context)
        state = {"tool_name": c.name, "tool_arguments": c.arguments, "conversation": text, "conversation_roles": roles,
                 "user_request": request}
        clash = [k for k in facts if k in state or k in BUILTIN
                 or (c.name in self.tools and k in self.tools[c.name].arguments)]
        if clash:
            raise ValueError(f"facts {clash} collide with the call's own facts; rename them")
        state.update(facts)
        if c.name not in self.tools:
            state = {"tool_name": c.name, "tool_arguments": c.arguments, "tools_known": sorted(self.tools)}
        return state

    def check(self, call, context=None, facts=None, store=True):
        """Decide on a proposed call without making it → GuardDecision (allow / deny / escalate, with the reasons and
        the solvi response). context: the conversation (see `messages`); facts: what your app knows (a user's role,
        a budget left) — given facts of the decision, recorded in its trace."""
        c = ToolCall.parse(call)
        s = self.system(c.name)
        res = s.ask(self._state(c, context, facts), store=False)
        return self._decision(c, s, res, store)

    async def acheck(self, call, context=None, facts=None, store=True):
        """check, on an event loop (policies may be `async def`: System.aask)."""
        c = ToolCall.parse(call)
        s = self.system(c.name)
        state = self._state(c, context, facts)
        res = await s.aask(state, store=False) if s.is_async else s.ask(state, store=False)
        return self._decision(c, s, res, store)

    def call(self, call, context=None, facts=None):
        """Check a proposed call and, when it is allowed, make it: solvi runs the tool's registered function with the
        validated arguments → GuardDecision with `result` (or `error` when the tool raised). A tool without a function
        (declared for a framework) is not run: `executed` stays False."""
        d = self.check(call, context, facts, store=False)
        if d.allowed:
            self._run(d)
        self._save(d)
        return d

    async def acall(self, call, context=None, facts=None):
        """call, on an event loop: an `async def` tool is awaited."""
        d = await self.acheck(call, context, facts, store=False)
        if d.allowed:
            await self._arun(d)
        self._save(d)
        return d

    def _run(self, d):
        t = self.tools[d.tool]
        if t.func is None:
            return
        d.executed = True
        try:
            r = t.func(**d.arguments)
            if inspect.isawaitable(r):
                from ..runtime import run_sync
                r = run_sync(r)
            d.result = r
        except Exception as e:  # noqa: BLE001 — the tool's own failure: reported, not raised
            d.error = f"{type(e).__name__}: {e}"

    async def _arun(self, d):
        t = self.tools[d.tool]
        if t.func is None:
            return
        d.executed = True
        try:
            r = t.func(**d.arguments)
            d.result = await r if inspect.isawaitable(r) else r
        except Exception as e:  # noqa: BLE001
            d.error = f"{type(e).__name__}: {e}"

    def resolve(self, decision, approve, reviewer=None, note=None, execute=True):
        """A person's answer to an escalated call: recorded in the store as a correction of its verdict (who, the
        stored decision it answers, a note) and, when approved, the call is made (execute=False: not made — the
        framework makes it). → the decision, with outcome "allow" (approved_by) or "deny"."""
        if decision.outcome != "escalate":
            raise ValueError(f"only an escalated call is resolved by a person; this one is {decision.outcome!r}")
        d = dataclasses.replace(decision, outcome="allow" if approve else "deny", approved_by=reviewer if approve else None,
                                reasons=list(decision.reasons) + ([f"approved by {reviewer or 'a person'}"] if approve
                                                                  else [f"rejected by {reviewer or 'a person'}"]
                                                                  + ([note] if note else [])))
        if approve and execute:
            self._run(d)
        if self.storage is not None:
            meta = {"tool": d.tool, "reviewer": reviewer, "note": note, "of": decision.stored_id, "executed": d.executed}
            meta.update(_outcome_meta(d))
            self.storage.save_correction("verdict", decision.response.trace.init, d.outcome, meta={"guard": meta})
        return d

    # --- the decision
    def _decision(self, c, system, res, store):
        r = res["verdict"]
        outcome = r.answer if r.status in ("ok", "forced") and r.answer in VERDICTS else "escalate"
        known = c.name in self.tools
        args = res.values.get("call_arguments", c.arguments) if known else c.arguments
        from ..runtime import MISSING
        if args is MISSING:
            args = c.arguments
        d = GuardDecision(outcome, c.name, args, self._reasons(c, system, res, r, outcome), res, id=c.id,
                          catalog=system.catalog)
        g = res.values.get("grounding") if known else None
        if isinstance(g, dict):
            d.evidence = [(a, q[0], q[1], q[2], q[3]) for a, qs in g.get("found", {}).items() for q in qs]
        if store:
            self._save(d)
        return d

    def _reasons(self, c, system, res, r, outcome):
        if outcome == "allow":
            return []
        if c.name not in self.tools:
            return [f"unknown tool {c.name!r}: the catalog has {sorted(self.tools)}"]
        cat, vals = system.catalog, res.values
        out = []
        order = {n: i for i, n in enumerate(cat.parts)}          # the catalog's order: the order checks decide in
        failed = sorted((r for r in res.trace.records if r.value is False and r.name in cat.parts
                         and cat.parts[r.name].kind == "check" and cat.parts[r.name].hard), key=lambda r: order[r.name])
        for rec in failed:
            p = cat.parts[rec.name]
            n = p.name
            if n == "arguments_valid":
                out.append("invalid arguments: " + "; ".join(vals.get("argument_errors") or []))
            elif n == "arguments_grounded":
                out.append("not in the conversation: " + ", ".join(vals["grounding"]["missing"]))
            elif n == "no_injected_arguments":
                out.append("; ".join(vals["grounding"]["injected"]))
            elif n == "schema_readable":
                out.append(f"the tool's input schema could not be read ({vals.get('schema_error')}): a person decides")
            elif n == "no_instructions_in_tool_outputs":
                out.append("a tool output in the conversation carries instruction-like text")
            elif n == "request_authorizes":
                p_yes = (res.trace and next((x.probs for x in res.trace.records if x.name == "authorized"), None)) or {}
                py = p_yes.get(True, p_yes.get("yes"))
                out.append("the conversation does not authorize this call"
                           + (f" (authorizer: P(yes) = {py:.2f})" if isinstance(py, (int, float)) else ""))
            else:
                doc = (inspect.getdoc(p.func) or "").strip().splitlines()
                out.append(f"{n}" + (f": {doc[0]}" if doc else " is false") + f" [{p.then.get('verdict', 'deny')}]")
        if not out:
            unresolved = res.flow.unresolved.get("verdict") or []
            if unresolved:                            # a check reads a fact nobody gave: name the facts
                have = set(res.trace.init) | set(res.values)
                for n in unresolved:
                    p = cat.parts.get(n)
                    lack = [x for x in (p.inputs if p is not None else []) if x not in have]
                    out.append(f"cannot evaluate {n}: not given: {', '.join(lack)}" if lack else f"cannot evaluate {n}")
            else:
                from ..runtime import MISSING
                for rec in res.trace.records:             # a fact a check reads failed: its error is the reason
                    if rec.value is MISSING and rec.error and cat.parts.get(rec.name) is not None \
                            and cat.parts[rec.name].kind != "check":
                        who = "the authorizer escalated" if rec.name == "authorized" else f"{rec.name} failed"
                        out.append(f"{who}: {rec.error}")
                if not out:
                    out.append(r.why)
        return out

    def _save(self, d):
        if self.storage is None or d.stored_id is not None:
            return
        meta = {"tool": d.tool, "outcome": d.outcome, "reasons": d.reasons, "executed": d.executed}
        if d.id:
            meta["call_id"] = d.id
        meta.update(_outcome_meta(d))
        d.stored_id = self.storage.save(d.response, meta={"guard": meta})

    # --- the store
    def replay(self, stored_id):
        """Re-compute a stored decision from its recorded inputs with the tool's current checks → solvi's replay report
        ({"ok", "steps", "mismatches", "catalog", ...}): a recorded step that now computes differently (a changed policy,
        schema or authorizer) is a mismatch; "catalog" says whether the tool's checks changed since the decision."""
        if self.storage is None:
            raise ValueError("the guard has no storage")
        rec = self.storage.record(stored_id)
        tool = ((rec.get("meta") or {}).get("guard") or {}).get("tool")
        s = self.system(tool) if tool is not None else self._unknown_system()
        res = self.storage.get(stored_id, s)
        return res.trace.replay(s.catalog)

    def replay_all(self):
        """replay every stored decision → [{"id", "tool", "mismatches"}] of those that do not replay."""
        bad = []
        for st in self.storage.iter():
            v = self.replay(st.id)
            if not v["ok"]:
                bad.append({"id": st.id, "tool": ((st.meta or {}).get("guard") or {}).get("tool"),
                            "mismatches": v["mismatches"]})
        return bad

    def session(self, context=None, facts=None, max_messages=None, max_chars=None):
        """A conversation the guard follows: calls made through it add their results to its context as tool outputs,
        so a later call's grounding and injection checks see them; max_messages / max_chars cap the context kept (see
        Session). → Session."""
        return Session(self, context, facts, max_messages, max_chars)


def _outcome_meta(d):
    """What a made call left for the store: its error, or the hash of its result (the result itself is not stored)."""
    if not d.executed:
        return {}
    if d.error:
        return {"error": d.error}
    from ..runtime import vhash
    try:
        return {"result_hash": vhash(d.result)}
    except Exception:  # noqa: BLE001 — a result that cannot be hashed as data
        return {"result_hash": None}


def _names(tools):
    if tools is None:
        return None
    return {tools} if isinstance(tools, str) else set(tools)


class Session:
    """A conversation with an agent: `session.call(proposal)` checks and makes calls in its context and appends each
    made call's result as a tool output (`add` appends other messages).

    max_messages / max_chars: the context kept for checking (None: all of it) — the oldest messages are dropped first,
    and a message longer than max_chars keeps its beginning plus any instruction-like sentence of the rest (so the taint
    of a long tool output is not cut away). Every decision's trace records the context it was checked against, so the cap
    also bounds what each stored decision holds; a value or an instruction that has left the window no longer grounds a
    value or taints a call."""

    def __init__(self, guard, context=None, facts=None, max_messages=None, max_chars=None):
        self.guard = guard
        self.max_messages, self.max_chars = max_messages, max_chars
        self.context = []
        for r, t in messages(context):
            self._append(r, t)
        self.facts = dict(facts or {})
        self.decisions = []

    def _append(self, role, text):
        if self.max_chars is not None and len(text) > self.max_chars:
            text = _clip(text, self.max_chars)
        self.context.append((role, text))
        if self.max_messages is not None and len(self.context) > self.max_messages:
            del self.context[:len(self.context) - self.max_messages]
        if self.max_chars is not None:
            while len(self.context) > 1 and sum(len(t) for _, t in self.context) > self.max_chars:
                del self.context[0]

    def add(self, role, text):
        self._append(ROLES.get(role, role), _text(text))
        return self

    def check(self, call):
        d = self.guard.check(call, self.context, self.facts)
        self.decisions.append(d)
        return d

    def call(self, call):
        d = self.guard.call(call, self.context, self.facts)
        self.decisions.append(d)
        if d.executed:
            self.add("tool", f"{d.tool}: {d.error if d.error else _text(d.result)}")
        return d

    async def acall(self, call):
        d = await self.guard.acall(call, self.context, self.facts)
        self.decisions.append(d)
        if d.executed:
            self.add("tool", f"{d.tool}: {d.error if d.error else _text(d.result)}")
        return d


def _clip(text, n):
    """A text cut to about n characters: its beginning, and the instruction-like passages of the rest (whose taint must
    survive the cut) — a long one by its 400-character windows that are instruction-like themselves."""
    from ..perturb import instruction_like, instruction_spans
    rest = []
    for a, b in instruction_spans(text, actions=True):
        if b <= n:
            continue
        a = max(a, n)
        if b - a <= 400:
            rest.append(text[a:b])
            continue
        wins = [text[i:i + 400] for i in range(a, b, 200)]
        rest += [w for w in wins if instruction_like(w, actions=True)] or [text[a:a + 400]]
    tail = "\n[…]" + ("\n" + "\n".join(rest) if rest else "")
    return text[: max(0, n - len(tail))] + tail
