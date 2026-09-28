"""`solvi serve`: a System's questions over HTTP (FastAPI) or as MCP tools, and a decider behind the System One API.

    solvi serve myapp/decisions.py:system --store decisions.db            # HTTP on 127.0.0.1:8000
    solvi serve myapp.decisions:system --decider solvi-ai/solvi-base      # + POST /v1/systemone
    solvi serve --decider ./my-decider --model-name kev-latest             # only POST /v1/systemone
    solvi serve myapp.decisions:system --mcp                              # an MCP server over stdio
    solvi serve --guard catalog.py:guard --upstream "CMD"                 # an MCP proxy: the guard checks every tool call

HTTP (`solvi[serve]`: fastapi, uvicorn):
  POST /ask              {"state": {...}, "questions": [names] (default: all), "store": true} → Response.to_dict() plus
                         "stored_id" (its id in the store, or null) and "trace_hash" (the hash at the end of its trace)
  POST /ask/{question}   the state itself as the body → the same response, for that question only
  POST /ask_text         {"text": "...", "question": null, "store": true, "today": null} → a free text through
                         System.ask_text: the question it asks (routed by the decider), the fields read with their quotes,
                         the missing ones and a clarifying question ("read"), and the answers as for /ask
  GET  /questions        each question: its text, answer type and the JSON schema of the input state it reads
  GET  /health           solvi's version, the catalog's fingerprint, the store, the decider
  POST /v1/systemone     the System One API, answered by a solvi decider (`--decider`): a drop-in for a Jev / Kev client
                         (solvi.systemone is the client side)

The OpenAPI schema (/openapi.json, /docs) comes from the same pydantic types: each question's input schema from the types of
the given facts its flow reads (System(inputs=...) fields, else the types its typed readers declare) and each response's
answers as their closed sets (System.response_schema). Inputs are not validated by the web layer: the state goes to
System.ask as it is, so a wrong-typed field is what it is in solvi — the fact is missing, the answers that need it
abstain, and the trace and the audit say why (safeguard type_rejected). With `--store`, every answer is stored with its
whole trace in a TraceStorage (hash-chained), and `solvi verify` / `replay` / `diff` work on that store.

MCP (`--mcp`): each question is a tool whose input schema is the question's input state schema; a call answers that
question and returns its result (answer, confidence, status, why, safeguards) with the stored id and trace hash; the tool
`ask_text` takes a free text (as POST /ask_text). It uses
the official `mcp` SDK (2.x, `solvi[mcp]`) when it is installed, else a built-in stdio JSON-RPC server with the subset of
the protocol that tools need (initialize, ping, tools/list, tools/call)."""
import json
import sys
import threading
import time
import typing
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

REF = "#/components/schemas/{model}"          # where the OpenAPI document keeps the models (see create_app)


# ------------------------------------------------------------------------------------------------ what a question reads
def question_inputs(system, name):
    """The given facts a question's flow reads → {"properties": [fact], "required": [fact]}. Planned with every given fact
    of the catalog present; a fact is required when the question cannot be answered without it (the strategist leaves it
    unresolved), so the inputs of alternative producers are optional."""
    from .strategist import PlanError, given_facts, plan
    cat, q = system.catalog, system.questions[name]
    given = given_facts(cat, system.questions.values())
    try:
        flow = plan(cat, [q], given, system.heads)
    except PlanError:
        return {"properties": [], "required": []}
    used = {x for s in flow.steps for x in s.part.inputs if x in given}
    used |= set(q.uses or ()) & given
    head = system.heads.get(name)
    if head is not None:
        used |= set(head.features) & given
    required = []
    for f in sorted(used):
        try:
            if plan(cat, [q], given - {f}, system.heads).unresolved.get(name):
                required.append(f)
        except PlanError:
            pass
    return {"properties": sorted(used), "required": required}


def _fact_type(system, fact):
    """The type of a given fact → (type or Any, description): System(inputs=...)'s field, else the one type its typed
    readers declare (Any when they disagree or nobody declares one)."""
    readers = system.catalog.readers.get(fact) or {}
    m = system.inputs
    if m is not None and fact in m.model_fields:
        return m.model_fields[fact].annotation, m.model_fields[fact].description
    types = []
    for t in readers.values():
        if t not in types:
            types.append(t)
    return (types[0] if len(types) == 1 else Any), None


def _schema_ok(t):
    from pydantic import TypeAdapter
    try:
        TypeAdapter(t).json_schema()
        return True
    except Exception:  # noqa: BLE001 — a type with no JSON schema (an arbitrary class): documented as any value
        return False


def input_model(system, name):
    """A pydantic model of the input state a question reads (extra keys allowed) — for the schema; solvi validates."""
    from .typed import type_name
    info = question_inputs(system, name)
    readers = system.catalog.readers
    fields = {}
    for i, f in enumerate(info["properties"]):
        t, desc = _fact_type(system, f)
        if t is not Any and not _schema_ok(t):
            desc, t = f"{type_name(t)} (no JSON schema)", Any
        who = sorted(readers.get(f) or ())
        desc = desc or (f"read by {', '.join(who)}" if who else None)
        default = ... if f in info["required"] else None
        fields[f"f{i}"] = (t, Field(default, alias=f, title=f, description=desc))
    return create_model(_camel(name) + "Input", __config__=ConfigDict(extra="allow", arbitrary_types_allowed=True),
                        __doc__=f"The input state of question {name!r}", **fields)


def input_schema(system, name, ref_template="#/$defs/{model}"):
    """The JSON schema of the input state a question reads."""
    return input_model(system, name).model_json_schema(ref_template=ref_template)


def _camel(name):
    s = "".join(w[:1].upper() + w[1:] for w in str(name).replace("-", "_").split("_") if w)
    return s if s.isidentifier() else "Question"


def questions_info(system):
    """GET /questions: [{name, text, answer, min_confidence, require_evidence, input_schema}]."""
    from .schema import dump
    out = []
    for q in system.questions.values():
        d = dump(q, "json")
        d["input_schema"] = input_schema(system, q.name)
        out.append(d)
    return out


# ------------------------------------------------------------------------------------------------ the service
class Service:
    """What the HTTP app and the MCP server call: asks a System (one at a time: a System learns costs and counts stats
    in place; a System with async parts is asked with System.aask, concurrently on the server's event loop), answers
    System One requests with a decider."""

    def __init__(self, system=None, decider=None, storage=None, model_name=None, textin=None):
        if system is None and decider is None:
            raise ValueError("solvi serve needs a System, a decider (--decider), or both")
        self.system, self.decider = system, decider
        self._textin = textin                         # a solvi.textin.TextIn (synonyms, patterns, ...), else made on use
        if system is not None and storage is not None:
            from .storage import open_storage
            system.storage = open_storage(storage, system)
        self.model_name = model_name or (getattr(decider, "model_id", None) if decider is not None else None)
        self._lock = threading.Lock()
        self._schemas = {}

    # --- a System
    def ask(self, state, names=None, store=True):
        """→ Response.to_dict() with "stored_id" and "trace_hash"."""
        resp = self._ask(state, names, store)
        d = resp.to_dict()
        d["stored_id"], d["trace_hash"] = resp.stored_id, trace_hash(resp)
        return d

    def _checked(self, state, names):
        s = self.system
        if s is None:
            raise LookupError("this server has no System (only POST /v1/systemone)")
        if not isinstance(state, dict):
            raise TypeError(f"the state is a JSON object of given facts, not {type(state).__name__}")
        bad = [n for n in names or () if n not in s.questions]
        if bad:
            raise KeyError(f"no such question: {', '.join(bad)}")
        return s

    def _ask(self, state, names, store):
        s = self._checked(state, names)
        with self._lock:
            if self.is_async:                         # async parts: awaited concurrently within the ask (System.aask)
                from .runtime import run_sync
                return run_sync(s.aask(state, names=list(names) if names else None, store=store))
            return s.ask(state, names=list(names) if names else None, store=store)

    @property
    def is_async(self):
        """Does the System have parts that aask awaits? Then asks go through System.aask."""
        if getattr(self, "_async", None) is None:
            self._async = self.system is not None and self.system.is_async
        return self._async

    async def _aask(self, state, names, store):
        """System.aask on the server's event loop: asks run concurrently (a System's costs and stats are updated between
        awaits, so they need no lock)."""
        s = self._checked(state, names)
        return await s.aask(state, names=list(names) if names else None, store=store)

    async def aask(self, state, names=None, store=True):
        """ask, for an async System (System.aask)."""
        resp = await self._aask(state, names, store)
        d = resp.to_dict()
        d["stored_id"], d["trace_hash"] = resp.stored_id, trace_hash(resp)
        return d

    def tool(self, name, state):
        """An MCP tool call: one question → its result (answer, confidence, status, why, ...), the safeguards that fired
        for it, "stored_id" and "trace_hash"."""
        return self._tool_result(name, self._ask(state, [name], True))

    async def atool(self, name, state):
        """tool, for an async System (System.aask)."""
        return self._tool_result(name, await self._aask(state, [name], True))

    # --- a free text (System.ask_text)
    def textin(self, today=None):
        """The TextIn that reads texts for this server: the one given, else TextIn(system, decider) — made once; `today`
        (a date or ISO string; default: the server's date at the request, recorded in the trace) on a copy per request."""
        import copy
        import datetime as dt

        from .textin import TextIn
        if self.system is None:
            raise LookupError("this server has no System (only POST /v1/systemone)")
        if self._textin is None:
            self._textin = TextIn(self.system, self.decider)
        tin = copy.copy(self._textin)
        if today is not None:
            tin.today = dt.date.fromisoformat(today) if isinstance(today, str) else today
        elif tin.today is None:
            tin.today = dt.date.today()
        return tin

    def _read(self, text, question, today):
        if not isinstance(text, str) or not text.strip():
            raise TypeError("the text is a non-empty string")
        tin = self.textin(today)
        if question is not None and question not in tin.entry_points:
            raise KeyError(f"no such entry point: {question}")
        if question is None and len(tin.entry_points) > 1 and tin.decider is None:
            raise ValueError("choosing the question a text asks needs a decider: start the server with --decider "
                             "(or pass question=)")
        return tin.read(text, question=question)

    @staticmethod
    def _text_result(resp):
        d = resp.to_dict()
        read = resp.textin
        d["read"] = {**read.to_dict(), "clarify": read.clarify(), "escalated": read.escalated}
        d["stored_id"], d["trace_hash"] = resp.stored_id, trace_hash(resp)
        return d

    def ask_text(self, text, question=None, store=True, today=None):
        """A free text → Response.to_dict() of System.ask_text plus "read" (the question it asks, the fields read with
        their quotes, the missing ones, a clarifying question), "stored_id" and "trace_hash"."""
        read = self._read(text, question, today)
        with self._lock:
            if self.is_async:
                from .runtime import run_sync
                resp = run_sync(self.system.aask_text(read, store=store))
            else:
                resp = self.system.ask_text(read, store=store)
        return self._text_result(resp)

    async def aask_text(self, text, question=None, store=True, today=None):
        """ask_text, for an async System (System.aask_text)."""
        read = self._read(text, question, today)
        return self._text_result(await self.system.aask_text(read, store=store))

    @staticmethod
    def _tool_result(name, resp):
        d = resp.to_dict()
        out = {"question": name, **d["results"][name]}
        out["safeguards"] = [e for e in d["safeguards"] if name in (e.get("questions") or [name])]
        out["stored_id"], out["trace_hash"] = resp.stored_id, trace_hash(resp)
        return out

    def health(self):
        from . import __version__
        d = {"status": "ok", "solvi": __version__}
        if self.system is not None:
            st = self.system.storage
            d.update(questions=list(self.system.questions), catalog=self.system.fingerprint()["catalog"],
                     store=None if st is None else str(getattr(st, "path", type(st).__name__)))
        if self.decider is not None:
            d["decider"] = {"model": self.model_name, "id": self.decider.model_id, "backend": self.decider.backend}
        return d

    # --- System One
    def systemone(self, body):
        """A System One request (a dict, see SystemOneRequest) → the response dict, answered by the decider: choice →
        the probability of each option (criteria in their order), noul → P(yes), score → the probability of each level
        (criteria in order, lowest first) and the expected level index. `other` / `none` options are scored like any
        other (the API has no abstain option). The answers carry no act / escalate signal: the client decides."""
        if self.decider is None:
            raise LookupError("this server has no decider: start it with --decider")
        req = body if isinstance(body, SystemOneRequest) else SystemOneRequest.model_validate(body)
        t0 = time.perf_counter()
        m = self.decider
        passes = m.passes
        answers = {}
        with self._lock:
            for name, q in req.questions.items():
                answers[name] = self._one(m, req.state, q)
        return {"model": self.model_name, "answers": answers,
                "usage": {"questions": len(answers), "passes": m.passes - passes},
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3)}

    @staticmethod
    def _one(m, state, q):
        crit = dict(q.criteria or {})
        if q.type == "noul":
            d = m.decide(state, q.instructions, ["yes", "no"], {k: v for k, v in crit.items() if k in ("yes", "no") and v},
                         kind="noul")
            return {"type": "noul", "noul": float(d.probs["yes"])}
        opts = list(crit)
        if len(opts) < 2:
            raise ValueError(f"a {q.type} question needs at least two criteria (options), got {opts}")
        desc = {k: v for k, v in crit.items() if v}
        if q.type == "choice":
            d = m.decide(state, q.instructions, opts, desc, other=False, kind="choice")
            return {"type": "choice", "choice": d.value, "confidence": float(d.conf),
                    "probabilities": {o: float(d.probs[o]) for o in opts}}
        d = m.decide(state, q.instructions, opts, desc, kind="score")
        p = [float(d.probs[o]) for o in opts]
        return {"type": "score", "score": sum(i * x for i, x in enumerate(p)), "confidence": float(d.conf),
                "legend": opts, "probabilities": dict(zip(opts, p))}


def trace_hash(resp):
    """The hash at the end of a response's trace (its last record's; the input's hash when nothing ran)."""
    tr = resp.trace
    return tr.records[-1].hash if tr.records else tr.init_hash


# --- the System One API (the wire format solvi.systemone speaks as a client)
class SystemOneQuestion(BaseModel):
    type: Literal["choice", "noul", "score"]
    instructions: str
    criteria: Optional[dict[str, Optional[str]]] = None      # option (level) → description; levels lowest first


class SystemOneRequest(BaseModel):
    state: Any                                               # a text, or a JSON state (serialized as the decider reads it)
    model: Optional[str] = None
    questions: dict[str, SystemOneQuestion]


class ChoiceAnswer(BaseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float]


class NoulAnswer(BaseModel):
    type: Literal["noul"]
    noul: float = Field(description="P(yes)")


class ScoreAnswer(BaseModel):
    type: Literal["score"]
    score: float = Field(description="the expected level index (0 = legend[0], the lowest)")
    confidence: float
    legend: list[str]
    probabilities: dict[str, float]


class SystemOneResponse(BaseModel):
    model: Optional[str]
    answers: dict[str, typing.Union[ChoiceAnswer, NoulAnswer, ScoreAnswer]]
    usage: dict[str, int]
    latency_ms: float


# ------------------------------------------------------------------------------------------------ HTTP
class AskRequest(BaseModel):
    state: dict[str, Any] = Field(description="the given facts")
    questions: Optional[list[str]] = Field(None, description="ask only these questions (default: all)")
    store: bool = Field(True, description="save the response to the server's store (if it has one)")


class AskTextRequest(BaseModel):
    text: str = Field(description="a free text: a message, an e-mail, a chat turn")
    question: Optional[str] = Field(None, description="the question it asks (default: the decider picks the entry point)")
    store: bool = Field(True, description="save the response to the server's store (if it has one)")
    today: Optional[str] = Field(None, description="ISO date for year-less and relative dates (default: the server's date)")


def create_app(system=None, decider=None, storage=None, model_name=None, title=None, textin=None):
    """The FastAPI app (see the module docs). `system`: a System; `decider`: a DecideModel for POST /v1/systemone and for
    routing texts (POST /ask_text); `storage`: a TraceStorage or a path (every ask is stored); `model_name`: the model name
    System One answers carry; `textin`: a solvi.textin.TextIn for POST /ask_text (default: TextIn(system, decider))."""
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import JSONResponse

    from . import __version__
    svc = Service(system, decider, storage, model_name, textin)
    app = FastAPI(title=title or "solvi", version=__version__,
                  description="Decisions from a solvi catalog: the model proposes, code decides, everything is in the trace.")
    app.state.service = svc

    def call(f, *args):
        try:
            return JSONResponse(f(*args))
        except LookupError as e:                      # KeyError too: no such question / no System / no decider
            raise HTTPException(404, str(e.args[0] if e.args else e)) from None
        except (TypeError, ValueError) as e:
            raise HTTPException(422, str(e)) from None

    async def acall(f, *args):
        try:
            return JSONResponse(await f(*args))
        except LookupError as e:
            raise HTTPException(404, str(e.args[0] if e.args else e)) from None
        except (TypeError, ValueError) as e:
            raise HTTPException(422, str(e)) from None
    run_async = system is not None and svc.is_async   # async parts: the questions are async endpoints (System.aask)

    @app.get("/health", tags=["service"])
    def health():
        return svc.health()

    documented = {}                                   # path → (request model or None, response model)
    if system is not None:
        from .schema import response_model

        def with_ids(M, name):
            return create_model(name, __base__=M, stored_id=(Optional[str], None), trace_hash=(str, ""))

        @app.get("/questions", tags=["questions"])
        def questions():
            return questions_info(system)

        if run_async:
            @app.post("/ask", tags=["questions"], response_model=None)
            async def ask(body: AskRequest):
                return await acall(svc.aask, body.state, body.questions, body.store)
        else:
            @app.post("/ask", tags=["questions"], response_model=None)
            def ask(body: AskRequest):
                return call(svc.ask, body.state, body.questions, body.store)
        documented["/ask"] = (None, with_ids(response_model(system), "AskResponse"))

        if run_async:
            @app.post("/ask_text", tags=["questions"], response_model=None,
                      summary="A free text: the question it asks, its fields read with quotes, the answers")
            async def ask_text(body: AskTextRequest):
                return await acall(svc.aask_text, body.text, body.question, body.store, body.today)
        else:
            @app.post("/ask_text", tags=["questions"], response_model=None,
                      summary="A free text: the question it asks, its fields read with quotes, the answers")
            def ask_text(body: AskTextRequest):
                return call(svc.ask_text, body.text, body.question, body.store, body.today)

        def route(name):
            if run_async:
                async def ask_one(state: dict = Body(...)):
                    return await acall(svc.aask, state, [name])
            else:
                def ask_one(state: dict = Body(...)):
                    return call(svc.ask, state, [name])
            ask_one.__name__ = f"ask_{name}"
            app.post(f"/ask/{name}", tags=["questions"], response_model=None, summary=system.questions[name].text or name,
                     description=f"Ask {name!r}: the body is the input state.")(ask_one)
            documented[f"/ask/{name}"] = (input_model(system, name),
                                          with_ids(response_model(system, [name]), _camel(name) + "Response"))
        for n in system.questions:
            route(n)

    if decider is not None:
        @app.post("/v1/systemone", tags=["system one"], response_model=SystemOneResponse)
        def systemone(body: SystemOneRequest):
            return call(svc.systemone, body)

    def openapi():
        """FastAPI's document, with each question's input and response schemas from solvi's pydantic types."""
        if app.openapi_schema:
            return app.openapi_schema
        doc = get_openapi(title=app.title, version=app.version, description=app.description, routes=app.routes)
        comps = doc.setdefault("components", {}).setdefault("schemas", {})

        def hoist(M):
            s = M.model_json_schema(ref_template=REF)
            comps.update(s.pop("$defs", {}))
            comps[M.__name__] = s
            return {"$ref": REF.format(model=M.__name__)}
        for path, (req, resp) in documented.items():
            op = doc["paths"][path]["post"]
            if req is not None:
                op["requestBody"] = {"required": True, "content": {"application/json": {"schema": hoist(req)}}}
            op["responses"]["200"] = {"description": "the response: answers, flow, trace, stored id",
                                      "content": {"application/json": {"schema": hoist(resp)}}}
        app.openapi_schema = doc
        return doc
    app.openapi = openapi
    return app


# ------------------------------------------------------------------------------------------------ MCP
PROTOCOL_VERSIONS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")


def mcp_tools(svc):
    """The questions as MCP tools: [{"name", "description", "inputSchema"}] (a name outside [A-Za-z0-9_-] is mapped)."""
    tools = []
    for q in svc.system.questions.values():
        at = q.answer
        opts = f": one of {at.options}" if at.options and at.kind in ("choice", "ordinal") else \
            f": any of {at.options}" if at.kind == "multi" else ""
        desc = (f"{q.text} — answers {at.kind}{opts}. solvi decides with the catalog's code and checks; the result says "
                "how (why, safeguards) and may abstain.")
        tools.append({"name": tool_name(q.name), "description": desc, "inputSchema": input_schema(svc.system, q.name)})
    names = sorted(svc.system.questions)
    tools.append({"name": text_tool_name(svc), "description":
                  "A free text (a customer's message): solvi picks the question it asks among " + ", ".join(names) +
                  ", reads that question's input fields from the text with quotes, and answers — or says which fields "
                  "are missing and asks a clarifying question ('read.clarify'). Fields are never guessed.",
                  "inputSchema": {"type": "object", "properties": {
                      "text": {"type": "string", "description": "the text"},
                      "question": {"type": "string", "enum": names,
                                   "description": "the question it asks, when known (skips routing)"}},
                      "required": ["text"]}})
    return tools


def text_tool_name(svc):
    """The MCP tool that takes a free text: "ask_text" (or "solvi_ask_text" when a question is named ask_text)."""
    return "solvi_ask_text" if "ask_text" in {tool_name(q) for q in svc.system.questions} else "ask_text"


def tool_name(q):
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", q)[:64]


def call_tool(svc, name, arguments):
    """tools/call → (result dict, is_error)."""
    names = {tool_name(q): q for q in svc.system.questions}
    if name == text_tool_name(svc):
        a = dict(arguments or {})
        try:
            return svc.ask_text(a.get("text"), a.get("question")), False
        except Exception as e:  # noqa: BLE001
            return {"error": f"{type(e).__name__}: {e}"}, True
    if name not in names:
        raise KeyError(name)
    try:
        return svc.tool(names[name], dict(arguments or {})), False
    except Exception as e:  # noqa: BLE001 — a tool error is reported to the agent, not a protocol error
        return {"error": f"{type(e).__name__}: {e}"}, True


async def acall_tool(svc, name, arguments):
    """tools/call on an event loop → (result dict, is_error): an async System is asked with aask, a sync one in a worker
    thread (its lock serializes the asks)."""
    names = {tool_name(q): q for q in svc.system.questions}
    if name not in names and name != text_tool_name(svc):
        raise KeyError(name)
    if not svc.is_async:
        import asyncio
        return await asyncio.to_thread(call_tool, svc, name, arguments)
    try:
        if name == text_tool_name(svc):
            a = dict(arguments or {})
            return await svc.aask_text(a.get("text"), a.get("question")), False
        return await svc.atool(names[name], dict(arguments or {})), False
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}, True


def run_builtin(svc, stdin=None, stdout=None):
    """A stdio MCP server without the SDK: JSON-RPC 2.0, one message per line; initialize, ping, tools/list, tools/call
    (notifications are read and ignored)."""
    from . import __version__
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout

    def send(msg):
        stdout.write(json.dumps(msg, ensure_ascii=False, default=repr) + "\n")
        stdout.flush()

    def error(id_, code, msg):
        send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": msg}})
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            error(None, -32700, "parse error")
            continue
        if not isinstance(msg, dict) or "method" not in msg:
            if not (isinstance(msg, dict) and ("result" in msg or "error" in msg)):    # a response to us: none expected
                error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
            continue
        method, params, id_ = msg["method"], msg.get("params") or {}, msg.get("id")
        if "id" not in msg:                            # a notification (notifications/initialized, cancelled, ...)
            continue
        if method == "initialize":
            v = params.get("protocolVersion")
            send({"jsonrpc": "2.0", "id": id_, "result": {
                "protocolVersion": v if v in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "solvi", "version": __version__},
                "instructions": "Each tool is a question of a solvi decision system: pass the input state, get the answer "
                                "with its confidence, why, safeguards and the id of the stored trace."}})
        elif method == "ping":
            send({"jsonrpc": "2.0", "id": id_, "result": {}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": id_, "result": {"tools": mcp_tools(svc)}})
        elif method == "tools/call":
            try:
                out, bad = call_tool(svc, params.get("name"), params.get("arguments"))
            except KeyError:
                error(id_, -32602, f"unknown tool: {params.get('name')}")
                continue
            send({"jsonrpc": "2.0", "id": id_, "result": {
                "content": [{"type": "text", "text": json.dumps(out, ensure_ascii=False, default=repr)}],
                "structuredContent": out, "isError": bad}})
        else:
            error(id_, -32601, f"method not found: {method}")


def run_sdk(svc):
    """A stdio MCP server with the official SDK (mcp 2.x: handlers passed to the low-level Server)."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    from . import __version__

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(name=t["name"], description=t["description"],
                                                       input_schema=t["inputSchema"]) for t in mcp_tools(svc)])

    async def on_call(ctx, params):
        try:
            out, bad = await acall_tool(svc, params.name, params.arguments)
        except KeyError:
            out, bad = {"error": f"unknown tool: {params.name}"}, True
        return types.CallToolResult(content=[types.TextContent(type="text", text=json.dumps(out, ensure_ascii=False,
                                                                                            default=repr))],
                                    structured_content=out, is_error=bad)
    server = Server("solvi", version=__version__, on_list_tools=list_tools, on_call_tool=on_call)

    async def main():
        async with stdio_server() as (r, w):
            await server.run(r, w, server.create_initialization_options())
    anyio.run(main)


def sdk_available():
    """Is the official MCP SDK (2.x, handlers in the Server constructor) installed?"""
    try:
        import inspect

        from mcp.server.lowlevel import Server
    except ImportError:
        return False
    return "on_list_tools" in str(inspect.signature(Server.__init__))


def run_mcp(svc, impl="auto"):
    if svc.system is None:
        raise ValueError("an MCP server needs a System (its questions are the tools)")
    if impl == "sdk" or (impl == "auto" and sdk_available()):
        return run_sdk(svc)
    return run_builtin(svc)


# ------------------------------------------------------------------------------------------------ the command
def cmd_serve(a):
    """`solvi serve` (see solvi.cli) → exit status."""
    from .cli import _fail, load_object, load_system
    if getattr(a, "guard", None) or getattr(a, "upstream", None):
        return _serve_guard(a, _fail, load_object)
    system = load_system(a.system) if a.system else None
    decider = None
    if a.decider:
        if a.decider.startswith(("systemone:", "llm:")):           # a service: solvi.models reads the spec
            from .models import load
            decider = load(a.decider, api_key=getattr(a, "api_key", None))
        else:
            from .decide import DecideModel
            decider = DecideModel.load(a.decider, backend=a.backend)
    if system is None and decider is None:
        _fail("serve: name a System (module:attr or file.py:attr) and / or a decider (--decider)")
    if a.store and system is None:
        _fail("serve: --store needs a System")
    if a.mcp:
        if system is None:
            _fail("serve --mcp: needs a System (its questions are the tools)")
        if a.mcp_impl == "sdk" and not sdk_available():
            _fail("serve --mcp-impl sdk: the MCP SDK (mcp>=2) is not installed: pip install 'solvi[mcp]'")
        svc = Service(system, decider, a.store)          # the decider routes texts for the ask_text tool
        run_mcp(svc, a.mcp_impl)
        return 0
    try:
        import uvicorn
    except ImportError:
        _fail("serve: HTTP needs FastAPI and uvicorn: pip install 'solvi[serve]'")
    app = create_app(system, decider, a.store, a.model_name, title=f"solvi: {a.system}" if a.system else "solvi")
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level)
    return 0


def _serve_guard(a, _fail, load_object):
    """`solvi serve --guard catalog.py:guard --upstream CMD`: the MCP proxy with a Guard (solvi.agents.mcp)."""
    if not (a.guard and a.upstream):
        _fail("serve --guard / --upstream: both are needed — the guard (module:attr or file.py:attr) and the MCP server's "
              "command line")
    if a.system or a.decider:
        _fail("serve --guard: a proxy serves the upstream server's tools; drop the System / --decider")
    from .agents import Guard
    from .agents.mcp import run_proxy
    guard = load_object(a.guard)
    if not isinstance(guard, Guard):
        _fail(f"--guard {a.guard}: not a solvi.agents.Guard")
    if a.store:
        from .storage import open_storage
        guard.storage = open_storage(a.store)
    try:
        facts = json.loads(a.facts) if a.facts else None
    except ValueError as e:
        _fail(f"--facts: not JSON: {e}")
    if facts is not None and not isinstance(facts, dict):
        _fail("--facts: a JSON object of facts")
    run_proxy(guard, a.upstream, facts=facts, escalate=a.escalate)
    return 0


def add_parser(sub):
    """The `serve` subcommand's options (solvi.cli)."""
    s = sub.add_parser("serve", help="serve a System's questions over HTTP (FastAPI) or MCP, and a decider as System One")
    s.add_argument("system", nargs="?", help="module:attr or file.py:attr — a System or a function returning one")
    s.add_argument("--store", help="save every answer to this TraceStorage (.db / .sqlite: SQLite, else JSON lines)")
    s.add_argument("--decider", help="a decider (a checkpoint folder, a Hugging Face id, systemone:URL#model or "
                   "llm:URL#model) behind POST /v1/systemone and routing POST /ask_text")
    s.add_argument("--api-key", help="for a systemone: / llm: decider (default $SOLVI_SYSTEMONE_API_KEY / "
                   "$SOLVI_LLM_API_KEY)")
    s.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="the decider's backend")
    s.add_argument("--model-name", help="the model name System One answers carry (default: the decider's id)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info")
    s.add_argument("--mcp", action="store_true", help="an MCP server over stdio instead of HTTP: each question is a tool")
    s.add_argument("--mcp-impl", default="auto", choices=["auto", "sdk", "builtin"],
                   help="the official MCP SDK (auto: when installed) or the built-in JSON-RPC subset")
    s.add_argument("--guard", help="module:attr or file.py:attr — a solvi.agents.Guard: an MCP proxy that checks every "
                                   "tools/call of --upstream (solvi.agents.mcp)")
    s.add_argument("--upstream", help="the command line of the MCP server (stdio) behind the guard")
    s.add_argument("--facts", help="a JSON object of facts the guard's policies read (with --guard)")
    s.add_argument("--escalate", default="elicit", choices=["elicit", "deny"],
                   help="with --guard: ask the user about an escalated call (MCP elicitation, when the client supports it) "
                        "or return it as an error")
    return s
