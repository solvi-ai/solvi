"""`solvi serve`: a System's questions over HTTP (FastAPI) or as MCP tools, and a decider behind the System One API.

    solvi serve myapp/decisions.py:system --store decisions.db            # HTTP on 127.0.0.1:8000
    solvi serve myapp.decisions:system --decider solvi-ai/solvi-base      # + POST /v1/systemone (a cached model; --pull)
    solvi serve --decider ./my-decider --model-name kev-latest             # only POST /v1/systemone
    solvi serve myapp.decisions:system --mcp                              # an MCP server over stdio

HTTP (`solvi[serve]`: fastapi, uvicorn):
  POST /ask              {"state": {...}, "questions": [names] (default: all), "store": true} → Response.to_dict() plus
                         "stored_id" (its id in the store, or null) and "trace_hash" (the hash at the end of its trace)
  POST /ask/{question}   the state itself as the body → the same response, for that question only
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
question and returns its result (answer, confidence, status, why, safeguards) with the stored id and trace hash. It uses
the official `mcp` SDK (2.x, `solvi[mcp]`) when it is installed, else a built-in stdio JSON-RPC server with the subset of
the protocol that tools need (initialize, ping, tools/list, tools/call).

Security (see docs/guide.md, Serving): `--token` / $SOLVI_SERVE_TOKEN requires `Authorization: Bearer <token>` on every
HTTP request (constant-time compare); a request body / MCP message is at most `--max-body` bytes and `--max-depth` levels
of JSON; a request takes at most `--timeout` seconds (async Systems: through System.aask's part timeout, so the answer
abstains rather than the request failing); a failure the client did not cause is logged here and answered with an
incident id, never a traceback or a path; CORS headers only with `--cors ORIGIN`; nothing is imported or loaded from
request data; `--decider` never downloads without `--pull`."""
import hmac
import json
import logging
import os
import sys
import threading
import time
import typing
import uuid
from dataclasses import dataclass
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


# ------------------------------------------------------------------------------------------------ limits and errors
log = logging.getLogger("solvi.serve")


@dataclass
class Limits:
    """What one request may be (see the module docs, Security). max_body: bytes of an HTTP body / an MCP message;
    max_depth: nesting of JSON objects and arrays in it; timeout: seconds a request may take (None: no limit) — HTTP
    answers 504 after it, an MCP tool call an error; an async System's parts are given 80% of it as System.aask's timeout
    (unless System(timeout=) or the part sets one), so a slow part makes its questions abstain (safeguard `timeout`) and
    the request still answers."""
    max_body: int = 1_000_000
    max_depth: int = 32
    timeout: Optional[float] = 60.0


class RequestError(Exception):
    """A request the server refuses; its message is written by solvi (never an exception text from the catalog's code) and
    is safe to return to the client. `status`: the HTTP status."""
    status = 422

    def __init__(self, message, status=None):
        super().__init__(message)
        if status is not None:
            self.status = status


class NotFound(RequestError, LookupError):
    status = 404


class BadRequest(RequestError, ValueError, TypeError):
    status = 422


class RequestTimeout(RequestError, TimeoutError):
    status = 504


def too_deep(v, max_depth):
    """Does a JSON value nest objects / arrays deeper than max_depth? (Iterative: no recursion on hostile input.)"""
    stack = [(v, 1)]
    while stack:
        x, d = stack.pop()
        if isinstance(x, dict):
            if d > max_depth:
                return True
            stack.extend((y, d + 1) for y in x.values())
        elif isinstance(x, (list, tuple)):
            if d > max_depth:
                return True
            stack.extend((y, d + 1) for y in x)
    return False


def parse_json(data, limits):
    """Bytes / text of a request → the JSON value, within the limits (BadRequest / RequestError 413 otherwise)."""
    if len(data) > limits.max_body:
        raise RequestError(f"the request is larger than {limits.max_body} bytes", 413)
    try:
        v = json.loads(data)
    except RecursionError:
        raise BadRequest(f"the request nests JSON deeper than {limits.max_depth} levels") from None
    except ValueError:
        raise BadRequest("the request is not JSON") from None
    if too_deep(v, limits.max_depth):
        raise BadRequest(f"the request nests JSON deeper than {limits.max_depth} levels")
    return v


def internal_error(where):
    """Log the exception being handled (with its traceback) on the server → the message for the client: an incident id,
    nothing of the exception (its text may carry paths, data or code)."""
    incident = uuid.uuid4().hex[:12]
    log.exception("solvi serve: %s failed (incident %s)", where, incident)
    return f"internal error (incident {incident}; the details are in the server log)"


# ------------------------------------------------------------------------------------------------ the service
class Service:
    """What the HTTP app and the MCP server call: asks a System (one at a time: a System learns costs and counts stats
    in place; a System with async parts is asked with System.aask, concurrently on the server's event loop), answers
    System One requests with a decider. `limits`: a Limits (the request size, JSON depth and timeout)."""

    def __init__(self, system=None, decider=None, storage=None, model_name=None, limits=None):
        if system is None and decider is None:
            raise ValueError("solvi serve needs a System, a decider (--decider), or both")
        self.system, self.decider = system, decider
        if system is not None and storage is not None:
            from .storage import open_storage
            system.storage = open_storage(storage, system)
        self.model_name = model_name or (getattr(decider, "model_id", None) if decider is not None else None)
        self.limits = limits or Limits()
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
            raise NotFound("this server has no System (only POST /v1/systemone)")
        if not isinstance(state, dict):
            raise BadRequest(f"the state is a JSON object of given facts, not {type(state).__name__}")
        if too_deep(state, self.limits.max_depth):
            raise BadRequest(f"the state nests JSON deeper than {self.limits.max_depth} levels")
        bad = [n for n in names or () if n not in s.questions]
        if bad:
            raise NotFound(f"no such question: {', '.join(map(str, bad))}")
        return s

    @property
    def part_timeout(self):
        """System.aask's timeout for the parts of an async System: System(timeout=) if set, else 80% of the request's."""
        t = getattr(self.system, "timeout", None)
        if t is not None or self.limits.timeout is None:
            return t
        return 0.8 * self.limits.timeout

    def _ask(self, state, names, store):
        s = self._checked(state, names)
        with self._lock:
            if self.is_async:                         # async parts: awaited concurrently within the ask (System.aask)
                from .runtime import run_sync
                return run_sync(s.aask(state, names=list(names) if names else None, store=store,
                                       timeout=self.part_timeout))
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
        return await s.aask(state, names=list(names) if names else None, store=store, timeout=self.part_timeout)

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

    @staticmethod
    def _tool_result(name, resp):
        d = resp.to_dict()
        out = {"question": name, **d["results"][name]}
        out["safeguards"] = [e for e in d["safeguards"] if name in (e.get("questions") or [name])]
        out["stored_id"], out["trace_hash"] = resp.stored_id, trace_hash(resp)
        return out

    def health(self):
        """{"status", "solvi", ...}: the questions, the catalog's fingerprint, the store (its file name only: no paths
        leave the server), the decider."""
        from . import __version__
        d = {"status": "ok", "solvi": __version__}
        if self.system is not None:
            st = self.system.storage
            where = None if st is None else getattr(st, "path", None)
            d.update(questions=list(self.system.questions), catalog=self.system.fingerprint()["catalog"],
                     store=None if st is None else os.path.basename(str(where)) if where else type(st).__name__)
        if self.decider is not None:
            d["decider"] = {"model": self.model_name, "id": self.decider.model_id, "backend": self.decider.backend}
        return d

    # --- System One
    def systemone(self, body):
        """A System One request (a dict, see SystemOneRequest) → the response dict, answered by the decider: choice →
        the probability of each option (criteria in their order), noul → P(yes), score → the probability of each level
        (criteria in order, lowest first) and the expected level index. `other` / `none` options are scored like any
        other (the API has no abstain option). The answers carry no act / escalate signal: the client decides. The
        request's "model" is only a name echoed back: nothing is loaded from request data."""
        if self.decider is None:
            raise NotFound("this server has no decider: start it with --decider")
        if isinstance(body, SystemOneRequest):
            req = body
        else:
            from pydantic import ValidationError
            if too_deep(body, self.limits.max_depth):
                raise BadRequest(f"the request nests JSON deeper than {self.limits.max_depth} levels")
            try:
                req = SystemOneRequest.model_validate(body)
            except ValidationError as e:
                raise BadRequest(f"not a System One request: {e.error_count()} problem(s), e.g. "
                                 f"{'.'.join(map(str, e.errors()[0]['loc']))}: {e.errors()[0]['msg']}") from None
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
            raise BadRequest(f"a {q.type} question needs at least two criteria (options), got {opts}")
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


class Guard:
    """ASGI middleware in front of the app: the bearer token (constant-time compare), the body size (Content-Length, and
    the bytes actually received) and the JSON depth of a request body — refused before FastAPI parses it."""

    def __init__(self, app, limits, token=None):
        self.app, self.limits = app, limits
        self.token = None if token is None else token.encode()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or ())
        if self.token is not None:
            got = headers.get(b"authorization", b"")
            scheme, _, cred = got.partition(b" ")
            if not (scheme.lower() == b"bearer" and hmac.compare_digest(cred.strip(), self.token)):
                return await _reply(send, 401, "a bearer token is required (Authorization: Bearer ...)",
                                    [(b"www-authenticate", b"Bearer")])
        if scope.get("method") not in ("POST", "PUT", "PATCH"):
            return await self.app(scope, receive, send)
        n = self.limits.max_body
        try:
            declared = int(headers.get(b"content-length", b"0") or 0)
        except ValueError:
            return await _reply(send, 400, "a bad Content-Length")
        if declared > n:
            return await _reply(send, 413, f"the request is larger than {n} bytes")
        body, more = b"", True
        while more:
            msg = await receive()
            if msg["type"] == "http.disconnect":
                return
            body += msg.get("body", b"")
            more = msg.get("more_body", False)
            if len(body) > n:
                return await _reply(send, 413, f"the request is larger than {n} bytes")
        if body:
            try:
                parse_json(body, self.limits)
            except RequestError as e:
                if str(e) != "the request is not JSON":        # not JSON: FastAPI answers 422 as usual
                    return await _reply(send, e.status if e.status != 422 else 400, str(e))
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        return await self.app(scope, replay, send)


async def _reply(send, status, detail, headers=()):
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                            *headers]})
    await send({"type": "http.response.body", "body": body})


def create_app(system=None, decider=None, storage=None, model_name=None, title=None, limits=None, token=None,
               cors=None):
    """The FastAPI app (see the module docs). `system`: a System; `decider`: a DecideModel for POST /v1/systemone;
    `storage`: a TraceStorage or a path (every ask is stored); `model_name`: the model name System One answers carry;
    `limits`: a Limits (request size, JSON depth, timeout); `token`: every request must carry `Authorization: Bearer
    <token>` (None: no authentication); `cors`: the origins browsers may call it from (None: no CORS headers at all)."""
    import asyncio

    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import JSONResponse

    from . import __version__
    svc = Service(system, decider, storage, model_name, limits)
    lim = svc.limits
    app = FastAPI(title=title or "solvi", version=__version__,
                  description="Decisions from a solvi catalog: the model proposes, code decides, everything is in the trace.")
    app.state.service = svc
    app.add_middleware(Guard, limits=lim, token=token)
    if cors:                                          # added last: outermost, so preflight requests need no token
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(CORSMiddleware, allow_origins=list(cors), allow_methods=["GET", "POST"],
                           allow_headers=["Authorization", "Content-Type"])

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):     # never a traceback or a path to the client
        return JSONResponse({"detail": internal_error(f"{request.method} {request.url.path}")}, status_code=500)

    async def respond(where, work):
        """Run a request's work within the timeout → a JSONResponse, or an HTTP error that says what solvi refused."""
        try:
            out = await (asyncio.wait_for(work, lim.timeout) if lim.timeout is not None else work)
        except RequestError as e:
            raise HTTPException(e.status, str(e)) from None
        except asyncio.TimeoutError:
            raise HTTPException(504, f"the request did not finish within {lim.timeout:g} s") from None
        except Exception:  # noqa: BLE001 — logged with its traceback, the client gets an incident id
            raise HTTPException(500, internal_error(where)) from None
        return JSONResponse(out)

    def call(where, f, *args):                        # a sync System / the decider: in a worker thread
        return respond(where, asyncio.to_thread(f, *args))

    def acall(where, f, *args):                       # an async System: on the event loop (System.aask)
        return respond(where, f(*args))
    run_async = system is not None and svc.is_async   # async parts: the questions are asked with System.aask
    ask_with = acall if run_async else call

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

        @app.post("/ask", tags=["questions"], response_model=None)
        async def ask(body: AskRequest):
            return await ask_with("/ask", svc.aask if run_async else svc.ask, body.state, body.questions, body.store)
        documented["/ask"] = (None, with_ids(response_model(system), "AskResponse"))

        def route(name):
            async def ask_one(state: dict = Body(...)):
                return await ask_with(f"/ask/{name}", svc.aask if run_async else svc.ask, state, [name])
            ask_one.__name__ = f"ask_{name}"
            app.post(f"/ask/{name}", tags=["questions"], response_model=None, summary=system.questions[name].text or name,
                     description=f"Ask {name!r}: the body is the input state.")(ask_one)
            documented[f"/ask/{name}"] = (input_model(system, name),
                                          with_ids(response_model(system, [name]), _camel(name) + "Response"))
        for n in system.questions:
            route(n)

    if decider is not None:
        @app.post("/v1/systemone", tags=["system one"], response_model=SystemOneResponse)
        async def systemone(body: SystemOneRequest):
            return await call("/v1/systemone", svc.systemone, body)

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
    return tools


def tool_name(q):
    import re
    return re.sub(r"[^A-Za-z0-9_-]", "_", q)[:64]


def call_tool(svc, name, arguments):
    """tools/call → (result dict, is_error). A refused request says why; any other failure is logged on the server and
    the agent gets an incident id (never the exception's text)."""
    names = {tool_name(q): q for q in svc.system.questions}
    if name not in names:
        raise KeyError(name)
    try:
        return svc.tool(names[name], _arguments(svc, arguments)), False
    except RequestError as e:
        return {"error": str(e)}, True
    except Exception:  # noqa: BLE001 — a tool error is reported to the agent, not a protocol error
        return {"error": internal_error(f"tools/call {name}")}, True


def _arguments(svc, arguments):
    if arguments is None:
        return {}
    if not isinstance(arguments, dict):
        raise BadRequest(f"the arguments are a JSON object of given facts, not {type(arguments).__name__}")
    return dict(arguments)


async def acall_tool(svc, name, arguments):
    """tools/call on an event loop → (result dict, is_error): an async System is asked with aask, a sync one in a worker
    thread (its lock serializes the asks); either within the request timeout."""
    import asyncio
    names = {tool_name(q): q for q in svc.system.questions}
    if name not in names:
        raise KeyError(name)
    t = svc.limits.timeout
    try:
        if not svc.is_async:
            return await asyncio.wait_for(asyncio.to_thread(call_tool, svc, name, arguments), t)
        try:
            return await asyncio.wait_for(svc.atool(names[name], _arguments(svc, arguments)), t), False
        except RequestError as e:
            return {"error": str(e)}, True
        except asyncio.TimeoutError:
            raise
        except Exception:  # noqa: BLE001
            return {"error": internal_error(f"tools/call {name}")}, True
    except asyncio.TimeoutError:
        return {"error": f"the call did not finish within {t:g} s"}, True


def _readline(stdin, limit):
    """One line of at most `limit` characters → (line, too_long): a longer line is read to its end and dropped."""
    line = stdin.readline(limit + 1)
    if len(line) <= limit or line.endswith("\n"):
        return line, False
    while True:                                       # the rest of an over-long line: read and discard, a chunk at a time
        more = stdin.readline(65536)
        if not more or more.endswith("\n"):
            return "", True


def run_builtin(svc, stdin=None, stdout=None):
    """A stdio MCP server without the SDK: JSON-RPC 2.0, one message per line; initialize, ping, tools/list, tools/call
    (notifications are read and ignored). A message is at most `svc.limits.max_body` characters and max_depth deep; a
    tools/call runs in a worker thread and fails after `svc.limits.timeout` seconds (the thread cannot be stopped: it
    finishes in the background, and the System's lock makes later calls wait for it)."""
    from concurrent.futures import ThreadPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout

    from . import __version__
    from .schema import dumps
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    lim = svc.limits
    pool = ThreadPoolExecutor(4, thread_name_prefix="solvi-mcp")

    def send(msg):
        stdout.write(dumps(msg, ensure_ascii=False, default=repr) + "\n")
        stdout.flush()

    def error(id_, code, msg):
        send({"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": msg}})
    try:
        while True:
            line, too_long = _readline(stdin, lim.max_body)
            if too_long:
                error(None, -32600, f"invalid request: a message is at most {lim.max_body} characters")
                continue
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                msg = parse_json(line, lim)
            except RequestError as e:
                error(None, -32700 if str(e) == "the request is not JSON" else -32600,
                      "parse error" if str(e) == "the request is not JSON" else f"invalid request: {e}")
                continue
            if not isinstance(msg, dict) or "method" not in msg:
                if not (isinstance(msg, dict) and ("result" in msg or "error" in msg)):    # a response to us: none expected
                    error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
                continue
            method, params, id_ = msg["method"], msg.get("params") or {}, msg.get("id")
            if "id" not in msg:                            # a notification (notifications/initialized, cancelled, ...)
                continue
            if not isinstance(params, dict):
                error(id_, -32602, "invalid params: an object is expected")
                continue
            if method == "initialize":
                v = params.get("protocolVersion")
                send({"jsonrpc": "2.0", "id": id_, "result": {
                    "protocolVersion": v if v in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "solvi", "version": __version__},
                    "instructions": "Each tool is a question of a solvi decision system: pass the input state, get the "
                                    "answer with its confidence, why, safeguards and the id of the stored trace."}})
            elif method == "ping":
                send({"jsonrpc": "2.0", "id": id_, "result": {}})
            elif method == "tools/list":
                send({"jsonrpc": "2.0", "id": id_, "result": {"tools": mcp_tools(svc)}})
            elif method == "tools/call":
                name = params.get("name")
                try:
                    out, bad = pool.submit(call_tool, svc, name, params.get("arguments")).result(lim.timeout)
                except KeyError:
                    error(id_, -32602, f"unknown tool: {str(name)[:64]}")
                    continue
                except FutureTimeout:
                    out, bad = {"error": f"the call did not finish within {lim.timeout:g} s"}, True
                send({"jsonrpc": "2.0", "id": id_, "result": {
                    "content": [{"type": "text", "text": dumps(out, ensure_ascii=False, default=repr)}],
                    "structuredContent": out, "isError": bad}})
            else:
                error(id_, -32601, f"method not found: {str(method)[:64]}")
    finally:
        pool.shutdown(wait=False)


def run_sdk(svc):
    """A stdio MCP server with the official SDK (mcp 2.x: handlers passed to the low-level Server). The SDK reads the
    messages; the arguments of a tools/call are held to the same size and depth limits, the call to the timeout."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    from . import __version__
    from .schema import dumps

    async def list_tools(ctx, params):
        return types.ListToolsResult(tools=[types.Tool(name=t["name"], description=t["description"],
                                                       input_schema=t["inputSchema"]) for t in mcp_tools(svc)])

    async def on_call(ctx, params):
        try:
            parse_json(json.dumps(params.arguments or {}), svc.limits)    # the size and depth limits
            out, bad = await acall_tool(svc, params.name, params.arguments)
        except KeyError:
            out, bad = {"error": f"unknown tool: {str(params.name)[:64]}"}, True
        except RequestError as e:
            out, bad = {"error": str(e)}, True
        return types.CallToolResult(content=[types.TextContent(type="text", text=dumps(out, ensure_ascii=False,
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
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def load_decider(spec, backend="auto", pull=False):
    """`--decider`: a folder, a Hugging Face id already in the local cache, systemone:URL#model or module:attr (as
    `solvi ask --decider`, solvi.models.load). Nothing is downloaded unless `pull`: then a Hugging Face id that is not
    cached is pulled first (the ONNX weights, or the safetensors for backend="torch" / when onnxruntime is missing)."""
    from .models import ModelError, cached_path, kind_of, load
    from .models import pull as pull_model
    if pull and kind_of(spec) == "hub" and spec.count("/") == 1 and cached_path(spec) is None:
        which = backend if backend in ("onnx", "torch") else "onnx"
        if backend == "auto":
            try:
                import onnxruntime  # noqa: F401
            except ImportError:
                which = "torch"
        print(f"solvi serve: pulling {spec} ({which})", file=sys.stderr)
        pull_model(spec, which)
    try:
        return load(spec, backend)
    except ModelError as e:
        raise ModelError(f"{e} (or start with --pull)" if "solvi models pull" in str(e) else str(e)) from None


def cmd_serve(a):
    """`solvi serve` (see solvi.cli) → exit status."""
    from .cli import _fail, load_system
    system = load_system(a.system) if a.system else None
    decider = None
    if a.decider:
        from .models import ModelError
        try:
            decider = load_decider(a.decider, a.backend, a.pull)
        except ModelError as e:
            _fail(f"serve --decider: {e}")
    if system is None and decider is None:
        _fail("serve: name a System (module:attr or file.py:attr) and / or a decider (--decider)")
    if a.store and system is None:
        _fail("serve: --store needs a System")
    for k in ("max_body", "max_depth"):
        if getattr(a, k) < 1:
            _fail(f"serve --{k.replace('_', '-')}: must be at least 1")
    limits = Limits(a.max_body, a.max_depth, a.timeout if a.timeout and a.timeout > 0 else None)
    if a.mcp:
        if system is None:
            _fail("serve --mcp: needs a System (its questions are the tools)")
        if a.mcp_impl == "sdk" and not sdk_available():
            _fail("serve --mcp-impl sdk: the MCP SDK (mcp>=2) is not installed: pip install 'solvi[mcp]'")
        svc = Service(system, None, a.store, limits=limits)
        run_mcp(svc, a.mcp_impl)
        return 0
    try:
        import uvicorn
    except ImportError:
        _fail("serve: HTTP needs FastAPI and uvicorn: pip install 'solvi[serve]'")
    token = a.token or os.environ.get("SOLVI_SERVE_TOKEN") or None
    if token is None and a.host not in LOOPBACK:
        print(f"solvi serve: warning: listening on {a.host} without a token — anyone who can reach it can ask and store "
              "decisions (set SOLVI_SERVE_TOKEN)", file=sys.stderr)
    app = create_app(system, decider, a.store, a.model_name, title=f"solvi: {a.system}" if a.system else "solvi",
                     limits=limits, token=token, cors=a.cors)
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level, server_header=False)
    return 0


def add_parser(sub):
    """The `serve` subcommand's options (solvi.cli)."""
    s = sub.add_parser("serve", help="serve a System's questions over HTTP (FastAPI) or MCP, and a decider as System One")
    s.add_argument("system", nargs="?", help="module:attr or file.py:attr — a System or a function returning one")
    s.add_argument("--store", help="save every answer to this TraceStorage (.db / .sqlite: SQLite, else JSON lines)")
    s.add_argument("--decider", help="the decider behind POST /v1/systemone: a checkpoint folder, a cached Hugging Face id "
                                     "(see solvi models), systemone:URL#model or module:attr")
    s.add_argument("--pull", action="store_true", help="download --decider first when it is a Hugging Face id that is not "
                                                       "cached (without it, serve never downloads)")
    s.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="the decider's backend")
    s.add_argument("--model-name", help="the model name System One answers carry (default: the decider's id)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info")
    d = Limits()
    s.add_argument("--token", help="require `Authorization: Bearer TOKEN` on every HTTP request (default: "
                                   "$SOLVI_SERVE_TOKEN; prefer the variable: arguments are visible to other local users)")
    s.add_argument("--max-body", type=int, default=d.max_body, metavar="BYTES",
                   help=f"the largest request body / MCP message (default {d.max_body})")
    s.add_argument("--max-depth", type=int, default=d.max_depth,
                   help=f"the deepest nesting of JSON objects and arrays in a request (default {d.max_depth})")
    s.add_argument("--timeout", type=float, default=d.timeout, metavar="SECONDS",
                   help=f"seconds a request may take: then 504 / an MCP error (default {d.timeout:g}; 0: no limit)")
    s.add_argument("--cors", action="append", metavar="ORIGIN",
                   help="let browsers on this origin call the API (repeat; default: no CORS headers)")
    s.add_argument("--mcp", action="store_true", help="an MCP server over stdio instead of HTTP: each question is a tool")
    s.add_argument("--mcp-impl", default="auto", choices=["auto", "sdk", "builtin"],
                   help="the official MCP SDK (auto: when installed) or the built-in JSON-RPC subset")
    return s
