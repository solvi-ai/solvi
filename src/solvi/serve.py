"""`solvi serve`: a System's questions over HTTP (FastAPI) or as MCP tools, and a decider behind the System One API.

    solvi serve myapp/decisions.py:system --store decisions.db            # HTTP on 127.0.0.1:8000
    solvi serve myapp.decisions:system --decider solvi-ai/solvi-base      # + POST /v1/systemone (a cached model; --pull)
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
whole trace in a TraceStorage (hash-chained), and `solvi verify` / `replay` / `diff` work on that store — a request's
"store": false is honoured only with `--allow-client-no-store`.

MCP (`--mcp`): each question is a tool whose input schema is the question's input state schema; a call answers that
question and returns its result (answer, confidence, status, why, safeguards) with the stored id and trace hash; the tool
`ask_text` takes a free text (as POST /ask_text). It uses
the official `mcp` SDK (2.x, `solvi[mcp]`) when it is installed, else a built-in stdio JSON-RPC server with the subset of
the protocol that tools need (initialize, ping, tools/list, tools/call).

Security (see docs/guide.md, Serving): `--token` / $SOLVI_SERVE_TOKEN requires `Authorization: Bearer <token>` on every
HTTP request (constant-time compare); a request body / MCP message is at most `--max-body` bytes and `--max-depth` levels
of JSON; a System One request at most `--max-questions` questions of `--max-options` options; at most `--max-inflight`
requests run or wait at once (an async System's too), each sync one waiting at most `--queue-timeout` s for the
System (then 503 busy); a request
takes at most `--timeout` seconds (async Systems: through System.aask's part timeout, so the answer
abstains rather than the request failing); a failure the client did not cause is logged here and answered with an
incident id, never a traceback or a path — also inside an answer: a part that raised is in the response as its
exception's type and an incident id (records[].error, the alternatives tried, why, the safeguards' details), its text
only in the stored trace and the server log; CORS headers only with `--cors ORIGIN`; nothing is imported or loaded from
request data; `--decider` never downloads without `--pull`."""
import contextlib
import hmac
import json
import logging
import os
import re
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
    unresolved), so the inputs of alternative producers are optional. Planned by the system's own strategist
    (System(strategist=)), as `ask` plans."""
    from .strategist import PlanError, given_facts
    cat, q = system.catalog, system.questions[name]
    given = given_facts(cat, system.questions.values())
    try:
        flow = system._plan([q], given)
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
            if system._plan([q], given - {f}).unresolved.get(name):
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
    the request still answers. max_questions / max_options: questions in one System One request, options (criteria) of
    one of them — more is refused (422). max_inflight: requests answered at once — by a worker thread (running or
    waiting for the System; a request whose thread timed out still counts until the thread ends) or, for an async
    System, on the event loop — more are refused at once
    with 503 "busy"; queue_timeout: seconds a request waits for the System while another is being answered (None: as
    long as it takes) — then 503 "busy", instead of piling up threads behind a slow one."""
    max_body: int = 1_000_000
    max_depth: int = 32
    timeout: Optional[float] = 60.0
    max_questions: int = 32
    max_options: int = 64
    max_inflight: int = 8
    queue_timeout: Optional[float] = 10.0


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


class Busy(RequestError):
    """The server is answering as many requests as it may (Limits.max_inflight), or the System stayed busy longer than
    Limits.queue_timeout: try again later."""
    status = 503


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


# an exception's text as the runtime records it: "Type: message" (a step), "error: Type: message" (an alternative tried)
_EXC = re.compile(r"^(?:error: )?([A-Z][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*): ")


def redact(d, where="an answer"):
    """A response's data for a client, without the text of any exception a part raised → d (changed in place): each
    exception text in the trace's records (their error, the alternatives tried) — and wherever it is repeated (why, the
    safeguards' details, a group's "no producer accepted") — becomes "Type (incident …)". The full texts are logged on
    the server under the incident id and stay in the stored trace; the response's trace_hash is the stored trace's."""
    found = {}
    for r in ((d.get("trace") or {}).get("records") or []) if isinstance(d, dict) else []:
        cands = [r.get("error")] + [t[1] for t in (r.get("tried") or []) if isinstance(t, (list, tuple)) and len(t) > 1]
        for c in cands:
            if isinstance(c, str) and (m := _EXC.match(c)):
                text = c[len("error: "):] if c.startswith("error: ") else c
                found.setdefault(text, (m.group(1), r.get("name")))
    if not found:
        return d
    incident = uuid.uuid4().hex[:12]
    for text, (typ, part) in found.items():
        log.error("solvi serve: %s — part %s raised (incident %s): %s", where, part, incident, text)
    subs = sorted(found.items(), key=lambda kv: -len(kv[0]))           # the longest first: a text may contain another

    def clean(v):
        if isinstance(v, str):
            for text, (typ, _) in subs:
                if text in v:
                    v = v.replace(text, f"{typ} (incident {incident})")
            return v
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, list):
            return [clean(x) for x in v]
        if isinstance(v, tuple):
            return tuple(clean(x) for x in v)
        return v
    for k in list(d):
        if k not in ("stored_id", "trace_hash"):
            d[k] = clean(d[k])
    return d


# ------------------------------------------------------------------------------------------------ the service
class Service:
    """What the HTTP app and the MCP server call: asks a System (one at a time: a System learns costs and counts stats
    in place; a System with async parts is asked with System.aask, concurrently on the server's event loop), answers
    System One requests with a decider. `limits`: a Limits (the request size, JSON depth and timeout)."""

    def __init__(self, system=None, decider=None, storage=None, model_name=None, limits=None, textin=None,
                 allow_client_no_store=False):
        if system is None and decider is None:
            raise ValueError("solvi serve needs a System, a decider (--decider), or both")
        self.system, self.decider = system, decider
        self._textin = textin                         # a solvi.textin.TextIn (synonyms, patterns, ...), else made on use
        if system is not None and storage is not None:
            from .storage import open_storage
            system.storage = open_storage(storage, system)
        self.model_name = model_name or (getattr(decider, "model_id", None) if decider is not None else None)
        self.limits = limits or Limits()
        self.allow_client_no_store = bool(allow_client_no_store)   # else storing is the server's policy
        self._lock = threading.Lock()
        self._slots = threading.BoundedSemaphore(max(1, int(self.limits.max_inflight)))
        self._schemas = {}

    @contextlib.contextmanager
    def exclusive(self):
        """The System (or the decider) for one sync request: a slot among Limits.max_inflight (none free: Busy at once),
        then the lock, waited for at most Limits.queue_timeout (then Busy) — never an unbounded queue of threads."""
        if not self._slots.acquire(blocking=False):
            raise Busy(f"the server is busy ({self.limits.max_inflight} requests in flight): try again later")
        try:
            t = self.limits.queue_timeout
            if not self._lock.acquire(timeout=-1 if t is None else max(0.0, float(t))):
                raise Busy("the server is busy (another request holds the System): try again later")
            try:
                yield
            finally:
                self._lock.release()
        finally:
            self._slots.release()

    @contextlib.asynccontextmanager
    async def aslot(self):
        """An async request's slot among Limits.max_inflight (shared with the sync ones; none free: Busy at once) — an
        async System answers requests concurrently, so this is what bounds them."""
        if not self._slots.acquire(blocking=False):
            raise Busy(f"the server is busy ({self.limits.max_inflight} requests in flight): try again later")
        try:
            yield
        finally:
            self._slots.release()

    def storing(self, store):
        """Whether a request is stored: always (a server with a store keeps every answer), unless the server allows
        clients to opt out (allow_client_no_store) and this one did."""
        return bool(store) or not self.allow_client_no_store

    # --- a System
    def ask(self, state, names=None, store=True):
        """→ Response.to_dict() with "stored_id" and "trace_hash". store=False is honoured only when the server allows
        clients to opt out of storing (allow_client_no_store)."""
        resp = self._ask(state, names, self.storing(store))
        d = resp.to_dict()
        d["stored_id"], d["trace_hash"] = resp.stored_id, trace_hash(resp)
        return redact(d, "ask")

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
        clash = sorted(k for k in state if k in s.catalog.parts)
        if clash:                                     # a caller must not be able to stand in for a check or a computed fact
            raise BadRequest(f"the state has {', '.join(map(repr, clash))}: not a given fact of this system (the system "
                             "computes it)")
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
        with self.exclusive():
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
        async with self.aslot():
            resp = await self._aask(state, names, self.storing(store))
        d = resp.to_dict()
        d["stored_id"], d["trace_hash"] = resp.stored_id, trace_hash(resp)
        return redact(d, "ask")

    def tool(self, name, state):
        """An MCP tool call: one question → its result (answer, confidence, status, why, ...), the safeguards that fired
        for it, "stored_id" and "trace_hash"."""
        return self._tool_result(name, self._ask(state, [name], True))

    async def atool(self, name, state):
        """tool, for an async System (System.aask)."""
        async with self.aslot():
            return self._tool_result(name, await self._aask(state, [name], True))

    # --- a free text (System.ask_text)
    def textin(self, today=None):
        """The TextIn that reads texts for this server: the one given, else TextIn(system, decider) — made once; `today`
        (a date or ISO string; default: the server's date at the request, recorded in the trace) on a copy per request."""
        import copy
        import datetime as dt

        from .textin import TextIn
        if self.system is None:
            raise NotFound("this server has no System (only POST /v1/systemone)")
        if self._textin is None:
            self._textin = TextIn(self.system, self.decider)
        tin = copy.copy(self._textin)
        if today is not None:
            try:
                tin.today = dt.date.fromisoformat(today) if isinstance(today, str) else today
            except ValueError:
                raise BadRequest(f"today: not an ISO date: {str(today)[:32]!r}") from None
        elif tin.today is None:
            tin.today = dt.date.today()
        return tin

    def _reader(self, text, question, today):
        """The TextIn for one request, after the checks that need no model (a refused request takes no slot)."""
        if not isinstance(text, str) or not text.strip():
            raise BadRequest("the text is a non-empty string")
        tin = self.textin(today)
        if question is not None and question not in tin.entry_points:
            raise NotFound(f"no such entry point: {str(question)[:64]}")
        if question is None and len(tin.entry_points) > 1 and tin.decider is None:
            raise BadRequest("choosing the question a text asks needs a decider: start the server with --decider "
                             "(or pass question=)")
        return tin

    def _read_alone(self, tin, text, question):
        """tin.read under the lock (the decider answers one request at a time), waited for at most
        Limits.queue_timeout — then Busy."""
        t = self.limits.queue_timeout
        if not self._lock.acquire(timeout=-1 if t is None else max(0.0, float(t))):
            raise Busy("the server is busy (another request holds the decider): try again later")
        try:
            return tin.read(text, question=question)
        finally:
            self._lock.release()

    @staticmethod
    def _text_result(resp):
        d = resp.to_dict()
        read = resp.textin
        d["read"] = {**read.to_dict(), "clarify": read.clarify(), "escalated": read.escalated}
        d["stored_id"], d["trace_hash"] = resp.stored_id, trace_hash(resp)
        return redact(d, "ask_text")

    def ask_text(self, text, question=None, store=True, today=None):
        """A free text → Response.to_dict() of System.ask_text plus "read" (the question it asks, the fields read with
        their quotes, the missing ones, a clarifying question), "stored_id" and "trace_hash"."""
        tin = self._reader(text, question, today)
        with self.exclusive():                        # the decider's passes (routing, the fields) are inside the slot
            read = tin.read(text, question=question)  # and the lock, like the ask itself
            if self.is_async:
                from .runtime import run_sync
                resp = run_sync(self.system.aask_text(read, store=self.storing(store)))
            else:
                resp = self.system.ask_text(read, store=self.storing(store))
        return self._text_result(resp)

    async def aask_text(self, text, question=None, store=True, today=None):
        """ask_text, for an async System (System.aask_text)."""
        import asyncio
        tin = self._reader(text, question, today)
        async with self.aslot():
            # the decider's passes block: in a worker thread (one at a time), never on the event loop
            read = await asyncio.to_thread(self._read_alone, tin, text, question)
            return self._text_result(await self.system.aask_text(read, store=self.storing(store)))

    @staticmethod
    def _tool_result(name, resp):
        d = redact(resp.to_dict(), f"tool {name}")
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
        lim = self.limits
        if len(req.questions) > lim.max_questions:
            raise BadRequest(f"a System One request has at most {lim.max_questions} questions, got {len(req.questions)}")
        wide = [n for n, q in req.questions.items() if len(q.criteria or {}) > lim.max_options]
        if wide:
            raise BadRequest(f"a question has at most {lim.max_options} options (criteria): {str(wide[0])[:64]} has "
                             f"{len(req.questions[wide[0]].criteria or {})}")
        t0 = time.perf_counter()
        m = self.decider
        passes = m.passes
        answers = {}
        with self.exclusive():
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
    store: bool = Field(True, description="save the response to the server's store (if it has one); false is honoured only "
                                          "when the server was started with --allow-client-no-store")


class AskTextRequest(BaseModel):
    text: str = Field(description="a free text: a message, an e-mail, a chat turn")
    question: Optional[str] = Field(None, description="the question it asks (default: the decider picks the entry point)")
    store: bool = Field(True, description="save the response to the server's store (if it has one); false is honoured only "
                                          "when the server was started with --allow-client-no-store")
    today: Optional[str] = Field(None, description="ISO date for year-less and relative dates (default: the server's date)")


class Guard:
    """ASGI middleware in front of the app: the bearer token (constant-time compare), the body size (Content-Length, and
    the bytes actually received) and the JSON depth of a request body — refused before FastAPI parses it."""

    def __init__(self, app, limits, token=None):
        self.app, self.limits = app, limits
        if token is not None and not str(token).strip():
            raise ValueError("an empty bearer token would let any request with 'Authorization: Bearer ' in: "
                             "give a token, or None for no authentication")
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
               cors=None, textin=None, allow_client_no_store=False):
    """The FastAPI app (see the module docs). `system`: a System; `decider`: a DecideModel for POST /v1/systemone and for
    routing texts (POST /ask_text); `storage`: a TraceStorage or a path (every ask is stored); `model_name`: the model name
    System One answers carry; `limits`: a Limits (request size, JSON depth, timeout); `token`: every request must carry
    `Authorization: Bearer <token>` (None: no authentication); `cors`: the origins browsers may call it from (None: no
    CORS headers at all); `textin`: a solvi.textin.TextIn for POST /ask_text (default: TextIn(system, decider));
    `allow_client_no_store`: honour a request's "store": false (default: with a store, every answer is saved — the
    server's policy, not the client's). A token that is empty or blank is a configuration error (ValueError)."""
    if token is not None and not str(token).strip():
        raise ValueError("create_app(token=...): an empty token would accept 'Authorization: Bearer ' — give a token, "
                         "or None for no authentication")
    import asyncio

    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import JSONResponse

    from . import __version__
    svc = Service(system, decider, storage, model_name, limits, textin, allow_client_no_store)
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

        @app.post("/ask_text", tags=["questions"], response_model=None,
                  summary="A free text: the question it asks, its fields read with quotes, the answers")
        async def ask_text(body: AskTextRequest):
            return await ask_with("/ask_text", svc.aask_text if run_async else svc.ask_text, body.text, body.question,
                                  body.store, body.today)

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
    """tools/call → (result dict, is_error). A refused request says why; any other failure is logged on the server and
    the agent gets an incident id (never the exception's text)."""
    names = {tool_name(q): q for q in svc.system.questions}
    if name not in names and name != text_tool_name(svc):
        raise KeyError(name)
    try:
        if name == text_tool_name(svc):
            a = _arguments(svc, arguments)
            return svc.ask_text(a.get("text"), a.get("question")), False
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
    if name not in names and name != text_tool_name(svc):
        raise KeyError(name)
    t = svc.limits.timeout
    try:
        if not svc.is_async:
            return await asyncio.wait_for(asyncio.to_thread(call_tool, svc, name, arguments), t)
        try:
            if name == text_tool_name(svc):
                a = _arguments(svc, arguments)
                return await asyncio.wait_for(svc.aask_text(a.get("text"), a.get("question")), t), False
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


MCP_INSTRUCTIONS = ("Each tool is a question of a solvi decision system: pass the input state, get the answer with its "
                    "confidence, why, safeguards and the id of the stored trace.")


def run_builtin(svc, stdin=None, stdout=None):
    """A stdio MCP server without the SDK: JSON-RPC 2.0, one message per line; initialize, ping, tools/list, tools/call
    (notifications are read and ignored). A message is at most `svc.limits.max_body` characters and max_depth deep; a
    tools/call runs in a worker thread and fails after `svc.limits.timeout` seconds (the thread cannot be stopped: it
    finishes in the background, and a later call waits for the System at most `queue_timeout` seconds, then is told the
    server is busy). A malformed message is answered with a JSON-RPC error; nothing in a message stops the server."""
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
            try:
                if method == "initialize":
                    v = params.get("protocolVersion")
                    send({"jsonrpc": "2.0", "id": id_, "result": {
                        "protocolVersion": v if v in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[-1],
                        "capabilities": {"tools": {"listChanged": False}},
                        "serverInfo": {"name": "solvi", "version": __version__}, "instructions": MCP_INSTRUCTIONS}})
                elif method == "ping":
                    send({"jsonrpc": "2.0", "id": id_, "result": {}})
                elif method == "tools/list":
                    send({"jsonrpc": "2.0", "id": id_, "result": {"tools": mcp_tools(svc)}})
                elif method == "tools/call":
                    name, arguments = params.get("name"), params.get("arguments")
                    if not isinstance(name, str):
                        error(id_, -32602, "invalid params: tools/call takes a tool name (a string)")
                        continue
                    try:
                        out, bad = pool.submit(call_tool, svc, name, arguments).result(lim.timeout)
                    except KeyError:
                        error(id_, -32602, f"unknown tool: {name[:64]}")
                        continue
                    except FutureTimeout:
                        out, bad = {"error": f"the call did not finish within {lim.timeout:g} s"}, True
                    send({"jsonrpc": "2.0", "id": id_, "result": {
                        "content": [{"type": "text", "text": dumps(out, ensure_ascii=False, default=repr)}],
                        "structuredContent": out, "isError": bad}})
                else:
                    error(id_, -32601, f"method not found: {str(method)[:64]}")
            except Exception:  # noqa: BLE001 — a malformed message never stops the server; the details are in the log
                error(id_, -32603, internal_error(f"MCP {str(method)[:64]}"))
    finally:
        pool.shutdown(wait=False)


def run_sdk(svc):
    """A stdio MCP server with the official SDK (mcp 2.x: handlers passed to the low-level Server). The SDK reads the
    messages; the arguments of a tools/call are held to the same size and depth limits, the call to the timeout. It
    answers as the built-in server does: the same instructions at initialize, and an unknown tool is a JSON-RPC error
    (-32602), not a tool result. Two differences are the SDK's own: arguments that are not an object are its
    "invalid request parameters" error (the built-in server answers with a tool error), and a call still running when
    stdin closes gets no answer."""
    import anyio
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server
    from mcp.shared.exceptions import MCPError

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
            raise MCPError(types.INVALID_PARAMS, f"unknown tool: {str(params.name)[:64]}") from None
        except RequestError as e:
            out, bad = {"error": str(e)}, True
        return types.CallToolResult(content=[types.TextContent(type="text", text=dumps(out, ensure_ascii=False,
                                                                                       default=repr))],
                                    structured_content=out, is_error=bad)
    server = Server("solvi", version=__version__, instructions=MCP_INSTRUCTIONS, on_list_tools=list_tools,
                    on_call_tool=on_call)

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


def load_decider(spec, backend="auto", pull=False, api_key=None):
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
        return load(spec, backend, api_key=api_key)
    except ModelError as e:
        raise ModelError(f"{e} (or start with --pull)" if "solvi models pull" in str(e) else str(e)) from None


def cmd_serve(a):
    """`solvi serve` (see solvi.cli) → exit status."""
    from .cli import _fail, load_object, load_system
    if getattr(a, "guard", None) or getattr(a, "upstream", None):
        return _serve_guard(a, _fail, load_object)
    system = load_system(a.system) if a.system else None
    decider = None
    if a.decider:
        from .models import ModelError
        try:
            decider = load_decider(a.decider, a.backend, a.pull, getattr(a, "api_key", None))
        except ModelError as e:
            _fail(f"serve --decider: {e}")
    if system is None and decider is None:
        _fail("serve: name a System (module:attr or file.py:attr) and / or a decider (--decider)")
    if a.store and system is None:
        _fail("serve: --store needs a System")
    for k in ("max_body", "max_depth"):
        if getattr(a, k) < 1:
            _fail(f"serve --{k.replace('_', '-')}: must be at least 1")
    for k in ("max_questions", "max_options", "max_inflight"):
        if getattr(a, k) < 1:
            _fail(f"serve --{k.replace('_', '-')}: must be at least 1")
    limits = _limits(a)
    if a.mcp:
        if system is None:
            _fail("serve --mcp: needs a System (its questions are the tools)")
        if a.mcp_impl == "sdk" and not sdk_available():
            _fail("serve --mcp-impl sdk: the MCP SDK (mcp>=2) is not installed: pip install 'solvi[mcp]'")
        svc = Service(system, decider, a.store, limits=limits,      # the decider routes texts for the ask_text tool
                      allow_client_no_store=a.allow_client_no_store)
        run_mcp(svc, a.mcp_impl)
        return 0
    try:
        import uvicorn
    except ImportError:
        _fail("serve: HTTP needs FastAPI and uvicorn: pip install 'solvi[serve]'")
    if a.token is not None and not a.token.strip():
        _fail("serve --token: an empty token would accept 'Authorization: Bearer ' — give a token (or leave it out)")
    env = os.environ.get("SOLVI_SERVE_TOKEN")
    if env is not None and not env.strip() and not a.token:
        print("solvi serve: warning: SOLVI_SERVE_TOKEN is set but empty — no token is required", file=sys.stderr)
    token = a.token or (env if env and env.strip() else None)
    if token is None and a.host not in LOOPBACK:
        print(f"solvi serve: warning: listening on {a.host} without a token — anyone who can reach it can ask and store "
              "decisions (set SOLVI_SERVE_TOKEN)", file=sys.stderr)
    app = create_app(system, decider, a.store, a.model_name, title=f"solvi: {a.system}" if a.system else "solvi",
                     limits=limits, token=token, cors=a.cors, allow_client_no_store=a.allow_client_no_store)
    uvicorn.run(app, host=a.host, port=a.port, log_level=a.log_level, server_header=False)
    return 0


def _limits(a):
    """The Limits of `solvi serve`'s options (0 or less: no timeout / no queue timeout)."""
    return Limits(a.max_body, a.max_depth, a.timeout if a.timeout and a.timeout > 0 else None,
                  max_questions=a.max_questions, max_options=a.max_options, max_inflight=a.max_inflight,
                  queue_timeout=a.queue_timeout if a.queue_timeout and a.queue_timeout > 0 else None)


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
    for k in ("context_messages", "context_chars"):
        if getattr(a, k) < 0:
            _fail(f"serve --{k.replace('_', '-')}: must be at least 0 (0: no limit)")
    run_proxy(guard, a.upstream, facts=facts, escalate=a.escalate,
              limits=_limits(a), context_messages=a.context_messages or None, context_chars=a.context_chars or None)
    return 0


def add_parser(sub):
    """The `serve` subcommand's options (solvi.cli)."""
    s = sub.add_parser("serve", help="serve a System's questions over HTTP (FastAPI) or MCP, and a decider as System One")
    s.add_argument("system", nargs="?", help="module:attr or file.py:attr — a System or a function returning one")
    s.add_argument("--store", help="save every answer to this TraceStorage (.db / .sqlite: SQLite, else JSON lines)")
    s.add_argument("--decider", help="a decider behind POST /v1/systemone and routing POST /ask_text: a checkpoint "
                   "folder, a cached Hugging Face id (see solvi models), systemone:URL#model, llm:URL#model or module:attr")
    s.add_argument("--pull", action="store_true", help="download --decider first when it is a Hugging Face id that is not "
                                                       "cached (without it, serve never downloads)")
    s.add_argument("--api-key", help="for a systemone: / llm: decider (default $SOLVI_SYSTEMONE_API_KEY / "
                   "$SOLVI_LLM_API_KEY)")
    s.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="the decider's backend")
    s.add_argument("--model-name", help="the model name System One answers carry (default: the decider's id)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--log-level", default="info", choices=["critical", "error", "warning", "info", "debug", "trace"],
                   help="uvicorn's log level (default info)")
    d = Limits()
    s.add_argument("--token", help="require `Authorization: Bearer TOKEN` on every HTTP request (default: "
                                   "$SOLVI_SERVE_TOKEN; prefer the variable: arguments are visible to other local users)")
    s.add_argument("--max-body", type=int, default=d.max_body, metavar="BYTES",
                   help=f"the largest request body / MCP message (default {d.max_body})")
    s.add_argument("--max-depth", type=int, default=d.max_depth,
                   help=f"the deepest nesting of JSON objects and arrays in a request (default {d.max_depth})")
    s.add_argument("--timeout", type=float, default=d.timeout, metavar="SECONDS",
                   help=f"seconds a request may take: then 504 / an MCP error (default {d.timeout:g}; 0: no limit)")
    s.add_argument("--max-questions", type=int, default=d.max_questions, metavar="N",
                   help=f"the most questions in one POST /v1/systemone request (default {d.max_questions})")
    s.add_argument("--max-options", type=int, default=d.max_options, metavar="N",
                   help=f"the most options (criteria) of one System One question (default {d.max_options})")
    s.add_argument("--max-inflight", type=int, default=d.max_inflight, metavar="N",
                   help=f"requests answered at once by worker threads; more get 503 busy (default {d.max_inflight})")
    s.add_argument("--queue-timeout", type=float, default=d.queue_timeout, metavar="SECONDS",
                   help=f"seconds a request waits for the System while another is answered, then 503 busy (default "
                        f"{d.queue_timeout:g}; 0: no limit)")
    s.add_argument("--allow-client-no-store", action="store_true",
                   help='honour a request\'s "store": false (default: with --store, every answer is saved)')
    s.add_argument("--cors", action="append", metavar="ORIGIN",
                   help="let browsers on this origin call the API (repeat; default: no CORS headers)")
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
    from .agents.mcp import CONTEXT_CHARS, CONTEXT_MESSAGES
    s.add_argument("--context-messages", type=int, default=CONTEXT_MESSAGES, metavar="N",
                   help=f"with --guard: the tool outputs the session keeps for checking (default {CONTEXT_MESSAGES}; 0: all)")
    s.add_argument("--context-chars", type=int, default=CONTEXT_CHARS, metavar="N",
                   help=f"with --guard: their characters in all (default {CONTEXT_CHARS}; 0: no limit)")
    return s
