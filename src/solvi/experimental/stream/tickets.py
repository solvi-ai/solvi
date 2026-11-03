"""Tickets: an agent (or a person) as the oracle of a Dispatcher, over MCP — blind, durable, never silently dropped
(experimental; stage 3a).

    from solvi.experimental.stream import AgentOracle, TicketDesk

    d = Dispatcher(s1, AgentOracle(s1), audit="adaptive", promise=sp, detector=det, on_drift="recertify",
                   secret=key, storage=store)
    desk = TicketDesk(d, ttl=3600)
    out = desk.decide(state)          # {"status": "decided", ...} or a blind ticket {"status": "needs_verdict", ...}
    desk.resolve(out["ticket"], "billing")              # the agent's verdict: the decision is completed and recorded

The oracle. `AgentOracle` is the Dispatcher's slow path when the slow path is an agent outside the process: it never
runs in-process (`think` raises); every decision the slow path would take — an escalation (think) or a check (an audit
drawn by the calibration coin, a supervision draw, a drift check) — becomes a ticket.

The ticket (what the agent sees). {"status": "needs_verdict", "ticket", "question", "options", "input",
"instructions", "expires_in_s"}: the question's name, all of its options, the input as given, the question's text,
the time left. No model guess, no probabilities, no reason, no marker of kind: an audit of an answer System 1 gave
alone and an escalation have the same payload fields (the kind is kept server-side only, in the record). This hides
the kind in an isolated ticket; it does not make the whole interaction indistinguishable. An oracle with access to
`decide` can submit the same state as a new decision: an input System 1 can answer may then return `decided`, revealing
its answer, while an escalation can produce another ticket. The calibration coin is drawn per decision,
not fixed per input. Reusing a `request_id` returns the original payload, but a new request is another decision.

The certificate assumes that the oracle's label does not depend on whether the decision was sampled. Equal ticket
fields alone do not enforce this assumption: the application must account for the oracle's access to repeated
`decide` calls and other information about System 1 when relying on it.

Durability. A ticket is a hash-chained record of kind "pending" written when it is issued: "pending" — {ticket,
question, n, created, ttl, kind_hidden, rho, probation, sampled} (KEPT: a redaction leaves it) — and "issue", the
decision begun (`Dispatcher.start`: the input, System 1's response, the decision's number, draws, spend, drift flag
and dispatch). The decision's number, its coin and its rate are fixed at issue; the dispatch record (kind
"dispatch", with "pending": the ticket) is written when it is completed (`Dispatcher.complete`). A restart rebuilds
everything from the store: a new TicketDesk over a freshly built dispatcher on the same storage resumes it
(`Dispatcher.resume`: numbers, spend, the stream certificate, its detector and rate) and lists the tickets issued and
not completed; they resolve as before.

The verdict. `resolve(ticket, label, source="s2")`: the label must be one of the options (else TicketError, and the
ticket stays open); the slow path's contract, if any, applies as it does to an in-process slow path. The verdict is the
slow path's Thought (mode "agent"; its record {"oracle": {"ticket", "label_source", "label_of"}}); the dispatcher
completes the decision with it as with any slow path (on_disagree for a check, think="agree", ...). label_source:
"s2" (the agent is the slow path; the default, and what the MCP tool records) or "human" (a person gave it); label_of:
the stored id of the ticket's record. On a sampled decision the label goes to the stream certificate with both, in
the order of the decisions' numbers: a verdict that comes before an earlier sampled ticket's waits for it
(`sampling.StreamDriver`), so an agent slow on the hard cases cannot certify on the easy ones first.

Expiry and failure. A ticket not resolved within its ttl (checked on every call and when the desk starts) or given
up (`expire(ticket, why)`) gets a record of kind "pending_expired" and is completed with no verdict: the input goes to
a person (by "human", whatever the action was — an audited answer System 1 gave does not stand unchecked) and, on a
sampled decision, the label is missing: a loss at every grid point where System 1 would have answered alone, and a
loss of 1 for the detector (design §5.4: both factors are ≤ 1 for the wealth and ≥ 1 for the detector, so the
certificate and the detector keep their statements under any missingness, adversarial included). That loss of 1
whatever System 1 would have done makes the oracle's expiry rate part of the rate the detector tests — a stream at
risk r with expiry rate e reads as e + (1 − e)·r — so the expiry rate must sit well below α: at α = 5 %, an oracle
that expires 10 % of its tickets alarms by construction on a clean stream, and each alarm's restart opens probation
(ρ = 1: every decision a ticket), which raises the expiry rate further until the loop exhausts the budget (the
share-of-expired stop, `max_drop`, is deferred — see "Not here"). An unsampled
expired ticket is not a sample (adding it would raise the wealth at the strict points); it is recorded the same way.
A resolve after expiry is refused. Nothing is dropped.

Not here (later stages or not built): the share-of-expired stop (`max_drop`), repeated tickets for the oracle's
self-disagreement (ε̂), several servers on one store (one desk per store and question), the free-text input. A
replay of a store with tickets: `Dispatcher.replay_all()` and `replay_stream()` (stored=None) with this module
imported (the agent's Thoughts are read back by AgentOracle). Unmeasured."""
from __future__ import annotations

import math
import hashlib
import json
import copy
import os
import weakref
import secrets
import threading
import time

from ...core.costs import cost_of
from ...core.dispatch import Dispatcher, SlowPath, Started, Thought, _answer_type, _echo_type

TTL = 3600.0
UNKNOWN = "unknown, expired or already resolved ticket"
SOURCES = ("s2", "human")
_OWNER_LOCK = threading.Lock()
_POSTGRES_OWNERS = set()


class TicketError(ValueError):
    """A ticket that cannot be resolved (unknown, expired or already resolved) or a label that is not an answer."""


class AgentOracle(SlowPath):
    """The slow path when the slow path is an agent outside the process: answers come only through a TicketDesk. system:
    the System whose question the agent answers (its options, its contract — usually System 1 itself)."""

    mode = "agent"

    def think(self, state, question, **kw):
        raise RuntimeError("AgentOracle answers through tickets (TicketDesk.decide / resolve), not in the process: "
                           "Dispatcher.ask cannot run it")

    def fingerprint(self):
        from ...core.provenance import digest
        return digest("AgentOracle", self.mode, self.system.fingerprint(), self.question)

    def not_applicable(self, state, question, price=None):
        """The Thought of a path whose contract does not apply to the input (it is not asked), else None."""
        c = self.contract_for(self._q(question))
        if c is None:
            return None
        ok, why = c.applies(state)
        if ok:
            return None
        return Thought(self.mode, None, False, f"not applicable: {why}", None, cost=cost_of([], price, 0.0, []),
                       contract={"applicable": False, "why": why})

    def verdict(self, state, question, label, *, source="s2", ticket=None, of=None, price=None):
        """The agent's label → the slow path's Thought (accepted unless the contract refuses it)."""
        from ...core.contract import refusal
        q = self._q(question)
        th = Thought(self.mode, label, True, None, {"oracle": {"ticket": ticket, "label_source": source,
                                                               "label_of": of}})
        c = self.contract_for(q)
        if c is not None:
            v = c.violations(label, c.echo_inputs(state), _answer_type(self, q), echo_type=_echo_type(self, q))
            if v:
                th.accepted, th.why = False, refusal(v, label)
                th.contract = {"applicable": True, "violations": [x.to_dict() for x in v]}
        th.cost = cost_of([], price, 0.0, [])
        return th

    def lost(self, why, *, ticket=None, price=None):
        """No verdict (an expired or failed ticket) → the Thought that sends the input to a person."""
        return Thought(self.mode, None, False, why, {"oracle": {"ticket": ticket, "label_source": None,
                                                                "label_of": None, "lost": True}},
                       cost=cost_of([], price, 0.0, []))


class TicketDesk:
    """The tickets of one Dispatcher whose slow path is an AgentOracle (see the module docs). ttl: seconds a ticket
    waits for a verdict; clock: the time source (seconds; tests pass their own). One active desk per persistent
    store; use close() or a context manager before restarting. Memory/custom stores guard only the same object."""

    Error = TicketError                                  # what a server reports to the agent as its error

    def __init__(self, dispatcher, *, ttl: float = TTL, clock=time.time):
        if not isinstance(dispatcher, Dispatcher):
            raise TypeError("TicketDesk takes a solvi Dispatcher")
        if not isinstance(dispatcher.slow, AgentOracle):
            raise ValueError("the dispatcher's slow path is the agent: Dispatcher(system, AgentOracle(system), ...)")
        if dispatcher.storage is None:
            raise ValueError("tickets live in the store: Dispatcher(..., storage=TraceStorage)")
        ttl = float(ttl)
        if not ttl > 0 or math.isinf(ttl) or math.isnan(ttl):
            raise ValueError("ttl is a positive number of seconds (a ticket that never expires is never accounted)")
        self.dispatcher, self.ttl, self.clock = dispatcher, ttl, clock
        self.question = dispatcher.question
        self._lock = threading.RLock()
        self._open = {}                                  # ticket → {"started", "id", "created", "ttl", "kind"}
        self._requests = {}
        self._request_results = {}
        self._closed = False
        self._release_owner = self._claim_owner(dispatcher.storage)
        self._finalizer = weakref.finalize(self, self._release_owner)
        try:
            self._load()
            self.sweep()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _claim_owner(store):
        """Claim before replay or expiry. File leases survive object aliases, not process exit."""
        from ...core.store import JSONLStorage, SQLiteStorage, DuckDBStorage, PostgresStorage
        message = "TicketDesk store already has an active owner; close that desk before restarting"
        if isinstance(store, (JSONLStorage, SQLiteStorage, DuckDBStorage)) and store.path != ":memory:":
            from ...core.store import fcntl
            if fcntl is None:
                raise RuntimeError("TicketDesk persistent ownership requires POSIX file locks")
            path = os.path.realpath(os.path.abspath(store.path)) + ".tickets.owner"
            fh = open(path, "a+b")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                raise TicketError(message) from None
            except BaseException:
                fh.close()
                raise
            # Never unlink the lease file: another process may already have opened its inode.
            return fh.close
        # Object-local protection for memory/custom stores; no cross-process promise.
        # PostgreSQL additionally holds a session advisory lock for the actual head relation in this database.
        with _OWNER_LOCK:
            lock = getattr(store, "_ticket_owner_lock", None)
            if lock is None:
                lock = store._ticket_owner_lock = threading.Lock()
            if not lock.acquire(blocking=False):
                raise TicketError(message)
        if isinstance(store, PostgresStorage):
            import hashlib
            try:
                # Let PostgreSQL resolve the same unquoted identifier as _SQLStorage:
                # folding, identifier byte truncation and search_path belong to the server.
                relation = store.db.execute("SELECT to_regclass(%s)::oid", (store.prefix + "meta",)).fetchone()[0]
                if relation is None:
                    raise RuntimeError("TicketDesk store head relation does not exist")
                key = int.from_bytes(hashlib.sha256(("solvi TicketDesk:" + str(relation)).encode()).digest()[:8],
                                     "big", signed=True)
                connection_key = (id(store.db), relation)
            except BaseException:
                lock.release()
                raise
            with _OWNER_LOCK:
                if connection_key in _POSTGRES_OWNERS:
                    lock.release()
                    raise TicketError(message)
                _POSTGRES_OWNERS.add(connection_key)
            try:
                if not store.db.execute("SELECT pg_try_advisory_lock(%s)", (key,)).fetchone()[0]:
                    raise TicketError(message)
            except BaseException:
                with _OWNER_LOCK:
                    _POSTGRES_OWNERS.remove(connection_key)
                lock.release()
                raise
            def release():
                try:
                    store.db.execute("SELECT pg_advisory_unlock(%s)", (key,))
                finally:
                    with _OWNER_LOCK:
                        _POSTGRES_OWNERS.remove(connection_key)
                    lock.release()
            return release
        return lock.release

    def close(self):
        """Release ownership without expiring tickets; closed desks cannot be used again."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._finalizer()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *exc):
        self.close()

    def _ensure_open(self):
        if self._closed:
            raise TicketError("TicketDesk is closed")

    # --- the store
    def _load(self):
        d, q = self.dispatcher, self.question
        issued, done, any_record = {}, set(), False
        outcomes = {}
        redacted_dispatch = False
        for s in d.storage.iter(kind=None, redacted=True):
            r = s.data
            k = r.get("kind")
            if k == "dispatch" and r.get("redacted"):
                # Redaction removes question and decision number, so no retry key can
                # safely be proved unrelated to this outcome. Never re-decide it.
                redacted_dispatch = True
            if k == "ticket_request" and (r.get("ticket_request") or {}).get("question") == q:
                intent = r["ticket_request"]
                self._requests[intent["request_id"]] = intent
            if k == "dispatch" and r.get("question") == q:
                outcomes.setdefault(r["n"], ("dispatch", s.id, r))
                any_record = True
                if r.get("pending"):
                    done.add(r["pending"])
            elif k == "pending" and (r.get("pending") or {}).get("question") == q:
                any_record = True
                issued[r["pending"]["ticket"]] = (s.id, r)
                outcomes[r["pending"]["n"]] = ("pending", s.id, r)
        if redacted_dispatch and self._requests:
            raise TicketError("idempotent request outcome was redacted")
        for key, intent in self._requests.items():
            outcome = outcomes.get(intent["n"])
            if outcome is None:
                continue
            kind, rid, record = outcome
            if kind == "pending":
                b = record.get("issue")
                if b is None:
                    raise TicketError("idempotent request outcome was redacted")
                result = self.ticket_payload(record["pending"]["ticket"], b["input"])
                result["expires_in_s"] = round(record["pending"]["ttl"], 3)
            else:
                from ...core.dispatch import Dispatched
                from ...core.runtime import trace_hash
                out = Dispatched.from_dict(record, d.system)
                result = {"status": "decided", "answer": record["answer"], "by": record["by"],
                          "stored_id": rid, "trace_hash": trace_hash(out.s1)}
            self._request_results[key] = result
        if any_record and d.n == 0:
            d.resume()
        for t, (rid, r) in issued.items():
            if t in done:
                continue
            if r.get("issue") is None:
                raise ValueError(f"ticket {t} is open and its record was redacted: it cannot be completed (complete or "
                                 "expire open tickets before redacting them)")
            p = r["pending"]
            self._open[t] = {"started": Started.from_dict(r["issue"], d.system), "id": rid,
                             "created": float(p["created"]), "ttl": float(p["ttl"]), "kind": p.get("kind_hidden")}

    # --- the tools
    def decide(self, state, *, request_id=None) -> dict:
        """One input → {"status": "decided", "answer", "by", "stored_id", "trace_hash"} (System 1 answered alone, or a
        person gets it: by "human", answer None) or a blind ticket (see the module docs).
        request_id is an optional durable retry key, separate from state. Reusing it returns the original
        payload (even after resolution/expiry); another state is refused. An interrupted call requires replay
        of a fresh dispatcher before retry. Storage must durably append, and all decisions must use this desk."""
        from ...core.store import plain
        d, q = self.dispatcher, self.question
        with self._lock:
            self._ensure_open()
            if request_id is not None and (not isinstance(request_id, str) or not request_id.strip()):
                raise TicketError("request_id must be a nonempty string")
            digest = hashlib.sha256(json.dumps(plain(state), sort_keys=True, ensure_ascii=False,
                                               separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            intent = self._requests.get(request_id)
            if intent is not None:
                if intent["state_digest"] != digest:
                    raise TicketError("request_id already belongs to a different state")
                if request_id in self._request_results:
                    return copy.deepcopy(self._request_results[request_id])
            unresolved = set(self._requests) - set(self._request_results)
            if unresolved - {request_id}:
                raise TicketError("an interrupted request must be retried with its original request_id first")
            self.sweep()
            if intent is not None and d.n + 1 != intent["n"]:
                raise TicketError("interrupted request decision number changed; cannot safely retry")
            if request_id is not None and intent is None:
                from ...core.store import FORMAT
                intent = {"v": FORMAT, "kind": "ticket_request", "question": q, "request_id": request_id,
                          "state_digest": digest, "n": d.n + 1}
                try:
                    d.storage._append({"v": FORMAT, "kind": "ticket_request", "ticket_request": intent})
                except BaseException:
                    self.close()
                    raise
                self._requests[request_id] = intent
            try:
                return self._decide_request(state, request_id)
            except BaseException:
                # start/complete may already have mutated memory or committed a record. Replay is mandatory.
                self.close()
                raise

    def _decide_request(self, state, request_id):
        from ...core.runtime import trace_hash
        from ...core.store import plain
        d, q = self.dispatcher, self.question
        with self._lock:
            b = d.start(state)
            th = None
            if b.needs_slow:
                th = d.slow.not_applicable(state, q, d.price)
                if th is None:
                    result = self._issue(b)
                    if request_id is not None:
                        self._request_results[request_id] = copy.deepcopy(result)
                    return result
            out = d.complete(b, th)
            result = {"status": "decided", "answer": plain(out.answer), "by": out.by, "stored_id": out.stored_id,
                      "trace_hash": trace_hash(out.s1)}
            if request_id is not None:
                self._request_results[request_id] = copy.deepcopy(result)
            return result

    def _issue(self, b: Started) -> dict:
        from ...core.store import FORMAT, plain
        d, q = self.dispatcher, self.question
        t = secrets.token_urlsafe(18)                    # 24 url-safe characters
        now = float(self.clock())
        kind = "audit" if b.action == "check" else "think"
        rec = d.storage._append({"v": FORMAT, "kind": "pending", "init_hash": b.res.trace.init_hash,
                                 "pending": {"ticket": t, "question": q, "n": b.n, "created": now, "ttl": self.ttl,
                                             "kind_hidden": kind, "rho": b.rho, "probation": b.probation,
                                             "sampled": b.sampled},
                                 "issue": b.to_dict()})
        self._open[t] = {"started": b, "id": rec["id"], "created": now, "ttl": self.ttl, "kind": kind}
        return self.ticket_payload(t, plain(b.state))

    def ticket_payload(self, ticket, state, expires_in_s=None) -> dict:
        """What the agent sees of a ticket: a function of the ticket id, the question and the input only."""
        qq = self.dispatcher.system.questions[self.question]
        return {"status": "needs_verdict", "ticket": ticket, "question": self.question,
                "options": list(qq.answer.options), "input": state, "instructions": qq.text,
                "expires_in_s": round(self.ttl if expires_in_s is None else expires_in_s, 3)}

    def resolve(self, ticket, label, source: str = "s2") -> dict:
        """The verdict on an open ticket → {"status": "resolved", "answer" (what the dispatcher gives: on_disagree for
        a check), "stored_id"}. TicketError: an unknown, expired or already resolved ticket, or a label that is not an
        answer of the question (the ticket stays open)."""
        from ...core.store import plain
        if source not in SOURCES:
            raise ValueError(f"source is one of {SOURCES}: the agent as the slow path, or a person")
        d, q = self.dispatcher, self.question
        with self._lock:
            self._ensure_open()
            self.sweep()
            t = self._open.get(ticket) if isinstance(ticket, str) else None
            if t is None:
                raise TicketError(UNKNOWN)
            at = d.system.questions[q].answer
            try:
                label = at.normalize(label)
            except (ValueError, TypeError) as e:
                raise TicketError(f"the label is not an answer of {q!r}: {e}") from None
            if isinstance(label, tuple):
                label = list(label)
            b = t["started"]
            th = d.slow.verdict(b.state, q, label, source=source, ticket=ticket, of=t["id"], price=d.price)
            out = d.complete(b, th, pending=ticket)
            del self._open[ticket]
            return {"status": "resolved", "answer": plain(out.answer), "stored_id": out.stored_id}

    def expire(self, ticket, why=None):
        """Give up an open ticket now (a failed oracle) → its Dispatched: recorded as lost (see the module docs)."""
        with self._lock:
            self._ensure_open()
            if ticket not in self._open:
                raise TicketError(UNKNOWN)
            return self._lose(ticket, why or "the oracle gave up the ticket: a person decides", float(self.clock()))

    def sweep(self, now=None) -> list:
        """Expire every open ticket past its ttl → their Dispatched (each recorded as lost)."""
        with self._lock:
            self._ensure_open()
            now = float(self.clock()) if now is None else float(now)
            late = sorted((t for t, v in self._open.items() if now >= v["created"] + v["ttl"]),
                          key=lambda t: self._open[t]["started"].n)
            return [self._lose(t, f"the ticket expired after {self._open[t]['ttl']:g} s without a verdict: a person "
                                  "decides", now) for t in late]

    def _lose(self, ticket, why, now):
        from ...core.store import FORMAT
        d = self.dispatcher
        t = self._open[ticket]
        b = t["started"]
        d.storage._append({"v": FORMAT, "kind": "pending_expired",
                           "pending": {"ticket": ticket, "question": self.question, "n": b.n, "at": now,
                                       "why": why}})
        out = d.complete(b, d.slow.lost(why, ticket=ticket, price=d.price), pending=ticket)
        del self._open[ticket]
        return out

    def open(self) -> list:
        """The open tickets: [{"ticket", "age_s"}], oldest first (no content, no kind)."""
        now = float(self.clock())
        with self._lock:
            self._ensure_open()
            return [{"ticket": t, "age_s": round(now - v["created"], 3)}
                    for t, v in sorted(self._open.items(), key=lambda kv: kv[1]["started"].n)]

    # --- MCP (solvi serve --mcp --agent-oracle)
    def mcp_tools(self) -> list:
        """Decide, resolve, recover open tickets, or explicitly give up."""
        from ...core._inputs import input_schema
        qq = self.dispatcher.system.questions[self.question]
        return [
            {"name": "decide", "description":
                f"Decide one case of the question {self.question!r} ({qq.text}). The result is either the decision "
                "(status 'decided') or a ticket (status 'needs_verdict'): read the case and the question, choose one "
                "of the options yourself, and give it with the tool 'resolve' before the ticket expires.",
             "inputSchema": {"type": "object", "properties": {
                 "state": {**input_schema(self.dispatcher.system, self.question), "description": "the case: its input "
                                                                                                  "fields"},
                 "request_id": {"type": "string", "minLength": 1,
                                "description": "Optional durable retry key; reuse only with the same state."},
                 "question": {"type": "string", "enum": [self.question], "description": "the question (optional)"}},
                 "required": ["state"]}},
            {"name": "resolve", "description":
                "Give your verdict on a ticket from 'decide': the ticket id and one of its options as the label.",
             "inputSchema": {"type": "object", "properties": {
                 "ticket": {"type": "string", "description": "the ticket id"},
                 "label": {"enum": list(qq.answer.options), "description": "one of the ticket's options"}},
                 "required": ["ticket", "label"]}},
            {"name": "open_tickets", "description":
                "Recover outstanding tickets and their inputs after a lost response. Returns open tickets only; "
                "For an idempotent retry, reuse the original request_id with the original state in decide.",
             "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
            {"name": "give_up", "description":
                "Refuse a ticket now: record a failed oracle and send the case to a person without a verdict.",
             "inputSchema": {"type": "object", "properties": {
                 "ticket": {"type": "string", "description": "the ticket id"},
                 "why": {"type": "string", "description": "the reason for refusing (optional)"}},
                 "required": ["ticket"]}}]

    def call_tool(self, name, arguments) -> dict:
        """An MCP tool call → its result (TicketError and ValueError are the agent's errors; KeyError: no such tool)."""
        a = arguments or {}
        if name == "decide":
            if a.get("question") not in (None, self.question):
                raise TicketError(f"this server decides the question {self.question!r} only")
            state = a.get("state")
            if not isinstance(state, dict):
                raise TicketError("decide takes the case as 'state': a JSON object of its input fields")
            return self.decide(state, request_id=a.get("request_id"))
        if name == "resolve":
            if "label" not in a:
                raise TicketError("resolve takes a ticket and a label")
            return self.resolve(a.get("ticket"), a["label"])
        if name == "open_tickets":
            from ...core.store import plain
            with self._lock:
                self._ensure_open()
                self.sweep()
                now = float(self.clock())
                return {"tickets": [self.ticket_payload(t, plain(v["started"].state),
                                                        max(0, v["created"] + v["ttl"] - now))
                                    for t, v in sorted(self._open.items(), key=lambda kv: kv[1]["started"].n)]}
        if name == "give_up":
            ticket, why = a.get("ticket"), a.get("why")
            if not isinstance(ticket, str):
                raise TicketError("give_up takes a ticket id")
            if why is not None and not isinstance(why, str):
                raise TicketError("give_up reason 'why' must be a string")
            out = self.expire(ticket, why)
            return {"status": "expired", "ticket": ticket, "stored_id": out.stored_id}
        raise KeyError(name)


__all__ = ["AgentOracle", "SOURCES", "TTL", "TicketDesk", "TicketError", "UNKNOWN"]
