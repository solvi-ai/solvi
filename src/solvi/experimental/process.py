"""A process of several solvi systems (experimental, stages 0 to 3): the nodes, the edges with predicates written as
data, merges, loops with a mandatory iteration limit and the declared promise — a JSON specification with a hash, a
structural diff and a drawing (stage 0) — and running it: each node's decision in its own store, chained into one run,
with an index record of the path (stage 1a), and questions put to a person through a queue, a run waiting for the
answer and going on when it comes (stage 1b) — replaying a run from its records (stage 2) — and the path promise in
the modes union and per_node, with a watch on the mix of paths (stage 3). The end_to_end promise (stage 4) and
retraction come in later stages.

    from solvi.experimental.process import Process, MemoryQueue

    queue = MemoryQueue()                                  # or your own: class MyChannel(HumanChannel): ...
    p = Process("claim", storage="st/process.jsonl", human=queue)
    p.node("triage", triage, reads=["claim"], writes={"kind": "answer"})          # triage = solvi.build(...)
    p.node("fraud", fraud, reads=["claim", "kind"], writes={"fraud_risk": "answer"})
    p.node("payout", payout, reads=["claim", "kind", "fraud_risk"], writes={"payout": "answer"}, end=True)
    p.node("review", kind="human", question="fraud_risk", writes={"fraud_risk": "answer"})
    p.edge("triage", "fraud", when={"answer": "kind", "in": ["auto", "property"]})
    p.edge("triage", "payout", when={"answer": "kind", "eq": "glass"})
    p.edge("fraud", "payout", when={"status": "ok"})
    p.edge("fraud", "review", when={"status": "abstain"}); p.edge("review", "payout")
    p.edge("fraud", "triage", when={"drift": True}, loop=True, max_iter=2)    # a cycle without max_iter: ValueError
    p.promise(mode="union", risk=0.05)                                         # split="equal": r / K_auto per node

    spec = p.spec()                     # a JSON-ready dict; spec["hash"]: SHA-256 of its canonical JSON without "hash"
    canonical(spec)                     # the canonical JSON text (sort_keys, compact separators, as docs/trace_spec.md §1)
    q = Process.from_spec(spec, systems={"triage": triage, "fraud": fraud, "payout": payout})
    p.diff(q)                           # [] — or one entry per change: {"path", "change", "before", "after", "say"}
    p.render("mermaid")                 # or "text"

    r = p.run({"claim": claim})         # ProcessResult: status ("decided" | "abstain" | "pending" | "cancelled"),
                                        #   answer, path, nodes, facts, provenance, run, stored_id, ticket (pending)
    queue.answer(r.ticket, "low", reviewer="anna")      # the built-in queue; your channel answers its own way
    p.pending()                         # the waiting questions: [{run, node, ticket, question, asked_at, stored_id}]
    r2 = p.resume(r.run)                # polls the channel: an answer → the run goes on (a new segment), else pending
    print(p.explain(r2))                # the path in words: who answered at each node, which edge was taken
    chronology(stores, run=r.run)       # solvi.core.store.chronology: the node decisions and the records, in order

The format (`"solvi_process": "1.0"`). A node has a `kind` ∈ {decision, system, guard, human}: a `solvi.build` result,
a bare System, a Guard, or a question put to a person; `reads` (the facts it reads), `writes` ({fact: what of the
node's result}: "answer", "answer:<question>", "confidence", "by" or "status"), `question`, the system's `fingerprint`,
optionally its `store` and its own `promise` record. An edge is {"from", "to", "when"} (`when` null: always), and
{"loop": true, "max_iter": n} on an edge that goes back: every cycle of the graph has to pass through such an edge, so
a process cannot cycle without a limit. `merge` says per node whether it waits for all its inputs or any; `start` is the
one node without an incoming edge (or the one given); `end` lists the nodes marked end=True. `promise` is the declared
process promise (mode, risk, split, family, delta; the thresholds and the calibration are null until a later stage
calibrates them). Keys are sorted when hashed, so the hash does not depend on the order of keys; the order of edges is
kept (it is the order in which they are tried) and is part of the hash.

The predicate language is closed and is data, not Python: {"answer": fact, "eq" | "in" | "lt" | "gt": value} (a fact
written upstream or given), {"status": "ok" | "abstain" | "forced" | "human"}, {"by": "s1" | "s2" | "human"},
{"confidence_below": c}, {"budget_over": bool}, {"drift": bool}, {"and": [p, ...]}, {"or": [p, ...]}, {"not": p}.
`check_predicate` validates one, `holds(when, outcome)` evaluates it on an outcome {"facts", "status", "by",
"confidence", "budget_over", "drift"}, `describe` says it in words.

`from_spec` checks the hash and the format's major version, and the fingerprint of every system it is given against
the recorded one: a different system is an error (`ProcessSpecError`), or with strict=False is loaded and listed in
`p.mismatched` ({node: {"recorded", "actual"}}; the spec then carries the actual fingerprint). A node given no system
stays unbound (`p.unbound`): its spec is kept as recorded, so a process can be diffed and drawn without its systems.
A system's fingerprint: `System.fingerprint()` for a System; that and `Dispatcher.config()` for a `solvi.build` result;
the tools and the code of the policies for a Guard (stage 0: the adapters of stage 1 may refine it); an object with its
own `fingerprint()` otherwise (a Guard's is `Guard.fingerprint()`).

Running (stage 1a). `run(facts)` starts at the start node and follows one path: a node is given the facts it reads
(plain values: what the caller gave and what upstream nodes wrote — a fact not yet written is left out), its decision
is made by its own system and stored in its own store, and the first edge out of it whose predicate holds on its outcome
(in the order the edges were added; a loop edge until it has been taken max_iter times) leads to the next node. A node
is run through a NodeRunner — DecisionNode (a solvi.build result: ask → Dispatched, a "dispatch" record), SystemNode
(a System: ask → Response, an "ask" record), GuardNode (a Guard: call(facts["call"], context=facts["context"],
facts=the rest) → GuardDecision; an escalation has status "human") — so the decisions are exactly the ones calling the
systems by hand in the same order makes. Every fact a node writes carries a provenance label {"from": node, "decision":
stored_id, "by", "confidence", "trace_hash"} (result.provenance); a node that reads the fact "provenance" is given the
labels of the facts it reads as that given fact, and is otherwise given plain values only. Shared facts go through a
solvi.Knowledge the systems were built with (build(knowledge=)); the process passes nothing through it, and records the
hash of the snapshot a node was given. The run opens run_context(run=id) and each node runs inside
run_context(parent=the previous node's stored id), so its record carries the run and the decision it follows from, and
chronology(stores, run=id) puts the run in order without the process. With storage= the process appends one record of
kind "process" to its own store: {"process", "spec_hash", "input_hash", "mode", "promise", "path": [{node, kind, store,
stored_id, trace_hash, by, status, answers_hash, facts_in_hash, facts_out_hash, knowledge, cost, edge: {to, when,
loop}}], "outcome": {status, at, answer_hash}} — an index of hashes and ids; the decisions are in the node stores.
The run ends "decided" at an end node that answered, or "abstain" where it stopped (`why`). A node that abstains, or
whose decision goes to a person, with no edge for it: on_abstain / on_human = "abstain" stops the run there;
"handoff" (the default) puts the node's question to a person. A node that merges "all" of several inputs raises
NotImplementedError (this stage follows one path). Not yet: `retract`.

The channel to a person (stage 1b) is a queue: the process does not know when the person answers. `HumanChannel` is an
abstract class — submit(HumanRequest) → ticket, poll(ticket) → HumanAnswer | None, cancel(ticket, why),
fingerprint() — that you subclass with your own way to ask (check it with
solvi.testing.conformance.check_human_channel); built in: `MemoryQueue` (queue.answer(ticket, value, reviewer=),
queue.open()) and `CallbackChannel(fn)` (fn(request) → HumanAnswer or None, called at submit and at each poll). A
question is put by a node of kind "human" (an edge to it is an explicit hand-over; it is given the facts it reads, or
every fact when it reads none) or by the hand-off of a node. The process submits it and polls once: an answer is
recorded and the run goes on; none → the run ends "pending" with its ticket, as a "process" record (a segment) with
status "pending", the ticket, the channel's fingerprint, the request, and what the run needs to go on (the facts, their
labels, the loop counts — the one record that holds fact values). `p.pending()` lists the waiting questions;
`p.resume(run)` polls again: no answer → "pending" (nothing written); an answer → a record of kind "human" in the
process store ({ticket, node, question, reason, value, declined, reviewer, note, verified, asked_at, answered_at,
request_hash, channel}, run parent: the pending segment) and the run goes on from the node that asked as a new segment
("continues": the pending segment's id); a process loaded afresh from the same spec and stores resumes it. The person's
answer is the node's answer: its writes take the value, its outcome is status "ok", by "human" (so an edge for "ok"
holds, `{"by": "human"}` tells it apart), and the facts carry the label {"from", "decision": the human record,
"by": "human", "reviewer"}. One ticket, one answer: an answer already recorded stands (a run that broke after
recording it resumes from it). HumanAnswer(declined=True) — the person or the channel's own deadline gives no
answer — stops the run "abstain"; `p.cancel(run, why)` withdraws the question (the channel's cancel, a "human" record
{cancelled, why}, a segment "cancelled"). Process(user_verifier=fn(answer, request) → identity) binds an answer to
who gave it (answer.token is checked, not recorded). A run that asks a person needs storage=.

Replay (stage 2). `p.replay(run)` re-checks one run from the records alone → {"ok", "run", "status", "segments",
"steps", "at", "kind", "reason", "unchecked"} (+ "pending": ticket, + "cancelled": ticket). It walks the segments by
`continues` and each recorded path as `run` walks it, each node's result loaded from its store instead of asked:
"record" (the record intact in its store, of this run, after the decision before it, agreeing with the step),
"trace_hash", "replay" (the decision re-checked by its own means: Dispatcher.replay, Trace.replay, Guard.replay),
"facts" (the facts given are those the recorded answers upstream and the input give, in the step and in the node's
own record), "edge" (the first edge that holds on the recorded answers is the recorded one); "promise" (the
end_to_end rule) waits for stage 4. A person's answer is a leaf, "recorded, not recomputed": its "human" record must
be intact, follow from the pending segment, be the ticket's only answer, come through the same channel and answer the
request asked. "at" is the first divergence {"node", "step", "segment", "position"}; "kind" is "integrity" (records
disagree), "recompute" (a decision does not recompute, or the spec changed) or "missing_part" (a node without its
system or store). A pending run replays up to its question: {"ok": True, "pending": ticket}. `p.replay_all()` replays
every run and verify()s every store it reads. What the records do not keep is listed in "unchecked" (a guard's tool
result is stored as its hash).

The path promise (stage 3). `p.guarantee()` says what the declared promise gives → {"mode", "risk", "in_force",
"bound", "statement", "assumptions", "nodes", "problems", "warnings", "delta"}; each run that ends carries
`automatic` (decided with no person's answer on its path) and `risk` (the process risk while the union promise is in
force, else None), and `explain` ends with the promise. A run ends automatically when an end node answered and no
person answered on its path; E is "it ends automatically and its end answer is wrong". mode "union" (the default):
P(E) ≤ Σ_i r_i·m_i ≤ risk, summed over every automatic node of the graph (not over the path a run took), m_i an upper
bound on the times node i can run in one process run (1 plus the max_iter of each loop edge that can bring a run back
to it). split="equal" gives r_i = risk / K_auto; {node: r_i} or callable(spec) → {node: r_i} gives your own (e.g. by
the cost of a node's error), with Σ r_i ≤ risk checked when the spec is made. Why it holds: if the end is wrong, the
first wrong node of the path was given right inputs, so it is wrong on its own question, and it answered alone (else
the end is not automatic); E lies in the union of "node i answered alone and was wrong", each of probability ≤ r_i·m_i.
No independence between nodes is assumed (correlated errors make the bound looser, not wrong). It is in force when
every automatic node keeps a promise at its share — its own CRC risk ≤ r_i (node(promise=), else the dispatcher's
calibrated policy, else System 1's guarantee: build(max_risk=r_i)); an LTT node counts at its error level and the
promise then holds with probability ≥ 1 − Σ δ_i (Bonferroni, a warning) —, Σ r_i·m_i ≤ risk (a loop can push an equal
split over), no node was loaded against another fingerprint (strict=False), and the path watch has not flagged. It
assumes (A2) that each node was calibrated on what reaches it in the process — in place, on upstream's answers with
their errors and the routing of the edges; a node calibrated on clean inputs breaks it. mode "per_node": each node
keeps its own promise, and the statement says the end error is not bounded. mode "end_to_end" is stage 4
(guarantee() raises NotImplementedError; a run runs and carries no risk).

`PathWatch` watches the mix of paths: `p.monitor(PathWatch(alpha=, horizon=), reference=runs of the calibration
stream)`; every run that ends is observed (result.path_drift: {"drift", "why", "edge", "S", "h", "seen"}). Each
edge's share of runs is followed by two Bernoulli CUSUMs (its odds up and down by `odds`), h simulated by resampling
the reference runs with the reference's own error, so an unchanged mix is flagged within `horizon` runs with chance
at most alpha. A flag is latched until watch.reset() and puts the union promise out of force.
"""
from __future__ import annotations

import abc
import dataclasses
import hashlib
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import warn_on_import

warn_on_import(__name__)

FORMAT = "1.0"
KINDS = ("decision", "system", "guard", "human")
WRITES = ("answer", "confidence", "by", "status")          # or "answer:<question>"
STATUSES = ("ok", "abstain", "forced", "human")
BYS = ("s1", "s2", "human")
MERGES = ("all", "any")
ON = ("handoff", "abstain")                               # on_abstain / on_human
MODES = ("union", "end_to_end", "per_node")
FAMILIES = ("shared_rank", "per_node_fixed")
_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")


class ProcessSpecError(ValueError):
    """A process specification that is not valid (the message names the place), or one whose systems, channel or hash
    are not the recorded ones."""


def canonical(spec):
    """The canonical JSON text of a spec (or any JSON value): sort_keys, compact separators, UTF-8 as is."""
    return json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(spec):
    """SHA-256 (64 hex digits) of the canonical JSON of a spec without its "hash"."""
    body = {k: v for k, v in spec.items() if k != "hash"}
    return hashlib.sha256(canonical(body).encode("utf-8")).hexdigest()


# --- the predicate language

def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ProcessSpecError(f"{where}: a number is needed, not {v!r}")
    return v


def check_predicate(p, where="when"):
    """Validate a predicate (see the module docs); None is "always". → p. ProcessSpecError names the bad place."""
    if p is None:
        return p
    if not isinstance(p, dict) or not p:
        raise ProcessSpecError(f"{where}: a predicate is a non-empty dict, not {p!r}")
    keys = set(p)
    if "answer" in keys:
        ops = keys - {"answer"}
        if len(ops) != 1 or not ops <= {"eq", "in", "lt", "gt"}:
            raise ProcessSpecError(f"{where}: {{'answer': fact}} takes exactly one of eq, in, lt, gt; got {sorted(keys)}")
        if not isinstance(p["answer"], str) or not p["answer"]:
            raise ProcessSpecError(f"{where}: 'answer' names a fact (a non-empty string), not {p['answer']!r}")
        op = ops.pop()
        if op == "in" and not isinstance(p["in"], list):
            raise ProcessSpecError(f"{where}: 'in' takes a list, not {p['in']!r}")
        if op in ("lt", "gt"):
            _num(p[op], f"{where}.{op}")
        return p
    if len(keys) != 1:
        raise ProcessSpecError(f"{where}: one form per predicate (combine with and / or), got {sorted(keys)}")
    (k, v), = p.items()
    if k == "status" and v not in STATUSES:
        raise ProcessSpecError(f"{where}: status is one of {', '.join(STATUSES)}, not {v!r}")
    elif k == "by" and v not in BYS:
        raise ProcessSpecError(f"{where}: by is one of {', '.join(BYS)}, not {v!r}")
    elif k == "confidence_below":
        if not 0.0 <= _num(v, f"{where}.confidence_below") <= 1.0:
            raise ProcessSpecError(f"{where}: confidence_below is in [0, 1], not {v!r}")
    elif k in ("budget_over", "drift"):
        if not isinstance(v, bool):
            raise ProcessSpecError(f"{where}: {k} takes true or false, not {v!r}")
    elif k in ("and", "or"):
        if not isinstance(v, list) or not v:
            raise ProcessSpecError(f"{where}: {k} takes a non-empty list of predicates")
        for i, q in enumerate(v):
            if q is None:
                raise ProcessSpecError(f"{where}.{k}[{i}]: null is not a predicate here")
            check_predicate(q, f"{where}.{k}[{i}]")
    elif k == "not":
        if v is None:
            raise ProcessSpecError(f"{where}.not: null is not a predicate here")
        check_predicate(v, f"{where}.not")
    elif k not in ("status", "by"):
        raise ProcessSpecError(f"{where}: unknown predicate {k!r} (answer, status, by, confidence_below, budget_over, "
                               "drift, and, or, not)")
    return p


def holds(p, outcome):
    """Whether predicate p holds for a node's outcome {"facts": {name: value}, "status", "by", "confidence",
    "budget_over", "drift"} (missing keys: None / False). A fact that is absent satisfies no comparison."""
    if p is None:
        return True
    if "answer" in p:
        facts = outcome.get("facts") or {}
        if p["answer"] not in facts:
            return False
        v = facts[p["answer"]]
        if "eq" in p:
            return v == p["eq"]
        if "in" in p:
            return v in p["in"]
        try:
            return v < p["lt"] if "lt" in p else v > p["gt"]
        except TypeError:                             # a value that does not compare with a number
            return False
    (k, v), = p.items()
    if k in ("status", "by"):
        return outcome.get(k) == v
    if k == "confidence_below":
        c = outcome.get("confidence")
        return c is not None and c < v
    if k in ("budget_over", "drift"):
        return bool(outcome.get(k, False)) == v
    if k == "and":
        return all(holds(q, outcome) for q in v)
    if k == "or":
        return any(holds(q, outcome) for q in v)
    return not holds(v, outcome)


def describe(p):
    """A predicate in words (for render and diff)."""
    if p is None:
        return "always"
    if "answer" in p:
        op = next(k for k in p if k != "answer")
        val = p[op]
        sym = {"eq": "=", "lt": "<", "gt": ">"}.get(op)
        if sym:
            return f"{p['answer']} {sym} {json.dumps(val, ensure_ascii=False)}"
        return f"{p['answer']} in [{', '.join(json.dumps(x, ensure_ascii=False) for x in val)}]"
    (k, v), = p.items()
    if k in ("and", "or"):
        return "(" + f" {k} ".join(describe(q) for q in v) + ")"
    if k == "not":
        return f"not {describe(v)}"
    if k == "confidence_below":
        return f"confidence < {v}"
    if k in ("budget_over", "drift"):
        return k.replace("_", " ") if v else f"no {k.replace('_', ' ')}"
    return f"{k} = {v}"


# --- fingerprints

def fingerprint_of(system, kind=None):
    """What a node's system is, as JSON-ready data (see the module docs). kind: the node's kind when known."""
    if kind == "guard" or (kind is None and _is_guard(system)):
        return {"guard": system.fingerprint()}
    disp = getattr(system, "dispatcher", None)
    if disp is not None and callable(getattr(disp, "config", None)):
        return {"system": system.system.fingerprint(), "dispatcher": disp.config()}
    fp = getattr(system, "fingerprint", None)
    if callable(fp):
        v = fp()
        return {"system": v if isinstance(v, (dict, str)) else str(v)}
    raise ProcessSpecError(f"{type(system).__name__}: no fingerprint (a solvi.build result, a System, a Guard, or an "
                           "object with fingerprint())")


def _is_guard(system):
    """A solvi Guard, by its public surface (the guard lives in the agent layer, which this module does not import)."""
    return (type(system).__name__ == "Guard" and isinstance(getattr(system, "tools", None), dict)
            and all(callable(getattr(system, m, None)) for m in ("call", "fingerprint", "replay")))


def _kind_of(system):
    if _is_guard(system):
        return "guard"
    if callable(getattr(getattr(system, "dispatcher", None), "config", None)):
        return "decision"
    return "system"


def _channel(human):
    if human is None:
        return None
    fp = getattr(human, "fingerprint", None)
    return {"channel": type(human).__name__, "fingerprint": str(fp()) if callable(fp) else None}


def _plain(v):
    """JSON-ready copy (tuples → lists); raises on what JSON cannot hold."""
    return json.loads(json.dumps(v, ensure_ascii=False))


# --- running a node (stage 1): one protocol over the three kinds of system

@dataclass
class NodeResult:
    """What one node decided. answer: its main answer (the node's question's; for a System without one question, the
    dict of its answers; for a Guard, the outcome allow / deny / escalate); answers: {question: answer} ("answer:<q>"
    writes read them; a Guard's are "verdict" and "result", the tool's return value); by: "s1" | "s2" | "human";
    status: "ok" | "forced" | "abstain" | "human" (a person has to decide: a solvi.build result that sent it to a
    person, a Guard that escalated); confidence: the answer's (None when the slow path or nobody answered);
    stored_id / trace_hash: where the decision is in its store and the hash at the end of its trace; cost: {"s1",
    "s2", "total"} as dicts (a solvi.build result) or None; drift, budget_over: the dispatcher's flags; raw: the
    system's own result (a Dispatched, a Response or a GuardDecision); store: the TraceStorage it was written to."""
    answer: Any
    answers: dict
    by: str
    status: str
    confidence: float | None = None
    stored_id: str | None = None
    trace_hash: str | None = None
    cost: dict | None = None
    drift: bool = False
    budget_over: bool = False
    raw: Any = field(default=None, repr=False)
    store: Any = field(default=None, repr=False)

    def value(self, what):
        """What a `writes` entry takes from this result: "answer", "answer:<question>", "confidence", "by", "status"."""
        if what == "answer":
            return self.answer
        if what.startswith("answer:"):
            q = what[7:]
            if q not in self.answers:
                raise ProcessRunError(f"no answer to {q!r} in this node's result (it has {sorted(self.answers)})")
            return self.answers[q]
        return {"confidence": self.confidence, "by": self.by, "status": self.status}[what]


class ProcessRunError(RuntimeError):
    """A process that cannot go on: a node given no system, a node result that lacks what the node writes, a guard
    node without the fact "call"."""


class NodeRunner:
    """The protocol of a node: run(facts) → NodeResult, where facts are the node's given facts ({name: plain value}, the
    ones it reads); the decision is written to the system's own store (inside the process's run_context, so the record
    carries the run and the decision it follows from). A runner of your own subclasses it (and sets `kind`);
    `runner_for` picks the built-in one for a node."""
    kind = None

    def __init__(self, system, spec):
        self.system, self.spec = system, spec

    def run(self, facts):
        raise NotImplementedError

    @property
    def store(self):
        return getattr(self.system, "storage", None)


class DecisionNode(NodeRunner):
    """A solvi.build result (a DecisionSystem): ask(facts) → Dispatched; stored as a "dispatch" record."""
    kind = "decision"

    def run(self, facts):
        return _decision_result(self.system.ask(facts), self.store)


def _decision_result(d, store):
    """A Dispatched → the NodeResult of a decision node (a run and its replay read it the same way)."""
    from ..core.runtime import trace_hash
    r1 = d.s1.results.get(d.question) if d.s1 is not None else None
    if d.by == "human":
        status, conf = "human", None
    elif d.by == "s1":
        status, conf = (r1.status if r1 is not None else "ok"), (r1.confidence if r1 is not None else None)
    else:
        status, conf = "ok", None
    return NodeResult(d.answer, {d.question: d.answer}, d.by, status, conf, d.stored_id,
                      trace_hash(d.s1) if d.s1 is not None else None,
                      {k: v.to_dict() for k, v in (d.cost or {}).items()} or None,
                      bool(d.drift), bool(d.over_budget), d, store)


class SystemNode(NodeRunner):
    """A bare System: ask(facts) → Response; stored as an "ask" record when the System has storage. The main answer is
    the node's question's (question=, or the System's one question); without one, the dict of every answer."""
    kind = "system"

    def run(self, facts):
        return _system_result(self.system.ask(facts), self.spec, self.store)


def _system_result(resp, spec, store):
    """A Response → the NodeResult of a system node (a run and its replay read it the same way)."""
    from ..core.runtime import trace_hash
    q = spec.get("question")
    if q is None and len(resp.results) == 1:
        q = next(iter(resp.results))
    answers = {k: r.answer for k, r in resp.results.items()}
    if q is not None:
        if q not in resp.results:
            raise ProcessRunError(f"the system answered no question {q!r} (it answered {sorted(resp.results)})")
        r = resp.results[q]
        answer, status, conf = r.answer, r.status, r.confidence
    else:
        answer, status, conf = answers, ("ok" if resp.complete else "abstain"), resp.confidence
    return NodeResult(answer, answers, "s1", status, conf, resp.stored_id, trace_hash(resp), None, False, False,
                      resp, store)


class GuardNode(NodeRunner):
    """A Guard: call(facts["call"], context=facts["context"], facts=the other facts) → GuardDecision (the tool runs when
    allowed, as Guard.call does); the answer is the outcome, status "human" for an escalation; answers "verdict" and
    "result" (the tool's return value)."""
    kind = "guard"

    def run(self, facts):
        if "call" not in facts:
            raise ProcessRunError("a guard node reads the proposed call as the fact 'call' (reads=['call', ...])")
        rest = {k: v for k, v in facts.items() if k not in ("call", "context")}
        d = self.system.call(facts["call"], context=facts.get("context"), facts=rest or None)
        r = d.response.results.get("verdict") if d.response is not None else None
        return NodeResult(d.outcome, {"verdict": d.outcome, "result": d.result}, "s1",
                          "human" if d.outcome == "escalate" else "ok", r.confidence if r is not None else None,
                          d.stored_id, d.trace_hash if d.response is not None else None, None, False, False, d,
                          self.store)


class HumanNode(NodeRunner):
    """A question put to a person: the process asks it through its HumanChannel (Process(human=)), so a human node
    has no runner of its own to call; run() says so."""
    kind = "human"

    def run(self, facts):
        raise ProcessRunError("a human node is asked through the process's channel (Process(human=...)), not run")


RUNNERS = {"decision": DecisionNode, "system": SystemNode, "guard": GuardNode, "human": HumanNode}


def runner_for(kind, system, spec):
    """The built-in NodeRunner of a node kind for its system (a NodeRunner given as the system is used as is)."""
    if isinstance(system, NodeRunner):
        return system
    return RUNNERS[kind](system, spec)


@dataclass
class ProcessResult:
    """One run of a process (one segment of it). status: "decided" (an end node answered), "abstain" (it stopped
    before an end node answered: `why` says where), "pending" (a question waits for a person: `ticket`; resume(run)
    goes on), "cancelled" (the question was withdrawn: cancel(run, why)); answer: the end node's answer (None unless decided); path: the steps as recorded
    ([{node, kind, store, stored_id, trace_hash, by, status, answers_hash, facts_in_hash, facts_out_hash, knowledge,
    cost, edge}]; a person's answer is a step of kind "human" with its "ticket" and "handoff"); nodes: {node: NodeResult} (the last result of each node); facts: every fact at the end; provenance:
    {fact: {"from", "decision", "by", "confidence", "trace_hash"}} for the facts nodes wrote; run: the run id;
    stored_id: the "process" record (None without storage=); spec_hash; why: why it stopped (abstain);
    ticket: the question waiting for a person (status "pending"); continues: the pending segment this run goes on
    from (a resumed run); automatic: decided with no person's answer on this segment's path (a resumed run starts
    with the person's answer, so it is never automatic), None while pending; risk: the process risk the promise puts
    on an automatic end when it is in force (Process.guarantee()), else None; path_drift: the path watch's report
    (Process.monitor) when one watches the process."""
    status: str
    answer: Any
    path: list
    nodes: dict
    facts: dict
    provenance: dict
    run: str
    stored_id: str | None = None
    spec_hash: str | None = None
    why: str | None = None
    ticket: str | None = None
    continues: str | None = None
    automatic: bool | None = None
    risk: float | None = None
    path_drift: dict | None = None


# --- the channel to a person (stage 1b): a queue — the process does not know when the person answers

@dataclass
class HumanRequest:
    """A question put to a person. process, run, node: who asks; question: what (a human node's question, or the
    question of the node that handed its decision over); reason: "node" (a human node), "abstain" or "human" (a node
    that abstained or sent its decision to a person, on_abstain / on_human="handoff"); facts: what the node was given
    (plain values); provenance: the labels of those facts written upstream ({fact: {"from", "decision", "by", ...}});
    path: the nodes run so far in this segment; answer: the answer type ({"kind", "options"}) when the node's system
    names one, else None; deadline: a hint for the channel (the library has no timeouts: a channel that gives up
    answers HumanAnswer(declined=True)); asked_at: seconds since the epoch."""
    process: str
    run: str
    node: str
    question: str
    reason: str
    facts: dict
    provenance: dict = field(default_factory=dict)
    path: list = field(default_factory=list)
    answer: dict | None = None
    deadline: float | None = None
    asked_at: float | None = None

    def to_dict(self):
        return _plain(dataclasses.asdict(self))

    @classmethod
    def from_dict(cls, d):
        return cls(**{f.name: d.get(f.name) for f in dataclasses.fields(cls) if f.name in d})


@dataclass
class HumanAnswer:
    """A person's answer to one ticket: value (JSON data: what the node writes as its answer), reviewer (who answered:
    required), note, at (seconds since the epoch; default: when the process records it), token (a proof of who
    answered, checked by Process(user_verifier=); not recorded), declined=True: the person (or the channel, on its
    own deadline) gives no answer — the run stops "abstain" there. ticket: filled in by the process."""
    value: Any
    reviewer: str
    note: str | None = None
    at: float | None = None
    token: Any = field(default=None, repr=False)
    declined: bool = False
    ticket: str | None = None


class HumanChannel(abc.ABC):
    """How a process asks a person — a queue: submit a question, get a ticket, poll the ticket later. Subclass it with
    your own way to ask (a ticket system, a review UI, a chat) and check it with
    solvi.testing.conformance.check_human_channel. The protocol:

    submit(request: HumanRequest) → ticket (a non-empty string, new for every question)
    poll(ticket) → HumanAnswer, or None while there is no answer (KeyError for a ticket it never gave); once a ticket
                   has an answer, every later poll gives the same answer (one ticket, one answer)
    cancel(ticket, why) → None: withdraw the question; a later poll gives None
    fingerprint() → str: the class and its configuration (recorded in the spec and in every record of a question;
                   the same for two channels that ask the same way)
    """

    @abc.abstractmethod
    def submit(self, request):
        raise NotImplementedError

    @abc.abstractmethod
    def poll(self, ticket):
        raise NotImplementedError

    @abc.abstractmethod
    def cancel(self, ticket, why):
        raise NotImplementedError

    @abc.abstractmethod
    def fingerprint(self):
        raise NotImplementedError


def _new_ticket(prefix):
    return f"{prefix}-{uuid.uuid4().hex[:20]}"


class MemoryQueue(HumanChannel):
    """A queue in memory: questions wait until `answer(ticket, value, reviewer=)` (or `decline(ticket, why)`);
    `open()` lists the waiting requests. For tests, a CLI, a stand. Thread-safe; lost with the process that holds it
    (the process store keeps every question and answer)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._q = {}                                  # ticket → {"request", "answer", "cancelled"}

    def submit(self, request):
        t = _new_ticket("mq")
        with self._lock:
            self._q[t] = {"request": request, "answer": None, "cancelled": None}
        return t

    def poll(self, ticket):
        with self._lock:
            e = self._q[ticket]
            return None if e["cancelled"] is not None else e["answer"]

    def cancel(self, ticket, why):
        with self._lock:
            self._q[ticket]["cancelled"] = str(why)

    def fingerprint(self):
        return "MemoryQueue/1"

    def answer(self, ticket, value, reviewer, note=None, token=None):
        """A person answers a waiting question. A ticket is answered once (ValueError again, KeyError unknown)."""
        return self._set(ticket, HumanAnswer(value, reviewer, note, token=token))

    def decline(self, ticket, why=None, reviewer="queue"):
        """No answer will come (a deadline passed, the person refused): the run stops "abstain" at that node."""
        return self._set(ticket, HumanAnswer(None, reviewer, why, declined=True))

    def _set(self, ticket, ans):
        import time
        ans = dataclasses.replace(ans, at=time.time())
        with self._lock:
            e = self._q[ticket]
            if e["cancelled"] is not None:
                raise ValueError(f"ticket {ticket} was cancelled ({e['cancelled']})")
            if e["answer"] is not None:
                raise ValueError(f"ticket {ticket} is already answered (by {e['answer'].reviewer})")
            e["answer"] = ans
        return ans

    def open(self):
        """The waiting questions → [(ticket, HumanRequest)] in the order asked."""
        with self._lock:
            return [(t, e["request"]) for t, e in self._q.items() if e["answer"] is None and e["cancelled"] is None]

    def __len__(self):
        return len(self.open())


class CallbackChannel(HumanChannel):
    """A function asks the person: fn(request) → a HumanAnswer, or None (no answer yet). submit calls it, and so does
    every poll until it answers (a wrapper for manual input, a web hook, a synchronous review service). A plain value
    instead of a HumanAnswer is an answer by `reviewer` (required then). The fingerprint is the function's code."""

    def __init__(self, fn, reviewer=None):
        if not callable(fn):
            raise TypeError("CallbackChannel(fn): fn(request) → HumanAnswer or None")
        self.fn, self.reviewer = fn, reviewer
        self._lock = threading.Lock()
        self._q = {}

    def _call(self, request):
        a = self.fn(request)
        if a is None or isinstance(a, HumanAnswer):
            return a
        if self.reviewer is None:
            raise TypeError("the callback returned a plain value: return HumanAnswer(value, reviewer=...) or give "
                            "CallbackChannel(fn, reviewer=...)")
        return HumanAnswer(a, self.reviewer)

    def submit(self, request):
        t = _new_ticket("cb")
        with self._lock:
            self._q[t] = {"request": request, "answer": None, "cancelled": False}
        a = self._call(request)
        with self._lock:
            if self._q[t]["answer"] is None:
                self._q[t]["answer"] = a
        return t

    def poll(self, ticket):
        with self._lock:
            e = self._q[ticket]
            if e["cancelled"]:
                return None
            if e["answer"] is not None:
                return e["answer"]
            req = e["request"]
        a = self._call(req)
        with self._lock:
            if e["answer"] is None and not e["cancelled"]:
                e["answer"] = a
            return None if e["cancelled"] else e["answer"]

    def cancel(self, ticket, why):
        with self._lock:
            self._q[ticket]["cancelled"] = True

    def fingerprint(self):
        from ..core.provenance import code_fingerprint
        return f"CallbackChannel/1:{code_fingerprint(self.fn)}" + (f":{self.reviewer}" if self.reviewer else "")


def _answer_type(system, question):
    """The answer type of a node's question when its system names one → {"kind", "options"} or None."""
    if system is None:
        return None
    try:
        sysm = getattr(system, "system", None) or system
        qs = getattr(sysm, "questions", None)
        at = getattr(qs.get(question), "answer", None) if isinstance(qs, dict) else None
        return None if at is None else _plain({"kind": at.kind, "options": list(at.options or [])})
    except Exception:  # noqa: BLE001 — a hint for the person only
        return None


# --- the process

class Process:
    """A process of solvi systems (see the module docs). name: the process's name; storage: a path or a TraceStorage
    where each run's "process" record is appended (not part of the spec; None: no record, the node stores still hold
    every decision; a run that asks a person needs it); human: the channel to a person (a HumanChannel: MemoryQueue,
    CallbackChannel or your own; recorded in the spec by its fingerprint); start: the start node (default: the one
    node without an incoming edge); on_abstain, on_human: "handoff" (to the channel) or "abstain"; user_verifier:
    fn(answer: HumanAnswer, request: HumanRequest) → the verified identity of who answered (from answer.token), or
    None to refuse it (PermissionError; the identity must be the answer's reviewer), as Guard(user_verifier=)."""

    def __init__(self, name, storage=None, human=None, start=None, on_abstain="handoff", on_human="handoff",
                 user_verifier=None):
        if not isinstance(name, str) or not name:
            raise ProcessSpecError(f"a process name is a non-empty string, not {name!r}")
        for k, v in (("on_abstain", on_abstain), ("on_human", on_human)):
            if v not in ON:
                raise ProcessSpecError(f"{k} is one of {', '.join(ON)}, not {v!r}")
        self.name, self.storage, self.human, self._start = name, storage, human, start
        self.on_abstain, self.on_human = on_abstain, on_human
        if user_verifier is not None and not callable(user_verifier):
            raise TypeError("user_verifier is a function (answer, request) → the verified identity, or None")
        self.user_verifier = user_verifier
        self.nodes: dict[str, dict] = {}              # name → the node's spec (without systems)
        self.systems: dict[str, object] = {}          # name → the bound system
        self.edges: list[dict] = []
        self.merges: dict[str, str] = {}
        self._promise = None                          # the declared promise: values, or a callable split kept until spec()
        self._human_spec = None                       # a channel as recorded, kept for a loaded spec without a channel
        self.examples = None                          # end_to_end calibration examples (used by a later stage)
        self.mismatched: dict[str, dict] = {}
        self.unbound: list[str] = []
        self._store = None                            # the opened storage= (on the first run)
        self.path_watch = None                        # a PathWatch on the mix of paths (monitor())
        self._edges_of: dict[str, set] = {}           # run → the edges its pending segments took (for the watch)

    # --- building

    def node(self, name, system=None, *, kind=None, reads=(), writes=None, question=None, end=False, merge=None,
             store=None, promise=None, fingerprint=None):
        """Add a node. system: a solvi.build result, a System or a Guard (kind inferred), or None for kind="human" (a
        question put to a person: question= is required). reads: fact names; writes: {fact: "answer" | "answer:<q>" |
        "confidence" | "by" | "status"}; end: an end node; merge: "all" | "any" (for a node with several inputs);
        store: where its decisions are stored (default: the system's storage path); promise: the node's own promise
        record (Promise.record()); fingerprint: overrides the computed one. → self."""
        if not isinstance(name, str) or not _NAME.match(name):
            raise ProcessSpecError(f"a node name is a letter or _ then letters, digits, _ or -: not {name!r}")
        if name in self.nodes:
            raise ProcessSpecError(f"node {name!r} is already in the process")
        if kind is None:
            kind = "human" if system is None else _kind_of(system)
        if kind not in KINDS:
            raise ProcessSpecError(f"node {name!r}: kind is one of {', '.join(KINDS)}, not {kind!r}")
        if kind == "human" and system is not None:
            raise ProcessSpecError(f"node {name!r}: a human node has no system (the process's channel asks the person)")
        if question is None and isinstance(getattr(system, "question", None), str):
            question = system.question                # a solvi.build result's question
        if kind == "human" and not question:
            raise ProcessSpecError(f"node {name!r}: a human node needs question=")
        if question is not None and not isinstance(question, str):
            raise ProcessSpecError(f"node {name!r}: question is a string, not {question!r}")
        if store is None and system is not None:
            path = getattr(getattr(system, "storage", None), "path", None)
            store = path if isinstance(path, str) else None
        if fingerprint is None and system is not None:
            fingerprint = fingerprint_of(system, kind)
        spec = {"kind": kind, "reads": list(reads), "writes": dict(writes or {})}
        if question is not None:
            spec["question"] = question
        if end:
            spec["end"] = True
        if store is not None:
            spec["store"] = str(store)
        if fingerprint is not None:
            spec["fingerprint"] = fingerprint
        if promise is not None:
            spec["promise"] = promise
        spec = _plain(spec)
        _check_node(name, spec)
        self.nodes[name] = spec
        if system is not None:
            self.systems[name] = system
        if merge is not None:
            self.merge(name, merge)
        return self

    def merge(self, node, how):
        """Say whether `node` waits for all its inputs ("all") or the first ("any"). → self."""
        if node not in self.nodes:
            raise ProcessSpecError(f"merge: no node {node!r}")
        if how not in MERGES:
            raise ProcessSpecError(f"merge of {node!r} is one of {', '.join(MERGES)}, not {how!r}")
        self.merges[node] = how
        return self

    def edge(self, src, dst, when=None, loop=False, max_iter=None):
        """Add an edge src → dst taken when the predicate `when` holds on src's outcome (None: always). An edge that
        goes back (closes a cycle) is loop=True with max_iter ≥ 1; a cycle without such an edge is a ProcessSpecError
        (a ValueError). → self."""
        e = {"from": src, "to": dst, "when": _plain(when)}
        if loop or max_iter is not None:
            e["loop"], e["max_iter"] = bool(loop), max_iter
        _check_edge(e, self.nodes, f"edge {src} -> {dst}")
        self.edges.append(e)
        try:
            _check_acyclic(self.nodes, self.edges)
        except ProcessSpecError:
            self.edges.pop()
            raise
        return self

    def promise(self, mode="union", risk=None, split="equal", delta=None, examples=None, family=None):
        """Declare the process promise (computed by a later stage; here its configuration is recorded and checked).
        mode: "union" (risk r split over the automatic nodes: split="equal" — r / K_auto each —, {node: r_i} with
        Σ r_i ≤ r, or callable(spec) → {node: r_i}), "end_to_end" (risk, delta, examples=[(input, end_label)],
        family="shared_rank" | "per_node_fixed" | {"kind": "per_node_fixed", "order": [...]} | a ThresholdFamily
        object with fingerprint()) or "per_node" (each node keeps its own promise). → self."""
        if mode not in MODES:
            raise ProcessSpecError(f"promise mode is one of {', '.join(MODES)}, not {mode!r}")
        pr = {"mode": mode, "risk": None, "split": None, "family": None, "delta": None, "thresholds": None,
              "calibrated_on": None}
        if mode in ("union", "end_to_end"):
            if not 0.0 < _num(risk, "promise.risk") < 1.0:
                raise ProcessSpecError(f"promise risk is in (0, 1), not {risk!r}")
            pr["risk"] = risk
        elif risk is not None:
            raise ProcessSpecError("promise mode per_node takes no risk (each node keeps its own)")
        if mode == "union":
            if callable(split):
                pr["split"] = split                    # called at spec() with the spec so far
            elif split == "equal":
                pr["split"] = {"kind": "equal"}
            elif isinstance(split, dict):
                pr["split"] = {"kind": "given", "per_node": _plain(split)}
            else:
                raise ProcessSpecError(f'promise split is "equal", {{node: r_i}} or a callable, not {split!r}')
        if mode == "end_to_end":
            if not 0.0 < _num(delta, "promise.delta") < 1.0:
                raise ProcessSpecError(f"promise delta is in (0, 1), not {delta!r}")
            pr["delta"] = delta
            pr["family"] = _family(family if family is not None else "shared_rank")
            self.examples = None if examples is None else list(examples)
        elif family is not None or delta is not None:
            raise ProcessSpecError(f"promise mode {mode} takes no family or delta (end_to_end does)")
        self._promise = pr
        return self

    # --- the spec

    def spec(self):
        """The process as data (see the module docs), validated, with its "hash". → a JSON-ready dict."""
        _check_graph(self.nodes, self.edges, self.merges, self._start)
        out = {"solvi_process": FORMAT, "name": self.name, "nodes": _plain(self.nodes), "edges": _plain(self.edges),
               "start": _start_of(self.nodes, self.edges, self._start),
               "end": sorted(n for n, s in self.nodes.items() if s.get("end")),
               "merge": dict(sorted(self.merges.items())), "on_abstain": self.on_abstain, "on_human": self.on_human,
               "human": _channel(self.human) if self.human is not None else self._human_spec, "promise": None}
        if self._promise is not None:
            out["promise"] = self._promise_spec(out)
        out = _plain(out)
        out["hash"] = spec_hash(out)
        return out

    def _promise_spec(self, spec):
        pr = dict(self._promise)
        auto = [n for n, s in self.nodes.items() if s["kind"] != "human"]
        sp = pr["split"]
        if callable(sp):
            from ..core.provenance import code_fingerprint
            per = sp(_plain({k: v for k, v in spec.items() if k != "promise"}))
            pr["split"] = {"kind": "callable", "fingerprint": code_fingerprint(sp), "per_node": _plain(per)}
        elif isinstance(sp, dict) and sp.get("kind") == "equal":
            pr["split"] = {"kind": "equal", "per_node": {n: pr["risk"] / len(auto) for n in auto} if auto else {}}
        if isinstance(pr["split"], dict) and pr["split"].get("kind") in ("given", "callable"):
            _check_split(pr["split"]["per_node"], pr["risk"], auto)
        fam = pr["family"]
        if isinstance(fam, dict) and fam["kind"] == "per_node_fixed" and fam["order"] is None:
            fam = dict(fam, order=_topological(self.nodes, self.edges, auto))
        if isinstance(fam, dict) and fam.get("order") is not None:
            bad = [n for n in fam["order"] if n not in auto]
            if bad or len(set(fam["order"])) != len(fam["order"]):
                raise ProcessSpecError(f"promise family order: unknown, human or repeated nodes {bad or fam['order']}")
        pr["family"] = fam
        return pr

    @classmethod
    def from_spec(cls, spec, systems=None, human=None, storage=None, strict=True, user_verifier=None):
        """A process from its spec (a dict or JSON text). systems: {node: system} to bind (each checked against its
        recorded fingerprint); human: the channel (checked likewise); storage, user_verifier: as Process() — a process
        loaded with the store of the one that asked resumes its pending runs. strict=False loads a mismatch and lists it in
        `mismatched` instead of raising. A node given no system is listed in `unbound`."""
        if isinstance(spec, (str, bytes)):
            spec = json.loads(spec)
        if not isinstance(spec, dict) or "solvi_process" not in spec:
            raise ProcessSpecError("not a process spec (no 'solvi_process' version)")
        ver = str(spec["solvi_process"]).split(".")
        if ver[0] != FORMAT.split(".")[0]:
            raise ProcessSpecError(f"process spec format {spec['solvi_process']} (this solvi reads {FORMAT.split('.')[0]}.x)")
        if "hash" in spec and spec["hash"] != spec_hash(spec) and strict:
            raise ProcessSpecError(f"process spec hash {spec['hash']} does not match its content ({spec_hash(spec)}): "
                                   "the spec was changed after it was written (strict=False loads it anyway)")
        systems = dict(systems or {})
        unknown = sorted(set(systems) - set(spec.get("nodes") or {}))
        if unknown:
            raise ProcessSpecError(f"systems for nodes the spec does not have: {unknown}")
        p = cls(spec["name"], storage=storage, start=spec.get("start"), on_abstain=spec.get("on_abstain", "handoff"),
                on_human=spec.get("on_human", "handoff"), user_verifier=user_verifier)
        for name, ns in (spec.get("nodes") or {}).items():
            if name in systems:
                sysm = systems[name]
                actual = fingerprint_of(sysm, ns.get("kind"))
                rec = ns.get("fingerprint")
                if canonical(actual) != canonical(rec):
                    if strict:
                        raise ProcessSpecError(f"node {name!r}: the system given is not the recorded one (fingerprint "
                                               f"{_short(actual)} against {_short(rec)}; diff the specs to see what "
                                               "changed, or strict=False to load it marked)")
                    p.mismatched[name] = {"recorded": rec, "actual": actual}
                    ns = dict(ns, fingerprint=actual)
                p.systems[name] = sysm
            elif ns.get("kind") != "human":
                p.unbound.append(name)
            ns = _plain(ns)
            if not isinstance(name, str) or not _NAME.match(name):
                raise ProcessSpecError(f"a node name is a letter or _ then letters, digits, _ or -: not {name!r}")
            _check_node(name, ns)
            p.nodes[name] = ns
        for i, e in enumerate(spec.get("edges") or []):
            e = _plain(e)
            _check_edge(e, p.nodes, f"edges[{i}]")
            p.edges.append(e)
        for n, how in (spec.get("merge") or {}).items():
            p.merge(n, how)
        rec_h = spec.get("human")
        if human is not None:
            act = _channel(human)
            if rec_h is not None and canonical(act) != canonical(rec_h):
                if strict:
                    raise ProcessSpecError(f"the channel given ({act['channel']} {act['fingerprint']}) is not the "
                                           f"recorded one ({rec_h.get('channel')} {rec_h.get('fingerprint')})")
                p.mismatched["human"] = {"recorded": rec_h, "actual": act}
            p.human = human
        else:
            p._human_spec = rec_h
        pr = spec.get("promise")
        if pr is not None:
            p._promise = _load_promise(pr)
        _check_graph(p.nodes, p.edges, p.merges, p._start)
        return p

    # --- running (stage 1a; the channel to a person and pending / resume: stage 1b)

    def run(self, facts, run=None):
        """Run the process once on the given facts ({name: value}) → ProcessResult (see the module docs). run: the run
        id (default: the enclosing run_context's, else a new one). Every node's decision is written to its own store
        inside run_context(run=..., parent=the previous node's stored id); with storage= the process writes one record
        of kind "process" (the index of the path: hashes, ids, the edges taken) whose parent is the last node's
        decision. A question put to a person that is not answered at once ends the run "pending" with its ticket:
        `resume(run)` goes on when the person has answered."""
        from ..core.store import current_run, run_context
        spec = self.spec()
        self._check_bound()
        outer = current_run()
        facts = dict(facts)
        w = {"node": spec["start"], "facts": facts, "prov": {}, "taken": [0] * len(self.edges),
             "parent": outer["parent"] if outer else None, "input_hash": _vh(facts), "continues": None}
        with run_context(run=run) as rc:
            return self._go(spec, rc.id, w)

    def _check_bound(self):
        lack = [n for n, ns in self.nodes.items() if ns["kind"] != "human" and n not in self.systems]
        if lack:
            raise ProcessRunError(f"nodes without a system: {lack} (Process.from_spec(spec, systems=...))")

    def _given(self, ns, facts, prov):
        """The facts a node is given: the ones it reads (a human node that reads nothing: every fact), plain values,
        with the labels as the fact "provenance" when it reads it."""
        reads = ns["reads"] or (list(facts) if ns["kind"] == "human" else [])
        state = {r: facts[r] for r in reads if r in facts and r != "provenance"}
        if "provenance" in reads:
            state["provenance"] = {r: prov[r] for r in reads if r in prov}
        return state

    def _go(self, spec, run_id, w, first=None):
        """Follow the path from w["node"] (inside run_context(run=run_id)) until an end, a stop or a question left
        pending. first: (NodeResult, state) of a person's answer that the first node takes (a resumed run)."""
        from ..core.store import run_context
        facts, prov, taken, parent = w["facts"], w["prov"], w["taken"], w["parent"]
        path, results = [], {}
        node, status, answer, why = w["node"], None, None, None

        def step(node, ns, res, state, kind):
            st, edge = self._step(node, ns, res, state, kind, facts, prov, taken)
            results[node] = res
            path.append(st)
            return edge

        while True:
            ns = self.nodes[node]
            if self.merges.get(node) == "all" and sum(1 for e in self.edges if e["to"] == node
                                                      and not e.get("loop")) > 1:
                raise NotImplementedError(f"node {node!r} merges 'all' of several inputs: this stage follows one "
                                          "path (the first edge that holds); waiting for several branches comes "
                                          "later")
            if first is not None:
                (res, state), first = first, None
                kind = "human"
            elif ns["kind"] == "human":
                state = self._given(ns, facts, prov)
                res = self._ask(spec, run_id, node, ns["question"], "node", state, w, path, parent)
                if isinstance(res, ProcessResult):
                    return res
                kind = "human"
            else:
                runner = runner_for(ns["kind"], self.systems[node], ns)
                state = self._given(ns, facts, prov)
                with run_context(parent=parent):
                    res = runner.run(state)
                kind = ns["kind"]
            edge = step(node, ns, res, state, kind)
            if res.stored_id is not None:
                parent = res.stored_id
            if edge is None and res.status in ("human", "abstain") and kind != "human":   # not a person's answer
                on = self.on_human if res.status == "human" else self.on_abstain
                if on == "handoff":                   # the node's decision goes to a person
                    q = ns.get("question") or node
                    res = self._ask(spec, run_id, node, q, res.status, state, w, path, parent)
                    if isinstance(res, ProcessResult):
                        return res
                    edge = step(node, ns, res, state, "human")
                    if res.stored_id is not None:
                        parent = res.stored_id
            if edge is not None:
                node = edge["to"]
                continue
            if res.by == "human" and res.status == "abstain":
                status, why = "abstain", f"{node}: the person did not answer ({res.raw.note or 'declined'})"
            elif res.status in ("human", "abstain"):
                status, why = "abstain", f"{node}: {'a person has to decide' if res.status == 'human' else 'abstained'}"
            elif ns.get("end"):
                status, answer = "decided", res.answer
            else:
                status, why = "abstain", f"{node}: no edge holds and it is not an end node"
            break
        with run_context(parent=parent):              # the record follows from the last node's decision
            rec_id = self._record(spec, w["input_hash"], path, status, answer, node, continues=w["continues"])
        res = ProcessResult(status, answer, path, results, facts, prov, run_id, rec_id, spec["hash"], why,
                            continues=w["continues"])
        return self._finish(spec, res)

    def _finish(self, spec, res):
        """A run that ended (not pending): automatic or not, the risk in force, the path watch's look at it."""
        res.automatic = res.status == "decided" and not any(st["kind"] == "human" or st["by"] == "human"
                                                            for st in res.path)
        edges = self._edges_of.pop(res.run, set()) | _taken(self.edges, res.path)
        if self.path_watch is not None:
            res.path_drift = self.path_watch.observe(sorted(edges), names=_edge_names(self.edges))
        pr = spec.get("promise")
        if pr is not None and pr["mode"] == "union":           # end_to_end: stage 4; per_node bounds no end risk
            g = self.guarantee(spec)
            res.risk = g["risk"] if g["in_force"] else None
        return res

    def _step(self, node, ns, res, state, kind, facts, prov, taken):
        """One step of a path: the node's writes and their labels go into facts / prov, the first edge out of the node
        that holds (counted in taken) is chosen → (the step as recorded, the edge or None). A run and its replay both
        take a step through here, so the replay recomputes exactly what the run recorded."""
        declined = res.by == "human" and res.status == "abstain"
        out = {} if declined else {f: res.value(what) for f, what in ns["writes"].items()}
        label = {"from": node, "decision": res.stored_id, "by": res.by, "confidence": res.confidence,
                 "trace_hash": res.trace_hash}
        if res.by == "human" and isinstance(res.raw, HumanAnswer):
            label["reviewer"] = res.raw.reviewer
        facts.update(out)
        prov.update({f: dict(label) for f in out})
        outcome = {"facts": facts, "status": res.status, "by": res.by, "confidence": res.confidence,
                   "budget_over": res.budget_over, "drift": res.drift}
        edge = None
        if not declined:                              # a person who declined: the run stops
            for i, e in enumerate(self.edges):
                if e["from"] != node or (e.get("loop") and taken[i] >= e["max_iter"]):
                    continue
                if holds(e.get("when"), outcome):
                    edge, taken[i] = e, taken[i] + 1
                    break
        kh = state.get("knowledge")
        if kh is None and res.raw is not None and getattr(res.raw, "s1", None) is not None:
            kh = res.raw.s1.trace.init.get("knowledge")
        st = {"node": node, "kind": kind, "store": ns.get("store") if kind != "human" else self._store_name(),
              "stored_id": res.stored_id, "trace_hash": res.trace_hash, "by": res.by, "status": res.status,
              "answers_hash": _vh_kept(res.answers), "facts_in_hash": _vh_kept(state),
              "facts_out_hash": _vh_kept(out), "knowledge": None if kh is None else _vh(kh), "cost": res.cost,
              "edge": None if edge is None else {"to": edge["to"], "when": edge.get("when"),
                                                 "loop": bool(edge.get("loop"))}}
        if kind == "human":
            st["ticket"] = res.raw.ticket if isinstance(res.raw, HumanAnswer) else None
            st["handoff"] = ns["kind"] != "human"
        return st, edge

    def _open(self):
        from ..core.store import open_storage
        if self._store is None:
            self._store = open_storage(self.storage)
        return self._store

    def _store_name(self):
        path = getattr(self._open(), "path", None) if self.storage is not None else None
        return None if path is None else str(path)

    def _record(self, spec, input_hash, path, status, answer, at, continues=None, extra=None):
        """A "process" record (an index: the node stores hold the decisions) → its id, or None without storage."""
        if self.storage is None:
            return None
        from ..core.store import FORMAT as RECORD_FORMAT
        pr = spec.get("promise")
        body = {"v": RECORD_FORMAT, "kind": "process", "process": spec["name"], "spec_hash": spec["hash"],
                "input_hash": input_hash, "mode": pr["mode"] if pr else None, "promise": pr, "path": _plain(path),
                "outcome": {"status": status, "at": at, "answer_hash": _vh(answer)}, "continues": continues}
        body.update(extra or {})
        return self._open()._append(body)["id"]

    # --- the channel to a person (stage 1b)

    def _channel_obj(self):
        ch = self.human
        if ch is None:
            raise ProcessRunError("a question for a person, and the process has no channel to one: Process(human="
                                  "MemoryQueue() or your HumanChannel), or on_abstain / on_human='abstain'")
        if not isinstance(ch, HumanChannel):
            raise ProcessRunError(f"the channel {type(ch).__name__} is not a HumanChannel (subclass it: submit, poll, "
                                  "cancel, fingerprint)")
        return ch

    def _ask(self, spec, run_id, node, question, reason, state, w, path, parent):
        """Put a question to a person: submit it; an answer at once is recorded and returned as the node's
        NodeResult; otherwise the run ends with a "pending" segment → ProcessResult."""
        ch = self._channel_obj()
        if self.storage is None:
            raise ProcessRunError("a question for a person is recorded with its answer: Process(storage=...) is needed")
        import time
        resume = {"node": node, "reason": reason, "facts": _plain(w["facts"]), "provenance": _plain(w["prov"]),
                  "taken": list(w["taken"]), "state": _plain(state)}   # raises before submitting if not JSON data
        req = HumanRequest(process=self.name, run=run_id, node=node, question=question, reason=reason,
                           facts=_plain({k: v for k, v in state.items() if k != "provenance"}),
                           provenance=_plain({k: w["prov"][k] for k in state if k in w["prov"]}),
                           path=[st["node"] for st in path], answer=_answer_type(self.systems.get(node), question),
                           asked_at=time.time())
        ticket = ch.submit(req)
        if not isinstance(ticket, str) or not ticket:
            raise ProcessRunError(f"{type(ch).__name__}.submit returned {ticket!r}: a ticket is a non-empty string")
        if self._ticket_seen(ticket):
            raise ProcessRunError(f"{type(ch).__name__} gave ticket {ticket!r} again: a ticket names one question")
        ans = ch.poll(ticket)
        if ans is None:
            from ..core.store import run_context
            with run_context(parent=parent):
                rec_id = self._record(spec, w["input_hash"], path, "pending", None, node, continues=w["continues"],
                                      extra={"ticket": ticket, "channel": _channel(ch), "asking": req.to_dict(),
                                             "resume": resume})
            self._edges_of.setdefault(run_id, set()).update(_taken(self.edges, path))
            return ProcessResult("pending", None, path, {}, w["facts"], w["prov"], run_id, rec_id, spec["hash"],
                                 f"{node}: waiting for a person (ticket {ticket})", ticket, continues=w["continues"])
        return self._answered(ans, req, ticket, parent, ch)

    def _ticket_seen(self, ticket):
        st = self._open()
        return any(s.data.get("ticket") == ticket for k in ("process", "human") for s in st.iter(k))

    def _human_record(self, ticket):
        for s in self._open().iter("human"):
            if s.data.get("ticket") == ticket and s.data.get("process") == self.name:
                return s
        return None

    def _answered(self, ans, req, ticket, parent, ch):
        """A person's answer → the "human" record (once per ticket) and the NodeResult the node takes."""
        from ..core.store import FORMAT as RECORD_FORMAT
        from ..core.store import run_context
        import time
        old = self._human_record(ticket)
        if old is None and not isinstance(ans, HumanAnswer):
            raise ProcessRunError(f"{type(ch).__name__}.poll returned {type(ans).__name__}: a HumanAnswer or None")
        if old is None and (not isinstance(ans.reviewer, str) or not ans.reviewer):
            raise ProcessRunError(f"the answer to ticket {ticket} names no reviewer: HumanAnswer(value, reviewer=...)")
        if old is not None:                           # one ticket, one answer: the recorded one stands
            d = old.data
            ans = HumanAnswer(d.get("value"), d.get("reviewer"), d.get("note"), d.get("answered_at"),
                              declined=bool(d.get("declined")), ticket=ticket)
            rec_id = old.id
        else:
            verified = None
            if self.user_verifier is not None and not ans.declined:
                who = self.user_verifier(ans, req)
                if not who:
                    raise PermissionError(f"the answer to ticket {ticket} by {ans.reviewer!r} did not verify")
                if str(who) != ans.reviewer:
                    raise PermissionError(f"the answer was given by {who!r}, not by the reviewer {ans.reviewer!r}")
                verified = str(who)
            try:
                value = _plain(ans.value)
            except (TypeError, ValueError) as e:
                raise ProcessRunError(f"the answer to ticket {ticket} is not JSON data: {e}") from None
            body = {"v": RECORD_FORMAT, "kind": "human", "process": self.name, "ticket": ticket, "node": req.node,
                    "question": req.question, "reason": req.reason, "value": None if ans.declined else value,
                    "declined": bool(ans.declined), "reviewer": ans.reviewer, "note": ans.note,
                    "verified": verified, "asked_at": req.asked_at,
                    "answered_at": ans.at if ans.at is not None else time.time(),
                    "request_hash": _vh(req.to_dict()), "channel": _channel(ch)}
            with run_context(run=req.run, parent=parent):
                rec_id = self._open()._append(body)["id"]
            ans = dataclasses.replace(ans, ticket=ticket)
        if ans.declined:
            return NodeResult(None, {}, "human", "abstain", None, rec_id, None, raw=ans, store=self._store)
        return NodeResult(ans.value, {req.question: ans.value}, "human", "ok", None, rec_id, None, raw=ans,
                          store=self._store)

    def _segments(self):
        return [s for s in self._open().iter("process") if s.data.get("process") == self.name] \
            if self.storage is not None else []

    def _tip(self, run):
        rid = run.run if isinstance(run, ProcessResult) else run
        segs = [s for s in self._segments() if s.run == rid]
        if not segs:
            raise ProcessRunError(f"no record of run {rid!r} in this process's store")
        return rid, segs[-1]

    def pending(self):
        """The questions waiting for a person: the pending segments with no continuation → [{"run", "node", "ticket",
        "question", "asked_at", "stored_id"}] in the order they were asked."""
        segs = self._segments()
        cont = {s.data.get("continues") for s in segs}
        return [{"run": s.run, "node": s.data["outcome"]["at"], "ticket": s.data["ticket"],
                 "question": s.data["asking"]["question"], "asked_at": s.data["asking"]["asked_at"],
                 "stored_id": s.id}
                for s in segs if s.data["outcome"]["status"] == "pending" and s.id not in cont]

    def _resumable(self, run):
        rid, tip = self._tip(run)
        d = tip.data
        if d["outcome"]["status"] != "pending":
            raise ProcessRunError(f"run {rid} is not pending (its last segment is {d['outcome']['status']!r})")
        ch = self._channel_obj()
        if canonical(_channel(ch)) != canonical(d.get("channel")):
            raise ProcessRunError(f"run {rid} asked through {d['channel']}, not through this channel ({_channel(ch)})")
        spec = self.spec()
        if spec["hash"] != d["spec_hash"]:
            raise ProcessRunError(f"run {rid} was run by process spec {d['spec_hash'][:12]}, this one is "
                                  f"{spec['hash'][:12]}: resume it with the process that asked")
        return rid, tip, ch, spec

    def resume(self, run):
        """Go on with a pending run (its id or its ProcessResult): poll the channel for the ticket; no answer yet → a
        ProcessResult "pending" again (nothing is written); an answer → it is recorded (kind "human", parent: the
        pending segment) and the run goes on from the node that asked, as a new segment ("continues": the pending
        segment's id) → its ProcessResult. A process loaded afresh from the same spec and stores resumes it."""
        from ..core.store import run_context
        self._check_bound()
        rid, tip, ch, spec = self._resumable(run)
        d = tip.data
        ticket, req = d["ticket"], HumanRequest.from_dict(d["asking"])
        old = self._human_record(ticket)
        ans = None if old is not None else ch.poll(ticket)
        if old is None and ans is None:
            return ProcessResult("pending", None, [], {}, dict(d["resume"]["facts"]), dict(d["resume"]["provenance"]),
                                 rid, tip.id, spec["hash"], f"{req.node}: waiting for a person (ticket {ticket})",
                                 ticket, continues=d.get("continues"))
        r = d["resume"]
        with run_context(run=rid):
            res = self._answered(ans, req, ticket, tip.id, ch)
            w = {"node": r["node"], "facts": dict(r["facts"]), "prov": dict(r["provenance"]),
                 "taken": list(r["taken"]), "parent": res.stored_id, "input_hash": d["input_hash"],
                 "continues": tip.id}
            return self._go(spec, rid, w, first=(res, dict(r["state"])))

    def cancel(self, run, why):
        """Withdraw a pending run's question (a retracted fact, the case closed): the channel's cancel, a "human"
        record {"cancelled": true, "why"} and a "process" segment with status "cancelled" → its ProcessResult."""
        from ..core.store import FORMAT as RECORD_FORMAT
        from ..core.store import run_context
        rid, tip, ch, spec = self._resumable(run)
        d = tip.data
        ticket, req = d["ticket"], HumanRequest.from_dict(d["asking"])
        if self._human_record(ticket) is not None:
            raise ProcessRunError(f"ticket {ticket} is already answered: resume the run, a recorded answer stands")
        ch.cancel(ticket, why)
        with run_context(run=rid, parent=tip.id):
            hid = self._open()._append({"v": RECORD_FORMAT, "kind": "human", "process": self.name, "ticket": ticket,
                                        "node": req.node, "question": req.question, "reason": req.reason,
                                        "cancelled": True, "why": str(why), "channel": _channel(ch)})["id"]
        with run_context(run=rid, parent=hid):
            rec = self._record(spec, d["input_hash"], [], "cancelled", None, req.node, continues=tip.id)
        return self._finish(spec, ProcessResult("cancelled", None, [], {}, dict(d["resume"]["facts"]),
                                                dict(d["resume"]["provenance"]), rid, rec, spec["hash"],
                                                f"{req.node}: cancelled ({why})", ticket, continues=tip.id))

    # --- the path promise (stage 3: union and per_node; end_to_end is stage 4)

    def guarantee(self, spec=None):
        """What the declared promise says about this process now (stage 3) → {"mode", "risk", "in_force", "bound",
        "statement", "assumptions", "nodes", "problems", "warnings", "delta"}.

        union: P(a run ends automatically and its end answer is wrong) ≤ Σ_i r_i·m_i ≤ risk — a union bound over every
        automatic node of the graph (not over the path a run took), m_i the most runs node i can make in one process
        run (1, plus each loop's max_iter that can bring the run back to it). It is in force when every automatic node
        keeps a promise at its share (a CRC risk ≤ r_i; an LTT node's error level ≤ r_i, with the process promise then
        holding with probability ≥ 1 − Σ δ_i, Bonferroni, and a warning), Σ r_i·m_i ≤ risk, no system was loaded
        against another fingerprint, and the path watch (monitor()) has not flagged. per_node: every node keeps its
        own promise, and the end error is not bounded. nodes: {node: {"r_i", "visits", "promise", "ok", "why"}}."""
        spec = spec if spec is not None else self.spec()
        pr = spec.get("promise")
        out = {"mode": None if pr is None else pr["mode"], "risk": None if pr is None else pr.get("risk"),
               "in_force": False, "bound": None, "statement": None, "assumptions": [], "nodes": {}, "problems": [],
               "warnings": [], "delta": None}
        if pr is None:
            out["statement"] = "no process promise is declared (p.promise(mode=...))"
            out["problems"].append(out["statement"])
            return out
        if pr["mode"] == "end_to_end":
            raise NotImplementedError("the end_to_end promise (a threshold family calibrated on end-labelled runs) is "
                                      "stage 4 of the process; this stage computes union and per_node")
        auto = [n for n, s in spec["nodes"].items() if s["kind"] != "human"]
        visits = _visits(spec["nodes"], spec["edges"])
        for n in auto:
            npr = _node_promise(spec["nodes"][n], self.systems.get(n))
            out["nodes"][n] = {"r_i": None, "visits": visits[n], "promise": npr, "ok": npr is not None,
                               "why": None if npr is not None else "keeps no promise of its own"}
        for n, m in self.mismatched.items():
            out["problems"].append(f"{n}: loaded against another fingerprint than the recorded one (strict=False)")
        if pr["mode"] == "per_node":
            kept = [f"{n} ({_say_promise(v['promise'])})" for n, v in out["nodes"].items() if v["promise"] is not None]
            bare = [n for n, v in out["nodes"].items() if v["promise"] is None]
            out["statement"] = ("per node: each node keeps its own promise — " + (", ".join(kept) or "none does")
                                + (f"; {', '.join(bare)} keep none" if bare else "")
                                + ". The end error is not bounded: a node's promise is about its own answers, and an "
                                "error upstream reaches the end through nodes that answer right on what they are given")
            out["assumptions"] = ["A2: each node's calibration examples are exchangeable with what reaches it in the "
                                  "process"]
            out["in_force"] = not out["problems"]
            return out
        risk, per = pr["risk"], (pr.get("split") or {}).get("per_node") or {}
        deltas = []
        for n, v in out["nodes"].items():
            r_i = float(per.get(n, 0.0))
            v["r_i"] = r_i
            npr = v["promise"]
            if npr is None:
                out["problems"].append(f"{n}: keeps no promise of its own (a union bound needs one at r_i = {r_i:g})")
                continue
            method = npr.get("method")
            level = npr.get("risk") if method == "crc" else npr.get("error")
            if level is None:
                v["ok"], v["why"] = False, f"its promise ({method}) has no risk or error level"
            elif method != "crc":
                deltas.append(float(npr.get("delta") or 0.0))
                out["warnings"].append(f"{n}: an {method} promise (error ≤ {level:g} among its own answers, with "
                                       f"probability ≥ {1 - float(npr.get('delta') or 0.0):g}); it is counted at its "
                                       "error level and the union holds with probability ≥ 1 − Σ δ_i (Bonferroni): "
                                       "honest, and weak — calibrate it with max_risk= (CRC) instead")
                if float(level) > r_i * (1 + 1e-9):
                    v["ok"], v["why"] = False, f"its error level {level:g} is over its share r_i = {r_i:g}"
            elif float(level) > r_i * (1 + 1e-9):
                v["ok"], v["why"] = False, (f"calibrated at risk {level:g}, over its share r_i = {r_i:g}: rebuild it "
                                            f"with max_risk={r_i:g}")
            if not v["ok"]:
                out["problems"].append(f"{n}: {v['why']}")
        bound = sum(v["r_i"] * v["visits"] for v in out["nodes"].values())
        out["bound"] = bound
        loops = {n: v["visits"] for n, v in out["nodes"].items() if v["visits"] > 1}
        if bound > risk * (1 + 1e-9):
            out["problems"].append(f"Σ r_i·m_i = {bound:.6g} is over the process risk {risk:g}: a node a loop can run "
                                   f"m_i times counts m_i times ({', '.join(f'{n} ×{m}' for n, m in loops.items())}); "
                                   "give a split that leaves room for it")
        if self.path_watch is not None and self.path_watch.flagged:
            out["problems"].append(f"the mix of paths drifted ({'; '.join(self.path_watch.why)}): the stream that "
                                   "reaches the nodes is not the one they were calibrated on (A2)")
        if deltas:
            out["delta"] = sum(deltas)
        kind = (pr.get("split") or {}).get("kind")
        shares = ", ".join(f"{n} {v['r_i']:.4g}" + (f" ×{v['visits']}" if v["visits"] > 1 else "")
                           for n, v in out["nodes"].items())
        out["statement"] = (f"P(a run ends automatically and its end answer is wrong) ≤ {risk:g}"
                            + (f" with probability ≥ {1 - out['delta']:g}" if deltas else "")
                            + f": a union bound over the {len(auto)} automatic nodes of the graph, not over the path "
                            f"a run took; each node keeps its share r_i ({kind} split: {shares}). A run ends "
                            "automatically when an end node answered and no person answered on its path")
        out["assumptions"] = [
            "A1: none on the errors — no independence; correlated errors only make the bound looser",
            "A2: each node's calibration examples are exchangeable with what reaches it in the process — calibrate a "
            "node in place, on what upstream gives it; a node calibrated on clean inputs, a node changed after its "
            "calibration, or a stream that drifted breaks the bound",
        ]
        out["in_force"] = not out["problems"]
        return out

    def monitor(self, watch, reference=None):
        """Watch the mix of paths: watch is a PathWatch; reference: runs (ProcessResults, or lists of the edges each
        took) from the stream the nodes were calibrated on, to set its reference (else set it yourself). Every run
        that ends is then observed (result.path_drift); a flag puts the union promise out of force until
        watch.reset(). → self."""
        if not isinstance(watch, PathWatch):
            raise TypeError("monitor(watch): a PathWatch")
        if reference is not None:
            watch.set_reference([_edge_list(self.edges, r) for r in reference], names=_edge_names(self.edges))
        self.path_watch = watch
        return self

    def records(self):
        """The "process" records of this process's store → [Stored] (stage 1: the index of each run)."""
        if self.storage is None:
            return []
        from ..core.store import open_storage
        if self._store is None:
            self._store = open_storage(self.storage)
        return [s for s in self._store.iter("process") if s.data.get("process") == self.name]

    def explain(self, result):
        """A run in words: each node, who answered and how, the edge taken; then how the run ended."""
        lines = [f"process {self.name}, run {result.run}" + (f" (record {result.stored_id})" if result.stored_id else "")]
        for i, st in enumerate(result.path, 1):
            r = result.nodes.get(st["node"]) if i == _last_index(result.path, st["node"]) else None
            ans = f" answered {json.dumps(_plain_or_repr(r.answer), ensure_ascii=False)}" if r is not None else ""
            conf = f" (confidence {st_conf:.3g})" if r is not None and (st_conf := r.confidence) is not None else ""
            e = st["edge"]
            nxt = (f"; → {e['to']} because {describe(e['when'])}" + (" (loop)" if e["loop"] else "")) if e else "; stop"
            where = f" [{st['stored_id']}]" if st["stored_id"] else ""
            lines.append(f"  {i}. {st['node']} ({st['kind']}): by {st['by']}, {st['status']}{ans}{conf}{where}{nxt}")
        if result.status == "decided":
            lines.append(f"  decided: {json.dumps(_plain_or_repr(result.answer), ensure_ascii=False)}"
                         + ("" if result.automatic is None else
                            " (automatically)" if result.automatic else " (with a person on the path)"))
        else:
            lines.append(f"  {result.status}: {result.why}")
        if self._promise is not None and self._promise["mode"] != "end_to_end":
            g = self.guarantee()
            lines.append(f"  promise ({g['mode']}): {g['statement']}"
                         + ("" if g["mode"] == "per_node" else
                            "; in force" if g["in_force"] else f"; NOT in force: {'; '.join(g['problems'])}"))
        if result.path_drift is not None and result.path_drift.get("drift"):
            lines.append(f"  paths: the mix drifted ({'; '.join(result.path_drift['why'])})")
        return "\n".join(lines)

    # --- replay (stage 2)

    def replay(self, run, trust_models=False):
        """Re-check one run (its id or its ProcessResult) from the records alone, across all its segments → {"ok",
        "run", "status", "segments", "steps", "at", "kind", "reason", "unchecked"} (+ "pending": the ticket of a run
        that waits for a person — ok True, replayed up to the question; + "cancelled": the withdrawn ticket). Each step
        of the recorded path, in order: (1) "record": the node's record is in its store, intact (its hash and id), of
        this run, after the decision before it, and says what the path says (who, status, answers, knowledge, cost);
        (2) "trace_hash": the hash at the end of its trace is the recorded one; (3) "replay": the node's decision
        re-checked by its own means (Dispatcher.replay, Trace.replay, Guard.replay; trust_models as there);
        (4) "facts": the facts it was given are the ones the recorded answers upstream and the input give, in the path
        and in its own record; (5) "edge": the first edge that holds on the recorded answers is the recorded one;
        (6) "promise": the end_to_end decision rule (stage 4: not checked yet). A person's answer is a leaf, recorded
        and not recomputed ("human"): its "human" record is intact, in this run, after the pending segment that asked
        (or the decision before it), the ticket's only answer, through the same channel, to the request asked; its
        value is what the next facts carry. The segments are checked as a chain ("segment": `continues`, the same
        spec and input, the pending segment's question and the facts it kept). "at": the first divergence {"node",
        "step", "segment", "position"} (None when ok); "kind": "integrity" (the records disagree: data changed after
        the run), "recompute" (a decision does not recompute: its system or the spec changed) or "missing_part" (a
        node without its system or store). "steps": per checked step {"segment", "node", "kind", "stored_id",
        "checked"} (+ "human": "recorded, not recomputed"). "unchecked": what the records do not let it check (a
        guard's tool result is kept as a hash; an input fact no node read)."""
        if self.storage is None:
            raise ProcessRunError("a process replays from its records: Process(storage=...)")
        rid = run.run if isinstance(run, ProcessResult) else run
        segs = [s for s in self._open().iter("process") if s.data.get("process") == self.name and s.run == rid]
        if not segs:
            raise ProcessRunError(f"no record of run {rid!r} in this process's store")
        return _Replay(self, rid, trust_models).walk(segs)

    def replay_all(self, trust_models=False):
        """Replay every run in this process's store and verify() every store it reads → {"ok", "runs", "failed":
        [the reports of the runs that do not replay], "pending": [the run ids that wait for a person], "verify":
        {store: {"ok", "problems"}}} — ok when every run replays and every chain verifies (publish the process store's
        head() elsewhere to catch a rewrite of the whole store: verify(anchor=))."""
        if self.storage is None:
            raise ProcessRunError("a process replays from its records: Process(storage=...)")
        runs = list(dict.fromkeys(s.run for s in self.records()))
        reports = [self.replay(r, trust_models=trust_models) for r in runs]
        stores, seen = {"process": self._open()}, {id(self._open())}
        for n in self.nodes:
            st = getattr(self.systems.get(n), "storage", None)
            if st is not None and id(st) not in seen:
                seen.add(id(st))
                stores[n] = st
        verify = {}
        for n, st in stores.items():
            v = st.verify()
            verify[n] = {"ok": v["ok"], "problems": v["problems"]}
        failed = [r for r in reports if not r["ok"]]
        return {"ok": not failed and all(v["ok"] for v in verify.values()), "runs": len(runs), "failed": failed,
                "pending": [r["run"] for r in reports if r.get("pending")], "verify": verify}

    # --- reading it

    def diff(self, other):
        """The changes from this process to `other` (a Process or a spec): see `diff`."""
        return diff(self.spec(), other.spec() if isinstance(other, Process) else other)

    def render(self, fmt="mermaid"):
        """The process drawn: "mermaid" (a flowchart) or "text"."""
        return render(self.spec(), fmt)

    def __repr__(self):
        return f"Process({self.name!r}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


# --- replay (stage 2): a run re-checked from its records, segment by segment, node by node

REPLAY_STEPS = ("record", "trace_hash", "replay", "facts", "edge", "promise", "human", "segment", "outcome", "spec",
                "index")
_RECORD_FIELDS = ("node", "kind", "store", "stored_id", "by", "status", "answers_hash", "knowledge", "cost")
_UNKNOWN = object()


class _Diverged(Exception):
    def __init__(self, node, step, kind, reason):
        super().__init__(reason)
        self.node, self.step, self.kind, self.reason = node, step, kind, reason


def _refs(p, out=None):
    """The facts a predicate reads (its "answer" leaves)."""
    out = set() if out is None else out
    if isinstance(p, dict):
        if "answer" in p:
            out.add(p["answer"])
        for k in ("and", "or"):
            for q in p.get(k) or []:
                _refs(q, out)
        if "not" in p:
            _refs(p["not"], out)
    return out


def _cj(v):
    """A value as canonical JSON, to compare a recomputed step with the recorded one."""
    return canonical(_plain_or_repr(v))


def _intact(store, sid):
    """The record with this id, checked against its own hash and id → (record, None) or (None, why)."""
    from ..core.store import chained, record_hash
    if not isinstance(sid, str) or not sid:
        return None, "the path names no record for this step"
    d = store._find(sid)
    if d is None:
        return None, f"no record {sid} in its store (deleted, or the path was changed)"
    if not chained(d) or d.get("hash") != record_hash(d):
        return None, f"record {sid} was changed after it was stored (its hash does not match)"
    if d.get("id") != d["hash"][:16]:
        return None, f"record {sid}: its id does not match its hash"
    if d.get("redacted"):
        return None, f"record {sid} was redacted: its content is gone"
    return d, None


def _why(rep):
    ms = rep.get("mismatches") or []
    return "; ".join(f"{m[-2] if len(m) > 2 else m[0]}: {m[-1]}" for m in ms[:3]) or "it does not replay"


class _Replay:
    """One run replayed from its records (Process.replay): the walk of Process._go over the recorded path, each node's
    result loaded from its store instead of asked, each recomputed step compared with the recorded one."""

    def __init__(self, p, rid, trust):
        self.p, self.rid, self.trust = p, rid, trust
        self.spec = p.spec()
        self.pst = p._open()
        self.facts, self.prov, self.inputs, self.absent = {}, {}, {}, set()
        self.taken = [0] * len(p.edges)
        self.parent = _UNKNOWN
        self.input_hash = None
        self.steps, self.unchecked = [], []
        self.seg = self.pos = 0
        self.segs, self._ahead = [], None

    def fail(self, node, step, kind, reason):
        raise _Diverged(node, step, kind, reason)

    # the run

    def walk(self, segs):
        out = {"ok": True, "run": self.rid, "status": segs[-1].data.get("outcome", {}).get("status"),
               "segments": [s.id for s in segs], "steps": self.steps, "at": None, "kind": None, "reason": None,
               "unchecked": self.unchecked}
        index_bad = None
        self.segs = segs
        for s in segs:
            d, why = _intact(self.pst, s.id)
            if d is None and index_bad is None:
                index_bad = why
        try:
            first, prev = None, None
            for k, s in enumerate(segs):
                self.seg, self.pos = k, 0
                d = s.data
                self.check_segment(k, s, prev, first)
                if d.get("outcome", {}).get("status") == "cancelled":
                    out["cancelled"] = self.cancelled(s, prev, first, k == len(segs) - 1)
                    break
                res = self.segment(s, first)
                first = None
                if res[0] == "end":
                    self.ended(s, res, k == len(segs) - 1)
                else:
                    first = self.asked(s, res)
                    if k == len(segs) - 1:
                        out["pending"] = d.get("ticket")
                prev = s
            if index_bad is not None:
                self.fail(None, "index", "integrity", f"a process record of this run: {index_bad}")
        except _Diverged as e:
            out.update(ok=False, kind=e.kind, reason=e.reason,
                       at={"node": e.node, "step": e.step, "segment": self.seg, "position": self.pos})
        return out

    def check_segment(self, k, s, prev, first):
        d, sp = s.data, self.spec
        node = first["node"] if first else sp["start"]
        if d.get("spec_hash") != sp["hash"]:
            self.fail(None, "spec", "recompute", f"segment {s.id} was run by process spec {str(d.get('spec_hash'))[:12]}"
                                                 f", this process is {sp['hash'][:12]}")
        pr = sp.get("promise")
        if d.get("mode") != (pr["mode"] if pr else None) or _cj(d.get("promise")) != _cj(pr):
            self.fail(None, "spec", "integrity", f"segment {s.id} records another promise than the spec's")
        if k == 0:
            if d.get("continues") is not None:
                self.fail(node, "segment", "integrity", "the first segment of the run continues another")
            self.input_hash = d.get("input_hash")
            return
        if prev is None or prev.data.get("outcome", {}).get("status") != "pending":
            self.fail(node, "segment", "integrity", "a segment after one that did not stop for a person")
        if d.get("continues") != prev.id:
            self.fail(node, "segment", "integrity", f"segment {s.id} continues {d.get('continues')}, not the pending "
                                                    f"segment {prev.id}")
        if d.get("input_hash") != self.input_hash:
            self.fail(node, "segment", "integrity", f"segment {s.id} records another input than the run's")

    def segment(self, s, first):
        """Walk one segment's path as Process._go walks it → ("end", status, answer, node) or ("ask", node, reason,
        state, known, nodes so far)."""
        path = s.data.get("path") or []
        node = first["node"] if first else self.spec["start"]
        i, seen = 0, []
        while True:
            ns = self.p.nodes.get(node)
            if ns is None:
                self.fail(node, "record", "missing_part", f"the run reaches node {node!r}, not in this process")
            self.pos = i
            if first is not None:
                state, known = first["state"], first["known"]
                res = self.human(path, i, node, ns, first["reason"], first["pending"], state, known, seen)
                kind, first = "human", None
                init = rep = None
            elif ns["kind"] == "human":
                state, known = self.given(ns)
                if i >= len(path):
                    return ("ask", node, "node", state, known, seen)
                res = self.human(path, i, node, ns, "node", None, state, known, seen)
                kind, init, rep = "human", None, None
            else:
                if i >= len(path):
                    self.fail(node, "record", "integrity", "the recorded path stops before this node's step")
                res, init, rep = self.load(path[i], node, ns)
                self.learn(ns, init)
                state, known = self.given(ns)
                kind = ns["kind"]
            edge = self.check(path[i], node, ns, res, state, known, kind, init, rep)
            seen.append(node)
            i += 1
            if res.stored_id is not None:
                self.parent = res.stored_id
            if edge is None and res.status in ("human", "abstain") and kind != "human":   # as run() does
                on = self.p.on_human if res.status == "human" else self.p.on_abstain
                if on == "handoff":
                    self.pos = i
                    if i >= len(path):
                        return ("ask", node, res.status, state, known, seen)
                    res = self.human(path, i, node, ns, res.status, None, state, known, seen)
                    edge = self.check(path[i], node, ns, res, state, known, "human", None, None)
                    seen.append(node)
                    i += 1
                    if res.stored_id is not None:
                        self.parent = res.stored_id
            if edge is not None:
                node = edge["to"]
                continue
            if res.by == "human" and res.status == "abstain":
                status, answer = "abstain", None
            elif res.status in ("human", "abstain"):
                status, answer = "abstain", None
            elif ns.get("end"):
                status, answer = "decided", res.answer
            else:
                status, answer = "abstain", None
            if i != len(path):
                self.fail(node, "edge", "integrity", f"the run ends at {node}, the recorded path goes on to "
                                                     f"{path[i].get('node')!r}")
            return ("end", status, answer, node)

    def ended(self, s, res, last):
        _, status, answer, node = res
        oc = s.data.get("outcome") or {}
        if oc.get("status") != status or oc.get("at") != node or oc.get("answer_hash") != _vh(answer):
            self.fail(node, "outcome", "integrity", f"the run ends {status} at {node}, the record says "
                                                    f"{oc.get('status')} at {oc.get('at')} (or another answer)")
        if self.parent is not _UNKNOWN and s.parent != self.parent:
            self.fail(node, "segment", "integrity", "the process record does not follow from the run's last decision")
        if not last:
            self.fail(node, "segment", "integrity", "a segment after the run ended")

    def asked(self, s, res):
        """A segment that stops for a person: its record keeps the question and what the run goes on with."""
        _, node, reason, state, known, seen = res
        d, ns = s.data, self.p.nodes[node]
        r = d.get("resume") or {}
        if not known:                                 # the input facts no node read so far: the facts the run kept
            for k, v in (r.get("facts") or {}).items():
                if k not in self.facts:
                    self.facts[k] = self.inputs[k] = v
            state, known = self.given(ns)
        oc = d.get("outcome") or {}
        if oc.get("status") != "pending" or oc.get("at") != node:
            self.fail(node, "outcome", "integrity", f"the run asks a person at {node} here, the record says "
                                                    f"{oc.get('status')} at {oc.get('at')}")
        q = ns.get("question") or node
        a = d.get("asking") or {}
        if (a.get("process"), a.get("run"), a.get("node"), a.get("question"), a.get("reason"), a.get("path")) != \
                (self.p.name, self.rid, node, q, reason, list(seen)) or (r.get("node"), r.get("reason")) != (node, reason):
            self.fail(node, "human", "integrity", "the recorded question is not the one the run asks here")
        if not d.get("ticket"):
            self.fail(node, "human", "integrity", "a pending segment without its ticket")
        if known:
            plain_state = {k: v for k, v in state.items() if k != "provenance"}
            if _vh(a.get("facts")) != _vh(_plain(plain_state)) or _vh(r.get("state")) != _vh(_plain(state)) or \
                    _vh(a.get("provenance")) != _vh(_plain({k: self.prov[k] for k in state if k in self.prov})):
                self.fail(node, "facts", "integrity", "the person is asked on other facts than the recorded path gives")
        else:
            self.unchecked.append(f"{node}: the facts of the question (an input no node read, or a value not kept)")
        rf = r.get("facts") or {}
        for k, v in self.facts.items():
            if isinstance(v, _NotKept):
                continue
            if k not in rf or _vh(rf[k]) != _vh(_plain_or_repr(v)):
                self.fail(node, "facts", "integrity", f"the fact {k!r} kept for the rest of the run is not the one "
                                                      "the recorded path gives")
        if _vh(r.get("provenance")) != _vh(_plain(self.prov)):
            self.fail(node, "facts", "integrity", "the labels kept for the rest of the run are not the recorded ones")
        if list(r.get("taken") or []) != self.taken:
            self.fail(node, "segment", "integrity", "the loop counts kept for the rest of the run are not the path's")
        if self.parent is not _UNKNOWN and s.parent != self.parent:
            self.fail(node, "segment", "integrity", "the pending segment does not follow from the decision before it")
        return {"node": node, "reason": reason, "pending": s, "state": dict(r.get("state") or {}),
                "known": known}

    def cancelled(self, s, prev, first, last):
        node = first["node"] if first else None
        d, t = s.data, prev.data.get("ticket")
        hum = [x for x in self.pst.iter("human") if x.data.get("ticket") == t and x.data.get("process") == self.p.name]
        if len(hum) != 1 or not hum[0].data.get("cancelled"):
            self.fail(node, "human", "integrity", f"ticket {t}: a cancelled run with {len(hum)} records for it")
        h, why = _intact(self.pst, hum[0].id)
        if h is None:
            self.fail(node, "human", "integrity", why)
        if (h.get("run") or {}).get("parent") != prev.id or h.get("node") != node:
            self.fail(node, "human", "integrity", "the withdrawal does not follow from the pending segment")
        oc = d.get("outcome") or {}
        if d.get("path") or oc.get("status") != "cancelled" or oc.get("at") != node or s.parent != h["id"]:
            self.fail(node, "segment", "integrity", "the cancelled segment does not record the withdrawal")
        if not last:
            self.fail(node, "segment", "integrity", "a segment after the run was cancelled")
        self.steps.append({"segment": self.seg, "node": node, "kind": "human", "stored_id": h["id"],
                           "checked": ["human"], "human": "cancelled, recorded"})
        return t

    # the facts

    def learn(self, ns, init):
        """The input facts a node read that no node read before: what its own record says it was given."""
        for r in ns["reads"]:
            if r == "provenance" or r in self.facts:
                continue
            if init is None:
                continue
            if r in init:
                self.facts[r] = self.inputs[r] = init[r]
            else:
                self.absent.add(r)

    def complete(self):
        return self.input_hash is not None and _vh(self.inputs) == self.input_hash

    def lookahead(self):
        """The input facts no node has read yet, from the records further on: taken only when, with them, the input is
        exactly the recorded one (its hash) — so a node that reads an input before any decision does is checked too."""
        if self._ahead is not None or self.input_hash is None:
            return
        self._ahead = {}
        written = {f for ns in self.p.nodes.values() for f in ns["writes"]}
        for s in self.segs:
            for k, v in ((s.data.get("resume") or {}).get("facts") or {}).items():
                self._ahead.setdefault(k, v)
            for e in s.data.get("path") or []:
                init = self._peek(e)
                for k, v in (init or {}).items():
                    self._ahead.setdefault(k, v)
        cand = dict(self.inputs)
        for k, v in self._ahead.items():
            if k not in cand and k not in written and k != "knowledge":
                cand[k] = v
        if _vh(cand) == self.input_hash:
            for k, v in cand.items():
                if k not in self.facts:
                    self.facts[k] = v
            self.inputs = cand

    def _peek(self, e):
        """The facts a recorded node decision says it was given, or None (no checks: the walk checks it in its turn)."""
        ns, sysm = self.p.nodes.get(e.get("node")), self.p.systems.get(e.get("node"))
        if ns is None or sysm is None or e.get("kind") not in ("decision", "system") or isinstance(sysm, NodeRunner):
            return None
        try:
            store = sysm.storage
            if e["kind"] == "decision":
                from ..core.dispatch import Dispatched
                disp = sysm.dispatcher
                return Dispatched.from_dict(store.record(e["stored_id"]), disp.system).s1.trace.init
            return store.rederive(e["stored_id"], sysm, trust_models=True).trace.init
        except Exception:  # noqa: BLE001 — a record that does not load gives nothing here
            return None

    def given(self, ns):
        if not self.complete() and any(r not in self.facts and r != "provenance" for r in ns["reads"] or [None]):
            self.lookahead()
        state = self.p._given(ns, self.facts, self.prov)
        if not ns["reads"] and ns["kind"] == "human":
            known = self.complete()
        else:
            known = all(r == "provenance" or r in self.facts or r in self.absent for r in ns["reads"]) or \
                self.complete()
        known = known and not any(isinstance(v, _NotKept) for v in state.values())
        return state, known

    # a node's step

    def check_run(self, node, rec, step):
        r = rec.get("run") if isinstance(rec.get("run"), dict) else {}
        if r.get("id") != self.rid:
            self.fail(node, step, "integrity", f"record {rec.get('id')} belongs to run {r.get('id')!r}, not to this one")
        if self.parent is not _UNKNOWN and r.get("parent") != self.parent:
            self.fail(node, step, "integrity", f"record {rec.get('id')} follows from {r.get('parent')}, not from the "
                                               f"decision before it ({self.parent})")

    def load(self, entry, node, ns):
        """A node's decision loaded from its store → (NodeResult, the facts its record says it was given or None,
        a function that replays it by its own means)."""
        if entry.get("node") != node or entry.get("kind") != ns["kind"]:
            self.fail(node, "record", "integrity", f"the recorded path has {entry.get('node')!r} "
                                                   f"({entry.get('kind')}) where the run reaches {node!r}")
        sysm = self.p.systems.get(node)
        if sysm is None:
            self.fail(node, "record", "missing_part", "no system is bound to this node (Process.from_spec(systems=))")
        if isinstance(sysm, NodeRunner):
            self.fail(node, "replay", "missing_part", "a node runner of one's own: the replay does not know its records")
        store = getattr(sysm, "storage", None)
        if store is None:
            self.fail(node, "record", "missing_part", "the node's system keeps no store: nothing to replay")
        sid = entry.get("stored_id")
        rec, why = _intact(store, sid)
        if rec is None:
            self.fail(node, "record", "integrity", why)
        self.check_run(node, rec, "record")
        from ..core.runtime import trace_hash
        try:
            if ns["kind"] == "decision":
                from ..core.dispatch import Dispatched
                if rec.get("kind") != "dispatch":
                    self.fail(node, "record", "integrity", f"record {sid} is not a decision of a dispatcher")
                disp = sysm.dispatcher
                dd = Dispatched.from_dict(rec, disp.system, disp.slow.system if disp.slow is not None else None)
                dd.stored_id = sid
                res, init = _decision_result(dd, store), (dd.s1.trace.init if dd.s1 is not None else None)

                def rep():
                    return disp.replay(dd, trust_models=self.trust)
            elif ns["kind"] == "system":
                if rec.get("kind", "ask") != "ask":
                    self.fail(node, "record", "integrity", f"record {sid} is not a decision of a System")
                resp = store.rederive(sid, sysm, trust_models=self.trust)
                resp.stored_id = sid
                res, init = _system_result(resp, ns, store), resp.trace.init

                def rep():
                    return resp.trace.replay(sysm, resp.flow, trust_models=self.trust)
            else:
                meta = (rec.get("meta") or {}).get("guard") or {}
                if rec.get("kind", "ask") != "ask" or not meta:
                    self.fail(node, "record", "integrity", f"record {sid} is not a decision of a guard")
                tool = meta.get("tool")
                resp = store.get(sid, sysm.system(tool) if tool is not None else sysm._unknown_system())
                v = resp.results.get("verdict")
                out = meta.get("outcome")
                result = _NotKept(meta.get("result_hash")) if meta.get("executed") and not meta.get("error") else None
                res = NodeResult(out, {"verdict": out, "result": result}, "s1", "human" if out == "escalate" else "ok",
                                 v.confidence if v is not None else None, sid, trace_hash(resp), None, False, False,
                                 None, store)
                init = None
                if result is not None:
                    self.unchecked.append(f"{node}: the tool's result (its store keeps its hash only)")

                def rep():
                    return sysm.replay(sid)
        except _Diverged:
            raise
        except Exception as e:  # noqa: BLE001 — a record that does not load is damaged data
            self.fail(node, "record", "integrity", f"record {sid} does not load: {type(e).__name__}: {e}")
        return res, init, rep

    def human(self, path, i, node, ns, reason, pending, state, known, seen):
        """A person's answer: a leaf, recorded and not recomputed — its "human" record checked against the run."""
        if i >= len(path):
            self.fail(node, "human", "integrity", "the recorded path stops before the person's answer")
        entry = path[i]
        if entry.get("node") != node or entry.get("kind") != "human":
            self.fail(node, "record", "integrity", f"the recorded path has {entry.get('node')!r} "
                                                   f"({entry.get('kind')}) where the run has a person's answer")
        sid = entry.get("stored_id")
        rec, why = _intact(self.pst, sid)
        if rec is None:
            self.fail(node, "record", "integrity", why)
        if rec.get("kind") != "human" or rec.get("process") != self.p.name or rec.get("node") != node or \
                rec.get("cancelled"):
            self.fail(node, "record", "integrity", f"record {sid} is not a person's answer at this node")
        if pending is not None:
            self.parent = pending.id                  # the answer follows from the pending segment
        self.check_run(node, rec, "human")
        t, q = rec.get("ticket"), ns.get("question") or node
        n = sum(1 for x in self.pst.iter("human") if x.data.get("ticket") == t and x.data.get("process") == self.p.name)
        if n != 1:
            self.fail(node, "human", "integrity", f"ticket {t} has {n} records: one ticket, one answer")
        if rec.get("question") != q or rec.get("reason") != reason:
            self.fail(node, "human", "integrity", f"the answer is to {rec.get('question')!r} ({rec.get('reason')}), "
                                                  f"the run asks {q!r} ({reason})")
        if not isinstance(rec.get("reviewer"), str) or not rec.get("reviewer"):
            self.fail(node, "human", "integrity", "the answer names no reviewer")
        if pending is not None:
            pd = pending.data
            if t != pd.get("ticket"):
                self.fail(node, "human", "integrity", f"the answer is to ticket {t}, the run asked {pd.get('ticket')}")
            if _cj(rec.get("channel")) != _cj(pd.get("channel")):
                self.fail(node, "human", "integrity", "the answer came through another channel than the question")
            if rec.get("request_hash") != _vh(pd.get("asking")):
                self.fail(node, "human", "integrity", "the answer is to another request than the one asked")
        else:
            if self.p.human is not None and _cj(rec.get("channel")) != _cj(_channel(self.p.human)):
                self.fail(node, "human", "integrity", "the answer came through another channel than the process's")
            if known:
                req = HumanRequest(process=self.p.name, run=self.rid, node=node, question=q, reason=reason,
                                   facts=_plain({k: v for k, v in state.items() if k != "provenance"}),
                                   provenance=_plain({k: self.prov[k] for k in state if k in self.prov}),
                                   path=list(seen), answer=_answer_type(self.p.systems.get(node), q),
                                   asked_at=rec.get("asked_at"))
                if rec.get("request_hash") != _vh(req.to_dict()):
                    self.fail(node, "facts", "integrity", "the person was asked on other facts than the recorded path "
                                                          "gives")
        ans = HumanAnswer(rec.get("value"), rec.get("reviewer"), rec.get("note"), rec.get("answered_at"),
                          declined=bool(rec.get("declined")), ticket=t)
        if ans.declined:
            return NodeResult(None, {}, "human", "abstain", None, sid, None, raw=ans, store=self.pst)
        return NodeResult(ans.value, {q: ans.value}, "human", "ok", None, sid, None, raw=ans, store=self.pst)

    def check(self, entry, node, ns, res, state, known, kind, init, rep):
        """Steps 1–5 of one recorded step: the recomputed step (Process._step) against the recorded one."""
        before = list(self.taken)
        st, edge = self.p._step(node, ns, res, state, kind, self.facts, self.prov, self.taken)
        checked = ["record"]
        for f in _RECORD_FIELDS:
            if f == "answers_hash" and st[f] is None:
                continue
            if _cj(st.get(f)) != _cj(entry.get(f)):
                self.fail(node, "record", "integrity", f"the path records {f} {entry.get(f)!r}, the node's record "
                                                       f"gives {st.get(f)!r}")
        if kind != "human":
            if st["trace_hash"] != entry.get("trace_hash"):
                self.fail(node, "trace_hash", "integrity", "the hash at the end of the node's trace is not the "
                                                           "recorded one")
            r = rep()
            if not r["ok"]:
                self.fail(node, "replay", "recompute", f"the node's decision does not replay: {_why(r)}")
            checked += ["trace_hash", "replay"]
        if known:
            if st["facts_in_hash"] != entry.get("facts_in_hash"):
                self.fail(node, "facts", "integrity", "the facts the node was given are not the ones the recorded "
                                                      "answers upstream and the input give")
            for k, v in (state.items() if init is not None else ()):
                if k not in init or _vh(init[k]) != _vh(v):
                    self.fail(node, "facts", "integrity", f"the node's record was given {k!r} other than the recorded "
                                                          "path passes it")
        else:
            self.unchecked.append(f"{node}: the facts it was given (an input no node read, or a value not kept)")
        if st["facts_out_hash"] is not None and st["facts_out_hash"] != entry.get("facts_out_hash"):
            self.fail(node, "facts", "integrity", "the facts the node wrote are not its recorded answers")
        checked.append("facts")
        refs = set()
        for e in self.p.edges:
            if e["from"] == node:
                _refs(e.get("when"), refs)
        blind = [f for f in refs if isinstance(self.facts.get(f), _NotKept) or
                 (f not in self.facts and f not in self.absent and not self.complete())]
        if blind:                                     # the predicate reads what the records do not keep
            self.taken[:] = before
            rec_e = entry.get("edge")
            edge = None
            if rec_e is not None:
                for j, e in enumerate(self.p.edges):
                    if e["from"] == node and e["to"] == rec_e.get("to") and _cj(e.get("when")) == _cj(rec_e.get("when")) \
                            and not (e.get("loop") and before[j] >= e["max_iter"]):
                        edge, self.taken[j] = e, before[j] + 1
                        break
                if edge is None:
                    self.fail(node, "edge", "integrity", f"the recorded edge to {rec_e.get('to')!r} is not an edge "
                                                         "of the process here")
            self.unchecked.append(f"{node}: the edge's predicate (it reads {sorted(blind)})")
        elif _cj(st["edge"]) != _cj(entry.get("edge")):
            got = st["edge"]["to"] if st["edge"] else None
            rec_to = (entry.get("edge") or {}).get("to")
            self.fail(node, "edge", "integrity", f"the recorded answers lead to {got!r}, the path records {rec_to!r}")
        checked.append("edge")
        step = {"segment": self.seg, "node": node, "kind": kind, "stored_id": res.stored_id, "checked": checked}
        if kind == "human":
            if (st.get("ticket"), st.get("handoff")) != (entry.get("ticket"), entry.get("handoff")):
                self.fail(node, "human", "integrity", "the path records another ticket or hand-off for this answer")
            step["checked"].append("human")
            step["human"] = "recorded, not recomputed"
        self.steps.append(step)
        return edge


def _last_index(path, node):
    return max(i for i, st in enumerate(path, 1) if st["node"] == node)


def _plain_or_repr(v):
    try:
        return _plain(v)
    except (TypeError, ValueError):
        return repr(v)


def _vh(v):
    """vhash of a value, or of its repr when it is not data (a tool's return value)."""
    from ..core.runtime import vhash
    try:
        return vhash(v)
    except Exception:  # noqa: BLE001 — any value a tool may return
        return vhash(repr(v))


class _NotKept:
    """A value its node's store does not keep (a guard's tool result: the store keeps its hash only)."""

    def __init__(self, h=None):
        self.hash = h

    def __repr__(self):
        return "<not kept>"


def _vh_kept(v):
    """_vh, or None when the value holds something its store does not keep (a replay then cannot check that hash)."""
    if isinstance(v, _NotKept) or (isinstance(v, dict) and any(isinstance(x, _NotKept) for x in v.values())):
        return None
    return _vh(v)


def _short(fp):
    return hashlib.sha256(canonical(fp).encode()).hexdigest()[:12] if fp is not None else "none"


def _family(f):
    if isinstance(f, str):
        if f not in FAMILIES:
            raise ProcessSpecError(f"promise family is one of {', '.join(FAMILIES)} or a ThresholdFamily, not {f!r}")
        return {"kind": f, "order": None, "fingerprint": None}
    if isinstance(f, dict):
        if f.get("kind") not in FAMILIES + ("custom",) or set(f) - {"kind", "order", "fingerprint", "name"}:
            raise ProcessSpecError(f"promise family: {{'kind': shared_rank | per_node_fixed, 'order': [...]}}, not {f!r}")
        out = {"kind": f["kind"], "order": f.get("order"), "fingerprint": f.get("fingerprint")}
        if "name" in f:
            out["name"] = f["name"]
        if out["kind"] == "shared_rank" and out["order"] is not None:
            raise ProcessSpecError("promise family shared_rank has no order")
        return _plain(out)
    fp = getattr(f, "fingerprint", None)
    if not callable(fp):
        raise ProcessSpecError(f"a custom promise family needs fingerprint(): {type(f).__name__}")
    return {"kind": "custom", "name": type(f).__name__, "order": None, "fingerprint": str(fp())}


def _load_promise(pr):
    """A recorded promise → the declared values (checked like promise()); a recorded split is kept as recorded."""
    keys = {"mode", "risk", "split", "family", "delta", "thresholds", "calibrated_on"}
    if not isinstance(pr, dict) or set(pr) - keys or "mode" not in pr:
        raise ProcessSpecError(f"promise: a dict of {sorted(keys)}, got {pr!r}")
    probe = Process("_")
    sp = pr.get("split")
    if pr["mode"] == "union":
        if not isinstance(sp, dict) or sp.get("kind") not in ("equal", "given", "callable"):
            raise ProcessSpecError(f"promise split: kind equal / given / callable, not {sp!r}")
        probe.promise("union", pr.get("risk"), split="equal")
    elif pr["mode"] == "end_to_end":
        probe.promise("end_to_end", pr.get("risk"), delta=pr.get("delta"), family=pr.get("family"))
    else:
        probe.promise(pr["mode"])
    out = dict(probe._promise)
    out["split"] = _plain(sp) if pr["mode"] == "union" and sp.get("kind") != "equal" else out["split"]
    out["thresholds"], out["calibrated_on"] = pr.get("thresholds"), pr.get("calibrated_on")
    return _plain(out)


def _check_split(per, risk, auto):
    if not isinstance(per, dict) or not per:
        raise ProcessSpecError(f"promise split: {{node: r_i}}, not {per!r}")
    bad = sorted(set(per) - set(auto))
    if bad:
        raise ProcessSpecError(f"promise split names nodes that are not automatic nodes of the process: {bad}")
    for n, r in per.items():
        if not 0.0 <= _num(r, f"promise split {n}") < 1.0:
            raise ProcessSpecError(f"promise split {n}: r_i is in [0, 1), not {r!r}")
    if sum(per.values()) > risk * (1 + 1e-12):
        raise ProcessSpecError(f"promise split: Σ r_i = {sum(per.values())} is over the process risk {risk}")


def _check_node(name, s):
    allowed = {"kind", "reads", "writes", "question", "end", "store", "fingerprint", "promise"}
    if set(s) - allowed:
        raise ProcessSpecError(f"node {name!r}: unknown fields {sorted(set(s) - allowed)}")
    if s.get("kind") not in KINDS:
        raise ProcessSpecError(f"node {name!r}: kind is one of {', '.join(KINDS)}, not {s.get('kind')!r}")
    if not isinstance(s.get("reads"), list) or not all(isinstance(r, str) and r for r in s["reads"]):
        raise ProcessSpecError(f"node {name!r}: reads is a list of fact names")
    w = s.get("writes")
    if not isinstance(w, dict) or not all(isinstance(k, str) and k for k in w):
        raise ProcessSpecError(f"node {name!r}: writes is {{fact: what}}")
    for f, what in w.items():
        if not (what in WRITES or (isinstance(what, str) and what.startswith("answer:") and len(what) > 7)):
            raise ProcessSpecError(f"node {name!r}: writes[{f!r}] is one of {', '.join(WRITES)} or 'answer:<question>', "
                                   f"not {what!r}")
    if s["kind"] == "human" and not s.get("question"):
        raise ProcessSpecError(f"node {name!r}: a human node needs a question")
    if "end" in s and s["end"] is not True:
        raise ProcessSpecError(f"node {name!r}: end is true or absent")


def _check_edge(e, nodes, where):
    if set(e) - {"from", "to", "when", "loop", "max_iter"} or "from" not in e or "to" not in e:
        raise ProcessSpecError(f"{where}: an edge is {{from, to, when[, loop, max_iter]}}, got {sorted(e)}")
    for k in ("from", "to"):
        if e[k] not in nodes:
            raise ProcessSpecError(f"{where}: no node {e[k]!r}")
    check_predicate(e.get("when"), f"{where}.when")
    if "loop" in e or "max_iter" in e:
        m = e.get("max_iter")
        if e.get("loop") is not True:
            raise ProcessSpecError(f"{where}: max_iter is for a loop edge (loop=True)")
        if m is None:
            raise ProcessSpecError(f"{where}: a loop edge needs max_iter (a cycle without a limit is not allowed)")
        if isinstance(m, bool) or not isinstance(m, int) or m < 1:
            raise ProcessSpecError(f"{where}: max_iter is an integer ≥ 1, not {m!r}")


def _check_acyclic(nodes, edges):
    """Every cycle has to pass through a loop edge: the graph without loop edges has no cycle."""
    succ = {n: [] for n in nodes}
    for e in edges:
        if not e.get("loop"):
            succ[e["from"]].append(e["to"])
    state, stack = {}, []

    def visit(n):
        state[n] = 1
        stack.append(n)
        for m in succ[n]:
            if state.get(m) == 1:
                cyc = stack[stack.index(m):] + [m]
                raise ProcessSpecError(f"cycle without max_iter: {' -> '.join(cyc)} (mark the edge that goes back "
                                       "loop=True, max_iter=N)")
            if m not in state:
                visit(m)
        stack.pop()
        state[n] = 2

    for n in nodes:
        if n not in state:
            visit(n)


def _start_of(nodes, edges, start):
    if start is not None:
        return start
    has_in = {e["to"] for e in edges if not e.get("loop")}
    roots = [n for n in nodes if n not in has_in]
    if len(roots) != 1:
        raise ProcessSpecError(f"the start node is ambiguous ({roots or 'none'}): give Process(start=)")
    return roots[0]


def _topological(nodes, edges, keep):
    order, seen = [], set()
    succ = {n: [e["to"] for e in edges if e["from"] == n and not e.get("loop")] for n in nodes}

    def visit(n):
        seen.add(n)
        for m in succ[n]:
            if m not in seen:
                visit(m)
        order.append(n)

    for n in nodes:
        if n not in seen:
            visit(n)
    return [n for n in reversed(order) if n in keep]


def _check_graph(nodes, edges, merges, start):
    if not nodes:
        raise ProcessSpecError("a process has at least one node")
    _check_acyclic(nodes, edges)
    s = _start_of(nodes, edges, start)
    if s not in nodes:
        raise ProcessSpecError(f"start: no node {s!r}")
    if not any(v.get("end") for v in nodes.values()):
        raise ProcessSpecError("no end node (node(..., end=True))")
    seen, todo = {s}, [s]
    while todo:
        n = todo.pop()
        for e in edges:
            if e["from"] == n and e["to"] not in seen:
                seen.add(e["to"])
                todo.append(e["to"])
    lost = [n for n in nodes if n not in seen]
    if lost:
        raise ProcessSpecError(f"nodes not reachable from the start {s!r}: {lost}")
    for n in merges:
        if n not in nodes:
            raise ProcessSpecError(f"merge: no node {n!r}")


# --- diff and render

def _edge_ids(edges):
    """Edge ids "a -> b" ("a -> b #2" for a second edge between the same nodes)."""
    count, out = {}, []
    for e in edges:
        k = f"{e['from']} -> {e['to']}"
        count[k] = count.get(k, 0) + 1
        out.append(k if count[k] == 1 else f"{k} #{count[k]}")
    return out


def _walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            p = f"{path}.{k}" if path else str(k)
            if k not in b:
                out.append({"path": p, "change": "removed", "before": a[k], "after": None})
            elif k not in a:
                out.append({"path": p, "change": "added", "before": None, "after": b[k]})
            else:
                _walk(a[k], b[k], p, out)
    elif canonical(a) != canonical(b):
        out.append({"path": path, "change": "changed", "before": a, "after": b})


def _say(c):
    p, ch = c["path"], c["change"]
    if p.startswith("edges.") and p.endswith(".when"):
        return f"edge {p.split('.')[1]}: predicate {describe(c['before'])} → {describe(c['after'])}"
    if ch == "added":
        return f"{p} added: {canonical(c['after'])}"
    if ch == "removed":
        return f"{p} removed (was {canonical(c['before'])})"
    if ch == "reordered":
        return f"edges reordered: {c['before']} → {c['after']}"
    return f"{p}: {canonical(c['before'])} → {canonical(c['after'])}"


def diff(a, b):
    """The structural changes from spec a to spec b, by node and edge id: one entry per change {"path" (e.g.
    "nodes.fraud.fingerprint.system.parts.score", "edges.triage -> fraud.when", "promise.risk"), "change" ∈ {added,
    removed, changed, reordered}, "before", "after", "say" (in words)}. A node or an edge added or removed is one entry;
    a change inside one is named at the deepest key that differs. [] when the specs describe the same process."""
    a = {k: v for k, v in a.items() if k != "hash"}
    b = {k: v for k, v in b.items() if k != "hash"}
    ea, eb = a.pop("edges", []) or [], b.pop("edges", []) or []
    ia, ib = _edge_ids(ea), _edge_ids(eb)
    out = []
    _walk(a, b, "", out)
    da, db = dict(zip(ia, ea)), dict(zip(ib, eb))
    for i in ia:
        if i not in db:
            out.append({"path": f"edges.{i}", "change": "removed", "before": da[i], "after": None})
    for i in ib:
        if i not in da:
            out.append({"path": f"edges.{i}", "change": "added", "before": None, "after": db[i]})
    for i in ia:
        if i in db:                                   # a predicate is compared whole: it is one condition
            wa, wb = da[i].get("when"), db[i].get("when")
            if canonical(wa) != canonical(wb):
                out.append({"path": f"edges.{i}.when", "change": "changed", "before": wa, "after": wb})
            _walk({k: v for k, v in da[i].items() if k != "when"}, {k: v for k, v in db[i].items() if k != "when"},
                  f"edges.{i}", out)
    common_a, common_b = [i for i in ia if i in db], [i for i in ib if i in da]
    if common_a != common_b:
        out.append({"path": "edges", "change": "reordered", "before": common_a, "after": common_b})
    for c in out:
        c["say"] = _say(c)
    return out


def _mid(n):
    return n.replace("-", "_")


def render(spec, fmt="mermaid"):
    """A spec drawn: "mermaid" (flowchart; a person's node as a hexagon, an end node with a double border, a loop edge
    dotted with its limit) or "text" (one line per node and per edge)."""
    nodes, edges = spec["nodes"], spec["edges"]
    if fmt == "text":
        lines = [f"process {spec['name']} (start {spec['start']}; end {', '.join(spec['end'])})"]
        for n, s in nodes.items():
            q = f" asks {s['question']}" if s.get("question") else ""
            m = f"; merge {spec['merge'][n]}" if n in (spec.get("merge") or {}) else ""
            lines.append(f"  {n}: {s['kind']}{q}; reads {', '.join(s['reads']) or '-'}; writes "
                         f"{', '.join(s['writes']) or '-'}{m}{'; end' if s.get('end') else ''}")
        for i, e in zip(_edge_ids(edges), edges):
            lp = f" (loop, at most {e['max_iter']})" if e.get("loop") else ""
            lines.append(f"  {i} when {describe(e.get('when'))}{lp}")
        pr = spec.get("promise")
        if pr:
            lines.append(f"  promise {pr['mode']}" + (f", risk {pr['risk']}" if pr.get("risk") is not None else ""))
        return "\n".join(lines)
    if fmt != "mermaid":
        raise ValueError(f'render: "mermaid" or "text", not {fmt!r}')
    out = ["flowchart TD"]
    for n, s in nodes.items():
        label = f"{n}<br/>{s['kind']}" + (f": {s['question']}" if s.get("question") else "")
        label = label.replace('"', "'")
        if s["kind"] == "human":
            out.append(f'  {_mid(n)}{{{{"{label}"}}}}')
        elif s.get("end"):
            out.append(f'  {_mid(n)}((("{label}")))')
        else:
            out.append(f'  {_mid(n)}["{label}"]')
    for e in edges:
        w = describe(e.get("when")) if e.get("when") is not None else ""
        if e.get("loop"):
            w = f"{w}; at most {e['max_iter']}".lstrip("; ")
        w = w.replace('"', "'").replace("|", "/")
        arrow = "-.->" if e.get("loop") else "-->"
        out.append(f"  {_mid(e['from'])} {arrow}{f'|{w}|' if w else ''} {_mid(e['to'])}")
    return "\n".join(out)


# --- the path promise (stage 3): helpers and the watch on the mix of paths

def _visits(nodes, edges):
    """An upper bound on the times each node can run in one process run: 1, plus the max_iter of every loop edge that
    can come between two of its runs — taken after it (its source reachable from the node) and leading back to it
    without another loop edge (the node reachable from its target over edges that are not loops). A return to a node
    takes a loop edge, and a loop edge is taken at most max_iter times in a run."""
    adj = {n: set() for n in nodes}
    fwd = {n: set() for n in nodes}
    for e in edges:
        adj[e["from"]].add(e["to"])
        if not e.get("loop"):
            fwd[e["from"]].add(e["to"])

    def reach(a, g):
        seen, todo = {a}, [a]
        while todo:
            for b in g[todo.pop()]:
                if b not in seen:
                    seen.add(b)
                    todo.append(b)
        return seen
    r = {n: reach(n, adj) for n in nodes}
    out = {n: 1 for n in nodes}
    for e in edges:
        if e.get("loop"):
            back = reach(e["to"], fwd)
            for n in back:
                if e["from"] in r[n]:
                    out[n] += int(e["max_iter"])
    return out


def _node_promise(ns, system):
    """A node's own promise as a Promise.record()-like dict: the one given to node(promise=), else its system's (the
    dispatcher's calibrated policy, else System 1's guarantee on the node's question) → dict or None."""
    if ns.get("promise") is not None:
        return dict(ns["promise"])
    if system is None or ns.get("kind") == "guard":
        return None
    pol = getattr(getattr(system, "dispatcher", None), "policy", None)
    if isinstance(pol, dict) and pol.get("method") in ("crc", "ltt", "empirical") and pol.get("level") is not None:
        out = {"method": pol["method"], "n": pol.get("n"), "by": "dispatcher"}
        out["risk" if pol["method"] == "crc" else "error"] = pol["level"]
        if pol.get("delta") is not None:
            out["delta"] = pol["delta"]
        return out
    guards = getattr(getattr(system, "system", system), "guards", None)
    if not isinstance(guards, dict) or not guards:
        return None
    q = ns.get("question")
    g = guards.get(q) if q in guards else (next(iter(guards.values())) if len(guards) == 1 else None)
    rec = getattr(g, "record", None) or getattr(getattr(g, "promise", None), "record", None)   # a QuestionGuard's
    return dict(rec()) if callable(rec) else None


def _say_promise(pr):
    if pr is None:
        return "none"
    if pr.get("method") == "crc":
        return f"risk ≤ {pr.get('risk'):g}"
    return f"{pr.get('method')} error ≤ {pr.get('error')}" + (f", δ {pr['delta']:g}" if pr.get("delta") else "")


def _edge_names(edges):
    names, seen = [], {}
    for e in edges:
        k = f"{e['from']} -> {e['to']}"
        seen[k] = seen.get(k, 0) + 1
        names.append(k if seen[k] == 1 else f"{k} #{seen[k]}")
    return names


def _taken(edges, path):
    """The edges (their indices) a recorded path took."""
    out = set()
    for st in path:
        e = st.get("edge")
        if e is None:
            continue
        for i, x in enumerate(edges):
            if x["from"] == st["node"] and x["to"] == e["to"] and _cj(x.get("when")) == _cj(e.get("when")) \
                    and bool(x.get("loop")) == bool(e.get("loop")):
                out.add(i)
                break
    return out


def _edge_list(edges, r):
    if isinstance(r, ProcessResult):
        return sorted(_taken(edges, r.path))
    names = _edge_names(edges)
    return sorted(names.index(x) if isinstance(x, str) else int(x) for x in r)


class PathWatch:
    """A watch on the mix of paths of a process (stage 3): which edges the runs take. The union promise assumes each
    node is calibrated on the stream that reaches it (A2); a shift in the share of runs that take an edge is a shift in
    that stream. Each edge's share is followed by two Bernoulli CUSUMs (a rise and a fall of its odds by `odds`), the
    reference shares from runs of the calibration stream, the threshold h simulated by resampling those runs (their
    dependence kept: the edges out of a node exclude each other) with the reference's own sampling error, so that an
    unchanged mix is flagged within `horizon` runs with chance at most alpha. A flag is latched until reset(); it puts
    the union promise out of force (Process.guarantee()). What it reports is "the paths changed", not why.

        watch = PathWatch(alpha=0.01, horizon=1000)
        p.monitor(watch, reference=calibration_runs)      # ProcessResults (or lists of edges) of the calibration stream
        r = p.run(facts); r.path_drift                    # {"drift", "why", "edge", "S", "h", "seen"}

    The same CUSUM as AuditWatch and DriftMonitor (solvi.core.guarantees.drift.Cusum)."""

    def __init__(self, *, alpha=0.01, horizon=1000, odds=2.0, sims=None, seed=0):
        if not float(odds) > 1.0:
            raise ValueError("odds: the rise (and fall) of an edge's odds to detect, above 1")
        if not 1e-4 <= float(alpha) < 1:
            raise ValueError("alpha between 1e-4 and 1")
        self.alpha, self.horizon, self.odds, self.sims, self.seed = float(alpha), int(horizon), float(odds), sims, seed
        self.names = self.p0 = self.cusum = self.h = None
        self.n_ref = None
        self.seen, self._flagged, self.why, self.edge = 0, False, [], None

    @property
    def flagged(self):
        return self._flagged

    @staticmethod
    def _inc(p0, odds):
        o = p0 / (1 - p0)
        up, dn = o * odds / (1 + o * odds), (o / odds) / (1 + o / odds)
        return (np.log(up / p0), np.log((1 - up) / (1 - p0)), np.log(dn / p0), np.log((1 - dn) / (1 - p0)))

    def set_reference(self, runs, names=None):
        """runs: one list of edge indices per run of the calibration stream (at least 20); names: the edges' names
        (default "e0", "e1", ...). → self (h simulated here)."""
        import warnings
        rows = [sorted({int(i) for i in r}) for r in runs]
        if len(rows) < 20:
            raise ValueError(f"a reference of {len(rows)} runs: at least 20 runs of the calibration stream")
        n_e = len(names) if names else max([max(r) + 1 for r in rows if r] or [1])
        if any(r and (r[0] < 0 or r[-1] >= n_e) for r in rows):
            raise ValueError(f"an edge index outside the {n_e} edges")
        self.names = list(names) if names else [f"e{i}" for i in range(n_e)]
        R = np.zeros((len(rows), len(self.names)))
        for j, r in enumerate(rows):
            R[j, r] = 1.0
        n = len(rows)
        self.n_ref, self._R = n, R
        self.p0 = (R.sum(0) + 0.5) / (n + 1.0)
        odds = self.odds
        inc = self._inc

        def make_null(rng, sims):
            counts = rng.multinomial(n, np.full(n, 1.0 / n), size=sims)        # the reference, redrawn (its own error)
            q0 = (counts @ R + 0.5) / (n + 1.0)
            u1, u0, d1, d0 = inc(q0, odds)

            def nxt():
                x = R[rng.integers(n, size=sims)]                              # runs drawn as they came, edges together
                return np.concatenate([np.where(x > 0, u1, u0), np.where(x > 0, d1, d0)], axis=1)
            return nxt
        from ..core.guarantees.drift import Cusum
        chans = [f"{k} up" for k in self.names] + [f"{k} down" for k in self.names]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.cusum = Cusum.calibrate(chans, make_null, alpha=self.alpha, horizon=self.horizon, sims=self.sims,
                                         seed=self.seed)
        self.h = self.cusum.h
        self._u1, self._u0, self._d1, self._d0 = inc(self.p0, odds)
        self.reset()
        return self

    def reset(self):
        """The stream starts again: the CUSUMs and the flag cleared; the reference kept."""
        if self.cusum is not None:
            self.cusum.reset()
        self.seen, self._flagged, self.why, self.edge = 0, False, [], None

    def observe(self, taken, names=None):
        """One run's edges (indices) → {"drift", "why", "edge", "S", "h", "seen"}."""
        if self.cusum is None:
            raise RuntimeError("set_reference() first (or Process.monitor(watch, reference=runs))")
        if names is not None and list(names) != self.names:
            raise ValueError("the process's edges are not the ones the reference was taken on")
        self.seen += 1
        x = np.zeros(len(self.names))
        x[[int(i) for i in taken]] = 1.0
        if not self._flagged:
            self.cusum.step(np.concatenate([np.where(x > 0, self._u1, self._u0), np.where(x > 0, self._d1, self._d0)]))
            ch, S, since = self.cusum.top()
            if S >= self.h:
                k = len(self.names)
                j = int(np.argmax(self.cusum.S))
                e, way = self.names[j % k], ("more" if j < k else "fewer")
                self._flagged, self.edge = True, e
                self.why = [f"{way} runs take {e} than in the reference ({self.p0[j % k]:.3g} of {self.n_ref}), "
                            f"since about run {self.seen - since} (S {S:.3g} ≥ h {self.h:.3g})"]
        _, S, _ = self.cusum.top()
        return {"drift": self._flagged, "why": list(self.why), "edge": self.edge, "S": S, "h": self.h,
                "seen": self.seen}

    def fingerprint(self):
        from ..core.provenance import digest
        return digest("PathWatch", self.alpha, self.horizon, self.odds, self.seed,
                      None if self.p0 is None else [round(float(v), 12) for v in self.p0], self.n_ref)


__all__ = ["FORMAT", "CallbackChannel", "DecisionNode", "GuardNode", "HumanAnswer", "HumanChannel", "HumanNode",
           "HumanRequest", "MemoryQueue", "NodeResult", "NodeRunner", "PathWatch", "Process", "ProcessResult",
           "ProcessRunError", "ProcessSpecError", "RUNNERS", "SystemNode", "canonical", "check_predicate", "describe",
           "diff", "fingerprint_of", "holds", "render", "runner_for", "spec_hash"]
