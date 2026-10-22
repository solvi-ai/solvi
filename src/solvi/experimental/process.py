"""A process of several solvi systems (experimental, stages 0 and 1a): the nodes, the edges with predicates written as
data, merges, loops with a mandatory iteration limit and the declared promise — a JSON specification with a hash, a
structural diff and a drawing (stage 0) — and running it: each node's decision in its own store, chained into one run,
with an index record of the path (stage 1a). Replay, the process guarantee and the channel to a person come in later
stages.

    from solvi.experimental.process import Process

    p = Process("claim")
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

    p = Process("claim", storage="st/process.jsonl", on_abstain="abstain", on_human="abstain")   # ... nodes, edges
    r = p.run({"claim": claim})         # ProcessResult: status ("decided" | "abstain"), answer, path, nodes, facts,
                                        #   provenance, run, stored_id (the "process" record)
    print(p.explain(r))                 # the path in words: who answered at each node, which edge was taken
    chronology(stores, run=r.run)       # solvi.core.store.chronology: the node decisions and the record, in order

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
"handoff" (the default) and a node of kind "human" go to the channel to a person — stage S1b; this stage raises
NotImplementedError there. A node that merges "all" of several inputs raises NotImplementedError too (this stage
follows one path). Not yet: `replay`, `retract`, `pending`/`resume` and the `HumanChannel` classes; `human=` takes any
object with a `fingerprint()` for now and only records it.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any

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
        from ..core.runtime import trace_hash
        d = self.system.ask(facts)
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
                          bool(d.drift), bool(d.over_budget), d, self.store)


class SystemNode(NodeRunner):
    """A bare System: ask(facts) → Response; stored as an "ask" record when the System has storage. The main answer is
    the node's question's (question=, or the System's one question); without one, the dict of every answer."""
    kind = "system"

    def run(self, facts):
        from ..core.runtime import trace_hash
        resp = self.system.ask(facts)
        q = self.spec.get("question")
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
                          resp, self.store)


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
    """A question put to a person through the process's channel (stage S1b: HumanChannel, pending / resume)."""
    kind = "human"

    def run(self, facts):
        raise NotImplementedError("a human node asks a person through the process's channel: that comes in the next "
                                  "stage (S1b, HumanChannel with pending / resume)")


RUNNERS = {"decision": DecisionNode, "system": SystemNode, "guard": GuardNode, "human": HumanNode}


def runner_for(kind, system, spec):
    """The built-in NodeRunner of a node kind for its system (a NodeRunner given as the system is used as is)."""
    if isinstance(system, NodeRunner):
        return system
    return RUNNERS[kind](system, spec)


@dataclass
class ProcessResult:
    """One run of a process. status: "decided" (an end node answered), "abstain" (it stopped before an end node
    answered: `why` says where); answer: the end node's answer (None unless decided); path: the steps as recorded
    ([{node, kind, store, stored_id, trace_hash, by, status, answers_hash, facts_in_hash, facts_out_hash, knowledge,
    cost, edge}]); nodes: {node: NodeResult} (the last result of each node); facts: every fact at the end; provenance:
    {fact: {"from", "decision", "by", "confidence", "trace_hash"}} for the facts nodes wrote; run: the run id;
    stored_id: the "process" record (None without storage=); spec_hash; why: why it stopped (abstain);
    pending, ticket: for the channel to a person (stage S1b)."""
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


# --- the process

class Process:
    """A process of solvi systems (see the module docs). name: the process's name; storage: a path or a TraceStorage
    where each run's "process" record is appended (not part of the spec; None: no record, the node stores still hold
    every decision); human: the channel to a person (recorded by fingerprint only, for
    now); start: the start node (default: the one node without an incoming edge); on_abstain, on_human: "handoff" (to
    the channel) or "abstain"."""

    def __init__(self, name, storage=None, human=None, start=None, on_abstain="handoff", on_human="handoff"):
        if not isinstance(name, str) or not name:
            raise ProcessSpecError(f"a process name is a non-empty string, not {name!r}")
        for k, v in (("on_abstain", on_abstain), ("on_human", on_human)):
            if v not in ON:
                raise ProcessSpecError(f"{k} is one of {', '.join(ON)}, not {v!r}")
        self.name, self.storage, self.human, self._start = name, storage, human, start
        self.on_abstain, self.on_human = on_abstain, on_human
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
    def from_spec(cls, spec, systems=None, human=None, storage=None, strict=True):
        """A process from its spec (a dict or JSON text). systems: {node: system} to bind (each checked against its
        recorded fingerprint); human: the channel (checked likewise). strict=False loads a mismatch and lists it in
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
                on_human=spec.get("on_human", "handoff"))
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

    # --- running (stage 1a)

    def run(self, facts, run=None):
        """Run the process once on the given facts ({name: value}) → ProcessResult (see the module docs). run: the run
        id (default: the enclosing run_context's, else a new one). Every node's decision is written to its own store
        inside run_context(run=..., parent=the previous node's stored id); with storage= the process writes one record
        of kind "process" (the index of the path: hashes, ids, the edges taken) whose parent is the last node's
        decision."""
        from ..core.store import current_run, run_context
        spec = self.spec()
        lack = [n for n, ns in self.nodes.items() if ns["kind"] != "human" and n not in self.systems]
        if lack:
            raise ProcessRunError(f"nodes without a system: {lack} (Process.from_spec(spec, systems=...))")
        outer = current_run()
        facts = dict(facts)
        given = dict(facts)
        prov, path, results = {}, [], {}
        taken = [0] * len(self.edges)
        node, status, answer, why = spec["start"], None, None, None
        with run_context(run=run) as rc:
            parent = outer["parent"] if outer else None
            while True:
                ns = self.nodes[node]
                if self.merges.get(node) == "all" and sum(1 for e in self.edges if e["to"] == node
                                                          and not e.get("loop")) > 1:
                    raise NotImplementedError(f"node {node!r} merges 'all' of several inputs: this stage follows one "
                                              "path (the first edge that holds); waiting for several branches comes "
                                              "later")
                if ns["kind"] == "human":
                    return self._human(node, ns, facts, prov, path)
                runner = runner_for(ns["kind"], self.systems[node], ns)
                state = {r: facts[r] for r in ns["reads"] if r in facts and r != "provenance"}
                if "provenance" in ns["reads"]:
                    state["provenance"] = {r: prov[r] for r in ns["reads"] if r in prov}
                with run_context(parent=parent):
                    res = runner.run(state)
                out = {f: res.value(what) for f, what in ns["writes"].items()}
                label = {"from": node, "decision": res.stored_id, "by": res.by, "confidence": res.confidence,
                         "trace_hash": res.trace_hash}
                facts.update(out)
                prov.update({f: dict(label) for f in out})
                results[node] = res
                outcome = {"facts": facts, "status": res.status, "by": res.by, "confidence": res.confidence,
                           "budget_over": res.budget_over, "drift": res.drift}
                edge = None
                for i, e in enumerate(self.edges):
                    if e["from"] != node or (e.get("loop") and taken[i] >= e["max_iter"]):
                        continue
                    if holds(e.get("when"), outcome):
                        edge, taken[i] = e, taken[i] + 1
                        break
                kh = state.get("knowledge")
                if kh is None and res.raw is not None and getattr(res.raw, "s1", None) is not None:
                    kh = res.raw.s1.trace.init.get("knowledge")
                path.append({"node": node, "kind": ns["kind"], "store": ns.get("store"), "stored_id": res.stored_id,
                             "trace_hash": res.trace_hash, "by": res.by, "status": res.status,
                             "answers_hash": _vh(res.answers), "facts_in_hash": _vh(state),
                             "facts_out_hash": _vh(out), "knowledge": None if kh is None else _vh(kh),
                             "cost": res.cost,
                             "edge": None if edge is None else {"to": edge["to"], "when": edge.get("when"),
                                                                "loop": bool(edge.get("loop"))}})
                if res.stored_id is not None:
                    parent = res.stored_id
                if edge is not None:
                    node = edge["to"]
                    continue
                if res.status == "human" or (res.status == "abstain"):
                    on = self.on_human if res.status == "human" else self.on_abstain
                    if on == "handoff":
                        return self._handoff(node, res, facts, prov, path)
                    status, why = "abstain", f"{node}: {'a person has to decide' if res.status == 'human' else 'abstained'}"
                elif ns.get("end"):
                    status, answer = "decided", res.answer
                else:
                    status, why = "abstain", f"{node}: no edge holds and it is not an end node"
                break
            with run_context(parent=parent):          # the record follows from the last node's decision
                rec_id = self._record(spec, given, path, status, answer, node)
        return ProcessResult(status, answer, path, results, facts, prov, rc.id, rec_id, spec["hash"], why)

    def _record(self, spec, given, path, status, answer, at):
        """The "process" record (an index: the node stores hold the decisions) → its id, or None without storage."""
        if self.storage is None:
            return None
        from ..core.store import FORMAT as RECORD_FORMAT
        from ..core.store import open_storage
        if self._store is None:
            self._store = open_storage(self.storage)
        pr = spec.get("promise")
        rec = self._store._append({"v": RECORD_FORMAT, "kind": "process", "process": spec["name"],
                                   "spec_hash": spec["hash"], "input_hash": _vh(given),
                                   "mode": pr["mode"] if pr else None, "promise": pr, "path": _plain(path),
                                   "outcome": {"status": status, "at": at, "answer_hash": _vh(answer)}})
        return rec["id"]

    def _human(self, node, ns, facts, prov, path):
        """A human node: the question goes to the channel (stage S1b)."""
        return runner_for("human", None, ns).run(facts)

    def _handoff(self, node, res, facts, prov, path):
        """on_abstain / on_human = "handoff": the decision goes to the channel (stage S1b)."""
        raise NotImplementedError(f"node {node!r} hands its decision to a person ({res.status}): the channel to a "
                                  "person comes in the next stage (S1b); Process(on_abstain='abstain', "
                                  "on_human='abstain') stops the run there instead")

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
            lines.append(f"  decided: {json.dumps(_plain_or_repr(result.answer), ensure_ascii=False)}")
        else:
            lines.append(f"  {result.status}: {result.why}")
        return "\n".join(lines)

    # --- reading it

    def diff(self, other):
        """The changes from this process to `other` (a Process or a spec): see `diff`."""
        return diff(self.spec(), other.spec() if isinstance(other, Process) else other)

    def render(self, fmt="mermaid"):
        """The process drawn: "mermaid" (a flowchart) or "text"."""
        return render(self.spec(), fmt)

    def __repr__(self):
        return f"Process({self.name!r}, {len(self.nodes)} nodes, {len(self.edges)} edges)"


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


__all__ = ["FORMAT", "DecisionNode", "GuardNode", "HumanNode", "NodeResult", "NodeRunner", "Process", "ProcessResult",
           "ProcessRunError", "ProcessSpecError", "RUNNERS", "SystemNode", "canonical", "check_predicate", "describe",
           "diff", "fingerprint_of", "holds", "render", "runner_for", "spec_hash"]
