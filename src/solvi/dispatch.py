"""Who answers: the fast system within its guarantee, a slow path when the fast one is unsure, or a person — one recorded
decision per input, with a budget.

    from solvi.dispatch import Budget, Dispatcher, SlowPath

    slow = SlowPath(system2)                       # a System that answers the same question slowly (an LLM part, ...)
    d = Dispatcher(system, slow, question="intent", price=(0.04, 0.17),
                   budget=Budget(usd=0.002, calls=4), total=Budget(usd=5.0), supervise=0.05)
    res = d.ask({"text": "my card was charged twice"})
    res.answer, res.by, res.reasons, res.cost      # by: "s1", "s2" or "human"; why it went that way; what it cost
    res.replay(d)                                  # re-derives the dispatch; the LLM outputs are checked, not re-asked

The fast system (System 1) is an ordinary System: a catalog of rules, checks, fitted heads and deciders, with a
guarantee on the question (`System.guarantee`, an open-set gate). It is always asked first — it is cheap. Its response
then says whether its answer can be given alone; the dispatcher reads the signals the library already has:

    signal        wakes the slow path when                                              action
    guarantee     the answer is below the question's guarantee (it abstained, "low_confidence")   think
    openset       below an open-set gate's threshold (the input is unlike the calibration)        think
    abstain       System 1 abstained otherwise (a model escalated, a fact is missing, a rule abstained)  think
    constraint    the answers break a constraint (`res.feasible` is False)                       think
    agreement     an agreement share the catalog computes (`agree`) is below its minimum             think
    drift         the open-set gate or a DriftMonitor has flagged a change of the stream          check
    supervise     a sampled share of the answers System 1 gives alone                             check

"think": System 1's answer is not given alone; the slow path answers, and when it cannot (its checks or its own
guarantee do not accept its answer, its budget runs out) a person does. "check": System 1's answer stands within its
guarantee and the slow path answers too; a disagreement is recorded (the material a teacher of System 1 learns from)
and, with `on_disagree="human"` or `"s2"`, goes to a person or to the slow path's accepted answer instead. A hard check
that forces System 1's answer is never re-thought: the check decides. A signal left out of `wake` sends its inputs to a
person instead of the slow path.

The slow path (`SlowPath`) is built from the existing parts and judged by a System's checks:
  - a System that answers the question itself — an LLM decision part (`solvi.llm`), generated candidates with
    `solvi.generate` and `solvi.agree` in its catalog, typed facts with quotes — with its own guarantee;
  - `propose=` (a `Generator.proposer`): `solvi.refine` — propose, check with the System's hard checks, re-ask with the
    reasons of the failed checks, within the budget;
  - `space=`: `solvi.search` — the candidates of an enumerable space through the checks.
Its answer is accepted when the System does not abstain on it (and, with refine and search, its checks accept it).

The budget. `budget=` is per decision (the slow path's dollars, model calls and milliseconds), `total=` for the
dispatcher's life. Dollars come from the tokens every model output records in the trace (`extra["llm"]["usage"]`,
`extra["generated"]["usage"]`) times `price` (dollars per million input and output tokens, or a function), so a
stored decision's cost is recomputed on replay. Before the slow path starts, its expected cost (the mean of the runs so
far) must fit what is left of both budgets; between the rounds of a refinement the spend so far must; otherwise the
input goes to a person — "no budget left", never a guess. A single System ask cannot be stopped half-way: a run that
went over the per-decision budget keeps its answer and records how far over it went.

The record. `ask` returns a `Dispatched`: the answer, who gave it, the action and its reasons, System 1's response,
the slow path's record (its responses, rounds or search, each a replayable trace), the cost of both and what had been
spent before. `replay(dispatcher)` re-checks all of it without calling a model: System 1's trace replays, the dispatch
follows from that response, the recorded supervision draw (a hash of the seed, the input and the decision's number),
the recorded budget and drift state; the slow path's record replays (the LLM outputs re-read through their schemas, as
`solvi.generate` and `solvi.llm` replay them); the cost recomputes; the answer follows. With `storage=` every decision
is one hash-chained record of kind "dispatch" in a TraceStorage (`stored()`, `replay_all()`).

Not done here: the slow path does not teach System 1 (the disagreements and the slow path's accepted answers are
recorded as material, `disagreements()`, not used); several questions at once (one dispatcher per question); parallel
asks (the budget, the drift state and the decision numbers follow the order of the asks); a drift flag is latched until
`reset_drift()` and is taken as recorded on replay, since it depends on the stream before the decision."""
from __future__ import annotations

import hashlib
import json
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any

SIGNALS = ("guarantee", "openset", "abstain", "constraint", "agreement", "drift", "supervise")
THINK = ("guarantee", "openset", "abstain", "constraint", "agreement")
PATHS = ("s1", "s2", "human")


# ------------------------------------------------------------------------------------------------ budget and cost
@dataclass(frozen=True)
class Budget:
    """Limits on the slow path: dollars, model calls, milliseconds (None: no limit on that one)."""
    usd: float | None = None
    calls: int | None = None
    ms: float | None = None

    def __post_init__(self):
        for k in ("usd", "calls", "ms"):
            v = getattr(self, k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0 or v != v):
                raise ValueError(f"Budget {k} must be a number ≥ 0 or None, not {v!r}")

    def over(self, cost, plus=None):
        """The first limit that `cost` (+ `plus`, an expected further cost) goes over → its reason, or None."""
        for k in ("usd", "calls", "ms"):
            lim = getattr(self, k)
            if lim is None:
                continue
            v = getattr(cost, k) + (getattr(plus, k) if plus is not None else 0)
            if v > lim + 1e-12:
                more = f" + {getattr(plus, k):.4g} expected" if plus is not None else ""
                return f"{k} {getattr(cost, k):.4g}{more} > {lim:g}"
        return None

    def used_up(self, cost):
        """The first limit `cost` has reached → its reason, or None."""
        for k in ("usd", "calls", "ms"):
            lim = getattr(self, k)
            if lim is not None and getattr(cost, k) >= lim - 1e-12:
                return f"{k} {getattr(cost, k):.4g} of {lim:g} used"
        return None

    def to_dict(self):
        return {"usd": self.usd, "calls": self.calls, "ms": self.ms}


@dataclass
class Cost:
    """What a decision (or a part of one) cost: dollars (None when no price is known), model calls, milliseconds and
    tokens."""
    usd: float | None = 0.0
    calls: int = 0
    ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, o):
        usd = None if self.usd is None or o.usd is None else self.usd + o.usd
        return Cost(usd, self.calls + o.calls, self.ms + o.ms, self.input_tokens + o.input_tokens,
                    self.output_tokens + o.output_tokens)

    def to_dict(self):
        return {"usd": None if self.usd is None else round(self.usd, 9), "calls": self.calls, "ms": round(self.ms, 3),
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens}

    @classmethod
    def from_dict(cls, d):
        return cls(d.get("usd"), int(d.get("calls", 0)), float(d.get("ms", 0.0)), int(d.get("input_tokens", 0)),
                   int(d.get("output_tokens", 0)))


def _usages(extra):
    """The model calls recorded in one trace record's extra → [(model, usage)]."""
    out = []
    if not isinstance(extra, dict):
        return out
    lm = extra.get("llm")
    if isinstance(lm, dict) and isinstance(lm.get("usage"), dict):
        out.append((lm.get("model"), lm["usage"]))
    gen = extra.get("generated")
    for m in gen if isinstance(gen, list) else [gen] if isinstance(gen, dict) else []:
        if isinstance(m, dict) and isinstance(m.get("usage"), dict):
            out.append((m.get("model"), m["usage"]))
    return out


def recorded_calls(responses, generated=()):
    """The model calls recorded in responses' traces (and in generator records outside a trace, as refine keeps the
    proposer's) → [(model, usage)]."""
    out = []
    for res in responses:
        if res is None:
            continue
        for r in res.trace.records:
            out += _usages(getattr(r, "extra", None))
    for g in generated:
        out += _usages({"generated": g})
    return out


def price_of(price, calls):
    """Dollars of recorded calls: price (dollars per million input and output tokens), a function (model, usage) →
    dollars, or None (unknown → None)."""
    if price is None:
        return None if calls else 0.0
    total = 0.0
    for model, u in calls:
        if callable(price):
            total += float(price(model, u))
        else:
            pin, pout = price
            total += (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout) / 1e6
    return total


def cost_of(responses, price, ms=0.0, generated=()):
    """The Cost of what responses (and generator records) recorded, with `ms` as the time."""
    calls = recorded_calls(responses, generated)
    return Cost(price_of(price, calls), len(calls), float(ms), sum(u.get("input_tokens", 0) for _, u in calls),
                sum(u.get("output_tokens", 0) for _, u in calls))


class BudgetStop(RuntimeError):
    """The slow path was stopped between two steps: what is left of the budget would not cover the next one."""


# ------------------------------------------------------------------------------------------------ the slow path
@dataclass
class Thought:
    """What the slow path did for one input: its mode ("ask", "refine", "search"), its answer (None when it has none),
    whether that answer was accepted, why not, its record (a Response, a Refinement or a SearchRun — replayable) and
    its cost."""
    mode: str
    answer: Any = None
    accepted: bool = False
    why: str | None = None
    record: Any = None
    cost: Cost = field(default_factory=Cost)
    stopped: str | None = None           # the budget stopped it between steps

    @property
    def responses(self):
        r = self.record
        if r is None:
            return []
        if self.mode == "ask":
            return [r]
        if self.mode == "refine":
            return [x.response for x in r.rounds if x.response is not None]
        return [r.response] if r.response is not None else []

    def generated(self):
        """The generator records kept outside the traces (a refinement's proposals)."""
        if self.mode != "refine" or self.record is None:
            return []
        return [x.generated for x in self.record.rounds if x.generated is not None]

    def to_dict(self):
        from .storage import plain
        rec = None if self.record is None else self.record.to_dict()
        return {"mode": self.mode, "answer": plain(self.answer), "accepted": self.accepted, "why": self.why,
                "record": rec, "cost": self.cost.to_dict(), "stopped": self.stopped}

    @classmethod
    def from_dict(cls, d, system=None):
        rec = d.get("record")
        if rec is not None:
            if d["mode"] == "ask":
                from .system import Response
                rec = Response.model_validate(rec, catalog=system)
            elif d["mode"] == "refine":
                from .refine import Refinement
                rec = Refinement.from_dict(rec, catalog=system)
            else:
                from .search import SearchRun
                rec = SearchRun.from_dict(rec, catalog=system)
        return cls(d["mode"], d.get("answer"), d["accepted"], d.get("why"), rec, Cost.from_dict(d.get("cost") or {}),
                   d.get("stopped"))


class SlowPath:
    """System 2: a slow, checked way to answer the question (see the module docs).

    system: the System that answers or judges — one whose question is answered by a model (an LLM decision part,
    generated candidates with agreement), or System 1 itself as the judge of proposals. question: its question's name
    when it differs from the dispatcher's. propose: a proposer for solvi.refine (`Generator.proposer(...)`) — the
    proposal is given to the System as the fact `into`, checked, and re-asked with the reasons up to `rounds` times.
    space: a space for solvi.search (with `into`, `search=` its other options: objective, prune, keep, budget).
    accept: what accepts a refined or searched proposal ("checks", an answer or answers, a function of the Response),
    as in solvi.refine; the System must also not abstain on the question. feedback: refine's feedback function."""

    def __init__(self, system, *, question=None, propose=None, into=None, rounds=3, space=None, search=None,
                 accept="checks", feedback=None):
        if propose is not None and space is not None:
            raise ValueError("a slow path refines proposals (propose=) or searches a space (space=), not both")
        if (propose is not None or space is not None) and not isinstance(into, str) and not isinstance(space, dict):
            raise ValueError("into= names the given fact a proposal or a candidate is passed as")
        if propose is not None and not callable(propose):
            raise TypeError("propose is a function (state, rounds) → a proposal, e.g. Generator.proposer(...)")
        if int(rounds) < 1:
            raise ValueError("rounds must be at least 1")
        self.system, self.question, self.propose, self.into = system, question, propose, into
        self.rounds, self.space, self.search_kw = int(rounds), space, dict(search or {})
        self.accept, self.feedback = accept, feedback
        self.mode = "refine" if propose is not None else "search" if space is not None else "ask"

    def __repr__(self):
        return f"SlowPath({self.mode})"

    def fingerprint(self):
        from .provenance import code_fingerprint, digest
        acc = code_fingerprint(self.accept) if callable(self.accept) else repr(self.accept)
        return digest("SlowPath", self.mode, self.system.fingerprint(), self.question, self.into, self.rounds, acc,
                      sorted(self.search_kw), code_fingerprint(self.propose) if self.propose is not None else None)

    def _q(self, question):
        q = self.question or question
        if q not in self.system.questions:
            raise ValueError(f"the slow path's System has no question {q!r} (question= names it)")
        return q

    def run(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        """Think about one input → a Thought. budget: the per-decision Budget, checked between the rounds of a
        refinement (with expected_round, the expected Cost of one more round)."""
        q = self._q(question)
        t0 = time.perf_counter()
        if self.mode == "ask":
            res = self.system.ask(dict(state), [q], store=store)
            r = res[q]
            ok = r.status != "abstain"
            th = Thought("ask", r.answer if ok else _would(r), ok, None if ok else f"the slow path abstained: {r.why}",
                         res)
        elif self.mode == "refine":
            th = self._refine(state, q, price, budget, expected_round, t0, store)
        else:
            th = self._search(state, q, store)
        th.cost = cost_of(th.responses, price, (time.perf_counter() - t0) * 1000, th.generated())
        return th

    def _refine(self, state, q, price, budget, expected_round, t0, store):
        from .refine import refine
        inner = self.propose

        def propose(st, rounds):
            if budget is not None and rounds:
                so_far = cost_of([x.response for x in rounds], price, (time.perf_counter() - t0) * 1000,
                                 [x.generated for x in rounds if x.generated is not None])
                why = budget.over(so_far, expected_round)
                if why:
                    raise BudgetStop(f"no budget for another round ({why})")
            return inner(st, rounds)
        run = refine(self.system, dict(state), q, propose, into=self.into, rounds=self.rounds, accept=self.accept,
                     feedback=self.feedback, store=store)
        r = run.result
        ok = bool(run.accepted) and r is not None and r.status != "abstain"
        stopped = None
        last = run.rounds[-1] if run.rounds else None
        if last is not None and last.error and "BudgetStop" in last.error:
            stopped = last.error
        why = None if ok else (run.escalation or (f"the slow path abstained: {r.why}" if r is not None else "no answer"))
        ans = (r.answer if ok else _would(r)) if r is not None else None
        return Thought("refine", ans, ok, why, run, stopped=stopped)

    def _search(self, state, q, store):
        from .search import search
        run = search(self.system, dict(state), q, self.space, into=self.into, accept=self.accept, store=store,
                     **self.search_kw)
        res = run.response
        r = res[q] if res is not None else None
        ok = run.best is not None and r is not None and r.status != "abstain"
        why = None if ok else (run.escalation or "no candidate of the space was accepted")
        return Thought("search", r.answer if ok else None, ok, why, run)

    def replay(self, thought, trust_models=False):
        """Re-check a recorded Thought: its record replays under this path's System (no model is called), and its
        answer and acceptance follow from the record → {"ok", "mismatches": [(what, why)]}."""
        bad = []
        rec = thought.record
        q = self.question
        if rec is None:
            if thought.accepted:
                bad.append(("record", "an accepted answer without its record"))
            return {"ok": not bad, "mismatches": bad}
        if thought.mode != self.mode:
            return {"ok": False, "mismatches": [("mode", f"recorded {thought.mode}, the slow path is {self.mode}")]}
        if thought.mode == "ask":
            rep = rec.trace.replay(self.system, trust_models=trust_models)
            bad += [("trace", f"{m[1]}: {m[2]}") for m in rep["mismatches"]]
            qn = q or next(iter(rec.results))
            r = rec[qn]
            ok = r.status != "abstain"
            ans = r.answer if ok else _would(r)
        elif thought.mode == "refine":
            rep = rec.replay(self.system, accept=None if not callable(self.accept) else self.accept,
                             trust_models=trust_models)
            bad += [("refine", f"round {m[0]} {m[1]}: {m[2]}") for m in rep["mismatches"]]
            r = rec.result
            ok = bool(rec.accepted) and r is not None and r.status != "abstain"
            ans = (r.answer if ok else _would(r)) if r is not None else None
        else:
            rep = rec.replay(self.system, accept=self.accept, objective=self.search_kw.get("objective"),
                             trust_models=trust_models)
            bad += [("search", f"{m[0]}: {m[1]}") for m in rep["mismatches"]] if rec.response is not None else []
            r = rec.response[rec.question] if rec.response is not None else None
            ok = rec.best is not None and r is not None and r.status != "abstain"
            ans = r.answer if ok else None
        if ok != thought.accepted:
            bad.append(("accepted", f"recorded {thought.accepted}, the record gives {ok}"))
        if _vh(ans) != _vh(thought.answer):
            bad.append(("answer", f"recorded {thought.answer!r}, the record gives {ans!r}"))
        return {"ok": not bad, "mismatches": bad}


# ------------------------------------------------------------------------------------------------ one decision
@dataclass
class Dispatched:
    """One dispatched decision. answer: the answer given (None when a person takes it); by: "s1", "s2" or "human";
    action: "accept" (System 1 alone), "think" (the slow path was asked to answer), "check" (it checked System 1's
    answer), "human" (straight to a person); reasons: why the dispatch went that way (and why a person got it);
    s1: System 1's Response; s2: the slow path's Thought (None when it did not run); candidates: what each path would
    have answered, for the person; disagreement: System 1's and the slow path's answers when they differ; cost:
    {"s1", "s2", "total"} Costs; n: the decision's number; draw: its supervision draw; spent_before: the dispatcher's
    total spend before it; drift: the drift flag it saw."""
    question: str
    answer: Any
    by: str
    action: str
    reasons: list
    s1: Any
    s2: Thought | None = None
    candidates: dict = field(default_factory=dict)
    disagreement: dict | None = None
    cost: dict = field(default_factory=dict)
    n: int = 0
    draw: float = 1.0
    spent_before: Cost = field(default_factory=Cost)
    drift: str | None = None
    expected: Cost | None = None
    over_budget: str | None = None
    config: str | None = None
    stored_id: str | None = None

    @property
    def supervised(self):
        return self.action == "check"

    def to_dict(self):
        from .storage import plain
        return {"question": self.question, "answer": plain(self.answer), "by": self.by, "action": self.action,
                "reasons": list(self.reasons), "s1": self.s1.to_dict(),
                "s2": self.s2.to_dict() if self.s2 is not None else None, "candidates": plain(self.candidates),
                "disagreement": plain(self.disagreement), "cost": {k: v.to_dict() for k, v in self.cost.items()},
                "n": self.n, "draw": self.draw, "spent_before": self.spent_before.to_dict(), "drift": self.drift,
                "expected": None if self.expected is None else self.expected.to_dict(), "over_budget": self.over_budget,
                "config": self.config}

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_dict(cls, d, system=None, slow_system=None):
        """A stored decision back: System 1's response restored with `system`'s types, the slow path's with
        `slow_system`'s (default: the same)."""
        from .system import Response
        s1 = Response.model_validate(d["s1"], catalog=system)
        s2 = Thought.from_dict(d["s2"], slow_system or system) if d.get("s2") is not None else None
        return cls(d["question"], d.get("answer"), d["by"], d["action"], list(d["reasons"]), s1, s2,
                   d.get("candidates") or {}, d.get("disagreement"),
                   {k: Cost.from_dict(v) for k, v in (d.get("cost") or {}).items()}, d.get("n", 0), d.get("draw", 1.0),
                   Cost.from_dict(d.get("spent_before") or {}), d.get("drift"),
                   Cost.from_dict(d["expected"]) if d.get("expected") is not None else None, d.get("over_budget"),
                   d.get("config"))

    def replay(self, dispatcher, trust_models=False):
        """Re-check this decision against `dispatcher` without calling a model → {"ok", "mismatches": [(what, why)]}:
        see Dispatcher.replay."""
        return dispatcher.replay(self, trust_models=trust_models)

    def __repr__(self):
        return f"Dispatched({self.question}={self.answer!r} by {self.by}, {self.action}: {'; '.join(self.reasons)[:120]})"


# ------------------------------------------------------------------------------------------------ the dispatcher
class Dispatcher:
    """System 1 first; the slow path or a person when its signals say so; within a budget. See the module docs.

    system: System 1. slow: a SlowPath (None: every input System 1 cannot answer alone goes to a person). question:
    the question dispatched (default: the System's only question). budget: a Budget per decision; total: a Budget for
    the dispatcher's life. price: dollars per million (input, output) tokens, or a function (model, usage) → dollars —
    required for a budget in dollars. wake: the signals that wake the slow path (default: all of SIGNALS); an input
    whose signal is not among them goes to a person. agreement: {fact: minimum share} — an agreement fact of System 1
    below its minimum wakes the slow path. monitor: a solvi.drift.DriftMonitor fed every System 1 response. supervise:
    the share of System 1's answers given alone that the slow path checks (drawn by a hash of seed, the input and the
    decision's number). think: "s2" (default: the slow path's accepted answer is given) or "agree" (given only when it
    equals what System 1 would have answered; otherwise a person). on_disagree: what a check that disagrees does —
    "record" (default: System 1's answer stands, the disagreement is recorded), "human", or "s2" (the slow path's
    accepted answer replaces it). storage: a TraceStorage that keeps every decision (kind "dispatch"). asked: the
    questions System 1 is asked together (default: all of its questions, so that constraints between them apply);
    store_responses: whether System 1's and the slow path's Systems store their responses in their own storage."""

    def __init__(self, system, slow=None, *, question=None, budget=None, total=None, price=None, wake=SIGNALS,
                 agreement=None, monitor=None, supervise=0.0, seed=0, think="s2", on_disagree="record", storage=None,
                 store_responses=True, asked=None):
        if question is None:
            if len(system.questions) != 1:
                raise ValueError(f"the System asks {sorted(system.questions)}: say which with question=")
            question = next(iter(system.questions))
        if question not in system.questions:
            raise ValueError(f"no question {question!r} in System 1")
        bad = [w for w in wake if w not in SIGNALS]
        if bad:
            raise ValueError(f"unknown signal(s) {bad}; the signals are {SIGNALS}")
        if not 0.0 <= float(supervise) <= 1.0:
            raise ValueError("supervise is a share between 0 and 1")
        if think not in ("s2", "agree"):
            raise ValueError('think must be "s2" or "agree"')
        if on_disagree not in ("record", "human", "s2"):
            raise ValueError('on_disagree must be "record", "human" or "s2"')
        for b in (budget, total):
            if b is not None and not isinstance(b, Budget):
                raise TypeError("budget= and total= take a Budget(usd=, calls=, ms=)")
            if b is not None and b.usd is not None and price is None:
                raise ValueError("a budget in dollars needs price= (dollars per million input and output tokens)")
        if price is not None and not callable(price):
            if not (isinstance(price, (tuple, list)) and len(price) == 2 and all(isinstance(x, (int, float)) for x in price)):
                raise ValueError("price is (dollars per million input tokens, per million output tokens) or a function")
            price = (float(price[0]), float(price[1]))
        if slow is not None and not isinstance(slow, SlowPath):
            raise TypeError("slow= takes a SlowPath")
        if slow is not None:
            slow._q(question)
        for f in agreement or {}:
            if not isinstance(f, str):
                raise TypeError("agreement= is {fact name: minimum share}")
        if asked is None:
            asked = list(system.questions)
        asked = [asked] if isinstance(asked, str) else list(asked)
        if question not in asked:
            asked.append(question)
        self.system, self.slow, self.question, self.asked = system, slow, question, asked
        self.budget, self.total, self.price = budget, total, price
        self.wake = tuple(wake)
        self.agreement = {str(k): float(v) for k, v in (agreement or {}).items()}
        self._require(list(self.agreement))
        self.monitor, self.supervise, self.seed = monitor, float(supervise), int(seed)
        self.think_policy, self.on_disagree = think, on_disagree
        from .storage import open_storage
        self.storage = open_storage(storage)
        self.store_responses = bool(store_responses)
        self.n = 0
        self.spent = Cost()
        self.runs = []                    # the slow path's costs so far (its expected cost is their mean)
        self.counts = {p: 0 for p in PATHS}
        self.drift = None
        self._lock = threading.Lock()

    def _require(self, facts):
        """Facts the dispatch reads (agreement shares) that the catalog computes become required parts of the
        question, so every flow computes them (as System.guarantee does for its signal)."""
        import dataclasses
        q = self.system.questions[self.question]
        add = [f for f in facts if f in self.system.catalog.parts and f not in q.requires]
        if add:
            self.system.questions[self.question] = dataclasses.replace(q, requires=list(q.requires) + add)

    def __repr__(self):
        return f"Dispatcher({self.question!r}, slow={self.slow!r}, answered {self.counts})"

    def config(self):
        """The fingerprint of everything a dispatch depends on besides the input (recorded with every decision)."""
        from .provenance import code_fingerprint, digest
        pr = code_fingerprint(self.price) if callable(self.price) else self.price
        return digest("Dispatcher", self.question, self.wake, sorted(self.agreement.items()), self.supervise, self.seed,
                      self.think_policy, self.on_disagree, None if self.budget is None else self.budget.to_dict(),
                      None if self.total is None else self.total.to_dict(), pr,
                      self.slow.fingerprint() if self.slow is not None else None)

    # --- the decision
    def draw(self, init_hash, n):
        """The supervision draw of decision n on this input: a number in [0, 1) from a hash (reproducible)."""
        h = hashlib.sha256(f"{self.seed}|{init_hash}|{n}".encode()).hexdigest()
        return int(h[:13], 16) / 16 ** 13

    def expected(self):
        """The expected Cost of one slow-path run: the mean of the runs so far (None before the first)."""
        if not self.runs:
            return None
        k = len(self.runs)
        usd = None if any(c.usd is None for c in self.runs) else sum(c.usd for c in self.runs) / k
        return Cost(usd, round(sum(c.calls for c in self.runs) / k), sum(c.ms for c in self.runs) / k)

    def signals(self, res):
        """System 1's response → [(signal, reason)] that are up for it (before `wake` filters them)."""
        q = self.question
        r = res[q]
        out = []
        if r.status == "forced":
            return out                                 # a hard check decided: nothing to re-think
        g = (r.extra or {}).get("guarantee") or {}
        if r.status == "abstain":
            if g and not g.get("answered", True):
                kind = "openset" if g.get("method") == "open-set" else "guarantee"
                out.append((kind, f"System 1 is below its guarantee: {r.why}"))
            elif r.guard == "low_confidence":         # Question(min_confidence=) or a decider's own threshold
                out.append(("guarantee", f"System 1 is below its threshold: {r.why}"))
            else:
                out.append(("abstain", f"System 1 abstained: {r.why}"))
        if not res.feasible:
            out.append(("constraint", f"the answers break {', '.join(res.violations or ['a constraint'])}"))
        for f, lo in self.agreement.items():
            v = res.values.get(f)
            v = getattr(v, "value", v)
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v < lo:
                out.append(("agreement", f"{f} = {v:.3g} < {lo:.3g}"))
            elif not isinstance(v, (int, float)) or isinstance(v, bool):
                out.append(("agreement", f"the agreement {f} was not computed"))
        state = g.get("state") or {}
        if state.get("flag_at") is not None:
            out.append(("drift", f"the open-set gate flagged a change of the stream at decision {state['flag_at']}: "
                                  f"{state.get('why')}"))
        return out

    def decide(self, res, draw, spent_before, drift, expected=None):
        """The dispatch of one System 1 response → (action, reasons). Pure: the same response, draw, spend, drift
        state and expected cost give the same dispatch (replay re-derives it)."""
        sig = self.signals(res)
        if drift and not any(s == "drift" for s, _ in sig):
            sig.append(("drift", drift))
        reasons = []
        think = [(s, w) for s, w in sig if s in THINK]
        check = [(s, w) for s, w in sig if s == "drift"]
        if not think and res[self.question].status == "ok" and draw < self.supervise:
            check.append(("supervise", f"sampled for supervision (draw {draw:.4f} < {self.supervise:g})"))
        if think:
            reasons += [w for _, w in think]
            if not all(s in self.wake for s, _ in think):
                off = sorted({s for s, _ in think if s not in self.wake})
                return "human", reasons + [f"{', '.join(off)} does not wake the slow path: a person decides"]
            if self.slow is None:
                return "human", reasons + ["no slow path: a person decides"]
            why = self._no_budget(spent_before, expected)
            if why:
                return "human", reasons + [why]
            return "think", reasons
        check = [(s, w) for s, w in check if s in self.wake]
        if check and self.slow is not None:
            why = self._no_budget(spent_before, expected)
            if why:
                return "accept", [w for _, w in check] + [why + "; System 1's answer stands unchecked"]
            return "check", [w for _, w in check]
        return "accept", ["System 1 answered within its guarantee" if (res[self.question].extra or {}).get("guarantee")
                          else ("a hard check decided" if res[self.question].status == "forced"
                                else "System 1 answered")]

    def _no_budget(self, spent, expected):
        if self.total is not None:
            why = self.total.used_up(spent) or (self.total.over(spent, expected) if expected is not None else None)
            if why:
                return f"no budget left in total ({why})"
        if self.budget is not None and expected is not None:
            why = self.budget.over(Cost(), expected)
            if why:
                return f"one run of the slow path is expected to cost more than the budget per decision ({why})"
        return None

    # --- asking
    def ask(self, state):
        """One input → a Dispatched (see the class docs)."""
        q = self.question
        res = self.system.ask(state, self.asked, store=self.store_responses)
        if self.monitor is not None:
            rep = self.monitor.observe(res, question=q)
            if rep.get("drift") and self.drift is None:
                self.drift = "DriftMonitor: " + "; ".join(rep.get("why") or ["drift"])
        with self._lock:
            self.n += 1
            n = self.n
            spent = Cost(**vars(self.spent))
        draw = self.draw(res.trace.init_hash, n)
        expected = self.expected()
        action, reasons = self.decide(res, draw, spent, self.drift, expected)
        r1 = res[q]
        would1 = r1.answer if r1.status != "abstain" else _would(r1)
        cand = {"s1": would1}
        c1 = cost_of([res], self.price, res.ms)
        th, over, disagreement = None, None, None
        if action in ("think", "check"):
            per_round = None if expected is None else _per_round(expected, self.slow)
            th = self.slow.run(state, q, price=self.price, budget=self.budget, expected_round=per_round,
                               store=self.store_responses)
            cand["s2"] = th.answer
            if self.budget is not None:
                over = self.budget.over(th.cost)
            with self._lock:
                self.runs.append(th.cost)
        answer, by = self._outcome(action, r1, would1, th, reasons)
        if th is not None and action == "check" and _vh(th.answer) != _vh(would1):
            disagreement = {"s1": would1, "s2": th.answer, "s2_accepted": th.accepted}
        c2 = th.cost if th is not None else Cost(0.0 if self.price is not None else None)
        cost = {"s1": c1, "s2": c2, "total": c1 + c2}
        with self._lock:
            self.spent = self.spent + c2
            self.counts[by] += 1
        d = Dispatched(q, answer, by, action, reasons, res, th, cand, disagreement, cost, n, draw, spent, self.drift,
                       expected, over, self.config())
        if self.storage is not None:
            from .storage import FORMAT
            rec = self.storage._append({"v": FORMAT, "kind": "dispatch", **json.loads(d.to_json())})
            d.stored_id = rec["id"]
        return d

    def _outcome(self, action, r1, would1, th, reasons):
        """(answer, by) from the action and the two paths' answers; appends why a person gets it."""
        if action == "human":
            return None, "human"
        if action == "accept":
            return r1.answer, "s1"
        if action == "think":
            if not th.accepted:
                reasons.append(f"the slow path did not answer: {th.why or th.stopped}")
                return None, "human"
            if self.think_policy == "agree" and _vh(th.answer) != _vh(would1):
                reasons.append(f"the slow path answered {th.answer!r}, System 1 would have answered {would1!r}: a "
                               "person decides")
                return None, "human"
            return th.answer, "s2"
        # check
        if _vh(th.answer) == _vh(would1):
            return r1.answer, "s1"
        reasons.append(f"the slow path says {th.answer!r}" + ("" if th.accepted else " (not accepted)")
                       + f", System 1 {would1!r}")
        if self.on_disagree == "human":
            return None, "human"
        if self.on_disagree == "s2" and th.accepted:
            return th.answer, "s2"
        return r1.answer, "s1"

    def reset_drift(self):
        """Forget the drift flag (after the stream was looked at, System 1 recalibrated, ...)."""
        self.drift = None

    # --- what happened
    def disagreements(self, stored=None):
        """The decisions where the slow path and System 1 differed (checks) or the slow path answered for System 1
        (think) → [{"n", "init", "s1", "s2", "accepted", "action", "by", "stored_id"}] — the material for teaching System 1
        (not used here). stored: Dispatched decisions (default: the dispatcher's store)."""
        out = []
        for d in stored if stored is not None else self.stored():
            if d.s2 is None:
                continue
            if d.action == "check" and d.disagreement is None:
                continue
            out.append({"n": d.n, "init": d.s1.trace.init, "s1": d.candidates.get("s1"), "s2": d.s2.answer,
                        "accepted": d.s2.accepted, "action": d.action, "by": d.by, "stored_id": d.stored_id})
        return out

    def stored(self):
        """The stored decisions, in order → [Dispatched] (needs storage=)."""
        if self.storage is None:
            raise ValueError("the dispatcher has no storage: Dispatcher(..., storage=TraceStorage)")
        slow_sys = self.slow.system if self.slow is not None else None
        out = []
        for s in self.storage.iter(kind="dispatch"):
            d = Dispatched.from_dict(s.data, self.system, slow_sys)
            d.stored_id = s.id
            out.append(d)
        return out

    def replay(self, d, trust_models=False):
        """Re-check a decision without calling a model → {"ok", "mismatches": [(what, why)], "s1": the trace replay,
        "s2": the slow path's replay}. Checked: the dispatcher is configured as it was (config); System 1's trace
        replays; the draw recomputes; the dispatch follows from the response, the draw and the recorded spend, drift
        flag and expected cost of a run; the slow path's record replays and its cost recomputes; the answer and who
        gave it follow."""
        bad = []
        if d.config is not None and d.config != self.config():
            bad.append(("config", "the dispatcher is not configured as it was for this decision"))
        rep1 = d.s1.trace.replay(self.system, trust_models=trust_models)
        bad += [("s1", f"{m[1]}: {m[2]}") for m in rep1["mismatches"]]
        if d.s1._system is None:
            d.s1._system = self.system
        draw = self.draw(d.s1.trace.init_hash, d.n)
        if abs(draw - d.draw) > 1e-12:
            bad.append(("draw", f"recorded {d.draw}, recomputed {draw}"))
        action, reasons = self.decide(d.s1, d.draw, d.spent_before, d.drift, d.expected)
        if action != d.action:
            bad.append(("dispatch", f"recorded {d.action}, the response gives {action}"))
        elif reasons != d.reasons[:len(reasons)]:
            bad.append(("reasons", "the recorded reasons are not those the response gives"))
        rep2 = None
        if d.s2 is not None:
            if self.slow is None:
                bad.append(("s2", "a slow-path record, and the dispatcher has no slow path"))
            else:
                rep2 = self.slow.replay(d.s2, trust_models=trust_models)
                bad += [("s2 " + w, why) for w, why in rep2["mismatches"]]
                c = cost_of(d.s2.responses, self.price, d.s2.cost.ms, d.s2.generated())
                if _cost_key(c) != _cost_key(d.s2.cost):
                    bad.append(("cost", f"the slow path's recorded cost {d.s2.cost.to_dict()} is not what its record "
                                        f"gives {c.to_dict()}"))
        elif d.action in ("think", "check"):
            bad.append(("s2", f"action {d.action} without the slow path's record"))
        c1 = cost_of([d.s1], self.price, d.s1.ms)
        if "s1" in d.cost and _cost_key(c1) != _cost_key(d.cost["s1"]):
            bad.append(("cost", "System 1's recorded cost is not what its trace gives"))
        if d.action == action:
            r1 = d.s1[self.question]
            would1 = r1.answer if r1.status != "abstain" else _would(r1)
            ans, by = self._outcome(d.action, r1, would1, d.s2, []) if d.s2 is not None or d.action in ("accept", "human") \
                else (None, "human")
            if by != d.by or _vh(ans) != _vh(d.answer):
                bad.append(("answer", f"recorded {d.answer!r} by {d.by}, the record gives {ans!r} by {by}"))
        return {"ok": not bad, "mismatches": bad, "s1": rep1, "s2": rep2}

    def replay_all(self, trust_models=False):
        """Every stored decision replayed → [(n, stored_id, mismatches)] — empty when all replay."""
        out = []
        for d in self.stored():
            rep = self.replay(d, trust_models=trust_models)
            if not rep["ok"]:
                out.append((d.n, d.stored_id, rep["mismatches"]))
        return out

    def summary(self):
        """{"decisions", "by": {path: count}, "spent": Cost dict, "slow_runs"}."""
        return {"decisions": self.n, "by": dict(self.counts), "spent": self.spent.to_dict(), "slow_runs": len(self.runs)}


def _per_round(expected, slow):
    """The expected cost of one more round of a refinement: the mean run's cost spread over its rounds."""
    k = max(1, slow.rounds if slow.mode == "refine" else 1)
    return Cost(None if expected.usd is None else expected.usd / k, int(math.ceil(expected.calls / k)), expected.ms / k)


def _cost_key(c):
    return (None if c.usd is None else round(c.usd, 9), c.calls, c.input_tokens, c.output_tokens)


def _would(r):
    """What an abstained Result would have answered: its answer, else the most probable option (None when unknown)."""
    if r is None:
        return None
    if r.answer is not None:
        return r.answer
    p = r.probs or {}
    if p:
        best = max(p, key=lambda k: p[k])
        return best
    return None


def _vh(v):
    from .runtime import vhash
    from .storage import plain
    return vhash(plain(v))


__all__ = ["Budget", "BudgetStop", "Cost", "Dispatched", "Dispatcher", "PATHS", "SIGNALS", "SlowPath", "Thought",
           "cost_of", "price_of", "recorded_calls"]
