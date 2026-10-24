"""Who answers: the fast system within its guarantee, a slow path when the fast one is unsure, or a person — one recorded
decision per input, with a budget.

    from solvi.core.dispatch import AskPath, Budget, Dispatcher

    slow = AskPath(system2)                        # a System that answers the same question slowly (an LLM part, ...)
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
person instead of the slow path. After a drift flag System 1's guarantee no longer covers its answers: when no slow
path checks them they go to a person (`on_drift="accept"`: they stand, marked out of the guarantee), and none is
reported "within its guarantee" until `reset_drift()`.

The slow path (a `SlowPath`) is built from the existing parts and judged by a System's checks:
  - `AskPath(system)`: a System that answers the question itself — an LLM decision part (`solvi.core.deciders.llm`),
    generated candidates with `solvi.core.slow.generate` and `solvi.core.slow.agree` in its catalog, typed facts with
    quotes — with its own guarantee;
  - `RefinePath(system, propose=, into=)` (a `Generator.proposer`, any Proposer): `solvi.core.slow.refine` — propose,
    check with the System's hard checks, re-ask with the reasons of the failed checks, within the budget;
  - `SearchPath(system, space=, into=)`: `solvi.core.slow.search` — the candidates of an enumerable space through the
    checks;
  - a subclass of your own (`mode`, `think`, `fingerprint`; see SlowPath).
Its answer is accepted when the System does not abstain on it (and, with refine and search, its checks accept it).

Calibrating the hard slice. The inputs System 1 hands over are the hard ones, and a slow path's accuracy (or its own
guarantee) calibrated on an average sample does not carry over to them: an LLM rarely wrong on product pairs overall can
be wrong on many of those a fitted head is unsure of. `calibrate(examples, max_risk= | max_error=)` measures it
where it matters: every labelled example goes through the same dispatch; the ones System 1 answers alone count as its
answers, the ones it hands over form a slice per waking signal (below the guarantee, open-set, abstention, constraint,
agreement), and on each slice System 1's own would-be answer, the slow path's answer and "the slow path when it agrees
with System 1" are measured. One answerer and one threshold on its confidence is chosen per slice — or a person — so
that all the answers given alone together (System 1's and the slices') keep the promise on these examples, answering
as many as possible; a slice with fewer than `min_slice` examples gets a person. The choice (`policy`, with each
slice's size and measured accuracies) is part of the config every decision records; a think then runs the slow path
only when its slice's answerer is the slow path. Use examples System 1's own guarantee was not calibrated on: there its
answers are tuned to the edge of the promise and leave nothing for the slices.

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
follows from that response, the recorded supervision draw (a hash of the seed, the input and the decision's number;
an HMAC with `secret=`, recomputed on replay only where the secret is given), the recorded budget and drift state; the slow path's record replays (the LLM outputs re-read through their schemas, as
`solvi.core.slow.generate` and `solvi.core.deciders.llm` replay them); the cost recomputes; the answer follows. With `storage=` every decision
is one hash-chained record of kind "dispatch" in a TraceStorage (`stored()`, `replay_all()`), and `calibrate` adds one
of kind "policy" (the policy and the config fingerprint the later decisions record), so that a report read from the
store alone (`solvi.core.store.sysreport`) knows the promise each decision was made under.

Not done here: the slow path does not teach System 1 (the disagreements and the slow path's accepted answers are
recorded as material, `disagreements()`, not used); several questions at once (one dispatcher per question); parallel
asks (the budget, the drift state and the decision numbers follow the order of the asks); a drift flag is latched until
`reset_drift()` and is taken as recorded on replay, since it depends on the stream before the decision."""
from __future__ import annotations

import abc
import contextlib
import copy
import dataclasses
import hashlib
import hmac
import json
import math
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar

# defined in solvi.core.costs since 1.0 (the system report, solvi.core.slow.generate and solvi.core.slow.refine use them too); re-exported here
from .costs import Budget, BudgetStop, Cost, _usages, cost_of, price_of, recorded_calls   # noqa: F401

SIGNALS = ("guarantee", "openset", "abstain", "constraint", "agreement", "drift", "supervise")
THINK = ("guarantee", "openset", "abstain", "constraint", "agreement")
PATHS = ("s1", "s2", "human")
SAMPLE_REASON = "sampled for the stream certificate (calibration coin < ρ)"
PROBATION_REASON = "probation after a recertification: System 1 does not answer alone"
SECRET_ENV = "SOLVI_AUDIT_SECRET"   # noqa: S105 — the name of the variable read for Dispatcher(secret=None)


class PredictableDrawWarning(UserWarning):
    """A Dispatcher samples decisions for supervision or audit without a secret: the draw is a public hash of the seed,
    the input and the decision's number, so whoever can vary the inputs can predict which decisions are checked and
    steer around them. Give `secret=` (or set SOLVI_AUDIT_SECRET); filter this class where the draw need not be
    hidden (tests, offline backtests)."""


_predictable_warned = False          # the warning above is given once per process


class _Secret:
    """The draw key: kept out of repr, the config fingerprint, the records and the logs."""
    __slots__ = ("_key",)

    def __init__(self, key: bytes):
        self._key = key

    def mac(self, msg: str) -> str:
        return hmac.new(self._key, msg.encode(), hashlib.sha256).hexdigest()

    def __repr__(self):
        return "<secret>"


# ------------------------------------------------------------------------------------------------ the slow path
@dataclass
class Thought:
    """What the slow path did for one input: its mode ("ask", "refine", "search", or a SlowPath subclass's own), its
    answer (None when it has none), whether that answer was accepted, why not, its record (a Response, a Refinement, a
    SearchRun — replayable — or a subclass's own record) and its cost.

    The mode names the SlowPath class that made it (`SlowPath.modes`, filled as subclasses are defined): a stored
    Thought is read back by that class (`restore_record`), and its responses and generator records are found by it
    (`responses_of`, `generated_of`) — so a Dispatched decision of a path of your own round-trips through a store as
    the built-ins' do. Stability: stable (the record format too)."""
    mode: str
    answer: Any = None
    accepted: bool = False
    why: str | None = None
    record: Any = None
    cost: Cost = field(default_factory=Cost)
    stopped: str | None = None           # the budget stopped it between steps
    contract: dict | None = None         # what the slow path's contract did (solvi.core.contract): {"applicable": False,
    #                                      "why"} — not called; {"applicable": True, "violations": [...]} — not accepted

    @property
    def responses(self):
        """The Responses the slow path's System gave (each a replayable trace; their model outputs are its cost)."""
        if self.record is None:
            return []
        return list(SlowPath.path_of(self.mode).responses_of(self.record))

    def generated(self):
        """The generator records kept outside the traces (a refinement's proposals)."""
        if self.record is None:
            return []
        return list(SlowPath.path_of(self.mode).generated_of(self.record))

    def to_dict(self):
        from .store import plain
        rec = self.record
        if rec is not None:
            rec = rec.to_dict() if callable(getattr(rec, "to_dict", None)) else plain(rec)
        d = {"mode": self.mode, "answer": plain(self.answer), "accepted": self.accepted, "why": self.why,
             "record": rec, "cost": self.cost.to_dict(), "stopped": self.stopped}
        if self.contract is not None:                 # a record without a contract is written as before
            d["contract"] = plain(self.contract)
        return d

    @classmethod
    def from_dict(cls, d, system=None):
        """A stored Thought back; its record restored by the SlowPath class of its mode (with `system`'s types)."""
        rec = d.get("record")
        if rec is not None:
            rec = SlowPath.path_of(d["mode"]).restore_record(rec, system)
        return cls(d["mode"], _restored(d.get("answer")), d["accepted"], d.get("why"), rec,
                   Cost.from_dict(d.get("cost") or {}), d.get("stopped"), d.get("contract"))


class _SlowPathType(abc.ABCMeta):
    """SlowPath(...) itself is the 0.9 factory (kept for 1.0.x, with a warning); a subclass is constructed as usual and
    must name its mode."""

    def __call__(cls, *args, **kw):
        if cls is SlowPath:
            from .._deprecate import _warn_from_caller
            _warn_from_caller("SlowPath(system, ...) as a constructor is deprecated in 1.0: SlowPath is the base class "
                              "of the slow paths — use AskPath(system), RefinePath(system, propose=, into=) or "
                              "SearchPath(system, space=, into=); the factory is removed in 1.1")
            return _slow_path(*args, **kw)
        obj = super().__call__(*args, **kw)
        if not isinstance(getattr(type(obj), "mode", None), str):
            raise TypeError(f"{cls.__qualname__} is a SlowPath without a mode: set the class attribute mode = \"...\" "
                            "(a name of its own; its Thoughts are read back by it)")
        return obj


class SlowPath(metaclass=_SlowPathType):
    """System 2: a slow, checked way to answer the question — the base class of the slow paths (see the module docs).

    The built-ins: AskPath (a System that answers the question itself), RefinePath (propose → check → re-ask with the
    reasons, solvi.core.slow.refine) and SearchPath (the candidates of a space through the checks,
    solvi.core.slow.search).

    You implement (a subclass): `mode` — a class attribute, a name of its own (its Thoughts are read back by it);
    `think(state, question, *, price, budget, expected_round, store)` → a Thought of that mode (the answer, accepted or
    not and why, the record); `fingerprint()` → everything the path's answers depend on (its System's fingerprint,
    its settings). Override when you have more: `replay(thought, trust_models=False)` (re-derive the answer and the
    acceptance from the record; the default checks the mode and replays a record that has `replay(system,
    trust_models=)`), `signal(thought, question)` (the number a calibrated threshold is put on: default 1 for an
    accepted answer), `restore_record(d, system)` / `responses_of(record)` / `generated_of(record)` (classmethods:
    how a stored record is read back, which Responses and generator records it holds — for the cost), `steps` (the
    rounds of one run, for the expected cost of one more round).

    You get for free (`run`, `Dispatcher`): routing by System 1's signals, a budget per decision and in total checked
    before the path starts (and between the rounds of a refinement), the cost in dollars, calls and tokens from the
    model outputs its Responses record (recomputed on replay), `Dispatcher.calibrate` choosing per slice whether its
    answers are taken and above what signal, the dispatch record in a store, replay of every decision, the system
    report.

    Stability: stable to use; provisional to subclass in 1.0 (`think` may gain keyword arguments; take **kw).
    `SlowPath(system, propose=..., space=...)` — the 0.9 constructor — still builds the matching built-in for 1.0.x,
    with a SolviDeprecationWarning; removed in 1.1."""

    mode: ClassVar[str]                  # a subclass's name for its Thoughts ("ask", "refine", "search", yours)
    modes: ClassVar[dict] = {}           # mode → the SlowPath class whose Thoughts have it (Thought.from_dict reads it)
    steps = 1                            # rounds of one run (a refinement's rounds): the expected cost of one more

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        m = cls.__dict__.get("mode")
        if m is None:
            return
        if not isinstance(m, str) or not m:
            raise TypeError(f"{cls.__qualname__}.mode is a name (a non-empty str), not {m!r}")
        old = SlowPath.modes.get(m)
        if old is not None and (old.__module__, old.__qualname__) != (cls.__module__, cls.__qualname__):
            raise ValueError(f"the slow-path mode {m!r} is {old.__module__}.{old.__qualname__}'s: give "
                             f"{cls.__qualname__} a mode of its own")
        SlowPath.modes[m] = cls

    @classmethod
    def path_of(cls, mode):
        """The SlowPath class whose Thoughts have this mode (KeyError when no class of that mode is defined: import the
        module that defines it before reading its decisions)."""
        p = SlowPath.modes.get(mode)
        if p is None:
            raise KeyError(f"no slow path of mode {mode!r} is defined (known: {', '.join(sorted(SlowPath.modes))}): "
                           "import the module that defines it before reading its records")
        return p

    def __init__(self, system, *, question=None, contract=None):
        from .contract import Contract
        if contract is not None and not isinstance(contract, Contract):
            raise TypeError("contract= takes a solvi.core.contract.Contract")
        self.system, self.question, self.contract = system, question, contract

    def __repr__(self):
        return f"{type(self).__name__}({self.mode})"

    @abc.abstractmethod
    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        """One input → a Thought of this path's mode (its cost is filled in by run). question: the question asked of
        the path's System (checked by run). budget: the per-decision Budget — a path that takes several steps checks
        it between them (raise or stop with Thought.stopped); expected_round: the expected Cost of one more step."""

    @abc.abstractmethod
    def fingerprint(self):
        """What this path's answers depend on (recorded in the dispatcher's config with every decision)."""

    def run(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        """Think about one input → a Thought, with its cost: the model outputs its Responses and generator records
        hold, at `price`, and the time it took (the cost a replay recomputes)."""
        q = self._q(question)
        t0 = time.perf_counter()
        c = self.contract_for(q)
        if c is not None:                             # checked before the path is called: called by applicability
            ok, why = c.applies(state)
            if not ok:
                th = Thought(self.mode, None, False, f"not applicable: {why}", None,
                             contract={"applicable": False, "why": why})
                th.cost = cost_of([], price, (time.perf_counter() - t0) * 1000, [])
                return th
        th = self.think(state, q, price=price, budget=budget, expected_round=expected_round, store=store)
        if not isinstance(th, Thought):
            raise TypeError(f"{type(self).__qualname__}.think returned {type(th).__name__}, not a Thought")
        if th.mode != self.mode:
            raise ValueError(f"{type(self).__qualname__}.think returned a Thought of mode {th.mode!r}, not its own "
                             f"{self.mode!r}")
        if c is not None and th.accepted:             # a hard refusal: whatever the path's confidence
            v = c.violations(th.answer, c.echo_inputs(state), _answer_type(self, q), echo_type=_echo_type(self, q))
            if v:
                from .contract import refusal
                th.accepted, th.why = False, refusal(v, th.answer)
                th.contract = {"applicable": True, "violations": [x.to_dict() for x in v]}
        th.cost = cost_of(th.responses, price, (time.perf_counter() - t0) * 1000, th.generated())
        return th

    def contract_for(self, question):
        """The contract the slow path's answers are held to: its own (`contract=`), else the contract its System's
        catalog declares for the question (`Catalog.contract`), else None."""
        if getattr(self, "contract", None) is not None:
            return self.contract
        return (getattr(self.system.catalog, "contracts", None) or {}).get(self.question or question)

    def _q(self, question):
        q = self.question or question
        if q not in self.system.questions:
            raise ValueError(f"the slow path's System has no question {q!r} (question= names it)")
        return q

    def replay(self, thought, trust_models=False):
        """Re-check a recorded Thought without calling a model → {"ok", "mismatches": [(what, why)]}. The default: the
        mode is this path's, an accepted answer has its record, and a record with `replay(system, trust_models=)`
        replays under this path's System. Override to re-derive the answer and its acceptance from the record (the
        built-ins do)."""
        bad = []
        rec = thought.record
        if thought.mode != self.mode:
            return {"ok": False, "mismatches": [("mode", f"recorded {thought.mode}, the slow path is {self.mode}")]}
        if rec is None:
            if thought.accepted:
                bad.append(("record", "an accepted answer without its record"))
            return {"ok": not bad, "mismatches": bad}
        rp = getattr(rec, "replay", None)
        if callable(rp):
            rep = rp(self.system, trust_models=trust_models)
            bad += [("record", "; ".join(map(str, m)) if isinstance(m, (tuple, list)) else str(m))
                    for m in (rep or {}).get("mismatches", [])]
        return {"ok": not bad, "mismatches": bad}

    def signal(self, thought, question):
        """The slow path's signal for its answer, what Dispatcher.calibrate thresholds: 1 for an accepted answer, −inf
        otherwise (AskPath: its answer's confidence)."""
        return 1.0 if thought.accepted else -math.inf

    @classmethod
    def restore_record(cls, d, system=None):
        """A stored record (a dict) → the record (default: the dict itself)."""
        return d

    @classmethod
    def responses_of(cls, record):
        """The Responses a record holds (default: its `responses`, if it has them)."""
        r = getattr(record, "responses", None)
        return list(r() if callable(r) else r or [])

    @classmethod
    def generated_of(cls, record):
        """The generator records a record keeps outside its traces (default: none)."""
        return []


class _Checked(SlowPath):
    """The built-in paths' shared settings and fingerprint (as 0.9's single SlowPath class computed it)."""

    def __init__(self, system, *, question=None, propose=None, into=None, rounds=3, space=None, search=None,
                 accept="checks", feedback=None, contract=None):
        if int(rounds) < 1:
            raise ValueError("rounds must be at least 1")
        super().__init__(system, question=question, contract=contract)
        self.propose, self.into = propose, into
        self.rounds, self.space, self.search_kw = int(rounds), space, dict(search or {})
        self.accept, self.feedback = accept, feedback

    def fingerprint(self):
        from .provenance import code_fingerprint, digest
        acc = code_fingerprint(self.accept) if callable(self.accept) else repr(self.accept)
        return digest("SlowPath", self.mode, self.system.fingerprint(), self.question, self.into, self.rounds, acc,
                      sorted(self.search_kw), code_fingerprint(self.propose) if self.propose is not None else None)


class AskPath(_Checked):
    """The slow path that asks a System answering the question itself — an LLM decision part, generated candidates
    with agreement (solvi.core.slow.generate / agree), typed facts with quotes — with its own guarantee. Its answer is
    accepted when the System does not abstain on it; its signal is the answer's confidence. Stability: stable."""
    mode = "ask"

    def __init__(self, system, *, question=None, contract=None):
        super().__init__(system, question=question, contract=contract)

    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        res = self.system.ask(dict(state), [question], store=store)
        r = res[question]
        ok = r.status != "abstain"
        return Thought("ask", r.answer if ok else _would(r, res, question), ok,
                       None if ok else f"the slow path abstained: {r.why}", res)

    def replay(self, thought, trust_models=False):
        bad = []
        rec = thought.record
        if thought.mode != self.mode:
            return {"ok": False, "mismatches": [("mode", f"recorded {thought.mode}, the slow path is {self.mode}")]}
        if rec is None:
            return SlowPath.replay(self, thought, trust_models)
        rep = rec.trace.replay(self.system, trust_models=trust_models)
        bad += [("trace", f"{m[1]}: {m[2]}") for m in rep["mismatches"]]
        qn = self.question or next(iter(rec.results))
        r = rec[qn]
        ok = r.status != "abstain"
        return _same_outcome(thought, ok, r.answer if ok else _would(r, rec, qn), bad)

    def signal(self, thought, question):
        res = thought.record
        if res is None:
            return -math.inf
        r = res[self._q(question)]
        if r.status == "abstain" and r.guard not in ("low_confidence", None):
            return -math.inf
        return float(r.confidence)

    @classmethod
    def restore_record(cls, d, system=None):
        from .response import Response
        return Response.model_validate(d, catalog=system)

    @classmethod
    def responses_of(cls, record):
        return [record]


class RefinePath(_Checked):
    """The slow path that refines proposals (solvi.core.slow.refine): `propose(state, rounds)` (a Proposer, e.g.
    `Generator.proposer(...)`) gives a proposal, the System gets it as the fact `into`, its hard checks judge it, and
    the reasons of the failed checks are fed back, up to `rounds` times; the budget is checked between the rounds.
    accept: what accepts a proposal ("checks", an answer or answers, a function of the Response); the System must also
    not abstain on the question. feedback: refine's feedback function. Stability: stable."""
    mode = "refine"

    def __init__(self, system, *, propose, into, question=None, rounds=3, accept="checks", feedback=None, contract=None):
        if not callable(propose):
            raise TypeError("propose is a function (state, rounds) → a proposal, e.g. Generator.proposer(...)")
        if not isinstance(into, str):
            raise ValueError("into= names the given fact a proposal or a candidate is passed as")
        super().__init__(system, question=question, propose=propose, into=into, rounds=rounds, accept=accept,
                         feedback=feedback, contract=contract)

    @property
    def steps(self):
        return self.rounds

    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        from .slow.refine import refine
        inner = self.propose
        assert inner is not None
        t0 = time.perf_counter()

        def propose(st, rounds):
            if budget is not None and rounds:
                so_far = cost_of([x.response for x in rounds], price, (time.perf_counter() - t0) * 1000,
                                 [x.generated for x in rounds if x.generated is not None])
                why = budget.over(so_far, expected_round)
                if why:
                    raise BudgetStop(f"no budget for another round ({why})")
            return inner(st, rounds)
        run = refine(self.system, dict(state), question, propose, into=self.into, rounds=self.rounds,
                     accept=self.accept, feedback=self.feedback, store=store)
        r = run.result
        ok = bool(run.accepted) and r is not None and r.status != "abstain"
        stopped = None
        last = run.rounds[-1] if run.rounds else None
        if last is not None and last.error and "BudgetStop" in last.error:
            stopped = last.error
        why = None if ok else (run.escalation or (f"the slow path abstained: {r.why}" if r is not None else "no answer"))
        ans = (r.answer if ok else _would(r, run.response, question)) if r is not None else None
        return Thought("refine", ans, ok, why, run, stopped=stopped)

    def replay(self, thought, trust_models=False):
        rec = thought.record
        if thought.mode != self.mode or rec is None:
            return SlowPath.replay(self, thought, trust_models)
        rep = rec.replay(self.system, accept=None if not callable(self.accept) else self.accept,
                         trust_models=trust_models)
        bad = [("refine", f"round {m[0]} {m[1]}: {m[2]}") for m in rep["mismatches"]]
        r = rec.result
        ok = bool(rec.accepted) and r is not None and r.status != "abstain"
        ans = (r.answer if ok else _would(r, rec.response, rec.question)) if r is not None else None
        return _same_outcome(thought, ok, ans, bad)

    @classmethod
    def restore_record(cls, d, system=None):
        from .slow.refine import Refinement
        return Refinement.from_dict(d, catalog=system)

    @classmethod
    def responses_of(cls, record):
        return [x.response for x in record.rounds if x.response is not None]

    @classmethod
    def generated_of(cls, record):
        return [x.generated for x in record.rounds if x.generated is not None]


class SearchPath(_Checked):
    """The slow path that searches a space (solvi.core.slow.search): the candidates of `space` (a list, a dict of
    domains, a Tree or any Space, or a function of the facts) are given to the System as the fact `into` and run
    through its checks; `search=` takes search's other options (objective, prune, keep, budget). accept: as in
    RefinePath. Stability: stable."""
    mode = "search"

    def __init__(self, system, *, space, into=None, question=None, search=None, accept="checks", contract=None):
        if space is None:
            raise ValueError("space= is the space searched (a list, a dict of domains, a Tree, a function of the facts)")
        if not isinstance(into, str) and not isinstance(space, dict):
            raise ValueError("into= names the given fact a proposal or a candidate is passed as")
        super().__init__(system, question=question, space=space, into=into, search=search, accept=accept,
                         contract=contract)

    def think(self, state, question, *, price=None, budget=None, expected_round=None, store=True):
        from .slow.search import search
        run = search(self.system, dict(state), question, self.space, into=self.into, accept=self.accept, store=store,
                     **self.search_kw)
        res = run.response
        r = res[question] if res is not None else None
        ok = run.best is not None and r is not None and r.status != "abstain"
        why = None if ok else (run.escalation or "no candidate of the space was accepted")
        return Thought("search", r.answer if ok else None, ok, why, run)

    def replay(self, thought, trust_models=False):
        rec = thought.record
        if thought.mode != self.mode or rec is None:
            return SlowPath.replay(self, thought, trust_models)
        rep = rec.replay(self.system, accept=self.accept, objective=self.search_kw.get("objective"),
                         trust_models=trust_models)
        bad = [("search", f"{m[0]}: {m[1]}") for m in rep["mismatches"]] if rec.response is not None else []
        r = rec.response[rec.question] if rec.response is not None else None
        ok = rec.best is not None and r is not None and r.status != "abstain"
        return _same_outcome(thought, ok, r.answer if ok else None, bad)

    @classmethod
    def restore_record(cls, d, system=None):
        from .slow.search import SearchRun
        return SearchRun.from_dict(d, catalog=system)

    @classmethod
    def responses_of(cls, record):
        return [record.response] if record.response is not None else []


def _answer_type(slow, question):
    """The question's answer type, for the contract's type clause (None for a primitive answer: a span or an estimate
    the System has already resolved)."""
    q = slow.system.questions.get(slow.question or question)
    a = getattr(q, "answer", None)
    return None if a is None or a.primitive and a.kind != "estimate" else a


def _echo_type(slow, question):
    """The question's answer type, for the echo default of a contract (free text is checked, a choice is not)."""
    return getattr(slow.system.questions.get(slow.question or question), "answer", None)


def _replay_thought(slow, th, state, question, trust_models=False):
    """A recorded Thought replayed with its contract: a "not applicable" must still not apply (and nothing ran); a
    path that ran must still apply; the record's own replay with the acceptance the path gave before the contract;
    the violations the contract gives now must be the recorded ones."""
    c = slow.contract_for(question)
    rec, bad = th.contract, []
    if rec is not None and rec.get("applicable") is False:
        if c is None:
            bad.append(("contract", "recorded as not applicable, and the slow path has no contract"))
        else:
            ok, why = c.applies(state)
            if ok:
                bad.append(("contract", "recorded as not applicable; the contract applies to the input now"))
            elif why != rec.get("why"):
                bad.append(("contract", f"recorded {rec.get('why')!r}, the contract says {why!r} now"))
        if th.record is not None or th.accepted:
            bad.append(("contract", "a slow path that did not apply has no record and no accepted answer"))
        return {"ok": not bad, "mismatches": bad}
    if c is not None:
        ok, why = c.applies(state)
        if not ok:
            bad.append(("contract", f"the slow path ran, and its contract does not apply to the input now ({why})"))
    refused = rec is not None and "violations" in rec
    inner = dataclasses.replace(th, accepted=True, why=None, contract=None) if refused else th
    rep = slow.replay(inner, trust_models=trust_models)
    bad += list(rep.get("mismatches") or [])
    if inner.accepted and (c is not None or refused):
        now = [x.to_dict() for x in c.violations(inner.answer, c.echo_inputs(state), _answer_type(slow, question),
                                                 echo_type=_echo_type(slow, question))] \
            if c is not None else []
        want = list((rec or {}).get("violations") or [])
        if now != want:
            bad.append(("contract", f"recorded violations {want}, the contract gives {now}"))
    return {"ok": not bad, "mismatches": bad}


def _same_outcome(thought, ok, ans, bad):
    """The replay's verdict: the record's acceptance and answer are the recorded ones."""
    if ok != thought.accepted:
        bad.append(("accepted", f"recorded {thought.accepted}, the record gives {ok}"))
    if _vh(ans) != _vh(thought.answer):
        bad.append(("answer", f"recorded {thought.answer!r}, the record gives {ans!r}"))
    return {"ok": not bad, "mismatches": bad}


def _slow_path(system, *, question=None, propose=None, into=None, rounds=3, space=None, search=None, accept="checks",
               feedback=None):
    """The 0.9 SlowPath(...) constructor: the built-in path its arguments name, with every setting as 0.9 kept it (so
    its fingerprint, and the config of the decisions it made, are 0.9's)."""
    if propose is not None and space is not None:
        raise ValueError("a slow path refines proposals (propose=) or searches a space (space=), not both")
    if (propose is not None or space is not None) and not isinstance(into, str) and not isinstance(space, dict):
        raise ValueError("into= names the given fact a proposal or a candidate is passed as")
    if propose is not None and not callable(propose):
        raise TypeError("propose is a function (state, rounds) → a proposal, e.g. Generator.proposer(...)")
    cls = RefinePath if propose is not None else SearchPath if space is not None else AskPath
    obj = object.__new__(cls)
    _Checked.__init__(obj, system, question=question, propose=propose, into=into, rounds=rounds, space=space,
                      search=search, accept=accept, feedback=feedback)
    return obj


# ------------------------------------------------------------------------------------------------ one decision
@dataclass
class Dispatched:
    """One dispatched decision. answer: the answer given (None when a person takes it); by: "s1", "s2" or "human";
    action: "accept" (System 1 alone), "think" (the slow path was asked to answer), "check" (it checked System 1's
    answer), "human" (straight to a person); reasons: why the dispatch went that way (and why a person got it);
    s1: System 1's Response; s2: the slow path's Thought (None when it did not run); candidates: what each path would
    have answered, for the person; disagreement: System 1's and the slow path's answers when they differ; cost:
    {"s1", "s2", "total"} Costs; n: the decision's number; draw: its supervision draw; spent_before: the dispatcher's
    total spend before it; drift: the drift flag it saw; keyed: the draws were keyed by the dispatcher's secret (stored
    as "draw_keyed": true; absent from records made without one); stream: with a stream certificate (promise=), the
    decision's sample, certificate and detector state (stored as "stream"; absent otherwise); pending: the ticket a
    deferred decision completes (`start` / `complete`: an oracle outside the process answered it; stored as "pending",
    absent otherwise); learning: with an adapter loop attached (`loop=`), the loop's view of the decision — its number,
    the version served and its fingerprint, the candidate, the coin (rho, sampled), whether the loop's rule answered
    alone, its events; "stopped" when the loop had stopped (stored as "learning"; absent otherwise)."""
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
    slice: str | None = None
    stored_id: str | None = None
    audit: bool = False
    keyed: bool = False
    stream: dict | None = None
    pending: str | None = None
    learning: dict | None = None

    @property
    def supervised(self):
        return self.action == "check"

    def to_dict(self):
        from .store import plain
        return {"question": self.question, "answer": plain(self.answer), "by": self.by, "action": self.action,
                "reasons": list(self.reasons), "s1": self.s1.to_dict(),
                "s2": self.s2.to_dict() if self.s2 is not None else None, "candidates": _stored(self.candidates),
                "disagreement": _stored(self.disagreement), "cost": {k: v.to_dict() for k, v in self.cost.items()},
                "n": self.n, "draw": self.draw, "spent_before": self.spent_before.to_dict(), "drift": self.drift,
                "expected": None if self.expected is None else self.expected.to_dict(), "over_budget": self.over_budget,
                "config": self.config, "slice": self.slice, **({"audit": True} if self.audit else {}),
                **({"draw_keyed": True} if self.keyed else {}), **({"stream": self.stream} if self.stream is not None else {}),
                **({"pending": self.pending} if self.pending is not None else {}),
                **({"learning": self.learning} if self.learning is not None else {})}

    def to_json(self, indent=None):
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent, default=str)

    @classmethod
    def from_dict(cls, d, system=None, slow_system=None):
        """A stored decision back: System 1's response restored with `system`'s types, the slow path's with
        `slow_system`'s (default: the same)."""
        from .response import Response
        s1 = Response.model_validate(d["s1"], catalog=system)
        s2 = Thought.from_dict(d["s2"], slow_system or system) if d.get("s2") is not None else None
        return cls(d["question"], _restored(d.get("answer")), d["by"], d["action"], list(d["reasons"]), s1, s2,
                   _restored(d.get("candidates") or {}), _restored(d.get("disagreement")),
                   {k: Cost.from_dict(v) for k, v in (d.get("cost") or {}).items()}, d.get("n", 0), d.get("draw", 1.0),
                   Cost.from_dict(d.get("spent_before") or {}), d.get("drift"),
                   Cost.from_dict(d["expected"]) if d.get("expected") is not None else None, d.get("over_budget"),
                   d.get("config"), d.get("slice"), audit=bool(d.get("audit", False)),
                   keyed=bool(d.get("draw_keyed", False)), stream=d.get("stream"), pending=d.get("pending"),
                   learning=d.get("learning"))

    def replay(self, dispatcher, trust_models=False):
        """Re-check this decision against `dispatcher` without calling a model → {"ok", "mismatches": [(what, why)]}:
        see Dispatcher.replay."""
        return dispatcher.replay(self, trust_models=trust_models)

    def __repr__(self):
        return f"Dispatched({self.question}={self.answer!r} by {self.by}, {self.action}: {'; '.join(self.reasons)[:120]})"


@dataclass
class Started:
    """A decision begun (`Dispatcher.start`): System 1 has answered, the decision has its number, its draws, the spend,
    expected cost and drift flag it saw, and its dispatch; the slow path has not run. `Dispatcher.complete(started,
    thought)` ends it — at once (what `ask` does), or later, when an oracle outside the process gave its verdict (a
    ticket, solvi.experimental.stream.tickets). `to_dict` / `from_dict`: what a ticket's record keeps so that a restart
    can complete it. Stability: provisional (1.4)."""
    state: Any
    res: Any
    n: int
    draw: float
    spent: Cost
    expected: Cost | None
    drift: str | None
    action: str
    reasons: list
    slice: str | None
    needs_slow: bool
    keyed: bool = False
    rho: float | None = None
    probation: bool = False
    sampled: bool = False

    def to_dict(self):
        from .store import plain
        return {"input": plain(self.state), "s1": self.res.to_dict(), "n": self.n, "draw": self.draw,
                "spent_before": self.spent.to_dict(), "expected": None if self.expected is None else self.expected.to_dict(),
                "drift": self.drift, "action": self.action, "reasons": list(self.reasons), "slice": self.slice,
                "needs_slow": self.needs_slow, "keyed": self.keyed, "rho": self.rho, "probation": self.probation,
                "sampled": self.sampled}

    @classmethod
    def from_dict(cls, d, system=None):
        from .response import Response
        return cls(d["input"], Response.model_validate(d["s1"], catalog=system), d["n"], d["draw"],
                   Cost.from_dict(d.get("spent_before") or {}),
                   Cost.from_dict(d["expected"]) if d.get("expected") is not None else None, d.get("drift"),
                   d["action"], list(d["reasons"]), d.get("slice"), bool(d["needs_slow"]), bool(d.get("keyed")),
                   d.get("rho"), bool(d.get("probation")), bool(d.get("sampled")))


# ------------------------------------------------------------------------------------------------ the dispatcher
class Dispatcher:
    """System 1 first; the slow path or a person when its signals say so; within a budget. See the module docs.
    Stability: stable (final: use it, do not subclass; extend it with a SlowPath, a Monitor, a TraceStorage).

    system: System 1. slow: a SlowPath (None: every input System 1 cannot answer alone goes to a person). question:
    the question dispatched (default: the System's only question). budget: a Budget per decision; total: a Budget for
    the dispatcher's life. price: dollars per million (input, output) tokens, or a function (model, usage) → dollars —
    required for a budget in dollars. wake: the signals that wake the slow path (default: all of SIGNALS); an input
    whose signal is not among them goes to a person. agreement: {fact: minimum share} — an agreement fact of System 1
    below its minimum wakes the slow path. monitor: a solvi.core.guarantees.drift.DriftMonitor fed every System 1 response. supervise:
    the share of System 1's answers given alone that the slow path checks (drawn by a hash of seed, the input and the
    decision's number). think: "s2" (default: the slow path's accepted answer is given) or "agree" (given only when it
    equals what System 1 would have answered; otherwise a person). on_disagree: what a check that disagrees does —
    "record" (default: System 1's answer stands, the disagreement is recorded), "human", or "s2" (the slow path's
    accepted answer replaces it); in a check the slow path's answer is compared whether or not its own checks accepted
    it, and only an accepted one replaces System 1's. same: a function (answer, answer) → bool that says when two answers are the same
    (default: equal values). unknown: what the slow path's "not stated" (solvi.Unknown) is — "answer" (default: a real
    answer) or "human" (none of the options fits: a person decides). storage: a TraceStorage that keeps every decision
    (kind "dispatch"). asked: the questions System 1 is asked together (default: all of its questions, so that constraints between them apply);
    store_responses: whether System 1's and the slow path's Systems store their responses in their own storage.
    on_drift: what System 1's answers become after a drift flag when no slow path checks them (none, "drift" not in
    wake, or no budget left) — "human" (default: a person decides; its guarantee was calibrated on a stream that has
    changed) or "accept" (System 1's answer stands, its reasons say it is out of its guarantee). Either way no answer
    after a flag is reported "within its guarantee"; reset_drift() ends it. audit: the share of the decisions drawn at
    random (a hash of seed, the input and the decision's number, independent of the supervision draw) and marked
    for labelling — `Dispatched.audit` and "audit": true in the stored record; label them with
    `teach(..., label_source="audit", of=<stored id>)` (the store refuses an audit label of an unmarked decision), and the
    system report measures the promise on that random sample, which corrections made where an answer looked wrong do
    not give. Default 0: nothing is marked and the records are as before. secret: a key (bytes or str; None reads the
    environment variable SOLVI_AUDIT_SECRET, unset or empty: no key) for the supervision and audit draws — with one
    they are HMAC-SHA256(secret, seed | input | decision number) instead of the plain SHA-256 of the same, so a caller
    who can vary the inputs cannot predict which decisions get supervised or audited. The secret enters no record, no
    config fingerprint and no repr; a record says only "draw_keyed": true. Replay takes the recorded draw: it needs the
    secret only to recompute and check it (a dispatcher without the secret replays a keyed record and does not
    recompute its draw). Without a secret the draws are as before (bit for bit), and a dispatcher that samples (supervise
    or audit strictly between 0 and 1; at 1 every decision is checked) gives a PredictableDrawWarning once per
    process.

    A stream certificate (experimental; the objects come from the experimental stream package, this module does not
    import it): promise= the StreamPromise installed on the question (`system.guarantee(question, promise=sp)`), with
    audit="adaptive" (the two go together): a calibration coin, `coin(n)` — the hash (HMAC with the secret) of
    "sample|question|seed|n", no input bytes — below a rate from the past only (1 until certified and in probation,
    else min(1, max(floor, budget / the share answered alone over the last window)); sampling= {"floor", "budget",
    "window"}, defaults 0.05, 0.05, 500). A sampled decision System 1 would answer alone is checked by the slow path
    whatever `wake` says (one extra call); a sampled escalation is free; the slow path's accepted answer is the label,
    none is a loss. detector= a RiskDetector fed each sample's deployed loss; on its alarm on_drift="recertify"
    restarts the certificate and escalates every decision for probation= decisions (default 300) and until it is
    certified again, "human" / "accept" latch the alarm as the drift flag. Each decision records `stream` (sample,
    certificate, detector) and each state change a record of kind "stream"; `replay_stream()` rebuilds them all.
    Without promise= nothing of this runs and the records are as before."""

    def __init__(self, system, slow=None, *, question=None, budget=None, total=None, price=None, wake=SIGNALS,
                 agreement=None, monitor=None, supervise=0.0, seed=0, think="s2", on_disagree="record", storage=None,
                 store_responses=True, asked=None, same=None, unknown="answer", on_drift="human", audit=0.0,
                 secret: bytes | str | None = None, promise=None, detector=None, probation=None, sampling=None,
                 loop=None):
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
        adaptive = isinstance(audit, str)
        if adaptive and audit != "adaptive":
            raise ValueError('audit is a share between 0 and 1 (the decisions marked for labelling) or "adaptive"')
        if not adaptive and not 0.0 <= float(audit) <= 1.0:
            raise ValueError("audit is a share between 0 and 1 (the decisions marked for labelling)")
        self.audit = 0.0 if adaptive else float(audit)
        if adaptive != (promise is not None):
            raise ValueError('audit="adaptive" and promise= go together: the adaptive coin draws the samples of the '
                             "stream certificate given as promise=, and that certificate learns from no other sampling")
        if detector is not None and promise is None:
            raise ValueError("detector= watches the deployed risk of a stream certificate: give promise= too")
        if on_drift == "recertify" and detector is None:
            raise ValueError('on_drift="recertify" restarts the stream certificate on the detector\'s alarm: give '
                             "promise= and detector=")
        if probation is not None and on_drift != "recertify":
            raise ValueError('probation= is the escalation after a recertification: it needs on_drift="recertify"')
        if secret is None:
            secret = os.environ.get(SECRET_ENV) or None
        if secret is not None:
            if isinstance(secret, str):
                secret = secret.encode()
            if not isinstance(secret, (bytes, bytearray)) or not secret:
                raise TypeError("secret is a non-empty bytes or str (the key of the supervision and audit draws)")
            secret = _Secret(bytes(secret))
        self._secret = secret
        if think not in ("s2", "agree"):
            raise ValueError('think must be "s2" or "agree"')
        if on_disagree not in ("record", "human", "s2"):
            raise ValueError('on_disagree must be "record", "human" or "s2"')
        if unknown not in ("answer", "human"):
            raise ValueError('unknown must be "answer" or "human"')
        if on_drift not in ("human", "accept", "recertify"):
            raise ValueError('on_drift must be "human", "accept" or "recertify"')
        if same is not None and not callable(same):
            raise TypeError("same is a function (answer, answer) → bool")
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
        self.same, self.unknown, self.on_drift = same, unknown, on_drift
        from .store import open_storage
        self.storage = open_storage(storage)
        self.store_responses = bool(store_responses)
        self.n = 0
        self.spent = Cost()
        self._runs = [0, 0.0, 0, 0.0, False]  # slow-path runs so far: count, dollars, calls, ms, a run without a price
        self.counts = {p: 0 for p in PATHS}
        self.drift = None
        self.policy = None                # calibrate(): who answers on each slice System 1 hands over
        self._lock = threading.Lock()
        self._stream = self._stream0 = None
        if promise is not None:
            guard = getattr(system, "guards", {}).get(question)
            if guard is None or guard.promise is not promise:
                raise ValueError("promise= is the stream certificate installed on the question: "
                                 f"system.guarantee({question!r}, promise=sp) first")
            if not callable(getattr(promise, "stream_driver", None)):
                raise TypeError("promise= takes a stream certificate (a StreamPromise: it has stream_driver)")
            self._stream = promise.stream_driver(detector, on_drift=on_drift,
                                                 probation=300 if probation is None else probation, **(sampling or {}))
            self._stream0 = copy.deepcopy(self._stream)          # what replay_stream rebuilds the states from
        self.loop = None
        self._learn_lock = threading.RLock()     # a loop's decisions one at a time (outermost: before part and loop)
        if loop is not None:
            loop.attach_to(self)                 # sets self.loop (experimental: solvi.experimental.stream)
        global _predictable_warned
        if secret is None and (any(0 < x < 1 for x in (self.supervise, self.audit)) or adaptive) \
                and not _predictable_warned:
            _predictable_warned = True
            warnings.warn("Dispatcher: the supervision / audit draw is a public hash of the seed, the input and the "
                          "decision's number, so whoever can vary the inputs can predict which decisions are checked; "
                          f"give secret= (or set {SECRET_ENV}) to key it", PredictableDrawWarning, stacklevel=2)

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
                      self.think_policy, self.on_disagree, self.unknown,
                      *([("on_drift", self.on_drift)] if self.on_drift != "human" else []),
                      *([("audit", self.audit)] if self.audit else []),
                      code_fingerprint(self.same) if self.same is not None else None,
                      None if self.budget is None else self.budget.to_dict(),
                      None if self.total is None else self.total.to_dict(), pr,
                      self.slow.fingerprint() if self.slow is not None else None,
                      None if self.policy is None else self.policy["fingerprint"],
                      *([("contract", self.slow.contract.fingerprint())]       # a decision made under another contract
                        if self.slow is not None and getattr(self.slow, "contract", None) is not None else []),
                      *([("stream", "adaptive", self._stream.config())] if self._stream is not None else []))

    # --- the decision
    @property
    def keyed(self) -> bool:
        """Are the draws keyed by a secret?"""
        return self._secret is not None

    def _hash(self, msg, keyed):
        if keyed:
            if self._secret is None:
                raise ValueError("a keyed draw needs the dispatcher's secret")
            return self._secret.mac(msg)
        return hashlib.sha256(msg.encode()).hexdigest()

    def draw(self, init_hash, n, *, keyed=None):
        """The supervision draw of decision n on this input: a number in [0, 1) from a hash (reproducible) — HMAC with
        the secret when the dispatcher has one. keyed: force the plain (False) or keyed (True) hash (default: keyed
        when there is a secret)."""
        h = self._hash(f"{self.seed}|{init_hash}|{n}", self.keyed if keyed is None else keyed)
        return int(h[:13], 16) / 16 ** 13

    def audit_draw(self, init_hash, n, *, keyed=None):
        """The audit draw of decision n on this input: a number in [0, 1) from a hash, independent of `draw` (keyed
        as `draw`)."""
        h = self._hash(f"audit|{self.seed}|{init_hash}|{n}", self.keyed if keyed is None else keyed)
        return int(h[:13], 16) / 16 ** 13

    def coin(self, n, *, keyed=None):
        """The calibration coin of decision n (audit="adaptive"): a number in [0, 1) from the hash of
        "sample|question|seed|n" — no input bytes, so the coin cannot depend on the input (keyed as `draw`)."""
        h = self._hash(f"sample|{self.question}|{self.seed}|{n}", self.keyed if keyed is None else keyed)
        return int(h[:13], 16) / 16 ** 13

    def expected(self):
        """The expected Cost of one slow-path run: the mean of the runs so far (None before the first)."""
        k, usd, calls, ms, unpriced = self._runs
        if not k:
            return None
        return Cost(None if unpriced else usd / k, round(calls / k), ms / k)

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

    def decide(self, res, draw, spent_before, drift, expected=None, *, probation=False, sampled=False, hold=None):
        """The dispatch of one System 1 response → (action, reasons). Pure: the same response, draw, spend, drift
        state and expected cost give the same dispatch (replay re-derives it). With a stream certificate: probation
        (System 1 answers nothing alone: a think) and sampled (the calibration coin came up: an answer System 1 gives
        alone is checked by the slow path, whatever `wake` says). hold: with an adapter loop, why its rule does not
        answer alone here (System 1 answers nothing alone: a think, as in probation)."""
        sig = self.signals(res)
        if drift and not any(s == "drift" for s, _ in sig):
            sig.append(("drift", drift))
        reasons = []
        think = [(s, w) for s, w in sig if s in THINK]
        if probation and not think and res[self.question].status == "ok":
            think = [("guarantee", PROBATION_REASON)]
        if hold and not think and res[self.question].status == "ok":
            think = [("guarantee", hold)]
        check = [(s, w) for s, w in sig if s == "drift"]
        if not think and res[self.question].status == "ok" and draw < self.supervise:
            check.append(("supervise", f"sampled for supervision (draw {draw:.4f} < {self.supervise:g})"))
        if sampled and not think and res[self.question].status == "ok":
            check.append(("sample", SAMPLE_REASON))
        if think:
            reasons += [w for _, w in think]
            if not all(s in self.wake for s, _ in think):
                off = sorted({s for s, _ in think if s not in self.wake})
                return "human", reasons + [f"{', '.join(off)} does not wake the slow path: a person decides"]
            if self.policy is not None and not self._needs_slow(think[0][0]):
                return "think", reasons                      # System 1's own answer or a person: no slow path, no cost
            if self.slow is None:
                return "human", reasons + ["no slow path: a person decides"]
            why = self._no_budget(spent_before, expected)
            if why:
                return "human", reasons + [why]
            return "think", reasons
        drifted = [w for s, w in check if s == "drift"] if res[self.question].status != "forced" else []
        check = [(s, w) for s, w in check if s in self.wake or s == "sample"]
        if check and self.slow is not None:
            why = self._no_budget(spent_before, expected)
            if not why:
                return "check", [w for _, w in check]
            if drifted and self.on_drift != "accept":
                return "human", [w for _, w in check] + [why + "; System 1 is out of its guarantee since the drift "
                                                               "flag: a person decides"]
            return "accept", [w for _, w in check] + [why + "; System 1's answer stands unchecked"
                                                      + (", out of its guarantee since the drift flag" if drifted else "")]
        if drifted:
            if self.on_drift != "accept":
                return "human", drifted + ["System 1 is out of its guarantee since the drift flag and no slow path "
                                           "checks it: a person decides"]
            return "accept", drifted + ["System 1 answered, out of its guarantee since the drift flag (on_drift="
                                        "\"accept\"; reset_drift() after recalibrating)"]
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
        if self.loop is not None:
            return self._ask_loop(state)
        b = self.start(state)
        th = None
        if b.needs_slow:
            per_round = None if b.expected is None else _per_round(b.expected, self.slow)
            th = self.slow.run(state, self.question, price=self.price, budget=self.budget, expected_round=per_round,
                               store=self.store_responses)
        return self.complete(b, th)

    def _ask_loop(self, state):
        """`ask` with an adapter loop: the loop reads the decision before the dispatch (its coin, whether its rule
        answers alone — outside System 1's call, so no part's lock is held while the loop's is taken) and takes it, with
        the slow path's label, after. The loop's decisions run one at a time (`_learn_lock`, taken before the part's
        and the loop's locks and never while either is held), the slow path's call included: with a loop attached a
        Dispatcher's throughput is at most 1 / the slow path's latency (on a stand-in: 40 decisions in
        four threads with a 50 ms slow path took 3.7 × as long as without a loop). The loop's controls take the same
        lock first, so a control waits for the decision in flight. The lock is needed as built: the decision's
        number and coin are read before System 1 and the step after it, and a second decision in between would read
        the same number."""
        loop = self.loop
        with self._learn_lock:
            x = pre = None
            try:
                x = loop._input(self.system, state)
                pre = loop._pre(x)
            except Exception as e:  # noqa: BLE001 — a stopped or failing loop: the dispatch goes on without it
                pre = {"stopped": f"{type(e).__name__}: {e}"}
            b = self._start(state, pre)
            th = None
            if b.needs_slow:
                per_round = None if b.expected is None else _per_round(b.expected, self.slow)
                th = self.slow.run(state, self.question, price=self.price, budget=self.budget,
                                   expected_round=per_round, store=self.store_responses)
            return self._complete(b, th, pre=pre, x=x)

    def _hold(self, pre):
        """(hold reason, sampled) for decide from the loop's view of a decision (None: no loop or a stopped one)."""
        if not pre or "stopped" in pre:
            return None, False
        hold = None
        if pre.get("probation"):
            hold = PROBATION_REASON
        elif not pre.get("alone"):
            hold = (f"the learning loop's version {pre.get('served')} does not answer alone here (not certified, or "
                    "its score is below its certified threshold)")
        return hold, bool(pre.get("sampled"))

    def start(self, state):
        """The first half of `ask`: System 1 answers, the decision gets its number, draws and dispatch → a Started
        (`needs_slow`: the slow path is to run). Provisional (1.4): for an oracle that answers later (tickets). Not
        with an adapter loop attached (its decisions run whole, through `ask`)."""
        if self.loop is not None:
            raise ValueError("start / complete with an adapter loop attached are not built: use ask()")
        return self._start(state)

    def _start(self, state, pre=None):
        q = self.question
        res = self.system.ask(state, self.asked, store=self.store_responses)
        if self.monitor is not None:
            rep = self.monitor.observe(res, question=q)
            if rep.get("drift") and self.drift is None:
                self.drift = "DriftMonitor: " + "; ".join(rep.get("why") or ["drift"])
                if self.slow is None or "drift" not in self.wake:
                    from .guarantees.guarantee import GuaranteeWarning
                    warnings.warn(f"drift flagged ({self.drift}) and no slow path checks System 1: its answers are "
                                  + ("given to a person" if self.on_drift != "accept" else "given out of its guarantee")
                                  + " until reset_drift()", GuaranteeWarning, stacklevel=3)
        with self._lock:
            self.n += 1
            n = self.n
            spent = Cost(**vars(self.spent))
        draw = self.draw(res.trace.init_hash, n)
        expected = self.expected()
        rho, prob, sampled = None, False, False
        if self._stream is not None:
            rho, prob = self._stream.before()
            sampled = self.coin(n) < rho
            self._stream.issue(n, sampled)                 # its sample is read in the order of the numbers
        hold = None
        if pre is not None:
            hold, sampled = self._hold(pre)
        action, reasons = self.decide(res, draw, spent, self.drift, expected, probation=prob, sampled=sampled,
                                      hold=hold)
        sl = self.slice_of(res) if action == "think" else None
        needs = action == "check" or (action == "think" and self._needs_slow(sl))
        return Started(state, res, n, draw, spent, expected, self.drift, action, reasons, sl, needs, self.keyed, rho,
                       prob, sampled)

    def complete(self, b, th=None, *, pending=None):
        """The second half of `ask`: a Started and the slow path's Thought (None when it was not to run) → the
        Dispatched, recorded. pending: the ticket it completes (recorded as "pending"). A Thought whose record is a dict
        with "oracle": {"label_source", "label_of", "lost"} is an oracle's verdict given outside the process: its source
        and the record holding it go to the stream sample; lost (no verdict: an expired ticket) sends the input to a
        person and counts as a missing label. Provisional (1.4)."""
        if self.loop is not None:
            raise ValueError("start / complete with an adapter loop attached are not built: use ask()")
        return self._complete(b, th, pending=pending)

    def _complete(self, b, th=None, *, pending=None, pre=None, x=None):
        q = self.question
        if b.needs_slow and th is None:
            raise ValueError("this decision needs the slow path's Thought to complete")
        res, action, reasons = b.res, b.action, list(b.reasons)
        paid = action == "check" and reasons[:1] == [SAMPLE_REASON]
        r1 = res[q]
        would1 = r1.answer if r1.status != "abstain" else _would(r1, res, q)
        cand = {"s1": would1}
        c1 = cost_of([res], self.price, res.ms)
        over, disagreement = None, None
        if th is not None:
            cand["s2"] = th.answer
            if self.budget is not None:
                over = self.budget.over(th.cost)
            with self._lock:
                r = self._runs
                r[0], r[2], r[3] = r[0] + 1, r[2] + th.cost.calls, r[3] + th.cost.ms
                r[1] += th.cost.usd or 0.0
                r[4] = r[4] or th.cost.usd is None
        answer, by = self._outcome(action, r1, would1, th, reasons, b.slice)
        if th is not None and action == "check" and not self.agrees(th.answer, would1):
            disagreement = {"s1": would1, "s2": th.answer, "s2_accepted": th.accepted}
        c2 = th.cost if th is not None else Cost(0.0 if self.price is not None else None)
        cost = {"s1": c1, "s2": c2, "total": c1 + c2}
        with self._lock:
            self.spent = self.spent + c2
            self.counts[by] += 1
        stream, events = None, []
        if self._stream is not None:
            stream, events, why = self._stream.after(**self._stream_inputs(action, paid, r1, would1, th),
                                                     sampled=b.sampled, rho=b.rho, probation=b.probation, n=b.n)
            self._stream_told(events, why)
        learning = None if pre is None else self._learned(pre, x, action, paid, r1, would1, th)
        d = Dispatched(q, answer, by, action, reasons, res, th, cand, disagreement, cost, b.n, b.draw, b.spent, b.drift,
                       b.expected, over, self.config(), slice=b.slice,
                       audit=bool(self.audit) and self.audit_draw(res.trace.init_hash, b.n) < self.audit,
                       keyed=b.keyed, stream=stream, pending=pending, learning=learning)
        if self.storage is not None:
            from .store import FORMAT
            rec = self.storage._append({"v": FORMAT, "kind": "dispatch", **json.loads(d.to_json())})
            d.stored_id = rec["id"]
            for e in events:
                self.storage._append({"v": FORMAT, "kind": "stream",
                                      "stream": json.loads(json.dumps({**e, "question": q, "decision": b.n}))})
        return d

    def _learned(self, pre, x, action, paid, r1, would1, th):
        """The loop's step of this decision (outside System 1's call and the dispatcher's lock) → the record's
        "learning". The label is the slow path's accepted answer (an oracle's verdict with its own source); a label
        whose source the loop's policy refuses is a missing one for the loop. A loop that fails here stops (its own
        "error" event); the decision stands."""
        if "stopped" in pre:
            return {"stopped": pre["stopped"]}
        out = {k: pre[k] for k in ("n", "served", "candidate", "fingerprint", "alone", "sampled", "rho", "probation")}
        inp = self._stream_inputs(action, paid, r1, would1, th)
        try:
            step = self.loop._post(pre, x, inp["label"], inp["label_source"], inp.get("label_of"))
        except Exception as e:  # noqa: BLE001 — recorded; the dispatcher's answer is not the loop's
            out["error"] = f"{type(e).__name__}: {e}"
            return out
        out.update(label=_stored(step["label"]), label_source=step["label_source"], by=step["by"],
                   events=step["events"])
        if not step["same"]:
            out["step"] = {"served": step["served"], "candidate": step["candidate"]}
        return out

    # --- the stream certificate (promise=, audit="adaptive")
    def _stream_inputs(self, action, paid, r1, would1, th):
        """What the stream step reads of one decision: the gate's signal and threshold, System 1's would-be answer,
        whether it was given alone, the slow path's accepted answer as the label (None: missing), whether the sample
        was free (the slow path ran anyway); for an oracle's verdict given outside the process (`_oracle`), its source,
        the record that holds it (label_of) and whether it was lost (no verdict)."""
        from .catalog import Unknown
        g = (r1.extra or {}).get("guarantee") or {}
        o = _oracle(th)
        lost = bool(o.get("lost"))
        label = None
        if th is not None and th.accepted and not lost and not (self.unknown == "human" and th.answer is Unknown):
            label = would1 if self.agrees(th.answer, would1) else th.answer
        out = {"score": g.get("value"), "in_force": g.get("threshold"), "would": would1,
               "alone": r1.status == "ok" and action in ("accept", "check"), "label": label,
               "label_source": (o.get("label_source") or "s2") if label is not None else None,
               "free": th is not None and not paid}
        if o:                                          # only an oracle's verdict: the records of stage 2 are unchanged
            out["label_of"] = o.get("label_of") if label is not None else None
            out["lost"] = lost
        return out

    def _stream_told(self, events, why):
        """An alarm: a person is told (a GuaranteeWarning); with on_drift other than "recertify" it is the latched
        drift flag."""
        if not any(e["event"] == "alarm" for e in events):
            return
        from .guarantees.guarantee import GuaranteeWarning
        if why is not None:
            if self.drift is None:
                self.drift = why
            warnings.warn(f"{why}: the stream certificate's answers go "
                          + ("to a person" if self.on_drift == "human" else "out of its guarantee")
                          + " until reset_drift()", GuaranteeWarning, stacklevel=3)
        else:
            alarm = next(e for e in events if e["event"] == "alarm")
            warnings.warn(f"{alarm['why']}: the stream certificate restarts (version index "
                          f"{self._stream.promise.v}, threshold +inf) and System 1 answers nothing alone for "
                          f"{self._stream.probation} decisions and until it is certified again", GuaranteeWarning,
                          stacklevel=3)

    def replay_stream(self, stored=None):
        """Rebuild the stream certificate, its detector and its rate from the stored decisions in order and compare at
        every decision → {"ok", "mismatches": [(n, what, why)], "coin": "verified" | "unverified ...", "decisions"}.
        Starts from the state the dispatcher was built with; re-reads each decision's gate signal (the guard's
        recorded state is compared before it is observed), recomputes the calibration coin (only with the secret for
        a keyed record; else the recorded flag is taken and the coin is reported unverified), feeds the same samples
        and compares the recorded sample, certificate and detector fields; with storage, the "stream" records too.
        A decision completed later through a ticket (`start` / `complete`) is read in two places, as it was made: its
        gate, rate and coin where its ticket was issued (the "pending" record), its sample where it was completed — so
        decisions with tickets are replayed from the storage (stored=None), not from a list."""
        if self._stream is None:
            raise ValueError('replay_stream needs a dispatcher with a stream certificate (promise=, audit="adaptive")')
        items = self._stored_items() if stored is None else [("dispatch", d, None) for d in stored]
        bad, events, unverified, k, _ = self._rebuild_stream(copy.deepcopy(self._stream0), items)
        if stored is None and self.storage is not None:
            got = [s.data.get("stream") for s in self.storage.iter(kind="stream")]
            got = [e for e in got if (e or {}).get("question") == self.question]
            if got != events:
                bad.append((None, "stream records", f"{len(got)} stored, {len(events)} rebuilt, or they differ"))
        coin = "verified" if not unverified else (f"unverified on {unverified} keyed decision(s): this dispatcher has "
                                                  "no secret")
        return {"ok": not bad, "mismatches": bad, "coin": coin, "decisions": k}

    def _stored_items(self):
        """This question's decisions and tickets in the store's order → [("dispatch", Dispatched, None) |
        ("pending", Started, the ticket's "pending" fields)]."""
        if self.storage is None:
            raise ValueError("the dispatcher has no storage: Dispatcher(..., storage=TraceStorage)")
        slow_sys = self.slow.system if self.slow is not None else None
        out = []
        for s in self.storage.iter(kind=None):
            d = s.data
            if d.get("kind") == "dispatch" and d.get("question") == self.question:
                x = Dispatched.from_dict(d, self.system, slow_sys)
                x.stored_id = s.id
                out.append(("dispatch", x, None))
            elif d.get("kind") == "pending" and (d.get("pending") or {}).get("question") == self.question \
                    and d.get("issue") is not None:
                out.append(("pending", Started.from_dict(d["issue"], self.system), {**d["pending"], "id": s.id}))
        return out

    def _rebuild_stream(self, drv, items):
        """Feed a stream driver the recorded decisions → (mismatches, the "stream" events, unverified coins, decisions,
        the drift flag latched after the last one)."""
        p, q = drv.promise, self.question
        bad, events, unverified, k, latched = [], [], 0, 0, None
        issued = {}

        def gate(n, res, keyed, rho_rec, prob_rec, sampled):
            nonlocal unverified
            r1 = res[q]
            g = (r1.extra or {}).get("guarantee") or {}
            if g.get("method") == p.method and g.get("value") is not None:
                if g.get("state") != _json(p.state()):
                    bad.append((n, "guard state", "the gate's recorded state is not the rebuilt one"))
                p.observe(g["value"])
            rho, prob = drv.before()
            if rho != rho_rec or prob != prob_rec:
                bad.append((n, "rate", f"recorded ρ {rho_rec} (probation {prob_rec}), rebuilt {rho} ({prob})"))
            if keyed and not self.keyed:
                unverified += 1
            elif (self.coin(n, keyed=keyed) < rho) != sampled:
                bad.append((n, "coin", f"recorded sampled={sampled}, the coin gives {not sampled}"))
            drv.issue(n, sampled)

        for kind, x, extra in items:
            if kind == "pending":
                gate(x.n, x.res, x.keyed, x.rho, x.probation, x.sampled)
                issued[extra.get("ticket")] = x
                continue
            d = x
            k += 1
            st = d.stream
            if st is None:
                bad.append((d.n, "stream", "a decision without the stream record"))
                continue
            sam = st.get("sample") or {}
            sampled = bool(sam.get("sampled"))
            if d.pending is None:
                gate(d.n, d.s1, d.keyed, sam.get("rho"), sam.get("probation"), sampled)
                prob = sam.get("probation")
            else:
                t = issued.pop(d.pending, None)
                if t is None or t.n != d.n:
                    bad.append((d.n, "ticket", f"the decision completes ticket {d.pending}, which was not issued "
                                               "before it (or for another decision)"))
                    continue
                if (t.rho, t.probation, t.sampled) != (sam.get("rho"), sam.get("probation"), sampled):
                    bad.append((d.n, "ticket", "the decision's rate, probation or coin is not its ticket's"))
                prob = t.probation
            r1 = d.s1[q]
            would1 = r1.answer if r1.status != "abstain" else _would(r1, d.s1, q)
            paid = d.action == "check" and d.reasons[:1] == [SAMPLE_REASON]
            rec, ev, why = drv.after(**self._stream_inputs(d.action, paid, r1, would1, d.s2), sampled=sampled,
                                     rho=sam.get("rho"), probation=prob, n=d.n)
            latched = d.drift or why
            rec = _json(rec)
            for part in ("sample", "certificate", "detector"):
                if rec.get(part) != st.get(part):
                    bad.append((d.n, part, f"recorded {st.get(part)}, rebuilt {rec.get(part)}"))
            events += [_json({**e, "question": q, "decision": d.n}) for e in ev]
        return bad, events, unverified, k, latched

    def resume(self):
        """Continue from the store after a restart: a freshly built dispatcher (configured as before, the same
        storage, nothing asked yet) reads its question's stored decisions and tickets in order → the number of the
        last decision. Restored: the decision count and numbers, the spend, the counts per path, the slow path's mean
        run (the expected cost), the drift flag of the last decision (and the one a detector's alarm latched after it),
        and — with a stream certificate — the certificate, detector, rate and probation, rebuilt and compared as
        `replay_stream` does (a mismatch raises ValueError: the store does not continue this dispatcher). Not restored:
        a calibrate() policy, a DriftMonitor's state, a reset_drift() made after the last decision. Tickets issued
        and not completed keep their numbers; complete them with their Started (solvi.experimental.stream.tickets).
        Provisional (1.4)."""
        if self.n:
            raise ValueError("resume() continues a fresh dispatcher from its store: this one has decided already")
        items = self._stored_items()
        latched = None
        if self._stream is not None:
            bad, _, _, _, latched = self._rebuild_stream(self._stream, items)
            if bad:
                raise ValueError(f"the store does not continue this dispatcher: {bad[:3]}")
        n, last = 0, None
        for kind, x, _ in items:
            n = max(n, x.n)
            if kind != "dispatch":
                continue
            last = x
            self.counts[x.by] = self.counts.get(x.by, 0) + 1
            c2 = x.cost.get("s2") or Cost(0.0 if self.price is not None else None)
            self.spent = self.spent + c2
            if x.s2 is not None:
                r = self._runs
                r[0], r[2], r[3] = r[0] + 1, r[2] + c2.calls, r[3] + c2.ms
                r[1] += c2.usd or 0.0
                r[4] = r[4] or c2.usd is None
        self.n = n
        if last is not None:
            self.drift = last.drift if self._stream is None else latched
        return n

    def _needs_slow(self, sl):
        """Does a think on this slice run the slow path? Always before calibrate(); after it, only when the slice's
        calibrated answerer is the slow path."""
        if self.policy is None:
            return True
        p = self.policy["slices"].get(sl)
        return p is not None and p["answer"] in ("s2", "agree")

    def _outcome(self, action, r1, would1, th, reasons, sl=None):
        """(answer, by) from the action and the two paths' answers; appends why a person gets it."""
        if action == "human":
            return None, "human"
        if action == "accept":
            return r1.answer, "s1"
        if _oracle(th).get("lost"):                    # no verdict (an expired ticket): a person, whatever the action
            reasons.append(th.why or "the oracle gave no verdict: a person decides")
            return None, "human"
        from .catalog import Unknown
        if action == "think" and self.policy is not None:
            return self._calibrated(sl, r1, would1, th, reasons)
        ok = th.accepted
        if ok and self.unknown == "human" and th.answer is Unknown:
            ok = False
            if action == "think":
                reasons.append("the slow path says none of the options (not stated): a person decides")
                return None, "human"
        if action == "think":
            if not ok:
                reasons.append(f"the slow path did not answer: {th.why or th.stopped}")
                return None, "human"
            if self.think_policy == "agree" and not self.agrees(th.answer, would1):
                reasons.append(f"the slow path answered {th.answer!r}, System 1 would have answered {would1!r}: a "
                               "person decides")
                return None, "human"
            return th.answer, "s2"
        # check
        if self.agrees(th.answer, would1):
            return r1.answer, "s1"
        reasons.append(f"the slow path says {th.answer!r}" + ("" if ok else " (not accepted)") + f", System 1 {would1!r}")
        if self.on_disagree == "human":
            return None, "human"
        if self.on_disagree == "s2" and ok:
            return th.answer, "s2"
        return r1.answer, "s1"

    def _calibrated(self, sl, r1, would1, th, reasons):
        """A think after calibrate(): the slice's answerer, above its threshold, or a person."""
        p = self.policy["slices"].get(sl)
        if p is None:
            reasons.append(f"no calibrated answerer for the slice {sl!r}: a person decides")
            return None, "human"
        a, t = p["answer"], p["threshold"]
        where = f"calibrated on the slice {sl!r} ({p['n']} examples)"
        if a == "human":
            reasons.append(f"{where}: a person decides ({p['why']})")
            return None, "human"
        if a == "s1":
            sig = _s1_signal(r1)
            if would1 is not None and sig >= t:
                reasons.append(f"{where}: System 1's own answer, {sig:.4g} ≥ {t:.4g}")
                return would1, "s1"
            reasons.append(f"{where}: System 1's own answer, {sig:.4g} < {t:.4g}: a person decides")
            return None, "human"
        sig = self._s2_signal(th)
        agree = a == "s2" or self.agrees(th.answer, would1)
        if sig >= t and agree:
            reasons.append(f"{where}: the slow path's answer" + (" (it agrees with System 1)" if a == "agree" else "")
                           + f", {sig:.4g} ≥ {t:.4g}")
            return th.answer, "s2"
        why = (f"{sig:.4g} < {t:.4g}" if sig < t else f"it says {th.answer!r}, System 1 {would1!r}")
        reasons.append(f"{where}: the slow path's answer is not taken ({why}): a person decides")
        return None, "human"

    def _s2_signal(self, th):
        """The slow path's signal: its answer's confidence (ask), 1 for an accepted refinement or search; −inf when it
        has no answer, or says "not stated" with unknown="human"."""
        from .catalog import Unknown
        if th is None or th.answer is None or (self.unknown == "human" and th.answer is Unknown):
            return -math.inf
        return float(self.slow.signal(th, self.question))

    def slice_of(self, res):
        """The slice of System 1's response: the first signal that wakes the slow path (None: none does)."""
        kinds = [s for s, _ in self.signals(res) if s in THINK]
        return kinds[0] if kinds else None

    # --- calibrating who answers on the hard slice
    def calibrate(self, examples, *, max_risk=None, max_error=None, method=None, delta=0.10, correct=None,
                  answerers=("s1", "s2", "agree"), min_slice=20, min_support=5):
        """Choose who answers on each slice System 1 hands over, and with what threshold, so that the dispatcher's
        answers given alone keep a promise — calibrated on labelled examples drawn through this same dispatch, not on
        an average sample. See the module docs ("Calibrating the hard slice").

        examples: [(state, correct answer)] — not the ones System 1's own guarantee was calibrated on (its answers
        must be out of sample here, or they use up the promise). correct(answer, label, response) → bool for
        answers judged otherwise than by equality (response: the Response that produced the answer). max_risk= (crc,
        a share of all examples) or max_error= (ltt: among the answers given alone; method="empirical" possible): the
        promise for all the dispatcher's answers given alone — System 1's within its guarantee and the slices' together.
        answerers: who may answer a slice — "s1" (System 1's own answer, thresholded on its confidence), "s2" (the slow
        path's, on its confidence), "agree" (the slow path's when it equals System 1's); a person is always possible.
        min_slice: a slice with fewer examples gets a person (the record says so); min_support: an answerer's threshold
        must let at least this many examples of the slice through. → the report; the choice is kept as `policy` (in
        the config every decision records) and every later think follows it. Calling the slow path on the slice's
        examples costs what it costs (report["cost"])."""
        from .calibration import _binom_cdf, check_rate
        from .catalog import Unknown
        from .guarantees.guarantee import promise_text
        from .provenance import digest
        if (max_risk is None) == (max_error is None):
            raise ValueError("give the promise as max_risk= (crc) or max_error= (ltt; method=\"empirical\" possible)")
        method = method or ("crc" if max_risk is not None else "ltt")
        if method not in ("crc", "ltt", "empirical") or (method == "crc") != (max_risk is not None):
            raise ValueError('method: "crc" with max_risk=, "ltt" or "empirical" with max_error=')
        level = check_rate("risk" if method == "crc" else "error", max_risk if method == "crc" else max_error)
        bad = [a for a in answerers if a not in ("s1", "s2", "agree")]
        if bad:
            raise ValueError(f"unknown answerer(s) {bad}: s1, s2, agree")
        if self.slow is None and any(a != "s1" for a in answerers):
            answerers = tuple(a for a in answerers if a == "s1")
        examples = list(examples)
        if not examples:
            raise ValueError("calibrate needs labelled examples [(state, correct answer)]")
        q = self.question
        at = self.system.questions[q].answer

        def judge(ans, label, res):
            if ans is None:
                return False
            if correct is not None:
                return bool(correct(ans, label, res))
            try:
                return _vh(ans) == _vh(at.normalize(label))
            except ValueError:
                return _vh(ans) == _vh(label)
        saved, self.policy = self.policy, None
        n_all, a1, w1, people = len(examples), 0, 0, 0
        slices, cost = {}, Cost(0.0 if self.price is not None else None)
        try:
            for state, label in examples:
                res = self.system.ask(state, self.asked, store=False)
                r1 = res[q]
                kinds = [s for s, _ in self.signals(res) if s in THINK]
                if not kinds:
                    if r1.status == "abstain":
                        people += 1
                    else:
                        a1 += 1
                        w1 += not judge(r1.answer, label, res)
                    continue
                if not all(k in self.wake for k in kinds):
                    people += 1
                    continue
                would1 = _would(r1, res, q)
                row = {"s1": (_s1_signal(r1) if would1 is not None else -math.inf, judge(would1, label, res))}
                if any(a in ("s2", "agree") for a in answerers):
                    th = self.slow.run(state, q, price=self.price, store=False)
                    cost = cost + th.cost
                    sig = self._s2_signal(th)
                    rec = th.responses[-1] if th.responses else None
                    ok2 = th.answer is not Unknown or self.unknown != "human"
                    right2 = judge(th.answer, label, rec) if ok2 else False
                    row["s2"] = (sig, right2)
                    row["agree"] = (sig if self.agrees(th.answer, would1) else -math.inf, right2)
                slices.setdefault(kinds[0], []).append(row)
        finally:
            self.policy = saved
        # per slice: the options (answerer, threshold) → (answered, wrong); a person: (0, 0)
        options, info = {}, {}
        for k, rows in slices.items():
            n = len(rows)
            acc = {a: round(sum(r[a][1] for r in rows) / n, 4) for a in answerers}
            info[k] = {"n": n, "accuracy": acc}
            opts = [("human", None, 0, 0)]
            if n < min_slice:
                info[k]["why"] = f"too few examples on this slice ({n} < min_slice {min_slice}): a person decides"
                options[k] = opts
                continue
            for a in answerers:
                pts = sorted(((r[a][0], r[a][1]) for r in rows if math.isfinite(r[a][0])), key=lambda x: -x[0])
                m = w = 0
                for i, (sg, ok) in enumerate(pts):
                    m, w = m + 1, w + (not ok)
                    if (i + 1 == len(pts) or pts[i + 1][0] != sg) and m >= min_support:
                        opts.append((a, sg, m, w))
            options[k] = opts
        # every combination of one option per slice, kept on its Pareto front (fewest wrong for each number answered)
        front = {(0, 0): {}}
        for k, opts in options.items():
            nxt = {}
            for (m0, w0), ch in front.items():
                for a, t, m, w in opts:
                    key = (m0 + m, w0 + w)
                    if key not in nxt:
                        nxt[key] = {**ch, k: (a, t, m, w)}
            best = {}
            for (m, w), ch in nxt.items():
                if m not in best or w < best[m][0]:
                    best[m] = (w, ch)
            front = {(m, w): ch for m, (w, ch) in best.items()}
        g = max(1, len(front))

        def keeps(m, w):
            ans, wrong = a1 + m, w1 + w
            if method == "crc":
                return (wrong + 1) / (n_all + 1) <= level + 1e-12
            if method == "empirical":
                return ans > 0 and wrong <= level * ans + 1e-9
            return ans > 0 and _binom_cdf(wrong, ans, level) <= delta / g
        ok = [(m, w) for (m, w) in front if keeps(m, w)]
        why_none = None
        if ok:
            m, w = max(ok, key=lambda x: (x[0], -x[1]))
            choice = front[(m, w)]
        else:
            m, w, choice = 0, 0, {}
            why_none = ("System 1's own answers already break the promise on these examples: every slice goes to a "
                        "person")
        pol = {}
        for k in slices:
            a, t, mk, wk = choice.get(k, ("human", None, 0, 0))
            e = {**info[k], "answer": a, "threshold": t, "answered": mk, "wrong": wk}
            if a == "human":
                e["why"] = info[k].get("why") or why_none or "no answerer keeps the promise on this slice"
            pol[k] = e
        promise = promise_text(method, level, delta if method == "ltt" else None)
        report = {"method": method, "level": level, "delta": delta if method == "ltt" else None, "n": n_all,
                  "s1_answered": a1, "s1_wrong": w1, "to_person": people, "slices": pol,
                  "answered": (a1 + m) / n_all, "wrong": w1 + w,
                  "risk": (w1 + w) / n_all, "error": (w1 + w) / max(1, a1 + m), "cost": cost.to_dict(),
                  "promise": f"the dispatcher's answers given alone: {promise} — calibrated on {n_all} examples "
                             f"through this dispatch, each slice System 1 hands over on its own examples"}
        if why_none:
            report["why"] = why_none
        report["fingerprint"] = digest("DispatchPolicy", method, level, report["delta"],
                                       sorted((k, v["answer"], v["threshold"]) for k, v in pol.items()), n_all,
                                       digest(*[_vh(s) for s, _ in examples]))
        self.policy = report
        if self.storage is not None:              # the promise in force, for a report from the store alone
            from .schema import tag_floats
            from .store import FORMAT
            self.storage._append({"v": FORMAT, "kind": "policy", "question": q, "config": self.config(),
                                  "policy": tag_floats(_stored(report))})
        return report

    def agrees(self, a, b):
        """Are two answers the same? same(a, b) when given (overlapping quotes, numbers within a tolerance), else
        equal values. A same() that raises: not the same."""
        if self.same is None:
            return _vh(a) == _vh(b)
        try:
            return bool(self.same(a, b))
        except Exception:  # noqa: BLE001 — a comparison that fails is not an agreement
            return False

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
        replays; the draw recomputes (a keyed draw only when this dispatcher has the secret; without it the recorded
        draw is taken); the dispatch follows from the response, the recorded draw and the recorded spend, drift
        flag and expected cost of a run; the slow path's record replays and its cost recomputes; the answer and who
        gave it follow. With an adapter loop attached (`loop=`), a decision made under another version than the one
        served now is replayed with that version installed for the duration (`AdapterLoop.serving`), and the dispatch
        follows from the loop's recorded view of it (its coin and whether its rule answered alone)."""
        bad = []
        if d.config is not None and d.config != self.config():
            bad.append(("config", "the dispatcher is not configured as it was for this decision"))
        lr = d.learning or {}
        ctx = contextlib.nullcontext()
        if lr.get("served") is not None:
            if self.loop is None:
                bad.append(("learning", "a decision made with an adapter loop, and the dispatcher has none attached"))
            else:
                ctx = self.loop.serving(lr["served"])
        try:
            with ctx:
                rep1 = d.s1.trace.replay(self.system, trust_models=trust_models)
        except ValueError as e:                    # the version's files cannot be found or are not the recorded ones
            bad.append(("learning", f"version {lr.get('served')}: {e}"))
            rep1 = {"mismatches": []}
        bad += [("s1", f"{m[1]}: {m[2]}") for m in rep1["mismatches"]]
        if d.s1._system is None:
            d.s1._system = self.system
        if not d.keyed or self.keyed:     # a keyed draw is recomputed only with the secret; else the record stands
            draw = self.draw(d.s1.trace.init_hash, d.n, keyed=d.keyed)
            if abs(draw - d.draw) > 1e-12:
                bad.append(("draw", f"recorded {d.draw}, recomputed {draw}"))
        prob = sampled = False
        if (d.stream is None) != (self._stream is None):
            bad.append(("stream", "a stream record, and the dispatcher has no stream certificate" if d.stream is not None
                        else "no stream record, and the dispatcher has a stream certificate"))
        elif d.stream is not None:
            sam = d.stream.get("sample") or {}
            prob, sampled = bool(sam.get("probation")), bool(sam.get("sampled"))
            if (not d.keyed or self.keyed) and (self.coin(d.n, keyed=d.keyed) < float(sam.get("rho", 1.0))) != sampled:
                bad.append(("coin", f"recorded sampled={sampled}, the coin gives {not sampled}"))
        hold = None
        if d.learning is not None:
            hold, sampled = self._hold(d.learning)
        action, reasons = self.decide(d.s1, d.draw, d.spent_before, d.drift, d.expected, probation=prob,
                                      sampled=sampled, hold=hold)
        if action != d.action:
            bad.append(("dispatch", f"recorded {d.action}, the response gives {action}"))
        elif reasons != d.reasons[:len(reasons)]:
            bad.append(("reasons", "the recorded reasons are not those the response gives"))
        rep2 = None
        if d.s2 is not None:
            if self.slow is None:
                bad.append(("s2", "a slow-path record, and the dispatcher has no slow path"))
            else:
                rep2 = _replay_thought(self.slow, d.s2, d.s1.trace.init, self.question, trust_models)
                bad += [("s2 " + w, why) for w, why in rep2["mismatches"]]
                c = cost_of(d.s2.responses, self.price, d.s2.cost.ms, d.s2.generated())
                if _cost_key(c) != _cost_key(d.s2.cost):
                    bad.append(("cost", f"the slow path's recorded cost {d.s2.cost.to_dict()} is not what its record "
                                        f"gives {c.to_dict()}"))
        c1 = cost_of([d.s1], self.price, d.s1.ms)
        if "s1" in d.cost and _cost_key(c1) != _cost_key(d.cost["s1"]):
            bad.append(("cost", "System 1's recorded cost is not what its trace gives"))
        if d.action == action:
            r1 = d.s1[self.question]
            would1 = r1.answer if r1.status != "abstain" else _would(r1, d.s1, self.question)
            sl = self.slice_of(d.s1) if d.action == "think" else None
            if sl != d.slice:
                bad.append(("slice", f"recorded {d.slice!r}, the response gives {sl!r}"))
            if d.s2 is None and d.action in ("think", "check") and self._needs_slow(sl):
                bad.append(("s2", f"action {d.action} without the slow path's record"))
            ans, by = self._outcome(d.action, r1, would1, d.s2, [], sl)
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
        return {"decisions": self.n, "by": dict(self.counts), "spent": self.spent.to_dict(), "slow_runs": self._runs[0]}


def _oracle(th):
    """An oracle's verdict given outside the process: the "oracle" dict of a Thought's dict record, else {}."""
    rec = getattr(th, "record", None)
    o = rec.get("oracle") if isinstance(rec, dict) else None
    return o if isinstance(o, dict) else {}


def _json(x):
    """As stored: JSON data (tuples → lists)."""
    return json.loads(json.dumps(x))


def _per_round(expected, slow):
    """The expected cost of one more round of a refinement: the mean run's cost spread over its rounds."""
    k = max(1, int(getattr(slow, "steps", 1) or 1))
    return Cost(None if expected.usd is None else expected.usd / k, int(math.ceil(expected.calls / k)), expected.ms / k)


def _cost_key(c):
    return (None if c.usd is None else round(c.usd, 9), c.calls, c.input_tokens, c.output_tokens)


def _s1_signal(r):
    """System 1's signal for its own answer on the slice: the answer's confidence (kept when it abstained)."""
    c = getattr(r, "confidence", None)
    return float(c) if isinstance(c, (int, float)) and math.isfinite(c) else -math.inf


def _would(r, res=None, question=None):
    """What an abstained Result would have answered: its answer, else the most probable option, else the answer its
    rule or head recorded in the trace before a guarantee or a threshold held it back (None when unknown)."""
    if r is None:
        return None
    if r.answer is not None:
        return r.answer
    p = r.probs or {}
    if p:
        return max(p, key=lambda k: p[k])
    if res is not None and question is not None:
        from .runtime import MISSING
        rec = next((x for x in res.trace.records if x.name == f"answer:{question}"), None)
        v = getattr(rec, "value", None)
        if v is not None and v is not MISSING and getattr(rec, "error", None) is None:
            return getattr(v, "value", v) if hasattr(v, "probs") else v
    return None


def _stored(v):
    """A value as stored, dicts and lists all the way down (storage.plain writes "not stated" as its key)."""
    from .store import plain
    if isinstance(v, dict):
        return {str(k): _stored(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_stored(x) for x in v]
    return plain(v)


def _restored(v):
    """A stored value back: "not stated" (stored as its key) is solvi.Unknown again, in dicts and lists too."""
    from .catalog import NOT_STATED_KEY, Unknown
    if isinstance(v, str) and v == NOT_STATED_KEY:
        return Unknown
    if isinstance(v, dict):
        return {k: _restored(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_restored(x) for x in v]
    return v


def _vh(v):
    from .runtime import vhash
    from .store import plain
    return vhash(plain(v))


__all__ = ["AskPath", "Budget", "BudgetStop", "Cost", "Dispatched", "Dispatcher", "PATHS", "PROBATION_REASON",
           "PredictableDrawWarning", "SAMPLE_REASON", "Started",
           "RefinePath", "SearchPath", "SECRET_ENV", "SIGNALS", "SlowPath", "Thought", "cost_of", "price_of",
           "recorded_calls"]
