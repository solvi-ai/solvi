"""System: catalog + questions → ask(init_state) → answers with confidence, flow, computed_state, trace; fit / teach; storage
of responses and corrections (solvi.storage; journal= is a JSONL store)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

from .core import Catalog, Serial
from .heads import Head
from .provenance import model_info
from .runtime import MISSING, Record, Result, execute, now_ms, path_confidence, srepr, vhash
from .strategist import computable, plan


@dataclass
class _Prepared:
    state: dict
    known: dict | None
    rejected: list | None
    questions: list
    flow: object
    order: object
    policy: object
    textin: object = None               # a solvi.textin.TextRead (ask_text): its records go into the trace


@dataclass
class Response(Serial):
    """The answers of one ask, with the flow, the trace, every fact's value and the safeguards.

    A response keeps a strong reference to the System that answered it (for `report()` and `counterfactual()`: the
    questions, the answer heads, the replay). Deliberately not a weak one: `System(cat, qs).ask(state).report()` is common
    and the temporary System would be gone before the report runs. A response you keep for long keeps its System (and its
    models) alive; keep `res.to_dict()` or the stored record instead, or drop the reference with `res._system = None`
    (then pass `system=` to report / counterfactual)."""
    results: dict
    flow: object
    trace: object
    values: dict
    ms: float
    feasible: bool = True               # do the answers satisfy every applicable constraint?
    violations: list | None = None      # names of the constraints still broken (fixed answers that conflict)
    catalog: object = None              # the catalog that answered (for the audit)
    safeguards: list | None = None      # safeguard events of this response (see solvi.audit.collect)
    model_outputs: int = 0              # outputs produced by models in this response
    stored_id = None                    # its id in a TraceStorage once saved (System(storage=...) saves every ask)
    textin = None                       # ask_text: the solvi.textin.TextRead the question and state were read from
    _system = None                      # the System that answered (reports, counterfactuals; see the class docs)
    _heads = None                       # its answer heads (the audit)

    def __getitem__(self, q):
        return self.results[q]

    @property
    def confidence(self):
        """Overall confidence: the probability that every answered question is right — the product of the answers'
        confidences (errors taken as independent, so it is conservative when answers agree through constraints). Abstained
        questions are left out; see `complete`. 1.0 when nothing was answered by a model or a learned part."""
        import math
        return math.prod(r.confidence for r in self.results.values() if r.status != "abstain")

    @property
    def complete(self):
        """Did every question get an answer (none abstained)?"""
        return all(r.status != "abstain" for r in self.results.values())

    @property
    def weakest(self):
        """(question, confidence) of the least confident answer, or None when every question abstained."""
        got = [(q, r.confidence) for q, r in self.results.items() if r.status != "abstain"]
        return min(got, key=lambda t: t[1]) if got else None

    @property
    def not_stated(self):
        """Questions answered "not stated" (solvi.Unknown): answered — they count in `confidence` and `complete` — but
        the text does not state the value."""
        from .core import Unknown
        return [q for q, r in self.results.items() if r.answer is Unknown and r.status != "abstain"]

    @property
    def overall(self):
        """The whole response in one line of data: confidence, weakest answer, answered / abstained, complete, feasible;
        the questions answered "not stated"; and per answer kind the answered count and the product of their confidences."""
        ab = [q for q, r in self.results.items() if r.status == "abstain"]
        kinds = {}
        for r in self.results.values():
            if r.status != "abstain":
                k = kinds.setdefault(r.kind or "?", [0, 1.0])
                k[0] += 1
                k[1] *= r.confidence
        return {"confidence": self.confidence, "weakest": self.weakest, "answered": len(self.results) - len(ab),
                "abstained": ab, "complete": not ab, "feasible": self.feasible, "not_stated": self.not_stated,
                "by_kind": {k: {"answered": n, "confidence": c} for k, (n, c) in sorted(kinds.items())}}

    def to_dict(self):
        """model_dump(mode="json"): answers, flow, trace, values, safeguards as JSON-ready data (see Serial)."""
        return self.model_dump("json")

    lang = "en"                         # the language of rendering (System(lang=...)); not a field: never serialized or hashed

    @property
    def computed_state(self):
        """Each computed fact with its value; a quote's offsets; for anything not computed by plain code, its provenance
        (quoted by a model, decided, learned) and the model; errors, including rejected (ungrounded) model outputs."""
        return self.computed_state_text()

    def computed_state_text(self, lang=None):
        """computed_state in a language (solvi.i18n; default: the System's)."""
        from . import i18n
        lang = i18n.check(self.lang if lang is None else lang)
        t = i18n.t
        lines = []
        for r in self.trace.records:
            if r.kind in ("rule", "head"):
                continue
            v = "—" if r.value is MISSING else repr(r.value)
            extra = t("cs.quote", lang, s=r.quote[0], e=r.quote[1]) if r.quote else ""
            if r.confidence < 1:
                extra += t("cs.confidence", lang, c=f"{r.confidence:.2f}")
            if r.origin not in ("computed", "quoted"):
                extra += f"   {i18n.provenance(r.origin, lang)}"
                if r.probs:
                    extra += " (" + ", ".join(f"{k} {float(p):.2f}" for k, p in sorted(r.probs.items(), key=lambda t_: -t_[1])[:3]) + ")"
            if r.model is not None:
                extra += t("cs.model", lang, m=f"{r.model['type']} {r.model['id']} #{r.model['fp'][:8]}")
            if r.error:
                extra += t("cs.error", lang, err=i18n.msg(r.error, lang))
            lines.append(f"{r.name:24s} = {v}{extra}")
        return "\n".join(lines)

    def audit(self, question=None, lang=None):
        """What each answer rests on and which safeguards fired: given inputs → computed facts → quotes (offsets and quoted
        text) → model decisions (model, probabilities) → learned parts → checks, rule, constraints → answer; plus the share
        of the support that is deterministic. `print(res.audit())`, or `res.audit("q").to_dict()` for data.
        lang: the language str() renders in (solvi.i18n; default: the System's, "en"); the data is the same in every one.
        → an Audit of every answer, or the AnswerAudit of one question when `question` is given."""
        from .audit import build
        a = build(self, question, lang=self.lang if lang is None else lang)
        return a[question] if isinstance(question, str) else a

    def counterfactual(self, question, max_changes=2, over=None, target=None, domains=None, system=None,
                       max_evals=5000):
        """The smallest change of the given inputs that changes this answer: "approve if amount ≤ 1000 (now 1200)".
        Only the deterministic flow is re-run — every model-backed part (extractor, decision, learned head) is held at the
        proposal it recorded in this trace, and no model is called; the result says which parts were held.
        over: the given facts to change (default: those the question's flow reads); numbers and dates are searched for
        the nearest threshold crossing (bisection — exact for inputs the answer is monotone in), booleans, enums and
        Literal-typed inputs enumerated; `domains` = {fact: [values]} or {fact: (lo, hi)} adds or bounds a domain.
        max_changes: 1 or 2 inputs changed together (two only when no single change does it). target: an answer to reach
        (default: any other answer). → solvi.counterfactual.Counterfactuals (print it; .best; .to_dict())."""
        from .counterfactual import search
        return search(self, question, max_changes=max_changes, over=over, target=target, domains=domains, system=system,
                      max_evals=max_evals)

    def report(self, format="md", question=None, system=None, replay="trusted"):
        """A human-readable report of this decision for an auditor or a customer: each answer, what it rests on, the quotes
        highlighted in the source text with their offsets, the safeguards that fired, the guarantee line, the models'
        fingerprints, the trace's hashes and the replay status. format: "md" (Markdown), "html" (one self-contained page,
        every value escaped) or "data" (a dict). replay: "trusted" (default: deterministic steps re-run, model outputs
        verified from the record — no model is called), "full" (models re-run too) or False. See solvi.report."""
        from .report import decision, render
        return render(decision(self, question, system, replay), format)


class System:
    def __init__(self, catalog: Catalog, questions, journal: str | None = None, workers: int = 1, order: str = "default",
                 producers: str = "declared", learn: bool | None = None, inputs=None, strategist=None, storage=None,
                 timeout: float | None = None, costs="declared", lang: str = "en"):
        """order: "default" (hard checks and their inputs first, all together) or "learned" (hard checks one at a time, most
        expected saving first — see learn_order). producers: "declared" (alternative producers of a fact are tried in
        declaration order) or "learned" (a policy picks the order per input and learns from outcomes). learn: after every
        ask, update part costs and the order / producer models from what happened (milliseconds).
        inputs: a pydantic model of init_state (optional): a dict passed to ask is validated against it — its fields (with
        defaults) are the given facts, a field that fails is left out and reported (safeguard type_rejected). ask also takes
        a BaseModel instance directly, with or without `inputs`.
        strategist: an object with plan(catalog, questions, init_keys, heads) → Flow used by ask instead of the deterministic
        strategist (experimental: solvi.strategy.ModelStrategist; its plan is recorded in the trace, see docs/strategist.md).
        storage: a solvi.storage.TraceStorage (or a path: .db / .sqlite → SQLite, else JSON lines) — every ask saves its
        response (answers, flow, whole trace) there, hash-chained across responses, and teach saves the correction; the
        response's `stored_id` is its id in the store. journal="file.jsonl" is the same as storage=JSONLStorage("file.jsonl").
        timeout: seconds a part's call may take under `aask` when the part declares no `timeout=` (None: no limit).
        producers="equivalent": the producers of a fact are interchangeable — the cost-optimal planner
        (solvi.strategy.ModelStrategist(producers="equivalent")) picks one per fact, the cheapest valid plan.
        costs: what that planner's costs are — "declared" (`cost=`, 1 when undeclared) or "measured" (the run times
        system.costs measures, after a warm-up; or a solvi.learned.MeasuredCosts with its settings). freeze_costs() fixes
        them; the plan record in each trace says which cost decided each choice.
        lang: the language of what solvi renders for people — res.audit(), solvi.show, safeguard_report() — "en" (default)
        or "ru" (solvi.i18n). Only the rendering changes: traces, stored responses, hashes and `why` stay in English."""
        from . import i18n
        from .learned import CostBook, OrderModel, ProducerPolicy
        self.lang = i18n.check(lang)
        self.catalog = catalog
        self.inputs = inputs
        self.strategist = strategist              # None: the deterministic strategist (solvi.strategist.plan)
        self.questions = {q.name: self._typed_question(q) for q in questions}
        self.heads: dict[str, Head] = {}
        from .storage import JSONLStorage, open_storage
        if journal and storage is not None:
            raise ValueError("pass journal= or storage=, not both (journal= is a JSONL storage)")
        self.journal = Path(journal) if journal else None
        self.storage = JSONLStorage(self.journal) if journal else open_storage(storage)
        if self.storage is not None and self.storage.catalog is None:
            self.storage.catalog = self               # typed values of stored responses are restored with this system
        self.workers = workers                    # >1: independent steps run in parallel threads
        self.timeout = timeout                    # aask: default seconds per call of a part
        self.calib: dict[str, tuple] = {}         # question → (a, b): confidence' = σ(a·logit(confidence) + b)
        self.learned_rules = {}                   # question → RuleList (readable rules learned from examples)
        if order not in ("default", "learned"):
            raise ValueError('order must be "default" or "learned"')
        if producers not in ("declared", "learned", "equivalent"):
            raise ValueError('producers must be "declared", "learned" or "equivalent"')
        if producers == "equivalent":
            from .strategy import ModelStrategist
            if strategist is None:
                self.strategist = strategist = ModelStrategist(producers="equivalent")
            elif getattr(strategist, "producers", None) != "equivalent":
                raise ValueError('producers="equivalent" with a strategist: give it producers="equivalent" too')
        from .learned import MeasuredCosts
        if isinstance(costs, MeasuredCosts):
            self.cost_policy = costs
        elif costs == "measured":
            self.cost_policy = MeasuredCosts()
        elif costs == "declared":
            self.cost_policy = None
        else:
            raise ValueError('costs must be "declared", "measured" or a MeasuredCosts')
        if self.cost_policy is not None and getattr(self.strategist, "producers", None) != "equivalent":
            raise ValueError('costs="measured" needs the cost-optimal planner: System(..., producers="equivalent") or '
                             'strategist=ModelStrategist(producers="equivalent")')
        # online learning of order / producer choice costs time inside ask (periodic refits), so it is on only when a learned
        # policy will use it (or when asked explicitly); plain systems keep flat, predictable decision times
        if learn is None:
            learn = order != "default" or producers == "learned"
        self.order, self.producers, self.learn = order, producers, learn
        self.costs = CostBook()                   # moving average of each part's run time, ms
        if self.cost_policy is not None and self.cost_policy.alpha is not None:
            self.costs.alpha = self.cost_policy.alpha
        self.order_model = OrderModel()           # P(hard check fails | cheap facts)
        self.producer_policy = ProducerPolicy()   # which producer of a fact to try first
        from .audit import STATS
        self.stats = {k: 0 for k in STATS}        # lifetime counts: model outputs and the safeguards that caught them

    def _typed_question(self, q):
        """A question without an answer type takes it from its rule's return type; a typed rule must fit its question."""
        rule = self.catalog.rules.get(q.name)
        rt = rule.returns if rule is not None else None
        if q.answer is None:
            if rt is None:
                raise ValueError(f"question {q.name!r} has no answer type: pass answer=, or give its rule a return type "
                                 "(bool, Literal[...], an Enum, list[Literal[...]])")
            from dataclasses import replace
            from .core import Answer
            return replace(q, answer=Answer.from_type(rt))
        if rt is not None:
            from .typed import check_answer
            check_answer(rule, q)
        return q

    def _state(self, init_state):
        """init_state → (dict of given facts, [(fact, why)] rejected by `inputs`)."""
        if self.inputs is None and type(init_state) is dict:
            return init_state, None
        from .typed import state_of
        return state_of(init_state, self.inputs)

    def response_schema(self):
        """The JSON schema of this system's responses, with each question's answer as its closed set of options."""
        from .schema import response_schema
        return response_schema(self)

    # --- answers
    def ask(self, init_state, names=None, workers=None, order=None, store=True):
        """init_state: a dict of given facts, or a pydantic BaseModel instance (its fields). order: override the system's
        order for this ask — "default", "learned", or an object with p_fail(check, row) and row(vals, init_keys) (e.g. an
        oracle for experiments). store=False: do not save this response to the system's storage.
        An `async def` part is awaited in an event loop of its own, one call at a time: use `aask` for such catalogs."""
        t0 = now_ms()
        p = self._prepare(init_state, names, order)
        trace, vals = execute(self.catalog, p.flow, p.state, workers=workers or self.workers, order=p.order,
                              costs=self.costs, policy=p.policy, known=p.known)
        return self._respond(p, trace, vals, t0, store)

    # --- text in
    def entry_points(self, names=None):
        """The questions as entry points: each question's name, text and the typed input state it reads — every given fact
        its flow reads, with its type, description and whether the question needs it (the schemas `solvi serve` publishes).
        → [solvi.textin.EntryPoint]; `ep.tool()` is the function-calling form."""
        from .textin import entry_points
        return entry_points(self, names)

    def _textin(self, text, decider, textin, question):
        from .textin import TextIn, TextRead
        if isinstance(text, TextRead):                # its fields are re-derived with textin's specs when given
            return text if textin is None else dataclasses.replace(text, reader=textin)
        if textin is None:
            textin = TextIn(self, decider)
        elif decider is not None:
            raise ValueError("pass a decider or a TextIn, not both")
        return textin.read(text, question=question)

    def _prepare_text(self, read, order):
        from .textin import rederive
        read = rederive(self, read)                   # each value re-derived from its quote: a built TextRead is not trusted
        if read.question is not None:
            p = self._prepare(read.init_state(), [read.question], order)
        else:                                         # the entry point escalated: nothing is asked
            state, rejected = self._state(read.init_state())
            p = _Prepared(state, None, rejected, [], plan(self.catalog, [], state.keys(), self.heads), None, None)
        p.textin = read
        return p

    def ask_text(self, text, decider=None, *, textin=None, question=None, store=True, workers=None, order=None):
        """A free text → the answer of the question it asks, in one trace: a solvi.textin.TextIn (made from `decider`, or
        `textin=`) picks the entry point and reads its input fields with quotes, then the question is asked on that state.
        `text` may be a TextRead already (TextIn.read / update): each field is re-derived from its quote with the field's
        own parser arguments (those of `textin=`, else of the TextIn that read it) before it is used. question=: skip
        routing.
        The trace holds the text (init_state[read.source]), the entry-point decision and one record per field (kind
        "textin": quote, parser, the model that found it); the audit counts the fields as quoted by a model, not given.
        When the entry point escalates, nothing runs: the likely questions abstain (guard "escalated"). A required field the
        text does not state is not guessed: the question abstains for lack of it. `res.textin` is the TextRead
        (`res.textin.missing`, `res.textin.clarify()`)."""
        read = self._textin(text, decider, textin, question)
        t0 = now_ms()
        p = self._prepare_text(read, order)
        trace, vals = execute(self.catalog, p.flow, p.state, workers=workers or self.workers, order=p.order,
                              costs=self.costs, policy=p.policy, known=p.known)
        return self._respond(p, trace, vals, t0, store)

    async def aask_text(self, text, decider=None, *, textin=None, question=None, store=True, timeout=None,
                        speculate=False, order=None):
        """ask_text with aask (async parts awaited)."""
        from .runtime import aexecute
        read = self._textin(text, decider, textin, question)
        t0 = now_ms()
        p = self._prepare_text(read, order)
        trace, vals = await aexecute(self.catalog, p.flow, p.state, order=p.order, costs=self.costs, policy=p.policy,
                                     known=p.known, timeout=self.timeout if timeout is None else timeout,
                                     speculate=speculate)
        return self._respond(p, trace, vals, t0, store)

    async def aask(self, init_state, names=None, order=None, store=True, timeout=None, speculate=False):
        """`ask` on an event loop: `async def` parts (database lookups, HTTP APIs, model servers) are awaited, parts marked
        `blocking=True` run in worker threads (asyncio.to_thread), plain sync parts inline; steps whose inputs are ready run
        concurrently. The answers, records and hashes are those of `ask` on the same input: records are written in flow
        order after the run, whatever finished first.
        timeout: seconds per call of a part without its own `timeout=` (default: System(timeout=)); a call that does not
        finish in time fails with "timed out after ... s" (safeguard `timeout`), and the questions that need it abstain —
        a producer of a fact that times out is followed by the next one. A timeout cannot stop a plain sync part running
        inline: mark it blocking, or make it async.
        speculate=False: hard checks and the steps they read first, then what the open questions need (as `ask`: no call
        starts that `ask` would not make). speculate=True: every step starts once its inputs are ready, and a failed hard
        check cancels the pending calls that only the questions it settles needed (lower latency; some paid calls may start
        and be cancelled). Cancelling `aask` itself cancels every pending call."""
        from .runtime import aexecute
        t0 = now_ms()
        p = self._prepare(init_state, names, order)
        trace, vals = await aexecute(self.catalog, p.flow, p.state, order=p.order, costs=self.costs, policy=p.policy,
                                     known=p.known, timeout=self.timeout if timeout is None else timeout,
                                     speculate=speculate)
        return self._respond(p, trace, vals, t0, store)

    @property
    def is_async(self):
        """Does the catalog have parts that `aask` awaits (`async def`, or marked blocking=True)? `solvi serve` then
        answers with aask."""
        from .runtime import async_parts
        return bool(async_parts(self.catalog))

    def freeze_costs(self):
        """Fix the costs the planner uses (System(costs="measured")) at what was measured so far — a producer that never
        ran: its declared cost, else 1 — so the choice of producers stops changing; measuring goes on. → {producer: ms}"""
        return self._measured().freeze(self.costs, _producers(self.catalog))

    def unfreeze_costs(self):
        """Back to the measured costs (see freeze_costs)."""
        self._measured().frozen = None

    def _measured(self):
        if self.cost_policy is None:
            raise ValueError('costs are declared: freeze_costs needs System(..., costs="measured")')
        return self.cost_policy

    def _prepare(self, init_state, names, order):
        """What ask and aask share before running: the given facts, the questions, the flow, the order and the policy."""
        known = None
        if self.inputs is not None or type(init_state) is not dict:
            from .typed import field_types, is_model
            model = type(init_state) if is_model(init_state) else self.inputs
            init_state, rejected = self._state(init_state)
            if model is not None:                     # validated fields: typed parts reading them skip re-validation
                ft = field_types(model)
                known = {k: t for k, t in ft.items() if k in init_state}   # rejected fields are not in init_state
        else:
            rejected = None
        qs = [self.questions[n] for n in (names or self.questions)]
        if self.cost_policy is not None:              # costs from measurements (see MeasuredCosts)
            c, src = self.cost_policy.costs(self.costs, _producers(self.catalog))
            flow = self.strategist.plan(self.catalog, qs, init_state.keys(), self.heads, costs=c)
            _why_costs(self.catalog, flow, init_state.keys(), c, src, self.cost_policy.frozen is not None)
        else:
            flow = (plan if self.strategist is None else self.strategist.plan)(self.catalog, qs, init_state.keys(),
                                                                               self.heads)
        mode = self.order if order is None else order
        om = None if mode == "default" else (self.order_model if mode == "learned" else mode)
        policy = self.producer_policy if self.producers == "learned" else None
        return _Prepared(init_state, known, rejected, qs, flow, om, policy)

    def _respond(self, p, trace, vals, t0, store):
        """What ask and aask share after running: the trace's plan record and fingerprint, costs, answers, safeguards,
        storage."""
        init_state, rejected, qs, flow, policy = p.state, p.rejected, p.questions, p.flow, p.policy
        if rejected:
            trace.rejected = rejected
        if getattr(flow, "strategy", None) is not None and getattr(self.strategist, "record", True):
            from .strategy import plan_record         # the strategist's plan, hashed into the trace (solvi.strategy)
            _append(trace, plan_record(flow), len(flow.steps))
        if p.textin is not None:                      # ask_text: the entry point and the fields read from the text
            for rec in p.textin.records():
                _append(trace, rec, len(flow.steps))
        trace.fingerprint = self._fingerprint(flow)
        for name, ms in trace.timings.items():         # cost tracking is cheap: always on
            self.costs.observe(name, ms)
        if self.cost_policy is not None:
            self.cost_policy.observe(trace.timings)
        if self.learn:
            self._observe(trace, init_state, vals, policy)
        results, feasible, violations = self._results(qs, flow, trace, vals, p.textin)
        resp = Response(results, flow, trace, vals, now_ms() - t0, feasible, violations, self.catalog)
        resp._heads = self.heads                      # for the audit: which features a learned head could not use
        resp._system = self                           # for reports and counterfactuals (questions, answer heads, replay)
        resp.textin = p.textin
        if self.lang != "en":
            resp.lang = self.lang                     # rendering only (audit, show): nothing recorded depends on it
        self._count(resp)
        if self.storage is not None and store:
            self.storage.save(resp)                   # sets resp.stored_id
        return resp

    def _results(self, qs, flow, trace, vals, textin=None):
        """The answers from an executed flow: rules / hard checks / heads, calibration, constraints, the low-confidence
        safeguard → (results, feasible, violations). No side effects besides an answer head's record in the trace."""
        by = {r.name: r for r in trace.records}
        results = {}
        for q in qs:
            r = self._answer(q, flow, trace, vals, by)
            r.kind = q.answer.kind
            if q.name in self.calib and r.status == "ok":
                r.confidence = _platt(r.confidence, *self.calib[q.name])
            results[q.name] = r
        if textin is not None:
            _read_results(textin, results)
        feasible, violations = self._joint(results)
        for q in qs:                                  # low-confidence safeguard: abstain rather than answer unsure
            r = results[q.name]
            if q.min_confidence is not None and r.status == "ok" and r.confidence < q.min_confidence:
                results[q.name] = Result(None, r.confidence, f"low confidence {r.confidence:.2f} < {q.min_confidence}; "
                                         f"would have answered {r.answer!r} ({r.why})", "abstain", r.probs, r.provenance,
                                         r.source, "low_confidence", r.repaired, r.kind, r.evidence, r.extra)
        return results, feasible, violations

    def fingerprint(self):
        """What makes this system's decisions: {"catalog": the catalog's fingerprint (every part's code and declarations,
        see solvi.provenance.catalog_fingerprint), "questions": the questions' (answer types, thresholds, calibration),
        "parts": {part: fingerprint}, "models": {part or "answer:<question>": model fingerprint} for model-backed parts and
        answer heads}. Every trace records the catalog and question fingerprints and those of the parts in its flow
        (trace.fingerprint); the models it used are recorded with the steps they produced."""
        from .provenance import catalog_fingerprint, fingerprint
        c = catalog_fingerprint(self.catalog)
        models = {}
        for n, p in list(self.catalog.parts.items()) + [(p.name, p) for p in self.catalog.rules.values()]:
            for a in (p.alternatives or [p]):
                if a.model is not None:
                    models[n if a is p else f"{n} ({a.name})"] = fingerprint(a.model)
        for q, h in self.heads.items():
            models["answer:" + q] = fingerprint(h)
        return {"catalog": c["fp"], "questions": self._questions_fp(), "parts": dict(c["parts"]), "models": models}

    def _questions_fp(self):
        """The questions' fingerprint (answer types, min_confidence, checkpoints, calibration); cached while the question
        objects and the calibration are the same."""
        from .provenance import digest
        from .schema import dump
        key = (tuple((n, id(q)) for n, q in self.questions.items()), tuple(sorted(self.calib.items())))
        cached = getattr(self, "_qfp", None)
        if cached is None or cached[0] != key:
            cached = self._qfp = (key, digest(sorted((q.name, dump(q, "json")) for q in self.questions.values()), list(key[1])))
        return cached[1]

    def _fingerprint(self, flow):
        """trace.fingerprint: the catalog's and questions' fingerprints and those of the parts in this flow."""
        from .provenance import catalog_fingerprint
        c = catalog_fingerprint(self.catalog)
        return {"catalog": c["fp"], "questions": self._questions_fp(),
                "parts": {s.part.name: c["parts"][s.part.name] for s in flow.steps if s.part.name in c["parts"]}}

    def _count(self, resp):
        from .audit import STAT_KEYS, collect
        events, n_model = collect(resp, self.catalog)
        resp.safeguards, resp.model_outputs = events, n_model
        st = self.stats
        st["asks"] += 1
        st["model_outputs"] += n_model
        for r in resp.results.values():
            st["abstained" if r.status == "abstain" else "answers"] += 1
        seen = set()
        for e in events:                              # a rejected fact shared by several questions counts once
            key = (e["kind"], e["fact"], e["detail"])
            if key not in seen:
                seen.add(key)
                st[STAT_KEYS[e["kind"]]] += 1

    def safeguard_report(self, lang=None):
        """The lifetime stats as text: how many model outputs, and how many were caught by each safeguard.
        lang: solvi.i18n (default: the System's)."""
        from . import i18n
        from .audit import QUIET, STAT_KEYS
        lang = i18n.check(self.lang if lang is None else lang)
        st = self.stats
        w = i18n.width(["sg." + k for k in STAT_KEYS], lang, en=22)
        lines = [i18n.t("rp.head", lang, **{k: st[k] for k in ("asks", "answers", "abstained", "model_outputs")})]
        for k, key in STAT_KEYS.items():
            if k not in QUIET or st[key]:             # "evidence missing" is listed once it fires
                lines.append(f"  {i18n.label(k, lang):{w}s} {st[key]}")
        return "\n".join(lines)

    def _answer(self, q, flow, trace, vals, by):
        facts = flow.per_question.get(q.name, [])
        # hard checks: a false hard check in the question's flow decides the answer (the model cannot override it).
        # A hard check with `then` governs only the questions listed there (and those that name it as a checkpoint);
        # for other questions it is an ordinary failed check.
        in_flow = set(facts)
        for f in [n for n in self.catalog.parts if n in in_flow]:        # several failed: the first declared in the catalog decides
            part = self.catalog.parts.get(f)
            r = by.get(f)
            if part is not None and part.kind == "check" and part.hard and r is not None and r.value is False:
                if not governs(part, q):
                    continue
                if q.name in part.then:
                    return Result(q.answer.normalize(part.then[q.name]), 1.0, f"hard check {f} is false", "forced",
                                  provenance=r.origin, source=f, guard="hard_check")
                return Result(None, 0.0, f"hard check {f} is false and no answer is set for it", "abstain", source=f,
                              guard="hard_check")
        for f in [n for n in self.catalog.parts if n in in_flow]:        # a governing hard check that could not be evaluated:
            part, r = self.catalog.parts.get(f), by.get(f)                # the answer is unknown, never "passed"
            if part is not None and part.kind == "check" and part.hard and r is not None and r.value is MISSING \
                    and governs(part, q):
                return Result(None, 0.0, f"hard check {f} could not be evaluated: {r.error or 'no value'}", "abstain",
                              source=f, guard="hard_check")
        missing =[f for f in facts if f in by and by[f].value is MISSING]
        if flow.unresolved.get(q.name):
            return Result(None, 0.0, "cannot compute: " + ", ".join(flow.unresolved[q.name]), "abstain")
        rule = self.catalog.rules.get(q.name)
        soft_failed = []
        for f in facts:
            part = self.catalog.parts.get(f)
            if part is not None and part.kind == "check" and by.get(f) is not None and by[f].value is False:
                if not part.hard or not governs(part, q):
                    soft_failed.append(f)
        if rule is not None:
            r = by.get(rule.name)
            if r is not None and r.value is MISSING and r.error and r.probs is not None:
                from .provenance import classify
                g = classify(r.error)
                if g in ("escalated", "low_confidence", "instruction", "memory"):  # the model's answer step escalated: abstain
                    return Result(None, r.confidence, r.error, "abstain", dict(r.probs), r.origin,
                                  rule.func.__name__ if rule.func is not None else rule.name, g)
            if r is None or r.value is MISSING:
                from .provenance import TIMED_OUT       # a call that did not finish in time (aask): safeguard "timeout"
                late = any(TIMED_OUT in (by[f].error or "") for f in list(facts) + [rule.name]
                           if f in by and by[f].value is MISSING)
                from .provenance import classify
                grounding = r is not None and classify(r.error) == "grounding"     # a model rule's quote not in the text
                return Result(None, 0.0, "rule not computed: " + (r.error if r else "no step") +
                              (f"; missing {', '.join(missing)}" if missing else ""), "abstain",
                              guard="timeout" if late else "grounding" if grounding else None)
            pc = path_confidence(self.catalog, trace, rule.inputs)
            conf = min(pc, r.confidence)
            why = "; ".join(f"{x} = {srepr(vals.get(x))}" for x in rule.inputs)
            src = rule.func.__name__ if rule.func is not None else rule.name
            if r.value is None:                       # the rule itself declined to answer (e.g. a split vote): a deliberate abstention
                return Result(None, 0.0, f"the rule abstained (returned None); {why}", "abstain", provenance=r.origin,
                              source=src, guard="rule_abstained")
            if q.answer.primitive or q.require_evidence or (r.extra is not None and "evidence" in r.extra):
                return _resolved(q, r, pc, why, src, trace.init)
            try:
                return Result(q.answer.normalize(r.value), conf, why, probs=dict(r.probs or {}), provenance=r.origin,
                              source=src)
            except ValueError:
                return Result(None, 0.0, f"rule returned {srepr(r.value)}, not one of the answer options; {why}", "abstain",
                              provenance=r.origin, source=src, guard="outside_options")
        head = self.heads.get(q.name)
        if head is None:
            return Result(None, 0.0, "no rule and the answer head is not fitted (fit)", "abstain")
        lost = [f for f in head.features if f not in vals]
        if lost:                                      # a head never guesses from facts that could not be computed
            return Result(None, 0.0, "head features not computed: " + ", ".join(lost), "abstain")
        p = head.predict(vals)
        if not all(math.isfinite(v) for v in p.values()):   # a NaN / inf feature: the head cannot say, so it does not answer
            bad = [f for f in head.features if isinstance(vals.get(f), float) and not math.isfinite(vals[f])]
            return Result(None, 0.0, "the answer head gave no finite probabilities"
                          + (" (non-finite " + ", ".join(bad) + ")" if bad else ""), "abstain", source=type(head).__name__,
                          guard="missing_facts")
        if q.answer.kind == "multi":                  # p: option -> probability it applies
            a = tuple(o for o in q.answer.options if p[o] >= 0.5)
            base = min(max(v, 1 - v) for v in p.values())
        elif q.answer.kind == "ordinal":              # median of the distribution over ordered levels
            acc = 0.0
            for a in q.answer.options:
                acc += p[a]
                if acc >= 0.5:
                    break
            base = p[a]
        else:
            a = max(p, key=p.get)
            base = p[a]
        contrib = head.contributions(vals)
        why = ", ".join(f"{f} = {srepr(vals.get(f))} ({c:+.2f})" for f, c in sorted(contrib.items(), key=lambda t: -abs(t[1]))[:4])
        if soft_failed:
            why += "; failed checks: " + ", ".join(soft_failed)
        conf = base * path_confidence(self.catalog, trace, head.features)
        _append(trace, Record(step=0, kind="head", name="answer:" + q.name, inputs={f: vhash(vals[f]) for f in head.features},
                              value=a, confidence=base, provenance="learned", model=model_info(head), probs=dict(p)),
                len(flow.steps))
        return Result(a, conf, why, probs=p, provenance="learned", source=type(head).__name__)

    # --- the learned strategist
    def _observe(self, trace, init_state, vals, policy):
        row = None
        for r in trace.records:
            part = self.catalog.parts.get(r.name)
            if part is not None and part.kind == "check" and part.hard and isinstance(r.value, bool):
                if row is None:
                    row = self.order_model.row(vals, list(init_state))
                self.order_model.observe(r.name, row, r.value is False)
        if policy is not None:
            for fact, o in getattr(trace, "_outs", {}).items():
                if o.row is not None:
                    policy.observe(self.catalog.parts[fact], o.row, o.outcomes)

    def learn_order(self, examples=None, features=None):
        """Learn which hard checks tend to fail on which inputs, and switch this system to the learned order.
        examples: [init_state] — each runs only its hard checks and what they read (no early exit), which also measures their
        costs. Without examples, the models learned from past asks are used as they are (every ask feeds them).
        features: computed facts to use besides init_state (cheap ones: they are computed before the hard checks)."""
        import copy
        if features is not None:
            self.order_model.features = list(features)
        for st in examples or ():
            flow = plan(self.catalog, list(self.questions.values()), st.keys(), self.heads)
            names = {s.part.name: s for s in flow.steps}
            keep = set()

            def up(n):
                if n in names and n not in keep:
                    keep.add(n)
                    for x in names[n].part.inputs:
                        up(x)
            for s in flow.steps:
                if s.part.kind == "check" and s.part.hard:
                    up(s.part.name)
            for f in self.order_model.features:
                up(f)
            sub = copy.copy(flow)
            sub.steps = [s for s in flow.steps if s.part.name in keep]
            trace, vals = execute(self.catalog, sub, st, early_exit=False, costs=self.costs)
            self._observe(trace, st, vals, None)
        self.order = "learned"
        return self.order_model

    # --- constraints between answers: joint decoding
    def _joint(self, results, max_combos=50_000):
        import itertools
        import math
        cons = [c for c in self.catalog.constraints.values() if all(q in results for q in c.inputs)]
        if not cons:
            return True, []

        def ok(assign, c):
            if any(assign.get(q) is None for q in c.inputs):
                return True                           # an abstained answer: nothing to check
            try:
                return bool(c.func(**{q: assign[q] for q in c.inputs}))
            except Exception:  # noqa: BLE001
                return False
        current = {q: r.answer for q, r in results.items()}
        if all(ok(current, c) for c in cons):
            return True, []
        # candidates: learned answers may change (their distribution), rule / forced / abstained answers are fixed
        qs = sorted({q for c in cons for q in c.inputs})
        cands = {}
        for q in qs:
            r, at = results[q], self.questions[q].answer
            rc = None
            if at.kind == "rank" and r.status == "ok" and r.probs:
                from .primitives import rank_candidates
                rc = rank_candidates(at, r.probs)
            if rc is not None:
                cands[q] = rc
            elif r.status == "ok" and r.probs and at.kind not in ("span", "estimate", "rank"):
                if at.kind == "multi":
                    opts = []
                    for bits in itertools.product([0, 1], repeat=len(at.options)):
                        combo = tuple(o for o, b in zip(at.options, bits) if b)
                        lp = sum(math.log(max(1e-9, r.probs[o] if b else 1 - r.probs[o])) for o, b in zip(at.options, bits))
                        opts.append((combo, lp))
                else:
                    opts = [(o, math.log(max(1e-9, pr))) for o, pr in r.probs.items()]
                cands[q] = sorted(opts, key=lambda t: -t[1])
            else:
                cands[q] = [(r.answer, 0.0)]
        k = max(len(v) for v in cands.values())
        while k > 1 and math.prod(min(len(v), k) for v in cands.values()) > max_combos:
            k -= 1
        best = None
        for combo in itertools.product(*[cands[q][:k] for q in qs]):
            assign = dict(current)
            assign.update({q: a for q, (a, _) in zip(qs, combo)})
            if all(ok(assign, c) for c in cons):
                score = sum(lp for _, lp in combo)
                if best is None or score > best[0]:
                    best = (score, assign)
        if best is None:
            return False, [c.name for c in cons if not ok(current, c)]
        for q in qs:
            r = results[q]
            if best[1][q] != r.answer:
                was = r.answer
                r.answer = best[1][q]
                if r.probs:
                    r.confidence = (math.exp(dict(cands[q])[r.answer]) if self.questions[q].answer.kind in ("multi", "rank")
                                    else r.probs.get(r.answer, r.confidence))
                broken = [c.name for c in cons if q in c.inputs and not ok({**current}, c)]
                r.why += f"; changed from {was!r} to satisfy {', '.join(broken)}"
                r.repaired = (was, broken)
        return True, []

    # --- task-specific training
    def facts_for(self, init_state):
        """All computable facts (for head training): a "compute everything" flow without rules."""
        from .core import Question
        from .strategist import plan as _plan
        init_state = self._state(init_state)[0]
        q = Question("__all__", "", None)
        flow = _plan(self.catalog, [q], init_state.keys())
        flow.steps = [s for s in flow.steps if s.part.kind != "rule"]
        _, vals = execute(self.catalog, flow, init_state, early_exit=False)
        return vals

    def fit(self, question, examples):
        """examples: [(init_state, answer)] → the answer head and its features (and hence the question's flow)."""
        q = self.questions[question]
        examples = _learnable(q, examples)
        rows = [self.facts_for(s) for s, _ in examples]
        ans = [q.answer.normalize(a) for _, a in examples]
        keys = set(self._state(examples[0][0])[0].keys())
        cands = sorted(f for f in computable(self.catalog, keys) - keys)
        if q.answer.kind == "multi":
            self.heads[question] = MultiHead(q.answer.options, lambda: Head(["yes", "no"]),
                                             lambda h, ys: h.fit(rows, ys, cands)).fit(ans)
        else:
            self.heads[question] = Head(q.answer.options).fit(rows, ans, cands)
        return self.heads[question]

    def fit_fast(self, question, examples, features=None, lam=None):
        """Fast answer head (closed-form ridge, milliseconds): examples — [(init_state, answer)]. Features: the given facts
        (numbers, booleans, categories or vectors such as a document embedding), by default every computed fact. Unlike fit it
        keeps all features and learns online: every `teach` for this question updates it instantly."""
        from .fast import FastHead
        import time
        q = self.questions[question]
        examples = _learnable(q, examples)
        t0 = time.perf_counter()
        rows = [self.facts_for(s) for s, _ in examples]
        if features is None:
            keys = set(self._state(examples[0][0])[0].keys())
            features = sorted(f for f in computable(self.catalog, keys) - keys)
        ans = [q.answer.normalize(a) for _, a in examples]
        if q.answer.kind == "multi":
            head = MultiHead(q.answer.options, lambda: FastHead(["yes", "no"], lam=lam),
                             lambda h, ys: h.fit(rows, ys, list(features))).fit(ans)
        else:
            head = FastHead(q.answer.options, lam=lam).fit(rows, ans, list(features))
        head.fit_ms = (time.perf_counter() - t0) * 1000
        dropped = getattr(head, "dropped", None) or {}
        if dropped and features is not None:           # features the caller asked for explicitly must not vanish silently
            import warnings
            warnings.warn(f"fit_fast({question!r}): features not used — " +
                          "; ".join(f"{f}: {why}" for f, why in dropped.items()), stacklevel=2)
        self.heads[question] = head
        return head

    def learn_rule(self, question, examples, facts, **kw):
        """An answer rule learned from examples (solvi.rules.RuleList): a readable "if feature then answer" list, installed in the
        catalog as a regular rule (deterministic, replayable)."""
        from .core import Part
        from .rules import RuleList
        q = self.questions[question]
        rows = [self.facts_for(s) for s, _ in examples]
        rl = RuleList(facts, **kw).fit(rows, [q.answer.normalize(a) for _, a in examples])

        def learned(**args):
            return rl.predict(args)[0]
        learned.__name__ = f"learned_{question}"
        self.catalog.rules[question] = Part(kind="rule", name="answer:" + question, inputs=list(facts), func=learned,
                                            doc=str(rl), question=question, model=rl, provenance="learned")
        self.learned_rules[question] = rl
        return rl

    def calibrate(self, question, examples, truth):
        """Calibrate answer confidence (Platt scaling) on held-out examples: examples is [init_state], truth is [correct answer]."""
        import numpy as np
        xs, ys = [], []
        for st, y in zip(examples, truth):
            r = self.ask(st, [question])[question]
            if r.status != "ok":
                continue
            c = min(max(r.confidence, 1e-4), 1 - 1e-4)
            xs.append(np.log(c / (1 - c)))
            ys.append(float(r.answer == y))
        xs, ys = np.array(xs), np.array(ys)
        if len(xs) < 10 or ys.min() == ys.max():
            self.calib[question] = (1.0, float(np.log((ys.mean() + 1e-3) / (1 - ys.mean() + 1e-3))) if len(xs) else 0.0)
            return self.calib[question]
        a, b = 1.0, 0.0
        for _ in range(500):
            p = 1 / (1 + np.exp(-(a * xs + b)))
            ga, gb = ((p - ys) * xs).mean(), (p - ys).mean()
            a, b = a - 0.5 * ga, b - 0.5 * gb
        self.calib[question] = (float(a), float(b))
        return self.calib[question]

    def learning(self, storage=None, parts=None, ladder=None, gates=None, **options):
        """Learning from corrections with gates and rollback — experimental, off until you call this (see
        solvi.learning): → a Learning loop over `storage` (default: this system's) for the questions answered by the
        decision parts in `parts` (default: all of them). Labels come only from human corrections, outcomes and rule
        rejections; loop.run() proposes an update by the ladder (ladder=: fit_below, memory_below, adapter, memory), runs
        the gates (gates=: min_gain, min_holdout, tolerance, risk, max_change, shadow_limit, max_conflict,
        min_calibration, honesty) and promotes it only when all pass; loop.rollback(version) restores any promoted
        version. options: changelog= (another TraceStorage for the update records), holdout=0.3, calibration=0.2,
        gate_teach=True (teach only stores corrections while the loop is attached), harvest_rules=False."""
        from .learning import Learning
        return Learning(self, storage, parts, ladder, gates, **options)

    def teach(self, question, init_state, correct, *, source="human", by=None, of=None):
        """Human correction. A fast head (fit_fast) absorbs it at once; so does a model decision that answers the question
        (a solvi.decide decision part as the question's rule, or a rule passing a decided fact on): its per-option shift is
        updated. Any correction goes to the storage (journal) for the next fit. Returns the update time in ms when something learned
        at once, else None. source ("human", "outcome", "rule"), by (who) and of (the stored id of the decision it corrects)
        are stored with it (TraceStorage.save_correction). With a learning loop (System.learning(..., gate_teach=True))
        nothing learns at once: the correction is only stored, and the loop's gates decide whether it is learned."""
        from .decide import decision_of
        from .fast import FastHead
        from .storage import check_source
        check_source(source)
        ms = None
        init_state = self._state(init_state)[0]
        loop = getattr(self, "_learning", None)
        if loop is not None and loop.gate_teach:
            if self.storage is None:
                raise ValueError("a learning loop reads corrections from the storage: System(storage=...)")
            self.storage.save_correction(question, init_state, correct, source=source, by=by, of=of)
            return None
        head = self.heads.get(question)
        dec = decision_of(self.catalog, question)
        if isinstance(head, (FastHead, MultiHead)) and getattr(head, "online", True):
            ms = head.update(self.facts_for(init_state), self.questions[question].answer.normalize(correct))
        if dec is not None:
            ans = self.questions[question].answer.normalize(correct)
            try:
                label = dec.spec.label(ans)               # the decision's own label (a bool decision: "yes" / "no")
            except ValueError:
                label = None                              # an answer the decision cannot give (e.g. set by a hard check)
            if label is not None:
                vals = init_state if all(f in init_state for f in dec.facts) else self.facts_for(init_state)
                ms = dec.teach(dec.text_of(vals), label)
        if self.storage is not None:
            self.storage.save_correction(question, init_state, correct, source=source, by=by, of=of)
        return ms


class MultiHead:
    """A multi-label answer as one yes/no head per option (each learned with fit or fit_fast)."""

    def __init__(self, options, make, train):
        self.options, self.make, self.train = list(options), make, train
        self.heads = {}

    def fit(self, answers):
        for o in self.options:
            self.heads[o] = self.train(self.make(), ["yes" if o in a else "no" for a in answers])
        h = next(iter(self.heads.values()))
        self.features = sorted({f for hh in self.heads.values() for f in hh.features})
        self.online = hasattr(h, "update")
        self.loo_acc = getattr(h, "loo_acc", None)
        return self

    @property
    def cv_acc(self):
        return self.loo_acc

    def predict(self, row):
        return {o: h.predict(row)["yes"] for o, h in self.heads.items()}

    def contributions(self, row):
        out = {}
        for h in self.heads.values():
            for f, c in h.contributions(row).items():
                out[f] = out.get(f, 0.0) + c
        return out

    def update(self, row, answer):
        return sum(h.update(row, "yes" if o in answer else "no") for o, h in self.heads.items())


def _append(trace, rec, n_steps):
    """Add a record after the flow's steps (an answer head's decision), chained to the last one."""
    last = trace.records[-1] if trace.records else None
    rec.step = max(n_steps, last.step if last else 0) + 1
    rec.prev = last.hash if last else trace.init_hash
    rec.hash = vhash(rec.body())
    trace.records.append(rec)


def _resolved(q, r, pc, why, src, init):
    """An answer primitive (not stated, span, rank, estimate) or an answer with evidence, from the rule's record (see
    solvi.primitives)."""
    from .core import Unknown
    from .primitives import Rejected, fmt, resolve
    try:
        out = resolve(q.answer, r, init)
    except Rejected as e:
        return Result(None, 0.0, f"{e.why}; {why}", "abstain", dict(r.probs or {}), r.origin, src, e.guard)
    a, ev = out["answer"], out["evidence"]
    if q.require_evidence and a is not Unknown and not ev:
        return Result(None, 0.0, f"no supporting quote (require_evidence); would have answered "
                                 f"{fmt(a, q.answer.kind, out['extra'])}; {why}", "abstain", out["probs"], r.origin, src,
                      "evidence_missing", extra=out["extra"])
    if a is Unknown:
        why = f"not stated; {why}"
    return Result(a, min(pc, out["confidence"]), why, probs=out["probs"], provenance=r.origin, source=src, evidence=ev,
                  extra=out["extra"])


def _read_results(read, results):
    """ask_text: an escalated entry point → the likely questions abstain (guard "escalated"); an answer rests on the
    reading too, so its confidence is at most the entry point's and the read fields' (min, as along a flow); a question
    that abstains for lack of a required field says the text does not state it."""
    if read.question is None:
        rt = read.route
        for q in rt.get("candidates") or []:
            results[q] = Result(None, 0.0, f"the entry point is unsure, nothing was asked: {rt.get('escalated')}",
                                "abstain", dict(rt.get("probs") or {}), "decided", "textin", "escalated")
        return
    r = results.get(read.question)
    if r is None:
        return
    confs = [float(read.route.get("confidence", 1.0))] + [f.confidence for f in read.fields.values() if f.status == "read"]
    if r.status in ("ok", "forced"):
        r.confidence = min([r.confidence] + confs)
    elif r.status == "abstain" and read.missing:
        r.why = f"not stated in the text: {', '.join(read.missing)}; {r.why}"


def governs(part, question):
    """Does a failed hard check decide this question? Yes if the question is in its `then`, names it as a checkpoint, or the
    check has no `then` at all."""
    return not part.then or question.name in part.then or part.name in question.checkpoints


def _learnable(q, examples):
    """Examples a learned head can learn from: its kinds are yes/no, choice, ordinal and multi-label, and it does not
    answer "not stated" (examples answered Unknown are left out)."""
    from .core import PRIMITIVES, Unknown
    if q.answer.kind in PRIMITIVES:
        raise ValueError(f"question {q.name!r} is a {q.answer.kind}: answer it with a rule or a model decision (a learned "
                         "head answers yes/no, choice, ordinal and multi-label questions)")
    return [(s, a) for s, a in examples if a is not Unknown] if q.answer.unknown else examples


def _platt(c, a, b):
    import math
    c = min(max(c, 1e-4), 1 - 1e-4)
    return 1 / (1 + math.exp(-(a * math.log(c / (1 - c)) + b)))


def _producers(catalog):
    """Every producer in a catalog: plain parts and each alternative producer of a fact."""
    return [a for p in catalog.parts.values() for a in (p.alternatives if p.alternatives is not None else [p])]


def _why_costs(catalog, flow, init_keys, costs, src, frozen):
    """Record in the flow's strategy (and so in the trace's plan record) the costs behind each choice among producers."""
    s = getattr(flow, "strategy", None)
    if s is None or not s.get("choice"):
        return
    from .strategy import reachable, usable
    reach = reachable(catalog, set(init_keys))

    def say(n):
        how, runs = src.get(n, ("declared", 0))
        return f"{n} {how} {costs.get(n, 1.0):.3f} ms" + (f" ×{runs}" if runs else "")
    facts = {}
    for f, chosen in s["choice"].items():
        p = catalog.parts.get(f)
        if p is None or p.alternatives is None:
            continue
        us = usable(catalog, f, reach)
        if len(us) < 2:
            continue
        others = sorted((a.name for a in us if a.name != chosen), key=lambda n: costs.get(n, 1.0))
        facts[f] = {"chosen": chosen,
                    "producers": {a.name: {"ms": round(costs.get(a.name, 1.0), 3), "from": src.get(a.name, ("declared",))[0],
                                           "runs": src.get(a.name, (None, 0))[1]} for a in us},
                    "why": "cheapest plan: " + "; ".join(say(n) for n in [chosen] + others)}
    s["costs"] = {"mode": "frozen" if frozen else "measured", "facts": facts}

