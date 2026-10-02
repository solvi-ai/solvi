"""System: catalog + questions → ask(init_state) → answers with confidence, flow, computed_state, trace; fit / teach; storage
of responses and corrections (solvi.storage)."""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from . import _deprecate
from .core import Catalog, Serial
from .provenance import model_info
from .runtime import MISSING, Record, Result, execute, now_ms, path_confidence, srepr, vhash
from .strategist import computable, plan

if TYPE_CHECKING:                                 # numpy loads with the heads, on first use: `import solvi` stays light
    from .heads import FastHead


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

    def __post_init__(self):
        if hasattr(self.trace, "records") and isinstance(self.results, dict):
            self.trace.answers = self.results      # replay(system) checks that these are the answers the trace gives

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

    def signature(self, alg="syndrome"):
        """The signature of this response's trace (solvi.signature.sign: the input, then every record) — a few numbers to
        keep next to the stored id; solvi.signature.locate(res, sig) later names the one record that changed."""
        from .signature import sign
        return sign(self, alg)

    @property
    def computed_state(self):
        """Deprecated (removed in 0.9): `state_text()`."""
        _deprecate.renamed("Response.computed_state", "Response.state_text()")
        return self.state_text()

    def computed_state_text(self, lang=None):
        """Deprecated (removed in 0.9): `state_text(lang)`."""
        _deprecate.renamed("Response.computed_state_text()", "Response.state_text()")
        return self.state_text(lang)

    def state_text(self, lang=None):
        """The computed state for people: each computed fact with its value; a quote's offsets; for anything not
        computed by plain code, its provenance (quoted by a model, decided, learned) and the model; errors, including
        rejected (ungrounded) model outputs — in a language (solvi.i18n; default: the System's). The facts as data:
        `res.values`."""
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


def _speculate_note(speculate, p):
    """aask(speculate=True) under a learned order: the hard checks run one at a time in that order, so nothing starts
    early — say so rather than ignore the option."""
    if speculate and p.order is not None:
        import warnings
        warnings.warn("aask(speculate=True) is ignored under a learned order (System(order=\"learned\"), learn_order() "
                      "or order=...): the hard checks run one at a time in the learned order, nothing starts early",
                      stacklevel=3)


def _clash(catalog, names, head):
    """The message for given facts named like parts of the catalog (a given value would replace the part)."""
    parts = catalog.parts
    own = [k for k in names if k in parts[k].inputs]  # `def savings(savings)`: the part reads the field it is named after
    return (head + " " + ", ".join(f"{k!r} (the catalog's {parts[k].kind} of that name)" for k in names)
            + ": a given fact cannot stand in for a part of the catalog — rename the input key or the part"
            + (f"; {', '.join(own)} {'reads its own name' if len(own) == 1 else 'read their own names'}: name a part "
               f"after what it computes (e.g. {own[0]}_points), not after the input it reads" if own else ""))


def _jsonl(path):
    from .storage import JSONLStorage
    return JSONLStorage(Path(path))


class System:
    inputs = _deprecate.attr("inputs", "input_model", "System")       # 0.7 names, removed in 0.9
    costs = _deprecate.attr("costs", "cost_book", "System")

    @_deprecate.kwargs(journal=("storage", _jsonl, "storage=JSONLStorage(path)"), inputs="input_model",
                       costs="cost_policy")
    def __init__(self, catalog: Catalog, questions, *, workers: int = 1, order: str = "default",
                 producers: str = "declared", learn: bool | None = None, input_model=None, strategist=None, storage=None,
                 timeout: float | None = None, cost_policy="declared", lang: str = "en", early_exit: bool = True):
        """order: "default" (hard checks and their inputs first, all together) or "learned" (hard checks one at a time, most
        expected saving first — see learn_order). producers: "declared" (alternative producers of a fact are tried in
        declaration order) or "learned" (a policy picks the order per input and learns from outcomes). learn: after every
        ask, update the order / producer models from what happened (which hard check failed on which input, which
        producer was accepted); default: on when order="learned" or producers="learned", else off (part costs are
        measured either way). learn_order() turns it on.
        input_model: a pydantic model of init_state (optional; `inputs=` in 0.7): a dict passed to ask is validated against it — its fields (with
        defaults) are the given facts, a field that fails is left out and reported (safeguard type_rejected). ask also takes
        a BaseModel instance directly, with or without `input_model`. A given fact cannot share a name with a part of the
        catalog (ask refuses such a key: the value would replace the part), so a model field named like a part is refused
        here.
        strategist: an object with plan(catalog, questions, init_keys, heads) → Flow used by ask instead of the deterministic
        strategist (experimental: solvi.strategy.ModelStrategist; its plan is recorded in the trace, see docs/strategist.md).
        storage: a solvi.storage.TraceStorage (or a path: .db / .sqlite → SQLite, else JSON lines) — every ask saves its
        response (answers, flow, whole trace) there, hash-chained across responses, and teach saves the correction; the
        response's `stored_id` is its id in the store. journal=path (deprecated, removed in 0.9) is
        storage=JSONLStorage(path).
        Every option after `questions` is keyword-only.
        timeout: seconds a part's call may take under `aask` when the part declares no `timeout=` (None: no limit).
        producers="equivalent": the producers of a fact are interchangeable — the cost-optimal planner
        (solvi.strategy.ModelStrategist(producers="equivalent")) picks one per fact, the cheapest valid plan.
        cost_policy: what that planner's costs are — "declared" (`cost=`, 1 when undeclared) or "measured" (the run times
        system.cost_book measures, after a warm-up; or a solvi.costs.MeasuredCosts with its settings; `costs=` in 0.7). freeze_costs() fixes
        them; the plan record in each trace says which cost decided each choice.
        early_exit: True (default) — when a hard check fails, the steps only the questions it settles needed are skipped
        (the expensive rest is not paid for); the decision's record then holds no values for them, and a part listed in
        `requires` may not have run. False — every step of the flow runs anyway: the answers are the same (the failed
        hard check still decides), `res.values` and the trace hold every fact and rule value, and the trace says so
        (`trace.early_exit` is False). `ask(..., early_exit=...)` overrides it for one ask.
        lang: the language of what solvi renders for people — res.audit(), solvi.show, safeguard_report() — "en" (default)
        or "ru" (solvi.i18n). Only the rendering changes: traces, stored responses, hashes and `why` stay in English."""
        from . import i18n
        from .costs import CostBook
        from .strategist import OrderModel, ProducerPolicy
        self.lang = i18n.check(lang)
        self.catalog = catalog
        for fact, name, lost in catalog.unreadable_validates():
            raise ValueError(f"validate of {name} reads {lost}, which no producer of {fact} takes as an input: validate "
                             f"gets the value and, by name, inputs of the fact's producers — as it is, it cannot run and "
                             f"every output of {name} would be rejected")
        inputs = self.input_model = input_model
        if inputs is not None:                        # a declared field named like a part could never be given (ask refuses
            from .typed import field_types           # the key): say it now, not at the first ask
            clash = sorted(k for k in field_types(inputs) if k in catalog.parts)
            if clash:
                raise ValueError(_clash(catalog, clash, f"System(input_model={getattr(inputs, '__name__', inputs)}) declares"))
        self.strategist = strategist              # None: the deterministic strategist (solvi.strategist.plan)
        questions = list(questions)
        dup = sorted({q.name for q in questions if sum(x.name == q.name for x in questions) > 1})
        if dup:                                       # the last one would silently win
            raise ValueError(f"two questions named {', '.join(map(repr, dup))}: question names are unique in a System")
        self.questions = {q.name: self._typed_question(q) for q in questions}
        for c in catalog.constraints.values():        # argument names are question names: one that is not (a typo) means
            lost = [x for x in c.inputs if x not in self.questions]   # the constraint would never apply, without a word
            if lost:
                raise ValueError(f"constraint {c.name} reads {', '.join(lost)}, which "
                                 f"{'is not a question' if len(lost) == 1 else 'are not questions'} of this system "
                                 f"({', '.join(self.questions)}): a constraint's arguments are question names, and it "
                                 "applies only when all of them are asked")
        self.heads: dict[str, FastHead] = {}
        from .storage import open_storage
        self.storage = open_storage(storage)
        if self.storage is not None and self.storage.catalog is None:
            self.storage.catalog = self               # typed values of stored responses are restored with this system
        self.workers = workers                    # >1: independent steps run in parallel threads
        self.timeout = timeout                    # aask: default seconds per call of a part
        self.early_exit = bool(early_exit)        # False: the whole flow runs although a hard check failed
        self.calib: dict[str, tuple] = {}         # question → (a, b): confidence' = σ(a·logit(confidence) + b)
        self.guards = {}                          # question → solvi.guarantee.QuestionGuard (System.guarantee)
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
        from .costs import MeasuredCosts
        costs = cost_policy
        if isinstance(costs, MeasuredCosts):
            self.cost_policy = costs
        elif costs == "measured":
            self.cost_policy = MeasuredCosts()
        elif costs == "declared":
            self.cost_policy = None
        else:
            raise ValueError('cost_policy must be "declared", "measured" or a MeasuredCosts')
        if self.cost_policy is not None and getattr(self.strategist, "producers", None) != "equivalent":
            raise ValueError('cost_policy="measured" needs the cost-optimal planner: System(..., producers="equivalent") or '
                             'strategist=ModelStrategist(producers="equivalent")')
        # online learning of order / producer choice costs time inside ask (periodic refits), so it is on only when a learned
        # policy will use it (or when asked explicitly); plain systems keep flat, predictable decision times
        if learn is None:
            learn = order != "default" or producers == "learned"
        self.order, self.producers, self.learn = order, producers, learn
        self.cost_book = CostBook()               # moving average of each part's run time, ms
        if self.cost_policy is not None and self.cost_policy.alpha is not None:
            self.cost_book.alpha = self.cost_policy.alpha
        self.order_model = OrderModel()           # P(hard check fails | cheap facts)
        self.producer_policy = ProducerPolicy()   # which producer of a fact to try first
        from .audit import STATS
        self.stats = {k: 0 for k in STATS}        # lifetime counts: model outputs and the safeguards that caught them
        for part in catalog.parts.values():       # a `then` answer outside the question's options would only show when the
            for qn, ans in (part.then or {}).items() if part.kind == "check" and part.hard else ():   # check fails
                if qn in self.questions:
                    try:
                        self.questions[qn].answer.normalize(ans)
                    except ValueError as e:
                        import warnings
                        warnings.warn(f"hard check {part.name}: `then` answers {qn!r} with {ans!r}: {e} — when the check "
                                      f"fails, {qn!r} will abstain", stacklevel=2)

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
        if self.input_model is None and type(init_state) is dict:
            return init_state, None
        from .typed import state_of
        return state_of(init_state, self.input_model)

    def response_schema(self):
        """The JSON schema of this system's responses, with each question's answer as its closed set of options."""
        from .schema import response_schema
        return response_schema(self)

    # --- answers
    @_deprecate.kwargs(names="questions")
    def ask(self, init_state, questions=None, *, workers=None, order=None, store=True, early_exit=None):
        """init_state: a dict of given facts, or a pydantic BaseModel instance (its fields). questions: the questions to
        ask — a name or a list of names (None: all; an unknown name raises KeyError; `names=` is its deprecated
        spelling). Every other option is keyword-only. order: override the system's
        order for this ask — "default", "learned", or an object with p_fail(check, row) and row(vals, init_keys) (e.g. an
        oracle for experiments). store=False: do not save this response to the system's storage.
        early_exit: None — the system's (System(early_exit=), True by default: after a failed hard check the steps only
        the settled questions needed are skipped). False — compute the whole flow anyway: the answers are the same, and
        `res.values` and the trace hold every fact and rule value of a decision a hard check forced (and every part
        listed in `requires`); the trace records it (`res.trace.early_exit`), and replay checks no step is missing.
        An `async def` part is awaited in an event loop of its own, one call at a time: use `aask` for such catalogs."""
        if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or workers < 1):
            raise ValueError(f"workers must be a positive int, not {workers!r}")
        t0 = now_ms()
        p = self._prepare(init_state, questions, order)
        trace, vals = execute(self.catalog, p.flow, p.state, workers=workers or self.workers, order=p.order,
                              costs=self.cost_book, policy=p.policy, known=p.known, early_exit=self._early(early_exit))
        return self._respond(p, trace, vals, t0, store)

    def _early(self, early_exit):
        return self.early_exit if early_exit is None else bool(early_exit)

    # --- text in
    def entry_points(self, questions=None):
        """The questions as entry points: each question's name, text and the typed input state it reads — every given fact
        its flow reads, with its type, description and whether the question needs it (the schemas `solvi serve` publishes).
        → [solvi.textin.EntryPoint]; `ep.tool()` is the function-calling form."""
        from .textin import entry_points
        return entry_points(self, questions)

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

    def ask_text(self, text, decider=None, *, textin=None, question=None, store=True, workers=None, order=None,
                 early_exit=None):
        """A free text → the answer of the question it asks, in one trace: a solvi.textin.TextIn (made from `decider`, or
        `textin=`) picks the entry point and reads its input fields with quotes, then the question is asked on that state.
        `text` may be a TextRead already (TextIn.read / update): each field is re-derived from its quote with the field's
        own parser arguments (those of `textin=`, else of the TextIn that read it) before it is used. question=: skip
        routing.
        The trace holds the text (init_state[read.source]), the entry-point decision and one record per field (kind
        "textin": quote, parser, the model that found it); the audit counts the fields as quoted by a model, not given.
        When the entry point escalates, nothing runs: the likely questions abstain (guard "escalated"). A required field the
        text does not state is not guessed: the question abstains for lack of it. `res.textin` is the TextRead
        (`res.textin.missing`, `res.textin.clarify()`). early_exit: as for `ask`."""
        read = self._textin(text, decider, textin, question)
        t0 = now_ms()
        p = self._prepare_text(read, order)
        trace, vals = execute(self.catalog, p.flow, p.state, workers=workers or self.workers, order=p.order,
                              costs=self.cost_book, policy=p.policy, known=p.known, early_exit=self._early(early_exit))
        return self._respond(p, trace, vals, t0, store)

    async def aask_text(self, text, decider=None, *, textin=None, question=None, store=True, timeout=None,
                        speculate=False, order=None, early_exit=None):
        """ask_text with aask (async parts awaited)."""
        from .runtime import aexecute
        read = self._textin(text, decider, textin, question)
        t0 = now_ms()
        p = self._prepare_text(read, order)
        _speculate_note(speculate, p)
        trace, vals = await aexecute(self.catalog, p.flow, p.state, order=p.order, costs=self.cost_book, policy=p.policy,
                                     known=p.known, timeout=self.timeout if timeout is None else timeout,
                                     speculate=speculate, early_exit=self._early(early_exit))
        return self._respond(p, trace, vals, t0, store)

    @_deprecate.kwargs(names="questions")
    async def aask(self, init_state, questions=None, *, order=None, store=True, timeout=None, speculate=False,
                   early_exit=None):
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
        and be cancelled). Cancelling `aask` itself cancels every pending call. Under a learned order speculate=True is
        ignored, with a UserWarning (the hard checks run one at a time in that order). An `async def` part is awaited
        whatever `blocking` says. early_exit: as for `ask` (False: every step runs, whatever the hard checks say)."""
        from .runtime import aexecute
        t0 = now_ms()
        p = self._prepare(init_state, questions, order)
        _speculate_note(speculate, p)
        trace, vals = await aexecute(self.catalog, p.flow, p.state, order=p.order, costs=self.cost_book, policy=p.policy,
                                     known=p.known, timeout=self.timeout if timeout is None else timeout,
                                     speculate=speculate, early_exit=self._early(early_exit))
        return self._respond(p, trace, vals, t0, store)

    @property
    def is_async(self):
        """Does the catalog have parts that `aask` awaits (`async def`, or marked blocking=True)? `solvi serve` then
        answers with aask."""
        from .runtime import async_parts
        return bool(async_parts(self.catalog))

    def freeze_costs(self):
        """Fix the costs the planner uses (System(cost_policy="measured")) at what was measured so far — a producer that never
        ran: its declared cost, else 1 — so the choice of producers stops changing; measuring goes on. → {producer: ms}"""
        return self._measured().freeze(self.cost_book, _producers(self.catalog))

    def unfreeze_costs(self):
        """Back to the measured costs (see freeze_costs)."""
        self._measured().frozen = None

    def _measured(self):
        if self.cost_policy is None:
            raise ValueError('costs are declared: freeze_costs needs System(..., cost_policy="measured")')
        return self.cost_policy

    def _prepare(self, init_state, names, order):
        """What ask and aask share before running: the given facts, the questions, the flow, the order and the policy."""
        known = None
        if self.input_model is not None or type(init_state) is not dict:
            from .typed import field_types, is_model
            model = type(init_state) if is_model(init_state) else self.input_model
            init_state, rejected = self._state(init_state)
            if model is not None:                     # validated fields: typed parts reading them skip re-validation
                ft = field_types(model)
                known = {k: t for k, t in ft.items() if k in init_state}   # rejected fields are not in init_state
        else:
            rejected = None
        clash = sorted(k for k in init_state if k in self.catalog.parts)
        if clash:                                     # the planner would take the given value for the part's own: a check
            raise ValueError(_clash(self.catalog, clash, "the input has"))   # named in the input would never run
        qs = [self.questions[n] for n in self._names(names)]
        flow = self._plan(qs, init_state.keys(), why_costs=True)
        if not (order is None or order in ("default", "learned") or hasattr(order, "p_fail")):
            raise ValueError(f'order must be "default", "learned" or an object with p_fail(check, row), not {order!r}')
        mode = self.order if order is None else order
        om = None if mode == "default" else (self.order_model if mode == "learned" else mode)
        policy = self.producer_policy if self.producers == "learned" else None
        return _Prepared(init_state, known, rejected, qs, flow, om, policy)

    def _names(self, names):
        """The questions to ask: None → all; a name, or a list of names, each one of this system's questions."""
        if names is None:
            return list(self.questions)
        names = [names] if isinstance(names, str) else list(names)
        lost = [n for n in names if n not in self.questions]
        if lost:
            raise KeyError(f"no question {', '.join(map(repr, lost))} in this system ({', '.join(self.questions)})")
        return names

    def _plan(self, questions, init_keys, why_costs=False):
        """The flow of these questions on these given facts, planned as `ask` plans it: by System(strategist=) (the
        deterministic solvi.strategist.plan when none is set), with measured costs under System(cost_policy="measured").
        Everything that plans for this system goes through here — ask, answers_of, facts_for / fit, learn_order, the input
        schemas of `solvi serve` and `solvi check` — so they all see the same flow. why_costs: note in the plan record
        which cost decided each choice (ask)."""
        if self.cost_policy is not None:              # costs from measurements (see MeasuredCosts)
            c, src = self.cost_policy.costs(self.cost_book, _producers(self.catalog))
            flow = self.strategist.plan(self.catalog, questions, init_keys, self.heads, costs=c)
            if why_costs:
                _why_costs(self.catalog, flow, init_keys, c, src, self.cost_policy.frozen is not None)
            return flow
        return (plan if self.strategist is None else self.strategist.plan)(self.catalog, questions, init_keys, self.heads)

    def _computable(self, init_keys):
        """The facts this system's strategist can compute from these given facts (with a strategist set, a fact needs one
        usable producer; the deterministic strategist needs the inputs of every alternative)."""
        if self.strategist is None:
            return computable(self.catalog, init_keys)
        from .strategy import reachable
        return reachable(self.catalog, init_keys)

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
            self.cost_book.observe(name, ms)
        if self.cost_policy is not None:
            self.cost_policy.observe(trace.timings)
        if self.learn:
            self._observe(trace, init_state, vals, policy)
        results, feasible, violations = self._results(qs, flow, trace, vals, p.textin)
        if self.guards:                               # calibrated thresholds with a promise (solvi.guarantee)
            from .guarantee import apply_guards
            apply_guards(self, results, trace, vals, flow)
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
        hard = [(n, p) for n, p in self.catalog.parts.items() if p.kind == "check" and p.hard]   # in catalog order
        pcache = {}                                   # the questions share one walk of path_confidence
        results = {}
        for q in qs:
            r = self._answer(q, flow, trace, vals, by, hard, pcache)
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

    def answers_of(self, trace, names, flow=None):
        """The answers a recorded trace gives under this system: nothing is re-run — the facts are the trace's, the hard
        checks, rules, heads, constraints and the low-confidence safeguard are applied to them as `ask` does → {question:
        Result}. names: the questions to answer; flow: the response's flow (planned again when not given)."""
        import copy
        qs = [self.questions[n] for n in names]
        if flow is None:
            flow = self._plan(qs, trace.init.keys())
        t = copy.copy(trace)
        t.records = list(trace.records)                # an answer head appends its record: to the copy
        vals = dict(trace.init)
        for r in trace.records:
            if r.value is not MISSING:
                vals[r.name] = r.value
        results = self._results(qs, flow, t, vals)[0]
        if self.guards:                               # the recorded guard verdicts, re-derived (solvi.guarantee)
            from .guarantee import apply_guards
            apply_guards(self, results, t, vals, flow, live=False)
        return results

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
        """The questions' fingerprint (answer types, min_confidence, required parts, calibration); cached while the questions'
        contents and the calibration are the same (a question changed in place — `q.min_confidence = 0.9` — changes
        it: the cache is keyed by what the questions hold, not by the objects)."""
        from .core import plain_json, question_data
        from .provenance import digest

        def data(q):                                  # as solvi.schema.dump(q, "json"); pydantic only for other values
            d = question_data(q)
            if plain_json(d):
                return d
            from .schema import jsonable
            return jsonable(d)

        def content(q):
            a = q.answer
            ans = None if a is None else (a.kind, tuple(map(repr, a.options)), repr(sorted(a.descriptions.items(), key=str)),
                                          a.unknown, a.k, tuple(a.bins or ()), a.coverage, a.unit, a.source, repr(a.type))
            return (q.text, ans, tuple(q.requires), tuple(q.uses or ()) if q.uses is not None else None,
                    q.min_confidence, q.require_evidence)
        key = (tuple((n, content(q)) for n, q in self.questions.items()), tuple(sorted(self.calib.items())),
               tuple(sorted((n, g.fingerprint()) for n, g in self.guards.items())))     # solvi.guarantee
        cached = getattr(self, "_qfp", None)
        if cached is None or cached[0] != key:
            cached = self._qfp = (key, digest(sorted((q.name, data(q)) for q in self.questions.values()), list(key[1]),
                                              *([list(key[2])] if key[2] else [])))
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

    def _answer(self, q, flow, trace, vals, by, hard, pcache=None):
        facts = flow.per_question.get(q.name, [])
        # hard checks: a false hard check in the question's flow decides the answer (the model cannot override it).
        # A hard check with `then` governs only the questions listed there (and those that require it);
        # for other questions it is an ordinary failed check. hard: the catalog's hard checks, in catalog order.
        in_flow = set(facts)
        hard = [(f, part) for f, part in hard if f in in_flow]
        for f, part in hard:                          # several failed: the first declared in the catalog decides
            r = by.get(f)
            if r is not None and r.value is False:
                if not governs(part, q):
                    continue
                if q.name in part.then:
                    try:
                        forced = q.answer.normalize(part.then[q.name])
                    except ValueError as e:           # a `then` answer that is not an answer of this question: the check
                        return Result(None, 0.0, f"hard check {f} is false and its `then` answer is not usable: {e}",   # still
                                      "abstain", source=f, guard="hard_check")                                         # decides
                    because = (r.extra or {}).get("reasons") if isinstance(r.extra, dict) else None   # refine.Fail
                    return Result(forced, 1.0, f"hard check {f} is false" + (": " + "; ".join(because) if because else ""),
                                  "forced",
                                  provenance=r.origin, source=f, guard="hard_check")
                return Result(None, 0.0, f"hard check {f} is false and no answer is set for it", "abstain", source=f,
                              guard="hard_check")
        for f, part in hard:                          # a governing hard check that could not be evaluated: the answer
            r = by.get(f)                             # is unknown, never "passed"
            if r is not None and r.value is MISSING and governs(part, q):
                return Result(None, 0.0, f"hard check {f} could not be evaluated: {r.error or 'no value'}"
                              + _caused_by(by, r), "abstain", source=f, guard="hard_check")
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
                              (f"; missing {', '.join(missing)}" if missing else "") + _caused_by(by, r), "abstain",
                              guard="timeout" if late else "grounding" if grounding else None)
            pc = path_confidence(self.catalog, trace, rule.inputs, pcache)
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
        conf = base * path_confidence(self.catalog, trace, head.features, pcache)
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
        costs. Without examples, the models learned from past asks are used as they are: asks feed them only while
        learning is on (System(order="learned"), producers="learned" or learn=True), so on a default System that never
        learned they are empty and every check counts as failing half the time. From this call on every ask feeds them
        (`self.learn` becomes True).
        features: computed facts to use besides init_state (cheap ones: they are computed before the hard checks)."""
        import copy
        if features is not None:
            self.order_model.features = list(features)
        for st in examples or ():
            flow = self._plan(list(self.questions.values()), st.keys())
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
            trace, vals = execute(self.catalog, sub, st, early_exit=False, costs=self.cost_book)
            self._observe(trace, st, vals, None)
        self.order = "learned"
        self.learn = True                             # the learned order goes on learning from the asks, as order="learned" does
        return self.order_model

    # --- constraints between answers: joint decoding
    def _joint(self, results, max_combos=50_000):
        import itertools
        import math
        cons = [c for c in self.catalog.constraints.values() if all(q in results for q in c.inputs)]
        if not cons:
            return True, []

        raised = {}                                   # constraint → the exception it raised on the answers as given

        def ok(assign, c, note=False):
            if any(assign.get(q) is None for q in c.inputs):
                return True                           # an abstained answer: nothing to check
            try:
                return bool(c.func(**{q: assign[q] for q in c.inputs}))
            except Exception as e:  # noqa: BLE001 — a constraint that raises counts as broken, and says so (below)
                if note:
                    raised[c.name] = f"{type(e).__name__}: {str(e)[:120]}"
                return False
        current = {q: r.answer for q, r in results.items()}
        if all([ok(current, c, True) for c in cons]):
            return True, []
        for c in cons:                                # the error is in the reason of every answer the constraint reads
            if c.name in raised:
                for q in c.inputs:
                    results[q].why += f"; constraint {c.name} raised {raised[c.name]} (it counts as broken)"
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
            broken = [c.name for c in cons if not ok(current, c)]
            total = math.prod(len(v) for v in cands.values())
            if total > max_combos:                    # not every combination was tried: say so, the answers stand as given
                for q in qs:
                    if len(cands[q]) > 1 and any(q in c.inputs for c in cons if c.name in broken):
                        results[q].why += (f"; not repaired: {', '.join(broken)} broken, and joint decoding tried only the "
                                           f"{k} most probable answer(s) of each question ({total:,} combinations of "
                                           f"{len(qs)} answers exceed its limit of {max_combos:,})")
            return False, broken
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
        """All computable facts (for head training): a "compute everything" flow without rules, planned by the system's
        strategist (so a fact `ask` computes around a dead-end producer is a feature candidate too)."""
        from .core import Question
        init_state = self._state(init_state)[0]
        q = Question("__all__", "", None)
        flow = self._plan([q], init_state.keys())
        flow.steps = [s for s in flow.steps if s.part.kind != "rule"]
        _, vals = execute(self.catalog, flow, init_state, early_exit=False)
        return vals

    def fit(self, question, examples, features=None, *, select=None, min_gain=0.0, lam=None, refit=2.0,
            refit_until=2000):
        """An answer head for a question without a rule, learned from examples — [(init_state, answer)]: a closed-form
        ridge head (solvi.heads.FastHead, milliseconds to seconds), which every `teach` for this question updates at once.

        features: the facts it may read, by default every fact computable from the examples' init_state keys, the given
        keys included. select: keep only the facts that help — greedy forward selection by the exact leave-one-out
        squared error (solvi.heads.select_features); a fact is kept while it lowers that error by more than `min_gain` ×
        the error of the answers' shares. The kept facts become the question's flow, so later requests compute only
        what the head reads. Default (None): select when `features` is not given, keep every fact listed when it is.
        lam, refit, refit_until: as FastHead (a refit keeps the selected facts; it does not choose again).
        `head.selection` says what each kept fact did to the error. Before 0.8, fit was a logistic head chosen by
        cross-validated accuracy (+1 point), which kept nothing on imbalanced questions; fit_fast was this without the
        selection (now fit(..., select=False))."""
        from .heads import FastHead, select_features
        import time
        q = self.questions[question]
        examples = _learnable(q, examples)
        t0 = time.perf_counter()
        rows = [self.facts_for(s) for s, _ in examples]
        explicit = features is not None
        keys = set(self._state(examples[0][0])[0].keys())
        if features is None:
            features = sorted(self._computable(keys))      # the given keys too
        features = list(features)
        select = not explicit if select is None else bool(select)
        ans = [q.answer.normalize(a) for _, a in examples]

        def make():
            return FastHead(["yes", "no"] if q.answer.kind == "multi" else q.answer.options, lam=lam, refit=refit,
                            refit_until=refit_until)

        def train(h, ys):
            if not select:
                return h.fit(rows, ys, features)
            chosen, path, unusable = select_features(rows, ys, h.options, features, min_gain)
            h.fit(rows, ys, chosen)
            h.dropped, h.selection = unusable, path       # dropped: facts that cannot be encoded (not: not chosen)
            return h
        head = MultiHead(q.answer.options, make, train).fit(ans) if q.answer.kind == "multi" else train(make(), ans)
        head.fit_ms = (time.perf_counter() - t0) * 1000
        dropped = getattr(head, "dropped", None) or {}
        if dropped and explicit:           # features the caller asked for explicitly must not vanish silently
            import warnings
            warnings.warn(f"fit({question!r}): features not used — " +
                          "; ".join(f"{f}: {why}" for f, why in dropped.items()), stacklevel=2)
        if not explicit:                   # and a head left without any feature says why
            self._warn_no_features(question, head, features, keys, ans, select)
        self.heads[question] = head
        return head

    def _warn_no_features(self, question, head, cands, keys, answers, select):
        """A head without features answers the same for every input: say so, with the reason (a UserWarning)."""
        if head.features:
            return
        import inspect
        import warnings
        from collections import Counter
        if not [c for c in cands if c not in keys]:     # the given keys alone did not do (see head.dropped)
            have, blocked = self._computable(keys), []
            for p in self.catalog.parts.values():
                missing = [x for x in p.inputs if x not in have]
                if p.name in have or not missing:
                    continue
                try:
                    defaults = [x for x in missing if inspect.signature(p.func).parameters[x].default is not inspect.Parameter.empty]
                except (TypeError, ValueError, KeyError):
                    defaults = []
                blocked.append(f"{p.name} needs {missing}" + (f" ({defaults}: a parameter with a default value is still a "
                                                              "fact the part reads — bind it in a closure instead)"
                                                              if defaults else ""))
            why = (f"no fact can be computed from the examples' inputs {sorted(keys)}"
                   + (": " + "; ".join(blocked[:5]) + (f"; and {len(blocked) - 5} more" if len(blocked) > 5 else "")
                      if blocked else " (the catalog has no parts over them)"))
            unusable = getattr(head, "dropped", None) or {}
            if unusable:
                why += "; the inputs themselves cannot be used — " + "; ".join(f"{f}: {w}" for f, w in
                                                                              list(unusable.items())[:5])
        elif not select or set(getattr(head, "dropped", None) or ()) >= set(cands):
            why = "none of the facts can be used — " + "; ".join(f"{f}: {w}" for f, w in list(head.dropped.items())[:5])
        else:
            top = Counter(str(a) for a in answers).most_common(1)[0][1] / max(1, len(answers))
            why = (f"none of the {len(cands)} facts lowered the leave-one-out error of the answers' shares (the most "
                   f"frequent answer is {top:.0%} of the examples): the selection kept nothing — the facts do not tell "
                   "the answers apart on these examples; select=False keeps every fact")
        warnings.warn(f"fit({question!r}): the head has no features — every input gets the same answer. {why}",
                      stacklevel=3)

    def fit_fast(self, question, examples, features=None, lam=None, refit=2.0, refit_until=2000):
        """Deprecated (0.8; removed in 0.9): `fit(question, examples, features, select=False, ...)` — the same ridge head
        on every feature, now built by fit."""
        import warnings
        warnings.warn("fit_fast is deprecated: use fit (fit_fast(...) is fit(..., select=False)); it will be removed in "
                      "0.9", DeprecationWarning, stacklevel=2)
        return self.fit(question, examples, features, select=False, lam=lam, refit=refit, refit_until=refit_until)

    @_deprecate.kwargs(facts="features")
    def learn_rule(self, question, examples, features, **kw):
        """An answer rule learned from examples (solvi.rulelist.RuleList): a readable "if feature then answer" list, installed in the
        catalog as a regular rule (deterministic, replayable). examples: [(init_state, answer)], at least one; features: the
        facts the list reads (`facts=` in 0.7) — parts of the catalog or given facts of the examples (a name that is neither raises, and
        nothing is installed)."""
        from .core import Part
        from .rulelist import RuleList
        q = self.questions[question]
        facts = [features] if isinstance(features, str) else list(features)
        if not examples:
            raise ValueError(f"learn_rule({question!r}): no examples to learn from")
        rows = [self.facts_for(s) for s, _ in examples]
        lost = [f for f in facts if f not in self.catalog.parts and not any(f in r for r in rows)]
        if lost or not facts:
            raise ValueError(f"learn_rule({question!r}): " + (f"{', '.join(lost)} is not a part of the catalog or a given "
                             "fact of the examples" if lost else "facts is empty") + " — the rule could never be computed")
        rl = RuleList(facts, **kw).fit(rows, [q.answer.normalize(a) for _, a in examples])

        def learned(**args):
            return rl.predict(args)[0]
        learned.__name__ = f"learned_{question}"
        self.catalog.replace_rule(Part(kind="rule", name="answer:" + question, inputs=list(facts), func=learned,
                                       doc=str(rl), question=question, model=rl, provenance="learned"))
        self.learned_rules[question] = rl
        return rl

    def calibrate(self, question, examples, truth=None):
        """Calibrate answer confidence (Platt scaling) on held-out examples [(init_state, correct answer)], as `fit` takes
        them (the answer written as for `fit` and `teach`: True / False for a yes/no question, an Enum member, ...); the
        0.7 form `calibrate(question, states, truth)` still works with a DeprecationWarning (removed in 0.9).
        → (a, b): confidence' = σ(a·logit(confidence) + b), applied to every later answer of the question.
        The held-out examples are run without the question's current calibration (so a second call fits the same thing
        again, not a correction of the first), are not saved to the storage and do not count in `stats`. Fewer than 10
        answered examples, all of them right (or wrong), or confidences that do not vary (a rule's answers: 1.0): only
        the shift b is fitted (a = 1) — the calibrated confidence is then the share of right answers."""
        import numpy as np
        q = self.questions[question]
        examples = list(examples)
        if truth is not None:
            _deprecate.renamed("System.calibrate(question, states, truth)",
                               "System.calibrate(question, [(state, answer), ...])")
            truth = list(truth)
            if len(truth) != len(examples):
                raise ValueError(f"calibrate: {len(examples)} examples and {len(truth)} correct answers")
            examples = list(zip(examples, truth))
        if not all(isinstance(e, tuple) and len(e) == 2 for e in examples):
            raise ValueError("calibrate: examples are [(init_state, correct answer), ...]")
        examples, truth = [s for s, _ in examples], [q.answer.normalize(y) for _, y in examples]
        was = self.calib.pop(question, None)          # raw confidences: not the previously calibrated ones
        xs, ys = [], []
        try:
            for st, y in zip(examples, truth):
                p = self._prepare(st, [question], None)
                trace, vals = execute(self.catalog, p.flow, p.state, workers=self.workers, order=p.order, costs=self.cost_book,
                                      policy=p.policy, known=p.known)
                r = self._results(p.questions, p.flow, trace, vals)[0][question]
                if r.status != "ok":
                    continue
                xs.append(_logit(r.confidence))
                ys.append(float(vhash(r.answer) == vhash(y)))
        except BaseException:
            if was is not None:
                self.calib[question] = was
            raise
        self.calib[question] = _platt_fit(np.array(xs), np.array(ys))
        return self.calib[question]

    def guarantee(self, question, examples=None, **kw):
        """A calibrated threshold with a stated promise on this question's answer, from labelled examples
        [(init_state, correct answer)]: max_risk= (P(answered alone and wrong) ≤ it, conformal risk control), max_error= (the
        error among the answers given alone ≤ it with probability ≥ 1 − delta, learn-then-test) or method="empirical";
        on the answer's confidence, a computed fact or any signal (signal=), one-sided (answer=), per group (groups=).
        Below the threshold the question abstains with the reason; the promise is recorded with every answer.
        examples=False removes it. → the calibration report. See solvi.guarantee.guard_question."""
        from .guarantee import guard_question
        return guard_question(self, question, examples, **kw)

    def learning(self, storage=None, parts=None, ladder=None, gates=None, **options):
        """Learning from corrections with gates and rollback — experimental, off until you call this (see
        solvi.learning): → a Learning loop over `storage` (default: this system's) for the questions answered by the
        decision parts in `parts` (default: all of them). Labels come only from human corrections, outcomes and rule
        rejections; loop.run() proposes an update by the ladder (ladder=: fit_below, memory_below, adapter, memory), runs
        the gates (gates=: min_gain, min_holdout, tolerance, risk, max_change, shadow_limit, max_conflict,
        min_calibration, honesty) and promotes it only when all pass; loop.rollback(version) restores any promoted
        version. options: changelog= (another TraceStorage for the update records), holdout=0.3, calibration=0.2,
        gate_teach=True (teach only stores corrections while the loop is attached)."""
        from .learning import Learning
        return Learning(self, storage, parts, ladder, gates, **options)

    @_deprecate.kwargs(source="label_source")
    def teach(self, question, init_state, correct, *, label_source="human", by=None, of=None):
        """Human correction. A fast head (fit_fast) absorbs it at once; so does a model decision that answers the question
        (a solvi.decide decision part as the question's rule, or a rule passing a decided fact on): its per-option shift is
        updated. Any correction goes to the storage for the next fit. Returns the update time in ms when something learned
        at once, else None. label_source ("human", "outcome", "rule"; `source=` in 0.7), by (who) and of (the stored id of the decision it corrects)
        are stored with it (TraceStorage.save_correction). With a learning loop (System.learning(..., gate_teach=True))
        nothing learns at once: the correction is only stored, and the loop's gates decide whether it is learned.
        An unknown question raises KeyError and an answer that is not one of the question's options ValueError, before
        anything is learned or stored; the answer is stored normalized (True → "yes"). When nothing learned at once and
        there is no storage, the correction is lost: a UserWarning says so."""
        from .decide import decision_of
        from .heads import FastHead
        from .storage import check_source
        source = label_source
        check_source(source)
        if question not in self.questions:
            raise KeyError(f"teach: no question {question!r} in this system ({', '.join(self.questions)})")
        correct = self.questions[question].answer.normalize(correct)   # ValueError: not one of the options
        ms = None
        init_state = self._state(init_state)[0]
        loop = getattr(self, "_learning", None)
        if loop is not None and loop.gate_teach:
            if self.storage is None:
                raise ValueError("a learning loop reads corrections from the storage: System(storage=...)")
            self.storage.save_correction(question, init_state, correct, label_source=source, by=by, of=of)
            return None
        head = self.heads.get(question)
        dec = decision_of(self.catalog, question)
        if isinstance(head, (FastHead, MultiHead)) and getattr(head, "online", True):
            ms = head.update(self.facts_for(init_state), correct)
        if dec is not None:
            try:
                label = dec.spec.label(correct)           # the decision's own label (a bool decision: "yes" / "no")
            except ValueError:
                label = None                              # an answer the decision cannot give (e.g. set by a hard check)
            if label is not None:
                vals = init_state if all(f in init_state for f in dec.facts) else self.facts_for(init_state)
                ms = dec.teach(dec.text_of(vals), label)
        if self.storage is not None:
            self.storage.save_correction(question, init_state, correct, label_source=source, by=by, of=of)
        elif ms is None:
            import warnings
            warnings.warn(f"teach({question!r}): nothing learned it — no online head (fit_fast) or model decision "
                          "answers this question, and the system has no storage to keep it for the next fit: the "
                          "correction is lost", stacklevel=2)
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


MISSING_INPUTS = "missing inputs: "


def _caused_by(by, r):
    """A step that could not run for lack of an input → "; caused by <part>: <its error>": the part further up whose
    own failure (it raised, its output was rejected) left the facts missing — else the reason would name only the
    last link, a check or a rule that never ran. "" when the step failed by itself, or nothing above it has an error."""
    roots, seen = {}, set()
    todo = [r] if r is not None and (r.error or "").startswith(MISSING_INPUTS) else []
    while todo:
        rec = todo.pop(0)
        for x in rec.error[len(MISSING_INPUTS):].split(", "):
            up = by.get(x)
            if up is None or x in seen or up.value is not MISSING or not up.error:
                continue
            seen.add(x)
            if up.error.startswith(MISSING_INPUTS):
                todo.append(up)
            else:
                roots[x] = up.error
    return "; caused by " + " and ".join(f"{x}: {e}" for x, e in roots.items()) if roots else ""


def _resolved(q, r, pc, why, src, init):
    """An answer primitive (not stated, span, rank, estimate) or an answer with evidence, from the rule's record (see
    solvi.primitives)."""
    from .core import Unknown
    from .primitives import NO_EVIDENCE, Rejected, fmt, resolve
    try:
        out = resolve(q.answer, r, init)
    except Rejected as e:
        return Result(None, 0.0, f"{e.why}; {why}", "abstain", dict(r.probs or {}), r.origin, src, e.guard)
    a, ev = out["answer"], out["evidence"]
    if q.require_evidence and a is not Unknown and not ev:
        return Result(None, 0.0, f"no supporting quote (require_evidence); would have answered "
                                 f"{fmt(a, q.answer.kind, out['extra'])}; {why}", "abstain", out["probs"], r.origin, src,
                      NO_EVIDENCE, extra=out["extra"])
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
    """Does a failed hard check decide this question? Yes if the question is in its `then`, requires it, or the
    check has no `then` at all."""
    return not part.then or question.name in part.then or part.name in question.requires


def _learnable(q, examples):
    """Examples a learned head can learn from: its kinds are yes/no, choice, ordinal and multi-label, and it does not
    answer "not stated" (examples answered Unknown are left out)."""
    from .core import PRIMITIVES, Unknown
    if q.answer.kind in PRIMITIVES:
        raise ValueError(f"question {q.name!r} is a {q.answer.kind}: answer it with a rule or a model decision (a learned "
                         "head answers yes/no, choice, ordinal and multi-label questions)")
    return [(s, a) for s, a in examples if a is not Unknown] if q.answer.unknown else examples


def _logit(c):
    c = min(max(c, 1e-4), 1 - 1e-4)
    return math.log(c / (1 - c))


def _platt(c, a, b):
    return 1 / (1 + math.exp(-(a * _logit(c) + b)))


def _platt_shift(xs, ys):
    """Platt scaling with the slope fixed at 1: the shift b at which the mean calibrated confidence is the share of
    right answers (kept inside [0.001, 0.999]) — the maximum-likelihood b; found by bisection."""
    import numpy as np
    target = min(max(float(ys.mean()), 1e-3), 1 - 1e-3)
    lo, hi = -40.0, 40.0
    for _ in range(80):
        b = (lo + hi) / 2
        if float((1 / (1 + np.exp(-(xs + b)))).mean()) < target:
            lo = b
        else:
            hi = b
    return 1.0, (lo + hi) / 2


def _platt_fit(xs, ys):
    """(a, b) of confidence' = σ(a·logit(confidence) + b) by maximum likelihood (Newton steps, halved until the loss
    falls). Only the shift is fitted — see _platt_shift — when the slope cannot be told from the data: fewer than 10
    examples, every answer right (or wrong), confidences that do not vary, or a fit that does not settle."""
    import numpy as np
    if not len(xs):
        return 1.0, 0.0
    if len(xs) < 10 or ys.min() == ys.max() or float(xs.std()) < 1e-6:
        return _platt_shift(xs, ys)

    def nll(a, b):
        z = a * xs + b
        return float((np.logaddexp(0.0, z) - ys * z).mean())
    a, b = _platt_shift(xs, ys)
    loss = nll(a, b)
    for _ in range(100):
        p = 1 / (1 + np.exp(-(a * xs + b)))
        w = p * (1 - p)
        g = np.array([((p - ys) * xs).mean(), (p - ys).mean()])
        h = np.array([[(w * xs * xs).mean(), (w * xs).mean()], [(w * xs).mean(), w.mean()]]) + 1e-9 * np.eye(2)
        try:
            da, db = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            return _platt_shift(xs, ys)
        t = 1.0
        while t > 1e-6 and not nll(a - t * da, b - t * db) < loss:
            t /= 2
        if t <= 1e-6:
            break
        a, b = a - t * da, b - t * db
        was, loss = loss, nll(a, b)
        if was - loss < 1e-12:
            break
    if not (math.isfinite(a) and math.isfinite(b)) or abs(a) > 1e3 or abs(b) > 1e3:   # separable data: no finite optimum
        return _platt_shift(xs, ys)
    return float(a), float(b)


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

