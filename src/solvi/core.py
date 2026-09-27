"""solvi — a catalog of functions and checks, questions with typed answers.

A part's contract comes from its signature: argument names are the facts it reads, the function name is the fact it sets.
Part kinds: extract (pull a value from text, with a quote), fn (computation), check (test → bool), rule (answer to a question).
A part may be backed by a model (`model=`): its outputs are fuzzy, so they are grounded (a quote must be literally at its
offsets, a decision must be among its options) and the model's identity is recorded in the trace (see solvi.provenance)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Quote:
    """A value extracted from text, with its location (doc[start:end] is the supporting quote)."""
    value: Any
    start: int
    end: int
    source: str = "doc"
    confidence: float = 1.0


@dataclass
class Decision:
    """A model's choice among declared options, with probabilities. Return it from a model-backed part (or rule): the value is
    accepted only if it is one of the part's `options` (or of the probability keys), and downstream parts receive the plain
    value."""
    value: Any
    probs: dict = field(default_factory=dict)
    confidence: float | None = None       # default: probs[value] (1.0 without probabilities)

    @property
    def conf(self):
        if self.confidence is not None:
            return float(self.confidence)
        try:
            return float(self.probs.get(self.value, 1.0))
        except TypeError:                   # an unhashable value (a list for a multi-label decision)
            return 1.0


def unwrap(v):
    """A part's output → (plain value, quote (start, end, source) or None, confidence, probs or None)."""
    if isinstance(v, Quote):
        return v.value, (v.start, v.end, v.source), v.confidence, None
    if isinstance(v, Decision):
        return v.value, None, v.conf, dict(v.probs)
    return v, None, 1.0, None


@dataclass
class AnswerType:
    kind: str                       # yes_no | choice | ordinal | multi
    options: list
    descriptions: dict = field(default_factory=dict)   # option -> what it means (for people and for zero-shot scoring)

    def normalize(self, v):
        if self.kind == "multi":
            vals = [v] if isinstance(v, str) else list(v or [])
            bad = [x for x in vals if x not in self.options]
            if bad:
                raise ValueError(f"answer {bad!r} not among {self.options}")
            return tuple(o for o in self.options if o in set(vals))      # canonical: in option order, no duplicates
        if self.kind == "yes_no" and isinstance(v, bool):
            return "yes" if v else "no"
        if v not in self.options:
            raise ValueError(f"answer {v!r} is not one of {self.options}")
        return v

    def rank(self, v):
        """Position of an ordinal (or choice) answer: 0 for the first option."""
        return self.options.index(v)


def _opts(options):
    if isinstance(options, dict):
        return list(options), dict(options)
    return list(options), {}


class Answer:
    @staticmethod
    def yes_no() -> AnswerType:
        return AnswerType("yes_no", ["yes", "no"])

    @staticmethod
    def choice(options) -> AnswerType:
        """One of the options. `options` is a list, or a dict {option: description}."""
        return AnswerType("choice", *_opts(options))

    @staticmethod
    def ordinal(levels) -> AnswerType:
        """One of ordered levels, lowest first (e.g. ["low", "medium", "high"]). A learned head answers with the median of its
        distribution instead of the most likely level, so it never jumps over the middle."""
        return AnswerType("ordinal", *_opts(levels))

    @staticmethod
    def multi(options) -> AnswerType:
        """Any subset of the options (possibly empty), returned as a tuple in option order. A rule may return a list or set."""
        return AnswerType("multi", *_opts(options))


@dataclass
class Question:
    name: str
    text: str
    answer: AnswerType
    checkpoints: list = field(default_factory=list)   # parts required in every flow for this question
    uses: list | None = None                          # hint to the strategist: which facts matter (when there is no rule or fit)
    min_confidence: float | None = None               # an answer below this confidence abstains (a low-confidence safeguard)


@dataclass
class Part:
    kind: str                       # extract | fn | check | rule
    name: str                       # fact it sets (for rule: "answer:<question>")
    inputs: list
    func: Callable
    doc: str = ""
    hard: bool = False              # check only: hard check
    then: dict = field(default_factory=dict)   # check only: {question: answer} when the check is false
    question: str | None = None     # rule only
    cost: float | None = None       # declared cost in ms (a prior for scheduling until run times are measured)
    provides: str | None = None     # alternative producer: the fact it provides (its own name is the function name)
    validate: Callable | None = None   # alternative producer: accepts the value? validate(value, [its inputs by name])
    min_confidence: float | None = None  # a Quote / Decision below this confidence is rejected
    alternatives: list | None = None     # group of alternative producers of one fact, in declaration order
    features: Callable | None = None     # group only: cheap features of the input for the producer policy
    model: Any = None                    # the model behind the part (extractor, head, rule list, classifier, ...)
    provenance: str | None = None        # declared provenance kind (default: from the output and the model)
    options: list | None = None          # a model's closed set of outputs: anything else is rejected
    exact: bool | None = None            # a Quote's value must be literally the text at its offsets (default: if model-backed)

    def strict(self):
        """Must a Quote from this part be literally at its offsets? Yes for model-backed parts (unless exact=False)."""
        from .provenance import FUZZY
        return self.exact if self.exact is not None else (self.model is not None or self.provenance in FUZZY)


def _group_func(group):
    """The group as a plain function: alternatives in declaration order, the first accepted value wins (used by code that
    calls part.func directly; the executor runs groups itself, with the producer policy and the trace)."""
    def run(**args):
        why = []
        for alt in group.alternatives:
            try:
                v = alt.func(**{x: args[x] for x in alt.inputs})
            except Exception as e:  # noqa: BLE001
                why.append(f"{alt.name}: {type(e).__name__}")
                continue
            ok, reason = accepts(alt, v, args)
            if ok:
                return v
            why.append(f"{alt.name}: {reason}")
        raise ValueError("no producer accepted: " + "; ".join(why))
    run.__name__ = group.name
    return run


def ground(part, v, init_state=None):
    """Is a part's output grounded? → None, or the reason it is rejected. A Quote must lie inside its source text and, for a
    strict (model-backed) part, its value must be literally the text at its offsets (strings up to whitespace, numbers as
    written); a Decision (or any value of a part with `options`) must be among the options; min_confidence is enforced."""
    from .provenance import NOT_GROUNDED, OUTSIDE_OPTIONS, QUOTE_OUTSIDE, matches
    conf = 1.0
    if isinstance(v, Quote):
        conf = v.confidence
        if init_state is not None:
            src = init_state.get(v.source, "")
            if not (isinstance(src, str) and 0 <= v.start <= v.end <= len(src)):
                return QUOTE_OUTSIDE
            if part.strict() and v.value is not None and matches(v.value, src[v.start:v.end]) is False:
                return f"{NOT_GROUNDED}: {_short(v.value)!r} is not the text at [{v.start}:{v.end}] ({_short(src[v.start:v.end])!r})"
    elif isinstance(v, Decision) or part.options is not None:
        value = v.value if isinstance(v, Decision) else v
        opts = part.options if part.options is not None else list(v.probs)
        vals = list(value) if isinstance(value, (list, tuple, set)) else [value]
        if opts and any(x not in opts for x in vals):
            return f"decision {_short(value)!r} is {OUTSIDE_OPTIONS} {list(opts)}"
        if isinstance(v, Decision):
            conf = v.conf
    if part.min_confidence is not None and conf < part.min_confidence:
        return f"confidence {conf:.2f} < {part.min_confidence}"
    return None


def _short(v, n=60):
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def validated(part, value, args):
    """Run a part's `validate(value, [its inputs by name])` → None or the rejection reason."""
    from .provenance import VALIDATE
    if part.validate is None:
        return None
    names = list(inspect.signature(part.validate).parameters)[1:]
    try:
        ok = part.validate(value, **{x: (args[x].value if isinstance(args[x], Quote) else args[x]) for x in names if x in args})
    except Exception as e:  # noqa: BLE001
        return f"validate raised {type(e).__name__}"
    return None if ok else VALIDATE


def accepts(alt, v, args, init_state=None):
    """Does an alternative producer's output pass its checks? → (ok, reason). None means "not found"; otherwise the output
    must be grounded (see ground: a quote in the text — literally, for a model — a decision among the options,
    min_confidence) and pass `validate`, which gets the plain value plus any of the producer's inputs it names."""
    if v is None or (isinstance(v, (Quote, Decision)) and v.value is None):
        return False, "no value"
    why = ground(alt, v, init_state) or validated(alt, unwrap(v)[0], args)
    return (False, why) if why else (True, "accepted")


class Catalog:
    """Everything the system can do; the strategist decides which parts each question needs."""

    def __init__(self):
        self.parts: dict[str, Part] = {}
        self.rules: dict[str, Part] = {}
        self.constraints: dict[str, Part] = {}     # rules between answers of different questions

    def _add(self, kind, f, **kw):
        sig = inspect.signature(f)
        # a function made by a model (e.g. extractor.field(...)) says so itself: its model and provenance come with it
        if kw.get("model") is None and getattr(f, "__solvi_model__", None) is not None:
            kw["model"] = f.__solvi_model__
        if kw.get("provenance") is None and getattr(f, "__solvi_provenance__", None) is not None:
            kw["provenance"] = f.__solvi_provenance__
        if kind in ("fn", "rule") and kw.get("options") is None and getattr(f, "__solvi_options__", None) is not None:
            kw["options"] = list(f.__solvi_options__)          # a decision part brings its closed set of options
        if kw.get("provenance") is not None:
            from .provenance import GIVEN, KINDS
            if kw["provenance"] not in KINDS or kw["provenance"] == GIVEN:
                raise ValueError(f"provenance must be one of {[k for k in KINDS if k != GIVEN]}")
        kw = {k: v for k, v in kw.items() if v is not None or k in ("then",)}
        p = Part(kind=kind, name=f.__name__, inputs=list(sig.parameters), func=f, doc=(f.__doc__ or "").strip(), **kw)
        if p.provides is not None:
            return self._add_alternative(p)
        if kind == "rule":
            p.name = "answer:" + p.question
            self.rules[p.question] = p
        elif kind == "constraint":
            self.constraints[p.name] = p
        else:
            if p.name in self.parts:
                raise ValueError(f"part {p.name} is already in the catalog")
            self.parts[p.name] = p
        return f

    def _add_alternative(self, p):
        """Several parts may provide one fact: they form a group (a fallback chain) in the catalog under the fact's name."""
        fact = p.provides
        if p.name == fact:
            raise ValueError(f"an alternative producer of {fact} needs its own function name (e.g. {fact}_regex)")
        if fact in self.rules or fact in self.constraints:
            raise ValueError(f"{fact} is not a fact")
        g = self.parts.get(fact)
        if g is None:
            g = Part(kind=p.kind, name=fact, inputs=[], func=None, doc=f"alternative producers of {fact}", alternatives=[])
            g.func = _group_func(g)
            self.parts[fact] = g
        elif g.alternatives is None:
            if g.kind not in ("fn", "extract"):
                raise ValueError(f"{fact} is a {g.kind}; only fn and extract parts can have alternatives")
            first = Part(kind=g.kind, name=g.func.__name__ + "__declared", inputs=g.inputs, func=g.func, doc=g.doc,
                         cost=g.cost, provides=fact, validate=g.validate, min_confidence=g.min_confidence, model=g.model,
                         provenance=g.provenance, options=g.options, exact=g.exact)   # a plain producer declared first
                                                                                        # becomes the first alternative
            g = Part(kind=g.kind, name=fact, inputs=list(g.inputs), func=None, doc=f"alternative producers of {fact}",
                     alternatives=[first])
            g.func = _group_func(g)
            self.parts[fact] = g
        if any(a.name == p.name for a in g.alternatives):
            raise ValueError(f"producer {p.name} of {fact} is already in the catalog")
        if p.kind not in ("fn", "extract"):
            raise ValueError("only fn and extract parts can be alternative producers")
        g.alternatives.append(p)
        g.inputs = list(dict.fromkeys(x for a in g.alternatives for x in a.inputs))
        if any(a.kind == "extract" for a in g.alternatives):
            g.kind = "extract"
        return p.func

    def _deco(self, kind, f, **kw):
        if f is None:
            return lambda g: self._add(kind, g, **kw)
        return self._add(kind, f, **kw)

    def extract(self, f=None, *, provides=None, cost=None, validate=None, min_confidence=None, model=None, provenance=None,
                exact=None):
        """Extract a value from text (returns a Quote). With `provides="fact"` the function is one of several alternative
        producers of that fact: its output is used only if it passes `validate` / `min_confidence`, otherwise the next
        alternative runs. `cost` (ms) is a prior for scheduling until run times are measured.
        `model`: the model behind it (recorded in the trace with its fingerprint); a model's Quote must be literally the text
        at its offsets or it is rejected (`exact=False` turns that off, `exact=True` turns it on for hand-written code)."""
        return self._deco("extract", f, provides=provides, cost=cost, validate=validate, min_confidence=min_confidence,
                          model=model, provenance=provenance, exact=exact)

    def fn(self, f=None, *, provides=None, cost=None, validate=None, model=None, provenance=None, options=None,
           min_confidence=None):
        """A computation. `provides`, `validate`, `cost`: as for extract. A model-backed fn (`model=`) with `options` is a
        model decision: it returns a Decision (or a plain value) that must be one of the options, else it is rejected."""
        return self._deco("fn", f, provides=provides, cost=cost, validate=validate, model=model, provenance=provenance,
                          options=None if options is None else list(options), min_confidence=min_confidence)

    def check(self, f=None, *, hard=False, then=None, cost=None, model=None, provenance=None):
        if f is None:
            return lambda g: self._add("check", g, hard=hard, then=then or {}, cost=cost, model=model, provenance=provenance)
        return self._add("check", f)

    def features(self, fact):
        """Cheap features of the input for choosing among a fact's alternative producers (the learned producer policy):
        `@cat.features("total") def total_features(doc): return {"length": len(doc), "has_total": "TOTAL" in doc}`.
        Arguments must be among the producers' inputs; values are numbers, booleans or short strings."""
        def deco(f):
            g = self.parts.get(fact)
            if g is None or g.alternatives is None:
                raise ValueError(f"declare the alternative producers of {fact} before its features")
            extra = [x for x in inspect.signature(f).parameters if x not in g.inputs]
            if extra:
                raise ValueError(f"features of {fact} read {extra}, which no producer of {fact} reads")
            g.features = f
            return f
        return deco

    def rule(self, question, *, model=None, provenance=None):
        """The answer rule of a question. With `model=` the rule is a model's decision: it may return a Decision with
        probabilities; an answer outside the question's options abstains, as for any rule."""
        return lambda f: self._add("rule", f, question=question, model=model, provenance=provenance)

    def constraint(self, f):
        """A rule between answers: argument names are question names, it returns True when the answers fit together
        (e.g. `def unsafe_if_harm(verdict, harm): return harm == "none" or verdict == "unsafe"`). When learned answers break it,
        solvi picks the most probable combination that satisfies every constraint; answers from rules and hard checks stay."""
        return self._add("constraint", f)

    def producer(self, fact):
        return self.parts.get(fact)

    def alternative(self, fact, name):
        """One alternative producer of a fact, by its function name."""
        g = self.parts[fact]
        for a in g.alternatives or ():
            if a.name == name:
                return a
        raise KeyError(f"{name} is not a producer of {fact}")
