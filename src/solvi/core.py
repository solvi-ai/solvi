"""solvi — a catalog of functions and checks, questions with typed answers.

A part's contract comes from its signature: argument names are the facts it reads, the function name is the fact it sets.
Part kinds: extract (pull a value from text, with a quote), fn (computation), check (test → bool), rule (answer to a question).
A part may be backed by a model (`model=`): its outputs are fuzzy, so they are grounded (a quote must be literally at its
offsets, a decision must be among its options) and the model's identity is recorded in the trace (see solvi.provenance).
Type hints are the facts' types (see solvi.typed): checked between producers and consumers when a part is registered,
validated / coerced with pydantic at run time; untyped parts cost nothing."""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from . import _deprecate


class Serial:
    """pydantic-backed export of solvi's data classes (see solvi.schema; pydantic is imported on first use):
    `x.model_dump(mode="python"|"json")`, `x.to_json()`, `Cls.model_validate(d)`, `Cls.from_json(s)`, `Cls.model_json_schema()`.
    Traces and responses take `catalog=` on load to restore typed values (dates, enums, models) from the facts' types."""

    def model_dump(self, mode="python"):
        from .schema import dump
        return dump(self, mode)

    def to_json(self, indent=None):
        return json.dumps(self.model_dump("json"), ensure_ascii=False, indent=indent, allow_nan=False)

    @classmethod
    def model_validate(cls, data, catalog=None):
        from .schema import load
        return load(cls, data, catalog)

    @classmethod
    def from_json(cls, s, catalog=None):
        return cls.model_validate(json.loads(s), catalog)

    @classmethod
    def model_json_schema(cls):
        from .schema import json_schema
        return json_schema(cls)


NOT_STATED_KEY = "<not stated>"          # how solvi.Unknown is written as a key (probabilities) and in JSON
NOT_STATED = "not stated"                # how a remote model is asked for it: an option of its own (llm, systemone)


class ExperimentalWarning(UserWarning):
    """A feature whose API and behaviour may still change (solvi.learning, solvi.lora)."""


def gone_in_1_0(old, new, owner=""):
    """A method or attribute that left its class in 1.0 (it would make a stable class import an experimental module, or
    it moves into the knowledge memory): reading it raises an AttributeError that names the call to use instead."""
    label = f"{owner}.{old}" if owner else old

    def gone(self, *_):
        raise AttributeError(f"{label} was removed in 1.0: use {new}")
    return property(gone, gone, doc=f"Removed in 1.0: use {new}.")


class NotStated:
    """The type of `solvi.Unknown`: the answer "the text does not state it". It is a real answer — the evidence says the
    input does not state the value, with a confidence — unlike an abstention (None: solvi refuses to answer). Declare it
    with `Maybe[T]` (or `T | NotStated`); constraints see it as `solvi.Unknown` (falsy; `x is Unknown`)."""
    _it = None

    def __new__(cls):
        if cls._it is None:
            cls._it = super().__new__(cls)
        return cls._it

    def __repr__(self):
        return "Unknown"

    def __str__(self):
        return NOT_STATED_KEY

    def __bool__(self):
        return False

    def __reduce__(self):
        return (NotStated, ())

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from pydantic_core import core_schema

        def check(v):
            if v is Unknown or v == NOT_STATED_KEY:
                return Unknown
            raise ValueError("not solvi.Unknown")
        return core_schema.no_info_plain_validator_function(
            check, serialization=core_schema.plain_serializer_function_ser_schema(lambda v: NOT_STATED_KEY))


Unknown = NotStated()


def unknown_key(k):
    """A probability key read back from JSON: "<not stated>" → Unknown."""
    return Unknown if k == NOT_STATED_KEY else k


@dataclass
class Quote:
    """A value extracted from text, with its location (doc[start:end] is the supporting quote). As evidence (Claim,
    Decision, Result.evidence) the value is the quoted text itself."""
    value: Any
    start: int
    end: int
    source: str = "doc"
    confidence: float = 1.0


@dataclass
class Claim:
    """A value with the quotes that support it — what a plain rule (or any part) returns to attach evidence and / or a
    confidence to its answer: `return Claim("billing", evidence=["charged twice"])`. Evidence items are Quotes (their
    offsets are checked: the text must be literally there) or strings (located in `source`: by default the part's only text
    input, or "doc" — as whole words and numbers: "3" is not evidence when the text says "30"); an item not in its text
    rejects the output (safeguard "grounding rejected"). The provenance stays the part's own (a hand-written rule:
    computed). `extra`: details recorded in the trace with the fact (`record.extra`) — a check's reasons
    (solvi.refine.Fail), a generator's request (solvi.generate.Generated)."""
    value: Any
    evidence: list = field(default_factory=list)
    confidence: float = 1.0
    source: str | None = None
    extra: dict = field(default_factory=dict)


@dataclass
class Decision:
    """A model's choice among declared options, with probabilities. Return it from a model-backed part (or rule): the value is
    accepted only if it is one of the part's `options` (or of the probability keys), and downstream parts receive the plain
    value."""
    value: Any
    probs: dict = field(default_factory=dict)
    confidence: float | None = None       # default: probs[value] (1.0 without probabilities)
    escalate: str | None = None           # the model hands this one to a person: why (the output is rejected, see ground)
    extra: dict = field(default_factory=dict)   # details recorded in the trace: act probability, expected level, shared pass
    evidence: list = field(default_factory=list)   # supporting quotes (Quote / str, as for Claim), checked like a Claim's

    @property
    def act(self):
        """Does the model act on this decision (False: it escalates)?"""
        return self.escalate is None

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
        if isinstance(v.value, Quote):            # a model's span: the quoted text, at its offsets
            q = v.value
            return q.value, (q.start, q.end, q.source), v.conf, dict(v.probs)
        return v.value, None, v.conf, dict(v.probs)
    if isinstance(v, Claim):
        if isinstance(v.value, Quote):
            q = v.value
            return q.value, (q.start, q.end, q.source), min(float(v.confidence), q.confidence), None
        return v.value, None, float(v.confidence), None
    return v, None, 1.0, None


def has_evidence(v):
    return isinstance(v, (Claim, Decision)) and bool(v.evidence)


def find_whole(text, t):
    """Where `text` is written in `t` as whole words and numbers → the start of its first such occurrence, or -1.
    An occurrence that begins or ends inside a longer word or number is not one: "3" is not found in "30", in "3.5" or
    in "1,300", "cat" not in "category" — but "30" is found in "30." and "30%", "charged twice" in "was charged twice,".
    (Only the ends are held to this: what the evidence itself contains is compared literally.)"""
    if not text:
        return -1
    n, i = len(text), t.find(text)
    while i >= 0:
        j = i + n
        left = not text[0].isalnum() or i == 0 or not (
            t[i - 1].isalnum() or (text[0].isdigit() and t[i - 1] in ".," and i > 1 and t[i - 2].isdigit()))
        right = not text[-1].isalnum() or j == len(t) or not (
            t[j].isalnum() or (text[-1].isdigit() and t[j] in ".," and j + 1 < len(t) and t[j + 1].isdigit()))
        if left and right:
            return i
        i = t.find(text, i + 1)
    return -1


def cuts_number(t, i, j):
    """Does t[i:j] begin or end inside a longer number written in t? A quote [26:27] of "30" is the "3" of "30", one of
    "3.5" or "1,300" is part of that number. (Words are not held to this: a quote with offsets may end inside a word.)"""
    if not 0 <= i < j <= len(t):
        return False
    left = t[i].isdigit() and i > 0 and (t[i - 1].isdigit() or (t[i - 1] in ".," and i > 1 and t[i - 2].isdigit()))
    right = t[j - 1].isdigit() and j < len(t) and (
        t[j].isdigit() or (t[j] in ".," and j + 1 < len(t) and t[j + 1].isdigit()))
    return left or right


def _sources(part, v, init_state):
    """The labelled texts a string of evidence may come from, in the order they are searched: Claim.source / the part's
    `source` when set (a name, or a list of names), else "doc" if the part reads it, else its only given text input, else
    "doc" — and after that (since 1.0) the part's other given text inputs, in argument order."""
    src = getattr(v, "source", None) or part.source
    if src is not None:
        return [src] if isinstance(src, str) else list(src)
    texts = [x for x in part.inputs if isinstance(init_state.get(x), str)]
    first = "doc" if ("doc" in texts or len(texts) != 1) else texts[0]
    return [first] + [x for x in texts if x != first]


def find_quote(quote, sources, *, quotes="normalized", whole=True):
    """Where a quote is in one or several labelled texts → Quote(the source's own substring, start, end, label) or None.
    sources: a text (labelled "doc") or {label: text}, searched in order ("notes", "dialogues", "map", ...); the first
    text that holds the quote wins. A literal occurrence is preferred in every text before the normalized view is tried
    (quotes="normalized", the default: Unicode NFKC, no-break spaces and hyphens, dash variants, "…" for "...",
    whitespace runs, curly quotes — see solvi.provenance.norm_view); quotes="literal" finds the text as written only.
    whole: as whole words and numbers ("3" is not found in "30")."""
    from .provenance import find_normalized, quote_mode
    quote_mode(quotes)
    srcs = {"doc": sources} if isinstance(sources, str) else dict(sources)
    q = str(quote)
    for label, t in srcs.items():
        if isinstance(t, str) and q:
            i = find_whole(q, t) if whole else t.find(q)
            if i >= 0:
                return Quote(q, i, i + len(q), label)
    if quotes == "normalized":
        for label, t in srcs.items():
            at = find_normalized(t, q, whole)
            if at is not None:
                return Quote(t[at[0]:at[1]], at[0], at[1], label)
    return None


def _kept(written, src, s, e, mode):
    """A quote with offsets whose written text is not literally the source text there (0.9's matches is False): in the
    normalized view, is it that text? → the source's own text at [s:e], or None."""
    if mode != "normalized" or not isinstance(written, str) or not isinstance(src, str) or not 0 <= s <= e <= len(src):
        return None
    from .provenance import matches_normalized
    t = src[s:e]
    return t if matches_normalized(written, t) else None


def _normalized_quote(part, v, init_state, mode, written):
    """A model-backed part's quoted value (a Quote, or one inside a Claim / Decision) written otherwise than its source
    text at its offsets, but the same in the normalized view → the output with the source's own text as its value
    (`written` gets what the part wrote). Anything else: as it is."""
    import dataclasses
    from .provenance import matches
    q = v.value if isinstance(v, (Claim, Decision)) and isinstance(v.value, Quote) else v
    if not isinstance(q, Quote) or not isinstance(q.value, str) or not part.strict():
        return v
    src = init_state.get(q.source)
    if not isinstance(src, str) or not 0 <= q.start <= q.end <= len(src) or matches(q.value, src[q.start:q.end]) \
            is not False:
        return v
    t = _kept(q.value, src, q.start, q.end, mode)
    if t is None:
        return v
    written["value"] = q.value
    q = dataclasses.replace(q, value=t)
    return dataclasses.replace(v, value=q) if q is not v and not isinstance(v, Quote) else q


def locate(part, v, init_state):
    """A Claim's / Decision's evidence with every item as a Quote: a string is located in the output's source text —
    Claim.source, else the part's `source`, else "doc" if the part reads it, else its only given text input, else "doc";
    then the part's other text inputs (several labelled sources: "notes", "dialogues", "map" — the Quote records which
    one) — at its first occurrence as whole words and numbers (find_whole: "3" does not quote "30"); a string that is
    not there becomes Quote(text, -1, -1, source), which ground rejects. An item given as a Quote keeps its own offsets.

    Quote matching (the part's `quotes`, Catalog(quotes=...)): "normalized" (default) — a quote that is not literally in
    its text is looked for, and compared at its offsets, in the normalized view (solvi.provenance.norm_view: NFKC,
    no-break spaces and hyphens, dashes, "…", whitespace runs, curly quotes); the Quote kept is then the source's own
    substring at offsets into the original text, and the output carries extra["quote_match"] = {"form", "written"} —
    what the part wrote — so the record says so and a replay re-checks it. A literal match never needs the view and
    leaves the output as it was. "literal": as 0.9 (the text as written, up to whitespace runs at given offsets).
    Outputs without evidence or quotes are returned as they are (no work)."""
    if init_state is None:
        return v
    mode = getattr(part, "quotes", "normalized")
    written = {}
    if isinstance(v, (Quote, Claim, Decision)):
        v = _normalized_quote(part, v, init_state, mode, written)
    if not has_evidence(v):
        return _noted(v, written)
    import dataclasses
    from .provenance import matches
    srcs = None
    out = []
    for k, e in enumerate(v.evidence):
        if isinstance(e, Quote):
            t = init_state.get(e.source)
            if e.value is None:
                e = dataclasses.replace(e, value=t[e.start:e.end] if isinstance(t, str) and 0 <= e.start <= e.end <= len(t)
                                        else "")
            elif isinstance(t, str) and 0 <= e.start <= e.end <= len(t) and matches(str(e.value), t[e.start:e.end]) \
                    is False:
                kept = _kept(str(e.value), t, e.start, e.end, mode)
                if kept is not None:
                    written[f"evidence {k}"] = str(e.value)
                    e = dataclasses.replace(e, value=kept)
            out.append(e)
            continue
        text = str(e)
        if srcs is None:
            srcs = _sources(part, v, init_state)
        t = init_state.get(srcs[0])                   # 0.9: the first source, literally — the same quote as before
        i = find_whole(text, t) if isinstance(t, str) else -1
        if i >= 0:
            out.append(Quote(text, i, i + len(text), srcs[0]))
            continue
        q = find_quote(text, {s: init_state.get(s) for s in srcs}, quotes=mode)
        if q is None:
            out.append(Quote(text, -1, -1, srcs[0]))
            continue
        if q.value != text:
            written[f"evidence {k}"] = text
        out.append(q)
    return _noted(dataclasses.replace(v, evidence=out), written)


def _noted(v, written):
    """An output whose quotes needed the normalized view, with that recorded: extra["quote_match"] (a bare Quote becomes
    a Claim of it, which unwraps to the same value, offsets and confidence)."""
    if not written:
        return v
    import dataclasses
    from .provenance import NORM_FORM
    note = {"quote_match": {"form": NORM_FORM, "written": written}}
    if isinstance(v, Quote):
        return Claim(v, confidence=v.confidence, extra=note)
    return dataclasses.replace(v, extra={**(v.extra or {}), **note})


def evidence_rows(v):
    """Located evidence → [[start, end, source, text]] (what the trace records in `extra["evidence"]`)."""
    return [[int(e.start), int(e.end), e.source, e.value] for e in v.evidence]


def check_evidence(evidence, init_state):
    """Every evidence Quote must lie in its (given) source text and be literally the text at its offsets → None or the
    rejection reason (a grounding reason: "quote outside the text" / "not grounded")."""
    from .provenance import NOT_GROUNDED, QUOTE_OUTSIDE, matches
    for e in evidence:
        src = init_state.get(e.source, "") if init_state is not None else ""
        if e.start < 0:
            return f"{NOT_GROUNDED}: evidence {_short(e.value)!r} is not in {e.source}"
        if not (isinstance(src, str) and 0 <= e.start <= e.end <= len(src)):
            return f"{QUOTE_OUTSIDE}: evidence [{e.start}:{e.end}] of {e.source}"
        if matches(str(e.value), src[e.start:e.end]) is False:
            return (f"{NOT_GROUNDED}: evidence {_short(e.value)!r} is not the text at {e.source}[{e.start}:{e.end}] "
                    f"({_short(src[e.start:e.end])!r})")
        if cuts_number(src, e.start, e.end):          # Quote("3", 26, 27) over "30": the text there is "3", but the
            return (f"{NOT_GROUNDED}: evidence {_short(e.value)!r} is not the text at {e.source}[{e.start}:{e.end}] "   # number
                    f"({_short(_number_around(src, e.start, e.end))!r})")                                               # is 30
    return None


def _number_around(t, i, j):
    """t[i:j] widened to the whole number it cuts (for the rejection message)."""
    while i > 0 and (t[i - 1].isdigit() or (t[i - 1] in ".," and i > 1 and t[i - 2].isdigit())):
        i -= 1
    while j < len(t) and (t[j].isdigit() or (t[j] in ".," and j + 1 < len(t) and t[j + 1].isdigit())):
        j += 1
    return t[i:j]


PRIMITIVES = ("span", "rank", "estimate")         # answer kinds resolved by solvi.primitives (with their value's details)


@dataclass
class AnswerType(Serial):
    kind: str                       # yes_no | choice | ordinal | multi | span | rank | estimate
    options: list
    descriptions: dict = field(default_factory=dict)   # option -> what it means (for people and for zero-shot scoring)
    unknown: bool = False           # "not stated" (solvi.Unknown) is a valid answer (Maybe[T])
    k: int | None = None            # rank: how many options the ranking returns (None: all)
    bins: list | None = None        # estimate: bin edges, ascending (the options are their labels); None: a plain number
    coverage: float | None = None   # estimate: the probability mass of the reported interval (default 0.8)
    unit: str | None = None         # estimate: the value's unit, for display
    source: str | None = None       # span: the given text fact the span lies in (default "doc")
    type: Any = None                # span: the value's type (the quoted text is coerced to it; None: str)

    @property
    def primitive(self):
        """Is this answer resolved with details (a span, a ranking, an estimate, or one that may be "not stated")?"""
        return self.unknown or self.kind in PRIMITIVES

    def normalize(self, v):
        if v is Unknown:
            if self.unknown:
                return v
            raise ValueError(f"answer Unknown (not stated) is not allowed: declare Maybe[...] for {self.kind}")
        if self.kind in PRIMITIVES:
            from .primitives import normalize
            return normalize(self, v)
        if isinstance(v, Enum):                        # an Enum answer (typed rule) → its value
            v = v.value
        if self.kind == "multi":
            vals = [v] if isinstance(v, str) else [x.value if isinstance(x, Enum) else x for x in (v or [])]
            bad = [x for x in vals if x not in self.options]
            if bad:
                raise ValueError(f"answer {bad!r} not among {self.options}")
            return tuple(o for o in self.options if o in set(vals))      # canonical: in option order, no duplicates
        if self.kind == "yes_no" and isinstance(v, bool):
            return "yes" if v else "no"
        if v not in self.options:
            raise ValueError(f"answer {v!r} is not one of {self.options}")
        return v

    rank = _deprecate.removed_attr("rank(v)", "options.index(v)", "AnswerType")


def _opts(options):
    """options (a list, or a dict {option: description}) → (options, descriptions); no options, or an option twice, raises
    ValueError (an answer type that can never answer, or a model scoring one option twice)."""
    if isinstance(options, (str, bytes)):
        raise ValueError(f"options are a list or a dict of options, not the string {options!r}")
    opts, desc = (list(options), dict(options)) if isinstance(options, dict) else (list(options), {})
    if not opts:
        raise ValueError("an answer type needs at least one option")
    dup = [o for i, o in enumerate(opts) if o in opts[:i]]
    if dup:
        raise ValueError(f"option {dup[0]!r} is given twice")
    return opts, desc


def _num(x):
    return str(int(x)) if float(x).is_integer() else repr(round(float(x), 6)).rstrip("0").rstrip(".")


def bin_labels(edges, integer=True, unit=None):
    """Bin edges e0 < … < e_last → the labels of their len(edges) + 1 bins (−∞, e0), [e0, e1), …, [e_last, ∞) — exactly the
    labels of the answer-primitives decider (its training code's `bin_labels`): "less than e0", then "a" (a one-wide integer
    bin), "a–(b−1)" (integers) or "a to b", then "e_last or more"; the unit after a space."""
    u = f" {unit}" if unit else ""
    out = [f"less than {_num(edges[0])}{u}"]
    for a, b in zip(edges[:-1], edges[1:]):
        if integer:
            out.append(f"{_num(a)}{u}" if b - a == 1 else f"{_num(a)}–{_num(b - 1)}{u}")
        else:
            out.append(f"{_num(a)} to {_num(b)}{u}")
    out.append(f"{_num(edges[-1])} or more{u}")
    return out


class Answer:
    @staticmethod
    def yes_no() -> AnswerType:
        return AnswerType("yes_no", ["yes", "no"])

    @staticmethod
    def maybe(answer_type) -> AnswerType:
        """The same answer type, with "not stated" (solvi.Unknown) as a valid answer — distinct from "no" and from an
        abstention. `Maybe[T]` in a type hint does the same."""
        import dataclasses
        return dataclasses.replace(answer_type, unknown=True)

    @staticmethod
    def span(source="doc", type=None) -> AnswerType:
        """An exact substring of a given text fact (`source`), with its offsets: always grounded (the text must be literally
        at the offsets), then coerced to `type` with pydantic (e.g. float: "12.50" → 12.5; a failure is "type rejected").
        The answer is the value; `result.span` is the Quote (text, start, end, source). `Span[float]` in a type hint."""
        return AnswerType("span", [], source=source, type=type)

    @staticmethod
    def rank(options, k=None) -> AnswerType:
        """An ordering of the options, best first: the top k (all by default) as a tuple, with a score per option
        (`result.scores`). A rule returns {option: score} (e.g. from a key function) or an ordered list; a model its
        probabilities (Plackett–Luce). `Rank[Literal[...], k]` in a type hint."""
        opts, desc = _opts(options)
        if k is not None and not 1 <= int(k) <= len(opts):
            raise ValueError(f"k must be between 1 and {len(opts)}")
        return AnswerType("rank", opts, desc, k=None if k is None else int(k))

    @staticmethod
    def estimate(bins=None, *, lo=None, hi=None, step=None, coverage=0.8, unit=None, integer=None) -> AnswerType:
        """A number with its uncertainty: a distribution over bins cut at the edges `bins` (ascending; or lo, hi, step) —
        len(bins) + 1 bins, the first and last open: (−∞, e0), [e0, e1), …, [e_last, ∞) — whose labels are the options
        ("less than 0", "0–6", "7–13", "14 or more" for integer edges; `integer=` overrides). The value is the middle of the
        median bin (an open bin: its edge), `result.interval` the bins holding the central `coverage` of the probability
        ((1 − c)/2 to (1 + c)/2 of the cumulative; None for an open end), and the confidence their probability mass. A rule
        returns a plain number (interval [x, x], confidence 1) or a distribution ({bin label or index: p}, or a list of p per
        bin); a model its probabilities over the bins as ordered options. Without bins the estimate is a plain number (rules
        only). `Estimate[0, 7, 14]` in a type hint."""
        if bins is not None and (lo, hi, step) != (None, None, None):
            raise ValueError("estimate takes bins, or lo=, hi= and step= — not both")
        if bins is None and (lo is not None or hi is not None) and step is None:
            raise ValueError("estimate with lo= and hi= needs step= (the width of a bin)")
        if bins is None and step is not None:
            if lo is None or hi is None:
                raise ValueError("estimate with step= needs lo= and hi=")
            if not float(step) > 0 or not hi > lo:
                raise ValueError("estimate needs lo < hi and step > 0")
            n = int(round((hi - lo) / step))
            bins = [lo + i * step for i in range(n + 1)]
        if bins is not None:
            bins = [b if isinstance(b, int) and not isinstance(b, bool) else float(b) for b in bins]
            if len(bins) < 1 or any(b >= c for b, c in zip(bins, bins[1:])):
                raise ValueError("estimate bins are ascending edges")
        if not 0 < float(coverage) < 1:
            raise ValueError("coverage is a probability between 0 and 1")
        if integer is None:
            integer = bins is not None and all(float(b).is_integer() for b in bins)
        labels = [] if bins is None else bin_labels(bins, integer, unit)
        return AnswerType("estimate", labels, bins=bins, coverage=float(coverage), unit=unit)

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

    @staticmethod
    def from_type(t, ordinal=False) -> AnswerType:
        """From a Python type: bool → yes_no; Literal[...] / an Enum → choice (`ordinal=True`: ordinal, in declaration
        order); list[Literal[...]] (or set, tuple, of an Enum) → multi. `X | None` is X (None = abstain)."""
        from .typed import answer_type_of
        return answer_type_of(t, ordinal)


@_deprecate.removed_kwargs(checkpoints="requires")
@dataclass
class Question(Serial):
    """A question the System answers: its name, text and answer type (None: from its rule's return type), the parts
    that must run in its flow (`requires`; `checkpoints=` in 0.7, removed in 0.9), the facts that matter for a question without a rule (`uses`), and the
    abstentions it asks for (`min_confidence`, `require_evidence`)."""
    name: str
    text: str
    answer: AnswerType | None = None                  # None: from the return type of the question's rule (Answer.from_type)
    requires: list = field(default_factory=list)      # parts required in every flow for this question
    uses: list | None = None                          # hint to the strategist: which facts matter (when there is no rule or fit)
    min_confidence: float | None = None               # an answer below this confidence abstains (a low-confidence safeguard)
    require_evidence: bool = False                    # an answer without supporting quotes abstains ("evidence missing")
    checkpoints = _deprecate.removed_attr("checkpoints", "requires", "Question")    # 0.7 name (not a field)


@dataclass
class Part:
    kind: str                       # extract | fn | check | rule
    name: str                       # fact it sets (for rule: "answer:<question>")
    inputs: list
    func: Callable | None
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
    source: str | None = None            # extract only: the given fact its quotes point into, when it is not "doc"
    types: dict | None = None            # typed arguments: {argument: type} (see solvi.typed); None — untyped
    returns: Any = None                  # the return type (the fact's type; for a Quote / Decision, of its value); None — untyped
    timeout: float | None = None         # System.aask: seconds a call may take (else System(timeout=)); then the step fails
    blocking: bool = False               # System.aask: a sync part that blocks (I/O, a model) runs in a worker thread
    quotes: str = "normalized"           # quote matching: "normalized" (1.0 default) or "literal" (0.9); Catalog(quotes=)
    then_parts: dict | None = field(default=None, repr=False, compare=False)   # check only: question → Part of a `then`
    #                                                                            function (its typed inputs; see check)
    tin: dict | None = field(default=None, repr=False, compare=False)   # compiled validators of the typed arguments
    tout: Any = field(default=None, repr=False, compare=False)          # ... and of the return type

    def strict(self):
        """Must a Quote from this part be literally at its offsets? Yes for model-backed parts (unless exact=False)."""
        from .provenance import FUZZY
        return self.exact if self.exact is not None else (self.model is not None or self.provenance in FUZZY)


def _then_parts(check):
    """A check's `then` functions as parts (question → Part): their argument names are the facts they read, their type
    hints validate / coerce those facts as for any part. Constants need none."""
    out = {}
    for q, fn in check.then.items():
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            raise ValueError(f"check {check.name}: the `then` function for {q!r} has no signature to read its facts "
                             "from") from None
        bad = [x for x, p in params.items() if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
        if bad:
            raise ValueError(f"check {check.name}: the `then` function for {q!r} takes *{bad[0]}: its arguments are "
                             "the facts it reads, by name")
        tp = Part(kind="fn", name=f"{check.name}.then[{q}]", inputs=list(params), func=fn)
        if getattr(fn, "__annotations__", None):
            from .typed import compile_part, hints
            tp.types, _ = hints(fn, "fn")
            if tp.types:
                compile_part(tp)
        out[q] = tp
    return out or None


def _group_func(group):
    """The group as a plain function: alternatives in declaration order, the first accepted value wins (used by code that
    calls part.func directly; the executor runs groups itself, with the producer policy and the trace)."""
    def run(**args):
        why = []
        for alt in group.alternatives:
            a = {x: args[x] for x in alt.inputs}
            if alt.tin is not None:
                from .typed import typed_in
                a, reason = typed_in(alt, a)
                if reason:
                    why.append(f"{alt.name}: {reason}")
                    continue
            try:
                from .runtime import resolved
                v = resolved(alt.func(**a))
            except Exception as e:  # noqa: BLE001
                why.append(f"{alt.name}: {type(e).__name__}")
                continue
            ok, reason = accepts(alt, v, {**args, **a} if alt.tin else args)
            if ok:
                return v
            why.append(f"{alt.name}: {reason}")
        raise ValueError("no producer accepted: " + "; ".join(why))
    run.__name__ = group.name
    return run


def _quote_source(f, sig, source):
    """The text an extract part's quotes point into (see Catalog.extract)."""
    if source is not None:
        if not isinstance(source, str) or not source:
            raise ValueError(f"extract {f.__name__}: source must be the name of a given fact (a text)")
        return source
    params = list(sig.parameters)
    if not params or "doc" in params:
        return "doc"
    if len(params) == 1:
        return params[0]
    texts = [x for x in params if sig.parameters[x].annotation in (str, "str")]
    if len(texts) == 1:
        return texts[0]
    raise ValueError(f"extract {f.__name__}({', '.join(params)}): cannot tell which argument is the text its quotes point "
                     f"into — pass it, e.g. @cat.extract(source={params[0]!r}), or annotate that one argument as str")


def _sourced(f, source):
    """f, with a returned Quote that kept the default source ("doc") pointed into `source` instead."""
    import dataclasses
    import functools

    def fix(v):
        if isinstance(v, Quote) and v.source == "doc":
            return dataclasses.replace(v, source=source)
        return v
    from .runtime import is_async_func
    if is_async_func(f):
        @functools.wraps(f)
        async def arun(*args, **kw):
            return fix(await f(*args, **kw))
        return arun

    @functools.wraps(f)
    def run(*args, **kw):
        return fix(f(*args, **kw))
    return run


def ground(part, v, init_state=None):
    """Is a part's output grounded? → None, or the reason it is rejected. A Quote must lie inside its source text and, for a
    strict (model-backed) part, its value must be literally the text at its offsets (strings up to whitespace, numbers as
    written); a Decision (or any value of a part with `options`) must be among the options; min_confidence is enforced."""
    from .provenance import NOT_GROUNDED, OUTSIDE_OPTIONS, QUOTE_OUTSIDE, matches
    conf = 1.0
    if has_evidence(v) and init_state is not None:
        why = check_evidence(v.evidence, init_state)
        if why:
            return why
    if isinstance(v, Claim):                  # a value with evidence: checked as the value itself
        conf = float(v.confidence)
        v = v.value
    elif isinstance(v, Decision) and isinstance(v.value, Quote):    # a model's span: escalation, then the quote
        if v.escalate:
            return v.escalate
        conf = v.conf
        v = v.value
    if isinstance(v, Quote):
        conf = min(conf, v.confidence)
        if init_state is not None:
            src = init_state.get(v.source, "")
            if not (isinstance(src, str) and 0 <= v.start <= v.end <= len(src)):
                return QUOTE_OUTSIDE
            if part.strict() and v.value is not None and (matches(v.value, src[v.start:v.end]) is False
                                                           or cuts_number(src, v.start, v.end)):
                return (f"{NOT_GROUNDED}: {_short(v.value)!r} is not the text at [{v.start}:{v.end}] "
                        f"({_short(_number_around(src, v.start, v.end))!r})")
    elif isinstance(v, Decision) or part.options is not None:
        value = v.value if isinstance(v, Decision) else v
        opts = part.options if part.options is not None else list(v.probs)
        if part.options is None and isinstance(v, Decision) and "interval" in v.extra:
            opts = None                       # a number: its value summarizes the distribution over the bins (checked as the answer)
        vals = list(value) if isinstance(value, (list, tuple, set)) else [value]
        if isinstance(v, Decision) and v.escalate and value is None:
            return v.escalate                 # no output at all (an LLM's invalid reply): its reason, not "outside"
        if opts and any(x not in opts for x in vals):
            return f"decision {_short(value)!r} is {OUTSIDE_OPTIONS} {list(opts)}"
        if isinstance(v, Decision):
            if v.escalate:                    # the model (or its calibrated threshold) escalates: rejected, like min_confidence
                return v.escalate
            conf = v.conf
    if part.min_confidence is not None and conf < part.min_confidence:
        return f"confidence {conf:.2f} < {part.min_confidence}"
    return None


def _short(v, n=60):
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def validate_misses(part, inputs):
    """The arguments a part's `validate` requires and cannot be given → [names]. `validate(value, ...)` gets, by name,
    inputs of the part — for an alternative producer, inputs of any producer of its fact (`inputs`: those names). An
    argument with a default, *args and **kwargs ask for nothing."""
    if part.validate is None:
        return []
    try:
        params = list(inspect.signature(part.validate).parameters.values())[1:]
    except (TypeError, ValueError):                     # a callable without a signature: nothing to check
        return []
    return [p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.default is p.empty
            and p.name not in inputs]


def validated(part, value, args):
    """Run a part's `validate(value, [its inputs by name])` → None or the rejection reason."""
    from .provenance import VALIDATE
    if part.validate is None:
        return None
    lost = validate_misses(part, args)
    if lost:                                            # else every output would be rejected as "validate raised TypeError"
        return f"validate cannot run: it reads {', '.join(lost)}, which is not an input of {part.provides or part.name}"
    names = list(inspect.signature(part.validate).parameters)[1:]
    try:
        ok = part.validate(value, **{x: (args[x].value if isinstance(args[x], Quote) else args[x]) for x in names if x in args})
    except Exception as e:  # noqa: BLE001
        return f"validate raised {type(e).__name__}"
    return None if ok else VALIDATE


def accept(alt, v, args, init_state=None):
    """Does an alternative producer's output pass its checks? → (ok, reason, plain value — coerced to its return type). None
    means "not found"; otherwise the output must be grounded (see ground: a quote in the text — literally, for a model — a
    decision among the options, min_confidence), pass its return type (typed parts) and `validate`, which gets the plain value
    plus any of the producer's inputs it names."""
    if v is None or (isinstance(v, (Quote, Decision, Claim)) and v.value is None):
        return False, "no value", None
    v = locate(alt, v, init_state)
    why = ground(alt, v, init_state)
    value = unwrap(v)[0]
    if not why and alt.tout is not None:
        from .typed import typed_out
        value, why = typed_out(alt, value)
    why = why or validated(alt, value, args)
    return (False, why, value) if why else (True, "accepted", value)


def accepts(alt, v, args, init_state=None):
    """accept() without the value → (ok, reason)."""
    ok, why, _ = accept(alt, v, args, init_state)
    return ok, why


def answer_data(at):
    """An answer type as plain data (its serialized form, solvi.schema; only what is set, so answer types of 0.4 dump
    as before)."""
    if at is None:
        return None
    # descriptions are keyed by the option's text (JSON keys are strings; an int option keeps its own type in `options`)
    d = {"kind": at.kind, "options": list(at.options), "descriptions": {str(k): v for k, v in at.descriptions.items()}}
    for k in ("unknown", "k", "bins", "coverage", "unit", "source"):
        v = getattr(at, k)
        if v not in (None, False):
            d[k] = list(v) if k == "bins" else v
    if at.type is not None:                           # a typed span: its name as the dump writes it (solvi.schema)
        from .schema import _span_type_name
        d["type"] = _span_type_name(at.type)
    return d


def question_data(q):
    """A question as plain data (its serialized form, solvi.schema) — without importing pydantic."""
    d = {"name": q.name, "text": q.text, "answer": answer_data(q.answer), "checkpoints": list(q.requires),
         "uses": None if q.uses is None else list(q.uses), "min_confidence": q.min_confidence}
    if q.require_evidence:
        d["require_evidence"] = True
    return d


def plain_json(v):
    """Is v JSON data as it is — str / int / bool / None, finite floats, lists and tuples, dicts with str keys — so that
    solvi.schema.jsonable would give it back unchanged (up to tuples as lists)?"""
    t = type(v)
    if v is None or t in (str, int, bool):
        return True
    if t is float:
        return math.isfinite(v)
    if t in (list, tuple):
        return all(plain_json(x) for x in v)
    if t is dict:
        return all(type(k) is str and plain_json(x) for k, x in v.items())
    return False


class Catalog:
    """Everything the system can do; the strategist decides which parts each question needs."""

    def __init__(self, *, quotes="normalized"):
        """quotes: how quotes and evidence are matched against their source text — "normalized" (default since 1.0: a
        quote that differs from its text only in Unicode form, no-break spaces or hyphens, dashes, "…", whitespace
        runs or curly quotes is found, and the source's own substring is kept, see solvi.core.locate) or "literal"
        (0.9: the text as written, up to whitespace runs). It applies to the parts registered after it is set."""
        from .provenance import quote_mode
        self.quotes = quote_mode(quotes)
        self.parts: dict[str, Part] = {}
        self.rules: dict[str, Part] = {}
        self.constraints: dict[str, Part] = {}     # rules between answers of different questions
        self.types: dict = {}                      # fact → type (its producer's return type; see solvi.typed)
        self.readers: dict = {}                    # fact → {typed part that reads it: the type it reads}
        self.decisions = 0                         # decision parts registered (solvi.decide): 0 — no batching work at all

    def _add(self, kind, f, **kw):
        sig = inspect.signature(f)
        # a function made by a model (e.g. extractor.field(...)) says so itself: its model and provenance come with it
        if kw.get("model") is None and getattr(f, "__solvi_model__", None) is not None:
            kw["model"] = f.__solvi_model__
        if kw.get("provenance") is None and getattr(f, "__solvi_provenance__", None) is not None:
            kw["provenance"] = f.__solvi_provenance__
        if kind in ("fn", "rule") and kw.get("options") is None and getattr(f, "__solvi_options__", None) is not None:
            kw["options"] = list(f.__solvi_options__)          # a decision part brings its closed set of options
        if getattr(f, "__solvi_decision__", None) is not None:
            self.decisions += 1                                # the strategist groups decisions that can share a pass
        if kw.get("provenance") is not None:
            from .provenance import GIVEN, KINDS
            if kw["provenance"] not in KINDS or kw["provenance"] == GIVEN:
                raise ValueError(f"provenance must be one of {[k for k in KINDS if k != GIVEN]}")
        if kind == "extract":
            src = _quote_source(f, sig, kw.get("source"))
            kw["source"] = None if src == "doc" else src
        kw = {k: v for k, v in kw.items() if v is not None or k in ("then",)}
        p = Part(kind=kind, name=f.__name__, inputs=list(sig.parameters), func=f, doc=(f.__doc__ or "").strip(),
                 quotes=getattr(self, "quotes", "normalized"), **kw)
        if p.source is not None:                       # a Quote without its own source points into this text
            p.func = _sourced(f, p.source)
        if kind == "check" and p.then:
            p.then_parts = _then_parts(p)
        commit = None
        if getattr(f, "__annotations__", None) and kind != "constraint":      # typed facts (untyped parts skip all this)
            from .typed import compile_part, hints, register
            p.types, p.returns = hints(f, kind)
            if p.types or p.returns is not None:
                compile_part(p)
                commit = register(self, p)                 # checks now (FactTypeError), records once the part is in
        if p.provides is not None:
            self._add_alternative(p)
            if commit:
                commit()
            return f
        lost = validate_misses(p, p.inputs)
        if lost:
            raise ValueError(f"validate of {p.name} reads {lost}, which {p.name} does not take as an input: validate gets "
                             "the value and, by name, the part's own inputs")
        if kind == "rule":
            p.name = "answer:" + p.question
            if self.readers and commit is None:
                from .typed import forget_rule
                forget_rule(self, p.question)
            self.rules[p.question] = p
        elif kind == "constraint":
            if p.name in self.constraints:            # as for parts: a second one of the same name replaced the first
                raise ValueError(f"constraint {p.name} is already in the catalog")
            self.constraints[p.name] = p
        else:
            if p.name in self.parts:
                raise ValueError(f"part {p.name} is already in the catalog")
            self.parts[p.name] = p
        if commit:
            commit()
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
            # a plain producer declared first becomes the first alternative
            first = Part(kind=g.kind, name=g.func.__name__ + "__declared", inputs=g.inputs, func=g.func, doc=g.doc,
                         cost=g.cost, provides=fact, validate=g.validate, min_confidence=g.min_confidence, model=g.model,
                         provenance=g.provenance, options=g.options, exact=g.exact, types=g.types, returns=g.returns,
                         tin=g.tin, tout=g.tout, source=g.source, timeout=g.timeout, blocking=g.blocking)
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
                exact=None, source=None, timeout=None, blocking=None):
        """Extract a value from text (returns a Quote). With `provides="fact"` the function is one of several alternative
        producers of that fact: its output is used only if it passes `validate` / `min_confidence`, otherwise the next
        alternative runs. `cost` (ms) is a prior for scheduling until run times are measured.
        `model`: the model behind it (recorded in the trace with its fingerprint); a model's Quote must be literally the text
        at its offsets or it is rejected (`exact=False` turns that off, `exact=True` turns it on for hand-written code).
        `source`: the given fact (text) its quotes point into. By default: "doc" if the function reads doc, else its only
        argument, else its only str-typed argument; if that is ambiguous, registration raises and asks for source=. A
        returned Quote that names another source (Quote(..., source="notes")) keeps it.
        The function may be `async def` (awaited by System.aask). `timeout` (seconds) and `blocking` (a sync function that
        waits on I/O or a model: aask runs it in a worker thread) apply under aask; see System.aask."""
        return self._deco("extract", f, provides=provides, cost=cost, validate=validate, min_confidence=min_confidence,
                          model=model, provenance=provenance, exact=exact, source=source, timeout=timeout,
                          blocking=blocking)

    def fn(self, f=None, *, provides=None, cost=None, validate=None, model=None, provenance=None, options=None,
           min_confidence=None, timeout=None, blocking=None):
        """A computation. `provides`, `validate`, `cost`, `timeout`, `blocking`: as for extract. A model-backed fn (`model=`)
        with `options` is a model decision: it returns a Decision (or a plain value) that must be one of the options, else
        it is rejected."""
        return self._deco("fn", f, provides=provides, cost=cost, validate=validate, model=model, provenance=provenance,
                          options=None if options is None else list(options), min_confidence=min_confidence,
                          timeout=timeout, blocking=blocking)

    def check(self, f=None, *, hard=False, then=None, cost=None, model=None, provenance=None, timeout=None, blocking=None):
        """A check: a function of facts that answers True or False (or `solvi.refine.Fail("why")` for False with its
        reasons). hard=True: when it is False it decides the questions it governs — the model cannot override it.
        then: {question: answer} — the answer a question gets when this hard check is False (a question listed here is
        governed by it; without `then` the check governs every question whose flow runs it, and they abstain). The check
        is put into the flow of every question its `then` names (since 1.0; no `requires=` needed). An answer may be a
        function of facts instead of a constant — `then={"step": free_side}` with `def free_side(position, walls) ->
        str` — whose argument names are the facts it reads (typed like any part's); it runs only when the check is
        False, its value must be one of the question's answers (else the question abstains), and it is recorded in the
        trace (record kind "then") and re-checked by replay."""
        if then is not None and not isinstance(then, dict):
            raise TypeError(f"then= is {{question: answer or function of facts}}, not {type(then).__name__}: e.g. "
                            "then={\"step\": free_side}")
        return self._deco("check", f, hard=hard, then=then or {}, cost=cost, model=model, provenance=provenance,
                          timeout=timeout, blocking=blocking)

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

    def rule(self, question, *, model=None, provenance=None, timeout=None, blocking=None):
        """The answer rule of a question. With `model=` the rule is a model's decision: it may return a Decision with
        probabilities; an answer outside the question's options abstains, as for any rule. `timeout`, `blocking`: as for
        extract."""
        return lambda f: self._add("rule", f, question=question, model=model, provenance=provenance, timeout=timeout,
                                   blocking=blocking)

    def replace_rule(self, part):
        """Install a ready rule Part (System.learn_rule: a learned rule list) as its question's rule, in place of the one
        registered before; what the catalog recorded about the replaced rule's typed arguments is dropped with it."""
        if part.kind != "rule" or not part.question:
            raise ValueError("replace_rule takes a rule part with its question")
        if self.readers:
            from .typed import forget_rule
            forget_rule(self, part.question)
        self.rules[part.question] = part
        return part

    def constraint(self, f):
        """A rule between answers: argument names are question names, it returns True when the answers fit together
        (e.g. `def unsafe_if_harm(verdict, harm): return harm == "none" or verdict == "unsafe"`). When learned answers break it,
        solvi picks the most probable combination that satisfies every constraint; answers from rules and hard checks stay."""
        return self._add("constraint", f)

    def unreadable_validates(self):
        """Producers whose `validate` requires an argument no producer of their fact takes as an input → [(fact, producer,
        [names])]. Such a validate cannot run, so every output of that producer would be rejected. (A part that is not
        an alternative producer is checked when it is declared; a fact's producers only once all of them are.)"""
        return [(g.name, a.name, lost) for g in self.parts.values() for a in g.alternatives or ()
                for lost in [validate_misses(a, g.inputs)] if lost]

    def alternative(self, fact, name):
        """One alternative producer of a fact, by its function name."""
        g = self.parts[fact]
        for a in g.alternatives or ():
            if a.name == name:
                return a
        raise KeyError(f"{name} is not a producer of {fact}")


__all__ = ["ExperimentalWarning", "NOT_STATED", "accept", "accepts", "Answer", "AnswerType", "bin_labels", "Catalog", "check_evidence", "Claim",
           "cuts_number", "Decision", "evidence_rows", "find_whole", "ground", "has_evidence", "locate",
           "NOT_STATED_KEY", "NotStated", "Part", "plain_json", "PRIMITIVES", "Question", "question_data", "Quote",
           "Serial", "Unknown", "unknown_key", "unwrap", "validated"]
