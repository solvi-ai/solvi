"""Text in: a free text → which question is asked (an entry point) and its typed input state, every value read with a quote.

    tin = TextIn(system, decider)
    read = tin.read("Please refund 1.5 million RUB for order A-10457, bought on 12 September 2026")
    read.question        # "request_refund" — the decider's choice among the entry points (escalates when unsure)
    read.state           # {"order_id": "A-10457", "amount": 1500000.0, "currency": "RUB", "purchase_date": date(2026, 9, 12)}
    read.fields["amount"].quote       # Quote("1.5 million", 21, 32, "request_text") — where the value was read
    read.missing         # required fields the text does not state → ask for them (read.clarify()), never guessed
    res = system.ask_text(read)       # the question answered from that state, all in one trace

Entry points are the system's questions with the typed input state each one reads (`system.entry_points()`, from the same
schemas as `solvi serve`). The model does two small things: the decider picks the entry point (a choice over the entry
points with their descriptions; below `min_confidence` or a near tie it escalates, and nothing is asked), and an extractor
points at the piece of text that holds each field (the decider's own span pointer when the checkpoint has one, else the
deterministic `CueExtractor`). Code does the rest: a deterministic parser per type turns the quote into the value —
numbers ("1,500", "1.5 million", "2k", "полтора миллиона"), dates ("2026-09-12", "12.09.2026", "12 September", "12 сентября";
without a year only with `today=`), enums by label or synonym, booleans, strings — and a quote that does not parse leaves
the field unread. A field the text does not state is "not stated"; a required one is listed in `missing` for a clarifying
question instead of a guess.

Provenance: a field read from the text is `quoted` by a model (the extractor's identity and fingerprint are recorded), never
`given`, so the audit counts it among the model outputs ("quoted by model"), not in the deterministic share; the choice
of entry point is `decided` by the decider. The records (kind "textin") are hash-chained into the answer's trace, and replay
re-checks each one: the quote is literally in the text at its offsets, the parser gives the recorded value from it, and the
value is the one the flow read. Nothing is executed from the text: the "call" is data — a question name from the closed
set of entry points and typed values — that solvi checks and then asks itself."""
from __future__ import annotations

import dataclasses
import datetime as _dt
import re
import typing
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Literal

from .core import Quote, Unknown
from .provenance import ESCALATED, digest, model_info

SOURCE = "request_text"             # the init_state key the text is given under (quotes point into it)
ROUTE = "textin"                    # the name of the entry-point record in the trace; a field's record is "textin:<field>"
SEP = "\n"                          # between turns of a dialogue (TextIn.update)


# ------------------------------------------------------------------------------------------------ entry points
@dataclass
class EntryField:
    """One input field of an entry point: its type, description and whether the question needs it."""
    name: str
    type: Any
    description: str | None = None
    required: bool = False

    def to_dict(self):
        from .typed import type_name
        return {"name": self.name, "type": type_name(self.type), "description": self.description, "required": self.required}


@dataclass
class EntryPoint:
    """A question as an entry point: its name, text and the typed input state it reads (`schema`: the JSON schema)."""
    name: str
    text: str
    fields: dict
    schema: dict

    @property
    def required(self):
        return [f for f, x in self.fields.items() if x.required]

    def tool(self, description=None):
        """As a function-calling tool: {"type": "function", "function": {"name", "description", "parameters"}}."""
        return {"type": "function", "function": {"name": self.name, "description": description or self.text,
                                                 "parameters": self.schema}}

    def to_dict(self):
        return {"name": self.name, "text": self.text, "fields": {f: x.to_dict() for f, x in self.fields.items()},
                "required": self.required, "schema": self.schema}


def entry_points(system, names=None):
    """The questions of a system as entry points (see System.entry_points)."""
    from .serve import _fact_type, input_schema, question_inputs
    out = []
    for q in system.questions.values():
        if names is not None and q.name not in names:
            continue
        info = question_inputs(system, q.name)
        fields = {}
        for f in info["properties"]:
            t, desc = _fact_type(system, f)
            fields[f] = EntryField(f, t, desc, f in info["required"])
        out.append(EntryPoint(q.name, q.text, fields, input_schema(system, q.name)))
    if names is not None:
        missing = [n for n in names if n not in system.questions]
        if missing:
            raise KeyError(f"no such question: {', '.join(missing)}")
    return out


_entry_points = entry_points           # TextIn has an attribute of that name


# ------------------------------------------------------------------------------------------------ deterministic parsers
class ParseError(ValueError):
    """A quote that does not parse as the field's type."""


_NUM_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
              "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
              "fifty": 50, "hundred": 100, "half": Decimal("0.5"), "one and a half": Decimal("1.5"),
              "один": 1, "одна": 1, "одну": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7,
              "восемь": 8, "девять": 9, "десять": 10, "двадцать": 20, "сто": 100, "полтора": Decimal("1.5"),
              "полторы": Decimal("1.5"), "пол": Decimal("0.5")}
_SCALES = {"hundred": 100, "thousand": 10 ** 3, "thousands": 10 ** 3, "k": 10 ** 3, "тыс": 10 ** 3, "тысяча": 10 ** 3,
           "тысячи": 10 ** 3, "тысяч": 10 ** 3, "million": 10 ** 6, "millions": 10 ** 6, "mln": 10 ** 6, "mn": 10 ** 6,
           "m": 10 ** 6, "млн": 10 ** 6, "миллион": 10 ** 6, "миллиона": 10 ** 6, "миллионов": 10 ** 6,
           "billion": 10 ** 9, "billions": 10 ** 9, "bn": 10 ** 9, "b": 10 ** 9, "млрд": 10 ** 9, "миллиард": 10 ** 9,
           "миллиарда": 10 ** 9, "миллиардов": 10 ** 9}
_DIGITS = r"[-+−]?(?:\d{1,3}(?:[   ',]\d{3})+(?:[.,]\d+)?|\d{1,3}(?:\.\d{3}){2,}(?:,\d+)?|\d+(?:[.,]\d+)?)"
_SCALE_RE = "|".join(sorted((re.escape(s) for s in _SCALES), key=len, reverse=True))
_WORD_RE = "|".join(sorted((re.escape(w) for w in _NUM_WORDS), key=len, reverse=True))
_CUR = r"(?:[$€£₽¥]|usd|eur|rub|gbp)"
NUMBER_RE = re.compile(
    rf"(?<![\w.,\-/])(?:{_CUR}\s?)?(?:(?P<d>{_DIGITS})(?:\s?(?P<s1>{_SCALE_RE})\.?(?!\w))?|"
    rf"(?P<w>{_WORD_RE})\s+(?P<s2>{_SCALE_RE})(?!\w))(?![\w]|[.,/\-]\d)", re.I)


def _digits(s):
    """A digit string with thousands separators and a decimal point or comma → Decimal."""
    s = s.replace("−", "-").replace(" ", " ").replace(" ", " ")
    for sep in (" ", "'"):
        s = s.replace(sep, "")
    if "," in s and "." in s:                               # the last one is the decimal separator
        s = s.replace(",", "") if s.rfind(".") > s.rfind(",") else s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", "") if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", s) else s.replace(",", ".")
    elif re.fullmatch(r"[-+]?\d{1,3}(\.\d{3}){2,}", s):      # 1.500.000
        s = s.replace(".", "")
    try:
        return Decimal(s)
    except InvalidOperation:
        raise ParseError(f"not a number: {s!r}") from None


def parse_number(s, spec=None):
    """A quote → the number it states, as a canonical string ("1500000", "12.5"): digits with thousands separators, a
    decimal point or comma, a scale word ("1.5 million", "2k", "3 млн") or a number word with one ("a million", "полтора
    миллиона"); a currency sign or code around it is allowed. Exactly one number: two numbers in the quote are an error.
    spec {"integer": True}: the number must be whole."""
    spec = spec or {}
    found = list(NUMBER_RE.finditer(s))
    if len(found) != 1:
        raise ParseError(f"{'no number' if not found else 'more than one number'} in {s!r}")
    m = found[0]
    if m.group("d") is not None:
        v = _digits(m.group("d"))
        sc = m.group("s1")
    else:
        v = Decimal(_NUM_WORDS[m.group("w").lower()])
        sc = m.group("s2")
    if sc:
        v *= _SCALES[sc.lower()]
    if spec.get("integer") and v != v.to_integral_value():
        raise ParseError(f"{s!r} is not a whole number")
    return _canon_decimal(v)


def _canon_decimal(v):
    t = format(v.normalize(), "f")
    return t[:-2] if t.endswith(".0") else t


_MONTHS = {}
for _i, _names in enumerate([
        ("january", "jan", "января", "январь", "янв"), ("february", "feb", "февраля", "февраль", "фев"),
        ("march", "mar", "марта", "март", "мар"), ("april", "apr", "апреля", "апрель", "апр"),
        ("may", "мая", "май"), ("june", "jun", "июня", "июнь", "июн"), ("july", "jul", "июля", "июль", "июл"),
        ("august", "aug", "августа", "август", "авг"), ("september", "sep", "sept", "сентября", "сентябрь", "сен", "сент"),
        ("october", "oct", "октября", "октябрь", "окт"), ("november", "nov", "ноября", "ноябрь", "ноя"),
        ("december", "dec", "декабря", "декабрь", "дек")], 1):
    for _n in _names:
        _MONTHS[_n] = _i
_MON = "|".join(sorted((re.escape(m) for m in _MONTHS), key=len, reverse=True))
_RELATIVE = {"today": 0, "yesterday": -1, "tomorrow": 1, "сегодня": 0, "вчера": -1, "завтра": 1}
DATE_RES = [
    ("iso", re.compile(r"(?<![\d.])(?P<y>\d{4})-(?P<m>\d{1,2})-(?P<d>\d{1,2})(?![\d])")),
    ("numeric", re.compile(r"(?<![\d.])(?P<a>\d{1,2})[./](?P<b>\d{1,2})[./](?P<y>\d{4}|\d{2})(?![\d])")),
    ("day_month", re.compile(rf"(?<!\w)(?P<d>\d{{1,2}})(?:st|nd|rd|th|-?го|-?е)?\s+(?:of\s+)?(?P<mon>{_MON})\.?"
                             rf"(?:,?\s+(?P<y>\d{{4}}))?(?!\w)", re.I)),
    ("month_day", re.compile(rf"(?<!\w)(?P<mon>{_MON})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y>\d{{4}}))?(?![\w])",
                             re.I)),
    ("relative", re.compile(r"(?<!\w)(?:today|yesterday|tomorrow|сегодня|вчера|завтра)(?!\w)", re.I)),
]


def parse_date(s, spec=None):
    """A quote → the date it states, ISO ("2026-09-12"): 2026-09-12; 12.09.2026 / 12/09/26 (day first; spec
    {"dayfirst": False}: month first); 12 September 2026, September 12, 2026, 12 Sep, 12 сентября; today / yesterday /
    tomorrow. A date without a year, or a relative one, needs spec {"today": "YYYY-MM-DD"} (TextIn(today=...)): without it
    it is an error, never a guessed year. Exactly one date in the quote."""
    spec = spec or {}
    today = _dt.date.fromisoformat(spec["today"]) if spec.get("today") else None
    hits = []
    for kind, rx in DATE_RES:
        for m in rx.finditer(s):
            if not any(a < m.end() and m.start() < b for a, b, _, _ in hits):
                hits.append((m.start(), m.end(), kind, m))
    if len(hits) != 1:
        raise ParseError(f"{'no date' if not hits else 'more than one date'} in {s!r}")
    _, _, kind, m = hits[0]
    if kind == "relative":
        if today is None:
            raise ParseError(f"{m.group()!r} is relative: pass today= to read it")
        return (today + _dt.timedelta(days=_RELATIVE[m.group().lower()])).isoformat()
    if kind == "iso":
        y, mo, d = int(m.group("y")), int(m.group("m")), int(m.group("d"))
    elif kind == "numeric":
        a, b = int(m.group("a")), int(m.group("b"))
        d, mo = (a, b) if spec.get("dayfirst", True) else (b, a)
        y = int(m.group("y"))
        y = y + 2000 if y < 100 else y
    else:
        d, mo = int(m.group("d")), _MONTHS[m.group("mon").lower()]
        y = m.group("y")
        if y is None:
            if today is None:
                raise ParseError(f"{m.group()!r} has no year: pass today= to read it")
            y = today.year
        y = int(y)
    try:
        return _dt.date(y, mo, d).isoformat()
    except ValueError as e:
        raise ParseError(f"not a date: {m.group()!r} ({e})") from None


_YES = {"yes", "y", "true", "on", "1", "yep", "sure", "да", "истина", "верно"}
_NO = {"no", "n", "false", "off", "0", "none", "нет", "ложь"}
# a negation word ("not", "isn't", "not at all", "hardly", "не", "ни"): near a yes / no cue it makes the cue unreadable
_NEG_WORDS = r"(?:not|no|non|never|nor|neither|without|cannot|hardly|barely|scarcely|не|нет|ни|без)"
NEGATION_RE = re.compile(rf"(?<!\w){_NEG_WORDS}(?!\w)|n['’]t(?!\w)", re.I)
_CLAUSE_END = re.compile(r"[.;:!?,\n—–]")
_NEG_WINDOW = 40                    # how far before a cue a negation still counts (chars, within the clause)


def _norm(s):
    return re.sub(r"[\s_\-]+", " ", re.sub(r"[^\w\s\-€$£₽%]", " ", str(s).casefold())).strip()


def parse_bool(s, spec=None):
    """A quote → True / False: yes / no words (en, ru); a declared negative cue (spec {"negatives": [...]}: "not urgent",
    "no rush") → False; the quote is exactly one of the field's cues (spec {"cues": [...]}: its name and the `cues=` words)
    → True. A cue with a negation near it ("isn't urgent", "not at all urgent", "hardly urgent", "не срочно") is neither:
    it does not parse — never True, and False only through a declared negative cue."""
    spec = spec or {}
    t = _norm(s)
    if t in _YES:
        return True
    if t in _NO:
        return False
    if t and t in {_norm(c) for c in spec.get("negatives") or ()}:
        return False
    cues = [_norm(c) for c in spec.get("cues") or () if _norm(c)]
    neg = NEGATION_RE.search(str(s))
    for c in sorted(cues, key=len, reverse=True):
        if re.search(rf"(?<!\w){re.escape(c)}(?!\w)", t):
            if neg:
                raise ParseError(f"{s!r} negates {c!r} ({neg.group()!r}): not read as a yes or a no "
                                 "(declare negative cues to read it as no)")
            if t == c:
                return True
    raise ParseError(f"not a yes / no: {s!r}")


def parse_enum(s, spec):
    """A quote → one of the labels (spec {"labels": {label: [synonym, ...]}}): the quote is a label or a synonym (case,
    spaces, underscores and punctuation ignored), or contains exactly one of them as a whole word."""
    labels = spec["labels"]
    t = _norm(s)
    for lab, syn in labels.items():
        if t in {_norm(x) for x in [lab, *syn]}:
            return lab
    hit = set()
    for lab, syn in labels.items():
        for x in [lab, *syn]:
            n = _norm(x)
            if n and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", t):
                hit.add(lab)
    if len(hit) == 1:
        return hit.pop()
    raise ParseError(f"{s!r} is {'none' if not hit else 'more than one'} of {sorted(labels)}")


def parse_text(s, spec=None):
    """A quote → the string, trimmed of spaces and surrounding punctuation (spec {"pattern": regex}: the whole string must
    match it)."""
    t = s.strip().strip(" \t\n.,;:!?\"'«»()")
    if not t:
        raise ParseError("an empty quote")
    pat = (spec or {}).get("pattern")
    if pat and not re.fullmatch(pat, t):
        raise ParseError(f"{t!r} does not match {pat!r}")
    return t


PARSERS = {"number": parse_number, "integer": lambda s, sp=None: parse_number(s, {**(sp or {}), "integer": True}),
           "date": parse_date, "bool": parse_bool, "enum": parse_enum, "text": parse_text}


# ------------------------------------------------------------------------------------------------ a field and its type
@dataclass
class FieldSpec:
    """What the extractor and the parser know of a field: kind (number, integer, date, bool, enum, text, or unsupported),
    cue words, enum labels with synonyms, a pattern."""
    name: str
    type: Any
    kind: str
    description: str | None = None
    required: bool = False
    cues: list = field(default_factory=list)
    labels: dict | None = None            # enum: {label: [synonyms]}
    values: dict | None = None            # enum: {label: the typed value (a Literal value, an Enum member)}
    pattern: str | None = None
    hints: list = field(default_factory=list)       # description words: they rank candidates, never decide a value
    negatives: list = field(default_factory=list)   # bool: declared phrases that mean False ("not urgent", "no rush")

    def parser_spec(self, today=None, dayfirst=True):
        """The parser's arguments, recorded in the trace so a replay parses the quote the same way."""
        if self.kind == "enum":
            return {"labels": self.labels}
        if self.kind == "bool":
            return {"cues": list(self.cues), **({"negatives": list(self.negatives)} if self.negatives else {})}
        if self.kind == "date":
            return {"today": today.isoformat() if today else None, "dayfirst": dayfirst}
        if self.kind == "text":
            return {"pattern": self.pattern} if self.pattern else {}
        return {}

    def value(self, canonical):
        """The parser's canonical output → the typed value."""
        t = self.type
        if self.kind == "enum":
            return self.values[canonical]
        if self.kind == "date":
            return _dt.date.fromisoformat(canonical)
        if self.kind in ("number", "integer"):
            d = Decimal(canonical)
            if t is int or self.kind == "integer":
                return int(d)
            return d if t is Decimal else float(d)
        return canonical


_STOP = {"the", "a", "an", "of", "for", "to", "in", "on", "is", "are", "and", "or", "by", "with", "from", "this", "that",
         "what", "which", "when", "how", "id", "number", "value", "read", "given", "none"}
_BOOL_PREFIX = {"is", "are", "was", "has", "have", "had", "needs", "need", "wants", "want", "should", "must", "can",
                "requires", "require", "be"}


def _base_type(t):
    from .typed import _members, _strip
    t = _strip(t)
    ms = [m for m in _members(t) if m is not type(None)]
    return _strip(ms[0]) if len(ms) == 1 else t


def field_spec(ef, synonyms=None, cues=None, pattern=None, extra=None, negatives=None):
    """An EntryField → FieldSpec: the kind from the type, cue words from the name (plus `cues`), hint words from the
    description (they rank candidates, never decide), enum labels from Literal values / Enum members (plus `synonyms`
    {label: [...]}, keyed by the value or the member name). A bool field's value cues are only its name ("urgent", "is
    urgent" → "urgent") and `cues`; `negatives` (or json_schema_extra "negative_cues") are phrases that mean False."""
    extra = extra or {}
    t = _base_type(ef.type)
    kind, labels, values = "unsupported", None, None
    if t is bool:
        kind = "bool"
    elif t is int:
        kind = "integer"
    elif t in (float, Decimal):
        kind = "number"
    elif t is _dt.date:
        kind = "date"
    elif t is str:
        kind = "text"
    elif typing.get_origin(t) is Literal:
        kind, values = "enum", {str(v): v for v in typing.get_args(t)}
    elif isinstance(t, type) and issubclass(t, Enum):
        kind, values = "enum", {str(m.value): m for m in t}
    if kind == "enum":
        syn = {**(extra.get("synonyms") or {}), **(synonyms or {})}
        labels = {}
        for lab, v in values.items():
            al = [str(v.name).replace("_", " ")] if isinstance(v, Enum) and str(v.name) != lab else []
            for key in (lab, getattr(v, "name", None), v):
                try:
                    al += [str(x) for x in syn.get(key, ())] if key is not None else []
                except TypeError:
                    pass
            labels[lab] = list(dict.fromkeys(a for a in al if _norm(a) != _norm(lab)))
    phrase = ef.name.replace("_", " ").lower()
    words = [w for w in re.split(r"[_\W]+", ef.name.lower()) if w and w not in _STOP]
    desc = [w for w in re.findall(r"\w+", (ef.description or "").lower()) if len(w) > 2 and w not in _STOP]
    explicit = [str(c).lower() for c in extra.get("cues") or ()] + [str(c).lower() for c in cues or ()]
    if kind == "bool":                   # only the name and explicit cues can make a field True
        core = " ".join(w for w in re.split(r"[_\W]+", ef.name.lower()) if w and w not in _BOOL_PREFIX)
        cue = list(dict.fromkeys([c for c in (phrase, core) if c] + explicit))
        hints = [w for w in dict.fromkeys(words + desc) if w not in cue]
    else:
        cue = list(dict.fromkeys([phrase] + words + explicit))
        hints = [w for w in dict.fromkeys(desc) if w not in cue]
    neg = list(dict.fromkeys(str(c).lower() for c in [*(extra.get("negative_cues") or ()), *(negatives or ())]))
    return FieldSpec(ef.name, ef.type, kind, ef.description, ef.required, cue, labels, values,
                     pattern or extra.get("pattern"), hints, neg)


# ------------------------------------------------------------------------------------------------ extractors
class CueExtractor:
    """A deterministic extractor: candidates of the field's type in the text (numbers, dates, enum labels and synonyms,
    cue words for a yes / no, a pattern — or, for a string, the words after a cue and a connector: "address: ...", "address
    is ..." — up to the end of the clause), the one nearest after a cue word of the
    field (its name, its description's words, `cues=`) first. It never calls a model, but the choice of the span is still
    a guess, so a value it reads is recorded as quoted by it (its identity in the trace) and counted with model outputs."""
    model_id = "solvi.textin.CueExtractor"
    version = "1"

    def fingerprint(self):
        return digest("CueExtractor", self.version)

    def find(self, text, fs):
        """→ [Quote] candidates, best first (confidence 1.0 near a cue, lower without one); [] when not stated."""
        cues = self._cues(text, fs)
        if fs.kind in ("number", "integer"):
            spans = [(m.start(), m.end()) for m in NUMBER_RE.finditer(text)]
            dates = [(m.start(), m.end()) for _, rx in DATE_RES for m in rx.finditer(text)]
            spans = [s for s in spans if not any(a < s[1] and s[0] < b for a, b in dates)]
        elif fs.kind == "date":
            spans = [(m.start(), m.end()) for _, rx in DATE_RES for m in rx.finditer(text)]
        elif fs.kind == "enum":
            spans = []
            for lab, syn in fs.labels.items():
                for x in [lab, *syn]:
                    if x.strip():
                        spans += [(m.start(), m.end()) for m in
                                  re.finditer(rf"(?<!\w){re.escape(x.strip())}(?!\w)", text, re.I)]
            return self._rank(text, spans, cues, own=True)
        elif fs.kind == "bool":
            spans = []
            for c in [*fs.cues, *fs.negatives]:
                for m in re.finditer(rf"(?<!\w){re.escape(c)}(?!\w)", text, re.I):
                    spans.append((_negated_from(text, m.start()), m.end()))
            return self._rank(text, spans, self._cues(text, fs, hints_only=True), own=True)
        elif fs.kind == "text":
            if fs.pattern:
                spans = [(m.start(), m.end()) for m in re.finditer(fs.pattern, text)]
            else:
                spans = []
                for a, b in cues:
                    m = re.match(r"(?:\s*[:=\-–—]\s*|\s+(?:is|are|to|as|was|should be|will be|будет|на)\s+)([^.;\n]+)", text[b:])
                    if m and m.group(1).strip():
                        s = b + m.start(1)
                        e = b + m.end(1)
                        while e > s and text[e - 1] in " ,":
                            e -= 1
                        spans.append((s, e))
                return [Quote(text[s:e], s, e, SOURCE, 0.7) for s, e in spans[:1]]
        else:
            return []
        return self._rank(text, spans, cues)

    @staticmethod
    def _cues(text, fs, hints_only=False):
        out = []
        for c in (fs.hints if hints_only else [*fs.cues, *fs.hints]):
            out += [(m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(c)}\w*", text, re.I)]
        return sorted(out)

    @staticmethod
    def _rank(text, spans, cues, own=False):
        """Candidates best first: nearest after a cue in the same sentence (confidence 1.0), else by position — 0.9 for a
        candidate that is itself a label / cue (own), 0.6 for a lone candidate of the type, 0.3 when there are several."""
        spans = sorted(set(spans))
        spans = [s for s in spans if not any(o != s and o[0] <= s[0] and s[1] <= o[1] for o in spans)]
        scored = []
        for s, e in spans:
            best = None
            for a, b in cues:
                gap = text[b:s] if b <= s else None
                if gap is not None and len(gap) <= 60 and not re.search(r"[.;!?\n]", gap):
                    best = len(gap) if best is None else min(best, len(gap))
            if best is not None:
                scored.append((0, best, s, e, 1.0))
            else:
                scored.append((1, s, s, e, 0.9 if own else 0.6 if len(spans) == 1 else 0.3))
        scored.sort()
        return [Quote(text[s:e], s, e, SOURCE, c) for _, _, s, e, c in scored]


def _negated_from(text, start):
    """Where a yes / no cue at `start` begins once a negation shortly before it in the same clause is included ("it isn't
    really urgent" → from "isn't"): the parser then sees the negation and does not read the cue as True."""
    lo = max(0, start - _NEG_WINDOW)
    window = text[lo:start]
    cut = [m.end() for m in _CLAUSE_END.finditer(window)]
    base = lo + (cut[-1] if cut else 0)
    negs = list(NEGATION_RE.finditer(text, base, start))
    if not negs:
        return start
    s = negs[0].start()
    while s > base and (text[s - 1].isalnum() or text[s - 1] in "'’"):     # the whole word of "isn't"
        s -= 1
    return s


class DeciderExtractor:
    """The decider's own span pointer: for each field a span question ("What is the <field>?" or its description), "not
    stated" allowed when the checkpoint has it. Needs a checkpoint with a pointer (DecideModel.has_pointer)."""

    def __init__(self, decider):
        if not getattr(decider, "has_pointer", False):
            raise ValueError("this decider cannot point at its input (no span pointer): use CueExtractor")
        self.decider = decider

    @property
    def model_id(self):
        return self.decider.model_id

    def fingerprint(self):
        return digest("DeciderExtractor", self.decider.fingerprint())

    def find(self, text, fs):
        task = fs.description or f"What is the {fs.name.replace('_', ' ')}?"
        d = self.decider.decide(text, task, [], kind="span", unknown=bool(getattr(self.decider, "has_unknown", False)))
        if d.value is Unknown or not isinstance(d.value, Quote) or d.escalate or d.value.end <= d.value.start:
            return []
        q = d.value
        return [Quote(text[q.start:q.end], q.start, q.end, SOURCE, float(d.conf))]


# ------------------------------------------------------------------------------------------------ what was read
@dataclass
class FieldRead:
    """One field as read from the text. status: read | not_stated | unparsed (a quote that does not parse as the type) |
    unsure (below min_field_confidence) | unsupported (a type no parser reads) | given (passed by the caller to update)."""
    name: str
    status: str
    value: Any = None
    quote: Quote | None = None
    confidence: float = 0.0
    canonical: Any = None
    parser: str | None = None
    spec: dict | None = None
    why: str | None = None
    required: bool = False
    model: dict | None = None
    was: Any = None                        # update(): the value before this turn (when it changed)

    @property
    def ok(self):
        return self.status in ("read", "given")


@dataclass
class Change:
    """A field a dialogue turn changed (TextIn.update): old and new value, and the quote of the new one."""
    field: str
    old: Any
    new: Any
    quote: Quote | None


@dataclass
class TextRead:
    """A text read as a call: the question (None when the entry point escalated), each field of its input state, the
    route (probabilities over the entry points), `missing` required fields, and `changes` after TextIn.update."""
    text: str
    question: str | None
    fields: dict
    route: dict
    source: str = SOURCE
    changes: list = field(default_factory=list)
    entry_points: list = field(default_factory=list)
    router: dict | None = None             # the decider's identity (model_info)

    @property
    def state(self):
        """{field: value} of the fields read from the text."""
        return {f: r.value for f, r in self.fields.items() if r.ok}

    @property
    def missing(self):
        """Required fields the text does not give (not stated, unparsed, unsure, unsupported)."""
        return [f for f, r in self.fields.items() if r.required and not r.ok]

    @property
    def escalated(self):
        return self.route.get("escalated")

    @property
    def ok(self):
        """A question was chosen and every required field was read."""
        return self.question is not None and not self.missing

    def init_state(self):
        """The init_state System.ask reads: the text itself (under `source`) and the fields read from it."""
        return {self.source: self.text, **self.state}

    def clarify(self):
        """A clarifying question for what is missing (None when nothing is): the entry point when it escalated, else the
        required fields not read."""
        if self.question is None:
            c = self.route.get("candidates") or []
            return "Which of these do you mean: " + "; ".join(c) + "?" if c else "What would you like to do?"
        if not self.missing:
            return None
        parts = []
        for f in self.missing:
            r = self.fields[f]
            what = f.replace("_", " ")
            parts.append(what + (f" (I read {r.quote.value!r} but {r.why})" if r.status in ("unparsed", "unsure")
                                 and r.quote is not None else ""))
        return "Please tell me the " + ", ".join(parts[:-1]) + (" and the " if len(parts) > 1 else "") + parts[-1] + "."

    def to_dict(self):
        from .runtime import _canon
        return {"question": self.question, "route": _canon(self.route), "missing": self.missing,
                "state": _canon(self.state),
                "fields": {f: {"status": r.status, "value": _canon(r.value) if r.ok else None, "confidence": r.confidence,
                               "quote": None if r.quote is None else [r.quote.value, r.quote.start, r.quote.end],
                               "why": r.why, "required": r.required} for f, r in self.fields.items()},
                "changes": [{"field": c.field, "old": _canon(c.old), "new": _canon(c.new),
                             "quote": [c.quote.value, c.quote.start, c.quote.end]} for c in self.changes]}

    def records(self):
        """The trace records of this read (kind "textin"): the entry point, then one per field — fresh Records, chained by
        the System when it appends them."""
        from .runtime import Record, vhash
        h = {self.source: vhash(self.text)}
        rt = self.route
        by = rt.get("by")
        err = rt.get("escalated")
        recs = [Record(step=0, kind="textin", name=ROUTE, inputs=dict(h), value=self.question, error=err,
                       confidence=float(rt.get("confidence", 1.0)),
                       provenance="decided" if by == "decider" else "given" if by == "given" else "computed",
                       model=self.router if by == "decider" else None,
                       probs=dict(rt["probs"]) if rt.get("probs") else None,
                       extra={"entry_points": list(self.entry_points), "candidates": list(rt.get("candidates") or []),
                              "by": by})]
        for f, r in self.fields.items():
            ex = {"field": f, "status": r.status, "required": r.required}
            if r.parser:
                ex.update(parser=r.parser, spec=r.spec)
            if r.quote is not None:
                ex["quoted"] = r.quote.value
            if r.ok:
                ex["canonical"] = r.canonical
            if r.was is not None:
                from .runtime import srepr
                ex["was"] = srepr(r.was)
            recs.append(Record(step=0, kind="textin", name=f"{ROUTE}:{f}", inputs=dict(h),
                               value=r.value if r.ok else Unknown,
                               quote=None if r.quote is None else (r.quote.start, r.quote.end, self.source),
                               confidence=float(r.confidence), error=None if r.ok or r.status == "not_stated" else r.why,
                               provenance="given" if r.status == "given" else "quoted",
                               model=None if r.status == "given" else r.model, extra=ex))
        return recs


# ------------------------------------------------------------------------------------------------ TextIn
class TextIn:
    """Free text → (question, init_state) over a System's entry points.

    decider: picks the entry point (a DecideModel, or any object with decide(text, task, options, descriptions=,
    kind="choice") → Decision); not needed with one entry point or when the question is given. extractor: finds each
    field's span — an object with find(text, FieldSpec) → [Quote] (best first), or a list of them tried in order; default:
    the decider's pointer (DeciderExtractor) when it has one, else CueExtractor.

    entry_points: the question names to choose from (default: every question). descriptions: {question: text} for the
    router (default: the question's text). synonyms: {field: {label: [synonym]}} for enum fields; cues: {field: [word]}
    extra cue words; patterns: {field: regex} for string fields; negatives: {field: [phrase]} for yes / no fields — the
    phrases that mean False ("not urgent", "no rush"; without one a negated cue does not parse) (a System(inputs=...)
    field's json_schema_extra may carry "synonyms", "cues", "negative_cues", "pattern" too). today: a date for year-less and relative dates (recorded in the trace).
    dayfirst: 12/09 is 12 September. min_confidence / min_margin: the router escalates below this probability or when the
    two best entry points are closer than the margin. min_field_confidence: a span found with less confidence is "unsure"
    (not used). task: the routing question the decider reads. source: the init_state key the text is given under."""

    def __init__(self, system, decider=None, extractor=None, *, entry_points=None, descriptions=None, synonyms=None,
                 cues=None, patterns=None, negatives=None, today=None, dayfirst=True, min_confidence=0.6, min_margin=0.1,
                 min_field_confidence=0.5, task="Which request is this text making?", source=SOURCE):
        self.system, self.decider = system, decider
        if extractor is None:
            extractor = DeciderExtractor(decider) if getattr(decider, "has_pointer", False) else CueExtractor()
        self.extractors = list(extractor) if isinstance(extractor, (list, tuple)) else [extractor]
        self.entry_points = {e.name: e for e in _entry_points(system, entry_points)}
        self.descriptions = dict(descriptions or {})
        self.synonyms, self.cues, self.patterns = dict(synonyms or {}), dict(cues or {}), dict(patterns or {})
        self.negatives = dict(negatives or {})
        if isinstance(today, str):
            today = _dt.date.fromisoformat(today)
        self.today, self.dayfirst = today, dayfirst
        self.min_confidence, self.min_margin, self.min_field_confidence = min_confidence, min_margin, min_field_confidence
        self.task, self.source = task, source
        self._specs = {}

    # --- routing
    def route(self, text, question=None):
        """→ {"question", "probs", "confidence", "candidates", "escalated", "by"}: the decider's choice among the entry
        points, escalated (question None) below min_confidence, on a near tie or when the decider escalates."""
        names = sorted(self.entry_points)
        if question is not None:
            if question not in self.entry_points:
                raise KeyError(f"{question!r} is not an entry point: {', '.join(names)}")
            return {"question": question, "probs": None, "confidence": 1.0, "candidates": [question], "escalated": None,
                    "by": "given"}
        if len(names) == 1:
            return {"question": names[0], "probs": None, "confidence": 1.0, "candidates": names, "escalated": None,
                    "by": "single"}
        if self.decider is None:
            raise ValueError("choosing among several entry points needs a decider (or question=)")
        desc = {n: self.descriptions.get(n) or self.entry_points[n].text or n.replace("_", " ") for n in names}
        d = self.decider.decide(text, self.task, names, descriptions=desc, kind="choice")
        probs = {str(k): float(v) for k, v in d.probs.items() if k in self.entry_points}
        ranked = sorted(probs.items(), key=lambda kv: -kv[1])
        best, p1 = ranked[0]
        p2 = ranked[1][1] if len(ranked) > 1 else 0.0
        cands = [n for n, p in ranked if p >= 0.1][:3] or [best]
        why = None
        if getattr(d, "escalate", None):
            why = d.escalate if d.escalate.startswith(ESCALATED) else f"{ESCALATED}: {d.escalate}"
        elif p1 < self.min_confidence:
            why = f"{ESCALATED}: entry point unsure — {best} {p1:.2f} < {self.min_confidence:.2f}"
        elif p1 - p2 < self.min_margin:
            why = f"{ESCALATED}: entry point unsure — {best} {p1:.2f} vs {ranked[1][0]} {p2:.2f} (margin < {self.min_margin:g})"
        return {"question": None if why else best, "probs": probs, "confidence": p1, "candidates": cands,
                "escalated": why, "by": "decider"}

    # --- fields
    def spec(self, question, name):
        key = (question, name)
        if key not in self._specs:
            ef = self.entry_points[question].fields[name]
            extra = {}
            m = getattr(self.system, "inputs", None)
            if m is not None and name in m.model_fields and isinstance(m.model_fields[name].json_schema_extra, dict):
                extra = m.model_fields[name].json_schema_extra
            self._specs[key] = field_spec(ef, self.synonyms.get(name), self.cues.get(name), self.patterns.get(name), extra,
                                          self.negatives.get(name))
        return self._specs[key]

    def _field(self, text, fs, avoid=None):
        """Read one field: the first candidate (of the first extractor that finds any) that parses; with `avoid` (the
        value before a dialogue turn) a candidate with another value is preferred ("not A-10457 but A-10475")."""
        if fs.kind == "unsupported":
            from .typed import type_name
            return FieldRead(fs.name, "unsupported", why=f"no parser reads {type_name(fs.type)}", required=fs.required)
        parser, spec = fs.kind, fs.parser_spec(self.today, self.dayfirst)
        first_bad = None
        for ex in self.extractors:
            cands = ex.find(text, fs)
            got = []
            for q in cands:
                try:
                    c = PARSERS[parser](q.value, spec)
                except ParseError as e:
                    first_bad = first_bad or (q, str(e), ex)
                    continue
                got.append((q, c))
            if not got:
                continue
            if avoid is not None and len(got) > 1 and fs.value(got[0][1]) == avoid:
                other = [g for g in got if fs.value(g[1]) != avoid]
                if other:
                    got = other + got
            q, c = got[0]
            mi = model_info(ex)
            if q.confidence < self.min_field_confidence:
                return FieldRead(fs.name, "unsure", quote=q, confidence=q.confidence, parser=parser, spec=spec,
                                 why=f"found with confidence {q.confidence:.2f} < {self.min_field_confidence:.2f}",
                                 required=fs.required, model=mi)
            return FieldRead(fs.name, "read", fs.value(c), q, q.confidence, c, parser, spec, required=fs.required, model=mi)
        if first_bad is not None:
            q, why, ex = first_bad
            return FieldRead(fs.name, "unparsed", quote=q, confidence=q.confidence, parser=parser, spec=spec,
                             why=f"cannot parse as {fs.kind}: {why}", required=fs.required, model=model_info(ex))
        return FieldRead(fs.name, "not_stated", parser=parser, spec=spec, why="not stated in the text", required=fs.required,
                         model=model_info(self.extractors[0]))

    def _fields(self, question, text, avoid=None):
        return {f: self._field(text, self.spec(question, f), (avoid or {}).get(f))
                for f in self.entry_points[question].fields}

    def read(self, text, question=None):
        """A text → TextRead: the entry point (routed, or `question`), and each input field read with its quote."""
        if not isinstance(text, str):
            raise TypeError(f"TextIn reads a text, not {type(text).__name__}")
        rt = self.route(text, question)
        q = rt["question"]
        fields = self._fields(q, text) if q is not None else {}
        return TextRead(text, q, fields, rt, self.source, entry_points=sorted(self.entry_points),
                        router=model_info(self.decider) if rt["by"] == "decider" else None)

    def update(self, prev, new_text, question=None):
        """A dialogue turn: `prev` (a TextRead, or a state dict with question=) and the next message → a new TextRead over
        the whole dialogue (the turns joined by a new line; quotes point into it) whose `changes` list the fields the turn
        states anew — old value, new value, quote. Fields the turn does not state keep their value and quote; a turn that
        repeats the old value next to a new one ("not A-10457 but A-10475") changes it to the new one. The entry point stays
        the one already chosen (an escalated read is routed again on the whole dialogue)."""
        if isinstance(prev, TextRead):
            base, old_fields, q = prev.text, dict(prev.fields), question or prev.question
            rt = prev.route if q == prev.question else None
        else:
            base, old_fields, q, rt = "", {}, question, None
            if q is None:
                raise ValueError("update(state_dict, text) needs question=")
            prev_state = dict(prev or {})
            for f, v in prev_state.items():
                if q in self.entry_points and f in self.entry_points[q].fields:
                    old_fields[f] = FieldRead(f, "given", v, None, 1.0, required=self.spec(q, f).required)
        text = base + SEP + new_text if base else new_text
        off = len(base) + len(SEP) if base else 0
        if q is None:                                  # not routed yet: route the whole dialogue
            return self.read(text)
        if rt is None:
            rt = self.route(text, q)
        avoid = {f: r.value for f, r in old_fields.items() if r.ok}
        fresh = self._fields(q, new_text, avoid)
        fields, changes = {}, []
        for f, r in fresh.items():
            old = old_fields.get(f)
            if r.quote is not None:
                r.quote = dataclasses.replace(r.quote, start=r.quote.start + off, end=r.quote.end + off)
            if r.ok:
                if old is not None and old.ok and old.value != r.value:
                    r.was = old.value
                    changes.append(Change(f, old.value, r.value, r.quote))
                elif old is None or not old.ok:
                    changes.append(Change(f, None, r.value, r.quote))
                fields[f] = r
            else:
                fields[f] = old if old is not None and (old.ok or r.status == "not_stated") else r
        return TextRead(text, q, fields, rt, self.source, changes, sorted(self.entry_points),
                        router=prev.router if isinstance(prev, TextRead) else None)


# ------------------------------------------------------------------------------------------------ replay
def replay_record(r, init):
    """Re-check a "textin" trace record against the recorded input → [(step, name, reason)]: the entry point is one of the
    recorded entry points; a field's quote is literally in the text at its offsets, the recorded parser gives the recorded
    value from it, and the flow read that value (the given fact equals it); a field not read was not given either."""
    bad = []
    ex = r.extra if isinstance(r.extra, dict) else {}
    if r.name == ROUTE:
        eps = ex.get("entry_points") or []
        if r.value is not None and eps and r.value not in eps:
            bad.append((r.step, r.name, f"entry point {r.value!r} is not one of {eps}"))
        if r.probs and eps and set(r.probs) - set(eps):
            bad.append((r.step, r.name, "route probabilities over names that are not entry points"))
        return bad
    f = ex.get("field", r.name.split(":", 1)[-1])
    src = r.quote[2] if r.quote else None
    if ex.get("status") == "read":
        text = init.get(src) if src else None
        if not isinstance(text, str):
            return [(r.step, r.name, f"the text {src!r} is not in the recorded input")]
        s, e = r.quote[0], r.quote[1]
        if not (0 <= s <= e <= len(text)) or text[s:e] != ex.get("quoted"):
            return [(r.step, r.name, f"quote {ex.get('quoted')!r} is not at [{s}:{e}] of {src}")]
        try:
            c = PARSERS[ex["parser"]](text[s:e], ex.get("spec") or {})
        except (ParseError, KeyError) as err:
            return [(r.step, r.name, f"the quote does not parse any more: {err}")]
        if c != ex.get("canonical"):
            bad.append((r.step, r.name, f"parsed {c!r} ≠ recorded {ex.get('canonical')!r}"))
        if f in init and init[f] != r.value:
            bad.append((r.step, r.name, f"the flow read {f} = {init[f]!r}, not the value read from the text {r.value!r}"))
    elif ex.get("status") == "given":
        if f in init and init[f] != r.value:
            bad.append((r.step, r.name, f"the flow read {f} = {init[f]!r}, not the given {r.value!r}"))
    elif f in init:
        bad.append((r.step, r.name, f"{f} was not read from the text ({ex.get('status')}) but is in the input"))
    return bad
