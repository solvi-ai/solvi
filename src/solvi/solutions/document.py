"""solvi.document — a decision over a document: a model reads one fact per field, code computes the answer.

    import solvi
    from pydantic import BaseModel, Field
    from solvi import Span
    from solvi.models import llm

    class Operands(BaseModel):
        a: Span[float] = Field(description="the first number the question names, exactly as written")
        b: Span[float] = Field(description="the second number, exactly as written")

    recipe = solvi.document(Operands, llm(URL, MODEL, api_key=KEY),
                            task="The text starts with a QUESTION, then a table and notes.",
                            compute=lambda f, s: f.a - f.b,                  # f: the facts, s: the rest of the input
                            checks={"b_not_zero": lambda f, s: f.b != 0},
                            storage="decisions.jsonl")
    recipe.calibrate([({"doc": d, "op": op}, gold) for d, op, gold in labelled], max_risk=0.10)
    res = recipe.ask({"doc": text, "op": "diff"})
    res["answer"].answer, res["answer"].status, res["answer"].why, res.stored_id
    recipe.verify(), recipe.replay_all()

The scheme it sets up (all of it existing low-level blocks, wired):

- one read per field: each field of the schema is one decision part (`DecideModel.decision`), its description the
  question (after `task=`), the text fact `text=` (default "doc") its input — one model call per field, the same text
  each call. A field with `json_schema_extra={"options": {option: description}}` asks with the plain option list and
  the descriptions in the question's text (a model that echoes "option: description" back is spared the chance).
- grounding: a `Span[...]` field must be found in the text (normalized view; a number as a number of its own, not
  inside a longer one — solvi.core.catalog.locate, cuts_number); not found, the field escalates and the answer
  abstains with the reason. The kept value is the text's own substring.
- numbers: a `Span[float]` (int, Decimal) is read from what was quoted, as the document writes it: one scale rule —
  a scale is never multiplied, the number is kept in the document's own unit ("1,250 million", "$1.2bn", "1.2M",
  "3k" → 1250, 1.2, 1.2, 3); currency signs and codes dropped; a percentage as its number ("12.5%" → 12.5); a sign in
  any dash ("-", "−", "–", "—"), before or after a currency sign, or trailing ("1,234-"); parentheses as an accounting
  negative ("(1,234)", "$(1,234) million", "EUR (1,234)" → −1234) — only around a number with a separator, decimals
  or a currency ("(2019)", "(1)" do not read: a year, a footnote). Anything else — a range, two numbers, an exponent
  ("1e3"), "1_000", NaN or infinity, an ambiguous "2.500" — does not read, and the answer abstains with the reason. A
  number field also abstains when the text writes a sign the quote left out (a "-" or "(" right before it): the
  quote "1,234" of "-1,234" or "(1,234)" is not read as +1234. A `Span[date]` and other types: solvi.core.types.span_value.
- code computes: `compute=(facts, state) → value` is the question's rule; `facts` is the schema filled with the read
  values, `state` the input without the text. The answer is a number (`Answer.estimate()`) unless `answer=` says
  otherwise (`solvi.Answer.choice(["pay", "hold"])`, `Answer.yes_no()`); its confidence is the weakest fact's.
  A result that is not a finite number (NaN, infinity) abstains. Without `compute=` a schema of one field answers
  with that field's value: a number (estimate), or a `Literal` / `Enum` / `bool` field's option; a `Span[str]` or
  `Span[date]` field needs compute= and answer=.
- checks: `checks={name: (facts, state) → bool}` are hard checks on the answer: one that fails makes it abstain
  ("failed check").
- confidence: an LLM made with `logprobs="auto"` (llm()'s default) is rebuilt to ask for one stated confidence per
  read (`logprobs=False, ask="confidence"`), with a UserWarning that says so; a gateway that mixes logprob and stated
  replies cannot then split a calibration. A model made with `logprobs=True` or `False`, and any other decider, is
  used as given. `recipe.model` is the model the reads go to (its usage counters are there).
- calibration: `calibrate(examples, max_risk=)` is System.guarantee on the answer (conformal risk control; max_error=
  for learn-then-test; groups=, correct=, weak= pass through): below the threshold the answer abstains. Fewer than
  100 examples: a GuaranteeWarning (the threshold rests on few examples and can move); a signal that does not
  separate right from wrong raises, as System.guarantee does. A number is judged right at the label's own precision:
  the label (a number, or a string read like a quote: "1,500", "33.33") with d decimals matches an answer within half
  a unit of its last digit (0.5·10^-d); pass correct=(result, label) → bool to judge otherwise.
- record: `storage=` (a path: JSONLStorage, `record="full"` so every decision replays) keeps every ask hash-chained;
  `verify()` checks the chain, `replay_all()` re-checks every stored decision against this system.
  `save_calibration(path)` / `document(..., calibration=path)` keep the calibrated threshold across processes
  (System.save_guarantee / load_guarantee: a file made for another catalog, another model or other prompts — the
  task, a field's description — is refused).

The input of `ask` is a dict with the text under `text=` and anything else the rule and checks read; the system is
given the text and the rest as one fact named "state", made plain JSON data first (numpy / pandas scalars as
numbers, tuples and sets as lists; anything else that JSON cannot store is refused), so a stored decision replays.
DocumentSystem holds a solvi.System (`recipe.system`) and does not subclass it (System is final): the low-level calls
(`recipe.system.ask`, `.guarantee`, `.report`) stay open."""
from __future__ import annotations

import inspect
import math
import re
import typing
import warnings

from ..core.catalog import Answer, Catalog, Question

QUESTION = "answer"
STATE = "state"                     # the fact that carries the input without the text
MIN_CALIBRATION = 100               # below this, calibrate warns: the threshold rests on few examples

_DASHES = str.maketrans({"−": "-", "–": "-", "—": "-", "‑": "-", "‒": "-"})
_CURRENCY = re.compile(r"[$€£¥₽]|\b[A-Z]{3}\b")
_PERCENT = re.compile(r"\s*(?:%|\bper\s?cent\b)", re.IGNORECASE)
_SCALE_WORD = re.compile(r"\s*\b(?:thousands?|millions?|billions?|trillions?|mn|bn|tn)\b\.?", re.IGNORECASE)
_SCALE_SUFFIX = re.compile(r"(?<=\d)\s?(?:bn|mn|tn|k|K|m|M|b|B)\b", re.IGNORECASE)
_CORE = re.compile(r"[0-9][0-9,. ]*|\.[0-9]+")
_SIGN_BEFORE = re.compile(r"(?:[-−–—]|\()\s?(?:[$€£¥₽]|[A-Z]{3})?\s?$")


def _read_number(vtype, text):
    """A quoted number → vtype (see the module docs, "numbers"): the scale never multiplied, currency dropped, a
    percentage as its number, a sign in any dash before or after the currency or trailing, parentheses as a negative
    around a number with a separator, decimals or a currency; the digits and separators as
    solvi.core.types.span_value reads them. ValueError when it is not one number."""
    import unicodedata

    from ..core.types import span_value
    if not isinstance(text, str):
        raise ValueError(f"not a quoted number: {text!r}")
    t = unicodedata.normalize("NFKC", text).translate(_DASHES).strip()
    if re.search(r"[0-9][eE][+-]?[0-9]|_", t):
        raise ValueError("an exponent or an underscore is not how a document writes a number")
    if re.search(r"%\s*[0-9]", t):
        raise ValueError("a percent sign before the number")
    cur = bool(_CURRENCY.search(t))
    for rx in (_CURRENCY, _SCALE_WORD, _SCALE_SUFFIX, _PERCENT):
        t = rx.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\(\s*", "(", re.sub(r"\s*\)", ")", t))
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
        if not cur and not re.search(r"[.,]", t):
            raise ValueError("a bare number in parentheses (a year, a footnote?) is not read as a negative")
    elif t.startswith("-"):
        neg, t = True, t[1:].strip()
    elif t.endswith("-"):
        neg, t = True, t[:-1].strip()
    elif t.startswith("+"):
        t = t[1:].strip()
    if not _CORE.fullmatch(t or "-"):
        raise ValueError("not one number")
    v = span_value(vtype, t)
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("not a finite number")
    return -v if neg else v


def _sign_dropped(raw, doc):
    """A number quote whose text writes a sign right before it ("-1,234", "(1,234)", "-$1,234") that the quote left
    out → that context, else None (the quote is located as the grounding locates it)."""
    from ..core.deciders.llm import locate
    if not isinstance(raw, str) or not isinstance(doc, str) or raw.lstrip()[:1] in ("-", "−", "–", "—", "(", "+"):
        return None
    got = locate(raw, doc)
    if got is None:
        return None
    a, b = got
    m = _SIGN_BEFORE.search(doc[max(0, a - 8):a])
    if not m:
        return None
    if m.group().lstrip().startswith("(") and ")" not in doc[b:b + 24].split("\n")[0]:
        return None
    return doc[a - len(m.group()):b + (1 if doc[b:b + 1] == ")" else 0)]


def _plain(v, where="state"):
    """An input value → plain data the record stores and a replay reads back as the same: numpy / pandas scalars as
    Python numbers, tuples as lists, sets as sorted lists; dates as they are. TypeError for anything else."""
    import datetime
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    item = getattr(v, "item", None)
    if callable(item) and type(v).__module__.split(".")[0] in ("numpy", "pandas"):
        return _plain(item(), where)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: the key {k!r} is not a string (the input is stored as JSON)")
            out[k] = _plain(x, f"{where}[{k!r}]")
        return out
    if isinstance(v, (list, tuple)):
        return [_plain(x, f"{where}[{i}]") for i, x in enumerate(v)]
    if isinstance(v, (set, frozenset)):
        try:
            items = sorted(v)
        except TypeError:
            raise TypeError(f"{where}: a set whose values do not sort; pass a list") from None
        return [_plain(x, where) for x in items]
    raise TypeError(f"{where}: a {type(v).__name__} is not stored as JSON — give numbers, strings, dates, and lists "
                    "or dicts of them")


def _value_reader(fi):
    """A field → a function: the raw value its part gave → the field's value. A span of a number: _read_number; a span
    of another type (date, ...): span_value; a span of str and any other field (a choice): as it is. A value that is
    not a string (solvi.Unknown for a Maybe[...] field) passes as it is."""
    from decimal import Decimal

    from ..core.types import span_value
    t = fi.annotation
    if _span_source(fi) is None or t is str:
        return lambda raw: raw
    if t in (int, float, Decimal):
        return lambda raw: _read_number(t, raw) if isinstance(raw, str) else raw
    return lambda raw: span_value(t, raw) if isinstance(raw, str) else raw


def _span_source(fi):
    from ..core.types import SpanOf
    return next((m.source for m in fi.metadata if isinstance(m, SpanOf)), None)


def _part(name, inputs, body, wrapped=None):
    """A catalog part named `name` reading the facts `inputs` (by name) and calling body({input: value}). `wrapped`:
    the person's function behind it — the catalog's fingerprint hashes its code (inspect.unwrap), so a changed compute=
    or check is a changed catalog."""
    def run(**kw):
        return body(kw)
    run.__name__ = run.__qualname__ = name
    run.__signature__ = inspect.Signature([inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD) for x in inputs])
    if wrapped is not None:
        run.__wrapped__ = wrapped
    return run


def _stated_confidence(model):
    """An LLM decider made with logprobs="auto" → the same server and settings with logprobs=False, ask="confidence"
    (one stated confidence per read); any other model as it is."""
    from ..core.deciders import DecideModel
    from ..core.deciders.llm import LLMScorer
    sc = getattr(model, "scorer", None)
    if type(sc) is not LLMScorer or sc.logprobs != "auto":
        return model
    new = LLMScorer(sc.url, sc.model, sc._key, timeout=sc.timeout, retries=sc.retries, backoff=sc.backoff,
                    response_format=sc.response_format, logprobs=False, ask="confidence", max_tokens=sc.max_tokens,
                    seed=sc.seed, headers=sc._headers, extra_body=sc.extra_body, workers=sc.workers, opener=sc.opener,
                    sleep=sc.sleep)
    m = DecideModel(new, meta=model.meta, model_id=new.model_id, backend="llm")
    m.deterministic = False
    warnings.warn(f"solvi.document: {sc.model_id} was made with logprobs=\"auto\"; the recipe reads with "
                  "logprobs=False, ask=\"confidence\" (one stated confidence per read) on a copy of it — its usage "
                  "counters are on recipe.model. Make the model with logprobs=True or False to keep your own", UserWarning,
                  stacklevel=3)
    return m


class DocumentSystem:
    """What solvi.document() returns (see the module docs). system: the solvi.System; catalog, schema, model (the one the
    reads go to), text (the text fact's name), fields (field names in order), checks (check names), calibration (the
    last calibration report, or None)."""

    def __init__(self, system, schema, model, text, fields, checks):
        self.system, self.schema, self.model, self.text = system, schema, model, text
        self.fields, self.checks = list(fields), list(checks)
        self.catalog = system.catalog
        self.question = QUESTION
        self.calibration = None

    def given(self, state):
        """The input of ask → the given facts: the text, and the rest as the fact "state"."""
        if hasattr(state, "model_dump"):
            state = state.model_dump()
        if not isinstance(state, dict):
            raise TypeError(f"the input is a dict with the text under {self.text!r}, not {type(state).__name__}")
        if self.text not in state:
            raise KeyError(f"the input has no {self.text!r} (the text the fields are read from); it has "
                           f"{sorted(state)}")
        return {self.text: state[self.text],
                STATE: _plain({k: v for k, v in state.items() if k != self.text})}

    def ask(self, state, **kw):
        """One input → a solvi.Response: `res["answer"]` (answer, status "ok" / "abstain", confidence, why),
        `res.stored_id` when storage= was given. Keyword options go to System.ask."""
        return self.system.ask(self.given(state), [QUESTION], **kw)

    def calibrate(self, examples, *, max_risk=None, max_error=None, correct=None, **kw):
        """A calibrated threshold on the answer from labelled examples [(input, correct answer)] (System.guarantee:
        max_risk= or max_error=, and groups=, weak=, delta=, method= ...) → the calibration report (also kept as
        `self.calibration`). Fewer than 100 examples: a GuaranteeWarning. A number is judged right at the label's own
        precision (see the module docs) unless correct=(result, label) → bool is given."""
        from ..core.guarantees.guarantee import GuaranteeWarning
        examples = [(self.given(st), y) for st, y in examples]
        n = len(examples)
        small = None
        if 0 < n < MIN_CALIBRATION:
            small = (f"calibrating on {n} examples: the threshold rests on them and can move with a few more "
                     f"({MIN_CALIBRATION} or more recommended)")
            warnings.warn(small, GuaranteeWarning, stacklevel=2)
        default = correct is None and self.system.questions[QUESTION].answer.kind == "estimate"
        if default:
            correct = _same_number
        try:
            rep = self.system.guarantee(QUESTION, examples, max_risk=max_risk, max_error=max_error, correct=correct,
                                        **kw)
        except ValueError as e:
            if default:
                raise ValueError(f"{e} — the answers were judged by the default number match (at the label's "
                                 "precision): if right answers count as wrong, pass correct=(result, label) → bool") from e
            raise
        if small:
            rep.setdefault("warnings", []).append(small)
        self.calibration = rep
        return rep

    def save_calibration(self, path):
        """Write the calibration in force to a JSON file (System.save_guarantee) → path."""
        return self.system.save_guarantee(QUESTION, path)

    def load_calibration(self, path, **kw):
        """Attach a calibration written by save_calibration (System.load_guarantee: refuses a file made for another
        catalog) → its report."""
        rep = self.system.load_guarantee(path, **kw)
        self.calibration = rep
        return rep

    @property
    def storage(self):
        return self.system.storage

    def verify(self, **kw):
        """Check the store's hash chain (TraceStorage.verify) → its report; needs storage=."""
        return self._store().verify(**kw)

    def replay_all(self, trust_models=False, **filters):
        """Replay every stored decision against this system (TraceStorage.replay_all) → the ones that fail; needs
        storage=."""
        return self._store().replay_all(self.system, trust_models=trust_models, **filters)

    def _store(self):
        if self.system.storage is None:
            raise ValueError("this system keeps no record: give document(..., storage=\"decisions.jsonl\")")
        return self.system.storage

    def __repr__(self):
        return (f"DocumentSystem({self.schema.__name__}: {', '.join(self.fields)}; checks {self.checks or 'none'}; "
                f"{'calibrated' if QUESTION in self.system.guards else 'not calibrated'})")


def _same_number(r, y):
    """The default judge of a number answer: right when it rounds to the label at the label's precision — a label with
    d decimals (a number, or a string read as a quote: "1,500", "33.33", "(2.5)") matches within 0.5·10^-d."""
    try:
        if r.answer is None or isinstance(y, bool):
            return False
        a = float(r.answer)
        if isinstance(y, str):
            g, written = float(_read_number(float, y)), y
        else:
            g = float(y)
            written = repr(y) if isinstance(y, float) else str(y)
        if not (math.isfinite(a) and math.isfinite(g)):
            return False
        m = re.search(r"[.,](\d+)\D*$", written.strip())
        d = len(m.group(1)) if m and not (isinstance(y, float) and written.endswith(".0")) else 0
        if isinstance(y, str) and m and re.fullmatch(r"\d{3}", m.group(1)) and "," in written and "." not in written:
            d = 0                                      # "1,500": a thousands separator, not three decimals
        return abs(a - g) <= 0.5 * 10 ** -d + 1e-9 * abs(g)
    except (TypeError, ValueError, OverflowError):
        return False


def document(schema, model, *, task=None, text="doc", compute=None, answer=None, checks=None, storage=None,
             calibration=None, record="full", workers=1):
    """A decision over a document (see the module docs) → DocumentSystem.

    schema: a pydantic model class, one field per fact to read (`Span[float]`, `Span[date]`, `Span[str]`, a
    `Literal[...]` choice ...). model: a decider (`solvi.models.llm(url, name)`, `decider("solvi-base")`). task: text
    put before every field's description (what the document is). text: the name of the text fact in the input
    (default "doc"). compute: (facts, state) → the answer. answer: its answer type (default `Answer.estimate()`, a
    number). checks: {name: (facts, state) → bool}, hard checks. storage: a path (JSONLStorage with `record=`) or a
    TraceStorage. calibration: a file written by save_calibration, attached at once. workers: parallel steps inside
    one ask (System(workers=))."""
    from ..core.store import JSONLStorage
    from ..core.system import System
    if not (isinstance(schema, type) and isinstance(getattr(schema, "model_fields", None), dict)):
        raise TypeError(f"schema is a pydantic model class (one field per fact to read), not {schema!r}")
    if not schema.model_fields:
        raise ValueError(f"{schema.__name__} has no fields: nothing to read")
    if not isinstance(text, str) or not text:
        raise TypeError("text= is the name of the text fact in the input (default \"doc\")")
    fields = list(schema.model_fields)
    for bad in (text, STATE, QUESTION):
        if bad in fields:
            raise ValueError(f"{schema.__name__}.{bad}: a field cannot be named {bad!r} (the "
                             + ("text fact" if bad == text else "input fact" if bad == STATE else "question") + ")")
    if text == STATE:
        raise ValueError(f"text= cannot be {STATE!r}: that fact carries the rest of the input")
    checks = dict(checks or {})
    for name, f in checks.items():
        if not isinstance(name, str) or not name.isidentifier() or name in fields or name in (text, STATE, QUESTION) \
                or name in {f"{x}_sign" for x in fields} or name == "compute":
            raise ValueError(f"check name {name!r}: a Python identifier other than a field, <field>_sign, compute, "
                             f"{text!r}, {STATE!r} or {QUESTION!r}")
        if not callable(f):
            raise TypeError(f"check {name!r} is a function (facts, state) → bool, not {type(f).__name__}")
    if compute is not None and not callable(compute):
        raise TypeError(f"compute= is a function (facts, state) → the answer, not {type(compute).__name__}")
    if compute is None and len(fields) != 1:
        raise ValueError(f"{schema.__name__} has {len(fields)} fields: give compute=(facts, state) → the answer "
                         "(without it, a schema of one field answers with that field's value)")
    from decimal import Decimal

    from ..core.types import Bins, Ordinal, RankOf, SpanOf, closed_values
    for name, fi in schema.model_fields.items():           # a plain float / str field is not read from the text
        if _span_source(fi) is None and fi.annotation is not bool and closed_values(fi.annotation) is None:
            t = _type_name(fi)
            raise TypeError(f"{schema.__name__}.{name}: {t} is not a fact read from the text — write Span[{t}] (a "
                            "value quoted from it), or a Literal[...] / Enum / bool for a choice")
    if compute is None and answer is None:
        fi = schema.model_fields[fields[0]]
        if _span_source(fi) is not None:
            if fi.annotation not in (int, float, Decimal):
                raise TypeError(f"{schema.__name__}.{fields[0]} is a {_type_name(fi)}: without compute= the answer "
                                "is a number or a choice — give compute=(facts, state) → the answer and its answer=")
        else:
            answer = Answer.from_type(fi.annotation)
    model = _stated_confidence(model)
    cat = Catalog()
    readers = {}
    for name, fi in schema.model_fields.items():
        src = _span_source(fi)
        if src is not None and src != text:
            raise ValueError(f"{schema.__name__}.{name} is a span of {src!r}, but the text is {text!r}: write "
                             f"Span[..., {text!r}] or document(..., text={src!r})")
        t = fi.annotation
        marks = [m for m in fi.metadata if isinstance(m, (Ordinal, SpanOf, RankOf, Bins))]
        if marks:                                      # pydantic moves Annotated metadata to the field: put solvi's back
            t = typing.Annotated[(t, *marks)]
        extra = fi.json_schema_extra if isinstance(fi.json_schema_extra, dict) else {}
        desc = fi.description or fi.title or name.replace("_", " ").capitalize() + "?"
        opts = extra.get("options") or ()
        if isinstance(opts, dict):                     # the plain option list; the descriptions go into the question
            desc = desc.rstrip() + " The options mean: " + "; ".join(f"{o} — {d}" if d else str(o) for o, d in opts.items()) + "."
            opts = () if closed_values(fi.annotation) is not None else list(opts)   # a Literal / Enum: from the type
        args = {k: extra[k] for k in ("min_confidence", "min_act", "use_act", "max_error", "other", "score_value",
                                      "kind", "evidence", "coverage", "unit") if k in extra}
        q = f"{task.rstrip()} {desc}" if task else desc
        cat.fn(model.decision(name, q, text, options=opts, type=t, **args))
        readers[name] = _value_reader(fi)

    def facts_of(kw):
        vals = {}
        for name in fields:
            raw = kw[name]
            try:
                vals[name] = readers[name](raw)
            except ValueError as e:
                raise ValueError(f"{name}: {raw!r} does not read as {_type_name(schema.model_fields[name])}: {e}") from None
        return schema.model_construct(**vals)

    from decimal import Decimal as _Dec

    from ..core.slow.refine import Fail
    signed = []
    for name, fi in schema.model_fields.items():           # a number whose sign the quote left out abstains
        if _span_source(fi) is not None and fi.annotation in (int, float, _Dec):
            def sign_kept(kw, name=name):
                around = _sign_dropped(kw[name], kw[text])
                return True if around is None else Fail(f"{name}: the text writes {around!r}, the quote "
                                                        f"{kw[name]!r} left out its sign")
            cat.check(_part(f"{name}_sign", [name, text], sign_kept), hard=True)
            signed.append(f"{name}_sign")

    def finite(v):
        if isinstance(v, (float, _Dec)) and not isinstance(v, bool) and not math.isfinite(v):
            raise ValueError(f"the computed answer is {v!r}, not a finite number")
        return v

    inputs = fields + [STATE]
    for name, f in checks.items():
        cat.check(_part(name, inputs, lambda kw, f=f: f(facts_of(kw), kw[STATE]), wrapped=f), hard=True)
    if compute is None:
        only = fields[0]
        rule = _part("compute", [only], lambda kw: finite(readers[only](kw[only])))
    else:
        rule = _part("compute", inputs, lambda kw: finite(compute(facts_of(kw), kw[STATE])), wrapped=compute)
    cat.rule(QUESTION)(rule)
    if answer is None:
        answer = Answer.estimate()
    q = Question(QUESTION, task or f"{schema.__name__} → the computed answer", answer, requires=signed + list(checks))
    if isinstance(storage, str) or hasattr(storage, "__fspath__"):
        storage = JSONLStorage(storage, record=record)
    s = System(cat, [q], storage=storage, workers=workers)
    out = DocumentSystem(s, schema, model, text, fields, checks)
    if calibration is not None:
        out.load_calibration(calibration)
    return out


def _type_name(fi):
    from ..core.types import type_name
    return type_name(fi.annotation)


__all__ = ["DocumentSystem", "document"]
