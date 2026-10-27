"""solvi.document — a decision over a document: a model reads one fact per field, code computes the answer.

    import solvi
    from pydantic import BaseModel, Field
    from solvi import Span
    from solvi.models import llm

    class Operands(BaseModel):
        a: Span[float] = Field(description="the first number the question names, exactly as written")
        b: Span[float] = Field(description="the second number, exactly as written")

    sys = solvi.document(Operands, llm(URL, MODEL, api_key=KEY),
                         task="The text starts with a QUESTION, then a table and notes.",
                         compute=lambda f, s: f.a - f.b,                     # f: the facts, s: the rest of the input
                         checks={"b_not_zero": lambda f, s: f.b != 0},
                         storage="decisions.jsonl")
    sys.calibrate([({"doc": d, "op": op}, gold) for d, op, gold in labelled], max_risk=0.10)
    res = sys.ask({"doc": text, "op": "diff"})
    res["answer"].answer, res["answer"].status, res["answer"].why, res.stored_id
    sys.verify(), sys.replay_all()

The scheme it sets up (all of it existing low-level blocks, wired):

- one read per field: each field of the schema is one decision part (`DecideModel.decision`), its description the
  question (after `task=`), the text fact `text=` (default "doc") its input — one model call per field, the same text
  each call. A field with `json_schema_extra={"options": {option: description}}` asks with the plain option list and
  the descriptions in the question's text (a model that echoes "option: description" back is spared the chance).
- grounding: a `Span[...]` field must be found in the text (normalized view; a number as a number of its own, not
  inside a longer one — solvi.core.catalog.locate, cuts_number); not found, the field escalates and the answer
  abstains with the reason. The kept value is the text's own substring.
- numbers: a `Span[float]` (int, Decimal) is read from what was quoted: "$1,250", "1,250 million" (the scale word is
  left alone: 1250, in the document's own unit), "(1,234)" (an accounting negative: −1234), "12.5%" (12.5); a
  `Span[date]` and other types by solvi.core.types.span_value. A value that does not read abstains with the reason.
- code computes: `compute=(facts, state) → value` is the question's rule; `facts` is the schema filled with the read
  values, `state` the input without the text. The answer is a number (`Answer.estimate()`) unless `answer=` says
  otherwise (`solvi.Answer.choice(["pay", "hold"])`, `Answer.yes_no()`); its confidence is the weakest fact's.
  Without `compute=` a schema of one field answers with that field's value.
- checks: `checks={name: (facts, state) → bool}` are hard checks on the answer: one that fails makes it abstain
  ("failed check").
- confidence: an LLM made with `logprobs="auto"` (llm()'s default) is asked for one stated confidence per read
  (`logprobs=False, ask="confidence"`) — the signal measured on this shape of task; a gateway that mixes logprob and
  stated replies cannot then split a calibration. A model made with `logprobs=True` or `False` is used as given.
  `sys.model` is the model the reads go to.
- calibration: `calibrate(examples, max_risk=)` is System.guarantee on the answer (conformal risk control; max_error=
  for learn-then-test; groups=, correct=, weak= pass through): below the threshold the answer abstains. Fewer than
  100 examples: a GuaranteeWarning (the threshold rests on few examples and can move); a signal that does not
  separate right from wrong raises, as System.guarantee does. A number is judged right when it is equal up to float
  noise (math.isclose); pass correct= for a rounding tolerance.
- record: `storage=` (a path: JSONLStorage, `record="full"` so every decision replays) keeps every ask hash-chained;
  `verify()` checks the chain, `replay_all()` re-checks every stored decision against this system.
  `save_calibration(path)` / `document(..., calibration=path)` keep the calibrated threshold across processes
  (System.save_guarantee / load_guarantee: a file made for another catalog is refused).

The input of `ask` is a dict with the text under `text=` and anything else the rule and checks read; the system is
given the text and the rest as one fact named "state". DocumentSystem holds a solvi.System (`sys.system`) and does
not subclass it (System is final): the low-level calls (`sys.system.ask`, `.guarantee`, `.report`) stay open."""
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

_SCALE = re.compile(r"\s*\b(?:thousands?|millions?|billions?|trillions?|mn|bn|tn)\b\.?\s*$", re.IGNORECASE)
_PERCENT = re.compile(r"\s*(?:%|percent|per cent)\s*$", re.IGNORECASE)
_ACCOUNTING = re.compile(r"([$€£¥]?)\s*\((.+)\)\s*(%?)")


def _read_number(vtype, text):
    """A quoted number → vtype: currency signs and thousands separators as solvi.core.types.span_value reads them, a
    trailing scale word left alone ("1,250 million" → 1250, the document's unit), a percentage as its number
    ("12.5%" → 12.5), parentheses as an accounting negative ("(1,234)" → −1234). ValueError when it is not one."""
    from ..core.types import span_value
    t = text.strip()
    neg = False
    m = _ACCOUNTING.fullmatch(t)
    if m:
        neg, t = True, (m.group(1) + m.group(2) + m.group(3)).strip()
    t = _SCALE.sub("", t)
    t = _PERCENT.sub("", t)
    v = span_value(vtype, t)
    return -v if neg else v


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
    if not isinstance(sc, LLMScorer) or sc.logprobs != "auto":
        return model
    new = LLMScorer(sc.url, sc.model, sc._key, timeout=sc.timeout, retries=sc.retries, backoff=sc.backoff,
                    response_format=sc.response_format, logprobs=False, ask="confidence", max_tokens=sc.max_tokens,
                    seed=sc.seed, headers=sc._headers, extra_body=sc.extra_body, workers=sc.workers, opener=sc.opener,
                    sleep=sc.sleep)
    m = DecideModel(new, meta=model.meta, model_id=new.model_id, backend="llm")
    m.deterministic = False
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
        return {self.text: state[self.text], STATE: {k: v for k, v in state.items() if k != self.text}}

    def ask(self, state, **kw):
        """One input → a solvi.Response: `res["answer"]` (answer, status "ok" / "abstain", confidence, why),
        `res.stored_id` when storage= was given. Keyword options go to System.ask."""
        return self.system.ask(self.given(state), [QUESTION], **kw)

    def calibrate(self, examples, *, max_risk=None, max_error=None, correct=None, **kw):
        """A calibrated threshold on the answer from labelled examples [(input, correct answer)] (System.guarantee:
        max_risk= or max_error=, and groups=, weak=, delta=, method= ...) → the calibration report (also kept as
        `self.calibration`). Fewer than 100 examples: a GuaranteeWarning. A number is judged right up to float noise
        unless correct=(result, label) → bool is given."""
        from ..core.guarantees.guarantee import GuaranteeWarning
        examples = [(self.given(st), y) for st, y in examples]
        n = len(examples)
        small = None
        if 0 < n < MIN_CALIBRATION:
            small = (f"calibrating on {n} examples: the threshold rests on them and can move with a few more "
                     f"({MIN_CALIBRATION} or more recommended)")
            warnings.warn(small, GuaranteeWarning, stacklevel=2)
        if correct is None and self.system.questions[QUESTION].answer.kind == "estimate":
            correct = _same_number
        rep = self.system.guarantee(QUESTION, examples, max_risk=max_risk, max_error=max_error, correct=correct, **kw)
        if small and isinstance(rep, dict):
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
    try:
        return r.answer is not None and math.isclose(float(r.answer), float(y), rel_tol=1e-9, abs_tol=1e-9)
    except (TypeError, ValueError):
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
        if not isinstance(name, str) or not name.isidentifier() or name in fields or name in (text, STATE):
            raise ValueError(f"check name {name!r}: a Python identifier other than a field, {text!r} or {STATE!r}")
        if not callable(f):
            raise TypeError(f"check {name!r} is a function (facts, state) → bool, not {type(f).__name__}")
    if compute is not None and not callable(compute):
        raise TypeError(f"compute= is a function (facts, state) → the answer, not {type(compute).__name__}")
    if compute is None and len(fields) != 1:
        raise ValueError(f"{schema.__name__} has {len(fields)} fields: give compute=(facts, state) → the answer "
                         "(without it, a schema of one field answers with that field's value)")
    model = _stated_confidence(model)

    from ..core.types import Bins, Ordinal, RankOf, SpanOf, closed_values
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

    inputs = fields + [STATE]
    for name, f in checks.items():
        cat.check(_part(name, inputs, lambda kw, f=f: f(facts_of(kw), kw[STATE]), wrapped=f), hard=True)
    if compute is None:
        only = fields[0]
        rule = _part("compute", [only], lambda kw: readers[only](kw[only]))
    else:
        rule = _part("compute", inputs, lambda kw: compute(facts_of(kw), kw[STATE]), wrapped=compute)
    cat.rule(QUESTION)(rule)
    if answer is None:
        answer = Answer.estimate()
    q = Question(QUESTION, task or f"{schema.__name__} → the computed answer", answer, requires=list(checks))
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
