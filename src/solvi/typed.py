"""Typed facts: a catalog function's type hints are the types of the facts it reads and sets.

  @cat.fn
  def risk_score(risk_points: dict[str, float]) -> float: ...

- registration: the catalog records each fact's type (`cat.types`, `cat.readers`) and checks every producer's return type
  against every consumer's argument type; a definite mismatch is a FactTypeError naming both functions (the check is
  conservative: what may pass, e.g. `int | None` into `int`, is left to run time);
- run time: a typed part's inputs (given or computed) are validated / coerced with pydantic before the call, its output after
  it; a value that fails is rejected — the fact is missing (the next producer runs, else dependent answers abstain) and the
  step's error starts with "type rejected" (safeguard `type_rejected`). A Literal / Enum return type is a closed set: a value
  outside it is rejected as "outside the options";
- untyped parts (no annotation, Any, object) cost nothing: nothing is validated, and pydantic is not even imported.

pydantic is imported on the first typed part (or BaseModel input, or serialization), so `import solvi` stays light."""
from __future__ import annotations

import inspect
import sys
import typing
import warnings
from enum import Enum
from typing import Any, Literal, Union, get_args, get_origin

from .core import NotStated, Quote
from .provenance import TYPE_REJECTED

_NOHINT = object()


class FactTypeError(TypeError):
    """A catalog error: a producer's return type cannot be what a consumer reads, or a rule's return type does not fit its
    question's answer type."""


def _untyped(t):
    return t is _NOHINT or t is Any or t is object or t is inspect.Parameter.empty


def _strip(t):
    """Annotated[X, ...] → X, NewType → its base."""
    while True:
        if get_origin(t) is typing.Annotated:
            t = get_args(t)[0]
        elif hasattr(t, "__supertype__"):
            t = t.__supertype__
        else:
            return t


def _is_union(t):
    o = get_origin(t)
    return o is Union or (sys.version_info >= (3, 10) and o is __import__("types").UnionType)


def _members(t):
    return get_args(t) if _is_union(t) else (t,)


def type_name(t):
    """A short readable name: dict[str, float], Literal['a', 'b'], Invoice | None."""
    if t is type(None):
        return "None"
    if t is Any or t is Ellipsis:
        return "Any" if t is Any else "..."
    o, a = get_origin(t), get_args(t)
    if _is_union(t):
        return " | ".join(type_name(m) for m in a)
    if o is Literal:
        return "Literal[" + ", ".join(repr(v) for v in a) + "]"
    if o is typing.Annotated:
        return type_name(a[0])
    if o is not None and a:
        return f"{getattr(o, '__name__', repr(o).replace('typing.', ''))}[{', '.join(type_name(x) for x in a)}]"
    if isinstance(t, type):
        return t.__name__
    return repr(t).replace("typing.", "")


# --- hints of a catalog function
def hints(f, kind):
    """→ ({argument: type} for the typed arguments, or None; the return type, or None when untyped). A Quote / Decision return
    annotation says nothing about the fact's type (the fact is the plain value), so it counts as untyped."""
    if not (inspect.isfunction(f) or inspect.ismethod(f)) or not getattr(f, "__annotations__", None):
        return None, None
    try:
        h = typing.get_type_hints(f, include_extras=True)
    except Exception:  # noqa: BLE001 — a name that cannot be resolved: resolve one by one, skip (with a warning) what fails
        h, g = {}, getattr(f, "__globals__", {})
        for k, v in f.__annotations__.items():
            if isinstance(v, str):
                try:
                    v = eval(v, g)  # noqa: S307
                except Exception:  # noqa: BLE001
                    warnings.warn(f"{f.__name__}: cannot resolve the type {v!r} of {k}; it is not checked", stacklevel=5)
                    continue
            h[k] = v
    ret = h.pop("return", _NOHINT)
    params = list(inspect.signature(f).parameters)
    ins = {x: h[x] for x in params if x in h and not _untyped(h[x])}
    if kind == "constraint":
        return None, None
    if not _untyped(ret):
        from .core import Decision
        if all(_strip(m) in (Quote, Decision) for m in _members(_strip(ret))):
            ret = _NOHINT
    return (ins or None), (None if _untyped(ret) else ret)


# --- validation (pydantic, lazily)
_ADAPTERS = {}


class Spec:
    """A compiled type: `cls` for the exact-type fast path (plain classes), the pydantic TypeAdapter, the closed set of values
    (Literal / Enum, else None) and a display name."""
    __slots__ = ("type", "cls", "ta", "closed", "name")

    def __init__(self, t, ta):
        s = _strip(t)
        self.type, self.ta, self.name = t, ta, type_name(t)
        self.cls = s if (s is t and isinstance(s, type) and get_origin(s) is None) else None
        cv = closed_values(t)
        self.closed = cv[0] if cv else None


def adapter(t):
    try:
        ta = _ADAPTERS.get(t)
    except TypeError:                                  # unhashable type (e.g. Annotated with a list): no cache
        ta = None
    if ta is None:
        from pydantic import ConfigDict, TypeAdapter
        from pydantic.errors import PydanticSchemaGenerationError
        try:
            ta = TypeAdapter(t)
        except PydanticSchemaGenerationError:          # arbitrary classes: an isinstance check
            ta = TypeAdapter(t, config=ConfigDict(arbitrary_types_allowed=True))
        try:
            _ADAPTERS[t] = ta
        except TypeError:
            pass
    return ta


def _NUMBER_TYPES():
    from decimal import Decimal
    return (int, float, Decimal)


def span_value(vtype, text, strict=False):
    """The value a typed span states (`Span[float]`, `Span[date]`, ...) → the value; ValueError when the text is not one.
    First the type's own reading of the text ("149.90", "2026-09-12") — except a number the guarded parser calls
    ambiguous ("2.500", "1.000": refused, as below). When that fails and the type is a date or a number
    (date, int, float, Decimal), the deterministic parsers of solvi.textin read what people write — "21 July 2026", "July
    21, 2026", "18 октября 2026 г.", "21.07.2026"; "41,908.56 USD", "EUR 18,851.12", "1 500 000 руб", "1.5 million" —
    and refuse what would be a guess: a numeric date that reads both ways ("03/04/2026", "12.09.2026": day or month
    first?), a date without a year, two dates or two numbers, "1.000", a percentage. strict=True: the type's own
    reading only."""
    t = text.strip()
    ta = adapter(vtype)
    if not strict and vtype in _NUMBER_TYPES():       # a number: what the guarded parser calls ambiguous ("2.500": two
        from .textin import ParseError, parse_number  # and a half, or two thousand five hundred?) is not read by the
        try:                                          # type's own reading either
            parse_number(t, {"integer": True} if vtype is int else None)
        except ParseError as why:
            if "ambiguous" in str(why):
                raise ValueError(str(why)) from None
    try:
        return ta.validate_python(t)
    except ValueError as err:
        if strict:
            raise
        import datetime
        from decimal import Decimal

        from .textin import ParseError, parse_date, parse_number
        try:
            if vtype is datetime.date:
                read = set()
                for dayfirst in (True, False):
                    try:
                        read.add(parse_date(t, {"dayfirst": dayfirst}))
                    except ParseError:
                        pass
                if len(read) > 1:
                    raise ParseError(f"{t!r} reads as {' or '.join(sorted(read))}: day or month first?")
                if not read:
                    parse_date(t)                      # raises with the reason
                canon = read.pop()
            elif vtype in (int, float, Decimal):
                canon = parse_number(t, {"integer": True} if vtype is int else None)
            else:
                raise err
        except ParseError as why:
            raise ValueError(f"{_msg(err)}; {why}") from None
        return ta.validate_python(canon)


def spec(t, where=""):
    """A Spec for type t, or None (with a warning) when pydantic cannot validate it."""
    try:
        return Spec(t, adapter(t))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"{where}: type {type_name(t)} cannot be validated ({type(e).__name__}); it is not checked", stacklevel=5)
        return None


def _short(v, n=60):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _msg(e):
    try:
        err = e.errors()[0]
        loc = ".".join(str(x) for x in err.get("loc", ()))
        return (f"{loc}: " if loc else "") + err["msg"]
    except Exception:  # noqa: BLE001
        return str(e).splitlines()[0][:120]


def check(s, v):
    """Validate / coerce one value → (value, None) or (None, pydantic's message)."""
    if type(v) is s.cls:
        return v, None
    try:
        return s.ta.validate_python(v), None
    except ValueError as e:                            # pydantic_core.ValidationError is a ValueError
        return None, _msg(e)


def typed_in(part, args, known=None):
    """Validate / coerce a typed part's inputs → (arguments for the call, None) or (args, the rejection reason). known: fact →
    the type its value already passed in this run; an argument of that same type is not validated again."""
    out = None
    for x, s in part.tin.items():
        v = args[x]
        if type(v) is s.cls or (known and known.get(x) == s.type):
            continue
        if isinstance(v, Quote):                      # a Quote given in init_state: its value is the fact
            v = v.value
        nv, err = check(s, v)
        if err is not None:
            return args, f"{TYPE_REJECTED}: {x} = {_short(v)} is not {s.name} ({err})"
        if nv is not args[x]:
            out = dict(args) if out is None else out
            out[x] = nv
    return (args if out is None else out), None


def typed_out(part, value):
    """Validate / coerce a typed part's plain output → (value, None) or (None, the rejection reason)."""
    s = part.tout
    nv, err = check(s, value)
    if err is None:
        return nv, None
    if s.closed is not None:
        from .provenance import OUTSIDE_OPTIONS
        return None, f"decision {_short(value)} is {OUTSIDE_OPTIONS} {s.closed} (return type {s.name})"
    return None, f"{TYPE_REJECTED}: returned {_short(value)}, not {s.name} ({err})"


def compile_part(p):
    """Compile a part's types (p.types, p.returns) into validators (p.tin, p.tout). A rule returning an answer primitive
    (Maybe, Span, Rank, Estimate) is checked when its answer is resolved instead."""
    if p.types:
        tin = {x: spec(t, f"{p.name}({x})") for x, t in p.types.items()}
        p.tin = {x: s for x, s in tin.items() if s is not None} or None
    if p.returns is not None and not (p.kind == "rule" and is_primitive(p.returns)):
        p.tout = spec(p.returns, f"{p.name} → return")


# --- the registration check
_NUMERIC = {(int, float), (int, complex), (float, complex), (bool, float)}
_SEQ = (list, tuple, set, frozenset)


def _parsed_from_str(c):
    import datetime
    import decimal
    import pathlib
    import uuid
    return issubclass(c, (datetime.date, datetime.time, datetime.timedelta, decimal.Decimal, uuid.UUID, pathlib.PurePath,
                          Enum, bytes))


def _record_like(c):
    import dataclasses
    return (hasattr(c, "model_fields") or dataclasses.is_dataclass(c) or typing.is_typeddict(c)
            if isinstance(c, type) else False)


def _cls_ok(p, c):
    from collections.abc import Mapping
    if issubclass(p, c) or (p, c) in _NUMERIC:
        return True
    if p in _SEQ and c in _SEQ:                         # pydantic converts between sequences
        return True
    if issubclass(p, Mapping) and _record_like(c):      # a dict validates into a model / dataclass / TypedDict
        return True
    if p is str and _parsed_from_str(c):                # "2026-09-27" → date, "12.50" → Decimal
        return True
    if p in (int, float) and c.__name__ == "Decimal":
        return True
    return False


def _value_fits(v, c):
    c = _strip(c)
    if _untyped(c) or _is_union(c):
        return _untyped(c) or any(_value_fits(v, m) for m in _members(c))
    if get_origin(c) is Literal:
        return v in get_args(c)
    cls = get_origin(c) or c
    if not isinstance(cls, type):
        return True
    return isinstance(v, cls) or (cls is float and type(v) is int) or (isinstance(v, str) and _parsed_from_str(cls))


def compatible(p, c):
    """Can a value of type p be read as type c? False only for a definite mismatch (no value of p can pass as c)."""
    p, c = _strip(p), _strip(c)
    if _untyped(p) or _untyped(c) or p == c:
        return True
    ps, cs = _members(p), _members(c)
    if len(ps) > 1 or len(cs) > 1:                      # some overlap may pass (validated at run time)
        return any(compatible(a, b) for a in ps for b in cs)
    po, co = get_origin(p), get_origin(c)
    if po is Literal:
        return any(_value_fits(v, c) for v in get_args(p))
    if co is Literal:
        pc = po or p
        return not isinstance(pc, type) or any(isinstance(v, pc) for v in get_args(c))
    pc, cc = po or p, co or c
    if not isinstance(pc, type) or not isinstance(cc, type):     # TypeVar, Callable, Protocol, ...: not decided here
        return True
    if not _cls_ok(pc, cc):
        return False
    pa, ca = get_args(p), get_args(c)
    if pa and ca and len(pa) == len(ca) and Ellipsis not in pa and Ellipsis not in ca:
        return all(compatible(a, b) for a, b in zip(pa, ca))
    if pa and ca and pc in _SEQ and cc in _SEQ:          # list[X] vs tuple[X, ...]
        return compatible(pa[0], ca[0])
    return True


def register(catalog, p):
    """Check a new part's types against the catalog (raises FactTypeError before anything is changed) → a function that
    records them."""
    fact = p.provides or p.name
    ins = dict(p.types or {}) if p.kind != "constraint" else {}
    for x, t in ins.items():
        for prod, pt in producers(catalog, x):
            if not compatible(pt, t):
                raise FactTypeError(f"fact {x!r}: {prod} returns {type_name(pt)}, but {_label(p)} reads "
                                    f"{x}: {type_name(t)}")
    if p.returns is not None and p.kind not in ("rule", "constraint"):
        for reader, t in (catalog.readers.get(fact) or {}).items():
            if not compatible(p.returns, t):
                raise FactTypeError(f"fact {fact!r}: {_fname(p)} returns {type_name(p.returns)}, but {reader} reads "
                                    f"{fact}: {type_name(t)}")

    def commit():
        who = _label(p)
        if p.kind == "rule":                            # a replaced rule drops what the old one read
            forget_rule(catalog, p.question)
        for x, t in ins.items():
            catalog.readers.setdefault(x, {})[who] = t
        if p.returns is not None and p.kind not in ("rule", "constraint"):
            catalog.types.setdefault(fact, p.returns)
    return commit


def forget_rule(catalog, question):
    for rs in catalog.readers.values():
        for k in [k for k in rs if k.startswith("rule ") and k.endswith(f"({question})")]:
            del rs[k]


def producers(catalog, fact):
    """[(function name, return type)] of the typed producers of a fact."""
    g = catalog.parts.get(fact)
    if g is None:
        return []
    return [(_fname(a), a.returns) for a in (g.alternatives or [g]) if a.returns is not None]


def _fname(p):
    return getattr(p.func, "__name__", p.name)


def _label(p):
    return f"rule {_fname(p)} ({p.question})" if p.kind == "rule" else _fname(p)


# --- closed sets and answer types
def closed_values(t):
    """Literal / Enum (optionally in a list / set / tuple, optionally | None) → (values, multi), else None."""
    t = _strip(t)
    if _is_union(t):
        ms = [m for m in get_args(t) if m is not type(None) and m is not NotStated]
        return closed_values(ms[0]) if len(ms) == 1 else None
    o = get_origin(t)
    if o is Literal:
        return [v.value if isinstance(v, Enum) else v for v in get_args(t)], False
    if isinstance(t, type) and issubclass(t, Enum):
        return [m.value for m in t], False
    if o in _SEQ:
        a = get_args(t)
        inner = closed_values(a[0]) if a else None
        if inner is not None and not inner[1]:
            return inner[0], True
    return None


class Ordinal:
    """Marks a closed set as ordered levels, lowest first: `Annotated[Literal["low", "medium", "high"], Ordinal()]`, or
    `Scale["low", "medium", "high"]`. An Enum class with `__solvi_ordinal__ = True` is ordinal too."""

    def __repr__(self):
        return "Ordinal()"

    def __eq__(self, other):
        return isinstance(other, Ordinal)

    def __hash__(self):
        return hash("solvi.Ordinal")


class Scale:
    """An ordinal type from its levels, lowest first: `Scale[Literal["low", "medium", "high"]]` (type checkers and linters
    read it as the Literal), `Scale["low", "medium", "high"]`, `Scale[1, 2, 3, 4, 5]`, or `Scale[AnEnum]` — that is
    `Annotated[Literal[...], Ordinal()]`. As a question: Answer.ordinal; as a decision: a score (see solvi.decide)."""

    def __class_getitem__(cls, levels):
        if get_origin(levels) is Literal or (isinstance(levels, type) and issubclass(levels, Enum)):
            return typing.Annotated[levels, Ordinal()]
        levels = levels if isinstance(levels, tuple) else (levels,)
        return typing.Annotated[Literal[levels], Ordinal()]


def _unoptional(t):
    """`X | None` and `Maybe[X]` (also inside Annotated) → X; anything else as it is."""
    s = _strip(t)
    if _is_union(s):
        ms = [m for m in get_args(s) if m is not type(None) and m is not NotStated]
        if len(ms) == 1:
            return ms[0]
    return t


# --- answer primitives: Maybe, Span, Rank, Estimate
class SpanOf:
    """Marks a type as a span of a text fact: `Span[float]` is `Annotated[float, SpanOf("doc")]`."""

    def __init__(self, source="doc"):
        self.source = source

    def __repr__(self):
        return f"SpanOf({self.source!r})"

    def __eq__(self, other):
        return isinstance(other, SpanOf) and other.source == self.source

    def __hash__(self):
        return hash(("solvi.SpanOf", self.source))


class RankOf:
    """Marks a tuple of options as a ranking (top k): `Rank[Literal[...], 2]`."""

    def __init__(self, k=None):
        self.k = k

    def __repr__(self):
        return f"RankOf({self.k!r})"

    def __eq__(self, other):
        return isinstance(other, RankOf) and other.k == self.k

    def __hash__(self):
        return hash(("solvi.RankOf", self.k))


class Bins:
    """Marks a number as an estimate over bins cut at `edges`: `Annotated[float, Bins([0, 7, 14, 30], coverage=0.9,
    unit="days")]`, or `Estimate[0, 7, 14, 30]` (see Answer.estimate)."""

    def __init__(self, edges, coverage=0.8, unit=None, integer=None):
        self.edges, self.coverage, self.unit, self.integer = tuple(edges), float(coverage), unit, integer

    def __repr__(self):
        return f"Bins({list(self.edges)}, coverage={self.coverage}, unit={self.unit!r})"

    def __eq__(self, other):
        return isinstance(other, Bins) and (other.edges, other.coverage, other.unit, other.integer) == \
            (self.edges, self.coverage, self.unit, self.integer)

    def __hash__(self):
        return hash(("solvi.Bins", self.edges, self.coverage, self.unit, self.integer))


class Maybe:
    """`Maybe[T]` is `T | NotStated`: the answer may be `solvi.Unknown` — "the text does not state it", a real answer with a
    confidence, distinct from "no" and from an abstention."""

    def __class_getitem__(cls, t):
        return typing.Union[t, NotStated]


class Span:
    """`Span[T]` (or `Span[T, "source"]`): an exact substring of a given text fact (default "doc"), coerced to T with
    pydantic (`Span[float]`: "12.50" → 12.5). As a question: Answer.span; as a decision: a pointer over the input."""

    def __class_getitem__(cls, params):
        t, src = (params if isinstance(params, tuple) else (params, "doc"))
        return typing.Annotated[t, SpanOf(src)]


class Rank:
    """`Rank[Literal["a", "b", "c"]]` (or `Rank[Literal[...], k]`, `Rank[AnEnum, k]`): the options best first, the top k
    (all by default), as a tuple. As a question: Answer.rank; as a decision: scores over the options."""

    def __class_getitem__(cls, params):
        t, k = (params if isinstance(params, tuple) else (params, None))
        return typing.Annotated[tuple[t, ...], RankOf(k)]


class Estimate:
    """`Estimate[0, 7, 14]`: a number estimated over the bins cut at these edges — less than 0, 0–6, 7–13, 14 or more —
    with an 80% interval (`Annotated[float, Bins(edges, coverage=, unit=)]` for other settings). As a question:
    Answer.estimate; as a decision ("number"): the bins as ordered options."""

    def __class_getitem__(cls, edges):
        return typing.Annotated[float, Bins(edges if isinstance(edges, tuple) else (edges,))]


def split_unknown(t):
    """→ (t without NotStated, whether NotStated was a member). `X | None` keeps its None."""
    s = _strip(t)
    if not _is_union(s):
        return t, False
    ms = get_args(s)
    if NotStated not in ms:
        return t, False
    rest = [m for m in ms if m is not NotStated]
    return (rest[0] if len(rest) == 1 else typing.Union[tuple(rest)]), True


def _marker(t, cls):
    """The first metadata instance of cls in Annotated[...] t (after `| None`), or None."""
    t = _unoptional(t)
    while get_origin(t) is typing.Annotated:
        for m in t.__metadata__:
            if isinstance(m, cls):
                return m
        t = get_args(t)[0]
    return None


def is_primitive(t):
    """Does a type declare an answer primitive (Maybe / NotStated, Span, Rank, Estimate)? Such a rule's output is checked
    when the answer is resolved (solvi.primitives), not by a return-type validator."""
    if t is None:
        return False
    t2, unk = split_unknown(t)
    return unk or any(_marker(t2, c) is not None for c in (SpanOf, RankOf, Bins))


def primitive_answer(t):
    """A primitive type (without NotStated) → its AnswerType, or None for an ordinary type."""
    from .core import Answer
    m = _marker(t, SpanOf)
    if m is not None:
        inner = _strip(_unoptional(t))
        return Answer.span(source=m.source, type=None if inner is str else inner)
    m = _marker(t, RankOf)
    if m is not None:
        cv = closed_values(_strip(_unoptional(t)))
        if cv is None:
            raise TypeError(f"Rank[...] needs a Literal or an Enum, not {type_name(t)}")
        return Answer.rank(cv[0], m.k)
    m = _marker(t, Bins)
    if m is not None:
        return Answer.estimate(list(m.edges), coverage=m.coverage, unit=m.unit, integer=m.integer)
    return None


def is_ordinal(t):
    """Is t marked as ordered levels (Scale[...], Annotated[..., Ordinal()], an Enum with __solvi_ordinal__)?"""
    if get_origin(t) is typing.Annotated:
        if any(isinstance(m, Ordinal) for m in t.__metadata__):
            return True
        return is_ordinal(get_args(t)[0])
    s = _strip(t)
    if _is_union(s):
        ms = [m for m in get_args(s) if m is not type(None)]
        return len(ms) == 1 and is_ordinal(ms[0])
    return isinstance(s, type) and issubclass(s, Enum) and bool(getattr(s, "__solvi_ordinal__", False))


def question_kind(t):
    """A Python type → how a model decides it: (kind, values, as_bool) with kind "noul" (bool, Literal["yes", "no"]),
    "score" (an ordinal type: Scale[...], 2–10 levels), "multi" (list / set / tuple of a Literal or Enum) or "choice"
    (a Literal or Enum). `X | None` is X. as_bool: the decision's value is True / False (a bool type)."""
    t = _unoptional(t)
    s = _strip(t)
    if s is bool:
        return "noul", ["yes", "no"], True
    cv = closed_values(s)
    if cv is None:
        raise TypeError(f"no decision for type {type_name(t)}: use bool, Literal[...], an Enum, Scale[...] or "
                        "list[Literal[...]]")
    vals, multi = cv
    if multi:
        return "multi", vals, False
    if is_ordinal(t):
        return "score", vals, False
    if sorted(map(str, vals)) == ["no", "yes"]:
        return "noul", ["yes", "no"], False
    return "choice", vals, False


def answer_type_of(t, ordinal=False):
    """A Python type → an AnswerType: bool → yes_no; Literal / Enum → choice (ordinal=True, or an ordinal type such as
    Scale[...]: ordinal, in declaration order; Literal["yes", "no"] → yes_no); list / set / tuple of a Literal or Enum →
    multi. `X | None` is X (None = abstain)."""
    from .core import Answer
    t, unknown = split_unknown(t)
    if unknown:
        return Answer.maybe(answer_type_of(t, ordinal))
    pa = primitive_answer(t)
    if pa is not None:
        return pa
    ordinal = ordinal or is_ordinal(t)
    s = _strip(t)
    if _is_union(s):
        ms = [m for m in get_args(s) if m is not type(None)]
        if len(ms) == 1:
            return answer_type_of(ms[0], ordinal)
    if s is bool:
        return Answer.yes_no()
    cv = closed_values(s)
    if cv is None:
        raise TypeError(f"no answer type for {type_name(t)}: use bool, Literal[...], an Enum, or list[Literal[...]]")
    vals, multi = cv
    if multi:
        return Answer.multi(vals)
    if ordinal:
        return Answer.ordinal(vals)
    return Answer.yes_no() if sorted(vals) == ["no", "yes"] else Answer.choice(vals)


def check_answer(rule, q):
    """A typed rule's return type must fit its question's answer type (raises FactTypeError)."""
    t = rule.returns
    at = q.answer
    if at is None or t is None:
        return
    if is_primitive(t):
        t2, unknown = split_unknown(t)
        got = answer_type_of(t2)
        if unknown and not at.unknown:
            raise FactTypeError(f"rule {_fname(rule)} may return Unknown (not stated), but question {q.name!r} does not "
                                "allow it: declare it with Maybe[...] / Answer.maybe(...)")
        if got.kind != at.kind:
            raise FactTypeError(f"rule {_fname(rule)} returns {type_name(t)} ({got.kind}), but question {q.name!r} is "
                                f"{at.kind}")
        extra = [v for v in got.options if got.kind == "rank" and v not in at.options]
        if extra:
            raise FactTypeError(f"rule {_fname(rule)} ranks {extra}, which are not options of question {q.name!r}")
        if got.kind in ("yes_no", "choice", "ordinal", "multi"):
            check_answer(type("R", (), {"returns": t2, "func": rule.func, "name": rule.name})(), q)
        return
    s = _strip(t)
    ms = [m for m in _members(s) if m is not type(None)]
    if len(ms) == 1 and _strip(ms[0]) is bool:
        if at.kind != "yes_no":
            raise FactTypeError(f"rule {_fname(rule)} returns bool, but question {q.name!r} is {at.kind} {at.options}")
        return
    cv = closed_values(s)
    if cv is None:
        return
    vals, multi = cv
    extra = [v for v in vals if v not in at.options]
    if extra:
        raise FactTypeError(f"rule {_fname(rule)} returns {type_name(t)}, but question {q.name!r} has options "
                            f"{at.options}: {extra} not among them")
    if multi and at.kind != "multi":
        raise FactTypeError(f"rule {_fname(rule)} returns a collection ({type_name(t)}), but question {q.name!r} is "
                            f"{at.kind}")


# --- typed input state
def is_model(obj):
    """A pydantic BaseModel instance? (pydantic is never imported for this: if it is not loaded, obj cannot be one)."""
    m = sys.modules.get("pydantic")
    return m is not None and isinstance(obj, m.BaseModel)


def model_facts(m):
    """A BaseModel instance → its fields as given facts (field values as they are, nested models included)."""
    d = {k: getattr(m, k) for k in type(m).model_fields}
    d.update(m.model_extra or {})
    return d


def state_of(obj, model=None):
    """init_state as a dict of given facts → (state, [(fact, rejection reason)]). A BaseModel instance gives its fields; with
    `model` (System(inputs=...)) a dict is validated against it: the model's fields (with defaults) become given facts, a field
    that fails validation is left out (the fact is missing) and reported; other keys pass through — unless the model
    forbids them (extra="forbid"): then each is left out and reported as rejected, like a field that failed.
    With `model` the facts come in the model's field order, then the other keys as they were given: the order is declared
    once, in the type, and does not depend on who built the dict (a decider reads a state's keys in their order)."""
    if is_model(obj):
        return model_facts(obj), []
    state = obj if isinstance(obj, dict) else dict(obj)
    if model is None:
        return state, []
    try:
        m = model.model_validate(state)
    except ValueError as e:
        return _partial(model, state, e)
    return _in_field_order(model, {**state, **model_facts(m)}), []


def _in_field_order(model, state):
    """The state with the model's fields first, in their declared order, then the remaining keys as given."""
    first = [k for k in model.model_fields if k in state]
    return {**{k: state[k] for k in first}, **{k: v for k, v in state.items() if k not in model.model_fields}}


def field_types(model):
    """{field: annotation} of a pydantic model class (cached on the class): the types its validated fields already have."""
    ft = model.__dict__.get("__solvi_types__")
    if ft is None:
        ft = {k: fi.annotation for k, fi in model.model_fields.items()}
        try:
            model.__solvi_types__ = ft
        except (AttributeError, TypeError):
            pass
    return ft


def _partial(model, state, e):
    bad = {}
    for err in e.errors():
        loc = err.get("loc") or ()
        k = str(loc[0]) if loc else None
        if k in state and k not in bad:
            bad[k] = (".".join(str(x) for x in loc[1:]) + ": " if len(loc) > 1 else "") + err["msg"]
    out = {k: v for k, v in state.items() if k not in bad}
    rejected = []
    for k, fi in model.model_fields.items():
        if k in bad:
            rejected.append((k, f"{TYPE_REJECTED}: given {k} = {_short(state[k])} is not {type_name(fi.annotation)} "
                                f"({bad[k]})"))
        elif k in state:
            s = spec(fi.annotation, f"inputs.{k}")
            if s is not None:
                nv, err = check(s, state[k])
                if err is None:
                    out[k] = nv
                else:
                    del out[k]
                    rejected.append((k, f"{TYPE_REJECTED}: given {k} = {_short(state[k])} is not {s.name} ({err})"))
        elif not fi.is_required():
            out[k] = fi.get_default(call_default_factory=True)
    for k, why in bad.items():                        # a key the model does not allow (extra="forbid"): left out like a
        if k not in model.model_fields:               # field that failed, and said so — never dropped without a word
            rejected.append((k, f"{TYPE_REJECTED}: given {k} = {_short(state[k])} is not a field of {model.__name__}, "
                                f"which forbids extra keys ({why})"))
    return _in_field_order(model, out), rejected
