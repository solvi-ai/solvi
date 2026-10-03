"""Provenance of facts and answers, model identity, and grounding.

Principle: fuzzy proposes, deterministic decides, everything is in the trace. Every fact and answer carries a provenance kind:

  given     — a key of init_state (the input)
  computed  — a plain function (fn, check, hand-written rule)
  quoted    — an extract part returning a Quote: the value is grounded at offsets in the source text
  decided   — a model's choice among declared options, with probabilities (a Decision)
  learned   — a fit answer head, a learn_rule list, or another trained function
  proposed  — a model that writes: a strategist's plan, a generator's text or JSON (solvi.generate); the deterministic
              layer verifies what it proposes

A part backed by a model (an extractor, a head, a rule list, any object) records the model's identity in the trace:
{"type", "id", "fp"} — the fingerprint is a stable hash of the model's configuration and weights, so a replay can tell
whether the model that made the decision is still the one in the catalog."""
from __future__ import annotations

import hashlib
import re
import sys

from ._deprecate import MOVED as _FP_MODULE   # new module (or "module:Name") → its 0.9 module; filled as code moves

GIVEN, COMPUTED, QUOTED, DECIDED, LEARNED, PROPOSED = "given", "computed", "quoted", "decided", "learned", "proposed"
KINDS = (GIVEN, COMPUTED, QUOTED, DECIDED, LEARNED, PROPOSED)
FUZZY = {DECIDED, LEARNED, PROPOSED}          # a model's output (a quote by a model is fuzzy too, but grounded)

# reasons an output is rejected (the executor, accepts() and replay share them; the audit classifies by them)
QUOTE_OUTSIDE = "quote outside the text"
NOT_GROUNDED = "not grounded"
OUTSIDE_OPTIONS = "outside the options"
VALIDATE = "rejected by validate"
TYPE_REJECTED = "type rejected"               # a typed fact failed its type (solvi.typed); the guard kind of the same
#                                               event is "type_rejected" (solvi.primitives.TYPE_REJECTED)
ESCALATED = "model escalated"                 # a decider's own act / escalate signal said "hand it to a person" (solvi.decide)
TIMED_OUT = "timed out"                       # a part did not finish within its timeout (System.aask)
INSTRUCTION = "answer depends on an instruction-like sentence"   # perturb=k: the answer changed without such a sentence
MEMORY = "memory of corrections disagrees"    # solvi.memory: similar corrected cases say another answer


def classify(reason):
    """A rejection reason → the safeguard that fired (grounding, type_rejected, outside_options, low_confidence, escalated,
    validator, timeout, instruction, memory) or None."""
    if not reason:
        return None
    if reason.startswith(INSTRUCTION):
        return "instruction"
    if reason.startswith(MEMORY):
        return "memory"
    if reason.startswith(ESCALATED):
        return "escalated"
    if reason.startswith((QUOTE_OUTSIDE, NOT_GROUNDED)):
        return "grounding"
    if reason.startswith(TYPE_REJECTED):
        return "type_rejected"
    if OUTSIDE_OPTIONS in reason:
        return "outside_options"
    if reason.startswith(("confidence ", "margin ")) and "<" in reason:     # min_margin: a near tie is unsure too
        return "low_confidence"
    if reason.startswith((VALIDATE, "validate raised")):
        return "validator"
    if reason.startswith(TIMED_OUT):
        return "timeout"
    return None


# --- fingerprints
def _feed(h, obj, depth=0):
    """Feed a value into a hash: arrays by dtype, shape and bytes; containers recursively; anything else by repr."""
    np = sys.modules.get("numpy")             # not imported yet: no value can be an array (and hashing stays light)
    if np is not None:
        if isinstance(obj, np.ndarray):
            a = np.ascontiguousarray(obj)
            h.update(f"nd{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
            return
        if isinstance(obj, np.generic):
            h.update(repr(obj.item()).encode())
            return
    if depth > 8:
        h.update(repr(obj).encode())
    elif isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=repr):
            _feed(h, k, depth + 1)
            _feed(h, obj[k], depth + 1)
        h.update(b"}")
    elif isinstance(obj, (list, tuple, set)):
        h.update(b"[")
        for x in (sorted(obj, key=repr) if isinstance(obj, set) else obj):
            _feed(h, x, depth + 1)
        h.update(b"]")
    else:
        h.update(repr(obj).encode())


def digest(*parts):
    h = hashlib.sha256()
    for p in parts:
        _feed(h, p)
    return h.hexdigest()[:16]


def fingerprint(model):
    """A stable hash of a model: its own `fingerprint()` if it has one; parameters for solvi heads and rule lists;
    otherwise a `version` attribute, or "unversioned:<type>" (then a changed model cannot be detected)."""
    fp = getattr(model, "fingerprint", None)
    if callable(fp):
        return str(fp())
    name = type(model).__name__
    if name == "MultiHead":
        return digest(name, model.options, {o: fingerprint(h) for o, h in model.heads.items()})
    if name == "RuleList":
        return digest(name, model.facts, model.rules, model.default)
    v = getattr(model, "version", None)
    return digest(name, v) if v is not None else f"unversioned:{name}"


def model_id(model):
    for a in ("model_id", "name", "model_name"):
        v = getattr(model, a, None)
        if isinstance(v, str) and v:
            return v
    return type(model).__name__


def model_info(model):
    """→ {"type", "id", "fp"} recorded in the trace for a model-backed step (None without a model)."""
    if model is None:
        return None
    return {"type": type(model).__name__, "id": model_id(model), "fp": fingerprint(model)}


def torch_fingerprint(modules, files_dir=None, samples=256):
    """Weights of torch modules, cheaply: every parameter's shape plus `samples` evenly spaced values (and the full tensor when
    it is small), plus the names and sizes of the weight files when the model was loaded from a directory."""
    import os
    h = hashlib.sha256()
    for m in modules:
        for name, p in m.state_dict().items():
            t = p.detach().flatten()
            h.update(f"{name}{tuple(p.shape)}".encode())
            if t.numel() > samples * 4:
                t = t[:: max(1, t.numel() // samples)][:samples]
            h.update(t.float().cpu().numpy().tobytes())
    if files_dir and os.path.isdir(files_dir):
        for f in sorted(os.listdir(files_dir)):
            p = os.path.join(files_dir, f)
            if os.path.isfile(p):
                h.update(f"{f}:{os.path.getsize(p)}".encode())
    return h.hexdigest()[:16]


def fp_module(module, qualname=None):
    """The module name a fingerprint records for an object defined in `module` (named `qualname` there): its 0.9 module
    when the object moved since (the table solvi._deprecate.MOVED), so that a move changes no fingerprint; otherwise
    `module` itself."""
    if not _FP_MODULE or not module:
        return module
    if qualname:
        old = _FP_MODULE.get(f"{module}:{qualname.split('.', 1)[0]}")
        if old is not None:
            return old
    return _FP_MODULE.get(module, module)


# --- catalog fingerprints: what the parts are (code and declarations), so a stored decision knows which catalog made it
_CODE = None                                 # function → (its fingerprint): weak, so a dropped catalog is not kept alive
_SIMPLE = (str, int, float, bool, type(None))
_ADDR = re.compile(r" at 0x[0-9a-fA-F]+")


def _r(v):
    """repr without memory addresses (they differ between processes)."""
    return _ADDR.sub("", repr(v))


def _simple(v, depth=0):
    if isinstance(v, _SIMPLE):
        return True
    if isinstance(v, (tuple, frozenset, set, list)) and depth < 3 and len(v) <= 256:
        return all(_simple(x, depth + 1) for x in v)
    if isinstance(v, dict) and depth < 3 and len(v) <= 256:
        return all(_simple(k, depth + 1) and _simple(x, depth + 1) for k, x in v.items())
    return False


def _code_names(co):
    """Global names a code object (and the functions and comprehensions nested in it) reads."""
    out = set(co.co_names)
    for c in co.co_consts:
        if hasattr(c, "co_names"):
            out |= _code_names(c)
    return out


def _bytecode(co):
    consts = [_bytecode(c) if hasattr(c, "co_code") else repr(c) for c in co.co_consts]
    return [co.co_code.hex(), consts, list(co.co_names), list(co.co_varnames)]


def _source_ast(f):
    """The function's definition as an AST dump — without decorators (they are declarations, fingerprinted as such), its
    docstring, comments and formatting — or None when the source is not available."""
    import ast
    import inspect
    import textwrap
    try:
        src = textwrap.dedent(inspect.getsource(f))
        tree = ast.parse(src)
    except (OSError, TypeError, SyntaxError, IndentationError):
        return None
    node = tree.body[0] if len(tree.body) == 1 else tree
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        node.decorator_list = []
        b = node.body
        if b and isinstance(b[0], ast.Expr) and isinstance(getattr(b[0], "value", None), ast.Constant) \
                and isinstance(b[0].value.value, str):
            node.body = b[1:] or [ast.Pass()]
    return _dump(node)


def _dump(node):
    """An AST as text, the same on every Python version: fields that are None or empty are left out (Python 3.12 adds
    `type_params=[]` to every function and 3.13 drops empty fields from ast.dump, so its text — and every code
    fingerprint built on it — changed with the interpreter, and a decision recorded on one Python did not replay its
    configuration on another)."""
    import ast
    if isinstance(node, ast.AST):
        fields = []
        for k in node._fields:
            v = getattr(node, k, None)
            if v is None or (isinstance(v, list) and not v):
                continue
            fields.append(f"{k}={_dump(v)}")
        return f"{type(node).__name__}({', '.join(fields)})"
    if isinstance(node, list):
        return "[" + ", ".join(_dump(x) for x in node) + "]"
    return repr(node)


def code_fingerprint(f, _seen=None):
    """A stable hash of a function's code: its definition's AST (without decorators, docstring, comments, formatting; the
    bytecode when there is no source), the simple values it closes over or reads as module globals (THRESHOLD = 30), and
    the code of the module's own functions it calls. A callable object (a model-backed part such as a decision) is
    identified by its type: its weights are the model's fingerprint. Cached per function: a module constant rebound after
    the first fingerprint is not noticed within the process."""
    import inspect
    import weakref
    global _CODE
    if _CODE is None:
        _CODE = weakref.WeakKeyDictionary()
    if f is None:
        return None
    f = inspect.unwrap(f)
    if not inspect.isfunction(f):
        if inspect.ismethod(f):
            return digest("method", type(f.__self__).__qualname__, code_fingerprint(f.__func__, _seen))
        return digest("object", fp_module(type(f).__module__, type(f).__qualname__), type(f).__qualname__)
    top = _seen is None
    try:
        if top and f in _CODE:
            return _CODE[f]
    except TypeError:
        top = False
    seen = _seen if _seen is not None else set()
    if id(f) in seen:
        return "recursive"
    seen.add(id(f))
    co = f.__code__
    body = _source_ast(f)
    parts = [f.__qualname__, body if body is not None else _bytecode(co)]
    cells = []
    for name, cell in zip(co.co_freevars, f.__closure__ or ()):
        try:
            v = cell.cell_contents
        except ValueError:
            cells.append([name, "empty"])
            continue
        cells.append([name, _value_fp(v, f, seen)])
    parts.append(cells)
    glob = []
    g = getattr(f, "__globals__", {}) or {}
    for name in sorted(_code_names(co)):
        if name in g:
            # co_names holds attribute names too: `type(x).__name__` reads the module's own __name__ here, recorded
            # as its 0.9 name
            v = _value_fp(fp_module(g[name]) if name == "__name__" and isinstance(g[name], str) else g[name], f, seen,
                          module_only=True)
            if v is not None:
                glob.append([name, v])
    parts.append(glob)
    if f.__defaults__:
        parts.append([_value_fp(v, f, seen) for v in f.__defaults__])
    fp = digest(*parts)
    if top:
        try:
            _CODE[f] = fp
        except TypeError:
            pass
    return fp


def _value_fp(v, owner, seen, module_only=False):
    """A value a function reads: simple data by value, a function of the same module by its code, a model by its
    fingerprint; anything else by its type (or, for a module global, not at all: imported modules and classes)."""
    import inspect
    if _simple(v):
        from .runtime import srepr
        return srepr(v)
    if inspect.isfunction(v):
        if module_only and fp_module(v.__module__, v.__qualname__) != fp_module(owner.__module__, owner.__qualname__):
            return None
        return code_fingerprint(v, seen)
    if module_only:
        return None
    if getattr(v, "fingerprint", None) is not None or type(v).__name__ in ("RuleList", "MultiHead"):
        return fingerprint(v)
    return "type:" + type(v).__qualname__


def type_fingerprint(t, depth=0):
    """A declared type as data: generics by their arguments, a pydantic model by its fields, an Enum by its members,
    Annotated by its constraints, any other class by module and name."""
    import enum
    import types
    import typing
    if depth > 6:
        return _r(t)
    o, a = typing.get_origin(t), typing.get_args(t)
    if o is typing.Annotated:
        return ["Annotated", type_fingerprint(a[0], depth + 1), [_r(m) for m in a[1:]]]
    if o is not None:
        # every union as "typing.Union": Python 3.14 made Optional[X], Union[X, Y] and X | Y one class, so the spelling
        # (typing.Union or types.UnionType on 3.11-3.13) can no longer be told apart
        head = "typing.Union" if o is typing.Union or o is types.UnionType else _r(o)
        return [head, [type_fingerprint(x, depth + 1) if not isinstance(x, _SIMPLE) else _r(x) for x in a]]
    if isinstance(t, type):
        fields = getattr(t, "model_fields", None)
        if isinstance(fields, dict):
            return ["model", fp_module(t.__module__, t.__qualname__), t.__qualname__,
                    {k: [type_fingerprint(fi.annotation, depth + 1), _r(fi.metadata), _r(fi.default)]
                     for k, fi in fields.items()}]
        if issubclass(t, enum.Enum):
            return ["enum", fp_module(t.__module__, t.__qualname__), t.__qualname__, [[m.name, _r(m.value)] for m in t]]
        return [fp_module(t.__module__, t.__qualname__), t.__qualname__]
    return _r(t)


def part_fingerprint(part):
    """The fingerprint of one catalog part: its declarations (kind, inputs, hard / then, options, validate, types, the
    model's type) and its code; a fact with alternative producers — each producer's, in order. The model's weights are not
    here: they are the model's own fingerprint, recorded with each step it produced."""
    d = {"kind": part.kind, "name": part.name, "inputs": list(part.inputs), "hard": part.hard,
         "then": sorted([str(k), code_fingerprint(v) if callable(v) else _r(v)] for k, v in (part.then or {}).items()),
         "question": part.question,
         "provides": part.provides, "options": None if part.options is None else [_r(o) for o in part.options],
         "exact": part.exact, "source": part.source, "min_confidence": part.min_confidence, "provenance": part.provenance,
         "validate": code_fingerprint(part.validate), "model": None if part.model is None else type(part.model).__qualname__,
         "types": None if part.types is None else {k: type_fingerprint(t) for k, t in part.types.items()},
         "returns": None if part.returns is None else type_fingerprint(part.returns)}
    if getattr(part, "quotes", "normalized") != "normalized":   # only when set otherwise: 0.9's fingerprints stay as
        d["quotes"] = part.quotes                                # they were
    if part.alternatives is not None:
        d["alternatives"] = [part_fingerprint(a) for a in part.alternatives]
        d["features"] = code_fingerprint(part.features)
    else:
        d["code"] = code_fingerprint(part.func)
    return digest(d)


def catalog_fingerprint(catalog):
    """→ {"fp": the catalog's fingerprint, "parts": {name: fingerprint}} over every part, answer rule ("answer:<question>")
    and constraint ("constraint:<name>"). Cached on the catalog while its parts are the same objects (a part edited in place
    after that is not noticed: register a new one)."""
    items = [(n, p) for n, p in catalog.parts.items()] + [(p.name, p) for p in catalog.rules.values()] + \
            [("constraint:" + n, p) for n, p in catalog.constraints.items()]
    key = tuple((n, id(p), id(p.func), id(p.model), len(p.alternatives or ()), len(p.inputs)) for n, p in items)
    cached = getattr(catalog, "_fp_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    parts = {n: part_fingerprint(p) for n, p in items}
    out = {"fp": digest(sorted(parts.items())), "parts": parts}
    catalog._fp_cache = (key, out)
    return out


# --- grounding
_WS = re.compile(r"\s+")
_NUM = re.compile(r"-?\d[\d,  ']*(?:\.\d+)?|-?\.\d+")
_DATEISH = re.compile(r"\d|today|tomorrow|yesterday|сегодня|завтра|вчера", re.I)


def _norm(s):
    return _WS.sub(" ", s).strip()


def _number(value):
    """A value that is a number (int, float, Decimal, Fraction, a numpy scalar; not a bool) → it as a Fraction or float,
    else None."""
    import decimal
    import fractions
    if isinstance(value, bool):
        return None
    if hasattr(value, "item") and getattr(value, "ndim", None) == 0 and not isinstance(value, (str, bytes)):
        value = value.item()                              # a numpy scalar
        if isinstance(value, bool):
            return None
    if isinstance(value, (int, decimal.Decimal, fractions.Fraction)):
        try:
            return fractions.Fraction(value)
        except (ValueError, OverflowError):               # Decimal("NaN"), an infinity: nothing in a text equals it
            return float("nan")
    return value if isinstance(value, float) else None


def matches(value, snippet):
    """Is `value` literally the quoted text? Strings: equal up to whitespace. Numbers (int, float, Decimal, Fraction, numpy
    scalars): a number in the snippet equals it (thousands separators allowed; a Decimal or a Fraction exactly, a float
    up to rounding). A date: the snippet reads as that date (solvi.textin.parse_date) or holds it in ISO form; a snippet
    with no date in it is not the date, one whose date needs a year or "today" to be read is not compared (None). Other
    types (datetimes, lists, booleans) cannot be compared → None (not checked)."""
    import datetime
    if isinstance(value, str):
        return _norm(value) == _norm(snippet)
    n = _number(value)
    if n is not None:
        for m in _NUM.finditer(snippet):
            t = re.sub(r"[,   ']", "", m.group())
            try:
                if isinstance(n, float):
                    if abs(float(t) - n) <= 1e-9 * max(1.0, abs(n)):
                        return True
                else:
                    import fractions
                    if fractions.Fraction(t) == n:
                        return True
            except (ValueError, ZeroDivisionError):
                continue
        return False
    if type(value) is datetime.date:
        if value.isoformat() in snippet:
            return True
        from .textin import ParseError, parse_date
        try:
            got = parse_date(snippet)
        except ParseError:                                # no date there at all → not the text; a date that needs a
            return None if _DATEISH.search(snippet) else False   # year or "today" to be read → cannot be compared
        return (got if isinstance(got, datetime.date) else datetime.date.fromisoformat(str(got))) == value
    return None


# --- quote matching on a normalized view (solvi 1.0). A quote a model writes may differ from its source in how characters
# are typed, not in what they say: NFKC forms (ligatures, full-width letters), no-break spaces, no-break and other
# hyphens / dashes, "…" for "...", runs of whitespace, curly for straight quotes, zero-width characters. QUOTES_NORMALIZED
# compares both texts in one normalized view; the quote that is kept is always the source's own substring at offsets
# into the ORIGINAL text, and a record whose quote needed the view says so (extra["quote_match"], see quote_match_problem).
QUOTES_LITERAL, QUOTES_NORMALIZED = "literal", "normalized"
QUOTE_MODES = (QUOTES_LITERAL, QUOTES_NORMALIZED)
NORM_FORM = "nfkc-ws-dash-quote/1"            # the normalization a record names: its rules never change under this name
_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―⁃−﹘﹣－"), "-")
_QUOTE_CHARS = {**dict.fromkeys(map(ord, "‘’‚‛‹›"), "'"),
                **dict.fromkeys(map(ord, "“”„‟«»"), '"')}
_CHAR_MAP = {**_DASHES, **_QUOTE_CHARS}
_ZERO_WIDTH = frozenset("­​‌‍⁠﻿")


def quote_mode(mode):
    """A quote-matching mode, checked: "normalized" (default) or "literal" (0.9: the text as written, up to whitespace)."""
    if mode not in QUOTE_MODES:
        raise ValueError(f'quotes must be "normalized" or "literal", not {mode!r}')
    return mode


def norm_view(s):
    """A text in the normalized view → (view, starts, ends): view[i] comes from s[starts[i]:ends[i]]. Each character with
    the combining marks after it is NFKC-normalized as a unit; dashes and hyphens become "-", curly and angle quotes
    straight ones, zero-width characters and soft hyphens go, and every run of whitespace is one space."""
    import unicodedata
    out, starts, ends = [], [], []
    i, n = 0, len(s)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(s[j]):
            j += 1
        unit = s[i:j]
        if not unit.isascii():
            unit = unicodedata.normalize("NFKC", unit).translate(_CHAR_MAP)
        for c in unit:
            if c in _ZERO_WIDTH:
                continue
            if c.isspace():
                if out and out[-1] == " ":
                    ends[-1] = j                  # a run of whitespace: one space for all of it
                    continue
                c = " "
            out.append(c)
            starts.append(i)
            ends.append(j)
        i = j
    return "".join(out), starts, ends


def normalized(s):
    """The normalized view of a text, without the offsets and without leading / trailing space."""
    return norm_view(s)[0].strip(" ")


def find_normalized(text, quote, whole=True):
    """Where `quote` is in `text` in the normalized view → (start, end) in the ORIGINAL text — text[start:end] is the
    source's own substring — or None. whole: as whole words and numbers (solvi.core.find_whole)."""
    q = normalized(quote)
    if not q or not isinstance(text, str):
        return None
    view, starts, ends = norm_view(text)
    if whole:
        from .core import find_whole
        i = find_whole(q, view)
    else:
        i = view.find(q)
    if i < 0:
        return None
    return starts[i], ends[i + len(q) - 1]


def matches_normalized(value, snippet):
    """Is the string `value` the text `snippet` in the normalized view?"""
    return isinstance(value, str) and isinstance(snippet, str) and normalized(value) == normalized(snippet)


def quote_match_problem(extra, value=None, rows=()):
    """A record's quote_match note (what a part wrote, matched in the normalized view) re-checked on replay → None, or why
    it does not hold: the form must be one this version knows, and each written text must be, in that view, the quote
    the record keeps (`value`: the record's quoted value; `rows`: its evidence rows [start, end, source, text])."""
    qm = extra.get("quote_match") if isinstance(extra, dict) else None
    if qm is None:
        return None
    if not isinstance(qm, dict) or qm.get("form") != NORM_FORM:
        return f"{NOT_GROUNDED}: quote matched in an unknown normalization {qm!r} (this version knows {NORM_FORM!r})"
    for where, written in (qm.get("written") or {}).items():
        kept = value if where == "value" else None
        if where.startswith("evidence "):
            k = int(where.split()[1])
            kept = rows[k][3] if 0 <= k < len(rows) else None
        if not matches_normalized(written, kept):
            return f"{NOT_GROUNDED}: {where} written as {written!r} is not {kept!r} in the normalized view"
    return None


def snippet(init, quote, width=60):
    """The quoted text doc[start:end] for a record's (start, end, source), shortened for display."""
    if not quote:
        return None
    s, e, src = quote
    text = init.get(src)
    if not isinstance(text, str):
        return None
    t = text[s:e]
    return t if len(t) <= width else t[: width - 1] + "…"


__all__ = ["catalog_fingerprint", "classify", "code_fingerprint", "digest", "ESCALATED", "find_normalized", "fingerprint",
           "FUZZY", "INSTRUCTION", "KINDS", "matches", "matches_normalized", "MEMORY", "model_id", "model_info",
           "NORM_FORM", "norm_view", "normalized", "NOT_GROUNDED", "OUTSIDE_OPTIONS", "QUOTE_MODES", "QUOTE_OUTSIDE",
           "quote_match_problem", "quote_mode", "QUOTES_LITERAL", "QUOTES_NORMALIZED", "snippet", "TIMED_OUT",
           "torch_fingerprint", "TYPE_REJECTED", "VALIDATE"]
