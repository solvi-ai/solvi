"""Provenance of facts and answers, model identity, and grounding.

Principle: fuzzy proposes, deterministic decides, everything is in the trace. Every fact and answer carries a provenance kind:

  given     — a key of init_state (the input)
  computed  — a plain function (fn, check, hand-written rule)
  quoted    — an extract part returning a Quote: the value is grounded at offsets in the source text
  decided   — a model's choice among declared options, with probabilities (a Decision)
  learned   — a fit / fit_fast answer head, a learn_rule list, or another trained function
  proposed  — reserved for a strategist model; the deterministic layer verifies what it proposes

A part backed by a model (an extractor, a head, a rule list, any object) records the model's identity in the trace:
{"type", "id", "fp"} — the fingerprint is a stable hash of the model's configuration and weights, so a replay can tell
whether the model that made the decision is still the one in the catalog."""
from __future__ import annotations

import hashlib
import re

GIVEN, COMPUTED, QUOTED, DECIDED, LEARNED, PROPOSED = "given", "computed", "quoted", "decided", "learned", "proposed"
KINDS = (GIVEN, COMPUTED, QUOTED, DECIDED, LEARNED, PROPOSED)
FUZZY = {DECIDED, LEARNED, PROPOSED}          # a model's output (a quote by a model is fuzzy too, but grounded)

# reasons an output is rejected (the executor, accepts() and replay share them; the audit classifies by them)
QUOTE_OUTSIDE = "quote outside the text"
NOT_GROUNDED = "not grounded"
OUTSIDE_OPTIONS = "outside the options"
VALIDATE = "rejected by validate"
TYPE_REJECTED = "type rejected"               # a typed fact failed its type (solvi.typed)
ESCALATED = "model escalated"                 # a decider's own act / escalate signal said "hand it to a person" (solvi.decide)
TIMED_OUT = "timed out"                       # a part did not finish within its timeout (System.aask)
INSTRUCTION = "answer depends on an instruction-like sentence"   # perturb=k: the answer changed without such a sentence


def classify(reason):
    """A rejection reason → the safeguard that fired (grounding, type_rejected, outside_options, low_confidence, escalated,
    validator, timeout, instruction) or None."""
    if not reason:
        return None
    if reason.startswith(INSTRUCTION):
        return "instruction"
    if reason.startswith(ESCALATED):
        return "escalated"
    if reason.startswith((QUOTE_OUTSIDE, NOT_GROUNDED)):
        return "grounding"
    if reason.startswith(TYPE_REJECTED):
        return "type_rejected"
    if OUTSIDE_OPTIONS in reason:
        return "outside_options"
    if reason.startswith("confidence ") and "<" in reason:
        return "low_confidence"
    if reason.startswith((VALIDATE, "validate raised")):
        return "validator"
    if reason.startswith(TIMED_OUT):
        return "timeout"
    return None


# --- fingerprints
def _feed(h, obj, depth=0):
    """Feed a value into a hash: arrays by dtype, shape and bytes; containers recursively; anything else by repr."""
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            a = np.ascontiguousarray(obj)
            h.update(f"nd{a.dtype}{a.shape}".encode())
            h.update(a.tobytes())
            return
        if isinstance(obj, np.generic):
            h.update(repr(obj.item()).encode())
            return
    except ImportError:  # pragma: no cover
        pass
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


# --- catalog fingerprints: what the parts are (code and declarations), so a stored decision knows which catalog made it
_CODE = None                                  # function → (its fingerprint): weak, so a dropped catalog is not kept alive
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
    return ast.dump(node, include_attributes=False)


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
        return digest("object", type(f).__module__, type(f).__qualname__)
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
            v = _value_fp(g[name], f, seen, module_only=True)
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
        if module_only and v.__module__ != owner.__module__:
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
    import typing
    if depth > 6:
        return _r(t)
    o, a = typing.get_origin(t), typing.get_args(t)
    if o is typing.Annotated:
        return ["Annotated", type_fingerprint(a[0], depth + 1), [_r(m) for m in a[1:]]]
    if o is not None:
        return [_r(o), [type_fingerprint(x, depth + 1) if not isinstance(x, _SIMPLE) else _r(x) for x in a]]
    if isinstance(t, type):
        fields = getattr(t, "model_fields", None)
        if isinstance(fields, dict):
            return ["model", t.__module__, t.__qualname__,
                    {k: [type_fingerprint(fi.annotation, depth + 1), _r(fi.metadata), _r(fi.default)]
                     for k, fi in fields.items()}]
        if issubclass(t, enum.Enum):
            return ["enum", t.__module__, t.__qualname__, [[m.name, _r(m.value)] for m in t]]
        return [t.__module__, t.__qualname__]
    return _r(t)


def part_fingerprint(part):
    """The fingerprint of one catalog part: its declarations (kind, inputs, hard / then, options, validate, types, the
    model's type) and its code; a fact with alternative producers — each producer's, in order. The model's weights are not
    here: they are the model's own fingerprint, recorded with each step it produced."""
    d = {"kind": part.kind, "name": part.name, "inputs": list(part.inputs), "hard": part.hard,
         "then": sorted([str(k), _r(v)] for k, v in (part.then or {}).items()), "question": part.question,
         "provides": part.provides, "options": None if part.options is None else [_r(o) for o in part.options],
         "exact": part.exact, "source": part.source, "min_confidence": part.min_confidence, "provenance": part.provenance,
         "validate": code_fingerprint(part.validate), "model": None if part.model is None else type(part.model).__qualname__,
         "types": None if part.types is None else {k: type_fingerprint(t) for k, t in part.types.items()},
         "returns": None if part.returns is None else type_fingerprint(part.returns)}
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


def _norm(s):
    return _WS.sub(" ", s).strip()


def matches(value, snippet):
    """Is `value` literally the quoted text? Strings: equal up to whitespace. Numbers: a number in the snippet equals it
    (thousands separators allowed). Other types (dates, lists, booleans) cannot be compared → None (not checked)."""
    if isinstance(value, str):
        return _norm(value) == _norm(snippet)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for m in _NUM.finditer(snippet):
            t = re.sub(r"[,  ']", "", m.group())
            try:
                if abs(float(t) - float(value)) <= 1e-9 * max(1.0, abs(float(value))):
                    return True
            except ValueError:
                continue
        return False
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
