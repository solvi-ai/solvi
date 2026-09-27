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


def classify(reason):
    """A rejection reason → the safeguard that fired (grounding, type_rejected, outside_options, low_confidence, escalated,
    validator) or None."""
    if not reason:
        return None
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
