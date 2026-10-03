"""Adapting one question: label-bias correction, few-shot shift and scale, temperature; the question spec. (Part of
solvi.core.deciders, which re-exports every name.)"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from ..catalog import Unknown
from .kinds import OTHER_NAMES, _kind
from .wire import Item


def _usable(Z):
    """Logits a learning call may take. A scorer that could not answer (a remote model that is down, a reply that breaks
    the contract) returns placeholder zeros marked `escalate`: a decision escalates on the mark, and adapt / fit / teach
    and the memory must not learn from the zeros."""
    Z = list(Z)
    bad = [z.escalate for z in Z if getattr(z, "escalate", None)]
    if bad:
        raise ValueError(f"the model gave no usable output for {len(bad)} of {len(Z)} input(s) — {bad[0]} — so nothing was "
                         "learned; repeat the call when the model answers")
    return Z


def _given(logits, n):
    """Precomputed logits for adapt / fit / teach → a list of n arrays."""
    Z = [np.asarray(z, float) for z in _usable(logits)]
    if len(Z) != n:
        raise ValueError(f"{len(Z)} logits for {n} input(s)")
    return Z


@dataclass
class Adaptation:
    """What solvi learned for one question (task, options, kind): the label-bias correction, the few-shot shift / scale, the
    temperature and the "other" threshold. Part of the fingerprint of every decision that uses it."""
    bias: list | None = None             # centered mean logit per option over unlabelled texts (subtracted)
    n_unlabelled: int = 0
    scale: float | None = None           # a in (a·z + b) / temperature (None: 1 / the model's temperature)
    shift: list | None = None            # b, per option (score / noul: from a tilt / a single bias, see _basis)
    temperature: float = 1.0             # fitted on out-of-fold predictions
    other_threshold: float | None = None  # fitted on labelled examples that include "other"
    n_labelled: int = 0
    examples: list = field(default_factory=list)   # [(raw logits, label)] kept for teach and refits

    def params(self):
        """Everything that changes the output (the examples only through the fitted parameters)."""
        d = asdict(self)
        d.pop("examples")
        return d


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _sig(z):
    return 1 / (1 + np.exp(-z))


def _basis(sp):
    """How the few-shot shift may move the logits: None — a free shift per option (choice, multi); score — a tilt towards
    higher / lower levels and a spread towards the middle / the ends (ordinal-aware: it cannot reorder the levels at random);
    noul — one yes−no bias."""
    K = len(sp.real)
    if sp.kind == "noul":
        return np.array([[0.5], [-0.5]])
    if sp.kind in ("score", "number"):
        r = (np.arange(K) - (K - 1) / 2) / max((K - 1) / 2, 1)
        cols = [r] + ([r ** 2 - np.mean(r ** 2)] if K >= 3 else [])
        return np.stack(cols, 1)
    return None


def _fit_shift(Z, Y, multi, a0, lam=1.0, iters=300, init=None, basis=None):
    """S: minimize the mean loss of a·z + b + λ/n·(‖c‖² + (a − a0)²) with L-BFGS, b = basis·c (basis None: b = c). Z [n, K];
    Y label indices (single choice) or a 0/1 matrix [n, K] (multi-label). → (a, b)."""
    from scipy.optimize import minimize
    n, K = Z.shape
    oh = Y if multi else np.eye(K)[Y]
    B = np.eye(K) if basis is None else np.asarray(basis, float)
    m = B.shape[1]

    def f(w):
        a, c = w[0], w[1:]
        b = B @ c
        s = a * Z + b
        if multi:
            p = _sig(s)
            loss = -np.mean(np.sum(oh * np.log(p + 1e-12) + (1 - oh) * np.log(1 - p + 1e-12), 1))
            g = (p - oh) / n
        else:
            mx = s.max(1, keepdims=True)
            ls = s - mx - np.log(np.exp(s - mx).sum(1, keepdims=True))
            loss = -np.mean((oh * ls).sum(1))
            g = (np.exp(ls) - oh) / n
        loss += lam / n * (np.sum(c ** 2) + (a - a0) ** 2)
        return loss, np.concatenate([[float((g * Z).sum()) + 2 * lam / n * (a - a0)], B.T @ g.sum(0) + 2 * lam / n * c])
    if init is None:
        w0 = np.concatenate([[a0], np.zeros(m)])
    else:
        init = np.asarray(init, float)
        c0 = init[1:] if basis is None else np.linalg.lstsq(B, init[1:], rcond=None)[0]
        w0 = np.concatenate([[init[0]], c0])
    w = minimize(f, w0, jac=True, method="L-BFGS-B", options={"maxiter": iters}).x
    return float(w[0]), B @ w[1:]


def _fit_temperature(oof, multi, prior=1.0):
    """Temperature on out-of-fold scores: minimum mean NLL + prior·(log t)²/n (a pull towards 1, so a few separable
    examples do not sharpen the model without limit). A single-choice row labelled None ("other": no option fits) has a
    uniform target — sharpening on texts that fit no option is penalized, which keeps the "other" threshold meaningful."""
    n = max(1, len(oof))
    best = None
    for t in np.exp(np.linspace(np.log(0.25), np.log(8), 60)):
        if multi:
            nll = -np.mean([np.mean(y * np.log(_sig(s / t) + 1e-12) + (1 - y) * np.log(1 - _sig(s / t) + 1e-12)) for s, y in oof])
        else:
            nll = -np.mean([np.log(_softmax(s / t)[y] + 1e-12) if y is not None else np.mean(np.log(_softmax(s / t) + 1e-12))
                            for s, y in oof])
        nll += prior * np.log(t) ** 2 / n
        if best is None or nll < best[0]:
            best = (nll, float(t))
    return best[1]


class _Spec:
    """A question: its kind, all its options, the ones the model scores (without "other"), their descriptions; for a bool
    question the labels are "yes" / "no" and the values True / False."""

    def __init__(self, task, options, descriptions=None, multi=False, other=None, kind=None, as_bool=False,
                 score_value="median", *, unknown=False, k=None, bins=None, integer=None, unit=None, coverage=0.8,
                 evidence=0, vtype=None):
        kind = _kind(kind, multi)
        if isinstance(options, dict):
            descriptions = {**options, **(descriptions or {})}
            options = list(options)
        options = list(options or ())
        d = dict(descriptions or {})
        self.unknown, self.k, self.edges, self.unit, self.coverage = bool(unknown), k, None, unit, float(coverage)
        self.evidence, self.vtype, self.integer = int(evidence or 0), vtype, integer
        if self.unknown or kind in ("rank", "number", "span"):
            if other not in (None, False):
                raise ValueError("'other' is not available with 'not stated' or with rank / number / span questions")
            other = False
        if kind == "number":
            if not bins:
                raise ValueError("a number decision needs bins (the bin edges)")
            from ..catalog import bin_labels
            self.edges = list(bins)
            if self.integer is None:
                self.integer = all(float(b).is_integer() for b in self.edges)
            options = options or bin_labels(self.edges, self.integer, unit)
            if len(options) != len(self.edges) + 1:
                raise ValueError(f"{len(self.edges)} bin edges give {len(self.edges) + 1} bins, not {len(options)} labels")
        if kind == "span":
            if options:
                raise ValueError("a span question has no options: the answer is a piece of the text")
            d = {}
        if kind == "noul":
            if options and options not in (["yes", "no"], [True, False]) and sorted(map(str, options)) != ["no", "yes"]:
                raise ValueError(f"a yes/no question has the options yes, no (or True, False), not {options}")
            as_bool = as_bool or options == [True, False]
            d = {"yes": d.get("yes", d.get(True)), "no": d.get("no", d.get(False))}
            options, other = ["yes", "no"], False
        if kind == "score":
            if not 2 <= len(options) <= 10:
                raise ValueError(f"a score has 2 to 10 levels, not {len(options)}")
            other = False
        if not options and kind != "span":
            raise ValueError("a decision needs options")
        if kind == "rank" and k is not None and not 1 <= int(k) <= len(options):
            raise ValueError(f"k must be between 1 and {len(options)}")
        if len(set(options)) != len(options):
            raise ValueError(f"duplicate options: {options}")
        if score_value not in ("median", "mode", "expected"):
            raise ValueError('score_value must be "median", "mode" or "expected"')
        if other is None:
            found = [o for o in options if isinstance(o, str) and o.strip().lower() in OTHER_NAMES]
            other = found[-1] if found else None
        elif other is False:
            other = None
        elif other not in options:
            raise ValueError(f"other={other!r} is not one of the options")
        self.task, self.options, self.kind, self.multi, self.other = task, options, kind, kind == "multi", other
        self.as_bool, self.score_value = bool(as_bool), score_value
        self.real = [o for o in options if o != other]
        if not self.real and kind != "span":
            raise ValueError("a decision needs at least one option besides 'other'")
        self.descriptions = {o: d[o] for o in options if d.get(o)}
        self.values = [True, False] if self.as_bool else list(options)
        self.pointer = kind == "span" or self.evidence > 0          # needs the pointer (full layout)
        if kind in ("number", "span"):
            self.values = None                     # no closed set: a number, a quote
        elif self.unknown:
            self.values = self.values + [Unknown]
        self.key = (task, tuple(self.real), tuple(d.get(o) or "" for o in self.real),
                    self.multi if kind in ("choice", "multi") else kind)
        if self.pointer:                           # the pointer's output is cached apart (it needs the full layout)
            self.key = self.key + ("pointer",)

    def item(self, text, wire="single", noul_labels=None):
        """The question as the scorer sees it; a native yes/no question uses the checkpoint's option labels ('l14f typed v1': "true",
        "false") with the yes / no descriptions."""
        desc = tuple(self.descriptions.get(o, "") for o in self.real)
        opts = tuple(str(o) for o in self.real)
        if wire == "noul" and noul_labels:
            opts = tuple(noul_labels)
        if self.pointer:
            return Item(self.task, opts, desc if any(desc) else None, text, wire == "multi",
                        "" if wire in ("single", "multi") else wire, True, self.unknown)
        return Item(self.task, opts, desc if any(desc) else None, text, wire == "multi",
                    "" if wire in ("single", "multi") else wire, unknown=self.unknown)

    def out(self, label):
        """A label → the decision's value (a bool question: True / False)."""
        return (label == "yes") if self.as_bool else label

    def label(self, y):
        """A value or an answer (True / "yes", an Enum member, a list for multi) → this question's label; ValueError if it
        is not one of the options."""
        if isinstance(y, Enum):
            y = y.value
        if y is Unknown or self.kind in ("rank", "span"):
            raise ValueError(f"{'not stated' if y is Unknown else self.kind} is not adapted from labels")
        if self.kind == "number":
            if isinstance(y, (int, float)) and not isinstance(y, bool):
                i = 0
                while i < len(self.edges) and y >= self.edges[i]:
                    i += 1
                return self.options[i]
            if y not in self.options:
                raise ValueError(f"{y!r} is not a number or one of the bins {self.options}")
            return y
        if self.kind == "noul":
            if y is True or (isinstance(y, (bool, np.bool_)) and y) or y == "yes":
                return "yes"
            if y is False or (isinstance(y, (bool, np.bool_)) and not y) or y == "no":
                return "no"
            raise ValueError(f"{y!r} is not yes / no")
        if self.multi:
            vals = [y] if isinstance(y, str) else [v.value if isinstance(v, Enum) else v for v in (y or [])]
            bad = [v for v in vals if v not in self.options]
            if bad:
                raise ValueError(f"{bad} not among {self.options}")
            return tuple(v for v in self.real if v in vals)
        if y not in self.options:
            raise ValueError(f"{y!r} is not one of {self.options}")
        return y

    def describe(self):
        d = {"task": self.task, "options": self.options, "descriptions": self.descriptions, "multi": self.multi,
             "other": self.other}
        if self.kind in ("score", "noul"):            # choice / multi questions describe (and hash) as before
            d["kind"] = self.kind
        if self.as_bool:
            d["as_bool"] = True
        if self.score_value != "median":
            d["score_value"] = self.score_value
        if self.kind in ("rank", "number", "span"):
            d["kind"] = self.kind
        for key, v in (("unknown", self.unknown), ("k", self.k), ("bins", self.edges),
                       ("integer", self.integer if self.kind == "number" else None), ("unit", self.unit),
                       ("coverage", self.coverage if self.kind == "number" else None), ("evidence", self.evidence)):
            if v not in (None, False, 0):          # only what is set: parts of 0.5 and earlier describe (and hash) as before
                d[key] = v
        return d


def _is_type(x):
    from typing import get_origin
    return isinstance(x, type) or get_origin(x) is not None


def _from_type(t, kind=None, options=None, descriptions=None):
    """A Python type → (kind, options, descriptions, as_bool, primitive settings) of a decision (solvi.core.types.question_kind;
    Maybe[T] — "not stated" allowed; Span[T], Rank[...], Estimate[...] — span, rank, number)."""
    from ..types import primitive_answer, question_kind, split_unknown
    t, unknown = split_unknown(t)
    extra = {"unknown": True} if unknown else {}
    pa = primitive_answer(t)
    if isinstance(options, dict):
        descriptions = {**options, **(descriptions or {})}
    elif options:
        raise ValueError("the options come from the type; pass descriptions as a dict {option: description}")
    if pa is not None:
        k = {"span": "span", "rank": "rank", "estimate": "number"}[pa.kind]
        if kind is not None and _kind(kind) != k:
            raise ValueError(f"type {t} is a {k} question, not {kind}")
        if k == "rank":
            extra["k"] = pa.k
        elif k == "number":
            extra.update(bins=pa.bins, unit=pa.unit, coverage=pa.coverage, labels=list(pa.options))
        else:
            extra["vtype"] = pa.type
        return k, (list(pa.options) if k == "rank" else []), descriptions, False, extra
    k, vals, as_bool = question_kind(t)
    if kind is not None and _kind(kind) != k and not (_kind(kind) == "score" and k == "choice"):
        raise ValueError(f"type {t} is a {k} question, not {kind}")
    return _kind(kind) if kind is not None else k, vals, descriptions, as_bool, extra


def lora_key(sp_or_key):
    """The question a LoRA adapter (solvi.experimental.lora) belongs to: the task, the scored options with their descriptions (in any
    order — option_order="average" asks rotations of the same question) and the kind. A _Spec or a _Spec.key."""
    k = sp_or_key.key if isinstance(sp_or_key, _Spec) else sp_or_key
    return (k[0], tuple(sorted(zip(map(str, k[1]), k[2]))), k[3])


# The adapter kinds a decision part's adapter slot can load from a file (part.load_lora, a calibration file that names
# its adapter): kind → the module that reads it (its load(part, path, strict, expect=)). Looked up by name when a file
# of that kind is loaded, so solvi.core.deciders imports no adapter module (LoRA is experimental).
ADAPTERS = {"lora": "solvi.experimental.lora"}


__all__ = ["ADAPTERS", "Adaptation", "lora_key"]
