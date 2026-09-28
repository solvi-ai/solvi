"""Decisions with a model: a cross-encoder that answers typed questions about a text or a state ("decider").

Types declare questions, the model proposes, checks decide. A question's kind comes from its type (solvi.typed):

  choice  Literal[...] / an Enum          one option (softmax); "other" / "none" may be an abstain threshold
  multi   list[Literal[...]]              every option that applies (a sigmoid per option)
  score   Scale[...] (2–10 levels)        ordered levels (softmax; the value is the median, the expected level is recorded)
  noul    bool / Literal["yes", "no"]     yes or no

The decider (solvi-decide, ModernBERT) reads one sequence per question — or, when the checkpoint says it can, several
questions about the same input in one sequence:

    [mode] task[opt] option 1[opt] option 2 ... [mode] task 2[opt] ... [SEP] input

and gives one logit per option marker (and, with an act head, one "act" logit per question). The input is a text, or a
state (a dict, a list, a pydantic model) serialized by `state_text` — the one serialization the training side uses too
(see docs/decide_format.md, which also defines the checkpoint's capability fields).

In solvi a decider is a catalog part like any other: `model.decision(...)` returns a function that returns
`solvi.Decision(value, probs)`. The value is one of the declared options by construction, the part's provenance is `decided`
and the model's identity (weights, calibration and adaptation of this part) is in the trace, so the closed set,
`min_confidence`, constraints with joint decoding, hard checks, the audit and the stats apply unchanged. A decision the model
escalates (its act head, or a calibrated confidence below the part's `escalate_below`) is rejected like an unsure one: the
fact is missing, the answer abstains, and the audit and stats say "model escalated" / "low confidence".

On top of the raw logits, per question (task, options, kind):
  - label-bias correction without labels (`adapt`): the mean logit of each option over unlabelled inputs of the domain is
    subtracted before the softmax (the decider likes some labels regardless of the text; +7 points in research L14b);
  - few-shot adaptation "S" (`fit`, `teach`): a shift and a shared scale fitted on k labelled examples (L-BFGS), with a
    temperature fitted on out-of-fold predictions, so confidences are calibrated; `teach` updates the shift at once. The
    shift is per option (choice, multi), a tilt / spread over the levels (score) or one yes−no bias (noul);
  - "other" / "none" as an abstain threshold: such an option is not scored by the model; it is chosen when the best real
    option's calibrated probability is below a threshold (fitted on labelled examples that include it, else the default);
  - `calibrate_for(examples, error=0.05)`: the escalation threshold for a target error rate.

Backends: "torch" (`solvi[model]`) or "onnx" (`solvi[onnx]`: onnxruntime + tokenizers, no torch), or any object with
`logits(items)` (and optionally `logits_pass(passes)`) — tests, other models: see DecideModel."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np

from .core import Decision, Quote, Unknown
from .provenance import ESCALATED, INSTRUCTION

OPT, ONE, MANY = "[unused0]", "[unused1]", "[unused2]"
MARKERS = {"option": OPT, "single": ONE, "multi": MANY, "score": "[unused3]", "noul": "[unused4]"}
OTHER_NAMES = ("other", "none", "none of the above", "none of these", "other / none", "nothing")
DEFAULT_T = {"l14b_decider v1": 1.45}          # temperature fitted on the training pool's validation split (L14b)
LEGACY_FORMAT, TYPED_FORMAT, FORMAT = "l14b_decider v1", "l14f typed v1", "solvi_decide v2"
TYPED2_FORMAT = "l14g typed v2"                 # L14g: rank, number, span, "not stated", evidence (subformat of solvi_decide v2)
ACT_FEATURES = ("confidence", "margin", "entropy", "act_logit", "n_options", "kind=choice", "kind=multi", "kind=score",
                "kind=noul")
ACT_FEATURES_V3 = ACT_FEATURES + ("p_unknown", "kind=rank", "kind=number", "kind=span")
KINDS = ("choice", "multi", "score", "noul")
KINDS_V3 = KINDS + ("rank", "number", "span")
V3_MARKERS = {"rank": "[unused5]", "number": "[unused6]", "span": "[unused7]"}
WIRE = {"choice": "single", "multi": "multi", "score": "score", "noul": "noul", "rank": "rank", "number": "number",
        "span": "span"}                           # question kind → mode on the wire
_KIND = {"choice": "choice", "single": "choice", "one": "choice", "multi": "multi", "many": "multi", "score": "score",
         "ordinal": "score", "scale": "score", "noul": "noul", "yes_no": "noul", "yesno": "noul", "bool": "noul",
         "rank": "rank", "ranking": "rank", "number": "number", "estimate": "number", "span": "span"}
NULL_SOURCE = "text"                            # the source of a pointer quote before it is bound to its fact


def _kind(kind, multi=False):
    if kind is None:
        return "multi" if multi else "choice"
    k = _KIND.get(str(kind).lower())
    if k is None:
        raise ValueError(f"kind must be one of {KINDS_V3}, not {kind!r}")
    return k


# ------------------------------------------------------------------------------------------------ state serialization
SERIALIZATIONS = ("paths", "tree", "json")      # the state serializations this solvi writes (state_text)
_KEY_OK = re.compile(r"^[A-Za-z0-9_\-]+$")


def state_text(obj, fmt="paths"):
    """The input a decider reads: a text as it is; any other value (a dict, a list, a pydantic model, a dataclass) first as
    JSON data (see `jsonable`), then serialized — by default "paths", one line per leaf with its full key path:

        customer.name: Anna
        items[0].sku: A-1
        items[0].qty: 2
        note: two lines of text

    Keys in their order (a pydantic model: field order); a key that is not [A-Za-z0-9_-]+ is written ["key"] (a JSON
    string); strings without quotes (a new line becomes a space); null / true / false; floats rounded to 6 decimals without
    trailing zeros; empty {} and [] kept; a scalar at the top is ".: value". "tree" is the YAML-like indented form, "json"
    is json.dumps with ", " / ": " separators. These are exactly the L14f training serializations
    (exps_v2/experiments/l14f_format.py `serialize`); docs/decide_format.md has the rules."""
    if isinstance(obj, Quote):
        obj = obj.value
    if isinstance(obj, str):
        return obj
    data = jsonable(obj)
    if fmt == "json":
        return json.dumps(data, ensure_ascii=False, separators=(", ", ": "))
    out = []
    if fmt == "paths":
        _paths(data, "", out)
    elif fmt == "tree":
        _tree(data, 0, out)
    else:
        raise ValueError(f"unknown state serialization {fmt!r} (this solvi writes {SERIALIZATIONS})")
    return "\n".join(out)


def jsonable(v):
    """A Python value → JSON data, deterministically: a pydantic model → its model_dump(), a dataclass → its fields, dates and
    times → ISO 8601, an Enum → its value, Decimal / UUID / other objects → str, bytes → UTF-8 text, tuples → lists, sets →
    lists sorted by their JSON text, numpy → Python numbers."""
    import datetime
    if v is None or isinstance(v, (str, bool, int, float)):
        return v.value if isinstance(v, Enum) else v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, Enum):
        return jsonable(v.value)
    if isinstance(v, (datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray)):
        return bytes(v).decode("utf-8", "replace")
    from .typed import is_model
    if is_model(v):
        return jsonable(v.model_dump())
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: jsonable(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, Mapping):
        return {_jkey(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((jsonable(x) for x in v), key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
    if isinstance(v, np.ndarray):
        return v.tolist()
    return str(v)


def _jkey(k):
    if isinstance(k, Enum):
        k = k.value
    return k if isinstance(k, (str, int, float, bool)) or k is None else str(k)


def _scalar(v):
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(round(v, 6)).rstrip("0").rstrip(".") if abs(v) < 1e15 else repr(v)
    return str(v).replace("\n", " ")


def _pkey(prefix, k):
    k = str(k)
    part = k if _KEY_OK.match(k) else json.dumps(k, ensure_ascii=False)
    if not _KEY_OK.match(k):
        return f"{prefix}[{part}]"
    return f"{prefix}.{part}" if prefix else part


def _paths(v, prefix, out):
    if isinstance(v, dict):
        if not v:
            out.append(f"{prefix or '.'}: {{}}")
        for k, x in v.items():
            _paths(x, _pkey(prefix, k), out)
    elif isinstance(v, (list, tuple)):
        if not v:
            out.append(f"{prefix or '.'}: []")
        for i, x in enumerate(v):
            _paths(x, f"{prefix}[{i}]", out)
    else:
        out.append(f"{prefix or '.'}: {_scalar(v)}")


def _tree(v, ind, out, key=None):
    pad = "  " * ind
    if isinstance(v, dict):
        if key is not None:
            out.append(f"{pad}{key}:" + (" {}" if not v else ""))
            ind += 1
        for k, x in v.items():
            _tree(x, ind, out, str(k))
    elif isinstance(v, (list, tuple)):
        if key is not None:
            out.append(f"{pad}{key}:" + (" []" if not v else ""))
        p2 = "  " * (ind + (1 if key is not None else 0))
        for x in v:
            if isinstance(x, dict) and x:
                sub = []
                _tree(x, 0, sub)
                out.append(f"{p2}- {sub[0]}")
                out.extend(f"{p2}  {s}" for s in sub[1:])
            elif isinstance(x, (list, tuple)):
                sub = []
                _tree(x, 0, sub)
                out.append(f"{p2}-")
                out.extend(f"{p2}  {s}" for s in sub)
            else:
                out.append(f"{p2}- {_scalar(x)}")
    else:
        out.append(f"{pad}{key}: {_scalar(v)}" if key is not None else f"{pad}{_scalar(v)}")


def _is_text(v):
    return v is None or isinstance(v, str) or (isinstance(v, Quote) and (v.value is None or isinstance(v.value, str)))


def _text(v, fmt="paths"):
    """A fact's value → the decider's input: a text as it is (a Quote: its value), a scalar (number, date, Enum) as its
    text, a container (dict, list, model) by state_text."""
    if isinstance(v, Quote):
        v = v.value
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    data = jsonable(v)
    return state_text(data, fmt) if isinstance(data, (dict, list)) else _scalar(data)


def _single(x):
    """Is x one input (a text, a Quote, a state) rather than a list of inputs?"""
    if isinstance(x, (str, Quote, Mapping)) or (dataclasses.is_dataclass(x) and not isinstance(x, type)):
        return True
    from .typed import is_model
    return is_model(x)


# ------------------------------------------------------------------------------------------------ what the scorer sees
@dataclass(frozen=True)
class Item:
    """One question to score: the scorer returns one logit (or a row of logits, one column per mode) per option, or
    {"logits": ..., "act": logit} when it has an act head."""
    task: str
    options: tuple
    descriptions: tuple | None
    text: str
    multi: bool = False
    kind: str = ""          # the mode on the wire when it is neither "single" nor "multi": "score", "noul", "rank", ...
    pointer: bool = False   # the question wants the pointer (a span answer, or evidence quotes): full layout only
    unknown: bool = False   # "not stated" is an answer to this question (scorers that ask in words, e.g. solvi.llm)

    @property
    def mode(self):
        return self.kind or ("multi" if self.multi else "single")


@dataclass(frozen=True)
class Pass:
    """Several questions about one input, scored in one forward pass (checkpoints with multi-question support)."""
    text: str
    items: tuple


def prompt(task, options, descriptions=None, multi=False, mode=None, markers=None):
    """A question's segment: "[mode] task[opt] option ..." (an option with a description is "label: description")."""
    mk = markers or MARKERS
    mode = mode or ("multi" if multi else "single")
    opts = [o if not descriptions or not descriptions[i] else f"{o}: {descriptions[i]}" for i, o in enumerate(options)]
    return f"{mk[mode]} {task}" + "".join(f"{mk['option']} {o}" for o in opts)


def pass_prompt(items, markers=None):
    """The first segment of a pass: the questions' segments joined by a space."""
    return " ".join(prompt(it.task, it.options, it.descriptions, mode=it.mode, markers=markers) for it in items)


class Logits(np.ndarray):
    """A question's logits [K] with an l14g checkpoint's extra outputs: `unknown` (the "not stated" logit) and `pointer`
    (decoded: {"null": p(null span), "spans": [(p, start, end, text)]}, most probable first) — and from a scorer that
    can fail on one question (solvi.llm): `escalate` (why the output is not usable: the decision escalates with it),
    `transient` (not cached: ask again next time) and `info` (recorded in the decision's extra)."""
    unknown = None
    pointer = None
    escalate = None
    transient = False
    info = None


def decode_pointer(ptr, text, max_span=40, top=20, temperature=1.0):
    """The pointer's raw output → {"null", "spans"}: a span's score is start_i + end_j over the input's tokens i ≤ j <
    i + max_span, the null span's start_m + end_m at the mode marker; p = softmax over the null span and every span
    (exactly `span_dist` of exps_v2/experiments/l14g_format.py). A span's text is the input's characters from token i's
    start to token j's end, without surrounding whitespace — so it is literally in the input. `ptr`: {"start": [T],
    "end": [T], "offsets": [(char start, char end)] per token, "null": [start_m, end_m] (or their sum)}. `temperature`
    (the checkpoint's `temperature.span`) divides every start / end score, the null span's too, before the softmax — as
    the L14g calibration fitted it."""
    t = float(temperature) if temperature and temperature > 0 else 1.0
    s = np.asarray(ptr["start"], dtype=np.float64).ravel() / t
    e = np.asarray(ptr["end"], dtype=np.float64).ravel() / t
    offs = [(int(a), int(b)) for a, b in ptr["offsets"]]
    nl = ptr.get("null", 0.0)
    null = float(np.sum(np.asarray(nl, dtype=np.float64))) / t
    L = len(s)
    if L == 0:
        return {"null": 1.0, "spans": []}
    M = s[:, None] + e[None, :]
    ii, jj = np.indices((L, L))
    M = np.where((jj >= ii) & (jj < ii + max_span), M, -np.inf)
    mx = max(null, float(M.max()))
    Z = math.exp(null - mx) + float(np.exp(M - mx).sum())
    spans, seen = [], set()
    for kk in np.argsort(M, axis=None)[::-1][: top * 4]:
        v = float(M.flat[kk])
        if not np.isfinite(v):
            break
        i, j = divmod(int(kk), L)
        a, b = offs[i][0], offs[j][1]
        while a < b and text[a:a + 1].isspace():
            a += 1
        while b > a and text[b - 1:b].isspace():
            b -= 1
        if a >= b or (a, b) in seen:
            continue
        seen.add((a, b))
        spans.append((math.exp(v - mx) / Z, a, b, text[a:b]))
        if len(spans) >= top:
            break
    return {"null": math.exp(null - mx) / Z, "spans": spans}


def _typed_span(spans, vtype):
    """For a typed span (`Span[float]`): (index, probability mass) of the most probable span inside the best span whose
    text parses as `vtype` — the best span trimmed to its value ('149.90 EUR' → '149.90'); the mass is that of every span
    between the two (they all give this value). Never a span outside the best one: when nothing inside it parses, the best
    span is kept and the answer's type check rejects it. (0, None) for str / untyped spans or when the best span parses."""
    if vtype is None or vtype is str:
        return 0, None
    from .typed import adapter
    try:
        ta = adapter(vtype)
    except Exception:  # noqa: BLE001
        return 0, None

    def parses(t):
        try:
            ta.validate_python(t.strip())
            return True
        except ValueError:
            return False

    _, a0, b0, t0 = spans[0]
    if parses(t0):
        return 0, None
    for k, (_, a, b, t) in enumerate(spans):
        if a0 <= a and b <= b0 and parses(t):
            return k, float(sum(q for q, x, y, _ in spans if a0 <= x <= a and b <= y <= b0))
    return 0, None


def pointer_evidence(ptr, threshold=0.15, max_spans=3):
    """Evidence quotes from a decoded pointer: greedily up to max_spans non-overlapping spans with p ≥ threshold; none
    when the null span is at least as probable as the best span (the l14g contract's `evidence`)."""
    spans = ptr["spans"]
    if not spans or ptr["null"] >= spans[0][0]:
        return []
    got = []
    for p, a, b, t in spans:
        if p < threshold or len(got) >= max_spans:
            break
        if all(b <= x or a >= y for _, x, y, _ in got):
            got.append((p, a, b, t))
    return [Quote(t, a, b, NULL_SOURCE, float(p)) for p, a, b, t in sorted(got, key=lambda g: g[1])]


# ------------------------------------------------------------------------------------------------ backends
class _Encoder:
    """Tokenization with the checkpoint's tokenizer.json (the `tokenizers` library).

    encode (one question, or the "concat" layout): [CLS] questions [SEP] input [SEP], only the input is truncated;
    encode_block (the "block" layout): [CLS] input [SEP] question 1 [SEP] question 2 [SEP] …, with position ids (each block
    continues the input's positions) and a block id per token (−1 input, j block j) for the attention mask.
    Both return the token ids and, per question, the position of its mode marker and of its option markers."""

    def __init__(self, path, max_len, markers=None):
        from tokenizers import Tokenizer
        f = os.path.join(path, "tokenizer.json")
        self.tok = Tokenizer.from_file(f)
        self.tok.no_padding()
        self.tok.enable_truncation(max_length=max_len, strategy="only_second")
        self.raw = Tokenizer.from_file(f)             # no truncation: the block layout budgets the input itself
        self.raw.no_padding()
        self.raw.no_truncation()
        self.max_len = max_len
        self.markers = {**MARKERS, **(markers or {})}
        self.opt_id = self.tok.token_to_id(self.markers["option"])
        pad = next((self.tok.token_to_id(t) for t in ("[PAD]", "<pad>") if self.tok.token_to_id(t) is not None), 0)
        self.pad_id = pad
        self.cls_id = next((self.tok.token_to_id(t) for t in ("[CLS]", "<s>") if self.tok.token_to_id(t) is not None), None)
        self.sep_id = next((self.tok.token_to_id(t) for t in ("[SEP]", "</s>") if self.tok.token_to_id(t) is not None), None)
        if self.opt_id is None:
            raise ValueError(f"the tokenizer has no {self.markers['option']} token: not a solvi-decide checkpoint")
        self.mode_ids = {i for m, t in self.markers.items() if m != "option" for i in [self.tok.token_to_id(t)]
                         if i is not None}

    def _groups(self, ids, keep, items):
        groups = []
        for i, t in enumerate(ids):
            if not keep(i):
                continue
            if t in self.mode_ids:
                groups.append((i, []))
            elif t == self.opt_id and groups:
                groups[-1][1].append(i)
        if len(groups) != len(items) or any(len(g[1]) != len(it.options) for g, it in zip(groups, items)):
            raise ValueError(f"found {sum(len(g[1]) for g in groups)} option markers for {sum(len(it.options) for it in items)} "
                             f"options (an option contains {self.markers['option']}, or the options do not fit in "
                             f"{self.max_len} tokens)")
        return groups

    def encode(self, items, text):
        p = pass_prompt(items, self.markers)
        try:
            enc = self.tok.encode(p, text or " ")
        except Exception as e:  # noqa: BLE001  (the questions alone are longer than max_len)
            raise ValueError(f"task and options do not fit in {self.max_len} tokens: {e}") from None
        seq = enc.sequence_ids
        groups = self._groups(enc.ids, lambda i: seq[i] == 0, items)
        if any(it.pointer for it in items):           # the input's tokens and their character offsets (for the pointer)
            pos = [i for i, x in enumerate(seq) if x == 1]
            return enc.ids, groups, None, None, (pos, [tuple(enc.offsets[i]) for i in pos])
        return enc.ids, groups

    def encode_block(self, items, text, max_len):
        if self.cls_id is None or self.sep_id is None:
            raise ValueError("the tokenizer has no [CLS] / [SEP] token: no block layout")
        blocks = [self.raw.encode(prompt(it.task, it.options, it.descriptions, mode=it.mode, markers=self.markers),
                                  add_special_tokens=False).ids + [self.sep_id] for it in items]
        budget = max_len - sum(len(b) for b in blocks) - 2
        if budget < 8:
            raise ValueError(f"{len(items)} questions do not fit in {max_len} tokens together")
        st = self.raw.encode(text or " ", add_special_tokens=False).ids[:budget]
        ids = [self.cls_id] + st + [self.sep_id]
        Ls = len(ids)
        pids, blk = list(range(Ls)), [-1] * Ls
        for j, b in enumerate(blocks):
            ids += b
            pids += list(range(Ls, Ls + len(b)))
            blk += [j] * len(b)
        groups = self._groups(ids, lambda i: blk[i] >= 0, items)
        return ids, groups, pids, blk


def block_masks(blk, pids, window):
    """blk [B, L] (−1 input, j ≥ 0 block j, −2 padding), pids [B, L] → (full, sliding) boolean masks [B, 1, L, L] (True =
    may attend): the input sees only the input, block j sees the input and itself, blocks do not see each other; the sliding
    (local) layers also need |pos_i − pos_j| ≤ window. Padding rows see themselves only."""
    qb, kb = blk[:, :, None], blk[:, None, :]
    see = ((kb == -1) & (qb != -2)) | ((qb == kb) & (qb >= 0))
    eye = np.eye(blk.shape[1], dtype=bool)[None]
    see = see | (eye & (qb == -2))
    near = np.abs(pids[:, :, None] - pids[:, None, :]) <= window
    return see[:, None], (see & (near | eye))[:, None]


class BlockUnsupported(ValueError):
    """The scorer cannot run the block layout at all (e.g. an ONNX export without its inputs): one question per sequence."""


def _batches(encs, bs):
    order = sorted(range(len(encs)), key=lambda i: len(encs[i][0]))
    for b in range(0, len(order), bs):
        yield order[b:b + bs]


class _NetScorer:
    """Shared by the ONNX and torch backends: encode, batch by length, read the logits at the markers (and the act logit at
    each question's mode marker)."""

    def _setup(self, path, max_len, bs, caps):
        caps = caps or capabilities({})
        self.enc = _Encoder(path, max_len, caps["markers"])
        self.bs = bs
        self.act_col = caps["act"]["column"] if caps["act"] is not None else None
        self.mq = caps["multi_question"]
        unk, ptr = caps.get("unknown"), caps.get("pointer")
        self.unk_col = unk["column"] if unk else None
        self.ptr_cols = (ptr["start"], ptr["end"]) if ptr else None

    def logits(self, items):
        return [o[0] for o in self._run([self.enc.encode((it,), it.text) for it in items])]

    def logits_pass(self, passes):
        if self.mq is not None and self.mq["layout"] == "block":
            return self._run([self.enc.encode_block(p.items, p.text, self.mq["max_len"]) for p in passes], block=True)
        return self._run([self.enc.encode(p.items, p.text) for p in passes])

    def _run(self, encs, block=False):
        out = [None] * len(encs)
        for ch in _batches(encs, self.bs):
            L = max(len(encs[i][0]) for i in ch)
            ids = np.full((len(ch), L), self.enc.pad_id, dtype=np.int64)
            att = np.zeros((len(ch), L), dtype=np.int64)
            pids = np.zeros((len(ch), L), dtype=np.int64) if block else None
            blk = np.full((len(ch), L), -2, dtype=np.int64) if block else None
            for r, i in enumerate(ch):
                n = len(encs[i][0])
                ids[r, :n], att[r, :n] = encs[i][0], 1
                if block:
                    pids[r, :n], blk[r, :n] = encs[i][2], encs[i][3]
            if block:
                lg = self._forward(ids, att, pids, block_masks(blk, pids, self.mq["window"]))
            else:
                lg = self._forward(ids, att)
            act = self.act_col is not None and self.act_col < lg.shape[-1]
            unk = self.unk_col is not None and self.unk_col < lg.shape[-1]
            ptr = self.ptr_cols is not None and max(self.ptr_cols) < lg.shape[-1]
            for r, i in enumerate(ch):
                row = []
                for mpos, opos in encs[i][1]:
                    if not (act or unk):
                        row.append(lg[r, opos])
                        continue
                    o = {"logits": lg[r, opos]}
                    if act:
                        o["act"] = float(lg[r, mpos, self.act_col])
                    if unk:
                        o["unknown"] = float(lg[r, mpos, self.unk_col])
                    if ptr and len(encs[i]) > 4:          # full layout with the input's tokens: the pointer (l14g)
                        pos, offs = encs[i][4]
                        a, b = self.ptr_cols
                        o["pointer"] = {"start": lg[r, pos, a], "end": lg[r, pos, b],
                                        "null": [float(lg[r, mpos, a]), float(lg[r, mpos, b])], "offsets": offs}
                    row.append(o)
                out[i] = row
        return out


class OnnxScorer(_NetScorer):
    """onnxruntime session over onnx/model_*.onnx (inputs input_ids, attention_mask; output logits [B, L, C]). The block
    layout needs an export with the inputs position_ids, full_attention_mask and sliding_attention_mask ([B, 1, L, L] bool);
    without them several questions fall back to one per pass."""

    def __init__(self, path, max_len=512, device=None, onnx_file=None, bs=16, caps=None):
        import onnxruntime as ort
        self._setup(path, max_len, bs, caps)
        self.file = onnx_file or _onnx_file(path)
        if self.file is None:
            raise FileNotFoundError(f"no ONNX model in {path}/onnx")
        providers = ["CPUExecutionProvider"]
        if device and str(device).startswith("cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider"] + providers
        self.sess = ort.InferenceSession(self.file, providers=providers)
        self.names = [i.name for i in self.sess.get_inputs()]
        self.tag = "onnx:" + os.path.basename(self.file)

    def logits_pass(self, passes):
        if self.mq is not None and self.mq["layout"] == "block" and not {"position_ids", "full_attention_mask",
                                                                          "sliding_attention_mask"} <= set(self.names):
            raise BlockUnsupported("this ONNX export has no block-layout inputs")
        return super().logits_pass(passes)

    def _forward(self, ids, att, pids=None, masks=None):
        feed = {"input_ids": ids, "attention_mask": att}
        if masks is not None:
            feed.update(position_ids=pids, full_attention_mask=masks[0], sliding_attention_mask=masks[1])
        return self.sess.run(None, {k: v for k, v in feed.items() if k in self.names})[0].astype(np.float32)


class TorchScorer(_NetScorer):
    """The checkpoint's encoder (transformers AutoModel from config.json) + the option head, weights from model.safetensors.
    The block layout passes per-layer-type attention masks and position ids (sdpa attention)."""

    def __init__(self, path, max_len=512, device=None, bs=16, caps=None):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._setup(path, max_len, bs, caps)
        sd = load_file(os.path.join(path, "model.safetensors"))
        cfg = AutoConfig.from_pretrained(path)
        if self.mq is not None and self.mq["layout"] == "block":
            cfg._attn_implementation = "sdpa"
            encoder = AutoModel.from_config(cfg, attn_implementation="sdpa")
        else:
            encoder = AutoModel.from_config(cfg)
        h = encoder.config.hidden_size
        n_out = sd["head.3.weight"].shape[0] if "head.3.weight" in sd else 2
        head = torch.nn.Sequential(torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.LayerNorm(h), torch.nn.Linear(h, n_out))

        class Decider(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc, self.head = encoder, head

            def forward(self, input_ids, attention_mask, position_ids=None):
                return self.head(self.enc(input_ids=input_ids, attention_mask=attention_mask,
                                          position_ids=position_ids).last_hidden_state)

        self.model = Decider()
        self.model.load_state_dict({k: v.float() for k, v in sd.items()})
        self.model.eval().to(self.device)
        self.tag = "torch"

    def _forward(self, ids, att, pids=None, masks=None):
        torch = self.torch
        dev = self.device
        mask = torch.from_numpy(att).to(dev)
        pos = None
        if masks is not None:
            mask = {"full_attention": torch.from_numpy(masks[0]).to(dev), "sliding_attention": torch.from_numpy(masks[1]).to(dev)}
            pos = torch.from_numpy(pids).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
            return self.model(torch.from_numpy(ids).to(dev), mask, pos).float().cpu().numpy()


def _onnx_file(path):
    d = os.path.join(path, "onnx")
    if not os.path.isdir(d):
        return None
    files = sorted(f for f in os.listdir(d) if f.endswith(".onnx"))
    for pref in ("model_fp16.onnx", "model.onnx", "model_fp32.onnx"):
        if pref in files:
            return os.path.join(d, pref)
    return os.path.join(d, files[0]) if files else None


def _file_fingerprint(paths, chunks=32, size=4096):
    """Names, sizes and evenly spaced 4 KB chunks of the checkpoint files: ~ms, and any retrained checkpoint differs."""
    h = hashlib.sha256()
    for p in paths:
        if not p or not os.path.isfile(p):
            continue
        n = os.path.getsize(p)
        h.update(f"{os.path.basename(p)}:{n}".encode())
        with open(p, "rb") as fh:
            if n <= chunks * size:
                h.update(fh.read())
            else:
                for k in range(chunks):
                    fh.seek(k * (n - size) // (chunks - 1))
                    h.update(fh.read(size))
    return h.hexdigest()[:16]


# ------------------------------------------------------------------------------------------------ checkpoint capabilities
def _multi_question(v):
    """→ {"layout", "max_questions", "max_len", "window"} or None (one question per pass)."""
    if v is None or v is False or v == 0:
        return None
    if v is True:
        v = {}
    elif isinstance(v, int) and not isinstance(v, bool):
        v = {"max_questions": v}
    if not isinstance(v, dict) or not v.get("enabled", True):
        return None
    mq = {"layout": str(v.get("layout", "block")), "max_questions": int(v.get("max_questions", 6)),
          "max_len": int(v.get("max_len", 1024)), "window": int(v.get("window", 64))}
    if mq["layout"] not in ("block", "concat"):
        raise ValueError(f"unknown multi-question layout {mq['layout']!r} (block, concat)")
    return mq if mq["max_questions"] > 1 else None


_DEFAULTS = {
    # L14b–L14e: choose-one / multi-label, text input, two head columns, one question per pass, no act head
    LEGACY_FORMAT: {"modes": ["single", "multi"], "columns": {"single": 0, "multi": 1}, "noul_labels": ["yes", "no"],
                    "state_serialization": ["text"], "act": None},
    # L14f: every kind natively, "paths" / "tree" / "json" states, three head columns (the third: act, at the mode token)
    TYPED_FORMAT: {"modes": ["single", "multi", "score", "noul"],
                   "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "act": 2}, "noul_labels": ["true", "false"],
                   "state_serialization": ["paths", "tree", "json"], "act": {}},
    # L14g: + rank, number (bins as ordered options), span; "not stated" (column 3 at the mode marker); a pointer
    # (columns 4 / 5 over the input's tokens, full layout only) for span answers and evidence quotes
    TYPED2_FORMAT: {"modes": ["single", "multi", "score", "noul", "rank", "number", "span"],
                    "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "rank": 0, "number": 0, "act": 2,
                                "unknown": 3, "span_start": 4, "span_end": 5},
                    "noul_labels": ["true", "false"], "state_serialization": ["paths", "tree", "json"], "act": {},
                    "unknown": {"column": 3}, "pointer": {"start": 4, "end": 5}},
}


def _v3(meta):
    """Is this a checkpoint of the answer-primitives contract (L14g: `subformat` 'l14g typed v2', or format 'l14g typed v2'
    / 'solvi_decide v3')? Only such checkpoints get the new capability fields (older ones hash as before)."""
    fmt, sub = str(meta.get("format", "")), str(meta.get("subformat", ""))
    return fmt.startswith(("l14g", "solvi_decide v3")) or sub.startswith("l14g")


def _unknown_caps(v, columns):
    """"not stated": {"column", "label", "joint" (kinds whose options compete with it in one softmax), "multi" ("sigmoid"),
    "span" ("null_span"), "threshold"} or None."""
    if v is None or v is False:
        return None
    v = {} if v is True else dict(v)
    return {"column": int(v.get("column", columns.get("unknown", 3))), "label": str(v.get("label", "not stated")),
            "joint": [str(k) for k in v.get("joint", ["single", "score", "noul", "rank", "number"])],
            "multi": str(v.get("multi", "sigmoid")), "span": str(v.get("span", "null_span")),
            "threshold": float(v.get("threshold", 0.5))}


def _pointer_caps(v, columns):
    """The pointer: {"start", "end" (columns), "layouts", "max_span_tokens", "null", "evidence": {"threshold",
    "max_spans"}} or None."""
    if v is None or v is False:
        return None
    v = {} if v is True else dict(v)
    ev = dict(v.get("evidence") or {})
    return {"start": int(v.get("start", columns.get("span_start", 4))), "end": int(v.get("end", columns.get("span_end", 5))),
            "layouts": [str(x) for x in v.get("layouts", ["full"])], "max_span_tokens": int(v.get("max_span_tokens", 40)),
            "null": str(v.get("null", "mode")),
            "evidence": {"threshold": float(ev.get("threshold", 0.15)), "max_spans": int(ev.get("max_spans", 3))}}


def capabilities(meta, multi_question=None, act=None):
    """What a checkpoint can do, from its solvi_decide.json (see docs/decide_format.md): the fields it declares over the
    defaults of its format ('l14b_decider v1': L14b–L14e; 'l14f typed v1': L14f; 'solvi_decide v2': the legacy defaults,
    everything else declared). multi_question / act: overrides (experiments), part of the fingerprint."""
    meta = meta or {}
    fmt = str(meta.get("format", ""))
    v3 = _v3(meta)
    base = (_DEFAULTS[TYPED2_FORMAT] if v3 and "l14g" in fmt + str(meta.get("subformat", "")) else
            _DEFAULTS[TYPED_FORMAT] if fmt.startswith("l14f") else _DEFAULTS[LEGACY_FORMAT])
    version = 3 if v3 else 2 if fmt.startswith(("l14f", "solvi_decide")) else 1     # other formats (stand-ins) hash as before
    modes = [str(m) for m in (meta.get("modes") or base["modes"])]
    markers = {**MARKERS, **(V3_MARKERS if v3 else {}), **(meta.get("markers") or {})}
    columns = {**base["columns"], **(meta.get("columns") or {})}
    ser = [str(x) for x in (meta.get("state_serialization") or base["state_serialization"])]
    known = [x for x in ser if x in SERIALIZATIONS]
    if version >= 2 and not known and ser != ["text"]:
        raise ValueError(f"the checkpoint reads states as {ser}; this solvi writes {SERIALIZATIONS}")
    a = meta.get("act", base["act"] if "act_head" not in meta else ({} if meta["act_head"] else None))
    if a is False:
        a = None
    if act is not None:
        a = (a if a is not None else {}) if act else None
    if a is not None:
        a = dict(a)
        a.setdefault("column", columns.get("act", 2))
        a.setdefault("temperature", float((meta.get("temperature") or {}).get("act", 1.0))
                     if isinstance(meta.get("temperature"), dict) else 1.0)
        cal = a.get("calibrator")
        if cal is not None:
            bad = [f for f in cal.get("features", []) if f not in (ACT_FEATURES_V3 if v3 else ACT_FEATURES)]
            if bad or len(cal.get("features", [])) != len(cal.get("weights", [])):
                raise ValueError(f"act calibrator: unknown features {bad} or features / weights of different lengths "
                                 f"(known: {ACT_FEATURES})")
    mq = _multi_question(meta.get("multi_question") if multi_question is None else multi_question)
    caps = {"format": fmt, "version": version, "modes": modes, "markers": markers, "columns": columns,
            "noul_labels": [str(x) for x in (meta.get("noul_labels") or base["noul_labels"])],
            "serialization": ser, "state_format": known[0] if known else "paths", "act": a, "multi_question": mq,
            "max_questions": mq["max_questions"] if mq else 0}
    if v3:                                        # the answer-primitives contract (older formats: the dict above, as before)
        raw_mq = meta.get("multi_question") if multi_question is None else multi_question
        if mq is not None:
            mq["pointer"] = bool(raw_mq.get("pointer", False)) if isinstance(raw_mq, dict) else False
        caps["subformat"] = str(meta.get("subformat", ""))
        caps["unknown"] = _unknown_caps(meta.get("unknown", base.get("unknown")), columns)
        caps["pointer"] = _pointer_caps(meta.get("pointer", base.get("pointer")), columns)
        num = meta.get("number") if isinstance(meta.get("number"), dict) else {}
        caps["number"] = {"interval": float(num.get("interval", 0.8))}
    return caps


# ------------------------------------------------------------------------------------------------ calibration of one decision
def _given(logits, n):
    """Precomputed logits for adapt / fit / teach → a list of n arrays."""
    Z = [np.asarray(z, float) for z in logits]
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
            from .core import bin_labels
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
        """The question as the scorer sees it; a native yes/no question uses the checkpoint's option labels (L14f: "true",
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
    """A Python type → (kind, options, descriptions, as_bool, primitive settings) of a decision (solvi.typed.question_kind;
    Maybe[T] — "not stated" allowed; Span[T], Rank[...], Estimate[...] — span, rank, number)."""
    from .typed import primitive_answer, question_kind, split_unknown
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


# ------------------------------------------------------------------------------------------------ the model
class DecideModel:
    """A decider: an input (a text or a state) + a question (task, options, kind) → a probability per option.

    DecideModel.load(path_or_hf_id, device=None, backend="auto") loads a solvi-decide checkpoint (a folder with config.json,
    tokenizer.json, solvi_decide.json, model.safetensors and/or onnx/model_fp16.onnx; or a Hugging Face id). `backend`:
    "torch", "onnx" or "auto" (ONNX when the file and onnxruntime are there, else torch). solvi_decide.json declares what the
    checkpoint can do (modes, act head, several questions per pass, temperatures, thresholds: docs/decide_format.md).

    DecideModel(scorer, meta=None, model_id=...) wraps any object with `logits(items)` → per Item an array [K] or [K, C] (a
    column per mode: [:, 0] choose-one, [:, 1] multi-label, more as `meta["columns"]` says) or {"logits": array, "act":
    logit}; optionally `logits_pass(passes)` → per Pass a list of those (one per question) and `fingerprint()`."""

    deterministic = True

    def __init__(self, scorer, meta=None, model_id=None, path=None, backend=None, cache_size=4096, multi_question=None,
                 act=None):
        self.scorer = scorer
        self.meta = dict(meta or {})
        self.model_id = model_id or getattr(scorer, "model_id", None) or type(scorer).__name__
        self.path = path
        self.backend = backend or getattr(scorer, "tag", type(scorer).__name__)
        self.caps = capabilities(self.meta, multi_question, act)
        self._overrides = {k: v for k, v in (("multi_question", multi_question), ("act", act)) if v is not None}
        fmt = self.meta.get("format", "")
        cal = self.meta.get("calibration", {}) if isinstance(self.meta.get("calibration"), dict) else {}
        T = self.meta.get("temperature", cal.get("temperature", DEFAULT_T.get(fmt, 1.0)))
        td = T if isinstance(T, dict) else {}
        self.temperature = float(td.get("single", td.get("choice", 1.0)) if td else T)
        self.temperature_multi = float(td.get("multi", self.meta.get("temperature_multi", cal.get("temperature_multi", 1.0))))
        a = self.caps["act"] or {}
        self.temperatures = {"single": self.temperature, "multi": self.temperature_multi,
                             "score": float(td.get("score", self.temperature)), "noul": float(td.get("noul", self.temperature)),
                             "act": float(a.get("temperature", 1.0))}
        if self.caps["version"] >= 3:               # rank, number, span (older checkpoints: the dict above, as before)
            self.temperatures.update(rank=float(td.get("rank", self.temperature)),
                                     number=float(td.get("number", td.get("score", self.temperature))),
                                     span=float(td.get("span", 1.0)))
        th = self.meta.get("thresholds") if isinstance(self.meta.get("thresholds"), dict) else {}
        self.other_threshold = float(th.get("other", self.meta.get("other_threshold", cal.get("other_threshold", 0.5))))
        self.multi_threshold = float(th.get("multi", self.meta.get("multi_threshold", cal.get("multi_threshold", 0.5))))
        self.act_threshold = float(a.get("threshold", th.get("act", 0.5)))
        self.act_thresholds = {float(k): float(v) for k, v in (a.get("threshold_for_error") or {}).items()}
        self.escalate_below = th.get("escalate_below")          # a default for parts that set none (None: no default)
        self.adaptations: dict[tuple, Adaptation] = {}
        self._cache, self._cache_size, self._lock = OrderedDict(), cache_size, threading.Lock()
        self._pass_lock = threading.Lock()
        self._block_failed = False
        self._wfp = None
        self.passes = 0                     # sequences the network encoded (a shared pass counts once)

    # --- loading and identity
    @classmethod
    def load(cls, path_or_id, device=None, backend="auto", max_len=None, bs=16, multi_question=None, act=None):
        """multi_question / act: override what solvi_decide.json declares (for experiments, e.g. testing a checkpoint
        in multi-question passes); both are part of the fingerprint."""
        path = str(path_or_id)
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download
            allow = None
            if backend == "onnx":
                allow = ["*.json", "onnx/*"]
            elif backend == "torch":
                allow = ["*.json", "*.safetensors"]
            path = snapshot_download(path, allow_patterns=allow)
        meta_file = os.path.join(path, "solvi_decide.json")
        if not os.path.isfile(meta_file):
            raise FileNotFoundError(f"{path} has no solvi_decide.json: not a solvi-decide checkpoint")
        meta = json.load(open(meta_file))
        fmt = str(meta.get("format", ""))
        if fmt and not (fmt.startswith("l14b_decider") or
                        re.match(r"^(l14f typed v1|l14g typed v2|solvi_decide v[23])(\.\d+)*$", fmt)):
            raise ValueError(f"unknown decider format {fmt!r} (this solvi reads {LEGACY_FORMAT!r}, {TYPED_FORMAT!r}, "
                             f"{FORMAT!r} (with subformat {TYPED2_FORMAT!r}) and 'solvi_decide v3')")
        caps = capabilities(meta, multi_question, act)
        n = int(max_len or meta.get("max_len", 512))
        if backend == "auto":
            try:
                import onnxruntime  # noqa: F401
                backend = "onnx" if _onnx_file(path) else "torch"
            except ImportError:
                backend = "torch"
        if backend == "onnx":
            scorer = OnnxScorer(path, n, device, bs=bs, caps=caps)
            weights = scorer.file
        elif backend == "torch":
            scorer = TorchScorer(path, n, device, bs=bs, caps=caps)
            weights = os.path.join(path, "model.safetensors")
        else:
            raise ValueError('backend must be "torch", "onnx" or "auto"')
        m = cls(scorer, meta, model_id=str(path_or_id), path=path, backend=scorer.tag, multi_question=multi_question, act=act)
        m._wfp = _file_fingerprint([os.path.join(path, f) for f in ("config.json", "solvi_decide.json", "tokenizer.json")]
                                   + [weights])
        return m

    @property
    def batchable(self):
        """Can several questions about one input share a forward pass (declared by the checkpoint, and the scorer has
        logits_pass)?"""
        return self.caps["max_questions"] > 1 and callable(getattr(self.scorer, "logits_pass", None))

    @property
    def has_act(self):
        """Does the checkpoint give an act / escalate signal per question?"""
        return self.caps["act"] is not None

    def act_threshold_for(self, error):
        """The act threshold the checkpoint ships for a target error rate (its `act.threshold_for_error` table: the entry
        with the largest error ≤ the target). ValueError when it ships none — use calibrate_for on your examples."""
        ok = [e for e in self.act_thresholds if e <= error + 1e-12]
        if not ok:
            raise ValueError(f"the checkpoint has no act threshold for error ≤ {error} (it has {sorted(self.act_thresholds)}); "
                             "use part.calibrate_for(examples, error=...)")
        return self.act_thresholds[max(ok)]

    def wire(self, kind):
        """How a question kind is asked: natively when the checkpoint was trained on it, else as a single choice (score:
        the levels; noul: the options "yes" / "no"; rank: the options, ordered by probability; number: the bins, as a score
        when the checkpoint has scores). A span is native only."""
        w = WIRE[kind]
        if w in ("single", "multi") or w in self.caps["modes"]:
            return w
        return "score" if w == "number" and "score" in self.caps["modes"] else "single"

    @property
    def has_unknown(self):
        """Does the checkpoint give a "not stated" output (an l14g checkpoint's `unknown`)?"""
        return self.caps.get("unknown") is not None

    @property
    def has_pointer(self):
        """Can the checkpoint point at a piece of its input (span answers, evidence quotes)?"""
        return self.caps.get("pointer") is not None and "span" in self.caps["modes"]

    def _T(self, sp):
        w = self.wire(sp.kind)
        return self.temperature_multi if w == "multi" else (self.temperature if w == "single" else self.temperatures[w])

    def weights_fingerprint(self):
        """A hash of the checkpoint (files, or the scorer's own fingerprint), the backend, the default calibration and (for
        checkpoints in the v2 format, or with overrides) the declared capabilities."""
        if self._wfp is None:
            fp = getattr(self.scorer, "fingerprint", None)
            self._wfp = str(fp()) if callable(fp) else f"unversioned:{type(self.scorer).__name__}"
        from .provenance import digest
        parts = [self._wfp, self.backend, self.temperature, self.temperature_multi, self.other_threshold, self.multi_threshold]
        if self.caps["version"] >= 2 or self._overrides:        # the 'l14b_decider v1' format hashes exactly as before
            parts.append({"caps": {k: v for k, v in self.caps.items()}, "temperatures": self.temperatures,
                          "act_threshold": self.act_threshold, "escalate_below": self.escalate_below,
                          "overrides": self._overrides})
        return digest("DecideModel", *parts)

    def fingerprint(self):
        """The checkpoint plus every adaptation (so a replay knows the calibration a decision used)."""
        from .provenance import digest
        return digest(self.weights_fingerprint(), {repr(k): a.params() for k, a in sorted(self.adaptations.items(), key=repr)})

    def metadata(self):
        """The checkpoint's metadata (without the training history), capabilities, the default calibration and every
        adaptation."""
        base = {k: v for k, v in self.meta.items() if k not in ("hist", "init_meta")}
        return {"model_id": self.model_id, "backend": self.backend, "weights": self.weights_fingerprint(),
                "capabilities": dict(self.caps), "temperature": self.temperature, "temperature_multi": self.temperature_multi,
                "temperatures": dict(self.temperatures), "other_threshold": self.other_threshold,
                "act_threshold": self.act_threshold, "meta": base,
                "adaptations": [{"task": k[0], "options": list(k[1]), "descriptions": list(k[2]), "multi": k[3], **a.params()}
                                for k, a in self.adaptations.items()]}

    @property
    def state_format(self):
        """The serialization of states for this checkpoint: the first of its declared ones that solvi writes ("paths" by
        default; a text-only checkpoint such as L14d reads the "paths" lines as text)."""
        return self.caps["state_format"]

    @property
    def max_len(self):
        """The tokens this checkpoint reads in one sequence (question and input): the encoder's, else the checkpoint's
        `max_len`, else 512."""
        enc = getattr(self.scorer, "enc", None)
        return int(getattr(enc, "max_len", 0) or self.meta.get("max_len") or getattr(self.scorer, "max_len", 0) or 512)

    def count_tokens(self, text):
        """Tokens of a text for this checkpoint: its tokenizer when it has one, else solvi.longdoc.approx_tokens."""
        tok = getattr(getattr(self.scorer, "enc", None), "tok", None)
        if tok is not None:
            return len(tok.encode(text, add_special_tokens=False).ids)
        from .longdoc import approx_tokens
        return approx_tokens(text)

    def text(self, v):
        """An input (a text, a Quote, a scalar or a state) → the text this checkpoint reads."""
        return _text(v, self.caps["state_format"])

    # --- raw logits
    def _pick(self, sp, o, text=None):
        """One scorer output → (the question's logits [K], the act logit or None). From an l14g checkpoint the logits also
        carry the "not stated" logit (`.unknown`) and the decoded pointer (`.pointer`)."""
        act = unk = ptr = why = info = None
        transient = False
        if isinstance(o, dict):
            why, info, transient = o.get("escalate"), o.get("info"), bool(o.get("transient"))
            o, act, unk, ptr = o.get("logits"), o.get("act"), o.get("unknown"), o.get("pointer")
            if o is None and sp.kind == "span":
                o = np.zeros(0)
        lg = np.asarray(o, dtype=np.float64)
        if lg.ndim == 2:
            col = self.caps["columns"].get(self.wire(sp.kind), 0)
            lg = lg[:, col if col < lg.shape[1] else 0]
        if sp.kind == "noul" and lg.shape == (1,):     # a single yes log-odds
            lg = np.array([lg[0], 0.0])
        if lg.shape != (len(sp.real),):
            raise ValueError(f"the scorer returned {lg.shape} logits for {len(sp.real)} options")
        if unk is not None or (ptr is not None and sp.pointer) or why or info:
            lg = lg.view(Logits)
            lg.unknown = None if unk is None else float(unk)
            if ptr is not None and sp.pointer:
                if "spans" in ptr:                    # already decoded by the scorer (quotes it located itself)
                    lg.pointer = {"null": float(ptr.get("null", 0.0)), "spans": list(ptr["spans"])}
                else:
                    pc = self.caps.get("pointer") or {}
                    lg.pointer = decode_pointer(ptr, text or "", pc.get("max_span_tokens", 40),
                                                temperature=self.temperatures.get("span", 1.0))
            lg.escalate, lg.info, lg.transient = (str(why) if why else None), info, transient
        return lg, (None if act is None else float(act))

    def _item(self, sp, text):
        return sp.item(text, self.wire(sp.kind), self.caps["noul_labels"])

    @property
    def block(self):
        """Does this model score in the block layout (declared by the checkpoint, and the scorer has logits_pass)? Then
        every question — alone or with others — is scored in that layout: its answer does not depend on the other
        questions of its pass, and adapt / fit / teach see the same logits as the runtime."""
        mq = self.caps["multi_question"]
        return mq is not None and mq["layout"] == "block" and callable(getattr(self.scorer, "logits_pass", None))

    def _score(self, pairs):
        """[(spec, text)] → scorer outputs, in order: block passes (one per text, at most max_questions each) for a block
        model — questions that do not fit together go one per pass, and an export without the block layout falls back to
        one question per sequence — else one sequence per question. Questions that need the pointer (span answers,
        evidence) always go one per sequence (the full layout: in the block layout the input does not see the question)."""
        if self.block and not self._block_failed and any(sp.pointer for sp, _ in pairs):
            ptr = [i for i, (sp, _) in enumerate(pairs) if sp.pointer]
            rest = [i for i, (sp, _) in enumerate(pairs) if not sp.pointer]
            out, used = [None] * len(pairs), False
            if rest:
                got, used = self._score([pairs[i] for i in rest])
                for i, o in zip(rest, got):
                    out[i] = o
            got = self.scorer.logits([self._item(*pairs[i]) for i in ptr])
            self.passes += len(ptr)
            for i, o in zip(ptr, got):
                out[i] = o
            return out, used
        if self.block and not self._block_failed:
            by_text = OrderedDict()
            for i, (sp, t) in enumerate(pairs):
                by_text.setdefault(t, []).append(i)
            n = self.caps["max_questions"]
            for size in (n, 1):
                chunks = [ix[k:k + size] for ix in by_text.values() for k in range(0, len(ix), size)]
                try:
                    got = self.scorer.logits_pass([Pass(pairs[c[0]][1], tuple(self._item(*pairs[i]) for i in c))
                                                   for c in chunks])
                except BlockUnsupported:
                    self._block_failed = True
                    break
                except ValueError:                    # too long together: smaller passes, then one per sequence
                    continue
                self.passes += len(chunks)
                out = [None] * len(pairs)
                for c, res in zip(chunks, got):
                    for i, o in zip(c, res):
                        out[i] = o
                return out, True
        items = [self._item(*pr) for pr in pairs]
        out = self.scorer.logits(items)
        self.passes += len(items)
        return out, False

    def _raw_full(self, specs_texts, info=None):
        """[(spec, text)] → [(raw logits of the mode's column [K], act logit or None)] (cached by text and question).
        info: a dict that receives "block": whether what was scored now went through block passes."""
        out, todo = [None] * len(specs_texts), []
        with self._lock:
            for i, (sp, t) in enumerate(specs_texts):
                k = (sp.key, t)
                if k in self._cache:
                    self._cache.move_to_end(k)
                    out[i] = self._cache[k]
                else:
                    todo.append(i)
        if todo:
            got, used_block = self._score([specs_texts[i] for i in todo])
            if info is not None:
                info["block"] = used_block
            with self._lock:
                for i, o in zip(todo, got):
                    sp = specs_texts[i][0]
                    out[i] = self._pick(sp, o, specs_texts[i][1])
                    if not getattr(out[i][0], "transient", False):     # a server that did not answer: ask again
                        self._cache[(sp.key, specs_texts[i][1])] = out[i]
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return out

    def _raw(self, specs_texts):
        return [z for z, _ in self._raw_full(specs_texts)]

    def _raw_pass(self, specs, text):
        """Several questions about one text in one forward pass → [(logits, act, shared)]. Block layout: the questions not
        cached yet go in one pass (an answer does not depend on its neighbours, so the cache is per question). Other
        layouts: the pass is cached as a whole; one question per pass (shared False) when the scorer has no logits_pass or
        the questions do not fit together."""
        if self.block:
            info = {"block": not self._block_failed}
            with self._pass_lock:                     # parallel steps of one pass: the first runs it, the others read it
                res = self._raw_full([(sp, text) for sp in specs], info)
            return [(z, a, info["block"]) for z, a in res]
        sig = tuple(sp.key for sp in specs)
        keys = [(sp.key, text, sig) for sp in specs]

        def cached():
            with self._lock:
                if all(k in self._cache for k in keys):
                    for k in keys:
                        self._cache.move_to_end(k)
                    return [self._cache[k] for k in keys]
            return None
        got = cached()
        if got is not None:
            return got
        with self._pass_lock:
            got = cached()
            if got is not None:
                return got
            outs = None
            if self.batchable:
                items = tuple(self._item(sp, text) for sp in specs)
                try:
                    outs = self.scorer.logits_pass([Pass(text, items)])[0]
                    self.passes += 1
                except ValueError:                    # too long together: one question per pass
                    outs = None
            if outs is None:
                res = [(z, a, False) for z, a in self._raw_full([(sp, text) for sp in specs])]
            else:
                res = [(*self._pick(sp, o, text), True) for sp, o in zip(specs, outs)]
            with self._lock:
                for k, v in zip(keys, res):
                    self._cache[k] = v
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            return res

    def logits(self, text, task, options, descriptions=None, multi=False, other=None, kind=None):
        """Raw logits of the scored options ("other" excluded): {option: logit}, or a list of them for a list of inputs."""
        sp = _Spec(task, options, descriptions, multi, other, kind)
        one = _single(text)
        zs = self._raw([(sp, self.text(t)) for t in ([text] if one else list(text))])
        res = [{o: float(v) for o, v in zip(sp.real, z)} for z in zs]
        return res[0] if one else res

    # --- calibrated scores
    def _scores(self, sp, z):
        """Raw logits → calibrated scores s (softmax / sigmoid input) with this question's adaptation."""
        a = self.adaptations.get(sp.key)
        z = np.asarray(z, float)
        if a is not None and a.bias is not None:
            z = z - np.asarray(a.bias)
        scale = a.scale if a is not None and a.scale is not None else 1.0 / self._T(sp)
        shift = np.asarray(a.shift) if a is not None and a.shift is not None else 0.0
        tau = a.temperature if a is not None else 1.0
        return (scale * z + shift) / tau

    def _decision(self, sp, z):
        """Calibrated logits → Decision. Questions of solvi 0.5 without "not stated" decide exactly as before; span, rank and
        number questions and those allowing "not stated" (from an l14g checkpoint's outputs) go through _decision_v3; a
        question asking for evidence gets the pointer's quotes."""
        u, ptr = getattr(z, "unknown", None), getattr(z, "pointer", None)
        if sp.kind == "span":
            d = self._span_decision(sp, ptr)
        else:
            if (sp.unknown and u is not None) or sp.kind in ("rank", "number"):
                d = self._decision_v3(sp, z, u if sp.unknown else None)
            else:
                d = self._decision_v1(sp, z)
            if sp.evidence and ptr is not None and d.value is not Unknown:
                ev = (self.caps.get("pointer") or {}).get("evidence") or {}
                d.evidence = pointer_evidence(ptr, ev.get("threshold", 0.15), min(sp.evidence, ev.get("max_spans", 3)))
        info, why = getattr(z, "info", None), getattr(z, "escalate", None)
        if info:
            d.extra.update(info)
        if why and d.escalate is None:              # the scorer could not give a usable output: never a guess
            d.escalate = f"{ESCALATED}: {why}"
        return d

    def _decision_v3(self, sp, z, u):
        """rank / number, and any kind with "not stated": for single, score, noul, rank, number the "not stated" logit u
        competes with the options in one softmax (the l14g contract), for multi it is a sigmoid. choice / noul: the most
        probable of the options and "not stated"; score / rank / number: "not stated" when p(not stated) ≥ the threshold
        (0.5), else the median level / the order by probability / the median bin, from the probabilities given it is
        stated. Confidence: p(answer) — for a ranking its Plackett–Luce probability, for a number the mass of its interval
        — times p(stated)."""
        thr = float((self.caps.get("unknown") or {}).get("threshold", 0.5))
        if sp.multi:
            d = self._decision_v1(sp, z)
            pu = float(_sig(u))
            if pu >= thr:
                return Decision(Unknown, {**d.probs, Unknown: pu}, confidence=pu)
            d.probs[Unknown] = pu
            d.confidence = min(d.conf, 1 - pu)
            return d
        s = self._scores(sp, z)
        K = len(sp.real)
        q_all = _softmax(np.append(s, u / self._T(sp))) if u is not None else _softmax(s)
        pn = float(q_all[K]) if u is not None else 0.0
        q = q_all[:K]
        cond = q / max(float(q.sum()), 1e-300)
        probs = {o: float(v) for o, v in zip(sp.real, q)}
        if u is not None:
            probs[Unknown] = pn
        if sp.kind in ("choice", "noul"):
            best = int(np.argmax(q_all))
            if best == K:
                return Decision(Unknown, probs, confidence=pn)
            return Decision(sp.out(sp.real[best]), probs, confidence=float(q[best]))
        if u is not None and pn >= thr:
            return Decision(Unknown, probs, confidence=pn)
        if sp.kind == "rank":
            from .primitives import plackett_luce
            order = tuple(sp.real[i] for i in sorted(range(K), key=lambda i: (-cond[i], i))[: sp.k or K])
            conf = plackett_luce(order, dict(zip(sp.real, cond))) * (1 - pn)
            return Decision(order, probs, confidence=conf, extra={"k": len(order)})
        if sp.kind == "number":
            from types import SimpleNamespace

            from .primitives import estimate_of
            value, interval, mass = estimate_of(SimpleNamespace(bins=sp.edges, coverage=sp.coverage), list(cond))
            return Decision(value, probs, confidence=mass * (1 - pn), extra={"interval": interval, "coverage": sp.coverage})
        levels = sp.real                               # score
        acc, med = 0.0, levels[-1]
        for o, p in zip(levels, cond):
            acc += p
            if acc >= 0.5 - 1e-12:
                med = o
                break
        rank = float(np.dot(np.arange(K), cond))
        numeric = all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in levels)
        expected = float(np.dot(np.asarray(levels, float), cond)) if numeric else rank
        value = {"median": med, "mode": levels[int(cond.argmax())], "expected": levels[int(round(rank))]}[sp.score_value]
        return Decision(value, probs, confidence=float(q[levels.index(value)]), extra={"expected": expected, "median": med})

    def _span_decision(self, sp, ptr):
        """The pointer's best span (its text literally from the input), or "not stated" when the null span is at least as
        probable and the question allows it. Confidence: p(span) among the null span and every span (renormalized without
        the null span when "not stated" is not allowed)."""
        if ptr is None:
            return Decision(Quote("", 0, 0, NULL_SOURCE), {}, confidence=0.0,
                            escalate="the checkpoint gave no pointer output for this span question")
        spans, pn = ptr["spans"], float(ptr["null"])
        if sp.unknown and (not spans or pn >= spans[0][0]):
            return Decision(Unknown, {Unknown: pn}, confidence=pn, extra={"p_null": pn})
        if not spans:
            return Decision(Quote("", 0, 0, NULL_SOURCE), {}, confidence=0.0, escalate="the pointer found no span")
        k, mass = _typed_span(spans, sp.vtype)
        p, a, b, t = spans[k]
        p = p if mass is None else mass
        conf = float(p) if sp.unknown else min(1.0, float(p) / max(1e-12, 1 - pn))
        extra = {"p_null": pn}
        if k:
            extra["trimmed"] = spans[0][3]            # the best span did not parse as the type; its part that did
        return Decision(Quote(t, a, b, NULL_SOURCE, conf), {Unknown: pn} if sp.unknown else {}, confidence=conf,
                        extra=extra)

    def _decision_v1(self, sp, z):
        s = self._scores(sp, z)
        a = self.adaptations.get(sp.key)
        if sp.multi:
            p = _sig(s)
            probs = {o: float(v) for o, v in zip(sp.real, p)}
            chosen = tuple(o for o in sp.real if probs[o] >= self.multi_threshold)
            conf = float(np.min(np.maximum(p, 1 - p)))
            if sp.other is not None:
                probs[sp.other] = float(1 - p.max())
                if not chosen:
                    chosen = (sp.other,)
            order = {o: i for i, o in enumerate(sp.options)}
            return Decision(tuple(sorted(chosen, key=order.get)), {o: probs[o] for o in sp.options}, confidence=conf)
        q = _softmax(s)
        if sp.kind == "noul":
            lab = "yes" if q[0] >= 0.5 else "no"
            return Decision(sp.out(lab), {"yes": float(q[0]), "no": float(q[1])}, confidence=float(q.max()))
        if sp.kind == "score":
            levels = sp.real
            acc, med = 0.0, levels[-1]
            for o, p in zip(levels, q):
                acc += p
                if acc >= 0.5 - 1e-12:
                    med = o
                    break
            rank = float(np.dot(np.arange(len(levels)), q))
            numeric = all(isinstance(o, (int, float)) and not isinstance(o, bool) for o in levels)
            expected = float(np.dot(np.asarray(levels, float), q)) if numeric else rank
            value = {"median": med, "mode": levels[int(q.argmax())], "expected": levels[int(round(rank))]}[sp.score_value]
            return Decision(value, {o: float(v) for o, v in zip(levels, q)}, confidence=float(q[levels.index(value)]),
                            extra={"expected": expected, "median": med})
        best = int(q.argmax())
        m = float(q[best])
        if sp.other is None:
            return Decision(sp.real[best], {o: float(v) for o, v in zip(sp.real, q)})
        thr = a.other_threshold if a is not None and a.other_threshold is not None else self.other_threshold
        thr = min(max(thr, 1e-6), 1 - 1e-6)
        g = thr * (1 - m) / (1 - thr)            # other's odds: other is the most probable option exactly when m < thr
        pi = g / (1 + g)
        probs = {o: float((1 - pi) * v) for o, v in zip(sp.real, q)}
        probs[sp.other] = float(pi)
        probs = {o: probs[o] for o in sp.options}
        if m < thr:
            return Decision(sp.other, probs, confidence=1 - m)
        return Decision(sp.real[best], probs, confidence=m)

    def act_probability(self, sp, d, act):
        """The probability that the model's answer is right, from its act logit: sigmoid(logit / T_act), or the checkpoint's
        act calibrator — a logistic regression over ACT_FEATURES of the calibrated decision (see docs/decide_format.md)."""
        cal = (self.caps["act"] or {}).get("calibrator")
        if not cal:
            return float(_sig(act / self.temperatures["act"]))
        x = act_features(sp, d, act)
        return float(_sig(float(cal.get("bias", 0.0)) + sum(float(w) * x[f] for f, w in zip(cal["features"], cal["weights"]))))

    def _finish(self, sp, d, act, escalate_below=None, act_threshold=None, use_act=None):
        """Act or escalate: the model's act signal (its probability is recorded; below the threshold → "model escalated"),
        else the calibrated confidence below escalate_below (→ "confidence … < …", the low-confidence safeguard)."""
        if act is not None:
            p = self.act_probability(sp, d, act)
            d.extra["act"] = p
            thr = self.act_threshold if act_threshold is None else act_threshold
            if use_act is not False and p < thr:
                d.escalate = f"{ESCALATED}: act {p:.2f} < {thr:.2f}; would have answered {d.value!r}"
        eb = self.escalate_below if escalate_below is None else escalate_below
        if d.escalate is None and eb is not None and d.conf < eb:
            d.escalate = f"confidence {d.conf:.2f} < {eb:.2f} (escalate_below); would have answered {d.value!r}"
        return d

    def decide(self, text, task, options, descriptions=None, multi=False, other=None, kind=None, escalate_below=None,
               **spec):
        """→ Decision(value, probs) (a list of them for a list of inputs). The value is always one of the options; a text
        or a state (dict, list, pydantic model: see state_text). `spec`: unknown=, k=, bins=, unit=, coverage=,
        evidence= (see decision)."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        one = _single(text)
        texts = [text] if one else list(text)
        raws = self._raw_full([(sp, self.text(t)) for t in texts])
        out = [self._finish(sp, self._decision(sp, z), a, escalate_below) for z, a in raws]
        return out[0] if one else out

    def score(self, text, task, options, descriptions=None, multi=False, other=None, kind=None):
        """→ {option: probability} (a list of them for a list of inputs): softmax over the options for a single choice, a
        score or yes/no, a sigmoid per option with multi=True; the adaptation of this question applied, "other" by its
        threshold."""
        d = self.decide(text, task, options, descriptions, multi, other, kind)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    def decide_pass(self, text, parts, names=None):
        """Several decision parts about one input, scored together → [Decision] (one forward pass when the checkpoint
        supports it, else one per question). What the runtime does for parts grouped in `flow.batches`."""
        parts = list(parts)
        t = self.text(text)
        if len(parts) > 1 and self.batchable and all(p.option_order != "average" for p in parts):
            raws = self._raw_pass([p.spec for p in parts], t)
        else:
            raws = [(*p._raw([t])[0], False) for p in parts]
        names = list(names) if names else [p.__name__ for p in parts]
        out = []
        for p, (z, a, shared) in zip(parts, raws):
            d = p._finish(self._decision(p.spec, z), a)
            d.extra["pass"] = {"with": names, "shared": shared}
            out.append(d)
        return out

    # --- adaptation
    def adaptation(self, task, options, descriptions=None, multi=False, other=None, create=False, kind=None):
        sp = _Spec(task, options, descriptions, multi, other, kind)
        if create and sp.key not in self.adaptations:
            self.adaptations[sp.key] = Adaptation()
        return self.adaptations.get(sp.key)

    def adapt(self, texts, task, options, descriptions=None, multi=False, other=None, kind=None, logits=None, **spec):
        """Label-bias correction without labels: the mean logit of each option over unlabelled inputs of the domain
        (centered over the options) is subtracted before the softmax / sigmoid. → the Adaptation. A later fit is kept
        consistent. logits: the inputs' logits already computed (one per text, in the question's option order) — what a
        DecisionPart passes, so the correction is fitted on the signal it decides on (option_order="average", long)."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        if sp.kind == "span":
            raise ValueError("a span question has no options to adapt")
        texts = [self.text(t) for t in texts]
        if not texts:
            raise ValueError("adapt needs unlabelled texts")
        Z = np.array(self._raw([(sp, t) for t in texts]) if logits is None else _given(logits, len(texts)))
        mean = Z.mean(0)
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.bias = [float(v) for v in mean - mean.mean()]
        a.n_unlabelled = len(texts)
        if a.examples:
            self._refit(sp, a)
        return a

    def fit(self, examples, task, options, descriptions=None, multi=False, other=None, lam=1.0, folds=4, kind=None,
            logits=None, **spec):
        """Few-shot adaptation "S" from labelled examples [(input, correct)]: a shift and a shared scale on the
        (bias-corrected) logits by L-BFGS (per option; a score: a tilt and a spread over the levels; yes/no: one bias), a
        temperature on out-of-fold predictions, and — when some examples are labelled "other" — the "other" threshold that
        maximizes out-of-fold accuracy. Replaces earlier examples. Examples labelled "not stated" are left out (the "not
        stated" logit is not adapted). logits: the examples' logits already computed (one per example; see adapt).
        → the Adaptation."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        examples = list(examples)
        given = None if logits is None else _given(logits, len(examples))
        keep = [i for i, (_, y) in enumerate(examples) if y is not Unknown]
        ex = [(self.text(examples[i][0]), examples[i][1]) for i in keep]
        Z = self._raw([(sp, t) for t, _ in ex]) if given is None else [given[i] for i in keep]
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.examples = [(list(map(float, z)), sp.label(y)) for z, (_, y) in zip(Z, ex)]
        self._refit(sp, a, lam, folds)
        return a

    def teach(self, text, correct, task, options, descriptions=None, multi=False, other=None, lam=1.0, kind=None,
              logits=None, **spec):
        """One labelled example, absorbed at once: the shift / scale is refitted from the kept examples (warm start, K + 1
        parameters or fewer — about a millisecond or two); the temperature and the "other" threshold stay until the next
        fit. logits: the input's logits already computed (see adapt). → the update time in ms (the model's forward pass,
        if the input was not scored before, is not included)."""
        sp = _Spec(task, options, descriptions, multi, other, kind, **spec)
        z = self._raw([(sp, self.text(text))])[0] if logits is None else _given([logits], 1)[0]
        t0 = time.perf_counter()
        a = self.adaptations.setdefault(sp.key, Adaptation())
        a.examples.append((list(map(float, z)), sp.label(correct)))
        self._refit(sp, a, lam, folds=0, warm=True)
        return (time.perf_counter() - t0) * 1000

    def _refit(self, sp, a, lam=1.0, folds=4, warm=False):
        a0 = 1.0 / self._T(sp)
        basis = _basis(sp)
        bias = np.asarray(a.bias) if a.bias is not None else 0.0
        if sp.multi:
            rows = [(np.asarray(z) - bias, y) for z, y in a.examples]
            Z = np.array([z for z, _ in rows])
            Y = np.array([[float(o in y) for o in sp.real] for _, y in rows])
            real_idx = list(range(len(rows)))
        else:
            rows = [(np.asarray(z) - bias, y) for z, y in a.examples]
            real_idx = [i for i, (_, y) in enumerate(rows) if y != sp.other]
            Z = np.array([rows[i][0] for i in real_idx]) if real_idx else np.zeros((0, len(sp.real)))
            Y = np.array([sp.real.index(rows[i][1]) for i in real_idx], dtype=int)
        a.n_labelled = len(a.examples)
        if len(real_idx) == 0:
            return
        init = None
        if warm and a.scale is not None and a.shift is not None:
            init = np.concatenate([[a.scale], np.asarray(a.shift)])
        a.scale, b = _fit_shift(Z, Y, sp.multi, a0, lam, iters=60 if warm else 300, init=init, basis=basis)
        a.shift = [float(v) for v in b]
        if warm:
            return
        # temperature and "other" threshold from out-of-fold predictions
        n = len(rows)
        fold = np.arange(n) % folds if folds and n >= 2 * folds else None
        oof = []                                  # (index into rows, scores before temperature)
        if fold is not None:
            for f in range(folds):
                tr = [j for j, i in enumerate(real_idx) if fold[i] != f]
                if len(tr) < 2:
                    continue
                af, bf = _fit_shift(Z[tr], Y[tr], sp.multi, a0, lam, basis=basis)
                for i in range(n):
                    if fold[i] == f:
                        oof.append((i, af * rows[i][0] + bf))
        a.temperature = 1.0
        if sp.multi:
            pairs = [(s, np.array([float(o in rows[i][1]) for o in sp.real])) for i, s in oof]
            if len(pairs) >= 4:
                a.temperature = _fit_temperature(pairs, True)
            return
        pairs = [(s, None if rows[i][1] == sp.other else sp.real.index(rows[i][1])) for i, s in oof]
        if sum(y is not None for _, y in pairs) >= 4:
            a.temperature = _fit_temperature(pairs, False)
        labels = [rows[i][1] for i, _ in oof]
        n_other = sum(y == sp.other for y in labels)
        if sp.other is not None and n_other >= 3 and len(labels) - n_other >= 3:
            # the threshold with the best out-of-fold accuracy; among equals, the one closest to the model's default
            ms = [(float(_softmax(s / a.temperature).max()), sp.real[int(np.argmax(s))], rows[i][1]) for i, s in oof]
            grid = sorted({0.0, *np.round(np.linspace(0.05, 0.95, 19), 2)})
            accs = [(float(np.mean([(sp.other if m < thr else p) == y for m, p, y in ms])), float(thr)) for thr in grid]
            top = max(acc for acc, _ in accs)
            a.other_threshold = min((t for acc, t in accs if acc >= top - 1e-12), key=lambda t: abs(t - self.other_threshold))

    def reset(self, task=None, options=None, descriptions=None, multi=False, other=None, kind=None, **spec):
        """Forget the adaptation of one question, or every adaptation."""
        if task is None:
            self.adaptations.clear()
        else:
            self.adaptations.pop(_Spec(task, options, descriptions, multi, other, kind, **spec).key, None)

    def save_adaptations(self, path):
        """Write every adaptation (with its kept examples' logits) to a JSON file, with the checkpoint's fingerprint."""
        data = {"weights": self.weights_fingerprint(), "model_id": self.model_id,
                "adaptations": [{"key": [k[0], list(k[1]), list(k[2]), k[3]], **asdict(a)} for k, a in self.adaptations.items()]}
        with open(path, "w") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1, default=_json_default)

    def load_adaptations(self, path, strict=True):
        """Read adaptations written by save_adaptations. strict: refuse them if the checkpoint differs (the bias and shift
        were fitted on another model's logits)."""
        data = json.load(open(path))
        if strict and data.get("weights") != self.weights_fingerprint():
            raise ValueError(f"adaptations were fitted on checkpoint #{data.get('weights')}, this is #{self.weights_fingerprint()}")
        for d in data["adaptations"]:
            k = d.pop("key")
            ex = [(list(z), tuple(y) if isinstance(y, list) else y) for z, y in d.pop("examples", [])]
            self.adaptations[(k[0], tuple(k[1]), tuple(k[2]), k[3] if isinstance(k[3], str) else bool(k[3]))] = \
                Adaptation(**d, examples=ex)
        return self

    # --- catalog parts
    def decision(self, name, task, text_fact="doc", options=(), descriptions=None, multi=False, other=None, *, kind=None,
                 type=None, escalate_below=None, act_threshold=None, use_act=None, target_error=None,
                 score_value="median", unknown=False, k=None, bins=None, unit=None, coverage=0.8, evidence=False,
                 option_order="canonical", permutations=4, min_margin=None, long=None, top_k=3, rerank=False,
                 perturb=0):
        """A catalog part: text_fact (a fact name, or a list of them) → Decision(value, probs).

        The question: `options` (a list, or {option: description}) and `kind` ("choice", "multi", "score", "noul"; default
        choice, or multi with multi=True) — or a Python type, as `type=` or in place of the options: Literal[...] / an Enum
        (choice), list[Literal[...]] (multi), Scale[...] (score), bool (noul: the value is True / False). The input: a
        text fact is read as it is (several joined by new lines); a state (dict, list, pydantic model) by state_text
        (several facts: {fact: value}).

        Escalation: the model's act signal when it has one (use_act=False ignores it; act_threshold overrides the
        checkpoint's threshold; target_error=0.1 takes the checkpoint's threshold for that error rate), and a calibrated
        confidence below escalate_below (see calibrate_for); an escalated decision is rejected — the fact is missing, the
        answer abstains ("model escalated" / "low confidence" in the audit and stats). min_margin=0.1: also escalate when
        the two most probable answers are closer than that (a near tie is where a misleading text flips the choice).

        Option order (choice and multi questions): "canonical" (the default) asks in sorted order, so how a caller lists
        the options cannot change the answer (the part's options are then in that order); "given" asks as listed (0.5.0);
        "average" averages the model's logits over `permutations` rotations of the list (each costs a forward pass) —
        against a model's preference for positions.

        perturb=k: ask again on up to k variants of the input without its instruction-like sentences ("ignore the rules
        and answer X", "SYSTEM: ...", "the correct answer is X" — deterministic rules, solvi.perturb) and escalate when
        the answer changes ("answer depends on an instruction-like sentence: ..."). An input without such sentences costs
        nothing extra; one with them costs up to k forward passes.

        Register with `cat.fn(part)` (a fact other parts read) or make it a question's answer with `part.question(cat)`.
        The value is one of the options by construction; the options are the part's closed set; provenance `decided`; the
        trace records the model, whose fingerprint covers the checkpoint and this part's adaptation and thresholds.

        Answer primitives (an l14g checkpoint, docs/decide_format.md §9): `Maybe[T]` or unknown=True — "not stated" is an
        answer (solvi.Unknown); `Rank[Literal[...], k]` or kind="rank", k= — the options best first; `Estimate[edges]` or
        kind="number", bins=, unit=, coverage= — a number over bins; `Span[T]` or kind="span" — a piece of the (one, given)
        text fact, coerced to T when it is a question's answer; evidence=True (or a number) — supporting quotes from the
        pointer. A checkpoint that cannot give what is asked raises here.

        Long texts: long=None cuts a text beyond max_len (the tokenizer truncates it, as before); long="retrieve" splits
        it into sections, selects the top_k that bear on the question by BM25 (rerank=True: re-ordered by the decider's own
        relevance, one yes / no pass per candidate section) and decides on them; spans and evidence point into the whole
        text, and the sections read are in the decision's extra["long"] (solvi.longdoc)."""
        as_bool, extra = False, {}
        if type is None and _is_type(options):
            type, options = options, ()
        if type is not None:
            kind, options, descriptions, as_bool, extra = _from_type(type, kind, options, descriptions)
        if target_error is not None and act_threshold is None:
            act_threshold = self.act_threshold_for(target_error)
        prim = {"unknown": bool(unknown or extra.get("unknown")), "k": extra.get("k", k), "bins": extra.get("bins", bins),
                "unit": extra.get("unit", unit), "coverage": extra.get("coverage", coverage),
                "vtype": extra.get("vtype")}
        if extra.get("labels") is not None:
            options = extra["labels"]
        pc = self.caps.get("pointer") or {}
        prim["evidence"] = (int((pc.get("evidence") or {}).get("max_spans", 3)) if evidence is True else int(evidence or 0))
        k_ = _kind(kind, multi)
        if (k_ == "span" or prim["evidence"]) and not self.has_pointer:
            raise ValueError(f"{name}: this checkpoint cannot point at its input (span answers, evidence): it declares no "
                             "'pointer' / 'span' mode (docs/decide_format.md §9)")
        if prim["unknown"] and not self.has_unknown:
            raise ValueError(f"{name}: this checkpoint has no 'not stated' output (declare 'unknown', docs/decide_format.md "
                             "§9); drop Maybe[...] / unknown=True")
        if option_order not in ("given", "canonical", "average"):
            raise ValueError('option_order must be "given", "canonical" or "average"')
        if long not in (None, "retrieve"):
            raise ValueError('long must be None (truncate) or "retrieve"')
        return DecisionPart(self, name, task, text_fact, options, descriptions, multi, other, kind=kind, as_bool=as_bool,
                            escalate_below=escalate_below, act_threshold=act_threshold, use_act=use_act,
                            option_order=option_order, permutations=permutations, min_margin=min_margin,
                            long=long, top_k=top_k, rerank=rerank, perturb=perturb,
                            score_value=score_value, **{x: v for x, v in prim.items() if v not in (None, False, 0)
                                                         or x == "coverage"})

    def decisions(self, schema, text_fact="doc", fields=None, **kw):
        """One decision part per field of a pydantic model class: the field's type is the question (bool, Literal[...],
        an Enum, Scale[...], list[Literal[...]]), its description the task (else its title, else its name), and
        `json_schema_extra` may carry "options" ({option: description}), "escalate_below", "act_threshold", "use_act",
        "target_error", "other", "score_value". → {field: DecisionPart} in field order."""
        import typing

        from .typed import Bins, Ordinal, RankOf, SpanOf
        out = {}
        for name, fi in schema.model_fields.items():
            if fields is not None and name not in fields:
                continue
            t = fi.annotation
            marks = [m for m in fi.metadata if isinstance(m, (Ordinal, SpanOf, RankOf, Bins))]
            if marks:                                  # pydantic moves Annotated metadata to the field: put solvi's back
                t = typing.Annotated[(t, *marks)]
            extra = fi.json_schema_extra if isinstance(fi.json_schema_extra, dict) else {}
            args = dict(kw)
            for k in ("escalate_below", "act_threshold", "use_act", "target_error", "other", "score_value", "kind",
                      "evidence", "coverage", "unit"):
                if k in extra:
                    args[k] = extra[k]
            task = fi.description or fi.title or name.replace("_", " ").capitalize() + "?"
            out[name] = self.decision(name, task, text_fact, options=extra.get("options") or (), type=t, **args)
        return out

    def questions(self, cat, schema, text_fact="doc", fields=None, min_confidence=None, **kw):
        """decisions(...) registered as the answers of questions named after the fields → [Question]."""
        return [p.question(cat, min_confidence=min_confidence) for p in self.decisions(schema, text_fact, fields, **kw).values()]


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return repr(o)


class Facts(dict):
    """An example input given as facts by name (`Facts(email=..., tier=...)`): each part reads its own facts, a route's
    predicates and a grouping (act_guard(groups=...)) read theirs. Any other input is the one input every part reads (a
    text or a state)."""


class GroupBy:
    """Which group an input belongs to, for thresholds per group (act_guard(groups=...)): a fact name ("domain"), a list of
    fact names — a hierarchy, top first (["domain", "task"]) — or a function whose parameters are fact names and which
    returns a group or a path (domain, task). A function with one parameter also takes an input that is not given as
    facts (a text or a state): it is called with the input itself. A state (dict) input gives facts by its keys."""

    def __init__(self, by):
        if isinstance(by, str):
            self.names, self.fn = [by], None
        elif isinstance(by, (list, tuple)) and by and all(isinstance(x, str) for x in by):
            self.names, self.fn = list(by), None
        elif callable(by):
            self.names, self.fn = list(inspect.signature(by).parameters), by
        else:
            raise TypeError(f"groups: a fact name, a list of fact names or a function, not {by!r}")
        self.by = by

    def describe(self):
        from .provenance import code_fingerprint
        return list(self.names) if self.fn is None else {"fn": code_fingerprint(self.fn), "reads": self.names}

    def label(self):
        return " → ".join(self.names) if self.fn is None else getattr(self.fn, "__name__", "a function")

    def path(self, vals=None, raw=None):
        """The input's group path (a tuple), or None when the input does not give it."""
        from .calibration import group_path
        if vals is None:
            if isinstance(raw, Mapping) and all(n in raw for n in self.names):
                vals = raw
            elif self.fn is not None and len(self.names) == 1:
                return group_path(self.fn(raw))
            else:
                return None
        if any(n not in vals for n in self.names):
            return None
        args = {n: (vals[n].value if isinstance(vals[n], Quote) else vals[n]) for n in self.names}
        return group_path(self.fn(**args) if self.fn is not None else tuple(args[n] for n in self.names))


def group_name(path):
    """A group path as people read it: "billing / refunds"; the whole stream: "(the rest of the stream)"."""
    return " / ".join(path) if path else "(the rest of the stream)"


def group_record(g, path, node, info):
    """A decision's guarantee under thresholds per group: the part's promise, the input's group, the group whose
    threshold applied (its own, or a parent's when the group had too few examples), that threshold and its examples."""
    pooled = tuple(node) != tuple(path)
    out = dict(g, group=list(path), applied=list(node), threshold=info["threshold"], n=info["n"])
    out["promise"] = (f"{g['promise']}; here: group {group_name(node)} (threshold {info['threshold']:.4g}, n = "
                      f"{info['n']})" + (f", pooled: {group_name(path)} had fewer than {g['min_group']} examples"
                                         if pooled else ""))
    return out


class DecisionPart:
    """A decider bound to one question (name, task, options, kind, the facts it reads). Callable as a catalog function; it
    is also the model recorded in the trace: `fingerprint()` covers the checkpoint and this question's adaptation and
    thresholds only, so teaching one decision does not mark the others as changed."""

    def __init__(self, model, name, task, text_fact, options, descriptions=None, multi=False, other=None, *, kind=None,
                 as_bool=False, escalate_below=None, act_threshold=None, use_act=None, score_value="median",
                 option_order="canonical", permutations=4, min_margin=None, long=None, top_k=3, rerank=False,
                 perturb=0, **prim):
        self.model = model
        self.long, self.top_k, self.rerank = long, max(1, int(top_k)), bool(rerank)
        self.spec = _Spec(task, options, descriptions, multi, other, kind, as_bool, score_value, **prim)
        sp = self.spec
        self._shown = None if sp.values is None else list(sp.values)   # the caller's order, for options and probs
        self._shown_labels = list(sp.options)
        if option_order == "canonical" and sp.kind in ("choice", "multi") and len(sp.real) > 1:
            ordered = sorted(sp.real, key=str) + ([sp.other] if sp.other is not None else [])
            if ordered != list(sp.options):         # the part asks, adapts and answers in the sorted order
                options, descriptions, other = ordered, dict(sp.descriptions), (sp.other if sp.other is not None else False)
                self.spec = _Spec(task, options, descriptions, multi, other, kind, as_bool, score_value, **prim)
        self._spec_args = (task, descriptions, multi, other, kind, as_bool, score_value, prim)
        if self.spec.kind not in ("choice", "multi") or len(self.spec.real) < 2:
            option_order = "given"                  # scores, numbers, rankings: the order is the meaning
        self.option_order, self.permutations, self.min_margin = option_order, max(1, int(permutations)), min_margin
        self.perturb = max(0, int(perturb or 0))  # re-ask without instruction-like sentences (solvi.perturb), up to k times
        self._orders = self._option_orders()
        self.facts = [text_fact] if isinstance(text_fact, str) else list(text_fact)
        self.escalate_below, self.act_threshold, self.use_act = escalate_below, act_threshold, use_act
        self.guarantee = None                   # what the escalation threshold promises (act_guard / calibrate_for)
        self.conformal_set = None               # the answer-set quantile (conformal)
        self.groups = None                      # thresholds per group (act_guard(groups=...)): {"by", "nodes", "signal"}
        self.correction_memory = None           # a solvi.memory.CorrectionMemory consulted on every decision (memory())
        self.__name__ = name
        self.__qualname__ = name
        self.__doc__ = task
        self.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for f in self.facts])
        self.__solvi_model__ = self
        self.__solvi_provenance__ = "decided"
        self.__solvi_options__ = None if self._shown is None else list(self._shown)
        self.__solvi_decision__ = self

    # identity recorded in the trace
    @property
    def model_id(self):
        return self.model.model_id

    @property
    def deterministic(self):
        """Does the model give the same output for the same input (replay re-runs it)? False for an LLM (solvi.llm):
        replay then checks the recorded output instead."""
        return getattr(self.model, "deterministic", True)

    @property
    def available(self):
        return getattr(self.model, "available", True)

    @property
    def options(self):
        """The values this decision can take (a bool question: True, False; with "not stated": also Unknown; None for a
        number or a span)."""
        return None if self._shown is None else list(self._shown)

    @property
    def labels(self):
        """The options as the model reads them (a bool question: "yes", "no")."""
        return list(self.spec.options)

    @property
    def kind(self):
        return self.spec.kind

    @property
    def multi(self):
        return self.spec.multi

    @property
    def task(self):
        return self.spec.task

    @property
    def adaptation(self):
        return self.model.adaptations.get(self.spec.key)

    def fingerprint(self):
        from .provenance import digest
        a = self.adaptation
        th = {k: v for k, v in (("escalate_below", self.escalate_below), ("act_threshold", self.act_threshold),
                                ("use_act", self.use_act), ("guarantee", self.guarantee),
                                ("conformal", self.conformal_set), ("min_margin", self.min_margin),
                                ("perturb", self.perturb or None),
                                ("groups", None if self.groups is None else
                                 (self.groups["by"].describe(), sorted((list(k), v["threshold"])
                                                                       for k, v in self.groups["nodes"].items()))),
                                ("option_order", None if self.option_order != "average" else
                                 (self.option_order, self.permutations)),
                                ("long", None if self.long is None else (self.long, self.top_k, self.rerank)),
                                ("memory", None if self.correction_memory is None else self.correction_memory.fingerprint()))
              if v is not None}
        if th:
            return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None, th)
        return digest("DecisionPart", self.model.weights_fingerprint(), self.spec.describe(), a.params() if a else None)

    def text_of(self, facts):
        """The input this part reads, from a dict of facts: text facts as they are (joined by new lines), a state by
        state_text (several facts: {fact: value})."""
        vals = [facts[f] for f in self.facts]
        if all(_is_text(v) for v in vals):
            return "\n".join(_text(v) for v in vals)
        if len(vals) == 1:
            return self.model.text(vals[0])
        return state_text({f: (v.value if isinstance(v, Quote) else v) for f, v in zip(self.facts, vals)},
                          self.model.state_format)

    def __call__(self, *args, **kw):
        vals = dict(zip(self.__signature__.parameters, args))
        vals.update(kw)
        text = self.text_of(vals)
        return self._bind(self._one(text, self._ctx(text, vals=vals)), vals)

    def _ctx(self, text, vals=None, raw=None):
        """What _finish needs besides the model's output: the input's text and, with thresholds per group, its group."""
        c = {"text": text}
        if self.groups is not None:
            c["path"] = self.groups["by"].path(vals, raw)
        return c

    def _input_text(self, x):
        return self.text_of(x) if isinstance(x, Facts) else self.model.text(x)

    def _bind(self, d, vals):
        """Point a span's / the evidence's quotes into the fact they were read from: a decision reading one given text fact
        (the pointer's offsets are into that text); otherwise a span escalates and evidence is dropped."""
        if not (isinstance(d.value, Quote) or d.evidence):
            return d
        f = self.facts[0] if len(self.facts) == 1 else None
        v = vals.get(f) if f is not None else None
        if not isinstance(v, str):
            d.evidence = []
            if isinstance(d.value, Quote) and d.escalate is None:
                d.escalate = f"a span is read from one text fact; {self.__name__} reads {self.facts}"
            return d
        if isinstance(d.value, Quote):
            d.value = dataclasses.replace(d.value, source=f)
        d.evidence = [dataclasses.replace(e, source=f) for e in d.evidence]
        return d

    def in_pass(self, siblings, args, names=None):
        """Called by the runtime for a part in a shared pass (flow.batches): score every sibling's question about the same
        input in one forward pass (cached, so the siblings read it) and return this part's decision; the decision's extra
        records the pass."""
        text = self.text_of(args)
        ctx = self._ctx(text, vals=args)
        m = self.model
        if self.long is not None and self._too_long(text):   # a long text: this part retrieves and decides on its own
            d = self._bind(self._one(text, ctx), args)
            d.extra["pass"] = {"with": list(names) if names else [s.__name__ for s in siblings], "shared": False}
            return d
        if (len(siblings) > 1 and m.batchable and all(s.model is m for s in siblings)
                and all(s.option_order != "average" for s in siblings)):
            z, a, shared = m._raw_pass([s.spec for s in siblings], text)[siblings.index(self)]
        else:
            (z, a), shared = self._raw([text])[0], False
        d = self._bind(self._finish(m._decision(self.spec, z), a, ctx=ctx), args)
        d.extra["pass"] = {"with": list(names) if names else [s.__name__ for s in siblings], "shared": shared}
        return d

    def __repr__(self):
        return f"DecisionPart({self.__name__!r}, {self.spec.kind}, options={self.options}, model={self.model.model_id!r})"

    def _group_threshold(self, ctx):
        """With thresholds per group: (the input's group path, the node whose threshold applies, its info) — None when
        the input does not say which group it is in."""
        path = (ctx or {}).get("path")
        if path is None:
            return None
        from .calibration import node_of
        node = node_of(path, self.groups["nodes"])
        return path, node, self.groups["nodes"][node]

    def _finish(self, d, act, threshold=None, ctx=None):
        """Act or escalate. threshold: a combination's shared threshold (solvi.multi) in place of this part's own
        act_threshold / escalate_below — on the part's signal (see _signal); −inf applies only the other safeguards.
        ctx: the input (see _ctx) — with thresholds per group, the threshold of the input's group applies."""
        grp = None
        if threshold is None:
            eb, at = self.escalate_below, self.act_threshold
            if self.groups is not None:
                grp = self._group_threshold(ctx)
                if grp is not None and self.groups["signal"] == "act":
                    at = grp[2]["threshold"]
                elif grp is not None:
                    eb = grp[2]["threshold"]
            d = self.model._finish(self.spec, d, act, eb, at, self.use_act)
            if self.groups is not None and grp is None:          # no group, no threshold that holds for it
                d.escalate = (f"group unknown: the thresholds are per group ({self.groups['by'].label()}) and this input "
                              f"does not give {self.groups['by'].names}; would have answered {d.value!r}")
        else:
            d = self.model._finish(self.spec, d, act, -math.inf, -math.inf, self.use_act)
        if self.option_order == "canonical" and d.probs:       # probabilities in the caller's order of the options
            at = {o: n for n, o in enumerate(self._shown_labels)}
            d.probs = dict(sorted(d.probs.items(), key=lambda kv: at.get(kv[0], len(at))))
            if self.spec.multi and isinstance(d.value, tuple):
                d.value = tuple(sorted(d.value, key=lambda o: at.get(o, len(at))))
        if self.min_margin is not None and not self.spec.multi and len(d.probs) > 1:
            (a1, p1), (a2, p2) = sorted(d.probs.items(), key=lambda kv: -kv[1])[:2]
            d.extra["margin"] = p1 - p2
            if d.escalate is None and p1 - p2 < self.min_margin:
                d.escalate = (f"margin {p1 - p2:.2f} < {self.min_margin:g} between {a1!r} ({p1:.2f}) and {a2!r} ({p2:.2f}); "
                              f"would have answered {d.value!r}")
        if threshold is not None:
            name, s = self._signal(d)
            if d.escalate is None and s < threshold:
                d.escalate = (f"{ESCALATED}: " if name == "act" else "") + \
                    f"{name} {s:.2f} < {threshold:.2f} (shared threshold); would have answered {d.value!r}"
        elif self.guarantee is not None:
            d.extra["guarantee"] = dict(self.guarantee) if grp is None else group_record(self.guarantee, *grp)
        if self.perturb and d.escalate is None and (ctx or {}).get("text") is not None:
            self._perturbed(d, ctx["text"])
        if self.correction_memory is not None and (ctx or {}).get("text") is not None:
            self.correction_memory.apply(d, ctx["text"], alone=threshold is None and not ctx.get("combined"))
        if self.conformal_set is not None and d.probs:
            cands = self.candidates(d)
            d.extra["candidates"] = cands
            if d.escalate:
                d.escalate += f"; candidates at {self.conformal_set['coverage']:.0%}: {cands!r}"
        return d

    def _perturbed(self, d, text):
        """The perturb=k safeguard: ask again on up to k variants of the input without its instruction-like sentences
        (solvi.perturb.variants — deterministic rules); when an answer differs, escalate. Records extra["perturb"]:
        {"variants", "calls" (extra forward passes), "removed" (per variant), "answers", "flipped"}. An input without
        such sentences has no variants and costs nothing."""
        from .perturb import variants
        vs = variants(text, self.perturb)
        if not vs:
            return d
        outs = self._read([v.text for v in vs])
        base, flip, answers = _vkey(d.value), None, []
        for v, (z, _) in zip(vs, outs):
            val = self.model._decision(self.spec, z).value
            answers.append(val)
            if flip is None and _vkey(val) != base:
                flip = (v, val)
        d.extra["perturb"] = {"variants": len(vs), "calls": len(vs) * len(self._orders if self.option_order == "average"
                                                                            else [0]),
                              "removed": [v.removed for v in vs], "answers": [jsonable(_shown(a)) for a in answers],
                              "flipped": flip is not None}
        if flip is not None:
            d.escalate = (f"{INSTRUCTION}: " + "; ".join(repr(r) for r in flip[0].removed)
                          + f" (without it: {_shown(flip[1])!r}); would have answered {_shown(d.value)!r}")
        return d

    def _signal(self, d):
        """The signal a threshold applies to for a finished decision → ("act", the act probability) when the model gave
        one and the part uses it, else ("confidence", the calibrated confidence) — as act_guard's signal="auto"."""
        if d.extra.get("act") is not None and self.use_act is not False:
            return "act", float(d.extra["act"])
        return "confidence", float(d.conf)

    def candidates(self, d):
        """The conformal answer set of a decision (after conformal(...)): the answers that cannot be ruled out at the
        calibrated coverage, most probable first — a short list for the person who handles an escalation."""
        from .calibration import set_scores
        keys = list(d.probs)
        s = set_scores([d.probs[k] for k in keys], self.conformal_set["ordinal"], Unknown in keys)
        keep = sorted((i for i in range(len(keys)) if s[i] <= self.conformal_set["quantile"]),
                      key=lambda i: -d.probs[keys[i]])
        return [keys[i] if keys[i] is Unknown else self.spec.out(keys[i]) for i in keep]

    def _option_orders(self):
        """The option lists the model is asked with: [the given one], [sorted], or `permutations` rotations."""
        real = list(self.spec.real)
        if self.option_order == "canonical":
            return [sorted(real, key=str)]
        if self.option_order == "average":
            k, n = len(real), min(self.permutations, len(real))
            return [real[s:] + real[:s] for s in sorted({(i * k) // n for i in range(n)})]
        return [real]

    def _raw(self, texts):
        """[text] → [(logits in the part's option order, act logit)], asked with each of the part's option orders and
        averaged (the act logit too); the question's adaptation applies afterwards, to the part's own spec."""
        if self.option_order != "average":           # given, or canonical (the spec itself is in sorted order)
            return self.model._raw_full([(self.spec, t) for t in texts])
        task, desc, multi, other, kind, as_bool, sv, prim = self._spec_args
        opts = (lambda o: o + [self.spec.other] if self.spec.other is not None and self.spec.other not in o else o)
        specs = [_Spec(task, opts(list(o)), desc, multi, other, kind, as_bool, sv, **prim) for o in self._orders]
        idx = {o: i for i, o in enumerate(self.spec.real)}
        runs = [self.model._raw_full([(sp, t) for t in texts]) for sp in specs]
        out = []
        for j in range(len(texts)):
            zs, us, acts = [], [], []
            for sp, run in zip(specs, runs):
                z, a = run[j]
                back = np.empty(len(self.spec.real))
                for pos, o in enumerate(sp.real):
                    back[idx[o]] = z[pos]
                zs.append(back)
                us.append(getattr(z, "unknown", None))
                acts.append(a)
            m = np.mean(zs, axis=0).view(Logits)
            m.unknown = None if any(u is None for u in us) else float(np.mean(us))
            m.pointer = getattr(runs[0][j][0], "pointer", None)
            m.escalate = next((w for w in (getattr(run[j][0], "escalate", None) for run in runs) if w), None)
            out.append((m, None if any(a is None for a in acts) else float(np.mean(acts))))
        return out

    def _one(self, text, ctx=None):
        ctx = ctx if ctx is not None else self._ctx(text)
        if self.long is not None and self._too_long(text):
            return self._retrieve(text, ctx)
        z, a = self._raw([text])[0]
        return self._finish(self.model._decision(self.spec, z), a, ctx=ctx)

    def _read(self, texts):
        """[text] → [(logits, act logit)] as a decision reads them: with long="retrieve", a text over the budget by the
        window of its retrieved sections (the signal the part answers on) — so calibration, fit / teach / adapt, the
        memory's features and the perturb re-asks see what a decision sees."""
        texts = list(texts)
        if self.long is not None:
            texts = [self._window(t)[2].text if self._too_long(t) else t for t in texts]
        return self._raw(texts)

    # --- long texts (long="retrieve", solvi.longdoc)
    def _prompt_tokens(self):
        sp = self.spec
        return self.model.count_tokens(" ".join([sp.task] + [str(o) for o in sp.options] +
                                                [str(v) for v in (sp.descriptions or {}).values() if v])) + len(sp.options) + 8

    def budget(self):
        """The tokens of input this decision can read in one pass: max_len minus its question."""
        return max(32, self.model.max_len - self._prompt_tokens())

    def _too_long(self, text):
        return self.model.count_tokens(text) > self.budget()

    def _window(self, text):
        """The top_k sections of a long text → (the LongDocument, [(section, score)], the window read, reranked?)."""
        from .longdoc import LongDocument
        budget = self.budget()
        doc = LongDocument(text, max_tokens=max(16, budget // self.top_k), count=self.model.count_tokens)
        sp = self.spec
        query = " ".join([sp.task] + [str(o) for o in sp.real] + [str(v) for v in (sp.descriptions or {}).values() if v])
        rr = self._relevance if self.rerank else None
        sel = doc.select(query, k=self.top_k, budget=budget, rerank=rr)
        return doc, sel, doc.window([s for s, _ in sel]), rr is not None

    def _retrieve(self, text, ctx=None):
        """Decide on the top_k sections of a long text; spans and evidence mapped back into the text; the sections read
        in extra["long"]. ctx: as for _finish (the group, and the whole text for perturb and the memory)."""
        sp = self.spec
        doc, sel, win, rr = self._window(text)
        z, a = self._raw([win.text])[0]
        d = self._finish(self.model._decision(sp, z), a, ctx=ctx if ctx is not None else self._ctx(text))
        score = {s.index: sc for s, sc in sel}
        d.extra["long"] = {"read": len(sel), "of": len(doc), "by": "bm25+decider" if rr else "bm25",
                           "sections": [[s.start, s.end, s.heading, round(float(score[s.index]), 6)] for s in win.sections]}
        if isinstance(d.value, Quote):
            got = win.to_doc(d.value.start, d.value.end)
            if got is None:
                d.escalate = d.escalate or "the span crosses two sections of the long text"
            else:
                d.value = dataclasses.replace(d.value, start=got[0], end=got[1])
        ev = []
        for e in d.evidence:
            if isinstance(e, Quote):
                got = win.to_doc(e.start, e.end)
                if got is not None:
                    ev.append(dataclasses.replace(e, start=got[0], end=got[1]))
            else:
                ev.append(e)
        d.evidence = ev
        return d

    def _relevance(self, texts):
        """The decider's own relevance of passages to this question: p(yes) of "Does this passage help answer: …?"."""
        task = f"Does this passage help answer the question: {self.spec.task}"
        ds = self.model.decide(list(texts), task, ["yes", "no"], kind="noul")
        return [float(d.probs.get("yes", 0.0)) for d in ds]

    # the model's methods for this decision
    def decide(self, text):
        """An input (a text, a state, or Facts by name) → Decision; a list of inputs → a list."""
        one = isinstance(text, Facts) or _single(text)
        xs = [text] if one else list(text)
        ts = [self._input_text(x) for x in xs]
        if self.long is not None and any(self._too_long(t) for t in ts):
            out = [self._one(t, ctx=self._ctx(t, vals=x if isinstance(x, Facts) else None, raw=x)) for t, x in zip(ts, xs)]
        else:
            out = [self._finish(self.model._decision(self.spec, z), a,
                                ctx=self._ctx(t, vals=x if isinstance(x, Facts) else None, raw=x))
                   for (z, a), t, x in zip(self._raw(ts), ts, xs)]
        return out[0] if one else out

    def score(self, text):
        d = self.decide(text)
        return d.probs if isinstance(d, Decision) else [x.probs for x in d]

    def _kw(self):
        sp = self.spec
        kw = dict(task=sp.task, options=sp.options, descriptions=sp.descriptions, multi=sp.multi, other=sp.other or False,
                  kind=sp.kind)
        for x, v in (("unknown", sp.unknown), ("k", sp.k), ("bins", sp.edges), ("unit", sp.unit), ("evidence", sp.evidence)):
            if v not in (None, False, 0):         # the same question key as the part's own spec
                kw[x] = v
        if sp.kind == "number":
            kw.update(coverage=sp.coverage, integer=sp.integer)
        return kw

    # adapt / fit / teach are fitted on the logits the part decides on (self._read: every option order averaged with
    # option_order="average", a long input's retrieved window), not on the model's single-order logits of the whole text
    def adapt(self, texts):
        """Label-bias correction from unlabelled inputs of the domain (see DecideModel.adapt)."""
        ts = [self._input_text(t) for t in texts]
        if self.spec.kind == "span" or not ts:
            return self.model.adapt(ts, **self._kw())             # raises the model's error
        return self.model.adapt(ts, logits=[z for z, _ in self._read(ts)], **self._kw())

    def fit(self, examples, lam=1.0, folds=4):
        """Few-shot "S" from [(input, correct)] (see DecideModel.fit)."""
        ex = [(self._input_text(t), self.spec.label(y)) for t, y in examples if y is not Unknown]
        return self.model.fit(ex, lam=lam, folds=folds, logits=[z for z, _ in self._read([t for t, _ in ex])],
                              **self._kw())

    def teach(self, text, correct):
        """One correction, absorbed at once (see DecideModel.teach). → ms."""
        t = self._input_text(text)
        return self.model.teach(t, self.spec.label(correct), logits=self._read([t])[0][0], **self._kw())

    def reset(self):
        self.model.reset(**self._kw())

    def _labelled(self, examples, signal):
        """Decide labelled examples [(input, correct)] → (the signal per example, correct 0/1 per example, "act" |
        "confidence", [Decision]). "Not stated" (solvi.Unknown) is a label like any other."""
        ex = [(self._input_text(t), y) for t, y in examples]
        if not ex:
            raise ValueError("calibration needs labelled examples")
        if signal not in ("auto", "act", "confidence"):
            raise ValueError('signal must be "auto", "act" or "confidence"')
        gold = [Unknown if y is Unknown else self.spec.label(y) for _, y in ex]
        conf, act, ok, ds = [], [], [], []
        for (z, a), y in zip(self._read([t for t, _ in ex]), gold):    # a long input: the window a decision reads
            d = self.model._decision(self.spec, z)
            ds.append(d)
            ok.append(float((Unknown if d.value is Unknown else self.spec.label(d.value)) == y))
            conf.append(d.conf)
            act.append(None if a is None else self.model.act_probability(self.spec, d, a))
        has_act = all(a is not None for a in act)
        if signal == "act" and not has_act:
            raise ValueError("the model gives no act signal: use signal='confidence'")
        use_act = signal == "act" or (signal == "auto" and has_act and self.use_act is not False)
        return (act if use_act else conf), ok, ("act" if use_act else "confidence"), ds

    def _set_threshold(self, sig, thr, guarantee, groups=None):
        """Set the calibrated threshold on its signal and clear the other signal's (an earlier calibration's threshold on
        the other signal would still escalate, so the new calibration's numbers would not describe the part); what was
        cleared is recorded in the guarantee as "cleared"."""
        other = "escalate_below" if sig == "act" else "act_threshold"
        old = getattr(self, other)
        setattr(self, other, None)
        if sig == "act":
            self.act_threshold = thr
        else:
            self.escalate_below = thr
        if old is not None:
            guarantee = {**guarantee, "cleared": {other: old}}
        self.guarantee = guarantee
        self.groups = groups
        extra = [n for n in (groups["by"].names if groups else []) if n not in self.facts]
        self.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                                                for f in self.facts + extra])

    def calibrate_for(self, examples, error=0.05, signal="auto", method="empirical", delta=0.10):
        """Choose the escalation threshold for a target error rate among the answers given alone, on labelled examples
        [(input, correct)]. method="empirical": the lowest threshold at which the calibration decisions it lets through
        are wrong at most `error` of the time — no guarantee on new inputs (it was 3–5× off on other data sets in our
        measurements); method="ltt" (learn-then-test): the error among the answered is ≤ `error` with probability
        ≥ 1 − delta for inputs like the examples — a strong promise, so it often lets nothing through. signal: "act"
        (the model's act probability → act_threshold), "confidence" (the calibrated confidence → escalate_below) or
        "auto" (act when the model has an act head). No threshold reaches the target → everything escalates (inf).
        Changes the part's fingerprint. → {"signal", "threshold", "coverage", "error", "n", "target_error", "method",
        "guarantee"}. For a guarantee on the share of all questions answered wrongly, see act_guard."""
        from .calibration import accuracy_at, check_rate, ltt_threshold
        examples = list(examples)
        if method not in ("empirical", "ltt"):
            raise ValueError('method must be "empirical" or "ltt"')
        check_rate("error", error, zero=method == "empirical")     # ltt cannot certify 0 (math domain error before)
        if method == "ltt":
            check_rate("delta", delta)
        sig, ok, name, _ = self._labelled(examples, signal)
        if method == "ltt":
            thr = ltt_threshold(sig, [1 - o for o in ok], error, delta)
            g = {"method": "ltt", "error": error, "delta": delta, "n": len(ok), "signal": name,
                 "promise": f"error among the answers given alone ≤ {error:g} with probability ≥ {1 - delta:g}, "
                            "for inputs like the calibration examples"}
        else:
            thr = _threshold(sig, ok, error)
            g = {"method": "empirical", "error": error, "n": len(ok), "signal": name,
                 "promise": "none: the error was measured on the calibration examples only"}
        self._set_threshold(name, thr, g)
        acc, cov = accuracy_at(sig, ok, thr)
        return {"signal": name, "threshold": thr, "coverage": cov, "error": (1 - acc) if cov else 0.0, "n": len(ok),
                "target_error": error, "method": method, "guarantee": g["promise"]}

    def act_guard(self, examples, risk=0.10, signal="auto", groups=None, min_group=100, delta=0.10):
        """Answer alone only as far as a guarantee allows (conformal risk control), from labelled examples of your own
        stream [(input, correct)] — a few hundred is typical: the escalation threshold is set so that, for inputs like
        the examples, P(answered alone AND wrong) ≤ risk — a share of all questions (answered or escalated), not of
        the answered ones. It holds for your stream, not under a shift of domain: recalibrate when the inputs change.
        Too few or too hard examples → everything escalates (threshold inf). Feasibility: when the model is wrong on
        a share μ > risk of the examples, any rule must escalate at least (μ − risk) / (1 − risk) of the inputs
        ("must_escalate_at_least"; arXiv 2606.29054) — a better signal can only get closer to that bound. Changes the part's fingerprint; the trace
        of every decision records the promise. → {"signal", "threshold", "answered" (share answered alone on the
        examples), "error" (among them), "risk" (answered and wrong, on the examples), "n", "guarantee"}.

        groups: a threshold per group — a fact name ("domain"), a hierarchy of fact names (["domain", "task"]) or a
        function of facts returning a group or a path (see GroupBy); the examples then give those facts (Facts(...) or
        a state with those keys). The promise over the whole stream allows a hard group to be answered wrongly far more
        often than `risk`; per group it holds inside each: every group with at least `min_group` examples gets its own
        threshold, a smaller one is pooled with the rest of its parent (whose threshold is calibrated on exactly those
        examples), the rest of the stream takes what is left. delta=0.10: with probability ≥ 90% over the examples,
        P(answered alone and wrong | group) ≤ risk in every group at once (a binomial bound per group at delta divided
        by the number of groups — Bonferroni; after HG-CRC, arXiv 2607.24562); delta=None: conformal risk control per
        group (each group on average). Every decision records its group and the group whose threshold applied; an
        input that does not give its group escalates. The group facts join the part's inputs: register the part in a
        catalog after act_guard. Adds "groups" ({path: {"threshold", "n", "answered", "error", "risk", "pooled"}}) to
        the result."""
        from .calibration import crc_threshold
        examples = list(examples)
        sig, ok, name, _ = self._labelled(examples, signal)
        s, o = np.asarray(sig, float), np.asarray(ok, float)
        base = float(1 - o.mean())
        if groups is None:
            thr = crc_threshold(sig, [1 - x for x in ok], risk)
            g = {"method": "crc", "risk": risk, "n": len(ok), "signal": name,
                 "promise": f"P(answered alone and wrong) ≤ {risk:g} for inputs like the calibration examples"}
            self._set_threshold(name, thr, g)
            auto = s >= thr
            out = {}
        else:
            by = GroupBy(groups)
            paths = [by.path(x if isinstance(x, Facts) else None, x) for x, _ in examples]
            if any(p is None for p in paths):
                i = next(i for i, p in enumerate(paths) if p is None)
                raise ValueError(f"example {i}: its group ({by.names}) is not given; give the examples as Facts(...) or "
                                 "states with those keys")
            nodes, info, auto = _group_guard(s, 1 - o, paths, risk, min_group, delta)
            g = {"method": "crc-groups" if delta is None else "group-bound", "risk": risk, "n": len(ok), "signal": name,
                 "groups": by.label(), "min_group": min_group, "delta": delta,
                 "promise": _group_promise(risk, delta, len(nodes))}
            self._set_threshold(name, nodes[()]["threshold"], g, {"by": by, "nodes": nodes, "signal": name})
            thr = nodes[()]["threshold"]
            out = {"groups": info}
        out = {"signal": name, "threshold": thr, "answered": float(auto.mean()),
               "error": float(1 - o[auto].mean()) if auto.any() else 0.0,
               "risk": float(((1 - o) * auto).mean()), "n": len(ok), "guarantee": g["promise"],
               "base_error": base, "must_escalate_at_least": max(0.0, (base - risk) / (1 - risk)), **out}
        return out

    def conformal(self, examples, coverage=0.90):
        """Conformal answer sets from labelled examples [(input, correct)]: afterwards every decision carries
        `extra["candidates"]` — the answers that cannot be ruled out, which contain the right one with probability
        ≥ coverage for inputs like the examples (score and number questions: one contiguous interval) — and an
        escalation's message lists them for the person who takes over. It does not change what is answered alone
        (see act_guard). Choice, yes/no, score and number questions. → {"coverage", "quantile", "n", "mean_size"}."""
        from .calibration import conformal_quantile, set_scores
        if self.spec.multi or self.kind in ("rank", "span"):
            raise ValueError(f"conformal sets need a single answer from a closed list; not for {self.kind!r} questions")
        examples = list(examples)                    # an iterator (zip, a generator) is read twice below
        _, _, _, ds = self._labelled(examples, "confidence")
        ordinal = self.kind in ("score", "number")
        scores, sizes = [], []
        for d, (_, y) in zip(ds, examples):
            keys = list(d.probs)
            g = Unknown if y is Unknown else self.spec.label(y)
            if g not in keys:
                raise ValueError(f"{y!r}: this question cannot answer it (its answers: {keys})")
            scores.append(float(set_scores([d.probs[k] for k in keys], ordinal, Unknown in keys)[keys.index(g)]))
        q = conformal_quantile(scores, 1 - coverage)
        self.conformal_set = {"coverage": coverage, "quantile": q, "n": len(scores), "ordinal": ordinal}
        for d in ds:
            sizes.append(len(self.candidates(d)))
        return {"coverage": coverage, "quantile": q, "n": len(scores), "mean_size": float(np.mean(sizes))}

    def save_calibration(self, path):
        """Write this decision's calibration — the escalation thresholds (per group too), the guarantee record, the
        conformal set — with the question and the fingerprint of the model and adaptation it was fitted on, to a JSON
        file (solvi.calibfile; `solvi calibrate` writes the same). → path."""
        from .calibfile import save
        return save(self, path)

    def load_calibration(self, path, groups=None, strict=True):
        """Apply a calibration file written by save_calibration / `solvi calibrate`: afterwards the part escalates, and
        records its guarantee, exactly as right after calibrating (the same fingerprint). Refuses (ValueError) a file made
        for another question, another checkpoint or another adaptation of this question — strict=False loads it anyway.
        Thresholds per group by a function: pass it again as groups=. Call it before registering the part in a catalog
        when the calibration has groups (the group facts join the part's inputs). While `solvi calibrate` loads a
        catalog, calibration files are not applied (the part is calibrated afresh). → self."""
        from .calibfile import load
        return load(self, path, groups, strict)

    def memory(self, memory=None, **settings):
        """A memory of corrected cases consulted on every decision of this part (solvi.memory.CorrectionMemory): its
        proposal, the cases it rests on and its fingerprint go into extra["memory"]; mode="check" (default) escalates when
        similar corrected cases say another answer, mode="answer" may also answer where the part escalated by its own
        threshold. settings: k, radius, min_strength, min_agreement, text, text_weight, mode. memory: an existing
        CorrectionMemory of this part to attach; False detaches. Its fingerprint is part of the part's. → the memory."""
        from .memory import CorrectionMemory
        if memory is False:
            self.correction_memory = None
            return None
        if memory is None:
            memory = self.correction_memory if self.correction_memory is not None and not settings else \
                CorrectionMemory(self, **settings)
        elif memory.part is not self:
            raise ValueError(f"this memory belongs to {memory.part.__name__!r}, not {self.__name__!r}")
        self.correction_memory = memory
        return memory

    def question(self, cat, name=None, text=None, min_confidence=None, checkpoints=None, require_evidence=False):
        """Make this decision the answer of a question: registers it as the question's rule (`cat.rule(name)(self)`) and
        returns the Question — choice, multi, ordinal (score) or yes_no (noul), with the option descriptions. System.teach on
        that question teaches this decision."""
        from .core import Answer, Question
        name = name or self.__name__
        cat.rule(name)(self)
        sp = self.spec
        order = self._shown_labels if self.option_order == "canonical" else sp.options
        opts = {o: sp.descriptions.get(o, "") for o in order} if sp.descriptions else list(order)
        if sp.kind == "noul":
            at = Answer.yes_no()
            at.descriptions = dict(sp.descriptions)
        elif sp.kind == "span":
            at = Answer.span(source=self.facts[0], type=sp.vtype)
        elif sp.kind == "rank":
            at = Answer.rank(opts, sp.k)
        elif sp.kind == "number":
            at = Answer.estimate(sp.edges, coverage=sp.coverage, unit=sp.unit, integer=sp.integer)
        else:
            at = {"multi": Answer.multi, "score": Answer.ordinal}.get(sp.kind, Answer.choice)(opts)
        if sp.unknown:
            at = Answer.maybe(at)
        return Question(name, text or sp.task, at, checkpoints=list(checkpoints or []), min_confidence=min_confidence,
                        require_evidence=require_evidence)


def _group_promise(risk, delta, n_groups):
    if delta is None:
        return (f"P(answered alone and wrong) ≤ {risk:g} within each group, for inputs like the calibration examples "
                "(conformal risk control per group)")
    return (f"P(answered alone and wrong) ≤ {risk:g} within every group at once ({n_groups} groups), with probability "
            f"≥ {1 - delta:g}, for inputs like the calibration examples")


def _group_info(nodes, owner, paths, auto, wrong):
    """The per-group report of act_guard(groups=...): each node's threshold, examples, answered share, error and risk on
    them, and the groups pooled into it."""
    info = {}
    for node, v in nodes.items():
        ix = [i for i, o in enumerate(owner) if o == node]
        a, w = auto[ix], wrong[ix]
        info[node] = {"threshold": v["threshold"], "n": v["n"], "answered": float(a.mean()) if ix else 0.0,
                      "error": float(w[a].mean()) if a.any() else 0.0, "risk": float((w * a).mean()) if ix else 0.0,
                      "pooled": sorted({paths[i] for i in ix if paths[i] != node})}
    return info


def _group_guard(score, wrong, paths, risk, min_group, delta):
    """Thresholds per group for one signal → (nodes {path: {"threshold", "n"}}, report per node, answered alone [n])."""
    from .calibration import _signal_losses, certify_groups
    nodes, owner = certify_groups(_signal_losses(score, wrong), paths, risk, min_group, delta)
    nodes = {k: {"threshold": v["threshold"], "n": v["n"]} for k, v in nodes.items()}
    auto = np.array([score[i] >= nodes[owner[i]]["threshold"] for i in range(len(score))], bool)
    return nodes, _group_info(nodes, owner, [tuple(p) for p in paths], auto, np.asarray(wrong, float)), auto


def _shown(v):
    """A decision value as it reads: a quote by its text, a multi-label answer as a tuple."""
    return v.value if isinstance(v, Quote) else v


def _vkey(v):
    """What must be equal for two answers to be the same (a quote by its text: offsets move when a sentence is removed;
    a multi-label answer by its set)."""
    if isinstance(v, Quote):
        return ("quote", v.value)
    if isinstance(v, tuple):
        return ("set", frozenset(v))
    return ("value", v if v is Unknown else jsonable(v))


def act_features(sp, d, act_logit):
    """The features of the act calibrator (ACT_FEATURES) for a calibrated decision: confidence (the decision's calibrated
    confidence: the top probability; multi-label: the least certain option's max(p, 1 − p)), margin (top − second
    probability; multi-label: the smallest |2p − 1|), entropy (of the probabilities, nats; multi-label: the mean binary
    entropy), act_logit (raw), n_options, kind=<kind> (1 / 0)."""
    if sp.multi:
        p = np.clip(np.array([d.probs[o] for o in sp.real], float), 1e-12, 1 - 1e-12)
        margin = float(np.min(np.abs(2 * p - 1)))
        ent = float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))
    elif d.probs:
        p = np.clip(np.sort(np.array(list(d.probs.values()), float))[::-1], 1e-12, 1)
        margin = float(p[0] - p[1]) if len(p) > 1 else 1.0
        ent = float(-(p * np.log(p)).sum())
    else:                                          # a span without "not stated": only its confidence
        margin, ent = 1.0, 0.0
    x = {"confidence": float(d.conf), "margin": margin, "entropy": ent, "act_logit": float(act_logit),
         "n_options": float(len(sp.real))}
    x.update({f"kind={k}": float(sp.kind == k) for k in KINDS})
    # the l14g features: always present (0 when the question does not allow "not stated"), so an l14g act calibrator can
    # score every kind — a plain yes / no or choice question on an l14g checkpoint as well
    x["p_unknown"] = float((d.probs or {}).get(Unknown, 0.0))
    x.update({f"kind={k}": float(sp.kind == k) for k in KINDS_V3[len(KINDS):]})
    return x


def _threshold(sig, ok, error):
    """The lowest threshold t (among the observed values) such that the cases with signal ≥ t are wrong at most `error` of
    the time — ties are kept together, so the error holds for everything the threshold lets through; inf if none."""
    sig, ok = np.asarray(sig, float), np.asarray(ok, float)
    best = math.inf
    for t in sorted(set(sig.tolist()), reverse=True):
        m = sig >= t
        if 1 - ok[m].mean() <= error + 1e-12:
            best = t
        else:
            break
    return best


def plan_batches(steps):
    """The strategist's grouping: decision parts in a flow that read the same facts with the same model, when the model can
    answer several questions in one forward pass → [[step name]] (chunks of at most the checkpoint's max_questions)."""
    groups = {}
    for st in steps:
        p = st.part
        if p.alternatives is not None or p.func is None:
            continue
        d = getattr(p.func, "__solvi_decision__", None)
        if not isinstance(d, DecisionPart) or not d.model.batchable or d.spec.pointer:
            continue                     # the pointer needs a pass of its own; a combination of models (solvi.multi) too
        groups.setdefault((id(d.model), tuple(d.facts)), (d.model, []))[1].append(p.name)
    out = []
    for model, names in groups.values():
        n = model.caps["max_questions"]
        for i in range(0, len(names), n):
            if len(names[i:i + n]) > 1:
                out.append(names[i:i + n])
    return out


def decision_of(catalog, question):
    """The DecisionPart behind a question's answer: its rule is a decision, or a rule that only passes a decided fact on
    (e.g. `def team(route): return route`). → the part or None."""
    rule = catalog.rules.get(question)
    if rule is None or rule.func is None:
        return None
    d = getattr(rule.func, "__solvi_decision__", None)
    if d is None and len(rule.inputs) == 1:
        part = catalog.parts.get(rule.inputs[0])
        if part is not None and part.alternatives is None:
            d = getattr(part.func, "__solvi_decision__", None)
    return d
