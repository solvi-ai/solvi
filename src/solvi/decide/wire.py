"""What the scorer sees: items, passes, prompts, logits, the pointer's spans and evidence. (Part of solvi.decide, which
re-exports every name.)"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..core import Quote
from .kinds import MARKERS, NULL_SOURCE


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
    max_len: int = 0        # tokens of the sequence to read (question and input); 0: the scorer's max_len. long="full"
    #                         sets the checkpoint's long-input length; scorers that read the whole text anyway ignore it

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
    (exactly `span_dist` of the training code). A span's text is the input's characters from token i's
    start to token j's end, without surrounding whitespace — so it is literally in the input. `ptr`: {"start": [T],
    "end": [T], "offsets": [(char start, char end)] per token, "null": [start_m, end_m] (or their sum)}. `temperature`
    (the checkpoint's `temperature.span`) divides every start / end score, the null span's too, before the softmax — as
    the answer-primitives calibration fitted it."""
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


def _respan(value, read, own):
    """A span's value after mapping it from a long text's window to the document: the document's own text when the value
    was the window's text (a span over neighbouring sections has the document's whitespace between them, not the
    window's separator), else the value as it was."""
    return own if isinstance(value, str) and value == read else value


def _typed_span(spans, vtype):
    """For a typed span (`Span[float]`): (index, probability mass) of the most probable span inside the best span whose
    text parses as `vtype` — the best span trimmed to its value ('149.90 EUR' → '149.90'); the mass is that of every span
    between the two (they all give this value). Never a span outside the best one, and never a piece of the best span
    that states another value ('851.12' inside 'EUR 18,851.12'): then the best span is kept and the answer's type check
    reads it (solvi.typed.span_value: 18851.12) or rejects it. (0, None) for str / untyped spans or when the best span
    parses."""
    if vtype is None or vtype is str:
        return 0, None
    from ..typed import adapter, span_value
    try:
        adapter(vtype)
    except Exception:  # noqa: BLE001
        return 0, None
    none = object()

    def value(t, strict=True):
        try:
            return span_value(vtype, t, strict=strict)
        except ValueError:
            return none

    _, a0, b0, t0 = spans[0]
    if value(t0) is not none:
        return 0, None
    whole = value(t0, strict=False)                    # what the best span states, read as people write it ("EUR 18,851.12")
    for k, (_, a, b, t) in enumerate(spans):
        if a0 <= a and b <= b0:
            v = value(t)
            if v is none or (whole is not none and v != whole):
                continue                               # a piece of the number ("851.12") is not its value: never trim to it
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


__all__ = ["decode_pointer", "Item", "Logits", "Pass", "pass_prompt", "pointer_evidence", "prompt"]
