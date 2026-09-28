"""Answer primitives: "not stated", evidence, spans, rankings and estimates — each a value with a confidence.

A question's answer type says which primitive it is (`Answer.span / rank / estimate`, `Answer.maybe(...)`, or the types
`Span[T]`, `Rank[...]`, `Estimate[...]`, `Maybe[T]` from solvi.typed); its rule, a learned part or a model decision proposes
the output, and `resolve` turns the rule's trace record into the answer, its confidence and its details — checking what
must hold (a span is literally in its text and parses as its type, a ranking uses only the options, a distribution is over
the declared bins). What a failure is:

  span not literally in the text / outside it   → abstain, guard "grounding"       (safeguard "grounding rejected")
  span text does not parse as its type          → abstain, guard "type_rejected"   (safeguard "type rejected")
  ranking / distribution not over the options   → abstain, guard "outside_options" (safeguard "outside the options")
  require_evidence and no supporting quote      → abstain, guard "evidence_missing" (safeguard "evidence missing")

Evidence quotes are checked when the part runs (solvi.core.ground): an output whose evidence is not literally in its text
is rejected like an ungrounded quote — the fact is missing, the next producer runs, else the answer abstains.

Confidence — the probability that the answer, as returned, is right — per kind:

  yes_no / choice / ordinal   p(answer)
  multi                       the least certain option's max(p, 1 − p)
  Unknown (not stated)        p(not stated)
  span                        p(this span) (a rule: 1, or its Claim's confidence)
  rank                        Plackett–Luce probability of the returned top k in this order (a rule: 1)
  estimate                    probability that the value lies in the interval: the mass of its bins (× p(stated));
                              a plain number: 1

and for every kind at most the confidence of the facts the answer rests on (quotes, decisions), as for any rule."""
from __future__ import annotations

import itertools
import math
from enum import Enum

from .core import Quote, Unknown

GROUNDING, TYPE_REJECTED, OUTSIDE, NO_EVIDENCE = "grounding", "type_rejected", "outside_options", "evidence_missing"


class Rejected(Exception):
    def __init__(self, guard, why):
        super().__init__(why)
        self.guard, self.why = guard, why


def _short(v, n=48):
    s = v if isinstance(v, str) else repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


# --- plain normalization (hard-check `then` values, fit / teach answers): no text to ground against
def normalize(at, v):
    if at.kind == "span":
        return v.value if isinstance(v, Quote) else v
    if at.kind == "rank":
        return _order(at, v, None)[0]
    if at.kind == "estimate":
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
        raise ValueError(f"an estimate is a number, not {v!r}")
    raise ValueError(f"unknown answer kind {at.kind}")


# --- rank
def _k(at):
    return at.k if at.k is not None else len(at.options)


def _order(at, v, probs):
    """A rule's / model's ranking output → (the top-k tuple, scores or None)."""
    opts = list(at.options)
    if isinstance(v, dict):
        bad = [o for o in v if o not in opts]
        if bad:
            raise ValueError(f"ranked {bad}, not among {opts}")
        pos = {o: i for i, o in enumerate(opts)}
        order = sorted(v, key=lambda o: (-float(v[o]), pos[o]))
        scores = {o: float(v[o]) for o in order}
    elif isinstance(v, (list, tuple)):
        order = [x.value if isinstance(x, Enum) else x for x in v]
        bad = [o for o in order if o not in opts]
        if bad or len(set(order)) != len(order):
            raise ValueError(f"ranking {list(v)!r} is not distinct options of {opts}")
        scores = None
    else:
        raise ValueError(f"a ranking is an ordered list or {{option: score}}, not {_short(v)}")
    k = _k(at)
    if len(order) < k:
        raise ValueError(f"ranking of {len(order)} options, {k} needed")
    if probs:
        scores = {o: float(probs[o]) for o in opts if o in probs}
    return tuple(order[:k]), scores


def plackett_luce(order, probs):
    """P(this top-k prefix, in this order) under Plackett–Luce with the options' probabilities as weights."""
    w = {o: max(float(p), 0.0) for o, p in probs.items() if o is not Unknown}
    total = sum(w.values())
    if total <= 0:
        return 0.0
    out, left = 1.0, total
    for o in order:
        if left <= 1e-15:
            return 0.0
        out *= w.get(o, 0.0) / left
        left -= w.get(o, 0.0)
    return out


def rank_candidates(at, probs, limit=2000):
    """Rankings to consider in joint decoding: every top-k order of the options (when there are at most `limit`), with its
    Plackett–Luce log-probability, most probable first; None when there are too many."""
    opts, k = list(at.options), _k(at)
    n = math.perm(len(opts), k)
    if n > limit:
        return None
    out = []
    for perm in itertools.permutations(opts, k):
        p = plackett_luce(perm, probs)
        out.append((tuple(perm), math.log(max(p, 1e-12))))
    return sorted(out, key=lambda t: -t[1])


# --- estimate
def distribution(at, v, probs):
    """A rule's distribution ({bin label or index: p}, or [p per bin]) or a model's probabilities → ([p per bin, summing
    to 1], p(not stated))."""
    if not at.bins:
        raise ValueError("this estimate has no bins: return a plain number")
    labels = list(at.options)
    src = probs if probs else v
    ns = 0.0
    if isinstance(src, dict):
        idx = {lab: i for i, lab in enumerate(labels)}
        p = [0.0] * len(labels)
        for key, x in src.items():
            if key is Unknown:
                ns = float(x)
            elif key in idx:
                p[idx[key]] += float(x)
            elif isinstance(key, int) and not isinstance(key, bool) and 0 <= key < len(labels):
                p[key] += float(x)
            else:
                raise ValueError(f"probability for {key!r}, not a bin of {labels}")
    elif isinstance(src, (list, tuple)) and len(src) == len(labels):
        p = [float(x) for x in src]
    else:
        raise ValueError(f"a distribution over the {len(labels)} bins {labels}, not {_short(src)}")
    s = sum(p)
    if any(x < 0 for x in p) or s <= 0:
        raise ValueError("a distribution has non-negative probabilities that do not all vanish")
    return [x / s for x in p], ns


def estimate_of(at, p):
    """A distribution over the bins → (value, [lo, hi], mass): the middle of the median bin (an open bin: its edge), the
    bins from the (1 − c)/2 to the (1 + c)/2 cumulative probability (None for an open end), and their probability mass —
    the probability that the value lies in the interval. The answer-primitives decider's `number_summary`, in solvi."""
    c = at.coverage or 0.8
    cdf, acc = [], 0.0
    for x in p:
        acc += x
        cdf.append(acc)

    def q(t):
        return next((i for i, v in enumerate(cdf) if v >= t - 1e-9), len(p) - 1)
    med, lo, hi = q(0.5), q((1 - c) / 2), q(1 - (1 - c) / 2)
    bounds = [None] + list(at.bins) + [None]
    a, b = bounds[med], bounds[med + 1]
    value = b if a is None else a if b is None else (a + b) / 2
    return value, [bounds[lo], bounds[hi + 1]], float(sum(p[lo:hi + 1]))


# --- the answer
def evidence_of(rec):
    """The supporting quotes recorded with an output (checked when the part ran) → [Quote(text, start, end, source)]."""
    x = rec.extra if isinstance(rec.extra, dict) else None
    rows = (x or {}).get("evidence") or ()
    return [Quote(t, s, e, src) for s, e, src, t in rows]


def resolve(at, rec, init):
    """The rule's record → {"answer", "confidence" (the answer's own), "probs", "evidence", "extra"}; raises Rejected."""
    from pydantic import TypeAdapter

    from .provenance import NOT_GROUNDED, QUOTE_OUTSIDE, matches
    v, probs = rec.value, dict(rec.probs) if rec.probs else None
    ev = evidence_of(rec)
    conf = float(rec.confidence)
    if v is Unknown:
        if not at.unknown:
            raise Rejected(OUTSIDE, "answered Unknown (not stated), which the question does not allow: declare Maybe[...]")
        return {"answer": Unknown, "confidence": conf, "probs": probs or {}, "evidence": ev, "extra": None}
    kind = at.kind
    ns = float(probs.get(Unknown, 0.0)) if probs else 0.0
    if kind == "span":
        src = at.source or "doc"
        if rec.quote:
            s, e, src = rec.quote
            text = init.get(src)
            if not (isinstance(text, str) and 0 <= s <= e <= len(text)):
                raise Rejected(GROUNDING, f"{QUOTE_OUTSIDE}: span [{s}:{e}] of {src}")
            t = text[s:e]
            if v is not None and matches(v, t) is False:
                raise Rejected(GROUNDING, f"{NOT_GROUNDED}: span {_short(v)!r} is not the text at {src}[{s}:{e}] ({_short(t)!r})")
        elif isinstance(v, str) and v:
            text = init.get(src)
            s = text.find(v) if isinstance(text, str) else -1
            if s < 0:
                raise Rejected(GROUNDING, f"{NOT_GROUNDED}: span {_short(v)!r} is not in {src}")
            e, t = s + len(v), v
        else:
            raise Rejected(GROUNDING, f"{NOT_GROUNDED}: a span answer is a Quote or a text in {src}, not {_short(v)}")
        value = t
        if at.type is not None and at.type is not str:
            try:
                value = TypeAdapter(at.type).validate_python(t.strip())
            except ValueError as err:
                from .typed import _msg, type_name
                raise Rejected(TYPE_REJECTED, f"type rejected: span {_short(t)!r} is not {type_name(at.type)} ({_msg(err)})") \
                    from None
        span = Quote(t, s, e, src, conf)
        return {"answer": value, "confidence": conf, "probs": probs or {}, "evidence": [span] + ev, "extra": None}
    if kind == "rank":
        try:
            order, scores = _order(at, v, probs)
        except ValueError as err:
            raise Rejected(OUTSIDE, f"ranking {_short(v)} is outside the options: {err}") from None
        if probs:
            conf = plackett_luce(order, probs) * (1 - ns)
        return {"answer": order, "confidence": conf, "probs": probs or {}, "evidence": ev,
                "extra": {"scores": scores} if scores else None}
    if kind == "estimate":
        if (isinstance(v, (int, float)) and not isinstance(v, bool)) and not probs:
            return {"answer": v, "confidence": conf, "probs": {}, "evidence": ev,
                    "extra": {"interval": [v, v], "coverage": 1.0, **({"unit": at.unit} if at.unit else {})}}
        try:
            p, ns2 = distribution(at, v, probs)
        except ValueError as err:
            raise Rejected(OUTSIDE, f"estimate {_short(v)} is outside the options: {err}") from None
        ns = ns or ns2
        value, interval, mass = estimate_of(at, p)
        dist = dict(zip(at.options, p))
        if ns:
            dist = {k: x * (1 - ns) for k, x in dist.items()}
            dist[Unknown] = ns
        return {"answer": value, "confidence": mass * (1 - ns), "probs": dist, "evidence": ev,
                "extra": {"interval": interval, "coverage": at.coverage or 0.8, **({"unit": at.unit} if at.unit else {})}}
    value = at.normalize(v)                       # yes_no / choice / ordinal / multi (may be "not stated")
    return {"answer": value, "confidence": conf, "probs": probs or {}, "evidence": ev, "extra": None}


def fmt(answer, kind=None, extra=None):
    """An answer for people: "not stated", an estimate with its interval, a ranking."""
    if answer is Unknown:
        return "not stated"
    if kind == "estimate" and extra and extra.get("interval") is not None:
        lo, hi = extra["interval"]
        u = f" {extra['unit']}" if extra.get("unit") else ""
        if lo == hi:
            return f"{answer:g}{u}"
        rng = (f"less than {hi:g}" if lo is None else f"{lo:g} or more" if hi is None else f"[{lo:g}, {hi:g})")
        return f"{answer:g}{u} ({extra.get('coverage', 0.8):.0%} interval: {rng})"
    return repr(answer)
