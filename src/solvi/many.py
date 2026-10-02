"""A choice among many options: dozens or hundreds of candidates, more than a decider reads in one pass.

A decider reads the question — its task, options and descriptions — and the input in one sequence of max_len tokens.
Thirty catalog rows as options no longer fit ("task and options do not fit"), and twenty that do fit leave the input a
few dozen tokens. An agent's step — the actions in a room, the elements of a page, the rows of a catalog — has that
many candidates, and they change with every step.

    from solvi.many import Many, decide_many
    d = decide_many(model, text, "Which action achieves the goal?", actions, many=Many(query=goal))
    d.value                # one of `actions`
    d.extra["many"]        # {"mode": "shortlist", "considered": 8, "of": 45, "unconsidered": 37, "calls": [...], ...}

Modes (`Many(mode=...)`):

  direct      one ordinary decision over all the options — when they fit
  shortlist   a selector ranks the options against a query (BM25 over the options' labels and descriptions, or your
              function), the decider chooses among the best k. One call. The options left out were not considered: the
              record says how many, and when the first one left out scores nearly as well as the last one kept
              (`gap`), the decision escalates
  tournament  the options in blocks of `block`: the decider chooses in each block, the winners meet in the next round,
              the last round decides. Every option is considered; about N / (block − 1) calls
  auto        direct when the question fits and leaves the input at least `min_input` tokens (default: half of
              max_len); else shortlist when there is a selector, else tournament

Whatever the mode, `extra["many"]` records the mode applied, how many options were considered of how many, and every
model call (its options, its answer, its probabilities) — a decision that replays from its inputs, since the selector
and the model are deterministic. The decision's `probs` cover the options of the last call only.

What it does not do. A guarantee calibrated on one set of options (act_guard, conformal) does not carry over to
options that change: a threshold holds for the question it was calibrated on. In shortlist mode the right option may
not be among the k (the selector's recall bounds the accuracy), which `unconsidered` and the gap escalation make
visible, not impossible. Measured with solvi-base on a text game with 20–45 actions per situation (30 situations):
direct 0.70 where the options fit, shortlist (BM25, k=8, the goal as the query) 0.63, tournament (blocks of 10) 0.57
in 5 calls; scoring each option alone was no better than chance and is not offered. On a catalog of 240 products the
shortlist found the product among its top 3 in 0.40 of the requests and the model picked it in almost none: there the
facts decide (price within budget, the category), and code should narrow the candidates before the model is asked."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .core import Decision
from .longdoc import BM25, terms

MODES = ("auto", "direct", "shortlist", "tournament")


@dataclass
class Many:
    """How to choose among many options (see the module docstring). k: the shortlist's size. block: the options per
    tournament call. query: what the selector searches by (default: the input's text — a short query, the goal or the
    request, finds better). selector: "bm25", a function (query, labels, texts) → one score per option, or None (no
    shortlist: auto goes to the tournament). min_input: the tokens the input must be left with for a direct decision
    (None: half of max_len). gap: shortlist escalates when (score of the last kept − score of the first left out) /
    the best score is below it and the one left out matched the query at all. max_options: more is an error at once."""
    mode: str = "auto"
    k: int = 8
    block: int = 10
    query: str | None = None
    selector: str | Callable | None = "bm25"
    min_input: int | None = None
    gap: float = 0.1
    max_options: int = 2000

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"Many.mode is one of {MODES}, not {self.mode!r}")
        if int(self.k) < 2 or int(self.block) < 2:
            raise ValueError("Many.k and Many.block are at least 2")
        if self.selector is not None and self.selector != "bm25" and not callable(self.selector):
            raise ValueError('Many.selector is "bm25", a function (query, labels, texts) → scores, or None')
        if self.mode == "shortlist" and self.selector is None:
            raise ValueError("a shortlist needs a selector")


def option_texts(labels, descriptions=None):
    """What a selector reads for each option: its label and, when it has one, its description."""
    d = descriptions or {}
    return [f"{o} {d[o]}" if d.get(o) else str(o) for o in labels]


def bm25_scores(query, labels, descriptions=None):
    """BM25 of each option (label and description) for the query → one score per option, in the options' order."""
    return BM25([terms(t) for t in option_texts(labels, descriptions)]).score(terms(query))


def fit(model, text, task, labels, descriptions=None, min_input=None):
    """Does the question with all its options fit one pass, and what is left for the input → {"question_tokens",
    "input_budget", "max_len", "min_input", "fits"} (token counts are the model's own, or approximate when the model has
    no tokenizer: an LLM, a hosted decision model)."""
    from .decide import _Encoder, _Spec
    max_len = int(getattr(model, "max_len", 0) or 512)
    enc = getattr(getattr(model, "scorer", None), "enc", None)
    if isinstance(enc, _Encoder):
        sp = _Spec(task, list(labels), descriptions)
        q = enc.question_tokens((model._item(sp, text if isinstance(text, str) else ""),))
    else:
        q = model.count_tokens(" ".join([task] + option_texts(labels, descriptions))) + len(labels) + 8
    need = max_len // 2 if min_input is None else int(min_input)
    return {"question_tokens": q, "input_budget": max_len - q, "max_len": max_len, "min_input": need,
            "fits": max_len - q >= max(need, 1)}


def decide_many(model, text, task, options, descriptions=None, *, many=None, **spec):
    """Choose one of many options → Decision (value: one of the options; probs: over the options of the last call;
    extra["many"]: what was done). model: a DecideModel (a local checkpoint, solvi.llm, solvi.systemone). options: a
    list, or {option: description}. many: a Many (default: Many()). spec: passed to model.decide (escalate_below, ...)."""
    many = many or Many()
    if isinstance(options, dict):
        descriptions = {**options, **(descriptions or {})} if descriptions else dict(options)
        options = list(options)
    labels = list(options)
    descs = {o: v for o, v in (descriptions or {}).items() if v} or None
    n = len(labels)
    if n < 2:
        raise ValueError("a choice needs at least two options")
    if len(set(labels)) != n:
        raise ValueError("duplicate options")
    if n > many.max_options:
        raise ValueError(f"{n} options are more than Many.max_options = {many.max_options}: narrow the candidates first "
                         "(a rule, a selector) or raise max_options")
    text = model.text(text)
    calls = []

    def ask(opts):
        d = model.decide(text, task, opts, {o: descs[o] for o in opts if o in descs} if descs else None, **spec)
        calls.append({"options": list(opts), "value": d.value, "confidence": round(float(d.conf), 6),
                      "act": None if d.extra.get("act") is None else round(float(d.extra["act"]), 6),
                      "probs": {str(o): round(float(p), 6) for o, p in d.probs.items()},
                      "escalate": d.escalate})
        return d

    info = {"requested": many.mode, "of": n, "fit": fit(model, text, task, labels, descs, many.min_input), "calls": calls}
    mode = many.mode
    if mode == "auto":
        mode = "direct" if info["fit"]["fits"] else "shortlist" if many.selector is not None else "tournament"

    if mode == "direct":
        try:
            d = ask(labels)
        except ValueError as e:
            if many.mode != "auto":
                raise ValueError(f"{e} — Many(mode=\"shortlist\") or \"tournament\" choose among options that do not "
                                 "fit one pass") from None
            mode = "shortlist" if many.selector is not None else "tournament"
        else:
            info.update(mode="direct", considered=n, unconsidered=0)
            d.extra["many"] = info
            return d

    if mode == "shortlist":
        query = many.query if many.query is not None else text
        if callable(many.selector):
            scores = [float(s) for s in many.selector(query, labels, option_texts(labels, descs))]
            if len(scores) != n:
                raise ValueError(f"the selector gave {len(scores)} scores for {n} options")
            info["selector"] = "callable"
        else:
            scores = bm25_scores(query, labels, descs)
            info["selector"] = "bm25"
        order = sorted(range(n), key=lambda i: (-scores[i], i))          # ties: the options' own order
        k = min(int(many.k), n)
        short = [labels[i] for i in order[:k]]
        best, last = scores[order[0]], scores[order[k - 1]]
        out = scores[order[k]] if n > k else None                        # the first option left out
        relevant = out is not None and out > 0                           # it matched the query: something was dropped
        gap = None if not relevant else ((last - out) / best if best > 0 else 0.0)
        d = ask(short)
        info.update(mode="shortlist", considered=k, unconsidered=n - k,
                    shortlist=[[labels[i], round(scores[i], 4)] for i in order[:k]],
                    left_out_best=None if out is None else round(out, 4), gap=None if gap is None else round(gap, 4))
        if gap is not None and gap < many.gap:
            why = (f"the shortlist left out {n - k} option(s), and the best of them scores close to the last one kept "
                   f"(gap {gap:.3f} < {many.gap:g}); would have answered {d.value!r}")
            d.escalate = why if d.escalate is None else f"{d.escalate}; {why}"
        d.extra["many"] = info
        return d

    pool, rounds, blocks = list(labels), 0, []                           # tournament
    while len(pool) > many.block:
        rounds += 1
        nxt = []
        for b in range(0, len(pool), many.block):
            blk = pool[b:b + many.block]
            if len(blk) == 1:                                            # a block of one goes on without a call
                nxt.append(blk[0])
                blocks.append({"round": rounds, "options": blk, "winner": blk[0], "confidence": None})
                continue
            d = ask(blk)
            nxt.append(d.value)
            blocks.append({"round": rounds, "options": blk, "winner": d.value, "confidence": round(float(d.conf), 6)})
        pool = nxt
    d = ask(pool)                                                        # the last round decides
    blocks.append({"round": rounds + 1, "options": pool, "winner": d.value, "confidence": round(float(d.conf), 6)})
    info.update(mode="tournament", considered=n, unconsidered=0, rounds=rounds + 1, blocks=blocks)
    d.extra["many"] = info
    return d
