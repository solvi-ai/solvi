"""Rules first, a model second: a composition table answers what it derives, a model answers only what it refuses.

    rf = RulesFirst(table, fallback=decider_fallback(model))       # table: a CompositionTable (or chains → Derivation)
    a = rf.answer([("father", "sister")], options)                 # one query, one or several chains
    a.value, a.who, a.reason                                       # "aunt", "rules", "derived"
    a.to_record()                                                  # journal-ready: the path, the cells or the model output
    rf.replay(record).ok                                           # derives again / re-reads the model's recorded output
    rf.stats                                                       # answered by rules / by the model / refused, per reason

The rules. `rules` is a `CompositionTable` (its `derive_paths` is called with the query's chains) or any callable
`(chains) -> Derivation`. A "derived" derivation is the answer: the model is never called, the question is not even
rendered. A refusal ("disagree", "underivable", "identity") goes to the fallback when one is given; with no fallback
it stays a refusal (`value` None, `who` None) with the rules' reason.

The fallback. `fallback(question, options)` → `(answer, confidence)` (or `(answer, confidence, raw)` with something
JSON-able to keep in the record, or a plain answer, or None for "the model refuses"). The question is rendered from the
chains by `render(chains, head, tail)`; the default, `render_chains`, reads a chain the table's way — (father, sister)
is "B is the father of A. C is the sister of B. What is C to A?" (the answer: C is A's aunt); several chains of one
query share the head and the tail and get their own people in between. An answer that is not one of the options is not
an answer (`value` None, `note` says so); the model's output stays in the record. `decider_fallback(model, task)` adapts
a `DecideModel` (its `decide(question, task, options)`; the confidence is the decision's, the probabilities are kept as
the raw output).

The answer. `RulesFirstAnswer(value, who, reason, derivation, model_output, confidence, question, options, chains,
note)`: `who` is the path that answered ("rules" | "model" | None); `reason` is always the rules' reason ("derived" on
the rules path; why the model was asked on the model path); `confidence` is the model's (the rules give none: their
precision is measured, not claimed per answer). `to_record()` is plain JSON.

Replay. `replay(record)` derives the record's chains again and requires the same relation, reason, candidates and cells
(the rules path is deterministic); on the model path it also renders the question again (it must be the recorded one)
and re-reads the recorded model output through the same option check — the fallback is never called, so a model path
replays without the model (`Replay.from_record` is True). An edited record or a changed table is a problem.

Stats. `stats` → {"rules", "model", "refused", "by_reason": {reason: {path: n}}}; a model answer outside the options
or a model refusal counts as "refused". A copy: the counters change only by `answer`.

What is not done: the same dispatcher over `forward_chain` (whole-graph questions over a knowledge store), the text
setting (a story to chains), and gating the model's answer by its confidence (a model answer is taken as given)."""
from __future__ import annotations

import string
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ._table import CompositionTable, Derivation

PATHS = ("rules", "model", "refused")


def _names():
    """A, B, ..., Z, AA, AB, ... — the people of a rendered question."""
    letters = string.ascii_uppercase
    n = 0
    while True:
        k, s = n, ""
        while True:
            s = letters[k % 26] + s
            k = k // 26 - 1
            if k < 0:
                break
        yield s
        n += 1


def render_chains(chains: Sequence[Sequence[str]], head: str = "A", tail: str | None = None) -> str:
    """The default question for a query reached by `chains` (each a sequence of relation names, read the table's way:
    (r1, r2) is "B is the r1 of A. C is the r2 of B."). head: the first person's name; tail: the last person's (default:
    the name after every person in between). Several chains share the head and the tail; the people in between are new
    for each chain."""
    chains = [tuple(c) for c in chains]
    gen = _names()
    used = {head}
    if tail is not None:
        used.add(tail)

    def fresh():
        for n in gen:
            if n not in used:
                used.add(n)
                return n
    inner = [[fresh() for _ in range(len(c) - 1)] for c in chains]
    if tail is None:
        tail = fresh()
    parts = []
    for c, mid in zip(chains, inner):
        people = [head] + mid + [tail]
        parts += [f"{people[i + 1]} is the {r} of {people[i]}." for i, r in enumerate(c)]
    return " ".join(parts + [f"What is {tail} to {head}?"])


def decider_fallback(model, task: str = "What is the relation?", **spec) -> Callable:
    """A fallback over a DecideModel (anything with `decide(text, task, options, **spec)` → Decision): → (value,
    confidence, {"probs": the decision's probabilities})."""
    def fallback(question, options):
        d = model.decide(question, task, list(options), **spec)
        return d.value, float(d.conf), {"probs": {str(k): float(v) for k, v in (d.probs or {}).items()}}
    fallback.model = model
    fallback.task = task
    return fallback


def _derivation_dict(d: Derivation | None) -> dict | None:
    if d is None:
        return None
    return {"relation": d.relation, "reason": d.reason,
            "candidates": {str(k): int(v) for k, v in sorted(d.candidates.items())},
            "cells": [[c.left, c.right, c.relation] for c in d.cells]}


@dataclass(frozen=True)
class RulesFirstAnswer:
    """See the module docstring."""
    value: Any
    who: str | None
    reason: str
    derivation: Derivation | None = None
    model_output: dict | None = None
    confidence: float | None = None
    question: str | None = None
    options: tuple = ()
    chains: tuple = ()
    note: str = ""
    by: str = "relations:RulesFirst"
    extra: dict = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.value is not None

    def to_record(self) -> dict:
        """Plain JSON: the path (`who`), the value, the rules' reason, the derivation (cells as [left, right, relation]),
        the question and the model's output (model path), the options, the chains, the confidence, a note."""
        return {"by": self.by, "who": self.who, "value": self.value, "reason": self.reason,
                "chains": [list(c) for c in self.chains], "options": list(self.options),
                "derivation": _derivation_dict(self.derivation), "question": self.question,
                "model_output": self.model_output, "confidence": self.confidence, "note": self.note}


def _as_chains(chains) -> tuple:
    chains = list(chains) if not isinstance(chains, str) else [chains]
    if chains and all(isinstance(x, str) for x in chains):
        chains = [chains]                                   # one chain given as a sequence of relation names
    out = tuple(tuple(c) for c in chains)
    if not out or any(not c for c in out):
        raise ValueError("a query needs at least one non-empty chain")
    return out


def _read_output(out) -> tuple[Any, float | None, Any]:
    """A fallback's return → (answer, confidence, raw)."""
    if out is None:
        return None, None, None
    if isinstance(out, tuple):
        ans = out[0] if len(out) > 0 else None
        conf = out[1] if len(out) > 1 else None
        raw = out[2] if len(out) > 2 else None
        return ans, (None if conf is None else float(conf)), raw
    return out, None, None


class RulesFirst:
    """See the module docstring. rules: a CompositionTable or a callable (chains) → Derivation; fallback: (question,
    options) → (answer, confidence) or None; render: (chains, head, tail) → question (default `render_chains`); by: the
    name written into every record."""

    def __init__(self, rules, *, fallback: Callable | None = None, render: Callable | None = None,
                 by: str = "relations:RulesFirst"):
        if isinstance(rules, CompositionTable):
            self._derive = rules.derive_paths
        elif callable(rules):
            self._derive = rules
        else:
            raise TypeError("rules is a CompositionTable or a callable (chains) -> Derivation")
        if fallback is not None and not callable(fallback):
            raise TypeError("fallback is a callable (question, options) -> (answer, confidence)")
        self.rules, self.fallback, self.render, self.by = rules, fallback, render or render_chains, by
        self._counts = Counter()
        self._by_reason = defaultdict(Counter)

    def _question(self, chains, head, tail):
        return self.render(list(chains), head, tail)

    def _model_answer(self, out, options):
        """The recorded model output → (value, note): the answer when it is one of the options."""
        ans, note = out.get("answer"), ""
        if ans is None:
            return None, "the model refused"
        if options and ans not in options:
            return None, f"the model's answer {ans!r} is not one of the options"
        return ans, note

    def answer(self, chains, options: Sequence[str] = (), *, head: str = "A", tail: str | None = None) -> RulesFirstAnswer:
        """Rules first: the derivation over `chains` (one chain, or several chains of one query); on a refusal the
        fallback answers the rendered question among `options`. → RulesFirstAnswer."""
        chains = _as_chains(chains)
        options = tuple(options)
        d = self._derive(chains)
        if d.reason == "derived" and d.relation is not None:
            a = RulesFirstAnswer(d.relation, "rules", "derived", d, None, None, None, options, chains, by=self.by)
            self._count("rules", "derived")
            return a
        if self.fallback is None:
            self._count("refused", d.reason)
            return RulesFirstAnswer(None, None, d.reason, d, None, None, None, options, chains,
                                    "refused by the rules; no fallback", by=self.by)
        q = self._question(chains, head, tail)
        ans, conf, raw = _read_output(self.fallback(q, list(options)))
        out = {"answer": ans, "confidence": conf}
        if raw is not None:
            out["raw"] = raw
        value, note = self._model_answer(out, options)
        self._count("model" if value is not None else "refused", d.reason)
        return RulesFirstAnswer(value, "model", d.reason, d, out, conf if value is not None else None, q, options,
                                chains, note, by=self.by)

    def _count(self, path, reason):
        self._counts[path] += 1
        self._by_reason[reason][path] += 1

    @property
    def stats(self) -> dict:
        return {**{p: self._counts[p] for p in PATHS},
                "by_reason": {r: dict(c) for r, c in sorted(self._by_reason.items())}}

    def replay(self, record: dict, *, head: str = "A", tail: str | None = None):
        """Replay a recorded answer (a `to_record()` dict) without calling the model → solvi.core.specialist.Replay
        (ok, problems, from_record=True)."""
        from ...core.specialist import Replay
        problems = []
        if record.get("by") != self.by:
            problems.append(f"recorded by {record.get('by')!r}, replayed by {self.by!r}")
        try:
            chains = _as_chains(record.get("chains") or [])
        except ValueError as e:
            return Replay(False, [f"the record's chains: {e}"])
        options = tuple(record.get("options") or ())
        d = self._derive(chains)
        if _derivation_dict(d) != record.get("derivation"):
            problems.append(f"the rules derive otherwise now: {_derivation_dict(d)} vs the recorded "
                            f"{record.get('derivation')}")
        who = record.get("who")
        if d.reason == "derived" and d.relation is not None:
            want = {"who": "rules", "value": d.relation, "reason": "derived", "model_output": None, "question": None}
        elif who is None:
            want = {"who": None, "value": None, "reason": d.reason, "model_output": None}
        else:
            out = record.get("model_output")
            if not isinstance(out, dict):
                return Replay(False, problems + ["a model-path record without the model's output"])
            value, _ = self._model_answer(out, options)
            want = {"who": "model", "value": value, "reason": d.reason,
                    "question": self._question(chains, head, tail),
                    "confidence": out.get("confidence") if value is not None else None}
        for k, v in want.items():
            if record.get(k) != v:
                problems.append(f"{k}: the record says {record.get(k)!r}, the replay gives {v!r}")
        return Replay(not problems, problems, from_record=True)


__all__ = ["RulesFirst", "RulesFirstAnswer", "decider_fallback", "render_chains"]
