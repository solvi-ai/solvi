"""solvi — a catalog of functions and checks, questions with typed answers.

A part's contract comes from its signature: argument names are the facts it reads, the function name is the fact it sets.
Part kinds: extract (pull a value from text, with a quote), fn (computation), check (test → bool), rule (answer to a question)."""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Quote:
    """A value extracted from text, with its location (doc[start:end] is the supporting quote)."""
    value: Any
    start: int
    end: int
    source: str = "doc"
    confidence: float = 1.0


@dataclass
class AnswerType:
    kind: str                       # yes_no | choice | ordinal | multi
    options: list
    descriptions: dict = field(default_factory=dict)   # option -> what it means (for people and for zero-shot scoring)

    def normalize(self, v):
        if self.kind == "multi":
            vals = [v] if isinstance(v, str) else list(v or [])
            bad = [x for x in vals if x not in self.options]
            if bad:
                raise ValueError(f"answer {bad!r} not among {self.options}")
            return tuple(o for o in self.options if o in set(vals))      # canonical: in option order, no duplicates
        if self.kind == "yes_no" and isinstance(v, bool):
            return "yes" if v else "no"
        if v not in self.options:
            raise ValueError(f"answer {v!r} is not one of {self.options}")
        return v

    def rank(self, v):
        """Position of an ordinal (or choice) answer: 0 for the first option."""
        return self.options.index(v)


def _opts(options):
    if isinstance(options, dict):
        return list(options), dict(options)
    return list(options), {}


class Answer:
    @staticmethod
    def yes_no() -> AnswerType:
        return AnswerType("yes_no", ["yes", "no"])

    @staticmethod
    def choice(options) -> AnswerType:
        """One of the options. `options` is a list, or a dict {option: description}."""
        return AnswerType("choice", *_opts(options))

    @staticmethod
    def ordinal(levels) -> AnswerType:
        """One of ordered levels, lowest first (e.g. ["low", "medium", "high"]). A learned head answers with the median of its
        distribution instead of the most likely level, so it never jumps over the middle."""
        return AnswerType("ordinal", *_opts(levels))

    @staticmethod
    def multi(options) -> AnswerType:
        """Any subset of the options (possibly empty), returned as a tuple in option order. A rule may return a list or set."""
        return AnswerType("multi", *_opts(options))


@dataclass
class Question:
    name: str
    text: str
    answer: AnswerType
    checkpoints: list = field(default_factory=list)   # parts required in every flow for this question
    uses: list | None = None                          # hint to the strategist: which facts matter (when there is no rule or fit)


@dataclass
class Part:
    kind: str                       # extract | fn | check | rule
    name: str                       # fact it sets (for rule: "answer:<question>")
    inputs: list
    func: Callable
    doc: str = ""
    hard: bool = False              # check only: hard check
    then: dict = field(default_factory=dict)   # check only: {question: answer} when the check is false
    question: str | None = None     # rule only


class Catalog:
    """Everything the system can do; the strategist decides which parts each question needs."""

    def __init__(self):
        self.parts: dict[str, Part] = {}
        self.rules: dict[str, Part] = {}
        self.constraints: dict[str, Part] = {}     # rules between answers of different questions

    def _add(self, kind, f, **kw):
        sig = inspect.signature(f)
        p = Part(kind=kind, name=f.__name__, inputs=list(sig.parameters), func=f, doc=(f.__doc__ or "").strip(), **kw)
        if kind == "rule":
            p.name = "answer:" + p.question
            self.rules[p.question] = p
        elif kind == "constraint":
            self.constraints[p.name] = p
        else:
            if p.name in self.parts:
                raise ValueError(f"part {p.name} is already in the catalog")
            self.parts[p.name] = p
        return f

    def extract(self, f):
        return self._add("extract", f)

    def fn(self, f):
        return self._add("fn", f)

    def check(self, f=None, *, hard=False, then=None):
        if f is None:
            return lambda g: self._add("check", g, hard=hard, then=then or {})
        return self._add("check", f)

    def rule(self, question):
        return lambda f: self._add("rule", f, question=question)

    def constraint(self, f):
        """A rule between answers: argument names are question names, it returns True when the answers fit together
        (e.g. `def unsafe_if_harm(verdict, harm): return harm == "none" or verdict == "unsafe"`). When learned answers break it,
        solvi picks the most probable combination that satisfies every constraint; answers from rules and hard checks stay."""
        return self._add("constraint", f)

    def producer(self, fact):
        return self.parts.get(fact)
