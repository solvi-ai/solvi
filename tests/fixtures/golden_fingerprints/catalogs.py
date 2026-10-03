"""Catalogs whose fingerprints are pinned (tests/test_golden_fingerprints.py): the kinds of parts the gallery, the
examples and the stand do not all cover — every typed answer (Maybe, Span, Scale, Rank, Estimate, Bins, NotStated),
pydantic and Enum types, a callable object as a part, decision parts registered as functions and as questions, a
cascade and a vote, an LLM decider (built, never called), a fitted answer head, a learned rule list, a guarantee and
solvi.build. Loaded under the module name "golden_catalogs": the module name is part of a type's fingerprint.

Edit this file and the pinned fingerprints change: rewrite them with `python tests/test_golden_fingerprints.py --write`
and say why in the commit."""
import enum
import random
import re
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel

from solvi import Answer, Bins, Catalog, Claim, Estimate, Maybe, Question, Quote, Rank, Scale, Span, System, Unknown
from solvi.core import NotStated
from solvi.core.deciders import DecideModel
from solvi.core.deciders.combine import Cascade, Vote

LEVELS = ["low", "medium", "high"]


class Severity(str, enum.Enum):
    minor = "minor"
    major = "major"


class Order(BaseModel):
    amount: float
    currency: Literal["EUR", "USD"]
    note: str


class Words:
    """A stand-in decider for every kind: keyword logits, a "not stated" logit, a pointer at the token after a colon."""
    model_id = "golden/words"
    META = {"format": "solvi_decide v2", "subformat": "l14g typed v2",
            "multi_question": {"layout": "block", "max_questions": 6}, "temperature": {"choice": 1.0}}

    def __init__(self, tag="v1"):
        self.tag = tag

    def fingerprint(self):
        return f"golden-words-{self.tag}"

    def _one(self, it):
        low = it.text.lower()
        o = {"act": 2.0, "unknown": -3.0 if ":" in low else 3.0}
        o["logits"] = np.zeros(0) if it.mode == "span" else \
            np.array([1.0 * low.count(str(x).lower()) for x in it.options] if it.options else [0.5])
        if it.pointer:
            toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", it.text)]
            st = np.array([4.0 if i and it.text[toks[i - 1][1] - 1] == ":" else -4.0 for i in range(len(toks))])
            o["pointer"] = {"start": st, "end": st.copy(), "null": [0.0, 0.0], "offsets": toks}
        return o

    def logits(self, items):
        return [self._one(it) for it in items]


class Scorer:
    """A callable object as a catalog part: identified by its type (its weights would be a model's fingerprint)."""
    __name__ = "n_words"

    def __call__(self, note: str) -> int:
        return len(note.split())


def model(tag="v1"):
    return DecideModel(Words(tag), Words.META)


def typed_rules():
    """Rules of every typed answer, a pydantic input, an Enum, a callable object, a quote, a claim, a hard check with
    then, a constraint."""
    cat = Catalog()
    cat.fn(Scorer())

    @cat.fn
    def order(amount: float, currency: Literal["EUR", "USD"], note: str) -> Order:
        return Order(amount=amount, currency=currency, note=note)

    @cat.extract
    def amount_text(note: str) -> Quote:
        m = re.search(r"\d+(?:\.\d+)?", note)
        return Quote(m.group(0), m.start(), m.end()) if m else None

    @cat.check(hard=True, then={"severity": "major"})
    def small(order: Order) -> bool:
        return order.amount < 1000

    @cat.rule("paid")
    def paid(note: str) -> Maybe[bool]:
        return True if "paid" in note else Unknown

    @cat.rule("total")
    def total(note: str) -> Maybe[Span[float, "note"]]:  # noqa: F821 — "note" is the source fact, not a name
        m = re.search(r"total:\s*(\S+)", note)
        return Quote(m.group(1), m.start(1), m.end(1), source="note") if m else Unknown

    @cat.rule("level")
    def level(n_words: int) -> Scale[Literal["low", "medium", "high"]]:
        return LEVELS[min(2, n_words // 5)]

    @cat.rule("level_num")
    def level_num(n_words: int) -> Scale[1, 2, 3]:
        return min(3, 1 + n_words // 5)

    @cat.rule("channels")
    def channels(note: str) -> Rank[Literal["email", "phone", "letter"], 2]:
        return {c: note.count(c) - 0.1 * i for i, c in enumerate(("email", "phone", "letter"))}

    @cat.rule("days")
    def days(n_words: int) -> Estimate[0, 3, 7]:
        return n_words

    @cat.rule("weeks")
    def weeks(n_words: int) -> Annotated[float, Bins([0, 2, 4], coverage=0.9, unit="weeks")]:
        return n_words / 7

    @cat.rule("severity")
    def severity(order: Order) -> Severity:
        return Severity.major if order.amount > 500 else Severity.minor

    @cat.rule("refund")
    def refund(order: Order, paid) -> bool:
        return Claim(order.amount < 100, evidence=[])

    @cat.fn
    def maybe_text(note: str) -> str | NotStated:
        return note[:10] or Unknown

    @cat.constraint
    def major_is_not_refunded(severity, refund):
        return severity != "major" or refund == "no"

    names = ["paid", "total", "level", "level_num", "channels", "days", "weeks", "severity", "refund"]
    return System(cat, [Question(n, f"{n}?") for n in names], input_model=Order)


def decisions():
    """Decision parts of every kind: as questions, one registered as a function (a callable object), a cascade, a
    vote."""
    m, m2 = model(), model("v2")
    cat = Catalog()
    cat.fn(m.decision("urgent", "Is it urgent?", "note", bool))
    qs = [m.decision("kind", "What kind?", "note", ["travel", "meals", "software"]).question(cat),
          m.decision("tags", "Which tags?", "note", list[Literal["a", "b", "c"]]).question(cat),
          m.decision("signed", "Signed?", "note", Maybe[bool]).question(cat),
          m.decision("amount", "Amount?", "note", Maybe[Span[float, "note"]]).question(cat),
          m.decision("level", "Level?", "note", Scale[Literal["low", "medium", "high"]]).question(cat),
          m.decision("channels", "Channels?", "note", Rank[Literal["email", "phone", "letter"], 2]).question(cat),
          m.decision("days", "Days?", "note", Maybe[Estimate[0, 3, 7, 14]]).question(cat),
          m.decision("damaged", "Damaged?", "note", bool, evidence=True).question(cat, require_evidence=True)]
    team = ["billing", "tech"]
    qs.append(Cascade([m.decision("team", "Which team?", "note", team),
                       m2.decision("team", "Which team?", "note", team)]).question(cat, "team", "Which team?"))
    qs.append(Vote([m.decision("route", "Which route?", "note", team),
                    m2.decision("route", "Which route?", "note", team)]).question(cat, "route", "Which route?"))
    return System(cat, qs)


def llm_decider():
    """An LLM decider part (an OpenAI-compatible server; built, never called)."""
    from solvi.core.deciders.llm import llm
    m = llm("http://127.0.0.1:9/v1", "golden-model", seed=1)
    cat = Catalog()
    return System(cat, [m.decision("kind", "What kind?", "note", ["travel", "meals"]).question(cat)])


def _examples(n=300, seed=3):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = {"amount": round(rng.uniform(1, 900), 2), "days": rng.randint(0, 9)}
        out.append((s, "yes" if s["amount"] < 300 or s["days"] > 6 else "no"))
    return out


def _numbers():
    cat = Catalog()

    @cat.fn
    def large(amount: float) -> bool:
        return amount >= 300

    @cat.fn
    def late(days: int) -> bool:
        return days > 6

    @cat.fn
    def small_share(amount: float) -> float:
        return round(1 - amount / 900, 3)
    return cat


def head():
    """A fitted answer head (solvi.core.deciders.heads.FastHead) with its weights rounded — as a head restored from a file carries
    fixed numbers: fitted ones differ in the last bits between numpy versions — and a guarantee on a computed fact."""
    s = System(_numbers(), [Question("ok", "OK?", Answer.yes_no())])
    ex = _examples()
    h = s.fit("ok", ex, features=["large", "late"])
    h.W, h._fp = np.round(h.W, 4), None
    s.guarantee("ok", ex, max_error=0.2, method="empirical", signal="small_share")
    return s


def rule_list():
    """A learned rule list installed as the answer rule."""
    s = System(_numbers(), [Question("ok", "OK?", Answer.yes_no())])
    s.learn_rule("ok", _examples(), ["large", "late"])
    return s


def auto_built():
    """solvi.build over a catalog whose rule answers the question: the promise on a computed fact, a slow path."""
    from solvi.solutions.decisions import build
    cat = _numbers()

    @cat.rule("ok")
    def ok(large: bool, late: bool) -> bool:
        return not large

    def slow(state):
        return "yes" if state["amount"] < 300 or state["days"] > 6 else "no"
    return build(Question("ok", "OK?", Answer.yes_no()), _examples(400), catalog=cat, max_risk=0.05, slow=slow)


SYSTEMS = {"typed_rules": typed_rules, "decisions": decisions, "llm_decider": llm_decider, "head": head,
           "rule_list": rule_list, "auto_built": auto_built}
