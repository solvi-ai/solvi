"""Catalogs that solvi 0.9.0 and later versions run unchanged, to prove that stores, calibration files and fingerprints
written by 0.9.0 still load, verify and replay after the 1.0 layout moves the modules (tests/test_store_0_9_0.py).
make.py wrote the files in this folder with the 0.9.0 sources.

system(): a typed input, a hard check, a function, a callable object, a rule, the answer primitives (Maybe[Span],
Rank, Estimate, a Claim with evidence) and a decision part on a stand-in decider, calibrated (calibration.json).
dispatch(): solvi.build — a rule as System 1 under a promise, a slow path, every decision dispatched and stored."""
import datetime as dt
import os
import random
import re
from typing import Literal

import numpy as np
from pydantic import BaseModel

from solvi import Answer, Catalog, Claim, Estimate, Maybe, Question, Quote, Rank, Span, System, Unknown
from solvi.core.deciders import DecideModel

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIONS = ["travel", "meals", "software"]


class Words:
    """A stand-in decider: a keyword count per option is its logit."""
    tag = "words"

    def fingerprint(self):
        return "words-v1"

    def logits(self, items):
        out = []
        for it in items:
            t = it.text.lower()
            out.append({"logits": np.array([3.0 * t.count(o.lower()) for o in it.options])})
        return out


class Expense(BaseModel):
    amount: float
    limit: float
    spent_on: dt.date
    note: str


class WordCount:
    """A callable object as a part."""
    __name__ = "n_words"

    def __call__(self, note: str) -> int:
        return len(note.split())


def category_part(calibration=None):
    """The decision part (a fresh model: teach changes the model it is given), with a calibration file applied."""
    model = DecideModel(Words(), {"format": "solvi_decide v2", "act": False})
    part = model.decision("category", "What kind of expense is it?", "note", OPTIONS)
    if calibration is not None:
        part.load_calibration(calibration)
    return part


def system(storage=None, calibration=None):
    """A fresh System."""
    cat = Catalog()
    cat.fn(WordCount())

    @cat.check(hard=True, then={"approve": "no"})
    def within_limit(amount: float, limit: float) -> bool:
        return amount <= limit

    @cat.fn
    def days_old(spent_on: dt.date) -> int:
        return (dt.date(2026, 10, 1) - spent_on).days

    @cat.rule("approve")
    def approve(within_limit, days_old) -> bool:
        return days_old <= 30

    @cat.rule("receipt_total")
    def receipt_total(note: str) -> Maybe[Span[float, "note"]]:  # noqa: F821 — "note" is the source fact, not a name
        m = re.search(r"total:?\s*(\d+(?:\.\d+)?)", note)
        return Quote(m.group(1), m.start(1), m.end(1), source="note") if m else Unknown

    @cat.rule("channels")
    def channels(note: str) -> Rank[Literal["email", "phone", "letter"], 2]:
        return {c: note.count(c) - 0.1 * i for i, c in enumerate(("email", "phone", "letter"))}

    @cat.rule("refund_days")
    def refund_days(n_words: int) -> Estimate[0, 3, 7]:
        return n_words

    @cat.rule("team_trip")
    def team_trip(note: str) -> bool:
        m = re.search(r"team|client", note)
        return Claim(bool(m), evidence=[m.group(0)] if m else [])

    questions = [Question("approve", "Approve the expense?", Answer.yes_no(), requires=["within_limit"]),
                 Question("receipt_total", "The total on the receipt?"), Question("channels", "How to reach them?"),
                 Question("refund_days", "Days to the refund?"),
                 Question("team_trip", "Is it for the team or a client?", require_evidence=True),
                 category_part(calibration).question(cat)]
    return System(cat, questions, storage=storage, input_model=Expense)


def calibrated(storage=None):
    """The System with the calibration file of this folder (`solvi replay ... --system task.py:calibrated`)."""
    return system(storage, os.path.join(HERE, "calibration.json"))


STATES = [
    {"amount": 120.0, "limit": 500.0, "spent_on": "2026-09-20", "note": "travel to the client, train tickets, total 120"},
    {"amount": 900.0, "limit": 500.0, "spent_on": "2026-09-25", "note": "software licence, email me"},
    {"amount": 40.0, "limit": 500.0, "spent_on": "2026-06-01", "note": "meals with the team, meals, total: 40.00"},
    {"amount": 300.0, "limit": 500.0, "spent_on": "2026-09-30", "note": "a thing, phone or email"},
]

# calibration examples: clear notes are right, notes naming two kinds are a coin flip for the stand-in decider
CALIBRATION = ([(f"{o} {o} expense number {i}", o) for i in range(20) for o in OPTIONS]
               + [(f"travel and meals, item {i}", "meals" if i % 2 else "travel") for i in range(20)])


# --- solvi.build: a rule under a promise, a slow path, the dispatcher's store
def _refund(rng):
    s = {"amount": round(rng.uniform(5, 400), 2), "days_late": rng.choice([0, 0, 1, 3, 6])}
    return s, "yes" if s["amount"] < 150 or s["days_late"] > 2 else "no"


EXAMPLES = [_refund(random.Random(i)) for i in range(1000)]
ASKED = [_refund(random.Random(1000 + i))[0] for i in range(30)]


def slow(state):
    """The slow path: the policy a person applies."""
    return "yes" if state["amount"] < 150 or state["days_late"] > 2 else "no"


def dispatch(storage=None):
    """A fresh AutoSystem (solvi.build) over the refund rule."""
    from solvi.auto import build
    cat = Catalog()

    @cat.fn
    def small(amount: float) -> bool:
        return amount < 150

    @cat.fn
    def share_left(amount: float) -> float:
        return round(1 - amount / 400, 3)

    @cat.rule("refund")
    def refund(small: bool, days_late: int) -> bool:
        return small or days_late > 4                   # wrong when three days late: the slow path knows better

    return build(Question("refund", "Refund without asking a person?", Answer.yes_no()), EXAMPLES, catalog=cat,
                 max_risk=0.06, slow=slow, storage=storage)
