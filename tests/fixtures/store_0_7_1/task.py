"""A small catalog that solvi 0.7.1 and 0.8 both run unchanged — a typed input, a hard check, a rule and a decision part
on a stand-in decider — to prove that stores written by 0.7.1 still load, verify and replay (tests/test_store_0_7_1.py).
make.py wrote the stores in this folder with the 0.7.1 sources."""
import datetime as dt
import inspect

import numpy as np
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.core.deciders import DecideModel


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


def system(storage=None):
    """A fresh System (and a fresh model: teach changes the model it is given)."""
    model = DecideModel(Words(), {"format": "solvi_decide v2", "act": False})
    cat = Catalog()

    @cat.check(hard=True, then={"approve": "no"})
    def within_limit(amount: float, limit: float) -> bool:
        return amount <= limit

    @cat.fn
    def days_old(spent_on: dt.date) -> int:
        return (dt.date(2026, 10, 1) - spent_on).days

    @cat.rule("approve")
    def approve(within_limit, days_old) -> bool:
        return days_old <= 30

    category = model.decision("category", "What kind of expense is it?", "note", ["travel", "meals", "software"])
    questions = [Question("approve", "Approve the expense?", Answer.yes_no(), ["within_limit"]),
                 category.question(cat)]
    new = "input_model" in inspect.signature(System.__init__).parameters          # 0.8's name of inputs=
    typed = {"input_model" if new else "inputs": Expense}
    return System(cat, questions, storage=storage, **typed)


STATES = [
    {"amount": 120.0, "limit": 500.0, "spent_on": "2026-09-20", "note": "travel to the client, train tickets"},
    {"amount": 900.0, "limit": 500.0, "spent_on": "2026-09-25", "note": "software licence"},
    {"amount": 40.0, "limit": 500.0, "spent_on": "2026-06-01", "note": "meals with the team, meals"},
    {"amount": 300.0, "limit": 500.0, "spent_on": "2026-09-30", "note": "a thing"},
]
