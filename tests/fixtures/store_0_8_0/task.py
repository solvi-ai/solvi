"""A small catalog that solvi 0.8.0 and 0.9 both run unchanged — a typed input, a hard check, a rule and a decision part
on a stand-in decider, calibrated — to prove that stores, calibration files and fingerprints written by 0.8.0 still load,
verify and replay (tests/test_store_0_8_0.py). make.py wrote the files in this folder with the 0.8.0 sources."""
import datetime as dt
import os

import numpy as np
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.decide import DecideModel

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

    @cat.check(hard=True, then={"approve": "no"})
    def within_limit(amount: float, limit: float) -> bool:
        return amount <= limit

    @cat.fn
    def days_old(spent_on: dt.date) -> int:
        return (dt.date(2026, 10, 1) - spent_on).days

    @cat.rule("approve")
    def approve(within_limit, days_old) -> bool:
        return days_old <= 30

    questions = [Question("approve", "Approve the expense?", Answer.yes_no(), requires=["within_limit"]),
                 category_part(calibration).question(cat)]
    return System(cat, questions, storage=storage, input_model=Expense)


def calibrated(storage=None):
    """The System with the calibration file of this folder (`solvi replay ... --system task.py:calibrated`)."""
    return system(storage, os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json"))


STATES = [
    {"amount": 120.0, "limit": 500.0, "spent_on": "2026-09-20", "note": "travel to the client, train tickets"},
    {"amount": 900.0, "limit": 500.0, "spent_on": "2026-09-25", "note": "software licence"},
    {"amount": 40.0, "limit": 500.0, "spent_on": "2026-06-01", "note": "meals with the team, meals"},
    {"amount": 300.0, "limit": 500.0, "spent_on": "2026-09-30", "note": "a thing"},
]

# calibration examples: clear notes are right, notes naming two kinds are a coin flip for the stand-in decider
CALIBRATION = ([(f"{o} {o} expense number {i}", o) for i in range(20) for o in OPTIONS]
               + [(f"travel and meals, item {i}", "meals" if i % 2 else "travel") for i in range(20)])
