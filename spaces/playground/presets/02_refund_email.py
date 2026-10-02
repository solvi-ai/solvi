"""Customer support: approve a refund from the customer's e-mail (after examples/04_refunds.py).
Fields are pulled from the text with regex / phrase matching; each value keeps a Quote with its offsets.
A hard check "within 30 days" overrides the rule. Try: change the order date to 2026-08-01."""
import re
from datetime import date

from solvi import Answer, Catalog, Question, Quote

cat = Catalog()

REASONS = {
    "defect": ["arrived broken", "is defective", "stopped working", "came damaged"],
    "does not fit": ["does not fit", "is too small", "is too big", "wrong size"],
    "changed mind": ["changed my mind", "no longer need it", "found it cheaper elsewhere"],
}
USED = {True: ["used it a few times", "after using it", "i have worn it"],
        False: ["still sealed", "never opened", "unused"]}


def prepare(state):
    state["today"] = date.fromisoformat(state["today"])
    return state


@cat.extract
def order_date(doc):
    "order date, quoted from the e-mail"
    m = re.search(r"(?:ordered on|order date|placed on)\s*:?\s*(\d{4}-\d{2}-\d{2})", doc, flags=re.I)
    if not m:
        raise ValueError("no order date in the e-mail")
    return Quote(date.fromisoformat(m.group(1)), m.start(1), m.end(1))


@cat.extract
def reason(doc):
    "reason for the return"
    low = doc.lower()
    for k, phrases in REASONS.items():
        for ph in phrases:
            i = low.find(ph)
            if i >= 0:
                return Quote(k, i, i + len(ph))
    return Quote("other", 0, 0, confidence=0.5)


@cat.extract
def used(doc):
    "did the customer use the item?"
    low = doc.lower()
    for v, phrases in USED.items():
        for ph in phrases:
            i = low.find(ph)
            if i >= 0:
                return Quote(v, i, i + len(ph))
    return Quote(False, 0, 0, confidence=0.5)      # not said: assume unused, but with low confidence


@cat.fn
def days_since(order_date, today):
    return (today - order_date).days


@cat.check(hard=True, then={"refund": "no"})
def within_30_days(days_since):
    """hard: no refunds after 30 days, whatever the reason"""
    return days_since <= 30


@cat.fn
def email_words(doc):          # not needed by any question: skipped
    return len(doc.split())


@cat.rule("refund")
def refund(reason, used):
    return reason == "defect" or (reason == "does not fit" and not used)


@cat.rule("tone")
def tone(doc):
    angry = any(w in doc.lower() for w in ("unacceptable", "furious", "worst", "!!!"))
    return "apologetic" if angry else "friendly"


QUESTIONS = [
    Question("refund", "Approve the refund?", Answer.yes_no(), requires=["within_30_days"]),
    Question("tone", "Tone of the reply?", Answer.choice(["friendly", "apologetic"])),
]
