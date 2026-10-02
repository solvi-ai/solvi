"""Customer support: approve a refund from the customer's email. The yes/no answer has no rule — it is learned from examples —
but a hard check "within 30 days" decides the answer whenever it fails, and the learned part cannot override it.
Fields are extracted from the email with phrase matching, each with a quote.

Run:  uv run python examples/04_refunds.py"""
from __future__ import annotations

import random
import re
from datetime import date, timedelta

from solvi import Answer, Catalog, Question, Quote, System
from solvi.show import show

cat = Catalog()
REASONS = {
    "defect": ["arrived broken", "is defective", "stopped working", "came damaged"],
    "does not fit": ["does not fit", "is too small", "is too big", "wrong size"],
    "changed mind": ["changed my mind", "no longer need it", "found it cheaper elsewhere"],
    "other": ["have a question about it", "want to discuss the order"],
}
USED = {True: ["I used it a few times", "after using it for a week", "I have worn it"],
        False: ["it is still sealed", "never opened", "unused, in original packaging"]}


@cat.extract
def order_date(doc):
    "order date"
    m = re.search(r"(?:ordered on|order date|placed on)\s*[:]?\s*(\d{4}-\d{2}-\d{2})", doc, flags=re.I)
    if not m:
        raise ValueError("no order date")
    return Quote(date.fromisoformat(m.group(1)), m.start(1), m.end(1))


@cat.extract
def reason(doc):
    "reason for the return, one of REASONS"
    for k, phrases in REASONS.items():
        for ph in phrases:
            i = doc.lower().find(ph)
            if i >= 0:
                return Quote(k, i, i + len(ph))
    return Quote("other", 0, 0)


@cat.extract
def used(doc):
    "whether the customer used the item"
    for v, phrases in USED.items():
        for ph in phrases:
            i = doc.lower().find(ph.lower())
            if i >= 0:
                return Quote(v, i, i + len(ph))
    return Quote(False, 0, 0, confidence=0.5)


@cat.fn
def days_since(order_date, today):
    return (today - order_date).days


@cat.check(hard=True, then={"refund": "no"})
def within_30_days(days_since):
    return days_since <= 30


@cat.fn
def email_words(doc):
    return len(doc.split())


QUESTIONS = [Question("refund", "Approve the refund?", Answer.yes_no(), requires=["within_30_days"])]
TODAY = date(2026, 9, 25)


def make(rng: random.Random, i: int):
    r = rng.choice(list(REASONS))
    u = rng.random() < 0.4
    days = rng.randint(1, 45)
    od = TODAY - timedelta(days=days)
    doc = (f"Hello, order #{5000 + i} {rng.choice(['ordered on', 'order date', 'placed on'])} {od}. The item "
           f"{rng.choice(REASONS[r])}; {rng.choice(USED[u])}. {rng.choice(['Please help.', 'Thanks.', 'Can I return it?'])}")
    ok = days <= 30 and (r == "defect" or (r == "does not fit" and not u))
    label = "yes" if ok else "no"
    if days <= 30 and rng.random() < 0.05:        # label noise only within 30 days (beyond that the hard check sets the answer)
        label = "no" if label == "yes" else "yes"
    return {"doc": doc, "today": TODAY}, {"refund": label, "within": days <= 30, "clean": "yes" if ok else "no"}


if __name__ == "__main__":
    rng = random.Random(0)
    system = System(cat, QUESTIONS)
    train = [make(rng, i) for i in range(400)]
    head = system.fit("refund", [(init, truth["refund"]) for init, truth in train])
    print(f"refund learned from {len(train)} emails; selected facts: {head.features}")
    test = [make(rng, 1000 + i) for i in range(400)]
    acc = sum(system.ask(init)["refund"].answer == truth["refund"] for init, truth in test) / len(test)
    late = [init for init, truth in test if not truth["within"]]
    forced = sum(system.ask(init)["refund"].status == "forced" for init in late)
    print(f"accuracy on 400 new emails: {acc:.3f}; late requests decided by the hard check: {forced}/{len(late)}")
    print("\n=== one email ===\n" + test[0][0]["doc"])
    show(system.ask(test[0][0]), cat)
