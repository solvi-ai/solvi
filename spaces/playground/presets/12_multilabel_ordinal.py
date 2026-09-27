"""Multi-label + ordinal answers (solvi 0.3+): a support ticket gets a set of tags and a priority.

- `tags` is an Answer.multi: any subset of the options, returned as a tuple in option order (empty = no tag). Here plain
  keyword rules give it, so every tag cites the words that triggered it.
- `priority` is an Answer.ordinal: low < normal < high < urgent. It has no rule; `setup(system)` learns it with fit_fast from
  60 synthetic tickets in a few milliseconds. An ordinal head answers with the MEDIAN of its distribution, not the most likely
  level, so a split between "normal" and "urgent" gives "high" instead of a jump. The Audit panel shows the probabilities.
- A constraint ties the two: an outage is never low or normal priority. If the learned priority breaks it, joint decoding
  repairs it and says so.

Try: remove "down" from the text, change the customer tier, or set hours_waiting to 40."""
import random
import re

from solvi import Answer, Catalog, Question

cat = Catalog()
TAG_WORDS = {"billing": r"\b(invoice|charged?|refund|payment)\b", "bug": r"\b(error|crash(es|ed)?|bug|broken)\b",
             "account": r"\b(password|login|log in|2fa|locked)\b", "outage": r"\b(down|outage|unavailable)\b"}
TIERS = {"free": 0, "pro": 1, "enterprise": 2}


@cat.fn
def matched(text):
    """which tag words occur in the ticket"""
    return {t: m.group() for t, p in TAG_WORDS.items() for m in [re.search(p, text, re.I)] if m}


@cat.fn
def n_tags(matched):
    return len(matched)


@cat.fn
def is_outage(matched):
    return int("outage" in matched)


@cat.fn
def tier_rank(tier):
    return TIERS[tier]


@cat.rule("tags")
def tags(matched):
    return list(matched)


@cat.constraint
def outage_is_high(tags, priority):
    """an outage is at least high priority"""
    return "outage" not in tags or priority in ("high", "urgent")


QUESTIONS = [
    Question("tags", "Which teams should see the ticket?", Answer.multi(["billing", "bug", "account", "outage"])),
    Question("priority", "Priority", Answer.ordinal(["low", "normal", "high", "urgent"]),
             uses=["n_tags", "is_outage", "tier_rank", "hours_waiting"]),
]


def synthetic(rng):
    words = rng.sample(["invoice", "crash", "password", "down", "hello"], rng.randint(1, 3))
    tier, hours = rng.choice(list(TIERS)), rng.randint(0, 48)
    score = 2 * ("down" in words) + TIERS[tier] + (hours > 24) + (len(words) > 2) + rng.choice([-1, 0, 0, 1])
    level = ["low", "normal", "high", "urgent"][max(0, min(3, score - 1))]
    return {"text": " ".join(words), "tier": tier, "hours_waiting": hours}, level


def setup(system):
    """optional hook, run once per code version: learn the priority from 60 labelled tickets"""
    rng = random.Random(7)
    system.fit_fast("priority", [synthetic(rng) for _ in range(60)],
                    features=["n_tags", "is_outage", "tier_rank", "hours_waiting"])
