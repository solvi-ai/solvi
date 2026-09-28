"""The honesty suite's core catalog: every part is plain code or a stub proposer, so the core set runs without model files.

One invoice / support-message text (`doc`) and a few given facts; each question exercises one honesty mechanism:

  paid      Maybe[bool] with a quote — "not stated" is an answer, distinct from "no" and from an abstention
  total     Maybe[Span[float]] — the total as a quote; two different totals in the text (conflicting sources) abstain
  currency  a choice of EUR / USD — a code outside the options (GBP) abstains instead of being forced into one
  method    a stub model proposer with an act signal and a quote: it escalates when unsure, and a quote it paraphrases
            instead of copying from the text is rejected (grounding)
  team      a decider (solvi.decide) over a keyword stub scorer with escalate_below — acts when sure, escalates when not;
            it can also be confidently wrong (the suite measures that, it does not hide it)
  approve   a hard check that raises on a malformed amount — the answer abstains, never "passed"

Loaded by solvi.honesty (`system()`), the way the gallery runners load a task.py."""
import re

import numpy as np

from solvi import Answer, Catalog, Claim, Decision, Maybe, Question, Quote, Span, System, Unknown
from solvi.decide import DecideModel
from solvi.provenance import ESCALATED

cat = Catalog()


# ------------------------------------------------------------------------------------- paid: not stated
PAID = (("not paid", False), ("unpaid", False), ("paid in full", True), ("payment received", True))


@cat.rule("paid")
def paid(doc: str) -> Maybe[bool]:
    """Yes / no with the phrase that says so; Unknown when the text does not say (the word "paid" alone is not enough)."""
    low = doc.lower()
    for phrase, value in PAID:
        i = low.find(phrase)
        if i >= 0:
            return Claim(value, evidence=[Quote(doc[i:i + len(phrase)], i, i + len(phrase))])
    return Unknown


# ------------------------------------------------------------------------------------- total: conflicting sources
TOTAL = re.compile(r"total(?: due)?:?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@cat.rule("total")
def total(doc: str) -> Maybe[Span[float]]:
    """The amount after "Total"; two different totals → None (the sources conflict: abstain); none → Unknown."""
    found = [(m.group(1), m.start(1), m.end(1)) for m in TOTAL.finditer(doc)]
    if not found:
        return Unknown
    if len({float(v) for v, _, _ in found}) > 1:
        return None
    v, s, e = found[0]
    return Quote(v, s, e)


# ------------------------------------------------------------------------------------- currency: outside the options
@cat.rule("currency")
def currency(doc):
    """The first ISO code in the text, whatever it is: the question's closed set decides whether it is an answer."""
    m = re.search(r"\b(EUR|USD|GBP|CHF|JPY)\b", doc)
    return m.group(1) if m else None


# ------------------------------------------------------------------------------------- method: act / escalate, quotes
class StubReader:
    """Stands in for a generative reader: picks the payment method by keyword and cites a canonical phrase for it — which
    it paraphrases when the text words it differently (the quote is then not in the text). Without a keyword its act
    signal is low: it escalates instead of guessing."""
    model_id, version = "tests/stub-reader", "1"
    CUES = {"card": ("card",), "transfer": ("transfer", "wire"), "cash": ("cash",)}
    PHRASE = {"card": "paid by card", "transfer": "paid by bank transfer", "cash": "paid in cash"}

    def __call__(self, doc):
        low = doc.lower()
        hits = {m: sum(low.count(c) for c in cues) for m, cues in self.CUES.items()}
        best = max(hits, key=hits.get)
        n = sum(hits.values())
        probs = {m: (hits[m] + 0.2) / (n + 0.6) for m in hits}
        if hits[best] == 0:
            return Decision(best, probs, escalate=f"{ESCALATED}: act 0.10 < 0.50; would have answered {best!r}")
        return Decision(best, probs, evidence=[self.PHRASE[best]])


READER = StubReader()


@cat.rule("method", model=READER)
def method(doc):
    return READER(doc)


# ------------------------------------------------------------------------------------- team: a decider that acts or escalates
KW = {"billing": ["charged", "refund", "invoice"], "technical": ["crash", "error", "bug"],
      "shipping": ["parcel", "delivery", "tracking"]}


class KeywordScorer:
    """A stand-in for the decider's network: keyword counts as logits (as tests/test_decide.py's FakeScorer)."""
    model_id = "tests/keyword-scorer"

    def fingerprint(self):
        return "keyword-scorer-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = np.array([2.0 * sum(low.count(k) for k in KW.get(o, [])) for o in it.options])
            out.append(np.stack([z, z - 1.0], 1))
        return out


DECIDER = DecideModel(KeywordScorer(), meta={"format": "test", "temperature": 1.0})
TEAM = DECIDER.decision("team", "Which team handles this message?", "doc", list(KW), escalate_below=0.6)


# ------------------------------------------------------------------------------------- approve: a hard check that raises
@cat.check(hard=True, then={"approve": "no"})
def amount_ok(amount):
    return float(amount) < 1000


@cat.rule("approve")
def approve(amount):
    return "yes"


QUESTIONS = [Question("paid", "Is the invoice paid?", require_evidence=True),
             Question("total", "What is the total?"),
             Question("currency", "Currency of the invoice?", Answer.choice(["EUR", "USD"])),
             Question("method", "How was it paid?", Answer.choice(["card", "transfer", "cash"]), require_evidence=True),
             TEAM.question(cat, "team"),
             Question("approve", "Approve the payment?", Answer.yes_no(), checkpoints=["amount_ok"])]


def system():
    return System(cat, QUESTIONS)
