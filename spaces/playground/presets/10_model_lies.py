"""A model lies, and grounding catches it (solvi 0.4, examples/12_grounded_audit.py).

An expense claim with three small stand-in "models" behind its parts (no torch, they run in the browser):
- an extractor quotes the total. With "extractor": "hallucinates" it returns 488.60 at the offsets of "48.60". Grounding
  checks that the quote is literally doc[start:end], so the value is rejected and the regular expression below takes over;
- a classifier decides the category. With "classifier": "rogue" it answers "luxury", which is not one of its options, so the
  output is rejected and the category question abstains instead of guessing;
- an urgency model says whether to pay out fast. It is only 55% sure, below the question's min_confidence of 0.7, so that
  question abstains too and says what it would have answered.

Look at the Audit panel: what each answer rests on (given, quoted, decided, computed), which safeguards fired, and how much of
the support is deterministic. Set both switches in init_state to "honest" and run again; then use "Replace the model" in
"Tamper with the trace" to see replay report that the model changed since the decision."""
import re

from solvi import Answer, Catalog, Decision, Question, Quote
from solvi.provenance import digest

CATEGORIES = ["travel", "meals", "equipment"]
KEYWORDS = {"travel": ["taxi", "train", "flight", "hotel"], "meals": ["lunch", "dinner", "restaurant"],
            "equipment": ["laptop", "monitor", "keyboard"]}


class StandInExtractor:
    """Stands in for a ModernBERT extractor. A real one is recorded the same way: model id + fingerprint of its weights."""
    model_id = "demo/extract-total"

    def __init__(self, version="1.0"):
        self.version, self.hallucinate = version, False

    def fingerprint(self):
        return digest("StandInExtractor", self.version)

    def find_total(self, doc):
        m = re.search(r"Total:\s*([0-9.,]+)", doc)
        if not m:
            return None
        value = m.group(1).replace(".", "8.") if self.hallucinate else m.group(1)   # "48.60" -> "488.60"
        return Quote(value, m.start(1), m.end(1), confidence=0.93)


class StandInClassifier:
    """Stands in for a text classifier: probabilities over the categories from keyword counts."""
    model_id = "demo/expense-category"
    version = "2.1"
    rogue = False

    def predict(self, doc):
        low = doc.lower()
        score = {c: 1 + sum(low.count(w) for w in ws) for c, ws in KEYWORDS.items()}
        n = sum(score.values())
        return {c: s / n for c, s in score.items()}


class StandInUrgency:
    """Stands in for a model that is not sure: 55% "yes" for any claim that mentions a client."""
    model_id = "demo/urgency"
    version = "0.3"

    def predict(self, doc):
        p = 0.55 if "client" in doc.lower() else 0.2
        return {"yes": p, "no": 1 - p}


EXTRACTOR, CLASSIFIER, URGENCY = StandInExtractor(), StandInClassifier(), StandInUrgency()
cat = Catalog()


def prepare(state):
    """the two switches set how the stand-in models behave; they are not facts of the claim"""
    EXTRACTOR.hallucinate = state.pop("extractor", "honest") == "hallucinates"
    CLASSIFIER.rogue = state.pop("classifier", "honest") == "rogue"
    return state


# ---------- the total: the model first, a regular expression as the fallback producer
@cat.extract(model=EXTRACTOR, provides="total")
def total_model(doc):
    q = EXTRACTOR.find_total(doc)
    if q is None:
        raise ValueError("no total found")
    return q


@cat.extract(provides="total")
def total_regex(doc):
    m = re.search(r"Total:\s*([0-9]+\.[0-9]{2})", doc)
    if m is None:
        raise ValueError("no total in the document")
    return Quote(m.group(1), m.start(1), m.end(1))


# ---------- model decisions: the value must be one of the options
@cat.fn(model=CLASSIFIER, options=CATEGORIES)
def category(doc):
    p = CLASSIFIER.predict(doc)
    return Decision("luxury" if CLASSIFIER.rogue else max(p, key=p.get), p)


@cat.fn(model=URGENCY, options=["yes", "no"])
def urgent(doc):
    p = URGENCY.predict(doc)
    return Decision(max(p, key=p.get), p)


# ---------- plain code decides
@cat.fn
def amount(total):
    return float(total.replace(",", ""))


@cat.check(hard=True, then={"approve": "no"})
def amount_positive(amount):
    return amount > 0


@cat.rule("approve")
def approve(amount, limit):
    return amount <= limit


@cat.rule("category")
def category_answer(category):
    return category


@cat.rule("pay_fast")
def pay_fast(urgent):
    return urgent == "yes"


QUESTIONS = [
    Question("approve", "Approve the claim (total within the limit)?", Answer.yes_no(), requires=["amount_positive"]),
    Question("category", "Expense category", Answer.choice(CATEGORIES), min_confidence=0.5),
    Question("pay_fast", "Pay out within 24 hours?", Answer.yes_no(), min_confidence=0.7),
]
