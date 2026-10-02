"""Grounded decisions: fuzzy proposes, deterministic decides, everything is in the trace.

One catalog answers an expense claim twice: first with plain code only, then with models behind two of its parts — an
extractor that quotes the total and a classifier that decides the category. The rules, the hard check and the questions are
the same; only the provenance of the facts differs, and `res.audit()` shows it. Then the models misbehave: the extractor
"hallucinates" a total that is not in the text (grounding rejects it and the regular expression takes over), and the classifier
answers with a category outside its options (rejected, the question abstains). Finally the extractor is retrained and replay
reports that the model changed since the decision. The models here are small stand-ins, so the example runs without torch;
a real extractor (solvi.extract_long.LongSpanExtractor.field) is recorded the same way.

Run:  uv run python examples/12_grounded_audit.py"""
from __future__ import annotations

import re

from solvi import Answer, Catalog, Decision, Question, Quote, System
from solvi.provenance import digest

CATEGORIES = ["travel", "meals", "equipment"]
KEYWORDS = {"travel": ["taxi", "train", "flight", "hotel"], "meals": ["lunch", "dinner", "restaurant"],
            "equipment": ["laptop", "monitor", "keyboard"]}

CLAIM = """Expense claim #2291
Vendor: City Taxi Ltd
Taxi from the airport to the client office, 14 km.
Total: 48.60 EUR
"""


class StandInExtractor:
    """Stands in for a ModernBERT extractor: finds a field with a pattern and quotes it. `hallucinate=True` makes it return
    a value that is not in the text (with plausible offsets) — the failure the grounding check exists for."""
    model_id = "demo/extract-total"

    def __init__(self, version="1.0", hallucinate=False):
        self.version, self.hallucinate = version, hallucinate

    def fingerprint(self):                          # a real extractor hashes its settings and weights
        return digest("StandInExtractor", self.version)

    def field(self, name, pattern):
        def f(doc):
            m = re.search(pattern, doc)
            if not m:
                return None
            value = m.group(1)
            if self.hallucinate:
                value = value.replace(".", "8.")        # "48.60" → "488.60": not what the document says
            return Quote(value, m.start(1), m.end(1), confidence=0.93)
        f.__name__ = name
        f.__solvi_model__ = self                    # like extractor.field(...): the part is model-backed
        return f


class StandInClassifier:
    """Stands in for a text classifier: probabilities over the categories from keyword counts. `rogue=True` answers with a
    category that was never declared."""
    model_id = "demo/expense-category"
    version = "2.1"

    def __init__(self, rogue=False):
        self.rogue = rogue

    def predict(self, doc):
        low = doc.lower()
        score = {c: 1 + sum(low.count(w) for w in ws) for c, ws in KEYWORDS.items()}
        n = sum(score.values())
        return {c: s / n for c, s in score.items()}


def build(extractor=None, classifier=None):
    """The same catalog with or without models behind `total` and `category`."""
    cat = Catalog()

    def regex_total(doc):
        m = re.search(r"Total:\s*([0-9]+\.[0-9]{2})", doc)
        if m is None:
            raise ValueError("no total in the document")
        return Quote(m.group(1), m.start(1), m.end(1))

    if extractor is None:                          # no model: a regular expression is the only producer of `total`
        @cat.extract
        def total(doc):
            return regex_total(doc)
    else:                                          # the model first, the regular expression as the fallback
        cat.extract(extractor.field("total_model", r"Total:\s*([0-9.,]+)"), provides="total")

        @cat.extract(provides="total")
        def total_regex(doc):
            return regex_total(doc)

    if classifier is not None:
        @cat.fn(model=classifier, options=CATEGORIES)
        def category(doc):
            p = classifier.predict(doc)
            best = max(p, key=p.get)
            return Decision("luxury" if classifier.rogue else best, p)
    else:
        @cat.fn
        def category(doc):
            low = doc.lower()
            return max(CATEGORIES, key=lambda c: sum(low.count(w) for w in KEYWORDS[c]))

    @cat.fn
    def amount(total):
        return float(total.replace(",", ""))

    @cat.check(hard=True, then={"approve": "no"})
    def amount_positive(amount):
        return amount > 0

    @cat.rule("approve")
    def approve(amount, category, limit):
        return amount <= limit[category]

    @cat.rule("category")
    def category_answer(category):
        return category

    return cat


QUESTIONS = [Question("approve", "Approve the claim?", Answer.yes_no(), requires=["amount_positive"]),
             Question("category", "Expense category", Answer.choice(CATEGORIES), min_confidence=0.5)]
REQUEST = {"doc": CLAIM, "limit": {"travel": 100, "meals": 60, "equipment": 800}}


if __name__ == "__main__":
    print("=== 1. no model: every fact comes from plain code ===")
    plain = System(build(), QUESTIONS)
    print(plain.ask(REQUEST).audit())

    print("\n=== 2. the same catalog with models behind `total` and `category` ===")
    extractor, classifier = StandInExtractor(), StandInClassifier()
    cat = build(extractor, classifier)
    system = System(cat, QUESTIONS)
    res = system.ask(REQUEST)
    print(res.audit())
    print("\ncomputed_state:\n" + res.state_text())

    print("\n=== 3. the extractor hallucinates a total: grounding rejects it, the regular expression takes over ===")
    extractor.hallucinate = True
    print(system.ask(REQUEST).audit("approve"))
    extractor.hallucinate = False

    print("\n=== 4. the classifier answers outside its options: rejected, the question abstains ===")
    classifier.rogue = True
    print(system.ask(REQUEST).audit("category"))
    classifier.rogue = False

    print("\n=== 5. an ambiguous claim: the category is below the question's min_confidence, so it abstains ===")
    vague = {**REQUEST, "doc": "Expense claim #2292\nLunch with the client, taxi back to the office.\nTotal: 36.00 EUR\n"}
    print(system.ask(vague).audit("category"))

    print("\n=== 6. the extractor is retrained after decision 2: replay says so ===")
    print("replay now:", system.ask(REQUEST).trace.replay(cat)["ok"])
    extractor.version = "1.1"
    rep = res.trace.replay(cat)
    print("replay of decision 2:", rep["ok"], rep["mismatches"])
    print("models:", rep["models"])
    extractor.version = "1.0"                      # back to the model that decided: re-run it, or trust it and only verify
    print("same model, trust_models=True:", res.trace.replay(cat, trust_models=True)["models"])

    print("\n=== lifetime safeguard stats ===")
    print(system.safeguard_summary())
