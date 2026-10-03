"""Answer primitives: every answer is a value AND a confidence — also "not stated", a quote, a ranking, a number range.

A claims desk reads delivery-damage claims. Each question's type says what kind of answer it is:

    signed:   Maybe[bool]                                   yes / no — or "not stated" (a real answer, not an abstention)
    amount:   Span[float]                                   an exact piece of the text, parsed into a float
    damaged:  bool, require_evidence=True                   yes / no, with the quote it rests on (checked in the text)
    contact:  Rank[Literal["email", "phone", "letter"], 2]  the two best channels, in order, with a score each
    repair:   Estimate[0, 3, 7, 14]                         days to repair: a distribution over bins → value + interval

  1. plain rules answer all five (no model): Unknown vs abstain, spans, evidence, a ranking, an estimate; the audit
  2. what goes wrong is caught: a quote not in the text, a span that is not a number, an answer without evidence
  3. a decider answers the same types (a stand-in with the typed v2 contract: "not stated", a pointer, rank, number)
  4. confidence means the same thing everywhere: the table, and the response's overall confidence per kind
  5. the response as JSON loads back, and its trace replays

The real decider runs when SOLVI_DECIDE_MODEL points to a typed v2 checkpoint ('l14g typed v2'); otherwise a keyword stand-in
with that contract plays its part.

Run:  uv run python examples/16_primitives.py"""
from __future__ import annotations

import os
import re
from typing import Literal

import numpy as np

from solvi import Catalog, Claim, Estimate, Maybe, Question, Quote, Rank, Response, Span, System, Unknown
from solvi.core.deciders import DecideModel
from solvi.core.primitives import fmt

CLAIMS = {
    "full": ("Claim 311. The parcel arrived with a cracked screen. Repair quote: 149.90 EUR. Signed by the customer. "
             "Please call me, phone is best; email is fine too. The shop says the repair takes about 5 days."),
    "sparse": "Claim 312. Box was wet on arrival, contents look fine. Amount: 20 EUR. Write to me.",
    "odd": "Claim 313. The lamp is broken. Amount: twenty euros. Not signed.",
}
CHANNELS = ["email", "phone", "letter"]


# --- 1. plain rules
def rule_catalog(bad_quote=False):
    cat = Catalog()

    @cat.rule("signed")
    def signed(doc: str) -> Maybe[bool]:
        if re.search(r"\bnot signed\b", doc, re.I):
            return False
        if re.search(r"\bsigned\b", doc, re.I):
            return True
        return Unknown                                    # the claim does not say: "not stated", with confidence 1

    @cat.rule("amount")
    def amount(doc: str) -> Span[float]:
        m = re.search(r"(?:quote|amount):\s*(\S+)", doc, re.I)
        return Quote(m.group(1), m.start(1), m.end(1)) if m else None      # None: the rule abstains

    @cat.rule("damaged")
    def damaged(doc: str) -> bool:
        m = re.search(r"cracked|broken|dented|wet", doc, re.I)
        quote = "shattered" if bad_quote else (m.group(0) if m else None)
        return Claim(bool(m), evidence=[quote] if quote else [])

    @cat.rule("contact")
    def contact(doc: str) -> Rank[Literal["email", "phone", "letter"], 2]:
        low = doc.lower()
        pref = {"email": low.count("email"), "phone": low.count("phone") + 2 * ("call" in low),
                "letter": 2 * ("write" in low)}
        return {c: pref[c] - 0.1 * i for i, c in enumerate(CHANNELS)}       # a key function's scores, best first

    @cat.rule("repair")
    def repair(doc: str) -> Estimate[0, 3, 7, 14]:
        m = re.search(r"about (\d+) days", doc)
        if m:
            return int(m.group(1))                        # a plain number: confidence 1, interval [5, 5]
        return {"0–2": 0.2, "3–6": 0.5, "7–13": 0.3}      # no word on it: the desk's usual distribution

    @cat.constraint
    def phone_first_only_when_signed(signed, contact):   # constraints see "not stated" as solvi.Unknown (≠ "yes", ≠ "no")
        return contact[0] != "phone" or signed == "yes"

    qs = [Question("signed", "Is the claim signed?"), Question("amount", "Claimed amount?"),
          Question("damaged", "Is the item damaged?", require_evidence=True), Question("contact", "Best channels?"),
          Question("repair", "Days to repair?")]
    return cat, qs


def show(res):
    for q, r in res.results.items():
        if r.status == "abstain":
            print(f"  {q:8s} — abstain [{r.guard}]: {r.why.split(';')[0]}")
            continue
        extra = ""
        if r.span:
            extra = f"   span doc[{r.span.start}:{r.span.end}] {r.span.value!r}"
        elif r.evidence:
            extra = "   evidence " + ", ".join(f"{e.value!r} [{e.start}:{e.end}]" for e in r.evidence)
        elif r.scores:
            extra = "   scores " + ", ".join(f"{k} {v:.2f}" for k, v in r.scores.items())
        print(f"  {q:8s} {fmt(r.answer, r.kind, r.extra):34s} confidence {r.confidence:.2f}{extra}")


# --- 3. a decider with the typed v2 contract
class StandIn:
    """Keyword logits for every kind; a "not stated" logit (high when the claim says nothing about the question); a pointer
    over whitespace tokens that points at the token after a cue word."""
    model_id = "demo/stand-in-typed-v2"
    META = {"format": "solvi_decide v2", "subformat": "l14g typed v2",
            "multi_question": {"layout": "block", "max_questions": 6}, "temperature": {"choice": 1.0}}
    CUES = {"signed": ["signed"], "amount": ["quote:", "amount:"], "damaged": ["cracked", "broken", "wet"],
            "contact": ["call", "phone", "email", "write"], "repair": ["days"]}

    def fingerprint(self):
        return "stand-in-typed-v2-1"

    def _one(self, it, text):
        low = text.lower()
        q = next((k for k in self.CUES if k in it.task.lower()), "")
        said = any(c in low for c in self.CUES.get(q, []))
        o = {"act": 3.0, "unknown": -4.0 if said else 4.0}
        if it.mode == "span":
            o["logits"] = np.zeros(0)
        elif it.mode == "noul":
            yes = {"signed": "signed" in low and "not signed" not in low, "damaged": said}.get(q, False)
            o["logits"] = np.array([2.5 if yes else -2.5])
        elif it.mode == "number":
            m = re.search(r"about (\d+) days", low)
            d = int(m.group(1)) if m else 5
            lab = next(i for i, e in enumerate([0, 3, 7, 14] + [10 ** 9]) if d < e)
            o["logits"] = np.array([3.0 if i == lab else -1.0 for i in range(len(it.options))])
        else:
            o["logits"] = np.array([1.5 * low.count(x) + 2.0 * (x == "phone" and "call" in low)
                                    + 2.0 * (x == "letter" and "write" in low) for x in it.options])
        if it.pointer:
            toks = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
            st, en = np.full(len(toks), -4.0), np.full(len(toks), -4.0)
            for i, (a, b) in enumerate(toks):
                w = text[a:b].lower()
                if q == "amount" and w in self.CUES["amount"] and i + 1 < len(toks):
                    st[i + 1] = en[i + 1] = 4.0
                elif q != "amount" and any(w.startswith(c.rstrip(":")) for c in self.CUES.get(q, [])):
                    st[i] = en[i] = 4.0
            o["pointer"] = {"start": st, "end": en, "null": [0.0, 0.0] if said else [4.0, 4.0], "offsets": toks}
        return o

    def logits(self, items):
        return [self._one(it, it.text) for it in items]

    def logits_pass(self, passes):
        return [[self._one(it, p.text) for it in p.items] for p in passes]


def load_model():
    src = os.environ.get("SOLVI_DECIDE_MODEL")
    if src:
        m = DecideModel.load(os.path.expanduser(src))
        if m.has_not_stated and m.has_pointer:
            return m
        print(f"  ({src} is not a typed v2 checkpoint: using the stand-in)")
    return DecideModel(StandIn(), StandIn.META)


def model_catalog(model):
    cat = Catalog()
    qs = [model.decision("signed", "Is the claim signed?", "doc", Maybe[bool]).question(cat),
          model.decision("amount", "What is the claimed amount?", "doc", Maybe[Span[float]]).question(cat),
          model.decision("damaged", "Is the item damaged?", "doc", bool, evidence=True).question(cat, require_evidence=True),
          model.decision("contact", "Which contact channels does the customer prefer?", "doc",
                         Rank[Literal["email", "phone", "letter"], 2]).question(cat),
          model.decision("repair", "How many days does the repair take?", "doc", Maybe[Estimate[0, 3, 7, 14]]).question(cat)]
    return cat, qs


if __name__ == "__main__":
    print("=== 1. plain rules: one claim each, every answer a value and a confidence ===")
    cat, qs = rule_catalog()
    system = System(cat, qs)
    for name, doc in CLAIMS.items():
        res = system.ask({"doc": doc})
        print(f"[{name}]  overall confidence {res.confidence:.2f}; not stated: {res.not_stated or '—'}; "
              f"abstained: {res.overall['abstained'] or '—'}")
        show(res)
    print(system.ask({"doc": CLAIMS["full"]}).audit("damaged"))

    print("\n=== 2. what goes wrong is caught ===")
    bad = System(*rule_catalog(bad_quote=True))
    r = bad.ask({"doc": CLAIMS["full"]})
    print(f"  a quote that is not in the text:  damaged → {r['damaged'].status} ({r['damaged'].why.split(';')[0]})")
    r = system.ask({"doc": CLAIMS["odd"]})
    print(f"  a span that is not a number:      amount  → {r['amount'].status} [{r['amount'].guard}]")
    r = system.ask({"doc": "Claim 314. Arrived. Amount: 5 EUR."})
    print(f"  an answer without evidence:       damaged → {r['damaged'].status} [{r['damaged'].guard}]")
    print("  " + system.safeguard_summary().replace("\n", "\n  "))

    print("\n=== 3. a decider answers the same types ===")
    model = load_model()
    print(f"  decider: {model.model_id}; not stated: {model.has_not_stated}; pointer: {model.has_pointer}")
    mcat, mqs = model_catalog(model)
    msys = System(mcat, mqs)
    for name, doc in CLAIMS.items():
        res = msys.ask({"doc": doc})
        print(f"[{name}]  overall confidence {res.confidence:.2f}; not stated: {res.not_stated or '—'}")
        show(res)
    full = msys.ask({"doc": CLAIMS["full"]})
    print(full.audit("amount"))

    print("\n=== 4. confidence means the same thing everywhere: P(this answer, as returned, is right) ===")
    for kind, meaning in [("yes_no / choice", "p(answer)"), ("not stated", "p(not stated)"),
                          ("span", "p(this span) — a rule: 1"), ("rank", "Plackett–Luce p(this top k, in this order)"),
                          ("estimate", "p(value in the interval) = the mass of its bins; a plain number: 1")]:
        print(f"  {kind:16s} {meaning}")
    o = full.overall
    print(f"  overall {o['confidence']:.2f} = the product over answered questions; per kind:")
    for k, v in o["by_kind"].items():
        print(f"    {k:9s} {v['answered']} answered, confidence {v['confidence']:.2f}")

    print("\n=== 5. JSON round trip and replay ===")
    back = Response.from_json(full.to_json(), catalog=msys)
    same = all((back[q].answer, back[q].evidence, back[q].extra) == (full[q].answer, full[q].evidence, full[q].extra)
               for q in full.results)
    sparse = msys.ask({"doc": CLAIMS["sparse"]})
    back2 = Response.from_json(sparse.to_json(), catalog=msys)
    print(f"  answers, quotes and intervals identical after JSON: {same}; not stated survives: "
          f"{[q for q in back2.results if back2[q].answer is Unknown]}")
    print(f"  replay: rules ok={system.ask({'doc': CLAIMS['full']}).trace.replay(cat)['ok']}, "
          f"decider ok={full.trace.replay(msys)['ok']}, loaded from JSON ok={back.trace.replay(msys)['ok']}")
