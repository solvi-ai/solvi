"""Answer primitives (solvi 0.5): every answer is a value AND a confidence — also "not stated", a span, a ranking, a range.

The rules read a delivery-damage claim; each question's type says what kind of answer it is:

- `signed: Maybe[bool]` — yes / no, or **"not stated"** (solvi.Unknown): a real answer ("the claim does not say"), not an
  abstention. Constraints see it too.
- `amount: Span[float]` — an exact piece of the text, located at its offsets and parsed into a float. "twenty euros" is a
  span but not a float: "type rejected", the question abstains.
- `damaged: bool` with `require_evidence=True` — the rule returns `Claim(value, evidence=[quote])`; every quote is checked
  literally in the text, and an answer without one abstains ("evidence missing").
- `contact: Rank[Literal[...], 2]` — the two best channels in order, with a score each.
- `repair: Estimate[0, 3, 7, 14]` — days to repair as bins → a value, an 80% interval and its mass as the confidence.

The Audit panel shows each answer's kind, quotes and confidence; the response's overall confidence is the product.

Try: delete "Signed by the customer.", write "Amount: twenty euros", remove "cracked", or remove "about 5 days"."""
import re
from typing import Literal

from solvi import Catalog, Claim, Estimate, Maybe, Question, Quote, Rank, Span, Unknown

CHANNELS = ["email", "phone", "letter"]
cat = Catalog()


@cat.rule("signed")
def signed(doc: str) -> Maybe[bool]:
    if re.search(r"\bnot signed\b", doc, re.I):
        return False
    if re.search(r"\bsigned\b", doc, re.I):
        return True
    return Unknown                                    # "not stated", with confidence 1


@cat.rule("amount")
def amount(doc: str) -> Span[float]:
    m = re.search(r"(?:quote|amount):\s*(\S+(?:\s+euros?)?)", doc, re.I)
    if not m:
        return None                                   # the rule abstains
    v = m.group(1).split()[0] if m.group(1).split()[0][:1].isdigit() else m.group(1)
    return Quote(v, m.start(1), m.start(1) + len(v))


@cat.rule("damaged")
def damaged(doc: str) -> bool:
    m = re.search(r"[^.]*\b(cracked|broken|dented|wet)\b[^.]*\.", doc, re.I)
    return Claim(bool(m), evidence=[m.group(0).strip()] if m else [])


@cat.rule("contact")
def contact(doc: str) -> Rank[Literal["email", "phone", "letter"], 2]:
    low = doc.lower()
    pref = {"email": low.count("email"), "phone": low.count("phone") + 2 * ("call" in low), "letter": 2 * ("write" in low)}
    return {c: pref[c] - 0.1 * i for i, c in enumerate(CHANNELS)}


@cat.rule("repair")
def repair(doc: str) -> Estimate[0, 3, 7, 14]:
    m = re.search(r"about (\d+) days", doc)
    if m:
        return int(m.group(1))                        # a plain number: confidence 1
    return {"0–2": 0.2, "3–6": 0.5, "7–13": 0.3}      # nothing said: the desk's usual distribution


@cat.constraint
def phone_first_only_when_signed(signed, contact):
    """call first only on a signed claim ("not stated" is neither yes nor no)"""
    return contact[0] != "phone" or signed == "yes"


QUESTIONS = [Question("signed", "Is the claim signed?"), Question("amount", "Claimed amount?"),
             Question("damaged", "Is the item damaged?", require_evidence=True),
             Question("contact", "Best contact channels?"), Question("repair", "Days to repair?")]
