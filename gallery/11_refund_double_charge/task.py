"""Support desk: "I was charged twice" -> is it a double charge?, refund automatically / manually / not at all, which reply.

The ticket text is only the customer's claim: what they say about being charged twice (claimed / denied / unclear / not
mentioned) and the amount they mention are extracted with a Quote (character offsets in the ticket); an unclear claim makes
the claim question abstain instead of guessing. The decision comes from the ledger: a double charge is two
settled charges of the same amount at the same merchant within 10 minutes that have not been refunded. A released card
authorisation is not a charge, and a monthly renewal is not a duplicate. The refund is the ledger amount, not the claimed
one. It is automatic up to 500 when the free balance of the payout account (balance - reserved) covers it; otherwise it goes
to a person. An open chargeback is a hard check: the bank is already returning the money, so refunding would pay twice.
Try: change the second charge's time to a day later, or the payout account's "reserved" to 1000."""
from __future__ import annotations

import re
from datetime import datetime

from solvi import Answer, Catalog, Question, Quote

cat = Catalog()

DOUBLE_WINDOW_MIN = 10          # two identical charges this close together are a double charge
AUTO_REFUND_LIMIT = 500.0

# ---------- reading the claim: a plain rule, clause by clause (no model; the same ticket always reads the same way)
# A claim is a word for taking money and a word for "twice" in one clause ("billed me two times", "duplicate transaction",
# "the same payment went through again"). "not" / "never" / "nobody" up to five words before it makes it a denial; "if",
# "maybe", "not sure whether" before it, or a yes/no question ("Was I charged twice?"), makes it unclear, and unclear
# abstains. What it cannot read is listed in the README.
MONEY_WORD = (r"charg\w*|bill\w*|debit\w*|deduct\w*|withdr[ae]w\w*|taken|took|paid|payments?|transactions?|money"
              r"|went through|gone through|go through|processed|hit|pulled|came out|collected")
TWICE = (r"twice|two times|2 times|2x|x2|a second time|one more time|once more|double[ds]?|duplicat\w*|repeated"
         r"|two (?:identical|equal) (?:charges|payments|transactions|debits|withdrawals)")
AGAIN = r"again|another|second|extra|additional|two (?:charges|payments|transactions|debits|withdrawals)"   # only with SAME
SAME = r"\bsame\b|\bidentical\b|right after|straight after|immediately|(?:minutes?|seconds?) later"
NOT_A_CLAIM = r"double[- ]?check\w*|twice (?:as|a (?:day|week|month|year))|two times (?:as|a (?:day|week|month|year))"
NEGATION = r"\b(?:not|never|no|nobody|no one|none|nor)\b|n['\u2019]t\b"
IDIOM = (r"can(?:no|['\u2019])t believe|don['\u2019]?t (?:know|understand|see) why|not sure why|no idea why"   # neither
         r"|not (?:happy|ok|okay|acceptable)|no (?:reason|explanation)|^\W*no\s*[,!]"             # deny nor doubt
         r"|if (?:possible|you can|you could)|as if|even if")
HEDGE = r"\b(?:if|whether|maybe|perhaps|possibly|might|may have|could have|unsure|not sure|wonder\w*|in case|thought)\b"
YES_NO_QUESTION = r"^\W*(?:was|were|did|is|are|have|has|am|do|does)\b"
CLAUSE = re.compile(r"(?:[^.!?;\n]|\.(?=\d))+[.!?;]*")             # sentences (a full stop inside "54.99" does not end one)
SPLIT = re.compile(r"\bbut\b|\bhowever\b|\s-\s|\s\u2013\s")         # ... cut again at "but", "however" and a dash
CLAIMED, DENIED, UNCLEAR, NONE = "claimed", "denied", "unclear", "not mentioned"  # "none" would mean "none of these" to a decider


def _blank(text, pattern):
    """the text with every match of pattern replaced by spaces (offsets stay the same)"""
    return re.sub(pattern, lambda m: " " * len(m.group()), text, flags=re.IGNORECASE)


def _mention(clause):
    """(start, end) of "money word ... twice" in a clause, or None"""
    c = _blank(clause, NOT_A_CLAIM)
    money = [m.span() for m in re.finditer(rf"\b(?:{MONEY_WORD})\b", c, re.IGNORECASE)]
    twice = [m.span() for m in re.finditer(rf"\b(?:{TWICE})\b", c, re.IGNORECASE)]
    if re.search(SAME, c, re.IGNORECASE):
        twice += [m.span() for m in re.finditer(rf"\b(?:{AGAIN})\b", c, re.IGNORECASE)]
    pairs = [(min(a[0], b[0]), max(a[1], b[1])) for a in money for b in twice]
    return min(pairs, key=lambda p: (p[1] - p[0], p[0])) if pairs else None


def read_claim(ticket):
    """(verdict, start, end): what the ticket says about being charged twice, and where"""
    found = []
    for m in CLAUSE.finditer(ticket):
        pieces, at = [], 0
        for s in SPLIT.finditer(m.group()):
            pieces.append((at, s.start()))
            at = s.end()
        pieces.append((at, len(m.group())))
        for a, b in pieces:
            clause = m.group()[a:b]
            span = _mention(clause)
            if span is None:
                continue
            before = _blank(clause[:span[0]], IDIOM)
            asks = clause.rstrip().endswith("?") and re.search(YES_NO_QUESTION, clause, re.IGNORECASE)
            if asks or re.search(HEDGE, before, re.IGNORECASE):
                verdict = UNCLEAR
            elif re.search(NEGATION, " ".join(before.split()[-5:]), re.IGNORECASE):   # a negation up to 5 words before it
                verdict = DENIED
            else:
                verdict = CLAIMED
            start = m.start() + a
            found.append((verdict, start + span[0], start + span[1]))
    for verdict in (CLAIMED, UNCLEAR, DENIED):                        # one clear claim is enough; then doubt; then a denial
        hit = next((f for f in found if f[0] == verdict), None)
        if hit:
            return hit
    return NONE, 0, 0


MONEY = re.compile(r"(?:[$€£]\s?|\b(?:USD|EUR|GBP)\s?)?(\d{1,3}(?:,\d{3})*\.\d{2})\b")


def prepare(state):
    """JSON has no datetimes: turn ISO timestamps into datetime."""
    state["ledger"] = [{**e, "ts": datetime.fromisoformat(e["ts"])} for e in state["ledger"]]
    return state


# ---------- what the customer says (cited)
@cat.extract
def double_charge_claim(ticket):
    """what the customer says about being charged twice: claimed / denied / unclear / not mentioned, cited (an empty quote at 0:0
    when not mentioned)"""
    verdict, start, end = read_claim(ticket)
    return Quote(verdict, start, end, "ticket")


@cat.extract(exact=True)                  # the amount must be the number written at its offsets (the audit checks it)
def claimed_amount(ticket):
    """the first amount of money in the ticket"""
    m = MONEY.search(ticket)
    if not m:
        raise ValueError("the ticket mentions no amount")
    return Quote(float(m.group(1).replace(",", "")), m.start(1), m.end(1), "ticket")


# ---------- what the ledger shows
@cat.fn
def duplicate_pairs(ledger):
    """settled charges with the same merchant and amount within DOUBLE_WINDOW_MIN minutes; each flagged if already refunded"""
    charges = sorted((e for e in ledger if e["type"] == "charge" and e["status"] == "settled"), key=lambda e: e["ts"])
    refunded = {e["refund_of"] for e in ledger if e["type"] == "refund"}
    pairs, used = [], set()
    for i, a in enumerate(charges):
        for b in charges[i + 1:]:
            gap = (b["ts"] - a["ts"]).total_seconds() / 60
            if b["id"] not in used and a["id"] not in used and b["merchant"] == a["merchant"] and b["amount"] == a["amount"] \
                    and gap <= DOUBLE_WINDOW_MIN:
                used |= {a["id"], b["id"]}
                pairs.append({"first": a["id"], "duplicate": b["id"], "merchant": a["merchant"], "amount": a["amount"],
                              "minutes_apart": round(gap, 1), "refunded": b["id"] in refunded})
    return pairs


@cat.fn
def outstanding_duplicates(duplicate_pairs):
    return [p for p in duplicate_pairs if not p["refunded"]]


@cat.fn
def refund_amount(outstanding_duplicates):
    return round(sum(p["amount"] for p in outstanding_duplicates), 2)


@cat.fn
def free_balance(payout_account):
    """what the payout account can pay now: balance minus reserved funds"""
    return round(payout_account["balance"] - payout_account["reserved"], 2)


@cat.check
def balance_covers(free_balance, refund_amount):
    return free_balance >= refund_amount


@cat.check
def within_auto_limit(refund_amount):
    return refund_amount <= AUTO_REFUND_LIMIT


@cat.check(hard=True, then={"refund": "none", "reply": "chargeback in progress"})
def no_open_chargeback(ledger):
    """hard: with a chargeback open the bank is already returning the money; a refund would pay it twice"""
    return not any(e["type"] == "chargeback" and e["status"] == "open" for e in ledger)


# ---------- answers
@cat.rule("customer_claims_double")
def customer_claims_double(double_charge_claim):
    """unclear -> None: the question abstains and a person reads the ticket"""
    return {CLAIMED: True, DENIED: False, NONE: False}.get(double_charge_claim)


@cat.rule("is_double_charge")
def is_double_charge(outstanding_duplicates):
    """decided by the ledger only"""
    return bool(outstanding_duplicates)


@cat.rule("refund")
def refund(outstanding_duplicates, balance_covers, within_auto_limit):
    if not outstanding_duplicates:
        return "none"
    return "auto" if balance_covers and within_auto_limit else "manual"


@cat.rule("reply")
def reply(double_charge_claim, duplicate_pairs, outstanding_duplicates, balance_covers, within_auto_limit):
    if outstanding_duplicates:
        return "confirm refund" if balance_covers and within_auto_limit else "under review"
    if duplicate_pairs:
        return "already refunded"
    if double_charge_claim == UNCLEAR:
        return None                       # the reply depends on what the customer meant: a person reads it
    return "no duplicate found" if double_charge_claim == CLAIMED else "not about a charge"


QUESTIONS = [
    Question("customer_claims_double", "Does the customer say they were charged twice?", Answer.yes_no(),
             checkpoints=["claimed_amount"]),
    Question("is_double_charge", "Does the ledger show a double charge still to refund?", Answer.yes_no()),
    Question("refund", "Refund", Answer.choice(["auto", "manual", "none"]), checkpoints=["no_open_chargeback"]),
    Question("reply", "Reply to send", Answer.choice(["confirm refund", "under review", "already refunded", "no duplicate found",
                                                      "chargeback in progress", "not about a charge"]),
             checkpoints=["no_open_chargeback"]),
]
