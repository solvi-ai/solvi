"""Support desk: "I was charged twice" -> is it a double charge?, refund automatically / manually / not at all, which reply.

The ticket text is only the customer's claim: whether they say they were charged twice, and the amount they mention, are
extracted with a Quote (character offsets in the ticket). The decision comes from the ledger: a double charge is two
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

CLAIM = re.compile(r"(charged|billed|debited|taken)\s+(me\s+)?(twice|two times|double)|double[- ]?(charged?|billed|billing|payment)"
                   r"|duplicate (charge|payment|transaction)|two (identical |same )?(charges|payments)", re.IGNORECASE)
MONEY = re.compile(r"(?:[$€£]\s?|\b(?:USD|EUR|GBP)\s?)?(\d{1,3}(?:,\d{3})*\.\d{2})\b")


def prepare(state):
    """JSON has no datetimes: turn ISO timestamps into datetime."""
    state["ledger"] = [{**e, "ts": datetime.fromisoformat(e["ts"])} for e in state["ledger"]]
    return state


# ---------- what the customer says (cited)
@cat.extract
def claims_double_charge(ticket):
    """the phrase in which the customer says they were charged twice (an empty quote at 0:0 when there is none)"""
    m = CLAIM.search(ticket)
    return Quote(True, m.start(), m.end(), "ticket") if m else Quote(False, 0, 0, "ticket")


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
def customer_claims_double(claims_double_charge):
    return claims_double_charge


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
def reply(claims_double_charge, duplicate_pairs, outstanding_duplicates, balance_covers, within_auto_limit):
    if outstanding_duplicates:
        return "confirm refund" if balance_covers and within_auto_limit else "under review"
    if duplicate_pairs:
        return "already refunded"
    return "no duplicate found" if claims_double_charge else "not about a charge"


QUESTIONS = [
    Question("customer_claims_double", "Does the customer say they were charged twice?", Answer.yes_no(),
             checkpoints=["claimed_amount"]),
    Question("is_double_charge", "Does the ledger show a double charge still to refund?", Answer.yes_no()),
    Question("refund", "Refund", Answer.choice(["auto", "manual", "none"]), checkpoints=["no_open_chargeback"]),
    Question("reply", "Reply to send", Answer.choice(["confirm refund", "under review", "already refunded", "no duplicate found",
                                                      "chargeback in progress", "not about a charge"]),
             checkpoints=["no_open_chargeback"]),
]
