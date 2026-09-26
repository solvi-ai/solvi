"""KYC / AML: a customer and their transactions -> risk level, file a suspicious activity report (SAR)?, freeze the account?

Name screening is a function (fuzzy match after normalising accents, case and word order, confirmed by date of birth), so
"Dmitriy Volkanov" still hits "Dmitri Volkanov" and a namesake with another birth date does not. Structuring (several cash
deposits just under the 10 000 reporting threshold within 7 days), velocity against the declared profile, pass-through wires
and high-risk-country exposure are computed facts. A sanctions hit is a hard check: the account is frozen and a SAR filed
whatever else the data says, and the expensive lookups (news search, counterparty screening) are skipped.
Try: rename the customer to "Dmitriy Volkanov" with dob 1968-03-14, or add a cash deposit of 9 800.
Data, names and lists are synthetic; the country list is illustrative."""
from __future__ import annotations

import difflib
import unicodedata
from datetime import date, timedelta

from solvi import Answer, Catalog, Question

cat = Catalog()

CTR_THRESHOLD = 10_000          # cash reporting threshold
JUST_UNDER = 0.90               # "just under" = 90-100% of the threshold
STRUCTURING_WINDOW_DAYS = 7
STRUCTURING_MIN_DEPOSITS = 3
NAME_MATCH = 0.88               # fuzzy name score that counts as a match
HIGH_RISK = {"IR", "KP", "MM", "SY", "YE"}      # illustrative high-risk jurisdictions
EXTERNAL = ("adverse_media", "counterparty_screening")   # paid / slow lookups


def prepare(state):
    """JSON has no dates: turn ISO strings into datetime.date."""
    d = date.fromisoformat
    state["today"] = d(state["today"])
    state["transactions"] = [{**t, "date": d(t["date"])} for t in state["transactions"]]
    return state


def _norm(name):
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return " ".join("".join(c if c.isalpha() else " " for c in s).split())


def _score(a, b):
    """Name similarity in [0, 1]: best of plain and word-order-independent comparison."""
    a, b = _norm(a), _norm(b)
    plain = difflib.SequenceMatcher(None, a, b).ratio()
    ordered = difflib.SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()
    return round(max(plain, ordered), 3)


# ---------- screening
@cat.fn
def sanctions_screen(customer, sanctions_list):
    """best fuzzy match of the customer against the sanctions list; a hit needs the name score AND no conflicting birth date"""
    best = max(sanctions_list, key=lambda e: _score(customer["name"], e["name"]))
    score = _score(customer["name"], best["name"])
    dob_match = None if not best.get("dob") else best["dob"] == customer.get("dob")
    return {"listed": best["name"], "score": score, "dob_match": dob_match,
            "hit": score >= NAME_MATCH and dob_match is not False}


@cat.check(hard=True, then={"risk": "high", "file_sar": "yes", "freeze": "yes"})
def customer_not_sanctioned(sanctions_screen):
    """hard: a confirmed sanctions hit freezes the account and files a SAR, regardless of anything else"""
    return not sanctions_screen["hit"]


@cat.fn
def pep_screen(customer, pep_list):
    best = max(pep_list, key=lambda n: _score(customer["name"], n))
    return {"listed": best, "score": _score(customer["name"], best)}


@cat.fn
def is_pep(pep_screen, customer):
    return pep_screen["score"] >= NAME_MATCH or bool(customer.get("pep_declared"))


@cat.fn
def adverse_media(customer, news_archive):
    """news search for financial-crime stories about the customer (a paid external API in production)"""
    return news_archive.get(_norm(customer["name"]), [])


@cat.fn
def counterparty_screening(transactions, sanctions_list):
    """every wire counterparty screened against the sanctions list (slow: one fuzzy search per counterparty)"""
    hits = []
    for name in sorted({t["counterparty"] for t in transactions if t.get("counterparty")}):
        best = max(sanctions_list, key=lambda e: _score(name, e["name"]))
        s = _score(name, best["name"])
        if s >= NAME_MATCH:
            hits.append({"counterparty": name, "listed": best["name"], "score": s})
    return hits


# ---------- transaction analytics
@cat.fn
def structuring(transactions):
    """most cash deposits of 90-100% of the threshold inside any 7-day window"""
    near = sorted((t["date"], t["amount"]) for t in transactions
                  if t["type"] == "cash_deposit" and JUST_UNDER * CTR_THRESHOLD <= t["amount"] < CTR_THRESHOLD)
    best = {"count": 0, "total": 0.0, "from": None, "to": None}
    for i, (d0, _) in enumerate(near):
        win = [(d, a) for d, a in near[i:] if d - d0 <= timedelta(days=STRUCTURING_WINDOW_DAYS - 1)]
        if len(win) > best["count"]:
            best = {"count": len(win), "total": round(sum(a for _, a in win), 2), "from": str(win[0][0]), "to": str(win[-1][0])}
    return best


@cat.fn
def velocity_ratio(transactions, customer, today):
    """volume of the last 30 days over the monthly volume declared at onboarding"""
    vol = sum(t["amount"] for t in transactions if today - t["date"] <= timedelta(days=30))
    return round(vol / customer["expected_monthly_volume"], 2)


@cat.fn
def high_risk_share(transactions, customer):
    """share of volume to or from high-risk jurisdictions (all of it when the customer lives in one)"""
    if customer.get("residence") in HIGH_RISK:
        return 1.0
    total = sum(t["amount"] for t in transactions) or 1.0
    return round(sum(t["amount"] for t in transactions if t.get("country") in HIGH_RISK) / total, 3)


@cat.fn
def pass_through(transactions):
    """incoming wires sent on within 2 days for at least 90% of the amount (a layering pattern)"""
    ins = [t for t in transactions if t["type"] == "wire_in"]
    outs = [t for t in transactions if t["type"] == "wire_out"]
    return [{"in": t["amount"], "out": o["amount"], "on": str(o["date"])} for t in ins for o in outs
            if timedelta(0) <= o["date"] - t["date"] <= timedelta(days=2) and 0.9 * t["amount"] <= o["amount"] <= t["amount"]]


# ---------- conclusions
@cat.fn
def risk_factors(is_pep, structuring, velocity_ratio, high_risk_share, adverse_media, counterparty_screening, pass_through):
    f = []
    if structuring["count"] >= STRUCTURING_MIN_DEPOSITS:
        f.append(("structuring", 3))
    if counterparty_screening:
        f.append(("sanctioned counterparty", 4))
    if pass_through:
        f.append(("pass-through wires", 2))
    if is_pep:
        f.append(("PEP", 2))
    if adverse_media:
        f.append(("adverse media", 2))
    if high_risk_share >= 0.25:
        f.append(("high-risk jurisdictions", 2))
    elif high_risk_share > 0:
        f.append(("some high-risk exposure", 1))
    if velocity_ratio >= 3:
        f.append(("volume 3x profile", 2))
    elif velocity_ratio >= 1.5:
        f.append(("volume 1.5x profile", 1))
    return f


@cat.fn
def sar_grounds(structuring, counterparty_screening, pass_through, adverse_media, high_risk_share):
    """reasons that oblige a suspicious activity report (not mere risk: a PEP alone is not suspicious)"""
    g = []
    if structuring["count"] >= STRUCTURING_MIN_DEPOSITS:
        g.append(f"structuring: {structuring['count']} cash deposits just under {CTR_THRESHOLD} "
                 f"({structuring['from']}..{structuring['to']}, total {structuring['total']})")
    if counterparty_screening:
        g.append("wire with sanctioned counterparty " + ", ".join(h["counterparty"] for h in counterparty_screening))
    if pass_through:
        g.append(f"{len(pass_through)} pass-through wire(s) within 2 days")
    if adverse_media and high_risk_share > 0:
        g.append("adverse media plus high-risk-jurisdiction flows")
    return g


@cat.rule("risk")
def risk(risk_factors):
    pts = sum(p for _, p in risk_factors)
    return "high" if pts >= 4 else ("medium" if pts >= 2 else "low")


@cat.rule("file_sar")
def file_sar(sar_grounds):
    return bool(sar_grounds)


@cat.rule("freeze")
def freeze(counterparty_screening):
    """a SAR alone does not freeze (that would tip the customer off); a sanctioned party does"""
    return bool(counterparty_screening)


CHECK = ["customer_not_sanctioned"]
QUESTIONS = [
    Question("risk", "Customer risk level", Answer.choice(["low", "medium", "high"]), checkpoints=CHECK),
    Question("file_sar", "File a suspicious activity report?", Answer.yes_no(), checkpoints=CHECK),
    Question("freeze", "Freeze the account?", Answer.yes_no(), checkpoints=CHECK),
]
