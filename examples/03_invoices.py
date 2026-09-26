"""Accounts payable: approve an invoice from its text. The catalog also holds parts that these questions do not need (other
documents, other data) — the strategist must leave them out. Fields are extracted with regular expressions, each with a quote;
swap in a model extractor (see 07_receipts_model.py) for messy real documents. The "risk" question has no rule and is learned.

Run:  uv run python examples/03_invoices.py"""
from __future__ import annotations

import random
import re
from datetime import date, timedelta

from solvi import Answer, Catalog, Question, Quote, System
from solvi.show import show

cat = Catalog()

MONEY = r"([0-9][0-9,]*\.[0-9]{2})"


def _find(doc, labels, pat, conv):
    for lab in labels:
        m = re.search(lab + r"\s*[:\-]?\s*" + pat, doc, flags=re.I)
        if m:
            return Quote(conv(m.group(1)), m.start(1), m.end(1))
    raise ValueError("not found: " + "/".join(labels))


# --- extraction
@cat.extract
def vendor(doc):
    "who issued the invoice"
    return _find(doc, ["Supplier", "Vendor", "From"], r"([A-Z][A-Za-z&.\- ]+?)(?:\n|$)", str.strip)


@cat.extract
def amount(doc):
    "total amount due"
    return _find(doc, ["Total due", "Amount payable", "Grand total"], MONEY, lambda s: float(s.replace(",", "")))


@cat.extract
def line_items(doc):
    "line item amounts"
    ms = list(re.finditer(r"^\s*-\s.*?\s" + MONEY + r"\s*$", doc, flags=re.M))
    if not ms:
        raise ValueError("no line items")
    return Quote([float(m.group(1).replace(",", "")) for m in ms], ms[0].start(1), ms[-1].end(1))


@cat.extract
def due_date(doc):
    "payment due date"
    return _find(doc, ["Due date", "Pay by", "Payment due"], r"(\d{4}-\d{2}-\d{2})", date.fromisoformat)


# --- computations
@cat.fn
def lines_total(line_items):
    return round(sum(line_items), 2)


@cat.fn
def days_to_due(due_date, today):
    return (due_date - today).days


@cat.fn
def vendor_history(vendor, payments_db):
    return [p for p in payments_db if p["vendor"] == vendor]


@cat.fn
def avg_vendor_amount(vendor_history):
    return round(sum(p["amount"] for p in vendor_history) / max(1, len(vendor_history)), 2)


@cat.fn
def n_lines(line_items):
    return len(line_items)


# --- checks
@cat.check
def total_matches(amount, lines_total):
    return abs(amount - lines_total) < 0.01


@cat.check
def known_vendor(vendor, vendors_db):
    return vendor in vendors_db


@cat.check
def not_duplicate(doc_id, prior_invoice_ids):
    return doc_id not in prior_invoice_ids


@cat.check
def amount_typical(amount, avg_vendor_amount):
    return avg_vendor_amount == 0 or amount <= 3 * avg_vendor_amount


@cat.check
def due_in_future(days_to_due):
    return days_to_due >= 0


# --- not needed for these questions (other documents, other data): the strategist must not run these
@cat.extract
def contract_parties(contract_doc):
    "contract parties"
    return Quote(["?"], 0, 0, "contract_doc")


@cat.fn
def penalty_amount(contract_parties, penalty_rate):
    return 0.0


@cat.fn
def shipping_eta(tracking_id, carrier_api):
    return 0


@cat.fn
def doc_length(doc):
    return len(doc)


@cat.fn
def vendor_upper(vendor):
    return vendor.upper()


@cat.check
def vendor_name_short(vendor_upper):
    return len(vendor_upper) < 40


# --- rules
@cat.rule("approve")
def approve_rule(not_duplicate, total_matches, known_vendor, amount_typical):
    if not not_duplicate:
        return "reject"
    return "approve" if (total_matches and known_vendor and amount_typical) else "escalate"


@cat.rule("duplicate")
def duplicate_rule(not_duplicate):
    return not not_duplicate


@cat.rule("urgent")
def urgent_rule(days_to_due, due_in_future):
    return due_in_future and days_to_due <= 5


QUESTIONS = [
    Question("approve", "Approve the invoice?", Answer.choice(["approve", "reject", "escalate"]), checkpoints=["not_duplicate"]),
    Question("duplicate", "Is it a duplicate of a paid invoice?", Answer.yes_no()),
    Question("urgent", "Is payment urgent?", Answer.yes_no()),
    Question("risk", "Risk level", Answer.choice(["low", "medium", "high"])),     # no rule — learned from examples
]

VENDORS = ["Acme Ltd", "Globex Corp", "Initech", "Umbrella Supplies", "Stark Parts"]
UNKNOWN = ["Shady Traders", "Quick Cash LLC", "Nowhere Inc"]
TODAY = date(2026, 9, 25)


def make(rng: random.Random, i: int):
    """Document + init_state + ground truth. Risk follows a hidden rule over computed facts, with 5% label noise."""
    known = rng.random() < 0.8
    v = rng.choice(VENDORS if known else UNKNOWN)
    lines = [round(rng.uniform(20, 900), 2) for _ in range(rng.randint(1, 5))]
    total = round(sum(lines), 2)
    mismatch = rng.random() < 0.15
    if mismatch:
        total = round(max(1.0, total + rng.choice([-1, 1]) * rng.uniform(5, 200)), 2)
    due = TODAY + timedelta(days=rng.randint(-5, 40))
    hist_avg = round(rng.uniform(200, 1500), 2)
    history = [{"vendor": v, "amount": round(hist_avg * rng.uniform(0.8, 1.2), 2)} for _ in range(rng.randint(0, 6))] if known else []
    if rng.random() < 0.12 and history:
        total = round(total * 4, 2)
        lines = [round(x * 4, 2) for x in lines]
        total = round(sum(lines), 2) if not mismatch else total
    doc_id = f"INV-{1000 + i}"
    dup = rng.random() < 0.1
    tpl = rng.randint(0, 2)
    lab_v, lab_t, lab_d = [("Supplier", "Total due", "Due date"), ("Vendor", "Amount payable", "Pay by"),
                           ("From", "Grand total", "Payment due")][tpl]
    body = "\n".join(f"  - item {j + 1} {x:,.2f}" for j, x in enumerate(lines))
    doc = f"INVOICE {doc_id}\n{lab_v}: {v}\nIssued: {TODAY - timedelta(days=rng.randint(1, 20))}\n{lab_d}: {due}\nItems:\n{body}\n{lab_t}: {total:,.2f}\n"
    init = {"doc": doc, "doc_id": doc_id, "today": TODAY, "vendors_db": list(VENDORS),
            "prior_invoice_ids": [doc_id] if dup else [f"INV-{rng.randint(1, 999)}"],
            "payments_db": [{"vendor": x["vendor"], "amount": x["amount"]} for x in history] + [{"vendor": "Initech", "amount": 500.0}]}
    # ground truth from the definitions (not through the catalog)
    lt = round(sum(lines), 2)
    tm = abs(total - lt) < 0.01
    hist = [p for p in init["payments_db"] if p["vendor"] == v]
    avg = round(sum(p["amount"] for p in hist) / max(1, len(hist)), 2)
    typ = avg == 0 or total <= 3 * avg
    appr = "reject" if dup else ("approve" if (tm and known and typ) else "escalate")
    days = (due - TODAY).days
    score = (0 if known else 2) + (0 if tm else 1) + (0 if typ else 2) + (1 if total > 2000 else 0)
    risk = "low" if score == 0 else ("medium" if score <= 2 else "high")
    if rng.random() < 0.05:
        risk = rng.choice([x for x in ("low", "medium", "high") if x != risk])
    truth = {"approve": appr, "duplicate": "yes" if dup else "no", "urgent": "yes" if (0 <= days <= 5) else "no", "risk": risk,
             "vendor": v, "amount": total}
    return init, truth


if __name__ == "__main__":
    rng = random.Random(0)
    system = System(cat, QUESTIONS)
    train = [make(rng, i) for i in range(300)]
    head = system.fit("risk", [(init, truth["risk"]) for init, truth in train])
    print(f"risk learned from {len(train)} invoices; selected facts: {head.features}")
    test = [make(rng, 1000 + i) for i in range(300)]
    for q in ("approve", "duplicate", "urgent", "risk"):
        acc = sum(system.ask(init, [q])[q].answer == truth[q] for init, truth in test) / len(test)
        print(f"  {q:10s} accuracy on 300 new invoices: {acc:.3f}")
    init, _ = test[0]
    print("\n=== one invoice ===\n" + init["doc"])
    show(system.ask(init), cat)
