"""Trusted, fixed catalogs for the "Business decisions" and "Learn from examples" tabs (they run in-process, no user code).

- leave request: the catalog of presets/01_leave_request.py (loaded from that file, so the form and the preset stay in sync);
- invoice approval: the regex catalog of solvi/examples/03_invoices.py, plus an `invoice_id` extractor, with the "risk"
  question learned from 300 synthetic invoices at startup;
- delivery zones: the address generator and catalog of solvi/examples/06_learned_rules.py.
"""
from __future__ import annotations

import importlib.util
import random
import re
from datetime import date, timedelta
from pathlib import Path

from solvi import Answer, Catalog, Question, Quote, System

HERE = Path(__file__).resolve().parent


# ====================================================================== leave request (shares the preset's catalog)
def _load_preset(name):
    spec = importlib.util.spec_from_file_location(f"preset_{name}", HERE / "presets" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LEAVE = _load_preset("01_leave_request")
LEAVE_SYSTEM = System(LEAVE.cat, LEAVE.QUESTIONS)
LEAVE_BLACKOUT = [(date(2026, 12, 20), date(2026, 12, 31))]


# ====================================================================== invoice approval (examples/03_invoices.py)
inv = Catalog()
MONEY = r"([0-9][0-9,]*\.[0-9]{2})"


def _find(doc, labels, pat, conv):
    for lab in labels:
        m = re.search(lab + r"\s*[:\-]?\s*" + pat, doc, flags=re.I)
        if m:
            return Quote(conv(m.group(1)), m.start(1), m.end(1))
    raise ValueError("not found: " + "/".join(labels))


@inv.extract
def invoice_id(doc):
    "invoice number"
    return _find(doc, ["Invoice", "Invoice no", "Invoice #"], r"([A-Z]{2,4}-\d+)", str)


@inv.extract
def vendor(doc):
    "who issued the invoice"
    return _find(doc, ["Supplier", "Vendor", "From"], r"([A-Z][A-Za-z&.\- ]+?)(?:\n|$)", str.strip)


@inv.extract
def amount(doc):
    "total amount due"
    return _find(doc, ["Total due", "Amount payable", "Grand total"], MONEY, lambda s: float(s.replace(",", "")))


@inv.extract
def line_items(doc):
    "line item amounts"
    ms = list(re.finditer(r"^\s*-\s.*?\s" + MONEY + r"\s*$", doc, flags=re.M))
    if not ms:
        raise ValueError("no line items")
    return Quote([float(m.group(1).replace(",", "")) for m in ms], ms[0].start(1), ms[-1].end(1))


@inv.extract
def due_date(doc):
    "payment due date"
    return _find(doc, ["Due date", "Pay by", "Payment due"], r"(\d{4}-\d{2}-\d{2})", date.fromisoformat)


@inv.fn
def lines_total(line_items):
    return round(sum(line_items), 2)


@inv.fn
def days_to_due(due_date, today):
    return (due_date - today).days


@inv.fn
def vendor_history(vendor, payments_db):
    return [p for p in payments_db if p["vendor"] == vendor]


@inv.fn
def avg_vendor_amount(vendor_history):
    return round(sum(p["amount"] for p in vendor_history) / max(1, len(vendor_history)), 2)


@inv.check
def total_matches(amount, lines_total):
    """the stated total equals the sum of the line items"""
    return abs(amount - lines_total) < 0.01


@inv.check
def known_vendor(vendor, vendors_db):
    return vendor in vendors_db


@inv.check
def not_duplicate(invoice_id, prior_invoice_ids):
    return invoice_id not in prior_invoice_ids


@inv.check
def amount_typical(amount, avg_vendor_amount):
    """at most 3x this vendor's average"""
    return avg_vendor_amount == 0 or amount <= 3 * avg_vendor_amount


@inv.check
def due_in_future(days_to_due):
    return days_to_due >= 0


# --- in the catalog but not needed for these questions: the strategist must not run them
@inv.extract
def contract_parties(contract_doc):
    "contract parties"
    return Quote(["?"], 0, 0, "contract_doc")


@inv.fn
def shipping_eta(tracking_id, carrier_api):
    return 0


@inv.fn
def doc_length(doc):
    return len(doc)


@inv.rule("approve")
def approve_rule(not_duplicate, total_matches, known_vendor, amount_typical):
    if not not_duplicate:
        return "reject"
    return "approve" if (total_matches and known_vendor and amount_typical) else "escalate"


@inv.rule("duplicate")
def duplicate_rule(not_duplicate):
    return not not_duplicate


@inv.rule("urgent")
def urgent_rule(days_to_due, due_in_future):
    return due_in_future and days_to_due <= 5


INV_QUESTIONS = [
    Question("approve", "Approve the invoice?", Answer.choice(["approve", "reject", "escalate"]), requires=["not_duplicate"]),
    Question("duplicate", "Is it a duplicate of a paid invoice?", Answer.yes_no()),
    Question("urgent", "Is payment urgent?", Answer.yes_no()),
    Question("risk", "Risk level (learned from 300 examples)", Answer.choice(["low", "medium", "high"])),
]
VENDORS = ["Acme Ltd", "Globex Corp", "Initech", "Umbrella Supplies", "Stark Parts"]
UNKNOWN = ["Shady Traders", "Quick Cash LLC", "Nowhere Inc"]
TODAY = date(2026, 9, 25)


def make_invoice(rng: random.Random, i: int):
    """Document + init_state + the risk label (a hidden rule over computed facts, 5% label noise), as in example 03."""
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
        lines = [round(x * 4, 2) for x in lines]
        total = round(sum(lines), 2) if not mismatch else round(total * 4, 2)
    doc_id = f"INV-{1000 + i}"
    lab_v, lab_t, lab_d = [("Supplier", "Total due", "Due date"), ("Vendor", "Amount payable", "Pay by"),
                           ("From", "Grand total", "Payment due")][rng.randint(0, 2)]
    body = "\n".join(f"  - item {j + 1} {x:,.2f}" for j, x in enumerate(lines))
    doc = (f"INVOICE {doc_id}\n{lab_v}: {v}\nIssued: {TODAY - timedelta(days=rng.randint(1, 20))}\n{lab_d}: {due}\n"
           f"Items:\n{body}\n{lab_t}: {total:,.2f}\n")
    dup = rng.random() < 0.1
    init = {"doc": doc, "today": TODAY, "vendors_db": list(VENDORS),
            "prior_invoice_ids": [doc_id] if dup else [f"INV-{rng.randint(1, 999)}"],
            "payments_db": history + [{"vendor": "Initech", "amount": 500.0}]}
    tm = abs(total - round(sum(lines), 2)) < 0.01
    hist = [p for p in init["payments_db"] if p["vendor"] == v]
    avg = round(sum(p["amount"] for p in hist) / max(1, len(hist)), 2)
    typ = avg == 0 or total <= 3 * avg
    score = (0 if known else 2) + (0 if tm else 1) + (0 if typ else 2) + (1 if total > 2000 else 0)
    risk = "low" if score == 0 else ("medium" if score <= 2 else "high")
    if rng.random() < 0.05:
        risk = rng.choice([x for x in ("low", "medium", "high") if x != risk])
    return init, risk


INV_SYSTEM = System(inv, INV_QUESTIONS)
_rng = random.Random(0)
RISK_HEAD = INV_SYSTEM.fit("risk", [make_invoice(_rng, i) for i in range(300)])

INVOICE_TEXT = """INVOICE INV-2041
Supplier: Globex Corp
Issued: 2026-09-18
Due date: 2026-09-29
Items:
  - 40 x steel brackets 312.00
  - 12 x hinge sets 148.80
  - freight 45.00
Total due: 505.80
"""
PAYMENTS_DB = [{"vendor": "Globex Corp", "amount": 480.0}, {"vendor": "Globex Corp", "amount": 610.0},
               {"vendor": "Acme Ltd", "amount": 1200.0}, {"vendor": "Initech", "amount": 500.0}]


# ====================================================================== delivery zones (examples/06_learned_rules.py)
ZONES = {"North": (["Aldmoor", "Kestrel Bay", "Northwick"], ["43", "44"]),
         "Central": (["Midtown", "Harlow", "Crossfield"], ["50", "51", "52"]),
         "South": (["Portsea", "Southend", "Rivermouth"], ["80", "81"]),
         "Islands": (["Isle of Fen", "Gullrock"], ["97"])}
STREETS = ["High St", "Mill Lane", "Station Rd", "Church Way", "Harbour Rd", "Oak Ave"]


def make_address(rng):
    zone = rng.choice(list(ZONES))
    cities, prefixes = ZONES[zone]
    city = rng.choice(cities)
    parts = [f"{rng.randint(1, 250)} {rng.choice(STREETS)}"]
    if rng.random() < 0.85:                                   # sometimes the city is missing
        parts.append(city if rng.random() < 0.9 else city.lower())
    if rng.random() < 0.9:                                    # sometimes the postcode is missing
        parts.append(rng.choice(prefixes) + f"{rng.randint(0, 999):03d}")
    return {"address": ", ".join(parts)}, zone


def zone_system():
    """A fresh catalog each time: learn_rule installs the learned rule into the catalog."""
    cat = Catalog()

    @cat.fn
    def address_upper(address):
        return address.upper()

    @cat.check
    def looks_complete(address):
        """soft: an address should carry a 5-digit postcode"""
        return any(t.isdigit() and len(t) == 5 for t in address.replace(",", " ").split())

    return cat, System(cat, [Question("zone", "Which delivery zone?", Answer.choice(list(ZONES)))])


SEED_ADDRESSES = [make_address(r) for r in [random.Random(0)] for _ in range(200)]
TEST_ADDRESSES = [make_address(r) for r in [random.Random(2026)] for _ in range(500)]
