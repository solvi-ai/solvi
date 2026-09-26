"""Accounts payable, 3-way match: purchase order + goods receipt + supplier invoice (all JSON) -> pay / hold / reject,
duplicate?, needs a higher approver?

Every invoice line is matched against the PO line and the received quantity: nothing may be billed beyond what was received,
and the unit price may differ from the PO price by at most 2% after both are converted to the base currency (a PO in EUR
can be invoiced in USD). The invoice total is converted at the day's rate before the approval limit of the approver's role
is applied: 8 200 GBP looks under a 10 000 limit but is 10 414 USD. Duplicates are found by a normalised invoice number
("INV-001187" = "inv 1187") from the same supplier. Supplier on hold, an invoice for another PO, or a duplicate: hard checks
that reject, and the line-by-line match is then skipped.
Try: set the supplier's status to "on_hold", change a unit price by 3%, or the invoice currency to "JPY" (no rate -> abstain)."""
from __future__ import annotations

import re

from solvi import Answer, Catalog, Question

cat = Catalog()

PRICE_TOLERANCE_PCT = 2.0       # unit price may differ from the PO by at most 2% (base currency)
TOTAL_TOLERANCE = 0.01          # header total vs sum of lines, invoice currency


def _norm_number(s):
    """'INV-001187', 'inv 1187', 'Inv#1187' -> '1187'"""
    s = re.sub(r"[^A-Z0-9]", "", s.upper())
    s = re.sub(r"^(INVOICE|INV|RE|NO)+", "", s)
    return s.lstrip("0") or "0"


# ---------- header checks
@cat.fn
def supplier_status(invoice, suppliers):
    return suppliers.get(invoice["supplier_id"], {}).get("status", "unknown")


@cat.check(hard=True, then={"payment": "reject"})
def supplier_payable(supplier_status):
    """hard: only active suppliers are paid (on hold, blocked or unknown -> reject)"""
    return supplier_status == "active"


@cat.check(hard=True, then={"payment": "reject"})
def invoice_matches_po(invoice, po):
    """hard: the invoice must reference this PO and come from the PO's supplier"""
    return invoice["po_number"] == po["po_number"] and invoice["supplier_id"] == po["supplier_id"]


@cat.fn
def duplicate_of(invoice, paid_invoices):
    """an already paid invoice from the same supplier with the same normalised number"""
    key = _norm_number(invoice["invoice_number"])
    for p in paid_invoices:
        if p["supplier_id"] == invoice["supplier_id"] and _norm_number(p["invoice_number"]) == key:
            return p["invoice_number"]
    return None


@cat.check(hard=True, then={"payment": "reject", "duplicate": "yes"})
def not_duplicate(duplicate_of):
    """hard: never pay the same invoice twice"""
    return duplicate_of is None


# ---------- currency
@cat.fn
def fx_rate(invoice, fx_rates):
    """rate from the invoice currency to the base currency (a missing rate is an error, not 1.0)"""
    return fx_rates[invoice["currency"]]


@cat.fn
def po_fx_rate(po, fx_rates):
    return fx_rates[po["currency"]]


@cat.fn
def invoice_total_base(invoice, fx_rate):
    return round(invoice["total"] * fx_rate, 2)


# ---------- 3-way match, line by line
@cat.fn
def line_match(po, receipt, invoice, fx_rate, po_fx_rate):
    po_lines = {x["line"]: x for x in po["lines"]}
    received = {x["line"]: x["qty_received"] for x in receipt["lines"]}
    out = []
    for x in invoice["lines"]:
        p = po_lines.get(x["line"])
        if p is None or p["sku"] != x["sku"]:
            out.append({"line": x["line"], "sku": x["sku"], "problem": "not on the PO"})
            continue
        got = received.get(x["line"], 0)
        var = (x["unit_price"] * fx_rate - p["unit_price"] * po_fx_rate) / (p["unit_price"] * po_fx_rate) * 100
        m = {"line": x["line"], "sku": x["sku"], "qty_invoiced": x["qty"], "qty_received": got,
             "price_var_pct": round(var, 2)}
        if x["qty"] > got:
            m["problem"] = f"billed {x['qty']}, received {got}"
        elif abs(var) > PRICE_TOLERANCE_PCT:
            m["problem"] = f"price {var:+.2f}% vs PO (tolerance ±{PRICE_TOLERANCE_PCT}%)"
        out.append(m)
    return out


@cat.fn
def lines_failing(line_match):
    return [f"line {m['line']} {m['sku']}: {m['problem']}" for m in line_match if "problem" in m]


@cat.check
def header_total_ok(invoice):
    """soft: the invoice total equals the sum of its lines"""
    return abs(sum(x["qty"] * x["unit_price"] for x in invoice["lines"]) - invoice["total"]) <= TOTAL_TOLERANCE


@cat.check
def within_approval_limit(invoice_total_base, approver, approval_limits):
    """soft: the amount in base currency is within the approver's role limit"""
    return invoice_total_base <= approval_limits[approver["role"]]


# ---------- answers
@cat.rule("payment")
def payment(lines_failing, header_total_ok, within_approval_limit):
    """any failing line, a total that does not add up, or an amount over the approver's limit -> hold; otherwise pay"""
    return "pay" if not lines_failing and header_total_ok and within_approval_limit else "hold"


@cat.rule("duplicate")
def duplicate(duplicate_of):
    return duplicate_of is not None


@cat.rule("escalate_approval")
def escalate_approval(within_approval_limit):
    return not within_approval_limit


QUESTIONS = [
    Question("payment", "Pay, hold or reject the invoice?", Answer.choice(["pay", "hold", "reject"]),
             checkpoints=["supplier_payable", "invoice_matches_po", "not_duplicate"]),
    Question("duplicate", "Already paid (duplicate)?", Answer.yes_no()),
    Question("escalate_approval", "Needs a higher approver?", Answer.yes_no()),
]
