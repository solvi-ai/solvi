"""Accounts payable, 3-way match: purchase order + goods receipt + supplier invoice (all JSON) -> pay / hold / reject,
duplicate?, needs a higher approver?

Every invoice line is matched against the PO line and the received quantity: nothing may be billed beyond what was received,
and the unit price may differ from the PO price by at most 2% after both are converted to the base currency (a PO in EUR
can be invoiced in USD). The invoice total is converted at the day's rate before the approval limit of the approver's role
is applied: 8 200 GBP looks under a 10 000 limit but is 10 414 USD. Duplicates are found by a normalised invoice number
("INV-001187" = "inv 1187") from the same supplier. Supplier on hold, an invoice for another PO, or a duplicate: hard checks
that reject, and the line-by-line match is then skipped.
The invoice rate comes from the rate table, or else from the daily feed (`fx_feed`) if the feed is dated the invoice day:
two producers of one fact, each with its own `validate`.
Typed (solvi 0.5): the documents are pydantic models and every function says what it reads and returns. The catalog checks
producer against consumer types when the functions are registered; at run time a malformed document (a line without a unit
price, a quantity "four hundred") is rejected with the field named (safeguard `type_rejected`) and the answers that need it
abstain, instead of a KeyError deep in the line match. `System(cat, QUESTIONS, inputs=Request)` validates a request once.
Try: set the supplier's status to "on_hold", change a unit price by 3%, or the invoice currency to "JPY" (no rate in the
table or the feed -> abstain; add "JPY": 0.0062 to the feed's rates -> the feed is used); set a line's qty to "many"."""

import re
from datetime import date
from typing import Any, Literal, Optional

from pydantic import BaseModel

from solvi import Answer, Catalog, Question

cat = Catalog()


# ---------- the documents
class POLine(BaseModel):
    line: int
    sku: str
    description: str = ""
    qty: int
    unit_price: float


class PurchaseOrder(BaseModel):
    po_number: str
    supplier_id: str
    currency: str
    lines: list[POLine]


class ReceiptLine(BaseModel):
    line: int
    sku: str
    qty_received: int


class GoodsReceipt(BaseModel):
    grn_number: str
    po_number: str
    lines: list[ReceiptLine]


class InvoiceLine(BaseModel):
    line: int
    sku: str
    qty: int
    unit_price: float


class Invoice(BaseModel):
    invoice_number: str
    supplier_id: str
    po_number: str
    currency: str
    date: date
    lines: list[InvoiceLine]
    total: float


class FxFeed(BaseModel):
    source: str = ""
    as_of: date
    rates: dict[str, float]


Status = Literal["active", "on_hold", "blocked", "unknown"]
Role = Literal["clerk", "manager", "director"]


class Supplier(BaseModel):
    name: str
    status: Status


class PaidInvoice(BaseModel):
    supplier_id: str
    invoice_number: str
    amount: float
    currency: str
    date: date


class Approver(BaseModel):
    name: str
    role: Role


class Request(BaseModel):
    """init_state of this task (System(..., inputs=Request))."""
    po: PurchaseOrder
    receipt: GoodsReceipt
    invoice: Invoice
    fx_rates: dict[str, float]
    fx_feed: FxFeed
    suppliers: dict[str, Supplier]
    paid_invoices: list[PaidInvoice]
    approver: Approver
    approval_limits: dict[Role, float]

PRICE_TOLERANCE_PCT = 2.0       # unit price may differ from the PO by at most 2% (base currency)
TOTAL_TOLERANCE = 0.01          # header total vs sum of lines, invoice currency


def _norm_number(s: str) -> str:
    """'INV-001187', 'inv 1187', 'Inv#1187' -> '1187'"""
    s = re.sub(r"[^A-Z0-9]", "", s.upper())
    s = re.sub(r"^(INVOICE|INV|RE|NO)+", "", s)
    return s.lstrip("0") or "0"


# ---------- header checks
@cat.fn
def supplier_status(invoice: Invoice, suppliers: dict[str, Supplier]) -> Status:
    s = suppliers.get(invoice.supplier_id)
    return s.status if s is not None else "unknown"


@cat.check(hard=True, then={"payment": "reject"})
def supplier_payable(supplier_status: Status) -> bool:
    """hard: only active suppliers are paid (on hold, blocked or unknown -> reject)"""
    return supplier_status == "active"


@cat.check(hard=True, then={"payment": "reject"})
def invoice_matches_po(invoice: Invoice, po: PurchaseOrder) -> bool:
    """hard: the invoice must reference this PO and come from the PO's supplier"""
    return invoice.po_number == po.po_number and invoice.supplier_id == po.supplier_id


@cat.fn
def duplicate_of(invoice: Invoice, paid_invoices: list[PaidInvoice]) -> Optional[str]:
    """an already paid invoice from the same supplier with the same normalised number"""
    key = _norm_number(invoice.invoice_number)
    for p in paid_invoices:
        if p.supplier_id == invoice.supplier_id and _norm_number(p.invoice_number) == key:
            return p.invoice_number
    return None


@cat.check(hard=True, then={"payment": "reject", "duplicate": "yes"})
def not_duplicate(duplicate_of: Optional[str]) -> bool:
    """hard: never pay the same invoice twice"""
    return duplicate_of is None


# ---------- currency
# the invoice rate has two producers, tried in order: the company's rate table, then the daily reference feed. A producer's
# output is used only if it passes its own validate; a missing rate is "not found" (None), never 1.0. The record of fx_rate
# says which producer was used and what happened to the ones tried before it; the audit shows it as a fallback.
@cat.fn(provides="fx_rate", cost=0.01, validate=lambda rate: rate > 0)
def fx_rate_table(invoice: Invoice, fx_rates: dict[str, float]) -> float:
    """rate from the invoice currency to the base currency, from the company's rate table"""
    return fx_rates.get(invoice.currency)


@cat.fn(provides="fx_rate", cost=5.0, validate=lambda rate, invoice, fx_feed: rate > 0 and fx_feed.as_of == invoice.date)
def fx_rate_feed(invoice: Invoice, fx_feed: FxFeed) -> float:
    """fallback: the daily reference feed, accepted only when its rates are from the invoice date (a stale rate is rejected)"""
    return fx_feed.rates.get(invoice.currency)


@cat.fn
def po_fx_rate(po: PurchaseOrder, fx_rates: dict[str, float]) -> float:
    return fx_rates[po.currency]


@cat.fn
def invoice_total_base(invoice: Invoice, fx_rate: float) -> float:
    return round(invoice.total * fx_rate, 2)


# ---------- 3-way match, line by line
@cat.fn
def line_match(po: PurchaseOrder, receipt: GoodsReceipt, invoice: Invoice, fx_rate: float,
               po_fx_rate: float) -> list[dict[str, Any]]:
    po_lines = {x.line: x for x in po.lines}
    received = {x.line: x.qty_received for x in receipt.lines}
    out = []
    for x in invoice.lines:
        p = po_lines.get(x.line)
        if p is None or p.sku != x.sku:
            out.append({"line": x.line, "sku": x.sku, "problem": "not on the PO"})
            continue
        got = received.get(x.line, 0)
        var = (x.unit_price * fx_rate - p.unit_price * po_fx_rate) / (p.unit_price * po_fx_rate) * 100
        m = {"line": x.line, "sku": x.sku, "qty_invoiced": x.qty, "qty_received": got, "price_var_pct": round(var, 2)}
        if x.qty > got:
            m["problem"] = f"billed {x.qty}, received {got}"
        elif abs(var) > PRICE_TOLERANCE_PCT:
            m["problem"] = f"price {var:+.2f}% vs PO (tolerance ±{PRICE_TOLERANCE_PCT}%)"
        out.append(m)
    return out


@cat.fn
def lines_failing(line_match: list[dict[str, Any]]) -> list[str]:
    return [f"line {m['line']} {m['sku']}: {m['problem']}" for m in line_match if "problem" in m]


@cat.check
def header_total_ok(invoice: Invoice) -> bool:
    """soft: the invoice total equals the sum of its lines"""
    return abs(sum(x.qty * x.unit_price for x in invoice.lines) - invoice.total) <= TOTAL_TOLERANCE


@cat.check
def within_approval_limit(invoice_total_base: float, approver: Approver, approval_limits: dict[Role, float]) -> bool:
    """soft: the amount in base currency is within the approver's role limit"""
    return invoice_total_base <= approval_limits[approver.role]


# ---------- answers (return types: the closed sets; each is checked against its question's options)
@cat.rule("payment")
def payment(lines_failing: list[str], header_total_ok: bool, within_approval_limit: bool) -> Literal["pay", "hold"]:
    """any failing line, a total that does not add up, or an amount over the approver's limit -> hold; otherwise pay
    (reject comes only from the hard checks)"""
    return "pay" if not lines_failing and header_total_ok and within_approval_limit else "hold"


@cat.rule("duplicate")
def duplicate(duplicate_of: Optional[str]) -> bool:
    return duplicate_of is not None


@cat.rule("escalate_approval")
def escalate_approval(within_approval_limit: bool) -> bool:
    return not within_approval_limit


QUESTIONS = [
    Question("payment", "Pay, hold or reject the invoice?", Answer.choice(["pay", "hold", "reject"]),
             checkpoints=["supplier_payable", "invoice_matches_po", "not_duplicate"]),
    Question("duplicate", "Already paid (duplicate)?", Answer.yes_no()),
    Question("escalate_approval", "Needs a higher approver?", Answer.yes_no()),
]
