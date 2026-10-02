"""Documents with a model: expense check on a scanned receipt. A ModernBERT extractor fine-tuned on receipts finds the date and
the amounts (each cited at its exact place in the OCR text); plain Python rules decide, including a computed check of the change. Needs `pip install "solvi[model]"`.

Run:  uv run --extra model python examples/07_receipts_model.py
Model: SOLVI_MODEL (a Hugging Face id or a local directory), default solvi-ai/extract-receipts.
To train your own on your documents, see LongSpanExtractor.fit (the dataset loaders are in benchmarks/datasets)."""
from __future__ import annotations

import os
import re
from datetime import date

from solvi import Answer, Catalog, Question, System
from solvi.extract_long import LongSpanExtractor
from solvi.show import show

RECEIPT = """SANYU STATIONERY SHOP
NO. 31G&33G, JALAN SETIA INDAH X ,U13/X
40170 SETIA ALAM
Mobile /Whatsapps : +6012-918 7937
Tel: +603-3362 4137
GST ID No: 001531760640
TAX INVOICE
Owned By :
SANYU SUPPLY SDN BHD (1135772-K)
CASH SALES COUNTER
1. 2012-0043 1 X 3.3000 3.30 SR
JOURNAL BOOK 80PGS A4 70G
CARD COVER (SJB-4013)
Total Sales Inclusive GST @6% 3.30
Discount 0.00
Total 3.30
Round Adjustment 0.00
Final Total 3.30
CASH 5.00
CHANGE 1.70
Date : 21/03/2018 Time: 14:53
"""

FIELDS = {"receipt_date": "the date of the purchase",
          "total": "the total amount to pay",
          "cash": "the cash amount given by the customer",
          "change": "the change returned to the customer"}

extractor = LongSpanExtractor.load(os.environ.get("SOLVI_MODEL", "solvi-ai/extract-receipts"))
cat = Catalog()
for name, description in FIELDS.items():
    cat.extract(extractor.field(name, description))


def money(text):
    return float(re.findall(r"\d+(?:\.\d+)?", text.replace(",", ""))[-1])


@cat.fn
def amount(total):
    return money(total)


@cat.fn
def purchase_date(receipt_date):
    d, m, y = (int(x) for x in re.findall(r"\d+", receipt_date)[:3])        # DD/MM/YYYY on these receipts
    return date(y if y > 99 else 2000 + y, m, d)


@cat.check(hard=True, then={"reimburse": "no"})
def not_too_old(purchase_date, today):
    """hard: receipts older than 90 days are never reimbursed"""
    return (today - purchase_date).days <= 90


@cat.rule("reimburse")
def reimburse(amount):
    return amount <= 200


@cat.rule("change_correct")
def change_correct(amount, cash, change):
    return abs(money(cash) - amount - money(change)) < 0.01


@cat.rule("weekend")
def weekend(purchase_date):
    return purchase_date.weekday() >= 5


QUESTIONS = [Question("reimburse", "Reimburse this receipt?", Answer.yes_no(), checkpoints=["not_too_old"]),
             Question("change_correct", "Is the change right (cash - total)?", Answer.yes_no()),
             Question("weekend", "Bought on a weekend?", Answer.yes_no())]

if __name__ == "__main__":
    system = System(cat, QUESTIONS)
    show(system.ask({"doc": RECEIPT, "today": date(2018, 4, 2)}), cat)
    print("\n=== same receipt, submitted a year later ===")
    show(system.ask({"doc": RECEIPT, "today": date(2019, 4, 2)}), cat, flow=False)
