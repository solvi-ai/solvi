"""CORD v2 receipts (Indonesia; 800 train / 100 validation / 100 test).

The text is assembled from the OCR lines; field spans are exact (from the labelled words); typed questions and their
ground truth come from the reference annotation."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("SOLVI_DATA", "data")) / "cord"
FIELDS = {"total": ("total.total_price", "the total amount to pay"),
          "subtotal": ("sub_total.subtotal_price", "the subtotal before tax and service"),
          "tax": ("sub_total.tax_price", "the tax amount"),
          "cash": ("total.cashprice", "the cash amount given by the customer"),
          "change": ("total.changeprice", "the change returned to the customer")}
# descriptions of all CORD field categories (for training a describe-the-field extractor); repeated fields: first value
CATEGORY_DESC = {
    "menu.nm": "the name of the first purchased item", "menu.price": "the price of the first purchased item",
    "menu.cnt": "the quantity of the first purchased item", "menu.unitprice": "the unit price of the first purchased item",
    "menu.num": "the item code of the first purchased item", "menu.discountprice": "the discount on the first purchased item",
    "menu.sub.nm": "the name of the first sub-item or add-on", "menu.sub.price": "the price of the first sub-item or add-on",
    "menu.sub.cnt": "the quantity of the first sub-item or add-on",
    "sub_total.subtotal_price": "the subtotal before tax and service", "sub_total.tax_price": "the tax amount",
    "sub_total.service_price": "the service charge", "sub_total.discount_price": "the discount on the whole bill",
    "sub_total.etc": "another amount listed in the subtotal section",
    "total.total_price": "the total amount to pay", "total.cashprice": "the cash amount given by the customer",
    "total.changeprice": "the change returned to the customer", "total.creditcardprice": "the amount paid by credit card",
    "total.emoneyprice": "the amount paid by e-money", "total.menuqty_cnt": "the total number of items",
    "total.menutype_cnt": "the number of different item types", "total.total_etc": "another amount listed in the total section"}
BANDS = ["under 50k", "50k to 100k", "100k to 300k", "over 300k"]


def num(s):
    """Indonesian amounts: "75,000", "75.000", "Rp. 1.250.000" -> number."""
    d = re.findall(r"\d[\d.,]*", str(s))
    if not d:
        raise ValueError(f"no number: {s!r}")
    x = d[-1]
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", x):
        return float(re.sub(r"[.,]", "", x))
    return float(x.replace(",", ""))


def band(t):
    return BANDS[0] if t < 50_000 else BANDS[1] if t < 100_000 else BANDS[2] if t < 300_000 else BANDS[3]


def load(split):
    out = []
    for i, line in enumerate((DATA / f"{split}.jsonl").read_text().splitlines()):
        g = json.loads(line)
        groups = g["valid_line"]
        # lines: words of one OCR row (row_id), rows sorted top to bottom, words left to right
        rows = {}
        for gi, grp in enumerate(groups):
            for w in grp["words"]:
                rows.setdefault(w["row_id"], []).append((w["quad"]["x1"], w["quad"]["y1"], w["text"], gi, w.get("is_key", 0)))
        order = sorted(rows.values(), key=lambda r: min(x[1] for x in r))
        text, spans_by_group = "", {}
        for r in order:
            r = sorted(r, key=lambda x: x[0])
            parts = []
            for x, y, t, gi, is_key in r:
                if parts or text and not text.endswith("\n"):
                    text += " "
                s = len(text)
                text += t
                if not is_key:                            # value span excludes key words ("Grand Total")
                    a, b = spans_by_group.get(gi, (s, s + len(t)))
                    spans_by_group[gi] = (min(a, s), max(b, s + len(t)))
                parts.append(t)
            text += "\n"
        spans, gold = {}, {}
        for f, (cat, _) in FIELDS.items():
            gis = [gi for gi, grp in enumerate(groups) if grp["category"] == cat and gi in spans_by_group]
            if gis:
                a, b = spans_by_group[gis[0]]
                spans[f] = (a, b)
                gold[f] = text[a:b]
            else:
                spans[f], gold[f] = None, None
        cats = {}                                          # all field categories: span of the first value
        for gi, grp in enumerate(groups):
            if gi in spans_by_group and grp["category"] not in cats:
                cats[grp["category"]] = spans_by_group[gi]
        out.append({"key": f"{split}{i}", "text": text, "spans": spans, "gold": gold, "cats": cats})
    return out


def labels(gold):
    lab = {}
    try:
        t = num(gold["total"]) if gold["total"] else None
    except ValueError:
        t = None
    if t is not None:
        lab["total_band"] = band(t)
    lab["has_tax"] = "yes" if gold["tax"] and _pos(gold["tax"]) else "no"
    lab["paid_cash"] = "yes" if gold["cash"] else "no"
    if t is not None and gold["cash"] and gold["change"]:
        try:
            lab["change_ok"] = "yes" if abs(num(gold["cash"]) - t - num(gold["change"])) < 1 else "no"
        except ValueError:
            pass
    return lab


def _pos(s):
    try:
        return num(s) > 0
    except ValueError:
        return False


LAYA_Q = {
    "total_band": {"type": "choice", "instructions": "What is the total amount to pay on this receipt (Indonesian rupiah)?",
                   "criteria": {b: f"The total is {b} rupiah." for b in BANDS}},
    "has_tax": {"type": "noul", "instructions": "The receipt charges a tax.",
                "criteria": {"false": "No tax is charged.", "true": "A tax is charged."}},
    "paid_cash": {"type": "noul", "instructions": "The customer paid in cash.",
                  "criteria": {"false": "Not paid in cash.", "true": "Paid in cash."}},
    "change_ok": {"type": "noul", "instructions": "The change equals the cash given minus the total.",
                  "criteria": {"false": "The change does not match.", "true": "The change matches."}},
}
