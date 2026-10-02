"""Which extractor reads the fields of a text best: the decider's span pointer, the cue finder, or one after the other.

    uv run --with onnxruntime --with tokenizers python benchmarks/textin_extractors.py [--model ID] [--json out.json]

The texts are the ones in this repository that carry typed fields, each read against the value the repository's own
hand-written code (or the text's author) gives it:

    shop       the README / guide refund-address-cancel system: 18 requests written for it (English and Russian) —
               order_id (pattern), amount, currency (synonyms), purchase_date, new_address, urgent
    refunds    examples/04_refunds.py: 40 generated e-mails — order_date, reason (the example's phrases as synonyms),
               used (its phrases as cues and negatives); gold = the example's own extractors
    invoices   examples/03_invoices.py: 40 generated invoices — vendor, amount (total due), due_date; gold = its extractors
    tickets    gallery/11_refund_double_charge: the 16 support tickets — the amount claimed; gold = the task's extractor
    claims     examples/16_primitives.py: the three damage claims — the amount
    strings    30 short texts written for this benchmark (a pre-registered set, counted apart from the rest): string
               fields with no pattern — order id, new address, customer name, vendor, invoice number — in "key: value"
               lists and in sentences, next to other fields, and with cue words that are followed by no value

Each field is read with the question given (routing apart) by TextIn with each extractor: the span pointer alone
(`DeciderExtractor`), `CueExtractor` alone, and the two in order both ways. A field is "right" when the value read is the
gold value, "wrong" when a value is read that is not the gold (or the gold says the text does not state it), "missed"
when the gold has a value and nothing is read. A quote that does not parse (unparsed, unsure) is not a value: missed.
The decider: --model (default solvi-ai/solvi-base, ONNX, from the local Hugging Face cache)."""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import inspect
import json
import os
import random
import sys
import time
from collections import Counter
from typing import Literal

from pydantic import BaseModel, Field

from solvi import Catalog, Question, System
from solvi.textin import CueExtractor, DeciderExtractor, TextIn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TODAY = dt.date(2026, 9, 28)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, path))
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.join(ROOT, path)))
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.pop(0)
    return mod


def _system(inputs, questions):
    """One question per entry: its rule reads the fields (typed by the `inputs` model)."""
    cat = Catalog()
    anns = inputs.model_fields
    for name, fields in questions.items():
        def rule(**kw):
            return "done"
        rule.__signature__ = inspect.Signature([inspect.Parameter(f, inspect.Parameter.KEYWORD_ONLY,
                                                                  annotation=anns[f].annotation) for f in fields])
        rule.__annotations__ = {**{f: anns[f].annotation for f in fields}, "return": Literal["done"]}
        rule.__name__ = rule.__qualname__ = f"do_{name}"
        cat.rule(name)(rule)
    return System(cat, [Question(n, n.replace("_", " ").capitalize()) for n in questions], input_model=inputs)


# --------------------------------------------------------------------------------------------------- shop
class Shop(BaseModel):
    order_id: str = Field(description="The order number", json_schema_extra={"pattern": r"[A-Z]-\d+"})
    amount: float = Field(description="The amount paid")
    currency: Literal["EUR", "USD", "RUB"] = Field(
        description="The currency paid in",
        json_schema_extra={"synonyms": {"EUR": ["euro", "euros", "€"], "RUB": ["rubles", "roubles", "руб", "рублей",
                                                                                "₽"], "USD": ["dollars", "$"]}})
    purchase_date: dt.date = Field(description="The day the order was paid")
    new_address: str = Field(description="The new delivery address")
    urgent: bool = Field(description="Whether the request is urgent", json_schema_extra={"negative_cues": ["not urgent"]})


SHOP_Q = {"request_refund": ["order_id", "amount", "currency", "purchase_date"],
          "change_delivery_address": ["order_id", "new_address"], "cancel_order": ["order_id", "urgent"]}
N = None
SHOP = [   # (question, text, gold)
    ("request_refund", "please refund order A-10457, 1.5 million rubles, paid 12 September 2026",
     {"order_id": "A-10457", "amount": 1.5e6, "currency": "RUB", "purchase_date": dt.date(2026, 9, 12)}),
    ("request_refund", "Please refund order A-10457: I paid 1.5 million rubles on 12 September.",
     {"order_id": "A-10457", "amount": 1.5e6, "currency": "RUB", "purchase_date": dt.date(2026, 9, 12)}),
    ("request_refund", "Hi, I want my money back for A-2231. It cost 49.90 EUR and I bought it on 2026-09-20.",
     {"order_id": "A-2231", "amount": 49.9, "currency": "EUR", "purchase_date": dt.date(2026, 9, 20)}),
    ("request_refund", "Order B-77: refund please, $120 paid on September 3.",
     {"order_id": "B-77", "amount": 120.0, "currency": "USD", "purchase_date": dt.date(2026, 9, 3)}),
    ("request_refund", "I was charged 2,300 euros for order A-9 on 15.09.2026 and the item never came. Refund it.",
     {"order_id": "A-9", "amount": 2300.0, "currency": "EUR", "purchase_date": dt.date(2026, 9, 15)}),
    ("request_refund", "Refund A-500 please, the headphones broke after a day.",
     {"order_id": "A-500", "amount": N, "currency": N, "purchase_date": N}),
    ("request_refund", "верните деньги за заказ A-10457, я заплатил 1 500 000 рублей 12 сентября",
     {"order_id": "A-10457", "amount": 1.5e6, "currency": "RUB", "purchase_date": dt.date(2026, 9, 12)}),
    ("request_refund", "Прошу вернуть 3200 руб. за заказ A-311, оплачен 1 сентября 2026.",
     {"order_id": "A-311", "amount": 3200.0, "currency": "RUB", "purchase_date": dt.date(2026, 9, 1)}),
    ("request_refund", "Refund for order A-8812 — I paid half a million roubles yesterday.",
     {"order_id": "A-8812", "amount": 5e5, "currency": "RUB", "purchase_date": dt.date(2026, 9, 27)}),
    ("request_refund", "Two weeks ago I paid 75 USD for A-61. It does not work. I want a refund.",
     {"order_id": "A-61", "amount": 75.0, "currency": "USD", "purchase_date": N}),
    ("change_delivery_address", "Please deliver order A-10457 to 14 Baker Street, London instead.",
     {"order_id": "A-10457", "new_address": "14 Baker Street, London"}),
    ("change_delivery_address", "New address for A-77: 5 Rue de Rivoli, Paris",
     {"order_id": "A-77", "new_address": "5 Rue de Rivoli, Paris"}),
    ("change_delivery_address", "Can you change the address of order A-12 to Hauptstrasse 3, Berlin?",
     {"order_id": "A-12", "new_address": "Hauptstrasse 3, Berlin"}),
    ("change_delivery_address", "I moved. Please send my order somewhere else.",
     {"order_id": N, "new_address": N}),
    ("cancel_order", "Cancel order A-10457, it's urgent!", {"order_id": "A-10457", "urgent": True}),
    ("cancel_order", "Please cancel A-31 when you have time, it is not urgent.", {"order_id": "A-31", "urgent": False}),
    ("cancel_order", "I'd like to cancel my order A-900.", {"order_id": "A-900", "urgent": N}),
    ("cancel_order", "Отмените заказ A-45, срочно", {"order_id": "A-45", "urgent": N}),     # "срочно" is no declared cue
]


def shop_rows():
    return _system(Shop, SHOP_Q), [(q, t, g) for q, t, g in SHOP]


# --------------------------------------------------------------------------------------------------- refunds e-mails
def refunds_rows(n=40):
    ex = _module("examples/04_refunds.py", "ex04_refunds")

    class Refund(BaseModel):
        order_date: dt.date = Field(description="The order date")
        reason: Literal[tuple(ex.REASONS)] = Field(description="The reason for the return",
                                                   json_schema_extra={"synonyms": ex.REASONS})
        used: bool = Field(description="Whether the customer used the item",
                           json_schema_extra={"cues": ex.USED[True], "negative_cues": ex.USED[False]})

    rng = random.Random(0)
    rows = []
    for i in range(n):
        doc = ex.make(rng, i)[0]["doc"]
        gold = {}
        for f in ("order_date", "reason", "used"):
            q = getattr(ex, f).fn(doc) if hasattr(getattr(ex, f), "fn") else getattr(ex, f)(doc)
            gold[f] = q.value if q.end > q.start else None
        rows.append(("refund", doc, gold))
    return _system(Refund, {"refund": ["order_date", "reason", "used"]}), rows


# --------------------------------------------------------------------------------------------------- invoices
def invoices_rows(n=40):
    ex = _module("examples/03_invoices.py", "ex03_invoices")

    class Invoice(BaseModel):
        vendor: str = Field(description="Who issued the invoice", json_schema_extra={"cues": ["supplier", "from"]})
        amount: float = Field(description="The total amount due")
        due_date: dt.date = Field(description="The payment due date")

    rng = random.Random(0)
    rows = []
    for i in range(n):
        doc = ex.make(rng, i)[0]["doc"]
        gold = {f: _call(getattr(ex, f), doc).value for f in ("vendor", "amount", "due_date")}
        rows.append(("approve", doc, gold))
    return _system(Invoice, {"approve": ["vendor", "amount", "due_date"]}), rows


def _call(part, doc):
    fn = getattr(part, "fn", None) or getattr(part, "func", None) or part
    return fn(doc)


# --------------------------------------------------------------------------------------------------- tickets, claims
def tickets_rows():
    task = _module("gallery/11_refund_double_charge/task.py", "g11_task")
    with open(os.path.join(ROOT, "gallery/11_refund_double_charge/cases.json")) as f:
        cases = json.load(f)

    class Ticket(BaseModel):
        claimed_amount: float = Field(description="The amount of money the customer mentions")

    rows = []
    for c in cases:
        try:
            gold = _call(task.claimed_amount, c["state"]["ticket"]).value
        except ValueError:
            gold = None
        rows.append(("review", c["state"]["ticket"], {"claimed_amount": gold}))
    return _system(Ticket, {"review": ["claimed_amount"]}), rows


def claims_rows():
    ex = _module("examples/16_primitives.py", "ex16_primitives")

    class Claim(BaseModel):
        amount: float = Field(description="The amount claimed")

    gold = {"full": 149.9, "sparse": 20.0, "odd": 20.0}
    return _system(Claim, {"pay": ["amount"]}), [("pay", t, {"amount": gold[k]}) for k, t in ex.CLAIMS.items()]


# --------------------------------------------------------------------------------------------------- strings
# Written for this benchmark before CueExtractor's reading of strings without a pattern was changed (pre-registered:
# the texts and their values were fixed first, then the code was measured on them): identifiers, addresses, names and
# vendors with no pattern given, in "key: value" lists and in sentences, with other fields next to them and traps where
# a cue word is followed by no value.
class Order(BaseModel):
    order_id: str = Field(description="The order number")
    new_address: str = Field(description="The new delivery address")
    customer_name: str = Field(description="The customer's name")


class Bill(BaseModel):
    vendor: str = Field(description="Who issued the invoice")
    invoice_number: str = Field(description="The invoice's number")


STRINGS_ORDER = [   # (text, order_id, new_address, customer_name)
    ("order: A-10457, amount: 1", "A-10457", N, N),
    ("order_id: A-5, 20 EUR, bought yesterday", "A-5", N, N),
    ("Order id: 88123. New address: 12 Elm Street, Springfield. Name: Jane Doe.", "88123", "12 Elm Street, Springfield",
     "Jane Doe"),
    ("Please send order A-77 to 5 Rue de Rivoli, Paris.", "A-77", "5 Rue de Rivoli, Paris", N),
    ("Hi, my name is Tom Baker and my order is A-4471. The new address is 221B Baker Street, London.", "A-4471",
     "221B Baker Street, London", "Tom Baker"),
    ("Customer: Anna Smith, order A-12, address: Hauptstrasse 3, Berlin", "A-12", "Hauptstrasse 3, Berlin", "Anna Smith"),
    ("New address for A-31: 10 Downing Street, London", "A-31", "10 Downing Street, London", N),
    ("I moved. Please update my order.", N, N, N),
    ("Order number: B-2207; deliver to: 7 King's Road, Chelsea; name: Lee Chang", "B-2207", "7 King's Road, Chelsea",
     "Lee Chang"),
    ("Заказ: A-900, новый адрес: ул. Ленина 5, Казань", "A-900", "ул. Ленина 5, Казань", N),
    ("order A-10457, please change the address to 14 Baker Street, London", "A-10457", "14 Baker Street, London", N),
    ("The order is A-555 and the customer is Maria Lopez.", "A-555", N, "Maria Lopez"),
    ("name: Ivan Petrov, order: A-1001, new address: Nevsky 28, St Petersburg", "A-1001", "Nevsky 28, St Petersburg",
     "Ivan Petrov"),
    ("Ticket about order A-3: the parcel never arrived.", "A-3", N, N),
    ("Address unchanged. Order: Z-99.", "Z-99", N, N),
    ("Customer name: O'Brien, Patrick. Order: A-640.", "A-640", N, "O'Brien, Patrick"),
    ("My order A-2 was sent to the old address, please use 9 High Street, Oxford instead.", "A-2", "9 High Street, Oxford",
     N),
    ("order: A-10457 amount: 15 EUR", "A-10457", N, N),
    ("Hello, this is Sarah Connor. Order A-1984. Please ship to 1 Main St, Los Angeles.", "A-1984", "1 Main St, Los Angeles",
     "Sarah Connor"),
    ("order: 4471-B / name: Kim", "4471-B", N, "Kim"),
]
STRINGS_BILL = [   # (text, vendor, invoice_number)
    ("Invoice number: INV-2231, vendor: Acme Corp, total: 1,200 EUR", "Acme Corp", "INV-2231"),
    ("Vendor: Globex Ltd. Invoice: 7781.", "Globex Ltd", "7781"),
    ("invoice INV-77 from Initech, due 2026-10-01", "Initech", "INV-77"),
    ("Invoice: 5521, supplier: Wayne Enterprises", "Wayne Enterprises", "5521"),
    ("Vendor is Stark Industries, invoice number is SI-0042.", "Stark Industries", "SI-0042"),
    ("Invoice: SI-9 (vendor: Umbrella)", "Umbrella", "SI-9"),
    ("Please pay the attached invoice.", N, N),
    ("vendor = Hooli; invoice = H-12", "Hooli", "H-12"),
    ("The vendor will be confirmed later. Invoice: X-1", N, "X-1"),
    ("Invoice for order A-5: issued by Soylent Inc, number SO-81.", "Soylent Inc", "SO-81"),
]


def strings_rows():
    """Two systems' worth of string fields in one: the order fields and the invoice fields, each its own question."""
    class Strings(Order, Bill):
        pass

    rows = [("change_order", t, {"order_id": o, "new_address": a, "customer_name": c}) for t, o, a, c in STRINGS_ORDER]
    rows += [("register_invoice", t, {"vendor": v, "invoice_number": i}) for t, v, i in STRINGS_BILL]
    return _system(Strings, {"change_order": ["order_id", "new_address", "customer_name"],
                             "register_invoice": ["vendor", "invoice_number"]}), rows


# --------------------------------------------------------------------------------------------------- measuring
def same(a, b):
    if isinstance(b, float) and isinstance(a, (int, float)):
        return abs(a - b) < 1e-6
    if isinstance(b, str) and isinstance(a, str):
        return a.strip().casefold() == b.strip().casefold()
    return a == b


def judge(fr, gold):
    if fr.ok:
        return "right" if gold is not None and same(fr.value, gold) else "wrong"
    return "missed" if gold is not None else "absent"


PREREGISTERED = ("strings",)     # counted apart: "all" stays the repository's texts


def run(system, rows, extractor, decider):
    tin = TextIn(system, decider, extractor, today=TODAY)
    c, per = Counter(), {}
    for q, text, gold in rows:
        read = tin.read(text, question=q)
        for f, g in gold.items():
            v = judge(read.fields[f], g)
            c[v] += 1
            per.setdefault(f, Counter())[v] += 1
    return c, per


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="solvi-ai/solvi-base")
    ap.add_argument("--backend", default="onnx")
    ap.add_argument("--json")
    a = ap.parse_args()
    from solvi.decide import DecideModel
    t0 = time.perf_counter()
    m = DecideModel.load(a.model, backend=a.backend)
    print(f"{a.model} ({m.backend}) loaded in {time.perf_counter() - t0:.1f} s; pointer {m.has_pointer}")
    sets = {"shop": shop_rows(), "refunds": refunds_rows(), "invoices": invoices_rows(), "tickets": tickets_rows(),
            "claims": claims_rows(), "strings": strings_rows()}
    ptr, cue = DeciderExtractor(m), CueExtractor()
    extractors = {"pointer": ptr, "cue": cue, "pointer, then cue": [ptr, cue], "cue, then pointer": [cue, ptr]}
    out = {"model": a.model, "backend": m.backend, "sets": {}}
    total = {k: Counter() for k in extractors}
    total_str = {k: Counter() for k in extractors}
    print(f"{'set':10} {'extractor':18} {'right':>6} {'wrong':>6} {'missed':>7} {'absent':>7}")
    for name, (system, rows) in sets.items():
        out["sets"][name] = {}
        for label, ex in extractors.items():
            c, per = run(system, rows, ex, m)
            (total_str if name in PREREGISTERED else total)[label].update(c)
            out["sets"][name][label] = {"total": dict(c), "per_field": {f: dict(v) for f, v in per.items()}}
            print(f"{name:10} {label:18} {c['right']:6} {c['wrong']:6} {c['missed']:7} {c['absent']:7}")
    print()
    for label, c in total.items():
        print(f"{'all':10} {label:18} {c['right']:6} {c['wrong']:6} {c['missed']:7} {c['absent']:7}")
    for label, c in total_str.items():
        print(f"{'strings':10} {label:18} {c['right']:6} {c['wrong']:6} {c['missed']:7} {c['absent']:7}")
    out["total"] = {k: dict(v) for k, v in total.items()}
    out["total_strings"] = {k: dict(v) for k, v in total_str.items()}
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
