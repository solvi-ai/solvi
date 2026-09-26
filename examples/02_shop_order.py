"""E-commerce: an online order. Two questions are answered by rules; the third, "is the order suspicious?", has NO rule — the
system learns it from labeled order history (fit) and picks by itself which computed facts matter.

Run:  uv run python examples/02_shop_order.py"""
from __future__ import annotations

import random

from solvi import Answer, Catalog, Question, System
from solvi.show import show

cat = Catalog()


@cat.fn
def order_total(items):
    return round(sum(q * p for _, q, p in items), 2)


@cat.fn
def n_items(items):
    return sum(q for _, q, _ in items)


@cat.fn
def in_stock(items, stock):
    return all(stock.get(sku, 0) >= q for sku, q, _ in items)


@cat.fn
def customer_orders(customer, orders_db):
    return [o for o in orders_db if o["customer"] == customer]


@cat.fn
def avg_customer_total(customer_orders):
    return round(sum(o["total"] for o in customer_orders) / max(1, len(customer_orders)), 2)


@cat.fn
def new_customer(customer_orders):
    return len(customer_orders) == 0


@cat.fn
def address_mismatch(billing_country, shipping_country):
    return billing_country != shipping_country


@cat.check
def total_unusual(order_total, avg_customer_total):
    return avg_customer_total > 0 and order_total > 5 * avg_customer_total


@cat.check(hard=True, then={"ship_now": "no"})
def paid(payment_status):
    return payment_status == "paid"


@cat.check
def stock_ok(in_stock):
    return in_stock


@cat.rule("ship_now")
def ship_now(stock_ok):
    return stock_ok


@cat.rule("free_shipping")
def free_shipping(order_total, new_customer):
    return order_total >= 100 or new_customer


QUESTIONS = [
    Question("ship_now", "Ship now?", Answer.yes_no(), checkpoints=["paid"]),
    Question("free_shipping", "Free shipping?", Answer.yes_no()),
    Question("suspicious", "Suspicious order?", Answer.choice(["low", "review", "block"])),   # no rule — learned from history
]

STOCK = {"A": 50, "B": 3, "C": 0, "D": 20}
PRICES = {"A": 12.5, "B": 80.0, "C": 45.0, "D": 5.0}


def make(rng, i, orders_db):
    cust = rng.choice(["c1", "c2", "c3", "c4", f"new{i}"])
    items = [(sku, rng.randint(1, 4), PRICES[sku]) for sku in rng.sample(list(PRICES), rng.randint(1, 3))]
    if rng.random() < 0.15:
        items = [(sku, q * rng.randint(8, 15), p) for sku, q, p in items]
    bill = rng.choice(["DE", "FR", "ES"])
    ship = bill if rng.random() < 0.8 else rng.choice(["DE", "FR", "ES", "NG", "BR"])
    st = {"customer": cust, "items": items, "stock": STOCK, "orders_db": orders_db, "billing_country": bill,
          "shipping_country": ship, "payment_status": "paid" if rng.random() < 0.9 else "pending"}
    # hidden fraud-labeling truth: from computed facts, with 5% label noise
    hist = [o for o in orders_db if o["customer"] == cust]
    total = round(sum(q * p for _, q, p in items), 2)
    avg = sum(o["total"] for o in hist) / max(1, len(hist))
    score = 2 * (bill != ship) + 2 * (avg > 0 and total > 5 * avg) + (len(hist) == 0) + (total > 500)
    lab = "low" if score <= 1 else ("review" if score <= 3 else "block")
    if rng.random() < 0.05:
        lab = rng.choice(["low", "review", "block"])
    return st, lab


if __name__ == "__main__":
    rng = random.Random(0)
    orders_db = [{"customer": c, "total": round(rng.uniform(20, 120), 2)} for c in ["c1", "c2", "c3", "c4"] for _ in range(5)]
    history = [make(rng, i, orders_db) for i in range(300)]
    system = System(cat, QUESTIONS)
    head = system.fit("suspicious", history)
    print(f"trained on {len(history)} orders; selected features: {head.features}; cross-validated accuracy {head.cv_acc:.2f}")
    test = [make(rng, 1000 + i, orders_db) for i in range(500)]
    acc = sum(system.ask(s, ["suspicious"])["suspicious"].answer == y for s, y in test) / len(test)
    print(f"accuracy on 500 new orders: {acc:.3f} (ceiling ~0.95 due to label noise)")
    order = {"customer": "c2", "items": [("B", 12, 80.0), ("A", 2, 12.5)], "stock": STOCK, "orders_db": orders_db,
             "billing_country": "DE", "shipping_country": "NG", "payment_status": "paid"}
    print("\n=== a large order from a regular customer shipped to another country ===")
    show(system.ask(order), cat)
