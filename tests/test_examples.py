"""The example scripts run and give the expected answers."""
import random
from datetime import date

from solvi import System
from examples_loader import load

L = load("01_leave_request")
S = load("02_shop_order")
from solvi.show import show


def test_leave_request(capsys):
    s = System(L.cat, L.QUESTIONS)
    r = s.ask(L.REQUEST)
    assert r["approve"].answer == "approve" and r["notify_hr"].answer == "yes"
    low = s.ask({**L.REQUEST, "balance": 5})["approve"]
    assert low.answer == "reject" and low.status == "forced"
    xmas = s.ask({**L.REQUEST, "start": date(2026, 12, 21), "end": date(2026, 12, 23)})["approve"]
    assert xmas.answer == "needs_manager"
    show(r, L.cat)
    assert "trace replay" in capsys.readouterr().out


def test_shop_order_fit_and_ask():
    rng = random.Random(0)
    db = [{"customer": c, "total": round(rng.uniform(20, 120), 2)} for c in ["c1", "c2", "c3", "c4"] for _ in range(5)]
    s = System(S.cat, S.QUESTIONS)
    s.fit("suspicious", [S.make(rng, i, db) for i in range(200)])
    test = [S.make(rng, 1000 + i, db) for i in range(200)]
    acc = sum(s.ask(a, ["suspicious"])["suspicious"].answer == y for a, y in test) / len(test)
    assert acc >= 0.8
    order = {"customer": "c2", "items": [("B", 12, 80.0)], "stock": S.STOCK, "orders_db": db,
             "billing_country": "DE", "shipping_country": "DE", "payment_status": "pending"}
    assert s.ask(order)["ship_now"].answer == "no"            # hard check "paid"
