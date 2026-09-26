"""Learning in milliseconds: a new question learned from a handful of labeled examples with a closed-form head, then corrected
one example at a time — each correction is absorbed instantly (about 0.1-0.2 ms), nothing is retrained, rules and hard checks are
untouched. Compare with `fit` (feature selection + logistic regression, seconds).

Run:  uv run python examples/10_learn_in_milliseconds.py"""
from __future__ import annotations

import importlib.util
import random
import time
from pathlib import Path

from solvi import System

spec = importlib.util.spec_from_file_location("shop", Path(__file__).with_name("02_shop_order.py"))
shop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shop)                              # the shop-order catalog: two rule questions + "suspicious?"


_db_rng = random.Random(0)
DB = [{"customer": c, "total": round(_db_rng.uniform(20, 120), 2)} for c in ["c1", "c2", "c3", "c4"] for _ in range(5)]


def orders(seed, n):                                        # same store (order history), different orders per seed
    rng = random.Random(seed)
    return [shop.make(rng, 1000 * seed + i, DB) for i in range(n)]


def accuracy(system, test):
    return sum(system.ask(s, ["suspicious"])["suspicious"].answer == y for s, y in test) / len(test)


if __name__ == "__main__":
    test = orders(99, 300)
    stream = orders(0, 200)

    slow = System(shop.cat, shop.QUESTIONS)
    t0 = time.perf_counter()
    slow.fit("suspicious", stream)
    print(f"fit on 200 examples:      {(time.perf_counter() - t0) * 1000:7.0f} ms, accuracy {accuracy(slow, test):.3f}")

    fast = System(shop.cat, shop.QUESTIONS)
    t0 = time.perf_counter()
    head = fast.fit_fast("suspicious", stream)
    print(f"fit_fast on 200 examples: {(time.perf_counter() - t0) * 1000:7.0f} ms, accuracy {accuracy(fast, test):.3f} "
          f"(exact leave-one-out {head.loo_acc:.3f}, ridge λ = {head.lam})")

    live = System(shop.cat, shop.QUESTIONS)
    live.fit_fast("suspicious", stream[:10])
    print(f"\nstart from 10 examples: accuracy {accuracy(live, test):.3f}; now a reviewer corrects orders one by one:")
    times = []
    for i, (state, label) in enumerate(stream[10:], 11):
        times.append(live.teach("suspicious", state, label))    # instant update, no retraining
        if i in (25, 50, 100, 200):
            print(f"  after {i:3d} labels: accuracy {accuracy(live, test):.3f}")
    times.sort()
    print(f"each correction took {times[len(times) // 2]:.2f} ms (median); the rule questions never changed:")
    s, _ = test[0]
    r = live.ask(s)
    print({q: r[q].answer for q in r.results}, "| why suspicious:", r["suspicious"].why)
