"""fit (closed-form ridge head; the facts selected by leave-one-out error) vs fit(select=False) (every fact) on the example
tasks: training time, accuracy on fresh examples, and online learning — start from 10 labeled examples and teach the rest one by one.

Run:  uv run python benchmarks/fast_head.py"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from examples_loader import load  # noqa: E402

from solvi import System  # noqa: E402


def tasks():
    S, R = load("02_shop_order"), load("04_refunds")

    def shop(seed, n):
        rng = random.Random(seed)
        db = [{"customer": c, "total": round(rng.uniform(20, 120), 2)} for c in ["c1", "c2", "c3", "c4"] for _ in range(5)]
        return [S.make(rng, i, db) for i in range(n)]

    def refunds(seed, n):
        rng = random.Random(seed)
        return [(i_, t["refund"]) for i_, t in (R.make(rng, i) for i in range(n))]
    return [("shop order: suspicious (3 classes)", S, "suspicious", shop), ("refund e-mail: approve (yes/no)", R, "refund", refunds)]


if __name__ == "__main__":
    for title, mod, q, gen in tasks():
        test = gen(99, 400)
        print(f"== {title}")
        for n in (20, 50, 200):
            train = gen(0, n)
            row = [f"{n:4d} examples:"]
            for kind, kw in (("fit", {}), ("select=False", {"select": False})):
                s = System(mod.cat, mod.QUESTIONS)
                t0 = time.perf_counter()
                s.fit(q, train, **kw)
                ms = (time.perf_counter() - t0) * 1000
                acc = np.mean([s.ask(a, [q])[q].answer == y for a, y in test])
                row.append(f"{kind} {acc:.3f} in {ms:7.1f} ms")
            print("  " + "   ".join(row))
        s = System(mod.cat, mod.QUESTIONS)
        stream = gen(0, 200)
        s.fit(q, stream[:10], select=False)
        curve, times = [], []
        for i, (st, y) in enumerate(stream[10:], 11):
            times.append(s.teach(q, st, y))
            if i in (20, 50, 100, 200):
                curve.append(f"{i}: {np.mean([s.ask(a, [q])[q].answer == yy for a, yy in test]):.3f}")
        print(f"  online (teach one by one from 10): {', '.join(curve)}; median update {np.median(times):.2f} ms")
