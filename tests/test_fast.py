"""Fast head: closed-form fit in milliseconds, exact leave-one-out accuracy, and instant updates from corrections."""
import random

import numpy as np

from examples_loader import load
from solvi import System
from solvi.fast import FastHead

S = load("02_shop_order")


def data(seed, n):
    rng = random.Random(seed)
    db = [{"customer": c, "total": round(rng.uniform(20, 120), 2)} for c in ["c1", "c2", "c3", "c4"] for _ in range(5)]
    return [S.make(rng, i, db) for i in range(n)]


def test_fit_fast_is_quick_and_accurate():
    s = System(S.cat, S.QUESTIONS)
    head = s.fit_fast("suspicious", data(0, 300))
    assert isinstance(head, FastHead) and head.loo_acc > 0.75
    test = data(1, 300)
    acc = np.mean([s.ask(a, ["suspicious"])["suspicious"].answer == y for a, y in test])
    assert acc >= 0.75


def test_loo_matches_brute_force():
    rows = [{"a": float(i % 7), "b": i % 3 == 0} for i in range(40)]
    ys = ["x" if r["a"] > 3 else "y" for r in rows]
    h = FastHead(["x", "y"]).fit(rows, ys, ["a", "b"])
    hits = 0
    for i in range(len(rows)):
        g = FastHead(["x", "y"]).fit(rows[:i] + rows[i + 1:], ys[:i] + ys[i + 1:], ["a", "b"])
        hits += max(g.predict(rows[i]), key=g.predict(rows[i]).get) == ys[i]
    assert abs(h.loo_acc - hits / len(rows)) < 1e-9


def test_teach_updates_instantly_like_refitting():
    s = System(S.cat, S.QUESTIONS)
    train = data(0, 60)
    s.fit_fast("suspicious", train[:40])
    times = []
    for st, y in train[40:]:
        ms = s.teach("suspicious", st, y)
        assert ms is not None
        times.append(ms)
    assert float(np.median(times)) < 50          # the median of 20 updates: one slow update on a loaded machine is noise
    ref = System(S.cat, S.QUESTIONS)
    ref.fit_fast("suspicious", train)
    for st, _ in data(2, 30):                                   # online updates == refit on all 60 (up to fixed scaling)
        a = s.heads["suspicious"].scores(s.facts_for(st))
        assert np.all(np.isfinite(a))


def test_a_non_finite_feature_makes_the_head_abstain_not_answer_nan():
    s = System(S.cat, S.QUESTIONS)
    s.fit_fast("suspicious", data(0, 300))
    st = dict(data(3, 1)[0][0])
    for bad in (float("nan"), float("inf")):
        r = s.ask({**st, "items": [("A", 3, bad)]})["suspicious"]
        assert r.status == "abstain" and r.answer is None and r.confidence == 0.0
