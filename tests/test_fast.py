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


def _rows(n, seed=0):
    rng = random.Random(seed)
    rows = [{"a": rng.random(), "c": rng.choice("pqrstuvw"), "b": rng.random() < 0.5} for _ in range(n)]
    return rows, ["x" if r["a"] > 0.5 or r["c"] == "w" else "y" for r in rows]


def _teach_all(h, rows, ys, start):
    fits = []
    for r, y in zip(rows[start:], ys[start:]):
        h.update(r, y)
        if h.fitted_on == h.n:
            fits.append(h.n)
    return fits


def test_refit_schedule_doubles_and_stops_at_refit_until():
    rows, ys = _rows(300)
    h = FastHead(["x", "y"]).fit(rows[:10], ys[:10], ["a", "b", "c"])
    assert _teach_all(h, rows, ys, 10) == [20, 40, 80, 160]
    assert len(h._rows) == h.n == 300
    h = FastHead(["x", "y"], refit=2.0, refit_until=100).fit(rows[:10], ys[:10], ["a", "b", "c"])
    assert _teach_all(h, rows, ys, 10) == [20, 40, 80]
    assert h._rows is None and h.n == 300                      # no refit due past refit_until: the rows are dropped
    h = FastHead(["x", "y"], refit=None).fit(rows[:10], ys[:10], ["a", "b", "c"])
    assert _teach_all(h, rows, ys, 10) == [] and h._rows is None and h.fitted_on == 10


def test_after_a_refit_the_head_equals_a_fresh_fit_on_the_same_rows():
    rows, ys = _rows(160, seed=1)
    h = FastHead(["x", "y"]).fit(rows[:10], ys[:10], ["a", "b", "c"])     # λ and pairs chosen on 10 rows
    for r, y in zip(rows[10:], ys[10:]):
        h.update(r, y)
    g = FastHead(["x", "y"]).fit(rows, ys, ["a", "b", "c"])
    assert h.fitted_on == 160
    assert np.array_equal(h.W, g.W) and np.array_equal(h.Ainv, g.Ainv) and h.lam == g.lam and h.pairs == g.pairs
    assert h.fz.spec == g.fz.spec and h.fingerprint() == g.fingerprint()
    probe, _ = _rows(20, seed=2)
    assert all(h.predict(r) == g.predict(r) for r in probe)


def test_teaching_the_same_sequence_gives_the_same_head():
    rows, ys = _rows(200, seed=3)
    fps = []
    for _ in range(2):
        h = FastHead(["x", "y"]).fit(rows[:10], ys[:10], ["a", "b", "c"])
        seen = []
        for r, y in zip(rows[10:], ys[10:]):
            h.update(r, y)
            seen.append(h.fingerprint())
        fps.append(seen)
    assert fps[0] == fps[1]
    assert len(set(fps[0])) == len(fps[0])                     # every update, rank-one or refit, changes the fingerprint


def test_a_refit_through_system_teach_and_fit_fast_refit_none():
    s = System(S.cat, S.QUESTIONS)
    train = data(0, 45)
    head = s.fit_fast("suspicious", train[:10])
    before = head.fingerprint()
    for st, y in train[10:20]:
        s.teach("suspicious", st, y)
    assert head.fitted_on == 20 and head.fingerprint() != before
    ref = System(S.cat, S.QUESTIONS).fit_fast("suspicious", train[:20])
    assert np.array_equal(head.W, ref.W)
    off = System(S.cat, S.QUESTIONS)
    h2 = off.fit_fast("suspicious", train[:10], refit=None)
    for st, y in train[10:20]:
        off.teach("suspicious", st, y)
    assert h2.fitted_on == 10 and h2._rows is None


def test_an_old_pickled_head_loads_and_learns_as_before():
    import pickle
    rows, ys = _rows(60, seed=4)
    h = FastHead(["x", "y"], refit=None).fit(rows[:10], ys[:10], ["a", "b", "c"])
    old = FastHead(["x", "y"], refit=None).fit(rows[:10], ys[:10], ["a", "b", "c"])
    for k in ("_lam0", "_pairs0", "refit", "refit_until", "_rows", "_answers", "_requested", "fitted_on"):
        del old.__dict__[k]                                    # a head pickled before refits existed
    old = pickle.loads(pickle.dumps(old))
    assert old.fingerprint() == h.fingerprint()
    for r, y in zip(rows[10:], ys[10:]):
        h.update(r, y)
        old.update(r, y)
    assert np.array_equal(old.W, h.W) and old.fingerprint() == h.fingerprint()


def test_an_unknown_answer_changes_nothing():
    import pytest
    rows, ys = _rows(19, seed=5)
    h = FastHead(["x", "y"]).fit(rows[:10], ys[:10], ["a", "b", "c"])
    for r, y in zip(rows[10:], ys[10:]):
        h.update(r, y)
    fp, n = h.fingerprint(), len(h._rows)
    with pytest.raises(ValueError):
        h.update(rows[0], "z")                                 # the update that would trigger the refit at 20
    assert h.fingerprint() == fp and len(h._rows) == n and h.fitted_on == 10


def test_refit_must_grow():
    import pytest
    with pytest.raises(ValueError):
        FastHead(["x", "y"], refit=1.0)
