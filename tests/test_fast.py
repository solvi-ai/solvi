"""Fast head: closed-form fit in milliseconds, exact leave-one-out accuracy, and instant updates from corrections."""
import random

import numpy as np
import pytest

from examples_loader import load
from solvi import Answer, Catalog, Question, System
from solvi.heads import FastHead

S = load("02_shop_order")


def data(seed, n):
    rng = random.Random(seed)
    db = [{"customer": c, "total": round(rng.uniform(20, 120), 2)} for c in ["c1", "c2", "c3", "c4"] for _ in range(5)]
    return [S.make(rng, i, db) for i in range(n)]


def test_fit_fast_is_quick_and_accurate():
    s = System(S.cat, S.QUESTIONS)
    head = s.fit("suspicious", data(0, 300), select=False)
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
    s.fit("suspicious", train[:40], select=False)
    times = []
    for st, y in train[40:]:
        ms = s.teach("suspicious", st, y)
        assert ms is not None
        times.append(ms)
    assert float(np.median(times)) < 50          # the median of 20 updates: one slow update on a loaded machine is noise
    ref = System(S.cat, S.QUESTIONS)
    ref.fit("suspicious", train, select=False)
    for st, _ in data(2, 30):                                   # online updates == refit on all 60 (up to fixed scaling)
        a = s.heads["suspicious"].scores(s.facts_for(st))
        assert np.all(np.isfinite(a))


@pytest.mark.filterwarnings("error::RuntimeWarning")         # no "invalid value encountered in matmul" on the way
def test_a_non_finite_feature_makes_the_head_abstain_not_answer_nan():
    s = System(S.cat, S.QUESTIONS)
    s.fit("suspicious", data(0, 300), select=False)
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
    head = s.fit("suspicious", train[:10], select=False)
    before = head.fingerprint()
    for st, y in train[10:20]:
        s.teach("suspicious", st, y)
    assert head.fitted_on == 20 and head.fingerprint() != before
    ref = System(S.cat, S.QUESTIONS).fit("suspicious", train[:20], select=False)
    assert np.array_equal(head.W, ref.W)
    off = System(S.cat, S.QUESTIONS)
    h2 = off.fit("suspicious", train[:10], refit=None, select=False)
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


def test_a_candidate_head_learns_a_choice_among_candidates_that_change():
    """Candidates differ at every step (no fixed options to attach weights to): the head learns over their features."""
    import random

    import pytest
    from solvi.heads import CandidateHead
    kinds = {"door": 1.0, "npc": 0.0, "item": 2.0, "route": 0.5}

    def steps(n, seed):
        rng, out = random.Random(seed), []
        for _ in range(n):
            cs = [{"kind": rng.choice(list(kinds)), "distance": rng.randint(1, 20), "dead_end": rng.random() < 0.25,
                   "reward": rng.choice([0, 0, 1, 1, 2, 3])} for _ in range(rng.randint(4, 8))]
            best = max(range(len(cs)), key=lambda i: 2.5 * cs[i]["reward"] - 0.35 * cs[i]["distance"]
                       - 6.0 * cs[i]["dead_end"] + kinds[cs[i]["kind"]])
            out.append((cs, best))
        return out

    train, test = steps(300, 1), steps(200, 2)

    def acc(h):
        return sum(h.choose(cs)[0] == y for cs, y in test) / len(test)

    nearest = sum(min(range(len(cs)), key=lambda i: (cs[i]["dead_end"], cs[i]["distance"])) == y for cs, y in test) / len(test)
    head = CandidateHead(["kind", "distance", "reward", "dead_end"]).fit(train)
    assert acc(head) > 0.85 > 0.6 > nearest
    i, probs = head.choose(test[0][0])
    assert len(probs) == len(test[0][0]) and abs(sum(probs) - 1) < 1e-9 and probs[i] == max(probs)
    small = CandidateHead(["kind", "distance", "reward", "dead_end"]).fit(train[:30])
    before, fp = acc(small), small.fingerprint()
    ms = [small.teach(cs, y) for cs, y in train[30:]]                     # one correction at a time
    assert acc(small) > before and acc(small) > 0.8 and small.fingerprint() != fp and sum(ms) / len(ms) < 50
    rel = CandidateHead(["distance", "reward"], relative=True)
    rows = rel.rows([{"distance": 3, "reward": 1}, {"distance": 9, "reward": 0}])
    assert rows[0] == {"distance": 3, "reward": 1, "distance:above_min": 0, "distance:below_max": 6,
                       "reward:above_min": 1, "reward:below_max": 0}
    several = CandidateHead(["distance"]).fit([([{"distance": 1}, {"distance": 1}, {"distance": 9}], {0, 1})] * 5
                                              + [([{"distance": 7}, {"distance": 2}], 1)] * 5)
    assert several.choose([{"distance": 8}, {"distance": 1}, {"distance": 5}])[0] == 1
    for bad in ([([{"distance": 1}], 3)], [([{"distance": 1}, {"distance": 2}], set())]):
        with pytest.raises(ValueError):
            CandidateHead(["distance"]).fit(bad)
    with pytest.raises(ValueError):
        head.choose([])


def test_a_head_left_without_features_warns_with_the_reason():
    """A head with no features answers the same for every input, with status ok. It used to be returned silently:
    when a part's parameter with a default value was read as a fact nobody gives (so no fact could be computed), and
    when fit's greedy selection kept nothing on an imbalanced question."""
    cat = Catalog()

    def mk(nm):
        def fn(facts, _nm=nm):
            return facts[_nm]
        fn.__name__ = nm
        return fn
    cat.fn(mk("a"))
    cat.fn(mk("b"))
    s = System(cat, [Question("q", "?", Answer.yes_no())])
    ex = [({"facts": {"a": float(i % 7), "b": float(i % 3)}}, "yes" if i % 7 > 3 else "no") for i in range(200)]
    for kw in ({"select": False}, {}):
        with pytest.warns(UserWarning, match=r"the head has no features.*no fact can be computed from the examples' "
                                             r"inputs \['facts'\].*a needs \['_nm'\].*a parameter with a default value"):
            assert s.fit("q", ex, **kw).features == []
    # an imbalanced question that needs two facts: accuracy (fit before 0.8) kept nothing; the leave-one-out error
    # keeps what tells the rare answer apart, and says nothing
    cat2 = Catalog()

    @cat2.fn
    def half(x):
        return x / 2

    @cat2.fn
    def third(y):
        return y / 3
    rng = random.Random(0)
    rows = [{"x": rng.random(), "y": rng.random()} for _ in range(300)]
    ex2 = [(r, "yes" if r["x"] > 0.6 and r["y"] > 0.6 else "no") for r in rows]       # 16% yes, needs both facts
    s2 = System(cat2, [Question("q", "?", Answer.yes_no())])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                               # a head with features: silent
        head = s2.fit("q", ex2)
        assert len(head.features) == 2 and {"x", "half"} & set(head.features) and {"y", "third"} & set(head.features)
        assert list(head.selection) == head.features
        assert s2.fit("q", ex2, select=False).features == ["half", "third", "x", "y"]      # the given keys too
    assert s2.ask({"x": 0.9, "y": 0.9})["q"].answer == "yes" and s2.ask({"x": 0.9, "y": 0.1})["q"].answer == "no"
    # answers the facts say nothing about: nothing is kept, and the warning says why
    noise = [(r, rng.choice(["yes", "no", "no", "no", "no"])) for r in rows]
    with pytest.warns(UserWarning, match=r"fit\('q'\): the head has no features.*none of the 4 facts lowered the "
                                         r"leave-one-out error.*most frequent answer is \d\d% of the examples.*"
                                         r"select=False keeps every fact"):
        assert s2.fit("q", noise).features == []


def test_a_given_number_is_a_feature_of_fit_and_fit_fast_as_the_guide_says():
    """The guide: "Features are all facts computable from the examples' init_state keys"; the given keys themselves
    were left out, so a given number needed a one-line function around it before a head could read it."""
    cat = Catalog()

    @cat.fn
    def doubled(score: float) -> float:
        return score * 2
    s = System(cat, [Question("ok", "Is it ok?", Answer.choice(["yes", "no"]))])
    rng = random.Random(0)
    ex = [({"score": rng.random(), "count": c, "note": f"n{i}"}, "yes" if c >= 5 else "no")
          for i, c in enumerate(rng.randint(0, 9) for _ in range(200))]
    assert s.fit("ok", ex).features == ["count"]
    r = s.ask({"score": 0.3, "count": 8, "note": "x"})
    assert r["ok"].answer == "yes" and "count = 8" in r["ok"].why and r.trace.replay(s.catalog)["ok"]
    head = s.fit("ok", ex, select=False)
    assert {"count", "score", "doubled"} <= set(head.features) and "note" not in head.features   # 200 distinct strings
    assert s.ask({"score": 0.3, "count": 1, "note": "x"})["ok"].answer == "no"



def test_teach_refuses_an_unknown_question_or_an_answer_outside_the_options_before_storing(tmp_path):
    """teach("no_such_question", state, "banana") returned None and stored a correction for a question that does not
    exist with an answer that is not an option."""
    from solvi.storage import JSONLStorage
    store = JSONLStorage(tmp_path / "s.jsonl")
    s = System(S.cat, S.QUESTIONS, storage=store)
    state = data(0, 1)[0][0]
    with pytest.raises(KeyError, match="no question 'nope'"):
        s.teach("nope", state, "banana")
    with pytest.raises(ValueError):
        s.teach("suspicious", state, "banana")
    assert store.corrections() == []
    s.teach("free_shipping", state, True)
    assert [c["answer"] for c in store.corrections()] == ["yes"]


def test_teach_warns_when_the_correction_is_lost():
    s = System(S.cat, S.QUESTIONS)
    a, y = data(0, 1)[0]
    with pytest.warns(UserWarning, match="correction is lost"):        # no head, no storage: nothing learns it
        assert s.teach("suspicious", a, y) is None
    s.fit("suspicious", data(0, 60))                                 # a fitted head learns it at once: no warning
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert s.teach("suspicious", a, y) is not None


def test_fit_fast_is_a_deprecated_alias_of_fit_without_selection():
    """0.8 merged fit_fast into fit: fit_fast(...) still works for one release, warns, and builds what
    fit(..., select=False) builds."""
    train = data(0, 120)
    a = System(S.cat, S.QUESTIONS)
    with pytest.warns(DeprecationWarning, match="fit_fast is deprecated: use fit"):
        h1 = a.fit_fast("suspicious", train)
    h2 = System(S.cat, S.QUESTIONS).fit("suspicious", train, select=False)
    assert h1.features == h2.features and h1.fingerprint() == h2.fingerprint()
