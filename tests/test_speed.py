"""Early exit on a failed hard check and parallel execution of independent steps: same answers, a valid trace, less work."""
import copy
import time

from solvi import Answer, Catalog, Question, System

calls = []


def build():
    cat = Catalog()

    @cat.fn
    def slow_a(x):
        calls.append("slow_a")
        time.sleep(0.2)
        return x + 1

    @cat.fn
    def slow_b(x):
        calls.append("slow_b")
        time.sleep(0.2)
        return x * 2

    @cat.check(hard=True, then={"ok": "no"})
    def allowed(flag):
        return flag

    @cat.rule("ok")
    def ok(slow_a, slow_b):
        return slow_a + slow_b > 3

    return cat, [Question("ok", "ok?", Answer.yes_no(), requires=["allowed"])]


def test_early_exit_skips_unneeded_steps():
    cat, qs = build()
    calls.clear()
    r = System(cat, qs).ask({"x": 1, "flag": False})
    assert r["ok"].answer == "no" and r["ok"].status == "forced"
    assert calls == []                                        # slow parts never ran
    assert {n for n, _ in r.trace.skipped} >= {"slow_a", "slow_b"}
    assert r.trace.replay(cat)["ok"]


def test_parallel_same_answer_valid_trace_and_the_independent_steps_overlap():
    cat, qs = build()
    seq = System(cat, qs)
    par = System(cat, qs, workers=4)
    r1 = seq.ask({"x": 1, "flag": True})
    r2 = par.ask({"x": 1, "flag": True})
    assert r1["ok"].answer == r2["ok"].answer == "yes"
    assert [r.hash for r in r1.trace.records] == [r.hash for r in r2.trace.records]   # scheduling does not change the trace
    assert r2.trace.replay(cat)["ok"]
    # the two slow steps ran at the same time (a timing ratio of two single runs is noise on a loaded machine)
    spans = {}
    cat2 = Catalog()

    def timed(name):
        def f(x):
            t0 = time.perf_counter()
            time.sleep(0.1)
            spans[name] = (t0, time.perf_counter())
            return x
        f.__name__ = name
        return f
    cat2.fn(timed("a"))
    cat2.fn(timed("b"))

    @cat2.rule("ok")
    def ok2(a, b):
        return a == b
    System(cat2, [Question("ok", "ok?", Answer.yes_no())], workers=2).ask({"x": 1})
    (a0, a1), (b0, b1) = spans["a"], spans["b"]
    assert a0 < b1 and b0 < a1
    System(cat2, [Question("ok", "ok?", Answer.yes_no())]).ask({"x": 1})
    (a0, a1), (b0, b1) = spans["a"], spans["b"]
    assert a1 <= b0 or b1 <= a0                                    # one worker: one after the other


def test_a_given_value_is_canonicalised_and_hashed_once_per_ask_and_the_hashes_are_the_same(monkeypatch):
    """A large input was canonicalised again for the input's hash after the steps had hashed it (and before the memo,
    once per step that read it): with a 3.7k-float input most of a decision's time was hashing."""
    import solvi.core.runtime as rt
    from solvi import Answer, Catalog, Question
    cat = Catalog()

    @cat.fn
    def mean(series):
        return sum(series) / len(series)

    @cat.fn
    def last(series):
        return series[-1]

    @cat.check(hard=True, then={"alert": "no"})
    def enough_history(series):
        return len(series) >= 3

    @cat.fn
    def window(series, n):
        return series[-n:]

    @cat.rule("alert")
    def alert(last, mean, window, label):
        return "yes" if last > mean and len(window) == 2 else "no"
    s = System(cat, [Question("alert", "?", Answer.yes_no(), requires=["enough_history"])])
    state = {"series": [0.5, 1.25, 2.0, 7.5], "n": 2, "label": "x" * 300, 5: "a key that is not a string"}
    seen = []
    canon = rt._canon
    monkeypatch.setattr(rt, "_canon", lambda v: (seen.append(id(v)), canon(v))[1])
    res = s.ask(state)
    for v in (state["series"], state["label"], res.values["window"]):           # read by 4 steps, by 1, made by a step
        assert seen.count(id(v)) == 1, v
    monkeypatch.setattr(rt, "_canon", canon)
    tr = res.trace
    assert tr.init_hash == rt.vhash(state) and tr.replay(s, res.flow)["ok"]            # byte for byte what vhash gives
    prev = tr.init_hash
    for r in tr.records:
        assert r.inputs == {x: rt.vhash(res.values[x]) for x in r.inputs}
        assert r.prev == prev and r.hash == rt.vhash(r.body())
        prev = r.hash
    assert res["alert"].answer == "yes"


def test_importing_solvi_and_the_first_ask_of_an_untyped_catalog_do_not_import_pydantic():
    """The docs say untyped parts cost nothing and pydantic is imported only for typed parts; the first ask of any
    catalog used to import it (the trace's fingerprint of the questions). What the docs say is what this pins."""
    import subprocess
    import sys
    code = ("import sys, solvi\n"
            "a = 'pydantic' in sys.modules\n"
            "cat = solvi.Catalog()\n"
            "cat.rule('q')(lambda x: 'yes')\n"
            "solvi.System(cat, [solvi.Question('q', '', solvi.Answer.yes_no())]).ask({'x': 1})\n"
            "print(a, 'pydantic' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "False"]


def _reference_vhash(v):
    """vhash as 0.7.1 computed it, kept here as the reference the faster code must match byte for byte."""
    import hashlib
    import json

    import numpy as np

    from solvi.core import Decision, Quote, Unknown
    from solvi.core.runtime import MISSING

    def canon(v):
        if isinstance(v, Quote):
            return {"quote": [canon(v.value), v.start, v.end, v.source]}
        if isinstance(v, Decision):
            return {"decision": [canon(v.value), canon(v.probs)]}
        if isinstance(v, (list, tuple)):
            return [canon(x) for x in v]
        if isinstance(v, (set, frozenset)):
            return sorted((canon(x) for x in v), key=lambda c: json.dumps(c, ensure_ascii=False, sort_keys=True))
        if isinstance(v, dict):
            return {str(k): canon(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
        if isinstance(v, float):
            return round(v, 9)
        if isinstance(v, (str, int, bool)) or v is None:
            return v
        if v is Unknown:
            return {"not_stated": True}
        if v is MISSING:
            return {"missing": True}
        if isinstance(v, (np.ndarray, np.generic)):
            return canon(v.tolist())
        return repr(v)
    return hashlib.sha256(json.dumps(canon(v), ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:16]


def test_vhash_and_the_input_hash_match_the_plain_canonical_json_on_random_values():
    """The hash of a value is taken over plain canonical JSON; the fast paths (lists of floats, string-keyed dicts, a
    scalar's JSON written directly, the input's JSON put together from its values' texts) must give exactly those bytes:
    stored traces replay only if every hash is what it always was."""
    import enum
    import math
    import random
    from datetime import date
    from decimal import Decimal

    import pytest
    np = pytest.importorskip("numpy")

    import solvi.core.runtime as rt
    from solvi.core import Decision, Quote, Unknown

    class Level(enum.IntEnum):
        LOW = 1

    rng = random.Random(7)
    atoms = [lambda: rng.uniform(-1e6, 1e6), lambda: rng.random() * 10 ** rng.randint(-12, 20), lambda: 0.1 + 0.2,
             lambda: rng.choice([math.inf, -math.inf, math.nan, -0.0, 1e-10, 5e-10, 1.0000000005]),
             lambda: rng.randint(-10**20, 10**20), lambda: rng.choice([True, False, None]), lambda: Level.LOW,
             lambda: rng.choice(["", "é ü 中文  ", 'q"\\\n\t', "plain"]), lambda: date(2026, 1, rng.randint(1, 28)),
             lambda: Decimal("1.10"), lambda: Unknown, lambda: np.float64(rng.random()), lambda: np.int64(5),
             lambda: Quote("x", 0, 1, "doc"), lambda: Decision("a", {"a": 0.25, "b": 0.75})]

    def value(depth=0):
        k = rng.random()
        if depth > 3 or k < 0.4:
            return rng.choice(atoms)()
        if k < 0.55:
            return [rng.uniform(-5, 5) for _ in range(rng.randint(0, 40))]          # a series
        if k < 0.6:
            return [rng.choice([None, rng.random()]) for _ in range(rng.randint(0, 20))]
        if k < 0.7:
            return tuple(value(depth + 1) for _ in range(rng.randint(0, 4)))
        if k < 0.8:
            return {rng.choice(["a", "b", "ключ", 3, 3.0, "3", (1, 2)]): value(depth + 1) for _ in range(rng.randint(0, 5))}
        if k < 0.85:
            return {rng.randint(0, 9) for _ in range(rng.randint(0, 5))}
        if k < 0.9:
            return np.array([rng.random() for _ in range(rng.randint(0, 6))])
        return [value(depth + 1) for _ in range(rng.randint(0, 5))]
    for _ in range(600):
        v = value()
        assert rt.vhash(v) == _reference_vhash(v), v
        state = {rng.choice(["x", "y", "z", "ä", 1, "1"]): value() for _ in range(rng.randint(0, 5))}
        assert rt.HashMemo(state).init_hash() == _reference_vhash(state) == rt.vhash(state), state


def test_a_hash_seed_lends_its_hashes_and_the_trace_is_byte_for_byte_the_same():
    """solvi.core.slow.search asks the same given text and held facts with every candidate: a HashSeed on the catalog hashes them
    once. The values it holds are not canonicalised again, and every hash of the trace is what an unseeded ask gives."""
    import solvi.core.runtime as rt
    cat = Catalog()

    @cat.fn
    def words(problem):
        return problem.split()

    @cat.rule("long")
    def long(words, n):
        return "yes" if len(words) > n else "no"
    s = System(cat, [Question("long", "?", Answer.yes_no())])
    problem = "a fairly long text " * 200
    plain = s.ask({"problem": problem, "n": 3})
    seeded_cat = copy.copy(cat)
    seeded_cat._hash_seed = rt.HashSeed().add(problem)
    seeded = System(seeded_cat, [Question("long", "?", Answer.yes_no())])
    seen = []
    canon = rt._canon
    rt._canon = lambda v: (seen.append(v), canon(v))[1]
    try:
        res = seeded.ask({"problem": problem, "n": 3})
    finally:
        rt._canon = canon
    assert not any(v is problem for v in seen)
    assert res.trace.init_hash == plain.trace.init_hash
    assert [(r.hash, r.inputs) for r in res.trace.records] == [(r.hash, r.inputs) for r in plain.trace.records]
