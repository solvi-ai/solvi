"""Early exit on a failed hard check and parallel execution of independent steps: same answers, a valid trace, less work."""
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

    return cat, [Question("ok", "ok?", Answer.yes_no(), checkpoints=["allowed"])]


def test_early_exit_skips_unneeded_steps():
    cat, qs = build()
    calls.clear()
    r = System(cat, qs).ask({"x": 1, "flag": False})
    assert r["ok"].answer == "no" and r["ok"].status == "forced"
    assert calls == []                                        # slow parts never ran
    assert {n for n, _ in r.trace.skipped} >= {"slow_a", "slow_b"}
    assert r.trace.replay(cat)["ok"]


def test_parallel_same_answer_valid_trace_and_faster():
    cat, qs = build()
    seq = System(cat, qs)
    par = System(cat, qs, workers=4)
    t0 = time.perf_counter()
    r1 = seq.ask({"x": 1, "flag": True})
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    r2 = par.ask({"x": 1, "flag": True})
    t_par = time.perf_counter() - t0
    assert r1["ok"].answer == r2["ok"].answer == "yes"
    assert [r.hash for r in r1.trace.records] == [r.hash for r in r2.trace.records]   # scheduling does not change the trace
    assert r2.trace.replay(cat)["ok"]
    assert t_par < 0.7 * t_seq


def test_a_given_value_is_canonicalised_and_hashed_once_per_ask_and_the_hashes_are_the_same(monkeypatch):
    """A large input was canonicalised again for the input's hash after the steps had hashed it (and before the memo,
    once per step that read it): with a 3.7k-float input most of a decision's time was hashing."""
    import solvi.runtime as rt
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
    s = System(cat, [Question("alert", "?", Answer.yes_no(), checkpoints=["enough_history"])])
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


def test_importing_solvi_does_not_import_pydantic_and_the_first_ask_does():
    """The docs said untyped parts cost nothing and pydantic is imported only for typed parts; the first ask of any
    catalog imports it (the trace's fingerprint). What the docs say now is what this pins."""
    import subprocess
    import sys
    code = ("import sys, solvi\n"
            "a = 'pydantic' in sys.modules\n"
            "cat = solvi.Catalog()\n"
            "cat.rule('q')(lambda x: 'yes')\n"
            "solvi.System(cat, [solvi.Question('q', '', solvi.Answer.yes_no())]).ask({'x': 1})\n"
            "print(a, 'pydantic' in sys.modules)\n")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out == ["False", "True"]
