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
