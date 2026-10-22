"""solvi.experimental.stream, stage 1: StreamPromise (an anytime-valid learn-then-test certificate on a threshold grid),
its e-processes and DeltaBudget."""
import math
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import DeltaBudget, MirrorBound, Mixture, StreamPromise, clean_need, fixed_grid

from solvi import Answer, Catalog, Decision, Question, System

INF = math.inf


def _need_with_failures(proc, delta, fails):
    """Samples until a fresh process reaches 1/delta, with a loss at each of the given 0-based positions."""
    level = math.log(1.0 / delta)
    t = 0
    while proc.log_wealth < level - 1e-12:
        proc.update(1.0 if t in fails else 0.0)
        t += 1
    return t


# --- known answers: the design's numbers (needs.py, agrapa.py at α = δ = 5 %)
def test_clean_needs_match_the_design():
    assert clean_need(0.05, 0.05) == 74                       # mixture at the full δ
    assert clean_need(0.05, 0.05, "agrapa") == 93             # MeanBound mirrored
    b = DeltaBudget(0.05)
    assert [clean_need(0.05, b.delta_at(v)) for v in (1, 2, 3, 5)] == [85, 116, 134, 156]
    assert clean_need(0.02, 0.05) == 187 and clean_need(0.10, 0.05) == 36


def test_needs_with_early_failures_match_the_design():
    for fails, mix, agr in (((10,), 123, 123), ((10, 40), 171, 158), ((5, 30, 60), 216, 182)):
        assert _need_with_failures(Mixture(0.05), 0.05, set(fails)) == mix
        assert _need_with_failures(MirrorBound(0.05), 0.05, set(fails)) == agr


def test_a_fresh_promise_certifies_after_85_clean_samples_then_116_after_a_restart():
    budget = DeltaBudget(0.05)
    sp = StreamPromise(max_risk=0.05, grid=[0.0], budget=budget)
    k = 0
    while math.isinf(sp.threshold):
        sp.learn(1.0, "a", "a")
        k += 1
    assert k == 85 and sp.threshold == 0.0 and [e["event"] for e in sp.events] == ["grid", "certified"]
    assert sp.restart("alarm") == (2, budget.delta_at(2)) and math.isinf(sp.threshold) and sp.n == 0
    k = 0
    while math.isinf(sp.threshold):
        sp.learn(1.0, "a", "a")
        k += 1
    assert k == 116 and sp.events[-2]["event"] == "recertify" and sp.events[-2]["why"] == "alarm"
    assert budget.spent == pytest.approx(budget.delta_at(1) + budget.delta_at(2))
    assert budget.spent < 0.05


def test_delta_budget_is_the_skill_certificates_schedule():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.skills import Certificate
    b, c = DeltaBudget(0.1), Certificate(delta=0.1)
    assert [b.next() for _ in range(3)] == [(v, c.delta_at(v)) for v in (1, 2, 3)]
    assert b.spent == pytest.approx(sum(c.delta_at(v) for v in (1, 2, 3)))
    with pytest.raises(ValueError):
        DeltaBudget(0.1, schedule="geometric")


# --- validity: the chance of ever holding a threshold whose risk is above α stays ≤ δ_v (seeded Monte Carlo)
def _toy(kind, rng):
    """score U(0, 1). joint: P(wrong | s) = 1 − s, so R(λ) = (1 − λ)²/2; selective: P(wrong | s) = (1 − s)², so
    e(λ) = (1 − λ)²/3. Both fall with λ, so the loosest threshold ever in force is the last one."""
    s = rng.random()
    p = (1.0 - s) if kind == "joint" else (1.0 - s) ** 2
    return s, rng.random() < p


def _risk(kind, lam):
    return 0.0 if math.isinf(lam) else (1.0 - lam) ** 2 / (2.0 if kind == "joint" else 3.0)


@pytest.mark.parametrize("kind", ["joint", "selective"])
def test_a_threshold_whose_risk_is_above_alpha_is_rarely_ever_in_force(kind):
    alpha, delta, streams, n = 0.05, 0.2, 100, 600            # seen: 0.08 of the streams (both kinds), δ_1 = 0.12
    grid = [0.5 + 0.025 * i for i in range(17)]              # the boundary (0.68 joint, 0.61 selective) inside it
    rng = random.Random(7)
    bad = certified = 0
    for _ in range(streams):
        sp = StreamPromise(delta=delta, grid=grid, **({"max_risk": alpha} if kind == "joint" else {"max_error": alpha}))
        last = INF
        for _ in range(n):
            s, wrong = _toy(kind, rng)
            sp.learn(s, "x", "y" if wrong else "x")
            assert sp.threshold <= last                       # only loosens
            last = sp.threshold
        bad += _risk(kind, sp.threshold) > alpha
        certified += not math.isinf(sp.threshold)
    assert certified >= streams // 2                          # it does certify something
    assert bad / streams <= 6 * delta / math.pi ** 2 + 0.03   # δ_1 ≈ 0.12 (+ Monte Carlo slack)


def test_shuffled_labels_never_certify():
    rng = random.Random(3)
    sp = StreamPromise(max_error=0.05, delta=0.05, grid_from=200)
    for _ in range(200):
        sp.observe(rng.random())
    assert sp.grid is not None and len(sp.grid) == 32
    for _ in range(10_000):
        sp.learn(rng.random(), rng.choice("ab"), rng.choice("ab"))
    assert math.isinf(sp.threshold) and sp.state()["certified"] == 0


def test_missing_labels_never_certify_more_than_full_labels():
    rng = random.Random(11)
    grid = [0.5 + 0.05 * i for i in range(10)]
    for kind in ("joint", "selective"):
        kw = {"max_risk": 0.1} if kind == "joint" else {"max_error": 0.1}
        full, part = StreamPromise(grid=grid, **kw), StreamPromise(grid=grid, **kw)
        for _ in range(3000):
            s, wrong = _toy(kind, rng)
            label = "y" if wrong else "x"
            full.learn(s, "x", label)
            part.learn(s, "x", None if rng.random() < 0.2 else label)
            assert part.threshold >= full.threshold
        assert part.missing > 0 and not math.isinf(full.threshold)


# --- the grid
def test_the_auto_grid_is_fixed_from_unlabelled_scores_and_earlier_samples_are_not_read():
    sp = StreamPromise(max_risk=0.05, grid_from=50)
    sp.learn(0.9, "a", "a")
    assert sp.dropped == 1 and sp.grid is None and sp.n == 0
    for i in range(50):
        sp.observe(i / 50)
    assert sp.grid == sorted(sp.grid, reverse=True) and len(sp.grid) == 32 and sp.events[0]["event"] == "grid"
    assert sp.state()["grid_fp"] is not None


def test_fixed_grid_is_kadis_shape():
    g = fixed_grid()
    assert g == sorted(g, reverse=True) and 0.05 in g and 0.95 in g and g[0] == pytest.approx(1 - 1e-4)
    assert fixed_grid(2.0, 4.0)[-1] == pytest.approx(2.1) and fixed_grid(2.0, 4.0)[0] == pytest.approx(4.0 - 2e-4)
    assert StreamPromise(max_error=0.1, grid="fixed").grid == g


def test_arguments_are_checked():
    with pytest.raises(ValueError):
        StreamPromise()
    with pytest.raises(ValueError):
        StreamPromise(max_risk=0.1, max_error=0.1)
    with pytest.raises(ValueError):
        StreamPromise(max_risk=0.1, kind="selective")
    with pytest.raises(ValueError):
        StreamPromise(max_risk=0.1, bets="kelly")
    with pytest.raises(ValueError):
        StreamPromise(max_risk=0.1, delta=0.1, budget=DeltaBudget(0.05))
    with pytest.raises(ValueError):
        StreamPromise(max_risk=0.1, grid=[0.5]).threshold_of(("a",))


# --- state, pickle, the Promise interface
def test_the_same_samples_give_the_same_state_and_a_pickle_continues_identically():
    rng = random.Random(5)
    rows = [(rng.random(), rng.random() < 0.1) for _ in range(400)]
    a = StreamPromise(max_error=0.1, grid_from=100, bets="agrapa")
    b = StreamPromise(max_error=0.1, grid_from=100, bets="agrapa")
    for s, _ in rows[:100]:
        a.observe(s)
        b.observe(s)
    for s, w in rows[100:250]:
        a.learn(s, "x", "y" if w else "x")
        b.learn(s, "x", "y" if w else "x")
    assert a.state() == b.state()
    c = pickle.loads(pickle.dumps(a))
    for s, w in rows[250:]:
        assert a.learn(s, "x", "y" if w else "x") == c.learn(s, "x", "y" if w else "x")
    assert a.fingerprint() == c.fingerprint() == b.fingerprint()


def test_it_has_the_promise_interface():
    sp = StreamPromise(max_error=0.05, grid=[0.9, 0.5])
    assert sp.stateful and sp.method == "stream-ltt" and sp.nodes is None and sp.level == 0.05 and sp.delta == 0.05
    assert sp.threshold_of() == (INF, None, None) and not sp.allows(0.99) and not sp.allows(None)
    rec = sp.record()
    assert rec["error"] == 0.05 and rec["method"] == "stream-ltt" and "error among the answers" in rec["promise"]
    assert sp.report["threshold"] is None and isinstance(sp.fingerprint(), str)
    fp = sp.fingerprint()
    for _ in range(100):
        sp.learn(1.0, "a", "a")
    assert sp.threshold == 0.5 and sp.allows(0.5) and not sp.allows(0.49)
    assert sp.fingerprint() == fp                             # the configuration, not the state


# --- installed on a System: recorded, replayed, rebuilt
def _system():
    cat = Catalog()

    @cat.rule("intent")
    def intent(p: float):
        return Decision("billing" if p > 0.5 else "cards", {"billing": p, "cards": 1 - p}, confidence=max(p, 1 - p))
    return System(cat, [Question("intent", "which?", Answer.choice(["billing", "cards"]))])


def test_installed_on_a_system_it_records_its_state_and_a_replay_rebuilds_it():
    s = _system()
    sp = StreamPromise(max_risk=0.1, grid_from=30)
    s.guarantee("intent", promise=sp)
    rng = random.Random(9)
    log, results = [], []                                     # (kind, args) in the order they happened
    for t in range(140):
        p = rng.uniform(0.5, 1.0)
        r = s.ask({"p": p})
        results.append(r)
        log.append(("observe", p))
        if t >= 30:                                           # every later decision labelled, all right
            sp.learn(p, "billing", "billing")
            log.append(("learn", p))
    assert sp.seen == 140 and not math.isinf(sp.threshold)
    first = next(x for x in results[0].trace.records if x.name == "guard:intent")
    last = next(x for x in results[-1].trace.records if x.name == "guard:intent")
    assert first.extra["method"] == "stream-ltt" and first.value is False and first.extra["state"]["threshold"] is None
    assert last.value is True and results[-1]["intent"].status == "ok"
    # replay re-derives each verdict from the recorded threshold and does not move the promise
    assert all(r.trace.replay(s)["ok"] for r in results[::10])
    assert sp.seen == 140
    # a fresh promise fed the same decisions and samples rebuilds the recorded state at every decision
    fresh = StreamPromise(max_risk=0.1, grid_from=30)
    recorded = iter(next(x for x in r.trace.records if x.name == "guard:intent").extra["state"] for r in results)
    for kind, p in log:
        if kind == "observe":
            assert fresh.state() == next(recorded)
            fresh.observe(p)
        else:
            fresh.learn(p, "billing", "billing")
    assert fresh.state() == sp.state()
