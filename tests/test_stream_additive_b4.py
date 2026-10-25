"""solvi.experimental.stream, additive coverage, stage B4 (the cheap valid fixes): the lazy δ charge (a StreamPromise
takes its (v, δ_v) at its first bet, not when built or restarted; a certificate that never bets spends nothing; no
threshold — so no alone answer — without an index) in both modes, and DefaultPolicy's maturity gate for the retrain
cadence (mature_patience, the alarm and backlog overrides; on by default in additive mode only). Spec tests 9 (lazy
part) and 10 (maturity gate)."""
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import (AdapterLoop, Budget, DefaultPolicy, DeltaBudget, LoopState, StreamPromise,
                                           VersionInfo)

KEY = b"additive-b4-test-key"


# ───────────────────────────── the lazy δ charge on a StreamPromise
def test_a_promise_takes_its_index_at_its_first_bet_not_when_built():
    budget = DeltaBudget(0.05)
    sp = StreamPromise(max_risk=0.05, grid_from=20, grid_size=8, budget=budget)
    assert sp.v is None and sp.delta_v is None and budget.v == 0 and math.isinf(sp.threshold)
    for i in range(10):                                    # before the grid: observed, a label is dropped, no bet
        sp.observe(i / 10)
        sp.learn(i / 10, "a", "a")
    assert sp.v is None and budget.v == 0 and sp.dropped == 10 and sp.state()["v_index"] is None
    for i in range(10):
        sp.observe(0.5 + i / 30)
    assert sp.grid is not None and sp.v is None            # the grid alone spends nothing
    sp.learn(0.9, "a", "a")                                # the first bet: the index first
    assert (sp.v, sp.delta_v) == (1, budget.delta_at(1)) and budget.v == 1 and sp.n == 1
    sp.learn(0.9, "a", "a")
    assert sp.v == 1 and budget.v == 1                     # once per certification


def test_the_index_order_is_the_order_of_first_bets_and_a_restart_that_never_bets_spends_nothing():
    budget = DeltaBudget(0.05)
    a = StreamPromise(max_risk=0.05, grid=[0.0], budget=budget)
    b = StreamPromise(max_risk=0.05, grid=[0.0], budget=budget)
    b.learn(1.0, "a", "a")
    a.learn(1.0, "a", "a")
    assert (b.v, a.v) == (1, 2)                            # issue order, not construction order
    assert a.restart("alarm") == (None, None) and a.v is None and math.isinf(a.threshold)
    a.restart("again")                                     # re-alarmed before its first post-alarm sample
    assert budget.v == 2
    a.learn(1.0, "a", "a")
    assert a.v == 3 and budget.v == 3 and budget.spent == pytest.approx(sum(budget.delta_at(v) for v in (1, 2, 3)))
    assert [e["event"] for e in a.events] == ["grid", "recertify", "recertify"]
    assert [e["v_index"] for e in a.events] == [None, None, None]


def test_certification_still_takes_85_then_116_clean_samples():
    budget = DeltaBudget(0.05)
    sp = StreamPromise(max_risk=0.05, grid=[0.0], budget=budget)
    k = 0
    while math.isinf(sp.threshold):
        sp.learn(1.0, "a", "a")
        k += 1
    assert k == 85 and sp.v == 1
    sp.restart("alarm")
    k = 0
    while math.isinf(sp.threshold):
        sp.learn(1.0, "a", "a")
        k += 1
    assert k == 116 and sp.v == 2


def test_no_finite_threshold_without_an_index():
    sp = StreamPromise(max_risk=0.05, grid=[0.0], budget=DeltaBudget(0.05))
    assert sp.threshold_of() == (math.inf, None, None) and not sp.allows(1.0)
    sp.threshold = 0.0                                     # forced: the invariant is checked where it is read
    with pytest.raises(AssertionError, match="without a δ index"):
        sp.threshold_of()
    with pytest.raises(AssertionError, match="without a δ index"):
        sp.allows(1.0)


def test_an_uncharged_dump_is_format_v2_and_restores_a_charged_one_stays_v1():
    budget = DeltaBudget(0.05)
    sp = StreamPromise(max_risk=0.05, grid=[0.5, 0.0], budget=budget)
    d = sp.dump()
    assert d["format"] == "solvi stream promise v2" and d["v"] is None
    back = StreamPromise(max_risk=0.05, grid=[0.5, 0.0], budget=budget).restore(d)
    assert back.v is None and back.state() == sp.state()
    bad = dict(d, threshold=0.5)
    with pytest.raises(ValueError, match="no δ index"):
        StreamPromise(max_risk=0.05, grid=[0.5, 0.0], budget=budget).restore(bad)
    for _ in range(30):
        sp.learn(0.7, "a", "a")
    d = sp.dump()
    assert d["format"] == "solvi stream promise v1" and d["v"] == 1
    back = StreamPromise(max_risk=0.05, grid=[0.5, 0.0], budget=budget).restore(d)
    assert back.state() == sp.state() and back.peak_at() == sp.peak_at() == 30
    old = {k: v for k, v in d.items() if k != "peaks"}     # a dump from before the peaks: set "now" (conservative)
    assert StreamPromise(max_risk=0.05, grid=[0.5, 0.0], budget=budget).restore(old).peak_at() == sp.n


def test_peak_at_follows_the_first_uncertified_point():
    sp = StreamPromise(max_risk=0.05, grid=[0.5, 0.0], budget=DeltaBudget(0.05))
    assert sp.peak_at() == 0
    for _ in range(20):
        sp.learn(0.3, "a", "a")                            # both points gain (joint: no loss at either)
    assert sp.peak_at() == 20
    for _ in range(5):
        sp.learn(0.3, "a", "b")                            # wrong, alone at 0.0 only: point 0.5 still gains
    assert sp.peak_at() == 25 and sp._peak[1][1] == 20     # the first open point (0.5) climbs; 0.0 fell
    while math.isinf(sp.threshold):
        sp.learn(0.3, "a", "a")
    assert sp.threshold == 0.5                             # point 0.5 certified: point 0.0 is the first open one
    assert sp.peak_at() == sp._peak[1][1] <= sp.n
    sp2 = StreamPromise(max_risk=0.05, grid=[0.0], budget=DeltaBudget(0.05))
    while math.isinf(sp2.threshold):
        sp2.learn(1.0, "a", "a")
    assert sp2.peak_at() is None                           # every point certified


# ───────────────────────────── the lazy charge in the loop (both modes) and the δ audit from the records
def _f_l0(x):
    return (x[0] if x[1] > 0.3 else "wrong", x[1])


def _f_good(x):
    return (x[0], x[2])


class _M:
    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, x):
        return self.f(x)

    def fingerprint(self):
        return f"m-{self.name}"


L0, GOOD = _M("l0", _f_l0), _M("good", _f_good)


class _Hand:
    def __init__(self, models):
        self.models, self._seen = list(models), 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        return self.models.pop(0) if len(self.models) > 1 else self.models[0]


def _policy(**kw):
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0, pre_freeze=False,
                         probe=False, **kw)


def _loop(models, *, additive, policy=None, **kw):
    return AdapterLoop(_Hand(models), question="q", policy=_policy() if policy is None else policy, max_risk=0.05,
                       secret=KEY, initial=L0, additive=additive, monitor=None, _detector_alone=True, wait=True,
                       clock=lambda: 0.0, **kw)


def _xs(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random(), r.random(), r.random()) for _ in range(n)]


def _run(loop, xs):
    for x in xs:
        loop.step(x, lambda x=x: x[0])


def _items(loop, kind):
    return [d for k, d in loop.log if k == kind]


def _events(loop, *kinds):
    return [e for e in _items(loop, "stream") if not kinds or e["event"] in kinds]


def _audit(loop):
    """The δ audit from the records alone: every index is handed out once, in order, Σ δ_v ≤ δ, an index appears on a
    version's records only after it had none (lazy) or at its freeze, and no alone answer of the served version (the
    owner in additive mode: its certification event) is given without an index."""
    seen = {}
    for e in _events(loop):
        if e.get("v_index") is not None:
            seen.setdefault(e["v_index"], e["delta_v"])
    for d in _items(loop, "loop"):
        for f in ("served_state", "candidate_state"):
            s = d["stream"][f]
            if s is not None and s["v_index"] is not None:
                seen.setdefault(s["v_index"], s["delta_v"])
    idx = sorted(seen)
    assert idx == list(range(1, len(idx) + 1))             # chronological, unique, no gap
    assert math.fsum(seen.values()) <= 0.05 + 1e-12
    assert idx[-1] == loop.budget.v if idx else loop.budget.v == 0
    for d in _items(loop, "loop"):
        r = d["stream"]
        if r["alone"] and "owner" not in r:                # replacement: the served version answered alone
            assert r["served_state"]["v_index"] is not None and r["threshold"] is not None
    for e in _events(loop, "certified", "loosened"):       # every certificate in force had its index first
        assert e["v_index"] is not None and e["delta_v"] is not None
    return idx


@pytest.mark.parametrize("additive", [False, True])
def test_a_candidate_discarded_before_its_first_sample_spends_nothing(additive):
    # stale=100 decisions < grid_from=200: the candidate's grid is never fixed, it never bets
    loop = _loop([GOOD, GOOD], additive=additive, policy=_policy(stale=100, mature_patience=None))
    _run(loop, _xs(0, 3000))
    fr, disc = _events(loop, "freeze"), _events(loop, "discard")
    assert fr and disc and disc[0]["version"] == fr[0]["version"]
    first = fr[0]["version"]
    assert all(r["candidate_state"]["v_index"] is None for d in _items(loop, "loop")
               for r in [d["stream"]] if r["candidate"] == first)
    assert loop._vs[first]["promise"] is None or loop._vs[first]["promise"].v is None
    idx = _audit(loop)
    assert loop.budget.v == len(idx) and loop.budget.v == 1  # version 0's only: every stale candidate spent nothing
    assert len(disc) >= 2


@pytest.mark.parametrize("additive", [False, True])
def test_the_delta_audit_holds_on_a_run_with_promotions(additive):
    loop = _loop([GOOD, GOOD, GOOD], additive=additive, policy=_policy(mature_patience=None))
    _run(loop, _xs(1, 6000))
    assert _events(loop, "promote")
    idx = _audit(loop)
    assert len(idx) >= 2


def test_a_restart_rolled_away_before_its_first_sample_spends_nothing_in_the_loop():
    loop = _loop([GOOD], additive=False, policy=_policy(mature_patience=None))
    _run(loop, _xs(2, 1500))
    v0 = loop.budget.v
    loop._restart(loop.served, loop.n, "test")             # a restart: no index yet
    loop._restart(loop.served, loop.n, "test again")
    assert loop.budget.v == v0 and loop._vs[loop.served]["promise"].v is None
    assert math.isinf(loop._threshold(loop.served))         # nothing alone under it until it re-certifies


# ───────────────────────────── the maturity gate (DefaultPolicy) on hand-built states
def _st(*, additive=True, samples=500, changed=None, peak=None, alarm=False, backlog=0, **kw):
    m = {"version": 0, "samples": samples, "changed_at": changed, "peak_at": peak, "last_freeze": 0,
         "labels_since_freeze": backlog, "alarm": alarm}
    extra = {"layers": (0,), "deepest": 0, "deepest_certified": True, "zone_share": {0: 1.0}} if additive else {}
    return LoopState(n=5000, labels={"train": 10 ** 4, "calib": 0, "holdout": 0},
                     versions=(VersionInfo(0, "served", certified=True, threshold=0.3),), maturity=m, **extra, **kw)


def test_the_gate_blocks_while_loosening_or_climbing_and_opens_when_mature():
    p = _policy()
    g = p.maturity_gate(_st(changed=400))
    assert g["loosening"] and g["trigger"] is None and g["patience"] == 150
    assert p.why_not_train(_st(changed=400)).startswith("maturity gate")
    g = p.maturity_gate(_st(changed=100, peak=420))
    assert not g["loosening"] and g["climbing"] and g["trigger"] is None
    g = p.maturity_gate(_st(changed=349, peak=300))
    assert g["trigger"] == "mature" and not (p.why_not_train(_st(changed=349, peak=300)) or "").startswith("maturity")


def test_the_alarm_and_backlog_overrides_open_the_gate():
    p = _policy()
    assert p.maturity_gate(_st(changed=450, alarm=True))["trigger"] == "alarm"
    assert p.maturity_gate(_st(changed=450, backlog=599))["trigger"] is None
    assert p.maturity_gate(_st(changed=450, backlog=600))["trigger"] == "backlog"      # 2 × first
    assert _policy(backlog_max=50).maturity_gate(_st(changed=450, backlog=50))["trigger"] == "backlog"
    with pytest.raises(ValueError, match="backlog_max"):
        _policy(backlog_max=0)


def test_the_gate_is_on_in_additive_mode_only_by_default():
    p = _policy()
    assert p.maturity_gate(_st(additive=False, changed=450)) is None
    assert p.maturity_gate(_st(additive=True, changed=450)) is not None
    assert _policy(mature_patience=None).maturity_gate(_st(changed=450)) is None
    g = _policy(mature_patience=10).maturity_gate(_st(additive=False, changed=450))
    assert g["patience"] == 10 and g["trigger"] == "mature"
    assert _policy().maturity_gate(_st(changed=450, mode="label")) is None
    with pytest.raises(ValueError, match="mature_patience"):
        _policy(mature_patience=0)


def test_the_loop_feeds_the_gate_and_records_it_in_the_train_events():
    loop = _loop([GOOD, GOOD], additive=True, max_layers=2)
    _run(loop, _xs(3, 4000))
    starts = [e for e in _events(loop, "train_start") if e["trigger"] == "schedule"]
    assert starts and all(e["gate"]["trigger"] in ("mature", "alarm", "backlog") for e in starts)
    trains = [e for e in _events(loop, "train") if e["trigger"] == "schedule"]
    assert trains and trains[0]["gate"] == starts[0]["gate"]
    m = loop._state().maturity
    assert m["version"] == loop._layers[-1] and m["samples"] == loop._vs[m["version"]]["promise"].n
    # replacement mode by default: no gate, no field
    rep = _loop([GOOD, GOOD], additive=False)
    _run(rep, _xs(3, 2000))
    assert _events(rep, "train_start") and all("gate" not in e for e in _events(rep, "train_start", "train"))
    assert rep._state().maturity["version"] == rep.served


def test_the_gate_delays_the_first_attach_until_the_base_is_mature():
    xs = _xs(4, 6000)
    on = _loop([GOOD], additive=True, max_layers=2)
    off = _loop([GOOD], additive=True, max_layers=2, policy=_policy(mature_patience=None))
    _run(on, xs)
    _run(off, xs)
    a_on, a_off = _events(on, "attach"), _events(off, "attach")
    assert a_off and a_on and a_on[0]["decision"] >= a_off[0]["decision"]
    g = [e for e in _events(on, "train_start") if e["trigger"] == "schedule"][0]["gate"]
    assert g["samples"] - max(g["changed_at"] or 0, g["peak_at"] or 0) >= 150 or g["trigger"] != "mature"
