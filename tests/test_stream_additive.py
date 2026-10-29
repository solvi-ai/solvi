"""solvi.experimental.stream, additive coverage, stage B1 (the routing core): AdapterLoop(additive=True) attaches a
candidate below the served chain — zones, owner routing, the top-down answer at the live thresholds (decision 21),
every layer learning on its zone, attach / unfreeze — and additive=False stays the replacement protocol, byte for
byte."""
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy

KEY = b"additive-test-key"


class _M:
    """A frozen model: x → f(x) = (would, score)."""

    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, x):
        return self.f(x)

    def fingerprint(self):
        return f"m-{self.name}"


# x = (label, u, w): L0 is right iff u > 0.3 with score u (it certifies near 0.3, so its zone below is u < ~0.3);
# GOOD is always right with score w (it certifies on the zone); BAD is always wrong with a high score (never certifies)
L0 = _M("l0", lambda x: (x[0] if x[1] > 0.3 else "wrong", x[1]))
GOOD = _M("good", lambda x: (x[0], x[2]))
BAD = _M("bad", lambda x: ("wrong", 0.99))


class _Hand:
    """A batch learner handing out the given models in turn (the last one again when they run out)."""

    def __init__(self, models):
        self.models, self._seen = list(models), 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        return self.models.pop(0) if len(self.models) > 1 else self.models[0]


def _policy(**kw):
    # the routing / layering mechanics run without the maturity gate (stage B4, on by default in additive mode):
    # their training timing stays the one they were written for; the gate has its own tests (test_stream_additive_b4)
    kw.setdefault("mature_patience", None)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0, pre_freeze=False,
                         probe=False, **kw)


def _loop(models, *, additive=True, policy=None, **kw):
    return AdapterLoop(_Hand(models), question="q", policy=_policy() if policy is None else policy, max_risk=0.05,
                       secret=KEY, initial=L0, additive=additive, monitor=None, _detector_alone=True, wait=True,
                       clock=lambda: 0.0, **kw)


def _xs(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random(), r.random()) for _ in range(n)]


def _run(loop, xs, oracle=None):
    return [loop.step(x, (lambda x=x: x[0]) if oracle is None else (lambda x=x: oracle(x))) for x in xs]


def _recs(loop):
    return [d["stream"] for k, d in loop.log if k == "loop"]


def _events(loop, *kinds):
    return [e for k, e in loop.log if k == "stream" and (not kinds or e["event"] in kinds)]


# ───────────────────────────── test 11: the opt-in, refusals, replacement mode unchanged
def test_additive_is_refused_in_label_mode_without_an_initial_model_and_below_two_layers():
    with pytest.raises(ValueError, match="stream mode"):
        AdapterLoop(_Hand([GOOD]), question="q", max_risk=0.05, secret=KEY, initial=L0, mode="label",
                    additive=True, policy=DefaultPolicy())
    with pytest.raises(ValueError, match="initial model"):
        AdapterLoop(_Hand([GOOD]), question="q", max_risk=0.05, secret=KEY, additive=True)
    for bad in (1, 0, True):
        with pytest.raises(ValueError, match="max_layers"):
            AdapterLoop(_Hand([GOOD]), question="q", max_risk=0.05, secret=KEY, initial=L0, additive=True,
                        max_layers=bad)


def test_additive_is_the_default_and_a_loop_without_initial_must_say_so():
    loop = AdapterLoop(_Hand([GOOD]), question="q", max_risk=0.05, secret=KEY, initial=L0, monitor=None,
                       _detector_alone=True, wait=True)
    assert loop.additive is True
    with pytest.raises(ValueError, match="initial"):
        AdapterLoop(_Hand([GOOD]), question="q", max_risk=0.05, secret=KEY, monitor=None, _detector_alone=True,
                    wait=True)


def test_additive_false_writes_the_same_records_and_events_as_explicit_false():
    xs = _xs(3, 2500)
    runs = []
    for kw in ({"additive": False}, {"additive": False, "max_layers": 5}):
        loop = AdapterLoop(_Hand([GOOD, GOOD]), question="q", policy=_policy(), max_risk=0.05, secret=KEY,
                           initial=L0, monitor=None, _detector_alone=True, wait=True, clock=lambda: 0.0, **kw)
        _run(loop, xs)
        runs.append(loop)
    assert _events(runs[0], "promote")                     # the replacement protocol ran a promotion
    def timed(e):                                          # a training's measured seconds are wall time
        if isinstance(e, dict):
            return {f: timed(v) for f, v in e.items() if f != "seconds"}
        if isinstance(e, (list, tuple)):
            return [timed(v) for v in e]
        return e
    for other in runs[1:]:
        assert timed(other.log) == timed(runs[0].log)
    st = runs[0]._state()
    assert st.layers is None and st.deepest is None and st.versions[0].layer is None
    assert all(k not in r for r in _recs(runs[0]) for k in ("owner", "path", "trial_zone"))


# ───────────────────────────── attach: the deepest layer certified, Λ frozen, the zone escalated
def _attached(n=2500, models=(GOOD,), **kw):
    kw.setdefault("policy", _policy(min_novel=10 ** 9))   # the gate never passes: the candidate stays in trial (B2
    loop = _loop(list(models), **kw)                       # layers a certified candidate by itself)
    _run(loop, _xs(0, n))
    return loop


def test_attach_sets_the_routing_threshold_drops_the_comparison_and_the_deepest_keeps_learning():
    loop = _attached()
    at = _events(loop, "attach")
    assert len(at) == 1 and at[0]["version"] == 1 and at[0]["zone_of"] == 0
    lam = at[0]["routing_thr"]
    assert lam == loop._route[0] and loop._threshold(0) <= lam and 0.2 < lam < 0.35   # Λ_0 = λ_0 at the attach
    assert loop._vs[1]["cmp"] is None and loop.candidate == 1
    st = loop._state()
    assert st.layers == (0,) and st.deepest == 0 and st.deepest_certified
    assert st.version(0).routing_thr == lam and st.version(0).learning is True and st.version(1).learning is True
    assert not _events(loop, "promote")                    # stage B1 never layers an additive candidate by itself
    recs = [r for r in _recs(loop) if r["n"] > at[0]["decision"]]
    assert all(r["path"][0][2] == lam for r in recs)       # Λ_0 constant: the routing (and the trial zone) fixed
    ns = [r["served_state"]["n"] for r in recs]
    assert ns == sorted(ns) and ns[-1] > ns[0]              # L0's promise keeps reading samples
    assert all(r["threshold"] is None or r["threshold"] <= lam for r in recs)   # λ_0 may loosen, never tighten


def test_attach_is_refused_below_an_uncertified_deepest_layer():
    # training at 50 labels: L0's grid needs 200 decisions, so it is not certified at the freeze
    # stage B2: the schedule starts no training below an uncertified deepest layer ("train_refused", once)
    loop = _attached(n=150, policy=_policy(first=50))
    tr = _events(loop, "train_refused")
    assert [e["why"] for e in tr] == ["deepest_uncertified"] and tr[0]["decision"] == 1 and tr[0]["version"] == 0
    assert not _events(loop, "train", "attach", "freeze") and loop.candidate is None
    st = loop._state()
    assert st.deepest_certified is False and "not certified" in loop.policy.why_not_train(st)
    # train_now ignores the schedule: its candidate is refused at the apply (the backstop) and discarded at birth
    loop.train_now(wait=True)
    d = _events(loop, "discard")
    assert d and "attach refused" in d[0]["why"] and not _events(loop, "attach", "freeze")
    assert loop.candidate is None and loop._route == {}


# ───────────────────────────── test 2: zone disjointness, one owner, Λ constant
def _check_routing(loop, recs):
    for r in recs:
        path = r["path"]
        assert r["owner"] == path[-1][0] and r["layer"] == r["zone"] == len(path) - 1
        for v, s, thr in path[:-1]:                         # every passed layer: below its routing threshold
            assert thr == loop._route[v] and (s is None or s < thr)
        v, s, thr = path[-1]
        deepest = r["layer"] == len(loop._layers) - 1 or r["owner"] == loop._layers[-1]
        if not deepest:                                     # a non-deepest owner: at or above its Λ
            assert s >= loop._route[v]
        assert r["alone"] == (r["answered_by"] is not None)
        if r["alone"]:                                      # one alone answer, by a layer on the path, at its λ
            j = r["answer_layer"]
            assert j <= r["layer"] and path[j][0] == r["answered_by"] and r["by"] in ("local", "oracle")
            assert r["answer_thr"] is not None and path[j][1] >= r["answer_thr"]
            if j < r["layer"]:                              # a passed layer answers below its Λ, at its loosened λ
                assert r["answer_thr"] <= path[j][1] < path[j][2]
        else:
            assert r["answer_layer"] is None and r["answer_thr"] is None
        if r["trial_zone"]:                                 # the candidate's zone: escalated, never answered alone
            assert not r["alone"] and r["candidate"] is not None and r["by"] in ("oracle", "person")


def test_every_decision_has_one_owner_and_the_trial_zone_is_the_escalation_zone_below_lambda():
    loop = _attached()
    lam = _events(loop, "attach")[0]["routing_thr"]
    at = _events(loop, "attach")[0]["decision"]
    recs = _recs(loop)
    _check_routing(loop, recs)
    zone = 0
    for r in recs:
        assert r["owner"] == 0 and len(r["path"]) == 1
        if r["n"] <= at:
            assert not r["trial_zone"] and r["shadow"] is None          # no candidate
            continue
        assert r["path"][0][2] == lam >= r["threshold"]                 # Λ constant in trial; λ_0 may loosen
        assert r["trial_zone"] == (r["score"] < lam)
        assert (r["shadow"] is not None) == r["trial_zone"]              # the candidate is scored in its zone only
        zone += r["trial_zone"]
    assert zone > 300


def test_a_layered_chain_routes_each_decision_to_one_owner_and_the_deeper_layer_answers_its_zone():
    loop = _attached()
    assert loop._threshold(1) < 0.5                                     # GOOD certified on the zone
    k = loop.candidate
    ev = loop._layer_in(k, loop.n)
    assert ev["event"] == "promote" and ev["additive"] and ev["zone_of"] == 0 and ev["routing_thr"] == loop._route[0]
    assert loop._layers == [0, 1] and loop.candidate is None and loop._vs[1]["state"] == "layer"
    loop.pause()                                                        # no next candidate yet: layer 1 learns
    before = len(_recs(loop))
    outs = _run(loop, _xs(7, 800))
    recs = _recs(loop)[before:]
    _check_routing(loop, recs)
    lam0 = loop._route[0]
    deep = [r for r in recs if r["owner"] == 1]
    assert deep and all(r["path"][0][1] < lam0 and r["path"][0][2] == lam0 for r in deep)
    assert all(r["owner"] == 0 and r["path"][0][1] >= lam0 for r in recs if r["layer"] == 0)
    # the deeper layer answers alone in its zone, with its own model's answer
    local = [(o, r) for o, r in zip(outs, recs) if r["answered_by"] == 1 and r["by"] == "local"]
    assert len(local) > 100 and all(o["answer"] in "ab" and o["owner"] == 1 for o, _ in local)
    # both layers learn on the base coin: L0 over its whole zone (layer 1's included), layer 1 over its own
    fed0, fed1 = set(loop.fed(0)), set(loop.fed(1))
    assert {r["n"] for r in recs if r["base"]} <= fed0 and any(r["n"] in fed0 for r in recs if r["owner"] == 1)
    assert {r["n"] for r in recs if r["owner"] == 1 and r["base"]} == {r["n"] for r in recs if r["n"] in fed1}
    assert all(r["base"] == r["sampled"] for r in recs)                  # both certified: ρ is the base rate
    # resumed: the next candidate attaches below layer 1 (Λ_1 := λ_1, layer 1 frozen), Λ_0 untouched
    loop.resume()
    _run(loop, _xs(8, 400))
    at = _events(loop, "attach")[-1]
    assert at["version"] == 2 and at["zone_of"] == 1 and loop._route == {0: lam0, 1: at["routing_thr"]}
    rest = [r for r in _recs(loop) if r["n"] > at["decision"]]
    _check_routing(loop, rest)
    assert all(r["trial_zone"] == (r["layer"] == 1 and r["path"][-1][1] < at["routing_thr"]) for r in rest)
    assert any(r["trial_zone"] for r in rest) and loop.fed(2) == [r["n"] for r in rest if r["trial_zone"]]
    # the cap: a chain of max_layers refuses the next attach
    assert loop._attach_refused() is None
    loop.max_layers = 2
    assert "max_layers" in loop._attach_refused()


# ───────────────────────────── test 3: trial reads
def test_the_candidate_reads_every_zone_decision_and_the_layer_above_reads_its_coin_samples():
    loop = _attached()
    at = _events(loop, "attach")[0]["decision"]
    recs = [r for r in _recs(loop) if r["n"] > at]
    zone_n = [r["n"] for r in recs if r["trial_zone"]]
    assert loop.fed(1) == zone_n                            # every zone decision, no coin filter, nothing outside
    st = loop._vs[1]["promise"].state()
    assert st["n"] + st["dropped"] == len(zone_n)           # read, or dropped before its grid was fixed
    assert st["seen"] == len(zone_n)                        # its grid reads zone scores only
    assert len(loop._vs[1]["samples"]) == len(zone_n)
    # L0 (base regime) reads the base coin over its whole zone, the trial zone included — not the owner's ρ there
    assert [n for n in loop.fed(0) if n > at] == [r["n"] for r in recs if r["base"]]
    assert any(r["trial_zone"] and r["base"] for r in recs)
    assert any(r["trial_zone"] and not r["base"] and r["label_source"] for r in recs)   # labelled, not L0's sample
    assert all(r["rho_base"] <= r["rho"] and (not r["base"] or r["sampled"]) for r in recs)


def test_a_missing_label_in_the_zone_counts_as_a_loss():
    loop = _loop([GOOD], policy=_policy(min_novel=10 ** 9))
    xs = _xs(0, 2500)
    _run(loop, xs[:600])
    assert loop.candidate == 1
    lam = loop._route[0]
    _run(loop, xs[600:], oracle=lambda x: None if x[1] < lam and x[2] < 0.1 else x[0])
    st = loop._vs[1]["promise"].state()
    assert st["missing"] > 0
    recs = [r for r in _recs(loop) if r["n"] > 600 and r["trial_zone"]]
    assert sum(r["missing"] for r in recs) > 0 and loop.fed(1)[-1] == recs[-1]["n"]


def test_a_discard_drops_the_routing_threshold_and_the_layer_above_learns_throughout():
    loop = _loop([BAD], policy=_policy(stale=500))
    _run(loop, _xs(0, 1500))
    at, un = _events(loop, "attach"), _events(loop, "unfreeze")
    assert at and un and un[0]["version"] == 0 and un[0]["routing_thr"] == at[0]["routing_thr"]
    d = _events(loop, "discard")[0]
    assert d["decision"] == un[0]["decision"] and "not certified" in d["why"]
    a, u = at[0]["decision"], un[0]["decision"]
    assert [n for n in loop.fed(0) if a < n <= u]           # learns during the trial (decision 21)
    assert [n for n in loop.fed(0) if n > u]                # and after the unfreeze
    rec_a = next(r for r in _recs(loop) if r["n"] == a)
    assert un[0]["v_index"] == rec_a["served_state"]["v_index"]   # the same index: no restart, one test
    rec_u = next(r for r in _recs(loop) if r["n"] == u + 1)
    assert rec_u["candidate"] in (None, 2) and (rec_u["candidate"] is not None or not rec_u["trial_zone"])


# ───────────────────────────── decision 21: the upstream layer keeps loosening and answers its band inside Z_1
# L0G: right above u = 0.6, wrong with probability 0.13 on (0.3, 0.6] and always below — its joint risk at λ ∈ [0.3,
# 0.6] is 0.13·(0.6 − λ): the strict points certify fast, the loose ones slowly (it keeps loosening after an attach)
L0G = _M("l0g", lambda x: (x[0] if x[1] > 0.6 or (x[1] > 0.3 and int(x[2] * 1e6) % 100 >= 13) else "wrong", x[1]))


def _early_attach():
    # training as soon as L0 certifies (first=50, no maturity gate): Λ_0 is a strict point, λ_0 loosens after it
    loop = AdapterLoop(_Hand([GOOD]), question="q", policy=_policy(first=50), max_risk=0.05, secret=KEY, initial=L0G,
                       additive=True, monitor=None, _detector_alone=True, wait=True, clock=lambda: 0.0, max_layers=2)
    _run(loop, _xs(0, 6000))
    return loop


def test_a_band_row_inside_layer_1s_zone_is_answered_by_l0_at_its_loosened_threshold():
    loop = _early_attach()
    at = _events(loop, "attach")[0]
    lam0 = at["routing_thr"]
    assert loop._layers == [0, 1] and loop._route == {0: lam0} and loop._threshold(0) < lam0   # λ_0 loosened
    recs = _recs(loop)
    _check_routing(loop, [r for r in recs if r["n"] > _events(loop, "promote")[0]["decision"]])   # the final chain
    band = [r for r in recs if r["owner"] == 1 and r["answered_by"] == 0]
    assert band                                             # owner 1, answered alone by L0 (its band [λ_0, Λ_0))
    items = {d["stream"]["n"]: d["loop"] for k, d in loop.log if k == "loop"}
    for r in band:
        assert r["alone"] and r["answer_layer"] == 0 and r["path"][0][2] == lam0
        assert r["answer_thr"] <= r["path"][0][1] < lam0 and r["answer_thr"] == r["threshold"]
        lp = items[r["n"]]
        if r["sampled"]:                                    # the label only by the coin
            assert r["by"] == "oracle" and lp["label"] is not None
        else:
            assert r["by"] == "local" and lp["label"] is None and lp["answer"] == lp["path_would"][0]
    assert any(not r["sampled"] for r in band)
    # nobody answers in the trial zone, though L0's loosened λ_0 would take part of it
    trial = [r for r in recs if r["trial_zone"]]
    assert any(r["threshold"] is not None and r["path"][0][1] >= r["threshold"] for r in trial)
    assert not any(r["alone"] for r in trial) and all(r["answered_by"] is None for r in trial)
    rep = loop.replay()
    assert rep["ok"], rep["mismatches"][:3]


# the checkpoint of an additive loop (stage B3): tests/test_stream_additive_b3.py


def test_a_one_layer_additive_run_replays():
    loop = _attached(n=1500)
    rep = loop.replay()
    assert rep["ok"], rep["mismatches"][:3]


# ───────────────────────────── a Dispatcher's _pre / _post with the chain changed or a layer restarted in between
def _chain_of_two():
    loop = _loop([GOOD, GOOD], policy=_policy(), max_layers=2)
    _run(loop, _xs(0, 4000))
    assert loop._layers == [0, 1]
    return loop


def _pre_owned_by(loop, v, alone=None):
    for x in _xs(9, 2000):
        pre = loop._pre(x)
        if pre.get("owner") == v and (alone is None or pre["alone"] == alone):
            return x, pre
    raise AssertionError(f"no decision owned by version {v}")


def test_a_detach_between_pre_and_post_makes_the_pre_route_stale_and_the_step_routes_again():
    loop = _chain_of_two()
    x, pre = _pre_owned_by(loop, 1)
    assert pre["layer"] == 1
    # rollback(1) (a control, or an alarm on another in-flight decision) detaches layer 1 before this decision's _post
    loop.rollback(1, by="test")
    out = loop._post(pre, x, x[0], "human")
    assert out["same"] is False and out["owner"] == 0 and out["layer"] == 0
    assert loop.status()["state"] == "running"
    rec = _recs(loop)[-1]
    assert rec["owner"] == 0 and [p[0] for p in rec["path"]] == [0]
    assert loop.replay()["ok"]


def test_a_restart_between_pre_and_post_records_the_threshold_in_force_at_the_draw():
    loop = _chain_of_two()
    x, pre = _pre_owned_by(loop, 1, alone=True)
    # λ_1 = +inf from here, nothing alone in its zone (the internal restart: no public control restarts a layer with
    # the decision number and the chain unchanged; the record must still read the threshold its draw used)
    loop._restart(1, loop.n)
    out = loop._post(pre, x, x[0], "human")
    assert out["same"] is True
    rec = _recs(loop)[-1]
    assert rec["owner"] == 1 and rec["path"][-1][2] is None and rec["alone"] is False
