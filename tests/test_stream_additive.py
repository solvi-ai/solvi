"""solvi.experimental.stream, additive coverage, stage B1 (the routing core): AdapterLoop(additive=True) attaches a
candidate below the served chain — zones, owner routing, zone-only learning, frozen promises, attach / unfreeze — and
additive=False stays the replacement protocol, byte for byte."""
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


def test_additive_false_writes_the_same_records_and_events_as_the_default():
    xs = _xs(3, 2500)
    runs = []
    for kw in ({}, {"additive": False}, {"additive": False, "max_layers": 5}):
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


def test_attach_freezes_the_deepest_layer_at_its_threshold_and_drops_the_comparison():
    loop = _attached()
    at = _events(loop, "attach")
    assert len(at) == 1 and at[0]["version"] == 1 and at[0]["zone_of"] == 0
    lam = at[0]["routing_thr"]
    assert lam == loop._route[0] == loop._threshold(0) and 0.2 < lam < 0.35
    assert loop._vs[1]["cmp"] is None and loop.candidate == 1
    st = loop._state()
    assert st.layers == (0,) and st.deepest == 0 and st.deepest_certified
    assert st.version(0).routing_thr == lam and st.version(0).learning is False and st.version(1).learning is True
    assert not _events(loop, "promote")                    # stage B1 never layers an additive candidate by itself


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
    assert loop.candidate is None and loop._route == {} and loop._frozen == set()


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
        if r["alone"]:                                      # one alone answer at most, the owner's, at its λ
            assert thr is not None and s >= thr and r["by"] in ("local", "oracle")
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
        assert r["path"][0][2] == lam == r["threshold"]                 # Λ (= the frozen λ) constant in trial
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
    assert 0 in loop._frozen and 1 not in loop._frozen
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
    local = [(o, r) for o, r in zip(outs, recs) if r["owner"] == 1 and r["by"] == "local"]
    assert len(local) > 100 and all(o["answer"] in "ab" and o["owner"] == 1 for o, _ in local)
    # the frozen L0 stays frozen; layer 1 (deepest, learning) reads coin samples only
    assert all(r["served_state"]["n"] == recs[0]["served_state"]["n"] for r in recs)
    fed1 = set(loop.fed(1))
    assert all(r["n"] in fed1 for r in recs if r["owner"] == 1 and r["sampled"]) and fed1
    assert not any(r["n"] in fed1 for r in recs if r["owner"] == 0)
    # resumed: the next candidate attaches below layer 1 (Λ_1 := λ_1, layer 1 frozen), Λ_0 untouched
    loop.resume()
    _run(loop, _xs(8, 400))
    at = _events(loop, "attach")[-1]
    assert at["version"] == 2 and at["zone_of"] == 1 and loop._route == {0: lam0, 1: at["routing_thr"]}
    assert loop._frozen == {0, 1}
    rest = [r for r in _recs(loop) if r["n"] > at["decision"]]
    _check_routing(loop, rest)
    assert all(r["trial_zone"] == (r["layer"] == 1 and r["path"][-1][1] < at["routing_thr"]) for r in rest)
    assert any(r["trial_zone"] for r in rest) and loop.fed(2) == [r["n"] for r in rest if r["trial_zone"]]
    # the cap: a chain of max_layers refuses the next attach
    assert loop._attach_refused() is None
    loop.max_layers = 2
    assert "max_layers" in loop._attach_refused()


# ───────────────────────────── test 3: trial reads
def test_the_candidate_reads_every_zone_decision_and_the_frozen_layer_reads_nothing():
    loop = _attached()
    at = _events(loop, "attach")[0]["decision"]
    recs = [r for r in _recs(loop) if r["n"] > at]
    zone_n = [r["n"] for r in recs if r["trial_zone"]]
    assert loop.fed(1) == zone_n                            # every zone decision, no coin filter, nothing outside
    st = loop._vs[1]["promise"].state()
    assert st["n"] + st["dropped"] == len(zone_n)           # read, or dropped before its grid was fixed
    assert st["seen"] == len(zone_n)                        # its grid reads zone scores only
    assert len(loop._vs[1]["samples"]) == len(zone_n)
    s0 = [r["served_state"] for r in recs]
    assert len({(s["n"], s["threshold"], s["evidence_fp"]) for s in s0}) == 1   # frozen: no learn, λ fixed
    assert not [n for n in loop.fed(0) if n > at]


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


def test_a_discard_unfreezes_the_layer_above_and_it_learns_again():
    loop = _loop([BAD], policy=_policy(stale=500))
    _run(loop, _xs(0, 1500))
    at, un = _events(loop, "attach"), _events(loop, "unfreeze")
    assert at and un and un[0]["version"] == 0 and un[0]["routing_thr"] == at[0]["routing_thr"]
    d = _events(loop, "discard")[0]
    assert d["decision"] == un[0]["decision"] and "not certified" in d["why"]
    a, u = at[0]["decision"], un[0]["decision"]
    assert not [n for n in loop.fed(0) if a < n <= u]       # frozen during the trial
    later = [n for n in loop.fed(0) if n > u]
    assert later                                            # learns again after the unfreeze (same index)
    rec_a = next(r for r in _recs(loop) if r["n"] == a)
    assert un[0]["v_index"] == rec_a["served_state"]["v_index"]   # the same index: a frozen process never bet
    rec_u = next(r for r in _recs(loop) if r["n"] == u + 1)
    assert rec_u["candidate"] in (None, 2) and (rec_u["candidate"] is not None or not rec_u["trial_zone"])


# the checkpoint of an additive loop (stage B3): tests/test_stream_additive_b3.py


def test_a_one_layer_additive_run_replays():
    loop = _attached(n=1500)
    rep = loop.replay()
    assert rep["ok"], rep["mismatches"][:3]
