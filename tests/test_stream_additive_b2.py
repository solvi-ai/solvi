"""solvi.experimental.stream, additive coverage, stage B2 (gate + drift): the additive gate (certified, novel,
novel_risk, probe — no share / margin comparison) layers a certified candidate below the deepest layer; the
max_layers cap and an uncertified deepest layer refuse a training ("train_refused"); one RiskDetector per layer, an
alarm restarts only its layer (its zone escalates until it re-certifies, a frozen layer is frozen again after);
rollback(j) detaches the layers ≥ j; on an attached Dispatcher a decision owned by a deeper layer is decided with
that layer's version installed. Spec tests 4, 7, 8."""
import random
import warnings

import pytest

from solvi.core.store import JSONLStorage
from test_stream_attach import KEY as AKEY
from test_stream_attach import Learner, _disp, _emails
from test_stream_attach import _policy as _apolicy

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, Paired, RiskDetector
    from solvi.experimental.stream.policy import LoopState, VersionInfo

KEY = b"additive-b2-test-key"


class _M:
    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, x):
        return self.f(x)

    def fingerprint(self):
        return f"m-{self.name}"


# x = (label, u, w, z): L0 right iff u > 0.3 (score u); GW right iff w > 0.4 (score w): it certifies near 0.4 on
# its zone, so a zone below it remains; GOOD2 always right (score z)
L0 = _M("l0", lambda x: (x[0] if x[1] > 0.3 else "wrong", x[1]))
GW = _M("gw", lambda x: (x[0] if x[2] > 0.4 else "wrong", x[2]))
GOOD = _M("good", lambda x: (x[0], x[2]))
GOOD2 = _M("good2", lambda x: (x[0], x[3]))


class _Hand:
    def __init__(self, models):
        self.models, self._seen = list(models), 0

    def update(self, x, y, **_):
        self._seen += 1

    def seen(self):
        return self._seen

    def train(self, job):
        return self.models.pop(0) if len(self.models) > 1 else self.models[0]


def _policy(cls=DefaultPolicy, **kw):
    # the routing / layering mechanics run without the maturity gate (stage B4, on by default in additive mode):
    # their training timing stays the one they were written for; the gate has its own tests (test_stream_additive_b4)
    kw.setdefault("mature_patience", None)
    return cls(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0, pre_freeze=False, probe=False,
               **kw)


def _loop(models, *, policy=None, **kw):
    kw.setdefault("monitor", None)
    return AdapterLoop(_Hand(models), question="q", policy=_policy() if policy is None else policy, max_risk=0.05,
                       secret=KEY, initial=L0, additive=True, _detector_alone=True, wait=True, clock=lambda: 0.0,
                       **kw)


def _xs(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random(), r.random(), r.random()) for _ in range(n)]


def _run(loop, xs, oracle=None):
    return [loop.step(x, (lambda x=x: x[0]) if oracle is None else (lambda x=x: oracle(x))) for x in xs]


def _recs(loop):
    return [d["stream"] for k, d in loop.log if k == "loop"]


def _events(loop, *kinds):
    return [e for k, e in loop.log if k == "stream" and (not kinds or e["event"] in kinds)]


def _layered(models=(GOOD,), n=2500, **kw):
    """L0 + one layer, then paused (no next candidate)."""
    loop = _loop(list(models), **kw)
    for x in _xs(0, 6000):
        loop.step(x, lambda x=x: x[0])
        if len(loop._layers) == 2:
            break
    assert loop._layers == [0, 1], _events(loop, "attach", "promote", "discard")
    loop.pause()
    return loop


# ───────────────────────────── test 4: the additive gate
def _st(layers, alpha=0.05):
    vs = (VersionInfo(0, "served"), VersionInfo(1, "candidate", certified=True, threshold=0.5))
    return LoopState(n=10, served=0, candidate=1, labels={}, versions=vs, holdout=(), alpha=alpha, alarms=(),
                     served_history=(0,), served_since=0, trainings=(), job=None, now=0.0, paused=False,
                     **({} if layers is None else {"layers": layers, "deepest": layers[-1],
                                                   "deepest_certified": True}))


def test_the_additive_gate_is_certified_novel_novel_risk_probe_and_drops_the_comparison():
    pol = DefaultPolicy(probe=True)
    good = Paired(200, 120, 0, 190, n_seen=0)                # served share 0.95 > candidate's 0.6: no margin
    rep = pol.promote_checks(_st(None), good)
    assert list(rep) == ["certified", "novel", "min_share", "margin", "novel_risk", "probe"]
    assert not rep["margin"]["ok"] and not pol.promote(_st(None), good)       # replacement: loses on the margin
    add = pol.promote_checks(_st((0,)), good)
    assert list(add) == ["certified", "novel", "novel_risk", "probe"] and all(g["ok"] for g in add.values())
    assert pol.promote(_st((0,)), good)                                         # additive: nothing to compare with
    risky = Paired(200, 120, 9, 0)                           # 9 / 200 wrong alone: Wilson upper bound > α
    assert not pol.promote_checks(_st((0,)), risky)["novel_risk"]["ok"] and not pol.promote(_st((0,)), risky)
    few = Paired(200, 20, 0, 0)                              # 20 novel alone answers < probe_min 30
    assert not pol.promote_checks(_st((0,)), few)["probe"]["ok"] and not pol.promote(_st((0,)), few)
    assert not pol.promote(_st((0,)), Paired(50, 50, 0, 0))                     # n_novel < min_novel


def test_a_certified_candidate_failing_the_gate_does_not_serve_and_is_discarded_with_the_gate_reason():
    loop = _loop([GOOD], policy=_policy(min_novel=10 ** 9, trial=600))
    _run(loop, _xs(0, 2500))
    assert not _events(loop, "promote") and loop._layers == [0]
    assert all(r["owner"] == 0 for r in _recs(loop))                          # nothing below L0 ever answered
    d = [e for e in _events(loop, "discard") if "additive gate" in e["why"]]
    assert d and "failed: novel" in d[0]["why"] and "share" not in d[0]["why"]
    un = _events(loop, "unfreeze")
    assert un and un[0]["decision"] == d[0]["decision"]                        # the layer above learns again


def test_a_candidate_passing_the_gate_owns_its_zone_from_the_next_decision():
    loop = _loop([GOOD], policy=_policy())
    _run(loop, _xs(0, 2500))
    pr = _events(loop, "promote")
    assert len(pr) >= 1
    p = pr[0]
    lam0 = p["routing_thr"]
    assert p["additive"] and p["version"] == 1 and p["zone_of"] == 0 and p["from"] == 0 and p["to"] == 1
    assert p["n_novel"] >= 100 and p["wrong_novel"] == 0 and p["thr_k"] < 1 and "install" not in p
    assert "l_s" not in p and "thr_s" not in p                                  # no comparison with L0
    at = p["decision"]
    recs = {r["n"]: r for r in _recs(loop)}
    assert recs[at]["owner"] == 0 and recs[at]["trial_zone"] == (recs[at]["score"] < lam0)
    after = [r for n, r in recs.items() if n > at]
    assert all((r["owner"] == 1) == (r["score"] < lam0) for r in after)       # Z_1 = {score_0 < Λ_0}
    assert any(r["owner"] == 1 and r["by"] == "local" for r in after)
    assert loop._vs[1]["state"] == "layer" and 1 in loop._dets
    st = loop._state()
    assert st.layers[:2] == (0, 1) and st.version(0).routing_thr == lam0


def test_the_promote_control_layers_a_certified_candidate_and_force_skips_only_the_gate():
    loop = _loop([GOOD], policy=_policy(min_novel=10 ** 9))
    xs = _xs(0, 3000)
    i = 0
    while loop.candidate is None or loop._threshold(loop.candidate) == float("inf"):
        loop.step(xs[i], lambda x=xs[i]: x[0])
        i += 1
    with pytest.raises(ValueError, match="does not promote"):
        loop.promote()
    e = loop.promote(force=True, by="op", why="test")
    assert e["event"] == "promote" and e["additive"] and e["forced"] and e["control"] and loop._layers == [0, 1]
    with pytest.raises(ValueError, match="not the candidate"):
        loop.promote(force=True)
    _run(loop, xs[i:i + 200])
    assert any(r["owner"] == 1 for r in _recs(loop)[-200:])


# ───────────────────────────── test 8: layering, nesting, the cap
def test_three_layers_nest_and_the_cap_refuses_the_next_training():
    loop = _loop([GW, GOOD2, GOOD2], max_layers=3)
    _run(loop, _xs(1, 12000))
    assert loop._layers == [0, 1, 2], (_events(loop, "promote", "discard"), loop._layers)
    lam0, lam1 = loop._route[0], loop._route[1]
    assert set(loop._route) == {0, 1} and loop._frozen == {0, 1}
    last = _events(loop, "promote")[-1]["decision"]
    for r in _recs(loop):
        if r["n"] <= last:
            continue
        s0 = r["path"][0][1]
        if s0 >= lam0:
            assert r["owner"] == 0 and len(r["path"]) == 1
        elif r["path"][1][1] >= lam1:
            assert r["owner"] == 1 and len(r["path"]) == 2
        else:
            assert r["owner"] == 2 and len(r["path"]) == 3               # Z_2 ⊂ Z_1 ⊂ Z_0
    tr = [e for e in _events(loop, "train_refused") if e["decision"] >= last]
    assert [e["why"] for e in tr] == ["max_layers"] and tr[0]["decision"] == last   # once, at the layering
    assert not [e for e in _events(loop, "train") if e["decision"] > last]
    job = loop.train_now(wait=True)
    assert job.state == "refused" and "max_layers" in job.why


# ───────────────────────────── test 7: drift per layer
def test_a_shift_in_layer_1s_zone_restarts_only_layer_1_and_it_serves_again_after_recertifying():
    loop = _layered(detector=RiskDetector(0.05, arl=50))
    lam0, thr0 = loop._route[0], loop._threshold(0)
    assert loop._dets[1] is not None and loop._dets[1] is not loop.detector
    n0 = loop.n
    xs = _xs(5, 6000)
    flip = lambda x: ("b" if x[0] == "a" else "a") if x[1] < lam0 and loop.n < n0 + 1500 else x[0]  # noqa: E731
    _run(loop, xs, oracle=flip)
    al = _events(loop, "alarm")
    assert al and all(a["version"] == 1 and a["layer"] == 1 for a in al)       # only layer 1 alarmed
    rs = _events(loop, "restart")
    assert rs and all(e["version"] == 1 and e["layer"] == 1 for e in rs)
    a = al[0]["decision"]
    assert loop._route == {0: lam0} and loop._threshold(0) == thr0 and loop._layers == [0, 1]   # Λ_0, L0 untouched
    recs = [r for r in _recs(loop) if r["n"] > a]
    l0 = [r for r in recs if r["owner"] == 0]
    assert l0 and sum(r["alone"] for r in l0) > 0.9 * len(l0)                   # layer 0 keeps serving
    after = [r for r in recs if r["owner"] == 1 and r["n"] < a + 200]
    assert after and not any(r["alone"] for r in after)                         # its zone escalates
    assert all(r["sampled"] for r in after)                                     # every zone decision labelled
    assert any(r["owner"] == 1 and r["alone"] for r in recs if r["n"] > n0 + 2500)   # recertified: serves again


def test_a_shift_in_l0s_zone_restarts_l0_only_layer_1_keeps_serving_and_l0_is_refrozen():
    loop = _layered(detector=RiskDetector(0.05, arl=50))
    lam0 = loop._route[0]
    n0 = loop.n
    flip = lambda x: ("b" if x[0] == "a" else "a") if x[1] >= lam0 and loop.n < n0 + 600 else x[0]  # noqa: E731
    _run(loop, _xs(6, 6000), oracle=flip)
    al = _events(loop, "alarm")
    assert al and al[0]["version"] == 0 and al[0]["layer"] == 0
    a = al[0]["decision"]
    assert loop._route == {0: lam0}                                             # Λ_0 fixed
    recs = [r for r in _recs(loop) if r["n"] > a]
    assert not any(r["alone"] for r in recs if r["owner"] == 0 and r["n"] < a + 200)   # L0's zone escalates
    l1 = [r for r in recs if r["owner"] == 1 and r["n"] < a + 300]
    assert l1 and sum(r["alone"] for r in l1) > 0.5 * len(l1)                  # layer 1 keeps serving
    rf = _events(loop, "refreeze")
    assert rf and rf[0]["version"] == 0 and 0 in loop._frozen and not loop._relearn   # frozen again once certified
    fed0 = loop.fed(0)
    assert [n for n in fed0 if a < n <= rf[0]["decision"]]                     # it learned between
    assert not [n for n in fed0 if n > rf[0]["decision"]]                      # and not after
    assert any(r["owner"] == 0 and r["alone"] for r in _recs(loop) if r["n"] > rf[0]["decision"])


class _Detaching(DefaultPolicy):
    def alarm(self, st, who, report):
        return f"rollback:{report['layer']}" if report.get("layer") else "recertify"


def test_an_alarm_answered_by_rollback_detaches_the_layer_and_the_one_above_learns_again():
    loop = _layered(detector=RiskDetector(0.05, arl=50), policy=_policy(_Detaching))
    lam0 = loop._route[0]
    flip = lambda x: ("b" if x[0] == "a" else "a") if x[1] < lam0 else x[0]  # noqa: E731
    _run(loop, _xs(7, 1500), oracle=flip)
    det = _events(loop, "detach")
    assert len(det) == 1 and det[0]["removed"] == [1] and det[0]["layer"] == 1 and det[0]["version"] == 0
    assert det[0]["routing_thr"] == lam0 and "control" not in det[0]
    d = det[0]["decision"]
    assert loop._layers == [0] and loop._route == {} and loop._frozen == set() and loop._vs[1]["state"] == "detached"
    assert loop._vs[1]["model"] is None and 1 not in loop._dets
    assert all(r["owner"] == 0 and len(r["path"]) == 1 for r in _recs(loop) if r["n"] > d)
    assert [n for n in loop.fed(0) if n > d]                                   # L0 learns again (deepest)


def test_the_rollback_control_detaches_the_layers_at_and_below_an_index():
    loop = _loop([GW, GOOD2, GOOD2], max_layers=3)
    _run(loop, _xs(1, 12000))
    assert loop._layers == [0, 1, 2]
    for bad in (0, 3, "1", True):
        with pytest.raises(ValueError, match="layer index"):
            loop.rollback(bad)
    lam0 = loop._route[0]
    e = loop.rollback(2, by="op", why="test")
    assert e["event"] == "detach" and e["removed"] == [2] and e["control"] and e["version"] == 1
    assert loop._layers == [0, 1] and loop._route == {0: lam0} and loop._frozen == {0}
    loop.resume() if loop.paused else None
    e = loop.rollback(1)
    assert e["removed"] == [1] and loop._layers == [0] and loop._route == {} and not loop._frozen
    assert loop.candidate is None and loop._refused_why is None


# ───────────────────────────── the Dispatcher path: an owner below L0 decides with its version installed
def test_a_dispatcher_decides_a_deeper_layers_zone_with_that_layers_version_installed(tmp_path):
    disp = _disp(storage=JSONLStorage(tmp_path / "d.jsonl"))
    loop = AdapterLoop.attach(disp, Learner(), question="team", max_risk=0.05, secret=AKEY,
                              initial=disp.system.holder.ver, policy=_apolicy(), wait=True, clock=lambda: 1000.0,
                              additive=True, versions_dir=tmp_path / "v")
    out = [disp.ask(x) for x in _emails(3000)]
    pr = [e for k, e in loop.log if k == "stream" and e["event"] == "promote"]
    assert pr and pr[0]["additive"], loop.versions()
    used = disp.system.holder.used
    assert len(used) == len(out)
    deep = [(d, fp) for d, fp in zip(out, used) if d.learning.get("owner", 0) != 0]
    assert deep
    for d, fp in zip(out, used):
        lr = d.learning
        assert lr["served"] == 0 and lr["fingerprint"] == loop._vs[0]["fingerprint"]
        assert fp == lr["owner_fingerprint"], lr                 # System 1 decided with the owner's version
    alone = [d for d, _ in deep if d.learning["alone"]]
    assert alone and all(d.by == "s1" for d in alone)
    assert disp.system.holder.ver.fingerprint() == loop._vs[0]["fingerprint"]   # L0 installed again after each
    assert disp.replay_all() == []                                # each decision under its owner's version
