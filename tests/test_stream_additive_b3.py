"""solvi.experimental.stream, additive coverage, stage B3 (persistence): the records carry every reached layer's score
and answer (path, path_would), replay recomputes the routing from them (1–3 layers, detach, per-layer restart), the
candidate's trace on its zone only, the checkpoint of an additive loop (format v2, the chain, per-layer detectors and
probation, the trial) continues exactly as the loop it was saved from, old / mismatched checkpoints, status() zones.
Spec tests 5, 6, 9 (audit part), 10 (status fields)."""
import json
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, RiskDetector
    from solvi.experimental.stream import checkpoint as ck

KEY = b"additive-b3-test-key"


# models at module level: a checkpoint pickles version 0 and the layers (HeadVersion)
def _f_l0(x):
    return (x[0] if x[1] > 0.3 else "wrong", x[1])


def _f_gw(x):
    return (x[0] if x[2] > 0.4 else "wrong", x[2])


def _f_good(x):
    return (x[0], x[2])


def _f_good2(x):
    return (x[0], x[3])


class _M:
    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, x):
        return self.f(x)

    def fingerprint(self):
        return f"m-{self.name}"


# x = (label, u, w, z): L0 right iff u > 0.3 (score u); GW right iff w > 0.4 (score w): a zone below it remains;
# GOOD right always (score w), GOOD2 right always (score z)
L0, GW, GOOD, GOOD2 = _M("l0", _f_l0), _M("gw", _f_gw), _M("good", _f_good), _M("good2", _f_good2)


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
    return cls(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0, pre_freeze=False, probe=False,
               **kw)


class _Detaching(DefaultPolicy):
    def alarm(self, st, who, report):
        return f"rollback:{report['layer']}" if report.get("layer") else "recertify"


def _clock():
    return 0.0


def _loop(models, *, policy=None, **kw):
    kw.setdefault("monitor", None)
    return AdapterLoop(_Hand(models), question="q", policy=_policy() if policy is None else policy, max_risk=0.05,
                       secret=KEY, initial=L0, additive=True, _detector_alone=True, wait=True, clock=_clock, **kw)


def _xs(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random(), r.random(), r.random()) for _ in range(n)]


def _truth(loop, x):
    return x[0]


def _run(loop, xs, oracle=_truth):
    return [loop.step(x, (lambda x=x: oracle(loop, x))) for x in xs]


def _items(loop, kind):
    return [d for k, d in loop.log if k == kind]


def _recs(loop):
    return [d["stream"] for d in _items(loop, "loop")]


def _events(loop, *kinds):
    return [e for e in _items(loop, "stream") if not kinds or e["event"] in kinds]


def _until(loop, xs, cond, oracle=_truth, start=0):
    i = start
    while not cond(loop):
        assert i < len(xs), "the condition never held"
        _run(loop, [xs[i]], oracle)
        i += 1
    return i


def _flip_zone1(lo, hi):
    """Labels flipped in layer 1's zone (u below Λ_0) on decisions lo..hi: a shift layer 1's detector sees."""
    def oracle(loop, x):
        lam = loop._route.get(0)
        if lam is not None and x[1] < lam and lo <= loop.n + 1 <= hi:
            return "b" if x[0] == "a" else "a"
        return x[0]
    return oracle


def _promoted_at(models, xs, **kw):
    loop = _loop(list(models), **kw)
    _until(loop, xs, lambda lp: len(lp._layers) == 2)
    return loop.n


# ───────────────────────────── records: the audit's view (spec test 9, audit part)
def _three():
    loop = _loop([GW, GOOD2, GOOD2], max_layers=3)
    _run(loop, _xs(1, 12000))
    assert loop._layers == [0, 1, 2], _events(loop, "promote", "discard")
    return loop


@pytest.fixture(scope="module")
def three():
    return _three()


def test_every_record_lets_an_auditor_recompute_the_routing_and_the_answer(three):
    loop = three
    since = {e["version"]: e["decision"] for e in _events(loop, "promote")}
    attach = {e["zone_of"]: e["routing_thr"] for e in _events(loop, "attach")}
    for d in _items(loop, "loop"):
        r, p = d["stream"], d["loop"]
        path, pw = r["path"], p["path_would"]
        assert len(pw) == len(path) and pw[0] == p["would"] and path[0][1] == r["score"]
        chain = [0] + [v for v, n in sorted(since.items(), key=lambda t: t[1]) if n < r["n"]]
        assert [e[0] for e in path] == chain[:len(path)]                      # the walk down the chain, in order
        for v, score, thr in path[:-1]:                                       # passed: below its routing threshold
            assert thr == attach[v] and score is not None and score < thr
        o, so, to = path[-1]
        assert r["owner"] == o and r["layer"] == r["zone"] == len(path) - 1
        if o != chain[-1]:                                                    # not the deepest: at or above Λ
            assert so >= attach[o]
        alone = not r["probation"] and to is not None and so is not None and so >= to
        assert r["alone"] == alone
        if r["by"] == "local":
            assert p["answer"] == pw[-1]                                      # the owner's answer, alone


def test_the_delta_indices_are_chronological_unique_and_within_delta(three):
    loop = three
    first = {}
    for e in _events(loop):
        if e.get("v_index") is not None and e["v_index"] not in first:
            first[e["v_index"]] = e["delta_v"]
    idx = list(first)
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert sum(first.values()) <= 0.05 + 1e-12
    for r in _recs(loop):                                                     # no alone answer without a certificate
        if r["alone"]:
            assert r["path"][-1][2] is not None


# ───────────────────────────── trace_candidate in additive mode: the candidate's zone only
def test_the_candidate_trace_is_on_its_zone_only_in_additive_mode():
    loop = _loop([GOOD], trace_candidate=True, max_layers=2)
    _run(loop, _xs(0, 2500))
    trial = [d for d in _items(loop, "loop") if d["stream"]["candidate"] is not None]
    assert trial and any(d["stream"]["trial_zone"] for d in trial) and not all(d["stream"]["trial_zone"] for d in trial)
    for d in trial:
        r, p = d["stream"], d["loop"]
        assert "candidate_score" in r and "candidate_would" in p
        if r["trial_zone"]:
            assert r["candidate_score"] == r["shadow"] is not None and p["candidate_would"] == p["shadow_would"]
        else:
            assert r["candidate_score"] is None and p["candidate_would"] is None and r["shadow"] is None
    for d in _items(loop, "loop"):
        if d["stream"]["candidate"] is None:
            assert "candidate_score" not in d["stream"]
    assert loop.replay()["ok"]


# ───────────────────────────── replay exactness (spec test 5)
def test_a_two_layer_run_with_a_per_layer_restart_and_a_detach_replays_and_rescores(tmp_path):
    xs = _xs(5, 6000)
    p = _promoted_at([GOOD], xs, max_layers=2, detector=RiskDetector(0.05, arl=50))
    loop = _loop([GOOD], max_layers=2, detector=RiskDetector(0.05, arl=50), versions_dir=tmp_path / "v")
    i = _until(loop, xs, lambda lp: lp.n >= p + 2500, oracle=_flip_zone1(p + 1, p + 1500))
    rs = _events(loop, "restart")
    assert rs and all(e["version"] == 1 and e["layer"] == 1 for e in rs)    # a restart of layer 1 only
    loop.rollback(1, by="op", why="test")                                     # the detach control
    _run(loop, xs[i:i + 600])
    assert _events(loop, "detach") and loop._layers[:1] == [0]
    rep = loop.replay(rescore=True)
    assert rep["ok"], rep["mismatches"][:3]
    assert any(len(r["path"]) == 2 for r in _recs(loop))


def test_a_three_layer_run_replays_and_rescores_every_layer(tmp_path):
    loop = _loop([GW, GOOD2, GOOD2], max_layers=3, versions_dir=tmp_path / "v")
    _run(loop, _xs(1, 12000))
    assert loop._layers == [0, 1, 2]
    loop.rollback(2, why="test")
    _run(loop, _xs(2, 400))
    rep = loop.replay(rescore=True)
    assert rep["ok"], rep["mismatches"][:3]
    assert any(len(r["path"]) == 3 for r in _recs(loop))


def test_an_alarm_detach_and_an_l0_refreeze_replay():
    loop = _loop([GOOD], max_layers=2, detector=RiskDetector(0.05, arl=50), policy=_policy(_Detaching))
    xs = _xs(7, 6000)
    i = _until(loop, xs, lambda lp: len(lp._layers) == 2)
    _run(loop, xs[i:i + 1500], oracle=_flip_zone1(0, 10 ** 9))
    assert _events(loop, "detach") and "control" not in _events(loop, "detach")[0]
    assert loop.replay()["ok"]
    loop = _loop([GOOD], max_layers=2, detector=RiskDetector(0.05, arl=50))
    i = _until(loop, xs, lambda lp: len(lp._layers) == 2)
    n0 = loop.n

    def l0_shift(lp, x):
        lam = lp._route.get(0)
        return ("b" if x[0] == "a" else "a") if lam is not None and x[1] >= lam and lp.n < n0 + 600 else x[0]
    _run(loop, xs[i:i + 3000], oracle=l0_shift)
    assert _events(loop, "refreeze") and _events(loop, "restart")[0]["version"] == 0
    rep = loop.replay()
    assert rep["ok"], rep["mismatches"][:3]


def test_replay_recomputes_the_routing_from_the_recorded_path_and_flags_a_changed_one():
    loop = _three()
    lam1 = loop._route[1]
    deep = next(d for d in _items(loop, "loop") if d["stream"]["owner"] == 2)
    deep["stream"]["path"][1][1] = max(lam1, 0.0) + 0.5                      # now it would stop at layer 1
    mid = next(d for d in _items(loop, "loop") if d["stream"]["owner"] == 1
               and d["stream"]["n"] > _events(loop, "promote")[-1]["decision"])
    mid["stream"]["path"][1][1] = -1.0                                        # now it would reach layer 2, unscored
    rep = loop.replay()
    assert not rep["ok"]
    at = {(n, what) for n, what, _ in rep["mismatches"]}
    assert (deep["stream"]["n"], "owner") in at or (deep["stream"]["n"], "error") in at
    assert any(n == mid["stream"]["n"] and what == "error" and "did not score" in why
               for n, what, why in rep["mismatches"])


# ───────────────────────────── the checkpoint (spec test 6)
def _snapshot(loop):
    return (list(loop._layers), dict(loop._route), set(loop._frozen), set(loop._relearn), dict(loop._left_of),
            loop._refused_why, {v: dict(z) for v, z in loop._zstat.items()}, loop.candidate, loop.left,
            {v: None if d is None else d.dump() for v, d in loop._dets.items()},
            None if loop.detector is None else loop.detector.dump())


def _resume_check(tmp_path, make, xs, cond, oracle=_truth, after=1500):
    a = make()
    i = _until(a, xs, cond, oracle)
    remaining = list(a.live.models)
    snap = _snapshot(a)
    a.save(tmp_path / "ck")
    _run(a, xs[i:i + after], oracle)
    b = AdapterLoop.load(tmp_path / "ck", _Hand(remaining), secret=KEY, clock=_clock)
    assert b.additive and _snapshot(b) == snap
    _run(b, xs[i:i + after], oracle)
    assert b.log == a.log                                                     # the same records and events, all
    assert b.status() == a.status()
    rep = b.replay()
    assert rep["ok"], rep["mismatches"][:3]
    return a, b


def test_a_checkpoint_mid_trial_continues_as_the_uninterrupted_run(tmp_path):
    def mid_trial(lp):
        c = lp.candidate
        return len(lp._layers) == 1 and c is not None and len(lp._vs[c]["samples"]) >= 30
    a, b = _resume_check(tmp_path, lambda: _loop([GOOD], max_layers=2), _xs(0, 6000), mid_trial)
    assert b._vs[b.candidate if b.candidate is not None else 1]["cmp"] is None    # no comparison certificate
    assert len(a._layers) == 2                                                # layered after the load, both


def test_a_checkpoint_mid_zone_with_a_second_candidate_in_trial_continues_exactly(tmp_path):
    def second(lp):
        c = lp.candidate
        return len(lp._layers) == 2 and c is not None and len(lp._vs[c]["samples"]) >= 20
    a, b = _resume_check(tmp_path, lambda: _loop([GW, GOOD2, GOOD2], max_layers=3), _xs(1, 12000), second,
                         after=4000)
    assert b._layers == [0, 1, 2] and b._frozen == {0, 1}                    # layered after the load, both
    assert any(len(r["path"]) == 3 for r in _recs(b))


def test_a_checkpoint_during_a_layer_restart_keeps_its_detector_probation_and_relearning(tmp_path):
    xs = _xs(5, 6000)
    det = lambda: RiskDetector(0.05, arl=50)  # noqa: E731
    p = _promoted_at([GOOD], xs, max_layers=2, detector=det())
    oracle = _flip_zone1(p + 1, p + 1500)
    a, b = _resume_check(tmp_path, lambda: _loop([GOOD], max_layers=2, detector=det()), xs,
                         lambda lp: lp._left_of.get(1, 0) > 0, oracle=oracle, after=2500)
    assert _events(b, "restart")


def test_a_checkpoint_in_l0_relearning_keeps_the_relearn_set(tmp_path):
    xs = _xs(7, 6000)
    det = lambda: RiskDetector(0.05, arl=50)  # noqa: E731

    def l0_shift(lp, x):
        lam = lp._route.get(0)
        start = next((e["decision"] for e in _events(lp, "promote")), None)
        return (("b" if x[0] == "a" else "a") if lam is not None and start is not None and x[1] >= lam
                and lp.n < start + 600 else x[0])
    a, b = _resume_check(tmp_path, lambda: _loop([GOOD], max_layers=2, detector=det()), xs,
                         lambda lp: bool(lp._relearn), oracle=l0_shift, after=2500)
    assert _events(b, "refreeze")


def _manifest(path):
    folder = path / (path / ck.CURRENT).read_text().strip()
    return folder, json.loads((folder / "manifest.json").read_bytes())


def _resign(folder, m):
    mb = ck._dumps(m)
    (folder / "manifest.json").write_bytes(mb)
    (folder / "manifest.sig").write_bytes(ck._dumps(ck._sig(KEY, mb)))


def test_a_replacement_checkpoint_stays_v1_and_loads_and_protocol_mismatches_are_refused(tmp_path):
    rl = AdapterLoop(_Hand([GOOD]), question="q", policy=_policy(), max_risk=0.05, secret=KEY, initial=L0,
                     monitor=None, _detector_alone=True, wait=True, clock=_clock)
    _run(rl, _xs(0, 800))
    rl.save(tmp_path / "r")
    folder, m = _manifest(tmp_path / "r")
    assert m["format"] == ck.FORMAT == "solvi adapter loop checkpoint v1"
    assert "additive" not in m["config"] and "max_layers" not in m["config"]
    assert "additive" not in json.loads((folder / "state.json").read_bytes())
    back = AdapterLoop.load(tmp_path / "r", _Hand([GOOD]), secret=KEY, clock=_clock)
    assert not back.additive and back.n == rl.n
    with pytest.raises(ValueError, match="replacement loop.*additive=True"):
        AdapterLoop.load(tmp_path / "r", _Hand([GOOD]), secret=KEY, clock=_clock, additive=True)
    al = _loop([GOOD], max_layers=2)
    _run(al, _xs(0, 800))
    al.save(tmp_path / "a")
    folder, m = _manifest(tmp_path / "a")
    assert m["format"] == ck.FORMAT_ADDITIVE != ck.FORMAT and "additive" in m["format"]
    assert m["config"]["additive"] is True and m["config"]["max_layers"] == 2
    with pytest.raises(ValueError, match="additive loop.*additive=False"):
        AdapterLoop.load(tmp_path / "a", _Hand([GOOD]), secret=KEY, clock=_clock, additive=False)
    assert AdapterLoop.load(tmp_path / "a", _Hand([GOOD]), secret=KEY, clock=_clock, additive=True).additive
    m2 = dict(m, config={**m["config"], "additive": False})                   # a v2 manifest claiming replacement
    _resign(folder, m2)
    with pytest.raises(ValueError, match="disagree"):
        AdapterLoop.load(tmp_path / "a", _Hand([GOOD]), secret=KEY, clock=_clock)
    _resign(folder, dict(m, format="solvi adapter loop checkpoint v9"))       # a newer library's format
    with pytest.raises(ValueError, match="newer solvi"):
        AdapterLoop.load(tmp_path / "a", _Hand([GOOD]), secret=KEY, clock=_clock)
    # an older library compares the format with v1 and names the one it found: the refusal says what is needed
    assert ck.FORMAT_ADDITIVE != "solvi adapter loop checkpoint v1" and "additive=True" in ck.FORMAT_ADDITIVE


# ───────────────────────────── status() zones (spec test 10, status fields)
def test_status_reports_each_zone_and_its_counts_match_the_records(three):
    loop = three
    st = loop.status()
    z = st["zones"]
    assert [row["version"] for row in z["layers"]] == [0, 1, 2] and z["deepest"] == 2 and z["max_layers"] == 3
    assert z["zone_share"][0] == 1.0 and 0 < z["zone_share"][2] < z["zone_share"][1] < 1
    since = {0: 0, **{e["version"]: e["decision"] for e in _events(loop, "promote")}}
    for row in z["layers"]:
        v = row["version"]
        assert row["routing_thr"] == loop._route.get(v) and row["learning"] == (v == 2)
        assert row["threshold"] == loop._threshold(v) and row["certified"]
        got = {"reach": 0, "owned": 0, "alone": 0, "sampled": 0, "s_alone": 0, "s_wrong": 0}
        for d in _items(loop, "loop"):
            r, p = d["stream"], d["loop"]
            if r["n"] <= since[v] or v not in [e[0] for e in r["path"]]:
                continue
            got["reach"] += 1
            if r["owner"] == v:
                got["owned"] += 1
                got["alone"] += r["alone"]
                if r["sampled"]:
                    got["sampled"] += 1
                    got["s_alone"] += r["alone"]
                    got["s_wrong"] += r["alone"] and (p["label"] is None or p["path_would"][-1] != p["label"])
        assert {k: row[k] for k in got} == got and row["since"] == since[v]
        assert row["zone_share"] == got["reach"] / (loop.n - since[v])
        assert row["risk"]["joint"] == (row["s_wrong"] / row["sampled"] if row["sampled"] else None)
    c = z["composite"]
    assert c["joint"] is not None and 0 <= c["joint"] <= 1 and c["selective"] is not None
    assert c["joint_bound"] == pytest.approx(0.05 * (1 + z["zone_share"][1] + z["zone_share"][2]))
    assert loop._state().zone_share == z["zone_share"]
    rl = AdapterLoop(_Hand([GOOD]), question="q", policy=_policy(), max_risk=0.05, secret=KEY, initial=L0,
                     monitor=None, _detector_alone=True, wait=True, clock=_clock)
    assert "zones" not in rl.status() and rl._state().zone_share is None
