"""solvi.experimental.stream, additive coverage: the usefulness certificate (decision 25). The pointwise procedure of
StreamPromise, the loop's construction checks, the fifth start gate `useful`, the composed answer threshold, the
promote event's fields, the δ_use audit from the records, checkpoint and replay with the certificate, and the
replacement / useful_error=None paths unchanged."""
import json
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.stream import AdapterLoop, Budget, DefaultPolicy, StreamPromise
    from solvi.experimental.stream._eprocess import DeltaBudget

KEY = b"usefulness-test-key"
GRID = [round(1 - 0.05 * i, 2) for i in range(1, 20)]


# models at module level: a checkpoint pickles them
def _f_l0(x):                                      # right iff u > 0.6, score u
    return (x[0] if x[1] > 0.6 else "wrong", x[1])


def _f_noise(x):                                   # wrong w.p. 0.9 everywhere; 2 % of the zone scores 0.97, the rest
    s = 0.97 if x[2] < 0.02 else 0.04 * x[2]       # below every grid point: joint risk ≈ 0.018 ≤ α, error 0.9
    return (("b" if x[0] == "a" else "a") if x[3] < 0.9 else x[0], s)


def _f_real(x):                                    # wrong w.p. 0.1 everywhere, score w
    return (("b" if x[0] == "a" else "a") if x[3] < 0.1 else x[0], x[2])


class _M:
    def __init__(self, name, f):
        self.name, self.f = name, f

    def score(self, x):
        return self.f(x)

    def fingerprint(self):
        return f"m-{self.name}"


L0, NOISE, REAL = _M("l0", _f_l0), _M("noise", _f_noise), _M("real", _f_real)


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
    kw.setdefault("mature_patience", None)
    return DefaultPolicy(Budget(min_interval_s=0.0, per_day=10 ** 6), calib=0.0, holdout=0.0, pre_freeze=False,
                         probe=False, **kw)


def _clock():
    return 0.0


def _loop(models, *, policy=None, **kw):
    kw.setdefault("max_layers", 2)
    return AdapterLoop(_Hand(models), question="q", policy=_policy() if policy is None else policy, max_risk=0.05,
                       secret=KEY, initial=L0, additive=True, monitor=None, _detector_alone=True, wait=True,
                       clock=_clock, promise_options={"grid": GRID}, **kw)


def _xs(seed, n):
    r = random.Random(seed)
    return [(r.choice("ab"), r.random(), r.random(), r.random()) for _ in range(n)]


def _run(loop, xs):
    for x in xs:
        loop.step(x, lambda x=x: x[0])


def _events(loop, name=None):
    return [d for k, d in loop.log if k == "stream" and (name is None or d.get("event") == name)]


# ───────────────────────────── StreamPromise(procedure="pointwise")
def test_pointwise_certifies_a_loose_point_before_a_strict_one_that_no_sample_reaches():
    pw = StreamPromise(max_error=0.5, grid=[0.9, 0.5], procedure="pointwise", delta=0.05)
    fx = StreamPromise(max_error=0.5, grid=[0.9, 0.5], delta=0.05)
    for _ in range(60):
        for p in (pw, fx):
            p.learn(0.6, "a", "a")                     # alone at 0.5 only, always right
    assert pw.certified_at(0.5) and not pw.certified_at(0.9) and pw.threshold == 0.5
    assert math.isinf(fx.threshold)                    # the fixed sequence starves on 0.9
    assert pw.level_now == pytest.approx(math.log(2 / pw.delta_v), abs=1e-9)
    assert fx.level_now == pytest.approx(math.log(1 / fx.delta_v), abs=1e-9)
    assert not pw.certified_at(0.7) and pw.log_wealth_at(0.7) is None and pw.log_wealth_at(0.5) > 0


def test_pointwise_is_named_in_state_dump_and_fingerprint_and_fixed_is_unchanged():
    fx = StreamPromise(max_error=0.5, grid=GRID)
    pw = StreamPromise(max_error=0.5, grid=GRID, procedure="pointwise")
    assert "procedure" not in fx.state() and "procedure" not in fx.dump() and "procedure" not in fx.record()
    assert pw.state()["procedure"] == "pointwise" and pw.dump()["procedure"] == "pointwise"
    assert fx.fingerprint() != pw.fingerprint() and "pointwise" in pw.promise and "pointwise" not in fx.promise
    assert StreamPromise(max_error=0.5, grid=GRID, procedure="fixed").fingerprint() == fx.fingerprint()
    with pytest.raises(ValueError, match="'pointwise' StreamPromise"):
        fx.restore(pw.dump())
    with pytest.raises(ValueError, match="procedure"):
        StreamPromise(max_error=0.5, procedure="bonferroni")
    for _ in range(40):
        pw.learn(0.97, "a", "a")
    back = StreamPromise(max_error=0.5, grid=GRID, procedure="pointwise").restore(pw.dump())
    assert back.state() == pw.state() and back.threshold == pw.threshold


# ───────────────────────────── construction
def test_useful_error_defaults_and_refusals():
    assert _loop([REAL]).useful_error == 0.5 and _loop([REAL]).useful_budget.delta == 0.05
    assert _loop([REAL], useful_error=None).useful_budget is None
    assert _loop([REAL], delta_use=0.1).useful_budget.delta == 0.1
    rl = AdapterLoop(_Hand([REAL]), question="q", policy=_policy(), max_risk=0.05, secret=KEY, initial=L0,
                     monitor=None, _detector_alone=True, wait=True, clock=_clock, additive=False)
    assert rl.useful_error is None and rl.useful_budget is None
    with pytest.raises(ValueError, match="replacement mode"):
        AdapterLoop(_Hand([REAL]), question="q", max_risk=0.05, secret=KEY, initial=L0, useful_error=0.5, additive=False)
    with pytest.raises(ValueError, match="delta_use"):
        _loop([REAL], useful_error=None, delta_use=0.1)
    for bad in (0.0, 1.0, True, "half"):
        with pytest.raises(ValueError):
            _loop([REAL], useful_error=bad)
    with pytest.raises(ValueError):
        _loop([REAL], delta_use=1.5)


# ───────────────────────────── the gate on a noise layer, and its control
def test_a_noise_layer_is_certified_but_never_useful_and_is_discarded_with_useful_in_the_reason():
    loop = _loop([NOISE], policy=_policy(stale=3000, trial=3000))
    _run(loop, _xs(3, 4000))
    assert not _events(loop, "promote")
    disc = _events(loop, "discard")
    assert disc and "useful" in disc[0]["why"] and "failed" in disc[0]["why"]
    ver = loop._vs[1]
    assert math.isfinite(ver["promise"].threshold)    # the joint certificate passed the noise layer
    assert ver["useful"].state()["certified"] == 0 and ver["useful"].v == 1
    # the policy's view while it was in trial
    loop2 = _loop([NOISE], policy=_policy(stale=3000, trial=3000))
    for x in _xs(3, 4000):
        loop2.step(x, lambda x=x: x[0])
        c = loop2.candidate
        if c is not None and math.isfinite(loop2._joint(c)):
            st = loop2._state()
            g = loop2.policy.promote_checks(st, loop2._paired(c))["useful"]
            assert g["ok"] is False and g["beta"] == 0.5 and g["thr"] is None and "does not hold" in g["why"]
            assert st.candidate_info.certified and st.candidate_info.useful is False
            break
    else:
        pytest.fail("the noise candidate never got a joint certificate")


def test_without_the_usefulness_certificate_the_same_noise_layer_starts_serving():
    loop = _loop([NOISE], policy=_policy(stale=3000, trial=3000), useful_error=None)
    _run(loop, _xs(3, 4000))
    assert _events(loop, "promote")                    # the hole decision 24 found: the control has teeth


# ───────────────────────────── a real layer: promoted, composed threshold, event fields, records
@pytest.fixture(scope="module")
def real():
    loop = _loop([REAL])
    _run(loop, _xs(5, 5000))
    return loop


def test_a_real_layer_is_promoted_with_the_usefulness_fields_in_its_promote_event(real):
    e = _events(real, "promote")
    assert len(e) == 1 and e[0]["to"] == 1
    e = e[0]
    assert e["useful"] is True and e["useful_lw"] >= e["useful_level"] > 0 and e["useful_error"] == 0.5
    assert e["useful_v_index"] == 1 and e["useful_delta_v"] == pytest.approx(DeltaBudget(0.05).delta_at(1))
    assert e["joint_thr"] == e["thr_k"] and e["alone_k"] > 0 and 0 <= e["wrong_k"] < e["alone_k"] / 2
    ucert = [d for d in _events(real) if d["event"] == "useful_certified" and d["version"] == 1]
    assert ucert and ucert[0]["decision"] <= e["decision"]


def test_answers_are_at_the_composed_threshold_and_records_carry_the_usefulness_state(real):
    assert real._threshold(1) == real._composed(1) and math.isfinite(real._threshold(1))
    assert real._threshold(1) >= real._joint(1)
    recs = [d["stream"] for k, d in real.log if k == "loop"]
    assert all("useful" in r for r in recs)
    by1 = [r for r in recs if r["answered_by"] == 1]
    assert by1 and all(r["answer_thr"] >= real._joint(1) for r in by1)
    last = dict(recs[-1]["useful"])
    assert set(last) == {1} and last[1] == real._vs[1]["useful"].state()
    z = real.status()["zones"]["layers"][1]
    assert z["useful"]["ok"] is True and z["threshold"] == real._threshold(1)
    assert z["useful"]["ok_at_answer"] is True                  # the composed point is useful by construction
    vi = next(r for r in real._state().versions if r.version == 1)
    assert vi.useful is True and vi.useful_at_answer is True and vi.useful_thr == real._threshold(1)
    assert "delta_use_spent" in real.status() and "useful_error" not in json.dumps(real.status()["zones"]["layers"][0])


def test_delta_use_audit_from_the_records(real):
    idx = {}
    for k, d in real.log:
        if k == "loop":
            for v, st in d["stream"]["useful"]:
                if st["v_index"] is not None:
                    idx.setdefault((v, st["v_index"]), st["delta_v"])
    vs = sorted(i for _, i in idx)
    assert vs == list(range(1, len(vs) + 1))           # chronological and unique
    assert sum(idx.values()) <= real.useful_budget.delta + 1e-12
    assert real.useful_budget.spent == pytest.approx(sum(idx.values()))
    off = _loop([REAL], useful_error=None)
    _run(off, _xs(5, 5000))
    assert off.budget.v == real.budget.v and off.budget.spent == real.budget.spent   # the loop's δ untouched


def test_a_candidate_discarded_before_its_first_zone_label_spends_no_usefulness_index():
    loop = _loop([REAL])
    for x in _xs(5, 5000):
        loop.step(x, lambda x=x: x[0])
        if loop.candidate is not None:
            break
    c = loop.candidate
    assert c is not None and loop._vs[c]["useful"].v is None
    loop._discard(c, loop.n, "test")
    loop._unfreeze(loop.n)
    _run(loop, _xs(6, 50))
    assert loop.useful_budget.v == 0 and loop.useful_budget.spent == 0


def test_replay_is_exact_with_the_usefulness_certificate(real):
    rep = real.replay()
    assert rep["ok"], rep["mismatches"][:3]


def test_a_checkpoint_mid_trial_restores_the_usefulness_state_and_continues_exactly(tmp_path):
    xs = _xs(5, 3000)
    a = _loop([REAL])
    cut = None
    for i, x in enumerate(xs):
        a.step(x, lambda x=x: x[0])
        c = a.candidate
        if c is not None and a._vs[c]["useful"].v is not None and a._vs[c]["useful"].n >= 20:
            cut = i + 1
            break
    assert cut is not None
    a.save(tmp_path / "ck")
    b = AdapterLoop.load(tmp_path / "ck", _Hand([REAL]), secret=KEY, clock=_clock)
    c = a.candidate
    assert b._vs[c]["useful"].state() == a._vs[c]["useful"].state()
    assert b.useful_budget.v == a.useful_budget.v and b.useful_error == 0.5
    for x in xs[cut:]:
        a.step(x, lambda x=x: x[0])
        b.step(x, lambda x=x: x[0])
    assert a.log[-200:] == b.log[-200:] and a.status()["zones"] == b.status()["zones"]


def test_an_alarm_restarts_both_certificates_of_a_layer():
    loop = _loop([REAL])
    _run(loop, _xs(5, 5000))
    assert len(loop._layers) == 2 and math.isfinite(loop._threshold(1))
    loop._restart(1, loop.n, "test")
    assert math.isinf(loop._threshold(1)) and loop._vs[1]["useful"].v is None
    assert loop._vs[1]["useful"].state()["certified"] == 0
