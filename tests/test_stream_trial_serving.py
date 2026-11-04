"""Certified trial serving retains a uniform, explicitly recorded candidate audit stream."""
import pytest

from test_stream_additive_b3 import GOOD, KEY, _Hand, _clock, _loop, _policy, _until, _xs
from solvi.experimental.stream import AdapterLoop


def prepared(mode="certified", *, band=True):
    lp = _loop([GOOD], policy=_policy(first=20, min_novel=10**9, stale=10**9), trial_serving=mode)
    _until(lp, _xs(0, 6000), lambda p: p.candidate is not None)
    lp.pause()
    for _ in range(40):
        lp.step(("a", 1.0, 0.8, 0.8), lambda: "a")
    # A conservative historical frozen boundary isolates the live-lambda cost band.
    # The live promise is genuinely certified; no threshold/evidence gate is mocked.
    if band:
        lp._route[0] = min(1.0, lp._threshold(0) + 0.2)
    score = (lp._threshold(0) + lp._route[0]) / 2
    assert lp._base_regime(0)
    if band:
        assert lp._threshold(0) < score < lp._route[0]
    return lp, ("a", score, 0.8, 0.8)


def record(lp):
    return [v["stream"] for k, v in lp.log if k == "loop"][-1]


def test_certified_base_serves_trial_band_without_unsampled_candidate_labels(monkeypatch):
    lp, x = prepared()
    c = lp.candidate
    fed = list(lp._vs[c]["fed"])
    samples = list(lp._vs[c]["samples"])
    boundary = dict(lp._route)
    monkeypatch.setattr(lp, "_coin", lambda n: 0.999999)
    out = lp.step(x, lambda: pytest.fail("unsampled certified answer must not ask oracle"))
    assert out["answer"] == "a" and not out["sampled"] and out["trial_zone"]
    assert lp._vs[c]["fed"] == fed and lp._vs[c]["samples"] == samples
    assert lp._route == boundary
    rec = record(lp)
    assert not rec["trial_sampled"] and 0 < rec["rho_trial"] == rec["rho_base"] < 1
    assert rec["answered_by"] == 0


@pytest.mark.parametrize("missing", [False, True])
def test_base_audit_feeds_candidate_and_missing_label_counts_as_loss(monkeypatch, missing):
    lp, x = prepared()
    c = lp.candidate
    before = len(lp._vs[c]["samples"])
    monkeypatch.setattr(lp, "_coin", lambda n: 0.0)
    lp.step(x, lambda: None if missing else "a")
    assert len(lp._vs[c]["samples"]) == before + 1
    assert lp._vs[c]["samples"][-1][2] is missing
    assert lp._vs[c]["fed"][-1] == lp.n
    assert record(lp)["trial_sampled"]


def test_incidental_escalation_label_is_not_a_candidate_sample(monkeypatch):
    lp, x = prepared()
    c = lp.candidate
    before = len(lp._vs[c]["samples"])
    monkeypatch.setattr(lp, "_coin", lambda n: 0.999999)
    x = ("a", 0.01, 0.8, 0.8)
    out = lp.step(x, lambda: "a")
    assert out["label"] == "a" and not out["sampled"]
    assert len(lp._vs[c]["samples"]) == before and not record(lp)["trial_sampled"]


def test_outside_zone_base_audit_updates_owner_but_not_candidate(monkeypatch):
    lp, x = prepared()
    c = lp.candidate
    candidate_fed, owner_fed = list(lp._vs[c]["fed"]), len(lp._vs[0]["fed"])
    monkeypatch.setattr(lp, "_coin", lambda n: 0.0)
    lp.step(("a", 1.0, 0.8, 0.8), lambda: "a")
    assert lp._vs[c]["fed"] == candidate_fed
    assert len(lp._vs[0]["fed"]) == owner_fed + 1
    assert record(lp)["base"] and not record(lp)["trial_sampled"]
    assert record(lp)["rho_trial"] is None


def test_restarted_base_cannot_serve_and_unsampled_trial_labels_still_are_not_fed(monkeypatch):
    lp, x = prepared()
    c = lp.candidate
    before = len(lp._vs[c]["samples"])
    lp._restart(0, lp.n)
    assert not lp._base_regime(0)
    monkeypatch.setattr(lp, "_coin", lambda n: 0.999999)
    out = lp.step(x, lambda: "a")
    assert out["label"] == "a" and record(lp)["answered_by"] is None
    assert not record(lp)["base"] and not record(lp)["trial_sampled"]
    assert len(lp._vs[c]["samples"]) == before


def test_attached_dispatcher_uses_same_serving_and_candidate_audit_gate(tmp_path, monkeypatch):
    from solvi.core.store import JSONLStorage
    from test_stream_attach import KEY as AKEY, Learner, _disp, _emails, _policy as attached_policy
    disp = _disp(storage=JSONLStorage(tmp_path / "dispatch.jsonl"))
    lp = AdapterLoop.attach(disp, Learner(), question="team", max_risk=0.05, secret=AKEY,
                            initial=disp.system.holder.ver,
                            policy=attached_policy(min_novel=10**9, stale=10**9, mature_patience=None),
                            wait=True, clock=lambda: 1000.0, additive=True, trial_serving="certified")
    for state in _emails(1200):
        disp.ask(state)
        if lp.candidate is not None:
            break
    assert lp.candidate is not None
    lp.pause()
    good = next(state for state in _emails(100, start=2000)
                if lp._vs[0]["model"].score(state)[1] == 0.95)
    for _ in range(40):
        disp.ask(good)
    lp._route[0] = 1.0  # conservative frozen bound, as in the direct-step fixture
    c = lp.candidate
    before = len(lp._vs[c]["samples"])
    monkeypatch.setattr(lp, "_coin", lambda n: 0.999999)
    out = disp.ask(good)
    assert out.by == "s1" and out.learning["alone"] and record(lp)["trial_zone"]
    assert len(lp._vs[c]["samples"]) == before
    assert not record(lp)["trial_sampled"]
    monkeypatch.setattr(lp, "_coin", lambda n: 0.0)
    disp.ask(good)
    assert len(lp._vs[c]["samples"]) == before + 1 and record(lp)["trial_sampled"]
    assert disp.replay_all() == []


def test_default_trial_still_escalates_and_reads_complete_zone(monkeypatch):
    lp, x = prepared("escalate")
    c = lp.candidate
    before = len(lp._vs[c]["samples"])
    monkeypatch.setattr(lp, "_coin", lambda n: 0.999999)
    out = lp.step(x, lambda: "a")
    assert out["label"] == "a" and len(lp._vs[c]["samples"]) == before + 1
    assert record(lp)["answered_by"] is None


def test_trial_audit_checkpoint_and_replay(tmp_path):
    a, x = prepared(band=False)
    a.save(tmp_path / "ck")
    b = AdapterLoop.load(tmp_path / "ck", _Hand(list(a.live.models)), secret=KEY, clock=_clock)
    assert b.trial_serving == "certified"
    for z in [x] * 30 + _xs(7, 30):
        a.step(z, lambda z=z: z[0])
        b.step(z, lambda z=z: z[0])
    assert a.log == b.log and a.status() == b.status()
    assert b.replay()["ok"]


@pytest.mark.parametrize("value", [None, True, "yes", [], {}])
def test_invalid_trial_serving_is_refused(value):
    with pytest.raises(ValueError, match="trial_serving"):
        _loop([GOOD], trial_serving=value)


def test_trial_serving_requires_additive_stream():
    with pytest.raises(ValueError, match="additive stream"):
        AdapterLoop(_Hand([GOOD]), question="q", additive=False, trial_serving="certified")
