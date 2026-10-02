"""A guarantee on any signal (solvi.guarantee): a calibrated threshold with a stated promise on a question's answer, a
computed fact or any scalar — the promise, the checks before it is made, ask time, the trace and replay."""
import math
import random

import numpy as np
import pytest

from solvi import Answer, Catalog, JSONLStorage, Question, System
from solvi.guarantee import GuaranteeWarning, calibrate


def _draw(rng, n, noise=0.15):
    """x in [0, 1]; the right answer is yes when x plus noise is above 0.5 — near 0.5 the answer is a coin flip."""
    out = []
    for _ in range(n):
        x = rng.random()
        out.append(({"x": x}, (x + rng.gauss(0, noise)) > 0.5))
    return out


def _system(storage=None):
    cat = Catalog()

    @cat.fn
    def score(x: float) -> float:
        return x

    @cat.fn
    def margin(x: float) -> float:
        return abs(x - 0.5)

    @cat.fn
    def side(x: float) -> str:
        return "high" if x > 0.5 else "low"
    s = System(cat, [Question("label", "label?", Answer.yes_no())], storage=storage)
    return s


def _head(seed=0, n=400):
    rng = random.Random(seed)
    s = _system()
    s.fit("label", _draw(rng, n), features=["score"])
    return s, rng


def _alone_and_wrong(s, examples):
    alone = wrong = 0
    for st, y in examples:
        r = s.ask(st, store=False)["label"]
        if r.status == "ok":
            alone += 1
            wrong += r.answer != ("yes" if y else "no")
    return alone, wrong


def test_crc_on_a_head_keeps_the_share_of_all_inputs_answered_alone_and_wrong():
    risks, answered = [], []
    for seed in range(8):
        s, rng = _head(seed)
        rep = s.guarantee("label", _draw(rng, 300), risk=0.05)
        assert rep["method"] == "crc" and rep["risk"] <= 0.05
        new = _draw(rng, 1500)
        a, w = _alone_and_wrong(s, new)
        risks.append(w / len(new))
        answered.append(a / len(new))
    assert np.mean(risks) <= 0.05 + 0.01
    assert np.mean(answered) > 0.3                      # it does answer alone


def test_ltt_bounds_the_error_among_the_answered_and_the_report_gives_both_shares():
    s, rng = _head(1)
    rep = s.guarantee("label", _draw(rng, 800), error=0.10)
    assert rep["method"] == "ltt" and math.isfinite(rep["threshold"])
    assert rep["error"] <= 0.10 and rep["risk"] == pytest.approx(rep["error"] * rep["answered"])
    assert "among the answers given alone" in rep["promise"] and "0.9" in rep["promise"]
    a, w = _alone_and_wrong(s, _draw(rng, 2000))
    assert a > 200 and w / a <= 0.10


def test_empirical_makes_no_promise_and_says_so():
    s, rng = _head(2)
    rep = s.guarantee("label", _draw(rng, 300), error=0.05, method="empirical")
    assert rep["promise"].startswith("none")


def test_a_signal_that_does_not_separate_right_from_wrong_is_refused_or_warned_loudly():
    s, rng = _head(3)
    cal = _draw(rng, 500)
    with pytest.raises(ValueError, match="does not separate"):
        s.guarantee("label", cal, risk=0.05, signal="score")        # x itself: the head errs on both sides of 0.5
    assert "label" not in s.guards
    with pytest.warns(GuaranteeWarning, match="does not separate"):
        rep = s.guarantee("label", cal, risk=0.05, signal="score", weak="warn")
    assert rep["warnings"] and "does not separate" in s.ask({"x": 0.9})["label"].extra["guarantee"]["warnings"][0]


def test_a_computed_fact_as_the_signal_is_computed_in_every_flow():
    s, rng = _head(4)
    rep = s.guarantee("label", _draw(rng, 600), error=0.10, signal="margin")
    assert rep["separation"]["auroc"] > 0.7
    assert "margin" in s.questions["label"].checkpoints
    r = s.ask({"x": 0.51})["label"]
    assert r.status == "abstain" and "margin" in r.why and "would have answered" in r.why
    assert s.ask({"x": 0.97})["label"].status == "ok"
    s.guarantee("label", False)
    assert "margin" not in s.questions["label"].checkpoints and s.ask({"x": 0.51})["label"].status == "ok"


def test_one_sided_answers_yes_alone_below_one_half_and_abstains_otherwise():
    s, rng = _head(5)
    rep = s.guarantee("label", _draw(rng, 600), risk=0.2, answer="yes")
    t = rep["threshold"]
    assert 0 < t < 1
    r = s.ask({"x": 0.99})["label"]
    assert r.status == "ok" and r.answer == "yes"
    r = s.ask({"x": 0.01})["label"]                        # the head says "no" for sure: one-sided, so it abstains
    assert r.status == "abstain" and "would have answered 'yes'" in r.why
    assert r.extra["guarantee"]["signal"] == "P('yes')"


def test_a_threshold_resting_on_too_few_examples_escalates_everything_with_the_reason():
    p = calibrate([0.99] + [0.5] * 30, [True] + [False] * 30, error=0.1, method="empirical", weak="warn")
    assert p.threshold == math.inf and "min_support" in p.report["why"]
    p = calibrate([0.9] * 8, [True] * 8, risk=0.05)
    assert p.threshold == math.inf and "cannot certify" in p.report["why"]


def test_thresholds_per_answer_hold_the_promise_inside_each_answer():
    s, rng = _head(6)
    rep = s.guarantee("label", _draw(rng, 800), risk=0.05, groups="answer", min_group=50, delta=None)
    assert set(rep["groups"]) >= {("yes",), ("no",)}
    r = s.ask({"x": 0.95})["label"]
    assert r.extra["guarantee"]["group"] == ["yes"] and r.extra["guarantee"]["applied"] == ["yes"]


def test_the_verdict_is_a_hashed_record_and_a_stored_decision_replays(tmp_path):
    rng = random.Random(7)
    s = _system(JSONLStorage(tmp_path / "d.jsonl"))
    s.fit("label", _draw(rng, 400), features=["score"])
    s.guarantee("label", _draw(rng, 600), error=0.1)
    res = s.ask({"x": 0.5})
    rec = next(r for r in res.trace.records if r.name == "guard:label")
    assert rec.kind == "guard" and rec.value is False and rec.extra["threshold"] > rec.extra["value"]
    assert res.trace.replay(s)["ok"]
    assert s.ask({"x": 0.97}).trace.replay(s)["ok"]
    assert s.storage.verify()["ok"] and s.storage.replay_all(s) == []
    s.guarantee("label", _draw(rng, 600), error=0.2)       # another guarantee: the stored decisions say so
    out = res.trace.replay(s)
    assert not out["ok"] and any("guarantee changed" in m[2] for m in out["mismatches"])


def test_a_recorded_verdict_that_does_not_follow_from_its_threshold_is_caught():
    s, rng = _head(8)
    s.guarantee("label", _draw(rng, 600), error=0.1)
    res = s.ask({"x": 0.5})
    rec = next(r for r in res.trace.records if r.name == "guard:label")
    rec.value = True
    out = res.trace.replay(s)
    assert not out["ok"] and any(m.kind == "integrity" for m in out["mismatches"])


def test_cross_fitting_scores_each_example_by_a_head_that_did_not_see_it():
    rng = random.Random(9)
    s = _system()
    ex = _draw(rng, 200)
    s.fit("label", ex, features=["score", "side"])
    rep = s.guarantee("label", ex, error=0.3, method="empirical", folds=5)
    assert rep["folds"] == 5 and any("approximate" in w for w in rep["warnings"])
    s2 = _system()
    s2.fit("label", ex)                                   # fit builds the same kind of head since 0.8: folds work too
    assert s2.guarantee("label", ex, error=0.3, method="empirical", folds=5)["folds"] == 5
    with pytest.raises(ValueError, match="head fitted with fit"):
        s3 = _system()
        s3.learn_rule("label", ex, facts=["score"])
        s3.guarantee("label", ex, error=0.3, folds=5)


def test_a_custom_judge_of_right_and_wrong():
    s, rng = _head(10)
    cal = [(st, st["x"] > 0.5) for st, _ in _draw(rng, 400)]
    rep = s.guarantee("label", cal, risk=0.05, correct=lambda result, right: (result.answer == "yes") == right)
    assert rep["base_error"] < 0.2


def test_the_promise_is_given_as_one_of_risk_or_error():
    s, rng = _head(11)
    with pytest.raises(ValueError, match="risk= .* or error="):
        s.guarantee("label", _draw(rng, 100))
    with pytest.raises(ValueError, match="takes risk="):
        s.guarantee("label", _draw(rng, 100), error=0.1, method="crc")
