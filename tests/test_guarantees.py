"""Escalation thresholds with a guarantee (conformal risk control, learn-then-test) and conformal answer sets."""
import numpy as np
import pytest

from solvi.calibration import conformal_quantile, crc_threshold, ltt_threshold, set_scores
from test_decide import TEAMS, model, texts


def _stream(rng, n, lo=0.3, gain=0.6):
    """A decider whose confidence is informative but over-confident: P(right) = lo + gain·conf."""
    conf = rng.uniform(0.3, 1.0, n)
    wrong = rng.uniform(size=n) > lo + gain * conf
    return conf, wrong.astype(float)


def test_crc_keeps_the_risk_of_answering_wrongly_alone_on_new_inputs():
    rng = np.random.default_rng(0)
    risks = []
    for _ in range(300):
        c, w = _stream(rng, 300)
        t = crc_threshold(c, w, 0.10)
        c2, w2 = _stream(rng, 2000)
        risks.append(float(((c2 >= t) * w2).mean()))
    assert np.mean(risks) <= 0.10 + 0.005
    assert np.mean(risks) >= 0.07                        # not trivially conservative: it does answer alone


def test_crc_escalates_everything_when_the_risk_cannot_be_met():
    assert crc_threshold([0.9, 0.8, 0.7], [1, 1, 1], 0.10) == float("inf")
    assert crc_threshold([0.9] * 5, [0] * 5, 0.10) == float("inf")      # 5 examples cannot certify 10%: (0 + 1) / 6 > 0.1


def test_ltt_bounds_the_error_among_the_answered_and_is_stricter():
    rng = np.random.default_rng(1)
    c, w = _stream(rng, 3000, 0.2, 0.8)
    t = ltt_threshold(c, w, error=0.10)
    c2, w2 = _stream(rng, 20000, 0.2, 0.8)
    assert t < float("inf") and w2[c2 >= t].mean() <= 0.10
    assert ltt_threshold(c[:50], w[:50], error=0.10) == float("inf")


def test_ordinal_sets_are_intervals_and_the_quantile_needs_enough_examples():
    p = [0.05, 0.1, 0.4, 0.3, 0.15]
    s = set_scores(p, ordinal=True)
    for q in (0.0, 0.3, 0.5, 0.75, 0.9):
        keep = [i for i in range(5) if s[i] <= q]
        assert keep == list(range(min(keep), max(keep) + 1))
    assert conformal_quantile([0.1] * 5, 0.10) == float("inf")
    assert conformal_quantile(list(np.linspace(0, 1, 99)), 0.10) == pytest.approx(0.9, abs=0.02)


def _labelled(n0=0):
    ex = []
    for team in TEAMS:
        ex += [(t, team) for t in texts(team, 60, n0)]
    ex += [(f"The app crashed after the refund of order {i}.", "billing" if i % 2 else "technical")
           for i in range(n0, n0 + 60)]
    return ex


def test_act_guard_sets_a_threshold_and_every_decision_records_its_promise():
    m = model(noise=2.0)
    part = m.decision("team", "Which team?", "email", TEAMS)
    fp = part.fingerprint()
    info = part.act_guard(_labelled(), risk=0.10)
    assert info["signal"] == "confidence" and part.escalate_below == info["threshold"]
    assert info["risk"] <= 0.10 and 0 < info["answered"] < 1 and info["n"] == 240
    assert "P(answered alone and wrong) ≤ 0.1" in info["guarantee"]
    assert part.fingerprint() != fp
    d = part(email="The app crashed after the refund of order 7.")
    assert d.extra["guarantee"]["method"] == "crc" and d.extra["guarantee"]["risk"] == 0.10


def test_calibrate_for_ltt_lets_nothing_through_on_few_examples():
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    info = part.calibrate_for(_labelled()[:40], error=0.02, method="ltt")
    assert info["method"] == "ltt" and info["threshold"] == float("inf") and info["coverage"] == 0
    assert part(email="I was charged twice, please refund order 3.").escalate
    with pytest.raises(ValueError):
        part.calibrate_for(_labelled(), method="magic")


def test_conformal_sets_cover_the_right_answer_and_reach_the_escalation_message():
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    info = part.conformal(_labelled(), coverage=0.90)
    assert info["n"] == 240 and 1 <= info["mean_size"] <= 3
    test = _labelled(1000)
    hits = [y in part(email=t).extra["candidates"] for t, y in test]
    assert np.mean(hits) >= 0.85
    part.escalate_below = 0.99
    d = part(email="The app crashed after the refund of order 5.")
    assert d.escalate and "candidates at 90%" in d.escalate
    assert set(d.extra["candidates"]) <= set(TEAMS)


class PositionBiased:
    """Keyword logits plus a strong preference for whatever option is listed first."""
    model_id = "test/position-biased"

    def fingerprint(self):
        return "position-biased-1"

    def logits(self, items):
        out = []
        for it in items:
            low = it.text.lower()
            z = [1.0 * ("refund" in low and o == "billing") + 1.0 * ("crash" in low and o == "technical")
                 + (1.5 if i == 0 else 0.0) for i, o in enumerate(it.options)]
            out.append(np.stack([np.array(z), np.array(z) - 1.0], 1))
        return out


def _biased():
    from solvi.decide import DecideModel
    return DecideModel(PositionBiased(), meta={"format": "test", "temperature": 1.0})


def test_option_order_canonical_and_average_remove_the_callers_order_from_the_answer():
    m = _biased()
    text = "A refund please."
    given = [m.decision("t", "Team?", "email", o).decide(text).value
             for o in (["billing", "technical"], ["technical", "billing"])]
    assert given == ["billing", "technical"]                       # the listed order decides: a position bias
    for mode in ("canonical", "average"):
        got = {m.decision("t", "Team?", "email", o, option_order=mode).decide(text).value
               for o in (["billing", "technical"], ["technical", "billing"])}
        assert len(got) == 1, mode
    avg = m.decision("t", "Team?", "email", ["technical", "billing"], option_order="average")
    assert avg.decide(text).probs["billing"] == pytest.approx(1 / (1 + np.exp(-1.0)), abs=1e-9)  # the bias cancels
    assert avg.fingerprint() != m.decision("t", "Team?", "email", ["technical", "billing"]).fingerprint()


def test_a_near_tie_escalates_with_min_margin():
    m = model()
    part = m.decision("team", "Which team?", "email", TEAMS, min_margin=0.2)
    d = part(email="The app crashed after the refund of order 7.")
    assert d.escalate and "margin" in d.escalate and d.extra["margin"] < 0.2
    assert not part(email="I was charged twice, please refund order 3.").escalate


def test_act_guard_reports_how_much_must_escalate_when_the_model_is_often_wrong():
    part = model(noise=2.0).decision("team", "Which team?", "email", TEAMS)
    info = part.act_guard(_labelled(), risk=0.05)
    assert info["base_error"] > 0.05
    assert info["must_escalate_at_least"] == pytest.approx((info["base_error"] - 0.05) / 0.95)
    assert 1 - info["answered"] >= info["must_escalate_at_least"] - 1e-9
