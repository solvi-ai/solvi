"""The label check before calibration (solvi.core.calibration.label_issues — confident learning, Northcutt, Jiang &
Chuang 2021): a hand case, planted label flips found, its false positives on a model's confident mistakes, and the
report of act_guard / calibrate_for / System.guarantee — which never changes the calibration."""
import numpy as np
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.calibration import label_issues
from solvi.core.deciders import DecideModel

TEAMS = ["billing", "shipping"]


def test_hand_case():
    """Labelled billing: p(billing) 0.9, 0.8, 0.1 → t_billing = 0.6; labelled shipping: p(shipping) 0.7, 0.9 → t = 0.8.
    The third example (p_billing 0.1 < 0.6, p_shipping 0.9 >= 0.8) is flagged with "shipping" suggested."""
    probs = [{"billing": 0.9, "shipping": 0.1}, {"billing": 0.8, "shipping": 0.2}, {"billing": 0.1, "shipping": 0.9},
             {"billing": 0.3, "shipping": 0.7}, {"billing": 0.1, "shipping": 0.9}]
    labels = ["billing", "billing", "billing", "shipping", "shipping"]
    r = label_issues(probs, labels, ids=["a", "b", "c", "d", "e"])
    assert r["thresholds"] == {"billing": pytest.approx(0.6), "shipping": pytest.approx(0.8)}
    assert r["ids"] == ["c"] and r["suggested"] == {"c": "shipping"} and r["count"] == 1 and r["checked"] == 5
    # "d": p_shipping 0.7 < 0.8, but p_billing 0.3 < 0.6 — no other answer is confident: not flagged
    assert label_issues(probs, [None] * 5)["checked"] == 0
    with pytest.raises(ValueError):
        label_issues(probs, labels[:2])


def _world(n, flips, seed, acc_noise=0.0):
    """n examples of 3 answers; the model's probability of the true answer ~ Beta(8, 2) (mostly sure and right);
    `acc_noise` of them the model is confidently wrong; `flips` labels changed to another answer."""
    rng = np.random.default_rng(seed)
    K = ["a", "b", "c"]
    true = rng.integers(0, 3, n)
    probs = []
    for t in true:
        p_top = rng.beta(8, 2)
        top = t if rng.uniform() >= acc_noise else (t + 1) % 3
        rest = (1 - p_top) * rng.dirichlet([1, 1])
        p = np.empty(3)
        p[top] = p_top
        p[[j for j in range(3) if j != top]] = rest
        probs.append({k: float(x) for k, x in zip(K, p)})
    labels = [K[t] for t in true]
    flipped = sorted(rng.choice(n, flips, replace=False).tolist())
    for i in flipped:
        labels[i] = K[(true[i] + 1 + rng.integers(0, 2)) % 3]
    return probs, labels, flipped


@pytest.mark.parametrize("seed", range(5))
def test_planted_flips_are_found(seed):
    probs, labels, flipped = _world(400, 12, seed)                      # 3% flips, a good model
    r = label_issues(probs, labels)
    hit = set(r["ids"]) & set(flipped)
    assert len(hit) >= 8                                              # most of the 12 flips found (8–12 over seeds 0–9)
    assert r["count"] - len(hit) <= 3                                 # few others flagged (0–2 over seeds 0–9)


def test_false_positives_are_the_models_confident_mistakes():
    """No flipped label, a model confidently wrong on 10% of the examples: the check flags them — it cannot tell a
    wrong label from a confident model error (documented)."""
    probs, labels, _ = _world(400, 0, 3, acc_noise=0.10)
    r = label_issues(probs, labels)
    wrong = [i for i, (p, y) in enumerate(zip(probs, labels)) if max(p, key=p.get) != y]
    assert r["count"] >= 0.5 * len(wrong) and set(r["ids"]) <= set(wrong)


class Scorer:
    """A local decider sure of billing on "charged", of shipping otherwise."""
    model_id = "kw"

    def logits(self, items):
        return [np.array([3.0, 0.0]) if "charged" in it.text else np.array([0.0, 3.0]) for it in items]


def _examples(flip):
    ex = [(f"charged {i}", "billing") for i in range(40)] + [(f"parcel {i}", "shipping") for i in range(40)]
    for i in flip:
        ex[i] = (ex[i][0], TEAMS[1 - TEAMS.index(ex[i][1])])
    return ex


def test_act_guard_and_calibrate_for_report_and_change_nothing():
    flip = [3, 17, 55]
    part = DecideModel(Scorer(), model_id="kw").decision("team", "Which team?", "email", TEAMS)
    on = part.act_guard(_examples(flip), max_risk=0.10, label_check=True)
    assert on["label_issues"]["ids"] == flip and on["label_issues"]["suggested"][3] == "billing"
    assert any("3 of 80 calibration labels look wrong" in w for w in on["warnings"])
    off = part.act_guard(_examples(flip), max_risk=0.10)                    # the default: off
    assert "label_issues" not in off and {k: v for k, v in on.items() if k not in ("label_issues", "warnings")} == \
        {k: v for k, v in off.items() if k != "warnings"}
    clean = part.act_guard(_examples([]), max_risk=0.10, label_check=True)
    assert clean["label_issues"]["count"] == 0 and not any("look wrong" in w for w in clean.get("warnings", []))
    cf = part.calibrate_for(_examples(flip), max_error=0.10, method="ltt", label_check=True)
    assert cf["label_issues"]["ids"] == flip and cf["threshold"] == \
        part.calibrate_for(_examples(flip), max_error=0.10, method="ltt", label_check=False)["threshold"]


def test_system_guarantee_reports_label_issues():
    flip = [5, 60]
    part = DecideModel(Scorer(), model_id="kw").decision("team", "Which team?", "email", TEAMS)
    cat = Catalog()
    part.question(cat, "route")
    s = System(cat, [Question("route", "Route", Answer.choice(TEAMS))])
    ex = [({"email": t}, y) for t, y in _examples(flip)]
    rep = s.guarantee("route", ex, max_risk=0.10, label_check=True)
    assert rep["label_issues"]["ids"] == flip
    assert any("2 of 80 calibration labels look wrong" in w for w in rep["warnings"])
    rep2 = s.guarantee("route", ex, max_risk=0.10, label_check=False)
    assert "label_issues" not in rep2 and rep2["threshold"] == rep["threshold"]


def test_a_combination_reports_on_its_own_decisions():
    from solvi.core.deciders.combine import Vote
    flip = [2, 41]
    a = DecideModel(Scorer(), model_id="kw").decision("team", "Which team?", "email", TEAMS)
    b = DecideModel(Scorer(), model_id="kw2").decision("team", "Which team?", "email", TEAMS)
    v = Vote([a, b])
    info = v.act_guard(_examples(flip), max_risk=0.10, label_check=True)
    assert info["label_issues"]["ids"] == flip
    assert "label_issues" not in v.act_guard(_examples(flip), max_risk=0.10, label_check=False)
    assert v.calibrate_for(_examples(flip), max_error=0.1, label_check=True)["label_issues"]["ids"] == flip


def test_in_sample_fast_head_gets_a_warning_and_folds_does_not():
    import random

    rng = random.Random(0)
    cat = Catalog()

    @cat.fn
    def score(x: float) -> float:
        return x
    s = System(cat, [Question("label", "label?", Answer.yes_no())])
    ex = [({"x": x}, x > 0.5) for x in (rng.random() for _ in range(80))]
    s.fit("label", ex, features=["score"])
    msg = "in-sample probabilities"
    rep = s.guarantee("label", ex, max_risk=0.2, label_check=True)
    assert any(msg in w for w in rep["warnings"])
    rep = s.guarantee("label", ex, max_risk=0.2, label_check=True, folds=4)
    assert not any(msg in w for w in rep["warnings"])
    rep = s.guarantee("label", ex, max_risk=0.2)                              # no label check: no such line
    assert not any(msg in w for w in rep["warnings"])
