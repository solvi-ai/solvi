"""A label with several right answers (solvi.core.AnyOf): calibration in `solvi.build`, `System.guarantee` and
`act_guard` counts an answer right when it is any of them; a plain label is the one right answer, as before."""
import random

import pytest

import solvi
from solvi import Answer, Catalog, Question
from solvi.core import AnyOf
from solvi.core.runtime import vhash

Q = Question("shut", "How is the dialog closed?", Answer.choice(["button", "escape", "leave"]))


def _cat():
    cat = Catalog()

    @cat.fn
    def margin(x: float) -> float:
        return abs(x - 0.5)

    @cat.rule("shut")
    def shut(x: float) -> str:
        return "escape" if x > 0.5 else "button"
    return cat


def _examples(n=400, seed=0, any_of=True):
    """Above 0.5 both the button and Escape close the dialog; the labeller wrote "button" there."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.random()
        y = (AnyOf("button", "escape") if any_of else "button") if x > 0.5 else "button"
        out.append(({"x": x}, y))
    return out


def test_build_counts_any_right_answer_as_right():
    ds = solvi.build(Q, _examples(), catalog=_cat(), max_risk=0.05)
    rep = ds.calibration["guarantee"]
    assert rep["answered"] > 0.9 and rep["error"] == 0.0
    plain = solvi.build(Q, _examples(any_of=False), catalog=_cat(), max_risk=0.05)
    assert plain.calibration["guarantee"]["answered"] < 0.6        # "escape" counted wrong: half the stream escalates


def test_a_custom_correct_is_asked_per_option():
    seen = []

    def correct(answer, label):
        seen.append(label)
        return answer == label
    ds = solvi.build(Q, _examples(200), catalog=_cat(), max_risk=0.05, correct=correct)
    assert ds.calibration["guarantee"]["error"] == 0.0
    assert not any(isinstance(x, AnyOf) for x in seen) and {"button", "escape"} <= set(seen)


def test_system_guarantee_reads_any_of():
    from solvi import System
    s = System(_cat(), [Q])
    rep = s.guarantee("shut", _examples(300), max_risk=0.05)
    assert rep["error"] == 0.0 and rep["answered"] > 0.9


def test_act_guard_on_a_decision_part_reads_any_of():
    from test_decide import TEAMS, FakeScorer
    from solvi.core.deciders import DecideModel
    m = DecideModel(FakeScorer(noise=0.5, version="v"), meta={"format": "test", "temperature": 1.0}, model_id="v")
    part = m.decision("team", "Which team?", "email", TEAMS)
    texts = ["I was charged twice on invoice 3, please refund.", "The app crashed after the refund of order 7."] * 60
    ds = part.decide(texts)
    wrong_first = [o for o in TEAMS if o != ds[0].value][0]
    plain = [(t, wrong_first) for t in texts]
    either = [(t, AnyOf(wrong_first, d.value)) for t, d in zip(texts, ds)]
    assert part.act_guard(plain, max_risk=0.1)["base_error"] > 0.4
    info = part.act_guard(either, max_risk=0.1)
    assert info["base_error"] == 0.0


def test_any_of_is_a_value():
    a = AnyOf("button", "escape")
    assert a == AnyOf(["button", "escape"]) and a.first == "button" and list(a) == ["button", "escape"]
    assert vhash(a) == vhash(AnyOf("button", "escape")) != vhash(AnyOf("escape", "button"))
    with pytest.raises(ValueError):
        AnyOf()
