"""build(slow=) with split=(0, 1, 0): System 1 is calibrated on every example and the slow path answers uncalibrated;
the default split is unchanged."""
import random
from typing import Literal

import solvi
from solvi import Answer, Catalog, Question
from solvi.core.costs import Cost

Q = Question("lab", "l", Answer.choice(["a", "b"]))


def _cat():
    cat = Catalog()

    @cat.fn
    def margin(x: float) -> float:
        return abs(x - 0.5)

    @cat.rule("lab")
    def lab(x: float) -> Literal["a", "b"]:
        return "a" if x > 0.5 else "b"
    return cat


def _teacher(state):
    return "a" if state["x"] > 0.5 else "b"


def _examples(n, seed, near=0.3, far=0.03):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = rng.random()
        y = "a" if x > 0.5 else "b"
        if rng.random() < (near if abs(x - 0.5) < 0.1 else far):
            y = "b" if y == "a" else "a"
        out.append(({"x": x}, y))
    return out


def test_split_with_no_dispatcher_share_calibrates_system_1_on_every_example():
    ex = _examples(600, 0, near=0.1, far=0.01)
    ds = solvi.build(Q, ex, catalog=_cat(), max_risk=0.05, slow=_teacher, split=(0, 1, 0))
    sp = ds.choices["split"]
    assert sp["to calibrate the dispatcher"] == 0
    assert sp["to calibrate System 1's guarantee"] + sp["to choose the signal"] == 600
    assert ds.dispatcher.policy is None                      # the slow path is not calibrated: it answers as it is
    assert "think" in [ds.dispatcher.decide(ds.system.ask({"x": 0.5001}, store=False), 0.5, Cost(), None)[0]]
    assert "no examples" in ds.explain()


def test_the_default_split_with_a_slow_path_is_unchanged():
    ds = solvi.build(Q, _examples(600, 0), catalog=_cat(), max_risk=0.05, slow=_teacher, min_slice=1)
    sp = ds.choices["split"]
    assert sp["to calibrate the dispatcher"] == 200 and ds.dispatcher.policy is not None


def test_explain_says_the_uncalibrated_slow_path_answers():
    ex = _examples(600, 0, near=0.1, far=0.01)
    text = solvi.build(Q, ex, catalog=_cat(), max_risk=0.05, slow=_teacher, split=(0, 1, 0)).explain()
    assert "uncalibrated" in text and "Every input System 1 does not answer alone goes to a person" not in text
