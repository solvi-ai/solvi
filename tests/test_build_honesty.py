"""build says when a promise is not kept: a dispatcher calibration that measures System 1 breaking the promise warns
(or refuses with strict=True)."""
import random
import warnings
from typing import Literal

import pytest

import solvi
from solvi import Answer, Catalog, Question
from solvi.core.guarantees.guarantee import GuaranteeWarning

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


def test_build_warns_when_the_dispatcher_measures_system_1_breaking_the_promise():
    with pytest.warns(GuaranteeWarning, match="break the promise"):
        ds = solvi.build(Q, _examples(600, 0), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=0)
    assert ds.calibration["dispatch"].get("why")
    assert "break the promise" in ds.explain()


def test_build_strict_refuses_a_measured_broken_promise():
    with pytest.raises(ValueError, match="break the promise"):
        solvi.build(Q, _examples(600, 0), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=0,
                    strict=True)


def test_a_kept_dispatcher_promise_does_not_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # the scenario as measured on the seeded shuffle (on the hash split's examples seed 1 measures a broken promise)
        ds = solvi.build(Q, _examples(600, 1), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=1,
                         split="shuffle")
    assert not ds.calibration["dispatch"].get("why")
    assert not [x for x in w if "break the promise" in str(x.message)]
