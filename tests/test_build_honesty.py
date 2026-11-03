"""A measured dispatcher violation fails closed by default; explicit diagnostic builds cannot answer alone."""
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
        ds = solvi.build(Q, _examples(600, 0), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=0,
                         strict=False)
    assert ds.calibration["dispatch"].get("why")
    assert "break the promise" in ds.explain()
    assert ds.calibration["dispatch"]["certified"] is False
    assert ds.calibration["dispatch"]["fallback"] == "human"
    assert "UNCERTIFIED" in ds.explain()
    for x in (.01, .49, .51, .99):
        assert ds.ask({"x": x}).by == "human"


def test_build_strict_refuses_a_measured_broken_promise():
    with pytest.raises(ValueError, match="break the promise"):
        solvi.build(Q, _examples(600, 0), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=0,
                    strict=True)
    with pytest.raises(ValueError, match="break the promise"):
        solvi.build(Q, _examples(600, 0), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=0)


def test_a_kept_dispatcher_promise_does_not_warn():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # the scenario as measured on the seeded shuffle (on the hash split's examples seed 1 measures a broken promise)
        ds = solvi.build(Q, _examples(600, 1), catalog=_cat(), max_risk=0.02, slow=_teacher, min_slice=1, seed=1,
                         split="shuffle")
    assert not ds.calibration["dispatch"].get("why")
    assert not [x for x in w if "break the promise" in str(x.message)]


def test_diagnostic_latch_overrides_a_forced_hard_check_and_report_mutation():
    cat = _cat()

    @cat.check(hard=True, then={"lab": "b"})
    def normal_range(x):
        return x >= -10

    with pytest.warns(GuaranteeWarning, match="automatic answers disabled"):
        ds = solvi.build(Q, _examples(600, 0), catalog=cat, max_risk=.02, slow=_teacher,
                         min_slice=1, seed=0, strict=False)
    config = ds.dispatcher.config()
    ds.calibration["dispatch"]["certified"] = True  # a mutable report cannot clear the private routing latch
    result = ds.ask({"x": -20.})
    assert result.s1["lab"].status == "forced"
    assert result.by == "human" and result.answer is None and result.action == "human"
    assert ds.dispatcher.config() == config
    assert ds.replay(result)["ok"]


def test_diagnostic_build_disables_every_actual_openset_threshold(monkeypatch):
    import math

    from solvi.core.guarantees.openset import OpenSetGate
    from solvi.solutions import decisions

    gate = OpenSetGate.calibrate([.2, .8] * 50, [True] * 100, [.01] * 100, max_error=.02, track=())
    gate.thresholds = dict.fromkeys(gate.thresholds, .3)  # deterministic over-permissive fixture, caught by own dispatch
    gate.cut = -1.  # no unrelated distribution-shift alarm can hide the deliberately permissive threshold
    gate.h = 1e100

    def fake_open_set(system, question, examples, outside, signal, correct, level, delta, seed):
        return gate, {"known": len(examples), "novel": len(outside), "given": len(outside)}

    monkeypatch.setattr(decisions, "_open_set_real", fake_open_set)
    catalog = _cat()

    @catalog.constraint
    def only_a(lab):
        return lab == "a"  # an actual non-openset handover slice keeps dispatcher calibration in this build path

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = solvi.build(Q, _examples(600, 0, near=.3, far=.3), catalog=catalog, max_risk=.02, slow=_teacher,
                         min_slice=1, seed=0, strict=False, signal="margin", novel=[{"x": -.1}] * 100)
    assert ds.calibration.get("dispatch", {}).get("certified") is False, ds.calibration
    assert ds.gate is gate and all(math.isinf(x) for x in gate.thresholds.values())
    result = ds.ask({"x": .99})
    assert result.by == "human" and result.answer is None
    assert ds.dispatcher._diagnostic_human_only
    assert ds.replay(result)["ok"]
