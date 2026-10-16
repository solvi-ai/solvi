"""After a drift flag no answer is reported "within its guarantee": with no slow path a person decides (or, with
on_drift="accept", System 1's answer stands marked out of its guarantee); build can attach a DriftMonitor."""
import random
import warnings
from typing import Literal

import pytest

import solvi
from solvi import Answer, Catalog, Question, System
from solvi.core.costs import Cost
from solvi.core.dispatch import Dispatcher
from solvi.core.guarantees.drift import DriftMonitor
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


def _guaranteed():
    cat = Catalog()

    @cat.rule("lab")
    def lab(x: float) -> str:
        return "a" if x > 0 else "b"
    s = System(cat, [Question("lab", "l", Answer.choice(["a", "b"]))])
    res = s.ask({"x": 1.0}, store=False)
    res["lab"].extra = {**(res["lab"].extra or {}), "guarantee": {"t": 0}}
    return s, res


def test_after_a_drift_flag_with_no_slow_path_a_person_decides():
    s, res = _guaranteed()
    d = Dispatcher(s, None, question="lab")
    action, why = d.decide(res, 0.5, Cost(), "DriftMonitor: act shift")
    assert action == "human"
    assert any("out of its guarantee" in w for w in why)
    assert not any("within its guarantee" in w for w in why)
    assert d.decide(res, 0.5, Cost(), None) == ("accept", ["System 1 answered within its guarantee"])


def test_on_drift_accept_keeps_system_1_answer_marked_out_of_guarantee():
    s, res = _guaranteed()
    d = Dispatcher(s, None, question="lab", on_drift="accept")
    action, why = d.decide(res, 0.5, Cost(), "DriftMonitor: act shift")
    assert action == "accept"
    assert any("out of its guarantee" in w for w in why)
    with pytest.raises(ValueError):
        Dispatcher(s, None, question="lab", on_drift="maybe")


def test_the_drift_flag_warns_once_when_no_slow_path_checks_system_1():
    s, _ = _guaranteed()

    class Flag:
        def observe(self, res, question=None):
            return {"drift": True, "why": ["planted"]}
    d = Dispatcher(s, None, question="lab", monitor=Flag())
    with pytest.warns(GuaranteeWarning, match="drift"):
        r = d.ask({"x": 1.0})
    assert r.by == "human"
    with warnings.catch_warnings():
        warnings.simplefilter("error", GuaranteeWarning)
        d.ask({"x": 1.0})                                       # latched: said once


def test_build_can_attach_a_drift_monitor():
    ex = _examples(600, 0, near=0.1, far=0.01)
    ds = solvi.build(Q, ex, catalog=_cat(), max_risk=0.05)
    assert ds.dispatcher.monitor is None                        # default "auto": too few calibration examples
    assert "Drift: none: monitor=\"auto\"" in ds.explain()     # and only explain() says so
    ds = solvi.build(Q, ex, catalog=_cat(), max_risk=0.05, monitor=True)
    mon = ds.dispatcher.monitor
    assert isinstance(mon, DriftMonitor) and len(mon.reference) > 0
    assert "drift" in ds.explain().lower()
    own = DriftMonitor(window=50)
    ds = solvi.build(Q, ex, catalog=_cat(), max_risk=0.05, monitor=own)
    assert ds.dispatcher.monitor is own


def test_build_monitor_auto_turns_on_only_with_enough_calibration_examples():
    from solvi.solutions import decisions
    big = _examples(4 * decisions.MONITOR_MIN_REFERENCE + 40, 1, near=0.1, far=0.01)
    ds = solvi.build(Q, big, catalog=_cat(), max_risk=0.05)    # "auto" is the default
    mon = ds.dispatcher.monitor
    assert isinstance(mon, DriftMonitor) and len(mon.reference) >= decisions.MONITOR_MIN_REFERENCE
    assert solvi.build(Q, big, catalog=_cat(), max_risk=0.05, monitor=None).dispatcher.monitor is None
    small = _examples(decisions.MONITOR_MIN_REFERENCE // 2, 1, near=0.1, far=0.01)
    ds = solvi.build(Q, small, catalog=_cat(), max_risk=0.1, monitor="auto")
    assert ds.dispatcher.monitor is None
    assert "monitor=\"auto\"" in ds.choices["monitor"] and str(decisions.MONITOR_MIN_REFERENCE) in ds.choices["monitor"]
    with pytest.raises(TypeError, match="auto"):
        solvi.build(Q, small, catalog=_cat(), max_risk=0.1, monitor="always")
