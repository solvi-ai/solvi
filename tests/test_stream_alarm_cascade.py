"""Explicit routing-dependency invalidation; legacy layer-local behavior stays covered in B2."""
import math

import pytest

from test_stream_additive_b3 import (GOOD, GOOD2, GW, _Hand, _events, _loop,
                                     _resume_check, _until, _xs)

from solvi.experimental.stream import AdapterLoop, RiskDetector


def test_cascade_checkpoint_and_rollback_keep_invalidated_layers_uncertified(tmp_path):
    def make():
        return _loop([GOOD], max_layers=2, detector=RiskDetector(0.05, arl=50), alarm_cascade="downstream")

    def shift(lp, x):
        route = lp._route.get(0)
        return ("b" if x[0] == "a" else "a") if len(lp._layers) == 2 and route is not None and x[1] >= route else x[0]

    a, b = _resume_check(tmp_path, make, _xs(6, 9000),
                         lambda lp: lp.left > 0 and lp._left_of.get(1, 0) > 0,
                         oracle=shift, after=100)
    assert a.alarm_cascade == b.alarm_cascade == "downstream"
    first = _events(b, "restart")[0]["decision"]
    assert [e["version"] for e in _events(b, "restart") if e["decision"] == first] == [0, 1]
    b.rollback(1)
    assert b._layers == [0] and not b._route
    assert b.replay()["ok"]


def test_upstream_alarm_discards_dependent_trial_and_spends_no_restart_budget():
    lp = _loop([GW, GOOD2], alarm_cascade="downstream", max_layers=3)
    _until(lp, _xs(1, 12000), lambda p: len(p._layers) == 2 and p.candidate is not None)
    c = lp.candidate
    routing = dict(lp._route)
    budget = lp._vs[0]["promise"].budget.v
    events = lp._on_alarm_layers(lp.n, {}, {"why": ["identity changed"]})
    assert lp.candidate is None and lp._vs[c]["state"] == "discarded"
    assert set(lp._route) == {lp._layers[0]}  # only the candidate's frozen boundary is removed
    assert lp._route[0] == routing[0]
    assert [e["version"] for e in events if e["event"] == "restart"] == [0, 1]
    assert all(math.isinf(lp._threshold(v)) for v in lp._layers)
    assert lp.left == lp.probation and lp._left_of[1] == lp.probation
    assert lp._vs[0]["promise"].budget.v == budget
    assert any(e["event"] == "discard" and "routing dependency" in e["why"] for e in events)


@pytest.mark.parametrize("value", [None, True, "all", [], {}])
def test_invalid_cascade_declarations_are_rejected(value):
    with pytest.raises(ValueError, match="alarm_cascade"):
        _loop([GOOD], alarm_cascade=value)


def test_cascade_requires_additive_dependencies():
    with pytest.raises(ValueError, match="additive routing"):
        AdapterLoop(_Hand([GOOD]), question="q", additive=False, alarm_cascade="downstream")
