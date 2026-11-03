"""Known-answer checks of multiple-testing rules, without empirical FDR claims."""
import math
from dataclasses import FrozenInstanceError

import pytest

from solvi.stat._multiple import ELOND, e_bh


def test_ebh_step_up_can_pass_after_first_rank_fails():
    # m=4, q=.5: thresholds 8,4,8/3,2. Rank one fails, rank two passes.
    assert e_bh([0, 4, 1, 4], q=0.5) == [1, 3]
    assert e_bh([2, 2, 2, 2], q=0.5) == [0, 1, 2, 3]
    assert e_bh([1, 0, 0], q=0.5) == []
    assert e_bh([], q=0.5) == []


def test_ebh_boundary_infinity_and_original_indices():
    assert e_bh([0, math.inf, 20], q=0.1) == [1, 2]
    assert e_bh([2], q=0.5) == [0]
    assert e_bh([math.nextafter(2, 0)], q=0.5) == []
    values = [0, 4, 4, 0]
    assert e_bh(iter(values), q=0.5) == [1, 2]
    assert values == [0, 4, 4, 0]


@pytest.mark.parametrize("q", [0, 1, -1, math.inf, -math.inf, math.nan])
def test_invalid_q_even_for_empty_family(q):
    with pytest.raises(ValueError):
        e_bh([], q=q)
    with pytest.raises(ValueError):
        ELOND(q=q)


@pytest.mark.parametrize("value", [-1, -math.inf, math.nan])
def test_invalid_evalues_are_rejected(value):
    with pytest.raises(ValueError):
        e_bh([100, value])
    rule = ELOND()
    allocation = rule.reserve()
    before = vars(rule).copy()
    with pytest.raises(ValueError):
        rule.finish(allocation.index, value)
    assert vars(rule) == before
    assert rule.finish(allocation.index, math.inf)


def test_uniform_known_sequence_and_no_refund():
    rule = ELOND(0.5, schedule="uniform", horizon=4)
    expected = [(0.125, 8, True), (0.25, 0, False), (0.25, 4, True), (0.375, 0, False)]
    for index, (alpha, evidence, rejected) in enumerate(expected, 1):
        allocation = rule.reserve()
        assert allocation.index == index and allocation.alpha == alpha
        assert allocation.threshold == 1 / alpha
        assert rule.finish(index, evidence) is rejected
    assert (rule.index, rule.finished, rule.discoveries, rule.pending) == (4, 4, 2, None)
    before = vars(rule).copy()
    with pytest.raises(RuntimeError, match="exhausted"):
        rule.reserve()
    assert vars(rule) == before


def test_basel_allocation_and_boundary():
    rule = ELOND(0.1)
    first = rule.reserve()
    assert first.alpha == pytest.approx(0.1 * 6 / math.pi ** 2)
    assert not rule.finish(1, math.nextafter(first.threshold, 0))
    second = rule.reserve()
    assert second.alpha == pytest.approx(first.alpha / 4)
    assert rule.finish(2, second.threshold)
    third = rule.reserve()
    assert third.alpha == pytest.approx(first.alpha * 2 / 9)


def test_fail_closed_reservations_cannot_use_future_discoveries():
    rule = ELOND(0.1, schedule="uniform", horizon=3)
    with pytest.raises(ValueError):
        rule.finish(1, math.inf)
    first = rule.reserve()
    before = vars(rule).copy()
    with pytest.raises(RuntimeError):
        rule.reserve()
    for bad_index in (0, 2, -1, True, 1.0):
        with pytest.raises(ValueError):
            rule.finish(bad_index, math.inf)
        assert vars(rule) == before
    with pytest.raises(FrozenInstanceError):
        first.alpha = 1
    assert rule.finish(1, first.threshold)
    with pytest.raises(ValueError):
        rule.finish(1, math.inf)
    second = rule.reserve()
    assert second.alpha == first.alpha * 2
    assert not rule.finish(2, 0)
    assert rule.reserve().alpha == second.alpha


@pytest.mark.parametrize("kwargs", [
    {"schedule": "other"}, {"schedule": "uniform"},
    {"schedule": "uniform", "horizon": 0}, {"schedule": "uniform", "horizon": True},
    {"schedule": "uniform", "horizon": 2.0}, {"horizon": 5},
])
def test_schedule_contract(kwargs):
    with pytest.raises(ValueError):
        ELOND(**kwargs)
