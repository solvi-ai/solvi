"""Certificate(alternatives=): the alarm as a mean of Bernoulli likelihood-ratio CUSUMs against r·promise."""
from __future__ import annotations

import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.skills import Certificate


def test_steps_are_the_likelihood_ratios():
    c = Certificate(promise=0.1, delta=0.1, alternatives=(2.0,), alarm=1e9)
    e = c.process()
    e.update(False, alone=True)                     # one failure: ratio r = 2
    assert e.detector == pytest.approx(2.0)
    e.update(True, alone=True)                      # an ok: (1 - 0.2) / (1 - 0.1)
    assert e.detector == pytest.approx(2.0 * 0.8 / 0.9)
    for _ in range(50):
        e.update(True, alone=True)                  # falls below 1 and restarts from 1 (CUSUM), never below
    e.update(False, alone=True)
    assert e.detector == pytest.approx(2.0)


def test_mixture_is_the_mean_over_alternatives():
    e = Certificate(promise=0.1, alternatives=(1.5, 3.0), alarm=1e9).process()
    e.update(False, alone=True)
    assert e.detector == pytest.approx((1.5 + 3.0) / 2)


def test_alarm_restarts_and_resets_the_detector():
    e = Certificate(promise=0.1, delta=0.1, alternatives=(2.0,), alarm=7.99).process()
    for _ in range(e.cert.need):
        e.update(True, alone=True)
    assert e.certified
    evs = [e.update(False, alone=True)["alarm"] for _ in range(3)]   # 2, 4, 8 → alarm at the third failure
    assert evs == [False, False, True] and not e.certified and e.alarms == 1 and e.detector == 1.0


def test_validation_and_default_unchanged():
    with pytest.raises(ValueError):
        Certificate(promise=0.1, alternatives=(1.0,))
    with pytest.raises(ValueError):
        Certificate(promise=0.4, alternatives=(3.0,))
    with pytest.raises(ValueError):
        Certificate(promise=0.1, alternatives=())
    c = Certificate(promise=0.05)
    assert c.alternatives is None and "alternatives" not in c.to_dict()
    e = c.process()
    for _ in range(50):
        e.update(True, alone=True)
    assert e.update(False, alone=True)["alarm"]     # the default bet: one failure alone at 5 % is still an alarm
    assert Certificate(promise=0.1, alternatives=(2, 3)).to_dict()["alternatives"] == [2.0, 3.0]


def test_not_alone_outcomes_do_not_move_the_detector():
    e = Certificate(promise=0.1, alternatives=(2.0,), alarm=1e9).process()
    e.update(False, alone=False)
    assert e.detector == 1.0


def test_false_alarms_at_the_promise_respect_the_union_bound():
    # at a failure rate equal to the promise, P(an alarm within L outcomes) <= L / alarm (each CUSUM start a martingale)
    rng = random.Random("test-cert-alt")
    p, L, A, n = 0.1, 200, 2000.0, 300
    hits = 0
    for _ in range(n):
        e = Certificate(promise=p, alternatives=(1.5, 2.0, 3.0), alarm=A).process()
        hits += any(e.update(rng.random() >= p, alone=True)["alarm"] for _ in range(L))
    assert hits / n <= L / A + 3 * math.sqrt(0.1 * 0.9 / n)


def test_a_doubled_rate_is_caught():
    rng = random.Random("test-cert-alt-break")
    p, caught = 0.1, 0
    for _ in range(50):
        e = Certificate(promise=p, alternatives=(1.5, 2.0, 3.0), alarm=30.0).process()
        caught += any(e.update(rng.random() >= 2 * p, alone=True)["alarm"] for _ in range(1500))
    assert caught >= 48


def test_default_level_with_alternatives_is_five_over_delta():
    assert Certificate(promise=0.1, delta=0.1, alternatives=(1.5, 2.0)).alarm == pytest.approx(50.0)
    assert Certificate(promise=0.1, delta=0.1).alarm == pytest.approx(10.0)
