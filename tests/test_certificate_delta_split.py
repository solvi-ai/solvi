"""The skill certificate spends delta over its restarts: the v-th certification of a process bets against
delta_v = 6·delta/(π²·v²), so every certificate it ever issues holds together with probability ≥ 1 − delta.
delta_split=False keeps the behaviour before 1.4 (the full delta at every restart: each certificate holds alone)."""
from __future__ import annotations

import math
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.skills import Certificate

# A phased stream at a 20% promise: 3% failures, then 50 uses at 50% every 500 (alarms and re-certifications).
# Recorded with the code before 1.4 (full delta at each restart): (outcome index, certified, alarm) and the final state.
OLD_EVENTS = [(11, True, False), (462, False, True), (488, False, True), (660, True, False), (975, False, True),
              (1147, True, False), (1484, False, True), (1499, False, True), (1523, True, False), (1983, False, True),
              (2217, True, False), (2979, False, True)]
OLD_STATE = {"certified": False, "log_wealth": -18.490917, "need_log_wealth": 2.302585, "remaining": 103,
             "outcomes": 3000, "failures": 239, "alarms": 7, "since_restart": 20, "loss_against": 0.0,
             "loss_outcomes": 0}


def stream(e, start=0, stop=3000):
    rng = random.Random(7)
    events = []
    for i in range(stop):
        ok, alone = rng.random() > (0.5 if i % 500 >= 450 else 0.03), rng.random() < 0.7
        if i < start:
            continue
        r = e.update(ok, alone=alone)
        if r["certified"] or r["alarm"]:
            events.append((i, r["certified"], r["alarm"]))
    return events


def test_the_delta_spent_over_any_number_of_restarts_stays_within_delta():
    cert = Certificate(promise=0.05, delta=0.1)
    assert cert.delta_split and cert.to_dict()["delta_split"] is True
    e = cert.process()
    assert e.v == 1 and e.delta_v == pytest.approx(0.6 / math.pi ** 2)
    for v in range(2, 5001):
        e.restart()
        assert e.v == v and e.delta_v == pytest.approx(0.6 / (math.pi ** 2 * v * v))
        if v % 500 == 0:
            assert e.delta_spent <= cert.delta
    assert e.delta_spent <= cert.delta and e.delta_spent > 0.0999          # Σ 6δ/(π²v²) → δ from below
    assert [cert.need_at(v) for v in (1, 2, 3)] == [61, 91, 108]
    old = Certificate(promise=0.05, delta=0.1, delta_split=False).process()
    for _ in range(4):
        old.restart()
    assert old.delta_v == 0.1 and old.delta_spent == pytest.approx(0.5)    # the old way: v·δ after v certifications
    with pytest.raises(ValueError):
        cert.delta_at(0)


def test_each_certification_records_its_v_and_delta_v():
    e = Certificate(promise=0.2, delta=0.1).process()
    events = stream(e)
    assert [i for i, c, _ in events if c][:3] == [13, 674, 1163]             # later certifications need more
    d = e.to_dict()
    assert d["v"] == 1 + d["alarms"] == 8
    assert d["delta_v"] == pytest.approx(0.6 / (math.pi ** 2 * 64))
    assert d["need_log_wealth"] == pytest.approx(math.log(1 / d["delta_v"]), abs=1e-6)
    assert d["delta_spent"] == pytest.approx(sum(0.6 / (math.pi ** 2 * v * v) for v in range(1, 9)))


def test_delta_split_false_reproduces_the_numbers_before_the_change():
    e = Certificate(promise=0.2, delta=0.1, delta_split=False).process()
    assert stream(e) == OLD_EVENTS
    d = e.to_dict()
    assert {k: d[k] for k in OLD_STATE} == OLD_STATE                        # the old keys, the old values
    assert d["v"] == 8 and d["delta_v"] == 0.1 and d["delta_spent"] == pytest.approx(0.8)


def test_a_certificate_pickled_before_the_change_loads_and_replays_as_it_did():
    e = Certificate(promise=0.2, delta=0.1, delta_split=False).process()
    stream(e, stop=1000)
    del e.cert.__dict__["delta_split"], e.__dict__["v"]                     # the attributes an old pickle lacks
    old = pickle.loads(pickle.dumps(e))
    assert not old.cert.delta_split and old.v == 1 and old.delta_v == 0.1
    rest = stream(old, start=1000)
    assert rest == [ev for ev in OLD_EVENTS if ev[0] >= 1000]
    d = old.to_dict()
    assert {k: d[k] for k in OLD_STATE} == OLD_STATE
