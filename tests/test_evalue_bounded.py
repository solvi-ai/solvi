"""A betting e-process for a bounded numeric outcome x in [0, 1] against "the mean is at most m" (a
claim e-value reduces x to {0, 1}); used by Certificate (EProcess.update(..., loss=)) and ShipWatch(outcome="loss")."""
from __future__ import annotations

import random
import warnings

import pytest

from solvi.experimental._evalue import MeanBound

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.core.knowledge import KnowledgeStore
    from solvi.experimental.changes import Change, Pipeline, ShipWatch, Static
    from solvi.experimental.skills import Certificate


def test_validation():
    for m in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            MeanBound(m)
    t = MeanBound(0.3)
    for x in (-0.01, 1.01, float("nan")):
        with pytest.raises(ValueError):
            t.update(x)


def test_ville_at_the_bound_with_spread_values():
    rng = random.Random(4)
    hits = 0
    for _ in range(300):
        t = MeanBound(0.3)
        for _ in range(400):
            t.update(rng.uniform(0.0, 0.6))          # mean exactly 0.3, not {0, 1}
            if t.evidence >= t.level(0.1):
                hits += 1
                break
    assert hits / 300 <= 0.1 + 0.035


def test_power_uses_the_magnitude_not_only_the_sign():
    rng = random.Random(5)
    t = MeanBound(0.3)
    n = 0
    while t.evidence < t.level(0.05) and n < 2000:
        t.update(min(1.0, max(0.0, rng.gauss(0.45, 0.05))))
        n += 1
    assert n < 200                                 # a mean 0.45 vs bound 0.3, low spread: found fast
    clean = MeanBound(0.3)
    for _ in range(500):
        clean.update(rng.uniform(0.0, 0.4))         # mean 0.2 < bound: never bets up
    assert clean.evidence <= 0.0


def test_the_certificate_reads_a_numeric_loss():
    e = Certificate(0.2, 0.1).process()
    for _ in range(100):
        e.update(True, loss=0.5)
    assert e.loss_refuted and e.to_dict()["loss_against"] > 0
    f = Certificate(0.2, 0.1).process()
    for _ in range(100):
        f.update(True)
    assert not f.loss_refuted and f.loss_test.n == 0


def test_shipwatch_on_a_numeric_loss_reverts_a_harmed_slice():
    pipe = Pipeline([Static()], None, store=KnowledgeStore(), approval={"fact": "gate"})
    v = pipe.propose(Change("person", payload={"kind": "fact", "body": {"s": "x", "r": "y", "o": 1}}, proposer="a"))
    w = ShipWatch(pipe, v.version, promise=0.2, slices=["main", "rare"], outcome="loss")
    rng = random.Random(6)
    ev = None
    for _ in range(5000):
        s = "rare" if rng.random() < 0.05 else "main"
        ev = w.observe(rng.uniform(0, 0.3) if s == "main" else rng.uniform(0.3, 0.7), slice=s)
        if ev:
            break
    assert ev["event"] == "reverted" and ev["slice"] == "rare" and pipe.current == 0
    with pytest.raises(ValueError):
        ShipWatch(pipe, 0, promise=0.2, slices=["a"], outcome="score")
