"""solvi.experimental.laws: coarse outcomes (`coarse(key)`: a law predicts the observation's class) and a base predictor
that backs off from specific to general (`BackoffBase`). Synthetic worlds: the law is known."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Atom, BackoffBase, Buffer, Grammar, LawSet, OutcomeClass, Transition,
                                         coarse)


def grammar():
    return Grammar([Atom(f"front={f}", (lambda s, f=f: s["front"] == f), feature="front")
                    for f in ("tree", "stone", "water")]
                   + [Atom(f"wood={w}", (lambda s, w=w: s["wood"] == w), feature="wood") for w in range(3)], k=2)


def world(n, seed, group):
    """Chopping at a tree gives wood (a random amount); anything else gives nothing (with a random message id)."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        s = {"front": rng.choice(["tree", "stone", "water"]), "wood": rng.randrange(3)}
        seen = ({"kind": "wood", "amount": rng.randint(1, 9)} if s["front"] == "tree"
                else {"kind": "nothing", "msg": rng.randrange(10 ** 6)})
        out.append(Transition(s, "chop", seen, group=group))
    return out


def run(**kw):
    g = grammar()
    held = Buffer(g)
    for w in range(4):
        held.extend(world(60, 100 + w, f"w{w}"))
    laws = LawSet(g, **kw)
    for t in world(80, 1, "live"):
        laws.step(t, held)
    return laws, held


def test_raw_observations_teach_nothing_and_coarse_classes_teach_the_law():
    raw, _ = run()
    assert raw.laws == []                                    # every observation is new: no law passes its test
    laws, held = run(**coarse(lambda seen: seen["kind"]))
    s = {"front": "tree", "wood": 0}
    assert laws.predict(s, "chop") == OutcomeClass("wood")
    assert laws.agrees(laws.predict(s, "chop"), {"kind": "wood", "amount": 7})
    assert not laws.agrees(laws.predict(s, "chop"), {"kind": "nothing", "msg": 3})
    rep = laws.report(held)
    assert rep["actions"]["chop"]["outcomes"] == 2           # counted by class, not 200 distinct observations


def test_backoff_base_uses_the_most_specific_level_with_enough_rows():
    rows = world(300, 7, "train")
    base = BackoffBase([("front", "wood"), ("front",), ()], min_rows=5, outcome=lambda seen: seen["kind"]).fit(rows)
    e = base.explain({"front": "tree", "wood": 1}, "chop")
    assert e["value"] == OutcomeClass("wood") and e["level"] == 0 and e["share"] == 1.0
    few = BackoffBase([("front", "wood"), ("front",), ()], min_rows=40, outcome=lambda seen: seen["kind"]).fit(rows)
    e = few.explain({"front": "tree", "wood": 1}, "chop")
    assert e["level"] == 1 and e["value"] == OutcomeClass("wood")             # about 33 per cell: one level up
    e = few.explain({"front": "lava", "wood": 1}, "chop")
    assert e["level"] == 2 and e["value"] == OutcomeClass("nothing")          # an unseen front: the action's majority
    assert few({"front": "tree"}, "dig") is None                               # no row of the action at any level
    with pytest.raises(ValueError):
        BackoffBase(min_rows=0)


def test_a_coarse_set_with_a_backoff_base_predicts_from_the_start():
    base = BackoffBase([("front",), ()], outcome=lambda seen: seen["kind"]).fit(world(200, 9, "train"))
    laws, held = run(base=base, **coarse(lambda seen: seen["kind"]))
    assert laws.predict({"front": "stone", "wood": 2}, "chop") == OutcomeClass("nothing")
    ok = sum(laws.agrees(laws.predict(t.state, t.action), t.observed) for t in world(100, 5, "test"))
    assert ok == 100
