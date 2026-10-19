"""solvi.experimental.laws (experimental, opt-in): `Buffer(dedup=True)` — a repeated situation is one row, not many.

Synthetic worlds only: every law is known, so a false accept is known exactly."""
from __future__ import annotations

import itertools
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition

# ───────────────────────────────────────────── dedup: a random walk over few objects
F = 8


class _Bit:
    """s[i], picklable."""
    def __init__(self, i):
        self.i = i

    def __call__(self, s):
        return s[self.i]


def walk_grammar():
    return Grammar([Atom(f"a{i}", _Bit(i), feature=f"a{i}") for i in range(F)], k=2)


def walk_world(seed, groups=10, objects=3, rows=20, law=False, dedup=False):
    """Per group (a world) 3 objects with persistent attributes (8 binary atoms at 0.3) and a persistent outcome; 20
    rows, each a visit to a random one of them (a random walk revisits). No law: the outcome is a coin per object,
    independent of the atoms. law=True: pass exactly where a0 & a1."""
    rng = random.Random(seed)
    g = walk_grammar()
    b = Buffer(g, dedup=dedup)
    for gi in range(groups):
        objs = []
        for _ in range(objects):
            s = tuple(rng.random() < 0.3 for _ in range(F))
            out = ("pass" if (s[0] and s[1]) else "block") if law else ("pass" if rng.random() < 0.5 else "block")
            objs.append((s, out))
        for r in range(rows):
            s, out = rng.choice(objs)
            b.add(Transition(s, "go", out, group=f"g{gi}", id=f"{gi}.{r}"))
    return g, b


def walk_candidates():
    names = [f"a{i}" for i in range(F)]
    conds = [c for r in (1, 2) for c in itertools.combinations(names, r)]
    return [Candidate("go", c, e) for e in ("pass", "block") for c in conds]


def walk_accepts(seeds, law, dedup, groups=10):
    n = 0
    for seed in range(seeds):
        g, b = walk_world(seed, groups=groups, law=law, dedup=dedup)
        got = LawSet(g).test(walk_candidates(), b, exclude="g0")
        if got:
            n += (got[0].condition == ("a0", "a1") and got[0].effect == "pass") if law else 1
    return n


def test_dedup_keeps_one_row_per_situation_and_group():
    g = walk_grammar()
    b = Buffer(g, dedup=True)
    s = tuple([True] + [False] * (F - 1))
    assert b.add(Transition(s, "go", "pass", group="g0")) is True
    assert b.add(Transition(s, "go", "pass", group="g0")) is False          # the same situation again
    assert b.add(Transition(s, "go", "block", group="g0")) is True          # another observation
    assert b.add(Transition(s, "go", "pass", group="g1")) is True           # another group
    assert b.add(Transition(s, "stay", "pass", group="g0")) is True         # another action
    assert len(b) == 4 and b.duplicates == 1
    nd = Buffer(g)
    assert nd.add(Transition(s, "go", "pass", group="g0")) and nd.add(Transition(s, "go", "pass", group="g0"))
    assert len(nd) == 2 and nd.duplicates == 0 and nd.situation(Transition(s, "go", "pass")) is None


def test_dedup_by_a_key_and_with_a_window():
    g = walk_grammar()
    b = Buffer(g, dedup=lambda t: t.state[0], window=2)                     # e.g. an object id
    s1, s2 = tuple([True] * F), tuple([True] + [False] * (F - 1))
    assert b.add(Transition(s1, "go", "pass", group=0)) and not b.add(Transition(s2, "go", "block", group=0))
    assert b.add(Transition(tuple([False] * F), "go", "pass", group=0))
    assert b.add(Transition(tuple([False] * F), "go", "pass", group=1))     # s1 leaves the window
    assert b.add(Transition(s1, "go", "pass", group=0))                     # its situation left with it
    with pytest.raises(ValueError):
        Buffer(g, dedup="yes")
    b = Buffer(g, dedup=True)
    b.add(Transition(s1, "go", "pass", group=0))
    b2 = pickle.loads(pickle.dumps(b))
    assert b2.dedup is True and not b2.add(Transition(s1, "go", "pass", group=0)) and b2.duplicates == 1


@pytest.mark.parametrize("dedup, expect", [(False, 872), (True, 0)])
def test_a_random_walk_over_few_objects_measured(dedup, expect):
    """No law exists (the outcome is a coin per object). Rows of a random walk revisit 3 objects per world: by default
    a condition matching a few objects gets their repeated rows as support (about 20–26 rows) and passes in 872 of
    1,000 buffers; with dedup each object counts once per world: 0 — within delta = 0.1."""
    got = walk_accepts(1000, False, dedup)
    assert got == expect and (not dedup or got <= 0.1 * 1000)


def test_dedup_still_finds_a_real_law():
    """pass exactly where a0 & a1, 20 worlds: the right law first in 591 of 1,000 buffers with dedup (473 without:
    there a false law passes first more often)."""
    assert walk_accepts(1000, True, True, groups=20) == 591
    assert walk_accepts(1000, True, False, groups=20) == 473

