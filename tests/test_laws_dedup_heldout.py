"""solvi.experimental.laws (experimental, opt-in): `Buffer(dedup=True)` — a repeated situation is one row, not many —
and `admit_by_probes(..., held_out=buffer)` — a probe admission the held-out rows of other groups refute is refused.

Synthetic worlds only: every law is known, so a false accept is known exactly."""
from __future__ import annotations

import itertools
import pickle
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Enumerate, Grammar, LawSet, Transition

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


# ───────────────────────────────────────────── held_out: a probe admission the other groups refute
FB = 8
ALL = [tuple(bits) for bits in itertools.product((False, True), repeat=FB)]


def binary_grammar(k=2):
    atoms = []
    for i in range(FB):
        atoms.append(Atom(f"x{i}", (lambda s, i=i: s[i]), feature=f"x{i}"))
        atoms.append(Atom(f"~x{i}", (lambda s, i=i: not s[i]), feature=f"x{i}"))
    return Grammar(atoms, k=k)


def holds(cond, s):
    return all((s[int(x.lstrip("~x"))] is not x.startswith("~")) for x in cond)


def off(s, a):
    return "off"


def probe_trial(seed, biased, held, noise=0.0, rounds=5, rows=60):
    """A hidden law of two atoms; the probe pool holds only states where its second atom is true when biased (so the
    probes cannot tell the one-atom law from the right one), else every state; 60 logged rows of three other groups,
    their observations flipped with probability `noise`. Each round: admit by probes, then `check` on the logged rows.
    → (admissions, wrong ones, right ones)."""
    rng = random.Random(seed)
    g = binary_grammar()
    feats = rng.sample(range(FB), 2)
    truth = tuple(sorted(("~" if rng.random() < 0.5 else "") + f"x{i}" for i in feats))

    def world(s, a):
        return "on" if holds(truth, s) else "off"
    s0 = next(s for s in rng.sample(ALL, len(ALL)) if holds(truth, s))
    states = [s for s in ALL if holds(truth[1:], s)] if biased else ALL
    probes = [(s, "go") for s in rng.sample(states, min(100, len(states)))]
    hb = Buffer(g)
    for r in range(rows):
        s = rng.choice(ALL)
        o = world(s, "go")
        if rng.random() < noise:
            o = "off" if o == "on" else "on"
        hb.add(Transition(s, "go", o, group=f"g{1 + r % 3}", id=f"h{r}"))
    ls = LawSet(g, base=off)
    cands = Enumerate().propose(ls, s0, "go", "on")
    n = wrong = right = 0
    for _ in range(rounds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adm = ls.admit_by_probes(cands, world, probes=probes, budget=8, exclude="g0",
                                     held_out=hb if held else None)
        if adm.status == "accepted":
            n += 1
            w = any(holds(adm.law.condition, s) != holds(truth, s) for s in ALL)
            wrong += w
            right += not w
        ls.check(hb)
    return n, wrong, right


def probe_totals(trials, biased, held, noise=0.0):
    tot = [0, 0, 0, 0]
    for seed in range(trials):
        n, w, r = probe_trial(seed, biased, held, noise)
        tot[0] += n
        tot[1] += w
        tot[2] += r
        tot[3] += r > 0
    return tuple(tot)


@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_held_out_refuses_the_readmission_cycle_measured(noise):
    """Biased pool: by default the wrong one-atom law is admitted, retired by the check on the logged rows and
    admitted again in every round — 1,500 admissions in 300 trials × 5 rounds, all wrong; with held_out: 0."""
    assert probe_totals(300, True, False, noise) == (1500, 1500, 0, 0)
    assert probe_totals(300, True, True, noise) == (0, 0, 0, 0)


@pytest.mark.parametrize("noise", [0.0, 0.05])
def test_held_out_keeps_a_right_law(noise):
    """Unbiased pool: the right law is admitted in the first round; with held_out it is refused at most delta of the
    time when the logged rows are noisy (a right law is right on every noise-free row: never refused)."""
    trials = 300
    base = probe_totals(trials, False, False, noise)
    held = probe_totals(trials, False, True, noise)
    assert held[1] <= base[1] and held[1] <= 0.1 * trials                  # wrong admissions
    assert held[3] >= base[3] - 0.1 * trials                               # trials with the right law
    assert (base, held) == EXPECT_UNBIASED[noise]


EXPECT_UNBIASED = {0.0: ((300, 0, 300, 300), (300, 0, 300, 300)),     # (admissions, wrong, right, trials right)
                   0.05: ((316, 0, 316, 300), (316, 0, 316, 300))}


def test_held_out_fit_and_the_set_left_as_it_was():
    g = binary_grammar(k=1)
    hb = Buffer(g)
    for i, s in enumerate(ALL):
        hb.add(Transition(s, "go", "on" if (s[0] and s[1]) else "off", group=f"g{i % 3 + 1}", id=str(i)))
    ls = LawSet(g, base=off)
    law = ls.add(Candidate("go", ("x0",), "on"))
    h, n, p = ls.held_out_fit(law, hb)
    assert (h, n) == (64, 128) and p < 1e-6
    ls.retire(law)
    v = ls.version
    pool = [(s, "go") for s in ALL if s[1]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        adm = ls.admit_by_probes([Candidate("go", ("x0",), "on")], lambda s, a: "on" if (s[0] and s[1]) else "off",
                                 probes=pool, budget=8, held_out=hb)
    assert adm.status == "undetermined" and "held-out rows refute" in adm.why and ls.refused_held_out == 1
    assert ls.version == v and ls.laws == [] and len(ls.retired) == 1
    ls2 = LawSet(binary_grammar(k=2), base=off)
    ok = ls2.add(Candidate("go", ("x0", "x1"), "on"))
    assert ls2.held_out_fit(ok, hb) == (64, 64, 1.0)
