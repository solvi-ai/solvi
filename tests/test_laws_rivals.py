"""solvi.experimental.laws: LawSet(rivals=True) — a candidate that passes is refused while another offered candidate
or active law with the same action and effect, and a condition neither wider nor narrower, matches exactly the same
held-out rows. Opt-in; the default (rivals=False) is unchanged.

Synthetic worlds only: every law is known, so a false accept is known exactly."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition

# ───────────────────────────────────────────── a grid-game-shaped world: an exception without its parent's atom
D = 6
ATOMS = ("mvO", "mvY", "qS", "qH") + tuple(f"d{i}" for i in range(D))
WRONG = Candidate("go", ("mvO", "qS"), "pass")       # key into anything STOP passes: wrong on key-into-wall rows
RIGHT = ("mvO", "qH")                                # key into a SHUT cell passes


def grid_world(walls, seed=0):
    """Per level: a key pushed into a door (mvO qS qH -> pass), four moves into walls (qS -> block), a rock into the
    door (qS qH -> block), four moves into empty cells (-> pass), and on the levels where walls[i] a key pushed into a
    wall (mvO qS -> block). d0..d5 are random noise atoms (p = 0.3). The door is STOP, so the exception
    `mvO & qH -> pass` overrides `qS -> block` without containing `qS`."""
    rng = random.Random(seed)
    g = Grammar([Atom(a, (lambda a: lambda s: s[a])(a), feature=a) for a in ATOMS], k=2)
    b = Buffer(g)
    for gi, wall in enumerate(walls):
        rows = [(dict(mvO=1, qS=1, qH=1), "pass")]
        rows += [(dict(mvY=j < 2, qS=1), "block") for j in range(4)]
        rows += [(dict(qS=1, qH=1), "block")]
        rows += [(dict(mvY=j < 2, mvO=j == 3), "pass") for j in range(4)]
        if wall:
            rows += [(dict(mvO=1, qS=1), "block")]
        for j, (s, o) in enumerate(rows):
            st = {a: False for a in ATOMS}
            st.update({k: bool(v) for k, v in s.items()})
            for i in range(D):
                st[f"d{i}"] = rng.random() < 0.3
            b.add(Transition(st, "go", o, group=f"g{gi}", id=f"{gi}.{j}"))
    return g, b


def grid_candidates(effect="pass"):
    return [Candidate("go", c, effect) for r in (1, 2) for c in itertools.combinations(ATOMS, r)]


def known_laws(g, rivals=False):
    ls = LawSet(g, rivals=rivals)
    ls.add(Candidate("go", (), "pass"))
    stop = ls.add(Candidate("go", ("qS",), "block"))
    return ls, stop


def groups_needed(rivals, wall=True, seed=1):
    """The fewest supporting levels (the proposing level held out) at which a law is accepted."""
    for n in range(2, 40):
        g, b = grid_world([wall] * (n + 1), seed)
        ls, _ = known_laws(g, rivals)
        got = ls.test(grid_candidates(), b, exclude=f"g{n}")
        if got:
            return n, got[0], ls
    return None, None, None


# ───────────────────────────────────────────── the rule
def test_rivals_is_off_by_default_and_validated_as_a_bool():
    g, _ = grid_world([True])
    assert LawSet(g).rivals is False and LawSet(g).refused_rival == 0
    assert LawSet(g, rivals=True).rivals is True


def test_a_rival_has_another_condition_neither_wider_nor_narrower_and_the_same_rows():
    g, b = grid_world([False] * 4, seed=1)
    ls, _ = known_laws(g)
    right = Candidate("go", RIGHT, "pass")
    others = [Candidate("go", ("mvO", "qS", "qH"), "pass"), Candidate("go", ("mvO", "qS"), "block")]
    assert ls.rival_of(b, right, others, "g3") is None                    # a refinement; another effect
    assert ls.rival_of(b, right, others + [WRONG], "g3") == WRONG.condition
    assert ls.rival_of(b, right, [], "g3") is None
    assert ls.rival_of(b, Candidate("stay", RIGHT, "pass"), [WRONG], "g3") is None   # another action: no rows
    ls.add(WRONG)                                                          # an active law is a rival too
    assert ls.rival_of(b, right, [], "g3") == WRONG.condition
    g, b = grid_world([True] * 4, seed=1)                                  # a key-into-wall row separates them
    assert known_laws(g)[0].rival_of(b, right, [WRONG], "g3") is None


def test_without_a_counterexample_neither_rival_is_accepted():
    """No key ever pushed into a wall: `mvO & qS` and `mvO & qH` match the same held-out rows. The default takes the
    earlier candidate — the wrong one; with rivals=True neither is accepted, and why_not names the rival."""
    nb, law_b, _ = groups_needed(False, wall=False)
    assert law_b.condition == WRONG.condition
    g, b = grid_world([False] * (nb + 1), seed=1)
    ls, _ = known_laws(g, rivals=True)
    got = ls.test(grid_candidates(), b, exclude=f"g{nb}")
    assert ls.refused_rival >= 2 and all(x.condition not in (RIGHT, WRONG.condition) for x in got)
    g, b = grid_world([False] * (nb + 1), seed=1)
    ls, _ = known_laws(g, rivals=True)
    w = ls.why_not(WRONG, b, exclude=f"g{nb}", others=grid_candidates())
    assert w["failed"] == "rival" and w["rival"] == RIGHT and not w["passes"]
    assert ls.why_not(WRONG, b, exclude=f"g{nb}")["passes"]               # no other candidate offered: no rival


def test_with_a_counterexample_the_right_law_is_not_delayed():
    nb, law_b, _ = groups_needed(False)
    nr, law_r, ls = groups_needed(True)
    assert law_b.condition == law_r.condition == RIGHT
    assert nb == nr == 14


def test_the_default_is_unchanged():
    """rivals=False: the same laws in the same order as before the option existed (rival_of is never asked)."""
    for walls in ([True] * 15, [False] * 15, [i % 3 == 0 for i in range(15)]):
        g, b = grid_world(walls, seed=2)
        ls, _ = known_laws(g)
        got = [x.text() for x in ls.test(grid_candidates(), b, exclude="g14", accept=3)]
        assert ls.refused_rival == 0
        g2, b2 = grid_world(walls, seed=2)
        ls2 = LawSet(g2)
        ls2.add(Candidate("go", (), "pass"))
        ls2.add(Candidate("go", ("qS",), "block"))
        assert [x.text() for x in ls2.test(grid_candidates(), b2, exclude="g14", accept=3)] == got


# ───────────────────────────────────────────── a wrong law whose counterexample is rare, levels one at a time
def _reference(r):
    rng = random.Random(12345)
    return grid_world([rng.random() < r for _ in range(3000)], 999)


def false_accepts(rivals, r, seeds=1000):
    """`seeds` seeded runs; a key-into-wall row on each level with probability r; levels arrive one at a time (up to
    20) and the first accepted law counts. A false accept: a law other than the right one whose precision on a
    3,000-level reference world is below min_precision 0.9 (the wrong `mvO & qS` is 0.907 at r = 0.1 and 0.769 at
    r = 0.3). → (false accepts, accepts of the wrong `mvO & qS`, accepts of the right law, runs with no accept)."""
    g_ref, b_ref = _reference(r)
    E = LawSet(g_ref)._effect_mask(b_ref, "go", "pass")

    def prec(cond):
        m = b_ref.match("go", cond)
        return (m & E).bit_count() / max(1, m.bit_count())
    false = wrong = right = none = 0
    for seed in range(seeds):
        rng = random.Random(seed)
        walls = [rng.random() < r for _ in range(20)]
        for L in range(2, 21):
            g, b = grid_world(walls[:L], seed)
            got = known_laws(g, rivals)[0].test(grid_candidates(), b, exclude=f"g{L - 1}")
            if got:
                c = got[0].condition
                false += c != RIGHT and prec(c) < 0.9
                wrong += c == WRONG.condition
                right += c == RIGHT
                break
        else:
            none += 1
    return false, wrong, right, none


@pytest.mark.parametrize("rivals, r, expect", [
    (False, 0.1, (0, 198, 763, 0)), (True, 0.1, (0, 0, 816, 65)),
    (False, 0.3, (17, 3, 983, 0)), (True, 0.3, (15, 0, 985, 0))])
def test_a_wrong_law_with_a_rare_counterexample_measured(rivals, r, expect):
    """The wrong `mvO & qS` is indistinguishable from the right `mvO & qH` while no key-into-wall row is held out.
    Without rivals it was the first accepted law in 198 of 1,000 streams at 10% walls; with rivals, 0, and the right
    law first in 816 (763 without) — in 65 streams nothing was accepted within 20 levels (the price: the right law
    waits until a key-into-wall row is held out). At 30% walls false accepts (precision below 0.9) 17 → 15. Within
    delta = 0.1 in every setting."""
    got = false_accepts(rivals, r)
    assert got == expect
    assert got[0] <= 0.1 * 1000


# ───────────────────────────────────────────── noise in the overridden law's rows: no exception exists
M = 12


def noisy_world(seed, p, n_groups=10, per=8):
    """q → block with `p` of q's rows passing at random (no exception exists); without q, pass at 85%."""
    rng = random.Random(seed)
    g = Grammar([Atom("q", lambda s: s["q"], feature="q")]
                + [Atom(f"a{i}", (lambda i: lambda s: s["a"][i])(i), feature=f"a{i}") for i in range(M)], k=3)
    b = Buffer(g)
    for gi in range(n_groups):
        for j in range(per):
            q = rng.random() < 0.5
            a = [rng.random() < 0.25 for _ in range(M)]
            eff = ("pass" if rng.random() < p else "block") if q else ("pass" if rng.random() < 0.85 else "block")
            b.add(Transition({"q": q, "a": a}, "go", eff, group=f"g{gi}", id=f"{gi}.{j}"))
    return g, b


def exceptions():
    out = [Candidate("go", ("q", f"a{i}"), "pass") for i in range(M)]
    return out + [Candidate("go", ("q", f"a{i}", f"a{j}"), "pass") for i, j in itertools.combinations(range(M), 2)]


@pytest.mark.parametrize("p, min_precision, learned", [(0.08, 0.9, True), (0.3, 0.6, False), (0.3, 0.6, True)])
def test_a_false_exception_is_rarely_accepted_with_rivals(p, min_precision, learned):
    """78 candidate exceptions on 80 rows, no true exception, 1,000 seeded buffers; the overridden law added by hand
    or learned on the same buffer. The stated bound is delta = 0.1 per buffer."""
    trials = accepts = 0
    for seed in range(1000):
        g, b = noisy_world(seed, p)
        ls = LawSet(g, min_precision=min_precision, retire_below=min(0.8, min_precision), rivals=True)
        par = Candidate("go", ("q",), "block")
        if learned:
            if not ls.test([par], b, exclude="g0"):
                continue
        else:
            ls.add(par)
        trials += 1
        accepts += bool(ls.test(exceptions(), b, exclude="g0"))
    assert trials >= 600
    assert accepts <= 0.1 * trials
    assert accepts == 0                       # measured: 0 of 715, 1,000 and 905 buffers (the default: the same)
