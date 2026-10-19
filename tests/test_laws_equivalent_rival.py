"""A merged congruent candidate stays a rival and the "too narrow" refutation reads every alias.

Known answer from a MiniGrid run (position rules, seed 6): facing west into an
open door, both encodings offered (relative `fwd`, absolute (dx, dy)) with `fwd ≡ abs(DIR_VEC[d])` declared under
`dir=d`. A logged forward-onto-goal row (a second missing law) refutes every `fwd` candidate as too narrow; the merge
had folded the guarded absolute twin into the refuted `fwd` head, so the unguarded `door open -> abs(-1, 0)` (wrong for
any facing but west) was the only class left and was admitted with 0 probes. Without declarations the guarded absolute
twin stays a class and forces the probe that refutes it."""
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Grammar, LawSet, Transition
    from solvi.experimental.laws._core import Candidate

DIRS = {"E": (1, 0), "S": (0, 1), "W": (-1, 0), "N": (0, -1)}
FRONTS = ("empty", "goal", "door_open", "wall")


def grammar():
    atoms = [Atom(f"front={f}", (lambda f: lambda s: s[0] == f)(f), feature="front") for f in FRONTS]
    atoms += [Atom(f"dir={d}", (lambda d: lambda s: s[1] == d)(d), feature="dir") for d in DIRS]
    return Grammar(atoms, k=2)


def world(s, a):
    return DIRS[s[1]] if s[0] != "wall" else (0, 0)


def apply(e, s):
    return DIRS[s[1]] if e == "fwd" else e


def variants(s, observed, rel_first=True):
    out = ["fwd", tuple(observed)] if observed != (0, 0) and tuple(observed) == DIRS[s[1]] else [tuple(observed)]
    return out if rel_first else out[::-1]


EQUIV = [("fwd", v, f"dir={d}") for d, v in DIRS.items()]


def setup(equivalent, rel_first=True):
    g = grammar()
    ls = LawSet(g, base=lambda s, a: (0, 0), apply=apply, effects=lambda s, o: variants(s, o, rel_first),
                equivalent=EQUIV if equivalent else None)
    ls.add(Candidate("forward", ("front=empty",), "fwd"))
    buf = Buffer(g)
    buf.add(Transition(("goal", "N"), "forward", (0, -1), group="w1", id="goal-north"))
    buf.add(Transition(("wall", "W"), "forward", (0, 0), group="w1", id="wall-west"))
    buf.add(Transition(("wall", "N"), "forward", (0, 0), group="w1", id="wall-north"))
    pool = [((f, d), "forward") for f in FRONTS for d in DIRS]
    return ls, buf, pool


def admit(ls, buf, pool, state=("door_open", "W"), observed=(-1, 0)):
    cands = ls.mismatch(state, "forward", observed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ls.admit_by_probes(cands, world, probes=pool, buffer=buf, budget=8, siblings="undetermined",
                                  coverage="ignore")


def wrong_somewhere(ls, law, pool):
    return any(ls.applied(s, a) is law and apply(law.effect, s) != world(s, a) for s, a in pool)


@pytest.mark.parametrize("rel_first", [True, False])
def test_seed6_no_false_law_with_equivalences(rel_first):
    ls, buf, pool = setup(True, rel_first)
    adm = admit(ls, buf, pool)
    if adm.status == "accepted":
        assert not wrong_somewhere(ls, adm.law, pool), adm.law.text()


def test_seed6_baseline_has_no_false_law_either():
    ls, buf, pool = setup(False)
    adm = admit(ls, buf, pool)
    if adm.status == "accepted":
        assert not wrong_somewhere(ls, adm.law, pool), adm.law.text()


@pytest.mark.parametrize("rel_first", [True, False])
def test_general_law_still_admitted_and_order_neutral(rel_first):
    """The congruent case the declarations are for: facing east into an empty cell with no law yet; the general
    `front=empty -> fwd` is admitted with the declarations whichever encoding is listed first, the merged guarded
    candidate being told apart on a logged or probed row."""
    g = grammar()
    ls = LawSet(g, base=lambda s, a: (0, 0), apply=apply, effects=lambda s, o: variants(s, o, rel_first),
                equivalent=EQUIV)
    buf = Buffer(g)
    buf.add(Transition(("empty", "N"), "forward", (0, -1), group="w1", id="empty-north"))
    buf.add(Transition(("wall", "E"), "forward", (0, 0), group="w1", id="wall-east"))
    pool = [((f, d), "forward") for f in FRONTS for d in DIRS]
    adm = admit(ls, buf, pool, state=("empty", "E"), observed=(1, 0))
    assert adm.status == "accepted", adm.why
    assert adm.law.condition == ("front=empty",) and adm.law.effect == "fwd"


def test_without_declarations_the_congruent_case_is_refused():
    g = grammar()
    ls = LawSet(g, base=lambda s, a: (0, 0), apply=apply, effects=variants)
    buf = Buffer(g)
    buf.add(Transition(("empty", "N"), "forward", (0, -1), group="w1", id="empty-north"))
    buf.add(Transition(("wall", "E"), "forward", (0, 0), group="w1", id="wall-east"))
    pool = [((f, d), "forward") for f in FRONTS for d in DIRS]
    adm = admit(ls, buf, pool, state=("empty", "E"), observed=(1, 0))
    assert adm.status == "undetermined"
