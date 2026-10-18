"""solvi.experimental.laws: declared effect equivalences (LawSet(equivalent=)): congruent candidates merged into one,
the accepted law's aliases, the row check of a declaration, and false-accept simulations with a declaration in place.

A one-step mover: "act" moves one cell in the action direction, "fwd" one cell along the facing; under "fa" (facing =
action) the two are one effect."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Enumerate, Grammar, LawSet, Transition

DIRS = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}


def grammar():
    return Grammar([Atom("p", lambda s: s["p"]), Atom("fa", lambda s, a: s["facing"] == a),
                    Atom("d0", lambda s: s["d0"]), Atom("d1", lambda s: s["d1"])], k=3)


def apply(effect, s):
    if effect == "stay":
        return s["pos"]
    d = DIRS[s["act"] if effect == "act" else s["facing"]]
    return (s["pos"][0] + d[0], s["pos"][1] + d[1])


def world(s, a):
    """The truth: a player (p) moves in the action direction; nothing else moves."""
    return apply("act", s) if s["p"] else s["pos"]


def state(rng, a, **fix):
    s = {"pos": (5, 5), "facing": rng.choice("UDLR"), "p": rng.random() < 0.5, "d0": rng.random() < 0.5,
         "d1": rng.random() < 0.5, "act": a}
    s.update(fix)
    return s


def variants(s, observed):
    """Both encodings of a move: relative to the action and relative to the facing (the caller does not know)."""
    out = [e for e in ("act", "fwd", "stay") if apply(e, s) == observed]
    return out


def lawset(equivalent=None, **kw):
    return LawSet(grammar(), base=lambda s, a: s["pos"], apply=lambda e, s: apply(e, s), effects=variants,
                  equivalent=equivalent, **kw)


EQ = [("act", "fwd", "fa")]


def pool(rng, n=60, **fix):
    out = []
    for _ in range(n):
        a = rng.choice("UDLR")
        out.append((state(rng, a, **fix), a))
    return out


def logged(rng, n=1, action=None, **fix):
    b = Buffer(grammar())
    for i in range(n):
        a = action or rng.choice("UDLR")
        s = state(rng, a, **fix)
        b.add(Transition(s, a, world(s, a), group="log", id=f"log-{i}"))
    return b


def test_congruent_only_under_a_guard_the_condition_holds():
    ls = lawset(EQ)
    assert ls.congruent(Candidate("U", ("fa", "p"), "act"), Candidate("U", ("fa", "p"), "fwd"))
    assert not ls.congruent(Candidate("U", ("p",), "act"), Candidate("U", ("p",), "fwd"))      # no guard
    assert not ls.congruent(Candidate("U", ("fa", "p"), "act"), Candidate("U", ("fa",), "fwd"))  # other condition
    assert not ls.congruent(Candidate("U", ("fa", "p"), "act"), Candidate("D", ("fa", "p"), "fwd"))
    assert not lawset().congruent(Candidate("U", ("fa", "p"), "act"), Candidate("U", ("fa", "p"), "fwd"))
    by_fn = lawset(lambda e1, e2, cond: {e1, e2} == {"act", "fwd"} and "fa" in cond)
    assert by_fn.congruent(Candidate("U", ("fa", "p"), "act"), Candidate("U", ("fa", "p"), "fwd"))


def test_a_guard_is_required():
    with pytest.raises(ValueError):
        lawset([("act", "fwd", None)])
    with pytest.raises(ValueError):
        lawset([("act", "fwd")])


def test_one_class_accepted_with_aliases_and_no_probe_between_the_congruent_pair():
    rng = random.Random(1)
    cands = [Candidate("U", ("fa", "p"), "act"), Candidate("U", ("fa", "p"), "fwd")]
    buf = logged(rng, n=1, action="U", p=True, facing="U")         # a logged move refutes the set without a law
    probes = [(s, "U") for s, _ in pool(rng)]
    for s, _ in probes:
        s["act"] = "U"
    # without the declaration the pair are two conditions of one length the pool cannot tell apart
    a0 = lawset().admit_by_probes(cands, world, probes=probes, buffer=buf, siblings="undetermined")
    assert a0.status == "undetermined" and "cannot tell apart" in a0.why
    ls = lawset(EQ)
    a = ls.admit_by_probes(cands, world, probes=probes, buffer=buf, siblings="undetermined")
    assert a.status == "accepted" and a.law.effect == "act" and a.law.aliases == ("fwd",)
    assert len(a.asked) == 0


def test_without_the_guard_the_probes_separate_act_from_fwd():
    rng = random.Random(2)
    cands = [Candidate("U", ("p",), "fwd"), Candidate("U", ("p",), "act")]
    probes = []
    for s, _ in pool(rng, n=40):
        s["act"] = "U"
        probes.append((s, "U"))
    buf = logged(rng, n=1, action="U", p=True, facing="U")
    a = lawset(EQ).admit_by_probes(cands, world, probes=probes, buffer=buf)
    assert a.status == "accepted" and a.law.effect == "act" and a.law.aliases == () and len(a.asked) >= 1


def test_mismatch_proposes_one_candidate_per_congruence_class():
    rng = random.Random(3)
    s = state(rng, "U", p=True, facing="U", d0=True, d1=False)
    plain, merged = lawset().mismatch(s, "U", world(s, "U")), lawset(EQ).mismatch(s, "U", world(s, "U"))
    with_fa = [c for c in plain if "fa" in c.condition]
    assert len(plain) - len(merged) == len(with_fa) // 2 > 0
    assert all(c.effect == "act" for c in merged if "fa" in c.condition)


def test_the_test_path_records_aliases():
    rng = random.Random(4)
    g = grammar()
    buf = Buffer(g)
    for i in range(60):
        a = rng.choice("UDLR")
        s = state(rng, a, p=True, facing=a)
        buf.add(Transition(s, a, world(s, a), group=i % 6, id=f"r{i}"))
    ls = LawSet(g, base=lambda s, a: s["pos"], apply=apply, effects=variants, equivalent=EQ, min_support=3,
                sized=False)
    s = state(rng, "U", p=True, facing="U")
    got = ls.test(ls.mismatch(s, "U", world(s, "U")), buf, exclude="none")
    assert got and all(law.effect == "act" for law in got)
    fa_laws = [law for law in got if "fa" in law.condition]
    assert all(law.aliases == ("fwd",) for law in fa_laws)


def test_a_false_declaration_is_reported_on_rows_and_the_law_falls_on_observation():
    rng = random.Random(5)
    bad = [("act", "fwd", "d0")]                       # wrong: under d0 the facing need not be the action
    cands = [Candidate("U", ("d0", "p"), "fwd"), Candidate("U", ("d0", "p"), "act")]
    probes = []
    for s, _ in pool(rng, n=40, p=True, d0=True, facing="U"):   # every probe faces the action: no probe separates
        s["act"] = "U"
        probes.append((s, "U"))
    buf = logged(rng, n=1, action="U", p=True, facing="U", d0=True)
    ls = lawset(bad, retire_min=3)
    a = ls.admit_by_probes(cands, world, probes=probes, buffer=buf)
    assert a.status == "accepted" and a.law.effect == "fwd" and a.law.aliases == ("act",)
    rows = Buffer(grammar())
    for i in range(8):
        s = state(rng, "U", p=True, d0=True, facing="L")
        rows.add(Transition(s, "U", world(s, "U"), group="w", id=f"w{i}"))
    with pytest.warns(UserWarning, match="contradicted"):
        found = ls.check_equivalent(rows)
    assert {(e1, e2) for e1, e2, _, _ in found} == {("act", "fwd"), ("fwd", "act")}
    assert found[0][3] == [f"w{i}" for i in range(8)]
    for t in rows.rows():
        ls.observe(t)
    assert a.law.status == "retired"
    assert lawset(EQ).check_equivalent(rows, warn=False) == []           # a true declaration: nothing to report


# ───────────────────────────────────────────── false accepts with a declaration in place
def test_false_accepts_on_null_buffers_with_an_equivalence_declared():
    rng = random.Random(0)
    n = 30
    g = Grammar([Atom(f"f{i}", (lambda s, i=i: s[i]), feature=f"f{i}") for i in range(n)], k=2)
    names = [a.name for a in g.atoms]
    conds = [c for size in range(3) for c in itertools.combinations(names, size)]
    eq = [("E", "E2", x) for x in names]
    acc = 0
    for _ in range(60):
        b = Buffer(g)
        for r in range(1500):
            s = tuple(rng.random() < 0.06 for _ in range(n))
            b.add(Transition(s, "do", "E" if rng.random() < 0.7 else "N", group=r % 10, id=f"r{r}"))
        ls = LawSet(g, base=lambda s, a: "N", apply=lambda e, s: "E" if e in ("E", "E2") else e, equivalent=eq)
        acc += bool(ls.test([Candidate("do", c, e) for c in conds for e in ("E", "E2")], b))
    assert acc / 60 <= 0.1                                              # measured: 0 of 60


F = 10
ALL = [tuple(bits) for bits in itertools.product((False, True), repeat=F)]


def holds(cond, s):
    return all((s[int(x.lstrip("~x"))] is not x.startswith("~")) for x in cond)


def admission_rates(trials, equivalent, seed=7):
    """The synthetic world of test_laws_sized_probes (a hidden conjunction of two of 10 binary features), every
    candidate offered twice — effect "on" and its congruent twin "on2" (declared one under every atom)."""
    rng = random.Random(seed)
    atoms = []
    for i in range(F):
        atoms += [Atom(f"x{i}", (lambda s, i=i: s[i]), feature=f"x{i}"),
                  Atom(f"~x{i}", (lambda s, i=i: not s[i]), feature=f"x{i}")]
    g = Grammar(atoms, k=3)
    eq = [("on", "on2", a.name) for a in atoms] if equivalent else None
    out = {"right": 0, "false": 0, "undetermined": 0}
    asked = 0
    for _ in range(trials):
        feats = rng.sample(range(F), 2)
        truth = tuple(sorted(("~" if rng.random() < 0.5 else "") + f"x{i}" for i in feats))
        world = lambda s, a, truth=truth: "on" if holds(truth, s) else "off"
        s0 = next(s for s in rng.sample(ALL, len(ALL)) if holds(truth, s))
        ls = LawSet(g, base=lambda s, a: "off", apply=lambda e, s: "on" if e in ("on", "on2") else e, equivalent=eq)
        base = Enumerate().propose(ls, s0, "do", "on")
        cands = [Candidate(c.action, c.condition, e) for c in base for e in ("on", "on2")]
        probes = [(s, "do") for s in rng.sample(ALL, 200)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            adm = ls.admit_by_probes(cands, world, probes=probes, budget=8, seed=rng.randrange(10 ** 9))
        asked += len(adm.asked)
        if adm.status != "accepted":
            out["undetermined"] += 1
        else:
            out["false" if any(holds(adm.law.condition, s) != holds(truth, s) for s in ALL) else "right"] += 1
    return {k: v / trials for k, v in out.items()}, asked / trials


def test_probe_admissions_with_a_planted_congruent_pair_are_never_wrong():
    with_eq, asked = admission_rates(150, True)
    without, asked0 = admission_rates(150, False)
    assert with_eq["false"] == 0.0
    assert with_eq["undetermined"] <= without["undetermined"]
    assert asked <= asked0 + 1e-9
