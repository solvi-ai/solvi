"""solvi.experimental.laws (experimental, opt-in): `Grammar(negations="refine" | "all")` — negated literals.

Known answer: a "transform unless the noun is itself" world, where the law is learned with negation and not without it.
Candidate counts per mismatch on the design's channel-like grammars ("refine" vs "all" vs positive). A false-accept
simulation on null buffers (40 sparse atoms, 2,000 rows, the effect independent of every atom): no law exists, so any
acceptance is false; the sized test must keep the rate ≤ δ under both modes."""
from __future__ import annotations

import random
import warnings
from math import comb

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition, check_compiled


class _Key:
    """Atom function reading one key of a dict state (a class, so grammars pickle)."""

    def __init__(self, k):
        self.k = k

    def __call__(self, s):
        return bool(s[self.k])


def word_grammar(negations=False, noise=4, k=2):
    atoms = [Atom("nsubj", _Key("nsubj"), code='s["nsubj"]'), Atom("nself", _Key("nself"), code='s["nself"]')]
    atoms += [Atom(f"d{i}", _Key(f"d{i}"), code=f's["d{i}"]') for i in range(noise)]
    return Grammar(atoms, k=k, negations=negations)


def word_state(rng, noise=4):
    s = {"nsubj": rng.random() < 0.5, "nself": rng.random() < 0.3}
    s.update({f"d{i}": rng.random() < 0.5 for i in range(noise)})
    return s


def word_truth(s):
    """The hidden law: a subject noun becomes, unless the noun is itself (then it stays)."""
    return "become" if s["nsubj"] and not s["nself"] else "same"


def run_words(negations, seed=0, logged=300, stream=60):
    """Log `logged` rows over 10 groups, then learn on a stream of new groups (each step predicted before learned)."""
    rng = random.Random(seed)
    g = word_grammar(negations)
    b = Buffer(g)
    for r in range(logged):
        s = word_state(rng)
        b.add(Transition(s, "turn", word_truth(s), group=f"log{r % 10}", id=f"l{r}"))
    ls = LawSet(g, base=lambda s, a: "same", min_support=5, min_precision=0.9)
    steps = [ls.step(Transition(s, "turn", word_truth(s), group=f"s{i}", id=f"s{i}"), b, remember=True)
             for i, s in enumerate(word_state(rng) for _ in range(stream))]
    return g, ls, b, steps


def wrong_laws(ls, seed=99, n=1000):
    """Laws whose prediction differs from the hidden law on some check state where they apply."""
    rng = random.Random(seed)
    states = [word_state(rng) for _ in range(n)]
    bad = []
    for law in ls.laws:
        for s in states:
            if law.holds(ls.grammar.true(s, "turn")) and law.effect != word_truth(s):
                bad.append(law)
                break
    return bad


# ───────────────────────────────────────────── the literal grammar
def test_true_gives_one_literal_per_atom_and_the_literal_shares_the_atoms_feature():
    g = word_grammar("all")
    s = {"nsubj": True, "nself": False, "d0": False, "d1": True, "d2": False, "d3": False}
    T = g.true(s, "turn")
    assert len(T) == len(g.atoms)
    assert set(T) == {"nsubj", "not:nself", "not:d0", "d1", "not:d2", "not:d3"}
    assert g.feature["not:nself"] == g.feature["nself"]
    with pytest.raises(ValueError):
        g.check(("nself", "not:nself"))                     # one literal per feature
    assert word_grammar(False).true(s, "turn") == ("d1", "nsubj")   # off: unchanged
    with pytest.raises(ValueError):
        Grammar([Atom("not:x", _Key("x"))], negations="refine")
    with pytest.raises(ValueError):
        Grammar([], negations="some")


def test_refine_allows_one_negation_and_only_where_the_atom_varies():
    g = word_grammar("refine")
    assert g.allows(("not:nself",)) and g.allows(("not:nself", "nsubj"))         # a lone negation is allowed
    assert not g.allows(("not:d0", "not:nself", "nsubj"))
    with pytest.raises(ValueError):
        g.check(("not:d0", "not:nself"))
    assert not Grammar(word_grammar().atoms, k=2, min_k=1, negations="refine").allows(("not:nself",))
    s = {"nsubj": True, "nself": False, "d0": False, "d1": False, "d2": False, "d3": False}
    T = g.true(s, "turn")
    assert all(not g.is_negated(x) for c in g.conditions(T) for x in c)        # no buffer: positive only
    b = Buffer(g)
    for i, (subj, me, d0) in enumerate([(1, 0, 0), (1, 1, 0), (1, 0, 0), (0, 0, 1)]):
        b.add(Transition({"nsubj": subj, "nself": me, "d0": d0, "d1": 0, "d2": 0, "d3": 0}, "turn", "x", id=f"r{i}"))
    # on the action's rows nself and d0 took both values (the lone literals); under nsubj only nself did
    assert g.refinements(T, b, "turn") == [("not:d0",), ("not:nself",), ("not:nself", "nsubj")]
    assert b.match("turn", ("nsubj", "not:nself")) == 0b101
    all_g = word_grammar("all")
    assert ("not:nself",) in all_g.conditions(all_g.true(s, "turn"))


def test_known_answer_transform_unless_itself_learned_with_negation_not_without():
    _, pos, _, _ = run_words(False)
    bad = wrong_laws(pos)
    assert not [law for law in pos.laws if law.effect == "become" and law not in bad], [x.text() for x in pos.laws]
    for mode in ("refine", "all"):
        _, ls, _, _ = run_words(mode)
        got = [law for law in ls.laws if law.effect == "become"]
        assert any(law.condition == ("not:nself", "nsubj") for law in got), (mode, [x.text() for x in ls.laws])
        assert wrong_laws(ls) == [], (mode, [x.text() for x in wrong_laws(ls)])


def test_a_law_with_a_negated_literal_compiles_and_matches_the_set():
    _, ls, b, _ = run_words("refine")
    law = next(x for x in ls.laws if x.condition == ("not:nself", "nsubj"))
    src = law.to_code(ls.grammar)
    assert "not (s[\"nself\"])" in src
    ns: dict = {}
    exec(src, ns)
    fn = next(v for k, v in ns.items() if k.startswith("law_"))
    rng = random.Random(5)
    for _ in range(300):
        s = word_state(rng)
        assert fn(s, "turn")[0] == law.holds(ls.grammar.true(s, "turn"))
    chk = check_compiled(ls.to_module(), b, ls)
    assert chk.rows == len(b) and chk.ok


def test_every_literal_candidate_scored_is_counted_in_n():
    _, ls, b, steps = run_words("refine")
    miss = [st for st in steps if st.mismatch and st.candidates]
    assert miss and sum(st.tested for st in steps) == b.tested           # N = every distinct candidate scored
    lit = [st for st in miss if st.tested > len(ls.grammar.conditions(ls.grammar.true(st.transition.state, "turn")))]
    assert lit                                                           # refinements were scored and counted


def run_unless_r(negations, seed=0, logged=300, stream=60):
    """A lone negation: every noun becomes unless R holds ("transform unless R" is the law `not:R`)."""
    rng = random.Random(seed)
    atoms = [Atom("R", _Key("R"), code='s["R"]')] + [Atom(f"d{i}", _Key(f"d{i}")) for i in range(4)]
    g = Grammar(atoms, k=2, negations=negations)

    def state():
        return {"R": rng.random() < 0.4, **{f"d{i}": rng.random() < 0.5 for i in range(4)}}

    def truth(s):
        return "same" if s["R"] else "become"

    b = Buffer(g)
    for r in range(logged):
        s = state()
        b.add(Transition(s, "turn", truth(s), group=f"log{r % 10}", id=f"l{r}"))
    ls = LawSet(g, base=lambda s, a: "same", min_support=5, min_precision=0.9)
    for i in range(stream):
        s = state()
        ls.step(Transition(s, "turn", truth(s), group=f"s{i}", id=f"s{i}"), b, remember=True)
    checks = [state() for _ in range(1000)]
    wrong = [law for law in ls.laws if any(law.holds(g.true(s, "turn")) and law.effect != truth(s) for s in checks)]
    return ls, wrong


def test_known_answer_a_lone_negated_law_is_learned_under_refine_not_without():
    pos, wrong = run_unless_r(False)
    assert not [law for law in pos.laws if law.effect == "become" and law not in wrong]
    ls, wrong = run_unless_r("refine")
    assert [law.condition for law in ls.laws if law.effect == "become"] == [("not:R",)], [x.text() for x in ls.laws]
    assert wrong == []


# ───────────────────────────────────────────── candidates per mismatch: "refine" vs "all" vs positive
def channel(n, t, v, rows=200, seed=0):
    """A channel-like grammar of n one-feature atoms; the mismatched state has atoms 0..t-1 true; on the logged rows
    atoms t..t+v-1 vary (true at 30 %), the rest never hold (they cannot vary under any condition)."""
    names = [f"a{i}" for i in range(n)]

    def grammar(mode):
        return Grammar([Atom(x, _Key(x)) for x in names], k=2, negations=mode)

    state = {x: i < t for i, x in enumerate(names)}
    out = {}
    for mode in (False, "refine", "all"):
        g = grammar(mode)
        b = Buffer(g)
        r2 = random.Random(seed)
        for r in range(rows):
            s = {x: (r2.random() < 0.5 if i < t else r2.random() < 0.3 if i < t + v else False)
                 for i, x in enumerate(names)}
            b.add(Transition(s, "do", "x", group=r % 5, id=f"r{r}"))
        out[mode] = len(g.candidates(g.true(state, "do"), b, "do"))
    return out


@pytest.mark.parametrize("n,t,v,all_ratio", [(10, 3, 2, 8), (10, 3, 5, 8), (12, 4, 5, 7), (20, 4, 5, 19),
                                             (45, 9, 8, 23)])
def test_candidate_multipliers_against_the_design(n, t, v, all_ratio):
    got = channel(n, t, v)
    pos = 1 + t + comb(t, 2)
    assert got[False] == pos
    assert got["all"] == 1 + n + comb(n, 2)                              # the full literal grammar at k = 2
    assert got["refine"] == pos + v + t * v              # the lone not:a and one negation beside each one-atom P
    assert abs(got["all"] / pos - all_ratio) < 0.6                       # design: 8×, 8×, 7×, 19×, 23×
    assert got["refine"] / pos <= 8                                      # design: 3–8× (measured 2.1–3.9×)
    assert got["refine"] < got["all"]


# ───────────────────────────────────────────── false accepts on null buffers
def null_rows(rng, n_atoms=40, rows=2000, p_atom=0.1, p_effect=0.7, groups=10):
    out = []
    for r in range(rows):
        s = {f"a{i}": rng.random() < p_atom for i in range(n_atoms)}
        out.append(Transition(s, "do", "E" if rng.random() < p_effect else "N", group=r % groups, id=f"r{r}"))
    return out


def null_accepts(reps, p_effect, seed=7, n_atoms=40):
    names = [f"a{i}" for i in range(n_atoms)]
    atoms = [Atom(x, _Key(x)) for x in names]
    g_all, g_ref, g_pos = (Grammar(atoms, k=2, negations=m) for m in ("all", "refine", False))
    every_all = g_all.literals()
    conds_all = g_all.conditions(every_all)
    assert len(conds_all) == 3201                                        # the design's literal grammar
    rng = random.Random(seed)
    acc = {"all": 0, "refine": 0, "fixed": 0}
    n_ref = 0
    for _ in range(reps):
        rows = null_rows(rng, n_atoms, p_effect=p_effect)
        for mode, g, conds in (("all", g_all, conds_all), ("refine", g_ref, None)):
            b = Buffer(g)
            b.extend(rows)
            if conds is None:
                conds = g.candidates(every_all, b, "do")
                n_ref = len(conds)
            ls = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9)
            acc[mode] += bool(ls.test([Candidate("do", c, "E") for c in conds], b))
            if mode == "all":                                            # the control: fixed thresholds
                fx = LawSet(g, base=lambda s, a: "N", min_support=5, min_precision=0.9, sized=False)
                acc["fixed"] += bool(fx.test([Candidate("do", c, "E") for c in conds], b))
    return acc, n_ref


@pytest.mark.parametrize("p_effect", [0.7, 0.03])
def test_null_false_accepts_stay_below_delta_under_both_modes(p_effect):
    reps = 20
    acc, n_ref = null_accepts(reps, p_effect)
    assert n_ref == 821 + 40 + 40 * 39                                   # positive + lone + one beside each atom
    assert acc["all"] / reps <= 0.1 and acc["refine"] / reps <= 0.1      # measured: 0 of 100 at both rates


@pytest.mark.slow
@pytest.mark.parametrize("p_effect", [0.7, 0.03])
def test_null_false_accepts_100_buffers(p_effect):
    acc, _ = null_accepts(100, p_effect)
    assert acc["all"] <= 10 and acc["refine"] <= 10
