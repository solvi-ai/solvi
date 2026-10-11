"""solvi.experimental.relations.synthesize_residual: top-down synthesis of a relation program through residuals of the
target (X = A ∘ Y ⟹ Y ⊆ A \\ Up = ¬(Aᵀ ∘ ¬Up)), iterative deepening on the number of atoms, a mask ∖ I anywhere;
the result is the package's program tuple, checked exactly by the package's evaluator on the examples, on `check`
worlds and by a `test` (HeldOut) — a candidate that fails is counted and the search goes on. Also the combination
`synthesize(method="residual-first")`: residuals first, the enumerator on what they do not find."""
from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import (Library, Relations, ResidualFound, evaluate, show, synthesize,
                                              synthesize_residual)
    from solvi.experimental.relations.library import HeldOut

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    spec = importlib.util.spec_from_file_location("relations_library_toy", ROOT / "benchmarks" / "relations_library_toy.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


K = _toy()


def _rels(worlds):
    return [Relations.from_dense(["E"], np.asarray(w["E"], bool)[None]) for w in worlds]


def _exact(program, worlds, f, library=None):
    got = evaluate(program, _rels(worlds), library)
    return all(np.array_equal(np.asarray(g, bool), f(w)) for g, w in zip(got, worlds))


def _ops(ast):
    if ast[0] in ("R", "I", "L"):
        return {ast[0]}
    return {ast[0]}.union(*(_ops(x) for x in ast[1:]))


def test_finds_kinship_chains_that_hold_on_check_worlds_of_another_shape():
    train, held, check = K.kin_worlds(3)
    test = HeldOut(held, {k: f for k, (_, f) in K.KIN.items()})
    for name in ("sibling", "cousin", "cousin1r"):
        f = K.KIN[name][1]
        r = synthesize_residual(train, f, name=name, test=test, depth=7)
        assert isinstance(r, ResidualFound) and r.stopped == "found" and r.program is not None
        assert _ops(r.program) <= {"R", "T", "COMP", "MINUS", "I"}            # the package's own program tuples
        assert _exact(r.program, train, f) and _exact(r.program, check, f)    # 0 false accepts on fresh worlds
        assert r.steps > 0 and r.seconds >= 0 and r.atoms >= 2
    assert show(synthesize_residual(train, K.KIN["cousin"][1], depth=5).program) == "(Eᵀ ∘ (((Eᵀ ∘ E) ∖ I) ∘ E))"


def test_relations_input_and_world_input_agree():
    train, _, _ = K.kin_worlds(1)
    f = K.KIN["uncle"][1]
    a = synthesize_residual(train, f, depth=4)                               # worlds + a function of a world
    b = synthesize_residual(_rels(train), [f(w) for w in train], depth=4)   # Relations + per-example matrices
    assert a.program == b.program and a.steps == b.steps and a.program is not None


def test_a_target_outside_the_grammar_is_not_found_and_nothing_is_accepted():
    train, _, _ = K.kin_worlds(0)
    rng = np.random.default_rng(0)
    junk = [rng.random(w["E"].shape) < 0.3 for w in train]
    r = synthesize_residual(train, junk, depth=4, budget=20_000)
    assert r.program is None and r.stopped in ("budget", "exhausted")
    anc = lambda w: _closure(w["E"])                                         # ancestor: a closure, not a chain
    r = synthesize_residual(train, anc, depth=3)
    assert r.program is None and r.stopped == "exhausted"


def _closure(E):
    R = E.copy()
    while True:
        nxt = R | ((R.astype(int) @ E.astype(int)) > 0)
        if np.array_equal(nxt, R):
            return R
        R = nxt


def _shallow(seed, n):
    """A forest of depth exactly 2: roots, children, grandchildren — no great-grandchild anywhere."""
    rng = np.random.default_rng(seed)
    E = np.zeros((n, n), bool)
    roots = list(range(max(1, n // 6)))
    mids = []
    for i in range(len(roots), n):
        if not mids or rng.random() < 0.35:
            E[int(rng.choice(roots)), i] = True
            mids.append(i)
        else:
            E[int(rng.choice(mids)), i] = True
    return {"E": E}


def test_the_held_out_test_rejects_a_program_that_only_fits_the_examples():
    # on forests of depth 2 the great-grandparent relation is empty, which a shorter program also gives (two parents
    # of one child: (E ∘ Eᵀ) ∖ I is empty on any forest); held-out deeper worlds tell them apart
    train = [_shallow(s, 18) for s in range(6)]
    _, held, check = K.kin_worlds(0)
    gg = K.KIN["ggrand"][1]
    plain = synthesize_residual(train, gg, depth=4)
    assert plain.program is not None and not _exact(plain.program, held, gg)   # the negative control: a false find
    tested = synthesize_residual(train, gg, depth=4, name="ggrand", test=HeldOut(held, {"ggrand": gg}))
    assert tested.rejected >= 1 and tested.program is not None
    assert _exact(tested.program, held, gg) and _exact(tested.program, check, gg)
    checked = synthesize_residual(train, gg, depth=4, check=(held, [gg(w) for w in held]))
    assert checked.program == tested.program and checked.rejected == tested.rejected


def test_a_test_needs_a_name_and_a_rejecting_test_moves_the_search_on():
    train, _, _ = K.kin_worlds(2)
    f = K.KIN["grand"][1]
    with pytest.raises(ValueError):
        synthesize_residual(train, f, test=lambda name, p, lib: {"passed": True})
    seen = []

    def first_fails(name, program, library):
        seen.append(program)
        return {"passed": len(seen) > 1}
    r = synthesize_residual(train, f, name="grand", test=first_fails, depth=4)
    assert r.rejected == 1 and seen[0] == ("COMP", ("R", "E"), ("R", "E"))
    assert r.program is not None and r.program != seen[0] and _exact(r.program, train, f)    # the next equal one


def _atom(rng):
    return ("T", ("R", "E")) if rng.random() < 0.5 else ("R", "E")


def _random_chain(rng, atoms, can_mask=True):
    """A random program of the grammar: an atom; an atom ∘ a chain or a chain ∘ an atom; a chain of ≥ 2 atoms ∖ I."""
    if atoms == 1:
        return _atom(rng)
    if can_mask and rng.random() < 0.3:
        return ("MINUS", _random_chain(rng, atoms, False), ("I",))
    rest = _random_chain(rng, atoms - 1)
    return ("COMP", _atom(rng), rest) if rng.random() < 0.5 else ("COMP", rest, _atom(rng))


def test_complete_on_random_chains_no_false_exhaustion():
    # the pruning (a pair without a witness), the forced pairs and the memo of failed subgoals must never rule out a
    # chain of the grammar that fits: every random one of ≤ 5 atoms is found with at most its own number of atoms
    train, _, _ = K.kin_worlds(0)
    rels = _rels(train)
    rng = np.random.default_rng(7)
    tried = 0
    for _ in range(60):
        n = int(rng.integers(1, 6))
        prog = _random_chain(rng, n)
        target = evaluate(prog, rels)
        if not any(t.any() for t in target):
            continue
        tried += 1
        r = synthesize_residual(rels, target, depth=n)
        assert r.program is not None and r.atoms <= n, show(prog)
        assert all(np.array_equal(a, b) for a, b in zip(evaluate(r.program, rels), target))
    assert tried >= 40


def test_the_limit_two_masked_segments_side_by_side_need_a_library_entry():
    # ((Eᵀ ∘ E) ∖ I) ∘ ((Eᵀ ∘ E) ∖ I): no end of it is a bare atom, so peeling cannot reach it (the documented limit)
    train, _, _ = K.kin_worlds(0)
    rels = _rels(train)
    sib = ("MINUS", ("COMP", ("T", ("R", "E")), ("R", "E")), ("I",))
    target = evaluate(("COMP", sib, sib), rels)
    r = synthesize_residual(rels, target, depth=4)
    assert r.program is None and r.stopped == "exhausted"
    lib = Library()
    lib.add(sib, "sibling")
    r = synthesize_residual(rels, target, depth=4, library=lib)
    assert r.program is not None and r.atoms == 2


def test_deterministic():
    train, held, _ = K.kin_worlds(4)
    f = K.KIN["cousin1r"][1]
    a = synthesize_residual(train, f, depth=6)
    b = synthesize_residual(train, f, depth=6)
    assert (a.program, a.steps, a.atoms, a.stopped) == (b.program, b.steps, b.atoms, b.stopped)


def test_hard_stops_by_steps_and_seconds():
    train, _, _ = K.kin_worlds(0)
    f = K.KIN["cousin2"][1]
    r = synthesize_residual(train, f, budget=50)
    assert r.program is None and r.stopped == "budget" and r.steps <= 51
    r = synthesize_residual(train, f, seconds=0.0)
    assert r.program is None and r.stopped == "seconds"
    lines = []
    synthesize_residual(train, K.KIN["uncle"][1], depth=3, log=lines.append)
    assert lines and all(isinstance(x, str) for x in lines)


def test_without_masks_or_converses_the_grammar_shrinks():
    train, _, _ = K.kin_worlds(0)
    assert synthesize_residual(train, K.KIN["sibling"][1], depth=3, masks=False).program is None
    assert synthesize_residual(train, K.KIN["sibling"][1], depth=3, converses=False).program is None
    assert synthesize_residual(train, K.KIN["ggrand"][1], depth=3, converses=False, masks=False).program is not None


def test_library_entries_are_atoms():
    train, _, check = K.kin_worlds(0)
    lib = Library()
    lib.add(("MINUS", ("COMP", ("T", ("R", "E")), ("R", "E")), ("I",)), "sibling")
    f = K.KIN["cousin"][1]
    flat = synthesize_residual(train, f, depth=4)
    r = synthesize_residual(train, f, depth=4, library=lib)
    assert ("L", "sibling") in _leaves(r.program) and r.atoms < flat.atoms
    assert _exact(r.program, check, f, lib)


def _leaves(ast):
    if ast[0] in ("R", "I", "L"):
        return {ast}
    return set().union(*(_leaves(x) for x in ast[1:]))


def test_synthesize_residual_first_combines_both_searches():
    train, _, _ = K.kin_worlds(0)
    rels = _rels(train)
    cousin = [K.KIN["cousin"][1](w) for w in train]
    flat = synthesize(rels, cousin, size=12, include_universal=False, max_behaviours=2_000)
    assert flat.ast is None and flat.by == "enumerate"                       # the enumerator alone: out of its cap
    both = synthesize(rels, cousin, size=12, include_universal=False, max_behaviours=2_000, method="residual-first")
    assert both.ast is not None and both.by == "residual" and both.steps > 0
    assert _exact(both.ast, train, K.KIN["cousin"][1])
    anc = [_closure(w["E"]) for w in train]                                  # a closure: residuals cannot, enumeration can
    fb = synthesize(rels, anc, size=3, method="residual-first", residual_budget=5_000)
    assert fb.ast is not None and fb.by == "enumerate" and fb.steps > 0 and "PLUS" in _ops(fb.ast)
    only = synthesize(rels, anc, size=3, method="residual", residual_budget=5_000)
    assert only.ast is None and only.by == "residual"
    with pytest.raises(ValueError):
        synthesize(rels, anc, size=3, method="bogus")
    sib = [K.KIN["sibling"][1](w) for w in train]                           # needs ∖ I; MINUS not among the ops
    no_minus = synthesize(rels, sib, size=3, method="residual-first", ops=("T", "COMP"))
    assert no_minus.ast is None and no_minus.by == "enumerate"


@pytest.mark.slow
def test_the_ladder_benchmark_finds_every_rung_without_a_library_and_accepts_nothing_false():
    spec = importlib.util.spec_from_file_location("relations_residual", ROOT / "benchmarks" / "relations_residual.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    res = mod.ladder(seeds=(0, 1), flat=False, say=lambda s: None)
    assert res["residual"]["found"] == res["residual"]["runs"] and res["residual"]["false_accepts"] == 0
    comp = mod.composite(seeds=(0,), budget=20_000, seconds=30, flat=False, say=lambda s: None)
    lv = comp["levels"]
    assert lv["grand"]["residual"]["found"] == 1                             # the one chain among the policy's levels
    assert all(x["residual"]["false_accepts"] == 0 for x in lv.values())
    assert lv["deny"]["residual"]["found"] == 0 and lv["deny"]["residual"]["stopped"] == ["exhausted"]
