"""solvi.experimental.relations.library: composites by enumeration with a library of accepted rules.

find_rules enumerates small programs over the base relations bottom-up (observational equivalence on the examples),
hands each program whose behaviour equals a target on the examples to the held-out test, admits what passes into the
library as a new terminal and enumerates again — so a composite becomes a small program over earlier finds. The test is
the judge: a program that fits the examples but not the held-out worlds is rejected, never admitted."""
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import library as L


def forest(rng, n):
    """A forest by generations: E[p, c] = p is the parent of c; nodes permuted."""
    E = np.zeros((n, n), bool)
    roots = max(1, n // 6)
    depth = {i: 0 for i in range(roots)}
    for i in range(roots, n):
        cand = [j for j in depth if depth[j] < 5]
        p = int(rng.choice(cand))
        E[p, i] = True
        depth[i] = depth[p] + 1
    perm = rng.permutation(n)
    return {"E": E[perm][:, perm]}


def mm(a, b):
    return (a.astype(np.int64) @ b.astype(np.int64)) > 0


def grand(w):
    return mm(w["E"], w["E"])


def sibling(w):
    E = w["E"]
    return mm(E.T, E) & ~np.eye(len(E), dtype=bool)


def uncle(w):
    return mm(sibling(w), w["E"])


def worlds(seed, k, lo, hi):
    rng = np.random.default_rng(seed)
    return [forest(rng, int(rng.integers(lo, hi + 1))) for _ in range(k)]


TRAIN, HELD = worlds(0, 16, 10, 18), worlds(1, 6, 30, 40)
TARGETS = {"grand": grand, "sibling": sibling, "uncle": uncle}


def test_the_ast_runs_on_any_world_and_prints():
    w = TRAIN[0]
    prog = ("MINUS", ("COMP", ("T", ("R", "E")), ("R", "E")), ("I",))
    assert np.array_equal(L.run(prog, w), sibling(w))
    assert np.array_equal(L.run(("PLUS", ("R", "E")), w), L.run(("PLUS", ("PLUS", ("R", "E"))), w))
    assert L.show(prog) == "((Eᵀ ∘ E) ∖ I)"
    assert L.size(prog) == 6
    dom = L.run(("CNT", 1, ("R", "E")), w)                  # a unary predicate as a diagonal: "has a child"
    assert np.array_equal(np.diag(dom), w["E"].any(1)) and not (dom & ~np.eye(len(dom), dtype=bool)).any()


def test_the_library_entry_has_the_agreed_fields_and_runs_as_a_terminal():
    e = L.Library(program=("COMP", ("R", "E"), ("R", "E")), name="grand", examples=(), test_record={"passed": True})
    assert (e.program, e.name, e.examples, e.test_record["passed"]) == (("COMP", ("R", "E"), ("R", "E")), "grand", (), True)
    w = HELD[0]
    assert np.array_equal(L.run(("COMP", ("L", "grand"), ("R", "E")), w, library=[e]), mm(grand(w), w["E"]))
    with pytest.raises(KeyError):
        L.run(("L", "nope"), w, library=[e])


def test_find_rules_admits_found_rules_and_builds_composites_on_them():
    test = L.HeldOut(HELD, TARGETS)
    r = L.find_rules(TRAIN, TARGETS, base=["E"], size=6, budget=20_000, library=(), test=test)
    assert set(r.found) == {"grand", "sibling", "uncle"}
    assert [e.name for e in r.library] == [n for n in r.order]                     # admitted in the order found
    unc = r.found["uncle"].program
    assert L.size(unc) <= 3 and ("L", "sibling") in _leaves(unc)                   # a small program over the library
    for name, e in r.found.items():
        assert e.test_record["passed"] and e.test_record["errors"] == 0
        for w in worlds(9, 3, 60, 80):                                             # runs on larger unseen worlds
            assert np.array_equal(L.run(e.program, w, library=r.library), TARGETS[name](w)), name
    assert r.records["uncle"].programs > r.records["grand"].programs              # cumulative cost, in order found


def cousin(w):
    return mm(w["E"].T, uncle(w))


def test_without_the_library_the_composite_costs_more_or_is_not_found():
    tg = dict(TARGETS, cousin=cousin)
    test = L.HeldOut(HELD, tg)
    lib = L.find_rules(TRAIN, tg, base=["E"], size=10, budget=50_000, library=(), test=test)
    flat = L.find_rules(TRAIN, tg, base=["E"], size=10, budget=50_000, library=(), test=test, grow=False)
    assert flat.library and all(L.size(e.program) == L.size(L.expand(e.program, flat.library)) for e in flat.library)
    assert "cousin" in lib.found and lib.found["cousin"].test_record["passed"] and "cousin" not in flat.found
    rec = lib.records["cousin"]
    assert rec.size <= 4 and rec.round_programs < rec.programs                     # its own round vs all rounds so far
    assert sum(r.round_programs for r in lib.records.values()) <= lib.programs


def test_the_held_out_test_rejects_a_program_that_only_fits_the_examples():
    """Train on worlds where nobody has a grandchild-of-sibling, so 'uncle' looks empty: the empty-on-examples program
    must not be admitted; it is listed as rejected."""
    star = [{"E": np.zeros((6, 6), bool)} for _ in range(4)]                       # no edges at all
    r = L.find_rules(star, {"uncle": uncle}, base=["E"], size=4, budget=2_000, library=(),
                     test=L.HeldOut(HELD, {"uncle": uncle}))
    assert "uncle" not in r.found and r.records["uncle"].rejected >= 1


def test_held_out_in_precision_mode_takes_noise():
    rng = np.random.default_rng(3)
    seq = []
    for w in HELD:                                       # labels with noise: 4% of true pairs lost, 2 false pairs added
        y = grand(w).copy()
        y[y & (rng.random(y.shape) < 0.04)] = False
        y[tuple(rng.integers(0, len(y), (2, 2)))] = True
        seq.append(y)
    exact = L.HeldOut(HELD, {"grand": seq})
    soft = L.HeldOut(HELD, {"grand": seq}, exact=False, min_precision=0.9, min_recall=0.8)
    prog = ("COMP", ("R", "E"), ("R", "E"))
    assert not exact("grand", prog, [])["passed"]
    rec = soft("grand", prog, [])
    assert rec["passed"] and rec["precision"] >= 0.9 and rec["net"] > 0 and rec["support"] >= 5


def test_find_rules_is_deterministic_and_order_free():
    test = L.HeldOut(HELD, TARGETS)
    a = L.find_rules(TRAIN, TARGETS, base=["E"], size=6, budget=20_000, library=(), test=test)
    b = L.find_rules(TRAIN, dict(reversed(list(TARGETS.items()))), base=["E"], size=6, budget=20_000, library=(),
                     test=test)
    assert {k: L.show(v.program) for k, v in a.found.items()} == {k: L.show(v.program) for k, v in b.found.items()}
    assert a.programs == b.programs


def test_the_budget_counts_enumerated_programs_and_stops():
    test = L.HeldOut(HELD, TARGETS)
    r = L.find_rules(TRAIN, {"uncle": uncle}, base=["E"], size=9, budget=300, library=(), test=test)
    assert r.programs <= 300 and "uncle" not in r.found and r.records["uncle"].programs is None


def test_a_given_library_is_used_and_not_changed():
    test = L.HeldOut(HELD, TARGETS)
    first = L.find_rules(TRAIN, {"sibling": sibling}, base=["E"], size=6, budget=20_000, library=(), test=test)
    given = tuple(first.library)
    r = L.find_rules(TRAIN, {"uncle": uncle}, base=["E"], size=4, budget=20_000, library=given, test=test)
    assert "uncle" in r.found and r.records["uncle"].programs < first.records["sibling"].programs
    assert tuple(r.library[:len(given)]) == given and len(given) == 1


class _Oracle:
    """A stand-in for a gradient-trained proposer: hands over programs, which are checked like any other candidate."""
    name = "oracle"

    def __init__(self, programs):
        self.programs, self.calls = programs, 0

    def propose(self, name, examples, target, library):
        self.calls += 1
        return self.programs.get(name, [])


def test_the_proposer_slot_is_a_fallback_whose_proposals_are_checked():
    test = L.HeldOut(HELD, TARGETS)
    good = ("COMP", ("MINUS", ("COMP", ("T", ("R", "E")), ("R", "E")), ("I",)), ("R", "E"))
    bad = ("COMP", ("R", "E"), ("R", "E"))
    p = _Oracle({"uncle": [bad, good]})
    r = L.find_rules(TRAIN, {"uncle": uncle}, base=["E"], size=3, budget=500, library=(), test=test, proposer=p)
    assert p.calls == 1 and "uncle" in r.found and r.found["uncle"].program == good
    assert r.found["uncle"].test_record["by"] == "oracle"
    p2 = _Oracle({"uncle": [bad]})
    r2 = L.find_rules(TRAIN, {"uncle": uncle}, base=["E"], size=3, budget=500, library=(), test=test, proposer=p2)
    assert "uncle" not in r2.found


def test_unknown_base_relation_and_bad_targets_are_refused():
    test = L.HeldOut(HELD, TARGETS)
    with pytest.raises(ValueError):
        L.find_rules(TRAIN, TARGETS, base=["F"], size=3, budget=100, library=(), test=test)
    with pytest.raises(ValueError):
        L.find_rules(TRAIN, {"x": [np.zeros((2, 2), bool)]}, base=["E"], size=3, budget=100, library=(), test=test)


def _leaves(ast):
    if ast[0] in ("R", "L", "I"):
        return [ast]
    return [x for a in ast[1:] if isinstance(a, tuple) for x in _leaves(a)]


def test_tolerance_finds_a_rule_behind_noisy_example_labels():
    rng = np.random.default_rng(4)
    noisy = []
    for w in TRAIN:                                       # 5% of the true pairs lost in the examples' labels
        y = grand(w).copy()
        y[y & (rng.random(y.shape) < 0.05)] = False
        noisy.append(y)
    test = L.HeldOut(HELD, {"grand": grand})
    exact = L.find_rules(TRAIN, {"grand": noisy}, base=["E"], size=4, budget=2_000, library=(), test=test)
    soft = L.find_rules(TRAIN, {"grand": noisy}, base=["E"], size=4, budget=2_000, library=(), test=test,
                        tolerance=0.1)
    assert "grand" not in exact.found
    assert L.show(soft.found["grand"].program) == "(E ∘ E)"
