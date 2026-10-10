"""solvi.experimental.relations: relations over a state as packed-bit matrices, the relation algebra, a bottom-up
enumerator of programs with observational equivalence on the full relations, scoring of a readout, a library of
accepted programs, compilation of a program into set code (refusing dense answers), the encoders (an egocentric
window, a path tree) and the proposer that hands relational atoms to the unchanged held-out test of a law set."""
from __future__ import annotations

import random
import warnings

import numpy as np
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Enumerate, Grammar, LawSet, Transition, check_compiled
    from solvi.experimental.relations import (OPS, DenseAnswer, Library, Relations, RelAlgProposer, check_program,
                                              closure, compile_ast, compile_program, complement, compose, converse,
                                              count_at_least, encode, encode_paths, enumerate_programs, evaluate,
                                              intersect, minus, readout, relational_atom, score, show, size,
                                              synthesize, top_programs, union, PathTree)

E0, E1, E2 = ("R", "E0"), ("R", "E1"), ("R", "E2")


def rand_rel(rng, n, k=3, p=None, names=("E0", "E1", "E2")):
    r = Relations(names[:k], n)
    for name in r.names:
        m = rng.random((n, n)) < (p if p is not None else 1.5 / n)
        np.fill_diagonal(m, False)
        r.set(name, m)
    return r


def worlds(seed=0, count=8, lo=5, hi=10, p=None):
    rng = np.random.default_rng(seed)
    return [rand_rel(rng, int(rng.integers(lo, hi + 1)), p=p) for _ in range(count)]


# ───────────────────────────────────────────── relations and the algebra
def test_relations_are_packed_bits_and_round_trip():
    r = Relations(["A", "B"], 11)
    r.add("A", 0, 10)
    r.update("A", [(3, 4), (4, 3)])
    r.mark("B", [2, 5])
    assert r.bits.dtype == np.uint8 and r.bits.shape == (2, 11, 2)       # 11 bits per row → 2 bytes
    assert r.pairs("A") == {(0, 10), (3, 4), (4, 3)} and r.pairs("B") == {(2, 2), (5, 5)}
    m = r.matrix("A")
    assert m.dtype == bool and m.shape == (11, 11) and m[0, 10] and not m[10, 0]
    r2 = Relations.from_dense(["A", "B"], r.dense())
    assert np.array_equal(r2.bits, r.bits) and r2.sets() == r.sets()
    with pytest.raises(KeyError):
        r.matrix("C")


def test_algebra_identities():
    rng = np.random.default_rng(1)
    for n in (1, 5, 12, 40):
        a, b, c = (rng.random((n, n)) < 0.2 for _ in range(3))
        mask = np.ones((n, n), bool)
        assert np.array_equal(converse(converse(a)), a)
        assert np.array_equal(converse(compose(a, b)), compose(converse(b), converse(a)))
        assert np.array_equal(compose(compose(a, b), c), compose(a, compose(b, c)))
        assert np.array_equal(compose(a, union(b, c)), union(compose(a, b), compose(a, c)))
        assert np.array_equal(complement(complement(a, mask), mask), a)
        assert np.array_equal(complement(union(a, b), mask), intersect(complement(a, mask), complement(b, mask)))
        assert np.array_equal(minus(a, b), intersect(a, complement(b, mask)))
        t = closure(a)
        assert np.array_equal(closure(t), t) and not (a & ~t).any()
        assert np.array_equal(t, union(a, compose(a, t)))                # the fixed point
        assert not (compose(a, b, k=2) & ~compose(a, b)).any()           # ≥2 paths ⊆ ≥1 path
        d = count_at_least(a, 2)
        assert not (d & ~np.eye(n, dtype=bool)).any()
        assert np.array_equal(np.diag(d), a.sum(1) >= 2)


def test_closure_is_transitive_on_a_chain_and_a_cycle():
    n = 6
    chain = np.zeros((n, n), bool)
    chain[np.arange(n - 1), np.arange(1, n)] = True
    assert np.array_equal(closure(chain), np.triu(np.ones((n, n), bool), 1))
    cyc = chain.copy()
    cyc[n - 1, 0] = True
    assert closure(cyc).all()                                            # every node reaches every node, itself too


# ───────────────────────────────────────────── programs
def test_show_size_and_evaluate():
    ast = ("MINUS", ("COMP", ("T", E2), E0), ("I",))
    assert show(ast) == "((E2ᵀ ∘ E0) ∖ I)" and size(ast) == 6
    assert show(("PLUS", E0)) == "(E0)⁺" and show(("C2", E0, E1)) == "(E0 #2∘ E1)"
    r = worlds()[0]
    want = compose(converse(r.matrix("E2")), r.matrix("E0")) & ~np.eye(r.n, dtype=bool)
    assert np.array_equal(evaluate(ast, r), want)
    assert np.array_equal(evaluate(("U",), r), np.ones((r.n, r.n), bool))
    assert np.array_equal(evaluate(("NOT", E0), r), ~r.matrix("E0"))


def test_the_enumerator_is_deterministic_and_keeps_one_program_per_behaviour():
    ex = worlds()
    a = [(p.ast, p.size) for p in enumerate_programs(ex, 4)]
    b = [(p.ast, p.size) for p in enumerate_programs(ex, 4)]
    assert a == b and len(a) > 100
    assert [s for _, s in a] == sorted(s for _, s in a)                  # bottom-up by size
    vals = {}
    for p in enumerate_programs(ex, 4):
        key = np.packbits(p.value).tobytes()
        assert key not in vals, (show(p.ast), show(vals.get(key, ("I",))))
        vals[key] = p.ast
    assert a[0][0] == E0 and ("I",) in [x for x, _ in a[:5]] and ("U",) in [x for x, _ in a[:5]]


def test_the_enumerator_finds_the_smallest_program_and_respects_ops_and_caps():
    ex = worlds(seed=3, p=0.25)
    target = [evaluate(("COMP", E0, ("T", E1)), r) for r in ex]
    f = synthesize(ex, target, 5)
    assert f.ast is not None and size(f.ast) == 4
    assert all(np.array_equal(evaluate(f.ast, r), t) for r, t in zip(ex, target))
    assert synthesize(ex, target, 3).ast is None                         # not expressible within 3 nodes
    en = enumerate_programs(ex, 4, ops=("T", "COMP"))
    progs = list(en)
    assert {p.ast[0] for p in progs} <= {"R", "I", "U", "T", "COMP"}
    capped = enumerate_programs(ex, 6, max_behaviours=50)
    assert len(list(capped)) <= 51 and capped.stopped == "cap"
    assert en.stopped == "exhausted" and en.evals >= en.behaviours == len(progs)
    assert "CNT2" in OPS and "C3" in OPS


def test_observational_equivalence_is_sound_under_composition_and_a_readout_key_is_not():
    # equal full relations stay equal under every operation with a third relation
    ex = worlds(seed=5)
    x, y = ("OR", E0, E1), ("OR", E1, E0)
    for r in ex:
        assert np.array_equal(evaluate(x, r), evaluate(y, r))
        for op in ("COMP", "AND", "OR", "MINUS", "C2"):
            assert np.array_equal(evaluate((op, x, E2), r), evaluate((op, y, E2), r))
    # two relations with the same readout (node 0 relates to something) but different elsewhere: x∘x differs
    r = Relations(["F", "G"], 3)
    r.update("F", [(0, 1)])
    r.update("G", [(0, 1), (1, 2)])
    assert readout(evaluate(("R", "F"), r)) == readout(evaluate(("R", "G"), r))
    assert readout(evaluate(("COMP", ("R", "F"), ("R", "F")), r)) != readout(evaluate(("COMP", ("R", "G"), ("R", "G")), r))
    # the enumerator keys on the full relation, so it keeps both as different behaviours
    kept = [p.ast for p in enumerate_programs([r], 1)]
    assert ("R", "F") in kept and ("R", "G") in kept


def test_a_complement_stays_inside_each_examples_real_pairs():
    small, big = Relations(["A"], 2), Relations(["A"], 5)
    small.add("A", 0, 1)
    for p in enumerate_programs([small, big], 2, ops=("NOT",)):
        assert not p.value[0, 2:, :].any() and not p.value[0, :, 2:].any(), show(p.ast)   # padding stays empty
    assert np.array_equal(evaluate(("NOT", ("R", "A")), small), [[True, False], [True, True]])


def test_search_on_a_subset_rechecks_on_all_examples_and_memory_is_capped():
    ex = worlds(seed=9, count=12, p=0.25)
    target = [evaluate(("COMP", E0, ("T", E1)), r) for r in ex]
    f = synthesize(ex[:4], target[:4], 5, check=(ex, target))
    assert f.ast is not None and all(np.array_equal(evaluate(f.ast, r), t) for r, t in zip(ex, target))
    # a target that only the subset satisfies: the subset's match fails the re-check and is not returned
    fake = [evaluate(E0, r) if i < 4 else evaluate(E1, r) for i, r in enumerate(ex)]
    assert synthesize(ex[:4], fake[:4], 3, check=(ex, fake)).ast is None
    en = enumerate_programs(ex, 6, max_bytes=200_000)
    list(en)
    assert en.stopped == "cap" and en.behaviours * en.bytes_per_behaviour <= 200_000 + en.bytes_per_behaviour


def test_score_counts_support_and_precision_of_the_readout():
    r = Relations(["A"], 3)
    r.update("A", [(0, 1)])
    empty = Relations(["A"], 3)
    ex = [r, r, empty, r]
    s = score(("R", "A"), ex, [True, False, True, True])
    assert (s.support, s.hits) == (3, 2) and s.precision == pytest.approx(2 / 3)
    assert score(("R", "A"), ex, [False] * 4).precision == 0.0
    assert score(("T", ("R", "A")), ex, [True] * 4).support == 0


def test_top_programs_do_not_stop_at_the_first_match_and_tolerate_noise():
    rng = np.random.default_rng(7)
    ex = []
    for _ in range(60):
        r = Relations(["FACE", "TREE", "STONE"], 4)
        r.add("FACE", 0, 1)
        r.mark("TREE" if rng.random() < 0.5 else "STONE", [1])
        if rng.random() < 0.3:
            r.mark("TREE", [2])
        ex.append(r)
    y = [bool(r.matrix("TREE")[1, 1]) for r in ex]
    noisy = [v if i % 10 else not v for i, v in enumerate(y)]            # 10% of the labels flipped
    top = top_programs(ex, noisy, size=3, k=5)
    assert 2 <= len(top) <= 5                                           # as many readout patterns as exist
    assert show(top[0].ast) == "(FACE ∘ TREE)"
    assert top[0].score.precision >= 0.85 and top[0].score.support >= 20
    pats = [tuple(t.readout) for t in top]
    assert len(set(pats)) == len(pats)                                   # one program per readout pattern (default)
    assert score(top[0].ast, ex, y).precision == 1.0                     # the clean rule, found under noisy labels
    assert len(top_programs(ex, noisy, size=3, k=20, per_readout=3)) > len(top_programs(ex, noisy, size=3, k=20))


# ───────────────────────────────────────────── the library
def test_the_library_holds_programs_as_terminals():
    lib = Library()
    anc = lib.add(("PLUS", ("T", ("R", "PARENT"))), "ancestor", examples=("s1", "s2"), test={"support": 9})
    assert (anc.name, anc.program, anc.examples, anc.test) == ("ancestor", ("PLUS", ("T", ("R", "PARENT"))),
                                                               ("s1", "s2"), {"support": 9})
    assert "ancestor" in lib and len(lib) == 1 and lib["ancestor"] is anc and list(lib) == [anc]
    with pytest.raises(ValueError):
        lib.add(("R", "PARENT"), "ancestor")
    ex = [encode_paths([p], focus=p, names=("secret",)) for p in
          ("/a/secret/b/c.txt", "/a/b/c.txt", "/secret/x.txt", "/q/r.txt", "/q/secret/r/s/t.md")]
    target = [evaluate(("COMP", ("PLUS", ("T", ("R", "PARENT"))), ("R", "NAME=secret")), r) for r in ex]
    flat = synthesize(ex, target, 5)
    with_lib = synthesize(ex, target, 5, library=lib)
    assert size(flat.ast) == 5 and with_lib.ast == ("COMP", ("L", "ancestor"), ("R", "NAME=secret"))
    assert show(with_lib.ast) == "(ancestor ∘ NAME=secret)" and with_lib.evals < flat.evals
    assert lib.expand(with_lib.ast) == ("COMP", ("PLUS", ("T", ("R", "PARENT"))), ("R", "NAME=secret"))


# ───────────────────────────────────────────── compiled set code
SPARSE_ASTS = [E0, ("T", E1), ("COMP", E0, E1), ("C2", E0, ("T", E0)), ("C3", ("OR", E0, E1), ("T", E2)),
               ("PLUS", E0), ("PLUS", ("OR", E0, ("T", E1))), ("MINUS", ("COMP", E0, E1), ("I",)),
               ("AND", E0, ("NOT", E1)), ("AND", ("NOT", E2), ("PLUS", E1)), ("MINUS", E0, ("NOT", E1)),
               ("XOR", E0, ("T", E0)), ("AND", ("U",), E2), ("CNT2", ("OR", E0, E1)), ("COMP", ("CNT3", E0), E1),
               ("MINUS", ("PLUS", E1), ("PLUS", E2))]


def test_compiled_programs_agree_with_the_interpreter_pair_for_pair():
    rng = np.random.default_rng(11)
    ws = [rand_rel(rng, n, p=p) for n in (12, 64) for p in (1.5 / n, 4.0 / n)] + [rand_rel(rng, 1000, p=1.5 / 1000)]
    for backend in ("sets", "arrays"):
        for ast in SPARSE_ASTS:
            chk = check_program(ast, ws, backend=backend)
            assert chk.ok and chk.differences == 0 and chk.worlds == len(ws), (backend, show(ast))
    small = ws[:2]
    for ast in (("NOT", E0), ("OR", ("U",), E1), ("AND", ("NOT", E0), ("NOT", E1))):     # dense, bounded by n
        for backend in ("sets", "arrays"):
            assert check_program(ast, small, dense=True, backend=backend).ok, (backend, show(ast))


def test_at_least_one_successor_is_an_opt_in_operator_that_compiles():
    """CNT1 (the nodes with a successor, as a diagonal) is known to the algebra, the enumerator and both compile
    backends, but not in the default OPS: the default search space stays as it was."""
    assert "CNT1" not in OPS and "CNT2" in OPS
    ex = worlds(seed=3)
    for r in ex:
        dom = evaluate(("CNT1", E0), r)
        assert np.array_equal(np.diag(dom), r.matrix("E0").any(1)) and not (dom & ~np.eye(r.n, dtype=bool)).any()
    assert show(("CNT1", E0)) == "#≥1(E0)" and size(("CNT1", E0)) == 2
    progs = list(enumerate_programs(ex, 2, ops=("CNT1",)))
    assert ("CNT1", E0) in {p.ast for p in progs}
    assert "CNT1" not in {p.ast[0] for p in enumerate_programs(ex, 2)}
    rng = np.random.default_rng(12)
    ws = [rand_rel(rng, n, p=1.5 / n) for n in (12, 64, 1000)]
    for backend in ("sets", "arrays"):
        for ast in (("CNT1", E0), ("COMP", ("CNT1", ("T", E1)), E2)):
            assert check_program(ast, ws, backend=backend).differences == 0, (backend, show(ast))


def test_compiled_code_is_plain_python_over_sets():
    src = compile_ast(("PLUS", E0), name="anc")
    assert src.startswith("def anc(R, n):") and "import" not in src
    fn = compile_program(("MINUS", ("COMP", ("T", E0), E0), ("I",)))
    R = {"E0": {(0, 1), (0, 2), (3, 4)}, "E1": set(), "E2": set()}
    assert fn(R, 5) == {(1, 2), (2, 1)}                                  # siblings


def test_dense_answers_are_refused_unless_the_domain_is_bounded():
    fn = compile_program(("AND", ("NOT", E0), ("U",)), dense=True)          # explicitly bounded by the node count
    assert fn({"E0": {(0, 1)}, "E1": set(), "E2": set()}, 2) == {(0, 0), (1, 0), (1, 1)}
    for ast in (("NOT", E0), ("U",), ("COMP", ("NOT", E0), E1), ("OR", E0, ("U",)), ("T", ("NOT", E1)),
                ("MINUS", ("NOT", E0), E1), ("PLUS", ("U",))):
        with pytest.raises(DenseAnswer):
            compile_ast(ast)
    assert issubclass(DenseAnswer, ValueError)


# ───────────────────────────────────────────── encoders
def crafter_state(front="tree", under="grass", facing=(0, 1), zombie_at=None, inv=None):
    mats = {(x, y): "grass" for x in range(-3, 4) for y in range(-3, 4)}
    mats[(0, 0)] = under
    mats[facing] = front
    mats[(1, 0)] = "water"
    objs = {zombie_at: "zombie"} if zombie_at else {}
    return {"pos": (0, 0), "facing": facing, "inv": inv or {"wood": 1}, "mats": mats, "objs": objs}


def test_the_egocentric_encoding():
    s = crafter_state(zombie_at=(-1, -1))
    r = encode(s)
    assert r.n == 10 and {"EAST", "SOUTH", "AT", "FACE", "MAT_tree", "MAT_water", "CRT_zombie"} <= set(r.names)
    assert not any(nm.startswith(("INV", "inv")) for nm in r.names)    # the inventory stays flat atoms
    assert len(r.pairs("EAST")) == 6 and len(r.pairs("SOUTH")) == 6
    assert r.pairs("AT") == {(0, 5)}                                     # the centre cell of the 3×3 window
    assert r.pairs("FACE") == {(0, 8)}                                   # facing (0, 1): the cell below the centre
    assert r.pairs("MAT_tree") == {(8, 8)} and r.pairs("MAT_water") == {(6, 6)}
    assert r.pairs("CRT_zombie") == {(1, 1)}
    assert readout(evaluate(("COMP", ("R", "FACE"), ("R", "MAT_tree")), r))
    assert readout(evaluate(("COMP", ("COMP", ("R", "AT"), ("R", "EAST")), ("R", "MAT_water")), r))
    r5 = encode(s, window=5)
    assert r5.n == 26 and len(r5.pairs("EAST")) == 20 and r5.pairs("AT") == {(0, 13)}
    assert r5.names == encode(crafter_state(front="stone"), window=5).names    # the names do not depend on the state


def test_the_path_tree_encoding(tmp_path):
    r = encode_paths(["/a/secret/b/c.txt", "/a/x.md", "/a/secret/"], focus="/a/secret/b/c.txt", names=("secret",))
    nodes = r.nodes
    assert nodes[0] == "/a/secret/b/c.txt"
    i = {p: k for k, p in enumerate(nodes)}
    assert (i["/a/secret/b"], i["/a/secret/b/c.txt"]) in r.pairs("PARENT")
    assert (i["/"], i["/a"]) in r.pairs("PARENT")
    assert (i["/a/x.md"], i["/a/secret"]) in r.pairs("SAME_DIR") and (i["/a/secret"], i["/a/x.md"]) in r.pairs("SAME_DIR")
    assert all(x != y for x, y in r.pairs("SAME_DIR"))
    assert (i["/a"], i["/a"]) in r.pairs("IS_DIR") and (i["/a/x.md"], i["/a/x.md"]) not in r.pairs("IS_DIR")
    assert r.pairs("NAME=secret") == {(i["/a/secret"], i["/a/secret"])}
    anc_secret = ("COMP", ("PLUS", ("T", ("R", "PARENT"))), ("R", "NAME=secret"))
    assert readout(evaluate(anc_secret, r))
    (tmp_path / "d" / "secret").mkdir(parents=True)
    (tmp_path / "d" / "secret" / "f.txt").write_text("x")
    (tmp_path / "d" / "g.txt").write_text("y")
    rd = encode_paths(str(tmp_path / "d"), names=("secret",))
    assert any(p.endswith("secret") for p in rd.nodes) and len(rd.pairs("NAME=secret")) == 1
    assert len(rd.pairs("IS_DIR")) == 2                                  # d and d/secret


# ───────────────────────────────────────────── the proposer and the held-out test
DIRS = ("a", "b", "c", "data", "src", "secret")


def tree_paths(rng):
    out = []
    for _ in range(12):
        depth = rng.randint(1, 4)
        parts = [rng.choice(DIRS) for _ in range(depth)]
        name = rng.choice(("f", "g", "secret")) + rng.choice((".txt", ".md", ""))    # a file may be named "secret"
        out.append("/" + "/".join(parts) + "/" + name)
    return out


def policy(path):
    return "denied" if "secret" in path.split("/")[:-1] else "ok"


def flat_grammar():
    atoms = [Atom(f"ext={e}", lambda s, e=e: s["path"].endswith("." + e), feature="ext", code=f"s['path'].endswith('.{e}')")
             for e in ("txt", "md")]
    atoms += [Atom(f"depth>={d}", lambda s, d=d: s["path"].count("/") >= d, feature="depth",
                   code=f"s['path'].count('/') >= {d}") for d in (2, 3, 4)]
    atoms += [Atom("parent=secret", lambda s: s["path"].split("/")[-2] == "secret", feature="parent",
                   code="s['path'].split('/')[-2] == 'secret'")]
    return Grammar(atoms, k=3)


def path_rows(seed, groups):
    rng = random.Random(seed)
    out = []
    for g in groups:
        for p in tree_paths(rng):
            out.append(Transition({"path": p}, "read", policy(p), group=g))
    return out


def run_policy(proposer_factory):
    g = flat_grammar()
    held = Buffer(g)
    held.extend(path_rows(1, [f"h{i}" for i in range(6)]))
    ls = LawSet(g, base=lambda s, a: "ok", min_support=3, sized=False)   # 72 held-out rows: the fixed thresholds
    ls.proposer = proposer_factory(g, held)
    for t in path_rows(2, [f"s{i}" for i in range(6)]):
        ls.step(t, held, remember=True)
    test = path_rows(3, [f"t{i}" for i in range(10)])
    acc = sum(ls.predict(t.state, t.action) == t.observed for t in test) / len(test)
    return ls, held, acc


def test_the_relational_proposer_finds_an_ancestor_rule_the_flat_grammar_cannot_say():
    enc = PathTree(key="path", names=("secret",))
    ls, held, acc = run_policy(lambda g, held: RelAlgProposer(g, k=20, encoder=enc, rows=held, size=5))
    _, _, flat_acc = run_policy(lambda g, held: Enumerate())
    rel_laws = [law for law in ls.laws if any(c.startswith("rel:") for c in law.condition)]
    assert rel_laws and all(law.proposer == "relalg" for law in rel_laws)
    assert acc == 1.0 and flat_acc < 1.0
    atom = ls.grammar.by_name[next(c for c in rel_laws[0].condition if c.startswith("rel:"))]
    assert atom.feature == "rel" and atom.code and atom.defs
    chk = check_compiled(ls.to_module(), held, ls)                      # the relational atom compiled to set code
    assert chk.rows == len(held) and chk.differences == 0


def test_a_relational_atom_reads_out_the_player_row_and_compiles():
    enc = PathTree(key="path", names=("secret",))
    ast = ("COMP", ("PLUS", ("T", ("R", "PARENT"))), ("R", "NAME=secret"))
    atom = relational_atom(ast, enc)
    assert atom.name == "rel:" + show(ast) and atom.feature == "rel"
    assert atom({"path": "/a/secret/b/c.txt"}, "read") and not atom({"path": "/a/b/c.txt"}, "read")
    ns: dict = {}
    for d in atom.defs:
        exec(d, ns)
    for p in ("/a/secret/b/c.txt", "/a/b/c.txt", "/secret/x", "/x/secret"):
        assert bool(eval(atom.code, ns, {"s": {"path": p}, "a": "read"})) == atom({"path": p}, "read")
    # a state's graph is bounded (the window, the path's chain): a complement there compiles against its nodes
    neg = relational_atom(("NOT", ("R", "PARENT")), enc)
    ns2: dict = {}
    for d in neg.defs:
        exec(d, ns2)
    for p in ("/a/b", "/"):
        assert bool(eval(neg.code, ns2, {"s": {"path": p}, "a": "read"})) == neg({"path": p}, "read")
