"""solvi.experimental.relations.CompositionTable: a composition table of relations learned by exact counting from
chains of length 2 (a cell is admitted at majority share ≥ m and support ≥ s), empty cells filled by residuals from
chains of length 3, a chain of any length derived by composition over every bracketing (CYK) with an agreement rule,
a refusal with its reason when the table cannot derive the answer, the cells used as the answer's reason, the export
as rule items of a KnowledgeStore, and the sized acceptance with a base-rate null as an optional admission hook."""
from __future__ import annotations

import itertools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import CompositionTable, Derivation, base_rates, sized_admit

from solvi.core.knowledge import KnowledgeStore

# a planted algebra over three relations: the cyclic group of order 3 written as names (associative, so every
# bracketing of a chain agrees); "r0" composes as the neutral element but is an ordinary label to the learner.
Z3 = {(a, b): f"r{(int(a[1]) + int(b[1])) % 3}" for a in ("r0", "r1", "r2") for b in ("r0", "r1", "r2")}


def z3_pairs(times=2):
    return [((a, b), r) for (a, b), r in sorted(Z3.items()) for _ in range(times)]


def z3_answer(chain):
    return f"r{sum(int(x[1]) for x in chain) % 3}"


def test_fit_learns_every_cell_of_a_planted_algebra_by_exact_counting():
    t = CompositionTable().fit(z3_pairs())
    assert len(t) == 9 and t.cells == Z3
    c = t.cell("r1", "r2")
    assert (c.relation, c.hits, c.support, c.origin) == ("r0", 2, 2, "pair")


def test_derive_composes_long_chains_over_every_bracketing():
    t = CompositionTable().fit(z3_pairs())
    rng = random.Random(0)
    for n in range(1, 11):
        chain = tuple(rng.choice(("r0", "r1", "r2")) for _ in range(n))
        d = t.derive(chain)
        assert isinstance(d, Derivation) and d.answered and d.relation == z3_answer(chain) and d.reason == "derived"
    d = t.derive(("r1", "r1", "r2", "r1"))
    assert d.bracketings == 5 and d.candidates == {"r2": 5}              # Catalan(3) bracketings, all agree


def test_explain_lists_the_cells_used():
    t = CompositionTable().fit(z3_pairs())
    used = t.explain(("r1", "r1", "r1"))
    assert {(c.left, c.right, c.relation) for c in used} == {("r1", "r1", "r2"), ("r2", "r1", "r0"),
                                                              ("r1", "r2", "r0")}
    assert t.derive(("r1", "r1", "r1")).cells == used
    assert t.explain(("r1", "zz")) == ()


def test_refuses_an_underivable_chain_with_the_reason():
    pairs = [p for p in z3_pairs() if p[0] != ("r1", "r2")]              # one cell never observed
    t = CompositionTable().fit(pairs)
    d = t.derive(("r1", "r2"))
    assert not d.answered and d.relation is None and d.reason == "underivable" and d.cells == ()
    d = t.derive(("r1", "unknown", "r2"))
    assert not d.answered and d.reason == "underivable"
    # the missing cell is bypassed where another bracketing derives the chain: r1 r2 r2 = r1 (r2 r2) = r1 r1 = r2
    d = t.derive(("r1", "r2", "r2"))
    assert d.answered and d.relation == "r2" and d.candidates == {"r2": 1}


def test_refuses_when_bracketings_disagree():
    # a non-associative planted table: (a∘b)∘c = c∘c = a, a∘(b∘c) = a∘a = b
    pairs = [((x, y), r) for (x, y), r in [(("a", "b"), "c"), (("c", "c"), "a"), (("b", "c"), "a"), (("a", "a"), "b")]
             for _ in range(3)]
    t = CompositionTable().fit(pairs)
    d = t.derive(("a", "b", "c"))
    assert not d.answered and d.reason == "disagree" and d.candidates == {"a": 1, "b": 1}
    m = CompositionTable(agree="majority").fit(pairs)
    assert not m.derive(("a", "b", "c")).answered                         # a tie stays a refusal
    assert CompositionTable(agree="majority").fit(pairs).derive(("a", "b")).relation == "c"


def test_majority_and_support_thresholds():
    pairs = [(("a", "b"), "x")] * 9 + [(("a", "b"), "y")] + [(("b", "a"), "x")] * 8 + [(("b", "a"), "y")] * 2
    pairs += [(("a", "a"), "z")]
    t = CompositionTable(majority=0.9, support=2).fit(pairs)
    assert t.cells == {("a", "b"): "x"}                                  # 0.9 in; 0.8 out; support 1 out
    assert t.cell("a", "b").hits == 9 and t.cell("a", "b").support == 10
    assert CompositionTable(majority=0.8, support=1).fit(pairs).cells == {("a", "b"): "x", ("b", "a"): "x",
                                                                          ("a", "a"): "z"}
    assert t.conflicts == 2                                              # cells whose votes name two relations


def test_residual_fill_from_chains_of_length_three():
    pairs = [(("a", "b"), "x")] * 2
    triples = [(("a", "b", "c"), "y")] * 2                               # a∘b = x is known, so x∘c = y
    t = CompositionTable().fit(pairs + triples)
    c = t.cell("x", "c")
    assert c is not None and c.relation == "y" and c.origin == "residual" and c.support == 2
    assert t.derive(("a", "b", "c")).relation == "y"
    assert CompositionTable(residual=False).fit(pairs + triples).cell("x", "c") is None
    # the other side: b∘c = x is known, so a∘x = y
    t2 = CompositionTable().fit([(("b", "c"), "x")] * 2 + [(("a", "b", "c"), "y")] * 2)
    assert t2.cell("a", "x").relation == "y"


def test_residual_fill_runs_to_a_fixed_point_and_counts_each_chain_once_per_round():
    pairs = [(("a", "b"), "x")] * 2
    triples = [(("a", "b", "c"), "y")] * 2 + [(("x", "c", "d"), "z")] * 2   # the second needs x∘c from the first
    t = CompositionTable().fit(pairs + triples)
    assert t.cell("y", "d").relation == "z" and t.rounds >= 2
    one = CompositionTable().fit(pairs + [(("a", "b", "c"), "y")])          # a single chain: support 1, not admitted
    assert one.cell("x", "c") is None


def test_fit_is_deterministic_and_order_free():
    data = z3_pairs(3) + [(("r1", "r1", "r1"), "r0")] * 2
    a = CompositionTable().fit(data)
    shuffled = list(data)
    random.Random(7).shuffle(shuffled)
    b = CompositionTable().fit(shuffled)
    assert a.cells == b.cells and [c for c in a] == [c for c in b]
    for chain in itertools.product(("r0", "r1", "r2"), repeat=4):
        assert a.derive(chain) == b.derive(chain)


def test_identity_option_learns_loops_from_length_three_chains():
    # x∘y∘c = c for every c ⟹ x∘y is the identity; the identity composes as neutral and is never an answer
    pairs = [(("p", "c"), "q")] * 2
    triples = [(("x", "y", "c"), "c")] * 2 + [(("x", "y", "p"), "p")] * 2
    plain = CompositionTable().fit(pairs + triples)
    assert not plain.derive(("x", "y", "c")).answered
    t = CompositionTable(identity="self").fit(pairs + triples)
    assert t.cell("x", "y").relation == "self"
    assert t.derive(("x", "y", "c")).relation == "c" and t.derive(("x", "y", "p", "c")).relation == "q"
    d = t.derive(("x", "y"))
    assert not d.answered and d.reason == "identity"


def test_to_knowledge_writes_rule_items_with_support_and_the_store_verifies():
    t = CompositionTable().fit(z3_pairs(3))
    store = KnowledgeStore()
    ids = t.to_knowledge(store, {"domain": "z3"})
    assert len(ids) == 9 and len(set(ids.values())) == 9
    iid = ids[("r1", "r2")]
    item = store.items[iid]
    assert item.kind == "rule" and item.body["o"] == "r0" and item.body["s"] == ["r1", "r2"]
    assert item.asserts[-1]["evidence"]["support"] == 3 and item.asserts[-1]["evidence"]["origin"] == "pair"
    assert store.verify()
    n = len(store.journal)
    assert t.to_knowledge(store, {"domain": "z3"}) == ids and len(store.journal) == n      # idempotent


def test_admit_hook_gates_cells_with_the_sized_rule_and_a_base_rate_null():
    rates = base_rates([((f"a{i}", "b"), f"r{i % 18}") for i in range(180)])
    assert abs(rates["r0"] - 1 / 18) < 1e-12 and abs(sum(rates.values()) - 1) < 1e-12
    admit = sized_admit(rates, delta=0.1)
    # P[Binomial(2, 1/18) ≥ 2] = 0.0031 > 0.1 / 100; three of three = 1.7e-4 passes
    assert not admit(("a", "b"), "r0", 2, 2, 100)
    assert admit(("a", "b"), "r0", 3, 3, 100)
    assert admit(("a", "b"), "r0", 2, 2, 10)                             # fewer cells tested: a looser level
    pairs = [(("a", "b"), "r0")] * 2 + [(("b", "a"), "r1")] * 3 + [((f"c{i}", "d"), "r2") for i in range(98)]
    t = CompositionTable(support=1, admit=admit).fit(pairs)
    assert t.cell("a", "b") is None and t.cell("b", "a").relation == "r1"
    assert t.tested >= 100 and t.refused_by_admit >= 1
    with pytest.raises(ValueError):
        sized_admit(rates, delta=1.5)


def test_validation_of_arguments():
    with pytest.raises(ValueError):
        CompositionTable(majority=0.4)
    with pytest.raises(ValueError):
        CompositionTable(support=0)
    with pytest.raises(ValueError):
        CompositionTable(agree="some")
    with pytest.raises(ValueError):
        CompositionTable().derive(())
