"""solvi.experimental.relations.forward_chain: forward chaining to a fixed point over composition rules kept in a
KnowledgeStore — derived facts written through the store's gates with their premises and the rule as derived_from,
a derivation never ranked above its premises, two derived values of one pair from equal-rank sources a dispute (the
table's «bracketings disagree → refuse» as a question for a person), JTMS retraction and promotion without a re-run,
equivalence with CompositionTable.derive on chains, explain() as the derivation read back from the store."""
from __future__ import annotations

import itertools
import math
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import (ChainLimit, ChainReport, CompositionTable, explain, forward_chain,
                                              pair_fact)

from solvi.core.knowledge import KnowledgeStore, Verdict

# the planted algebra of test_relations_table.py: the cyclic group of order 3 written as names (associative)
Z3 = {(a, b): f"r{(int(a[1]) + int(b[1])) % 3}" for a in ("r0", "r1", "r2") for b in ("r0", "r1", "r2")}
RULES = {"domain": "kin"}


def z3_pairs(times=2):
    return [((a, b), r) for (a, b), r in sorted(Z3.items()) for _ in range(times)]


def z3_answer(chain):
    return f"r{sum(int(x[1]) for x in chain) % 3}"


def story(store, chain, scope, source="outcome"):
    """A path story 0 → 1 → … → n, edge i labelled chain[i] → the fact ids in order."""
    return [store.add("fact", pair_fact(i, i + 1, r), scope, source=source, by="story") for i, r in enumerate(chain)]


def store_with(table, gates=None):
    ks = KnowledgeStore(gates=gates)
    table.to_knowledge(ks, RULES)
    return ks


def values(ks, x, y, scope):
    """{relation: status} of the live items on the pair (x, y) in scope."""
    out = {}
    for rec in ks.find(kind="fact", s=[x, y], scope=scope):
        if rec["scope"] == scope and rec["status"] not in ("retracted", "refuted", "expired"):
            out[rec["body"]["o"]] = rec["status"]
    return out


def outcome_of(ks, x, y, scope):
    """The store's reading of a pair: ("answered", relation, status) | ("disputed", values) | ("absent",)."""
    v = values(ks, x, y, scope)
    if not v:
        return ("absent",)
    if len(v) == 1:
        (r, st), = v.items()
        return ("answered", r, st)
    assert set(v.values()) == {"disputed"}
    return ("disputed", tuple(sorted(v)))


def run(ks, scope, **kw):
    return forward_chain(ks, scope=scope, rule_scope=RULES, **kw)


# ───────────────────────────── the store: an inferred outcome competes by rank
def test_an_inferred_outcome_disputes_where_an_observed_one_refutes():
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "p", "r": "q", "o": 1}, source="outcome")
    b = ks.add("fact", {"s": "p", "r": "w", "o": 1}, source="outcome")
    x = ks.add("fact", {"s": "k", "r": "v", "o": "A"}, source="outcome", derived_from=[a], inferred=True)
    y = ks.add("fact", {"s": "k", "r": "v", "o": "B"}, source="outcome", derived_from=[b], inferred=True)
    assert ks.status(x) == ks.status(y) == "disputed" and len(ks.disputes()) == 1
    plain = KnowledgeStore()                                   # the same without inferred: the later outcome refutes
    a = plain.add("fact", {"s": "p", "r": "q", "o": 1}, source="outcome")
    b = plain.add("fact", {"s": "p", "r": "w", "o": 1}, source="outcome")
    x = plain.add("fact", {"s": "k", "r": "v", "o": "A"}, source="outcome", derived_from=[a])
    y = plain.add("fact", {"s": "k", "r": "v", "o": "B"}, source="outcome", derived_from=[b])
    assert (plain.status(x), plain.status(y)) == ("refuted", "active")
    obs = ks.add("fact", {"s": "k", "r": "v", "o": "B"}, source="outcome")   # an observation ends the dispute
    assert ks.status(obs) == "active" and ks.status(y) == "active" and ks.status(x) == "refuted"
    assert ks.verify() and ks.rebuild().fingerprint() == ks.fingerprint()
    with pytest.raises(ValueError):
        ks.add("fact", {"s": "z", "r": "v", "o": 1}, source="outcome", inferred=True)   # no premises named


# ───────────────────────────── a planted algebra: the fixed point and the derivation
def test_a_story_over_a_planted_algebra_reaches_its_fixed_point_in_log_depth_rounds():
    table = CompositionTable().fit(z3_pairs())
    rng = random.Random(1)
    chain = tuple(rng.choice(("r0", "r1", "r2")) for _ in range(9))
    ks = store_with(table)
    sc = {"story": "s1"}
    base = story(ks, chain, sc)
    rep = run(ks, sc)
    assert isinstance(rep, ChainReport) and rep.fixed_point
    n = len(chain)
    assert rep.rounds == math.ceil(math.log2(n))                # spans double each round
    assert rep.facts_added == n * (n + 1) // 2 - n               # every span of length ≥ 2, once
    assert rep.disputes_opened == 0 and rep.hypotheses == 0 and rep.refused == 0
    assert sum(r["added"] for r in rep.per_round) == rep.facts_added and len(rep.per_round) == rep.rounds
    for i, j in itertools.combinations(range(n + 1), 2):
        assert outcome_of(ks, i, j, sc) == ("answered", z3_answer(chain[i:j]), "active")
    head = ks.active(ks.items[ks.find(kind="fact", s=[0, n], scope=sc)[0]["id"]].key)
    rec = ks.item(head)
    assert rec["source"] == "outcome" and rec["by"] == "relations:forward_chain"
    ev = rec["evidence"][0]
    assert set(ev) == {"rule", "via", "depth"} and 1 <= ev["depth"] <= rep.rounds
    ex = explain(ks, head)
    assert set(ex.facts) == set(base)                            # down to the original facts
    assert all(ks.items[r].kind == "rule" for r in ex.rules)
    assert set(ex.cells) == {(c.left, c.right, c.relation) for c in table.derive(chain).cells}
    tree = ex.tree()
    assert tree["body"]["o"] == z3_answer(chain) and len(tree["premises"]) == 2 and tree["rule"]["cell"]
    assert ks.verify()


def test_derived_from_names_two_facts_and_a_rule_and_the_source_is_the_weaker_premise():
    table = CompositionTable().fit(z3_pairs())
    ks = store_with(table)
    sc = {"story": "mixed"}
    a = ks.add("fact", pair_fact("x", "y", "r1"), sc, source="outcome")
    b = ks.add("fact", pair_fact("y", "z", "r1"), sc, source="person")
    run(ks, sc)
    (rec,) = ks.find(kind="fact", s=["x", "z"], scope=sc)
    assert rec["body"]["o"] == "r2" and rec["source"] == "person"     # never above the weaker premise
    prem = set(rec["derived_from"])
    assert {a, b} < prem and len(prem) == 3
    (rule,) = prem - {a, b}
    assert ks.items[rule].body == {"kind": "composition", "s": ["r1", "r1"], "r": "compose", "o": "r2"}
    assert rec["evidence"][0]["via"] == "y" and rec["evidence"][0]["rule"] == rule


# ───────────────────────────── equivalence with CompositionTable.derive
def random_table(rng, names=("a", "b", "c"), fill=0.8):
    """A random (generally non-associative) table: each cell present with probability `fill`."""
    pairs = []
    for x, y in itertools.product(names, repeat=2):
        if rng.random() < fill:
            pairs += [((x, y), rng.choice(names))] * 2
    return CompositionTable().fit(pairs)


def test_equivalence_with_the_table_on_random_chains_and_tables():
    rng = random.Random(0)
    seen, differences = {"derived": 0, "disagree": 0, "underivable": 0}, []
    for t in range(12):
        table = random_table(rng)
        ks = store_with(table)
        for s in range(10):
            chain = tuple(rng.choice(("a", "b", "c")) for _ in range(rng.randint(2, 7)))
            sc = {"story": f"{t}-{s}"}
            story(ks, chain, sc)
            run(ks, sc)
            d = table.derive(chain)
            seen[d.reason] += 1
            got = outcome_of(ks, 0, len(chain), sc)
            if d.reason == "derived":
                ok = got[:2] == ("answered", d.relation)
            elif d.reason == "disagree":
                ok = got == ("disputed", tuple(sorted(d.candidates)))
            else:
                ok = got == ("absent",)
            if not ok:
                differences.append((chain, d.reason, d.relation, got))
            if d.reason == "derived" and got[0] == "answered":
                ex = explain(ks, ks.find(kind="fact", s=[0, len(chain)], scope=sc, status=got[2])[0]["id"])
                assert set(ex.cells) == {(c.left, c.right, c.relation) for c in d.cells}
        assert ks.verify()
    assert differences == []                                     # the expectation written ahead: none
    assert min(seen.values()) >= 5                                # every kind of answer occurred


def test_a_disagreement_is_a_dispute_and_a_persons_answer_settles_it_without_a_rerun():
    # the non-associative table of test_relations_table.py: (a∘b)∘c = c∘c = a, a∘(b∘c) = a∘a = b
    pairs = [((x, y), r) for (x, y), r in [(("a", "b"), "c"), (("c", "c"), "a"), (("b", "c"), "a"), (("a", "a"), "b"),
                                           (("a", "d"), "c"), (("b", "d"), "b")] for _ in range(3)]
    table = CompositionTable().fit(pairs)
    ks = store_with(table)
    sc = {"story": "d"}
    story(ks, ("a", "b", "c", "d"), sc)
    rep = run(ks, sc)
    assert outcome_of(ks, 0, 3, sc) == ("disputed", ("a", "b")) and rep.disputes_opened >= 1
    assert table.derive(("a", "b", "c")).reason == "disagree"
    # (0, 4) rests on the disputed span: a∘d = c, b∘d = b → disputed downstream too (CYK: {"b", "c"} → disagree)
    assert outcome_of(ks, 0, 4, sc) == ("disputed", ("b", "c"))
    assert table.derive(("a", "b", "c", "d")).reason == "disagree"
    key = ks.items[ks.find(kind="fact", s=[0, 3], scope=sc)[0]["id"]].key
    ks.resolve(key, "a", by="ann")                               # the person's answer: (0, 3) is "a"
    assert outcome_of(ks, 0, 3, sc) == ("answered", "a", "active")
    assert outcome_of(ks, 0, 4, sc) == ("answered", "c", "active")   # the other side's consequence went with it
    assert forward_chain(ks, scope=sc, rule_scope=RULES).facts_added == 0
    assert ks.verify()


# ───────────────────────────── hypothesis rules, promotion and retraction (JTMS)
class HoldRules:
    """A write gate that holds every rule (a hypothesis until promoted)."""
    name = "hold-rules"

    def admit(self, store, item, shadow=None):
        return Verdict(item["kind"] != "rule", "rules wait for a person")


def test_hypothesis_rules_give_hypotheses_promotion_activates_and_retraction_takes_them_back():
    from solvi.core.knowledge.gates import default_gates
    table = CompositionTable().fit(z3_pairs())
    ks = store_with(table, gates=default_gates() + [HoldRules()])
    sc = {"story": "h"}
    base = story(ks, ("r1", "r1", "r2"), sc)
    assert run(ks, sc).facts_added == 0                          # rules="active": no active rule, nothing derived
    rep = run(ks, sc, rules="any")
    assert rep.facts_added == 3 and rep.hypotheses == 3
    assert outcome_of(ks, 0, 2, sc) == ("answered", "r2", "hypothesis")
    for cell in (("r1", "r1"), ("r2", "r2"), ("r1", "r2"), ("r2", "r1")):
        rid = next((r["id"] for r in ks.find(kind="rule", scope=RULES) if tuple(r["body"]["s"]) == cell), None)
        if rid is not None:
            ks.promote(rid)
    assert outcome_of(ks, 0, 2, sc) == ("answered", "r2", "active")   # promoted without re-running the chain
    assert outcome_of(ks, 0, 3, sc) == ("answered", "r1", "active")
    ks.retract(base[1], by="ann", why="wrong edge")
    assert outcome_of(ks, 0, 2, sc) == ("absent",) and outcome_of(ks, 0, 3, sc) == ("absent",)
    assert outcome_of(ks, 2, 3, sc) == ("answered", "r2", "active")    # an original edge stays
    assert ks.verify()


def test_a_cycle_terminates_and_no_derived_fact_supports_itself():
    table = CompositionTable().fit(z3_pairs())
    ks = store_with(table)
    sc = {"story": "cycle"}
    a = ks.add("fact", pair_fact(0, 1, "r1"), sc, source="person")
    ks.add("fact", pair_fact(1, 2, "r1"), sc, source="person")
    ks.add("fact", pair_fact(2, 0, "r1"), sc, source="person")
    rep = run(ks, sc)
    assert rep.fixed_point and rep.facts_added == 6               # (0,2) (1,0) (2,1) and the loops (0,0) (1,1) (2,2)
    assert outcome_of(ks, 0, 0, sc) == ("answered", "r0", "active")
    assert rep.circular > 0                                       # justifications that would rest on themselves
    ks.retract(a, by="ann")
    # every derived fact that needs edge (0, 1) goes: (0,2), (2,1), the three loops; (1, 0) = r1∘r1 over 1→2→0 stays
    assert outcome_of(ks, 1, 0, sc) == ("answered", "r2", "active")
    for x, y in ((0, 2), (2, 1), (0, 0), (1, 1), (2, 2)):
        assert outcome_of(ks, x, y, sc) == ("absent",), (x, y)
    assert ks.verify()


# ───────────────────────────── idempotence, determinism, scopes, the guards
def test_idempotent_deterministic_and_verified():
    table = CompositionTable().fit(z3_pairs())
    chain = ("r1", "r2", "r2", "r0", "r1", "r1")
    fps = []
    for _ in range(2):
        ks = store_with(table)
        story(ks, chain, {"story": "i"})
        first = run(ks, {"story": "i"})
        n = len(ks.journal)
        again = run(ks, {"story": "i"})
        assert again.facts_added == 0 and again.facts_proposed == 0 and len(ks.journal) == n
        assert first.facts_added > 0 and ks.verify()
        fps.append(ks.fingerprint())
    assert fps[0] == fps[1]


def test_scopes_do_not_mix_and_scope_none_chains_each_scope_apart():
    table = CompositionTable().fit(z3_pairs())
    ks = store_with(table)
    ks.add("fact", pair_fact(0, 1, "r1"), {"story": "A"}, source="person")
    ks.add("fact", pair_fact(1, 2, "r1"), {"story": "B"}, source="person")
    rep = forward_chain(ks, rule_scope=RULES)
    assert rep.facts_added == 0
    ks.add("fact", pair_fact(1, 2, "r2"), {"story": "A"}, source="person")
    assert forward_chain(ks, rule_scope=RULES).facts_added == 1
    assert outcome_of(ks, 0, 2, {"story": "A"}) == ("answered", "r0", "active")


def test_the_guards_max_rounds_and_the_hard_stop():
    table = CompositionTable().fit(z3_pairs())
    ks = store_with(table)
    story(ks, ("r1",) * 8, {"story": "g"})
    rep = run(ks, {"story": "g"}, max_rounds=1)
    assert rep.rounds == 1 and not rep.fixed_point and rep.facts_added == 7
    assert run(ks, {"story": "g"}).fixed_point                   # resumes to the fixed point
    ks2 = store_with(table)
    story(ks2, ("r1",) * 8, {"story": "g"})
    with pytest.raises(ChainLimit, match="max_facts"):
        run(ks2, {"story": "g"}, max_facts=5)
    with pytest.raises(ValueError):
        run(ks2, {"story": "g"}, rules="some")


def test_a_store_on_disk_replays_the_chain_and_its_disputes(tmp_path):
    pairs = [((x, y), r) for (x, y), r in [(("a", "b"), "c"), (("c", "c"), "a"), (("b", "c"), "a"), (("a", "a"), "b")]
             for _ in range(3)]
    path = tmp_path / "knowledge.jsonl"
    ks = KnowledgeStore(str(path))
    CompositionTable().fit(pairs).to_knowledge(ks, RULES)
    story(ks, ("a", "b", "c"), {"story": "disk"})
    run(ks, {"story": "disk"})
    assert outcome_of(ks, 0, 3, {"story": "disk"}) == ("disputed", ("a", "b"))
    again = KnowledgeStore(str(path))                              # replayed from the journal: inferred kept
    assert again.fingerprint() == ks.fingerprint() and again.verify()
    assert outcome_of(again, 0, 3, {"story": "disk"}) == ("disputed", ("a", "b"))
