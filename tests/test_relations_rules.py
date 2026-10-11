"""solvi.experimental.relations.GuardedRules / chain_rules: Datalog-lite rules learned by exact counting on labelled
query pairs — bodies r ← s, r ← s ∧ u, r ← s1 ∘ s2, r ← s1 ∘ s2 ∧ u(z), r ← s(x,z) ∧ z ≠ y ∧ u(y), and a negative
stratum r ← u1(x) ∧ u2(y) ∧ ¬s(x,y) applied after the positive fixed point; residual votes for bodies whose
intermediate fact is not yet derivable; rounds to a fixed point of the rule set; pairs carrying a set of relations;
the same answers from the in-memory fixed point and from a KnowledgeStore; explain over guards and the negative
stratum. A planted world with an independent oracle."""
from __future__ import annotations

import functools
import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import (GuardedRule, GuardedRules, Story, chain_rules, explain, forward_chain,
                                              pair_fact, unary_fact, CompositionTable)

from solvi.core.knowledge import KnowledgeStore

COMPLEMENTS = {"Not_living_in": "living_in"}


# ───────────────────────────── the planted world and its oracle (written independently of the learner)
def world(seed, families=3, places=3, known=0.5):
    """One story: families of three generations, colleagues, places; each person's home is drawn on its own (homes
    do not follow families). Returns (story, truth) where truth maps an ordered pair to the set of every relation
    that holds on it (the oracle; a residence holds when it is given or reachable through living_in_same_place)."""
    rng = random.Random(seed)
    nid = iter(range(10_000))
    unary, edges = [], []
    male, persons, place = {}, [], []
    parents = {}                                        # child → [parents]
    for _ in range(places):
        p = next(nid)
        place.append(p)
        unary.append((p, "is_place"))
    for _ in range(families):
        g = [next(nid) for _ in range(2)]               # a couple: one male, one female
        kids = [next(nid) for _ in range(rng.randint(2, 3))]
        grand = [next(nid) for _ in range(rng.randint(1, 3))]
        male[g[0]], male[g[1]] = True, False
        for k in kids + grand:
            male[k] = rng.random() < 0.5
        for k in kids:
            parents[k] = list(g)
        for c in grand:
            parents[c] = [kids[0]]
        persons += g + kids + grand
    home = {x: rng.choice(place) for x in persons}
    for x in persons:
        unary.append((x, "is_person"))
        unary.append((x, "is_male" if male[x] else "is_female"))
    for c, ps in sorted(parents.items()):
        for p in ps:
            edges.append((p, c, "parent_of"))
    for x in persons:                                   # colleagues / housemates: people of one home, one way
        for y in persons:
            if x < y and home[x] == home[y]:
                d = rng.random()
                if d < 0.12:
                    edges.append((x, y, "colleague_of"))
                elif d < 0.24:
                    edges.append((x, y, "living_in_same_place"))
    for x in persons:
        if rng.random() < known:
            edges.append((x, home[x], "living_in"))
    rng.shuffle(edges)
    return Story.from_facts(edges, unary), oracle(edges, unary)


def oracle(edges, unary):
    """Every relation on every ordered pair, by direct definitions (not by rules)."""
    U = {}
    for x, u in unary:
        U.setdefault(x, set()).add(u)
    parent = {(a, b) for a, b, r in edges if r == "parent_of"}
    coll = {(a, b) for a, b, r in edges if r == "colleague_of"}
    lsp0 = {(a, b) for a, b, r in edges if r == "living_in_same_place"}
    lives0 = {(a, b) for a, b, r in edges if r == "living_in"}
    people = sorted(x for x in U if "is_person" in U[x])
    places = sorted(x for x in U if "is_place" in U[x])
    lsp = set(lsp0) | set(coll)                         # both ways, transitively over the given edges
    lsp |= {(b, a) for a, b in lsp}
    while True:
        more = {(a, c) for a, b in lsp for b2, c in lsp if b == b2 and a != c} - lsp
        if not more:
            break
        lsp |= more
    lives = set(lives0)
    while True:
        more = {(a, p) for a, b in lsp for b2, p in lives if b == b2} - lives
        if not more:
            break
        lives |= more
    T = {}

    def put(a, b, r):
        if a != b:
            T.setdefault((a, b), set()).add(r)
    for a, b, r in edges:
        put(a, b, r)
    for a, b in parent:
        put(b, a, "child_of")
        put(a, b, "father_of" if "is_male" in U[a] else "mother_of")
    for a, b in parent:
        for b2, c in parent:
            if b == b2:
                put(a, c, "grandparent_of")
                put(c, a, "grandchild_of")
                put(a, c, "grandfather_of" if "is_male" in U[a] else "grandmother_of")
                if "is_male" in U[b]:
                    put(a, c, "paternal_grandparent_of")
    for a, b in parent:
        for a2, c in parent:
            if a == a2 and b != c:
                put(b, c, "sibling_of")
    for a, b in coll:
        put(b, a, "colleague_of")
    for a, b in lsp:
        put(a, b, "living_in_same_place")
    for a, p in lives:
        put(a, p, "living_in")
    for x in people:
        mine = {p for a, p in lives if a == x}
        for p in places:
            if p not in mine:
                put(x, p, "Not_living_in")             # elsewhere, or (residence unknown) the closed-world default
    return T


def examples(seeds, **kw):
    """Labelled query pairs: every ordered pair of distinct entities of each story, with its truth (possibly empty:
    no relation holds) — every pair labelled, so that no pair is queried because it happens to hold a relation."""
    out = []
    for s in seeds:
        st, T = world(s, **kw)
        ents = sorted(st.entities())
        out += [(st, (x, y), frozenset(T.get((x, y), ()))) for x in ents for y in ents if x != y]
    return out


PLANTED = {
    ("child_of", "B1", (("parent_of", 1),), ()),
    ("father_of", "B1u", (("parent_of", 0),), (("is_male", "x"),)),
    ("grandparent_of", "B2", (("parent_of", 0), ("parent_of", 0)), ()),
    ("paternal_grandparent_of", "B2z", (("parent_of", 0), ("parent_of", 0)), (("is_male", "z"),)),
    ("living_in_same_place", "B1", (("colleague_of", 0),), ()),
    ("Not_living_in", "BNE", (("living_in", 0),), (("is_place", "y"),)),
    ("Not_living_in", "NEG", (("living_in", 0),), (("is_person", "x"), ("is_place", "y"))),
}


@functools.lru_cache(maxsize=None)
def ex8():
    return tuple(examples(range(8)))


@pytest.fixture(scope="module")
def learned():
    return GuardedRules(0.9, 2, complements=COMPLEMENTS).fit(ex8())


def sig(r: GuardedRule):
    return (r.head, r.template, r.atoms, r.unary)


# ───────────────────────────── learning
def test_planted_rules_are_learned(learned):
    got = {sig(r) for r in learned}
    missing = PLANTED - got
    assert not missing, missing
    # a guarded head over a derived relation needs a second round: grandfather ← grandparent ∧ male(x)
    gf = {sig(r): r for r in learned if r.head == "grandfather_of"}
    key = ("grandfather_of", "B1u", (("grandparent_of", 0),), (("is_male", "x"),))
    assert key in gf and gf[key].round >= 2 and learned.rounds >= 2


def test_trivial_and_specific_bodies_are_filtered(learned):
    for r in learned:
        if r.template in ("B1", "B1u"):
            assert r.atoms[0] != (r.head, 0), r                 # r ← r (∧ u) never admitted
    heads_b1 = {(r.head, r.atoms) for r in learned if r.template == "B1"}
    heads_b2 = {(r.head, r.atoms) for r in learned if r.template == "B2"}
    for r in learned:
        if r.template == "B1u":
            assert (r.head, r.atoms) not in heads_b1, r          # the generalisation is admitted → not the guard
        if r.template == "B2z":
            assert (r.head, r.atoms) not in heads_b2, r


def test_every_rule_has_its_counts(learned):
    for r in learned:
        assert r.hits >= 2 and r.hits / r.support >= 0.9 - 1e-12 and r.origin in ("direct", "residual")
        assert r.round >= 1
    assert learned.by_template()["NEG"] >= 1


def test_fit_is_deterministic_and_order_free():
    ex = list(ex8())
    a = GuardedRules(complements=COMPLEMENTS, rounds=2).fit(ex)
    b = GuardedRules(complements=COMPLEMENTS, rounds=2).fit(list(reversed(ex)))
    assert [sig(r) for r in a] == [sig(r) for r in b]
    assert [(r.hits, r.support) for r in a] == [(r.hits, r.support) for r in b]


def test_check_slice_refuses_a_rule_it_contradicts():
    ex = list(ex8())
    rules = GuardedRules(complements=COMPLEMENTS, rounds=2).fit(ex)
    assert any(r.head == "child_of" for r in rules)
    # a check slice where every child_of pair is labelled something else
    bad = [(st, q, frozenset({"stranger"}) if "child_of" in y else y) for st, q, y in examples(range(100, 103))]
    checked = GuardedRules(complements=COMPLEMENTS, rounds=2).fit(ex, check=bad)
    assert ("child_of", "B1", (("parent_of", 1),), ()) not in {sig(r) for r in checked}   # contradicted there
    assert checked.refused_by_check >= 1
    strict = GuardedRules(complements=COMPLEMENTS, rounds=2, check_min=1).fit(ex, check=bad)
    assert not any(r.head == "child_of" for r in strict)          # no child_of hit on the check slice at all


# ───────────────────────────── prediction (in memory)
def test_predictions_on_new_stories_are_exact(learned):
    test = examples(range(500, 505))
    ok = sum(learned.predict(st, q) == y for st, q, y in test)
    assert ok / len(test) >= 0.98, ok / len(test)


def test_predict_returns_a_set_and_refuses_with_empty(learned):
    st, T = world(7)
    assert learned.predict(st, (10**6, 10**6 + 1)) == frozenset()


def test_negative_stratum_only_where_nothing_positive():
    # person 1 lives in 10 (given); 11 is a place: Not_living_in(1, 11) comes from BNE; person 2 has no residence: NEG
    st = Story.from_facts([(1, 10, "living_in")], [(1, "is_person"), (2, "is_person"), (10, "is_place"),
                                                    (11, "is_place")])
    rs = GuardedRules(complements=COMPLEMENTS).with_rules([
        GuardedRule("Not_living_in", "BNE", (("living_in", 0),), (("is_place", "y"),)),
        GuardedRule("Not_living_in", "NEG", (("living_in", 0),), (("is_person", "x"), ("is_place", "y")))])
    c = rs.closure(st)
    assert c.relations(1, 11) == {"Not_living_in"} and c.relations(1, 10) == {"living_in"}
    assert c.relations(2, 10) == {"Not_living_in"} and c.relations(2, 11) == {"Not_living_in"}
    assert c.stratum(2, 10, "Not_living_in") == 2 and c.stratum(1, 11, "Not_living_in") == 1


def test_complement_values_dispute_and_refuse():
    st = Story.from_facts([(1, 10, "living_in"), (1, 2, "colleague_of"), (2, 11, "living_in")],
                          [(1, "is_person"), (2, "is_person"), (10, "is_place"), (11, "is_place")])
    rs = GuardedRules(complements=COMPLEMENTS).with_rules([
        GuardedRule("living_in", "B2", (("colleague_of", 0), ("living_in", 0)), ()),
        GuardedRule("Not_living_in", "BNE", (("living_in", 0),), (("is_place", "y"),))])
    c = rs.closure(st)
    # 1 lives in 10 (given) and in 11 (derived): BNE gives Not_living_in on both → each key disputed → refused
    assert c.status(1, 11, "living_in") == "disputed" and c.status(1, 11, "Not_living_in") == "disputed"
    assert c.answer(1, 11) == frozenset()
    assert c.answer(2, 11) == frozenset({"living_in"})


def test_multi_relation_pairs_are_facts_not_disputes():
    st = Story.from_facts([(1, 2, "nibling_of")], [(1, "is_person"), (1, "is_female"), (2, "is_person")])
    rs = GuardedRules().with_rules([GuardedRule("niece_of", "B1u", (("nibling_of", 0),), (("is_female", "x"),))])
    assert rs.predict(st, (1, 2)) == frozenset({"nibling_of", "niece_of"})


def test_two_engines_give_the_same_facts_and_statuses(learned):
    stories = [world(s)[0] for s in range(700, 706)]
    stories.append(Story.from_facts([(1, 10, "living_in"), (1, 2, "colleague_of"), (2, 11, "living_in"),
                                     (3, 1, "colleague_of")],
                                    [(1, "is_person"), (2, "is_person"), (3, "is_person"), (10, "is_place"),
                                     (11, "is_place")]))
    for st in stories:
        a, b = learned.closure(st), learned.closure(st, justify=True)
        assert a.pairs.keys() == b.pairs.keys()
        assert {f: (a.stratum(*f)) for f in a.facts()} == {f: b.stratum(*f) for f in b.facts()}
        assert a.statuses == b.statuses


# ───────────────────────────── residual votes
def residual_world(seed, depth2):
    """Queries (x, p) labelled living_in through x -friend- a -colleague- b -living_in- p (length 3) and, for
    `depth2` of the stories, through a -colleague- b -living_in- p (length 2). friend ∘ living_in is never true on a
    query pair with raw facts: only a derived living_in(a, p) makes it visible."""
    out = []
    for s in range(seed, seed + 30):
        x, a, b, p = 4 * s, 4 * s + 1, 4 * s + 2, 4 * s + 3
        un = [(x, "is_person"), (a, "is_person"), (b, "is_person"), (p, "is_place")]
        st = Story.from_facts([(x, a, "friend_of"), (a, b, "colleague_of"), (b, p, "living_in")], un)
        out.append((st, (x, p), frozenset({"living_in"})))
        if s - seed < depth2:
            out.append((st, (a, p), frozenset({"living_in"})))
    return out


def test_residual_votes_make_a_chain_body_visible_in_one_round():
    ex = residual_world(0, depth2=10)
    with_r = GuardedRules(rounds=1, residual=True, neg=False).fit(ex)
    without = GuardedRules(rounds=1, residual=False, neg=False).fit(ex)
    target = ("living_in", "B2", (("friend_of", 0), ("living_in", 0)), ())
    assert target in {sig(r) for r in with_r}
    assert target not in {sig(r) for r in without}
    r = next(r for r in with_r if sig(r) == target)
    assert r.origin == "residual" and r.hits == 30
    # with rounds the derived fact makes the body visible without residual votes (the residual only accelerates)
    rounds = GuardedRules(rounds=5, residual=False, neg=False).fit(ex)
    assert target in {sig(r) for r in rounds}


def test_residual_never_votes_where_the_intermediate_fact_is_present():
    ex = residual_world(0, depth2=10)
    st, q, y = ex[0]
    st2 = Story.from_facts(list(st.facts) + [(q[0] + 1, q[1], "living_in")], list(st.unary))
    r = GuardedRules(rounds=1, residual=True, neg=False).fit([(st2, q, y)] * 3 + ex[1:])
    t = next(x for x in r if x.atoms == (("friend_of", 0), ("living_in", 0)))
    assert t.origin == "direct"                                   # its votes on st2 are direct, not residual


# ───────────────────────────── the store: chain_rules, explain, to_knowledge
def load(ks, st, scope):
    for x, y, rel in st.facts:
        ks.add("fact", pair_fact(x, y, rel, multi=True, family=COMPLEMENTS.get(rel, rel)), scope,
               source="outcome", by="story")
    for x, u in st.unary:
        ks.add("fact", unary_fact(x, u), scope, source="outcome", by="story")


def store_answer(ks, x, y, scope):
    return frozenset(r["body"]["o"] for r in ks.find(kind="fact", s=[x, y], scope=scope, status="active"))


def test_chain_rules_agrees_with_the_in_memory_fixed_point(learned):
    ks = KnowledgeStore()
    learned.to_knowledge(ks, {"domain": "planted"})
    diffs = 0
    for seed in range(600, 606):
        st, T = world(seed)
        scope = {"story": seed}
        load(ks, st, scope)
        rep = chain_rules(ks, scope=scope, rule_scope={"domain": "planted"}, complements=COMPLEMENTS)
        assert rep.fixed_point and rep.facts_added > 0
        c = learned.closure(st)
        for (x, y) in T:
            diffs += store_answer(ks, x, y, scope) != c.answer(x, y)
    assert diffs == 0
    assert ks.verify()


def test_chain_rules_is_idempotent_and_explains_guards_and_negation():
    rs = GuardedRules(complements=COMPLEMENTS).with_rules([
        GuardedRule("father_of", "B1u", (("parent_of", 0),), (("is_male", "x"),)),
        GuardedRule("Not_living_in", "NEG", (("living_in", 0),), (("is_person", "x"), ("is_place", "y")))])
    ks = KnowledgeStore()
    rs.to_knowledge(ks, {"d": 1})
    st = Story.from_facts([(1, 2, "parent_of")], [(1, "is_male"), (1, "is_person"), (2, "is_person"),
                                                   (9, "is_place")])
    load(ks, st, {"s": 1})
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert rep.facts_added == 3                                   # father_of(1,2), Not_living_in(1,9), (2,9)
    j = len(ks)
    rep2 = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert rep2.facts_proposed == 0 and len(ks) == j
    fid = ks.find(kind="fact", s=[1, 2], r="relation/father_of", scope={"s": 1})[0]["id"]
    e = explain(ks, fid)
    assert len(e.facts) == 2                                      # parent_of(1,2) and is_male(1): the guard is a premise
    assert e.cells == (("B1u", "parent_of(x,y) ∧ is_male(x)", "father_of"),)
    nid = ks.find(kind="fact", s=[1, 9], r="relation/living_in", scope={"s": 1})[0]["id"]
    d = explain(ks, nid).nodes[nid]["derivations"][0]
    assert d["absent"] == "living_in" and d["stratum"] == 2


def test_negative_fact_retracted_when_its_condition_ends():
    rs = GuardedRules(complements=COMPLEMENTS).with_rules([
        GuardedRule("Not_living_in", "NEG", (("living_in", 0),), (("is_person", "x"), ("is_place", "y")))])
    ks = KnowledgeStore()
    rs.to_knowledge(ks, {"d": 1})
    st = Story.from_facts([], [(1, "is_person"), (9, "is_place")])
    load(ks, st, {"s": 1})
    chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert store_answer(ks, 1, 9, {"s": 1}) == {"Not_living_in"}
    ks.add("fact", pair_fact(1, 9, "living_in", multi=True), {"s": 1}, source="outcome", by="later")
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert rep.retracted == 1 and store_answer(ks, 1, 9, {"s": 1}) == {"living_in"}


def test_disputes_become_store_questions():
    rs = GuardedRules(complements=COMPLEMENTS).with_rules([
        GuardedRule("living_in", "B2", (("colleague_of", 0), ("living_in", 0)), ()),
        GuardedRule("Not_living_in", "BNE", (("living_in", 0),), (("is_place", "y"),))])
    ks = KnowledgeStore()
    rs.to_knowledge(ks, {"d": 1})
    st = Story.from_facts([(1, 10, "living_in"), (1, 2, "colleague_of"), (2, 11, "living_in")],
                          [(1, "is_person"), (2, "is_person"), (10, "is_place"), (11, "is_place")])
    load(ks, st, {"s": 1})
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert rep.disputes_opened >= 1 and store_answer(ks, 1, 11, {"s": 1}) == frozenset()
    assert store_answer(ks, 1, 11, {"s": 1}) == rs.closure(st).answer(1, 11)


def test_to_knowledge_writes_one_rule_item_per_rule(learned):
    ks = KnowledgeStore()
    ids = learned.to_knowledge(ks, {"d": 1})
    assert len(ids) == len(learned) and len(set(ids.values())) == len(learned)
    assert learned.to_knowledge(ks, {"d": 1}) == ids                # idempotent
    rec = ks.item(next(iter(ids.values())))
    assert rec["body"]["kind"] == "guarded" and rec["status"] == "active"


# ───────────────────────────── multi pair facts in forward_chain; composition behaviour unchanged
def test_forward_chain_multi_keeps_two_relations_of_one_pair():
    t = CompositionTable(0.9, 1).fit([(("a", "b"), "c"), (("a", "b2"), "d")])
    ks = KnowledgeStore()
    t.to_knowledge(ks, {"k": 1})
    sc = {"s": 1}
    for x, y, r in ((0, 1, "a"), (1, 2, "b"), (1, 2, "b2")):
        ks.add("fact", pair_fact(x, y, r, multi=True), sc, source="outcome")
    rep = forward_chain(ks, scope=sc, rule_scope={"k": 1}, multi=True)
    got = {r["body"]["o"]: r["status"] for r in ks.find(kind="fact", s=[0, 2], scope=sc)}
    assert got == {"c": "active", "d": "active"} and rep.disputes_opened == 0


def test_pair_fact_default_unchanged():
    assert pair_fact(1, 2, "a") == {"s": [1, 2], "r": "relation", "o": "a"}
    assert pair_fact(1, 2, "Not_x", multi=True, family="x") == {"s": [1, 2], "r": "relation/x", "o": "Not_x"}
    assert unary_fact(3, "is_place") == {"s": 3, "r": "unary/is_place", "o": True}


def test_rule_text_and_validation():
    r = GuardedRule("paternal_grandparent_of", "B2z", (("parent_of", 0), ("parent_of", 0)), (("is_male", "z"),))
    assert r.text() == "parent_of(x,z) ∧ is_male(z) ∧ parent_of(z,y)"
    assert GuardedRule("x", "B2", (("a", 1), ("b", 0)), ()).text() == "a⁻(x,z) ∧ b(z,y)"
    assert GuardedRule("x", "BNE", (("a", 0),), (("u", "y"),)).text() == "a(x,z) ∧ z≠y ∧ u(y)"
    assert GuardedRule("x", "NEG", (("a", 0),), (("u", "x"), ("v", "y"))).text() == "u(x) ∧ v(y) ∧ ¬a(x,y)"
    with pytest.raises(ValueError):
        GuardedRule("x", "B3", (("a", 0),), ())
    with pytest.raises(ValueError):
        GuardedRules(majority=0.4)
