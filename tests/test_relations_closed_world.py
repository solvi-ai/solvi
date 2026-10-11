"""solvi.experimental.relations.GuardedRules(functional=...): a declared functional relation (a subject of a type has at
most one object for it) and the closed-world stratum after the positive fixed point and NEG — for a subject with a
known (active) positive r(x, z), the complement r̄(x, y) on every other candidate object y of the object type; never
from a hypothesis or a disputed positive, never for a subject without a known positive (open world stays). Statuses,
templates_on, why_not, chain_rules (the declaration as a spec item and a premise), explain, the learning flag
closed_world_in_rounds."""
from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import (TEMPLATES, GuardedRule, GuardedRules, Story, chain_rules, explain,
                                              pair_fact, unary_fact, why_not)

from solvi.core.knowledge import KnowledgeStore

from test_relations_rules import COMPLEMENTS, examples, world

FUNC = {"living_in": "person"}
P = [(1, "is_person"), (2, "is_person"), (3, "is_person"), (10, "is_place"), (11, "is_place"), (12, "is_place")]


def rules(*rs, **kw):
    return GuardedRules(complements=COMPLEMENTS, functional=FUNC, **kw).with_rules(list(rs))


COLL = GuardedRule("living_in", "B2", (("colleague_of", 0), ("living_in", 0)), ())


# ───────────────────────────── the declaration
def test_declaration_needs_a_complement_and_a_type():
    with pytest.raises(ValueError):
        GuardedRules(functional={"living_in": "person"})                      # no complement declared for living_in
    with pytest.raises(ValueError):
        GuardedRules(complements=COMPLEMENTS, functional={"living_in": ""})
    g = GuardedRules(complements=COMPLEMENTS, functional={"living_in": ("person", "place")})
    assert g.functional == {"living_in": ("is_person", "is_place")}
    assert GuardedRules(complements=COMPLEMENTS, functional=FUNC).functional == {"living_in": ("is_person", None)}
    assert GuardedRules(complements=COMPLEMENTS, functional={"living_in": "is_person"}).functional == \
        {"living_in": ("is_person", None)}


# ───────────────────────────── the stratum (in memory)
def test_closed_world_for_a_subject_with_a_known_positive_only():
    st = Story.from_facts([(1, 10, "living_in")], P)
    c = rules().closure(st)
    assert c.answer(1, 11) == {"Not_living_in"} and c.answer(1, 12) == {"Not_living_in"}
    assert c.answer(1, 10) == {"living_in"}
    assert c.info(1, 11, "Not_living_in") == {"status": "active", "stratum": "closed", "depth": 0}
    assert c.info(1, 10, "living_in")["stratum"] == "given"
    assert c.answer(2, 10) == frozenset() and c.answer(2, 11) == frozenset()     # no known positive: open world
    assert c.answer(1, 2) == frozenset()                                           # a person is not a candidate place
    assert c.closed[(1, 11, "Not_living_in")] == [(1, 10, "living_in")]


def test_without_the_declaration_nothing_changes():
    st = Story.from_facts([(1, 10, "living_in")], P)
    c = GuardedRules(complements=COMPLEMENTS).closure(st)
    assert c.answer(1, 11) == frozenset() and not c.closed


def test_a_derived_active_positive_is_known():
    st = Story.from_facts([(1, 2, "colleague_of"), (2, 10, "living_in")], P)
    c = rules(COLL).closure(st)
    assert c.answer(1, 10) == {"living_in"} and c.answer(1, 11) == {"Not_living_in"}
    assert c.answer(2, 12) == {"Not_living_in"}


def test_two_known_positives_contradict_the_declaration():
    # 1 lives in 10 (given) and in 11 (through 2); 3 lives in both through 1: the declaration is contradicted for 1 and
    # 3 — no closed world for them (a store cannot hold the two mutual disputes: the second would rest on the first)
    st = Story.from_facts([(1, 10, "living_in"), (1, 2, "colleague_of"), (2, 11, "living_in"),
                           (3, 1, "colleague_of")], P)
    c = rules(COLL).closure(st)
    assert c.answer(1, 10) == {"living_in"} and c.answer(1, 11) == {"living_in"}
    assert c.relations(1, 12) == set() and c.relations(3, 12) == set()
    assert c.closed_conflicts == [(1, "living_in", (10, 11)), (3, "living_in", (10, 11))]
    assert c.answer(2, 10) == {"Not_living_in"} and c.answer(2, 12) == {"Not_living_in"}
    assert not c.closed_disputes


def test_never_from_a_disputed_or_hypothetical_positive():
    # 6 lives in 10 (given) while a rule says it does not (disputed before the stratum); 7 lives in 10 only through 6
    away = GuardedRule("Not_living_in", "B1", (("away_from", 0),), ())
    st = Story.from_facts([(6, 10, "living_in"), (6, 10, "away_from"), (7, 6, "colleague_of")],
                          P + [(6, "is_person"), (7, "is_person")])
    c = rules(COLL, away).closure(st)
    assert c.status(6, 10, "living_in") == "disputed" and c.status(7, 10, "living_in") == "hypothesis"
    assert c.relations(6, 11) == set() and c.relations(7, 11) == set()            # no closed world from either
    assert not c.closed_disputes


AWAY = GuardedRule("Not_living_in", "B1", (("away_from", 0),), ())
# 9 lives in 11 while a rule says it does not (disputed); 8 lives in 10 (given) and in 11 only through 9 (a hypothesis)
DISPUTE = Story.from_facts([(9, 11, "living_in"), (9, 11, "away_from"), (8, 10, "living_in"), (8, 9, "colleague_of")],
                           P + [(8, "is_person"), (9, "is_person")])


def test_closed_fact_meeting_a_hypothetical_positive_is_a_dispute():
    c = rules(COLL, AWAY).closure(DISPUTE)
    assert c.status(8, 11, "living_in") == "disputed" and c.status(8, 11, "Not_living_in") == "disputed"
    assert c.answer(8, 11) == frozenset() and c.answer(8, 12) == {"Not_living_in"} and c.answer(8, 10) == {"living_in"}
    assert c.closed_disputes == [(8, 11, "living_in")]


def test_subject_of_another_type_and_object_type_declared():
    st = Story.from_facts([(10, 11, "living_in"), (1, 10, "living_in")], P + [(13, "is_town")])
    c = rules().closure(st)
    assert c.relations(10, 12) == set()                    # a place is not a person: no closed world for it
    c2 = GuardedRules(complements=COMPLEMENTS, functional={"living_in": ("person", "town")}).closure(st)
    assert c2.answer(1, 13) == {"Not_living_in"} and c2.relations(1, 11) == set()


def test_closed_facts_never_feed_the_positive_rules():
    far = GuardedRule("far_from", "B1", (("Not_living_in", 0),), ())
    st = Story.from_facts([(1, 10, "living_in"), (2, 11, "Not_living_in")], P)
    c = rules(far).closure(st)
    assert c.answer(2, 11) == {"Not_living_in", "far_from"}          # a given complement feeds the rule
    assert c.answer(1, 11) == {"Not_living_in"}                       # the closed one does not


def test_two_engines_agree_with_the_closed_world():
    learned = GuardedRules(0.9, 2, complements=COMPLEMENTS, functional=FUNC, rounds=3).fit(examples(range(4)))
    stories = [world(s)[0] for s in range(710, 716)]
    stories.append(Story.from_facts([(1, 10, "living_in"), (1, 2, "colleague_of"), (2, 11, "living_in"),
                                     (3, 1, "colleague_of")], P))
    for st in stories:
        a, b = learned.closure(st), learned.closure(st, justify=True)
        assert a.pairs.keys() == b.pairs.keys()
        assert {f: a.stratum(*f) for f in a.facts()} == {f: b.stratum(*f) for f in b.facts()}
        assert a.statuses == b.statuses
        assert a.closed == b.closed


def test_templates_on_and_statuses_name_the_stratum():
    st = Story.from_facts([(1, 10, "living_in")], P)
    g = rules()
    c = g.closure(st)
    assert g.templates_on(c, (1, 11), ["Not_living_in"]) == {"Not_living_in": ["closed"]}
    assert g.statuses(st, (1, 11)) == {"Not_living_in": {"status": "active", "stratum": "closed", "depth": 0}}


# ───────────────────────────── learning: untouched unless closed_world_in_rounds
def test_learning_is_untouched_by_default():
    ex = examples(range(4))
    a = GuardedRules(0.9, 2, complements=COMPLEMENTS, rounds=3).fit(ex)
    b = GuardedRules(0.9, 2, complements=COMPLEMENTS, functional=FUNC, rounds=3).fit(ex)
    assert [(r, r.hits, r.support) for r in a] == [(r, r.hits, r.support) for r in b]


def test_closed_world_in_rounds_makes_the_complement_visible_to_learning():
    ex = []
    for s in range(12):
        b = 100 * s
        st = Story.from_facts([(b + 1, b + 10, "living_in")], [(b + 1, "is_person"), (b + 10, "is_place"),
                                                                 (b + 11, "is_place")])
        ex.append((st, (b + 1, b + 11), frozenset({"Not_living_in", "far_from"})))
    target = GuardedRule("far_from", "B1", (("Not_living_in", 0),), ())
    off = GuardedRules(complements=COMPLEMENTS, functional=FUNC, templates=("B1",)).fit(ex)
    on = GuardedRules(complements=COMPLEMENTS, functional=FUNC, templates=("B1",), closed_world_in_rounds=True).fit(ex)
    assert target not in set(off) and target in set(on)
    with pytest.raises(ValueError):
        GuardedRules(complements=COMPLEMENTS, closed_world_in_rounds=True)          # nothing declared


def test_planted_world_elsewhere_by_declaration():
    """With BNE and NEG taken out of the templates, the planted world's Not_living_in comes from the declaration alone:
    every closed-world answer is right (persons with a known home), and nothing else changes."""
    pos = ("B1", "B1u", "B2", "B2z")
    g = GuardedRules(0.9, 2, complements=COMPLEMENTS, functional=FUNC, templates=pos).fit(examples(range(6)))
    plain = GuardedRules(0.9, 2, complements=COMPLEMENTS, templates=pos).fit(examples(range(6)))
    assert set(g) == set(plain)                                   # learning untouched
    test = examples(range(500, 503))
    closed = changed = 0
    for st, q, y in test:
        p, p0 = g.predict(st, q), plain.predict(st, q)
        if "Not_living_in" in p:
            closed += 1
            assert p == y, (q, p, y)
        else:
            changed += p != p0
    assert closed > 50 and changed == 0


# ───────────────────────────── why_not
def test_why_not_names_the_missing_positive():
    st = Story.from_facts([(1, 10, "living_in")], P)
    g = rules()
    w = why_not(g, st, 2, 11, "Not_living_in")
    assert w.status == "no_known_positive"
    assert w.agenda_items() == [{"kind": "positive", "goal": "learn a living_in fact for 2", "relation": "living_in",
                                 "subject": 2, "between": [2, 11]}]
    assert "no known living_in" in w.summary()
    assert why_not(g, st, 1, 11, "Not_living_in").status == "derivable"
    st2 = Story.from_facts([(1, 10, "living_in"), (1, 11, "living_in")], P)
    w2 = why_not(g, st2, 1, 12, "Not_living_in")
    assert w2.status == "functional_conflict" and w2.agenda_items()[0]["objects"] == [10, 11]
    assert why_not(rules(COLL, AWAY), DISPUTE, 8, 11, "Not_living_in").status == "disputed"
    assert why_not(g, st, 2, 11, "grandparent_of").status == "underivable"


# ───────────────────────────── the store
def load(ks, st, scope):
    for x, y, rel in st.facts:
        ks.add("fact", pair_fact(x, y, rel, multi=True, family=COMPLEMENTS.get(rel, rel)), scope,
               source="outcome", by="story")
    for x, u in st.unary:
        ks.add("fact", unary_fact(x, u), scope, source="outcome", by="story")


def store_answer(ks, x, y, scope):
    return frozenset(r["body"]["o"] for r in ks.find(kind="fact", s=[x, y], scope=scope, status="active"))


def test_chain_rules_writes_the_closed_world_with_the_declaration_as_premise():
    g = rules()
    ks = KnowledgeStore()
    ids = g.to_knowledge(ks, {"d": 1})
    decl = [r for r in ks.find(kind="rule", scope={"d": 1}) if r["body"].get("kind") == "functional"]
    assert len(decl) == 1 and ks.items[decl[0]["id"]].asserts[-1]["source"] == "spec"
    assert decl[0]["status"] == "active" and len(ids) == 0
    st = Story.from_facts([(1, 10, "living_in")], P)
    load(ks, st, {"s": 1})
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert rep.facts_added == 2
    assert store_answer(ks, 1, 11, {"s": 1}) == {"Not_living_in"} and store_answer(ks, 2, 11, {"s": 1}) == frozenset()
    fid = ks.find(kind="fact", s=[1, 11], r="relation/living_in", scope={"s": 1})[0]["id"]
    it = ks.items[fid].asserts[-1]
    assert it["source"] == "outcome" and decl[0]["id"] in it["just"] and it["evidence"]["stratum"] == 3
    e = explain(ks, fid)
    d = e.nodes[fid]["derivations"][0]
    assert d["stratum"] == 3 and decl[0]["id"] in d["premises"]
    assert d["cell"] == ("closed", "living_in functional for is_person", "Not_living_in")
    assert decl[0]["id"] in e.rules
    n = len(ks)
    rep2 = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    assert rep2.facts_proposed == 0 and len(ks) == n                     # idempotent
    ks.retract(decl[0]["id"], by="person", why="not functional here")    # the declaration taken back → the facts go
    assert store_answer(ks, 1, 11, {"s": 1}) == frozenset()


def test_chain_rules_functional_argument_writes_the_declaration():
    ks = KnowledgeStore()
    GuardedRules(complements=COMPLEMENTS).to_knowledge(ks, {"d": 1})
    st = Story.from_facts([(1, 10, "living_in")], P)
    load(ks, st, {"s": 1})
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS, functional=FUNC)
    assert rep.facts_added == 2 and store_answer(ks, 1, 12, {"s": 1}) == {"Not_living_in"}
    with pytest.raises(ValueError):
        chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, functional=FUNC)       # no complement for living_in


@pytest.mark.parametrize("which", ["dispute", "conflict"])
def test_chain_rules_opens_the_dispute_and_agrees_with_memory(which):
    g = rules(COLL, AWAY)
    ks = KnowledgeStore()
    g.to_knowledge(ks, {"d": 1})
    st = DISPUTE if which == "dispute" else Story.from_facts(
        [(1, 10, "living_in"), (1, 2, "colleague_of"), (2, 11, "living_in"), (3, 1, "colleague_of")], P)
    load(ks, st, {"s": 1})
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    if which == "dispute":
        assert rep.disputes_opened >= 2 and rep.closed_conflicts == 0      # (9, 11) by the rules, (8, 11) closed
    else:
        assert rep.closed_conflicts == 2
    c = g.closure(st)
    ents = sorted(st.entities())
    for x in ents:
        for y in ents:
            if x != y:
                assert store_answer(ks, x, y, {"s": 1}) == c.answer(x, y), (x, y)


def test_chain_rules_agrees_with_memory_on_planted_stories():
    learned = GuardedRules(0.9, 2, complements=COMPLEMENTS, functional=FUNC,
                           templates=tuple(t for t in TEMPLATES if t != "BNE")).fit(examples(range(6)))
    ks = KnowledgeStore()
    learned.to_knowledge(ks, {"domain": "planted"})
    diffs = 0
    for seed in range(620, 623):
        st, T = world(seed)
        scope = {"story": seed}
        load(ks, st, scope)
        chain_rules(ks, scope=scope, rule_scope={"domain": "planted"}, complements=COMPLEMENTS)
        c = learned.closure(st)
        ents = sorted(st.entities())
        diffs += sum(store_answer(ks, x, y, scope) != c.answer(x, y) for x in ents for y in ents if x != y)
    assert diffs == 0
    assert ks.verify()


def test_chain_rules_counts_each_new_item_once():
    learned = GuardedRules(0.9, 2, complements=COMPLEMENTS, functional=FUNC, rounds=3).fit(examples(range(4)))
    ks = KnowledgeStore()
    learned.to_knowledge(ks, {"d": 1})
    st, _ = world(900)
    load(ks, st, {"s": 1})
    before = sum(1 for it in ks.items.values() if it.kind == "fact")
    rep = chain_rules(ks, scope={"s": 1}, rule_scope={"d": 1}, complements=COMPLEMENTS)
    after = sum(1 for it in ks.items.values() if it.kind == "fact")
    assert rep.facts_added == after - before > 0 and rep.facts_proposed >= rep.facts_added
