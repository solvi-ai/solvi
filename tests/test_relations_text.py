"""solvi.experimental.relations.TextFrontend / GraphCheck: kinship sentences read into a relation graph by code — five
sentence patterns, the owner of «his / her REL [Y]» (the nearest preceding name of the pronoun's gender that is not
itself the object of a relation phrase), inverse edges by the relation's inverse table, gender facts with their source,
a global check (gender consistency, one relation per ordered pair, at most one father / mother / husband / wife) that
drops contradictions and names them, and the hand-off to the composition table as a Story and its shortest paths."""
from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import (FUNCTIONAL, KINSHIP, PATTERNS, CompositionTable, GraphCheck, Story,
                                              TextFrontend, TextGraph, forward_chain, pair_fact)

M, F = "male", "female"


def triples(g: TextGraph, inverse: bool | None = None):
    return {(e.x, e.relation, e.y) for e in g.edges if inverse is None or e.inverse == inverse}


def pattern_of(g: TextGraph, x, r, y):
    return next(e.pattern for e in g.edges if (e.x, e.relation, e.y) == (x, r, y))


# ───────────────────────────── the five patterns on planted sentences
@pytest.mark.parametrize("sentence, edge, pattern", [
    ("[Ann]'s son [Bob] went to the park.", ("Ann", "son", "Bob"), "x_rel_y"),
    ("[Ann]'s son, [Bob], went to the park.", ("Ann", "son", "Bob"), "x_rel_y"),
    ("They met [Bob], [Ann]'s son.", ("Ann", "son", "Bob"), "y_x_rel"),
    ("[Bob] is the son of [Ann].", ("Ann", "son", "Bob"), "y_is_rel_of_x"),
    ("[Bob], was a son of [Ann].", ("Ann", "son", "Bob"), "y_is_rel_of_x"),
    ("They asked [Bob], the son of [Ann], to stay.", ("Ann", "son", "Bob"), "y_the_rel_of_x"),
    ("[Ann] took her son [Bob] to school.", ("Ann", "son", "Bob"), "his_her_rel_y"),
    ("[Ann] called her son, [Bob].", ("Ann", "son", "Bob"), "his_her_rel_y"),
])
def test_each_pattern_reads_its_planted_sentence(sentence, edge, pattern):
    g = TextFrontend().read(sentence, genders={"Ann": F, "Bob": M})
    assert triples(g, inverse=False) == {edge}
    assert pattern_of(g, *edge) == pattern
    assert set(PATTERNS) == {"x_rel_y", "y_x_rel", "y_is_rel_of_x", "y_the_rel_of_x", "his_her_rel_y"}


def test_longest_relation_word_wins_and_unrelated_text_gives_nothing():
    g = TextFrontend().read("[Ann]'s grandmother [Cat] baked. [Dan] went home with [Eve].",
                            genders={"Ann": F, "Cat": F, "Dan": M, "Eve": F})
    assert triples(g, inverse=False) == {("Ann", "grandmother", "Cat")}
    assert TextFrontend().read("[Dan] went home with [Eve].").edges == ()


def test_owner_is_the_nearest_preceding_name_of_the_pronouns_gender():
    g = TextFrontend().read("[Joe] and [Ann] went out, and she took her daughter [Liz].",
                            genders={"Joe": M, "Ann": F, "Liz": F})
    assert triples(g, inverse=False) == {("Ann", "daughter", "Liz")}
    g = TextFrontend().read("[Joe] and [Ann] went out, and he took his daughter [Liz].",
                            genders={"Joe": M, "Ann": F, "Liz": F})
    assert triples(g, inverse=False) == {("Joe", "daughter", "Liz")}


def test_no_owner_in_the_sentence_is_a_refusal_not_a_guess():
    g = TextFrontend().read("[Joe] was late. She took her daughter [Liz].", genders={"Joe": M, "Liz": F})
    assert g.edges == ()


# ───────────────────────────── the v2 owner rule
def test_owner_rule_on_an_enumeration():
    text = "[R] took his son [P] and his daughter [J] to the zoo."
    g = TextFrontend().read(text, genders={"R": M, "P": M, "J": F})
    assert triples(g, inverse=False) == {("R", "son", "P"), ("R", "daughter", "J")}
    assert ("P", "daughter", "J") not in triples(g)


def test_owner_rule_skips_the_object_of_a_possessive_phrase():
    g = TextFrontend().read("[Sam]'s brother [Tom] and his wife [Una] came.", genders={"Sam": M, "Tom": M, "Una": F})
    assert ("Sam", "brother", "Tom") in triples(g, inverse=False)
    # the v2 rule: Tom, an object, is never the owner (the benchmark's stories keep «his» on the sentence's subject;
    # in free English the phrase is ambiguous — a stated limit)
    assert ("Sam", "wife", "Una") in triples(g, inverse=False)
    assert ("Tom", "wife", "Una") not in triples(g)


# ───────────────────────────── inverse edges and genders
def test_every_edge_carries_its_inverse_by_the_owners_gender():
    g = TextFrontend().read("[Ann] took her son [Bob] out. [Joe]'s sister [Kim] sang.",
                            genders={"Ann": F, "Bob": M, "Joe": M, "Kim": F})
    assert triples(g, inverse=True) == {("Bob", "mother", "Ann"), ("Kim", "brother", "Joe")}
    inv = next(e for e in g.edges if e.inverse and e.x == "Bob")
    assert inv.pattern == "his_her_rel_y" and inv.sentence == 0


def test_inverse_table_is_complete_and_gender_consistent():
    for r, (g, im, iff) in KINSHIP.items():
        assert g in (M, F)
        for x_gender, inv in ((M, im), (F, iff)):
            assert inv in KINSHIP and KINSHIP[inv][0] == x_gender
            back = KINSHIP[inv][1] if g == M else KINSHIP[inv][2]
            assert back == r                                        # the inverse of the inverse is the relation
    assert set(FUNCTIONAL) == {"father", "mother", "husband", "wife"}


def test_genders_inferred_from_relation_words_with_their_source():
    g = TextFrontend().read("[Ann]'s mother [Cat] sang. [Cat] and her son [Bob] left.")
    assert g.genders["Cat"].gender == F and g.genders["Cat"].source == "inferred:mother"
    assert g.genders["Bob"].gender == M and g.genders["Bob"].source == "inferred:son"
    assert g.genders["Ann"].gender is None
    assert ("Ann", "mother", "Cat") in triples(g) and not any(e.x == "Cat" and e.y == "Ann" for e in g.edges)
    assert ("Bob", "mother", "Cat") in triples(g, inverse=True)
    assert [e.relation for e in g.no_inverse] == ["mother"]          # Ann's gender unknown: no inverse, listed
    d = TextFrontend().read("[Ann]'s mother [Cat] sang.", genders={"Ann": F, "Cat": F})
    assert d.genders["Ann"].source == "declared" and ("Cat", "daughter", "Ann") in triples(d)
    assert ("Cat", "is_female") in d.story().unary


def test_names_given_as_a_list_are_bracketed():
    g = TextFrontend().read("Ann took her son Bob to Annapolis.", names=["Ann", "Bob"], genders={"Ann": F, "Bob": M})
    assert triples(g, inverse=False) == {("Ann", "son", "Bob")}
    assert g.names == ("Ann", "Bob")


# ───────────────────────────── the global check
def test_check_drops_a_gender_inconsistent_edge_and_names_it():
    text = "[Chris] and her mother [Will] went out. [Chris]'s sister [Kim] sang."
    sex = {"Chris": F, "Will": M, "Kim": F}
    g = TextFrontend().read(text, genders=sex)
    assert not any("Will" in (e.x, e.y) for e in g.edges)
    assert ("Chris", "sister", "Kim") in triples(g)
    assert [(d.edge.x, d.edge.relation, d.edge.y, d.reason) for d in g.dropped] == [("Chris", "mother", "Will", "gender")]
    raw = TextFrontend(check=False).read(text, genders=sex)
    assert ("Chris", "mother", "Will") in triples(raw) and raw.dropped == ()


def test_check_drops_a_second_father():
    text = "[Ann]'s father [Bob] sang. [Ann]'s father [Carl] danced. [Ann]'s brother [Dan] read."
    sex = {"Ann": F, "Bob": M, "Carl": M, "Dan": M}
    g = TextFrontend().read(text, genders=sex)
    assert not any(e.relation == "father" and e.x == "Ann" for e in g.edges)
    assert {d.reason for d in g.dropped} == {"functional"} and len(g.dropped) == 2
    assert not any("Bob" in (e.x, e.y) or "Carl" in (e.x, e.y) for e in g.edges)   # the inverses go along
    assert ("Ann", "brother", "Dan") in triples(g)
    raw = TextFrontend(check=False).read(text, genders=sex)
    assert {("Ann", "father", "Bob"), ("Ann", "father", "Carl")} <= triples(raw)


def test_check_drops_two_relations_on_one_pair_and_a_gender_conflict_of_inferred_genders():
    g = TextFrontend().read("[Ann]'s sister [Kim] sang. [Ann]'s aunt [Kim] danced.", genders={"Ann": F, "Kim": F})
    assert g.edges == () and {d.reason for d in g.dropped} == {"pair"}
    g = TextFrontend().read("[Ann]'s son [Pat] sang. [Bea]'s daughter [Pat] danced.")
    assert not any(e.y == "Pat" for e in g.edges) and {d.reason for d in g.dropped} == {"gender"}
    assert g.genders["Pat"].gender is None and g.genders["Pat"].source == "conflict"


def test_check_agrees_with_both_directions_stated():
    g = TextFrontend().read("[Ann]'s son [Bob] sang. [Bob]'s mother [Ann] danced.", genders={"Ann": F, "Bob": M})
    assert triples(g) == {("Ann", "son", "Bob"), ("Bob", "mother", "Ann")} and g.dropped == ()
    assert all(not e.inverse for e in g.edges)                       # both stated: no duplicate inverse


def test_graph_check_alone_and_its_report():
    gc = GraphCheck()
    g = TextFrontend(check=gc).read("[Ann]'s father [Bob] sang. [Ann]'s father [Carl] danced.",
                                    genders={"Ann": F, "Bob": M, "Carl": M})
    assert len(g.dropped) == 2 and all("father" in d.detail for d in g.dropped)
    with pytest.raises(ValueError):
        GraphCheck(functional=("cousin",))


# ───────────────────────────── hand-off to the rules
def test_story_and_paths_feed_the_composition_table():
    t = CompositionTable(support=1).fit([(("son", "sister"), "daughter"), (("father", "sister"), "aunt")])
    text = "[Ann] took her son [Bob] to the park. [Bob]'s sister [Kim] stayed home."
    fe = TextFrontend()
    sex = {"Ann": F, "Bob": M, "Kim": F}
    st = fe.story(text, genders=sex)
    assert isinstance(st, Story) and ("Ann", "Bob", "son") in st.facts and ("Bob", "is_male") in st.unary
    g = fe.read(text, genders=sex)
    assert g.paths("Ann", "Kim") == [("son", "sister")]
    assert t.derive_paths(g.paths("Ann", "Kim")).relation == "daughter"
    assert g.paths("Kim", "Nobody") == []


def test_story_feeds_forward_chain():
    from solvi.core.knowledge import KnowledgeStore
    t = CompositionTable(support=1).fit([(("son", "sister"), "daughter")])
    store = KnowledgeStore()
    t.to_knowledge(store, {"domain": "kin"})
    st = TextFrontend().story("[Ann] took her son [Bob] out. [Bob]'s sister [Kim] sang.",
                              genders={"Ann": F, "Bob": M, "Kim": F})
    for x, y, r in st.facts:
        store.add("fact", pair_fact(x, y, r), {"story": "s1"}, source="outcome", by="test")
    forward_chain(store, scope={"story": "s1"}, rule_scope={"domain": "kin"})
    got = {rec["body"]["o"] for rec in store.find(kind="fact", s=["Ann", "Kim"], scope={"story": "s1"})}
    assert got == {"daughter"}


def test_reading_is_deterministic():
    text = ("[R] took his son [P] and his daughter [J] to the zoo. [J]'s aunt [K] came. [P], [K]'s nephew, waved. "
            "[K] is the sister of [R].")
    sex = {"R": M, "P": M, "J": F, "K": F}
    a = [TextFrontend().read(text, genders=sex) for _ in range(3)]
    assert a[0].edges == a[1].edges == a[2].edges and a[0].story().facts == a[2].story().facts
    assert len(a[0].edges) == len(set(a[0].edges))


def test_custom_relations_and_bad_tables():
    rel = {"boss": (None, "report", "report"), "report": (None, "boss", "boss")}
    g = TextFrontend(relations=rel).read("[Ann]'s boss [Bob] smiled.")
    assert triples(g) == {("Ann", "boss", "Bob"), ("Bob", "report", "Ann")}   # one inverse for both genders
    rel2 = {"elder": (None, "nephew", "niece"), "nephew": ("male", "elder", "elder"), "niece": ("female", "elder", "elder")}
    g = TextFrontend(relations=rel2).read("[Ann]'s elder [Bob] smiled.")
    assert triples(g) == {("Ann", "elder", "Bob")} and len(g.no_inverse) == 1   # Ann's gender unknown: no inverse
    g = TextFrontend(relations=rel2).read("[Ann]'s elder [Bob] smiled.", genders={"Ann": F})
    assert ("Bob", "niece", "Ann") in triples(g)
    with pytest.raises(ValueError):
        TextFrontend(relations={"boss": ("other", "x", "x")})
