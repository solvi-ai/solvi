"""solvi.experimental.relations.why_not: why a relation between two entities was NOT derived — the mirror of explain.
Over the pair facts' paths x → y and a CompositionTable: derivable (with the derivation), disputed (the readings),
derives another relation, a missing cell r1 ∘ r2 → ? ranked by the number of paths it would unblock, or no path at all
(a missing fact). The refusal's missing cells are also the `missing` detail of CompositionTable.derive, and a report
turns into agenda goals («learn cell father ∘ sister») whose done check is the table itself. Also first_missing_step
for a reference derivation."""
from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import CompositionTable, WhyNot, first_missing_step, rule_schema, why_not

from solvi.core.knowledge import Agenda

# a small kinship algebra: (r1, r2) → r reads «y is x's r1, z is y's r2 ⟹ z is x's r»
PARENTS = ("father", "mother")
SIBS = {"sister": "aunt", "brother": "uncle"}
GRAND = {"father": "grandfather", "mother": "grandmother"}
GSIB = {"sister": "grandaunt", "brother": "granduncle"}
CELLS = {}
for p in PARENTS:
    for s, r in SIBS.items():
        CELLS[(p, s)] = r                                          # father ∘ sister → aunt
    for q in PARENTS:
        CELLS[(p, q)] = GRAND[q]                                   # father ∘ mother → grandmother
    for s, r in SIBS.items():
        CELLS[(p, r)] = GSIB[s]                                    # father ∘ aunt → grandaunt
for g in GRAND.values():
    for s, r in GSIB.items():
        CELLS[(g, s)] = r                                          # grandfather ∘ sister → grandaunt


def examples(without=()):
    return [(k, v) for k, v in sorted(CELLS.items()) if k not in without for _ in range(2)]


FULL = CompositionTable().fit(examples())
CUT = CompositionTable().fit(examples(without={("father", "sister")}))
FACTS = [("don", "father", "mike"), ("mike", "sister", "dot")]   # don —father→ mike —sister→ dot ⟹ dot is don's aunt


def test_positive_control_names_exactly_the_removed_cell():
    full = why_not(FULL, FACTS, "don", "dot", "aunt")
    assert full.status == "derivable" and full.derivation.relation == "aunt" and not full.gaps   # negative control
    rep = why_not(CUT, FACTS, "don", "dot", "aunt")
    assert isinstance(rep, WhyNot) and rep.status == "missing_cell" and not rep.readings
    assert [(g.left, g.right) for g in rep.gaps] == [("father", "sister")] and rep.gaps[0].unblocks == 1
    assert "father ∘ sister" in rep.summary()


def test_derive_refusals_carry_the_missing_cells():
    d = CUT.derive(("father", "sister"))
    assert d.reason == "underivable" and d.missing == (("father", "sister"),)
    assert FULL.derive(("father", "sister")).missing == ()                          # an answer has none
    # a longer chain: the shortest failing span names both cells that would close it
    cut2 = CompositionTable().fit(examples(without={("father", "aunt"), ("grandfather", "sister")}))
    d = cut2.derive(("father", "father", "sister"))
    assert d.reason == "underivable" and set(d.missing) == {("father", "aunt"), ("grandfather", "sister")}
    # one of the two is enough: the other bracketing derives the chain
    cut1 = CompositionTable().fit(examples(without={("father", "aunt")}))
    assert cut1.derive(("father", "father", "sister")).relation == "grandaunt"
    # pooled paths: the missing cells of the chains that derive nothing, most blocked first
    cut3 = CompositionTable().fit(examples(without={("father", "sister"), ("mother", "sister")}))
    d = cut3.derive_paths([("father", "sister"), ("mother", "sister"), ("father", "sister")])
    assert d.reason == "underivable" and d.missing == (("father", "sister"), ("mother", "sister"))


def test_gaps_are_ranked_by_the_paths_they_would_unblock():
    cut = CompositionTable().fit(examples(without={("father", "sister"), ("mother", "sister")}))
    facts = [("x", "father", "p1"), ("p1", "sister", "y"), ("x", "father", "p2"), ("p2", "sister", "y"),
             ("x", "mother", "q"), ("q", "sister", "y")]
    rep = why_not(cut, facts, "x", "y", "aunt")
    assert rep.status == "missing_cell" and rep.paths == 3
    assert [(g.left, g.right, g.unblocks) for g in rep.gaps] == [("father", "sister", 2), ("mother", "sister", 1)]
    assert rep.gaps[0].example == ("father", "sister")


def test_disputed_other_and_no_path():
    facts = [("x", "father", "p"), ("p", "sister", "y"), ("x", "mother", "q"), ("q", "brother", "y")]
    rep = why_not(FULL, facts, "x", "y", "aunt")
    assert rep.status == "disputed" and rep.readings == {"aunt": 1, "uncle": 1}
    assert rep.derivation.reason == "disagree"
    other = why_not(FULL, FACTS, "don", "dot", "uncle")
    assert other.status == "other" and other.readings == {"aunt": 1} and not other.gaps
    none = why_not(FULL, [("a", "father", "b")], "a", "z")
    assert none.status == "no_path" and none.paths == 0 and "fact" in none.summary()
    assert why_not(FULL, FACTS, "don", "dot").status == "derivable"


def test_inverse_relations_open_reverse_paths_and_the_cap_is_reported():
    facts = [("mike", "son", "don"), ("mike", "sister", "dot")]           # don's father is mike, written the other way
    assert why_not(FULL, facts, "don", "dot").status == "no_path"
    rep = why_not(FULL, facts, "don", "dot", "aunt", inverse={"son": "father"})
    assert rep.status == "derivable" and rep.derivation.relation == "aunt"
    many = [("x", "father", f"p{i}") for i in range(30)] + [(f"p{i}", "sister", "y") for i in range(30)]
    capped = why_not(FULL, many, "x", "y", "aunt", max_paths=5)
    assert capped.paths == 5 and capped.truncated and capped.status == "derivable"


def test_the_report_becomes_agenda_goals_whose_done_check_is_the_table():
    table = CompositionTable().fit(examples(without={("father", "sister")}))
    rep = why_not(table, FACTS, "don", "dot", "aunt")
    items = rep.agenda_items()
    assert items == [{"kind": "cell", "goal": "learn cell father ∘ sister", "left": "father", "right": "sister",
                      "unblocks": 1, "example": ["father", "sister"], "between": ["don", "dot"]}]
    ag = Agenda()
    assert rep.to_agenda(ag, table) == ["learn cell father ∘ sister"]
    ag.update({})
    assert "learn cell father ∘ sister" in ag.open()
    table.fit(examples())                                          # the cell observed later: the goal is done
    ag.update({})
    assert "learn cell father ∘ sister" in ag.done()
    assert rep.to_agenda(ag, table) == ["learn cell father ∘ sister"]   # idempotent: an existing goal is kept
    gap = why_not(table, [("a", "father", "b")], "a", "z").agenda_items()
    assert gap == [{"kind": "fact", "goal": "learn a fact linking a and z", "between": ["a", "z"]}]
    dis = why_not(FULL, [("x", "father", "p"), ("p", "sister", "y"), ("x", "mother", "q"), ("q", "brother", "y")],
                  "x", "y").agenda_items()
    assert dis[0]["kind"] == "dispute" and dis[0]["readings"] == {"aunt": 1, "uncle": 1}


def test_first_missing_step_of_a_reference_derivation():
    steps = [(("grandfather", "a", "c"), [("father", "a", "b"), ("father", "b", "c")]),
             (("grandaunt", "a", "d"), [("grandfather", "a", "c"), ("sister", "c", "d")])]
    head, body, schema = first_missing_step(steps, {("grandfather", "a", "c")})
    assert head == ("grandaunt", "a", "d") and body == [("grandfather", "a", "c"), ("sister", "c", "d")]
    assert schema == "grandaunt(X,Y) ⟸ grandfather(X,Z), sister(Z,Y)"
    assert first_missing_step(steps, {("grandfather", "a", "c"), ("grandaunt", "a", "d")}) is None
    assert first_missing_step(steps, set())[0] == ("grandfather", "a", "c")
    assert rule_schema(("r", 1, 2), [("p", 2, 1)]) == "r(X,Y) ⟸ p(Y,X)"


def test_why_not_validates_its_inputs():
    with pytest.raises(TypeError):
        why_not(CELLS, FACTS, "don", "dot")
    with pytest.raises(ValueError):
        why_not(FULL, FACTS, "don", "dot", max_len=0)
