"""solvi.rules.RuleList and System.learn_rule: the literals of a text, fitting twice, arguments that cannot work."""
from typing import Literal

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.rules import RuleList, literals


def _zones():
    cat = Catalog()

    @cat.fn
    def up(addr):
        return addr.upper()
    return cat, [Question("zone", "?", Answer.choice(["n", "s"]))]


def test_a_text_gives_one_literal_per_word_in_any_script():
    assert literals({"a": "12 North Street"}, ["a"]) == {"a has '12'", "a has 'NORTH'", "a has 'STREET'"}
    assert literals({"a": "ул. Северная, 12"}, ["a"]) == {"a has 'УЛ'", "a has 'СЕВЕРНАЯ'", "a has '12'"}
    assert literals({"a": "Zürich O'Neil_x 90210"}, ["a"]) == {"a has 'ZÜRICH'", "a has 'O'NEIL'", "a has 'X'",
                                                              "a has '90210'", "a has number starting '90'"}
    assert literals({"a": "Zürich"}, ["a"]) == {"a has 'ZÜRICH'"}          # a decomposed accent: the same word


def test_learn_rule_learns_a_rule_list_from_russian_street_names():
    cat, qs = _zones()
    s = System(cat, qs)
    ex = [({"addr": f"ул. Северная, {i}"}, "n") for i in range(6)] + [({"addr": f"ул. Южная, {i}"}, "s") for i in range(6)]
    rl = s.learn_rule("zone", ex, facts=["up"])
    assert {r["if"] for r in rl.rules} >= {"up has 'СЕВЕРНАЯ'"} and len(rl.rules) >= 1
    assert s.ask({"addr": "ул. Южная, 30"})["zone"].answer == "s"
    assert s.ask({"addr": "ул. Северная, 30"})["zone"].answer == "n"


def test_fitting_a_rule_list_twice_replaces_the_list_instead_of_appending_to_it():
    rows, answers = [{"a": "x north"}] * 4 + [{"a": "y south"}] * 4, ["n"] * 4 + ["s"] * 4
    rl = RuleList(["a"]).fit(rows, answers)
    first = [dict(r) for r in rl.rules]
    assert rl.fit(rows, answers).rules == first and len(first) == 2


def test_learn_rule_without_examples_or_with_an_unknown_fact_raises_and_installs_nothing():
    cat, qs = _zones()

    @cat.rule("zone")
    def zone(up):
        return "n" if "NORTH" in up else "s"
    s = System(cat, qs)
    en = [({"addr": f"{i} north st"}, "n") for i in range(6)] + [({"addr": f"{i} south st"}, "s") for i in range(6)]
    with pytest.raises(ValueError, match="no examples"):
        s.learn_rule("zone", [], facts=["up"])
    with pytest.raises(ValueError, match="upp is not a part of the catalog or a given fact"):
        s.learn_rule("zone", en, facts=["upp"])
    with pytest.raises(ValueError, match="rows and"):
        RuleList(["a"]).fit([], [])
    assert cat.rules["zone"].func is zone and "zone" not in s.learned_rules
    assert s.ask({"addr": "1 south st"})["zone"].answer == "s"
    assert s.learn_rule("zone", en, facts=["addr"]).rules                      # a given fact of the examples is fine


def test_a_learned_rule_that_replaces_a_typed_rule_takes_its_typed_bookkeeping_with_it():
    cat = Catalog()

    @cat.rule("zone")
    def zone(code: int) -> Literal["n", "s"]:
        return "n" if code > 5 else "s"
    s = System(cat, [Question("zone", "?")])
    s.learn_rule("zone", [({"code": i}, "n" if i > 5 else "s") for i in range(12)], facts=["code"])
    assert not any("zone" in k for rs in cat.readers.values() for k in rs)

    @cat.fn
    def code(text) -> str:                                                     # the replaced rule read code: int
        return text.strip()
    assert "code" in cat.parts
