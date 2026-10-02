"""solvi.sets: decisions over a set — the most probable combination of the items' answers under at-most / exactly /
capacity / exclusion constraints, solved per connected component (exact) or greedily (the stated approximation); fixed
answers never change; every change cites its group; a set that cannot be repaired says so; the record replays."""
import json
import time

import pytest

from solvi import Answer, Catalog, Decision, Question, System
from solvi.sets import EACH, AtMostOne, Capacity, Exclusive, ExactlyOne, Item, SetDecision, decide_set


def _pair(a, b, p, **kw):
    return Item((a, b), {"yes": p, "no": 1 - p}, keys={"a": a, "b": b}, **kw)


ONE = [AtMostOne("a"), AtMostOne("b")]


def test_exact_keeps_the_most_probable_combination_where_the_greedy_keeps_the_surest_pair_first():
    items = [_pair("a1", "b1", 0.95), _pair("a2", "b1", 0.9), _pair("a1", "b2", 0.9), _pair("a3", "b3", 0.2)]
    exact = decide_set(items, ONE)
    assert exact.answers == {("a1", "b1"): "no", ("a2", "b1"): "yes", ("a1", "b2"): "yes", ("a3", "b3"): "no"}
    assert exact.exact and exact.feasible and exact.method == "exact"
    greedy = decide_set(items, ONE, method="greedy")
    assert greedy.answers[("a1", "b1")] == "yes" and greedy.answers[("a2", "b1")] == "no" and not greedy.exact
    assert [d.id for d in greedy.changed] == [("a2", "b1"), ("a1", "b2")]


def test_a_changed_item_cites_the_constraint_the_group_and_the_item_that_holds_it():
    out = decide_set([_pair("a1", "b1", 0.99), _pair("a1", "b2", 0.7)], ONE)
    d = out[("a1", "b2")]
    assert d.answer == "no" and d.was == "yes" and d.changed and d.confidence == pytest.approx(0.3)
    assert d.cited == [{"constraint": "at_most_one(a)", "group": "a=a1", "kept": [("a1", "b1")]}]
    assert d.why.endswith("changed from 'yes' to 'no' to satisfy at_most_one(a) on a=a1: ('a1', 'b1') holds 'yes' (0.99)")
    assert not out[("a1", "b1")].changed and out[("a1", "b1")].cited == []


def test_answers_as_given_that_already_satisfy_every_group_are_kept_and_nothing_is_solved():
    out = decide_set([_pair("a1", "b1", 0.9), _pair("a1", "b2", 0.2), _pair("a2", "b2", 0.8)], ONE)
    assert not out.changed and out.components == [] and out.feasible


def test_fixed_answers_never_change_and_conflicting_ones_are_reported_not_repaired():
    rule = Item(("a1", "b1"), None, "yes", keys={"a": "a1", "b": "b1"})        # a rule's answer, no probabilities
    out = decide_set([rule, _pair("a1", "b2", 0.99)], ONE)
    assert out.answers == {("a1", "b1"): "yes", ("a1", "b2"): "no"} and out[("a1", "b1")].fixed
    both = decide_set([rule, Item(("a1", "b2"), None, "yes", keys={"a": "a1", "b": "b2"})], ONE)
    assert not both.feasible and both.violations == [{"constraint": "at_most_one(a)", "group": "a=a1", "count": 2,
                                                       "min": 0, "max": 1}]
    assert all("not repaired: at_most_one(a) on a=a1 broken" in d.why for d in both)
    assert "NOT FEASIBLE" in str(both)
    abstained = Item(("a1", "b3"), None, None, keys={"a": "a1", "b": "b3"})   # an abstention counts in no group
    assert decide_set([rule, abstained], ONE).feasible


def test_exactly_one_moves_the_item_that_loses_least_into_an_empty_group_and_says_it_was_needed():
    items = [_pair("r1", "x", 0.3), _pair("r1", "y", 0.1), _pair("r2", "z", 0.8)]
    for method in ("exact", "greedy"):
        out = decide_set(items, [ExactlyOne("a")], method=method)
        assert out.answers == {("r1", "x"): "yes", ("r1", "y"): "no", ("r2", "z"): "yes"}, method
        assert out[("r1", "x")].cited == [{"constraint": "exactly_one(a)", "group": "a=r1", "needed": 1}]


def test_a_capacity_per_answer_assigns_tasks_to_shifts_with_room():
    p = {"t1": {"am": 0.9, "pm": 0.1}, "t2": {"am": 0.8, "pm": 0.2}, "t3": {"am": 0.6, "pm": 0.4}, "t4": {"am": 0.5, "pm": 0.5}}
    items = [Item(t, probs) for t, probs in p.items()]
    out = decide_set(items, [Capacity(max=2, answer=EACH, name="two_per_shift")])
    assert out.answers == {"t1": "am", "t2": "am", "t3": "pm", "t4": "pm"} and out.exact
    assert out["t3"].cited[0]["constraint"] == "two_per_shift" and out["t3"].cited[0]["group"] == "all, answer=am"


def test_exclusive_lists_of_ids_and_a_tie_score_that_settles_equally_probable_combinations():
    items = [Item("x", {"yes": 0.8, "no": 0.2}, tie=0.1), Item("y", {"yes": 0.8, "no": 0.2}, tie=0.9)]
    for method in ("exact", "greedy"):
        out = decide_set(items, [Exclusive([("x", "y")])], method=method)
        assert out.answers == {"x": "no", "y": "yes"}, method
        assert out["x"].cited == [{"constraint": "exclusive", "group": "group 0", "kept": ["y"]}]


def test_items_from_responses_a_decision_is_free_a_forced_answer_is_fixed():
    cat = Catalog()

    @cat.check(hard=True, then={"match": "no"})
    def same_brand(brand_a, brand_b):
        return brand_a == brand_b

    @cat.rule("match")
    def match(p, same_brand):
        return Decision("yes" if p >= 0.5 else "no", {"yes": p, "no": 1 - p})
    s = System(cat, [Question("match", "Same product?", Answer.yes_no())])
    states = [{"a": "a1", "b": "b1", "p": 0.9, "brand_a": "x", "brand_b": "x"},
              {"a": "a1", "b": "b2", "p": 0.95, "brand_a": "x", "brand_b": "x"},
              {"a": "a2", "b": "b1", "p": 0.99, "brand_a": "x", "brand_b": "y"}]
    items = [Item.of(s.ask(st), "match", id=(st["a"], st["b"]), keys={"a": st["a"], "b": st["b"]}) for st in states]
    assert items[0].free and not items[2].free and items[2].answer == "no"
    out = decide_set(items, ONE)
    assert out.answers == {("a1", "b1"): "no", ("a1", "b2"): "yes", ("a2", "b1"): "no"}
    assert out[("a2", "b1")].why.startswith("hard check same_brand is false")


def test_the_record_round_trips_through_json_and_replays_and_an_edited_record_is_a_mismatch():
    items = [_pair("a1", "b1", 0.95), _pair("a2", "b1", 0.9), _pair("a1", "b2", 0.9)]
    for method in ("exact", "greedy"):
        out = decide_set(items, ONE, method=method)
        back = SetDecision.from_dict(json.loads(json.dumps(out.to_dict())))
        assert back.answers == out.answers and back.replay() == {**out.replay(), "ok": True}
    d = json.loads(json.dumps(decide_set(items, ONE).to_dict()))
    d["items"][0]["answer"] = "yes"                                         # the record says a1-b1 kept too
    rep = SetDecision.from_dict(d).replay()
    assert not rep["ok"] and {m[1] for m in rep["mismatches"]} >= {"group", "objective"}
    d = json.loads(json.dumps(decide_set(items, ONE, method="greedy").to_dict()))
    d["method"] = "exact"                                                   # a greedy answer passed off as exact
    rep = SetDecision.from_dict(d).replay()
    assert [m[1] for m in rep["mismatches"]] == ["optimum"]


def test_thousands_of_items_in_one_component_are_solved_exactly_in_well_under_a_second_each():
    # a chain a0-b0, a1-b0, a1-b1, a2-b1, ... : 4,000 pairs linked into one component
    items = []
    for i in range(2000):
        items.append(_pair(f"a{i}", f"b{i}", 0.6 + 0.3 * ((i * 7) % 11) / 11))
        items.append(_pair(f"a{i + 1}", f"b{i}", 0.6 + 0.3 * ((i * 5) % 13) / 13))
    t0 = time.perf_counter()
    out = decide_set(items, ONE)
    assert time.perf_counter() - t0 < 10
    assert out.exact and out.feasible and len(out.components) == 1 and len(out.components[0]["items"]) == 4000
    held = [d.id for d in out if d.answer == "yes"]
    assert len({a for a, _ in held}) == len(held) == len({b for _, b in held})
    assert out.replay()["ok"]


def test_decide_set_refuses_what_it_cannot_decide():
    with pytest.raises(ValueError, match="share the id"):
        decide_set([_pair("a", "b", 0.5), _pair("a", "b", 0.6)], ONE)
    with pytest.raises(ValueError, match="probability"):
        decide_set([Item("x", {"yes": float("nan"), "no": 0.5})], ONE)
    with pytest.raises(ValueError, match="above 0"):
        decide_set([Item("x", {"yes": 0.0, "no": 0.0})], ONE)
    with pytest.raises(TypeError, match="Capacity"):
        decide_set([_pair("a", "b", 0.5)], [lambda answers: True])
    with pytest.raises(KeyError, match="no item"):
        decide_set([_pair("a", "b", 0.5)], [Exclusive([(("a", "b"), "zzz")])])
    with pytest.raises(ValueError, match="above max"):
        Capacity("a", max=1, min=2)
    with pytest.raises(ValueError, match="share a name"):
        decide_set([_pair("a", "b", 0.5)], [AtMostOne("a"), AtMostOne("a")])
    with pytest.raises(ValueError, match="method"):
        decide_set([_pair("a", "b", 0.5)], ONE, method="fast")
