"""solvi.search: candidates through a System's checks — the first accepted in a space's order, the best by an
objective, a depth-first Tree pruned by failed hard checks and a bound, a budget that makes the result not exact; the
facts read from the state computed once (a model called once, not per candidate); the winner asked in full, stored
and replayable; nothing accepted escalates; what it cannot run is refused."""
import json
import os
import tempfile

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, JSONLStorage, Question, System
from solvi.refine import Fail
from solvi.search import SearchRun, Tree, search


class Busy(BaseModel):
    day: str
    start: int
    end: int


READS = {"n": 0}


def _calendar(storage=None):
    """A meeting slot: the problem text read into busy times (counted), a slot given as day and start."""
    cat = Catalog()

    @cat.fn
    def busy(problem: str) -> list[Busy]:
        READS["n"] += 1
        out = []
        for part in problem.split(";"):
            d, a, b = part.split()
            out.append(Busy(day=d, start=int(a), end=int(b)))
        return out

    @cat.check(hard=True, then={"ok": "no"})
    def in_hours(start: int) -> bool:
        return 9 <= start <= 16

    @cat.check(hard=True, then={"ok": "no"})
    def nobody_busy(busy: list[Busy], day: str, start: int) -> bool:
        clash = [f"busy {b.day} {b.start}-{b.end}" for b in busy if b.day == day and b.start <= start < b.end]
        return Fail(*clash) if clash else True

    @cat.rule("ok")
    def ok(in_hours, nobody_busy) -> bool:
        return True
    return System(cat, [Question("ok", "Does the slot work?", Answer.yes_no(), checkpoints=["in_hours", "nobody_busy"])],
                  storage=storage)


PROBLEM = "Mon 9 12;Mon 13 17;Tue 9 10"


def test_a_dict_of_domains_gives_the_first_accepted_slot_in_order_and_counts_what_rejected_the_others():
    READS["n"] = 0
    run = search(_calendar(), {"problem": PROBLEM}, "ok", {"day": ["Mon", "Tue"], "start": list(range(8, 18))})
    assert run.best == {"day": "Mon", "start": 12} and run.exact and run.accepted == 1
    assert run.asked == 5 and run.rejected == {"in_hours": 1, "nobody_busy": 3}
    assert run.response["ok"].answer == "yes" and run.held == ["busy"]
    assert READS["n"] == 2                       # once for the search, once for the winner's full ask
    assert "the first accepted in the space's order" in run.why_exact


def test_keep_two_without_an_objective_says_whether_the_first_is_the_only_one():
    run = search(_calendar(), {"problem": PROBLEM}, "ok", {"day": ["Mon"], "start": list(range(8, 18))}, keep=2)
    assert run.best == {"day": "Mon", "start": 12} and run.accepted == 1 and "(and the only one)" in run.why_exact
    run = search(_calendar(), {"problem": PROBLEM}, "ok", {"day": ["Tue"], "start": list(range(8, 18))}, keep=2)
    assert [c["start"] for c, _ in run.kept] == [10, 11] and "not the only one" in run.why_exact


def test_an_objective_keeps_the_best_by_a_function_or_by_a_fact_of_the_response_largest_or_smallest():
    cat = Catalog()

    @cat.fn
    def size(team: list) -> int:
        return len(team)

    @cat.check(hard=True, then={"ok": "no"})
    def no_rivals(team: list) -> bool:
        return not ({"ann", "bob"} <= set(team))

    @cat.rule("ok")
    def ok(no_rivals) -> bool:
        return True
    s = System(cat, [Question("ok", "A valid team?", Answer.yes_no(), checkpoints=["no_rivals", "size"])])
    teams = [["ann"], ["ann", "bob"], ["ann", "cy", "dee"], ["bob", "cy"], ["ann", "bob", "cy", "dee"]]
    run = search(s, {}, "ok", teams, into="team", objective=len, keep=2)
    assert run.best == ["ann", "cy", "dee"] and run.value == 3 and run.kept == [(["ann", "cy", "dee"], 3), (["bob", "cy"], 2)]
    assert run.rejected == {"no_rivals": 2} and run.objective == "len"
    small = search(s, {}, "ok", teams, into="team", objective="size", maximize=False)
    assert small.best == ["ann"] and small.value == 1 and small.objective == "fact size"


def _trip():
    """Orders of cities along direct flights; every city once (prunable), the flights (prunable), all cities (not)."""
    cat = Catalog()
    flights = {("A", "B"), ("B", "C"), ("C", "D"), ("B", "D"), ("D", "E"), ("A", "C")}

    @cat.check(hard=True, then={"ok": "no"})
    def once(order: list) -> bool:
        return len(set(order)) == len(order)

    @cat.check(hard=True, then={"ok": "no"})
    def direct(order: list) -> bool:
        return all((a, b) in flights or (b, a) in flights for a, b in zip(order, order[1:]))

    @cat.check(hard=True, then={"ok": "no"})
    def starts_at_a(order: list) -> bool:
        return not order or order[0] == "A"

    @cat.check(hard=True, then={"ok": "no"})
    def all_cities(order: list) -> bool:
        return len(order) == 5

    @cat.rule("ok")
    def ok(once, direct, starts_at_a, all_cities) -> bool:
        return True
    return System(cat, [Question("ok", "A valid trip?", Answer.yes_no(),
                                 checkpoints=["once", "direct", "starts_at_a", "all_cities"])])


CITIES = ["A", "B", "C", "D", "E"]
TREE = Tree([], lambda o: [o + [c] for c in CITIES if c not in o], complete=lambda o: len(o) == 5)


def test_a_tree_pruned_by_failed_hard_checks_finds_what_the_full_walk_finds_with_far_fewer_asks():
    s = _trip()
    full = search(s, {}, "ok", TREE, into="order", keep=10)
    cut = search(s, {}, "ok", TREE, into="order", keep=10, prune=["direct", "starts_at_a"])
    assert [c for c, _ in full.kept] == [c for c, _ in cut.kept] == [["A", "B", "C", "D", "E"], ["A", "C", "B", "D", "E"]]
    assert full.asked == 120 and cut.asked == 30 and cut.exact and full.exact
    assert set(cut.pruned) == {"direct", "starts_at_a"}
    assert "given that the prune checks direct, starts_at_a stay false below a node" in cut.why_exact


def test_a_bound_stops_the_walk_once_nothing_below_can_beat_what_is_kept():
    s = _trip()
    meet = Tree([], lambda o: [o + [c] for c in CITIES if c not in o], complete=lambda o: True, bound=lambda o: 3)
    cat = Catalog()

    @cat.check(hard=True, then={"ok": "no"})
    def short(order: list) -> bool:
        return len(order) <= 3

    @cat.rule("ok")
    def ok(short) -> bool:
        return True
    s = System(cat, [Question("ok", "?", Answer.yes_no(), checkpoints=["short"])])
    run = search(s, {}, "ok", meet, into="order", objective=len, prune=["short"])
    assert run.best == ["A", "B", "C"] and run.value == 3 and run.pruned.get("bound", 0) > 0 and run.asked < 20
    with pytest.raises(ValueError, match="bound needs an objective"):
        search(s, {}, "ok", meet, into="order")


def test_a_budget_stops_the_search_and_the_result_says_it_is_not_exact():
    run = search(_trip(), {}, "ok", TREE, into="order", keep=10, budget=40)
    assert run.exhausted and not run.exact and run.asked == 40
    assert run.why_exact.startswith("not exact: the budget of 40 asks stopped the search")
    assert run.best == ["A", "B", "C", "D", "E"]                   # found before the budget ran out
    back = Tree([], lambda o: [o + [c] for c in reversed(CITIES) if c not in o], complete=lambda o: len(o) == 5)
    none = search(_trip(), {}, "ok", back, into="order", budget=5)
    assert none.best is None and none.escalation.startswith("nothing accepted within the budget of 5 asks")


def test_a_model_reading_the_state_is_called_once_and_hold_false_calls_it_per_candidate():
    calls = {"n": 0}

    class Reader:
        model_id = "test/reader"
        version = "1"

    cat = Catalog()

    @cat.fn(model=Reader())
    def limit(problem: str) -> int:
        calls["n"] += 1
        return int(problem)

    @cat.check(hard=True, then={"ok": "no"})
    def under(limit: int, x: int) -> bool:
        return x <= limit

    @cat.rule("ok")
    def ok(under) -> bool:
        return True
    s = System(cat, [Question("ok", "?", Answer.yes_no(), checkpoints=["under"])])
    run = search(s, {"problem": "7"}, "ok", list(range(20)), into="x", objective=lambda x: x)
    assert run.best == 7 and run.asked == 20 and calls["n"] == 2 and run.held == ["limit"]
    calls["n"] = 0
    search(s, {"problem": "7"}, "ok", list(range(20)), into="x", objective=lambda x: x, hold=False)
    assert calls["n"] == 21


def test_the_space_can_be_read_from_the_facts_and_the_winner_is_stored_and_replays():
    with tempfile.TemporaryDirectory() as d:
        store = JSONLStorage(os.path.join(d, "s.jsonl"))
        s = _calendar(storage=store)
        run = search(s, {"problem": PROBLEM}, "ok",
                     lambda facts: {"day": sorted({b.day for b in facts["busy"]}), "start": list(range(9, 17))})
        assert run.best == {"day": "Mon", "start": 12}
        assert run.response.stored_id is not None and len(list(store.query())) == 1     # only the winner is stored
        assert s.stats["asks"] == 1                                                     # the searched asks are not counted
        assert run.replay(s)["ok"]
        back = SearchRun.from_dict(json.loads(json.dumps(run.to_dict())), catalog=s.catalog)
        assert back.best == run.best
        rep = back.replay(s)
        assert rep["ok"], rep["mismatches"]
        d2 = json.loads(json.dumps(run.to_dict()))
        d2["response"]["trace"]["init"]["start"] = 9                                    # an edited record
        assert not SearchRun.from_dict(d2, catalog=s.catalog).replay(s)["ok"]


def test_nothing_accepted_escalates_with_the_checks_that_rejected_and_an_abstention_is_never_accepted():
    run = search(_calendar(), {"problem": PROBLEM}, "ok", {"day": ["Mon"], "start": [9, 10, 13]})
    assert run.best is None and run.exact and run.escalation == "nothing accepted in the whole space: nobody_busy (3)"
    assert run.why_exact.startswith("exact") and "no candidate of the space is accepted" in run.why_exact
    bad = search(_calendar(), {"problem": PROBLEM}, "ok", [{"day": "Mon"}], into="slot")   # no day / start given
    assert bad.best is None and list(bad.rejected)[0].startswith("abstained")


def test_search_refuses_what_it_cannot_run():
    s = _trip()
    with pytest.raises(ValueError, match="not a check"):
        search(s, {}, "ok", TREE, into="order", prune=["dirct"])
    with pytest.raises(ValueError, match="into="):
        search(s, {}, "ok", TREE)
    with pytest.raises(ValueError, match="leave into=None"):
        search(s, {}, "ok", {"order": [["A"]]}, into="order")
    with pytest.raises(ValueError, match="already gives"):
        search(s, {"order": ["A"]}, "ok", TREE, into="order")
    with pytest.raises(ValueError, match="needs a Tree"):
        search(s, {}, "ok", [["A"]], into="order", prune=["direct"])
    with pytest.raises(ValueError, match="not a question"):
        search(s, {}, "nope", TREE, into="order")
    with pytest.raises(TypeError, match="space is"):
        search(s, {}, "ok", 5, into="order")
    with pytest.raises(ValueError, match="not computed for the question"):
        search(s, {}, "ok", [CITIES], into="order", objective="length")
