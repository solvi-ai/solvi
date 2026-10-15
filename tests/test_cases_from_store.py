"""Regression cases from recorded decisions (testing.cases_from_store) and boundary inputs around them
(testing.boundary): the cases run under solvi test, pin the current answers, and catch a changed threshold."""
import json

import pytest

from solvi import Answer, Catalog, Question, System, testing
from solvi.core.store import JSONLStorage


def _system(limit=100, storage=None):
    cat = Catalog()

    @cat.check(hard=True, then={"approve": False})
    def known(customer):
        return customer != "blocked"

    @cat.rule("approve")
    def approve(amount, customer) -> bool:
        return amount < limit

    @cat.rule("tier")
    def tier(amount):
        return "big" if amount >= 1000 else "small"

    qs = [Question("approve", "Approve?", Answer.yes_no(), requires=["known"]),
          Question("tier", "Tier?", Answer.choice(["big", "small"]))]
    return System(cat, qs, storage=storage)


@pytest.mark.parametrize("record", ["full", "compact"])
def test_cases_from_store_pin_the_recorded_answers(tmp_path, record):
    st = JSONLStorage(str(tmp_path / "d.jsonl"), record=record)
    s = _system(storage=st)
    for a, c in [(50, "ann"), (50, "ann"), (150, "bob"), (5000, "cy"), (20, "blocked"), (60, "dan")]:
        s.ask({"amount": a, "customer": c})
    cases = testing.cases_from_store(st)
    assert len(cases) == 5                            # the repeated input once
    assert cases[0]["state"] == {"amount": 50, "customer": "ann"}
    assert cases[0]["expected"] == {"approve": "yes", "tier": "small"}
    blocked = next(c for c in cases if c["state"]["customer"] == "blocked")
    assert blocked["status"]["approve"] == "forced" and blocked["expected"]["approve"] == "no"
    json.dumps(cases)
    for c in cases:
        assert testing.run_case(s, c, c["state"]).ok
    few = testing.cases_from_store(st, n=3)          # every distinct outcome first, then in store order
    outs = {json.dumps([c["expected"], c["status"]], sort_keys=True) for c in few}
    assert len(few) == 3 and len(outs) == 3


def test_boundary_inputs_pin_current_answers_and_catch_a_moved_threshold(tmp_path):
    s = _system()
    seeds = [{"name": "a", "state": {"amount": 50, "customer": "ann"}, "expected": {"approve": "yes"}},
             {"name": "b", "state": {"amount": 99, "customer": "bob"}, "expected": {"approve": "yes"}}]
    cases = testing.boundary(seeds, n=40, system=s)
    assert 10 <= len(cases) <= 40
    states = [json.dumps(c["state"], sort_keys=True) for c in cases]
    assert len(set(states)) == len(states) and not any(c["state"] in [x["state"] for x in seeds] for c in cases)
    assert any(c["state"].get("amount") == 100 for c in cases)       # just past the largest value seen
    assert any(c["state"].get("customer") == "" for c in cases)
    assert all(c["note"].startswith("boundary: ") for c in cases)
    assert all(testing.run_case(s, c, c["state"]).ok for c in cases)
    moved = _system(limit=101)                        # a changed threshold: a boundary case fails, the seeds pass
    assert all(testing.run_case(moved, c, c["state"]).ok for c in seeds)
    assert not all(testing.run_case(moved, c, c["state"]).ok for c in cases)
    assert testing.boundary(seeds, n=40, system=s) == cases            # deterministic


def test_boundary_without_a_system_gives_states_and_drops_nondeterministic_inputs():
    plain = testing.boundary([{"state": {"x": 3, "tags": ["a", "b"], "who": {"age": 30}}}], n=200)
    assert plain and all(set(c) == {"name", "state", "note"} for c in plain)
    assert any(c["state"]["tags"] == [] for c in plain)
    assert any(c["state"]["who"].get("age") == 31 for c in plain)    # one level into a dict
    cat = Catalog()
    n = [0]

    @cat.rule("q")
    def q(x) -> bool:
        n[0] += 1
        return x > 0 and n[0] % 2 == 0

    s = System(cat, [Question("q", "Q?", Answer.yes_no())])
    with pytest.warns(UserWarning, match="nondeterministic"):
        kept = testing.boundary([{"state": {"x": 3}}], n=10, system=s)
    assert kept and all(c["state"]["x"] <= 0 for c in kept)          # only the inputs whose answer does not move
