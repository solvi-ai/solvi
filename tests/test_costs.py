"""Costs from measurements: System(..., producers="equivalent", costs="measured") feeds the run times system.costs
measures to the cost-optimal planner — after a warm-up it picks the fastest of equivalent producers, switches when that one
slows down, rechecks a producer it stopped using, stops switching once frozen, and every plan record says which cost
(declared, warm-up, measured, recheck, frozen) decided each choice."""
import asyncio
import time

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.learned import MeasuredCosts
from solvi.strategy import ModelStrategist


def rates(delay):
    """A rate from a live feed (declared first) or from a local table: interchangeable, the same value."""
    cat = Catalog()

    @cat.fn(provides="rate")
    def rate_live(currency):
        time.sleep(delay["live"])
        return 1.08

    @cat.fn(provides="rate")
    def rate_table(currency):
        time.sleep(delay["table"])
        return 1.08

    @cat.rule("convert")
    def convert(rate):
        return "yes" if rate > 1 else "no"
    return cat, [Question("convert", "Convert?", Answer.yes_no())]


def used(res):
    return next(r for r in res.trace.records if r.name == "rate").producer


def plan_costs(res):
    return next(r for r in res.trace.records if r.kind == "plan").extra["costs"]


def test_declared_costs_keep_the_declaration_order():
    cat, qs = rates({"live": 0.02, "table": 0.0})
    s = System(cat, qs, producers="equivalent")
    assert isinstance(s.strategist, ModelStrategist) and s.cost_policy is None
    got = [used(s.ask({"currency": "EUR"})) for _ in range(4)]
    assert got == ["rate_live"] * 4                                         # unit costs: a tie, the first declared
    assert "costs" not in next(r for r in s.ask({"currency": "EUR"}).trace.records if r.kind == "plan").extra


def test_measured_costs_pick_the_fast_producer_switch_when_it_slows_and_freeze():
    delay = {"live": 0.03, "table": 0.0}
    cat, qs = rates(delay)
    s = System(cat, qs, producers="equivalent", costs=MeasuredCosts(min_samples=2, recheck=None))

    def ask():
        r = s.ask({"currency": "EUR"})
        assert r["convert"].answer == "yes" and r.trace.replay(s)["ok"]
        return r
    rs = [ask() for _ in range(6)]
    assert [used(r) for r in rs] == ["rate_live"] * 2 + ["rate_table"] * 4  # warm-up: each measured twice, then the fast
    first, last = plan_costs(rs[0])["facts"]["rate"], plan_costs(rs[-1])["facts"]["rate"]
    assert first["producers"]["rate_live"]["from"] == "warm-up" and first["chosen"] == "rate_live"
    assert last["chosen"] == "rate_table" and plan_costs(rs[-1])["mode"] == "measured"
    assert last["producers"]["rate_table"]["from"] == last["producers"]["rate_live"]["from"] == "measured"
    assert last["producers"]["rate_live"]["ms"] >= 25 and last["producers"]["rate_table"]["ms"] < 5
    assert last["why"].startswith("cheapest plan: rate_table measured")
    delay["table"] = 0.08                                                   # the table slows down: the planner switches
    got = [used(ask()) for _ in range(6)]
    assert got[0] == "rate_table" and got[-1] == "rate_live"
    s.freeze_costs()                                                        # frozen: no switching any more
    delay["live"], delay["table"] = 0.08, 0.0
    rs = [ask() for _ in range(5)]
    assert {used(r) for r in rs} == {"rate_live"} and plan_costs(rs[-1])["mode"] == "frozen"
    assert plan_costs(rs[-1])["facts"]["rate"]["producers"]["rate_live"]["from"] == "frozen: measured"
    s.unfreeze_costs()                                                      # measuring went on: back to the table
    got = [used(ask()) for _ in range(8)]
    assert got[-1] == "rate_table"


def test_a_producer_it_stopped_using_is_rechecked():
    delay = {"live": 0.0, "table": 0.04}
    cat, qs = rates(delay)
    s = System(cat, qs, producers="equivalent", costs=MeasuredCosts(min_samples=1, recheck=3, alpha=1.0))
    got = [used(s.ask({"currency": "EUR"})) for _ in range(3)]
    assert got == ["rate_live", "rate_table", "rate_live"]
    delay["live"], delay["table"] = 0.04, 0.0                               # the live feed got slow, the table fast
    rs = [s.ask({"currency": "EUR"}) for _ in range(12)]
    got = [used(r) for r in rs]
    assert got[-6:].count("rate_table") >= 4                                # back to the table once it was rechecked ...
    froms = [p["from"] for r in rs for p in plan_costs(r)["facts"]["rate"]["producers"].values()]
    assert "recheck" in froms                                               # ... and each unused one gets a trial now and then
    assert all(used(r) == "rate_live" for r in rs if plan_costs(r)["facts"]["rate"]["producers"]["rate_live"]["from"]
               == "recheck")


def test_declared_costs_are_the_prior_until_measured_and_aask_uses_the_same_planner():
    cat = Catalog()

    @cat.fn(provides="score", cost=300)
    async def score_api(customer):
        await asyncio.sleep(0.02)
        return 0.5

    @cat.fn(provides="score")
    def score_cache(customer):
        return 0.5

    @cat.rule("ok")
    def ok(score):
        return score < 0.9
    s = System(cat, [Question("ok", "OK?", Answer.yes_no())], producers="equivalent", costs="measured")
    for _ in range(4):
        r = asyncio.run(s.aask({"customer": "c1"}))
        assert next(x for x in r.trace.records if x.name == "score").producer == "score_cache"
    info = plan_costs(r)["facts"]["score"]["producers"]
    assert info["score_api"] == {"ms": 300.0, "from": "declared", "runs": 0} and info["score_cache"]["from"] == "measured"


def test_settings_and_errors():
    cat, qs = rates({"live": 0, "table": 0})
    with pytest.raises(ValueError, match="cost-optimal planner"):
        System(cat, qs, costs="measured")
    with pytest.raises(ValueError, match="costs must be"):
        System(cat, qs, producers="equivalent", costs="fastest")
    with pytest.raises(ValueError, match="producers=\"equivalent\" too"):
        System(cat, qs, producers="equivalent", strategist=ModelStrategist())
    with pytest.raises(ValueError):
        MeasuredCosts(min_samples=0)
    with pytest.raises(ValueError, match="freeze_costs"):
        System(cat, qs, producers="equivalent").freeze_costs()
    s = System(cat, qs, strategist=ModelStrategist(producers="equivalent"), costs=MeasuredCosts(alpha=0.5))
    assert s.costs.alpha == 0.5 and "MeasuredCosts" in repr(s.cost_policy)
    s.ask({"currency": "EUR"})
    frozen = s.freeze_costs()
    assert set(frozen) >= {"rate_live", "rate_table"} and frozen["rate_table"] == 1.0   # never ran: unit
