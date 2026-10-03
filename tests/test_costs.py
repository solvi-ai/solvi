"""Costs of the cost-optimal planner: the declared ones (cost=, 1 when undeclared). Planning on measured run times
(cost_policy="measured", MeasuredCosts, freeze_costs) was removed in 1.0: it showed no measured benefit."""
import time

import pytest

from solvi import Answer, Catalog, Question, System
from solvi.strategy import CostStrategist


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
    assert isinstance(s.strategist, CostStrategist)
    got = [used(s.ask({"currency": "EUR"})) for _ in range(4)]
    assert got == ["rate_live"] * 4                                         # unit costs: a tie, the first declared
    assert "costs" not in next(r for r in s.ask({"currency": "EUR"}).trace.records if r.kind == "plan").extra


def test_measured_costs_were_removed():
    """1.0 trim: cost_policy="measured", solvi.costs.MeasuredCosts and freeze_costs / unfreeze_costs showed no measured
    benefit; each says so and names what to use (declared cost= on the parts)."""
    import solvi.costs
    cat, qs = rates({"live": 0, "table": 0})
    with pytest.raises(ValueError, match=r'cost_policy="measured" .* was removed in 1.0: .* declare cost='):
        System(cat, qs, producers="equivalent", cost_policy="measured")
    with pytest.raises(ValueError, match='cost_policy must be "declared"'):
        System(cat, qs, producers="equivalent", cost_policy="fastest")
    with pytest.raises(AttributeError, match="MeasuredCosts .* was removed in 1.0"):
        _ = solvi.costs.MeasuredCosts
    s = System(cat, qs, producers="equivalent")
    for name in ("freeze_costs", "unfreeze_costs"):
        with pytest.raises(AttributeError, match=rf"System.{name}\(\) was removed in 1.0: use declared costs"):
            getattr(s, name)()
    with pytest.raises(ValueError, match="producers=\"equivalent\" too"):
        System(cat, qs, producers="equivalent", strategist=CostStrategist())
