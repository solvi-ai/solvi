"""The code strategist (solvi.strategy.CostStrategist): code plans, dead ends, mandatory checks, the plan record in the
trace and its replay, facts derivable from each other. (The model strategist and name matching were removed in 1.0.)"""
import time
from typing import Literal

import pytest

from solvi import Catalog, Question, System
from solvi.strategist import plan as det_plan
from solvi.strategy import CostStrategist, mandatory_checks, search, validate


def chain():
    """a2 has a dead-end producer (reads a feed nobody gives) and a costly shortcut; a3 a cheap and a costly producer."""
    cat = Catalog()

    @cat.fn(cost=1)
    def a1(g1: float) -> float:
        return g1 + 1

    @cat.fn(provides="a2", cost=1)
    def a2_feed(partner_feed: float) -> float:
        """Reads the partner feed. Fast."""
        return partner_feed

    @cat.fn(provides="a2", cost=3)
    def a2_std(a1: float) -> float:
        """Standard step."""
        return a1 * 2

    @cat.fn(provides="a3", cost=20)
    def a3_slow(a1: float, g2: float) -> float:
        """Slow: rebuilds it from scratch."""
        return a1 * 2 + g2

    @cat.fn(provides="a3")
    def a3_std(a2: float, g2: float) -> float:
        """Standard step."""
        return a2 + g2

    @cat.check(hard=True, then={"ok": "rejected"})
    def policy(a2: float) -> bool:
        return a2 < 100

    @cat.check
    def audit(a3: float) -> bool:
        return a3 > 0

    @cat.rule("ok")
    def ok(a3: float) -> Literal["approved", "review", "rejected"]:
        return "approved" if a3 > 5 else "review"
    return cat, [Question("ok", "approve?", None)]


ST = {"g1": 3.0, "g2": 1.0}


def test_dead_end_breaks_the_deterministic_strategist_not_this_one():
    cat, qs = chain()
    assert System(cat, qs).ask(ST)["ok"].status == "abstain"          # union of inputs: partner_feed is never given
    s = System(cat, qs, strategist=CostStrategist())
    r = s.ask(ST)
    assert r["ok"].answer == "approved"
    assert r.trace.replay(s, r.flow)["ok"]
    assert r.trace.records[-1].kind == "plan" and r.trace.records[-1].provenance == "computed"


def test_declared_mode_equals_deterministic_without_dead_ends():
    cat = Catalog()

    @cat.fn
    def x(a: int) -> int:
        return a + 1

    @cat.fn(provides="y")
    def y1(x: int) -> int:
        return x * 2

    @cat.fn(provides="y")
    def y2(a: int) -> int:
        return a * 3

    @cat.rule("q")
    def q(y: int) -> bool:
        return y > 4
    qs = [Question("q", "?", None)]
    for a in (1, 2, 5):
        d = System(cat, qs).ask({"a": a})["q"]
        m = System(cat, qs, strategist=CostStrategist()).ask({"a": a})["q"]
        assert (d.answer, d.status) == (m.answer, m.status)


def test_equivalent_mode_is_cheapest_and_keeps_mandatory_checks():
    cat, qs = chain()
    ms = CostStrategist(producers="equivalent", costs={"a3_std": 1.0})
    flow = ms.plan(cat, qs, set(ST))
    assert ms.last["choice"]["a3"] == "a3_std" and ms.last["choice"]["a2"] == "a2_std"
    assert ms.last["mandatory"] == {"ok": ["policy"]}
    assert "policy" in flow.per_question["ok"]
    assert validate(cat, flow, qs, set(ST), ms.last["mandatory"]) == []
    # a flow without the mandatory check fails validation
    bad = det_plan(cat, [Question("ok", "approve?", None)], set(ST))
    assert validate(cat, bad, qs, set(ST), {"ok": ["policy"]})


def test_search_milp_and_branch_and_bound_agree():
    cat, qs = chain()
    gov = mandatory_checks(cat, qs, set(ST))
    must = [c for cs in gov.values() for c in cs]
    for costs in (None, {"a3_std": 1.0}, {"a3_std": 50.0}):
        a = search(cat, qs, set(ST), costs, extra=must, method="milp")
        b = search(cat, qs, set(ST), costs, extra=must, method="bnb")
        assert a.cost == pytest.approx(b.cost) and a.choice == b.choice


def test_the_plan_is_recorded_replayed_and_a_tampered_plan_does_not_verify():
    cat, qs = chain()
    s = System(cat, qs, strategist=CostStrategist(producers="equivalent"))
    r = s.ask(ST)
    rec = r.trace.records[-1]
    assert rec.kind == "plan" and rec.provenance == "computed" and rec.model is None
    assert rec.extra["strategist"] == "code" and rec.extra["segments"] == []
    assert r.trace.replay(s, r.flow)["ok"] and r["ok"].answer == "approved"
    rec.value["choice"]["a3"] = "a3_nope"                 # tampering: hash broken and the plan no longer verifies
    assert not r.trace.replay(s, r.flow)["ok"]


def test_the_model_strategist_and_name_matching_are_gone():
    import importlib

    from solvi import strategy
    for mod in ("solvi.segment_model", "solvi.aliases"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(mod)
    assert not hasattr(strategy, "ModelStrategist") and not hasattr(strategy, "segments")


def test_deterministic_strategist_is_linear_on_diamonds():
    cat = Catalog()
    prev, prev2 = "g", "g"
    for i in range(60):                                    # every fact reachable by two routes: exponential without memo
        name = f"m{i}"
        cat.fn(_two(name, prev, prev2 if prev2 != prev else "h"))
        prev2, prev = prev, name

    @cat.rule("q")
    def q(m59) -> bool:
        return m59 > 0
    t0 = time.perf_counter()
    det_plan(cat, [Question("q", "?", None)], {"g", "h"})
    assert time.perf_counter() - t0 < 2.0


def _two(name, a, b):
    import inspect

    def f(**kw):
        return kw[a] + kw[b]
    f.__name__ = name
    f.__signature__ = inspect.Signature([inspect.Parameter(a, inspect.Parameter.POSITIONAL_OR_KEYWORD),
                                         inspect.Parameter(b, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
    return f


def test_cost_strategist_record_false_keeps_the_plan_out_of_the_trace_and_fallbacks_false_keeps_one_producer():
    cat, qs = chain()
    rec = System(cat, qs, strategist=CostStrategist(producers="equivalent")).ask(ST)
    assert rec.trace.records[-1].kind == "plan"
    off = System(cat, qs, strategist=CostStrategist(producers="equivalent", record=False)).ask(ST)
    assert all(r.kind != "plan" for r in off.trace.records) and off["ok"].answer == rec["ok"].answer
    one = System(cat, qs, strategist=CostStrategist(producers="equivalent", keep_alternatives=False)).ask(ST)
    assert all(len(st.part.alternatives or [st.part]) == 1 for st in one.flow.steps) and one["ok"].answer == "approved"


def test_plan_raises_plan_error_for_a_checkpoint_that_is_not_in_the_catalog():
    from solvi.strategist import PlanError
    cat, _ = chain()
    with pytest.raises(PlanError, match="requires .nope., which is not in the catalog"):
        det_plan(cat, [Question("ok", "", None, requires=["nope"])], set(ST))


# ---------------------------------------------------------------- facts derivable from each other
def _net_gross():
    cat = Catalog()

    @cat.fn(provides="net", cost=1)
    def net_given(net_in: float) -> float:
        return net_in

    @cat.fn(provides="net", cost=2)
    def net_from_gross(gross: float) -> float:
        return gross / 1.25

    @cat.fn(provides="gross", cost=1)
    def gross_given(gross_in: float) -> float:
        return gross_in

    @cat.fn(provides="gross", cost=2)
    def gross_from_net(net: float) -> float:
        return net * 1.25

    @cat.rule("big")
    def big(gross: float, net: float) -> bool:
        return gross > 100 and net > 0
    return cat, [Question("big", "", None)]


@pytest.mark.parametrize("strategist", [
    lambda: CostStrategist(), lambda: CostStrategist(producers="equivalent"),
    lambda: CostStrategist(producers="equivalent", keep_alternatives=False)], ids=["declared", "equivalent", "no_fallbacks"])
@pytest.mark.parametrize("state, order, answer", [({"net_in": 100.0}, ["net", "gross"], "yes"),
                                                  ({"gross_in": 100.0}, ["gross", "net"], "no")])
def test_facts_derivable_from_each_other_are_planned_run_and_replayed(strategist, state, order, answer):
    cat, qs = _net_gross()
    st = strategist()
    s = System(cat, qs, strategist=st)
    res = s.ask(dict(state))
    assert st.last["fallback"] is None
    assert [x.part.name for x in res.flow.steps] == order + ["answer:big"]
    assert (res["big"].answer, res["big"].status, res["big"].confidence) == (answer, "ok", 1.0)
    for step in res.flow.steps[:2]:                                # no producer kept in the flow reads its own fact back
        assert len(step.part.alternatives) == 1
    assert res.trace.replay(s, res.flow)["ok"]


def test_a_fallback_producer_is_dropped_only_when_it_would_read_its_own_fact():
    cat, qs = _net_gross()
    st = CostStrategist(producers="equivalent")
    res = System(cat, qs, strategist=st).ask({"net_in": 100.0, "gross_in": 130.0})
    alts = {s.part.name: [a.name for a in s.part.alternatives] for s in res.flow.steps[:2]}
    assert alts["net"][0] == "net_given" and alts["gross"][0] == "gross_given"   # both given: the cheap ones first,
    assert sorted(len(v) for v in alts.values()) == [1, 2]                       # and one derived fallback, not both
    assert res["big"].answer == "yes"


def test_path_confidence_follows_the_producer_that_ran_and_survives_a_ring_of_facts():
    from solvi.runtime import path_confidence
    cat, qs = _net_gross()
    res = System(cat, qs, strategist=CostStrategist(producers="equivalent", keep_alternatives=False)).ask({"net_in": 100.0})
    assert path_confidence(cat, res.trace, ["gross", "net"]) == 1.0            # the full catalog: net ⇄ gross


def test_solvi_check_calls_facts_derived_from_each_other_a_cycle_only_for_the_deterministic_strategist():
    from solvi.check import lint
    cat, qs = _net_gross()
    rep = lint(System(cat, qs, strategist=CostStrategist()))
    assert rep.ok and rep.codes() == ["mutual_producers"]
    rep = lint(System(cat, qs))                                     # the deterministic strategist cannot plan it
    assert "cycle" in rep.codes("error") and "CostStrategist" in str(rep)
    cat = Catalog()                                                 # a loop with no way in stays an error whatever plans

    @cat.fn
    def a(b): return b

    @cat.fn
    def b(a): return a

    @cat.rule("q")
    def q(a) -> bool: return True
    assert "cycle" in lint(System(cat, [Question("q", "", None)], strategist=CostStrategist())).codes("error")


def test_fit_serve_and_check_plan_with_the_systems_strategist_as_ask_does():
    from solvi.check import lint
    from solvi.serve import question_inputs
    cat, qs = chain()
    s = System(cat, qs, strategist=CostStrategist())
    assert s.ask(ST)["ok"].answer == "approved"
    assert {"a2", "a3"} <= set(s.facts_for(ST))                     # computed around the dead-end producer, as ask does
    info = question_inputs(s, "ok")
    assert "partner_feed" in info["properties"] and "partner_feed" not in info["required"]   # the dead end's input is
    assert lint(s).ok                                                                         # optional, not required
    det = System(*chain())
    assert "a2" not in det.facts_for(ST)                            # the deterministic strategist: unchanged
    assert "partner_feed" in question_inputs(det, "ok")["required"]


def test_example_17_runs(capsys):
    import runpy
    from pathlib import Path
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "examples" / "17_cost_strategist.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "abstain" in out and "approve = 'pay'" in out and "plan:strategy computed — replay ok: True" in out
    assert "chosen: fx_from_table" in out
