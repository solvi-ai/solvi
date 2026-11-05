import copy

import pytest

from solvi.core.costs import BudgetStop
from solvi.experimental.experience._common import WorkBudget, digest
from solvi.experimental.experience.acquisition import Probe, select_probe
from solvi.experimental.experience.worlds import WorldSet


def budget():
    return WorkBudget(100_000)


def test_joint_correlations_are_not_replaced_by_marginals():
    correlated = WorldSet([{"a": 0, "b": 0}, {"a": 1, "b": 1}])
    crossed = WorldSet([{"a": 0, "b": 1}, {"a": 1, "b": 0}])
    assert correlated.decide(lambda s: s["a"] == s["b"], budget()).value is True
    assert crossed.decide(lambda s: s["a"] == s["b"], budget()).value is False
    assert correlated.fingerprint != crossed.fingerprint


def test_null_answer_is_resolved_and_empty_model_is_not():
    assert WorldSet([{"x": 1}]).decide(lambda _: None, budget()).resolved
    result = WorldSet([]).decide(lambda _: "yes", budget())
    assert result.status == "contradiction" and not result.resolved


def test_exact_answer_types_and_incomplete_reasoning():
    w = WorldSet([{"x": True}, {"x": 1}])
    assert w.decide(lambda s: s["x"], budget()).status == "unknown"
    result = w.decide(lambda s: s["x"], WorkBudget(1))
    assert result.status == "budget" and not result.resolved and result.value is None


def test_external_observation_and_dependency_retraction_restore_joint_states():
    w = WorldSet([{"x": x, "y": y} for x in (0, 1) for y in (0, 1)])
    b = budget()
    a = w.observe("x", 1, "outside-x", b)
    both = a.observe("y", 0, "derived-y", b, depends_on=("outside-x",))
    assert both.worlds == ({"x": 1, "y": 0},)
    restored = both.retract("outside-x", b)
    assert restored.fingerprint == w.fingerprint
    assert len(restored) == 4 and restored.evidence == ()


def test_constraint_is_evaluated_on_full_base_for_later_retraction():
    w = WorldSet([{"x": 0}, {"x": 1}, {"x": 2}])
    b = budget()
    low = w.constrain("low", lambda s: s["x"] < 2, b)
    high = low.constrain("high", lambda s: s["x"] > 0, b)
    assert high.worlds == ({"x": 1},)
    assert high.retract("low", b).worlds == ({"x": 1}, {"x": 2})


def test_observation_contradiction_and_event_identity():
    w = WorldSet([{"x": 0}, {"x": 1}])
    b = budget()
    yes = w.observe("x", 1, "e", b)
    assert yes.observe("x", 1, "e", b) is yes
    with pytest.raises(ValueError, match="reused"):
        yes.observe("x", 0, "e", b)
    impossible = w.observe("x", 3, "e", b)
    assert impossible.decide(lambda s: s["x"], b).status == "contradiction"


def test_worlds_and_snapshots_do_not_share_mutable_input():
    raw = [{"x": [1]}]
    w = WorldSet(raw)
    raw[0]["x"].append(2)
    detached = w.worlds
    detached[0]["x"].append(3)
    snap = w.snapshot()
    restored = WorldSet.restore(snap)
    snap["base"][0]["x"].append(4)
    assert restored.worlds == w.worlds == ({"x": [1]},)


@pytest.mark.parametrize("worlds", [[{"x": 0}, {"y": 0}], [{"x": float("nan")}], [{1: "bad"}]])
def test_invalid_world_schema_values_and_limits(worlds):
    with pytest.raises(ValueError):
        WorldSet(worlds)
    with pytest.raises(ValueError, match="max_worlds"):
        WorldSet([{"x": x} for x in range(3)], max_worlds=2)


def test_restore_rejects_tampering_and_invalid_dependency_order():
    w = WorldSet([{"x": 0}, {"x": 1}]).observe("x", 0, "e", budget())
    s = w.snapshot()
    bad = copy.deepcopy(s)
    bad["base"][0]["x"] = 7
    with pytest.raises(ValueError, match="corrupt"):
        WorldSet.restore(bad)
    bad = copy.deepcopy(s)
    bad["constraints"][0]["depends_on"] = ["future"]
    bad["digest"] = digest({k: v for k, v in bad.items() if k != "digest"})
    with pytest.raises(ValueError, match="dependencies"):
        WorldSet.restore(bad)
    assert WorldSet.restore(s).fingerprint == w.fingerprint


def test_budget_failed_work_is_charged_and_restore_never_refills():
    b = WorkBudget(3)
    b.charge("attempt", 2)
    with pytest.raises(BudgetStop):
        b.charge("attempt", 2)
    r = WorkBudget.restore(b.snapshot())
    assert r.spent == 2 and r.remaining == 1 and r.counts == {"attempt": 2}
    with pytest.raises(BudgetStop):
        r.charge("other", 2)
    s = b.snapshot()
    s["spent"] = 0
    with pytest.raises(ValueError):
        WorkBudget.restore(s)


def test_debit_sink_runs_before_paid_action_and_failure_stops_it():
    records = []
    b = WorkBudget(2, on_charge=lambda s: records.append(s))
    b.charge("work")
    assert records[0]["spent"] == 1

    def fail(_):
        raise RuntimeError("write failed")

    b.on_charge = fail
    with pytest.raises(RuntimeError):
        b.charge("work")
    assert b.remaining == 0


def test_selects_sufficient_cheap_probe_and_ignores_irrelevant_field():
    w = WorldSet([{"x": x, "noise": n} for x in (0, 1) for n in (0, 1)])
    probes = [Probe("noise", lambda s: s["noise"], 1), Probe("x", lambda s: s["x"], 8)]
    b = budget()
    choice = select_probe(w, lambda s: s["x"], probes, b)
    assert choice.name == "x" and choice.worst_case_cost == 8
    assert "observation" not in b.counts  # planning is not an external observation


def test_invariant_answer_does_not_purchase_any_probe():
    w = WorldSet([{"x": 0}, {"x": 1}])
    touched = []
    c = select_probe(w, lambda _: "same", [Probe("x", lambda s: touched.append(s) or s["x"])], budget())
    assert c.status == "resolved" and c.name is None and not touched


def test_lookahead_handles_xor_synergy_without_fake_one_step_gain():
    w = WorldSet([{"a": a, "b": b} for a in (0, 1) for b in (0, 1)])
    probes = [Probe("a", lambda s: s["a"], 2), Probe("b", lambda s: s["b"], 3)]
    choice = select_probe(w, lambda s: s["a"] ^ s["b"], probes, budget())
    assert choice.name == "a" and choice.worst_case_cost == 5
    observed = w.observe("a", 1, "external-a", budget())
    assert select_probe(observed, lambda s: s["a"] ^ s["b"], probes, budget()).name == "b"


def test_probe_planning_budget_is_real_and_no_free_probe_is_executed():
    w = WorldSet([{"x": 0}, {"x": 1}])
    b = WorkBudget(1)
    c = select_probe(w, lambda s: s["x"], [Probe("x", lambda s: s["x"])], b)
    assert c.status == "budget" and c.name is None and b.spent == 1


def test_fallback_does_not_buy_independent_noise_without_answer_information():
    w = WorldSet([{"x": x, "noise": n} for x in (0, 1) for n in (0, 1)])
    b = budget()
    choice = select_probe(w, lambda s: s["x"], [Probe("noise", lambda s: s["noise"], 1)], b)
    assert choice.name is None and choice.status == "unavailable"


def test_fallback_checks_budget_after_all_planning_not_before_other_probes():
    w = WorldSet([{"a": a, "b": b} for a in (0, 1) for b in (0, 1)])
    b = WorkBudget(39)
    choice = select_probe(
        w, lambda s: s["a"] ^ s["b"], [Probe("a", lambda s: s["a"], 10), Probe("b", lambda s: s["b"], 20)], b, horizon=1
    )
    assert choice.name is None and choice.status == "budget"


@pytest.mark.parametrize("bad", [-1, True, 0, 1.5])
def test_bad_probe_costs(bad):
    with pytest.raises(ValueError):
        Probe("x", lambda s: s["x"], bad)
