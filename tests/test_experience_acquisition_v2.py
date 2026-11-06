"""Tiny independent exhaustive trees and exploratory development fixtures."""

import itertools
import random
from math import inf

import pytest

from solvi.core.costs import BudgetStop
from solvi.experimental.experience._common import WorkBudget, canonical
from solvi.experimental.experience.acquisition import Probe, select_probe
from solvi.experimental.experience.worlds import WorldSet


def exhaustive(states, question, probes, depth):
    """Enumerate every root and every child, with no pruning or shared cache."""
    if len({canonical(question(s)) for s in states}) <= 1:
        return 0, None
    if depth == 0:
        return inf, None
    trees = []
    for position, probe in enumerate(probes):
        groups = {}
        for state in states:
            groups.setdefault(canonical(probe.read(state)), []).append(state)
        if len(groups) < 2:
            continue
        rest = probes[:position] + probes[position + 1 :]
        costs = [exhaustive(group, question, rest, depth - 1)[0] for group in groups.values()]
        trees.append((probe.cost + max(costs), probe.name))
    result = min(trees, default=(inf, None))
    return result if result[0] < inf else (inf, None)


@pytest.mark.parametrize("seed", [5001, 5002, 5003, 5004])
def test_matches_independent_exhaustive_trees(seed):
    rng = random.Random(seed)
    states = [{"id": i, "y": rng.randrange(2), **{f"p{j}": rng.randrange(3) for j in range(4)}} for i in range(7)]
    question = lambda s: s["y"]
    probes = [Probe(name, lambda s, j=j: s[f"p{j}"], rng.randrange(1, 8)) for j, name in enumerate(["z", "a", "m", "b"])]
    for depth in (1, 2, 3, 4):
        expected_cost, expected_name = exhaustive(states, question, probes, depth)
        for ordered in (probes, list(reversed(probes))):
            result = select_probe(WorldSet(states), question, ordered, WorkBudget(100_000), horizon=depth)
            if expected_cost < inf:
                assert (result.worst_case_cost, result.name) == (expected_cost, expected_name)
            else:
                assert result.worst_case_cost is None


@pytest.mark.parametrize("names", [("z", "a"), ("b", "x")])
def test_equal_cost_root_uses_name_after_rename_and_reorder(names):
    worlds = WorldSet([{"x": 0}, {"x": 1}])
    for ordered in (names, names[::-1]):
        result = select_probe(worlds, lambda s: s["x"], [Probe(n, lambda s: s["x"], 2) for n in ordered], WorkBudget(1000))
        assert (result.name, result.worst_case_cost) == (min(names), 2)


@pytest.mark.parametrize("horizon", [1, 2])
def test_arbitrary_precision_cost_with_unresolved_or_finite_children(horizon):
    price = 10**400
    worlds = WorldSet([{"a": a, "b": b} for a, b in itertools.product((0, 1), repeat=2)])
    probes = [Probe("b", lambda s: s["b"], price + 1), Probe("a", lambda s: s["a"], price)]
    result = select_probe(worlds, lambda s: s["a"] ^ s["b"], probes, WorkBudget(3 * price), horizon=horizon)
    assert (result.status, result.name) == ("selected", "a")
    assert result.worst_case_cost == (None if horizon == 1 else 2 * price + 1)


def test_zero_and_stop_before_callback_or_lookup_work():
    calls = []
    worlds = WorldSet([{"x": 0}, {"x": 1}])
    assert select_probe(worlds, lambda s: calls.append(s) or s["x"], [], WorkBudget(0)).status == "budget"
    assert calls == []

    def stop(snapshot):
        if "probe_partition_lookup" in snapshot["counts"]:
            raise BudgetStop("test interruption after debit, before lookup")

    budget = WorkBudget(1000, on_charge=stop)
    with pytest.raises(BudgetStop, match="test interruption"):
        select_probe(worlds, lambda s: s["x"], [Probe("x", lambda s: s["x"])], budget)
    assert budget.counts["probe_partition_lookup"] == 1
    assert "probe_partition" not in budget.counts


def test_exhausted_after_model_matrix_does_not_sort_before_branch_debit(monkeypatch):
    from solvi.experimental.experience import acquisition

    def unexpected_sort(*args, **kwargs):
        raise AssertionError("unpaid sorting")

    monkeypatch.setattr(acquisition, "sorted", unexpected_sort, raising=False)
    worlds = WorldSet([{"x": 0}, {"x": 1}])
    probes = [Probe(n, lambda s: s["x"]) for n in ("z", "a")]
    budget = WorkBudget(6)  # Two labels and the complete 2x2 model matrix.
    assert select_probe(worlds, lambda s: s["x"], probes, budget).status == "budget"
    assert budget.remaining == 0 and "probe_branch" not in budget.counts


def test_xor_fallback_relevance_and_final_affordability():
    worlds = WorldSet([{"a": a, "b": b, "n": n} for a, b, n in itertools.product((0, 1), repeat=3)])
    probes = [Probe("noise", lambda s: s["n"], 1), Probe("a", lambda s: s["a"], 10), Probe("b", lambda s: s["b"], 20)]
    question = lambda s: s["a"] ^ s["b"]
    budget = WorkBudget(100_000)
    choice = select_probe(worlds, question, probes, budget, horizon=1)
    assert choice.name == "a" and choice.worst_case_cost is None
    tight = WorkBudget(budget.spent + 9)
    assert select_probe(worlds, question, probes, tight, horizon=1).status == "budget"


@pytest.mark.parametrize("error", [ValueError, BudgetStop])
@pytest.mark.parametrize("kind", ["probe_decision", "probe_model", "probe_branch", "probe_candidate", "probe_partition_lookup", "probe_partition", "probe_relevance"])
def test_sink_failure_stops_immediately_after_debit(error, kind):
    events = []

    def sink(snapshot):
        events.append(snapshot)
        if kind in snapshot["counts"]:
            raise error("sink failed")

    budget = WorkBudget(10000, on_charge=sink)
    worlds = WorldSet([{"a": a, "b": b} for a, b in itertools.product((0, 1), repeat=2)])
    probes = [Probe(n, lambda s, n=n: s[n]) for n in ("a", "b")]
    with pytest.raises(error, match="sink failed"):
        select_probe(worlds, lambda s: s["a"] ^ s["b"], probes, budget, horizon=1)
    assert budget.spent == events[-1]["spent"]
    assert sum(kind in event["counts"] for event in events) == 1


def test_fallback_reuses_three_root_partitions_with_six_builds():
    events = []
    previous = {}

    def sink(snapshot):
        nonlocal previous
        events.append(next(k for k, value in snapshot["counts"].items() if value != previous.get(k, 0)))
        previous = snapshot["counts"]

    worlds = WorldSet([dict(zip("abc", bits)) for bits in itertools.product((0, 1), repeat=3)])
    probes = [Probe(n, lambda s, n=n: s[n]) for n in "abc"]
    budget = WorkBudget(10000, on_charge=sink)
    result = select_probe(worlds, lambda s: sum(s.values()) % 2, probes, budget, horizon=1)
    assert result.name == "a" and result.worst_case_cost is None
    lookups = [i for i, kind in enumerate(events) if kind == "probe_partition_lookup"]
    # Each miss starts with a charged partition scan; hits skip that scan.
    misses = sum(events[i + 1] == "probe_partition" for i in lookups)
    assert (len(lookups) - misses, misses) == (3, 6)
    assert budget.counts["probe_partition"] == 3 * 8 + 3 * 2


def test_exploratory_fixture_pruning_reduces_legacy_work():
    # Baseline work is counted by its original unpruned solver, including its
    # existing memo hits; callback precomputation was already present there.
    states = [dict(zip(("a", "b", "c", "d"), bits)) for bits in itertools.product((0, 1), repeat=4)]
    worlds = WorldSet(states)
    probes = [Probe(name, lambda s, name=name: s[name], 2) for name in ("a", "b", "c", "d")]
    question = lambda s: s["a"] ^ s["b"] ^ s["c"] ^ s["d"]
    legacy = len(states) * (1 + len(probes))
    memo = {}

    def old(indices, available, depth):
        nonlocal legacy
        legacy += 1
        key = indices, available, depth
        if key in memo:
            return memo[key]
        if len({question(states[i]) for i in indices}) == 1:
            return 0
        if not depth:
            return inf
        best = inf
        for p in available:
            groups = {}
            for i in indices:
                legacy += 1
                groups.setdefault(probes[p].read(states[i]), []).append(i)
            if len(groups) > 1:
                rest = tuple(j for j in available if j != p)
                best = min(best, probes[p].cost + max(old(tuple(g), rest, depth - 1) for _, g in sorted(groups.items())))
        memo[key] = best
        return best

    expected = old(tuple(range(len(states))), tuple(range(len(probes))), 4)
    budget = WorkBudget(100_000)
    result = select_probe(worlds, question, probes, budget, horizon=4)
    assert result.worst_case_cost == expected == 8
    assert budget.spent < legacy * 0.85
    assert budget.counts["probe_partition_lookup"] > 0
    assert budget.counts["probe_partition"] < 400
