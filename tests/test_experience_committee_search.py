"""Ordered prefix, shared debits and partial results in finite committee search."""

import hashlib
import importlib.util
import os
from pathlib import Path
import sys

import pytest

from solvi.experimental.experience import programs
from solvi.experimental.experience._common import BudgetStop, WorkBudget


@pytest.fixture(scope="module")
def baseline():
    # CI can run current single-search comparisons without a historical worktree.
    # With a frozen source supplied, verify its bytes and execute exactly those.
    source = os.environ.get("SOLVI_COMMITTEE_BASELINE")
    if source is None:
        return programs
    raw = Path(source).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == "02b5bdfc504078d9e2a4d75a0e571f7a427d33457326427d354ac5b3590dfc60"
    name = "solvi.experimental.experience._committee_baseline"
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, source, "exec"), module.__dict__)
    return module


def fixtures(case):
    if case == "duplicate":
        return programs.Grammar(("x", "y"), constants=(0, 0, 1), operators=("add", "sub"), max_nodes=3), [
            ({"x": 0, "y": 0}, 0)
        ]
    if case == "bool":
        return programs.Grammar(("b", "n"), operators=("eq", "not", "lt", "or"), max_nodes=4), [
            ({"b": True, "n": 1}, True),
            ({"b": False, "n": 0}, False),
        ]
    if case == "overflow":
        return programs.Grammar(("x",), operators=("mul", "add", "sub"), max_abs=3, max_nodes=5), [
            ({"x": 2}, 2),
            ({"x": 3}, 3),
        ]
    return programs.Grammar(("x",), constants=(), operators=("sub",), max_nodes=1), [({"x": 1}, 2)]


@pytest.mark.parametrize("case", ["duplicate", "bool", "overflow", "absent"])
@pytest.mark.parametrize("size", [1, 2, 3, 8])
def test_matches_successive_frozen_exclusions(baseline, case, size):
    grammar, rows = fixtures(case)
    old_grammar = baseline.Grammar.from_dict(grammar.to_dict())
    fingerprints = []
    for _ in range(size):
        old = baseline.synthesize(rows, old_grammar, WorkBudget(100000), exclude=fingerprints)
        if old.candidate is None:
            break
        fingerprints.append(old.candidate.fingerprint)
    budget = WorkBudget(100000)
    result = programs.synthesize_committee(rows, grammar, budget, size=size)
    assert [p.fingerprint for p in result.candidates] == fingerprints
    assert result.reason == old.reason and result.tried == old.tried
    first_cost = WorkBudget(100000)
    baseline.synthesize(rows, old_grammar, first_cost)
    assert result.continuation_spent == budget.spent - first_cost.spent


@pytest.mark.parametrize("limit", [0, 1, 5, 18, 100, 50000])
def test_single_search_preserves_all_frozen_debits(baseline, limit):
    grammar, rows = fixtures("bool")
    old_grammar = baseline.Grammar.from_dict(grammar.to_dict())
    for excluded in [[], [programs.Program(("input", "b")).fingerprint]]:
        left = WorkBudget(limit)
        right = WorkBudget(limit)
        old = baseline.synthesize(rows, old_grammar, left, exclude=excluded)
        new = programs.synthesize(rows, grammar, right, exclude=excluded)
        assert (old.reason, old.tried) == (new.reason, new.tried)
        assert (old.candidate and old.candidate.ast) == (new.candidate and new.candidate.ast)
        assert left.snapshot() == right.snapshot()


def test_one_pass_and_zero_window_prefix():
    class SinglePass:
        calls = 0

        def __iter__(self):
            self.calls += 1
            assert self.calls == 1
            return iter([({"x": 0}, 0)])

    grammar = programs.Grammar(("x",), constants=(0,), operators=("add",), max_nodes=3)
    first = WorkBudget(100)
    single = programs.synthesize_committee(SinglePass(), grammar, first, size=1)
    assert first.counts == {
        "search_example": 1,
        "search_input": 1,
        "search_generate": 1,
        "program_node": 1,
        "search_compare": 1,
    }
    budget = WorkBudget(100)
    stopped = programs.synthesize_committee(SinglePass(), grammar, budget, size=3, max_additional_work=0, reserve=1000)
    assert stopped.candidates == single.candidates and stopped.reason == "continuation_limit"
    assert stopped.continuation_spent == 0 and budget.snapshot() == first.snapshot()
    budget = WorkBudget(1000)
    three = programs.synthesize_committee(SinglePass(), grammar, budget, size=3)
    assert len(three.candidates) == 3 and budget.counts["search_example"] == 1


@pytest.mark.parametrize("stop", ["work", "reserve", "parent", "candidates"])
def test_second_match_survives_insufficient_third(stop):
    from dataclasses import replace

    grammar, rows = fixtures("duplicate")
    b1 = WorkBudget(1000)
    programs.synthesize_committee(rows, grammar, b1, size=1)
    b2 = WorkBudget(1000)
    two = programs.synthesize_committee(rows, grammar, b2, size=2)
    budget = WorkBudget(b2.spent if stop == "parent" else 1000)
    options = {}
    if stop == "work":
        options["max_additional_work"] = b2.spent - b1.spent
    if stop == "reserve":
        options["reserve"] = budget.limit - b2.spent
    if stop == "candidates":
        grammar = replace(grammar, max_candidates=two.tried)
    result = programs.synthesize_committee(rows, grammar, budget, size=3, **options)
    assert result.candidates == two.candidates
    assert (
        result.reason
        == {
            "work": "continuation_limit",
            "reserve": "continuation_limit",
            "parent": "budget_exhausted",
            "candidates": "candidate_limit",
        }[stop]
    )
    assert result.continuation_spent == budget.spent - b1.spent
    if stop != "candidates":
        assert budget.spent == b2.spent


@pytest.mark.parametrize(
    "options",
    [
        {"size": True},
        {"size": 0},
        {"size": 33},
        {"reserve": False},
        {"reserve": -1},
        {"max_additional_work": False},
        {"max_additional_work": -1},
    ],
)
def test_strict_caps_and_exclusion_bound(options):
    grammar, rows = fixtures("bool")
    with pytest.raises(ValueError):
        programs.synthesize_committee(rows, grammar, WorkBudget(1000), **options)
    with pytest.raises(ValueError):
        programs.synthesize(rows, grammar, WorkBudget(1000), exclude=["name"] * 33)


@pytest.mark.parametrize("error", [BudgetStop("store"), ValueError("store")])
def test_post_debit_sink_failure_escapes_with_original_identity(error):
    grammar, rows = fixtures("duplicate")

    def fail(snapshot):
        if snapshot["counts"].get("search_exclusion"):
            raise error

    budget = WorkBudget(1000, on_charge=fail)
    with pytest.raises(type(error)) as caught:
        programs.synthesize_committee(rows, grammar, budget, size=3, max_additional_work=100)
    assert caught.value is error and budget.counts["search_exclusion"] == 1


@pytest.mark.parametrize("limit", [0, 2, 6])
def test_no_program_before_first_match_keeps_full_budget_behavior(baseline, limit):
    grammar, rows = fixtures("bool")
    old_budget, budget = WorkBudget(limit), WorkBudget(limit)
    old = baseline.synthesize(rows, baseline.Grammar.from_dict(grammar.to_dict()), old_budget)
    new = programs.synthesize_committee(rows, grammar, budget, size=3, max_additional_work=0, reserve=10**6)
    assert not new.candidates and old.candidate is None
    assert (new.reason, new.tried) == (old.reason, old.tried)
    assert new.continuation_spent == 0 and budget.snapshot() == old_budget.snapshot()


def test_existing_parent_debits_do_not_consume_continuation_window():
    grammar, rows = fixtures("duplicate")
    parent = WorkBudget(1000)
    parent.charge("prior_work", 29)
    first = WorkBudget(1000)
    programs.synthesize(rows, grammar, first)
    result = programs.synthesize_committee(rows, grammar, parent, size=3, max_additional_work=2)
    assert result.reason == "continuation_limit"
    assert parent.spent == 29 + first.spent + result.continuation_spent
    assert 0 < result.continuation_spent <= 2 and parent.counts["prior_work"] == 29


def test_empty_rows_and_mixed_target_types():
    grammar, _ = fixtures("bool")
    empty = programs.synthesize_committee([], grammar, WorkBudget(100), reserve=100)
    assert empty == programs.CommitteeSearchResult((), 0, "no_examples", 0)
    rows = [({"b": True, "n": 1}, True), ({"b": False, "n": 0}, 0)]
    with pytest.raises(ValueError, match="inconsistent proposal types"):
        programs.synthesize_committee(rows, grammar, WorkBudget(100))


@pytest.mark.parametrize(("size", "cap", "expected_length"), [(1, None, 1), (3, 0, 1), (3, None, 3)])
def test_first_checkpoint_matches_frozen_first_cost_before_any_continuation(baseline, size, cap, expected_length):
    grammar, rows = fixtures("duplicate")
    old_budget = WorkBudget(1000)
    old_budget.charge("prior", 13)
    first = baseline.synthesize(rows, baseline.Grammar.from_dict(grammar.to_dict()), old_budget)
    snapshots = []
    budget = WorkBudget(1000)
    budget.charge("prior", 13)

    def checkpoint(program):
        snapshots.append((program.fingerprint, budget.snapshot()))

    result = programs.synthesize_committee(
        rows, grammar, budget, size=size, max_additional_work=cap, on_first_match=checkpoint
    )
    assert len(result.candidates) == expected_length
    assert snapshots == [(first.candidate.fingerprint, old_budget.snapshot())]
    assert result.continuation_spent == budget.spent - old_budget.spent


@pytest.mark.parametrize("error", [ValueError("journal failure"), BudgetStop("journal unavailable")])
def test_checkpoint_failure_is_not_a_search_result(error):
    grammar, rows = fixtures("duplicate")
    budget = WorkBudget(1000)

    def checkpoint(program):
        assert isinstance(program, programs.Program)
        raise error

    with pytest.raises(type(error)) as caught:
        programs.synthesize_committee(rows, grammar, budget, on_first_match=checkpoint)
    assert caught.value is error
    assert "search_exclusion" not in budget.counts


def test_no_checkpoint_without_match_and_reject_debit_in_checkpoint():
    grammar, rows = fixtures("duplicate")
    budget = WorkBudget(0)
    result = programs.synthesize_committee(rows, grammar, budget, on_first_match=lambda p: pytest.fail("no match"))
    assert result.candidates == () and result.reason == "budget_exhausted"
    with pytest.raises(ValueError, match="callable"):
        programs.synthesize_committee(rows, grammar, budget, on_first_match=42)
    assert budget.spent == 0
    budget = WorkBudget(1000)
    with pytest.raises(ValueError, match="spent the parent budget"):
        programs.synthesize_committee(rows, grammar, budget, on_first_match=lambda p: budget.charge("unexpected"))
    assert budget.counts["unexpected"] == 1 and "search_exclusion" not in budget.counts
