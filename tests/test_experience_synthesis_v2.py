"""Development checks against the pre-vector, full-AST search algorithm."""

from itertools import chain, product

import pytest

from solvi.experimental.experience import programs
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.programs import Grammar, Program, SearchResult, synthesize


def full_ast_reference(rows, grammar, budget):
    """Independent enumeration: run the public interpreter on every row."""
    pools, seen, tried = {}, set(), 0

    def consider(ast):
        nonlocal tried
        if tried >= grammar.max_candidates:
            return SearchResult(None, tried, "candidate_limit")
        budget.charge("search_generate")
        tried += 1
        p = Program(ast)
        if p.depth > grammar.max_depth or p.nodes > grammar.max_nodes:
            return
        try:
            outputs = tuple(p.evaluate(x, budget, grammar) for x, _ in rows)
        except ValueError:
            return
        if all(type(v) is type(y) and v == y for v, (_, y) in zip(outputs, rows)):
            return SearchResult(p, tried, "found")
        signature = tuple((type(v).__name__, v) for v in outputs)
        if signature not in seen:
            seen.add(signature)
            pools.setdefault(p.nodes, []).append((p, type(outputs[0])))

    for ast in chain((("input", n) for n in grammar.inputs),
                     (("const", c) for c in grammar.constants),
                     (("const", False), ("const", True))):
        result = consider(ast)
        if result:
            return result
    for size in range(2, grammar.max_nodes + 1):
        for op in grammar.operators:
            for sizes in programs._partitions(size - 1, programs._ARITY[op]):
                for entries in product(*(pools.get(n, ()) for n in sizes)):
                    # Invalid types count as candidates, just as in the old
                    # early filter; full evaluation independently rejects them.
                    result = consider((op, *(entry[0].ast for entry in entries)))
                    if result:
                        return result
    return SearchResult(None, tried, "exhausted")


@pytest.mark.parametrize("rows,kwargs", [
    ([({"x": x}, 2 * x + 1) for x in (-2, 0, 3)],
     {"operators": ("add", "mul"), "max_nodes": 7}),
    ([({"x": x}, x < 0) for x in (-2, 0, 3)],
     {"operators": ("lt", "eq", "not"), "max_nodes": 4}),
    ([({"x": x}, not x) for x in (False, True)],
     {"operators": ("not", "and", "or", "if"), "max_nodes": 5}),
    ([({"x": x}, 0 if x < 0 else x) for x in (-2, 0, 3)],
     {"operators": ("lt", "if", "add"), "max_nodes": 6}),
    ([({"x": x}, -x) for x in (-2, 2)],
     {"operators": ("add", "mul", "sub", "if"), "max_abs": 2, "max_nodes": 5}),
])
@pytest.mark.parametrize("cap", [0, 7, 4000])
def test_same_first_match_and_candidate_accounting(rows, kwargs, cap):
    grammar = Grammar(("x",), max_candidates=cap, **kwargs)
    actual = synthesize(rows, grammar, WorkBudget(10**7))
    reference = full_ast_reference(rows, grammar, WorkBudget(10**7))
    assert actual == reference


def test_search_never_calls_full_interpreter(monkeypatch):
    def forbidden(*args):
        raise AssertionError("search re-evaluated a subtree")
    monkeypatch.setattr(Program, "evaluate", forbidden)
    budget = WorkBudget(100000)
    result = synthesize([({"x": -2}, -3), ({"x": 3}, 7)],
                        Grammar(("x",), operators=("add",), max_nodes=7), budget)
    assert result.reason == "found"
    assert budget.counts["search_cache_lookup"] > 0
    assert "program_input" not in budget.counts


@pytest.mark.parametrize("ast", [
    ("add", ("const", True), ("const", 1)),
    ("eq", ("const", True), ("const", 1)),
    ("if", ("const", False), ("add", ("input", "x"), ("const", 1)), ("const", 0)),
    ("sub", ("add", ("input", "x"), ("const", 1)), ("const", 1)),
])
def test_public_eager_strict_and_intermediate_overflow(ast):
    with pytest.raises(ValueError):
        Program(ast).evaluate({"x": 2}, WorkBudget(100), Grammar(("x",), max_abs=2))


def test_debit_happens_before_primitive_or_child_read(monkeypatch):
    original = programs._primitive
    events = []
    budget = WorkBudget(100000, on_charge=lambda s: events.append(dict(s["counts"])))

    def checked(op, args, maximum):
        assert events[-1].get("search_cache_lookup", 0) >= len(args)
        assert events[-1]["program_node"] > 0
        return original(op, args, maximum)

    monkeypatch.setattr(programs, "_primitive", checked)
    assert synthesize([({"x": -1}, -2), ({"x": 2}, 4)],
                      Grammar(("x",), operators=("add",), max_nodes=3), budget).reason == "found"


def test_failed_cache_debit_stops_before_primitive(monkeypatch):
    calls = []
    original = programs._primitive

    def primitive(*args):
        calls.append(args)
        return original(*args)

    def sink(snapshot):
        if snapshot["counts"].get("search_cache_lookup"):
            raise RuntimeError("durable debit sink failed")

    monkeypatch.setattr(programs, "_primitive", primitive)
    with pytest.raises(RuntimeError, match="durable debit"):
        synthesize([({"x": -1}, -2), ({"x": 2}, 4)],
                   Grammar(("x",), operators=("add",), max_nodes=3),
                   WorkBudget(100000, on_charge=sink))
    assert calls == []


def test_public_runtime_cost_unchanged():
    budget = WorkBudget(100)
    assert Program(("add", ("input", "x"), ("const", 1))).evaluate(
        {"x": 2}, budget, Grammar(("x",))) == 3
    assert budget.counts == {"program_node": 3, "program_input": 1}


@pytest.mark.parametrize("error_type", [ValueError, programs.BudgetStop])
@pytest.mark.parametrize("category", ["search_example", "program_node", "search_cache_lookup", "search_compare"])
def test_post_debit_sink_failure_propagates_unchanged(error_type, category):
    failure = error_type("one-shot persistence failure")
    events = []
    failed_snapshot = []

    def sink(snapshot):
        events.append(snapshot)
        if snapshot["counts"].get(category) and not failed_snapshot:
            failed_snapshot.append(snapshot)
            raise failure

    budget = WorkBudget(100000, on_charge=sink)
    with pytest.raises(error_type) as caught:
        synthesize([({"x": -1}, -2), ({"x": 2}, 4)],
                   Grammar(("x",), operators=("add",), max_nodes=3), budget)
    assert caught.value is failure
    assert events[-1] is failed_snapshot[0]
    assert budget.snapshot() == failed_snapshot[0]
    if category == "search_cache_lookup":
        assert budget.spent == 43


@pytest.mark.parametrize("limit", [0, 1, 42, 43, 48])
def test_normal_budget_refusal_remains_search_result(limit):
    budget = WorkBudget(limit)
    result = synthesize([({"x": -1}, -2), ({"x": 2}, 4)],
                        Grammar(("x",), operators=("add",), max_nodes=3), budget)
    assert result.reason == "budget_exhausted"
    assert result.candidate is None
    assert budget.spent <= limit
