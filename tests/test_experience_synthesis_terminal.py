"""A-only terminal-size early rejection; runtime semantics stay eager and strict."""
import importlib.util
import os
from pathlib import Path
import sys

import pytest

from solvi.experimental.experience import programs
from solvi.experimental.experience._common import WorkBudget, BudgetStop
from solvi.experimental.experience.programs import Grammar, Program, synthesize
from test_experience_synthesis_v2 import full_ast_reference


def normalized(result):
    return result.reason, result.tried, None if result.candidate is None else result.candidate.to_dict()


def load_source(variable, suffix):
    configured = os.environ.get(variable)
    if not configured:
        pytest.skip('set ' + variable + ' for historical source integration')
    path = Path(configured) / 'src/solvi/experimental/experience/programs.py'
    name = 'solvi.experimental.experience._terminal_comparison_' + suffix
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_max_nodes_one_typed_mismatch_and_full_success_rows():
    grammar = Grammar(('x',), max_nodes=1)
    rows = [({'x': 0}, True), ({'x': 1}, True)]
    budget = WorkBudget(10000)
    result = synthesize(rows, grammar, budget)
    assert result.candidate.ast == ('const', True)
    assert result.tried == 5
    assert budget.counts['program_node'] == 6
    assert normalized(result) == normalized(full_ast_reference(rows, grammar, WorkBudget(10000)))


@pytest.mark.parametrize('operator,rows', [
    ('add', [({'x': x}, x + 1) for x in (0, 1, 2)]),
    ('sub', [({'x': x}, x - 1) for x in (0, 1, 2)]),
    ('mul', [({'x': x}, x * x) for x in (0, 1, 2)]),
    ('min', [({'x': x}, min(x, 1)) for x in (0, 1, 2)]),
    ('max', [({'x': x}, max(x, 1)) for x in (0, 1, 2)]),
    ('lt', [({'x': x}, x < 1) for x in (0, 1, 2)]),
    ('eq', [({'x': x}, x == 1) for x in (0, 1, 2)]),
    ('and', [({'x': x}, x and False) for x in (True, False)]),
    ('or', [({'x': x}, x or True) for x in (False, True)]),
    ('not', [({'x': x}, not x) for x in (False, True)]),
    ('if', [({'x': x}, 1 if x else 0) for x in (False, True)]),
])
@pytest.mark.parametrize('source', [None, 'SOLVI_EXPERIENCE_V2_BASELINE', 'SOLVI_EXPERIENCE_V2_CANDIDATE02'])
def test_all_scalar_operators_preserve_first_match(operator, rows, source):
    grammar = Grammar(('x',), operators=(operator,), max_nodes=4, max_candidates=1000)
    if source is None:
        reference = full_ast_reference(rows, grammar, WorkBudget(1000000))
    else:
        before = load_source(source, source.rsplit('_', 1)[-1].lower())
        reference = before.synthesize(rows, before.Grammar.from_dict(grammar.to_dict()), WorkBudget(1000000))
    assert normalized(synthesize(rows, grammar, WorkBudget(1000000))) == normalized(reference)


def test_runtime_eager_if_and_intermediate_overflow_remain_strict():
    grammar = Grammar(('x',), max_abs=4)
    ast = ('if', ('const', True), ('const', 0), ('add', ('input', 'x'), ('input', 'x')))
    with pytest.raises(ValueError):
        Program(ast).evaluate({'x': 3}, WorkBudget(100), grammar)
    with pytest.raises(ValueError):
        Program(('add', ('const', True), ('const', 1))).evaluate({}, WorkBudget(100), grammar)


@pytest.mark.parametrize('error_type', [BudgetStop, ValueError])
@pytest.mark.parametrize('category', ['search_generate', 'program_node', 'search_compare', 'search_cache_lookup'])
def test_paid_sink_failures_are_propagated(error_type, category):
    def sink(snapshot):
        if snapshot['counts'].get(category, 0):
            raise error_type('persisted debit failure')
    budget = WorkBudget(100000, on_charge=sink)
    with pytest.raises(error_type, match='persisted debit failure'):
        synthesize([({'x': 0}, 1), ({'x': 2}, 3)],
                   Grammar(('x',), operators=('add',), max_nodes=3), budget)
    assert budget.spent > 0


def test_true_insufficient_budget_is_explicit():
    result = synthesize([({'x': 0}, 1), ({'x': 2}, 3)], Grammar(('x',), max_nodes=3), WorkBudget(1))
    assert result.reason == 'budget_exhausted'


def test_terminal_rejection_skips_later_overflow_without_changing_runtime(monkeypatch):
    before = load_source('SOLVI_EXPERIENCE_V2_CANDIDATE02', 'candidate02')
    rows = [({'x': 0}, 1), ({'x': 2}, 3), ({'x': 3}, 4)]
    grammar = Grammar(('x',), operators=('add',), max_nodes=3, max_abs=4)
    calls = {'before': [], 'after': []}
    for label, module in [('before', before), ('after', programs)]:
        original = module._primitive
        def count(op, args, maximum, label=label, original=original):
            calls[label].append((op, tuple(args)))
            return original(op, args, maximum)
        monkeypatch.setattr(module, '_primitive', count)
    prior = before.synthesize(rows, before.Grammar.from_dict(grammar.to_dict()), WorkBudget(1000000))
    after = synthesize(rows, grammar, WorkBudget(1000000))
    assert normalized(prior) == normalized(after)
    assert ('add', (3, 3)) in calls['before']
    assert ('add', (3, 3)) not in calls['after']
    assert len(calls['after']) < len(calls['before'])
    print('terminal primitive diagnostic:', len(calls['before']), '->', len(calls['after']))


def test_smaller_candidates_keep_complete_vectors_and_exact_costs():
    before = load_source('SOLVI_EXPERIENCE_V2_CANDIDATE02', 'candidate02')
    rows = [({'x': 0}, 1), ({'x': 2}, 3), ({'x': 3}, 4)]
    grammar = Grammar(('x',), operators=('add',), max_nodes=5, max_abs=4)
    prior_budget, actual_budget = WorkBudget(1000000), WorkBudget(1000000)
    reference = before.synthesize(rows, before.Grammar.from_dict(grammar.to_dict()), prior_budget)
    actual = synthesize(rows, grammar, actual_budget)
    assert normalized(reference) == normalized(actual)
    assert prior_budget.counts == actual_budget.counts


@pytest.mark.parametrize('source', ['SOLVI_EXPERIENCE_V2_BASELINE', 'SOLVI_EXPERIENCE_V2_CANDIDATE02'])
@pytest.mark.parametrize('family', ['inventory', 'eligibility', 'schedule'])
def test_actual_historical_sources_same_seven_node_family_first_match(source, family):
    prior = load_source(source, source.rsplit('_', 1)[-1].lower())
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from experience_v2 import FAMILIES, roles, sql_value
    names, _, operators, _ = FAMILIES[family]
    rows = [(dict(zip(names, row)), sql_value(family, row)) for row in roles(0, family)['proposal']]
    grammar = Grammar(names, operators=operators, max_nodes=7, max_depth=4, max_candidates=10000, max_abs=32)
    actual = synthesize(rows, grammar, WorkBudget(10000000))
    reference = prior.synthesize(rows, prior.Grammar.from_dict(grammar.to_dict()), WorkBudget(10000000))
    assert actual.reason != 'budget_exhausted' and reference.reason != 'budget_exhausted'
    assert normalized(actual) == normalized(reference)
