"""Finite S350: sound grouping only under an explicit builtin-integer parameter contract."""
import math

import pytest

from solvi.testing import mutants


def evaluate(source, value):
    namespace = {}
    exec(compile(source, '<finite-test>', 'exec'), namespace)
    return namespace['rule'](value)


def test_integer_boundary_and_constant_alias_are_one_explicit_group():
    source = 'def rule(age):\n    return age >= 18\n'
    raw = mutants.generate(source)
    grouped = mutants.generate(source, integer_parameters={'rule': ['age']})
    pair = next(m for m in grouped if m.description == '>= → >')
    assert {'kind': 'constant', 'line': 2, 'description': '18 → 19'} in pair.equivalent_mutations
    assert len(raw) == len(grouped) + 1
    for m in raw:
        assert any(all(evaluate(m.source, age) == evaluate(g.source, age) for age in range(-10, 41)) for g in grouped)
    for m in grouped:
        m.status = 'survived'
    report = mutants.Report('cases', 'task', grouped, integer_parameters={'rule': ['age']}).to_dict()
    assert report['score_unit'] == 'declared_integer_equivalence_groups'
    assert report['raw_mutations_represented'] == len(raw)
    assert report['equivalence_groups'][0]['aliases']


def test_integer_branch_swap_and_comparison_flip_are_one_group():
    source = 'def rule(age):\n    return "a" if age > 17 else "b"\n'
    raw = mutants.generate(source)
    grouped = mutants.generate(source, integer_parameters={'rule': ['age']})
    assert any({m.kind, *[a['kind'] for a in m.equivalent_mutations]} >= {'branch', 'flip'} for m in grouped)
    for m in raw:
        assert any(all(evaluate(m.source, age) == evaluate(g.source, age) for age in range(-10, 41)) for g in grouped)


def test_default_keeps_float_boundary_and_nan_complement_distinct():
    source = 'def rule(age):\n    return "a" if age > 17 else "b"\n'
    raw = mutants.generate(source)
    flip = next(m for m in raw if m.kind == 'flip')
    branch = next(m for m in raw if m.kind == 'branch')
    assert evaluate(flip.source, math.nan) != evaluate(branch.source, math.nan)
    boundary = mutants.generate('def rule(age):\n    return age >= 18\n')
    a = next(m for m in boundary if m.description == '>= → >')
    b = next(m for m in boundary if m.description == '18 → 19')
    assert evaluate(a.source, 18.5) != evaluate(b.source, 18.5)


@pytest.mark.parametrize('source', ['def rule(age):\n age = 1\n return age >= 18\n',
    'def rule(age):\n def nested():\n  age = 3\n return age >= 18\n'])
def test_reassignment_even_in_nested_scope_is_conservatively_refused(source):
    with pytest.raises(ValueError, match='reassigned'):
        mutants.generate(source, integer_parameters={'rule': ['age']})


def test_opaque_helper_expression_is_not_grouped_or_executed():
    source = 'def rule(age):\n return opaque(age) >= 18\n'
    raw = mutants.generate(source)
    grouped = mutants.generate(source, integer_parameters={'rule': ['age']})
    assert [m.source for m in raw] == [m.source for m in grouped]
    assert all(not m.equivalent_mutations for m in grouped)


def test_run_reports_group_denominator_and_progress_without_extra_execution(tmp_path):
    from test_testing_mutants import STRONG, TASK, suite
    path = suite(tmp_path, STRONG)
    (path.parent / 'task.py').write_text(TASK.replace('return amount >= LIMIT', 'return amount >= 100'))
    progress = []
    report = mutants.run(path, kinds=('boundary', 'constant'), integer_parameters={'big': ['amount']},
                         progress=lambda i, n, m: progress.append((i, n)), timeout=3)
    data = report.to_dict()
    assert data['score_unit'] == 'declared_integer_equivalence_groups'
    assert data['raw_mutations_represented'] > len(report.mutants)
    assert len(progress) == len(report.mutants)
    assert report.total == report.killed + report.survived
    assert data['equivalence_groups']
