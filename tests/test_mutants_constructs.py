"""Finite S348 patterns and explicit callback registration; no source discovery execution."""
import ast

import pytest

from solvi.testing import mutants


def test_literal_patterns_are_mutated_without_touching_guard_or_body():
    source = '''def rule(value):
    match value:
        case 0: return "zero"
        case -5: return "negative"
        case "yes": return "text"
        case True: return "boolean"
        case _: return "other"
'''
    generated = mutants.generate(source, kinds=('constant', 'flip'))
    assert len(generated) == 6  # two nudges per number, one string and one singleton flip
    for mutant in generated:
        compile(mutant.source, '<pattern>', 'exec')
        original = ast.parse(source).body[0].body[0]
        changed = ast.parse(mutant.source).body[0].body[0]
        assert [ast.dump(c.body[0]) for c in original.cases] == [ast.dump(c.body[0]) for c in changed.cases]
        assert sum(ast.dump(a.pattern) != ast.dump(b.pattern) for a, b in zip(original.cases, changed.cases)) == 1
    zero = next(m for m in generated if m.description == '0 → 1')
    namespace = {}
    exec(zero.source, namespace)
    assert namespace['rule'](0) == 'other' and namespace['rule'](1) == 'zero'


def test_helper_and_call_form_checks_use_explicit_names_without_executing_registration():
    source = '''raise RuntimeError("must not execute during generation")
def register(fn):
    raise RuntimeError("must not discover helpers by execution")
@register
def known(amount): return amount > 0
def positive(amount): return amount >= 0
cat.check(hard=True)(positive)
'''
    assert mutants.generate(source, kinds=('drop',)) == []
    generated = mutants.generate(source, kinds=('drop',), check_functions=['known', 'positive'])
    assert len(generated) == 2
    assert {m.description for m in generated} == {'check known() always passes', 'check positive() always passes'}
    for mutant in generated:
        compile(mutant.source, '<manifest>', 'exec')
        assert 'must not execute during generation' in mutant.source


def test_manifest_is_qualified_validated_and_respects_function_selection():
    source = 'class Rules:\n def check(self, value): return value > 0\ndef other(value): return value < 1\n'
    assert len(mutants.generate(source, kinds=('drop',), check_functions=['Rules.check'])) == 1
    assert mutants.generate(source, kinds=('drop',), functions=['other'], check_functions=['Rules.check']) == []
    for declaration in ['other', [1], ['missing']]:
        with pytest.raises(ValueError):
            mutants.generate(source, check_functions=declaration)
    with pytest.raises(ValueError, match='ambiguous'):
        mutants.generate('def check(): pass\ndef check(): pass\n', check_functions=['check'])


def test_multistatement_raise_guard_and_assert_get_individual_drop_mutants():
    source = '''def rule(amount):
    if amount < 0:
        print("invalid")
        raise ValueError("amount")
    assert amount < 10
    return amount
'''
    generated = mutants.generate(source, kinds=('drop',))
    assert len(generated) == 2
    guard = next(m for m in generated if m.description.startswith('guard'))
    namespace = {}
    exec(guard.source, namespace)
    assert namespace['rule'](-1) == -1
    with pytest.raises(AssertionError):
        namespace['rule'](11)


def test_unselected_structural_and_none_patterns_are_honestly_unsupported():
    assert mutants.generate('def rule(x):\n match x:\n  case None: return 0\n  case [a, b]: return a\n', kinds=('constant',)) == []


def test_registered_helper_drop_runs_in_existing_worker_and_preserves_score_unit(tmp_path):
    import json
    source = '''from solvi import Catalog, Question, Answer
cat = Catalog()
def register(fn):
    return cat.check(hard=True, then={"action": "block"})(fn)
@register
def positive(amount): return bool(amount >= 0)
@cat.rule("action")
def action(amount): return "pay"
QUESTIONS = [Question("action", "Action?", Answer.choice(["pay", "block"]))]
'''
    task = tmp_path / 'task.py'
    task.write_text(source)
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps([{'name': 'negative', 'state': {'amount': -1}, 'expected': {'action': 'block'}},
                                 {'name': 'positive', 'state': {'amount': 1}, 'expected': {'action': 'pay'}}]))
    report = mutants.run(cases, kinds=('drop',), check_functions=['positive'])
    assert report.total == report.killed == 1 and report.score == 1
    assert report.mutants[0].killed_by == 'negative'
    assert report.to_dict()['check_functions'] == ['positive']
    assert report.to_dict()['score_unit'] == 'distinct_ASTs'
    assert task.read_text() == source


def test_invalid_manifest_is_rejected_before_running_the_baseline(tmp_path, monkeypatch):
    import json
    task = tmp_path / 'task.py'
    task.write_text('def known(x): return x\n')
    cases = tmp_path / 'cases.json'
    cases.write_text(json.dumps([]))
    def must_not_run(*args, **kwargs):
        raise AssertionError('baseline must not run for an invalid manifest')
    monkeypatch.setattr(mutants, '_bounded', must_not_run)
    with pytest.raises(ValueError, match='unknown check'):
        mutants.run(cases, check_functions=['missing'])
