"""Only preregistered training configurations, paired seeds and independent base agents."""
import json
import random

import pytest

from solvi.solutions.wrap import select_components


class Base:
    def __init__(self, seed):
        self.rng = random.Random(seed)
        self.calls = 0
    def decide(self, state):
        self.calls += 1
        return {'kind': 'base', 'noise': self.rng.random()}


def factory(seed):
    return Base(seed)


def good(data):
    return {'action': {'kind': 'good', 'noise': data['default']['noise']},
            'default_loss': 2, 'loss': 1, 'evidence': ['training:good']}


def harmful(data):
    return {'action': {'kind': 'harmful', 'noise': data['default']['noise']},
            'default_loss': 2, 'loss': 1, 'evidence': ['training:harmful']}


POINTS = {'decide': {'good': good, 'harmful': harmful}}


def evaluate(agent, seed):
    action = agent.decide({'seed': seed})
    return action['noise'] + {'base': 0, 'good': 1, 'harmful': -1}[action['kind']]


def test_default_single_components_paired_and_baseline_rng_unchanged(tmp_path):
    result = select_components(factory, evaluate, decision_points=POINTS,
                               seeds=[7, 29, 101], directory=tmp_path/'training')
    assert len(result['cells']) == 9 and result['best_registered_config_training'] == ['good']
    assert {tuple(row['configuration']) for row in result['cells']} == {(), ('good',), ('harmful',)}
    baseline = result['cells'][:3]
    assert [row['utility'] for row in baseline] == [evaluate(factory(seed), seed) for seed in [7, 29, 101]]
    assert all(not __import__('pathlib').Path(row['override_journal']).exists() for row in baseline)
    assert [c['mean_paired_delta'] for c in result['comparisons']] == pytest.approx([1, -1])
    assert len({row['override_journal'] for row in result['cells']}) == 9
    assert json.loads((tmp_path/'training/result.json').read_text())['training_only'] is True


def test_only_declared_combination_is_evaluated_and_budget_is_counted(tmp_path):
    calls = []
    def evaluator(agent, seed):
        calls.append(seed)
        return evaluate(agent, seed)
    result = select_components(factory, evaluator, decision_points=POINTS, seeds=[1, 2],
        configurations=[('good', 'harmful')], max_evaluations=3,
        costs={(): 1, ('good', 'harmful'): 2}, directory=tmp_path/'budget')
    assert calls == [1, 2, 1] and len(result['cells']) == 3
    assert {tuple(r['configuration']) for r in result['cells']} == {(), ('good', 'harmful')}
    assert result['budget_exhausted'] and result['declared_total_cost'] == 4
    assert result['best_registered_config_training'] == []  # incomplete pair set cannot win


def test_exceptions_and_nan_are_saved_and_later_cells_continue(tmp_path):
    calls = []
    def evaluator(agent, seed):
        calls.append(seed)
        action = agent.decide({})
        if action['kind'] == 'harmful':
            if seed == 1:
                raise RuntimeError('training failure')
            return float('nan')
        return evaluate(agent, seed)
    result = select_components(factory, evaluator, decision_points=POINTS, seeds=[1, 2], directory=tmp_path/'errors')
    assert len(calls) == len(result['cells']) == 6
    assert [r['status'] for r in result['cells']][-2:] == ['error', 'error']
    lines = [json.loads(line) for line in (tmp_path/'errors/cells.jsonl').read_text().splitlines()]
    assert len(lines) == 6 and 'training failure' in lines[-2]['error'] and 'finite' in lines[-1]['error']
    assert result['best_registered_config_training'] == ['good']


def test_reused_agent_is_rejected_before_another_evaluation(tmp_path):
    shared = Base(1)
    calls = []
    def reuse(seed):
        return shared
    def evaluator(agent, seed):
        calls.append(seed)
        return evaluate(agent, seed)
    result = select_components(reuse, evaluator, decision_points=POINTS, seeds=[1, 2], directory=tmp_path/'reuse')
    assert calls == [1] and result['best_registered_config_training'] is None
    assert all('reused' in row['error'] for row in result['cells'][1:])


def test_invalid_configuration_never_calls_factory_or_evaluator(tmp_path):
    def never(*args):
        pytest.fail('not a declared configuration')
    with pytest.raises(ValueError, match='registered'):
        select_components(never, never, decision_points=POINTS, seeds=[1],
                          configurations=[('unknown',)], directory=tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()


def test_configuration_and_seed_snapshots_cannot_change_during_factory(tmp_path):
    points = {'decide': {'good': good}}
    seeds = [1, 2]
    def creating(seed):
        seeds[:] = [99]
        points['decide']['good'] = harmful
        return Base(seed)
    result = select_components(creating, evaluate, decision_points=points, seeds=seeds, directory=tmp_path/'frozen')
    assert [r['seed'] for r in result['cells']] == [1, 2, 1, 2]
    assert result['best_registered_config_training'] == ['good']


def test_finite_utilities_with_overflowing_delta_are_recorded_not_selected(tmp_path):
    def extreme(agent, seed):
        return -1e308 if agent.decide({})['kind'] == 'base' else 1e308
    result = select_components(factory, extreme, decision_points={'decide': {'good': good}},
        seeds=[1, 2], costs={(): 1e308, ('good',): 1e308}, directory=tmp_path/'overflow')
    assert all(row['status'] == 'ok' for row in result['cells'])
    assert result['best_registered_config_training'] == []
    assert len(result['comparisons'][0]['invalid_deltas']) == 2
    assert result['declared_total_cost'] is None and result['declared_total_cost_overflow']
    assert json.loads((tmp_path/'overflow/result.json').read_text())['training_only']
