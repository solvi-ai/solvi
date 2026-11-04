"""S122 finite external calls: bypass, strict evidence/loss and durable-before-return."""
import random

import pytest

import solvi
from solvi.core.store import JSONLStorage, SQLiteStorage


class Agent:
    def __init__(self):
        self.rng = random.Random(17)
        self.calls = 0
        self.last = None
    def decide(self, state):
        self.calls += 1
        self.last = {'value': self.rng.random(), 'input': state}
        return self.last
    def run(self, state):
        return self.decide(state)


def improve(analysis):
    return {'action': {'value': 'override'}, 'default_loss': 2, 'loss': 1, 'evidence': ['checked:1']}


def test_disabled_is_bit_identical_state_rng_and_never_opens_journal(tmp_path):
    reference, agent = Agent(), Agent()
    callbacks = []
    def never(data):
        callbacks.append(data)
        raise AssertionError('disabled proposer called')
    path = tmp_path / 'must-not-exist.jsonl'
    wrapped = solvi.wrap(agent, decision_points={'decide': never}, default=agent.decide,
                         enabled=False, storage=path)
    for i in range(12):
        assert wrapped.decide({'i': i}) == reference.decide({'i': i})
        assert agent.rng.getstate() == reference.rng.getstate()
    assert agent.calls == reference.calls == 12 and not callbacks and not path.exists()


@pytest.mark.parametrize('proposal', [None, {}, {'action': 'x', 'loss': 1, 'default_loss': 2},
    {'action': 'x', 'loss': 2, 'default_loss': 2, 'evidence': ['id']},
    {'action': 'x', 'loss': -1, 'default_loss': 2, 'evidence': ['id']},
    {'action': 'x', 'loss': float('nan'), 'default_loss': 2, 'evidence': ['id']},
    {'action': 'x', 'loss': 0, 'default_loss': float('inf'), 'evidence': ['id']},
    {'action': 'x', 'loss': False, 'default_loss': 2, 'evidence': ['id']},
    {'action': 'x', 'loss': 0, 'default_loss': 2, 'evidence': [' ']},
    {'action': object(), 'loss': 0, 'default_loss': 2, 'evidence': ['id']}])
def test_invalid_or_nonimproving_evidence_preserves_original_identity(tmp_path, proposal):
    agent = Agent()
    wrapped = solvi.wrap(agent, decision_points={'decide': lambda data: proposal}, storage=tmp_path/'none.jsonl')
    result = wrapped.decide({'x': 1})
    assert result is agent.last and agent.calls == 1 and not (tmp_path/'none.jsonl').exists()


@pytest.mark.parametrize('backend', ['jsonl', 'sqlite'])
def test_override_is_durable_before_return_and_survives_restart(tmp_path, backend):
    path = tmp_path / ('overrides.jsonl' if backend == 'jsonl' else 'overrides.db')
    agent = Agent()
    wrapped = solvi.wrap(agent, decision_points={'decide': {'doubt': improve}}, storage=path)
    assert wrapped.decide({'x': 1}) == {'value': 'override'} and agent.calls == 1
    reopened = JSONLStorage(path) if backend == 'jsonl' else SQLiteStorage(path)
    records = [row for _, row in reopened._raw()]
    assert len(records) == 1 and reopened.verify()['ok']
    row = records[0]
    assert row['kind'] == 'external_override' and row['component'] == 'doubt'
    assert row['proposal']['evidence'] == ['checked:1'] and row['proposer']['code']
    assert row['analysis']['default'] == agent.last


def test_proposer_cannot_mutate_inputs_default_or_other_component_analysis(tmp_path):
    agent = Agent()
    state = {'nested': [1]}
    def bad(data):
        data['args'][0]['nested'].append(99)
        data['default']['input']['nested'].append(88)
        return None
    def good(data):
        assert data['args'][0] == {'nested': [1]}
        assert data['default']['input'] == {'nested': [1]}
        return improve(data)
    wrapped = solvi.wrap(agent, decision_points={'decide': {'bad': bad, 'good': good}}, storage=tmp_path/'x.jsonl')
    assert wrapped.decide(state) == {'value': 'override'}
    assert state == {'nested': [1]} and agent.last['input'] == state and agent.calls == 1


def test_non_json_input_skips_proposers_and_returns_original_object():
    agent = Agent()
    wrapped = solvi.wrap(agent, decision_points={'decide': lambda data: pytest.fail('not JSON')})
    assert wrapped.decide(object()) is agent.last and agent.calls == 1


def test_components_empty_is_exact_bypass_and_internal_calls_are_not_patched(tmp_path):
    agent = Agent()
    wrapped = solvi.wrap(agent, decision_points={'decide': {'doubt': improve}}, components=[], storage=tmp_path/'x.jsonl')
    assert wrapped.decide({'x': 1}) is agent.last
    wrapped = solvi.wrap(agent, decision_points={'decide': improve}, storage=tmp_path/'x.jsonl')
    assert wrapped.run({'x': 2}) is agent.last and agent.calls == 2
    assert not (tmp_path/'x.jsonl').exists()


def test_outer_batch_and_failed_storage_never_return_an_unjournaled_override(tmp_path):
    agent = Agent()
    storage = SQLiteStorage(tmp_path/'x.db')
    wrapped = solvi.wrap(agent, decision_points={'decide': improve}, storage=storage)
    with storage.batch(), pytest.raises(ValueError, match='active storage batch'):
        wrapped.decide({'x': 1})
    assert agent.calls == 1 and list(storage._raw()) == []
    class Broken(JSONLStorage):
        def _append(self, body):
            raise OSError('journal unavailable')
    broken = Broken(tmp_path/'broken.jsonl', fsync=True)
    wrapped = solvi.wrap(agent, decision_points={'decide': improve}, storage=broken)
    with pytest.raises(OSError, match='journal unavailable'):
        wrapped.decide({'x': 2})
    assert agent.calls == 2


def test_identical_action_is_not_replaced_despite_inconsistent_lower_loss_claim(tmp_path):
    agent = Agent()
    def same(data):
        return {'action': data['default'], 'default_loss': 2, 'loss': 0, 'evidence': ['id']}
    wrapped = solvi.wrap(agent, decision_points={'decide': same}, storage=tmp_path/'none.jsonl')
    assert wrapped.decide({'x': 1}) is agent.last and agent.calls == 1
    assert not (tmp_path/'none.jsonl').exists()


def test_component_selection_never_calls_unselected_proposer(tmp_path):
    calls = []
    def first(data):
        pytest.fail('unselected proposer called')
    def second(data):
        calls.append('second')
        return improve(data)
    agent = Agent()
    wrapped = solvi.wrap(agent, decision_points={'decide': {'first': first, 'second': second}},
                         components=['second'], storage=tmp_path/'selected.jsonl')
    assert wrapped.decide({'x': 1}) == {'value': 'override'}
    assert calls == ['second'] and agent.calls == 1


def test_returned_override_cannot_mutate_the_journal_record(tmp_path):
    storage = JSONLStorage(tmp_path/'identity.jsonl', fsync=True)
    wrapped = solvi.wrap(Agent(), decision_points={'decide': improve}, storage=storage)
    result = wrapped.decide({'x': 1})
    result['value'] = 'changed later'
    row = next(storage._raw())[1]
    assert row['proposal']['action'] == {'value': 'override'}
    assert storage.verify()['ok']
