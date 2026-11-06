import pytest
import spam_budget_trial as trial


def prepared():
    def row(i):
        return dict(id=i, inputs={'f0': i % 8}, label=bool(i % 2))
    return dict(name='fixture', seed=3, split='dev', pool=[row(i) for i in range(256)],
                test=[row(i) for i in range(256, 768)], feature_names=['x'], source_sha='fixture',
                preprocess_costs={'preprocess_fields': 768}, group_ids={'pool': list(range(256)),
                'test': list(range(256, 768)), 'holdout': []}, thresholds={}, rawcounts={},
                quantized_test_rows_matching_pool=512)


class Model:
    def to_dict(self):
        return {'fixture': True}
    def predict(self, inputs, budget):
        budget.charge('fixture_predict')
        return bool(inputs['f0'] % 2)


@pytest.mark.parametrize('arm,n', [('RANDOM32',32),('RANDOM40',40),('RANDOM48',48),('MIXED32',32),('MIXED40',40),('FULL_LABEL',256)])
def test_paid_one_fit_seed_freeze(monkeypatch, arm, n):
    calls, events = [], []
    def fit(rows, budget, *, seed):
        calls.append((rows[:], seed))
        assert len([e for e in events if e[0] == 'label_receipt']) == n
        budget.charge('fixture_fit')
        return Model()
    monkeypatch.setattr(trial, 'fit_forest', fit)
    result = trial.real_trial('fixture', None, 3, 'dev', arm, 128, prepared=prepared(), emit=lambda k,v: events.append((k,v)))
    assert len(calls) == len(result['fits']) == 1
    assert calls[0][1] == 300000 + (0 if arm == 'FULL_LABEL' else 800)
    assert len(calls[0][0]) == n
    assert all(int(r['id']) < 256 for r in calls[0][0])
    assert len(result['predictions']) == 512
    assert events[-1][0] == 'frozen'
    if arm.startswith('RANDOM'):
        assert [r['id'] for r in result['receipts']] == list(map(str, range(n)))
        assert result['plan'] is None
    else:
        assert arm == 'FULL_LABEL' or len(result['plan']['names']) == 8


def test_budget_partial_fit_discarded_no_refund(monkeypatch):
    def fit(rows, budget, *, seed):
        budget.charge('expensive', 4000000)
    monkeypatch.setattr(trial, 'fit_forest', fit)
    result = trial.real_trial('fixture', None, 3, 'dev', 'RANDOM32', 128, prepared=prepared())
    assert not result['fits'] and result['frozen']['model'] is None
    assert result['frozen']['reason'] == 'budget'
    assert all(p['outcome'] == 'abstain' for p in result['predictions'])
    assert len(result['receipts']) == 32
    assert result['budget']['spent'] > 32 * 128


def test_plan_only_paid_pilot_and_bounded_unread(monkeypatch):
    import solvi.experimental.experience.stratified as core
    original = core.plan_stratified_observations
    seen = []
    def plan(examples, options, budget, **kwargs):
        assert [e.event_id for e in examples] == list(map(str, range(24)))
        assert all(e.split == 'proposal' and type(e.expected) is bool for e in examples)
        assert [o.name for o in options] == list(map(str, range(24, 88)))
        seen.append(True)
        return original(examples, options, budget, **kwargs)
    monkeypatch.setattr(core, 'plan_stratified_observations', plan)
    monkeypatch.setattr(trial, 'fit_forest', lambda *a, **k: Model())
    result = trial.real_trial('fixture', None, 3, 'dev', 'MIXED32', 128, prepared=prepared())
    assert seen == [True]
    assert all(int(name) < 256 for name in result['plan']['names'])


def test_worker_json_and_fsynced_journal(monkeypatch, tmp_path, capsys):
    import json
    import sys
    from pathlib import Path
    import spam_budget_worker as worker
    source = Path(worker.__file__).resolve().parents[1]
    job = dict(tier='budget_real', dataset='fixture', seed=3, split='dev', arm='RANDOM32', price=128)
    job_path = tmp_path / 'job.json'
    job_path.write_text(json.dumps(job))
    journal = tmp_path / 'journal.jsonl'
    syncs = []
    monkeypatch.setattr(worker.os, 'fsync', lambda fd: syncs.append(fd))
    def run(*a, emit, **k):
        emit('label_intent', {'request': 'r', 'id': '1'})
        emit('label_receipt', {'request': 'r', 'id': '1'})
        emit('frozen', {'model': None})
        return {'fixture': True}
    monkeypatch.setattr(trial, 'real_trial', run)
    monkeypatch.setattr(sys, 'argv', ['worker', '--source', str(source), '--job', str(job_path),
                                    '--data', str(tmp_path), '--roles', str(tmp_path/'roles.json'),
                                    '--journal', str(journal)])
    worker.main()
    result = json.loads(capsys.readouterr().out)
    assert result['job'] == job and result['payload'] == {'fixture': True}
    assert result['origin'].endswith('/stratified.py') and len(result['actor_sha256']) == 64
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    assert len(syncs) == len(events) == 3
    assert events[0]['payload'] == events[1]['payload']
