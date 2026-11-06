"""Synthetic end-to-end records checked by the independent packed replay."""
from copy import deepcopy

import pytest

from active_real_data import SOURCES
import packed_spam_trial as actor
from packed_spam_score import ARMS, audit_packed


def prepared():
    def row(i):
        x = i % 8
        return dict(id=i, inputs={'f0': x, 'f1': (i // 8) % 8}, label=bool(x >= 4))
    return dict(name='spambase', seed=3, split='dev', fold='f0',
                source_sha=SOURCES['spambase'].sha256, feature_names=['x', 'z'],
                pool=[row(i) for i in range(256)], test=[row(i) for i in range(256, 768)],
                group_ids=dict(pool=[f'p{i}' for i in range(256)],
                               test=[f't{i}' for i in range(512)],
                               holdout=[f'h{i}' for i in range(512)]),
                reserved_ids=[f'h{i}' for i in range(1536)], thresholds={'f0': [], 'f1': []},
                rawcounts={}, quantized_test_rows_matching_pool=512,
                preprocess_costs=dict(preprocess_sort=5120, preprocess_fields=1536,
                                      preprocess_compare=4608))


@pytest.fixture(scope='module')
def records():
    data = prepared()
    return data, {arm: actor.real_trial('spambase', None, 3, 'dev', arm, 128, prepared=data)
                  for arm in ARMS}


@pytest.mark.parametrize('arm', ARMS)
def test_all_arms_independent_roundtrip(records, arm):
    data, rows = records
    result = audit_packed(rows[arm], data)
    assert result['assigned'] == 512
    assert result['labels'] == (256 if arm == 'FULL_LABEL' else 48 if arm == 'RANDOM48' else 40)
    assert result['fits'] == 1
    assert result['positives'] + result['negatives'] == 512
    assert sum(result[k] for k in ('correct', 'wrong', 'abstain', 'budget')) == 512


def test_median_legacy_exact_policy_model_predictions(records):
    _, rows = records
    median, legacy = (rows[a]['record'] for a in ('MEDIAN40', 'LEGACY40'))
    assert [r['id'] for r in median['receipts']] == [r['id'] for r in legacy['receipts']]
    assert median['fits'][0]['model'] == legacy['fits'][0]['model']
    assert median['predictions'] == legacy['predictions']


def set_path(obj, path, value):
    for key in path[:-1]:
        obj = obj[key]
    obj[path[-1]] = value


@pytest.mark.parametrize('path,value', [
    (('record', 'controller', 'source'), 'foreign'),
    (('record', 'controller', 'features', 0), 'f99'),
    (('record', 'controller', 'thresholds', 0), 7),
    (('record', 'controller', 'steps', 0, 'observed'), False),
    (('record', 'controller', 'steps', 0, 'after_choose', 'spent'), 0),
    (('record', 'receipts', 0, 'expected'), True),
    (('record', 'receipts', 0, 'source'), 'foreign'),
    (('record', 'fits', 0, 'model', 'trees', 0, 'n'), 999),
    (('record', 'fits', 0, 'seed'), 42),
    (('record', 'fits', 0, 'model', 'seed'), 42),
    (('fold',), 'f1'),
    (('record', 'data_sha'), 'foreign'),
    (('record', 'partition_digest'), 'foreign'),
    (('record', 'trace', 0, 1), True),
])
def test_tampered_transcripts_rejected(records, path, value):
    data, rows = records
    bad = deepcopy(rows['ADAPT40'])
    # Invert booleans rather than accidentally assigning the original bit.
    if type(value) is bool and path[-1] in ('observed', 'expected'):
        obj = bad
        for key in path:
            obj = obj[key]
        value = not obj
    set_path(bad, path, value)
    with pytest.raises(ValueError):
        audit_packed(bad, data)


def test_assignment_denominator_is_fixed(records):
    data, rows = records
    bad = deepcopy(rows['RANDOM40'])
    bad['record']['predictions'].pop()
    with pytest.raises(ValueError):
        audit_packed(bad, data)
    short = deepcopy(data)
    short['test'].pop()
    with pytest.raises(ValueError):
        audit_packed(rows['RANDOM40'], short)


def test_native_borrows_host_inputs_without_portable(monkeypatch):
    data = prepared()
    before = deepcopy(data)
    original = actor.fit_forest
    seen = []
    def fit(rows, budget, **kwargs):
        for row in rows:
            source = data['pool'][int(row['id'])]['inputs']
            assert row['inputs'] is source
        seen.append(True)
        return original(rows, budget, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError('native feature serialization called portable')
    monkeypatch.setattr(actor, 'portable', forbidden)
    monkeypatch.setattr(actor, 'fit_forest', fit)
    result = actor.real_trial('spambase', None, 3, 'dev', 'ADAPT40', 128, prepared=data)
    assert seen == [True]
    assert data == before
    audit_packed(result, data)


def test_replay_does_not_call_actor_or_planner(records, monkeypatch):
    import solvi.experimental.experience.packed as core
    import solvi.experimental.experience.stratified as legacy
    import active_real_models as models
    def forbidden(*args, **kwargs):
        raise AssertionError('independent replay called implementation')
    monkeypatch.setattr(actor, 'real_trial', forbidden)
    monkeypatch.setattr(actor, 'fit_forest', forbidden)
    monkeypatch.setattr(models, 'fit_forest', forbidden)
    monkeypatch.setattr(models.Forest, 'predict', forbidden)
    monkeypatch.setattr(core.PackedStratifiedSampler, 'choose', forbidden)
    monkeypatch.setattr(core.PackedStratifiedSampler, 'observe', forbidden)
    monkeypatch.setattr(legacy, 'plan_stratified_observations', forbidden)
    data, rows = records
    for arm in ARMS:
        assert audit_packed(rows[arm], data)['assigned'] == 512
