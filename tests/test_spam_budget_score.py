"""Synthetic end-to-end actor versus independent audit regression checks."""
from copy import deepcopy
import pytest
from spam_budget_trial import ARMS, real_trial
from spam_budget_score import audit_budget


def data_fixture(one_class=False):
    def row(i):
        return dict(id=i, inputs={'f0': i % 8}, label=False if one_class else i % 8 >= 4)
    return dict(name='spambase', seed=3, split='dev', pool=[row(i) for i in range(256)],
                test=[row(i) for i in range(256, 768)], feature_names=['x'], source_sha='fixture',
                preprocess_costs={'preprocess_fields': 768},
                group_ids={'pool': list(range(256)), 'test': list(range(256,768)),
                           'holdout': list(range(768,1280))}, thresholds={}, rawcounts={},
                quantized_test_rows_matching_pool=512)


@pytest.fixture(scope='module')
def rows():
    data = data_fixture()
    return data, {arm: real_trial('spambase', None, 3, 'dev', arm, 128, prepared=data) for arm in ARMS}


@pytest.mark.parametrize('arm', ARMS)
def test_exact_reference_all_arms(rows, arm):
    data, results = rows
    audit = audit_budget(results[arm], None, None, prepared=data)
    assert audit['assigned'] == 512
    assert audit['correct'] + audit['wrong'] + audit['abstain'] + audit['budget'] == 512


@pytest.mark.parametrize('tamper', ['label', 'planname', 'feature', 'threshold', 'fitseed', 'node',
                                  'debit', 'rawcounts', 'holdoutduplicate', 'boolunits', 'extraassignment'])
def test_reference_rejects_tamper(rows, tamper):
    data, results = rows
    data, result = deepcopy(data), deepcopy(results['MIXED32'])
    if tamper == 'label':
        result['receipts'][0]['expected'] = not result['receipts'][0]['expected']
    elif tamper == 'planname':
        result['plan']['names'][0] = '999999'
    elif tamper == 'feature':
        result['plan']['features'][0] = 'f99'
    elif tamper == 'threshold':
        result['plan']['thresholds'][0] += 1
    elif tamper == 'fitseed':
        result['fits'][0]['seed'] += 1
    elif tamper == 'node':
        result['fits'][0]['model']['trees'][0]['root']['pos'] += 1
    elif tamper == 'debit':
        result['trace'][0][1] += 1
    elif tamper == 'rawcounts':
        result['rawcounts']['forged'] = 1
    elif tamper == 'holdoutduplicate':
        data['group_ids']['holdout'][-1] = data['group_ids']['holdout'][0]
    elif tamper == 'boolunits':
        result['trace'][0][1] = True
    elif tamper == 'extraassignment':
        result['predictions'].append(deepcopy(result['predictions'][0]))
    with pytest.raises(ValueError):
        audit_budget(result, None, None, prepared=data)


def test_single_class_fallback():
    data = data_fixture(one_class=True)
    result = real_trial('spambase', None, 3, 'dev', 'MIXED32', 128, prepared=data)
    assert result['plan']['reason'] == 'random_prefix'
    assert result['plan']['names'] == list(map(str, range(24,32)))
    assert audit_budget(result, None, None, prepared=data)['assigned'] == 512
