import pytest
from active_real_data import Dataset, Row, _group_id
from packed_spam_data import _prepare


def fixture():
    rows = tuple(Row(i + 1, (str(i),), bool(i % 2)) for i in range(2400))
    ds = Dataset('spambase', ('x',), rows, 'fixture', {})
    folds = {f'f{i}': [_group_id(ds.name, r.features) for r in rows[i*512:(i+1)*512]] for i in range(3)}
    return ds, dict(test_folds=folds, reserved=sum(folds.values(), []), previous_used=[])


def test_explicit_closed_and_prefix():
    ds, roles = fixture()
    dev = _prepare(ds, 7, 'dev', roles)
    final = _prepare(ds, 7, 'final', roles)
    assert dev['pool'] == final['pool']
    assert set(dev['group_ids']['holdout']) == set(roles['test_folds']['f0'])
    assert not set(dev['group_ids']['test']) & set(roles['test_folds']['f0'])
    assert set(final['group_ids']['test']) == set(roles['test_folds']['f0'])
    assert len(dev['pool']) == 256 and len(dev['test']) == 512


def test_exposure_rejected():
    ds, roles = fixture()
    roles['previous_used'] = ['foreign']
    with pytest.raises(ValueError, match='foreign'):
        _prepare(ds, 7, 'dev', roles)
    roles['previous_used'] = []
    roles['test_folds']['f0'][-1] = roles['test_folds']['f0'][0]
    with pytest.raises(ValueError, match='512 distinct'):
        _prepare(ds, 7, 'dev', roles)

def test_all_folds_reserved_and_disjoint():
    ds, roles = fixture()
    for fold in roles['test_folds']:
        dev = _prepare(ds, 7, 'dev', roles, fold)
        final = _prepare(ds, 7, 'final', roles, fold)
        assert not (set(dev['group_ids']['pool']) | set(dev['group_ids']['test'])) & set(roles['reserved'])
        assert set(final['group_ids']['test']) == set(roles['test_folds'][fold])
        assert set(dev['reserved_ids']) == set(roles['reserved'])
    roles['reserved'].pop()
    with pytest.raises(ValueError, match='union'):
        _prepare(ds, 7, 'dev', roles)
