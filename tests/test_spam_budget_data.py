import pytest
from active_real_data import Dataset, Row, _group_id
from spam_budget_data import _prepare


def fixture():
    rows = tuple(Row(i + 1, (str(i),), bool(i % 2)) for i in range(1300))
    ds = Dataset('spambase', ('x',), rows, 'fixture', {})
    holdout = [_group_id(ds.name, r.features) for r in rows[:512]]
    return ds, dict(holdout=holdout, previous_used=[])


def test_explicit_closed_and_prefix():
    ds, roles = fixture()
    dev = _prepare(ds, 7, 'dev', roles)
    final = _prepare(ds, 7, 'final', roles)
    assert dev['pool'] == final['pool']
    assert set(dev['group_ids']['holdout']) == set(roles['holdout'])
    assert not set(dev['group_ids']['test']) & set(roles['holdout'])
    assert set(final['group_ids']['test']) == set(roles['holdout'])
    assert len(dev['pool']) == 256 and len(dev['test']) == 512


def test_exposure_rejected():
    ds, roles = fixture()
    roles['previous_used'] = roles['holdout'][:1]
    with pytest.raises(ValueError, match='exposed'):
        _prepare(ds, 7, 'dev', roles)
    roles['previous_used'] = []
    roles['holdout'][-1] = roles['holdout'][0]
    with pytest.raises(ValueError, match='512 distinct'):
        _prepare(ds, 7, 'dev', roles)
