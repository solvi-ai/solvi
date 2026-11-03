"""Denied seed reads are auditable, but are not completed evaluations."""
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from solvi.experimental.changes import EvalLocked, EvaluatorLock


def test_denied_read_is_logged_without_opening_seeds(tmp_path, monkeypatch):
    log = tmp_path / 'access.jsonl'
    lock = EvaluatorLock(tmp_path / 'missing_seeds.json', access_log=log, env='SOLVI_TEST_EVAL_AUDIT')
    monkeypatch.setenv(lock.env, '1')
    with pytest.raises(EvalLocked):
        lock.seeds('forbidden probe', by='candidate')
    records = [record.data for record in lock.access_log.iter('eval_access')]
    assert len(records) == 1
    assert records[0]['granted'] is False
    assert records[0]['purpose'] == 'forbidden probe' and records[0]['by'] == 'candidate'
    assert lock.access_count() == 0
    assert 'seeds' not in records[0]
    assert lock.access_log.verify()['ok']


def test_success_count_survives_denied_reads_and_legacy_records(tmp_path, monkeypatch):
    seeds, log = tmp_path / 'seeds.json', tmp_path / 'access.jsonl'
    pin = EvaluatorLock.create(seeds, seeds=[11, 23])
    lock = EvaluatorLock(seeds, pinned=pin, access_log=log, env='SOLVI_TEST_EVAL_AUDIT')
    monkeypatch.setenv(lock.env, '1')
    with pytest.raises(EvalLocked):
        lock.seeds('denied')
    monkeypatch.delenv(lock.env)
    assert lock.seeds('authorized') == [11, 23]
    assert lock.access_count() == 1
    records = [record.data for record in lock.access_log.iter('eval_access')]
    assert len(records) == 2 and records[1]['granted'] is True
    # Legacy granted records had no granted field; their count remains valid.
    legacy = {k: v for k, v in records[1].items()
              if k not in {'granted', 'seq', 'time', 'prev', 'hash', 'id'}}
    lock.access_log._append(legacy)
    assert lock.access_count() == 2
    assert lock.access_log.verify()['ok']
    assert all('seeds' not in row for row in records)
