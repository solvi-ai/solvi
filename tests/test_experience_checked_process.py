"""Actual process death at checked-learning durability boundaries."""
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import pytest

from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget, digest
from solvi.experimental.experience.competences import CompetenceRegistry
from solvi.experimental.experience.curriculum import AutoCurriculum
from solvi.experimental.experience.domain_checks import DomainReceipt, FiniteDomain
from solvi.experimental.experience.durable import SQLiteEnvironment
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.loop import ExperienceRunner
from solvi.experimental.experience.programs import Grammar


CASES = {str(i): {'family': 'resource', 'x': i + 6, 'y': i % 3 + 1} for i in range(10)}
STAGES = ['learning_reserved', 'domain_read_intent', 'domain_read_receipt',
          'learning_committed', 'learn_completed']


class LoggedOracle:
    source_id = 'audit'
    source_revision = 'sql-sub-v1'

    def __init__(self, path):
        self.path = Path(path)

    def read(self, request_id, source_revision, point):
        assert source_revision == self.source_revision
        assert set(point) == {'x', 'y'}
        assert all(type(v) is int for v in point.values())
        with sqlite3.connect(':memory:') as conn:
            conn.execute('PRAGMA query_only=ON')
            expected = conn.execute('SELECT ? - ?', (point['x'], point['y'])).fetchone()[0]
        # The fsynced external read audit survives destruction of Python state.
        with self.path.open('a') as stream:
            stream.write(json.dumps({'request_id': request_id, 'point': point}) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        return DomainReceipt(request_id, 'read:' + request_id, self.source_id,
                             source_revision, digest(point), expected)


def fresh(directory, *, create=False):
    directory = Path(directory)
    environment = SQLiteEnvironment(directory / 'environment.db', CASES if create else None)
    store = KnowledgeStore(str(directory / 'knowledge.db'))
    journal = ExperienceJournal(store, 'checked-process', {})
    grammar = Grammar(('x', 'y'), constants=(), operators=('sub',),
                      max_depth=2, max_nodes=3, max_candidates=50)
    registry = CompetenceRegistry(grammar, store=store)
    oracle = LoggedOracle(directory / 'oracle.jsonl')
    domain = FiniteDomain('resource', 'v1', 'xy-int-v1', grammar, oracle.source_id,
                          oracle.source_revision,
                          [dict(x=c['x'], y=c['y']) for c in CASES.values()],
                          budget=WorkBudget(10000))
    return environment, store, journal, registry, {'resource': (domain, oracle)}


def child_main(directory, stage):
    directory = Path(directory)
    environment, store, journal, registry, contracts = fresh(directory, create=True)
    runner = ExperienceRunner(environment, journal, WorkBudget(100000), registry,
                              mode='fixed', checked_domains=contracts,
                              checked_store_policy='evidence_only')
    controller = AutoCurriculum(runner, list(CASES), max_attempts=1)

    def halt():
        with (directory / 'marker').open('w') as stream:
            json.dump({'stage': stage, 'spent': runner.budget.spent}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        while True:
            time.sleep(.05)

    append = journal.append

    def record(kind, payload):
        result = append(kind, payload)
        if kind == stage:
            halt()
        return result

    journal.append = record
    learn = runner.learn

    def learn_and_halt(*args, **kwargs):
        result = learn(*args, **kwargs)
        if stage == 'learn_completed':
            assert result is not None and result.accepted
            halt()
        return result

    runner.learn = learn_and_halt
    controller.run()
    raise AssertionError('fault boundary not reached')


def read_audit(directory):
    path = directory / 'oracle.jsonl'
    return [] if not path.exists() else [json.loads(line) for line in path.read_text().splitlines()]


@pytest.mark.parametrize('stage', STAGES)
@pytest.mark.skipif(os.name != 'posix', reason='SIGKILL requires POSIX')
def test_checked_learning_sigkill_fresh_recovery_no_oracle_replay(tmp_path, stage):
    with (tmp_path / 'child.log').open('w') as stream:
        child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                  '--child', str(tmp_path), stage], stdout=stream,
                                 stderr=subprocess.STDOUT, env=os.environ.copy())
        try:
            deadline = time.monotonic() + 120
            while child.poll() is None and not (tmp_path / 'marker').exists() and time.monotonic() < deadline:
                time.sleep(.02)
            assert (tmp_path / 'marker').exists(), (tmp_path / 'child.log').read_text()
            # Only the explicitly created child PID is killed.
            child.kill()
            child.wait(timeout=5)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=5)
    assert child.returncode == -9
    stopped = json.loads((tmp_path / 'marker').read_text())
    before_reads = read_audit(tmp_path)
    committed = stage in ('learning_committed', 'learn_completed')
    assert len(before_reads) == (10 if committed else 1 if stage == 'domain_read_receipt' else 0)

    # All objects, including the oracle and domain, are constructed after SIGKILL.
    environment, store, journal, registry, contracts = fresh(tmp_path)
    records = journal.records()
    controller_state = [e['payload']['controller_state'] for e in records if e['kind'] == 'curriculum'][-1]
    restored = AutoCurriculum.restore(
        {'controller': controller_state, 'runner': journal.latest()},
        environment=environment, journal=journal, registry=registry,
        checked_domains=contracts, checked_store_policy='evidence_only')
    assert restored.runner.budget.spent >= stopped['spent']
    results = restored.run()
    assert restored.state['status'] == 'done'
    assert len(results) == 10 and all(r['ok'] for r in results)
    assert all(s['effects'] == 1 for s in environment.state.values())
    assert len(restored.state['seen']) == 10
    assert restored.state['attempts'] == 1
    assert restored.state['learning'] is None
    assert read_audit(tmp_path) == before_reads
    assert restored.runner.budget.spent >= stopped['spent']
    assert sum(e['kind'] == 'learning_intent' for e in journal.records()) == 1
    packets = list(restored.state['packets'].values())
    if committed:
        assert all(p['transfer'] for p in packets)
        assert all(r['competence_id'] is not None for r in results[8:])
        assert all(c['status'] == 'active' for c in restored.runner.registry.snapshot()['competences'].values())
    else:
        assert all(not p['transfer'] and p['generation'] == 1 and not p['proposal'] and not p['admission'] for p in packets)
        assert not restored.runner.registry.snapshot()['competences']
        assert all(r['competence_id'] is None for r in results[8:])
    ledgers = []
    for event in journal.records():
        payload = event['payload']
        if event['kind'] == 'debit':
            ledgers.append(payload['ledger']['spent'])
        elif 'state' in payload:
            ledgers.append(payload['state']['ledger']['spent'])
    assert ledgers == sorted(ledgers)
    assert journal.verify()


if __name__ == '__main__' and sys.argv[1:2] == ['--child']:
    child_main(sys.argv[2], sys.argv[3])
