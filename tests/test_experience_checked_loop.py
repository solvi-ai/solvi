"""Checked runner/controller integration and durable failure boundaries."""
import pytest

from test_experience_curriculum import Host
from solvi.core.knowledge import KnowledgeStore
from solvi.experimental.experience._common import WorkBudget, BudgetStop, digest
from solvi.experimental.experience.competences import CompetenceRegistry
from solvi.experimental.experience.curriculum import AutoCurriculum
from solvi.experimental.experience.domain_checks import FiniteDomain, DomainReceipt
from solvi.experimental.experience.journal import ExperienceJournal
from solvi.experimental.experience.loop import ExperienceRunner
from solvi.experimental.experience.programs import Grammar


class IdentityOracle:
    source_id = 'audit'
    source_revision = 'v1'

    def __init__(self):
        self.calls = 0

    def read(self, request_id, revision, point):
        self.calls += 1
        return DomainReceipt(request_id, 'read:' + request_id, self.source_id,
                             revision, digest(point), point['x'])


def make(n=10, with_store=False):
    grammar = Grammar(('x',), operators=('add',), max_candidates=10)
    registry = CompetenceRegistry(grammar, store=KnowledgeStore() if with_store else None)
    oracle = IdentityOracle()
    domain = FiniteDomain('domain', 'm1', 's1', grammar, 'audit', 'v1',
                          [{'x': i} for i in range(n)], budget=WorkBudget(100000))
    contracts = {'domain': (domain, oracle)}
    runner = ExperienceRunner(Host(), ExperienceJournal(KnowledgeStore(), 'checked', {}),
                              WorkBudget(100000), registry, checked_domains=contracts,
                              checked_store_policy='evidence_only' if with_store else 'deny')
    return AutoCurriculum(runner, [str(i) for i in range(n)]), contracts


def resume(controller, snapshot, contracts):
    return AutoCurriculum.restore(snapshot, environment=controller.runner.environment,
                                  journal=controller.runner.journal, registry=controller.runner.registry,
                                  checked_domains=contracts,
                                  checked_store_policy=controller.runner.checked_store_policy)


def test_clearing_contracts_never_downgrades_to_empirical():
    controller, _ = make()
    controller.runner.checked_domains.clear()
    with pytest.raises(ValueError, match='contract changed'):
        controller.run()
    assert not controller.runner.environment.calls
    assert not controller.runner.registry.snapshot()['admissions']


@pytest.mark.parametrize('with_store', [False, True])
def test_checked_curriculum_committed_crash_recovers_without_oracle(monkeypatch, with_store):
    controller, contracts = make(with_store=with_store)
    snapshot = controller.snapshot()
    journal = controller.runner.journal
    append = journal.append

    def crash(kind, payload):
        result = append(kind, payload)
        if kind == 'learning_committed':
            raise RuntimeError('after committed write')
        return result

    monkeypatch.setattr(journal, 'append', crash)
    with pytest.raises(RuntimeError, match='after committed write'):
        controller.run()
    calls = contracts['domain'][1].calls
    assert calls == 10
    monkeypatch.setattr(journal, 'append', append)
    restored = resume(controller, snapshot, contracts)
    restored.run()
    assert restored.state['status'] == 'done'
    assert restored.state['attempts'] == 1
    assert restored.runner.learning is None
    assert contracts['domain'][1].calls == calls
    assert all(r['competence_id'] for r in restored.runner.results[8:])
    assert len(restored.runner.environment.calls) == 10
    with pytest.raises(ValueError, match='original host contracts'):
        AutoCurriculum.restore(snapshot, environment=controller.runner.environment,
                               journal=journal, registry=controller.runner.registry)


def test_failure_before_commit_burns_packet_without_oracle_retry(monkeypatch):
    controller, contracts = make(8)
    snapshot = controller.snapshot()
    journal = controller.runner.journal
    append = journal.append

    def crash(kind, payload):
        result = append(kind, payload)
        if kind == 'learning_reserved':
            raise RuntimeError('before check')
        return result

    monkeypatch.setattr(journal, 'append', crash)
    with pytest.raises(RuntimeError, match='before check'):
        controller.run()
    monkeypatch.setattr(journal, 'append', append)
    restored = resume(controller, snapshot, contracts)
    restored.run()
    assert contracts['domain'][1].calls == 0
    assert restored.state['attempts'] == 1
    assert restored.state['learning'] is None
    assert all(not p['proposal'] and not p['admission'] for p in restored.state['packets'].values())


@pytest.mark.parametrize('target', ['runner', 'controller'])
@pytest.mark.parametrize('error_type', [BudgetStop, ValueError])
def test_post_debit_sink_failure_is_not_normal_budget_and_blocks_followup(monkeypatch, target, error_type):
    controller, _ = make()
    runner = controller.runner
    if target == 'runner':
        runner.start('0')
        step = runner.step
    else:
        step = controller.step
    before = runner.budget.spent
    debit = runner.journal.debit

    def broken(ledger):
        debit(ledger)
        raise error_type('sink failed after write')

    monkeypatch.setattr(runner.journal, 'debit', broken)
    with pytest.raises(error_type, match='sink failed'):
        step()
    assert runner.budget.spent > before
    calls = len(runner.environment.calls)
    records = len(runner.journal.records())
    with pytest.raises(ValueError, match='restore before continuing'):
        step()
    assert len(runner.environment.calls) == calls
    assert len(runner.journal.records()) == records


@pytest.mark.parametrize('target', ['runner', 'controller'])
def test_normal_budget_exhaustion_returns_budget(target):
    controller, _ = make()
    runner = controller.runner
    if target == 'runner':
        runner.start('0')
        step = runner.step
    else:
        step = controller.step
    runner.budget.limit = runner.budget.spent
    assert step() == 'budget'
    assert not runner._debit_failed


def test_recovery_rejects_committed_packet_generation_mismatch(monkeypatch):
    controller, contracts = make(8)
    snapshot = controller.snapshot()
    journal = controller.runner.journal
    append = journal.append

    def crash(kind, payload):
        result = append(kind, payload)
        if kind == 'learning_committed':
            raise RuntimeError('committed')
        return result

    monkeypatch.setattr(journal, 'append', crash)
    with pytest.raises(RuntimeError, match='committed'):
        controller.run()
    monkeypatch.setattr(journal, 'append', append)
    restored = resume(controller, snapshot, contracts)
    pending = restored.state['learning']
    restored.state['packets'][pending['context']]['generation'] += 1
    calls = contracts['domain'][1].calls
    with pytest.raises(ValueError, match='another curriculum packet'):
        restored.step()
    assert contracts['domain'][1].calls == calls
    assert restored.runner.learning is not None


@pytest.mark.parametrize('change', ['revision', 'unavailable'])
def test_runner_rechecks_live_source_before_prediction_and_retracts_store(change):
    controller, contracts = make(with_store=True)
    controller.run()
    runner = controller.runner
    state = runner.registry.snapshot()
    cid, record = next(iter(state['competences'].items()))
    sid = record['store_id']
    assert runner.registry.store.usable(sid)
    oracle = contracts['domain'][1]
    calls = oracle.calls
    if change == 'revision':
        oracle.source_revision = 'changed'
    else:
        oracle.available = False
    runner.start('10')
    runner.step()
    revoked = runner.registry.snapshot()['competences'][cid]
    assert revoked['status'] == 'revoked'
    assert revoked['reason'] == 'source_retracted'
    assert not runner.registry.store.usable(sid)
    assert runner.competence_id is None
    assert oracle.calls == calls
    assert any(e['kind'] == 'domain_source_changed' for e in runner.journal.records())


def test_runner_rejects_mutated_domain_binding_before_external_action():
    controller, contracts = make()
    controller.run()
    runner = controller.runner
    original, oracle = contracts['domain']
    changed = FiniteDomain('domain', 'm1', 's1', runner.registry.grammar, 'audit', 'v1',
                           [{'x': 8}], budget=WorkBudget(10000))
    runner.checked_domains['domain'] = (changed, oracle)
    runner.start('10')
    calls = len(runner.environment.calls)
    applications = runner.registry.snapshot()['applications']
    with pytest.raises(ValueError, match='host domain contract changed'):
        runner.step()
    assert len(runner.environment.calls) == calls
    assert runner.registry.snapshot()['applications'] == applications


@pytest.mark.parametrize('change', ['revision', 'unavailable'])
def test_runner_rechecks_live_source_before_new_learning(change):
    from solvi.experimental.experience.competences import ExternalExample
    controller, contracts = make(8)
    controller.run()
    runner = controller.runner
    packet = next(iter(controller.state['packets'].values()))
    oracle = contracts['domain'][1]
    if change == 'revision':
        oracle.source_revision = 'changed'
    else:
        oracle.available = False
    before = sum(e['kind'] == 'learning_intent' for e in runner.journal.records())
    calls = oracle.calls
    with pytest.raises(ValueError, match='no checked contract'):
        runner.learn([ExternalExample.from_dict(e) for e in packet['proposal']],
                     [ExternalExample.from_dict(e) for e in packet['admission']])
    assert runner.learning is None
    assert oracle.calls == calls
    assert sum(e['kind'] == 'learning_intent' for e in runner.journal.records()) == before
    assert all(c['status'] == 'revoked' for c in runner.registry.snapshot()['competences'].values())
