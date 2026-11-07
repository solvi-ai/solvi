"""Finite teaching contract: proposal ambiguity, independent evidence, revocation."""
import copy

import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience.teaching import (
    ConceptRef, TeachingAuthority, TeachingSession,
)


def session(*, budget=None, authority=None, programs=None):
    grammar = Grammar(('x',), constants=(0, 1, 2, 3), max_depth=8, max_nodes=128)
    programs = programs or [Program(('lt', ('input', 'x'), ('const', 1))),
                            Program(('lt', ('input', 'x'), ('const', 2)))]
    return TeachingSession(grammar, programs, [{'x': i} for i in range(4)],
                           budget=budget or WorkBudget(100000), authority=authority)


def accepted(s):
    s.teach({'x': 1}, True)
    return s.admit('small', '1', [{'x': 0}, {'x': 2}], lambda p: p['x'] < 2,
                   source_id='independent-checker', source_revision='oracle-v1')


def test_question_discriminates_and_answer_learns():
    s = session()
    q = s.question()
    assert q.inputs == {'x': 1}
    assert q.predictions == (False, True)
    assert s.answer(q, True) == 1
    assert s.question() is None
    with pytest.raises(ValueError, match='stale'):
        s.answer(q, False)


def test_ambiguity_is_not_admission_and_separate_domain_required():
    s = session()
    assert s.admit('x', '1', [{'x': 0}], lambda p: True, source_id='o', source_revision='1') is None
    s.teach({'x': 1}, True)
    with pytest.raises(ValueError, match='separate'):
        s.admit('x', '1', [{'x': 1}], lambda p: True, source_id='o', source_revision='1')


def test_independent_domain_and_provenance():
    s = session()
    ref = accepted(s)
    assert s.apply(ref, {'x': 0}).status == 'verified'
    assert s.apply(ref, {'x': 2}).value is False
    assert s.apply(ref, {'x': 3}).status == 'unknown'
    record = next(iter(s.snapshot()['body']['concepts'].values()))
    assert record['source_id'] == 'independent-checker'
    assert record['source_revision'] == 'oracle-v1'


def test_wrong_oracle_never_installs():
    s = session()
    s.teach({'x': 1}, True)
    assert s.admit('wrong', '1', [{'x': 2}], lambda p: True, source_id='o', source_revision='1') is None
    assert s.apply(ConceptRef('wrong', '1'), {'x': 2}).status == 'unknown'


def test_exception_contradiction_retained_and_invalidates():
    s = session()
    ref = accepted(s)
    assert s.teach({'x': 1}, False, exception=True) == 0
    assert s.candidates == ()
    assert s.apply(ref, {'x': 0}).status == 'unknown'
    assert s.snapshot()['body']['examples'][-1]['exception'] is True


def test_composition_pins_and_transitive_retraction():
    s = session()
    ref = accepted(s)
    inverse = s.compose('large', '1', Program(('not', ('input', 'small'))), {'small': ref},
                        [{'x': 0}, {'x': 2}], lambda p: p['x'] >= 2,
                        source_id='o', source_revision='1')
    assert s.apply(inverse, {'x': 2}).value is True
    s.retract(ref)
    assert s.apply(inverse, {'x': 2}).status == 'unknown'
    with pytest.raises(ValueError, match='already used'):
        s.admit('small', '1', [{'x': 0}], lambda p: True, source_id='o', source_revision='1')


def test_paid_restore_without_oracle_and_tamper_rejection():
    authority = TeachingAuthority()
    s = session(authority=authority)
    ref = accepted(s)
    snapshot = s.snapshot()
    budget = s.budget
    oracle_before = budget.counts.get('teaching_oracle', 0)
    restored = TeachingSession.restore(snapshot, budget=budget, authority=authority)
    assert budget.counts['teaching_restore_node'] > 0
    assert budget.counts.get('teaching_oracle', 0) == oracle_before
    assert restored.apply(ref, {'x': 2}).value is False
    bad = copy.deepcopy(snapshot)
    bad['body']['concepts'].clear()
    with pytest.raises(ValueError, match='authentication'):
        TeachingSession.restore(bad, budget=budget, authority=authority)
    with pytest.raises(ValueError):
        TeachingSession.restore(snapshot, budget=budget, authority=TeachingAuthority())


def test_rollback_cannot_revive_retracted_proof():
    s = session()
    ref = accepted(s)
    snapshot = s.snapshot()
    s.retract(ref)
    with pytest.raises(ValueError, match='stale snapshot'):
        TeachingSession.restore(snapshot, budget=WorkBudget(100000), authority=s.authority)


def test_parallel_restored_copy_becomes_stale_after_mutation():
    s = session()
    ref = accepted(s)
    saved = s.snapshot()
    other = TeachingSession.restore(saved, budget=s.budget, authority=s.authority)
    other.retract(ref)
    assert s.apply(ref, {'x': 0}).status == 'unknown'
    with pytest.raises(ValueError, match='ledger|stale'):
        s.teach({'x': 3}, False)


@pytest.mark.parametrize('point,value', [({'x': True}, True), ({'x': 1}, 1), ({'x': 1.0}, False)])
def test_exact_scalar_types(point, value):
    s = session()
    with pytest.raises(ValueError):
        s.teach(point, value)


def test_bounded_infinite_iterator_and_prepaid_next():
    calls = []
    def endless():
        while True:
            calls.append(1)
            yield {'x': 0}
    with pytest.raises(ValueError, match='bound'):
        TeachingSession(Grammar(('x',)), [Program(('const', True))], endless(),
                        budget=WorkBudget(10000), max_items=3)
    assert len(calls) == 4
    calls.clear()
    with pytest.raises(BudgetStop):
        TeachingSession(Grammar(('x',)), [Program(('const', True))], endless(),
                        budget=WorkBudget(4), max_items=3)
    assert len(calls) == 1


def test_exhaustion_during_teach_invalidates_old_admission():
    s = session()
    accepted(s)
    # Enough to record the evidence/revision, insufficient to complete search.
    s.budget.charge('test_consume', s.budget.remaining - 5)
    with pytest.raises(BudgetStop):
        s.teach({'x': 3}, True, exception=True)
    assert next(iter(s._concepts.values()))['revision'] != s._revision
    assert s.candidates == ()


def test_admission_exhaustion_never_installs_partial_proof():
    s = session()
    s.teach({'x': 1}, True)
    s.budget.charge('test_consume', s.budget.remaining - 12)
    with pytest.raises(BudgetStop):
        s.admit('partial', '1', [{'x': 0}, {'x': 2}], lambda p: p['x'] < 2,
                source_id='o', source_revision='1')
    assert not s._concepts


def test_oracle_cannot_change_session_then_admit_old_proposal():
    s = session()
    s.teach({'x': 1}, True)
    def reentrant_oracle(point):
        s.teach({'x': 3}, True)
        return point['x'] < 2
    with pytest.raises(ValueError, match='changed during admission'):
        s.admit('stale', '1', [{'x': 0}], reentrant_oracle, source_id='o', source_revision='1')
    assert s.apply(ConceptRef('stale', '1'), {'x': 0}).status == 'unknown'


def test_trusted_authority_checkpoint_supports_new_process(tmp_path):
    import json
    import subprocess
    import sys

    s = session()
    ref = accepted(s)
    snapshot = s.snapshot()
    authority_state = s.authority.export_trusted_state(budget=s.budget)
    # Host secret checkpoint and user snapshot are separate files/trust channels.
    trusted_path, snapshot_path = tmp_path / 'host.json', tmp_path / 'snapshot.json'
    trusted_path.write_text(json.dumps(dict(authority=authority_state, ledger=s.budget.snapshot())))
    snapshot_path.write_text(json.dumps(snapshot))
    script = '''
import json, sys
from solvi.experimental.experience._common import WorkBudget
from solvi.experimental.experience.teaching import TeachingAuthority, TeachingSession, ConceptRef
trusted = json.load(open(sys.argv[1]))
budget = WorkBudget.restore(trusted['ledger'])
oracle_before = budget.counts.get('teaching_oracle', 0)
authority = TeachingAuthority.from_trusted_state(trusted['authority'], budget=budget)
s = TeachingSession.restore(json.load(open(sys.argv[2])), budget=budget, authority=authority)
r = s.apply(ConceptRef('small', '1'), {'x': 2})
assert r.status == 'verified' and r.value is False
assert budget.counts.get('teaching_oracle', 0) == oracle_before
s.retract(ConceptRef('small', '1'))
try:
    TeachingSession.restore(json.load(open(sys.argv[2])), budget=budget, authority=authority)
except ValueError:
    pass
else:
    raise AssertionError('rollback accepted after retraction')
'''
    subprocess.run([sys.executable, '-c', script, str(trusted_path), str(snapshot_path)], check=True)
    assert s.apply(ref, {'x': 2}).status == 'verified'


def test_trusted_authority_checkpoint_strict_validation():
    authority = TeachingAuthority()
    state = authority.export_trusted_state(budget=WorkBudget(100000))
    bad = dict(state, epochs={'a' * 32: True})
    with pytest.raises(ValueError):
        TeachingAuthority.from_trusted_state(bad, budget=WorkBudget(100000))
    with pytest.raises(BudgetStop):
        TeachingAuthority.from_trusted_state(state, budget=WorkBudget(0))


def test_restore_rejects_fresh_or_larger_ledger():
    from solvi.experimental.experience._common import digest
    s = session()
    accepted(s)
    saved = s.snapshot()
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget(s.budget.limit), authority=s.authority)
    larger = s.budget.snapshot()
    larger['limit'] += 100
    larger['digest'] = digest({k: v for k, v in larger.items() if k != 'digest'})
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(larger), authority=s.authority)


def test_restore_cannot_refund_apply_after_snapshot_or_move_cost_categories():
    from solvi.experimental.experience._common import digest
    s = session()
    ref = accepted(s)
    saved = s.snapshot()
    previous = s.budget.snapshot()
    assert s.apply(ref, {'x': 0}).status == 'verified'
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(previous), authority=s.authority)
    moved = s.budget.snapshot()
    moved['counts']['other'] = moved['counts'].pop('program_node')
    moved['digest'] = digest({k: v for k, v in moved.items() if k != 'digest'})
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(moved), authority=s.authority)


def test_public_configuration_is_readonly_or_detached():
    s = session()
    ref = accepted(s)
    detached = s.input_types
    detached['x'] = 'bool'
    assert s.input_types == {'x': 'int'}
    for name, value in [('grammar', Grammar(('z',))), ('input_types', {}),
                        ('authority', TeachingAuthority()), ('result_type', 'int'),
                        ('max_items', 9999), ('budget', WorkBudget(1000000))]:
        with pytest.raises(AttributeError):
            setattr(s, name, value)
    with pytest.raises(AttributeError):
        s.budget.limit = 1000000
    assert s.apply(ref, {'x': 0}).status == 'verified'


def test_restore_at_authority_capacity_has_no_temporary_registry_leak():
    s = session()
    ref = accepted(s)
    # Mechanically fill the trusted registry; these are host-owned dummy anchors.
    for i in range(9999):
        name = f'{i:032x}'
        s.authority._epochs[name] = 0
        s.authority._ledgers[name] = s.budget.snapshot()
    saved = s.snapshot()
    before = set(s.authority._epochs)
    restored = TeachingSession.restore(saved, budget=s.budget, authority=s.authority)
    assert set(s.authority._epochs) == before
    assert restored.apply(ref, {'x': 0}).status == 'verified'


def test_restore_cannot_refund_external_shared_ledger_work():
    shared = WorkBudget(100000)
    s = session(budget=shared)
    accepted(s)
    saved = s.snapshot()
    old = shared.snapshot()
    shared.charge('other', 500)
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(old), authority=s.authority)
    trusted = s.authority.export_trusted_state(budget=shared)
    exported_anchor = trusted['ledgers'][s._session]
    assert exported_anchor['counts']['other'] == 500
    restored_authority = TeachingAuthority.from_trusted_state(trusted, budget=shared)
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(old), authority=restored_authority)


def test_failed_replay_does_not_leak_sessions_and_keeps_debits():
    shared = WorkBudget(100000)
    s = session(budget=shared)
    accepted(s)
    saved = s.snapshot()
    before_sessions = set(s.authority._epochs)
    before_spent = shared.spent
    def fail_sink(state):
        if 'teaching_restore_record' in state['counts']:
            raise RuntimeError('host sink failed during record replay')
    shared.on_charge = fail_sink
    with pytest.raises(RuntimeError, match='host sink'):
        TeachingSession.restore(saved, budget=shared, authority=s.authority)
    assert set(s.authority._epochs) == before_sessions
    assert shared.spent > before_spent
    assert s.authority._ledgers[s._session]['spent'] == shared.spent
    shared.on_charge = None
    restored = TeachingSession.restore(saved, budget=shared, authority=s.authority)
    assert restored.candidates
    assert set(s.authority._epochs) == before_sessions


def test_trusted_import_cannot_refund_its_own_debits_or_fork_live_ledger():
    shared = WorkBudget(100000)
    s = session(budget=shared)
    accepted(s)
    saved = s.snapshot()
    trusted = s.authority.export_trusted_state(budget=shared)
    pre_import = shared.snapshot()
    imported = TeachingAuthority.from_trusted_state(trusted, budget=shared)
    assert shared.spent > pre_import['spent']
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(pre_import), authority=imported)
    shared.charge('other_after_import', 500)
    with pytest.raises(ValueError, match='ledger'):
        TeachingSession.restore(saved, budget=WorkBudget.restore(pre_import), authority=imported)
    with pytest.raises(ValueError, match='ledger'):
        TeachingAuthority.from_trusted_state(trusted, budget=WorkBudget(shared.limit))
    restored = TeachingSession.restore(saved, budget=shared, authority=imported)
    assert restored.candidates


def test_live_budget_identity_cannot_be_forked():
    grammar = Grammar(('x',), constants=(0, 1))
    budget = WorkBudget(100000)
    session = TeachingSession(grammar, [Program(('lt', ('input', 'x'), ('const', 1)))], [{'x': 0}], budget=budget)
    saved = session.snapshot()
    clone = WorkBudget.restore(budget.snapshot())
    with pytest.raises(ValueError, match='identity'):
        TeachingSession.restore(saved, budget=clone, authority=session.authority)
