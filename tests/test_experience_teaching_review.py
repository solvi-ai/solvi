"""Independent adversarial gates for finite teaching authority and composition."""
import copy
import itertools
import pytest

from solvi.experimental.experience._common import WorkBudget, BudgetStop, digest
from solvi.experimental.experience.programs import Grammar, Program
from solvi.experimental.experience.teaching import TeachingSession, TeachingAuthority, ConceptRef


def session(authority=None, budget=None):
    return TeachingSession(Grammar(('x',), constants=(0,1,2,3), max_depth=10, max_nodes=128),
        [Program(('lt', ('input','x'), ('const',1))), Program(('lt', ('input','x'), ('const',2)))],
        [{'x':i} for i in range(5)], budget=budget if budget is not None else WorkBudget(100000), authority=authority)


def admit(s):
    s.teach({'x':1}, True)
    return s.admit('small','v1',[{'x':0},{'x':2}],lambda p:p['x'] < 2,
                   source_id='independent',source_revision='oracle1')


def test_review_plain_hash_cannot_replace_authority_signature():
    s = session(); ref = admit(s)
    snap = copy.deepcopy(s.snapshot())
    for record in snap['body']['concepts'].values():
        record['evidence'][0]['value'] = False
    snap['signature'] = digest(snap['body'])
    with pytest.raises(ValueError, match='authentication'):
        TeachingSession.restore(snap,budget=s.budget,authority=s.authority)
    assert s.apply(ref, {'x':0}).value is True


def test_review_same_secret_without_epoch_registry_not_authorized():
    authority = TeachingAuthority(b'x'*32)
    s = session(authority); admit(s)
    with pytest.raises(ValueError, match='stale snapshot|unknown authority'):
        TeachingSession.restore(s.snapshot(),budget=WorkBudget(100000),authority=TeachingAuthority(b'x'*32))


def test_review_new_counterexample_invalidates_signed_saved_proof():
    s = session(); ref = admit(s)
    old = s.snapshot()
    s.teach({'x':4},True,exception=True)
    assert s.apply(ref, {'x':0}).status == 'unknown'
    with pytest.raises(ValueError, match='stale'):
        TeachingSession.restore(old,budget=s.budget,authority=s.authority)


def test_review_ambiguity_and_admission_failure_never_choose_arbitrarily():
    s = session()
    assert s.admit('ambiguous','v1',[{'x':10}],lambda p:False,source_id='o',source_revision='1') is None
    s.teach({'x':1},True)
    with pytest.raises(ValueError,match='separate'):
        s.admit('leak','v1',[{'x':1}],lambda p:True,source_id='o',source_revision='1')
    assert s.admit('wrong','v1',[{'x':0}],lambda p:False,source_id='o',source_revision='1') is None
    assert s.apply(ConceptRef('wrong','v1'),{'x':0}).status == 'unknown'


def test_review_genuine_composition_and_three_level_revocation():
    s = session(); base = admit(s)
    inverse = s.compose('inverse','v1',Program(('not',('input','slot'))),{'slot':base},
        [{'x':0},{'x':2}],lambda p:p['x'] >= 2,source_id='o',source_revision='1')
    double = s.compose('double','v1',Program(('not',('input','slot'))),{'slot':inverse},
        [{'x':0},{'x':2}],lambda p:p['x'] < 2,source_id='o',source_revision='1')
    assert s.apply(inverse,{'x':2}).value is True
    assert s.apply(double,{'x':2}).value is False
    snapshot = s.snapshot()
    restored = TeachingSession.restore(snapshot,budget=s.budget,authority=s.authority)
    restored.retract(base)
    assert restored.apply(inverse,{'x':2}).status == 'unknown'
    assert restored.apply(double,{'x':2}).status == 'unknown'
    assert s.apply(double,{'x':2}).status == 'unknown'
    with pytest.raises(ValueError,match='stale'):
        TeachingSession.restore(snapshot,budget=s.budget,authority=s.authority)


def test_review_composed_program_must_match_own_independent_oracle():
    s = session(); base = admit(s)
    wrong = s.compose('wrong','v1',Program(('not',('input','slot'))),{'slot':base},
        [{'x':0}],lambda p:True,source_id='o',source_revision='1')
    assert wrong is None
    assert s.apply(ConceptRef('wrong','v1'),{'x':0}).status == 'unknown'


def test_review_infinite_admission_iterator_is_bounded_and_cannot_install():
    s = session(); s.teach({'x':1},True)
    with pytest.raises(ValueError,match='bound'):
        s.admit('infinite','v1',itertools.repeat({'x':0}),lambda p:True,source_id='o',source_revision='1')
    assert s.apply(ConceptRef('infinite','v1'),{'x':0}).status == 'unknown'


def test_review_on_charge_failure_before_admission_mutation():
    class SinkFailure(Exception):
        pass
    enabled = [False]
    def sink(ledger):
        if enabled[0]:
            raise SinkFailure()
    budget = WorkBudget(100000,on_charge=sink)
    s = session(budget=budget)
    s.teach({'x':1},True)
    enabled[0] = True
    with pytest.raises(SinkFailure):
        s.admit('partial','v1',[{'x':0}],lambda p:True,source_id='o',source_revision='1')
    enabled[0] = False
    assert s.apply(ConceptRef('partial','v1'),{'x':0}).status == 'unknown'


def test_review_paid_restore_zero_budget_never_authenticates():
    s = session(); admit(s)
    with pytest.raises((BudgetStop, ValueError)):
        TeachingSession.restore(s.snapshot(),budget=WorkBudget(0),authority=s.authority)


def test_review_cycles_in_untrusted_snapshot_are_bounded():
    s = session(); snap=s.snapshot()
    snap['body']['cycle'] = snap['body']
    with pytest.raises((ValueError,BudgetStop)):
        TeachingSession.restore(snap,budget=WorkBudget(100),authority=s.authority)
