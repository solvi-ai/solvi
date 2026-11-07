from dataclasses import replace
import pytest
from solvi.experimental.experience._common import WorkBudget, BudgetStop, digest
from solvi.experimental.experience.preferences import (Alternative, Requirement, PreferenceSession, check_regret_certificate)


def session(**kw):
    return PreferenceSession([Alternative('a', (10, 0, 0)), Alternative('b', (0, 10, 0)), Alternative('c', (4, 4, 0))], [(1, 0, 0), (0, 1, 0)], budget=kw.pop('budget', WorkBudget(100000)), **kw)


def test_exact_choice_and_independent_certificate():
    s = session()
    c = s.choose()
    assert (c.choice, c.regret) == ('c', 6)
    assert check_regret_certificate(s, c, budget=s.budget)
    assert not check_regret_certificate(s, replace(c, regret=5), budget=s.budget)
    s.choose = lambda: (_ for _ in ()).throw(AssertionError('producer called'))
    assert check_regret_certificate(s, c, budget=s.budget)


def test_query_value_cost_and_declared_prior():
    s = session()
    q = s.next_question()
    assert q is not None and q.reduction == 6
    assert s.next_question(cost=6) is None
    assert s.next_question(criterion='expected', probabilities=(2, 1)) is not None
    with pytest.raises(ValueError):
        s.next_question(criterion='expected', probabilities=(True, 1))
    s.answer('one', q.left, q.right, source='author')
    assert s.choose().regret == 0
    assert s.next_question() is None


def test_conflicts_saved_revised_retracted_and_ties():
    s = session()
    s.answer('x', 'a', 'b', source='author')
    s.answer('y', 'b', 'a', source='author')
    assert s.choose().status == 'conflict'
    assert s.conflicts()['answers'] == (('x', 'y'),)
    s.retract('y', source='correction')
    assert s.choose().choice == 'a'
    s.answer('x', 'b', 'a', source='revision')
    assert s.choose().choice == 'b'
    assert [a.revision for a in s.answers if a.id == 'x'] == [1, 2]
    s.answer('tie', 'a', 'b', source='author', relation='tie')
    assert s.choose().status == 'conflict'


def test_strict_cycle_equal_weights_and_weak_semantics():
    s = PreferenceSession([Alternative('a',(1,0,0)), Alternative('b',(0,1,0))], [(1,1,0)], budget=WorkBudget(1000))
    s.answer('x','a','b',source='author',relation='weak')
    assert s.choose().status == 'ok'
    s.answer('x','a','b',source='author')
    assert s.choose().status == 'conflict'


def test_requirements_minimal_relaxations_and_hard_never_weakened():
    s = session(requirements=[Requirement('x',0,5,False), Requirement('y',1,5,False)])
    assert s.choose().status == 'no_feasible_alternative'
    assert s.conflicts()['requirements'] == (('x','y'),)
    assert s.relaxations() == (('x',), ('y',))
    assert len(s.requirements) == 2
    h = session(requirements=[Requirement('hard',2,1)])
    assert h.relaxations() == ()
    assert check_regret_certificate(h,h.choose(),budget=h.budget)


def test_replay_strict_revision_provenance_and_paid_restore():
    s = session(host_revision='r7')
    s.answer('x','a','b',source='author')
    s.retract('x',source='author correction')
    snap = s.snapshot()
    b = s.budget
    expected = s.choose()
    restored = PreferenceSession.restore(snap,budget=b,host_revision='r7', authority=s.authority)
    assert restored.snapshot()["answers"] == snap["answers"] and b.spent > snap["ledger"]["spent"]
    assert restored.choose() == expected
    with pytest.raises(ValueError):
        PreferenceSession.restore(snap,budget=b,authority=s.authority)
    bad = dict(snap, extra=1)
    with pytest.raises(ValueError):
        PreferenceSession.restore(bad,budget=b,host_revision='r7', authority=s.authority)
    bad = dict(snap)
    bad['answers'] = [dict(snap['answers'][0], source='')]
    bad['digest'] = digest({k:v for k,v in bad.items() if k != 'digest'})
    with pytest.raises(ValueError):
        PreferenceSession.restore(bad,budget=b,host_revision='r7', authority=s.authority)
    with pytest.raises(BudgetStop):
        PreferenceSession.restore(snap,budget=WorkBudget(0),host_revision='r7', authority=s.authority)


def test_strict_inputs_and_budget_no_false_proof():
    with pytest.raises(ValueError):
        Alternative('bad',(True,0,0))
    with pytest.raises(ValueError):
        PreferenceSession([], [], budget=WorkBudget(10))
    with pytest.raises(ValueError):
        PreferenceSession([Alternative('a',(1,0,0))],[(True,0,0)],budget=WorkBudget(10))
    s = session(budget=WorkBudget(2))
    with pytest.raises(BudgetStop):
        s.choose()
    assert s.budget.spent <= s.budget.limit
    with pytest.raises(BudgetStop):
        check_regret_certificate(session(),session().choose(),budget=WorkBudget(1))


def test_authority_persistence_rollback_and_nested_strict_certificate():
    from solvi.experimental.experience.preferences import PreferenceAuthority
    s = session()
    s.answer('x','a','b',source='author')
    checkpoint = s.snapshot()
    authority = PreferenceAuthority.from_trusted_state(s.authority.export_trusted_state(budget=s.budget), budget=s.budget)
    restored = PreferenceSession.restore(checkpoint,budget=s.budget,authority=authority)
    assert restored.choose() == s.choose()
    restored.retract('x',source='correction')
    with pytest.raises(ValueError,match='stale'):
        PreferenceSession.restore(checkpoint,budget=WorkBudget.restore(s.budget.snapshot()),authority=authority)
    forged = dict(checkpoint, answers=[])
    forged['digest'] = digest({k:v for k,v in forged.items() if k not in ('digest','signature')})
    with pytest.raises(ValueError,match='authentication'):
        PreferenceSession.restore(forged,budget=WorkBudget(10000),authority=authority)
    c = s.choose()
    assert not check_regret_certificate(s,replace(c,regret=False),budget=s.budget)
    assert not check_regret_certificate(s,replace(c,surviving_weights=(False,)),budget=s.budget)
    assert not check_regret_certificate(s,replace(c,regrets=(('a',False),)),budget=s.budget)


def test_public_answers_read_only_and_restore_no_budget_refund():
    s = session()
    s.answer('x','a','b',source='author')
    before = s.choose()
    with pytest.raises(AttributeError):
        s.answers.clear()
    detached = list(s.answers)
    detached.clear()
    assert s.choose() == before
    with pytest.raises(AttributeError):
        s.answers = []
    checkpoint = s.snapshot()
    for budget in (WorkBudget(s.budget.limit), WorkBudget(s.budget.limit+1)):
        with pytest.raises(ValueError,match='ledger refund|limit mismatch|identity'):
            PreferenceSession.restore(checkpoint,budget=budget,authority=s.authority)
    expected = s.choose()
    continued = s.budget
    restored = PreferenceSession.restore(checkpoint,budget=continued,authority=s.authority)
    assert restored.choose() == expected


def test_read_only_configuration_and_live_ledger_highwater():
    s = session()
    for name,value in [('requirements',()),('alternatives',()),('weights',()),('host_revision','evil'),('budget',WorkBudget(1)),('authority',None)]:
        with pytest.raises(AttributeError):
            setattr(s,name,value)
    checkpoint = s.snapshot()
    old = WorkBudget.restore(s.budget.snapshot())
    s.budget.charge('external-shared-work',500)
    with pytest.raises(ValueError,match='live preference ledger'):
        PreferenceSession.restore(checkpoint,budget=old,authority=s.authority)
    continued = s.budget
    restored = PreferenceSession.restore(checkpoint,budget=continued,authority=s.authority)
    assert restored.choose().status == 'ok'
    trusted = s.authority.export_trusted_state(budget=continued)
    from solvi.experimental.experience.preferences import PreferenceAuthority
    imported = PreferenceAuthority.from_trusted_state(trusted,budget=continued)
    checkpoint = restored.snapshot()
    old = WorkBudget.restore(continued.snapshot())
    continued.charge('more-shared-work',500)
    with pytest.raises(ValueError,match='live preference ledger'):
        PreferenceSession.restore(checkpoint,budget=old,authority=imported)


def test_live_restore_rejects_clone_and_transfers_session_epoch():
    s = session()
    snap = s.snapshot()
    clone = WorkBudget.restore(s.budget.snapshot())
    with pytest.raises(ValueError,match='identity'):
        PreferenceSession.restore(snap,budget=clone,authority=s.authority)
    expected = s.choose()
    restored = PreferenceSession.restore(snap,budget=s.budget,authority=s.authority)
    assert restored.choose() == expected
    for operation in (s.choose, s.snapshot, lambda:s.answer('x','a','b',source='author')):
        with pytest.raises(ValueError,match='stale'):
            operation()
