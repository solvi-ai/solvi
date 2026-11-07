"""Adversarial regression gates discovered by an independent review."""
import copy
import pytest

from solvi.experimental.experience._common import WorkBudget, digest
from solvi.experimental.experience.cooperation import FiniteProblem, CooperativeSearch, Evidence


def problem(state=None):
    state = state if state is not None else {'allowed': False, 'cost': 1}
    return FiniteProblem({'a': {'x': 0}}, {'allowed': lambda a: state['allowed']},
                         lambda a: state['cost'], revision='r1', budget=WorkBudget(1000))


def test_review_mutable_constraint_cannot_keep_verified_exclusion():
    state = {'allowed': False, 'cost': 1}
    s = CooperativeSearch(problem(state), WorkBudget(100))
    s.submit(Evidence('e', 'host', 'r1', 'exclusion', ('a',), 'allowed'))
    state['allowed'] = True
    # Either invalidation/revalidation or explicit refusal is required.
    assert s.residual().candidates == ()
    assert s.certify()['status'] == 'infeasible'


def test_review_mutable_objective_cannot_keep_stale_incumbent():
    state = {'allowed': True, 'cost': 1}
    s = CooperativeSearch(problem(state), WorkBudget(100))
    s.submit(Evidence('i', 'host', 'r1', 'incumbent', ('a',), 1))
    state['cost'] = 99
    assert s.residual().upper_bound == 1
    assert s.certify()['solution'] == (1, 'a')


def test_review_mutable_constraint_cannot_keep_infeasible_incumbent():
    state = {'allowed': True, 'cost': 1}
    s = CooperativeSearch(problem(state), WorkBudget(100))
    s.submit(Evidence('i', 'host', 'r1', 'incumbent', ('a',), 1))
    state['allowed'] = False
    assert s.residual().incumbent == 'a'
    assert s.certify()['solution'] == (1, 'a')


def test_review_mutable_objective_cannot_keep_false_lower_bound():
    state = {'allowed': True, 'cost': 10}
    s = CooperativeSearch(problem(state), WorkBudget(100))
    s.submit(Evidence('l', 'host', 'r1', 'lower_bound', value=10))
    state['cost'] = 0
    assert s.residual().lower_bound == 10
    assert s.certify()['solution'] == (10, 'a')


def test_review_forged_snapshot_cannot_mint_work_budget():
    s = CooperativeSearch(problem(), WorkBudget(100))
    s.submit(Evidence('e', 'host', 'r1', 'exclusion', ('a',), 'allowed'))
    forged = copy.deepcopy(s.snapshot())
    ledger = {'v': 1, 'limit': 1000000, 'spent': 0, 'counts': {}}
    forged['budget'] = {**ledger, 'digest': digest(ledger)}
    forged['digest'] = digest({k:v for k,v in forged.items() if k != 'digest'})
    with pytest.raises((ValueError, TypeError)):
        CooperativeSearch.restore(forged, problem(), s.budget, authority=s.authority)


def test_review_empty_certificate_requires_paid_enumeration():
    s = CooperativeSearch(FiniteProblem({}, {}, lambda a: 0, revision='r1', budget=WorkBudget(1000)), WorkBudget(0))
    assert s.certify()['status'] == 'unknown'


def test_review_on_charge_failure_is_atomic_for_evidence():
    class SinkFailure(Exception):
        pass
    def sink(ledger):
        raise SinkFailure()
    s = CooperativeSearch(problem(), WorkBudget(100, on_charge=sink))
    with pytest.raises(SinkFailure):
        s.submit(Evidence('e', 'host', 'r1', 'exclusion', ('a',), 'allowed'))
    assert s.budget.spent == 2
    s.budget.on_charge = None
    assert s.residual().candidates == ('a',)


def test_review_changed_domain_same_revision_snapshot_rejected():
    s = CooperativeSearch(problem(), WorkBudget(100))
    p = FiniteProblem({'different': {'x': 0}}, {}, lambda a: 0, revision='r1', budget=WorkBudget(1000))
    with pytest.raises(ValueError):
        CooperativeSearch.restore(s.snapshot(), p, s.budget, authority=s.authority)


def test_review_retraction_remains_revoked_after_replay():
    s = CooperativeSearch(problem(), WorkBudget(100))
    s.submit(Evidence('e', 'host', 'r1', 'exclusion', ('a',), 'allowed'))
    s.revoke('e')
    restored = CooperativeSearch.restore(s.snapshot(), problem(), s.budget, authority=s.authority)
    assert restored.residual().candidates == ('a',)
    with pytest.raises(ValueError):
        restored.submit(Evidence('e', 'host', 'r1', 'exclusion', ('a',), 'allowed'))


def test_review_infinite_solver_iterator_stops_under_budget():
    def solvers():
        while True:
            yield lambda residual, budget: None
    s = CooperativeSearch(problem(), WorkBudget(3))
    assert s.run(solvers())['reason'] == 'budget'
    assert s.budget.spent <= 3


def test_review_evidence_iterator_rejected_before_materialization():
    def unsafe_targets():
        # A real endless iterator hangs at list(self.targets), before a second debit.
        raise AssertionError('unbounded iterator was consumed')
        yield 'a'
    s = CooperativeSearch(problem(), WorkBudget(2))
    with pytest.raises((ValueError, TypeError)):
        s.submit(Evidence('e', 'host', 'r1', 'exclusion', unsafe_targets(), 'allowed'))


def test_review_assignment_iterator_rejected_before_materialization():
    def unsafe_assignments():
        raise AssertionError('unbounded iterator was consumed')
        yield ('a', {'x': 0})
    with pytest.raises((ValueError, TypeError)):
        FiniteProblem(unsafe_assignments(), {}, lambda a: 0, revision='r1', budget=WorkBudget(1000))
