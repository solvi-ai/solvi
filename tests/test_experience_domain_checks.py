from dataclasses import FrozenInstanceError, replace

import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget, digest
from solvi.experimental.experience.domain_checks import DomainReceipt, FiniteDomain, check_program
from solvi.experimental.experience.programs import Grammar, Program


class Oracle:
    source_id = 'table'
    source_revision = 'r1'

    def __init__(self, function=lambda p: p['x'] - p['y']):
        self.function, self.calls = function, []

    def read(self, request_id, source_revision, point):
        self.calls.append(point)
        return DomainReceipt(request_id, f'e{len(self.calls)}', self.source_id,
                             source_revision, digest(point), self.function(point))


def fixture(points=None, **kwargs):
    grammar = Grammar(('x', 'y'), max_abs=10, max_depth=5)
    domain = FiniteDomain('s', 'm', 'schema', grammar, 'table', 'r1',
                          points if points is not None else [{'x': 0, 'y': 0}, {'x': 3, 'y': 0}],
                          budget=kwargs.pop('budget', WorkBudget(10000)), **kwargs)
    return grammar, domain


def check(grammar, domain, program=None, oracle=None, budget=None, **kwargs):
    return check_program(domain, program or Program(('sub', ('input', 'x'), ('input', 'y'))),
                         grammar, oracle or Oracle(), budget or WorkBudget(10000), request_prefix='test', **kwargs)


def test_full_coverage_and_historical_edge():
    grammar, domain = fixture()
    outcome = check(grammar, domain)
    assert outcome.status == 'verified' and outcome.checked_count == 2 and outcome.proof_id
    wrong = Program(('sub', ('sub', ('input', 'x'), ('input', 'y')), ('add', ('const', 1), ('const', 1))))
    outcome = check(grammar, domain, wrong)
    assert outcome.status == 'rejected' and outcome.reason == 'counterexample'
    assert outcome.proof_id is None


def test_manifest_immutable_paid_roundtrip_membership():
    grammar, domain = fixture()
    with pytest.raises(FrozenInstanceError):
        domain.source_id = 'other'
    point = domain.point_dict(0)
    point['x'] = 9
    assert domain.contains({'x': 0, 'y': 0}, WorkBudget(100))
    assert not domain.contains({'x': False, 'y': 0}, WorkBudget(100))
    snapshot = domain.snapshot(budget=WorkBudget(1000))
    assert FiniteDomain.from_dict(snapshot, grammar, budget=WorkBudget(1000)) == domain
    snapshot['source_revision'] = 'r2'
    with pytest.raises(ValueError):
        FiniteDomain.from_dict(snapshot, grammar, budget=WorkBudget(1000))
    with pytest.raises(BudgetStop):
        fixture(budget=WorkBudget(0))


@pytest.mark.parametrize('points', [[], [{'x': 0, 'y': 0}] * 2, [{'x': True, 'y': 0}], [{'x': 11, 'y': 0}]])
def test_invalid_manifests(points):
    with pytest.raises(ValueError):
        fixture(points)


def test_boolean_output_is_not_integer():
    grammar, domain = fixture()
    assert check(grammar, domain, oracle=Oracle(lambda p: False)).reason == 'invalid_receipt'
    assert check(grammar, domain, program=Program(('const', False))).reason == 'counterexample'


@pytest.mark.parametrize('field,value', [('request_id', 'wrong'), ('source_id', 'other'),
                                        ('source_revision', 'r2'), ('input_digest', 'wrong')])
def test_tampered_receipt(field, value):
    grammar, domain = fixture()
    oracle = Oracle()
    original = oracle.read
    oracle.read = lambda *args: replace(original(*args), **{field: value})
    assert check(grammar, domain, oracle=oracle).reason == 'invalid_receipt'


def test_duplicate_event_and_changing_source():
    grammar, domain = fixture()
    oracle = Oracle()
    original = oracle.read
    oracle.read = lambda *args: replace(original(*args), event_id='same')
    result = check(grammar, domain, oracle=oracle)
    assert result.status == 'unknown' and result.checked_count == 1
    oracle = Oracle()
    original = oracle.read

    def change(*args):
        receipt = original(*args)
        oracle.source_revision = 'r2'
        return receipt

    oracle.read = change
    assert check(grammar, domain, oracle=oracle).reason == 'source_changed'


def test_oracle_debit_precedes_intent_and_external_read():
    grammar, domain = fixture(oracle_cost=7)
    budget, oracle, order = WorkBudget(10000), Oracle(), []

    def event(kind, payload):
        order.append(kind)
        assert budget.counts['domain_oracle'] == 7 * (len(oracle.calls) + (kind == 'domain_read_intent'))

    assert check(grammar, domain, oracle=oracle, budget=budget, on_event=event).status == 'verified'
    assert order == ['domain_read_intent', 'domain_read_receipt'] * 2
    oracle = Oracle()
    assert check(grammar, domain, oracle=oracle, budget=WorkBudget(0)).status == 'unknown'
    assert not oracle.calls
    with pytest.raises(ValueError):
        fixture(oracle_cost=0)


@pytest.mark.parametrize('error', [ValueError('sink'), BudgetStop('sink')])
def test_callback_and_debit_failures_propagate(error):
    grammar, domain = fixture()

    def fail(*args):
        raise error

    with pytest.raises(type(error), match='sink'):
        check(grammar, domain, on_event=fail)
    with pytest.raises(type(error), match='sink'):
        check(grammar, domain, budget=WorkBudget(10000, on_charge=fail))


def test_invalid_program_no_read_and_oracle_exception_unknown():
    grammar, domain = fixture()
    oracle = Oracle()
    result = check(grammar, domain, Program(('input', 'undeclared')), oracle=oracle)
    assert result.reason == 'invalid_program' and not oracle.calls
    oracle.read = lambda *args: (_ for _ in ()).throw(ValueError('unavailable'))
    assert check(grammar, domain, oracle=oracle).reason == 'oracle_error'


def test_exact_rare_edge_and_partial_coverage():
    grammar, domain = fixture([{'x': 3, 'y': 0}])
    wrong = Program(('sub', ('sub', ('input', 'x'), ('input', 'y')), ('add', ('const', 1), ('const', 1))))
    outcome = check(grammar, domain, program=wrong)
    assert outcome.reason == 'counterexample'
    assert dict(outcome.witness)['actual'] == 1
    assert dict(outcome.witness)['expected'] == 3
    grammar, domain = fixture()
    outcome = check(grammar, domain, budget=WorkBudget(19))
    assert outcome.status == 'unknown' and outcome.checked_count == 1 and outcome.proof_id is None


@pytest.mark.parametrize('kind', ['domain_read_intent', 'domain_read_receipt'])
@pytest.mark.parametrize('error', [BudgetStop('late sink'), ValueError('late sink')])
def test_late_event_failures_are_not_oracle_errors(kind, error):
    grammar, domain = fixture()

    def callback(event, payload):
        if event == kind:
            raise error

    with pytest.raises(type(error), match='late sink'):
        check(grammar, domain, on_event=callback)
