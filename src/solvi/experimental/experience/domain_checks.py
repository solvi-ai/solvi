"""Paid exhaustive checks of a host-declared finite domain, never activation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ._common import BudgetStop, canonical, digest, identifier, natural
from .programs import Grammar


class _ChargeFailure(Exception):
    def __init__(self, error):
        self.error = error


class _Paid:
    """Separate insufficient credit from failures of a post-debit durable sink."""
    def __init__(self, budget):
        self.budget = budget

    def charge(self, kind, units=1):
        before = self.budget.spent
        try:
            self.budget.charge(kind, units)
        except Exception as error:
            if self.budget.spent != before or not isinstance(error, BudgetStop):
                raise _ChargeFailure(error) from error
            raise


def _typed(value, kind, maximum):
    return type(value) is (int if kind == 'int' else bool) and (kind != 'int' or abs(value) <= maximum)


@dataclass(frozen=True, init=False)
class FiniteDomain:
    scope: str
    mechanism: str
    schema: str
    grammar_fingerprint: str
    source_id: str
    source_revision: str
    input_names: tuple
    input_types: tuple
    result_type: str
    oracle_cost: int
    max_points: int
    max_abs: int
    points: tuple
    fingerprint: str

    def __init__(self, scope, mechanism, schema, grammar, source_id, source_revision, points, *,
                 budget, input_types=None, result_type='int', oracle_cost=8, max_points=10000):
        budget.charge('domain_manifest')
        for name, value in [('scope', scope), ('mechanism', mechanism), ('schema', schema),
                            ('source_id', source_id), ('source_revision', source_revision)]:
            identifier(value, name)
        if not isinstance(grammar, Grammar):
            raise ValueError('domain requires Grammar')
        natural(oracle_cost, 'oracle_cost', positive=True)
        natural(max_points, 'max_points', positive=True)
        if result_type not in ('int', 'bool'):
            raise ValueError('invalid result type')
        types = {} if input_types is None else input_types
        budget.charge('domain_manifest', len(grammar.inputs) + 1)
        if input_types is None:
            types = {name: 'int' for name in grammar.inputs}
        if type(types) is not dict or set(types) != set(grammar.inputs) or any(
            type(t) is not str or t not in ('int', 'bool') for t in types.values()
        ):
            raise ValueError('invalid domain input types')
        rows, seen = [], set()
        for point in points:
            budget.charge('domain_manifest', len(grammar.inputs) + 1)
            if len(rows) >= max_points:
                raise ValueError('domain exceeds max_points')
            if type(point) is not dict or set(point) != set(grammar.inputs):
                raise ValueError('invalid domain point schema')
            row = tuple(point[name] for name in grammar.inputs)
            if any(not _typed(v, types[n], grammar.max_abs) for n, v in zip(grammar.inputs, row)):
                raise ValueError('invalid domain point type or bound')
            key = canonical(list(row))
            if key in seen:
                raise ValueError('duplicate domain point')
            seen.add(key)
            rows.append((key, row))
        if not rows:
            raise ValueError('empty domain')
        budget.charge('domain_manifest', len(rows) * (len(rows).bit_length() + 1))
        values = dict(scope=scope, mechanism=mechanism, schema=schema,
                      grammar_fingerprint=grammar.fingerprint, source_id=source_id,
                      source_revision=source_revision, input_names=grammar.inputs,
                      input_types=tuple(types[n] for n in grammar.inputs), result_type=result_type,
                      oracle_cost=oracle_cost, max_points=max_points, max_abs=grammar.max_abs,
                      points=tuple(row for _, row in sorted(rows)))
        for key, value in values.items():
            object.__setattr__(self, key, value)
        budget.charge('domain_manifest', len(rows) * (len(grammar.inputs) + 1))
        object.__setattr__(self, 'fingerprint', digest(self._body()))

    def point_dict(self, index):
        return dict(zip(self.input_names, self.points[index]))

    def _body(self):
        return dict(v=1, scope=self.scope, mechanism=self.mechanism, schema=self.schema,
                    grammar_fingerprint=self.grammar_fingerprint, source_id=self.source_id,
                    source_revision=self.source_revision, input_types=dict(zip(self.input_names, self.input_types)),
                    result_type=self.result_type, oracle_cost=self.oracle_cost, max_points=self.max_points,
                    points=[self.point_dict(i) for i in range(len(self.points))])

    def to_dict(self, *, budget):
        budget.charge('domain_manifest', len(self.points) * (len(self.input_names) + 1) + 1)
        return {**self._body(), 'fingerprint': self.fingerprint}

    snapshot = to_dict

    @classmethod
    def from_dict(cls, value, grammar, *, budget):
        budget.charge('domain_manifest')
        keys = {'v', 'scope', 'mechanism', 'schema', 'grammar_fingerprint', 'source_id', 'source_revision',
                'input_types', 'result_type', 'oracle_cost', 'max_points', 'points', 'fingerprint'}
        if type(value) is not dict or set(value) != keys or type(value['v']) is not int or value['v'] != 1:
            raise ValueError('invalid domain snapshot')
        args = {key: value[key] for key in keys - {'v', 'fingerprint', 'grammar_fingerprint'}}
        domain = cls(grammar=grammar, budget=budget, **args)
        if domain.fingerprint != value['fingerprint'] or grammar.fingerprint != value['grammar_fingerprint']:
            raise ValueError('domain fingerprint mismatch')
        return domain

    def contains(self, inputs, budget):
        budget.charge('domain_membership', len(self.input_names) + 1)
        if type(inputs) is not dict or set(inputs) != set(self.input_names):
            return False
        row = tuple(inputs[n] for n in self.input_names)
        if any(not _typed(v, t, self.max_abs) for v, t in zip(row, self.input_types)):
            return False
        key = canonical(list(row))
        for point in self.points:
            budget.charge('domain_membership', len(self.input_names) + 1)
            if key == canonical(list(point)):
                return True
        return False


@dataclass(frozen=True)
class DomainReceipt:
    request_id: str
    event_id: str
    source_id: str
    source_revision: str
    input_digest: str
    expected: int | bool

    def to_dict(self):
        return dict(self.__dict__)


class DomainOracle(Protocol):
    source_id: str
    source_revision: str

    def read(self, request_id: str, source_revision: str, point: dict) -> DomainReceipt: ...


@dataclass(frozen=True)
class DomainCheckResult:
    status: str
    reason: str
    program_fingerprint: str
    grammar_fingerprint: str
    domain_fingerprint: str
    checked_count: int
    receipts: tuple = ()
    actuals: tuple = ()
    witness: tuple | None = None
    proof_id: str | None = None

    def to_dict(self):
        return {**self.__dict__, 'receipts': [r.to_dict() for r in self.receipts],
                'actuals': list(self.actuals),
                'witness': None if self.witness is None else dict(self.witness)}


def check_program(domain, program, grammar, oracle, budget, *, request_prefix, on_event=None):
    """An exhaustive finite-list verdict; a DTO cannot authorize registration."""
    paid, receipts, actuals, evidence, events = _Paid(budget), [], [], [], set()
    count = 0

    def result(status, reason, witness=None, proof=None):
        return DomainCheckResult(status, reason, program.fingerprint, grammar.fingerprint,
                                 domain.fingerprint, count, tuple(receipts), tuple(actuals), witness, proof)

    def bound():
        return (getattr(oracle, 'source_id', None) == domain.source_id and
                getattr(oracle, 'source_revision', None) == domain.source_revision and
                getattr(oracle, 'available', True) is True)

    def emit(kind, value):
        if on_event is not None:
            try:
                on_event(kind, value)
            except Exception as error:
                raise _ChargeFailure(error) from error

    try:
        paid.charge('domain_check')
        identifier(request_prefix, 'request_prefix')
        if grammar.fingerprint != domain.grammar_fingerprint:
            return result('unknown', 'grammar_changed')
        for index in range(len(domain.points)):
            paid.charge('domain_check', len(domain.input_names) + 1)
            if not bound():
                return result('unknown', 'source_changed')
            point = domain.point_dict(index)
            request_id = f'{request_prefix}:{index}'
            input_digest = digest(point)
            try:
                actual = program.evaluate(point, paid, grammar)
            except ValueError as error:
                return result('rejected', 'invalid_program', (('index', index), ('error', str(error))))
            paid.charge('domain_oracle', domain.oracle_cost)
            emit('domain_read_intent', dict(request_id=request_id, source_id=domain.source_id,
                                          source_revision=domain.source_revision, input_digest=input_digest,
                                          point=point.copy()))
            try:
                receipt = oracle.read(request_id, domain.source_revision, point.copy())
            except Exception:  # noqa: BLE001 - an unavailable host oracle cannot certify a program
                return result('unknown', 'oracle_error')
            paid.charge('domain_receipt')
            if not bound():
                return result('unknown', 'source_changed')
            if (type(receipt) is not DomainReceipt or receipt.request_id != request_id or
                receipt.source_id != domain.source_id or receipt.source_revision != domain.source_revision or
                receipt.input_digest != input_digest or type(receipt.event_id) is not str or
                not receipt.event_id or receipt.event_id in events or
                not _typed(receipt.expected, domain.result_type, grammar.max_abs)):
                return result('unknown', 'invalid_receipt')
            events.add(receipt.event_id)
            receipts.append(receipt)
            actuals.append(actual)
            emit('domain_read_receipt', receipt.to_dict())
            paid.charge('domain_compare')
            if type(actual) is not type(receipt.expected) or actual != receipt.expected:
                return result('rejected', 'counterexample', (('index', index), ('actual', actual),
                                                            ('expected', receipt.expected),
                                                            ('request_id', request_id)))
            count += 1
            evidence.append([input_digest, actual, receipt.to_dict()])
        paid.charge('domain_proof', len(evidence) + 1)
        if not bound():
            return result('unknown', 'source_changed')
        proof = digest(dict(program=program.fingerprint, grammar=grammar.fingerprint,
                            domain=domain.fingerprint, evidence=evidence))
        return result('verified', 'complete_finite_domain', proof=proof)
    except _ChargeFailure as failure:
        raise failure.error from failure
    except BudgetStop:
        return result('unknown', 'budget')
