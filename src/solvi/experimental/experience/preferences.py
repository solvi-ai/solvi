"""Bounded, single-author finite utility elicitation with exact certificates."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import combinations
from fractions import Fraction
import hashlib
import hmac
import secrets

from ._common import canonical

from ._common import WorkBudget, digest, identifier, natural, portable


def _integer(x):
    if type(x) is not int:
        raise ValueError('expected strict integer')
    return x


@dataclass(frozen=True)
class Alternative:
    id: str
    attributes: tuple[int, ...]

    def __post_init__(self):
        identifier(self.id)
        if type(self.attributes) is not tuple or not 3 <= len(self.attributes) <= 5:
            raise ValueError('attributes must be a tuple of 3–5 integers')
        for x in self.attributes:
            _integer(x)


@dataclass(frozen=True)
class Requirement:
    id: str
    attribute: int
    minimum: int
    hard: bool = True

    def __post_init__(self):
        identifier(self.id)
        natural(self.attribute)
        _integer(self.minimum)
        if type(self.hard) is not bool:
            raise ValueError('hard must be bool')


@dataclass(frozen=True)
class PreferenceAnswer:
    id: str
    preferred: str
    other: str
    source: str
    revision: int = 1
    active: bool = True
    relation: str = "strict"


@dataclass(frozen=True)
class RegretCertificate:
    status: str
    choice: str | None
    regret: int | None
    surviving_weights: tuple[int, ...]
    regrets: tuple[tuple[str, int], ...]
    fingerprint: str


@dataclass(frozen=True)
class PreferenceQuestion:
    left: str
    right: str
    reduction: Fraction
    cost: int
    net_value: Fraction
    criterion: str


def _supports(left, right, relation):
    return left > right if relation == "strict" else left >= right if relation == "weak" else left == right


class PreferenceAuthority:
    """Host capability; persist trusted state separately from untrusted snapshots."""
    def __init__(self):
        self._secret = secrets.token_bytes(32)
        self._epochs = {}
        self._budgets = {}

    def _sign(self, body):
        return hmac.new(self._secret, canonical(body).encode(), hashlib.sha256).hexdigest()

    def export_trusted_state(self, *, budget):
        budget.charge('preference.authority.export', 1+len(self._epochs))
        return {'v': 1, 'secret': self._secret.hex(), 'epochs': dict(self._epochs), 'ledgers': {k: b.snapshot() for k, b in self._budgets.items()}}

    @classmethod
    def from_trusted_state(cls, state, *, budget):
        initial_ledger = budget.snapshot()
        budget.charge('preference.authority.restore')
        if type(state) is not dict or set(state) != {'v', 'secret', 'epochs', 'ledgers'} or type(state['v']) is not int or state['v'] != 1 or type(state['secret']) is not str or len(state['secret']) != 64 or type(state['epochs']) is not dict:
            raise ValueError('invalid trusted authority state')
        try:
            secret = bytes.fromhex(state['secret'])
        except ValueError as exc:
            raise ValueError('invalid authority secret') from exc
        if len(secret) != 32 or len(state['epochs']) > 100000:
            raise ValueError('invalid trusted authority state')
        budget.charge('preference.authority.epochs', len(state['epochs']))
        for key, value in state['epochs'].items():
            identifier(key)
            natural(value)
        obj = cls()
        if type(state['ledgers']) is not dict or set(state['ledgers']) != set(state['epochs']):
            raise ValueError('invalid authority ledger registry')
        obj._secret, obj._epochs = secret, dict(state['epochs'])
        for key, value in state['ledgers'].items():
            budget.charge('preference.authority.ledger')
            anchor = WorkBudget.restore(value)
            if initial_ledger['limit'] != anchor.limit or initial_ledger['spent'] < anchor.spent or any(initial_ledger['counts'].get(k, 0) < v for k, v in anchor.counts.items()):
                raise ValueError('authority ledger refund or limit mismatch')
            obj._budgets[key] = budget
        return obj


class PreferenceSession:
    """At most 64 alternatives, 256 utilities, 16 requirements, 128 answer IDs.

    Answers explicitly assert strict, weak, or tie relations. Every exhaustive
    step is debited before execution; BudgetStop never returns a partial proof.
    """
    def __init__(self, alternatives, weights, *, requirements=(), budget, host_revision="v1", authority=None):
        if not isinstance(budget, WorkBudget):
            raise TypeError('shared WorkBudget required')
        identifier(host_revision, "host revision")
        self._host_revision = host_revision
        if type(alternatives) not in (tuple, list) or type(weights) not in (tuple, list) or type(requirements) not in (tuple, list):
            raise ValueError("bounded concrete sequences required")
        if len(alternatives) > 64 or len(weights) > 256 or len(requirements) > 16:
            raise ValueError("input exceeds limits")
        if any(type(w) not in (tuple, list) or not 3 <= len(w) <= 5 for w in weights):
            raise ValueError("invalid weight vectors")
        self._budget = budget
        self.budget.charge('preference.input')
        self._alternatives = tuple(alternatives)
        self._weights = tuple(tuple(w) for w in weights)
        self._requirements = tuple(requirements)
        if not 1 <= len(self.alternatives) <= 64 or not 1 <= len(self.weights) <= 256 or len(self.requirements) > 16:
            raise ValueError('finite domain exceeds limits')
        if any(type(a) is not Alternative for a in self.alternatives) or any(type(r) is not Requirement for r in self.requirements):
            raise ValueError('typed alternatives and requirements required')
        n = len(self.alternatives[0].attributes)
        if len({a.id for a in self.alternatives}) != len(self.alternatives) or len({r.id for r in self.requirements}) != len(self.requirements):
            raise ValueError('duplicate IDs')
        if any(len(a.attributes) != n for a in self.alternatives) or any(r.attribute >= n for r in self.requirements):
            raise ValueError('attribute dimension mismatch')
        for w in self.weights:
            if len(w) != n:
                raise ValueError('weight dimension mismatch')
            for x in w:
                _integer(x)
        if len(set(self.weights)) != len(self.weights):
            raise ValueError('duplicate utilities')
        self._authority = PreferenceAuthority() if authority is None else authority
        if type(self.authority) is not PreferenceAuthority:
            raise ValueError('host PreferenceAuthority required')
        self._session = secrets.token_hex(16)
        self._epoch = 0
        self.authority._epochs[self._session] = 0
        self.authority._budgets[self._session] = budget
        self._answers = []
        self._by_id = {a.id: a for a in self.alternatives}

    @property
    def host_revision(self):
        return self._host_revision

    @property
    def budget(self):
        return self._budget

    @property
    def alternatives(self):
        return self._alternatives

    @property
    def weights(self):
        return self._weights

    @property
    def requirements(self):
        return self._requirements

    @property
    def authority(self):
        return self._authority

    @property
    def answers(self):
        """Detached immutable event history; edits cannot bypass revision tracking."""
        return tuple(self._answers)

    def _assert_current(self):
        if self.authority._epochs.get(self._session) != self._epoch:
            raise ValueError('stale preference session')

    def _change(self):
        self._assert_current()
        self._epoch += 1
        self.authority._epochs[self._session] = self._epoch

    def _u(self, a, w):
        self.budget.charge('preference.utility')
        return sum(x*y for x, y in zip(a.attributes, w))

    def _active(self):
        self._assert_current()
        latest = {}
        for a in self._answers:
            self.budget.charge('preference.history')
            latest[a.id] = a
        return tuple(a for a in latest.values() if a.active)

    def answer(self, id, preferred, other, *, source, revision=None, relation="strict"):
        if relation not in ("strict", "weak", "tie"):
            raise ValueError("unknown answer relation")
        identifier(id)
        identifier(source)
        identifier(preferred)
        identifier(other)
        if preferred not in self._by_id or other not in self._by_id or preferred == other:
            raise ValueError('distinct known alternatives required')
        self.budget.charge('preference.history.scan', 1+len(self._answers))
        previous = [a for a in self._answers if a.id == id]
        rev = previous[-1].revision + 1 if previous else 1
        if revision is not None and (type(revision) is not int or revision != rev):
            raise ValueError('revision must be the next integer')
        if not previous and len({a.id for a in self._answers}) >= 128 or len(self._answers) >= 512:
            raise ValueError('answer history limit')
        self.budget.charge('preference.answer')
        a = PreferenceAnswer(id, preferred, other, source, rev, True, relation)
        self._change()
        self._answers.append(a)
        return a

    def retract(self, id, *, source):
        identifier(source)
        self.budget.charge('preference.history.scan', 1+len(self._answers))
        previous = [a for a in self._answers if a.id == id]
        if not previous or not previous[-1].active or len(self._answers) >= 512:
            raise ValueError('unknown/inactive answer or history limit')
        self.budget.charge('preference.answer')
        old = previous[-1]
        a = PreferenceAnswer(id, old.preferred, old.other, source, old.revision+1, False, old.relation)
        self._change()
        self._answers.append(a)
        return a

    def _surviving_weights(self, answers):
        return tuple(i for i, w in enumerate(self.weights) if all(_supports(self._u(self._by_id[a.preferred], w), self._u(self._by_id[a.other], w), a.relation) for a in answers))

    def _feasible(self, requirements):
        result = []
        for a in self.alternatives:
            self.budget.charge('preference.feasibility', max(1, len(requirements)))
            if all(a.attributes[r.attribute] >= r.minimum for r in requirements):
                result.append(a)
        return tuple(result)

    def _regrets(self, alternatives, indices):
        table = []
        for a in alternatives:
            worst = 0
            for i in indices:
                own = self._u(a, self.weights[i])
                for b in alternatives:
                    worst = max(worst, self._u(b, self.weights[i])-own)
            table.append((a.id, worst))
        return tuple(table)

    def _fingerprint(self):
        self.budget.charge('preference.fingerprint', 1+len(self._answers))
        return digest(self._body())

    def choose(self):
        self._assert_current()
        feasible = self._feasible(self.requirements)
        indices = self._surviving_weights(self._active())
        status = 'no_feasible_alternative' if not feasible else 'conflict' if not indices else 'ok'
        table = self._regrets(feasible, indices) if status == 'ok' else ()
        best = min(table, key=lambda p: (p[1], p[0])) if table else (None, None)
        return RegretCertificate(status, best[0], best[1], indices, table, self._fingerprint())

    def next_question(self, *, cost=0, criterion='worst_case', probabilities=None):
        natural(cost)
        if criterion not in ('worst_case', 'expected'):
            raise ValueError('unknown criterion')
        cert = self.choose()
        if cert.status != 'ok':
            return None
        indices = cert.surviving_weights
        if criterion == 'expected':
            if type(probabilities) is not tuple or len(probabilities) != len(self.weights) or any(type(p) is not int or p < 0 for p in probabilities) or not sum(probabilities[i] for i in indices):
                raise ValueError('declare nonnegative integer utility masses')
        feasible = self._feasible(self.requirements)
        best = None
        for a, b in combinations(feasible, 2):
            branches = []
            for sign in (-1, 0, 1):
                branch = tuple(i for i in indices if ((d := self._u(a, self.weights[i])-self._u(b, self.weights[i])) > 0)-(d < 0) == sign)
                if branch:
                    regret = min(v for _, v in self._regrets(feasible, branch))
                    mass = sum(probabilities[i] for i in branch) if criterion == 'expected' else 1
                    branches.append((regret, mass))
            after = Fraction(max(r for r, _ in branches)) if criterion == 'worst_case' else Fraction(sum(r*m for r, m in branches), sum(m for _, m in branches))
            reduction = cert.regret-after
            q = PreferenceQuestion(a.id, b.id, reduction, cost, reduction-cost, criterion)
            if q.net_value > 0 and (best is None or q.net_value > best.net_value):
                best = q
        return best

    def _minimal(self, items, predicate):
        found = []
        for n in range(1, len(items)+1):
            for subset in combinations(items, n):
                self.budget.charge('preference.subset')
                ids = frozenset(x.id for x in subset)
                if not any(set(s) <= ids for s in found) and predicate(subset):
                    found.append(tuple(x.id for x in subset))
        return tuple(found)

    def conflicts(self):
        answers = self._active()
        return {'answers': self._minimal(answers, lambda s: not self._surviving_weights(s)),
                'requirements': self._minimal(self.requirements, lambda s: not self._feasible(s))}

    def relaxations(self):
        self._assert_current()
        if self._feasible(self.requirements):
            return ((),)
        soft = tuple(r for r in self.requirements if not r.hard)
        return self._minimal(soft, lambda s: bool(self._feasible(tuple(r for r in self.requirements if r not in s))))

    def _body(self):
        return {'v': 1, 'host_revision': self.host_revision, 'alternatives': [{'id': a.id, 'attributes': list(a.attributes)} for a in self.alternatives],
                'weights': [list(w) for w in self.weights], 'requirements': [asdict(r) for r in self.requirements],
                'answers': [asdict(a) for a in self._answers]}

    def snapshot(self):
        self.budget.charge('preference.snapshot', 1+len(self._answers))
        self._assert_current()
        body = {**self._body(), 'session': self._session, 'epoch': self._epoch, 'ledger': self.budget.snapshot()}
        return {**body, 'digest': digest(body), 'signature': self.authority._sign(body)}

    @classmethod
    def restore(cls, snapshot, *, budget, host_revision="v1", authority=None):
        initial_ledger = budget.snapshot()
        session_key = snapshot.get('session') if type(snapshot) is dict else None
        live_budget = authority._budgets.get(session_key) if type(authority) is PreferenceAuthority and type(session_key) is str else None
        live_ledger = live_budget.snapshot() if live_budget is not None else None
        budget.charge('preference.restore')
        if type(authority) is not PreferenceAuthority:
            raise ValueError('host PreferenceAuthority required')
        if type(snapshot) is not dict:
            raise ValueError('snapshot must be a dict')
        for key, limit in (('alternatives', 64), ('weights', 256), ('requirements', 16), ('answers', 512)):
            if type(snapshot.get(key)) is not list or len(snapshot[key]) > limit:
                raise ValueError('snapshot exceeds bounds')
        budget.charge('preference.restore.records', sum(len(snapshot[k]) for k in ('alternatives', 'weights', 'requirements', 'answers')))
        pending = [snapshot]
        visited = 0
        while pending:
            budget.charge('preference.restore.node')
            visited += 1
            if visited > 20000:
                raise ValueError('snapshot structure limit')
            item = pending.pop()
            if type(item) in (dict, list):
                if len(item) > 512 or len(pending)+len(item) > 20000:
                    raise ValueError('snapshot collection limit')
                budget.charge('preference.restore.edge', len(item))
                pending.extend(item.values() if type(item) is dict else item)
            elif type(item) not in (str, int, bool, type(None)):
                raise ValueError('invalid snapshot value')
            elif type(item) is str and len(item) > 10000:
                raise ValueError('snapshot string limit')
        s = portable(snapshot)
        if type(s) is not dict or set(s) != {'v', 'host_revision', 'alternatives', 'weights', 'requirements', 'answers', 'session', 'epoch', 'digest', 'signature', 'ledger'} or type(s['v']) is not int or s['v'] != 1:
            raise ValueError('invalid preference snapshot schema')
        if s['host_revision'] != host_revision:
            raise ValueError('host revision mismatch')
        body = {k: v for k, v in s.items() if k not in ('digest', 'signature')}
        if type(s['signature']) is not str or not hmac.compare_digest(authority._sign(body), s['signature']):
            raise ValueError('snapshot authentication failed')
        if type(s['epoch']) is not int or authority._epochs.get(s['session']) != s['epoch']:
            raise ValueError('stale preference snapshot')
        if live_budget is not budget:
            raise ValueError('live preference ledger identity required')
        if live_ledger is None or initial_ledger['limit'] != live_ledger['limit'] or initial_ledger['spent'] < live_ledger['spent'] or any(initial_ledger['counts'].get(k,0) < v for k,v in live_ledger['counts'].items()):
            raise ValueError('live preference ledger refund or limit mismatch')
        recorded_ledger = WorkBudget.restore(s['ledger'])
        if initial_ledger['limit'] != recorded_ledger.limit or initial_ledger['spent'] < recorded_ledger.spent or any(initial_ledger['counts'].get(k, 0) < v for k, v in recorded_ledger.counts.items()):
            raise ValueError('preference ledger refund or limit mismatch')
        if digest(body) != s['digest']:
            raise ValueError('snapshot digest mismatch')
        if any(type(s[k]) is not list for k in ('alternatives', 'weights', 'requirements', 'answers')) or len(s['answers']) > 512:
            raise ValueError('invalid snapshot lists')
        for a in s['alternatives']:
            if type(a) is not dict or set(a) != {'id', 'attributes'} or type(a['attributes']) is not list or not 3 <= len(a['attributes']) <= 5:
                raise ValueError('alternative schema')
        for r in s['requirements']:
            if type(r) is not dict or set(r) != {'id', 'attribute', 'minimum', 'hard'}:
                raise ValueError('requirement schema')
        obj = cls([Alternative(a['id'], tuple(a['attributes'])) for a in s['alternatives']], s['weights'], requirements=[Requirement(**r) for r in s['requirements']], budget=budget, host_revision=host_revision, authority=PreferenceAuthority())
        for a in s['answers']:
            if type(a) is not dict or set(a) != {'id', 'preferred', 'other', 'source', 'revision', 'active', 'relation'} or type(a['active']) is not bool:
                raise ValueError('answer schema')
            if a['active']:
                obj.answer(a['id'], a['preferred'], a['other'], source=a['source'], revision=a['revision'], relation=a['relation'])
            else:
                event = obj.retract(a['id'], source=a['source'])
                if asdict(event) != a:
                    raise ValueError('invalid retraction')
        if authority._epochs.get(s['session']) != s['epoch'] or authority._budgets.get(s['session']) is not live_budget:
            raise ValueError('session changed during preference restore')
        obj._authority = authority
        obj._session, obj._epoch = s['session'], s['epoch']+1
        authority._epochs[obj._session] = obj._epoch
        authority._budgets[obj._session] = budget
        return obj


def check_regret_certificate(session, certificate, *, budget):
    """Independent arithmetic enumeration; never calls the producer's solver."""
    if type(certificate) is not RegretCertificate:
        return False
    if type(certificate.status) is not str or type(certificate.fingerprint) is not str or (certificate.choice is not None and type(certificate.choice) is not str) or (certificate.regret is not None and type(certificate.regret) is not int):
        return False
    if type(certificate.surviving_weights) is not tuple or any(type(i) is not int for i in certificate.surviving_weights) or type(certificate.regrets) is not tuple:
        return False
    if any(type(row) is not tuple or len(row) != 2 or type(row[0]) is not str or type(row[1]) is not int for row in certificate.regrets):
        return False
    session._assert_current()
    if type(certificate) is not RegretCertificate:
        return False
    budget.charge('preference.check', 1+len(session.answers))
    active = {}
    for a in session.answers:
        budget.charge('preference.check.history')
        active[a.id] = a
    feasible = []
    for a in session.alternatives:
        budget.charge('preference.check.feasibility', max(1, len(session.requirements)))
        if all(a.attributes[r.attribute] >= r.minimum for r in session.requirements):
            feasible.append(a)
    utilities = {}
    for i, w in enumerate(session.weights):
        for a in session.alternatives:
            budget.charge('preference.check.utility')
            utilities[i, a.id] = sum(w[j]*a.attributes[j] for j in range(len(w)))
    indices = tuple(i for i in range(len(session.weights)) if all(_supports(utilities[i, a.preferred], utilities[i, a.other], a.relation) for a in active.values() if a.active))
    status = 'no_feasible_alternative' if not feasible else 'conflict' if not indices else 'ok'
    budget.charge('preference.check.regret', len(feasible)*len(feasible)*len(indices))
    table = tuple((a.id, max(utilities[i, b.id]-utilities[i, a.id] for i in indices for b in feasible)) for a in feasible) if status == 'ok' else ()
    choice, regret = min(table, key=lambda p: (p[1], p[0])) if table else (None, None)
    return certificate == RegretCertificate(status, choice, regret, indices, table, digest(session._body()))


__all__ = ['PreferenceAuthority', 'Alternative', 'Requirement', 'PreferenceAnswer', 'RegretCertificate', 'PreferenceQuestion', 'PreferenceSession', 'check_regret_certificate']
