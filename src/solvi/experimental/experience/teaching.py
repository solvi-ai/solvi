"""Interactive finite program teaching with host-authenticated domain admission.

A committee is a proposal, never a proof. Verification covers only the separately
checked finite domain. The authority is a host capability, not serialized data.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import secrets

from ._common import WorkBudget, canonical, identifier, natural, portable
from .programs import Grammar, Program


def _bounded(values, limit, budget, kind):
    """Charge before requesting each item, including the bounded overflow probe."""
    it = iter(values)
    out = []
    for _ in range(limit + 1):
        budget.charge(kind)
        try:
            item = next(it)
        except StopIteration:
            return out
        if len(out) == limit:
            raise ValueError('finite collection exceeds bound')
        out.append(item)
    raise ValueError('finite collection exceeds bound')


@dataclass(frozen=True)
class ConceptRef:
    name: str
    version: str

    def __post_init__(self):
        identifier(self.name)
        identifier(self.version)


@dataclass(frozen=True)
class TeachingQuestion:
    revision: int
    inputs_json: str
    predictions: tuple

    @property
    def inputs(self):
        import json
        return json.loads(self.inputs_json)


@dataclass(frozen=True)
class TeachingResult:
    status: str
    value: int | bool | None = None
    reason: str = ''


def _require_ledger(current, floor):
    """A restore may continue credit, never mint it or move category debits."""
    if (type(current['limit']) is not int or current['limit'] != floor['limit'] or
            current['spent'] < floor['spent'] or any(
                current['counts'].get(kind, 0) < count for kind, count in floor['counts'].items())):
        raise ValueError('teaching ledger rewind or limit change')


class _TeachingLedger(WorkBudget):
    """Read-only ledger facade that anchors every debit, including failed sinks."""
    def __init__(self, source, authority, session):
        while isinstance(source, _TeachingLedger):
            source = source._source
        self._source, self._authority, self._session = source, authority, session

    @property
    def limit(self):
        return self._source.limit

    @property
    def spent(self):
        return self._source.spent

    @property
    def remaining(self):
        return self._source.remaining

    @property
    def counts(self):
        return self._source.counts

    def snapshot(self):
        current = self._source.snapshot()
        _require_ledger(current, self._authority._ledgers[self._session])
        return current

    def charge(self, kind, units=1):
        self.snapshot()
        try:
            self._source.charge(kind, units)
        finally:
            self._authority._ledgers[self._session] = self._source.snapshot()


class TeachingAuthority:
    """Host-owned authentication capability for saved admission evidence.

    Retain this object, including its monotonic epoch registry, across trusted
    restores. A fresh authority cannot validate another authority's snapshots.
    This authenticates provenance, not truthfulness of the host oracle.
    """

    def __init__(self, secret=None):
        if secret is not None and (type(secret) is not bytes or len(secret) < 32):
            raise ValueError('authority secret requires at least 32 bytes')
        self._secret = secrets.token_bytes(32) if secret is None else secret
        self._epochs = {}
        self._ledgers = {}
        self._sources = {}

    def export_trusted_state(self, *, budget):
        """Export a secret host checkpoint; never attach this to user snapshots."""
        budget.charge('teaching_authority_export', len(self._epochs) + 1)
        budget.charge('teaching_authority_export_ledgers', sum(len(v['counts']) + 1 for v in self._ledgers.values()))
        for session, source in self._sources.items():
            current = source.snapshot()
            _require_ledger(current, self._ledgers[session])
            self._ledgers[session] = current
        return portable(dict(v=1, secret=self._secret.hex(), epochs=dict(self._epochs), ledgers=self._ledgers))

    @classmethod
    def from_trusted_state(cls, state, *, budget):
        """Import explicit host-owned state, not data submitted by a teacher.

        The host must preserve the newest checkpoint atomically with snapshots.
        Restoring an older host checkpoint is outside this trust boundary.
        """
        supplied_ledger = budget.snapshot()
        budget.charge('teaching_authority_restore')
        if type(state) is not dict or set(state) != {'v', 'secret', 'epochs', 'ledgers'}:
            raise ValueError('invalid trusted authority state')
        if type(state['v']) is not int or state['v'] != 1:
            raise ValueError('invalid trusted authority version')
        secret, epochs = state['secret'], state['epochs']
        if type(secret) is not str or len(secret) < 64 or len(secret) > 4096:
            raise ValueError('invalid trusted secret')
        if type(epochs) is not dict or len(epochs) > 10000:
            raise ValueError('invalid trusted epochs')
        budget.charge('teaching_authority_restore', len(epochs) + len(secret))
        try:
            raw = bytes.fromhex(secret)
        except ValueError as exc:
            raise ValueError('invalid trusted secret encoding') from exc
        obj = cls(raw)
        for session, epoch in epochs.items():
            if (type(session) is not str or len(session) != 32 or
                    any(c not in '0123456789abcdef' for c in session)):
                raise ValueError('invalid trusted session')
            natural(epoch, 'trusted epoch')
        if type(state['ledgers']) is not dict or set(state['ledgers']) != set(epochs):
            raise ValueError('invalid trusted ledger anchors')
        anchors = {}
        for session, ledger in state['ledgers'].items():
            budget.charge('teaching_authority_ledger_restore')
            if type(ledger) is not dict or type(ledger.get('counts')) is not dict or len(ledger['counts']) > 10000:
                raise ValueError('invalid trusted ledger anchor')
            budget.charge('teaching_authority_ledger_restore', len(ledger['counts']))
            anchors[session] = WorkBudget.restore(ledger).snapshot()
        for anchor in anchors.values():
            _require_ledger(supplied_ledger, anchor)
        # Import work is paid on the continuing shared ledger, never a new fork.
        current_ledger = budget.snapshot()
        source = budget
        while isinstance(source, _TeachingLedger):
            source = source._source
        obj._epochs = dict(epochs)
        obj._ledgers = {session: portable(current_ledger) for session in anchors}
        obj._sources = {session: source for session in anchors}
        return obj

    def _sign(self, body):
        return hmac.new(self._secret, canonical(body).encode(), hashlib.sha256).hexdigest()


class TeachingSession:
    """Learn from scalar examples in a host supplied bounded program library.

    All calls share one WorkBudget; exhaustion raises BudgetStop. Failed
    admission never installs a concept. Versions are immutable, including after
    retraction; dependents become unusable transitively.
    """

    def __init__(self, grammar, hypotheses, query_domain, *, budget,
                 input_types=None, result_type='bool', max_items=256, authority=None):
        budget.charge('teaching_init')
        if not isinstance(grammar, Grammar) or not isinstance(budget, WorkBudget):
            raise ValueError('Grammar and WorkBudget required')
        natural(max_items, positive=True)
        if max_items > 10000:
            raise ValueError('max_items exceeds implementation bound')
        if result_type not in ('int', 'bool'):
            raise ValueError('invalid result type')
        self._grammar, self._budget, self._max_items = grammar, budget, max_items
        if input_types is not None and type(input_types) is not dict:
            raise ValueError('input_types must be a finite dict')
        self._input_types = dict.fromkeys(grammar.inputs, 'int') if input_types is None else dict(input_types)
        if set(self.input_types) != set(grammar.inputs) or any(t not in ('int', 'bool') for t in self.input_types.values()):
            raise ValueError('invalid input types')
        self._result_type = result_type
        self._authority = TeachingAuthority() if authority is None else authority
        if type(self.authority) is not TeachingAuthority:
            raise ValueError('host TeachingAuthority required')
        self._library = tuple(_bounded(hypotheses, max_items, budget, 'teaching_library'))
        if not self._library or any(type(p) is not Program for p in self._library):
            raise ValueError('nonempty Program library required')
        if len({p.fingerprint for p in self._library}) != len(self._library):
            raise ValueError('duplicate program')
        self._domain = tuple(self._point(p) for p in _bounded(query_domain, max_items, budget, 'teaching_domain'))
        if not self._domain or len({canonical(p) for p in self._domain}) != len(self._domain):
            raise ValueError('nonempty unique question domain required')
        if len(self.authority._epochs) >= 10000:
            raise ValueError('authority session limit exceeded')
        self._session = secrets.token_hex(16)
        self._epoch = 0
        self.authority._epochs[self._session] = 0
        self.authority._ledgers[self._session] = budget.snapshot()
        self._budget = _TeachingLedger(budget, self.authority, self._session)
        self.authority._sources[self._session] = self._budget._source
        self._examples = []
        self._candidates = self._library
        self._concepts = {}
        self._revision = 0

    @property
    def grammar(self):
        return self._grammar

    @property
    def budget(self):
        return self._budget

    @property
    def max_items(self):
        return self._max_items

    @property
    def input_types(self):
        return dict(self._input_types)

    @property
    def result_type(self):
        return self._result_type

    @property
    def authority(self):
        return self._authority

    def _change(self):
        if self.authority._epochs.get(self._session) != self._epoch:
            raise ValueError('stale teaching session')
        self._epoch += 1
        self.authority._epochs[self._session] = self._epoch

    def _point(self, point):
        self.budget.charge('teaching_input', len(self.grammar.inputs) + 1)
        if type(point) is not dict or set(point) != set(self.input_types):
            raise ValueError('input schema mismatch')
        for name, kind in self.input_types.items():
            value = point[name]
            if type(value) is not (int if kind == 'int' else bool) or (type(value) is int and abs(value) > self.grammar.max_abs):
                raise ValueError('input type or bound mismatch')
        return dict(point)

    def _value(self, value):
        if type(value) is not (int if self.result_type == 'int' else bool) or (type(value) is int and abs(value) > self.grammar.max_abs):
            raise ValueError('result type or bound mismatch')
        return value

    def _evaluate(self, program, inputs):
        return self._value(program.evaluate(inputs, self.budget, self.grammar))

    @property
    def candidates(self):
        return self._candidates

    def teach(self, inputs, value, *, exception=False):
        """Add an ordinary example or explicit exception; contradictions persist."""
        self.budget.charge('teaching_example')
        if type(exception) is not bool:
            raise ValueError('exception must be bool')
        inputs, value = self._point(inputs), self._value(value)
        if len(self._examples) >= self.max_items:
            raise ValueError('example limit exceeded')
        # Revision changes atomically invalidate all old proofs, even on exhaustion.
        self._change()
        self._revision += 1
        self._examples.append({'inputs': inputs, 'value': value, 'exception': exception})
        self._candidates = ()
        survivors = []
        for program in self._library:
            self.budget.charge('teaching_search')
            try:
                if all(canonical(self._evaluate(program, e['inputs'])) == canonical(e['value']) for e in self._examples):
                    survivors.append(program)
            except ValueError:
                continue
        self._candidates = tuple(survivors)
        return len(self._candidates)

    def question(self):
        """Return the largest committee split over the finite query domain."""
        self.budget.charge('teaching_question')
        best = None
        score = 0
        taught = {canonical(e['inputs']) for e in self._examples}
        for point in self._domain:
            self.budget.charge('teaching_query_point')
            if canonical(point) in taught:
                continue
            predictions = []
            for p in self._candidates:
                try:
                    predictions.append(self._evaluate(p, point))
                except ValueError:
                    predictions.append(None)
            groups = {}
            for v in predictions:
                self.budget.charge('teaching_query_vote')
                groups[canonical(v)] = groups.get(canonical(v), 0) + 1
            split = len(predictions) - max(groups.values(), default=0)
            if split > score:
                score = split
                best = TeachingQuestion(self._revision, canonical(point), tuple(predictions))
        return best

    def answer(self, question, value):
        self.budget.charge('teaching_answer')
        if type(question) is not TeachingQuestion or question.revision != self._revision:
            raise ValueError('stale or invalid question')
        # Questions are proposals only; fabricated points cannot grant admission.
        if canonical(question.inputs) not in {canonical(p) for p in self._domain}:
            raise ValueError('question outside domain')
        return self.teach(question.inputs, value)

    @staticmethod
    def _key(ref):
        if type(ref) is not ConceptRef:
            raise ValueError('ConceptRef required')
        return canonical([ref.name, ref.version])

    def _live(self, key):
        # Explicit stack avoids recursion failure for long dependency chains.
        pending, done = [(key, False)], set()
        visiting = set()
        while pending:
            self.budget.charge('teaching_dependency')
            current, leaving = pending.pop()
            if leaving:
                visiting.remove(current)
                done.add(current)
                continue
            if current in done:
                continue
            if current in visiting:
                return False
            r = self._concepts.get(current)
            if (self.authority._epochs.get(self._session) != self._epoch or not r or
                    not r['active'] or r['revision'] != self._revision):
                return False
            visiting.add(current)
            pending.append((current, True))
            pending.extend((dep, False) for dep in r['dependencies'])
        return True

    def _admit(self, ref, program, dependencies, domain, oracle, source_id, source_revision):
        self.budget.charge('teaching_admit')
        start_epoch, start_revision = self._epoch, self._revision
        key = self._key(ref)
        if key in self._concepts:
            raise ValueError('concept version already used')
        if len(self._concepts) >= self.max_items:
            raise ValueError('concept limit exceeded')
        identifier(source_id, 'oracle source')
        identifier(source_revision, 'oracle revision')
        if not callable(oracle):
            raise ValueError('host oracle required')
        rows = [self._point(p) for p in _bounded(domain, self.max_items, self.budget, 'teaching_admission_domain')]
        keys = [canonical(p) for p in rows]
        if not rows or len(set(keys)) != len(rows) or set(keys) & {canonical(e['inputs']) for e in self._examples}:
            raise ValueError('admission requires unique points separate from teaching examples')
        evidence = []
        for point in rows:
            self.budget.charge('teaching_oracle')
            expected = self._value(oracle(dict(point)))
            actual = self._evaluate(program, point)
            evidence.append({'inputs': point, 'value': expected})
            if canonical(expected) != canonical(actual):
                return None
        if self._epoch != start_epoch or self._revision != start_revision:
            raise ValueError('session changed during admission')
        if not all(self._live(dep) for dep in dependencies):
            return None
        record: dict = dict(ref=[ref.name, ref.version], program=program.to_dict(), dependencies=dependencies,
                      revision=self._revision, source_id=source_id, source_revision=source_revision,
                      evidence=evidence, active=True)
        self.budget.charge('teaching_receipt', len(evidence) + 1)
        record['signature'] = self.authority._sign({k: v for k, v in record.items() if k != 'active'})
        self._change()
        self._concepts[key] = record
        return ref

    def admit(self, name, version, domain, oracle, *, source_id, source_revision):
        """Independently check the unique remaining proposal; never auto-select."""
        self.budget.charge('teaching_selection')
        if len(self._candidates) != 1:
            return None
        return self._admit(ConceptRef(name, version), self._candidates[0], [], domain, oracle, source_id, source_revision)

    def compose(self, name, version, template, bindings, domain, oracle, *, source_id, source_revision):
        """Inline immutable accepted programs into template input terminals."""
        self.budget.charge('teaching_compose')
        if type(template) is not Program or type(bindings) is not dict or len(bindings) > self.max_items:
            raise ValueError('bounded template and bindings required')
        replacements, dependencies = {}, []
        for slot, ref in bindings.items():
            self.budget.charge('teaching_binding')
            identifier(slot)
            key = self._key(ref)
            if slot not in template.input_names or not self._live(key):
                raise ValueError('unavailable binding')
            replacements[slot] = Program.from_dict(self._concepts[key]['program']).ast
            dependencies.append(key)

        def inline(node):
            self.budget.charge('teaching_inline')
            if node[0] == 'input':
                return replacements.get(node[1], node)
            if node[0] == 'const':
                return node
            return (node[0], *(inline(c) for c in node[1:]))
        program = Program(inline(template.ast))
        return self._admit(ConceptRef(name, version), program, dependencies, domain, oracle, source_id, source_revision)

    def apply(self, ref, inputs):
        self.budget.charge('teaching_apply')
        key, point = self._key(ref), self._point(inputs)
        if not self._live(key):
            return TeachingResult('unknown', reason='unavailable or retracted concept')
        record = self._concepts[key]
        for e in record['evidence']:
            self.budget.charge('teaching_membership')
            if canonical(e['inputs']) == canonical(point):
                value = self._evaluate(Program.from_dict(record['program']), point)
                return TeachingResult('verified', value)
        return TeachingResult('unknown', reason='outside independently checked domain')

    def retract(self, ref):
        self.budget.charge('teaching_retract')
        key = self._key(ref)
        if key not in self._concepts:
            raise ValueError('unknown concept')
        self._change()
        self._concepts[key]['active'] = False

    def snapshot(self):
        self.budget.charge('teaching_snapshot', len(self._examples) + len(self._concepts) + 1)
        body = dict(v=1, session=self._session, epoch=self._epoch, grammar=self.grammar.to_dict(), library=[p.to_dict() for p in self._library],
                    query_domain=list(self._domain), input_types=self.input_types, result_type=self.result_type,
                    max_items=self.max_items, examples=self._examples, revision=self._revision, concepts=self._concepts)
        if self.authority._epochs.get(self._session) != self._epoch:
            raise ValueError('stale teaching session')
        self.budget.charge('teaching_snapshot_node', sum(p.nodes for p in self._library) +
                           sum(len(r['evidence']) + Program.from_dict(r['program']).nodes for r in self._concepts.values()))
        body['ledger'] = self.budget.snapshot()
        return portable(dict(body=body, signature=self.authority._sign(body)))

    @classmethod
    def restore(cls, snapshot, *, budget, authority):
        """Authenticated paid replay; does not call an oracle or trust plain hashes."""
        supplied_ledger = budget.snapshot()
        claimed_body = snapshot.get('body') if type(snapshot) is dict else None
        claimed_session = claimed_body.get('session') if type(claimed_body) is dict else None
        entry_anchor = authority._ledgers.get(claimed_session) if type(authority) is TeachingAuthority and type(claimed_session) is str else None
        entry_live = authority._sources[claimed_session].snapshot() if type(authority) is TeachingAuthority and type(claimed_session) is str and claimed_session in authority._sources else None
        budget.charge('teaching_restore')
        if type(authority) is not TeachingAuthority or type(snapshot) is not dict or set(snapshot) != {'body', 'signature'}:
            raise ValueError('invalid snapshot or missing host authority')
        body = snapshot['body']
        # Bound and debit untrusted snapshot traversal before hashing or copying.
        pending = [body]
        visited = 0
        while pending:
            budget.charge('teaching_restore_node')
            visited += 1
            if visited > 100000:
                raise ValueError('snapshot structural limit exceeded')
            item = pending.pop()
            if type(item) in (dict, list):
                if len(item) > 10000:
                    raise ValueError('snapshot collection limit exceeded')
                if len(pending) + len(item) > 100000:
                    raise ValueError('snapshot pending-node limit exceeded')
                budget.charge('teaching_restore_edges', len(item))
                pending.extend(item.values() if type(item) is dict else item)
            elif type(item) not in (str, int, bool, type(None)):
                raise ValueError('nonportable snapshot')
            elif type(item) is str and len(item) > 100000:
                raise ValueError('snapshot string limit exceeded')
        if type(body) is not dict or set(body) != {'v', 'session', 'epoch', 'grammar', 'library', 'query_domain', 'input_types', 'result_type', 'max_items', 'examples', 'revision', 'concepts', 'ledger'}:
            raise ValueError('invalid snapshot schema')
        if type(snapshot['signature']) is not str or not hmac.compare_digest(authority._sign(body), snapshot['signature']):
            raise ValueError('snapshot authentication failed')
        if type(body['epoch']) is not int or authority._epochs.get(body['session']) != body['epoch']:
            raise ValueError('stale snapshot or unknown authority session')
        live_source = budget
        while isinstance(live_source, _TeachingLedger):
            live_source = live_source._source
        if authority._sources.get(body['session']) is not live_source:
            raise ValueError('live teaching ledger identity required')
        recorded_ledger = WorkBudget.restore(body['ledger']).snapshot()
        _require_ledger(supplied_ledger, recorded_ledger)
        if entry_anchor is None:
            raise ValueError('missing trusted ledger anchor')
        _require_ledger(supplied_ledger, entry_anchor)
        if entry_live is not None:
            _require_ledger(supplied_ledger, entry_live)
        if type(body['v']) is not int or body['v'] != 1:
            raise ValueError('unsupported snapshot')
        try:
            replay_authority = TeachingAuthority(authority._secret)
            obj = cls(Grammar.from_dict(body['grammar']), [Program.from_dict(p) for p in body['library']], body['query_domain'],
                      budget=budget, authority=replay_authority, input_types=body['input_types'], result_type=body['result_type'], max_items=body['max_items'])
            for example in body['examples']:
                obj.teach(example['inputs'], example['value'], exception=example['exception'])
            if obj._revision != body['revision'] or len(body['concepts']) > obj.max_items:
                raise ValueError('invalid snapshot revision or concept count')
            for key, saved in body['concepts'].items():
                budget.charge('teaching_restore_record')
                r = portable(saved)
                if key != obj._key(ConceptRef(*r['ref'])) or type(r['active']) is not bool or type(r['revision']) is not int or r['revision'] > obj._revision:
                    raise ValueError('invalid saved concept')
                if r['signature'] != authority._sign({k: v for k, v in r.items() if k not in ('active', 'signature')}):
                    raise ValueError('invalid admission receipt')
                program = Program.from_dict(r['program'])
                rows = _bounded(r['evidence'], obj.max_items, budget, 'teaching_restore_evidence')
                if not rows:
                    raise ValueError('empty admission evidence')
                for e in rows:
                    if canonical(obj._evaluate(program, obj._point(e['inputs']))) != canonical(obj._value(e['value'])):
                        raise ValueError('inconsistent evidence')
                obj._concepts[key] = r
            if any(dep not in obj._concepts for r in obj._concepts.values() for dep in r['dependencies']):
                raise ValueError('missing dependency')
            obj._session, obj._epoch = body['session'], body['epoch']
            obj._authority = authority
            authority._ledgers[obj._session] = budget.snapshot()
            obj._budget = _TeachingLedger(budget, authority, obj._session)
            return obj
        finally:
            authority._ledgers[body['session']] = budget.snapshot()
            source = budget
            while isinstance(source, _TeachingLedger):
                source = source._source
            authority._sources[body['session']] = source


__all__ = ['ConceptRef', 'TeachingAuthority', 'TeachingQuestion', 'TeachingResult', 'TeachingSession']
