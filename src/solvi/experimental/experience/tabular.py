"""Three bounded CSV-backed SQL mechanisms; private truth never evaluates a Program."""
from contextlib import contextmanager
import csv
import io
import json
from itertools import product
from pathlib import Path
import sqlite3

from ._common import canonical, digest, identifier
from .interfaces import ProbeSpec, Receipt, TaskSpec
from .domain_checks import DomainReceipt, FiniteDomain

_RULES = {
    'inventory': (('stock', 'reserved', 'damaged'), 4, 'SELECT max(0, ? - ? - ?)'),
    'eligibility': (('requested', 'available', 'expired'), 4, 'SELECT (? < ?) AND NOT ?'),
    'schedule': (('a', 'b', 'c', 'd'), 3, 'SELECT max(? + ?, ?) + ?'),
}


def _point(family, point):
    if family not in _RULES:
        raise ValueError('unknown tabular family')
    names, bound, _ = _RULES[family]
    if type(point) is not dict or set(point) != set(names):
        raise ValueError('invalid point schema')
    for name in names:
        value = point[name]
        if name == 'expired':
            if type(value) is not bool:
                raise ValueError('expired must be bool')
        elif type(value) is not int or not 0 <= value <= bound:
            raise ValueError('point outside bounded integer domain')
    return tuple(point[n] for n in names)


def _truth(conn, family, point):
    result = conn.execute(_RULES[family][2], _point(family, point)).fetchone()[0]
    return bool(result) if family == 'eligibility' else result


class TabularOracle:
    """Read-only finite rule SQL adapter, independent of tasks and learner code."""
    def __init__(self, family, *, source_id=None):
        if family not in _RULES:
            raise ValueError('unknown tabular family')
        _point(family, {n: False if n == 'expired' else 0 for n in _RULES[family][0]})
        self.family = family
        self.source_id = identifier(source_id or 'sql:' + family, 'source_id')
        self.source_revision = 'rule:' + digest([family, list(_RULES[family][0]), _RULES[family][1], _RULES[family][2], 'v1'])

    def read(self, request_id, source_revision, point):
        identifier(request_id, 'request_id')
        if source_revision != self.source_revision:
            raise ValueError('oracle revision mismatch')
        with sqlite3.connect(':memory:') as conn:
            conn.execute('PRAGMA query_only=ON')
            expected = _truth(conn, self.family, point)
        return DomainReceipt(request_id, 'domain:' + digest([self.source_id, self.source_revision,
                             request_id, point]), self.source_id, self.source_revision, digest(point), expected)

    def domain(self, grammar, *, budget):
        names, bound, _ = _RULES[self.family]
        points = (dict(zip(names, row)) for row in product(*[
            (False, True) if n == 'expired' else range(bound + 1) for n in names]))
        return FiniteDomain(self.family, 'v1', self.family + '-v1', grammar,
                            self.source_id, self.source_revision, points, budget=budget,
                            input_types={n: 'bool' if n == 'expired' else 'int' for n in names},
                            result_type='bool' if self.family == 'eligibility' else 'int')


class TabularEnvironment:
    """Initialize from CSV task_id,family and family input columns (other cells empty).

    The CSV remains pinned on reopen and on each operation. Effects and receipts
    commit in one SQLite transaction; a successful task gets one effect marker.
    """
    idempotent_attempts = True

    def __init__(self, path, csv_path=None, *, identity='tabular', source_revision='v1'):
        self.path = Path(path).resolve()
        self.identity = identifier(identity, 'identity')
        self.source_revision = identifier(source_revision, 'source_revision')
        self.csv_path = None if csv_path is None else Path(csv_path).resolve()
        with self._transaction(True) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS tabular_meta (config TEXT NOT NULL)')
            rows = conn.execute('SELECT config FROM tabular_meta').fetchall()
            if not rows:
                if self.csv_path is None:
                    raise ValueError('CSV required for initialization')
                raw, cases = self._csv(self.csv_path)
                meta = dict(identity=self.identity, revision=self.source_revision,
                            csv_path=str(self.csv_path), csv_digest=digest(raw), cases_digest=digest(cases), version=1)
                conn.execute('INSERT INTO tabular_meta VALUES (?)', (canonical(meta),))
                conn.execute('CREATE TABLE tabular_tasks (task_id TEXT PRIMARY KEY, family TEXT NOT NULL, inputs TEXT NOT NULL, effects INTEGER NOT NULL DEFAULT 0)')
                conn.execute('CREATE TABLE tabular_receipts (attempt TEXT PRIMARY KEY, intent TEXT NOT NULL, receipt TEXT NOT NULL)')
                conn.executemany('INSERT INTO tabular_tasks(task_id,family,inputs) VALUES (?,?,?)',
                                 [(t, c['family'], canonical(c['inputs'])) for t, c in cases.items()])
            self._validate(conn)

    @staticmethod
    def _csv(path):
        raw = path.read_bytes().decode('utf-8')
        reader = csv.DictReader(io.StringIO(raw))
        allowed = {'task_id', 'family'} | {n for rule in _RULES.values() for n in rule[0]}
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames) or not set(reader.fieldnames) <= allowed or not {'task_id', 'family'} <= set(reader.fieldnames):
            raise ValueError('invalid CSV header')
        cases = {}
        for row in reader:
            t, family = identifier(row['task_id'], 'task_id'), row['family']
            if family not in _RULES or t in cases or None in row:
                raise ValueError('invalid CSV task')
            names = _RULES[family][0]
            if any(v for n, v in row.items() if n not in {'task_id', 'family', *names}):
                raise ValueError('unexpected CSV input')
            point = {}
            for n in names:
                v = row.get(n)
                if n == 'expired':
                    if v not in ('0', '1', 'false', 'true'):
                        raise ValueError('invalid CSV boolean')
                    point[n] = v in ('1', 'true')
                else:
                    if v is None or v not in tuple(str(i) for i in range(_RULES[family][1] + 1)):
                        raise ValueError('invalid CSV integer')
                    point[n] = int(v)
            _point(family, point)
            cases[t] = dict(family=family, inputs=point)
        if not cases:
            raise ValueError('empty CSV')
        return raw, cases

    @contextmanager
    def _transaction(self, write=False):
        conn = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        try:
            conn.execute('PRAGMA synchronous=FULL')
            conn.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _validate(self, conn):
        rows = conn.execute('SELECT config FROM tabular_meta').fetchall()
        if len(rows) != 1:
            raise ValueError('invalid metadata')
        meta = json.loads(rows[0][0])
        if meta['version'] != 1 or meta['identity'] != self.identity or meta['revision'] != self.source_revision:
            raise ValueError('identity or revision mismatch')
        stored_path = Path(meta['csv_path'])
        if self.csv_path is not None and self.csv_path != stored_path:
            raise ValueError('CSV path mismatch')
        raw, cases = self._csv(stored_path)
        actual = {t: dict(family=f, inputs=json.loads(p)) for t, f, p in conn.execute('SELECT task_id,family,inputs FROM tabular_tasks')}
        if (digest(raw) != meta['csv_digest'] or digest(cases) != meta['cases_digest']
                or canonical(actual) != canonical(cases)):
            raise ValueError('CSV or SQL inputs changed')
        return cases

    def task(self, task_id):
        with self._transaction() as conn:
            cases = self._validate(conn)
            if task_id not in cases:
                raise ValueError('unknown task')
            c = cases[task_id]
        family = c['family']
        answers = (False, True) if family == 'eligibility' else range(5 if family == 'inventory' else 10)
        return TaskSpec(task_id, family, 'v1', family + '-v1', c['inputs'],
                        tuple({'answer': a} for a in answers), (ProbeSpec('measure', 'answer', 8),))

    def execute(self, task_id, kind, payload, attempt_id):
        identifier(attempt_id, 'attempt_id')
        spec = self.task(task_id)
        if type(payload) is not dict:
            raise ValueError('invalid payload')
        if kind == 'observe':
            if payload != {'probe': 'measure', 'field': 'answer'}:
                raise ValueError('invalid probe')
        elif kind == 'act':
            expected_type = bool if spec.scope == 'eligibility' else int
            if set(payload) != {'answer'} or type(payload['answer']) is not expected_type:
                raise ValueError('invalid action scalar type')
        else:
            raise ValueError('invalid action kind')
        intent = canonical(dict(task_id=task_id, kind=kind, payload=payload))
        with self._transaction(True) as conn:
            self._validate(conn)
            old = conn.execute('SELECT intent,receipt FROM tabular_receipts WHERE attempt=?', (attempt_id,)).fetchone()
            if old:
                if old[0] != intent:
                    raise ValueError('attempt conflict')
                return Receipt.from_dict(json.loads(old[1]))
            answer = _truth(conn, spec.scope, spec.inputs)
            if kind == 'observe':
                value = dict(field='answer', value=answer)
            else:
                effects = conn.execute('SELECT effects FROM tabular_tasks WHERE task_id=?', (task_id,)).fetchone()[0]
                ok = payload['answer'] == answer and effects == 0
                if ok:
                    conn.execute('UPDATE tabular_tasks SET effects=1 WHERE task_id=?', (task_id,))
                value = dict(answer=answer, ok=ok, refused=not ok, state={'effects': int(effects or ok)})
            receipt = Receipt('event:' + digest([self.identity, attempt_id]), attempt_id, task_id,
                              kind, self.identity, 'v1', value)
            conn.execute('INSERT INTO tabular_receipts VALUES (?,?,?)', (attempt_id, intent, canonical(receipt.to_dict())))
            return receipt

    def reconcile(self, attempt_id):
        identifier(attempt_id, 'attempt_id')
        with self._transaction() as conn:
            self._validate(conn)
            row = conn.execute('SELECT receipt FROM tabular_receipts WHERE attempt=?', (attempt_id,)).fetchone()
            return None if row is None else Receipt.from_dict(json.loads(row[0]))

    @property
    def state(self):
        with self._transaction() as conn:
            self._validate(conn)
            return {t: {'effects': e} for t, e in conn.execute('SELECT task_id,effects FROM tabular_tasks')}


__all__ = ['TabularEnvironment', 'TabularOracle']
