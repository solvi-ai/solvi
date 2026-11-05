import json
import math
import pytest
from solvi.stat import SourceOpinion, SourceTrust


class VisitMap(dict):
    visits = 0
    lookups = 0

    def __getitem__(self, key):
        self.lookups += 1
        return super().__getitem__(key)

    def items(self):
        for pair in super().items():
            self.visits += 1
            yield pair


def registry():
    trust = SourceTrust(window=10, min_labels=3)
    for source in range(20):
        for j in range(20):
            key = f'{source}:{j}'
            trust.assert_statement(key, statement='s', source=str(source),
                                   asserted_at=j % 10, now=20,
                                   parent=f'{source}:0' if j % 2 else None)
            if j % 3:
                trust.label(key, loss=(j % 5) / 4, now=20)
    return trust


def old_scan(trust, source, now, role):
    rows = [(k, v) for k, v in dict.items(trust._assertions)
            if v['source'] == source and
            (role == 'all' or (v['parent'] is None) == (role == 'origin'))]
    recent = [(k, v) for k, v in rows if now-trust.window < v['asserted_at'] <= now]
    losses = [trust._labels[k]['loss'] for k, _ in recent if k in trust._labels]
    n, total = len(losses), math.fsum(losses)
    last = max((v['asserted_at'] for _, v in rows), default=None)
    a, b = trust.prior
    scale = max(a, b, n, 1.)
    reliability = (a/scale+(n-total)/scale)/(a/scale+b/scale+n/scale)
    return SourceOpinion(source, role, now, 'supported' if n >= trust.min_labels else 'unknown',
                         reliability, n, len(recent)-n, len(rows)-len(recent),
                         total/n if n else None, last, now-last if last is not None else None)


def test_opinion_visits_only_requested_source_history():
    trust = registry()
    trust._assertions = VisitMap(trust._assertions)
    assert trust.opinion('7', now=20) == old_scan(trust, '7', 20, 'all')
    assert trust._assertions.visits == 0  # no full-registry items traversal
    assert trust._assertions.lookups == 20


def test_all_roles_windows_restore_and_late_registration_match_old_scan():
    trust = registry()
    trust.assert_statement('late', statement='s', source='7', asserted_at=19, now=21)
    trust.label('late', loss=.75, now=22)
    state = trust.snapshot()
    for obj in (trust, SourceTrust.restore(json.loads(json.dumps(state)))):
        for window in (1, 3, 10, 30):
            obj.window = window
            for now in (22, 29, 30, 40):
                for source in ('0', '7', '19', 'unknown'):
                    for role in ('all', 'origin', 'repeater'):
                        assert obj.opinion(source, now=now, role=role) == old_scan(obj, source, now, role)
    assert SourceTrust.restore(state).snapshot() == state


def test_invalid_assertions_leave_index_and_serialized_state_unchanged():
    trust = registry()
    before = trust.snapshot()
    for kwargs in (dict(assertion_id='0:0', source='other', asserted_at=1),
                   dict(assertion_id='bad', source='other', asserted_at=99),
                   dict(assertion_id='bad', source='other', asserted_at=1, parent='absent')):
        with pytest.raises(ValueError):
            trust.assert_statement(statement='s', now=20, **kwargs)
        assert trust.snapshot() == before
        assert trust.opinion('other', now=20).n == 0
        assert trust.opinion('other', now=20).last_asserted_at is None
