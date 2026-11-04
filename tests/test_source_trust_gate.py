import pytest
from solvi.stat import SourceTrust
from solvi.core.knowledge import KnowledgeStore, SourceGate, TrustGate

BODY = {'s': 'pump', 'r': 'max_pressure', 'o': 10}


def registry():
    trust = SourceTrust(window=10)
    trust.assert_statement('a', statement='s', source='lab', asserted_at=0, now=0)
    trust.label('a', loss=0, now=1)
    return trust


def test_store_admission_refusal_identity_and_independent_source_gate():
    trust = registry()
    before = trust.fingerprint()
    gate = TrustGate(trust, clock=lambda: 2, min_reliability=.6)
    store = KnowledgeStore(gates=[gate], writers={'lab': 'spec', 'stranger': 'spec'})
    assert isinstance(store.gates[0], SourceGate)
    assert store.add('fact', BODY, source='spec', by='lab') is not None
    assert store.add('fact', BODY, source='spec', by='stranger') is None
    assert store.journal[-1]['op'] == 'refused'
    assert store.journal[-1]['why'].startswith('trust:')
    assert store.add('fact', BODY, source='system', by='lab') is None
    assert store.journal[-1]['why'].startswith('source:')
    assert store.add('fact', BODY, source='outcome', by='lab') is None
    assert store.journal[-1]['why'].startswith('writers:')
    verdict = gate.admit(store, {'by': 'lab'})
    assert verdict.admit and verdict.measured['n'] == 1
    assert verdict.measured['timestamp'] == 2
    assert verdict.measured['registry_fingerprint'] == before
    assert trust.fingerprint() == before


def test_expiry_delayed_label_and_role_do_not_transfer_trust():
    trust = registry()
    trust.assert_statement('repeat', statement='s', source='pub', asserted_at=9, now=9, parent='a')
    origin = TrustGate(trust, clock=lambda: 9, min_reliability=.6, role='origin')
    repeater = TrustGate(trust, clock=lambda: 9, min_reliability=.6, role='repeater')
    assert origin.admit(None, {'by': 'lab'}).admit
    assert not repeater.admit(None, {'by': 'pub'}).admit
    trust.label('repeat', loss=0, now=11)
    assert TrustGate(trust, clock=lambda: 11, min_reliability=.6, role='repeater').admit(None, {'by': 'pub'}).admit
    late = TrustGate(trust, clock=lambda: 11, min_reliability=.6, role='origin').admit(None, {'by': 'lab'})
    assert not late.admit and late.measured['expired'] == 1


def test_invalid_clock_refuses_and_mapping_is_explicit():
    trust = registry()
    before = trust.fingerprint()
    for now in (float('nan'), True, -1, 'today'):
        gate = TrustGate(trust, clock=lambda: now, min_reliability=.6)
        store = KnowledgeStore(gates=[gate])
        assert store.add('fact', BODY, source='spec', by='lab') is None
        assert store.items == {} and store.journal[-1]['op'] == 'refused'
        assert trust.fingerprint() == before
    gate = TrustGate(trust, clock=lambda: 2, min_reliability=.6, writer_sources={'writer': 'lab'})
    assert gate.admit(None, {'by': 'writer'}).admit
    assert not gate.admit(None, {'by': 'lab'}).admit
    assert not gate.admit(None, {'source': 'lab'}).admit
    with pytest.raises(ValueError):
        TrustGate(trust, clock=lambda: 2, min_reliability=float('nan'))


@pytest.mark.parametrize('kind', ['fact', 'rule', 'skill', 'action'])
@pytest.mark.parametrize('case', ['unknown', 'expired', 'unreliable'])
def test_trust_refuses_behaviour_changing_items_before_any_assertion(kind, case):
    trust = registry()
    source, now, threshold = 'lab', 2, .6
    if case == 'unknown':
        source = 'stranger'
    elif case == 'expired':
        now = 10
    else:
        threshold = .9
    gate = TrustGate(trust, clock=lambda: now, min_reliability=threshold)
    store = KnowledgeStore(gates=[gate])
    before = trust.fingerprint()
    assert store.add(kind, BODY, source='spec', by=source) is None
    assert store.items == {}
    assert len(store.journal) == 1
    assert store.journal[0]['op'] == 'refused'
    assert store.journal[0]['why'].startswith('trust:')
    assert trust.fingerprint() == before


def test_other_custom_gates_still_hold_behaviour_changes():
    from solvi.core.knowledge import Verdict

    class ReviewGate:
        name = 'review'

        def admit(self, store, item, shadow=None):
            return Verdict(False, 'needs review', gate=self.name)

    store = KnowledgeStore(gates=[ReviewGate()])
    item = store.add('rule', BODY, source='spec', by='lab')
    assert item is not None
    assert any(row['op'] == 'held' for row in store.journal)
    assert not any(row['op'] == 'refused' for row in store.journal)
