import hashlib
import json
from unittest.mock import patch
import pytest
from solvi.stat import SourceTrust
from solvi.core.knowledge import TrustGate


def old_hash(trust):
    return hashlib.sha256(json.dumps(trust.snapshot(), sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def registry():
    trust = SourceTrust(window=10)
    trust.assert_statement('a', statement='s', source='lab', asserted_at=0, now=0)
    trust.label('a', loss=0, now=1)
    return trust


def test_twenty_native_admissions_serialize_history_once():
    trust = registry()
    gate = TrustGate(trust, clock=lambda: 2, min_reliability=.6)
    expected = old_hash(trust)
    with patch.object(trust, 'snapshot', wraps=trust.snapshot) as snapshot:
        for _ in range(20):
            verdict = gate.admit(None, {'by': 'lab'})
            assert verdict.admit
            assert verdict.measured['registry_fingerprint'] == expected
        assert snapshot.call_count == 1


def test_mutations_config_restore_and_detached_snapshots_keep_old_hash():
    trust = registry()
    previous = trust.fingerprint()
    trust.assert_statement('b', statement='s', source='lab', asserted_at=2, now=2)
    assert trust.fingerprint() == old_hash(trust) != previous
    previous = trust.fingerprint()
    trust.label('b', loss=.5, now=3)
    assert trust.fingerprint() == old_hash(trust) != previous
    for field, value in [('window', 20.), ('prior', [2., 3.]), ('min_labels', 2)]:
        previous = trust.fingerprint()
        setattr(trust, field, value)
        assert trust.fingerprint() == old_hash(trust) != previous
    previous = trust.fingerprint()
    trust.prior[0] = 4.
    assert trust.fingerprint() == old_hash(trust) != previous
    state = trust.snapshot()
    restored = SourceTrust.restore(state)
    assert restored.snapshot() == state
    assert restored.fingerprint() == old_hash(trust)
    state['assertions']['a']['source'] = 'mutated detached snapshot'
    assert restored.fingerprint() == old_hash(trust)


def test_invalid_writes_and_reads_preserve_cached_hash():
    trust = registry()
    expected = trust.fingerprint()
    with patch.object(trust, 'snapshot', wraps=trust.snapshot) as snapshot:
        with pytest.raises(ValueError):
            trust.assert_statement('a', statement='s', source='other', asserted_at=2, now=2)
        with pytest.raises(ValueError):
            trust.label('a', loss=.5, now=2)
        with pytest.raises(ValueError):
            trust.label('missing', loss=float('nan'), now=2)
        trust.opinion('lab', now=100)
        assert trust.fingerprint() == expected
        assert snapshot.call_count == 0
