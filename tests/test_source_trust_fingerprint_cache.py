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
    with patch('solvi.stat._source_trust.json.dumps', wraps=json.dumps) as dumps:
        for _ in range(20):
            verdict = gate.admit(None, {'by': 'lab'})
            assert verdict.admit
            assert verdict.measured['registry_fingerprint'] == expected
        assert sum(isinstance(c.args[0], dict) and 'assertions' in c.args[0]
                   and c.kwargs.get('sort_keys') is True for c in dumps.call_args_list) == 1


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
    with patch('solvi.stat._source_trust.json.dumps', wraps=json.dumps) as dumps:
        with pytest.raises(ValueError):
            trust.assert_statement('a', statement='s', source='other', asserted_at=2, now=2)
        with pytest.raises(ValueError):
            trust.label('a', loss=.5, now=2)
        with pytest.raises(ValueError):
            trust.label('missing', loss=float('nan'), now=2)
        trust.opinion('lab', now=100)
        assert trust.fingerprint() == expected
        assert not any(isinstance(c.args[0], dict) and 'assertions' in c.args[0]
                       for c in dumps.call_args_list)


@pytest.mark.parametrize('subclass', [True, False])
def test_custom_public_snapshot_reflects_external_state_every_time(subclass):
    external = registry()
    if subclass:
        class CustomTrust(SourceTrust):
            def snapshot(self):
                return external.snapshot()
        trust = CustomTrust(window=10)
    else:
        trust = SourceTrust(window=10)
        trust.snapshot = external.snapshot
    gate = TrustGate(trust, clock=lambda: 2, min_reliability=.6)
    previous = trust.fingerprint()
    external.assert_statement('b', statement='s', source='lab', asserted_at=2, now=2)
    external.label('b', loss=1, now=2)
    expected = old_hash(external)
    assert trust.fingerprint() == expected != previous
    assert gate.admit(None, {'by': 'lab'}).measured['registry_fingerprint'] == expected


def test_class_snapshot_override_does_not_reuse_native_cache():
    trust, external = registry(), registry()
    previous = trust.fingerprint()
    external_snapshot = external.snapshot
    with patch.object(SourceTrust, 'snapshot', lambda self: external_snapshot()):
        external.assert_statement('b', statement='s', source='lab', asserted_at=2, now=2)
        assert trust.fingerprint() == old_hash(external) != previous
        external.label('b', loss=1, now=2)
        assert trust.fingerprint() == old_hash(external)
    assert trust.fingerprint() == previous


def test_custom_registry_proxy_does_not_require_native_cache_configuration():
    from solvi.core.knowledge.gates import TrustGate

    external = SourceTrust(window=10)
    external.assert_statement('a', statement='s', source='writer', asserted_at=0, now=0)
    external.label('a', loss=0, now=0)

    class Proxy(SourceTrust):
        def __init__(self, registry):
            self.registry = registry
            self.snapshot_reads = 0

        @property
        def snapshot(self):
            self.snapshot_reads += 1
            return self.registry.snapshot

        def opinion(self, *args, **kwargs):
            return self.registry.opinion(*args, **kwargs)

    proxy = Proxy(external)
    assert proxy.fingerprint() == external.fingerprint()
    assert proxy.snapshot_reads == 1
    gate = TrustGate(proxy, clock=lambda: 0, min_reliability=.6)
    assert gate.admit(None, {'by': 'writer'}).admit
