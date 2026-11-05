"""Measured registry identity follows owner clock refresh before opinion evaluation."""
from solvi.core.knowledge.gates import TrustGate
from solvi.stat import SourceTrust


def test_clock_refresh_records_the_registry_used_for_the_verdict():
    registry = SourceTrust(window=10)
    registry.assert_statement('a', statement='s', source='writer', asserted_at=0, now=0)
    registry.label('a', loss=0, now=0)
    before = registry.fingerprint()
    calls = []

    def clock():
        calls.append(1)
        registry.assert_statement('b', statement='s2', source='writer', asserted_at=1, now=1)
        registry.label('b', loss=1, now=1)
        return 1

    gate = TrustGate(registry, clock=clock, min_reliability=.6)
    verdict = gate.admit(None, {'by': 'writer'})
    assert not verdict.admit
    assert verdict.measured['n'] == 2 and verdict.measured['reliability'] == .5
    assert verdict.measured['registry_fingerprint'] == registry.fingerprint() != before
    assert calls == [1]


def test_invalid_writer_does_not_call_clock_and_still_has_current_registry_identity():
    registry = SourceTrust(window=10)
    calls = []
    gate = TrustGate(registry, clock=lambda: calls.append(1), min_reliability=.6)
    verdict = gate.admit(None, {})
    assert not verdict.admit and calls == []
    assert verdict.measured['registry_fingerprint'] == registry.fingerprint()
