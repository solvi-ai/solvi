import pytest

from solvi.core.signed import HmacScheme, Record, SignatureCache, Verifier, sign


def fixture(**kw):
    s = HmacScheme({'A': b'a', 'M': b'm'})
    return s, Verifier(s, max_age=10, **kw)


def test_snapshot_and_tampering():
    s, v = fixture(replay=False)
    value = {'x': [1]}
    r = sign(s, 'A', value, 5, nonce='n')
    value['x'][0] = 9
    r.value['x'][0] = 8
    assert r.value == {'x': [1]} and v.check(r, 5).ok
    for kw in ({'source': 'M'}, {'time': 6}, {'scope': 'other'}, {'nonce': 'x'}, {'value': 2}):
        fields = dict(source=r.source, time=r.time, value=r.value, scope=r.scope, nonce=r.nonce, tag=r.tag)
        fields.update(kw)
        bad = v.check(Record(**fields), 6)
        assert bad.reason == 'bad_signature' and bad.authenticated_source is None and not bad.evidence
    with pytest.raises(TypeError):
        Record('A', 1, 1, tag=bytearray(b'x'))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), {1: 'x'}, object(), (1, 2)])
def test_strict_json(value):
    with pytest.raises(ValueError):
        Record('A', 0, value)


def test_cache_shared_rotation_namespace_and_negative():
    class Counting(HmacScheme):
        calls = 0
        def verify(self, *args):
            self.calls += 1
            return super().verify(*args)
    s = Counting({'A': b'a'})
    c = SignatureCache()
    r = sign(s, 'A', 1, 0)
    for _ in range(2):
        assert Verifier(s, max_age=10, replay=False, cache=c).check(r, 0).ok
    assert s.calls == 1
    s.register('A', b'b')
    assert not s.verify_cached('A', r.payload(), r.tag, c)
    assert not s.verify_cached('A', r.payload(), r.tag, c)
    assert s.calls == 2
    assert HmacScheme({'A': b'a'}).verify_cached('A', r.payload(), r.tag, c)
    s.revoke('A')
    assert not s.verify_cached('A', r.payload(), r.tag, c)


def test_cache_limits_and_exact_bytes():
    s, _ = fixture()
    c = SignatureCache(max_entries=1, max_bytes=400)
    for payload in (b'a', b'b', b'c' * 500):
        tag = s.sign('A', payload)
        assert s.verify_cached('A', payload, tag, c)
        assert len(c._items) <= 1 and c._bytes <= 400
        assert not s.verify_cached('A', payload + b'x', tag, c)
    assert not s.verify_cached('A', b'a', bytearray(b'a'), c)


def test_freshness_and_replay_not_cached():
    s, v = fixture(cache=SignatureCache())
    r = sign(s, 'A', 1, 5)
    assert v.check(r, 5).ok
    assert v.check(r, 5).reason == 'replay'
    assert v.check(r, 16).reason == 'stale'
    s, v = fixture(replay=False, cache=SignatureCache())
    r = sign(s, 'A', 1, 5)
    assert v.check(r, 4).reason == 'future'
    assert v.check(r, 5).ok and v.check(r, 15).ok
    assert v.check(r, 16).reason == 'stale'
    assert not v._seen


def test_replay_bounds_out_of_order_preview_and_clock():
    s, v = fixture(max_seen=2)
    a, b, c = [sign(s, 'A', x, t) for x, t in [(1, 9), (2, 1), (3, 10)]]
    assert v.check(a, 10, remember=False).ok
    assert not v._seen and v._last_now is None
    assert v.check(a, 10).ok and v.check(b, 10).ok
    assert v.check(c, 10).reason == 'replay_capacity'
    before = dict(v._seen)
    assert v.check(c, 12, remember=False).ok
    assert v._seen == before and v._last_now == 10
    assert v.check(c, 12).ok  # b expires despite arriving after a
    assert v.check(a, 11).reason == 'clock_rollback'
    assert v.check(a, 12, remember=False).reason == 'replay'


def test_sizes_and_attribution():
    s, v = fixture(max_seen_bytes=1)
    r = sign(s, 'A', 1, 0)
    assert v.check(r, 0).reason == 'replay_capacity'
    assert Verifier(s, max_age=10, max_payload_bytes=1).check(r, 0).reason == 'too_large'
    fake = Record('A', 0, 1, tag=b'wrong')
    verdict = v.check(fake, 0)
    assert verdict.claimed_source == 'A' and verdict.authenticated_source is None and not verdict.evidence
    assert v.check(Record('Z', 0, 1, tag=b'x'), 0).reason == 'unknown_source'
    assert v.check(Record('A', 0, 1), 0).reason == 'unsigned'
    for val in (None, float('nan'), float('inf'), -1):
        with pytest.raises((ValueError, TypeError)):
            Verifier(s, max_age=val)


def test_rotation_does_not_reset_replay():
    s, v = fixture()
    r = sign(s, 'A', 1, 0, nonce='n')
    assert v.check(r, 0).ok
    s.register('A', b'new')
    assert v.check(r, 0).reason == 'bad_signature'
    rotated = sign(s, 'A', 1, 0, nonce='n')
    assert v.check(rotated, 0).reason == 'replay'


def test_replay_expiry_uses_the_same_float_boundary_as_freshness():
    s = HmacScheme({'A': b'a'})
    v = Verifier(s, max_age=1e16)
    r = sign(s, 'A', 1, -1e16)
    assert v.check(r, 0).ok
    assert v.check(r, 1).reason == 'replay'
    assert v.check(r, 1).reason == 'replay'
