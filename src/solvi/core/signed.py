"""Signed JSON records. Authentication is not truth, public proof, or evidence of forgery.

Replay state is local to one verifier's lifetime. Operations are thread safe; key
rotation and MAC verification share a lock. HMAC verifiers possess signing secrets.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hmac
import json
import math
import os
from threading import RLock


def _finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _json(value):
    # Roundtrip rejects non-JSON objects; disallow non-string keys instead of coercing.
    def check(x):
        if x is None or isinstance(x, (str, bool, int)):
            return
        if isinstance(x, float) and math.isfinite(x):
            return
        if isinstance(x, list):
            for v in x:
                check(v)
            return
        if isinstance(x, dict) and all(isinstance(k, str) for k in x):
            for v in x.values():
                check(v)
            return
        raise ValueError("value must contain only finite JSON values and string keys")
    check(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


class SignatureCache:
    """Explicitly shared, bounded MAC-result LRU; no time or replay verdicts cached."""
    def __init__(self, max_entries=4096, max_bytes=4 * 1024 * 1024):
        if not isinstance(max_entries, int) or max_entries < 0 or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("cache bounds must be nonnegative integers")
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self._items, self._bytes, self._lock = OrderedDict(), 0, RLock()

    def _get(self, key):
        with self._lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key][0]

    def _put(self, key, result):
        size = 128 + len(key[1].encode()) + len(key[2]) + len(key[3])
        with self._lock:
            if not self.max_entries or size > self.max_bytes:
                return
            old = self._items.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._items[key] = (result, size)
            self._bytes += size
            while len(self._items) > self.max_entries or self._bytes > self.max_bytes:
                _, (_, n) = self._items.popitem(last=False)
                self._bytes -= n


class HmacScheme:
    """HMAC-SHA256 registry. Replacement/revocation invalidates cached decisions."""
    def __init__(self, registry=None):
        self._keys, self._lock, self._namespace = {}, RLock(), object()
        for source, secret in dict(registry or {}).items():
            self.register(source, secret)

    def register(self, source, secret=None):
        if not isinstance(source, str):
            raise TypeError("source must be a string")
        if secret is not None and not isinstance(secret, bytes):
            raise TypeError("secret must be bytes")
        with self._lock:
            if source in self._keys and secret is None:
                return self._keys[source]
            secret = os.urandom(32) if secret is None else secret
            self._keys[source] = secret
            self._namespace = object()
            return secret

    def revoke(self, source):
        with self._lock:
            self._keys.pop(source, None)
            self._namespace = object()

    def known(self, source):
        with self._lock:
            return source in self._keys

    def sign(self, source, payload):
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        with self._lock:
            return hmac.digest(self._keys[source], payload, "sha256")

    def verify(self, source, payload, tag):
        with self._lock:
            return (isinstance(payload, bytes) and isinstance(tag, bytes) and source in self._keys
                    and hmac.compare_digest(hmac.digest(self._keys[source], payload, "sha256"), tag))

    def verify_cached(self, source, payload, tag, cache=None):
        if not isinstance(source, str) or not isinstance(payload, bytes) or not isinstance(tag, bytes):
            return False
        with self._lock:
            if source not in self._keys:
                return False
            key = (self._namespace, source, payload, tag)
            hit = None if cache is None else cache._get(key)
            if hit is not None:
                return hit
            result = self.verify(source, payload, tag)
            if cache is not None:
                cache._put(key, result)
            return result


@dataclass(frozen=True, init=False)
class Record:
    """Immutable JSON snapshot; value returns a detached copy. All fields except tag signed."""
    source: str
    time: float
    scope: str
    nonce: str
    tag: bytes | None
    _payload: bytes

    def __init__(self, source, time, value, scope="*", nonce="", tag=None):
        if not all(isinstance(x, str) for x in (source, scope, nonce)):
            raise TypeError("source, scope and nonce must be strings")
        if tag is not None and not isinstance(tag, bytes):
            raise TypeError("tag must be bytes or None")
        time = _finite(time, "time")
        payload = _json([source, time, scope, value, nonce])
        for name, val in (("source", source), ("time", time), ("scope", scope),
                          ("nonce", nonce), ("tag", tag), ("_payload", payload)):
            object.__setattr__(self, name, val)

    @property
    def value(self):
        return json.loads(self._payload)[3]

    def payload(self):
        return self._payload


def sign(scheme, source, value, time, scope="*", nonce=None):
    rec = Record(source, time, value, scope, os.urandom(16).hex() if nonce is None else nonce)
    return Record(source, rec.time, rec.value, scope, rec.nonce, scheme.sign(source, rec.payload()))


@dataclass(frozen=True)
class SignedVerdict:
    ok: bool
    reason: str
    record: Record
    authenticated_source: str | None = None

    @property
    def claimed_source(self):
        return self.record.source

    @property
    def evidence(self):
        return False


class Verifier:
    """Finite freshness; optional local replay protection with fail-closed capacity.

    remember=False never changes replay state (including clock/pruning state).
    Byte limits count payload, tag and a fixed per-entry allowance, not Python RSS.
    """
    def __init__(self, scheme, *, max_age, max_skew=0, replay=True, cache=None,
                 max_seen=10000, max_seen_bytes=8 * 1024 * 1024, max_payload_bytes=1024 * 1024):
        self.max_age, self.max_skew = _finite(max_age, "max_age"), _finite(max_skew, "max_skew")
        if self.max_age < 0 or self.max_skew < 0:
            raise ValueError("time bounds must be nonnegative")
        for n in (max_seen, max_seen_bytes, max_payload_bytes):
            if not isinstance(n, int) or n < 0:
                raise ValueError("size bounds must be nonnegative integers")
        self.scheme, self.replay, self.cache = scheme, replay, cache
        self.max_seen, self.max_seen_bytes, self.max_payload_bytes = max_seen, max_seen_bytes, max_payload_bytes
        self._seen, self._seen_bytes, self._last_now, self._lock = {}, 0, None, RLock()

    def check(self, rec, now, *, remember=True):
        if not isinstance(rec, Record):
            raise TypeError("expected Record")
        now = _finite(now, "now")
        with self._lock, self.scheme._lock:
            authenticated = None
            def verdict(reason):
                return SignedVerdict(reason == "ok", reason, rec, authenticated)
            if len(rec.payload()) > self.max_payload_bytes or (rec.tag is not None and len(rec.tag) > 32):
                return verdict("too_large")
            if rec.tag is None:
                return verdict("unsigned")
            if not self.scheme.known(rec.source):
                return verdict("unknown_source")
            if not self.scheme.verify_cached(rec.source, rec.payload(), rec.tag, self.cache):
                return verdict("bad_signature")
            authenticated = rec.source
            if self.replay and self._last_now is not None and now < self._last_now:
                return verdict("clock_rollback")
            if now - rec.time > self.max_age:
                return verdict("stale")
            if rec.time - now > self.max_skew:
                return verdict("future")
            if not self.replay:
                return verdict("ok")
            # Exact signed content identity persists across key rotation.
            key = rec.payload()
            # Use exactly the freshness predicate above: adding time + max_age
            # can round differently and forget a message that is still fresh.
            live = {k: v for k, v in self._seen.items() if now - v[0] <= self.max_age}
            if remember:
                self._seen = live
                self._seen_bytes = sum(v[1] for v in live.values())
                self._last_now = now
            if key in live:
                return verdict("replay")
            size = len(key) + len(rec.tag) + 128
            if len(live) >= self.max_seen or sum(v[1] for v in live.values()) + size > self.max_seen_bytes:
                return verdict("replay_capacity")
            if remember:
                self._seen[key] = (rec.time, size)
                self._seen_bytes += size
            return verdict("ok")
