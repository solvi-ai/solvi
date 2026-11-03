# Signed JSON records

`solvi.core.signed` authenticates exact bytes using a trusted HMAC-SHA256 key registry.
HMAC authenticates a key holder, including any verifier holding that secret. It does
not prove a claim true or publicly establish authorship, intent, or forgery.

```python
from solvi.core.signed import HmacScheme, SignatureCache, Verifier, sign
scheme = HmacScheme({'lab': b'application-managed-secret'})
cache = SignatureCache(max_entries=4096, max_bytes=4 * 1024 * 1024)
record = sign(scheme, 'lab', {'status': 'ok'}, time=100, scope='pump')
verifier = Verifier(scheme, max_age=30, cache=cache)
result = verifier.check(record, now=101)
assert result.ok and result.authenticated_source == 'lab'
```

Records snapshot strict JSON (finite numbers, lists, string-keyed dictionaries,
strings, booleans, null). Unsupported objects, tuples and non-string keys fail.
`record.value` returns a detached copy. Source, time, scope, value and nonce are
signed; tag must be bytes. No transport identity or knowledge rank is inferred.

`scheme.verify_cached(source, payload, tag, cache=None)` accepts exact bytes and
shares only MAC results across verifiers. Registry replacement or revocation
invalidates prior cache decisions; independent registries never share decisions.
A cache bounds entries and accounted bytes (payload, tag, source, fixed allowance),
not total Python process RSS. Oversized entries are not cached. Applications should
bound incoming raw byte messages before calling this generic primitive.

Verifier requires finite nonnegative `max_age`; `max_skew=0` rejects future records.
Freshness and replay are recomputed on every call, including cache hits.
`replay=False` allows repeated reads without storing replay state.
`remember=False` previews checks without changing replay/clock state; previously
seen messages still return `replay`. Clock rollback relative to a remembered check
returns `clock_rollback` when replay protection is enabled.

Replay state uses exact signed payload identity and survives registry rotation;
rotating immediately replaces the old key without overlap. State is local to one
verifier lifetime, not persistent or shared across processes. Out-of-order signed
timestamps expire independently. Limits `max_seen`, `max_seen_bytes` fail closed
with `replay_capacity`; unexpired entries are never evicted to admit a new one.
`max_payload_bytes` rejects large records with `too_large`. Byte accounting includes
payload, tag and a fixed allowance; it is not a hard RSS limit. Registry, cache and
verifier operations use locks; key rotation cannot interrupt a verification check.

`SignedVerdict` exposes `ok`, `reason`, `record`, `claimed_source`,
`authenticated_source` and `evidence` (always false). Authentication is established
only after a valid MAC, even if freshness/replay policy subsequently refuses it.
Reasons: `ok`, `unsigned`, `unknown_source`, `bad_signature`, `stale`, `future`,
`replay`, `replay_capacity`, `clock_rollback`, `too_large`. A bad signature does not
identify who changed or delivered it. Replay may be a retry. Neither proves fraud.
No automatic accusation or Knowledge import is provided; normal write gates and
application source-rank authorization remain necessary.

`signed_messages(keys, cache=cache)` in Guard uses the same MAC cache primitive
while retaining its existing `sender.timestamp.hex` format and repeated-history
semantics. It checks age and future skew on every call, including cache hits.

::: solvi.core.signed
