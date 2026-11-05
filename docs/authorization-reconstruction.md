# Updating owner-managed authorization

Identity, key custody and retirement of old service instances belong to the host.
These recipes use existing APIs; they are not a global revocation service or a
promise to detect stolen credentials. See [signed records](api/signed.md) and the
[guard approval/resume contract](guide.md).

## Writer registry: reconstruct and retire old instances

`Knowledge(writers=...)` and `KnowledgeStore(writers=...)` copy the input registry.
Changing its original dictionary **does not revoke** existing instances:

```python
from solvi.core.knowledge import KnowledgeStore

registry = {"alice": "outcome"}
store = KnowledgeStore("knowledge.jsonl", writers=registry)
registry.clear()  # store still has its own original registry
replacement = KnowledgeStore("knowledge.jsonl", writers={})
# Route future authenticated writes only to replacement; retire all old handlers.
```

The replacement refuses new writes from the removed writer. Replayed historical
facts remain historical facts; permission changes do not implicitly retract them.
Use the existing explicit withdrawal/review APIs if the evidence itself is no
longer trusted. Do not obtain `by` from an untrusted payload and treat it as an
identity. Reconstructing one instance cannot revoke another still-exposed one.

## Guard: replace verifier, recheck pending calls

`approval_verifier(keys)` and `signed_messages(keys)` also copy their key mapping.
Replace the verifier or construct a new guard with the new mapping; editing the
original dictionary does not update an existing verifier:

```python
from solvi.solutions.guard import approval_verifier

guard.user_verifier = approval_verifier({})  # no reviewer is authorized now
```

Use `guard.policy(...)` for additional owner policies; subsequent checks rebuild
the compiled policy systems. A signed approval covers a snapshot of one call and
its reasons, not all future policy states. Before resuming a pending external
call, check it again under current policy/context and compare approval keys as
specified in the guide. If denied, do not execute; if the required approval has
changed, obtain a new one. `resolve(old_decision, ...)` is not an automatic current
policy recheck. Retire old guard/verifier references, authenticate the reviewer,
and enforce token freshness; never sign approvals inside an untrusted agent.

## Native HMAC registry: rotate/revoke cached authentication

```python
from solvi.core.signed import HmacScheme, SignatureCache

scheme = HmacScheme({"alice": b"EXAMPLE-ONLY-OLD-KEY"})
cache = SignatureCache()
body = b"example exact bytes"
tag = scheme.sign("alice", body)
assert scheme.verify_cached("alice", body, tag, cache)
scheme.register("alice", b"EXAMPLE-ONLY-NEW-KEY")
assert not scheme.verify_cached("alice", body, tag, cache)
scheme.revoke("alice")
```

The registry invalidates its own cached MAC decisions on replacement/revocation.
These fixed keys are documentation fixtures, not deployment credentials. HMAC
means a registry key holder authenticated exact bytes; it does not establish truth,
public authorship, transport identity or independent evidence. This registry is
separate from a guard verifier built from a copied dictionary: revoking one does
not automatically update the other. Do not infer cross-process propagation or
persistent replay state from the in-memory primitive.
