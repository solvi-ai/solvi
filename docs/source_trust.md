# Recent source reliability

`SourceTrust` summarizes observed source losses within an explicit time window.
Its Beta posterior mean is a descriptive model, not a truth certificate, confidence
bound, e-value, or permission to publish an otherwise inadmissible claim. Source
identities must be authenticated by the application; a signature alone does not
establish correctness.

```python
from solvi.stat import SourceTrust
from solvi.core.knowledge import KnowledgeStore, TrustGate

trust = SourceTrust(window=10, prior=(1, 1), min_labels=1)
trust.assert_statement(
    'original', statement='pump-pressure-at-t0', source='lab',
    asserted_at=0, now=0,
)
trust.label('original', loss=0, now=1)  # independently observed bounded loss
trust.assert_statement(
    'publication', statement='pump-pressure-at-t0', source='publisher',
    asserted_at=2, now=2, parent='original',
)
trust.label('publication', loss=0, now=3)

opinion = trust.opinion('publisher', now=3, role='repeater')
assert opinion.support == 'supported'
assert opinion.n == 1

store = KnowledgeStore(
    writers={'publisher': 'spec'},
    gates=[TrustGate(
        trust, clock=lambda: 3, min_reliability=0.6, role='repeater',
    )],
)
item = store.add(
    'fact', {'s': 'pump', 'r': 'pressure', 'o': 10},
    source='spec', by='publisher',
)
assert item is not None

restored = SourceTrust.restore(trust.snapshot())
assert restored.fingerprint() == trust.fingerprint()
```

Choose the window, fixed prior, minimum label count and gate threshold explicitly
for the application. In this example a single successful observation and the
uniform prior yield reliability 2/3. This is a small amount of model support,
not proof that another assertion will be correct.

## Time and delayed feedback

Supply finite numeric times representable as Python floats in consistent units.
Unrepresentable numeric inputs are refused with `ValueError` before registry
mutation; the same numeric domain applies to the window, prior and bounded loss. Registration and labeling use
monotone `now` values denoting when information became known. `asserted_at` may
be earlier than registration, but cannot be in the future. Opinions may be read
at or after the most recent known time, without advancing the registry clock.

The active interval is `(now - window, now]` in **assertion time**. A label arriving
after its assertion expires remains in the audit history but does not rejuvenate
that assertion. Labels arriving before expiry contribute only until the assertion
expires. Old evidence therefore cannot accumulate indefinitely as current support.
An observation received late is not evidence about a new assertion unless the
application explicitly establishes that connection.

## Attribution and interpretation

Every origin and repeat has its own unique assertion identifier, source and time.
A repeat names an existing parent of the same statement. Changing the proposition
or its target time requires a different statement identifier. A loss in `[0, 1]`
attaches exactly once to exactly one assertion; no label automatically spreads to
its ancestors or descendants. To hold both an originator and a repeater accountable,
the application must explicitly verify and label each accountable assertion.

`role='origin'` and `role='repeater'` keep those records separate;
`role='all'` pools a source's two roles. Opinions include recent labeled count,
pending count, expired count, mean loss, last assertion time and its age.
`support='unknown'` means fewer than `min_labels` recent labels are available.
`support='supported'` only means this count requirement was met. Counts are not
counts of independent observations; copied claims and repeated labels can be
correlated, and selectively observed outcomes can bias the estimate.

## Opt-in write gate

`TrustGate` requires an explicit clock callable and `min_reliability`. It reads the
writer's identity from `item.by`. An optional `writer_sources` mapping explicitly
maps application writer IDs to registry source IDs; no implicit trust transfers
from a rank such as `spec` or `outcome`. Authenticate writer IDs before supplying
them. WriterGate remains the separate mechanism that binds writers to allowed
source ranks.

Unknown, expired or below-threshold evidence refuses facts, rules, skills and
actions before insertion. Invalid clock values also refuse. Reads do not mutate
the registry. A verdict's `measured` data includes count, support status,
reliability, role, timestamp and registry fingerprint. Refusal reasons enter the
store's journal. SourceGate still runs first under the normal store gate setup;
reputation cannot authorize an unverified system answer or bypass WriterGate.
Other custom gates retain their existing hold behavior for behavior-changing
items. This feature does not alter the default gate list.

Snapshots are JSON-compatible detached dictionaries. Restore validates schema,
chronology, lineage and bounded losses before returning a new registry. A SHA-256
fingerprint identifies the serialized state; it is not a signature or a guarantee
of honest data. History is retained, so applications with long-lived registries
must plan storage growth. No automatic regime detection or selective-label
correction is performed.

## Lookup cost

Opinions scan only the requested source’s retained history, including expired
assertions. A derived index adds one assertion-ID reference per assertion plus
per-source containers; it does not prune history or bound work to the active
window. Snapshots keep their existing schema and restore rebuilds the index.

The unchanged registry caches one fingerprint and a serialized small configuration
key. Public assertion/label writes invalidate it; valid changes to `window`,
`prior` or `min_labels` are detected, including edits within a list prior. Reads
do not invalidate it. Direct mutation of underscore-prefixed internal state is
unsupported. The fingerprint identifies owner-managed state, not an integrity
attestation against arbitrary in-process memory edits. Detached snapshots and
restores retain the existing JSON schema.

Fingerprint caching applies only to exact native `SourceTrust` instances using
the native snapshot method. Subclasses and instance snapshot overrides recompute
the full hash on every call, preserving externally supplied snapshot behavior.
