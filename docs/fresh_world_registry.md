# Fresh-world registry for a finite candidate family

`FreshEnvironments` can opt into a persisted registry in its Pipeline journal:

```python
from solvi.core.store import JSONLStorage
from solvi.experimental.changes import FreshEnvironments, Pipeline

fresh = FreshEnvironments(
    env, agent, select=selection_seeds, confirm=confirmation_seeds,
    world_registry="owner-environment-v1",
    candidate_ids=["candidate-a", "candidate-b"],
    candidate_id=lambda change, ctx: change.proposer,
    world_id=lambda seed: f"owner-environment-v1:{seed}",
    alpha=0.05,
)
pipeline = Pipeline([fresh], JSONLStorage("changes.jsonl", fsync=True))
```

The owner supplies stable world identities, candidate identities and the complete finite family before trying its candidates. Identity is not inferred from seed equality, model semantics or a hidden environment state. Generate genuinely fresh, unpredictable worlds externally; the registry does not randomize selection or establish that a candidate has never seen a world elsewhere.

Selection and confirmation world IDs must be distinct. Before constructing either policy, the stage atomically checks prior reservations and records both sets. A repeated candidate or any reused world is refused before agent evaluation. Crashes and refused comparisons burn the reservation; restarting cannot retry those worlds. Family order, alpha and identity callback code fingerprints are frozen within a registry. Changing them requires a deliberately new namespace, which also starts separate accounting: do not relabel a family to evade its budget.

The registry requires nonrotating `JSONLStorage(fsync=True)` or file-backed SQLiteStorage with FULL/EXTRA synchronous and durable journaling. SQLite's actual main filename is checked with PRAGMA database_list; ':memory:' and empty-path temporary databases are refused even though they use the same storage class. Explicit synchronous OFF/NORMAL and journal_mode OFF/MEMORY are also refused. JSONL uses its existing file lock and batch write/fsync, SQLite its existing write transaction; read/check/reserve occur inside that transaction. The process lock alone is not the cross-process guard. Memory/custom backends and nondurable JSONL are refused. Call the stage outside any existing storage batch: an outer transaction could roll back an evaluated reservation or defer its fsync. Active JSONL/SQLite batches are rejected before selection, identity or agent callbacks. A redacted journal fails closed; registry history must remain complete. External journal replacement, malicious writers or reuse across unrelated stores are outside this contract.

Alpha/N and 1-alpha/N must both be representable strictly between zero and one; unusable tiny allocations are rejected at construction. Each candidate receives nominal alpha/N from the frozen family, and selection/confirmation/progress bootstrap intervals use level 1-alpha/N. Measurements record allocation and reservation ID. This is explicit multiple-attempt accounting for the existing bootstrap procedure, not a distribution-free family error guarantee: empirical bootstrap bounds need their own assumptions and adequate data/replicates. No new statistical validity, utility improvement or corrected-gain evaluation is claimed. The earlier dropped changes_multiplicity_v2 experiment stays dropped; this registry is a separate optional infrastructure feature. Defaults retain the previous caller-managed behavior.
