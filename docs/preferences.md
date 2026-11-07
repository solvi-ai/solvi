# Finite preference elicitation

`solvi.experimental.experience.preferences` implements a single author's finite
linear utility family. It does not interpret natural language or infer a
continuous weight polytope. Utility is the exact integer dot product of 3–5
integer attributes and declared integer weights. Larger utility is preferred;
encode costs as negative attributes/weights explicitly.

Create immutable `Alternative(id, attributes)` and
`Requirement(id, attribute, minimum, hard=True)` records, then pass concrete
lists/tuples to `PreferenceSession(alternatives, weights, requirements=(),
budget=shared_budget, host_revision="v1")`. Limits are 64 alternatives, 256
unique weight vectors, 16 requirements, 128 answer IDs and 512 historical events.
Boolean values are rejected wherever integers are required.

`choose()` returns a `RegretCertificate` containing a stable choice ID, exact
integer maximum regret, surviving weight indices, all candidate regrets, and a
fingerprint of the complete input/history. The comparator set consists of
alternatives satisfying **all** requirements. Minimax regret enumerates that
entire set for every surviving utility. Ties use lexicographic alternative IDs.
Empty feasibility is `no_feasible_alternative`; inconsistent preferences are
`conflict`, with no choice or numeric regret. These statuses do not prove the
original problem impossible beyond the declared finite family.

`check_regret_certificate(session, certificate, budget=shared_budget)` rebuilds
the utility table with independent enumeration and checks every certificate
field. It never calls `choose()` or the producer's regret routine. The host's
input and revision remain the trust boundary; the fingerprint is an integrity
check, not a signature or external outcome verification.

`next_question(cost=0, criterion="worst_case")` considers all feasible pairs.
Its three disjoint outcomes are strict left preference, tie, and strict right
preference. It chooses the greatest exact minimax-regret reduction minus the
declared integer question cost, returning `None` if no strictly positive value
remains. `criterion="expected", probabilities=(...)` instead requires declared
nonnegative integer masses over the original weight family and normalizes them
over survivors. Expectations use exact `Fraction` arithmetic. Query selection
work is charged separately; question cost is a decision penalty, not an
automatically purchased observation. Record an obtained answer through
`answer(id, preferred, other, source=..., relation="strict", revision=None)`.
`relation="weak"` means greater-or-equal, `"tie"` means equal. For a tie, pass
both question IDs in either order. Strict answers match the query's two strict
branches; weak answers deliberately supply less information.

Answer IDs persist; another answer with that ID supersedes it with the next
revision. `retract(id, source=...)` appends an inactive revision. Sources must be
nonempty provenance strings. Contradictory answers remain in history and create
explicit conflicts; no automatic repair occurs. `conflicts()` returns all
minimal-by-inclusion inconsistent sets of active answer IDs and separately of
requirement IDs. `relaxations()` returns all minimal-by-inclusion sets of
**negotiable** requirement IDs whose removal would restore feasibility. It
returns `((),)` when already feasible and `()` when hard requirements alone
prevent feasibility. These are proposals only: requirements are never mutated.

Every enumeration step debits the shared `WorkBudget` before work. Exhaustion
raises `BudgetStop` rather than emitting a partial certificate, conflict set or
question. Combinatorial conflict enumeration can exhaust a small budget even
when the declared domain is within bounds. Work units describe operations, not
wall time or dollars.

`snapshot()` includes the immutable problem, host revision, entire event history,
strict schema version, and digest. `PreferenceSession.restore(snapshot,
budget=shared_budget, host_revision="v1", authority=session.authority)` charges restoration and replays each
event through validation, rejecting corrupt schemas, invalid revisions and host
revision mismatches. Use the existing shared ledger: restoring preferences does
not reset prior spending. Provenance is retained but not authenticated.

Public records also include `PreferenceAnswer` (historical assertion) and
`PreferenceQuestion` (IDs, exact reduction, cost, net value and criterion).
See `examples/38_preference_elicitation.py` for a runnable interaction.

`PreferenceAuthority` is the host authentication capability. Keep it outside
untrusted snapshots; `restore(..., authority=session.authority)` requires it.
Snapshots carry an HMAC and monotonic session epoch, so recomputed plain hashes
cannot remove answers and old checkpoints cannot undo a later answer/retraction.
A mutation of one restored copy invalidates other copies sharing that authority.
`export_trusted_state(budget=shared_budget)` /
`PreferenceAuthority.from_trusted_state(state, budget=shared_budget)` enable
cross-process persistence of the secret and epoch registry. Store that state in
trusted host storage, atomically updating it after mutations. Restoring an old
trusted authority state also rolls back its protection; the payload alone is
never authority. The snapshot contains no spending permission: restoration uses
the supplied existing shared WorkBudget.

The public `answers` property returns an immutable tuple of frozen records;
mutating a detached copy cannot change the session or bypass its epoch. A signed
checkpoint also binds the current ledger. Restore requires the supplied ledger's
entry state to retain the same limit and at least every recorded category count
and total spending. For cross-process continuation, persist and restore the
trusted WorkBudget ledger alongside the separately trusted authority state;
a fresh empty or higher-limit budget is rejected. Replay costs are additional.
