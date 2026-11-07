# Interactive finite concept teaching

`solvi.experimental.experience.teaching` provides a bounded executable teaching
interface. It does not parse natural language or execute user code. The host
supplies a finite library of typed `Program` trees, a `Grammar`, a finite question
domain, scalar input/output types, and a shared `WorkBudget`.

```python
from solvi.experimental.experience import Grammar, Program, WorkBudget
from solvi.experimental.experience.teaching import TeachingSession

s = TeachingSession(
    Grammar(('x',), constants=(1, 2)),
    [Program(('lt', ('input', 'x'), ('const', n))) for n in (1, 2)],
    [{'x': 0}, {'x': 1}, {'x': 2}], budget=WorkBudget(10000),
)
question = s.question()              # x=1 separates the hypotheses
s.answer(question, True)             # one hypothesis remains
concept = s.admit('small', '1', [{'x': 0}, {'x': 2}],
                  lambda row: row['x'] < 2,
                  source_id='host-reference', source_revision='1')
assert s.apply(concept, {'x': 2}).value is False
assert s.apply(concept, {'x': 1}).status == 'unknown'
```

## Public API

- `TeachingSession(grammar, hypotheses, query_domain, *, budget, input_types=None,
  result_type='bool', max_items=256, authority=None)`: materializes bounded inputs
  with a paid overflow probe. `max_items` is at most 10000. Default input types
  are strict integers; values are bounded by `Grammar.max_abs`. Booleans and
  integers are distinct. Collection limits apply to hypotheses, query points,
  examples, concepts, and admission points.
- `teach(inputs, value, exception=False)` adds a labelled example and recomputes
  the finite committee. An exception is an explicit counterexample, not an
  implicit default rule. Contradictory labels remain in the history and leave
  an empty committee. Every new label invalidates prior admissions conservatively.
- `candidates` returns the immutable tuple of surviving programs. These are
  proposals, including when only one survives.
- `question()` exhaustively evaluates the committee on untaught query points and
  returns the largest split (minimizes the largest answer group), with stable
  first-point tie breaking. `TeachingQuestion` contains `revision`, portable
  `inputs_json`, `predictions`, and a defensive `inputs` property. No question
  means no split in the declared domain; it does not establish global equivalence.
- `answer(question, value)` checks the question revision and records its label.
  Stale questions are rejected. Questions themselves grant no verification rights.
- `admit(name, version, domain, oracle, *, source_id, source_revision)` requires a
  unique proposal and independently exhausts a **separate** finite domain using
  a host oracle. Admission points may not overlap labelled proposal examples.
  A mismatch returns `None`; exhaustion raises `BudgetStop` and installs nothing.
  Source ID/revision and exact reference labels are retained as provenance.
- `ConceptRef(name, version)` is an immutable version reference. Versions cannot
  be reused, even after retraction.
- `compose(name, version, template, bindings, domain, oracle, *, source_id,
  source_revision)` substitutes accepted concept programs into named template
  input terminals, pins dependencies to exact versions, and independently checks
  the resulting program. Bound dependencies use the same session input schema.
  Existing `Program`/`Grammar` depth, node, operator and constant bounds apply.
- `apply(ref, inputs)` returns `TeachingResult(status, value, reason)`. Only a live
  concept on its independently checked finite domain yields `status='verified'`.
  Untested inputs, absent concepts, stale revisions, and retracted dependencies
  return `status='unknown'`. No extrapolation is presented as verified.
- `retract(ref)` disables that version and transitively prevents dependent use.
- `snapshot()` and `TeachingSession.restore(snapshot, budget=..., authority=...)`
  provide paid, authenticated replay. The supplied ledger must retain the exact
  original limit and at least every previously paid category debit; a fresh or
  larger ledger is rejected. The signed snapshot and trusted authority anchor
  both carry a spending floor. Live shared-ledger work outside the session is
  checked too, and trusted authority export refreshes this high-water mark. Restore reevaluates stored reference labels
  without calling the oracle. Snapshot mutation is rejected.
- `TeachingAuthority(secret=None)` is a **trusted host capability**. Keep the same
  object and its monotonic epoch registry for restore. Saved JSON or a matching
  plain digest alone cannot grant admission. A new authority, even with the same
  secret but without the host epoch registry, cannot restore these receipts.
  Old snapshots cannot revive concepts after teaching, admission, or retraction;
  mutation through one restored copy makes other copies stale. This v1 does not
  implicitly persist the authority registry across process restarts.

For process restart, the host may explicitly save
`authority.export_trusted_state(budget=...)` into its **separate trusted secret
store**, then call `TeachingAuthority.from_trusted_state(state, budget=...)` in
the fresh process. This preserves the private authentication key,
monotonic session epochs and ledger high-water marks. Persist the actual shared
`WorkBudget.snapshot()` as well and reconstruct it with `WorkBudget.restore()`;
creating a fresh ledger with a new spending allowance is rejected. The host must atomically retain the newest authority
checkpoint together with its session snapshot. Import validates the caller
ledger against the checkpoint before accepting it, then anchors all import
charges and subsequent work on that same live shared ledger. An authority
checkpoint must cover sessions using one compatible shared ledger; use separate
authorities for incompatible budget limits. Teacher-supplied JSON must never
be imported as trusted authority state. Rolling back the host's own trusted
checkpoint is outside the capability boundary. Export/import are paid and
schema validated; replay still makes no oracle calls.

Session configuration properties are read-only. `input_types` returns a detached
dictionary; programs and grammar are immutable. `authority` remains the explicit
host capability, and `budget` exposes a read-only facade over the supplied shared
ledger. Python access to private implementation attributes is outside this API
contract. Replay uses an isolated temporary authority, so failed replay cannot
consume persistent session slots.

All exploration, program evaluation, domain checks, oracle calls, application,
and replay use the caller's ledger. Oracle calls cost one declared work unit in
addition to program-node costs; this is not a wall-time or monetary bound. The
host is responsible for bounding an oracle's own implementation. `BudgetStop`
propagates explicitly. A partially completed teaching operation retains its
new label but no partial committee or stale acceptance. A retry is a new paid
operation. Inputs/hypotheses are never consumed from unbounded iterators without
an explicit bound and pre-debit.

Admission proves agreement with the host's declared oracle only on the listed
finite points. It cannot establish real-world truth, verify the host oracle, or
identify arbitrary rules outside the supplied program library. Synthetic
interactive question savings do not demonstrate human teaching usability.
