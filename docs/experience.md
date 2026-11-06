# Finite reasoning and executable experience

`solvi.experimental.experience` joins finite reasoning, paid observations, host-owned outcomes and newly acquired
scalar operations. It is experimental. An accepted operation has passed its finite admission examples; this does
not prove it correct outside those examples. No general learning gain or real-world autonomous improvement is claimed.

## Joint alternatives and missing information

```python
from solvi.experimental.experience import Probe, WorkBudget, WorldSet, select_probe

budget = WorkBudget(10_000)
worlds = WorldSet([
    {"stock": 7, "reserved": 2},
    {"stock": 8, "reserved": 3},
])
decision = worlds.decide(lambda state: state["stock"] - state["reserved"], budget)
assert decision.resolved and decision.value == 5
```

The correlation between stock and reservations is retained. This answer is invariant **within these alternatives**;
the host remains responsible for the real state being represented. Empty alternatives mean contradiction, never
proof of an arbitrary answer. Resolved `None` is distinct from UNKNOWN. Values use exact portable JSON integers,
booleans, strings, null, lists and string-keyed dicts; floats are intentionally unsupported in this first contract.

`observe(field, value, evidence_id, budget, depends_on=...)` adds host-supplied evidence. `constrain` evaluates a pure
predicate on the full finite base and stores its accepted states. `retract` removes that evidence and transitively
dependent restrictions, restoring the alternatives that remain possible. Reusing an evidence ID with conflicting
content is an error. Snapshots retain the base and restrictions; they contain no executable Python callback.

`select_probe` plans over pure predicted observation models, paying for label evaluations, partitions and branch
expansions. Its bounded minimax tree uses a declared horizon; an incomplete fallback requires conditional relevance.
The selector purchases no actual observation. The host performs and pays for that separately. No probabilities are
invented, and neither optimality beyond the search bound nor truth of the observation model is promised.

## Acquired operations

`Grammar` fixes the input names, constants, allowed int/bool operators, tree limits, maximum candidate count and
integer magnitude. `Program` is a declarative immutable AST. It cannot import code, perform IO, modify checks or
inspect the judge. `if` eagerly evaluates both branches; their work is charged. `synthesize` sees proposal examples
only and returns a bounded-search candidate, not a certificate of generalization.

`ExternalExample` describes an observation supplied by a trusted host adapter. The registry accepts separate
proposal/admission task and event IDs, exact types and matching scope/mechanism/schema. Admission reserves its whole
fresh data package before executing a candidate. An interrupted or rejected package remains consumed. Eval examples
cannot be used for proposal/admission; actual eval outcomes can refute a previously applied operation.

`CompetenceRegistry.predict` records the applied program and its input/context. A matching external counterexample
revokes it and dependent operations. Changing the owner-declared mechanism/schema retires incompatible operations;
switching back to an old version token does not revive a revoked operation. The optional `KnowledgeStore` bridge
preserves outcome provenance and dependent retraction. Sources and IDs are host assertions, not authentication.

## Automatic learning within a finite task stream

`AutoCurriculum(runner, task_ids, proposal_quota=4, admission_quota=4, max_attempts=4)` drives the task stream and
starts learning itself. Assignments are journaled **before** a task runs. It collects action receipts into separate
proposal/admission packets for the declared scope, mechanism and schema; once both quotas are met it calls
`runner.learn`, then continues to unseen tasks. Results from evaluation or already applied operations are never
relabeled as admission data. An external counterexample or a lost/retracted competence opens a fresh packet from
future tasks. Version changes conservatively discard incompatible packets; frequent switches can prevent a packet
from filling. Attempts are bounded, and controller selection, collection and recovery work use the same ledger.

```python
from solvi.experimental.experience import AutoCurriculum

controller = AutoCurriculum(runner, task_ids, max_attempts=2)
results = controller.run()
snapshot = controller.snapshot()
# After reopening the environment and journal, with a registry of the same grammar:
restored = AutoCurriculum.restore(snapshot, environment=env, journal=journal, registry=runner.registry)
restored.run()
```

Use a fresh idle runner to create a curriculum; use `restore` to resume an existing one. `run(max_steps=...)` can
yield before completion. Budget exhaustion and ambiguous effects stop explicitly. Roles and pending learning share
the existing journal without replacing runner checkpoints. A completed learning transaction is not rerun after a
controller crash; an interrupted packet is discarded with its spent work and consumed identities preserved, and
only future tasks can supply a new attempt. Remaining finite tasks may be insufficient to learn again.

The host still supplies the finite task stream, observation model, grammar and trustworthy outcomes. This controller
does not invent open-ended goals or deploy itself as a production daemon. See
[`29_autonomous_curriculum.py`](https://github.com/solvi-ai/solvi/blob/dev/examples/29_autonomous_curriculum.py)
for automatic collection, acquisition of a composite arithmetic operation, unseen application and completed replay.
The example covers a declared branch of a resource mechanism; it is not a benchmark result.

## External loop and recovery

`TaskSpec` contains public inputs, finite alternatives and named `ProbeSpec` observations. `Environment.task` exposes
only that specification. The trusted adapter implements `execute(..., attempt_id)` and `reconcile(attempt_id)`;
the learner cannot issue receipts. Observation receipts name the measured field, and action receipts report the
independently observed correct answer and outcome, preserving failures as failures.

`ExperienceRunner` records intent before external execution, receipt before revision, and the resulting checkpoint.
It uses `ExperienceJournal` over the existing `KnowledgeStore.note` hash chain. The checkpoint binds the ledger,
worlds, registry and phase. Replay is pure. Restore reconciles a pending attempt with the host: an unknown effect
stops unresolved instead of retrying an irreversible action blindly. Exactly-once effects require idempotency or
reconciliation support from the external service. The journal does not supply that support by itself.

`SQLiteEnvironment(path, cases, identity="environment")` is a small durable host adapter for the finite resource
and workflow mechanisms. It commits each effect and its receipt together and deduplicates attempt IDs across
processes. It declares `idempotent_attempts=True`, allowing recovery to retry the **same** durable attempt if no
receipt exists. Other adapters without this explicit capability remain unresolved when reconciliation fails.
Use a separate persistent `KnowledgeStore("knowledge.db")` for the runner journal. Environment snapshots contain
private cases and belong to the host, never to the learner. The test suite kills a child process before/after an
effect and after receipt/fact persistence, then compares recovery with uninterrupted execution.

`WorkBudget` debits integer work units before paid work. Counts preserve the declared category vector; these units
are not dollars or a wall-time bound. Search failures, admission, lookup, program nodes, planning, observations and
actions must use the same ledger. A durable sink records debits before work; restore never replenishes the ledger.
Internal `ExperienceJournal.debit` appends use a constant-time head guard within a pure computation interval;
ordinary intent, checkpoint, replay and recovery boundaries verify the accumulated history before proceeding.

Host callbacks are pure where declared; arbitrary hostile host code is outside this contract. SHA digests detect
accidental corruption, not an adversary capable of rewriting the entire chain and its external anchors. Journal
ordinary append fingerprints its checked history. Full verification at effect/checkpoint boundaries still grows
with retained history; no general memory or runtime bound for the complete journal is promised.

## Evaluation scope

### Full finite-domain admission and tabular actions

The optional `FiniteDomain` contract pins the complete supported input set, strict input/result types,
mechanism/schema, grammar, oracle source and revision. `check_program` pays for manifest processing, program
execution, oracle reads, receipt checks and comparisons. A counterexample rejects; interrupted coverage or an
unavailable oracle returns `unknown`. A successful check is evidence only for that finite contract. It is not
a proof about arbitrary inputs, an untrusted oracle, or an updated source.

`CompetenceRegistry.admit_checked` performs this check itself after separate proposal and admission data.
Public result objects cannot activate an operation. Checked calls require the matching domain and paid membership
checking; missing contracts and inputs outside the domain produce `unknown`. `require_checked=True` excludes
empirical operations. Snapshot v2 explicitly records admission mode; restored v1 records remain empirical.
The registry's `allowed_sources` must include both external example sources and the independent oracle source.
`retract_source(source_id, source_revision, event_id=...)` revokes matching operations and their dependants.

With a `KnowledgeStore`, checked admission requires explicit `checked_store_policy="evidence_only"`.
It stores an `episode` whose writer is the oracle source, rather than publishing an executable generic `skill`.
The registry remains the execution authority. Custom knowledge policies that approve `skill` records are **not**
thereby approving these operations; hosts requiring that approval should retain the default `"deny"` policy.
Withdrawing the evidence writer in the store makes the operation unusable. A commit callback failure poisons
the current registry writer; only trusted durable recovery can continue.

Bind contracts to the complete loop with
`ExperienceRunner(..., checked_domains={scope: (domain, oracle)}, checked_store_policy="evidence_only")`.
The same bindings are required by `ExperienceRunner.restore` and `AutoCurriculum.restore`.
The runner records each read intent before the oracle call and a committed admission before returning it.
Curriculum recovery completes an already committed matching packet without paying for its oracle again;
an interrupted uncommitted packet remains burned. Source changes and durable debit failures fail closed.

`TabularEnvironment(path, csv_path, identity="tabular")` imports bounded CSV tasks and commits each effect and
receipt in one SQLite transaction. It supports stock/reservation/damage arithmetic (`inventory`), a boolean
eligibility rule (`eligibility`), and dependent completion times (`schedule`). `TabularOracle(family)` evaluates
independent fixed SQL rules and constructs the paid complete domain with `.domain(grammar, budget=budget)`.
CSV bytes and SQL task inputs remain pinned on reopening. This is a small local adapter, not an arbitrary SQL
agent. The host must keep the source CSV available. See
`examples/30_checked_csv_operations.py`
for a learned boolean rule, full-domain check, actual SQL effects and completed replay.

Search now reuses paid output vectors; planning caches partitions and prunes losing branches. Cache operations
are themselves paid. Runtime program evaluation prices remain unchanged. These changes can reduce search work,
but small problems may cost more; there is no general performance claim from component tests.
Expressions at the maximum permitted node count are rejected on their first mismatching example; they cannot
serve as children of any later valid expression. Smaller expressions keep complete vectors for reuse, and a
matching candidate still requires all examples. This changes search work, not runtime eager evaluation.

The accompanying benchmark compares finite resource and workflow mechanisms, separate proposal/admission/test data,
mechanism changes, full work budgets, explicit hand-written formulas/caches, observation-only controls and a no-call
ablation that still pays for learning. Its encoding control renames probe identifiers only; it does not establish
invariance to action identifiers or serialization order. The component benchmark does not measure the scheduling
policy of `AutoCurriculum`. Its authored finite generators are not an external benchmark. Cost improvements,
scientific verdicts and failures must be reported from the frozen experiment; successful replay alone establishes
none of them.

A separate CSV/SQLite component stand covers inventory, boolean eligibility and dependent schedules. It compares
pinned previous and current search, full measurements and known formulas with the complete declared work budget.
Its wire audit changes input/result/model fields, task/event/attempt/probe identifiers and serialization order.
This evaluates component contracts rather than `AutoCurriculum` scheduling. The mechanisms are authored small
finite rules, not an external benchmark. Full-domain oracle checks include the transfer inputs: later correct
application is not evidence of prediction on unseen oracle labels. Search-cost savings, checked success rates
and the cost of the complete strategy must be reported separately.
