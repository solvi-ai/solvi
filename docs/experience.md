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


## Counterexample-guided learning and useful observations

`RefinementSession` adds a bounded iterative learner on top of the same grammar, program evaluator,
finite-domain checker and shared work ledger. Each round proposes a program, then either buys an observation and starts a new proposal round,
or checks the candidate. Only the first counterexample is added to training before the next proposal.
Other verifier receipts remain audit evidence rather than extra training labels. The grammar remains fixed;
search can fail even when a solution exists. `max_rounds`, `max_observations`, committee size and the total
budget bound the work. A result with `status="verified"` covers only the declared verification domain.
It is not a registry entry, a promise about new inputs, or authorization for external effects.
Use the existing `CompetenceRegistry.admit_checked` with a candidate proposed or derived by that registry
and independent admission examples to authorize a checked operation; that path performs its own check rather than trusting a session result DTO.

`ObservationOption(name, inputs, cost)` declares one allowed input and its read price. The immutable option
contains no label. `select_observation(programs, options, grammar, budget)` pays to execute candidate
predictions and count typed pair disagreements, then selects the greatest disagreements/read-price ratio.
Ties follow host order; the selector requires the read to remain affordable after planning. Invalid execution is a separate marker; bool and int outputs remain distinct.
No disagreement means no purchase. The host pays for the actual read separately. This is a heuristic over
at most 32 supplied programs and 256 options, not an optimal information-gain guarantee.

`synthesize_committee(examples, grammar, budget, size=3, max_additional_work=None, reserve=0)` collects
distinct matching programs in one enumeration, reusing paid intermediate vectors. It returns
`CommitteeSearchResult(candidates, tried, reason, continuation_spent)`. Candidates retain enumeration order;
a budget, candidate or continuation limit preserves the completed prefix. The first match uses the ordinary
parent search budget. Only work after that match counts toward `max_additional_work` and must preserve
`reserve`. There is no promise that the first search leaves that reserve. `synthesize` keeps its original
first-match semantics; `exclude` still accepts at most 32 fingerprints. Neither search grants competence authority.

`select_observation_bounded(models, options, budget, predict=..., max_work=1024, reserve=0, result_type=None)`
is a separate backend-independent selector. The host supplies a pure
`predict(model, inputs, paid_budget)` callback, which charges its own prediction work before execution.
It receives no oracle or labels. A Program adapter can call `program.evaluate(inputs, paid_budget, grammar)`;
the host wrapper also owns input schema and bounds validation. External pure models need no library backend.
The same parent ledger pays selector metadata, comparisons, cache operations and predictions. Their total debit
is limited by `max_work`, while retaining `reserve`; observation prices are compared but never purchased here.
At most 32 models and 256 options are accepted. A one-model committee makes no predictions.

Typed bool and int outputs differ. Without `result_type`, invalid predictions share a separate marker.
An explicit `result_type=bool` or `int` is a strict host promise: a prediction `ValueError` or type violation raises
`ValueError`. This promise is not authentication of arbitrary host code. The selector can stop when the best
completed score/price cannot be improved by any unvisited row: the separated-pair bound is
`floor(k*k/4)` for the bool contract and `k*(k-1)/2` otherwise. Equal scores retain the first host option.
An interrupted row is discarded; fully scored rows remain eligible. A work cap can return
`reason="selected_partial"` with the best still-affordable completed row. Without an eligible row the reason
is `budget` or `no_disagreement`. A post-debit persistence failure propagates and is not a cheap budget refusal.
The original `select_observation` remains available with its previous full-scan behavior and prices.

Session modes are `none` (counterexamples only), `fixed`, `random` and `active`. New active sessions pin
`active_policy="onepass-bounded-v1"` and `active_work_limit=1024` in their configuration. The constructor accepts
a strict integer limit from 0 through 100000. Additional committee search and bounded selection share this
window; the essential first search, host filtering/validation and actual label purchases are outside that window
and still consume the parent ledger where charged. This is a core-planner work limit, not a total-session or
host-runtime bound. Planning reserves a conservative complete-domain check plus the largest unused read fee.
If extra candidates or selection do not fit, the completed candidate proceeds to checking. Buying a label does
not promise enough budget for the next essential search. Fixed/random purchase up to the same observation quota;
active can stop buying sooner. Comparing these policies does not isolate point choice from stopping.

Pass `journal=ExperienceJournal(...)` for durable operation. The session binds its grammar, finite domain,
source revision, options, caps and mode. It records each debit and saves a read intent before calling the
host oracle. `RefinementSession.restore(journal=..., grammar=..., domain=..., oracle=..., options=...)`
reopens the original bound state and preserves subsequent debits. It pins the recorded policy and work limit;
restore has no cap override. A legacy checkpoint without `active_policy` retains repeated synthesis with excluded
fingerprints and the original full selector, rather than silently migrating algorithms. Unknown policy versions
are rejected before oracle use. An observe-phase checkpoint retains the remaining planning window; later durable
debits reduce it on recovery. A first candidate is checkpointed before any optional committee continuation work; after a crash,
recovery keeps that candidate and deducts later durable continuation debits instead of replaying synthesis
with a fresh planning window. An uncertain read is never automatically
reissued: `oracle.reconcile(request_id)` must return its original `DomainReceipt`, otherwise the session
stays blocked. Reconciliation itself must be a side-effect-free host lookup. Source changes stop the session.
A durable session owns its debit sink and rejects a preexisting `WorkBudget.on_charge` callback before
writing state. Arbitrary callback recovery is unsupported. Pure interrupted computations may be repeated and paid again; a recovered receipt is not repurchased.
A debit/persistence sink failure poisons the current writer. Malformed reconciliation receipts raise an error
without training. A completed result is historical: replay does not revalidate a changed live source and grants
no current execution authority. `journal=None` is explicitly volatile and
provides no crash-durability guarantee. Journaling overhead and retained history need separate resource budgets.

Keep initial observation inputs, selectable observation inputs, verification inputs and closed transfer
inputs under host control. The API does not sandbox a malicious host. Do not expose transfer inputs or labels
to training or observation selection. Applying a frozen program outside its verification domain is empirical
prediction; it must not be presented as certified execution.

See `examples/31_counterexample_learning.py` for real SQLite receipts and knowledge journal, a counterexample
that causes a new proposal, an active paid observation, and completed-state recovery. The separate
`benchmarks/refinement_bench.py` evaluates twelve authored finite mechanisms across numeric tables, boolean
policies and multistep numeric dependencies. Its independent scorer checks held-out predictions and complete
budgets. It does not measure arbitrary strings, external real-world tasks, distribution shift or general
scientific superiority; the known-formula and direct-measurement controls have privileged access disclosed
by the benchmark protocol.


The benchmark is configured by a host-owned protocol file, supplied with `--protocol PATH`.
Final execution also requires `--final-claim PATH` naming one stable claim shared across output directories.
A host-authored `FINAL_APPROVAL.json` must bind the manifest digest, frozen dev raw digest, selected comparator,
absolute `output` path and absolute `final_claim` path. Copying an approval to a different output or supplying
another claim path is rejected before starting an actor. An interrupted claimed trial blocks for investigation;
completed trials can be reopened without replay. The source manifest pins actual module origins, benchmark/scorer,
project metadata, dependency lock and protocol contents. No private workspace layout is required.

See `examples/32_bounded_observations.py` for a tiny known-answer comparison of ordered committee fingerprints
and observation choices. It prints actual work-unit debits, not an asserted general speedup.

### External-data acquisition protocol

The separate [`active_real_bench.py`](https://github.com/solvi-ai/solvi/blob/dev/benchmarks/active_real_bench.py)
entry point supports a host-pinned paired component comparison and an external-data classification protocol.
Its source names are `banknote`, `spambase` and `wine_red`, referring to UCI Banknote Authentication,
Spambase and red Wine Quality. These are empirical classification tasks; noisy labels and finite held-out
accuracy do not provide a finite-domain certificate or competence authorization. The benchmark uses a small
host-owned learner rather than adding a training dependency to the library core. Source files, licenses,
checksums, preprocessing, splits and protocol are pinned by the host before comparison. Acquisition receives
only permitted pool features and paid training labels, never held-out labels.

All arms use declared work budgets and simulated label prices. Work units are neither currency nor measured
human annotation cost. Full accounting includes preprocessing, training, predictions, selection and purchased
labels; wall time and resource use must be reported separately. The CLI requires explicit `--protocol`,
`--baseline`, `--baseline-revision`, `--data`, `--output`, `--claim` and `--audit-wrapper` paths/identity;
`--python` selects the interpreter. Preparation, smoke/dev, freeze and final execution are separate stages.
These APIs and the runnable example establish no accuracy gain, general cost advantage or external deployment result.

### Paid label-aware observation plans

`plan_stratified_observations(examples, options, budget, limit=8,
candidate_limit=64)` accepts a bounded tuple/list of 1–64 boolean proposal
examples in one host context and an independently shuffled ordered pool of at
most 256 unique `ObservationOption` rows. Inputs use contiguous `f0`…`fp-1`
integer bins 0–7, with at most 128 features. Only the first bounded candidate
prefix is decoded; its shape must match the pilot even for fallback selection.

The planner computes each upper median using eight-bin histograms. Features
with at least four pilot rows on each side are ranked by the exact absolute
difference of Laplace-smoothed positive fractions, with numerical feature ties.
Up to two nonzero features define cells; round-robin selection preserves the
host order inside each cell. One-class or uninformative pilots return the host
prefix with `reason="random_prefix"`. Planning phases debit the shared budget
before their work; exhaustion and debit-sink errors propagate without a partial
success. No environment reads or oracle calls occur.

Observed labels remain host assertions. A plan neither authenticates evidence
nor admits a competence, grants read authority, or supplies a certificate. See
`examples/33_stratified_observations.py` for the separately paid planning step.

## Packed host observations

`PackedObservationFrame(records, budget, source="host")` snapshots 1..128 named
rows with 1..128 contiguous `f0`..`fp-1` integer bins (0..7). Boolean bins are
rejected. Its `names`, `features`, and `values` are immutable tuples; `index` is
read-only. Input dictionaries can subsequently change without changing the frame.

`PackedStratifiedSampler(frame, pilot, budget, strategy="round_robin",
cut_mode="all", max_steps=8)` accepts 1..64 unique `(name, bool)` pilot labels
and 1..64 remaining candidates. A call to `choose(budget)` creates one pending
ID; `observe(name, label, budget)` must supply matching strict boolean feedback
before another choice. Selection stops after `max_steps` (1..16). No oracle,
registry, authentication, or implicit label read is involved.

Pilot histograms choose up to two different features using absolute differences
of Laplace-smoothed positive rates, requiring at least four pilot rows on each
side. `all` examines thresholds 1..7; `median` uses the upper pilot median.
Equal scores prefer lower feature indices and lower thresholds. The first
feature is the least significant cell bit. One-class or ineligible pilots use
candidate order. Cuts stay fixed; observed labels update cell counts.
Round robin advances its cursor only on observation. Adaptive selection uses
exact rational `r*(a+1)*(n-a+1)/((n+2)^2*(n+1))`, preferring lower cell codes
on ties (`r`: remaining candidates, `n`: labels, `a`: positive labels).

`to_dict()` returns a native snapshot with tuples. `restore(frame, saved, budget)`
also accepts lists after a JSON IO boundary, checks the frame directly and
replays choices and supplied observations. Restore debits the current budget
again without refunds. Snapshots are host assertions and can be forged; they
are not authenticated evidence. All work groups are prepaid and debit/sink
exceptions propagate. See example `34_packed_observations.py`.

## Verified computation, selective updates and acquired operations

`ComputationPlan` describes a bounded, topologically ordered DAG: up to 32 input names,
128 scalar nodes and eight alternative producers per node. `ComputeNode` names ordered
source/dependency ports and producer alternatives. `ComputeOperator` binds a producer,
a separate verifier, their joint revision and their declared work prices. Values are
strict integers or booleans. `VerifiedComputation` executes only ancestors of requested
outputs. `fifo`, `cheap` and `critical` select ready, affordable attempts; the latter is
a fixed goal-contribution/cost heuristic, not learned or optimal metareasoning.

Every attempt prepays both execution and verification, including failed attempts. Only
strict `True` accepts a scalar proposal. An ordinary callback exception rejects that
alternative; another may be attempted from the remaining budget. An unknown interrupted
callback or failed persistence closes the session. `observe(name, reader, cost, revision)`
prepays an explicitly requested source read; an untrusted observed value does not bypass
the downstream output verifier. Use `step()` and its `missing` ports to drive a host read
loop. There is no implicit network access, external action commit or automatic oracle.

`update_inputs(values, revision)` invalidates the affected reverse dependency closure,
including a change from `1` to `True`. Disjoint accepted values survive. Producer or
verifier changes require a new operator revision via `update_operators()`. Optional
operator validity hooks are checked on live cached values; revocation removes their
results and descendants before another answer. These inspections and invalidations
consume the same ledger. Reading outputs can therefore exhaust the budget.

For durable execution give the session an `ExperienceJournal` over a persistent
`KnowledgeStore`. Reopen the same journal/configuration and construct
`VerifiedComputation` with the same operator versions and budget limit. Recovery checks
monotonic ledger history and independently re-verifies cached values for a paid fee.
Spent work is retained. A pending callback without a confirmed result blocks recovery;
this API deliberately does not repeat an unknown side effect. Writers must be serialized.
Snapshots describe state; they are not standalone authority to resume or authenticate
rewritten history. Callbacks, version labels and the journal writer are host trusted.

`OperationCatalog` acquires whole parameterized `Program` templates from accepted node
traces. `record_session()` extracts accepted bindings using host-supplied node/Program
metadata. At least two distinct bindings are needed. `acquire()` uses the existing
`CompetenceRegistry.propose_program` and `admit_checked` path: separate admission
examples and an independent oracle must check the declared finite domain. The callback
is then compiled to a bounded postfix interpreter, without Python `eval` or `exec`.
`operation()` returns a normal versioned `ComputeOperator`, usable as a node in a new
composition; each call still pays its input/AST work, guards and independent live check.
A counterexample revokes the operation. Finite admission does not certify other inputs.

This catalog does not invent a semantic specification or perform unrestricted subtree
search. The host asserts that Program metadata describes its accepted computation;
independent admission tests that assertion. Store-bound registries are not supported.
Callable serialization is unavailable. Save `catalog.snapshot(budget=budget)` alongside
the existing registry snapshot and the current ledger. After restoring the registry,
`OperationCatalog.restore(snapshot, registry=registry, budget=budget)` checks the saved
candidate/admission/domain bindings and recompiles the callable for a paid fee, without
another oracle purchase. Supply the same independent verifier and `verify_revision`
to `operation()` before reopening the graph journal. The budget must retain its saved
categories and all additional reconstruction work; a monotonic extension is accepted,
incompatible ledgers are rejected. The host must persist these coordinated snapshots
and durably record reconstruction debits. A growing registry requires rebuilding
existing operation wrappers. This is a limited executable vocabulary, not automatic general algorithm
invention. See `examples/35_verified_computation.py` for the complete connected path.

Declared units constrain prepaid work, not arbitrary Python runtime, memory or truth.
A separate function is not intrinsically an independent or reliable check. Supply an
independent specification/reference and use an external final scorer when evaluating
quality. The component does not modify the calibrated fast/slow `core.Dispatcher`.


## Decision value, cooperating solvers and interactive authorship

Four bounded experimental directions extend the finite runtime:

- [Decision-loss learning](decision_learning.md): train a small ranker from full outcomes, separately validate and explicitly activate; evaluate known-propensity one-step logs with IPS/SNIPS/DR.
- [Cooperating partial solvers](cooperation.md): share independently checked exclusions, incumbents and bounds over an immutable finite problem.
- [Teaching executable concepts](teaching.md): examples/exceptions, distinguishing questions, finite-domain admission and version-pinned composition.
- [Preference elicitation](preferences.md): exact minimax regret, costed questions, explicit conflicts and proposed soft relaxations without weakening hard requirements.

These modules share WorkBudget and keep proposals distinct from accepted evidence and external permissions. No foundation model is required or implicitly trained. A running integrated example is `examples/40_decision_workflow.py`. Bounded correctness and engineering readiness are separate from empirical superiority; consult each module's limits.
