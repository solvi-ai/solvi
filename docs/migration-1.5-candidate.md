# Breaking development changes: candidate migration notes

These changes are local unreleased development work. They do not announce a release or change the package version.

## Conditional checks and source accountability

`System(check_policies=...)` adds opt-in, recorded adaptive acquisition. Each request supplies `check_loss`;
existing calibration/guards must not be reused for acquisition. `check_intervals` may report conditional
uncertainty or use a lower score for decision thresholds. These are descriptive model calculations, not new
coverage or false-accept certificates. See [check acquisition](check_acquisition.md).

Check-combiner fingerprints now include fitted Dirichlet concentration state. Reconstructed Systems therefore
have new check-model identities. A previously fitted combiner without concentration state still supports point
prediction; refit it from its calibration observations before requesting uncertainty draws. Frozen snapshots
cannot be refitted. Recalibrate identities rather than relabeling old traces as decisions of a changed model.

`SourceTrust` and opt-in `TrustGate` provide windowed source accountability without changing SourceGate or
WriterGate defaults. An explicit trust refusal now refuses facts and behavior-changing items; other custom
measurement gates retain their existing hold semantics. [Source trust](source_trust.md) describes required
identity authentication and independent labels. [Failed-world policies](failure_policies.md) add explicit
environment-failure vetoes without changing the default score-only stage policy.

## Accepted answers, condition text and external names

An answer head returning nonfinite probabilities now leaves a finite, explicit failure record. Acceptance can be
restored without invoking the head, and independent replay verifies that failure. Recorded probabilities and
confidence are checked for consistency even with trusted models. Stateful guards retain their historical metadata
when a trace is re-derived after later learning; their current configuration fingerprint remains checked.
Record-free candidate evaluation reads the current guard threshold without advancing its observations or auto-grid.

Guard sessions now support `check_many(calls, advance=...)` / `call_many(...)` and async counterparts.
Every call is checked against its validated arguments before any registered tool function starts. A denied or
escalated member prevents the whole batch from executing. `advance(facts, decision)` returns predicted caller facts
after an allowed member, for example an accumulated budget; it receives private copies. No cost is guessed from an
arbitrary tool. The full plan and member index are bound into each decision's trace. `check_many` returns a dictionary
with `allowed`, `decisions`, `executed`, `handed_off` and `runtime_error`; `call_many` fills the execution fields.

A Session must have one owner, or access must be serialized by the application. This is preflight for one batch,
not an external transaction: runtime error or cancellation may leave earlier effects. An allowed framework tool is
handed off, not confirmed as externally completed. Policy/authorizer/advance callbacks may themselves perform I/O;
the no-execution property concerns registered tool functions. A rejected preflight leaves session history and
storage unchanged.

For a tool with explicit confirmation, a user's acceptance of a proposal naming the current argument values can
discharge the soft tool-taint check for those confirmed fields. The structured taint evidence remains recorded.
Subset confirmation does not discharge other fields; missing/user-only grounding, scan-user taint and
`injections="any"` are unchanged. This restores confirmation semantics when the improved lexical detector
recognizes a previously missed instruction.

Instruction detection retains its lexical scope: conventional invoice payments and website-for-details lines
can be document data, while explicit assistant-directed messages and task prerequisites are flagged in tool-output
mode. Unicode normalization and original span offsets are preserved. User-message mode (`actions=False`) retains
the existing override/role rules. This is not a calibrated classifier or proof that document data is safe;
provenance restrictions still govern tool argument authority. The original wider detector research request,
including an independent labelled corpus, is not declared complete by these bounded regressions.

TicketDesk now deduplicates identical serialized inputs while their ticket is open, including across restart
and new request ids. A repeated request returns the original ticket without running System 1 or drawing another
coin. Legacy duplicate tickets resume oldest first; resolving/expiring it reveals the next outstanding one.
Completed or expired inputs are new decisions unless an original request id is reused. Set
`deduplicate_open=False` only for independent per-call decisions; this restores the open-input re-probe side channel.
This is a breaking change to repeated-input accounting, not a whole-interaction blindness guarantee: modified
inputs and other access to System 1 can reveal information. Labels must remain sampling-independent.

`Catalog.answer_fact(question, name=...)` declares a fact holding the accepted, normalized answer of another
question in the same System. For a bool rule this is `"yes"` or `"no"`, not the raw bool. Sources run before consumers;
shared computations run once per request. Source hard checks, contracts, confidence thresholds and guarantees
apply before publication. An abstention becomes a missing fact carrying the source reason; `Unknown` remains a
valid distinct answer. The reference's confidence bounds consumers. Structured source field metadata is retained
in the hashed reference record. Raw inputs and probes cannot override an accepted-answer fact.

The source question must be declared. Cycles, including a reference within a joint-constraint component, are
rejected. Reference records bind the source Result and checked prefix and are independently verified on replay
with the System. Restoring acceptance from records does not invoke a missing head/then callback: incomplete records
produce an abstention. Input schemas and `depends` include transitive source inputs and applicability declarations.
Explicit head training may select accepted-answer features; default feature inference excludes answer references
to avoid training on the head's own answer. Ordinary requests without references keep their existing executor path.

```python
@cat.rule('payable', text='amount > 0')
def payable(amount: float) -> bool:
    return amount > 0

cat.answer_fact('payable')

@cat.rule('go')
def go(payable: str) -> bool:
    return payable == 'yes'
```

`Catalog.rule(..., text='condition')` adds an author-declared condition to `why` alongside the actual input values;
the text participates in the catalog fingerprint. It is not inferred from Python source and does not certify that
the condition is true. `because(...)` still supplies the reason for the selected branch.

`System.rename(inputs={'amount': 'total'}, answers={'go': 'proceed'})` returns a new System with external names,
adapted contracts, spans, checks and constraints; the original System is retained. Execution statistics and storage
start fresh. Trained heads and guard promises are shared objects, not independent trained copies; question-bound
guard wrappers are adapted to the new names. An input rename on a
System with `input_model` is explicitly refused: provide an external-field model and `Catalog.alias` instead, so
custom model validators are not silently discarded. A custom strategist's own hardcoded names remain its owner's
responsibility. Rename transfers preserve the original selection of automatic checks; ordinary `Catalog.alias`
continues to declare a computed fact.

## Knowledge identity and readers

`Knowledge.branch()` and `KnowledgeStore.branch()` provide isolated writable what-if copies. Writable
`rebuild()` now also copies its policy; a gate holding noncopyable resources must supply a safe independent
copy before it can be branched. Ordinary read-only `verify()` does not require copying the write gates.
Callback closures and external state are not copied. Persistent reopening still needs explicit Python policy,
writers, vocabulary and callbacks; inherited verified decision proofs are not exported automatically.

`KnowledgeStore.note()` now rejects reserved journal metadata (`n`, `kind`, `op`, `clock`, `v`, `seq`, `id`,
`time`, `prev`, `hash`, `run`) before writing. Use a distinct payload name such as `episode_n`; old recorded
notes retain their data during replay. The internal logical cursor is separate from note payloads.

The default belief fingerprint is `km2:sha256:<64 hex>`, a canonical Merkle root. The journal's hash chain and item IDs
retain their existing formats. Loading the same journal rebuilds the root; no history rewrite is required.
`store.legacy_fingerprint()` computes the previous 16-character belief fingerprint when auditing older records.
An old belief fingerprint and the new fingerprint are different identity formats. A replay or a saved calibration
that binds the old identity must not silently treat them as interchangeable. Rebuild/revalidate the consumer under
the new identity before using it for new decisions; retain the old environment for reproducing an old calibration.

The tree uses one node per present belief and expected logarithmic updates for changed belief rows. Resolving a large
connected component can still require visiting that component. This is not a constant-time guarantee for every write.
Use store methods to change beliefs; mutating internal `items` or `state` dictionaries bypasses resolution and indexes.

`Knowledge.value(snapshot, subject, relation)` raises `ValueError` when matching facts contain distinct values.
Pass `scope=` and/or `valid_at=` to select the intended facts, or call `Knowledge.values()` for all distinct matches.
Snapshots include scope so readers can make the selection explicitly. Equal duplicate values are not ambiguous.

## Law defaults

The default support population is `matched`: evidence includes observations the proposed law matches, including
ones a narrower law currently handles. `falsify="auto"` uses leftover probes for the supported exact single-law path;
noise and two-law admission keep their separate procedures. These checks cover a finite pool, not all unseen inputs.
Explicit `support_on="applied", falsify=None` retains the previous admission policy when it is deliberately needed.

## Coding hooks

The hook CLI does not record file content or prompts by default. Audit/replay of hook content requires explicit
`--record-content`; then full edited files, including pre-existing secrets, and prompts may enter the local store.
`--no-store` overrides recording. Passing a `store=` to the Python API remains an explicit request to record.
Old records are not deleted by changing this default.

The deterministic skill picker requires two distinct shared stems by default. `--min-shared=1` restores the previous
heuristic. Installation in HOME is refused. Obsolete `pre-edit --model` produces a structured ask/deny; reinstall with
`--decider` to use a model. Install/uninstall preserves files modified by their owner after installation.

## Stream ownership and retries

Close a `TicketDesk` before opening its persistent store from another desk, or use its context manager. A restart
does not mean two simultaneously live owners. CLI servers close their desk on exit. A timed-out client should retry
`decide` with the original `request_id` and the same state, or recover outstanding work with `open_tickets`.
Use `give_up` when a verdict is unavailable; missing outcomes retain conservative failure accounting.

## Checks and finite diagnostics

Protocol-only deciders no longer need private registration markers or a function name. Conformance calls actual
scorers, checks multiple/interleaved inputs and rejects input mutation. Finite checks do not prove future determinism.
An identity canary can return `unconfirmed`: an initial difference followed by a matching confirmation is not called
`same`. Probe diagnostics describe only the configured probes, not general provider-change detection coverage.

## Chart proposals

`chart(text)` accepts explicit nonempty `label: value [scale/unit]` rows. Unsupported residue or multiple RHS values
refuse the whole default proposal; natural-language releases no longer have their labels guessed automatically.
Choose an explicit recipe/proposer for that task. `RuleProposer(allow_prose=True)` retains the previous heuristic;
its output, like arbitrary `FixedProposer` output, is verified numerically against quotes, not proven semantically.

## Guard effects

Effect declarations belong to trusted application code. `effect="egress"` with `destinations=("url",)` requires a
full destination URL from the current authorized user request or trusted `approved_urls`. Prefix authorization is
insufficient. Host, scheme, path, query and fragment remain significant; only the corresponding scheme's default
port is an alias. Remote MCP read-only annotations cannot grant this authority.
`Guard(require_effects=True)` escalates unknown effects. The default compatibility mode keeps existing tools working
but records unknown effects as unverified. This boundary checks declared destination URLs; it does not observe
actual I/O, classify every tool automatically, inspect arbitrary body/header payloads, or replace context taint.

## Command discovery

`solvi --help` shows init/test/ask/serve/check/models. Use `solvi --all-help` for the full command list, or
`solvi COMMAND --help` for a specific command. Existing secondary commands remain available by name.

## Applicability input schemas

A callable `applicable` predicate reads the initial state before computing the question. Declare its dependencies:
`catalog.contract("answer", applicable=predicate, applicable_inputs=("enabled",))`. Use an explicit empty tuple when
it needs no input. Schema export cannot infer all reads of arbitrary Python and refuses an undeclared opaque predicate;
an explicitly configured `Question.uses` remains a compatibility fallback. Declared facts are required initial inputs,
not requests to compute catalog parts. Local `System.ask` keeps its existing applicability semantics.


## Base-dependent law exceptions

S235 adds an explicit opt-in `lapse_on_base_change=True, base_version="edition"` and
`set_base(base, version="new-edition")`. Default laws retain their absolute meaning. In the bound mode,
old active/suspended exceptions retire on a new edition; compile and Knowledge exports refuse until they
can preserve the dependency. Update the declared version for mutable bases. A base change invalidates the
law-set version and memo; it does not refund acceptance accounting.

## Structured answers

`Answer.structured(type=dict)` and inferred dictionary/list/model/TypedDict rule answers are now question results,
with optional field explanations in `Claim.extra['fields']`. See [the contract](api/structured-answers.md).
Closed Literal/Enum collections retain their previous behavior. Structured schemas require string object keys and
serializable types; local classes are refused. Return annotations remain enforced separately from question schemas.

`Answer.value()` explicitly accepts any finite JSON value, including scalar strings, booleans, numbers and `null`.
Use it when a rule's `None` is a real answer: with ordinary answer types, `None` still means abstention.
Non-JSON objects, non-string object keys, NaN and infinity are refused. Declared rule return types remain enforced;
`Unknown` still requires `Answer.maybe(...)`. Structured arrays retain their list shape after JSON restoration.
This API does not trace arbitrary Python functions or record their internal execution.

In decision test cases, `expected: {question: null}` still requests abstention. To test a literal JSON `null` or
the string `"abstain"` with `Answer.value()`, also set `status: {question: "ok"}`. Value comparisons distinguish
JSON booleans from numbers and preserve literal strings such as `"<not stated>"`.

`solvi.trace(fn, storage=store)` now captures eager synchronous JSON calls as native `kind="call"` records.
Replay, call diff and native-case export are explicit helpers; see [function call records](api/calls.md).
This does not capture generator consumption or replace the decision `cases.json` format.

`System.decide` is an explicit [unrecorded decision entry point](api/unrecorded-decisions.md), with strict configuration
checks and a separate result type. Use `ask` for recording, guards, models, calibration or adaptive systems.

`Knowledge.redecide` adds an explicit current-knowledge, per-input-date route; see its
[snapshot and callback contract](api/solutions_knowledge.md). Historical counterfactual removal keeps its default.
The new agreement diagnostic in `solvi.stat` is descriptive and does not automatically weight checks or certify them.

## Intermediate-fact probes

`System.probe(question, facts, override=True)` tests the real rule at explicit intermediate fact values. Results and
traces are durably hypothetical; no calibrated guarantee is applied. Checks/answer rules cannot be replaced.
The initial fact name `__solvi_hypothetical_probe__` is reserved. See [probe semantics](api/system-probe.md).

## Fact aliases

`Catalog.alias(internal, source, type=..., description=...)` adapts one fact name at the boundary; consumers retain
original signatures. Type inference requires one common existing consumer type; otherwise declare `type=`.
Supply external/source keys to ask and input models. Question/answer renaming is not implemented by this adapter.

`System.depends(question)` reports initial inputs of the declared flow, including aliases/checks/applicability.
It reuses input-schema planning and refuses undeclared opaque applicability reads; it is not causal analysis.

## Joint repair confidence

Joint decoding retains the evidence path when it changes an answer: rule outputs are capped by path confidence,
learned-head class probabilities are multiplied by it, and text entry-point confidence remains a cap. A repair cannot
cross `min_confidence` merely by discarding a low-confidence fact. Revalidate persisted calibration/thresholds when
upgrading a system that relies on joint repair. This change does not establish calibration of arbitrary probabilities.

Calibration now follows the final joint selection: the selected candidate's path-aware score receives the existing
Platt map, then independent fact and text-entry bounds are applied. The old answer's calibrated confidence is not
replaced by an uncalibrated class probability. Existing systems using joint repair should refit and revalidate their
calibration/guarantee under the new runtime; retain the old environment to replay old confidence values exactly.

## Required sampling declaration

`StreamPromise.learn(signal, answer, label, floor=None, *, sampling_policy)` now requires a sampling declaration on every call. Omission raises `TypeError` before the certificate reads or changes anything. Restoring a checkpoint does not supply a sampling policy for subsequent calls.

Use `sampling_policy="uniform"` only for an input-independent audit coin whose rate can depend on the past. Use `sampling_policy="complete"` when every decision in the certificate's population is labelled. Missing labels still count as losses. Input-dependent sampling is rejected before any evidence or delta budget changes; this certificate has no IPW/DR support. Separate per-slice certificates need independent uniform sampling within each slice and separately allocated delta budgets.

Dispatcher and StreamLoop supply declarations for the sampling paths they own. Manual callers must establish their actual sampling design; adding the word `uniform` does not make a slice-dependent coin valid. This API validates a declaration, cannot detect a dishonest caller, and does not solve adversarial re-probing (S334). Grid construction, statistical thresholds and label floors are unchanged.

## Trusted exact claim reassertion

Knowledge(..., writers={"wikidata": "outcome"}, reassert_writers={"wikidata"}) enables an opt-in capability. `tell(..., reassert=True)` accepts only a direct fact with exactly the same writer, source, content, scope and full validity as an assertion previously withdrawn by that writer with `assertion=`. Ordinary tell and the default empty capability retain pending-retest protection. Unsupported reassertions raise ValueError before any journal write. A registered writer is an application-bound identity, not a name copied from untrusted input.

This claim-scoped journal proof does not call retest, promote old claims, clear foreign copies, or waive external/global withdrawals, retract_source, rollback, forget, failed retests or active flags covering the item. Writer-wide `only_from` withdrawals are conservatively unsupported. Derived and consulted assertions cannot reassert. Replayed accepted proofs retain their historical authority even if the current process does not grant the capability for new writes. Applications must authenticate the adapter and grant this authority deliberately; it is not evidence that the upstream source is factually correct.

## Opt-in provider identity leases

`DecideModel(provider, identity_safe=True, identity_lease=provider.lease)` requires a provider-owned context manager that pins the entire deployment/router generation for the whole operation, including cache hits. It must yield a nonnegative monotonically increasing integer. No built-in remote adapter currently supplies this contract; an application must implement and authenticate it at the provider boundary.

Unknown, changed, invalid or unavailable identity fails closed and invalidates cached decisions and adaptations. Persisted calibration bindings are audit provenance and require fresh lease-bound Part calibration. Checkpoint-level calibrated metadata never recovers through Part calibration: start from an uncalibrated scorer. Unsupported model/Part mutation APIs reject before scoring or writing; combinations abstain at decide/call and reject state/final/fitting/calibration APIs. Direct public dictionary or attribute mutation is outside the contract. See [Provider identity leases](provider_identity_leases.md) for the exact supported boundaries. Default behavior remains unchanged; this does not establish detection of hidden provider swaps outside the lease contract or global sensitivity of fixed probes.

## Adaptive certificate reference re-takes

`Dispatcher(promise=..., monitor=...)` now requires an independent labelled-risk `detector=` when the monitor
provides `adapted()`. Certification, loosening and restart re-take that monitor's reference because routing changes
its share answered alone. A simultaneous confidence/distribution shift can enter the new reference and escape that
monitor. The constructor rejects this composition before the first decision when no label detector is provided.

For legacy applications that explicitly accept this blind spot, pass `allow_reference_retake=True`. Only actual
`bool` values are accepted; strings such as `"false"` and integers are rejected before constructor work. The enabled
waiver is included in the dispatch configuration fingerprint; the default False adds no fingerprint field, keeping
the earlier configuration format. Reconstruct an opted-out dispatcher with the same explicit True setting for
matching configuration fingerprints during audit/replay.

A `RiskDetector` continues across certificate threshold moves and monitors sampled deployed risk using accepted
slow-path labels. Its expected false-alarm run length is not a per-horizon probability, and its delay depends on
sampling and label availability. This guard does not prove detection of every confidence shift: a shift that leaves
labelled deployed risk unchanged can still evade the reference re-take. The existing confidence thresholds and
false-alarm calibration are unchanged. This is a bounded composition guard, not closure of the historical CLINC
false-alarm/shift benchmark.


## Measured build violations and hard-check reachability

`solvi.build` now defaults to `strict=True`: a violation reported by its own dispatcher calibration raises
`ValueError`. Explicit `strict=False` produces an uncertified diagnostic build, with automatic answering latched
off and requests sent to a person. The original measured report is retained. Editing that report does not enable
automatic answering; construct a fresh successfully calibrated build.

`System` rejects known unreachable hard checks by default. Declare the scope through `Question(..., requires=...)`
or `check(..., then=...)`. For computed-fact checks, request-specific missing dependencies are checked before
callbacks and cache hits. Explicit `allow_unreachable_hard_checks=True` preserves the legacy skip with a warning and
lint error; it is a diagnostic waiver, not evidence that a check executed.

## Law admission and numeric grounding

Automatic exact falsification now also checks the full chosen pair on unused probes before committing it. Noisy
law admission requires an explicit `falsify=None` acknowledgement of its conditional assumptions; the default no
longer silently selects that weaker path. Every attempted oracle call is charged, including calls that raise.
Refinement that needs negated proposals requires a buffered mismatch, and fails before proposal counters change
when that evidence is absent.

Ambiguous repeated numeric source occurrences now abstain instead of selecting the first matching number. Unique
label support can distinguish an occurrence, including an immediately following line; unrelated units and labels
are not borrowed from elsewhere in the document. This establishes source provenance, not general extraction accuracy.

## Past grids, acquisition and availability

`StreamPromise.prime_grid(scores, source=...)` fixes a bounded grid from past finite scores on a fresh automatic
promise, without transferring labels, bets or delta spending. Supported ordinary calibration can supply this past
grid to a subsequent dispatcher; insufficient or unsupported calibration keeps the existing warmup. Requested
pilot sizes and evidence floors are preserved.

The synchronous stream loop offers explicit bounded acquisition through `DefaultPolicy(acquisition=...)`. Its
extra-label budget and stopping conditions are recorded and replayed. Unsupported asynchronous acquisition is
rejected before queries. This changes acquisition effort, not the training, calibration or holdout floors.

`Dispatcher.availability()` reports observed and missing labels separately. `replay_stream()` now includes an
additional `availability` field: update exact dictionary expectations accordingly. Missing-label handling in the
certificate remains conservative; the conditional observed error rate is not a population risk guarantee.

## Domain-supplied check correlations

`CorrelationAwareCombiner` accepts `rho_by_domain`, `domain_counts` and `domain_shrinkage`, with an explicit `domain=`
at combination time. Positive shrinkage requires supported domain counts. Unknown domains are refused; omission
with configured domains warns before using the global matrix. Likelihood fitting remains pooled. Domain matrices
are supplied by the caller and must not use the current answer's truth to select a domain.

### Guard approvals require a trusted verifier

`Guard.resolve(..., approve=True)` now refuses with `PermissionError` when `user_verifier` is absent. It leaves the
escalation unresolved, writes no correction and executes no tool, including with `execute=False`. Configure an
existing `approval_verifier(keys)` and supply `sign_approval(...)` tokens, or your own trusted verifier callable.
Token validation and its binding to the exact call remain mandatory whenever a verifier is configured.

For a trusted in-process legacy reviewer callback only, explicit keyword-only `allow_unverified_approvals=True`
retains the old programmatic path. The flag requires an actual bool, emits an `UNVERIFIED` warning when used,
marks the returned reasons and stores `approval_verification="unverified-explicit-waiver"`. It supplies no identity
authentication and never bypasses a configured verifier. Rejections still need no token. Normal checks, calls and
batch policies are unchanged. Message-role authentication remains a separate boundary; this change does not close
all of S157 or establish the truth of authenticated messages.

MCP elicitation's plain `accept`/`approve: true` response supplies no authenticated identity or approval token.
Without an applicable trusted verifier/token, the proxy keeps the escalation unresolved and returns structured
`isError`/`solvi` decision content with `approval refused`; it does not call upstream. A negative elicitation response
still denies without a token. The proxy never enables the compatibility waiver automatically or manufactures a
signature from an MCP role. Only `PermissionError` from resolution becomes this structured refusal; unexpected
resolution errors keep their existing error path.

`Message(..., verified=True)` is now descriptive metadata only. With `message_verifier` configured, authentication
comes from successful verification of the raw message or opaque context provenance minted by that same Guard.
The private context stamp binds the exact role/text to the Guard and current verifier identity; copying flags or
stamp metadata onto changed content does not authenticate it. Changing verifier identity invalidates existing context
stamps. Preverified context from another Guard does not transfer authentication, even with equal verification keys. Same-Guard
Session clipping and batch deep copies retain their internal provenance. Sessions verify on entry/add and retain
that accepted context; they do not promise fresh age verification on every subsequent call. Repeated raw signed
contexts still pass through the verifier and its existing age checks. This boundary does not defend against hostile
Python code modifying private in-process Guard state, nor does it establish the truth of authenticated content.
