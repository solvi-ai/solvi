# Breaking development changes: candidate migration notes

These changes are local unreleased development work. They do not announce a release or change the package version.

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

Blind tickets hide audit/escalation kind within one payload. An oracle with access to repeated `decide` calls can
learn additional information; equal ticket fields do not enforce sampling-independent labels. Reusing a
`request_id` handles retry idempotency, not whole-interaction blindness.

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
