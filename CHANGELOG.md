# Changelog

## Unreleased (0.7)

### Reports for people

- `res.report(format="md" | "html" | "data")`: a report of one decision for an auditor or a customer — each answer, what it
  rests on (given, computed, quoted with offsets, decided with the model and probabilities, learned, checks, rule,
  evidence), the safeguards that fired, the guarantee line (the promise of the calibrated thresholds behind it, "none",
  or no model decided it), the source texts with every quote highlighted, every model that ran with its fingerprint, the
  trace's hashes and the replay status (`replay="trusted"` by default: no model is called).
- `store.report(since=, until=, question=, format=, examples=3)`: a report of a period — per question the counts by
  answer, status and safeguard, the escalation rate, the guarantee coverage of the answers a model took part in, the
  catalog and model fingerprints in use and their changes, and example stored ids.
- HTML is one self-contained page (inline CSS, light and dark, no scripts or external assets); every value is escaped.
  Markdown escapes every special character.
- `solvi report STORE [--since] [--until] [--question] [--id ID] [--html out.html] [--md out.md] [--json] [--system]`.
- A response keeps the System that answered (and one loaded with a System, its System) for reports.

### OpenTelemetry export

- `solvi.otel.export(res_or_store, tracer=None, **filters)`: decisions as OpenTelemetry spans — a root `solvi.decision`,
  one span per step (fact, provenance, value, confidence, error, producer, quote offsets, model id and fingerprint,
  probabilities, safeguards, the step's hash and its link) and one per answer; failed or rejected steps with status
  ERROR; the root is a child of the caller's current span. A store exports every stored decision, or a query's.
- `solvi.otel.to_otlp_json(...)`: the same spans as OTLP/JSON (an ExportTraceServiceRequest body) without OpenTelemetry;
  ids derived from the trace's hashes.
- New extra `otel` (`opentelemetry-api`, `opentelemetry-sdk`).

## 0.6.0 — 2026-09-28 — serving, catalog lint, several models, async, measured costs

### Async execution: aask

- `await system.aask(state, names=None, order=None, store=True, timeout=None, speculate=False)` next to `ask`:
  `async def` catalog parts (fn, extract, check, rule, alternative producers) are awaited; steps run concurrently as
  soon as the steps they read have finished; sync parts run inline, or in a worker thread (`asyncio.to_thread`) when
  declared `blocking=True`.
- Early exit: by default in the phases of `ask` (hard checks and what they read first), so no call starts that `ask`
  would not make; `speculate=True` starts every ready step at once and cancels the pending calls that a failed hard
  check makes unnecessary. Cancelling `aask` cancels every pending call.
- Timeouts: `timeout=` (seconds) on a part (`@cat.fn(timeout=2)`, extract, check, rule), per call (`aask(timeout=)`) or
  for the System (`System(timeout=)`). A call that does not finish fails with "timed out after 2 s"; the questions that
  need it abstain with guard `timeout` — a new safeguard in `res.safeguards`, the audit, `system.stats["timeouts"]` and
  `safeguard_report()` (listed once it fires) — and a producer that times out is followed by the next one. Replay does
  not re-run a step that timed out.
- The trace is the one `ask` writes: records in flow order, the same answers and hashes whatever finished first — tested
  on all 117 gallery cases and on examples 01, 03, 04, 09, 12 and 16, phased and speculative, and with storage,
  concurrent asks, batched decisions and `Cascade` / `Vote` / `Route`.
- `ask`, replay and `facts_for` still work on catalogs with `async def` parts (each call awaited in an event loop of its
  own). `System.is_async` (`solvi.runtime.async_parts(catalog)`) says whether a catalog has parts that `aask` awaits;
  `solvi serve` answers such a System with `aask` (async HTTP endpoints, concurrent asks; the MCP server too).
- `solvi.runtime.aexecute` is the async executor; `execute` and `aexecute` share one plan of phases.

### Costs from measurements

- `System(..., producers="equivalent", costs="measured")`: the cost-optimal planner (`ModelStrategist(producers=
  "equivalent")`; `producers="equivalent"` on the System is now a shortcut for it) plans with the run times
  `system.costs` measures instead of declared costs. Warm-up: a producer counts its measured time after `min_samples`
  runs; before that its declared `cost=`, or 0 ms when undeclared, so each is tried and measured. When the producer in use
  slows down, the next plans switch; a producer unused for `recheck` asks gets one more trial. Settings:
  `solvi.learned.MeasuredCosts(min_samples=3, recheck=50, alpha=None)` (`alpha`: the smoothing of `system.costs`).
- `system.freeze_costs()` fixes the planner's costs at what was measured (the choice stops changing; measuring goes on),
  `system.unfreeze_costs()` resumes.
- The plan record says why each path was chosen: `extra["costs"]` lists, per fact with several usable producers, each
  producer's cost and its source (`measured`, `declared`, `warm-up`, `recheck`, `frozen: ...`) and a `why` line.
- `ModelStrategist.plan(..., costs={producer: cost})` takes costs from the caller (under the strategist's own `costs=`).

### solvi serve: HTTP, MCP and System One

- `solvi serve module:attr` (or `file.py:attr`) serves a System's questions over HTTP (`solvi[serve]`: FastAPI, uvicorn):
  `POST /ask` (state in; `Response.to_dict()` out with `stored_id` and `trace_hash`), `POST /ask/{question}`,
  `GET /questions`, `GET /health`. The OpenAPI document comes from the same pydantic types: each question's input state
  schema (the given facts its flow reads, typed by `System(inputs=...)` or by their typed readers, the ones it cannot be
  answered without as required; `solvi.serve.question_inputs`) and each response's answers as closed sets. The state is
  not validated by the web layer: a wrong-typed field is rejected by solvi as usual (the answers that need it abstain,
  safeguard `type_rejected`). `--store PATH` saves every answer with its trace to a TraceStorage.
- `solvi serve --mcp`: an MCP server over stdio, each question a tool whose input schema is the question's input state
  schema; a call returns the answer, confidence, status, why and safeguards with the stored id. Uses the official `mcp`
  SDK (2.x, `solvi[mcp]`) when installed, else a built-in JSON-RPC server (initialize, ping, tools/list, tools/call).
- `POST /v1/systemone` backed by a solvi decider (`--decider path_or_hf_id`, `--model-name`): the System One protocol
  (choice → probabilities, noul → P(yes), score → expected level index with its legend), so solvi answers where a Jev /
  Kev client points; `solvi.systemone` round-trips against it. `solvi serve --decider X` alone serves only this endpoint.
- `System.response_schema` is built by `solvi.schema.response_model(system, names=None)` (the pydantic class);
  `solvi.strategist.given_facts(catalog)` lists the facts a catalog reads and no part produces.

### solvi check: catalog lint

- `solvi check module:attr` (`solvi.check.lint(system)`): catalog lint with exit status 0 (no errors) / 1 / 2 (usage),
  `--strict` (warnings fail), `--json`. Errors: a hard check whose `then=` question never runs it (not read by the rule,
  not in `checkpoints`: a failing check would be ignored), `then=` naming no question or an invalid answer, cycles,
  questions no input can answer, producer / consumer and `inputs=` type conflicts, constraints that cannot hold (alone or
  together; brute force over finite answer domains), constraints reading non-questions. Warnings: unused parts, `then=` on
  soft checks, rules reading question names, disagreeing reader types, options the constraints always rule out, raising
  constraints, and silent defaults — `x or <literal>` / `.get(k, <literal>)` in functions that read the input (`# solvi:
  ok` accepts one).

### Several models, one decision

- `solvi.multi.Cascade([small, large])`: ask the decision parts in order, answer with the first that does not escalate,
  escalate when all do. The next model is asked only when needed; `costs=[45, 137]` reports the expected cost.
- `solvi.multi.Vote([a, b], rule="all" | "majority")`: answer when the rule holds and every agreeing part is sure;
  disagreement escalates with the proposals listed. Parts of one model share a forward pass when they can.
- `solvi.multi.Route({predicate or fact name: part}, default=part)`: code picks the part per input; only its model runs.
- A combination is used wherever a decision part is (`cat.fn`, `.question(cat)`, `System.teach` teaches every part);
  the parts must answer the same question (checked at construction); combinations nest.
- `act_guard(examples, risk=0.10)` on the combination: one threshold on every part's signal, chosen by conformal risk
  control on the loss monotonized from above (a cascade's loss is not monotone in the threshold), so P(answered alone
  and wrong) ≤ risk holds for the whole. Measured with solvi-base → solvi-large at risk 0.10: the risk stayed ≤ 10% on
  every data set; the cascade answered 96% of ContractNLI alone at 64 ms per question against the large model's 97% at
  137 ms; voting lowered the error among automatic answers on JSON questions from 2.1% to 0.4%. Also `conformal`.
- The trace records every proposal (`extra["stages"]` / `["answered_by"]`, `["votes"]`, `["route"]` / `["routed"]`) and
  the models called (`extra["calls"]`); the audit lists each stage, vote or route; replay re-runs every stage and compares
  the proposals, or — trusted or unavailable models — checks that the answer follows from the recorded proposals.
- `examples/18_several_models.py`: cascade, vote and route under one guarantee, with keyword stand-ins.

## 0.5.1 — 2026-09-28 — escalation with a guarantee, any System One model, a release gate, stored decisions

### Escalation with a guarantee

Measured on the 0.5.0 deciders: the shipped act threshold for "10% error" let through answers that were wrong 32–39% of
the time on typed-decisions and Taskmaster-2 (it holds on ContractNLI and JSON questions). The thresholds below keep their
promise on inputs like your calibration examples.

- `part.act_guard(examples, risk=0.10)`: conformal risk control on a few hundred labelled examples of your stream —
  P(answered alone and wrong) ≤ risk, as a share of all questions. Measured on solvi-large with 300 examples: the risk stays
  at 9.6–10.0% on every data set (typed-decisions answers 32% alone, ContractNLI 97%, JSON questions 99.6%). The result
  also says how much must escalate at least when the model is often wrong (`must_escalate_at_least`).
- `part.calibrate_for(examples, error=..., method="ltt")`: learn-then-test — the error among the answers given alone ≤
  error with probability ≥ 1 − delta; stricter, it often lets nothing through. `method="empirical"` is the 0.5.0 behaviour.
- `part.conformal(examples, coverage=0.9)`: every decision carries `extra["candidates"]`, the answers that cannot be
  ruled out; an escalation's message lists them for the person who takes over.
- Every decision records what its threshold promises; the audit shows a `guarantee` line per answer, or says that there
  is none because the thresholds were not calibrated on your data.
- `solvi.calibration`: `crc_threshold`, `ltt_threshold`, `conformal_quantile`, `set_scores`.

### Safeguards

- **Changed default:** choice and multi-label decisions ask the model with the options in sorted order
  (`option_order="canonical"`), so how a caller lists them cannot change the answer. On an independent stress test
  (decision-models-under-pressure, 64 options) reordering the options changed 41% of solvi-large's answers in the given
  order and 0.5% in the canonical one, at about the same accuracy. Options, probabilities and multi-label answers are still
  shown in the caller's order. `option_order="given"` restores 0.5.0 (and its fingerprints); `"average"` averages over
  rotations of the list. Parts whose options were not already sorted get a new fingerprint.
- `min_margin=0.1`: escalate a near tie between the two most probable answers (where a misleading text flips a choice).
- An answer head with a NaN or infinite feature abstains instead of answering with confidence NaN (found by fuzzing).
- Quotes proposed by a model are shown in the audit as "in the text; support not checked" (the text match is checked;
  whether the quote supports the answer is not).

### Any System One model as a decider

- `solvi.systemone.systemone(base_url, model, api_key=None)`: a decider over `POST /v1/systemone` — Jev and open servers
  (Kev, Von, Laya-serve, Intern-Decision, …). Questions about one input go in one request; everything built on a decider
  works: act_guard, conformal, fit / teach, audit, trace (which records the endpoint and model name).

### Release gate and decision tests

- Honesty suite (`solvi.honesty`, `solvi honesty SET --baseline B`): abstaining, "not stated", act vs escalate and traps on
  a labelled set; three numbers — confident errors, coverage at 10% risk, share of quotes that back the answer (a proxy) —
  and a non-zero exit when any gets worse. Run in CI and before publishing a model (docs/honesty.md).
- `solvi test PATH` and a pytest plugin: decision regression tests from `cases.json` (the gallery format) — expected
  answers, statuses and safeguards per case, trace replay, `--fuzz N` input mutations (docs/testing.md).

### Models

- The deciders are now `solvi-ai/solvi-large` and `solvi-ai/solvi-base` (the old `decide-large` / `decide-base` ids
  redirect).

### Storage

- `TraceStorage` (`solvi.storage`): stored responses with their whole traces — `save`, `get(id)`, `query(question=,
  answer=, status=, safeguard=, model=, since=, until=)`, `iter`, `corrections`, `replay_all(system)`. Backends
  `JSONLStorage` (append-only, one record per line) and `SQLiteStorage` (stdlib sqlite3, indexed; several writers). A hash
  chain across stored records: `verify()` catches an edited, deleted, inserted or reordered record and a cut-off tail
  (the stored head; `verify(anchor=head)` against a head kept elsewhere). `quarantine(fact, value)` lists the stored
  decisions whose answers rest on a fact; `forget(fact, value)` reports what removing a given fact would touch (nothing is
  deleted).
- `System(..., storage=...)` saves every ask (`res.stored_id`) and every `teach`; `ask(..., store=False)` skips one.
  `journal="file.jsonl"` is now a `JSONLStorage`: the 0.5 line keys are kept (plus the whole response and the chain
  fields), 0.5 lines already in the file are kept and reported as `legacy`; `teach` lines store dates as ISO strings.
- Catalog fingerprint: `System.fingerprint()` and `trace.fingerprint` (the catalog's, the questions' and every flow
  part's fingerprint: declarations, declared types and the code's syntax tree with the constants and same-module helpers
  it reads; `solvi.provenance.catalog_fingerprint`). `Trace.replay` says whether the catalog changed since the trace was
  recorded and which parts; `TraceStorage.query(catalog=fp)`.
- `solvi.diff.diff(store, system)`: re-run stored decisions with a new catalog or model and list the answers, statuses,
  safeguards and confidences that change, each with the first step that differs and why. `Shadow(current, candidate,
  storage=...)`: answer with the current system, store the candidate's response and the differences.
- A `solvi` command (also `python -m solvi`): `solvi verify`, `solvi replay`, `solvi diff` over a store.
- `Result.why` shows set-valued facts in a fixed order (it depended on `PYTHONHASHSEED`), so stored responses hash the same
  in every process.

## 0.5.0 — 2026-09-28 — typed facts, typed decisions, answer primitives

Type hints on catalog functions are the types of the facts (pydantic v2); untyped catalogs behave and hash exactly as
before. The same types declare the questions a decider model answers: types declare questions, the model proposes, checks
decide.

Models: the decider checkpoints published with this release are previews; each model card on
[huggingface.co/solvi-ai](https://huggingface.co/solvi-ai) has its measured numbers and limits. The strategist's model
weights are not published.

### Typing

- Typed facts (`solvi.typed`): `def risk_score(risk_points: dict[str, float]) -> float` — the catalog records each fact's
  type (`cat.types`, `cat.readers`, `flow.types`) and checks every producer's return type against every consumer's argument
  type when a part is registered; a definite mismatch raises `FactTypeError` naming both functions (conservative: `int` →
  `float`, `str` → `date`, `dict` → model, `X | None` → `X` pass). A typed rule's return type is checked against its
  question's options when the `System` is built.
- Run time: a typed part's arguments (given or computed) and its output (a Quote's / Decision's value) are validated and
  coerced with pydantic `TypeAdapter`s (cached per type; exact-type fast path; values that already passed the same type in
  the run are not re-validated). A failure is rejected like an ungrounded quote — the fact is missing, the next producer
  runs or dependent answers abstain — and is a new safeguard, `type_rejected` ("type rejected"): in the step's error, the
  audit, `res.safeguards` and `System.stats`. A Literal / Enum return type is a closed set (outside it: `outside_options`);
  an Enum answer is returned as its value. `validate` gets the coerced value; replay re-runs the validation.
- Answer types from Python types: `Answer.from_type(bool | Literal[...] | Enum | list[Literal[...]], ordinal=False)`;
  `Question(name, text)` without `answer=` takes it from its rule's return type.
- Typed input state: `system.ask(model_instance)` (a pydantic `BaseModel`: its fields are the given facts);
  `System(..., inputs=Model)` validates dict requests — fields with defaults become given facts, a field that fails is left
  out and reported (`res.trace.rejected`, safeguard `type_rejected`).
- Serialization (`solvi.schema`, pydantic models): `model_dump(mode)`, `to_json()`, `model_validate(data, catalog=)`,
  `from_json(text, catalog=)`, `model_json_schema()` on `Response`, `Result`, `Trace`, `Record`, `Question`, `AnswerType`;
  `system.response_schema()` has each answer as its closed set. With `catalog=` (or the System), typed values that JSON
  cannot carry (dates, enums, models) are restored from the facts' types, so a loaded trace replays with the same hashes.
- pydantic (`>=2`) is a core dependency; it is imported only for typed parts, BaseModel inputs and serialization
  (`import solvi` does not load it; pydantic ships with Pyodide, so the browser playground can use it).
- Faster asks: a value read by several steps is hashed once per run (gallery runners up to 14% faster).
- Example 14 (typed customs desk); gallery 10 (procurement) retrofitted with pydantic documents and typed functions (same
  answers; the audit shows the given documents as models).
- Trace note: records of typed parts hash their coerced values; untyped traces are unchanged.
- Hand-written extractors: a Quote without its own `source` points into the extractor's text — `doc` if the function
  reads it, else its only argument, else its only `str`-typed argument; an ambiguous signature raises at registration and
  asks for the new `@cat.extract(source="...")`.

### Typed decisions

- The decider (`solvi.decide`) answers typed questions; L14b–L14e checkpoints (`l14b_decider v1`) load, score and hash
  exactly as before.
  - Question kinds from types: `choice` (`Literal[...]`, an Enum; with "other" as an abstain threshold), `multi`
    (`list[Literal[...]]`), `score` (`solvi.typed.Scale[Literal[...]]`, 2–10 ordered levels → an ordinal answer; the value
    is the median, the expected level is recorded), `noul` (`bool` → the value True / False, answered yes / no).
    `model.decision(name, task, fact, Scale[...])` (or `type=`, `kind=`), `model.decisions(PydanticModel, fact)` (one part
    per field: its type the kind, its description the task), `model.questions(cat, PydanticModel, fact)`;
    `Answer.from_type(Scale[...])` is ordinal; `solvi.typed.question_kind`, `Scale`, `Ordinal`.
  - Input: a text, or a state — a dict, list, pydantic model or dataclass — serialized by `solvi.decide.state_text` as key
    paths (`customer.tier: pro`), exactly the L14f training serialization ("paths"; also "tree" and "json", as the checkpoint
    declares). A decision reading several facts serializes `{fact: value}`.
  - Output per question: probabilities, a calibrated confidence (a temperature per kind) and act / escalate. The model's act
    signal (an act head, optionally through a shipped act calibrator) below its threshold rejects the decision as the new
    safeguard **model escalated** (`guard="escalated"`, `system.stats["model_escalated"]`, the audit); without one,
    `escalate_below=` escalates by calibrated confidence as **low confidence**. `act_threshold=`, `target_error=` (the
    checkpoint's threshold for an error rate), `use_act=False`; `part.calibrate_for(examples, error=0.05)` picks the
    threshold for a target error rate. An escalated decision's answer abstains saying what it would have answered; a
    fallback producer runs if there is one. Provenance stays `decided`; `record.extra` has the act probability.
  - Several questions per forward pass: when the checkpoint declares `multi_question`, the strategist groups decision
    parts reading the same facts with the same model (`flow.batches`) and the executor scores each group in one pass
    (`model.passes` counts them; the block layout of L14f — input encoded once, questions do not see each other — with a
    fallback to one question per pass); records name their shared pass (`extra["pass"]`) and replay re-scores it.
    `model.decide_pass(input, parts)`. Catalogs without decisions do no extra work.
  - `adapt` / `fit` / `teach` per kind: a free shift per option (choice, multi), an ordinal-aware tilt and spread over the
    levels (score), one yes−no bias (noul); `System.teach` maps answers to the decision's labels (`True` → yes).
  - Checkpoint capabilities in `solvi_decide.json` (formats `l14b_decider v1`, `l14f typed v1`, `solvi_decide v2`): modes,
    markers, head columns, noul labels, state serialization, multi-question layout, temperatures per kind, thresholds, act
    head (column, temperature, calibrator, thresholds per target error) — the contract is
    [docs/decide_format.md](docs/decide_format.md). `DecideModel.load(..., multi_question=, act=)` overrides them for
    experiments; `model.caps`.
  - Records, flows and their JSON carry the new `extra` / `batches` (records without them hash as before).

### Answer primitives

- Answer primitives — every answer is a value and a confidence, declared by types, from plain rules, learned parts and
  model decisions alike (`solvi.primitives`; guide: "Answer primitives"; [examples/16_primitives.py](examples/16_primitives.py)):
  - **"Not stated"**: `solvi.Unknown` (type `NotStated`; `Maybe[T]` = `T | NotStated`; `Answer.maybe(t)`) is a real answer —
    the text does not state it — with a confidence, distinct from "no" and from an abstention (`None`). `result.not_stated`,
    `res.not_stated`, `res.overall["not_stated"]`; constraints see `Unknown` and joint decoding can choose it; it
    round-trips through JSON (`"not_stated": true`, probability key `"<not stated>"`).
  - **Evidence**: `Claim(value, evidence=[Quote | str], confidence=, source=)` from any part, `Decision(..., evidence=)` from
    a model; strings are located in the text, every quote must be literally in its given text at its offsets — else the
    output is **rejected** (safeguard "grounding rejected": the fact is missing, the next producer runs, else the answer
    abstains). Recorded in `record.extra["evidence"]` (hashed, replayed, tampering caught), `result.evidence`, shown in the
    audit and counted in the support (`quoted` / `quoted_by_model`). `Question(require_evidence=True)`: an answer without a
    quote abstains — the new safeguard **evidence missing** (`guard="evidence_missing"`, `system.stats["evidence_missing"]`,
    listed by `safeguard_report()` once it fires).
  - **Span**: `Span[T]` / `Answer.span(source=, type=)` — an exact substring of a given text (a Quote, or a text that is
    located), always grounded, coerced to `T` with pydantic (a failure: "type rejected"); `result.span`.
  - **Rank**: `Rank[Literal[...], k]` / `Answer.rank(options, k=)` — a tuple of the top k with `result.scores`; from a rule's
    `{option: score}` (a key function) or an ordered list, or a model's probabilities (confidence: Plackett–Luce);
    constraints see the tuple and joint decoding repairs a model's ranking.
  - **Estimate**: `Estimate[edges]` / `Answer.estimate(bins | lo, hi, step, coverage=0.8, unit=, integer=)` — open-ended
    bins labelled like the L14g decider's; a rule returns a number (confidence 1) or a distribution; the value is the middle
    of the median bin, `result.interval` the bins holding the central coverage, the confidence their mass.
  - **Confidence as a primitive**: for every kind the probability that the answer, as returned, is right (the guide's
    table); `res.overall["by_kind"]` gives per kind the answered count and the product of their confidences. `Result`
    gains `kind`, `evidence`, `extra` (JSON too); `Question` gains `require_evidence`; `AnswerType` gains `unknown`, `k`,
    `bins`, `coverage`, `unit`, `source`, `type` (dumped only when set).
  - The decider (`solvi.decide`) requests them from a checkpoint that declares them — the L14g contract
    (`"subformat": "l14g typed v2"`, docs/decide_format.md §9, aligned with `exps_v2/experiments/l14g_format.py`): modes
    `rank`, `number`, `span`; the "not stated" logit (joint softmax with the options; sigmoid for multi; the null span for
    spans); a pointer (start / end columns over the input's tokens, full layout only) for span answers and evidence quotes
    (`evidence=True`); `model.decision(..., Maybe[...] | Span[T] | Rank[...] | Estimate[...])`, `model.has_unknown`,
    `model.has_pointer`, `solvi.decide.decode_pointer`. Pointer questions are scored one per sequence and never batched.
    Older checkpoints parse, score and hash exactly as before (rank / number are asked as a choice / a score there; a span,
    evidence or "not stated" raise when the decision is made).

### Overall confidence

- Overall confidence of a response: `res.confidence` (the probability that every answered question is right: the
  product of the answers' confidences), `res.complete`, `res.weakest`, and `res.overall` as data; shown in the first lines of
  `print(res.audit())` and included in `to_json()`.

### Code strategist

- `System(..., strategist=...)`: a pluggable strategist; the default is still `solvi.strategist.plan`
  ([docs/strategist.md](docs/strategist.md), [examples/17_model_strategist.py](examples/17_model_strategist.py)).
- `solvi.strategy.ModelStrategist()` (no model) — the deterministic plan with dead ends dropped: a producer whose inputs
  cannot be computed no longer makes its fact unreachable (the deterministic strategist needs the inputs of every producer).
- `producers="equivalent"`: interchangeable producers, the cheapest verified plan by declared `cost=` (an exact 0/1 program,
  scipy's HiGHS), with the hard checks that govern a question kept as mandatory milestones. The plan is one hashed trace
  record (kind `plan`); `trace.replay` re-verifies it.
- The deterministic strategist memoizes each fact once per question (it was exponential on catalogs where a fact is
  reachable by several routes); flows and answers are unchanged.

### Experimental

- `ModelStrategist.load(path)`: a segment model (the L3–L6 typed decomposer, compressed to 34.5M parameters; torch or ONNX,
  format `solvi_strategist v1`) proposes producers where declared costs do not settle the choice; every proposal and the
  whole plan are verified by code, a rejected one falls back to code's plan; provenance `proposed` with the model's
  fingerprint when the model chose. What it learned is roughly the cost hints in docstrings — declare `cost=` instead.
  **Its weights are not published**; load your own checkpoint.
- `solvi.aliases`: a name matcher (MiniLM + character CNN) proposes aliases for parameter names that match no fact;
  `accept` decides by labelled examples, probes and targeted questions (active mode); `apply` rewires the catalog. Accepted
  aliases are a suggestion to review, not proof. Weights not published.
- `DecideModel(..., multi_question=, act=)` overrides and the block layout (several questions in one pass) — the answers of
  one question can differ between the block layout and one question per pass; ONNX exports without block inputs fall back
  to one question per pass.

### Fixes and tooling

- Set-valued facts hash in a fixed order (sorted canonical elements) and are exported to JSON in that order: trace hashes
  no longer depend on `PYTHONHASHSEED`, and a set fact replays after a JSON round trip. Traces from 0.4.x that contain sets
  hash differently (their hashes depended on the process anyway).
- Replay catches a value written into a step that failed (a record with an error must carry no value), also when the
  attacker re-hashes the chain; before, such an edit replayed as ok.
- A hard check that raises while it runs now makes the questions it governs abstain ("hard check … could not be
  evaluated"); before, the question was answered as if the check had passed (also in 0.4.x).
- The pointer applies the checkpoint's `temperature.span` to the start / end scores and the null span (as the L14g
  calibration fitted it), for spans and evidence.
- A typed span (`Span[float]`) whose best span does not parse ('149.90 EUR') takes the best span's part that does ('149.90'),
  with the probability mass of the spans between them; never a span outside the best one (then "type rejected" as before).
- An l14g act calibrator scores plain yes / no and choice questions too (its `p_unknown` / `kind=` features were missing
  when a question did not allow "not stated": the decision failed with `KeyError`).
- `solvi.__version__`; `tools/smoke_decide.py` runs a decider checkpoint end to end through solvi on torch and ONNX (every
  kind, the pointer's tokenizer offsets, several questions per pass, replay, JSON, backend agreement, latency).
- CI runs examples 01–06 and 09–17 (13, 15, 16 and 17 with their stand-ins).

### Examples and docs

- Example 16 (answer primitives from rules and from a decider). Example 15 (typed decisions: a pydantic ticket, four typed
  questions in one pass, checks over the model, an escalation);
  example 13 adds escalation for a target error rate and a JSON ticket. The guide's "Types" and "Decisions with a model"
  sections are one section now, "Types, questions and model decisions".
- Example 17 (the code strategist, a model's proposal checked, aliases). New docs: [docs/decide_format.md](docs/decide_format.md)
  (the decider contract), [docs/strategist.md](docs/strategist.md). SECURITY.md, CODE_OF_CONDUCT.md, issue templates.

## 0.4.1 — 2026-09-27 — clearer audits

- The audit of a learned part now lists what the head reads (`reads …`) and which requested features it ignored and why
  (`ignored …`, e.g. a dict-valued fact); `fit_fast` warns when it drops an explicitly requested feature.
- A rule that returns `None` on purpose is reported as its own safeguard, `rule_abstained` ("rule abstained"), instead of
  "outside the options"; `System.stats` counts it separately.
- Gallery: 03 shows a surface-only head (constraints repair it, 8/10) next to one reading a computed risk score (10/10);
  the gallery audit summary no longer double-counts safeguard events.

## 0.4.0 — 2026-09-27 — grounded decisions

Fuzzy proposes, deterministic decides, everything is in the trace. Every fact now carries its provenance (given, computed,
quoted, decided, learned, proposed); model outputs are grounded or rejected; `Response.audit()` shows what an answer rests on;
decisions with a model (`solvi.decide`) plug into the same safeguards. The pretrained solvi-decide weights are not published
yet (training data licensing is being cleaned); `solvi.decide` loads any checkpoint in that format.

- Decisions with a model (`solvi.decide`): the solvi-decide cross-encoder ("[mode] task [opt] options … [SEP] text", one logit
  per option) as a catalog part.
  - `DecideModel.load(path_or_hf_id, device=None, backend="auto"|"torch"|"onnx")`, `fingerprint()`, `model_id`,
    `metadata()`; `score` / `decide` / `logits`, batched and cached. Any object with `logits(items)` can stand in.
  - `model.decision(name, task, text_fact=, options=, descriptions=, multi=)` → a function returning `Decision(value, probs)`:
    the value is one of the options by construction, provenance `decided`, the trace records the model with a fingerprint of
    the checkpoint and this decision's adaptation. `part.question(cat, ...)` makes it a question's answer. Catalog parts
    pick up a decision's options (`__solvi_options__`) as their closed set.
  - `adapt(unlabelled_texts)`: label-bias correction without labels (the mean logit per option is subtracted).
  - `fit(examples)`: few-shot shift per option + shared scale (L-BFGS) and a temperature on out-of-fold predictions;
    `teach(text, correct)` refits the shift at once (~1 ms). `System.teach` routes to it when a question's answer is a
    decision part (or a rule passing a decided fact on).
  - "other" / "none" among the options is an abstain threshold on the best real option's calibrated probability (fitted
    on labelled "other" examples, else from the checkpoint's metadata), not a label the model scores.
  - `save_adaptations` / `load_adaptations`.
  - New extra `solvi[onnx]` (onnxruntime, tokenizers, huggingface_hub): the decider without torch; `solvi[model]` runs it with
    torch.
- `solvi.calibration`: `reliability`, `ece`, `coverage_at`, `threshold_for`, `accuracy_at`, `summary`, `evaluate(system,
  question, examples)` — for any model or question.
- Example 13: support-email routing with a decision part, bias correction, 16 labelled examples, abstention, a constraint
  with a rule-based question, a hard check, the audit and teach (the real decider with `SOLVI_DECIDE_MODEL`, a stand-in
  otherwise).

- Online learning of the check order / producer choice is on only when a learned policy uses it (`learn=None` default); it refits inside `ask` and caused rare slow calls in long runs. Cost tracking stays on.

- Grounded decisions: fuzzy proposes, deterministic decides, everything is in the trace.
  - Provenance of every fact and answer: `given`, `computed`, `quoted`, `decided`, `learned`, `proposed` (`record.origin`,
    `result.provenance` / `.source`, `solvi.provenance`). Parts take `model=` and `provenance=` (`@cat.extract`, `@cat.fn`,
    `@cat.check`, `@cat.rule`); `@cat.fn(model=, options=)` returning `Decision(value, probs)` is a model decision.
    Extractor `field()` / `embedder()` functions carry their model. `computed_state` shows provenance and model.
  - Model identity in the trace: model-backed records store `{"type", "id", "fp"}`; extractors, `Head`, `FastHead`,
    `RuleList` have fingerprints. Answer-head decisions are trace records (`kind="head"`). `Trace.replay(catalog_or_system,
    trust_models=False)` re-runs a model when it is the same and deterministic, otherwise verifies the recorded output is
    grounded, and reports "model changed since this decision"; `rep["models"]` gives a verdict per model step.
  - Grounding: a model's quote must be literally `doc[start:end]` (strings up to whitespace, numbers as written;
    `exact=` per part); a quote outside its text is now rejected (the fact is missing) instead of being used with an error;
    `min_confidence` and `validate` apply to every part, not only alternative producers; `Question(min_confidence=)`
    abstains on low-confidence answers.
  - `Response.audit(question=None)`: what each answer rests on, the safeguards that fired, and the deterministic share of
    its support; `show` prints a compact version. `System.stats` / `safeguard_report()`: lifetime counts of model outputs,
    grounding rejections, answers outside options, low-confidence abstentions, hard-check decisions, constraint repairs,
    validator rejections and fallbacks.
  - Hashes: plain records hash exactly as before; model-backed and learned-rule records (and head records) are new.
  - Example 12: one catalog with and without models, a hallucination caught, a model changed since the decision.
- The learned strategist. `System.costs`: a moving average of each part's run time. `System(order="learned")` /
  `system.learn_order(examples)`: hard checks run one at a time, most expected saving first (P(fail) from a per-check online
  model × cost saved ÷ cost), with early exit; answers are identical to the default order (earlier-declared governing checks
  are always evaluated before a later one decides). `res.trace.explain_order()` and `show` explain the order.
- Several producers of one fact: `@cat.fn(provides="total", cost=, validate=)`, `@cat.extract(provides=..., min_confidence=)`,
  `@cat.features("total")`. A fallback chain in declaration order, or `System(producers="learned")` — a policy that orders
  producers per input by P(accepted), P(agrees with the reference producer) and cost, learning from every run (with shadow
  runs for exploration). Records carry `producer` and `tried`; replay recomputes with the producer that was used.

## 0.3.0 — 2026-09-27

- New answer types: `Answer.ordinal` (ordered levels; a learned head answers with the median) and `Answer.multi` (a subset;
  learned per option with fit / fit_fast, updated by teach). Options may carry descriptions (`{option: description}`).
- Constraints between answers (`@cat.constraint`) with joint decoding: contradictory learned answers are replaced by the most
  probable combination that satisfies every constraint; `Response.feasible` and `Response.violations` report the result.
- Example 11: a content guard with multi-label and ordinal answers tied by constraints.

## 0.2.2 — 2026-09-27

- Stronger `Trace.replay`: checks that the chain starts from the hash of the recorded input, flags inputs missing from the trace
  (a deleted step), re-runs steps recorded as failed (a faked error is caught), and with `replay(catalog, flow)` checks that every
  planned step was recorded or skipped at run time. Documented limit: a trace rebuilt honestly from a different input is
  consistent — compare `init_hash` with a receipt published elsewhere.
- Arcade: minesweeper, 20 questions, Mafia detective, hack the trace, bot arena.

## 0.2.1 — 2026-09-26

- A question's flow no longer depends on which other questions are asked (checks on computed facts are planned per question).
- A hard check with `then` governs only the questions listed there (and those naming it as a checkpoint); for others it is an
  ordinary failed check. Early exit follows the same rule. When several hard checks fail, the first declared in the catalog decides.
- Learned rule lists are deterministic (ties broken in sorted order).
- Gallery: twelve decision tasks with scenarios, runners and comparisons; synced into the browser playground (`tools/sync_gallery.py`).

## 0.2.0 — 2026-09-26

- `System.fit_fast`: a closed-form ridge answer head trained in milliseconds (ridge strength by exact leave-one-out accuracy,
  pairwise features when there are few), and `System.teach` now updates it instantly (rank-one update, ~0.2 ms).
- `LongSpanExtractor.embed` / `embedder()`: a document embedding usable as a `fit_fast` feature.
- A learned head abstains when its features could not be computed.
- `tools/export_onnx.py`: export an extractor to ONNX (fp32 / fp16 / int8) with an agreement check.
- Spaces now run entirely in the browser (Gradio-Lite + Pyodide): playground and arcade.
- Examples 09 (strategy at scale) and 10 (learning in milliseconds); benchmarks `strategist_scale.py`, `fast_head.py`.

## 0.1.1 — 2026-09-26

- Runs in the browser (Pyodide): parallel execution falls back to one-by-one where threads are unavailable.
- Example 07 uses the published receipts model's strong fields (date, total, cash, change) and checks the change.

## 0.1.0 — 2026-09-26

First public version.

- Catalog of `@extract`, `@fn`, `@check` (soft and hard) and `@rule` parts; contracts come from function signatures.
- Typed questions (yes/no, choice); the strategist plans only the parts the asked questions need, plus checkpoints.
- Answers with confidence, a quote or a formula, and a hash-chained trace that `replay` re-verifies.
- Early exit (hard checks first; a failed one skips what only the settled questions needed) and parallel execution of
  independent steps (`workers=`), with a scheduling-independent trace.
- Learning: answer heads from labeled examples (`fit`), readable rule lists (`learn_rule`), Platt calibration (`calibrate`).
- ModernBERT extractors: one-pass multi-field (`MultiSpanExtractor`), per-field QA (`SpanExtractor`), long documents by field
  description with "no answer" (`LongSpanExtractor`), save/load and Hugging Face loading.
