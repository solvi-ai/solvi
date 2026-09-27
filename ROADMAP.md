# Roadmap

What we plan to add or improve, roughly in priority order. Research items ship only if their pre-registered measurements
hold; the numbers are published either way (see the research notes linked from each release). Ideas and votes are welcome
in [Discussions → Ideas](https://github.com/solvi-ai/solvi/discussions/categories/ideas).

Status: `planned` · `in progress` · `research` (may not ship) · `done (version)`.

## Next (0.5.x)

| Item | What it gives | Status |
|---|---|---|
| **Costs from measurements** in the strategist | `System.costs` already measures every part's run time; feed it to the cost-optimal planner so that, with no hand-written `cost=`, solvi picks the fastest of equivalent producers (e.g. a local table over a 300 ms feed) and adapts when a source slows down. Opt-in like `producers="equivalent"`, with a switch to freeze the choice; the trace records why a path was chosen. Offline, 0.4-style plans were up to 48% costlier than optimal without declared costs. | planned |
| **Several questions per pass by default** | The decider answers many questions about one state in one forward pass (≈2–3× faster). On by default only once block and single-question answers agree on ≥ 99% of decisions; ONNX export with block inputs for CPU and browser. | in progress (L19) |
| **Guaranteed escalation thresholds** | Calibrate on your own labelled stream and get a statistical guarantee on answering alone: `act_guard(examples, risk=0.10)` (conformal risk control) or `max_error=` (learn-then-test), plus conformal answer sets (`part.conformal(...)`) as a short list for the person who handles an escalation. Measured: the guarantee holds on the calibrated stream but breaks under domain shift, so calibration must be on your own data. | planned (L20) |
| **TraceStorage** | An abstract store for responses and traces: `save`, `get(id)`, `query(question=, answer=, safeguard=, model=, since=, until=)`, `iter`, `replay_all`. Backends: **JSONL** (append-only, no dependencies; replaces today's `journal=`) and **SQLite** (stdlib, indexed by question, answer, safeguard, model fingerprint, time). A hash chain across stored traces makes deleting or editing a stored decision detectable. Uses: audit reports for a period, re-checking every stored decision after a model or rule change, drift, feeding human corrections to `teach`. Later backends: **Postgres** (several services writing), **DuckDB** (analytics; it reads JSONL/Parquet directly), RocksDB only if a high-rate key-value log is really needed. | planned |
| **Async execution** | `await system.aask(state)` next to `ask`: `async def` catalog parts (database lookups, HTTP APIs, model servers) are awaited, independent steps of the flow run concurrently, early exit cancels pending calls (a failed hard check stops paid lookups), per-part timeouts turn into abstentions; sync parts still work (inline or in a thread). The trace keeps flow order, so replay and hashes do not depend on timing. Worth it because solvi is called from async web servers and agents, where a blocking call stalls the event loop, and because Pyodide in the browser needs async for network calls. Plain CPU parts gain nothing: the sync `ask` stays the default. | planned |
| **Browser check of every Space on each release** | Automated smoke test of playground / arcade / documents / realms after a release. | planned |

## Models

| Item | What it gives | Status |
|---|---|---|
| **decide-large** | A larger decider (ModernBERT-large, 395M) trained with a data-agent generator, an LLM teacher ensemble with soft labels, evidence data and label-set augmentation; goal: beat GLiNER2.5-Decide on the Fast Decisions dev split; a distilled base student for CPU and the browser. | in progress (L19) |
| **Better evidence quotes** | Quotes that actually support the answer on long legal and business texts (today ~23% on ContractNLI). | research |
| **Business-judgement questions** | Better zero-shot answers on operational judgements (typed-decisions style); today use `fit` on 30–60 labeled examples. | research |
| **Numeric estimates** | Tighter, calibrated intervals for `Estimate`. | research |
| **extract-base v2** | A stronger zero-shot field extractor, unified with the decider's span pointer. | planned |

## Strategist

| Item | What it gives | Status |
|---|---|---|
| **Reliability-aware producer choice** | Choose among producers by how often each one is accepted and right, learned from outcomes (continues the learned producer policy). | planned |
| **Task in words → goal and plan** | A model that turns a natural-language task into the questions to ask and the goal to plan for over a catalog; code still verifies and executes. | research |
| **Name matching across teams** (`solvi.aliases`) | Link parameters and facts whose names differ; acceptance by active, targeted questions and units in types. Experimental today. | research |
| **Model strategist for plan segments** | The L3–L6 decomposer as a proposer for ambiguous segments. Experimental: adds little over the code planner when contracts and costs are declared. | research |

## Games and demos

| Item | What it gives | Status |
|---|---|---|
| **L17 strategist in realms** | A tiny learned policy behind hard laws; leads 23 of 32 new maps over 5,000 turns with zero law violations. | in progress |
| **Lookahead + policy mode** | Stronger (leads 29/32) but up to 0.9 s per decision; needs a faster search to fit the browser. | research |

## Docs

| Item | Status |
|---|---|
| How-to series: support triage, classification with guarantees, solvi as a tool for an LLM agent, moderation and PII spans, three-way invoice matching, KYC screening, learning from 10 examples, field extraction with quotes | in progress |

## Done

| Item | Version |
|---|---|
| Typed facts (pydantic), questions declared by types, answer primitives (not stated, evidence, span, rank, estimate), overall confidence, typed decider API with act/escalate, code strategist (dead ends, exact cost optimum, memoized planning) | 0.5.0 |
| Audit shows what a learned head reads and ignores; `rule_abstained` safeguard | 0.4.1 |
| Provenance, audit and safeguards; decisions with a model | 0.4.0 |

## Not planned (for now)

- Text generation: solvi decides and extracts; it does not write prose.
- A hosted service: solvi runs in your process, on your data.
