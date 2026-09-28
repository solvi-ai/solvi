# Roadmap

What we plan to add or improve, roughly in priority order. Research items ship only if their pre-registered measurements
hold; the numbers are published either way (see the research notes linked from each release). Ideas and votes are welcome
in [Discussions → Ideas](https://github.com/solvi-ai/solvi/discussions/categories/ideas).

Status: `planned` · `in progress` · `research` (may not ship) · `done (version)`.

## Next (0.5.x)

| Item | What it gives | Status |
|---|---|---|
| **Costs from measurements** in the strategist | `System.costs` already measures every part's run time; feed it to the cost-optimal planner so that, with no hand-written `cost=`, solvi picks the fastest of equivalent producers (e.g. a local table over a 300 ms feed) and adapts when a source slows down. Opt-in like `producers="equivalent"`, with a switch to freeze the choice; the trace records why a path was chosen. Offline, 0.4-style plans were up to 48% costlier than optimal without declared costs. | planned |
| **Several questions per pass by default** | The decider answers many questions about one state in one forward pass (≈2–3× faster). Trained with a consistency loss, the answers still differ from one-question-per-pass in ~3–10% of cases on real states, so it stays off; next: an architecture where questions are independent queries over a text encoded once. | research |
| **Guaranteed escalation thresholds** | Calibrate on your own labelled stream and get a statistical guarantee on answering alone: `act_guard(examples, risk=0.10)` (conformal risk control) or `max_error=` (learn-then-test), plus conformal answer sets (`part.conformal(...)`) as a short list for the person who handles an escalation. Measured: the guarantee holds on the calibrated stream but breaks under domain shift, so calibration must be on your own data. | planned (L20) |
| **TraceStorage** | An abstract store for responses and traces: `save`, `get(id)`, `query(question=, answer=, safeguard=, model=, since=, until=)`, `iter`, `replay_all`. Backends: **JSONL** (append-only, no dependencies; replaces today's `journal=`) and **SQLite** (stdlib, indexed by question, answer, safeguard, model fingerprint, time). A hash chain across stored traces makes deleting or editing a stored decision detectable. Uses: audit reports for a period, re-checking every stored decision after a model or rule change, drift, feeding human corrections to `teach`. Later backends: **Postgres** (several services writing), **DuckDB** (analytics; it reads JSONL/Parquet directly), RocksDB only if a high-rate key-value log is really needed. | planned |
| **Async execution** | `await system.aask(state)` next to `ask`: `async def` catalog parts (database lookups, HTTP APIs, model servers) are awaited, independent steps of the flow run concurrently, early exit cancels pending calls (a failed hard check stops paid lookups), per-part timeouts turn into abstentions; sync parts still work (inline or in a thread). The trace keeps flow order, so replay and hashes do not depend on timing. Worth it because solvi is called from async web servers and agents, where a blocking call stalls the event loop, and because Pyodide in the browser needs async for network calls. Plain CPU parts gain nothing: the sync `ask` stays the default. | planned |
| **`solvi check`: catalog lint** | Catches catalog mistakes before they bite: a hard check with `then=` that is not in its question's flow (it only governs a question when the question reads it or lists it in `checkpoints`), parts nothing uses, questions no input can answer, cycles, type conflicts, constraints that cannot all hold. | planned |
| **`solvi test` and a pytest plugin** | Decision regression tests from `cases.json` (the format every gallery entry already uses): expected answers, statuses and safeguards per case, run in CI; plus input fuzzing to find crashes and unhandled values. | planned |
| **Shadow mode and `solvi diff`** | "We changed a rule — which decisions change?": re-run stored decisions (TraceStorage) with a new catalog or model and list the answers that change and why; run a new version in the shadow of the current one before switching. The catalog's fingerprint (hash of its code) goes into every trace. | planned |
| **`solvi serve`: HTTP and MCP server** | `solvi serve catalog.py` exposes the questions as an HTTP API (FastAPI, OpenAPI schema from the same pydantic types) and as an MCP server, so any agent can call solvi's decisions as a tool; traces go to TraceStorage. | planned |
| **Counterfactual explanations** | The smallest change of the inputs that would change the answer ("approved if the amount were ≤ 1000", "refused: 40 days since purchase, the limit is 30"), computed by search over the deterministic parts; adverse-action reasons for lending, clear answers for support. | planned |
| **Human-readable reports and OpenTelemetry** | An HTML / Markdown report of a decision or a period for an auditor or a customer: the answer, what it rests on, quotes highlighted in the document, safeguards that fired; export of traces as OpenTelemetry spans. | planned |
| **Browser check of every Space on each release** | Automated smoke test of playground / arcade / documents / realms after a release. | planned |

## Models

| Item | What it gives | Status |
|---|---|---|
| **decide-large** and a distilled **decide-base** | Typed decisions with evidence quotes, not stated, spans, rankings and numbers; ahead of GLiNER2.5-Decide and Laya on typed questions over JSON states, behind GLiNER2.5-Decide on zero-shot choice questions (level with it after `fit` on ~64 examples). | done (0.5.0, preview) |
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
| **L17 strategist in realms** | A tiny learned policy behind hard laws; leads 23 of 32 new maps over 5,000 turns with zero law violations. | done (0.5.0) |
| **Lookahead + policy mode** | Stronger (leads 29/32) but up to 0.9 s per decision; needs a faster search to fit the browser. | research |

## Docs

| Item | Status |
|---|---|
| Documentation site (mkdocs on GitHub Pages) built from the guide, format specs and examples | planned |
| Explanations and safeguard messages in more languages (Russian first) | planned |
| How-to series: support triage, classification with guarantees, solvi as a tool for an LLM agent, moderation and PII spans, three-way invoice matching, KYC screening, learning from 10 examples, field extraction with quotes | in progress |

## Done

| Item | Version |
|---|---|
| decide-large / decide-base (preview); L17 strategist in realms | 0.5.0 |
| Typed facts (pydantic), questions declared by types, answer primitives (not stated, evidence, span, rank, estimate), overall confidence, typed decider API with act/escalate, code strategist (dead ends, exact cost optimum, memoized planning) | 0.5.0 |
| Audit shows what a learned head reads and ignores; `rule_abstained` safeguard | 0.4.1 |
| Provenance, audit and safeguards; decisions with a model | 0.4.0 |

## Not planned (for now)

- Text generation: solvi decides and extracts; it does not write prose.
- A hosted service: solvi runs in your process, on your data.
