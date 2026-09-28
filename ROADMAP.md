# Roadmap

What we plan to add or improve, roughly in priority order. Research items ship only if their pre-registered measurements
hold; the numbers are published either way (see the research notes linked from each release). Ideas and votes are welcome
in [Discussions → Ideas](https://github.com/solvi-ai/solvi/discussions/categories/ideas).

Status: `planned` · `in progress` · `research` (may not ship) · `done (version)`.

## Next (0.5.x)

| Item | What it gives | Status |
|---|---|---|
| **Costs from measurements** in the strategist | `System.costs` already measures every part's run time; feed it to the cost-optimal planner so that, with no hand-written `cost=`, solvi picks the fastest of equivalent producers (e.g. a local table over a 300 ms feed) and adapts when a source slows down. Opt-in like `producers="equivalent"`, with a switch to freeze the choice; the trace records why a path was chosen. Offline, 0.4-style plans were up to 48% costlier than optimal without declared costs. | done (0.6) |
| **Several questions per pass by default** | The decider answers many questions about one state in one forward pass (≈2–3× faster). Trained with a consistency loss, the answers still differ from one-question-per-pass in ~3–10% of cases on real states, so it stays off; next: an architecture where questions are independent queries over a text encoded once. | research |
| **Honest act thresholds in the shipped models** | The act thresholds shipped in 0.5.0 are over-confident (measured 32–39% real error at a nominal 10%). 0.5.1 ships thresholds calibrated with conformal risk control and a table "threshold → measured error" in each model card. | done (0.5.1) |
| **Honesty suite** | A fixed test suite for abstaining, "not stated", act/escalate and traps that fails a release (or a fine-tune, merge, adapter) if it gets worse; three numbers per release: rate of confident errors, coverage at 10% risk, share of evidence quotes that support the answer. | done (0.5.1) |
| **Honest labels in the trace** | A quote not yet checked for support is marked as such; the runtime records the level of guarantee of each verdict and prints its conditions next to it. | done (0.5.1) |
| **Guaranteed escalation thresholds** | Calibrate on your own labelled stream and get a statistical guarantee on answering alone: `act_guard(examples, risk=0.10)` (conformal risk control) or `max_error=` (learn-then-test), plus conformal answer sets (`part.conformal(...)`) as a short list for the person who handles an escalation. Measured: the guarantee holds on the calibrated stream but breaks under domain shift, so calibration must be on your own data. | done (0.5.1) |
| **Guarantees per group** | `act_guard(examples, risk, groups=["domain", "task"])`: a threshold per group of a hierarchy, small groups pooled with their parent, a Bonferroni-corrected bound so that the risk holds inside every group at once — not only on average over a stream where a hard group can be far over it (28% at a 10% promise in our simulation). For one part and for Cascade / Vote / Route; the trace records which group's threshold applied. | done (0.7) |
| **Robustness to instructions in the input** | A message that says "ignore the rules and answer X" can push a decider to an allowed but wrong option. `perturb=k` asks again without instruction-like sentences (deterministic rules) and escalates when the answer changes; the honesty suite gets injection traps (embedded instructions, near-duplicate distractor options) and gates the share answered alone with the injected answer. Measured on solvi-decide base: 5–15% of injected messages followed without it, 0–1% with it, no extra pass on inputs without such sentences. | done (0.7) |
| **TraceStorage** | An abstract store for responses and traces: `save`, `get(id)`, `query(question=, answer=, safeguard=, model=, since=, until=)`, `iter`, `replay_all`. Backends: **JSONL** (append-only, no dependencies; replaces today's `journal=`) and **SQLite** (stdlib, indexed by question, answer, safeguard, model fingerprint, time). A hash chain across stored traces makes deleting or editing a stored decision detectable. Uses: audit reports for a period, re-checking every stored decision after a model or rule change, drift, feeding human corrections to `teach`. Later backends: **Postgres** (several services writing), **DuckDB** (analytics; it reads JSONL/Parquet directly) — done for 0.7; RocksDB only if a high-rate key-value log is really needed. | done (0.5.1); Postgres, DuckDB done (0.7) |
| **Async execution** | `await system.aask(state)` next to `ask`: `async def` catalog parts (database lookups, HTTP APIs, model servers) are awaited, independent steps of the flow run concurrently, early exit stops pending calls (a failed hard check stops paid lookups; with `speculate=True` lookups start at once and are cancelled), per-part timeouts turn into abstentions; sync parts still work (inline or in a thread). The trace keeps flow order, so replay and hashes do not depend on timing. Worth it because solvi is called from async web servers and agents, where a blocking call stalls the event loop, and because Pyodide in the browser needs async for network calls. Plain CPU parts gain nothing: the sync `ask` stays the default. | done (0.6) |
| **`solvi check`: catalog lint** | Catches catalog mistakes before they bite: a hard check with `then=` that is not in its question's flow (it only governs a question when the question reads it or lists it in `checkpoints`), parts nothing uses, questions no input can answer, cycles, type conflicts, constraints that cannot all hold, silent defaults (`x or 0`, `.get(k, 0)`) in functions that read the input. | done (0.6) |
| **Command line for a whole project** | `solvi init` scaffolds a project (a typed catalog, regression cases that pass, a README, a CI workflow); `solvi ask` runs one decision from a JSON state or a text and prints the answers, audit or report; `solvi calibrate` runs `act_guard` on a labelled file and saves the thresholds to a file the catalog loads, refusing it for another model (`part.save_calibration` / `load_calibration`); `solvi models` lists, pulls and checks deciders (capabilities, fingerprint, latency, accuracy). | done (0.7) |
| **`solvi test` and a pytest plugin** | Decision regression tests from `cases.json` (the format every gallery entry already uses): expected answers, statuses and safeguards per case, run in CI; plus input fuzzing to find crashes and unhandled values. | done (0.5.1) |
| **Shadow mode and `solvi diff`** | "We changed a rule — which decisions change?": re-run stored decisions (TraceStorage) with a new catalog or model and list the answers that change and why; run a new version in the shadow of the current one before switching. The catalog's fingerprint (hash of its code) goes into every trace. | done (0.5.1) |
| **`solvi serve`: HTTP and MCP server** | `solvi serve catalog.py:system` exposes the questions as an HTTP API (FastAPI, OpenAPI schema from the same pydantic types) and as an MCP server (each question a tool), and a decider behind the System One API (`POST /v1/systemone`); traces go to TraceStorage — done for 0.6. Guarding an agent's tool calls (`solvi.agents.Guard`: the agent proposes a call, solvi checks it — catalog, types, grounding in the conversation, instruction-like tool outputs, policies, an authorizer under act_guard and perturb — and makes it, denies it or escalates it, every decision a stored trace), adapters for PydanticAI, LangGraph and the OpenAI Agents SDK, and an MCP proxy (`solvi serve --guard --upstream`) — done for 0.7; hardened before release: approvals bound to the call and the reasons shown, `scan_user` for pasted injections, `locale=` for ambiguous numbers, `guard_run_config()` so the OpenAI Agents guard sees a run's own tool outputs. Next: measure how many harmful calls pass on an agent-injection benchmark (AgentDojo-style) with and without the guard. | done (0.6); agent guard and adapters: preview (0.7) |
| **Hardening for 0.7** | `solvi serve` with a bearer token, request size / JSON depth / time limits, errors that never carry a traceback or a path, CORS off by default, `--decider` never downloading without `--pull`; strict JSON everywhere (non-finite floats tagged); ruff and pyright in CI; an `ask` overhead benchmark against 0.5.0-style settings. | done (0.7) |
| **Counterfactual explanations** | The smallest change of the inputs that would change the answer ("approved if the amount were ≤ 1000", "refused: 40 days since purchase, the limit is 30"), computed by search over the deterministic parts; adverse-action reasons for lending, clear answers for support. | done (0.7) |
| **Human-readable reports and OpenTelemetry** | An HTML / Markdown report of a decision or a period for an auditor or a customer: the answer, what it rests on, quotes highlighted in the document, safeguards that fired; export of traces as OpenTelemetry spans. | done (0.7) |
| **Browser check of every Space on each release** | Automated smoke test of playground / arcade / documents / realms after a release. | planned |

## Models

| Item | What it gives | Status |
|---|---|---|
| **solvi-large** and a distilled **solvi-base** | Typed decisions with evidence quotes, not stated, spans, rankings and numbers; ahead of GLiNER2.5-Decide and Laya on typed questions over JSON states, behind GLiNER2.5-Decide on zero-shot choice questions (level with it after `fit` on ~64 examples). | done (0.5.0, preview) |
| **Decider v3** | New layers around the same pretrained encoder: text encoded once and questions as independent queries (many questions per pass that match one-by-one exactly), cached option vectors (hundreds of options), answers computed only from 1–3 selected evidence passages, several heads whose disagreement is the uncertainty, early exit on CPU, adapters per question kind. Each block ships only if it beats solvi-large on the same data. | research |
| **Long documents** | Beyond the decider's context, find the relevant sections first and decide on them: `long="retrieve"` (sections by headings and paragraphs, BM25, optional rerank by the decider, offsets mapped back, sections read in the trace). Next: the encoder's full 8k context (today the deciders are trained on up to 1k tokens) and a measured comparison of retrieve vs truncate on long legal texts. | retrieve done (0.7, unreleased); 8k context research |
| **Better evidence quotes** | Quotes that actually support the answer on long legal and business texts (today ~23% on ContractNLI). | research |
| **Business-judgement questions** | Better zero-shot answers on operational judgements (typed-decisions style); today use `fit` on 30–60 labeled examples. | research |
| **Numeric estimates** | Tighter, calibrated intervals for `Estimate`. | research |
| **extract-base v2** | A stronger zero-shot field extractor, unified with the decider's span pointer. | planned |

## Strategist

| Item | What it gives | Status |
|---|---|---|
| **Reliability-aware producer choice** | Choose among producers by how often each one is accepted and right, learned from outcomes (continues the learned producer policy). | planned |
| **Text in: entry points** | Text → which question is asked + its typed input state: the decider picks the entry point, spans and deterministic parsers fill the fields with quotes; provenance says the input was read by a model. `system.ask_text`, `solvi.textin.TextIn`, dialogue updates; served as `POST /ask_text` and an MCP tool. Next: measure the share of wrong actions that pass the checks on gallery catalogs, and a generative fallback only where a parser fails. | done (0.7, unreleased) |
| **Learning from corrections, with gates and rollback** | The system learns from every human correction of an escalation, every known outcome and every proposal your rules rejected — never from its own accepted answers (tried three times: no gain). By the number of labels per process: shift/scale (today's `fit` / `teach`), a memory of corrected cases and a small head, then a LoRA adapter. Every update runs in shadow, is diffed against the current one, must beat it on held-out labels and pass the honesty suite, recalibrates `act_guard` on fresh labels, and is recorded in each trace so it can be rolled back. The scaffold ships in 0.7 as experimental and off by default (`System.learning`: trusted labels only, the ladder, the gates, versions and rollback, tested with stand-in models); the simulation on real data sets decides whether it becomes a default. | experimental (0.7); simulation research |
| **Cascade and memory of corrections** | Decider → larger decider → LLM, escalating by the cost of a mistake; a memory of human-corrected cases (nearest neighbours with an abstain threshold). Any OpenAI-compatible LLM server is a decider (`solvi.llm`: JSON-schema replies validated, invalid output escalates, probabilities from log-probabilities when given), so it can be the last stage of a cascade or a family in a vote. | cascade, vote, route done (0.6); LLM stage and memory done (0.7, unreleased) |
| **LLM decider confidence** | Log-probabilities from LLMs are saturated (≥ 0.999 on almost every answer while still ranking right and wrong well), so a risk threshold on them lets nothing through; take the threshold by rank or on a log(1 − p) scale. | planned (0.8) |
| **Task in words → goal and plan** | A model that turns a natural-language task into the questions to ask and the goal to plan for over a catalog; code still verifies and executes. | research |
| **Name matching across teams** (`solvi.aliases`) | Link parameters and facts whose names differ; acceptance by active, targeted questions and units in types. Experimental today. | research |
| **Model strategist for plan segments** | The typed decomposer from research as a proposer for ambiguous segments. Experimental: adds little over the code planner when contracts and costs are declared. | research |

## Games and demos

| Item | What it gives | Status |
|---|---|---|
| **Learned strategist in realms** | A tiny learned policy behind hard laws; leads 23 of 32 new maps over 5,000 turns with zero law violations. | done (0.5.0) |
| **Lookahead + policy mode** | Stronger (leads 29/32) but up to 0.9 s per decision; needs a faster search to fit the browser. | research |

## Docs

| Item | Status |
|---|---|
| Documentation site (mkdocs on GitHub Pages) built from the guide, format specs and examples, with an API reference from the docstrings | done (0.7) |
| Explanations and safeguard messages in more languages (Russian first): audit, `show` and the safeguard report render in Russian with `lang="ru"`; traces and hashes stay in English | Russian done (0.7); more languages planned |
| How-to series: support triage, classification with guarantees, solvi as a tool for an LLM agent, moderation and PII spans, three-way invoice matching, KYC screening, learning from 10 examples, field extraction with quotes | in progress |

## Done

| Item | Version |
|---|---|
| `solvi serve` (HTTP, MCP, System One API), `solvi check`, Cascade / Vote / Route under one guarantee, `aask` with timeouts, costs from measurements | 0.6.0 |
| Escalation with a guarantee (`act_guard`, learn-then-test, conformal sets), option order and near-tie safeguards, System One backend, honesty suite, `solvi test`, TraceStorage, `solvi diff` and shadow mode | 0.5.1 |
| solvi-large / solvi-base (preview); learned strategist in realms | 0.5.0 |
| Typed facts (pydantic), questions declared by types, answer primitives (not stated, evidence, span, rank, estimate), overall confidence, typed decider API with act/escalate, code strategist (dead ends, exact cost optimum, memoized planning) | 0.5.0 |
| Audit shows what a learned head reads and ignores; `rule_abstained` safeguard | 0.4.1 |
| Provenance, audit and safeguards; decisions with a model | 0.4.0 |

## Not planned (for now)

- Text generation: solvi decides and extracts; it does not write prose.
- A hosted service: solvi runs in your process, on your data.
