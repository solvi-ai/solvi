# Roadmap

What we plan to add or improve, roughly in priority order. Research items ship only if their pre-registered measurements
hold; the numbers are published either way (see the research notes linked from each release). Ideas and votes are welcome
in [Discussions → Ideas](https://github.com/solvi-ai/solvi/discussions/categories/ideas).

Status: `planned` · `in progress` · `research` (may not ship) · `done (version)`.

## Toward 1.0: what solvi is for, and everything that exists landed

First the goal and an honest account of what is there; the module cleanup and the split into two levels come after,
and follow from it.

| Item | What it gives | Status |
|---|---|---|
| **The goal and the mission, stated** | [docs/mission.md](docs/mission.md). solvi builds decision systems from fast solvers (rules, code, small models) and LLMs: the fast ones answer where they are sure, LLMs and search deliberate where they are not, a person decides what neither can; every decision is recorded, explainable and replayable. **Accountability and safety:** see how each model solves each task — what it read, who answered, what was promised, whether the promise held — and wrap any model in checks that keep it from doing harm. **Accumulated knowledge:** the system keeps verified knowledge about its environment — rules, action conditions, a world map, goals (an agenda with code done checks and gates); a written specification is compiled into rules, otherwise rules are learned from experience, each with its source, and every item can be checked and retracted. Honest limits: on a stream of one kind of decision (classification) solvi keeps its error promise but does not promise growth over time, and the promise does not hold between an abrupt shift and its detection. **Any model**, including a small local one so much runs without the cloud (no quality promise for it). **Where to use:** single decisions (tickets, documents and contracts, policy decisions) and sequential decisions in an environment (agents with tools, planning, games). **What it is not:** a text generator, a hosted service or a replacement for LLMs; where a guarantee cannot be given it hands over to a person. **Honesty:** only what measurably helps enters the library, negative results are published next to positive ones, and a promise is a number you can check. | done |
| **Everything that exists, landed** | Each module and feature in one of three places: works, with the measured value and the script behind the number; experimental, with what is missing; or removed. The README, the guide and the API reference follow that split. Removals done (CHANGELOG 1.0.0 "Removed"); `memory`, `episode` and `extract_multi` move into the knowledge memory. | in progress |

### Then, in 1.0: two levels and an experimental module

Built on the account above: what works becomes the stable low level, and the high level is put together from it.

| Item | What it gives | Status |
|---|---|---|
| Low level: building blocks with stable extension points | Runtime, parts, questions, storage and traces, guards, deciders, slow paths and dispatch as documented base classes and protocols you can subclass or replace, with a promise of which ones stay stable; your own pieces keep tracing, replay and guarantees | planned |
| High level: ready-made systems | A System 1 + System 2 system in one, where you choose the models and providers and state what you need (questions, examples, risk, budget); built only from the low level, so any piece can be swapped out | planned |
| Clear split in docs and imports | A short path for the high level; a reference for the low level | planned |
| `solvi.experimental` | Pieces that work but are not yet measured well enough for the stable promise live in one submodule, with what they still need to graduate | planned |

## Recipes: a separate repository

| Item | What it gives | Status |
|---|---|---|
| **solvi-ai/recipes** | Subprojects, one per task: how a real task is solved, why it is solved this way, and what the library does in it — and where it does not help. Kept apart from the library so a recipe can use any data, model or service. | planned |

## Library: open items from earlier releases

| Item | What it gives | Status |
|---|---|---|
| **Several questions per pass by default** | The decider answers many questions about one state in one forward pass (≈2–3× faster). Trained with a consistency loss, the answers still differ from one-question-per-pass in ~3–10% of cases on real states, so it stays off; next: an architecture where questions are independent queries over a text encoded once. | research |
| **Which record changed: a trace signature** | `solvi.signature`: 64 bytes kept next to the head; when a stored record was edited and every hash after it and the head recomputed, `verify(signature=...)` names the record and restores its content hash (`candidates=` finds the original in a backup). A syndrome code (two sums mod a 256-bit prime); the positional octonion code, kept for future tree-shaped (derivation) signatures, left the package in 0.8 for `benchmarks/octonion_signature.py`. One changed record: 2000/2000 located, 0 wrong; several: detected, not located. | preview (0.7) |
| **`solvi serve`: HTTP and MCP server** | `solvi serve catalog.py:system` exposes the questions as an HTTP API (FastAPI, OpenAPI schema from the same pydantic types) and as an MCP server (each question a tool), and a decider behind the System One API (`POST /v1/systemone`); traces go to TraceStorage — done for 0.6. Guarding an agent's tool calls (`solvi.agents.Guard`: the agent proposes a call, solvi checks it — catalog, types, grounding in the conversation, instruction-like tool outputs, policies, an authorizer under act_guard and perturb — and makes it, denies it or escalates it, every decision a stored trace), and an MCP proxy (`solvi serve --guard --upstream`) — done for 0.7 (adapters for PydanticAI, LangGraph and the OpenAI Agents SDK came too, and were removed in 1.0: no measured run went through them); hardened before release: approvals bound to the call and the reasons shown, `scan_user` for pasted injections, `locale=` for ambiguous numbers. Measured on AgentDojo (97 tasks, two open models): default settings cut successful attacks by 93–97% but blocked honest tasks that take payees or recipients from tool outputs (67% → 51%, 84% → 56% solved). Added for 0.7, opt-in: `tool_values="escalate"` (such a value goes to a person instead of a refusal), a `"url"` matcher, `require_request` policies for actions with no user-given value, and a wider detector of booking / event / visit commands. All together with a reviewer: attacks 0.6% / 1.6%, honest tasks 72% / 75%, a person asked in 28–31% of honest tasks. Next: measure the new detector rules and intents on attacks they were not written for; a reviewer UI that shows the injected instruction next to the escalated value. | done (0.6); agent guard: preview (0.7); framework adapters removed (1.0) |
| **Verifiable specialists** | Small models for one job each, under one contract (`solvi.specialist`): the model proposes a typed spec, code checks it against the source, code renders it, the trace replays to identical bytes; what does not verify is marked, never invented. First: **verified charts** (`solvi.charts`) — every number on the chart quoted from the text, with its unit and scale; a deterministic, accessible SVG. Next: tables and slides (the same checks, HTML / PPTX output), then checking layers over open speech models (speech to text: silence has no words, forced alignment, a term dictionary; text to speech: numbers and dates normalised by rules, the synthesised audio transcribed back and compared). | charts: preview (0.7); tables, slides, speech: planned |
| **Gallery: helpers for coding agents** | Three runnable entries for decisions a coding agent meets on every task: a pre-edit rule check (rules per path glob; code checks where code can, a decider's question per fuzzy rule under act_guard and perturb; allow / block / escalate), review triage (seven risk questions, quick review only for a confident "no" to all, P(a risky change goes to quick review) ≤ 10%), a skill picker with an honest "none" (ties escalate with their candidates). Offline with keyword stand-ins and synthetic labels. Next: calibrate solvi-large and an LLM decider on labelled real edits, changes and prompts, and publish what the promise costs in escalations. | done (0.7.1); calibration on real edits: planned |

## Models

| Item | What it gives | Status |
|---|---|---|
| **solvi-large** and a distilled **solvi-base** | Typed decisions with evidence quotes, not stated, spans, rankings and numbers; ahead of GLiNER2.5-Decide and Laya on typed questions over JSON states, behind GLiNER2.5-Decide on zero-shot choice questions (level with it after `fit` on ~64 examples). | done (0.5.0, preview) |
| **Decider v3** | New layers around the same pretrained encoder: text encoded once and questions as independent queries (many questions per pass that match one-by-one exactly), cached option vectors (hundreds of options), answers computed only from 1–3 selected evidence passages, several heads whose disagreement is the uncertainty, early exit on CPU, adapters per question kind. Each block ships only if it beats solvi-large on the same data. | research |
| **Long documents** | Beyond the decider's context, find the relevant sections first and decide on them: `long="retrieve"` (sections by headings and paragraphs, BM25, optional rerank by the decider, offsets mapped back, sections read in the trace; `top_k` follows the budget, sections of ≈ 170 tokens). `long="full"` reads a text whole up to the checkpoint's declared `max_len_long` and retrieves within it beyond that (a GPU mode). Measured on 4–8k-token documents: a solvi-large fine-tuned on inputs up to 8k tokens reads them whole at 85% (retrieve at 512 tokens: 78%; untrained solvi-large: 73%), and retrieve with `max_len=2048` matches that at about 3× a 512-token pass on a CPU (reading 4–8k tokens whole: 12–31×). Next: publish a long-input checkpoint that declares `max_len_long` once it keeps quotes and the act signal on short texts. | retrieve, `long="full"` code path: done (0.7); long-input checkpoint: research |
| **Better evidence quotes** | Quotes that actually support the answer on long legal and business texts (today ~23% on ContractNLI). | research |
| **Business-judgement questions** | Better zero-shot answers on operational judgements (typed-decisions style); today use `fit` on 30–60 labeled examples. | research |
| **A LoRA adapter per question** (`part.adapt_lora`) | When a question has ~100 labelled answers or more, `fit` levels off (it only moves the logits); a 3.2 MB LoRA adapter on solvi-base keeps improving: +3 / +4 / +6 / +9 points over `fit` at 32 / 100 / 300 / 1000 examples per process on typed decisions. Trained in-process on a CPU in minutes (the time is measured and reported first), act_guard on held-out labels in the same call (the adapted model is overconfident), the adapter in the fingerprint and every trace, save / load next to the calibration file, `remove_lora` rolls back; solvi-large and big k: `tools/adapt_lora_gpu.py` on a GPU. Next: use it as the learning loop's adapter rung, and decide on real use whether it leaves experimental. | experimental (0.7) |
| **Numeric estimates** | Tighter, calibrated intervals for `Estimate`. | research |
| **extract-base v2** | A stronger zero-shot field extractor, unified with the decider's span pointer. | planned |

## Strategist

| Item | What it gives | Status |
|---|---|---|
| **Reliability-aware producer choice** | Choose among producers by how often each one is accepted and right, learned from outcomes (continues the learned producer policy). | planned |
| **Text in: entry points** | Text → which question is asked + its typed input state: the decider picks the entry point, spans and deterministic parsers fill the fields with quotes; provenance says the input was read by a model. `system.ask_text`, `solvi.textin.TextIn`, dialogue updates; served as `POST /ask_text` and an MCP tool. Next: measure the share of wrong actions that pass the checks on gallery catalogs, and a generative fallback only where a parser fails. | done (0.7) |
| **Learning from corrections, with gates and rollback** | The system learns from every human correction of an escalation, every known outcome and every proposal your rules rejected — never from its own accepted answers (tried three times: no gain). By the number of labels per process: shift/scale (today's `fit` / `teach`), a memory of corrected cases and a small head, then a LoRA adapter (`part.adapt_lora`, experimental in 0.7; not yet wired into the ladder). Every update runs in shadow, is diffed against the current one, must beat it on held-out labels and pass the honesty suite, recalibrates `act_guard` on fresh labels, and is recorded in each trace so it can be rolled back. The scaffold ships in 0.7 as experimental and off by default (`System.learning`: trusted labels only, the ladder, the gates, versions and rollback, tested with stand-in models); the simulation on real data sets decides whether it becomes a default. | experimental (0.7); simulation research |
| **Cascade and memory of corrections** | Decider → larger decider → LLM, escalating by the cost of a mistake; a memory of human-corrected cases (nearest neighbours with an abstain threshold). Any OpenAI-compatible LLM server is a decider (`solvi.llm`: JSON-schema replies validated, invalid output escalates, probabilities from log-probabilities when given), so it can be the last stage of a cascade or a family in a vote. | cascade, vote, route done (0.6); LLM stage and memory done (0.7) |
| **LLM decider confidence** | Log-probabilities from LLMs are saturated (≥ 0.999 on almost every answer), yet the raw score ranks right and wrong well and a threshold taken from its distinct values on the calibration examples (as `act_guard` does) separates them — no transform needed. What broke were fixed scales around it: combinations can share one threshold on each model's rank among the calibration examples (`act_guard(scale="rank")`, opt-in: it helped on one of three measured sets and hurt on two, so compare it with the raw default), a cascade warns when a stage does nothing, and `calibrate_for(method="ltt")` takes its grid from the calibration scores (at most 64 quantiles) instead of a fixed 0.2–0.995. | done (0.7) |
| **Task in words → goal and plan** | A model that turns a natural-language task into the questions to ask and the goal to plan for over a catalog; code still verifies and executes. | research |
| **Name matching across teams** (`solvi.aliases`) | Link parameters and facts whose names differ. No checkpoint of the matcher was ever published; removed in 1.0. | removed (1.0) |
| **Model strategist for plan segments** | The typed decomposer from research as a proposer for ambiguous segments (`ModelStrategist`). No checkpoint was published and it planned no better than a keyword list; removed in 1.0 — the code planner `CostStrategist` stays. | removed (1.0) |

## Games and demos

| Item | What it gives | Status |
|---|---|---|
| **Learned strategist in realms** | A tiny learned policy behind hard laws; leads 23 of 32 new maps over 5,000 turns with zero law violations. | done (0.5.0) |
| **System 1 + System 2 on a game** | The world map of Pokémon Red (recorded, no ROM): rules over compiled routes, a search over the player's world map when they are unsure or surprised, consolidation after each goal; 147 → 2 slow decisions from the first run to the second. Example 23; a video of the run goes to the recipes repository (the Space in `spaces/pokemon/` is not published). Next: walking and battles inside a place, a second world. | done (0.9) |
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
| One entry point for System 1 and System 2 (`solvi.auto.build`): labelled examples, a promise and a slow path in, System 1 fitted, its guarantee and the slow path's slice calibrated on examples it did not see, the store wired, every choice explained; measured on four stand tasks against the hand-written setups | 0.9 |
| Who answers: System 1, the slow path or a person in one recorded decision within a budget (`solvi.dispatch`), calibrated per slice of what System 1 hands over (`Dispatcher.calibrate`) | 0.9 |
| The system report: who answered, the cost, the promise against the stored labels, drift — from the store alone (`System.report`, `solvi report --overview`) | 0.9 |
| A specification compiled into catalog parts without labels (`solvi.compile`, `solvi.sandbox`), a person settling what the drafts dispute (`review=`), a changed text recompiled with the decisions it moves, a compiled policy as an agent guard (`to_guard`) | 0.9 |
| System 2's verified answers as labels for a guarantee's calibration only (`label_source="verified"`); lean search (about twice as fast); the nine-task stand rerun in CI; the Pokémon world-map showcase; the names 0.8 renamed removed; an import structure without cycles | 0.9 |
| Local thinking decision models (Jeeves) through `solvi.systemone`, measured on the vs-LLM benchmark | 0.8 |
| One name per concept across the package (old names warned in 0.8, removed in 0.9), one decider protocol for parts and combinations, one client and error policy for every remote model (`solvi.remote`), `__all__` in every module, `solvi.decide` as a package of layers; README and guide around any model (an LLM first, a local checkpoint for offline use) | 0.8 |
| A model that writes, checked: `solvi.generate`, agreement of candidates (`solvi.agree`), the re-ask loop with checks that say why (`solvi.refine`, `Fail`) | 0.8 |
| A guarantee on any question or signal (`System.guarantee`, `solvi.guarantee`), inputs from outside the calibration set (`solvi.openset`), decisions over a set (`solvi.sets`), search through a System's checks (`solvi.search`) | 0.8 |
| Agent guard: "the user confirmed this" (`require_confirmation`, `accepts`), policies shown to the model, `once=True` everywhere; drift with a sequential test; `System.fit` as one entry point; `ask(early_exit=False)` | 0.8 |
| The audit of 0.7: checks that silently did nothing, replay that checks the answers, verified redactions, stores and calibrations written by 0.7.1 still load | 0.8 |
| A System One client for hosted decision services (`solvi.systemone`: provider routing, "not stated", multi-label, cost and latency); gallery entries for coding agents (pre-edit rules, review triage, skill picker) | 0.7.1 |
| Guarantees per group, robustness to instructions in the input (`perturb`), the project command line (`solvi init`, `ask`, `calibrate`, `models`), counterfactual explanations, reports for people and OpenTelemetry, Postgres and DuckDB stores, hardening of `solvi serve`, a browser check of every Space | 0.7 |
| Fast heads taught by corrections refit on all kept examples each time they double (`fit_fast(refit=2.0)` then, `fit(..., select=False)` since 0.9), so the ridge strength and features chosen on the first few examples do not stay frozen | 0.7 |
| `solvi serve` (HTTP, MCP, System One API), `solvi check`, Cascade / Vote / Route under one guarantee, `aask` with timeouts, costs from measurements | 0.6.0 |
| Honest act thresholds in the shipped models, honest labels in the trace | 0.5.1 |
| Escalation with a guarantee (`act_guard`, learn-then-test, conformal sets), option order and near-tie safeguards, System One backend, honesty suite, `solvi test`, TraceStorage, `solvi diff` and shadow mode | 0.5.1 |
| solvi-large / solvi-base (preview); learned strategist in realms | 0.5.0 |
| Typed facts (pydantic), questions declared by types, answer primitives (not stated, evidence, span, rank, estimate), overall confidence, typed decider API with act/escalate, code strategist (dead ends, exact cost optimum, memoized planning) | 0.5.0 |
| Audit shows what a learned head reads and ignores; `rule_abstained` safeguard | 0.4.1 |
| Provenance, audit and safeguards; decisions with a model | 0.4.0 |

## Not planned (for now)

- Writing prose: solvi decides and extracts, and checks what a model writes (`solvi.generate`: schemas, quotes); it does not
  write prose itself.
- A hosted service: solvi runs in your process, on your data.
