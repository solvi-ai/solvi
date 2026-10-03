# API reference

Generated from the docstrings of the public modules. The [guide](../guide.md) explains how the parts fit together;
this reference lists what each module exports and the signatures.

| Module | What it holds |
|---|---|
| [`solvi`](solvi.md) | the catalog, questions and answers, `System` and `Response`, typed facts, storage and shadow mode |
| [`solvi.decide`](decide.md) | the decider: typed questions about a text or a state answered by a model |
| [`solvi.calibration`](calibration.md) | reliability, expected calibration error, coverage at a target accuracy |
| [`solvi.guarantee`](guarantee.md) | a calibrated threshold with a stated promise on a question (`System.guarantee`), a computed fact or any scalar |
| [`solvi.openset`](openset.md) | inputs from outside the calibration set: a threshold sized for the share of them, estimated as the stream goes, and a change flag |
| [`solvi.systemone`](systemone.md) | a model behind the System One HTTP API as a decider |
| [`solvi.llm`](llm.md) | an OpenAI-compatible chat-completions server (an LLM) as a decider |
| [`solvi.remote`](remote.md) | the client every remote model shares: retries, token counts, one policy for HTTP errors |
| [`solvi.multi`](multi.md) | cascade, vote and route over several models |
| [`solvi.generate`](generate.md) | generation by an LLM: a text or validated JSON, quotes checked in a text, recorded as a model's output |
| [`solvi.agree`](agree.md) | agreement of generated candidates under a key: the chosen one, its share as a fact, the tally |
| [`solvi.refine`](refine.md) | a check that says why (`Fail`); the loop propose → check → re-ask with the reasons → escalate |
| [`solvi.compile`](compile.md) | a specification compiled by an LLM into catalog parts citing its clauses, accepted when two drafts agree and the spec's tests pass; large specs in groups; versions, recompile, the decisions a change moves |
| [`solvi.sandbox`](sandbox.md) | code a model wrote, held to pure functions: ast allowlist, a subprocess with limits, a restricted load |
| [`solvi.memory`](memory.md) | a memory of corrected cases: nearest neighbours with an abstain threshold |
| [`solvi.lora`](lora.md) | a LoRA adapter per question, fitted on a few hundred labelled examples |
| [`solvi.learning`](learning.md) | learning from corrections with gates and rollback (experimental) |
| [`solvi.serve`](serve.md) | `solvi serve`: HTTP, MCP and the System One API |
| [`solvi.agents`](agents.md) | guarding an agent's tool calls: `Guard`, the MCP proxy (adapters for PydanticAI, LangGraph, the OpenAI Agents SDK) |
| [`solvi.hooks`](hooks.md) | a coding agent's hooks: edits checked against rules, a skill picked for a prompt, install / uninstall (Claude Code; Codex, preview) |
| [`solvi.textin`](textin.md) | text in: a message → the question it asks and its typed input state, read with quotes |
| [`solvi.specialist`](specialist.md) | the specialist contract (preview): propose a typed spec, check it against the source, render, replay |
| [`solvi.charts`](charts.md) | verified charts (preview): a text with numbers → an SVG in which every number is quoted from the text |
| [`solvi.longdoc`](longdoc.md) | long texts: sections, BM25 retrieval, the window a decider reads |
| [`solvi.many`](many.md) | a choice among many options: a shortlist, then the decider |
| [`solvi.episode`](episode.md) | an agent's memory (what was tried, what failed, what worked) as an input |
| [`solvi.worldmap`](worldmap.md) | a map of an environment an agent builds by acting: claims with provenance |
| [`solvi.drift`](drift.md) | drift: has the stream moved away from the one the thresholds were calibrated on? |
| [`solvi.heads`](heads.md) | the learned answer heads: `System.fit`'s `FastHead` and `select_features`, `CandidateHead`, `Head` (`solvi.fast` up to 0.7) |
| [`solvi.extract_long`](extract_long.md) | `@extract` by description for long documents (`LongSpanExtractor`; torch) |
| [`solvi.extract_multi`](extract_multi.md) | a single-pass `@extract` over several fields (torch) |
| [`solvi.strategist`](strategist.md) | the strategist: a flow per request (`Flow`, `PlanError`), the learned order of hard checks and producer policy |
| [`solvi.strategy`](strategy.md) | the model strategist (experimental, `ModelStrategist`) |
| [`solvi.costs`](costs.md) | measured run times (`CostBook`) and the planner's measured costs (`MeasuredCosts`) |
| [`solvi.rulelist`](rulelist.md) | a readable rule list learned from examples (`System.learn_rule`; `solvi.rules` up to 0.7) |
| [`solvi.aliases`](aliases.md) | name matching between parameters and facts (experimental) |
| [`solvi.audit`](audit.md) | `res.audit()`: what each answer rests on and which safeguards fired; `to_dict()` as JSON-ready data |
| [`solvi.show`](show.md) | printing a response |
| [`solvi.report`](report.md) | reports of one decision or a period of stored decisions (Markdown, HTML, data) |
| [`solvi.sysreport`](sysreport.md) | the system report: who answered, the cost, the promise against the stored labels, drift — from the store alone |
| [`solvi.otel`](otel.md) | decisions as OpenTelemetry spans (the API, or OTLP/JSON) |
| [`solvi.sets`](sets.md) | decisions over a set: answers of many items made consistent under at-most / exactly-one / capacity constraints |
| [`solvi.search`](search.md) | search over alternatives: candidates through the System's checks, the best by an objective, pruned, recorded |
| [`solvi.dispatch`](dispatch.md) | who answers: System 1 within its guarantee, a slow path (a System, refine or search) or a person — one recorded decision per input, with a budget (experimental) |
| [`solvi.counterfactual`](counterfactual.md) | counterfactuals: the smallest change of the given inputs that changes an answer |
| [`solvi.perturb`](perturb.md) | instruction-like sentences in an input and the variants without them (`perturb=k`) |
| [`solvi.storage`](storage.md) | TraceStorage: JSONL, SQLite, PostgreSQL and DuckDB stores, queries, replay |
| [`solvi.signature`](signature.md) | a signature of a trace or a store that names the one changed record (preview) |
| [`solvi.diff`](diff.md) | `solvi diff` and shadow mode |
| [`solvi.testing`](testing.md) | decision regression tests from `cases.json` |
| [`solvi.honesty`](honesty.md) | the honesty suite |
| [`solvi.check`](check.md) | `solvi check`: catalog lint |
| [`solvi.calibfile`](calibfile.md) | calibration files (`save_calibration` / `load_calibration`) and `solvi calibrate` |
| [`solvi.models`](models.md) | `solvi models`: published and cached deciders, pull, check |
| [`solvi.scaffold`](scaffold.md) | `solvi init`: a new decision project |
| [`solvi.i18n`](i18n.md) | languages of rendering: the audit, `show` and the safeguard report in Russian |
