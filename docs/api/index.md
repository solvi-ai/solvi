# API reference

Generated from the docstrings of the public modules. The [guide](../guide.md) explains how the parts fit together;
this reference lists what each module exports and the signatures.

| Module | What it holds |
|---|---|
| [`solvi`](solvi.md) | the entry points `build`, `Agent`, `Guard` and `Knowledge`, `Budget`, and the shared vocabulary: catalog, questions and answers, `System` and `Response`, typed answers |
| [`solvi.core`](core.md) | the low level: the catalog and its value classes, and the extension points (protocols and base classes, see [Building blocks](../building_blocks.md)); one module per area under `solvi.core` |
| [`solvi.core.deciders`](decide.md) | the decider: typed questions about a text or a state answered by a model |
| [`solvi.core.deciders.protocols`](protocols.md) | the extension points of the parts: `Scorer`, `Decider`, `Adapter`, `Head` |
| [`solvi.core.guarantees.monitor`](monitor.md) | the `Monitor` protocol: a watcher of the decision stream |
| [`solvi.core.environment`](environment.md) | the `Environment` protocol and `Outcome`: a world an agent acts in |
| [`solvi.core.calibration`](calibration.md) | reliability, expected calibration error, coverage at a target accuracy |
| [`solvi.core.guarantees.guarantee`](guarantee.md) | a calibrated threshold with a stated promise on a question (`System.guarantee`), a computed fact or any scalar |
| [`solvi.core.guarantees.openset`](openset.md) | inputs from outside the calibration set: a threshold sized for the share of them, estimated as the stream goes, and a change flag |
| [`solvi.core.deciders.systemone`](systemone.md) | a model behind the System One HTTP API as a decider |
| [`solvi.core.deciders.llm`](llm.md) | an OpenAI-compatible chat-completions server (an LLM) as a decider |
| [`solvi.core.deciders._remote`](remote.md) | the client every remote model shares: retries, token counts, one policy for HTTP errors |
| [`solvi.core.deciders.combine`](multi.md) | cascade, vote and route over several models |
| [`solvi.core.slow.generate`](generate.md) | generation by an LLM: a text or validated JSON, quotes checked in a text, recorded as a model's output |
| [`solvi.core.slow.agree`](agree.md) | agreement of generated candidates under a key: the chosen one, its share as a fact, the tally |
| [`solvi.core.slow.refine`](refine.md) | a check that says why (`Fail`); the loop propose → check → re-ask with the reasons → escalate |
| [`solvi.experimental.compile`](compile.md) | a specification compiled by an LLM into catalog parts citing its clauses, accepted when two drafts agree and the spec's tests pass; a person for what the drafts dispute; versions, recompile, the decisions a change moves, an agent guard |
| [`solvi.experimental.compile.sandbox`](sandbox.md) | code a model wrote, held to pure functions: ast allowlist, a subprocess with limits, a restricted load |
| [`solvi.core.knowledge`](knowledge.md) | what a system knows across decisions: the knowledge store, write gates, the action model, risk policies, the agenda, the failure memory |
| [`solvi.core.knowledge.store`](knowledge_store.md) | the knowledge store: a hash-chained journal of items with provenance, disputes, retraction, staleness flags, snapshots |
| [`solvi.core.knowledge.gates`](knowledge_gates.md) | write gates: `WriteGate`, the source check, the consistency check |
| [`solvi.core.knowledge.actions`](knowledge_actions.md) | the action model: accept / refuse / unknown with risk and support, learned over an explicit vocabulary |
| [`solvi.core.knowledge.risk`](knowledge_risk.md) | protection (`Protect`), justified risk (`RiskBudget`), a bounded learned gate (`LearnedGate`) |
| [`solvi.core.knowledge.agenda`](knowledge_agenda.md) | goals with done checks in code, gates, order; a dry run that validates gates |
| [`solvi.core.knowledge.failures`](knowledge_failures.md) | do not repeat a plan that failed recently: a hard check with an expiry and a bound |
| [`solvi.core.knowledge.memory`](memory.md) | a memory of corrected cases: nearest neighbours with an abstain threshold; with a store, its correction facts |
| [`solvi.experimental.lora`](lora.md) | a LoRA adapter per question, fitted on a few hundred labelled examples |
| [`solvi.experimental.learning`](learning.md) | learning from corrections with gates and rollback (experimental) |
| [`solvi.experimental.oncalib`](oncalib.md) | recalibration on the fly from outcomes (experimental; the promise is not kept) |
| [`solvi.serve`](serve.md) | `solvi serve`: HTTP, MCP and the System One API |
| [`solvi.solutions.guard`](agents.md) | guarding an agent's tool calls: `Guard`, the MCP proxy |
| [`solvi.solutions.agent`](solutions_agent.md) | the environment agent: `Agent` — System 1 on what the knowledge predicts, System 2 a search, hard checks, replay |
| [`solvi.solutions.knowledge`](solutions_knowledge.md) | `Knowledge`: the store, agenda, action model and failure memory behind one object |
| [`solvi.experimental.hooks`](hooks.md) | a coding agent's hooks: edits checked against rules, a skill picked for a prompt, install / uninstall (Claude Code; Codex, preview) |
| [`solvi.core.textin`](textin.md) | text in: a message → the question it asks and its typed input state, read with quotes |
| [`solvi.core._inputs`](inputs.md) | what a question reads: its given facts, the model and JSON schema of its input state, entry points |
| [`solvi.experimental.specialist`](specialist.md) | the specialist contract (preview): propose a typed spec, check it against the source, render, replay |
| [`solvi.experimental.charts`](charts.md) | verified charts (preview): a text with numbers → an SVG in which every number is quoted from the text |
| [`solvi.core.deciders.longdoc`](longdoc.md) | long texts: sections, BM25 retrieval, the window a decider reads |
| [`solvi.core.knowledge.episodes`](episode.md) | an agent's memory (what was tried, what failed, what worked) as an input |
| [`solvi.core.knowledge.worldmap`](worldmap.md) | a map of an environment an agent builds by acting: claims with provenance, a view over "leads_to" facts |
| [`solvi.core.guarantees.drift`](drift.md) | drift: has the stream moved away from the one the thresholds were calibrated on? |
| [`solvi.core.deciders.heads`](heads.md) | the learned answer heads: `System.fit`'s `FastHead` and `select_features`, `CandidateHead`, `Head` (`solvi.fast` up to 0.7) |
| [`solvi.core.extract`](extract_long.md) | `@extract` by description for long documents (`LongSpanExtractor`; torch) |
| [`solvi.core.plan.strategist`](strategist.md) | the strategist: a flow per request (`Flow`, `PlanError`), the learned order of hard checks and producer policy |
| [`solvi.core.plan.cost`](strategy.md) | the code strategist (`CostStrategist`): dead ends dropped, the cheapest verified plan |
| [`solvi.core.costs`](costs.md) | measured run times (`CostBook`), the dollars of recorded model calls (`price_of`), `Budget` and `Cost` |
| [`solvi.core.deciders.rulelist`](rulelist.md) | a readable rule list learned from examples (`System.learn_rule`; `solvi.rules` up to 0.7) |
| [`solvi.core.store.audit`](audit.md) | `res.audit()`: what each answer rests on and which safeguards fired; `to_dict()` as JSON-ready data |
| [`solvi.show`](show.md) | printing a response |
| [`solvi.core.store.report`](report.md) | reports of one decision or a period of stored decisions (Markdown, HTML, data) |
| [`solvi.core.store.sysreport`](sysreport.md) | the system report: who answered, the cost, the promise against the stored labels, drift — from the store alone |
| [`solvi.core.sets`](sets.md) | decisions over a set: answers of many items made consistent under at-most / exactly-one / capacity constraints |
| [`solvi.core.slow.search`](search.md) | search over alternatives: candidates through the System's checks, the best by an objective, pruned, recorded |
| [`solvi.solutions.decisions`](auto.md) | one entry point: a question, labelled examples, a promise and a slow path → System 1 fitted, its guarantee and the dispatcher calibrated, the store wired, the choices explained (preview) |
| [`solvi.core.dispatch`](dispatch.md) | who answers: System 1 within its guarantee, a slow path (a System, refine or search) or a person — one recorded decision per input, with a budget (experimental) |
| [`solvi.experimental.counterfactual`](counterfactual.md) | counterfactuals: the smallest change of the given inputs that changes an answer (experimental) |
| [`solvi.core.deciders.perturb`](perturb.md) | instruction-like sentences in an input and the variants without them (`perturb=k`) |
| [`solvi.core.store`](storage.md) | TraceStorage: JSONL, SQLite, PostgreSQL and DuckDB stores, queries, replay |
| [`solvi.core.store.signature`](signature.md) | a signature of a trace or a store that names the one changed record (preview) |
| [`solvi.core.store.diff`](diff.md) | `solvi diff` and shadow mode |
| [`solvi.testing`](testing.md) | decision regression tests from `cases.json` |
| [`solvi.testing.conformance`](conformance.md) | conformance checks for your implementations of the extension points of `solvi.core` (stores, slow paths, strategists, heads, deciders, extractors, monitors, environments) |
| [`solvi.testing.honesty`](honesty.md) | the honesty suite |
| [`solvi.check`](check.md) | `solvi check`: catalog lint |
| [`solvi.core.calibfile`](calibfile.md) | calibration files (`save_calibration` / `load_calibration`) and `solvi calibrate` |
| [`solvi.models`](models.md) | the models by name — `decider("solvi-base")`, `llm(url, model)`, `systemone(url, model)` — and the published and cached deciders (`solvi models`: list, pull, check) |
| [`solvi.cli._scaffold`](scaffold.md) | `solvi init`: a new decision project |
| [`solvi.core._i18n`](i18n.md) | languages of rendering: the audit, `show` and the safeguard report in Russian |
