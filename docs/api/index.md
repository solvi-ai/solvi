# API reference

Generated from the docstrings of the public modules. The [guide](../guide.md) explains how the parts fit together;
this reference lists what each module exports and the signatures.

| Module | What it holds |
|---|---|
| [`solvi`](solvi.md) | the catalog, questions and answers, `System` and `Response`, typed facts, storage and shadow mode |
| [`solvi.decide`](decide.md) | the decider: typed questions about a text or a state answered by a model |
| [`solvi.calibration`](calibration.md) | reliability, expected calibration error, coverage at a target accuracy |
| [`solvi.systemone`](systemone.md) | a model behind the System One HTTP API as a decider |
| [`solvi.multi`](multi.md) | cascade, vote and route over several models |
| [`solvi.serve`](serve.md) | `solvi serve`: HTTP, MCP and the System One API |
| [`solvi.agents`](agents.md) | guarding an agent's tool calls: `Guard`, the MCP proxy (adapters for PydanticAI, LangGraph, the OpenAI Agents SDK) |
| [`solvi.storage`](storage.md) | TraceStorage: JSONL and SQLite stores, queries, replay |
| [`solvi.diff`](diff.md) | `solvi diff` and shadow mode |
| [`solvi.testing`](testing.md) | decision regression tests from `cases.json` |
| [`solvi.honesty`](honesty.md) | the honesty suite |
| [`solvi.check`](check.md) | `solvi check`: catalog lint |
| [`solvi.calibfile`](calibfile.md) | calibration files (`save_calibration` / `load_calibration`) and `solvi calibrate` |
| [`solvi.models`](models.md) | `solvi models`: published and cached deciders, pull, check |
| [`solvi.scaffold`](scaffold.md) | `solvi init`: a new decision project |
| [`solvi.i18n`](i18n.md) | languages of rendering: the audit, `show` and the safeguard report in Russian |
