# Compare a complete candidate process

`Process.shadow(candidate, storage=None, keep=1000)` compares two complete
sequential graphs of native `System` nodes. Unlike a comparison of one System,
each side routes its own upstream answers through its own downstream nodes.

```python
# live and candidate are separately assembled Process instances.
# Candidate must use on_abstain="abstain", on_human="abstain", without a human channel.
shadow = live.shadow(candidate, storage="shadow-index.jsonl")
served = shadow.run({"claim": claim})
comparison = shadow.last["comparison"]
candidate_result = shadow.last["candidate"]
```

The returned object has `run`, `stats`, `last`, `changed`, and `errors`. `run`
returns the current process's actual result. It runs the candidate once on a
separate copy of the original input. Candidate failure is counted and recorded
without replacing the served result. If the current process fails, its exception
propagates and the candidate is not called. Input-copy and configuration failures
happen before either run. A comparison-index write failure remains visible in
`errors`; it does not undo a successfully served result.

An output difference means status or final answer differs. `path_changed` reports
a changed sequence of node names separately. `keep` bounds in-memory histories.
The optional comparison journal stores hashes, specification identities and links
to both process records, without raw fact/answer values or exception text. A failed
candidate contributes its exception type and a message hash. Each side retains
its normal process/node records and can replay them using its own `replay` API;
the comparison journal does not replace these records.

Current and candidate must have distinct native System bindings, guards and
journals. The comparison journal must be separate too. Checks run before every
request and detect path, symlink and hardlink aliases.
Native JSONL, SQLite and DuckDB journals and file paths are supported; PostgreSQL
and custom backends are refused because their separate object identities do not
establish separate underlying relations. Journal sidecars and existing JSONL
rotation segments are included in alias checks. The candidate cannot have
human nodes, a human channel or automatic handoff. Dispatcher and Guard nodes
are not supported by this first graph-comparison API. A candidate may abstain.

Use `Knowledge.branch()` separately to prepare alternative knowledge state, and
bind that state into explicitly constructed candidate Systems. This API does not
clone arbitrary models, functions or their closures. Python callbacks and model
backends can still share external state or perform I/O; these effects are not
sandboxed or rolled back. Supply candidate callbacks whose effects are acceptable.
The two runs are sequential, without a cross-process transaction or concurrent
ownership guarantee. No statistical quality certificate, automatic promotion,
merge, or distributed protocol follows from a comparison.
