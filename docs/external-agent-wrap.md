# Explicit wrapping of an external agent

```python
import solvi

wrapped = solvi.wrap(
    agent,
    decision_points={"decide": {"doubt": proposer}},
    default=agent.decide,
    components=["doubt"],
    storage="overrides.jsonl",
)
action = wrapped.decide(state)
```

Each configured external method calls its base exactly once. A proposer receives a
fresh finite JSON copy of `{args, kwargs, default}`, never the agent. It returns
None, or `{action, default_loss, loss, evidence: ["explicit-id"]}`. Losses must be
finite, nonnegative and strictly improving; the action must differ. The first valid
selected component wins in declared order. Missing/invalid evidence, callback errors
or unsupported non-JSON analysis return the original base object unchanged. Loss
estimates, evidence validity, common units and proposer purity are owner assumptions:
this adapter neither proves utility nor sandboxes arbitrary Python callbacks.

`enabled=False` or selecting no components bypasses analysis and callbacks entirely:
base output identity, state and RNG continuation are preserved, with no journal open
or write. With several decision points, omit `default`; each forwards to that named
agent method. A supplied `default` belongs to one configured entry point only.
Other methods forward unchanged. An original `agent.run()` calling `self.decide()`
internally is **not** monkeypatched; invoke the wrapped decision point externally.

An accepted override is durably journaled before returning it. A path creates
fsynced nonrotating JSONL (or SQLite for .db/.sqlite/.sqlite3); explicit storage must
be durable JSONL/SQLite and outside an active batch. Missing or failed storage raises
explicitly after the one base call; an unjournaled override is never returned. The
chain records input/default/proposal/evidence, method/component and proposer code
identity; its authentication does not establish that a loss estimate is true.

Selecting components is supported. A training-only paired component-ablation and
selection helper is a separate next stage; no tuning/test search or historical arena
utility result is claimed by this adapter.
