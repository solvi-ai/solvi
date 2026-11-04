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
be durable JSONL/SQLite and outside an active batch. SQLite must have a real main
database file, synchronous FULL/EXTRA, and a disk journal; memory/temp databases
and OFF/MEMORY journal modes are refused without changing owner settings. Missing or failed storage raises
explicitly after the one base call; an unjournaled override is never returned. The
chain records input/default/proposal/evidence, method/component and proposer code
identity; its authentication does not establish that a loss estimate is true.

Selecting components and the training-only paired comparison below are supported.
No holdout tuning or historical arena utility result is claimed.

## Training-only component comparison

```python
from solvi.solutions.wrap import select_components

result = select_components(
    factory, evaluator, decision_points={"decide": {"doubt": proposer}},
    seeds=[7, 29, 101], directory="training-comparison-01",
)
```

`factory(seed)` constructs a fresh agent with its own seeded state/RNG; the helper
never clones a baseline or shares one agent between configurations. `evaluator(agent,
seed)` returns a finite utility. Baseline and each single component use the same
explicit seeds. Optional `configurations=[("doubt", "calibration")]` declares actual
combinations to evaluate instead of single components: positive single effects are
never assumed to combine. Configuration/seed snapshots are frozen before callbacks.

Only preregistered configurations are run. `max_evaluations` caps attempted cells,
including failed factories/evaluations. Optional `costs={(): 0, ("doubt",): 1}`
records declared per-cell costs; it neither measures costs nor subtracts them from
utility. Incorporate an actual utility/cost tradeoff explicitly in the evaluator.

Each cell gets a unique override-journal path. Disabled baseline does not create
that journal. Manifest and each utility/error record are fsynced incrementally;
exceptions/nonfinite utility are logged and later cells continue. Reused agent
objects are rejected before a second evaluation when detectable by identity. A new
output directory is required: partial results are preserved, not silently resumed
or overwritten. Objects lacking weak-reference support are retained for identity
checks, so there is no bounded-memory guarantee for arbitrary factories.

Results expose paired deltas, pair counts and missing/error cells. Only complete
paired configurations can be recommended; baseline wins a zero-gain tie. Overflowing
differences are explicitly invalid, not successful improvements. Selection is purely
descriptive on these training seeds. Holdout data is neither accepted nor touched;
there is no confidence interval, utility theorem or performance certificate.
