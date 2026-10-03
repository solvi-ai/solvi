# The code strategist: dead ends and costs

See also: [the guide](guide.md#how-the-strategist-plans-a-flow) for the default strategist.

solvi's default strategist plans a flow by exact names: a parameter's name is the fact it reads, and for a fact with several
producers (`provides=`) it needs the inputs of all of them — a fallback chain in declaration order. So it cannot
**plan around a dead end** — one producer of a fact whose inputs are never given makes the whole fact unreachable — nor
**choose between interchangeable producers** by what they cost. `solvi.strategy.CostStrategist` does both, in code: no
model takes part, and only verified plans run.

(Up to 0.9 solvi also had an experimental model strategist, `ModelStrategist` with `solvi.segment_model`, and a name
matcher for parameters whose names match no fact, `solvi.aliases`. Neither had a published checkpoint, the segment
model planned no better than a short keyword list over the docstrings, and declaring `cost=` makes both unnecessary;
both were removed in 1.0.)

## Planning: `CostStrategist`

```python
from solvi import System
from solvi.strategy import CostStrategist

System(cat, questions, strategist=CostStrategist())                                   # 1. dead ends dropped
System(cat, questions, strategist=CostStrategist(producers="equivalent"))             # 2. cheapest verified plan
```

1. **`producers="declared"` (default).** The deterministic strategist's plan, except that producers whose inputs cannot be
   computed are dropped (they no longer make the fact unreachable). The remaining producers keep their declaration order as
   a run-time fallback chain. By design, wherever the deterministic strategist answers, the answers are the same.
2. **`producers="equivalent"`.** You declare that the producers of a fact are interchangeable (any accepted output is the
   same fact). Then *points* are computed by code: an exact 0/1 program (scipy's HiGHS; branch and bound as a fallback)
   picks one producer per needed fact — the cheapest valid plan by declared `cost=` (a part without one counts 1).
   **Mandatory milestones**: every hard check that governs a question in the plan the deterministic strategist would build
   stays in every plan, so a shortcut that skips the fact such a check reads cannot drop the check. The other producers
   stay as run-time fallbacks when their inputs are computed anyway.

Facts that can be derived from each other (`net` from `gross` and `gross` from `net`, each also given directly) work in
both modes: the plan computes each fact from what is given, and a producer that would read its own fact back — through
any other fact — is not kept as a fallback, since the flow could not run it. `solvi check` reports such a loop as the
note `mutual_producers` for a System with a strategist (an error, `cycle`, only for the deterministic strategist, which
cannot plan it).

A plan that fails the final check (inputs bound, order acyclic, producer and reader types fit, every mandatory check
kept) is used with the failure recorded (`on_failure="code"`), or falls back to the deterministic strategist
(`"deterministic"`), or every question abstains (`"abstain"`). (`CostStrategist` was `ModelStrategist()` without a model
up to 0.7, and its options `fallback=` / `fallbacks=` are `on_failure=` / `keep_alternatives=`; the old spellings were
removed in 0.9.)

Everything that plans for a System uses its strategist, not only `ask`: `answers_of` / replay, `facts_for` and so the
feature candidates of `fit`, `learn_order`, the input schemas of `solvi serve` and `solvi check`.

### Costs from measurements

With no `cost=` declared, interchangeable producers tie and the first declared wins. `System(cat, questions,
producers="equivalent", cost_policy="measured")` (the same as `strategist=CostStrategist(producers="equivalent")` plus
`cost_policy="measured"`) plans with the run times solvi measures anyway (`system.cost_book`, a moving average in ms per part):

- **Warm-up.** A producer counts its measured time once it has run `min_samples` times (default 3); before that its
  declared `cost=`, or 0 ms when it declares none — so each undeclared producer gets chosen, and measured, in turn.
- **Smoothing and adapting.** The moving average weighs the newest run by `alpha` (CostBook's 0.3 unless set). When the
  producer in use slows down, its average rises above the others' and the next plans switch.
- **Recheck.** A producer not run for `recheck` asks (default 50) counts 0 ms for one plan, so a source that got faster
  again is noticed (`recheck=None`: never).
- **Freeze.** `system.freeze_costs()` fixes the planner's costs at what was measured (a producer that never ran: its
  declared cost, else 1) — the choice stops changing, measuring goes on; `system.unfreeze_costs()` resumes.

```python
from solvi.costs import MeasuredCosts
system = System(cat, questions, producers="equivalent", cost_policy=MeasuredCosts(min_samples=5, recheck=100, alpha=0.2))
```

Every plan record then carries `extra["costs"]`: `{"mode": "measured" | "frozen", "facts": {fact: {"chosen",
"producers": {name: {"ms", "from", "runs"}}, "why"}}}` — per fact with several usable producers, the cost of each and where
it came from (`measured`, `declared`, `warm-up`, `recheck`, `frozen: measured` ...), e.g. `"why": "cheapest plan: rate_table
measured 0.012 ms ×5; rate_live measured 301.448 ms ×3"`. The choice is made on the costs of the whole plan (a producer
that reads an expensive fact pays for it too). Since measured costs depend on timing, so does this record's hash — it
says what was known when the plan was made; the rest of the trace is as always.

`strategist.last` is the report of the last plan: the choice per fact, the mandatory checks, the plan's cost, the
fallback if any, and the time.

### In the trace

A planned flow adds one hashed record at the end of the trace: kind `plan`, name `plan:strategy`, value = the chosen
producer per fact and the mandatory checks; `extra` = the strategist (and the fallback, and the measured costs, if
any; `segments` is always empty since the model strategist was removed). Provenance is `computed`.
`trace.replay(system)` re-verifies the record against the catalog (every chosen producer exists, provides its fact, and
its inputs are given or chosen); a tampered record breaks the hash chain. Facts whose producer group was narrowed replay
with the producers that ran.
