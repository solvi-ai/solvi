# `solvi.experimental.process`

A process of several solvi systems (stages 0 to 3 of `solvi.Process`): nodes, edges with predicates written as data,
merges, loops with a mandatory `max_iter` and the declared process promise, as a JSON spec with a hash, a structural
diff and a drawing; `run` / `explain` through the node runners (`DecisionNode`, `SystemNode`, `GuardNode`), one run
chained across the node stores and a `"process"` index record; questions to a person through a queue — the abstract
`HumanChannel` (check your own with `solvi.testing.conformance.check_human_channel`), the built-ins `MemoryQueue` and
`CallbackChannel`, `HumanRequest` / `HumanAnswer`, a node of kind `"human"` and the hand-off, `pending` / `resume` /
`cancel` with each segment of a run a `"process"` record and each answer a `"human"` record; `replay` / `replay_all`
of a run from its records across its segments, a person's answer a leaf recorded and not recomputed, the first
divergence named by node and step; the path promise — `guarantee()` in the modes `union` (split `"equal"`,
`{node: r_i}` or a function, Σ r_i ≤ r, each node's own promise at its share, loops counted) and `per_node` ("the end
error is not bounded"), `ProcessResult.automatic` / `risk`, and `PathWatch` on the mix of paths (`monitor`) (see
[experimental](../experimental.md#process)).

::: solvi.experimental.process
