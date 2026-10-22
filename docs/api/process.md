# `solvi.experimental.process`

A process of several solvi systems (stages 0 and 1a of `solvi.Process`): nodes, edges with predicates written as data,
merges, loops with a mandatory `max_iter` and the declared process promise, as a JSON spec with a hash, a structural
diff and a drawing; `run` / `explain` through the node runners (`DecisionNode`, `SystemNode`, `GuardNode`; `HumanNode`
is the hook of the next stage), one run chained across the node stores and a `"process"` index record (see
[experimental](../experimental.md#process)).

::: solvi.experimental.process
