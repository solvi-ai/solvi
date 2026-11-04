# `solvi.experimental.process`

A process of several solvi systems (stages 0 to 4 of `solvi.Process`): nodes, edges with predicates written as data,
merges, loops with a mandatory `max_iter` and the declared process promise, as a JSON spec with a hash, a structural
diff and a drawing; `run` / `explain` through the node runners (`DecisionNode`, `SystemNode`, `GuardNode`), one run
chained across the node stores and a `"process"` index record; questions to a person through a queue — the abstract
`HumanChannel` (check your own with `check_human_channel`), the built-ins `MemoryQueue` and
`CallbackChannel`, `HumanRequest` / `HumanAnswer`, a node of kind `"human"` and the hand-off, `pending` / `resume` /
`cancel` with each segment of a run a `"process"` record and each answer a `"human"` record; `replay` / `replay_all`
of a run from its records across its segments, a person's answer a leaf recorded and not recomputed, the first
divergence named by node and step; the path promise — `guarantee()` in the modes `union` (split `"equal"`,
`{node: r_i}` or a function, Σ r_i ≤ r, each node's own promise at its share, loops counted), `per_node` ("the end
error is not bounded") and `end_to_end` (`calibrate` on end-labelled runs, learn then test over a `ThresholdFamily`:
the built-ins `SharedRank` — `"shared_rank"`, the default — and `PerNodeFixed` — `"per_node_fixed"` —, or your own;
the threshold applied at run time and re-checked by `replay`), `ProcessResult.automatic` / `risk`, and `PathWatch` on
the mix of paths (`monitor`) (see
[experimental](../experimental.md#process)).

`Process.node(..., input_contracts=..., output_contracts=...)` optionally declares
versioned boundaries for facts named in `reads`/`writes`. Each mapping value is
`{"version": "quote/1", "schema": {...}, "invariants": [...]}`. Versions are opaque
nonempty strings compared exactly. Declarations are copied and included in the
process spec/hash; normal strict System fingerprint checks still apply.

Schemas support an explicit `type` at every level: object, array, string, number,
integer, boolean or null. Supported keywords are `enum`; object `properties`,
`required`, boolean `additionalProperties`; array `items`, `minItems`, `maxItems`;
string `minLength`, `maxLength`; numeric `minimum`, `maximum`. Unsupported keywords
and incompatible keyword/type combinations are rejected. This is a bounded subset,
not a full JSON Schema implementation. Payloads are finite JSON with string object
keys, without coercion; booleans are not numbers. Nesting is limited to 32 levels.
`enum` uses canonical JSON representation equality recursively: object key order
does not matter, while `true`, `1` and `1.0` are distinct, including inside arrays
and objects. This stricter numeric equality is part of this subset's semantics.

Object payload invariants use the existing predicate forms `answer` with one of
`eq`/`in`/`lt`/`gt`, combined by `and`/`or`/`not`; `answer` names a payload field.
No arbitrary callbacks or runtime metadata predicates are accepted. An absent
field satisfies no comparison. Schema validation precedes invariants.

Output validation runs before the produced facts are published. Input validation
checks the exact producer version, schema and invariants before invoking the
consumer. Failures raise `ProcessRunError`; the producer may already have run and
saved its decision. Unversioned initial facts cannot satisfy an opted-in input
contract. Nodes without these declarations retain their existing behavior. Run
and replay use the same boundary checks; no compatibility negotiation is implied.

::: solvi.experimental.process
