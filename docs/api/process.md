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

## System participation declarations

A bare `System` can declare its role before it is assembled into a process:

```python
triage.participate(writes={"kind": "answer"})
payout.participate(after="triage", when={"answer": "kind", "eq": "glass"}, end=True)
p = Process("claims", storage="process.jsonl").participants({"triage": triage, "payout": payout})
```

`System.participate` returns the System. `after=None` declares an entry; otherwise
`after` names exactly one predecessor. `when` uses the existing closed Process
predicate language and observes that predecessor's outcome. An entry cannot have
`when` or `max_iter`. `reads`, `writes`, `question` and `end` have the same meaning
as on `Process.node`. Node names belong to the process mapping, so one System's
declaration can be used in different processes with the same names.

`max_iter=n` marks the incoming edge as a loop with at most `n` traversals per
run. It is a positive integer (bool and float are refused). It is **not** a bound
on total System invocations. For example, a self-loop with `max_iter=2` can run
its node three times: the initial entry plus two loop traversals. A graph with
such a loop still needs a reachable end node and may need an explicit
`Process(start=...)`. Mapping order sets outgoing edge priority. Existing graph,
loop and merge validation applies; no unbounded cycle or multi-input scheduler
is introduced. For multiple predecessors or more involved routing use the
existing explicit node/edge API on nodes without participation declarations.

`participants` validates and installs the batch atomically. It snapshots detached
JSON declarations and compiles them to existing nodes and edges; declarations
are included in the process specification and hash. Editing returned specs or
caller predicate dictionaries cannot change the process. `from_spec` validates
that declarations agree with compiled nodes and incoming edges; when binding a
System it requires the same declaration, including with `strict=False` (that
option relaxes fingerprints, not participation). Redeclaring a bound System does
not edit the frozen process: its next run fails before invoking any node. Build
a new process to adopt the changed declaration. Replay follows the recorded
specification and existing loop counters. The existing trace replay may recompute
catalog parts; it does not assemble a new graph from live declarations.

This experimental contract describes one sequential path. It does not implement
runtime negotiation, parallel joins, time triggers or distributed participation.
