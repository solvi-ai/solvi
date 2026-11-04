# `solvi.solutions.knowledge`

`solvi.Knowledge`: the knowledge store, the agenda, the action model and the failure memory behind one object that
`solvi.build`, `solvi.Guard` and `solvi.Agent` share. See [Using solvi: agents and knowledge](../agent.md).

`Knowledge.redecide(storage, decide, affected=[...], valid_at_key="date")` reconsiders decisions using current knowledge
at each decision input's explicitly named world date. Tell the replacement with its validity interval and retain the
earlier assertion for dates where it was valid. Global retraction removes that support and is a different operation.
The method preserves snapshot query scope and prepares all selected dates/snapshots before running callbacks.

It returns `answer_changes` and `justification_only`, without saving new decisions. Each side is evaluated by the
current callable or System: the old side reruns that callable on the historical snapshot, rather than loading the
historical model's prediction. Callbacks retain their external effects. Dates must be canonical ISO dates or date
objects; outputs must be finite JSON values. System `Unknown` answers carry an explicit marker plus
`old_unknown_paths` / `new_unknown_paths`, distinguishing them from literal JSON with the same marker shape.
Ordinary `retract` and the low-level historical counterfactual default remain available.

`km.branch(path=None)` creates an independent writable what-if at the current state. Facts,
write gates, writer/reassert permissions, agenda callbacks, native action model and failure
memory are copied; writes never change the parent or another branch. Custom action models
are refused before destination writes. Python callbacks retain access to their external state;
branching does not snapshot their closure or external services. There is no merge.

`km.store.branch(storage=None, upto=None)` also supports a historical journal prefix (`upto`
is a record count). A journal note records the parent head and original selected prefix hash.
The optional destination must be empty. Physical destination hashes are newly assigned;
the lineage note identifies the inherited history rather than claiming it was observed again.
Persistent branches containing inherited verified/`of` decision references are refused before
creating the destination, because decision proofs are not exported. Source/write gates remain active.

Reopen a persistent branch with the normal `Knowledge`/`KnowledgeStore` constructor and the
same explicit writers, gates, clock and native vocabulary configuration. Journal replay restores
knowledge; Python policy, agenda callbacks and volatile failure-memory state are not serialized.
The returned live branch preserves those runtime objects independently. A bare reopen is not a
policy/configuration export. Use the low-level store prefix API for historical branches; the facade
branches only its current runtime state.

::: solvi.solutions.knowledge
