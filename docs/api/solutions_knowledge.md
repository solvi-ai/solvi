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

::: solvi.solutions.knowledge
