# `solvi.core.catalog`

The catalog: `Catalog`, `Question`, `Answer` and the decorators that register parts (re-exported by `solvi.core` and `solvi`).

A model's `Decision` keeps its `value` when it escalates (what it would have answered) and the reason in `escalate`:
`Decision.alone` is True only for a value that was not escalated — count only those as the model's answers. In a
System, read an answer as `res["question"].answer`; an escalated one abstains there (`res["question"].status`).

::: solvi.core.catalog
