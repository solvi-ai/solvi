# `solvi.experimental.numguard`

Exact values through a side channel (experimental): the model writes `[k]` for an input numeral and `<calc>` /
`<date>` calls with `@k` for their results; `fill(text, source)` writes the exact values in, `check(text, source)`
names the source of every number left in a text, `NumberGuard(on_fail=...)` runs both and refuses, raises or reports,
`NumberGuard.run_with_retry(generate, source)` asks once more with what failed. Typed calls share the `@k`
numbering: `<table>` (cells, column and row aggregates of the source's pipe tables, `table_calc`), `<cmp>`
(comparisons, a yes/no, `cmp_calc`), `<scale>` (scale words, percent, time and speed units, `scale_calc`) and
`<count>` (counts and membership, `count_calc`). `MODES` is the registry of call kinds (slot grammar, result type,
validator, executor); `register_mode` adds one that `fill` runs. `PI` is an opt-in constant. It guarantees that every
number has a source, not that it is the right one ([status](../experimental.md#numguard)).

::: solvi.experimental.numguard
