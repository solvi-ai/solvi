# `solvi.experimental.numguard`

Exact values through a side channel (experimental): the model writes `[k]` for an input numeral and `<calc>` /
`<date>` calls with `@k` for their results; `fill(text, source)` writes the exact values in, `check(text, source)`
names the source of every number left in a text, `NumberGuard(on_fail=...)` runs both and refuses, raises or reports,
`NumberGuard.run_with_retry(generate, source)` asks once more with what failed. It guarantees that every number has a
source, not that it is the right one ([status](../experimental.md#numguard)).

::: solvi.experimental.numguard
