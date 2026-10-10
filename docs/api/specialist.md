# `solvi.core.specialist`

The specialist contract (stable since 1.1; until then `solvi.experimental.specialist`): a model proposes a typed spec,
code checks it against the source, code renders it; every run replays to identical bytes from its own record or from
its record of kind "specialist" in a TraceStorage (`replay_runs`). `align` matches two readers' readings by the overlap
of their spans (time, boxes, character offsets).

::: solvi.core.specialist

## The old path, until 1.2

`solvi.experimental.specialist` re-exports every name above, as the same objects, with an `ExperimentalWarning`.

::: solvi.experimental.specialist
    options:
      members: false
