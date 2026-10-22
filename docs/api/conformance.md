# `solvi.testing.conformance`

Conformance checks for the extension points of `solvi.core` (see [Building blocks](../building_blocks.md)): each runs
your implementation through the parts of solvi that rely on it and fails with the guarantee that does not hold.
`check_specialist` checks a specialist of `solvi.core.specialist` (identical bytes, replay, JSON records,
blocked invalid proposals, the runs stored in a TraceStorage and replayed from it, an edited stored run caught).
`check_human_channel` checks a channel to a person of `solvi.experimental.process` (a ticket per question, none lost,
one answer per ticket, cancel, pending / resume through a process loaded afresh).
`check_pipeline` runs attack families on a change pipeline of `solvi.experimental.changes` and reports which stage
caught what.

::: solvi.testing.conformance
