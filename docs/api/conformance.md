# `solvi.testing.conformance`

Conformance checks for the extension points of `solvi.core` (see [Building blocks](../building_blocks.md)): each runs
your implementation through the parts of solvi that rely on it and fails with the guarantee that does not hold.
`check_specialist` checks a specialist of `solvi.experimental.specialist` (identical bytes, replay, JSON records,
blocked invalid proposals).

::: solvi.testing.conformance
