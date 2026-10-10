# `solvi.core.contract`

The contract of an answer: its type matches what the question asks, it is not the input it was read from, and an
applicability precondition is checked before the answerer is called (`Catalog.contract`, a slow path's `contract=`,
`Specialist.applicable`).

::: solvi.core.contract
