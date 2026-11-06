# `solvi.experimental`

The experimental package itself: `STATUS` (for each piece: what it is, since when, what it is missing to graduate, the
script that measures it, the deadline), `DEADLINE`, and the helpers each piece uses to warn on import and to mark the
decisions it takes part in. Importing this package warns about nothing; each piece warns when it is imported. See
[Experimental](../experimental.md).

::: solvi.experimental


### Counterexample refinement (experience)

`RefinementSession` implements bounded paid counterexample-guided iteration and optional observations.
`ObservationOption`, `ObservationChoice` and `select_observation` expose finite committee disagreement selection.
These APIs do not activate registry operations or authorize side effects. See [experience](../experience.md)
for verified-domain scope, source contracts, budget accounting and durable read reconciliation.
