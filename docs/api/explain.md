# `solvi.core.explain`

Explaining a decision: the smallest change of its given inputs that changes the answer (counterfactuals; stable since
1.1, until then `solvi.experimental.counterfactual`). Only the deterministic flow is re-run, model parts held at their
recorded proposals; `text_cost=` and `costs=` set what a change costs.

::: solvi.core.explain

## The old path, until 1.2

`solvi.experimental.counterfactual` re-exports every name above, as the same objects, with an `ExperimentalWarning`.

::: solvi.experimental.counterfactual
    options:
      members: false
