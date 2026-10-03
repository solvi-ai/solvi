# `solvi.core.costs`

Run-time costs: `CostBook` (`system.cost_book`), the dollars of recorded model calls (`price_of`, `cost_of`), and
`Budget` / `Cost` / `BudgetStop` — the limits and costs `solvi.core.dispatch`, `solvi.core.slow.generate` and `solvi.core.slow.refine` share
(defined in `solvi.core.dispatch` until 1.0, which re-exports them); `MeasuredCosts`
(`System(cost_policy="measured")`) was removed in 1.0. Part of
`solvi.learned` up to 0.7; the learned order and producer policy are in `solvi.core.plan.strategist`.

::: solvi.core.costs
