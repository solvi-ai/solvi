# Check acquisition policy

`CheckPolicy` is the bounded finite-search policy for the selected answer's
correctness score. See [runtime acquisition](../check_acquisition.md) for a full
example and [fitted check models](../check_models.md) for conditional uncertainty.

```python
from solvi.core.plan.check_policy import CheckPolicy

policy = CheckPolicy(fitted_combiner, costs={"verified": .02},
                     max_checks=1, max_cost=.02,
                     risk_quantile=None, n=1000, seed=0)
action = policy.decide(prior=.7, outcomes={}, loss=1, available=["verified"])
```

`costs` must contain exactly all sources. Prices and budgets are finite and
nonnegative; `max_checks` is a nonnegative integer or `None`. `risk_quantile` is
`None` or a value in `[0, 1]`; `n` is a positive integer and `seed` a nonnegative
integer. The model must be a fitted native combiner with exactly the three
boolean-check outcomes `agree/disagree/fail`, with at most six sources.

`decide` returns the existing `Action` with kind `answer`, `abstain` or `check`,
plus utility `value`, current score `p_answer`, `why`, and the next `check` when
applicable. `available=None` allows every source; otherwise supply a unique subset
of sources. Availability only restricts future purchases. `outcomes` contains
actual observed outcomes and cannot use missing values as evidence. Completed
checks count toward budgets, but their costs are sunk in the returned future
utility. Ties retain stopping, then declared source order.

`risk_quantile` uses a seeded quantile of native `posterior_samples` for stopping;
future outcomes still use the point-score mixture of fitted class tables.
This is a conditional risk-averse surrogate, without Bayesian coherence or
coverage claims for correlated checks. No online fitting occurs.

`freeze()` creates another detached snapshot. `fingerprint()` hashes the fitted
model and policy configuration; per-request availability is not configuration.
`sources` returns the source tuple. Public configuration is read-only by
convention; change it by constructing another policy. Each decision builds a
fresh search tree, so there is no stale cross-request planning cache.

::: solvi.core.plan.check_policy
