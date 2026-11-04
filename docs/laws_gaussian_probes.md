# Numeric Gaussian experiments

`GaussianProbes` is an explicit finite-family numeric alternative to `split(score="eig", noise=epsilon)`, whose noise is categorical label flipping. Existing split/discriminate defaults are unchanged.

```python
from solvi.experimental.laws import GaussianProbes
planner = GaussianProbes([lambda x, action: x, lambda x, action: 2*x], sigma=1.0)
history = planner.run(
    lambda x, action: measure(x),
    safe=lambda x, action: action == "read",
    probes=[(1.0, "read"), (2.0, "read")],
    max_observations=20, max_cost=10, confidence=.95,
    cost=lambda x, action: 1,
)
```

The owner supplies numeric hypothesis means and a known positive measurement standard deviation in the same units (scalar, or `sigma(state, action)`). Independent measurements conditional on each hypothesis follow the specified Normal model. Repeated measurements at the same point remain eligible. Each actual observation updates normalized log posterior probabilities; finite family/prior/noise assumptions are explicit. Predictions and noise callbacks must remain frozen throughout the experiment. Observations do not fit hypothesis parameters or noise from the same data.

`choose(safe, probes, read_only=..., cost=..., per_cost=True)` returns the existing Probe record. Unsafe and unaffordable points are excluded before prediction callbacks. Selection orders read-only first, expected information (or information/cost) next, lower cost next, and original order last. Default cost is one; finite nonnegative costs are required. `observe(probe, value)` updates beliefs separately when an external measurement arrives. It rejects changed predicted means or standard deviations, and rejects probes not issued by this planner. These checks bind the declared numeric model; they do not detect hidden changes behind a callback that returns the same values.

Expected information integrates continuous Normal likelihoods using deterministic 64 midpoint-quantile nodes per component. Probe.info is approximate bits, not exact integration or a certified bound. The loop reselects after each observation, stopping at the predeclared observation/cost budget, no informative affordable safe probe, or the declared posterior threshold. It does not automatically increase budgets or retry failed measurements. Confidence is conditional posterior mass, not a distribution-free guarantee, test alpha, accuracy certificate, or guaranteed identification within 20 observations. Highly extreme unsupported numeric scales are rejected explicitly.

Fourier/dimensional formula discovery, learned noise, correlated measurements, historical formula-discovery accuracy and population power are outside this finite helper. Retain the historical S170 negative experiment; these focused tests establish the numeric likelihood and adaptive plumbing only.


Floating-point support is deliberately bounded: each standardized mean separation
`(mu_i-reference)/sigma` and centered observation `(y-reference)/sigma` must be
finite with absolute value at most 10,000. The same bound applies to the
quadrature's standardized component coordinates. Unsupported scales, including
astronomical observations such as ±1e100 at unit sigma, raise ValueError before
posterior/counter mutation. No saturated approximate update is silently accepted.
This domain concerns centered quantities, so representable translations of all
means and measurements (for example by 1e12 at unit sigma) remain supported.
Within this domain arithmetic and information integration are floating-point
approximations, not arbitrary-precision guarantees. An external measurement
callback may already have run when its returned value is rejected; its real-world
side effects or incurred charges cannot be rolled back by this helper.
