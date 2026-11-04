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

`choose(safe, probes, read_only=..., cost=..., per_cost=True)` returns the existing Probe record. Unsafe and unaffordable points are excluded before prediction callbacks. Selection orders read-only first, expected information (or information/cost) next, lower cost next, and original order last. Default cost is one; finite nonnegative costs are required. `observe(probe, value)` updates beliefs separately when an external measurement arrives. It rejects changed predicted means. Do not pass probes from another planner.

Expected information integrates continuous Normal likelihoods using deterministic 64 midpoint-quantile nodes per component. Probe.info is approximate bits, not exact integration or a certified bound. The loop reselects after each observation, stopping at the predeclared observation/cost budget, no informative affordable safe probe, or the declared posterior threshold. It does not automatically increase budgets or retry failed measurements. Confidence is conditional posterior mass, not a distribution-free guarantee, test alpha, accuracy certificate, or guaranteed identification within 20 observations. Highly extreme unsupported numeric scales are rejected explicitly.

Fourier/dimensional formula discovery, learned noise, correlated measurements, historical formula-discovery accuracy and population power are outside this finite helper. Retain the historical S170 negative experiment; these focused tests establish the numeric likelihood and adaptive plumbing only.
