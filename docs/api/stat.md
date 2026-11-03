# `solvi.stat`

Sequential evidence and multiple testing, without model or scientific-computing dependencies.
Existing experimental imports of the shared processes remain available and refer to the same classes.

```python
from solvi.stat import MeanBound, ELOND, e_bh

test = MeanBound(0.5)  # H0: conditional mean of each bounded outcome is at most 0.5
for outcome in [0.8, 0.7, 0.9]:
    test.update(outcome)
saved = test.dump()
resumed = MeanBound(0.5).restore(saved)

ledger = ELOND(q=0.1, schedule="uniform", horizon=50)
allocation = ledger.reserve()  # before inspecting this hypothesis's outcomes
rejected = ledger.finish(allocation.index, 2.0)  # a valid e-value, not an arbitrary score
family_rejections = e_bh([0.0, 25.0, 30.0], q=0.1)
```

`MeanBound.update` returns **log evidence**; `ELOND.finish` and `e_bh` take e-values on the ordinary
nonnegative scale. A marginal average-risk bound is insufficient for an anytime process: the stated null
must hold conditionally on the past. Selecting a promising score and calling it an e-value does not establish
that null validity. Data-only checkpoints preserve arithmetic; they do not authenticate the data's origin.

`Mixture` and `MirrorBound` certify low conditional mean loss, in the opposite direction to `MeanBound`.
`Certificate` configures `EProcess`, including restart budgets and a separate change detector. A detector
alarm has the bound stated in `Certificate`; it is not itself a level-delta certification test.

`DeltaBudget(schedule="horizon", horizon=T)` allocates `delta/T` at each of exactly `T` steps, then refuses
more. The default Basel schedule remains `6*delta/(pi*pi*v*v)` indefinitely. Reserve a budget before
observing the corresponding test; there is no implicit refund or reset.

`ELOND` supports `schedule="uniform", horizon=T` and default `schedule="basel"`. Its minimal sequential
interface requires alternating `reserve` and `finish`: only one reservation may be pending. Resolve an
abandoned test with e-value zero to consume its index conservatively. Do not create a new ledger to reuse
the same epoch's budget. Its FDR statement applies to fixed test prefixes under valid null e-values;
arbitrary stopping of an FDR report needs separate justification. `e_bh` is a fixed-family procedure.
Both allow positive infinity, reject negative values and NaN, and leave online state intact on invalid input.

References: [Wang and Ramdas, e-BH](https://sas.uwaterloo.ca/~wang/papers/2021Wang-Ramdas-JRSSB.pdf),
[Xu and Ramdas, e-LOND](https://proceedings.mlr.press/v238/xu24a/xu24a.pdf).

`agreement_diagnostics(rows, checkers=[...])` describes fixed, caller-normalized answers. Each row supplies `gold`,
`primary` and a `checkers` mapping. Answer tokens are exact strings; `None` and missing checker entries mean missing
data. The report includes agreement frequencies conditional on primary correctness, conditional pair tables/phi,
and the fraction of matching answers when both predictors were wrong. Denominators and missing counts are explicit;
undefined rates and correlations are `None`. This descriptive report does not fit weights, establish independence
or issue a certificate.

::: solvi.stat


## Pooled check models and provenance

`solvi.stat` also exports `Check`, `CheckRegistry`, `error_correlation`,
`NaiveCombiner`, and `CorrelationAwareCombiner`. These descriptive models are
separate from anytime evidence and certificates; no effective count is an
anytime guarantee or proof of independence. No names are added to top-level `solvi`.

`Check(name, family, method, shares_call_with=None, cost=1, note="")` is immutable.
`CheckRegistry(primary, checks=(), rho_call=.9, rho_method=.6, rho_family=.35,
rho_other=.05)` follows shared-call ancestry, warns about primary-call reuse and
same-family checks, and exposes `independent()`, `prior_rho(a,b)`, `warnings()`
and descriptive `n_effective(names, rho=None)`. Here "independent" only excludes
shared primary-call ancestry; it does not establish statistical independence.

`error_correlation(rows, sources, prior=None, strength=5)` takes mappings of source
to bool / integer 0 or 1 (correctness), or None (unobserved). Pairwise error phi
is shrunk toward `prior(a,b)` by `strength / (strength + n_inf)`, where `n_inf`
counts rows with at least one error. Degenerate phi is reported as NaN and the
prior is used; all supplied priors and parameters must be finite. Raw undefined
conditional rates also remain NaN, rather than invented estimates.

`NaiveCombiner(sources, outcomes=("agree","disagree","fail"), pool=2, alpha=.5)`
and `CorrelationAwareCombiner(sources, rho, ...)` fit `fit(obs,y)`. Both iterables
are consumed once and must have equal lengths. Targets are strict bool or integer
0/1. Names and outcome labels are distinct nonempty strings; rows may omit a
source or use None for unobserved, but unknown names and unsupported outcomes
are rejected. Pool and alpha are strictly positive. Empty training sets retain
proper smoothed priors. Prediction uses `log_lr(obs)` / `posterior(prior,obs)`;
prior must be in [0,1] and is clipped to [1e-6,1-1e-6] by the original method.

Per-class Dirichlet probabilities retain the original pooled pseudo-count
formula. Correlation rho is a pair mapping (unspecified pairs mean zero) or a
callable returning a symmetric finite correlation with unit diagonal, entries
in [-1,1]. Asymmetric input is rejected. The original identity blending lifts
the minimum eigenvalue to .05; observed-subset weights are
`clip(solve(R, ones), 0, 1)`. Weights are recomputed per call so mutable rho and
caller mutation of returned dictionaries cannot produce stale cached weights.
NumPy is imported only inside numerical helpers.

For candidate-aware answer decisions, use
`AnswerPolicy(check_model, weights=combiner.weights)` from
`solvi.core.plan.answer_policy`: the callback tempers evidence while that policy
retains actual candidate identities. This does not fit a domain-specific
correlation model or establish quality gains. The checks model is pooled across
the supplied records; label uncertainty is not inferred automatically.

Implementation provenance: archived `solvi_ext/checks.py` from handoff
2026-10-30; exact SHA256 is recorded in `_checks.py`. Fixed four-row tests pin
original likelihood, log LR and posterior values. No historical evaluation was
rerun. `JointCombiner` is intentionally outside this surface.


Smoothing is evaluated in log space, including for every finite strictly positive
`alpha` and `pool`; likelihood ratios subtract log probabilities. Public `lik`
is a floating-point introspection snapshot and extremely small probabilities may
underflow to zero there. Prediction uses internal log likelihoods, so such zeros
do not invalidate a fitted model. Editing `lik` does not edit the fitted model;
call `fit` to update it. Public smoothing configuration is revalidated before fit.
