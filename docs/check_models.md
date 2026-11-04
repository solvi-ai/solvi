# Fitted check models in System

`System(..., check_models={"question": fitted_combiner})` applies an explicitly fitted
`solvi.stat.CorrelationAwareCombiner` or `NaiveCombiner` to the confidence of the
selected answer. The model learns conditional outcome tables for correct and
incorrect answers; a single check reliability number cannot replace these tables.
Prefer a correlation-aware model when checks share errors. Selecting a naive model
is an explicit assumption of conditional independence; System never selects it automatically.

```python
from solvi import Answer, Catalog, Decision, Question, System
from solvi.stat import CorrelationAwareCombiner

catalog = Catalog()

@catalog.check
def verified(verification_ok: bool) -> bool:
    return verification_ok

@catalog.rule("category")
def category(score: float):
    return Decision("A", {"A": score, "B": 1 - score})

# Fit on separate labeled calibration observations, not on the request being answered.
model = CorrelationAwareCombiner(["verified"], rho={}).fit(
    [{"verified": "agree"}, {"verified": "disagree"}], [True, False]
)
system = System(catalog, [Question("category", "Category?", Answer.choice(["A", "B"]),
                                  requires=["verified"])],
                check_models={"category": model})
response = system.ask({"score": .7, "verification_ok": True})
evidence = response["category"].extra["check_evidence"]
```

The tiny table illustrates the API; it is insufficient to establish calibration.
For multiple checks, supply an independently estimated error-correlation mapping to
the correlation-aware combiner. Neither model certifies correctness, independence,
false-alarm rate or deployment quality.

Each modeled check must be explicitly listed in the question's `requires`. System
does not schedule extra unrequested checks. Executed `True` maps to `agree`, `False`
to `disagree`, and an execution error to `fail` when that outcome exists in the model.
Skipped checks supply no evidence. Inputs cannot impersonate registered checks.
Hard-check failure, forced answers and abstention retain their existing priority.
Checks already read by the primary rule/head, and checks declared to share its call
or a primary dependency's call, are refused to prevent repeated use of that evidence.
This validation cannot discover undeclared statistical dependence.

The posterior describes correctness of the selected answer. `probs` retains the
primary model's raw distribution; `extra["check_evidence"]` records prior, posterior,
observed outcomes, model fingerprint and domain. The posterior feeds existing
confidence calibration, guards and minimum-confidence abstention. It does not
select another answer or repair the primary distribution.

Construction freezes a detached model snapshot and resolves correlation callbacks.
Refitting or changing the original model does not change the System. Public
`combiner.freeze()` returns a snapshot that cannot be refitted; direct mutation of
that snapshot's configuration is unsupported. `combiner.fingerprint()` requires a
fitted model and hashes its fitted state. The System's question identity includes
the frozen model, so replay detects a different fitted model. Sync/async asks,
recorded-answer recalculation, storage and boundary/question renaming use the same
snapshot.

Current scope is structured `ask`/`aask` with scalar choice or yes/no answers.
Joint constraints, answer-fact composition, `ask_text`, unrecorded `decide` and
decision caching are refused. Domain selection is not inferred: this bridge uses
global correlation (`domain=None`); a model with domain-specific matrices retains
the combiner's warning about global fallback. No online fitting or independent calibration guarantee is added by this API.
For optional purchases instead of executing all modeled requirements, see
[check acquisition](check_acquisition.md).


## Conditional uncertainty

Both native combiners provide `posterior_samples(prior, outcomes, n=1000, seed=0)`
and `posterior_interval(prior, outcomes, level=.95, n=1000, seed=0)`. Keyword-only
`domain` selects an explicitly configured correlation domain when supported.
Sampling is seeded before draws and can be replayed with the same configuration.
The interval is an equal-tail Monte Carlo summary and includes sampling error.

Likelihood draws follow independent source/class Dirichlet distributions with
fitted counts plus pooled pseudo-counts. This is conditional empirical Bayes:
the estimated pooled prior, supplied answer prior and correlation weights are
fixed. Their uncertainty is excluded. For a correlation-aware combiner the draws
describe a weighted composite score, not a joint generative posterior. There is
no frequentist coverage, calibration or false-accept guarantee. Sparse or empty
source/class calibration produces a warning instead of disabling the check.

Enable reporting in System with:

```python
system = System(catalog, questions, check_models={"category": model},
                check_intervals={"category": {"level": .95, "n": 1000,
                                               "seed": 7, "mode": "report"}})
```

The same option works with `check_policies`. Settings default to `.95`, `1000`,
`0` and `"report"`; unknown keys and questions without a configured model are
rejected. Construction copies settings and includes them in question identity.
`extra["check_evidence"]["uncertainty"]` records settings, lower/upper bounds,
the model-conditional interpretation and fixed assumptions.

`mode="report"` retains point confidence. `mode="lower"` passes the lower bound
through subsequent confidence processing before minimum-confidence abstention;
a policy action cannot replace it with a higher point score. For ordinary fitted
models this processing includes existing calibration and guards. Policy questions
refuse existing calibration, guards and shift monitoring until acquisition-specific
calibration is available. Forced answers, existing abstentions and requests with
no executed modeled outcomes receive no interval evidence. Raw answer `probs`
remain the primary distribution in either mode.
