# Buy checks according to a fitted model

`CheckPolicy` chooses whether to answer, abstain, or purchase another soft check.
`System(..., check_policies={question_name: policy})` runs the selected checks and
records the acquisition decisions. Each `ask` or `aask` request must supply a
finite nonnegative `check_loss`: the utility penalty for an incorrect answer,
with reward one for a correct answer and zero for abstaining.

```python
from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.plan.check_policy import CheckPolicy
from solvi.stat import CorrelationAwareCombiner

catalog = Catalog()

@catalog.check
def verified(verification_ok: bool) -> bool:
    return verification_ok

@catalog.rule("category")
def category(score: float):
    return Decision("A", {"A": score, "B": 1 - score})

# Offline labeled observations: illustrative data, not a quality evaluation.
rows = [{"verified": "agree"}] * 20 + [{"verified": "disagree"}] * 20
model = CorrelationAwareCombiner(["verified"], rho={}).fit(rows, [1] * 20 + [0] * 20)
policy = CheckPolicy(model, costs={"verified": .02}, max_checks=1, max_cost=.02)
system = System(catalog, [Question("category", "Category?", Answer.choice(["A", "B"]),
                                  requires=["verified"])],
                check_policies={"category": policy},
                check_intervals={"category": {"mode": "report", "seed": 7}})
response = system.ask({"score": .7, "verification_ok": True},
                      check_loss=1, early_exit=True)
acquisition = response["category"].extra["check_acquisition"]
```

Configure either `check_models` or `check_policies` for a question. A policy
supplies its own detached fitted model. All its sources must be declared soft
checks in `Question.requires`; their outcomes must be exactly `agree`, `disagree`,
and `fail`. The policy accepts native `NaiveCombiner` or
`CorrelationAwareCombiner` instances, not custom subclasses, and supports at most
six sources. The answer is a scalar choice or yes/no; checks cannot replace the
selected answer. Existing hard-check decisions retain priority.

The runtime first executes primary and mandatory work, then executes only checks
selected by the policy. Previously completed shared dependencies are reused.
An unpurchased check is skipped; an unavailable check is not imputed as a failure
or any other outcome. Optional checks cannot be primary/hard dependencies or
depend on another optional check. The actual phase sequence, available sources,
request loss and policy decisions are recorded in a `check_policy` trace record.

## Utility and budgets

For model score `p`, stopping compares `p - check_loss * (1 - p)` with abstention
utility zero. Buying a check subtracts its declared price and averages future
values over its possible outcomes. The finite Bellman search considers all
remaining purchases within `max_checks` and `max_cost`. Completed checks count
toward these budgets; each purchased price is charged once.

`costs` must specify every source with a finite nonnegative utility price. These
are caller-supplied prices, not inferred tokens, money or latency. Include the
cost of prerequisite work when setting a check's price; dependency reuse does not
automatically discount that price. Mandatory primary work is outside these check
budgets. A budget does not authorize executing unselected optional checks.

Predictive outcome probabilities mix the fitted class tables using the current
point posterior. Correlation tempering and optional `risk_quantile` make this a
conditional VOI surrogate rather than a coherent joint Bayesian model. The search
is exact for the stated finite surrogate; it does not establish correctness,
calibration, interval coverage or a false-accept bound. Correlations, pooled
priors, likelihood tables and prices remain fixed throughout a request.

## Replay and limits

Construction snapshots model parameters and resolves correlation callbacks.
Policy/model fingerprints bind recorded decisions to their source configuration.
Sync/async execution, storage, replay, recorded-answer recalculation and renaming
preserve the acquisition record. Recalculation uses recorded work, not purchases
of new checks. A changed policy or incompatible record is rejected by replay.

Use one question per acquisition request, deterministic declared planning,
`early_exit=True`, and `speculate=False` for async calls. Online learning,
strategist-driven planning, joint answers, `ask_text`, diagnostic `probe`, unrecorded `decide` and
caching are outside this bridge. Existing confidence calibration, guards and
`shift_watch` cannot be reused for a policy question: they require calibration of
the acquisition process itself and are explicitly refused.
Static `facts_for`, `fit`, `learn_rule`, `learn_order` and in-place `teach` flows
are refused before calls on an acquisition-configured System; fit the primary
model in a separate System first. Static `calibrate` and `guarantee` are also
refused for the policy question before consuming examples or changing state.
Recorded-answer recalculation requires an acquisition record; a static trace is
not reinterpreted as a budgeted policy execution.
Acquisition responses require full storage records (`sample:1` is also full).
Compact and partially sampled storage are refused before paid work; `store=False`
may return a full in-memory response without writing. A legacy compact acquisition
record cannot be re-derived, and fails before any check is called.

Optional [check intervals](check_models.md#conditional-uncertainty) report a
model-conditional interval. `mode="lower"` uses its lower bound before
minimum-confidence checks; an acquisition decision cannot overwrite that bound.
A policy's `risk_quantile` controls its own search score separately and does not
turn either interval into a coverage guarantee.
