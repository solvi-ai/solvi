# Candidate answer policy

`solvi.core.plan.answer_policy` ports the archived `solvi_ext/answer_policy.py`:
SHA256 `6bd3e7e39ce82d76a5d0dc34d36f20a575e3040a581d501d5e0377a5ee16f30b`.
Primary source: `solvi_handoff_2026-10-30/solvi_ext/answer_policy.py` and
`solvi_ext/tests/test_answer_policy.py`; research context S103 and S117 in
`bag_reports/solvi_1.1.1.md`. The native tests pin a small deterministic record
set and numerical outputs computed using that exact source. No external archive
is needed to run tests. They do not reproduce historical model quality.

```python
from solvi.core.plan.answer_policy import AnswerPolicy, CheckModel
# Calibration records have answer, correct, gold, and checks:
# {name: {status: 'agree'|'disagree'|'fail', value: number|None, correct: bool}}
model = CheckModel().fit(calibration_records)
policy = AnswerPolicy(model, checks=['P', 'D'], cost=0.05)
action = policy.decide(q0=0.8, state={}, lam=30)
# Replay buys only stored checks, and returns the actual selected number.
result = policy.replay(q0=0.8, record=stored_record, lam=30)
```

`q0` is the calibrated probability the original answer is correct. State maps
configured check names to `agree`, `fail`, or `('val', candidate_id)`; equal ids
represent the same alternative number. `posterior` returns original, candidate,
and unseen-truth probabilities. `CheckModel.fit` preserves pooled conditional
counts and collision estimation from the prototype; unseen checks use the pooled
model. Conditional independence with optional weight tempering is an assumption,
not a guarantee or proof that agreeing checks are independent.

The original dynamic program compares answer utility `p - lam*(1-p)`, abstention
utility zero, and expected future value minus check price. `Action.kind` is
`answer`, `answer_check`, `check`, or `abstain`; `p_answer` belongs to the chosen
candidate. `allow_replace=False` preserves the original answer-only ablation.
`Replay` reports actual answer, correctness, primary-plus-check calls and used
checks. Numeric answers may be negative; `close` preserves the prototype's
relative comparison tolerance. `gold` is for offline scoring, not policy input.

`costs` overrides utility price per check; `units` supplies spending units;
`max_checks` and `max_units` restrict purchases. `BudgetPacer.configure` sets
prices and hard per-item allowance; `charge` accounts a Replay or spent units.
An overspend is rejected before accounting changes. These units are supplied by
the caller and are not inferred token prices or wall time.

Probability inputs must be finite in [0,1]; costs, units, stakes and pacer
parameters finite and nonnegative. Pooling counts and collision priors are
positive. Malformed states and records raise ValueError before weight callbacks.
Callbacks must return nonnegative finite weights for observed checks. Callbacks
and mutable configuration are caller-owned; thread safety is not provided.
Weights must be deterministic for the same observed subset during a decision and
must not mutate the model or configuration while planning.
Zero weight excludes that check from posterior candidates, collision terms and answer-source attribution.
An acquired check still counts toward purchase limits and spending, even when its evidence weight is zero.
Unlike the prototype's sticky cross-call memo, each decision uses a fresh memoized
DP tree so changed model/configuration/callback state cannot reuse stale values.
This preserves valid-method arithmetic while trading cross-call reuse for safety.
Two inherited candidate-name collisions are corrected: memo keys retain outcome
type, and simulated unseen candidates receive an ID distinct from observed IDs.
Renaming an observed candidate therefore leaves the selected action and utility
unchanged, including names such as `agree`, `fail` or `new1:P`.

This is a bounded functional port, not evidence of improved quality. Scientific
S103 original acceptance, frozen AIME calibration/cross-fitting, comparisons and
cost-quality claims remain open. It does not implement S128 agent refusal repair.

::: solvi.core.plan.answer_policy
