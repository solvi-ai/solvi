# From user examples to a checked decision

From a checkout of the [Solvi repository](https://github.com/solvi-ai/solvi/tree/main/examples),
run `python examples/40_decision_workflow.py` for a complete finite capacity
planning workflow. It imports public APIs, shares one paid `WorkBudget` and
performs no external action. The host provides a simulator for outcome labels;
replacing it with real business data is a separate integration task. The Python
examples are repository files; the installed wheel contains the library, not the
examples directory.

## Host setup versus user interaction

The **host developer** defines a typed `Grammar`, a finite `Program` hypothesis
library and query domain. The end user supplies labels through
`teacher.teach(inputs, value, exception=True)` or answers the distinguishing
`teacher.question()` using `teacher.answer(question, value)`. An exception is a
labelled counterexample; it does not create a program outside the host library.
Example37 demonstrates a rule with an exception.

The host then calls `admit(..., domain, oracle, source_id=...,
source_revision=...)` on separate finite points. Admission needs a unique
proposal and returns a concept reference or `None` on disagreement. A taught
point is not independently verified just because the user labelled it.
Always inspect the application status before making it a hard constraint:

```python
checked = teacher.apply(concept, {'capacity': 6})
if checked.status != 'verified':
    raise RuntimeError(f'Independent capacity check required: {checked.reason}')
allowed = checked.value is True  # verified False means a checked prohibition
```

Do not silently present `unknown` as a verified rejection. Request an independent
check of the new point instead. Source strings and revisions must be nonempty;
they record provenance and host semantics, not external authentication.

## Preference, feasible search and learning

1. Define finite `Alternative` attribute vectors and possible utility weights.
   Costs must be encoded explicitly, e.g. `(-cost, capacity, safety)`, because
   larger utility wins. `next_question(cost=...)` returns a useful pair or `None`.
   Show its IDs to the user, then record their actual answer with a unique answer
   ID and a source: `preferences.answer('manager-1', preferred, other,
   source='planning-manager')`. A tie uses `relation='tie'`.
   Question `cost` is a decision penalty, separate from charged computation.
2. Obtain `preferences.choose()` and independently call
   `check_regret_certificate(preferences, certificate, budget=budget)`. Handle
   `conflict` and `no_feasible_alternative` explicitly before using the choice.
   A preference certificate checks only that session's requirements. The taught
   capacity rule must also reach the final feasibility checks; a preferred
   alternative can otherwise be forbidden by the final host task.
3. Build a `FiniteProblem` with the checked constraints and chosen objective.
   `CooperativeSearch.run([ConstraintSolver(problem), ObjectiveSolver(problem)])`
   exchanges independently checked claims. A feasible incumbent still has
   `unknown` optimality. Call `certify()` and require `status == 'optimal'` before
   reporting an optimum. `infeasible` and `unknown` need separate handling.
4. Observe external outcomes. `OutcomeFeedback` requires outcomes for every
   action, not just the selected action. Train with separate `train` rows,
   admit against independent `validation` rows and an explicit baseline, inspect
   `accepted`, then explicitly activate. Equal regret passes; a passed gate is
   not a promise of improvement or population generalization. The next policy
   proposal remains subject to host `allowed` flags and creates no external
   effects. Selected-action logs use the separate `LoggedFeedback` evaluation
   interface; do not invent unobserved counterfactual outcomes.

Example40 prints the preferred alternative, the checked capacity flags, the
cooperative optimum, candidate/baseline validation regret, the next proposal and
total paid work. Its baseline is intentionally simple. It returns early without
activation if validation rejects the candidate. For more detail see
[preferences](preferences.md), [teaching](teaching.md),
[cooperation](cooperation.md), and [policy learning](decision_learning.md).

## Budgets and saved sessions

Use one shared ledger throughout; example40 declares `WorkBudget(200000)` as a
small demonstration allowance, not a sizing guarantee for other domains. Host
data preparation and external oracle/simulator work need their own declared
charges. `BudgetStop` propagates from most calls; cooperative `run` reports
`unknown` on exhaustion, and later residual reads can still raise. Do not turn
an exhausted partial result into a certificate or reset the ledger to retry.
For recovery, retain each API's trusted authority separately from the session
payload, together with the current ledger. The individual guides describe their
different ownership and restart contracts; the first workflow needs no restore.
