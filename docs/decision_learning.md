# Learning for final decision value

`solvi.experimental.experience.decision_learning` provides a finite CPU-only integer linear reward ranker, a separate validation/activation gate, and one-step logged-policy evaluation. It does not train or invoke a System One checkpoint. No torch, optimizer service or external API is required.

A `DecisionContext` contains named actions with immutable integer features, explicit costs and host-owned hard feasibility. `DecisionObjective` defines reward-weight × outcome minus cost-weight × cost. `OutcomeFeedback` supplies **all** action outcomes from a declared source, event and split. These are host evidence assertions, not model-generated truth. The host remains responsible for truthful sources, current feasibility, representative data and separating roles. There is no long-horizon RL or causal discovery.

`fit_decision_policy` performs deterministic bounded integer coordinate descent. `loss='decision'` minimizes empirical downstream regret; `loss='prediction'` minimizes squared reward error using the same model class. This is a simple decision-focused backend, not SPO+. Both candidate evaluation and training are paid through the caller's `WorkBudget`; local minima, ties and model misspecification remain possible. A context with no permitted action returns `None`. A high reward never overrides feasibility.

## First policy: public forms

Import these names from `solvi.experimental.experience.decision_learning` and
`WorkBudget` from `solvi.experimental.experience`:

```python
DecisionAction(name, features, cost=0, allowed=True)
DecisionContext(context_id, revision, actions)
OutcomeFeedback(event_id, source, context, outcomes, split='train')
DecisionObjective(reward_weight=1, cost_weight=1)
DecisionPolicy(weights, objective=DecisionObjective())
DecisionLearner(budget)
```

Use nonempty string IDs, revisions and sources; `source` identifies the host's
evidence origin, not an authenticated identity. Actions and features are tuples;
features, outcomes, weights and costs use strict integers rather than booleans.
Every context uses the same feature dimension as the policy. `outcomes` is a
tuple such as `(('standard', 4), ('express', 7))`, covering **all** actions,
including those marked `allowed=False`. `allowed` is a host-owned hard boolean
constraint. An action's explicit `cost` is subtracted by the objective, so do
not subtract it again from the outcome unless that is your intended objective.

`learner.train(feedback, objective=DecisionObjective(), loss='decision',
epochs=4, steps=(2, 1))` returns a registered candidate identifier. Training
uses a finite integer search that may retain ties or
reach a local minimum. `loss='prediction'` is the alternative squared-error
training objective. `fit_decision_policy(feedback, budget, ...)` uses those same
keyword options and returns a policy directly, without registration/admission.

`learner.admit(candidate, validation, baseline)` returns a dictionary containing
`accepted` (boolean), `receipt` (activation identifier when accepted), `candidate`, `regret`,
`baseline_regret`, the baseline policy and retained validation evidence. Lower
regret is better; equality passes this finite gate. A receipt does not itself
activate anything:

```python
# training and validation are independently obtained full-feedback rows.
candidate = learner.train(training)
admission = learner.admit(candidate, validation, DecisionPolicy((0, 0)))
if admission['accepted']:
    learner.activate(admission['receipt'])
    action_name = learner.choose(next_context)
else:
    # Retain the baseline / seek better evidence; do not activate this receipt.
    print('Rejected:', admission['regret'], admission['baseline_regret'])
```

The two-weight baseline above requires two-feature actions. `choose(context)`
returns an action name or `None` when no action is permitted; it requires an
explicitly activated live policy. `activate(receipt)` rejects failed admissions.
Budget exhaustion raises `BudgetStop`; retain the paid ledger instead of
silently retrying with a fresh allowance. For a complete runnable example use
`examples/39_decision_learning.py`; for host setup and all four APIs together,
see [the decision workflow guide](decision_workflows.md).

`DecisionLearner.train` registers a proposal. `admit(candidate, validation, baseline)` requires a separate `validation` split with disjoint event/context IDs, evaluates exact downstream regret and accepts only empirical non-inferiority to the explicit baseline with the same objective. It does **not** activate. `activate(receipt)` is an explicit host decision; `revoke(candidate_id)` prevents reuse. Admission on finite data is not a population guarantee or permission for external effects. Context IDs alone cannot detect a host relabelling duplicate data.

`snapshot` creates a checkpoint bound to a host-owned `DecisionAuthority`. `restore(snapshot, budget, authority=...)` validates exact current checkpoint identity and independently recomputes admission scores, without rereading an external oracle. It transfers sole ownership: the previous live learner becomes stale. Old checkpoints, forged hashes and changed revocations are rejected. Use the restored original ledger: refunding work or changing the budget limit is rejected. Host-side `export_trusted_state` / `from_trusted_state` support restart when the anchor and ledger are persisted in a **trusted channel separate from the untrusted payload**. An untrusted caller supplying both payload and its supposed authority is not authenticated. Coordinate persistence atomically at the host, e.g. in a KnowledgeStore transaction; these objects do not provide a distributed lock.

`LoggedFeedback` is a different schema: it contains only the selected action's outcome and the complete known logging distribution. `evaluate_logged_policy` computes exact rational IPS, SNIPS, DR (with a fixed independently fitted reward model), effective sample size and maximum weight. Targets are supplied for exact context fingerprints. It refuses unsupported/infeasible target mass, unknown propensities and insufficient ESS. It never invents probabilities or clips weights silently. There are no confidence intervals or population-identification guarantees: validity still requires suitable logging, support, fixed target/model and correct outcome evidence. Matching the full oracle on a balanced fixture checks formulas, not causal identification in field data.

All loops are structurally bounded: 32 features, 128 actions, 4096 feedback rows, 32 epochs, 16 coordinate scales, 128 registered candidates/admissions. Budget exhaustion raises `BudgetStop` before the unpaid operation; it never returns a partially fitted accepted candidate. Work units include feature arithmetic and checks, but are declared units rather than CPU instructions, seconds or money. Host parsing and generation must be priced separately in comparisons.

Run `examples/39_decision_learning.py` for training, explicit activation, paid recovery and IPS/SNIPS/DR; `examples/40_decision_workflow.py` composes all four decision directions. See the API reference and the checked research protocol for empirical results and limits.

Methods: [Smart Predict, then Optimize](https://arxiv.org/abs/1710.08005), [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601). These methods predate this integration; no mathematical novelty is claimed.
