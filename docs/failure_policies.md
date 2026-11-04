# Failure policies for fresh environments

`solvi.experimental.changes.FreshEnvironments` compares a candidate with the current agent on paired selection
worlds, then on separate confirmation worlds. Its default remains score-based: `failure_policy=None`, with a default
score of one for an episode without a failure and zero for an episode with a failure. A custom score can reward
other outcomes. Policy exceptions still independently refuse the candidate.

An optional failure policy adds a veto that a higher score cannot override. Choose the policy explicitly:

| `failure_policy` | Refusal condition on either set |
| --- | --- |
| `None` (default) | No additional environment-failure veto |
| `"zero_failures"` | At least one candidate episode failed |
| `"no_new_failures"` | A baseline episode was safe but the candidate episode with the same seed failed |
| `"max_failure_rate"` | Candidate failed episodes / episodes exceeds `max_failure_rate` |

For example, given an environment factory and an agent factory that accepts `(change, context)`:

```python
from solvi.experimental.changes import FreshEnvironments

stage = FreshEnvironments(
    env_factory, agent_factory,
    select=selection_seeds,
    confirm=confirmation_seeds,
    score=lambda row: row["progress"],
    failure_policy="no_new_failures",
)

# Alternatively, require no candidate failures at all:
strict_stage = FreshEnvironments(
    env_factory, agent_factory,
    select=selection_seeds, confirm=confirmation_seeds,
    failure_policy="zero_failures",
)

# Or choose an explicit observed-sample cap:
capped_stage = FreshEnvironments(
    env_factory, agent_factory,
    select=selection_seeds, confirm=confirmation_seeds,
    failure_policy="max_failure_rate", max_failure_rate=0.05,
)
```

Provide nonempty, disjoint fresh selection and confirmation seed sets for each candidate. The caller remains
responsible for their freshness. The failure policy does not change bootstrap seeds, confidence levels, score,
progress or replay requirements. Confirmation runs only after selection and replay pass.

`max_failure_rate` must be an explicit finite number in `[0, 1]` when that policy is selected; booleans are rejected.
Providing a cap with another policy is an error. Equality to the cap passes the failure guard; other stage bars
still apply. This cap concerns the observed episode fraction on each set separately. It provides no bound or
confidence guarantee for the population failure rate.

Failures follow the existing episode convention: the final environment outcome's `info["failed"]` is converted
with `bool`; missing metadata means false. A policy exception is recorded separately as an error and retains its
existing refusal priority. These guards rely on the environment reporting failures accurately.

With a policy enabled, each measured set includes `failure_policy` evidence: policy name, candidate failure count,
new paired failure count, observed failure rate, explicit cap, refusal flag and reason. Its `paired` list records
each seed with baseline and candidate failure flags, preserving order and repeated seeds. Existing aggregate
`failed` counts remain available. Evidence describes the worlds actually run; it does not create a cross-candidate
world registry or a multiple-testing correction.
