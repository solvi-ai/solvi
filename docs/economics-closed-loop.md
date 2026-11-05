# Offline action-cost accounting

Use `AnswerPolicy.decide` to choose an answer, check, or abstention with explicit
utility prices, spending units and purchase limits. `replay` buys stored checks;
its `calls` counts the primary answer plus purchased checks, not money.
Compute acquisition cost from actual `used` checks and declared prices. An
acquired check costs money even if it fails or contributes zero evidence weight.

Score an independently observed outcome separately: a wrong action is
`result.answered and result.answer != gold`; abstention is not a wrong action.
Include both acquisition cost and the declared loss of wrong actions. Do not
substitute the policy posterior, admission verdict or check agreement for gold.
The replay record's `correct` flag describes the original answer, so keep it
consistent with the independent outcome; use the selected answer when scoring
replacement actions. Compare baselines with the same hard spending limits and
prices, without forcing every policy to exhaust its allowance.

`Economics` uses explicit owner estimates before expensive stages; its predicted
net is not measured realized utility. `RegimeMemory` accepts delayed external
observations associated with issue-time regimes; it neither executes probes nor
installs LawSets. See [answer policy](api/answer_policy.md),
[economics/history](change-economics-history.md) and [regime memory](regime-memory.md).
These helpers do not learn causal loss, fix selective labels or establish an
empirical utility improvement.
