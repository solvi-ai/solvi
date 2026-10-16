# `solvi.experimental.ood`

A shift watch for `System.calibrate(question, examples, watch=ShiftWatch())`, next to the default Platt scaling: the
known black-box shift test (per-feature two-sample KS with Bonferroni, Rabanser et al. 2019) on the decider's answer
features. `shift(batch)` says when a batch's answers are unlike the calibration answers; "out of domain" in an answer's
`why` marks one such answer. It sees the decider's confidence profile, not the content (see
[experimental](../experimental.md#ood)).

::: solvi.experimental.ood
