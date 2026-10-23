# `solvi.experimental.laws`

Laws of an environment from prediction mismatches (experimental): `solvi.experimental.laws.LawSet(grammar, base=...)`,
accepted only by a held-out test on other groups; a probe planner for live systems; export into the knowledge store.
Where each check is expensive (a live probe, a planner call), `RankedProposer(inner, scorer, buffer)` reorders another
proposer's candidates — `JointRanker` (ranks candidate conditions by the joint statistic of the whole condition over the
logged rows, no training; with no rows it is the blind order), `BagOfAtomsRanker` (trained on synthetic tasks drawn from
your grammar), `PriorRanker` (a prior order over the condition's own features — its number of atoms, its atoms' kinds,
their prior usefulness and the co-occurrence of their features in true conditions — learned from synthetic tasks of the
grammar; it reads no rows and falls back to the blind order when it learned nothing), `HandoverRanker(prior, joint)` (by default a blend: the joint Jaccard plus 20 / (20 + rows) times the prior's rank
among the candidates; `mode="switch"`: the prior below `min_rows` logged rows of the
action, the joint statistic from there on; `last` says which scored) or the control `FeatureRanker` (a raw per-atom
statistic, no training); the candidates themselves never change.
`BudgetedProposer(inner, candidates, neg_budget=, mismatches=)` cuts another proposer's candidates to a task budget
in its order; with `Grammar(negations=...)` the candidates with a negated literal get their own budget (`neg_budget`,
default the same size; `"shared"`: one budget), so they never crowd out a later mismatch's positive candidates. Both are experimental and off by default; an
admitted negated law can steer the search away from the true law (numbers in the best practices).

Scope: environments whose behaviour follows discrete rules — the features were measured on grid games and relational
laws over generated graphs. On a real agent tool-call log (write calls of the public AgentDojo benchmark, laws learned
as exceptions to a provenance guard's verdicts) the opt-in 1.3.0 features showed no measured gain over the default
(coverage ratio 0.981, soundness inconclusive) and harmful writes let through stayed above 0.15 (0.177 with every
feature on, 0.218 with the default, at about 0.10 false interruptions), about a tuned gradient-boosting model on the
same atoms (0.188). For noisy behavioural logs use a calibrated scorer.

Numeric laws: the held-out test accepts row by row. `LawSet.test` counts a held-out row as a hit when the law's
prediction is within `tol` / `rtol` / `same` of the observed value, and accepts on that count. Nothing reads the
residuals together, so two cases go wrong. A law with a systematic bias — every residual inside the tolerance but on one
side, or growing with an input — passes as easily as the true law. When the law's constants are fitted from the data, a
tolerance set from the measurement error alone ignores the error of the fitted constants, and the true law can fail
rows it should pass. Before trusting an accepted numeric law, check its residuals as a whole: a `confirm=` closure over
the held-out rows that refits the constants and runs, for example, a χ² or a sign test on the residuals. Widen the
tolerance by the fitted constants' uncertainty. The χ² after a refit is built in as an opt-in,
`residual_check(refit, level=0.995)`, used with `tol=math.inf` (see its docstring and
[best practices](../best_practices.md) for the measurement and its cautions); the widened tolerance and the sign test
are not shipped.

::: solvi.experimental.laws

## `solvi.experimental.laws.factored`

A multi-law (factored) transition model: one `LawSet` and held-out `Buffer` per channel, δ split over the channels,
row provenance. Experimental: more accurate than one joint `LawSet` on MiniGrid, no saving in transitions.

::: solvi.experimental.laws.factored
