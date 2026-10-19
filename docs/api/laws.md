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
as exceptions to a provenance guard's verdicts) the opt-in 1.2.1 features showed no measured gain over the default
(coverage ratio 0.981, soundness inconclusive) and harmful writes let through stayed above 0.15 (0.177 with every
feature on, 0.218 with the default, at about 0.10 false interruptions), about a tuned gradient-boosting model on the
same atoms (0.188). For noisy behavioural logs use a calibrated scorer.

::: solvi.experimental.laws

## `solvi.experimental.laws.factored`

A multi-law (factored) transition model: one `LawSet` and held-out `Buffer` per channel, δ split over the channels,
row provenance. Experimental: more accurate than one joint `LawSet` on MiniGrid, no saving in transitions.

::: solvi.experimental.laws.factored
