# `solvi.experimental.laws`

Laws of an environment from prediction mismatches (experimental): `solvi.experimental.laws.LawSet(grammar, base=...)`,
accepted only by a held-out test on other groups; a probe planner for live systems; export into the knowledge store.
Where each check is expensive (a live probe, a planner call), `RankedProposer(inner, scorer, buffer)` reorders another
proposer's candidates — `BagOfAtomsRanker` (trained on synthetic tasks drawn from your grammar) or the control
`FeatureRanker` (a raw statistic, no training); the candidates themselves never change.

::: solvi.experimental.laws
