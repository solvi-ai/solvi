# `solvi.experimental.laws`

Laws of an environment from prediction mismatches (experimental): `solvi.experimental.laws.LawSet(grammar, base=...)`,
accepted only by a held-out test on other groups; a probe planner for live systems; export into the knowledge store.
Where each check is expensive (a live probe, a planner call), `RankedProposer(inner, scorer, buffer)` reorders another
proposer's candidates — `JointRanker` (ranks candidate conditions by the joint statistic of the whole condition over the
logged rows, no training; with no rows it is the blind order), `BagOfAtomsRanker` (trained on synthetic tasks drawn from
your grammar), `PriorRanker` (a prior order over the condition's own features — its number of atoms, its atoms' kinds,
their prior usefulness and the co-occurrence of their features in true conditions — learned from synthetic tasks of the
grammar; it reads no rows), `HandoverRanker(prior, joint, min_rows=20)` (the prior below `min_rows` logged rows of the
action, the joint statistic from there on; `last` says which scored) or the control `FeatureRanker` (a raw per-atom
statistic, no training); the candidates themselves never change.

::: solvi.experimental.laws
