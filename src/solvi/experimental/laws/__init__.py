"""Laws of an environment from prediction mismatches (experimental): the environment is the judge, never a model.

    from solvi.experimental.laws import Atom, Grammar, LawSet, Buffer, Transition, discriminate

    g = Grammar([Atom("front=tree", lambda s, a: s["front"] == "tree", feature="front"),
                 Atom("front=stone", lambda s, a: s["front"] == "stone", feature="front"),
                 Atom("wood>=1", lambda s: s["wood"] >= 1, feature="wood")], k=3)
    held_out = Buffer(g)
    held_out.extend(Transition(s, a, seen, group=world) for s, a, seen, world in other_worlds)
    laws = LawSet(g, base=my_map)                     # base: what you predict without laws
    for t in stream:                                  # each transition is predicted before it is learned from
        laws.step(t, held_out)                        # mismatch → candidates → the held-out test → accept / retire
    laws.predict(state, "do")                         # the most specific matching law, else the base
    laws.to_knowledge(ks, {"env": "grid"})            # "rule" items of kind "law", sources and evidence; retired → retire()

The loop. Before every transition the set predicts (most specific matching law first, ties to the earlier law, else the
base predictor). A wrong prediction — a mismatch — asks the proposer for candidates "for this action, under this
condition, the effect is this". A candidate is accepted only by a test on held-out transitions of other groups than the
one it was proposed from (another world, another session, another path): support ≥ min_support, precision ≥
min_precision, and it removes more errors than it adds where it would be the applied law (net > 0); of the passing
candidates the one with the largest net is accepted (ties: fewer atoms, higher precision). A law whose running held-out
precision (the buffer rows where it applies, plus later observations from other groups) falls below retire_below is
retired. At most `cap` laws are active.

The grammar is yours: boolean atoms over the state (`fn(state)` or `fn(state, action)`), grouped by feature, and
conjunctions of at most k of them (one atom per feature). Effects: by default the observation itself; `apply(effect,
state)` and `effects(state, observed)` let an effect be a change ("+1 wood") that predicts different observations in
different states.

Proposers (`Proposer`: `propose(lawset, state, action, observed)` → candidates). `Enumerate` (default): every condition
true in the mismatched state. `RandomProposer`: conditions drawn from the whole grammar — the control. `CallableProposer`:
any function, an LLM among them; it proposes, the test decides.

The probe planner. Where each test is a call into a live system, `split(hypotheses, safe, probes, read_only=, cost=)`
picks the safe probe whose predicted outcomes split the hypotheses most — read-only actions before writes — and
`discriminate(laws_a, laws_b, safe, probes)` the cheapest safe probe where two law sets differ; `lawset.hypothesis(c)`
is the set with one candidate added, so a mismatch's candidates become hypotheses to tell apart.

Ranking candidates where each check is expensive (a live probe, a planner call). `RankedProposer(inner, scorer,
buffer)` reorders another proposer's candidates and never adds, drops or changes one: the order is what a probe budget
spends (the hypotheses `split` / `discriminate` receive first). Features come from the buffer's bitsets
(`atom_features`: per atom, its precision, recall, Jaccard and anti-match against the effect on the logged rows, its
rate, the base rate, and the best of the pairs among the top atoms — a fixed length; `candidate_features` sums them per
condition). `BagOfAtomsRanker(seed).fit(grammar, sample)` is a numpy logistic regression from an atom's features to
"this atom is in the true condition", trained on synthetic tasks drawn from the grammar (`synthetic_task`: a random
condition labels sampled states); a condition's score is the sum of its atoms' log-odds, so no position matters.
`FeatureRanker(stat)` is the control: the raw statistic, no training. On synthetic tasks of one grammar (7 categorical
and 3 counter features, 30 atoms, k = 3; 60 fresh tasks with 2% label noise, trained on 150) the checks to the true
condition were 69.7 in the blind order (by length, then name), 2.1 with the trained ranker, and 24.6 / 30.4 / 42.8
with the raw Jaccard / recall / precision (the best raw threshold picked after the fact: 12.8; tests/test_laws_rank.py).
In distribution only:
never measured on a live system or a grammar the ranker was not trained on.

Knowledge. `to_knowledge(store, scope)` writes each law as a KnowledgeStore item of kind "rule" with body {"kind":
"law", "action", "if", "then"} (the sub-kind in the body, as the change pipeline's hazard rules), source "outcome",
evidence = support / hits / precision / net / the birth group / the ids of the held-out transitions it was accepted on;
a retired law is retire()d with its reason; writing again writes nothing new; the store's journal verifies and replays.

Measured (2026-10, two stands, one run each, fixed before the run): on a game simulator with written mechanics (20 fresh
worlds, 4,731 transitions) the laws closed 94% of the gap between a learned map and the written mechanics (0.892 →
0.988 against a ceiling of 0.994); random candidates from the same grammar with the same held-out test reached 0.977 —
the test does the work, the mismatch is a compact way to propose (61 laws against 103); an LLM proposer under the same
test ended 0.95 points below enumeration. On a real MCP filesystem server with a path policy the agent was not told,
probes planned from each mismatch learned every denial class in one session: refusals 6 → 0 in the second session,
no allowed call wrongly skipped, against 48 → 29 with random probes at the same budget. benchmarks/laws_toy.py
reproduces both findings in miniature (tests/test_laws_toy.py, `pytest -m slow`).

What is not done:
- no grammar that cannot be enumerated (where the mismatch should matter most) has been measured;
- retirement is by precision only; a weak law no observation contradicts stays;
- the store item's id is its content: a law retired and accepted again is the same item, asserted again;
- no wiring into a System's decisions or an Agenda gate (a law set predicts; using the prediction is yours)."""
from __future__ import annotations

from .. import warn_on_import
from ._core import (Atom, Buffer, CallableProposer, Candidate, Enumerate, Grammar, Law, LawSet, Proposer,
                    RandomProposer, Step, Transition)
from ._probes import Probe, discriminate, split
from ._rank import (FEATURES, BagOfAtomsRanker, FeatureRanker, RankedProposer, SyntheticTask, atom_features,
                    candidate_features, checks_to, synthetic_task)

warn_on_import(__name__)

__all__ = ["Atom", "BagOfAtomsRanker", "Buffer", "CallableProposer", "Candidate", "Enumerate", "FEATURES", "FeatureRanker",
           "Grammar", "Law", "LawSet", "Probe", "Proposer", "RandomProposer", "RankedProposer", "Step", "SyntheticTask",
           "Transition", "atom_features", "candidate_features", "checks_to", "discriminate", "split", "synthetic_task"]
