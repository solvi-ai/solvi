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

Compiled laws. An atom may carry its code — `Atom(name, fn, feature=, code="s['front'] == 'tree'", defs=...)`, a
Python expression over `s` and `a` that equals `fn`, with the module-level source it needs. `law.to_code(grammar)` is a
pure function for one law; `lawset.to_module()` one module that imports nothing from solvi (the laws in the set's own
order, most specific first; `predict(s, a, base=None, apply=None)` with your base predictor as the fallback). A law
whose atoms have no code is refused, naming them; so is an action or effect that is not a literal. Compiled code is
accepted only after the differential check: `check_compiled(module, buffer, lawset)` runs the module against
`lawset.predict` (and the applied law's id) on every row of the buffer, and the bar is 0 differences. A grammar may grow
(`grammar.add(atom)`, e.g. a relational atom a proposer invents); a buffer evaluates the new atom on its older rows the
first time a condition names it.

Sized acceptance. Fixed thresholds do not see how many candidates were tried on one buffer (a stand of ours tried about
52,000 at support ≥ 5). By default (`sized=True`) a candidate passes only when its hits would be unlikely if its true
precision were the null (min_precision; 0.95 when that is 1): P[Binomial(support, null) ≥ hits] ≤ δ_N, N = the
candidates scored on this buffer so far (`buffer.tested`), δ_N = δ / (N·(N + 1)) — summing to below δ however long the
stream runs — or δ / space when you declare `space=` (Bonferroni). A law right on every row then needs support ≥
`sized_need(N, δ, null)` ≈ (2·ln N + ln 1/δ) / ε, ε = 1 − null (at δ = 0.1, null 0.9: 29 rows at N = 1, 150 at N = 821, 228 at N = 52,000 or 125 with space=52,000); errors ask
for more. On a buffer with no law (2,000 rows; an effect at 70% independent of 40 sparse atoms; the 821 conditions of at
most two atoms) the fixed thresholds (support ≥ 5, precision ≥ 0.9) accepted a law in 100 of 100 buffers, the sized
rule in 0 of 100, and a real two-atom law on 250 rows was still accepted (tests/test_laws_w2.py). A small buffer
therefore accepts nothing — the rule working, not failing: declare `space=` or pass `sized=False` knowingly (the fixed
thresholds; the measured stands above ran so, and their false accepts are the known price). Each Law records `tested`, `need`,
`level` and `admitted` ("test", "probes", "hand").

Admission by probes. Where the world (or an exact model of it) answers questions, `lawset.admit_by_probes(candidates,
ask, probes=, safe=, budget=)` asks where the surviving candidates disagree, CEGIS-style: candidates refuted by logged
rows (`buffer=`) go first; the survivors are grouped into classes by their predictions on the probe pool (the set without
a new law is a hypothesis too); `split` picks the safe probe that splits the classes most; the refuted are dropped; the
shortest candidate is accepted only when one class survives (identifiability) — else an `Admission` "undetermined"
with the classes, never a guess. On a synthetic world (a hidden two-atom law, 176 candidates, 300 trials) at 8 checks a
random held-out sample with the shortest survivor accepted a wrong law in 0.593 of the trials; the active admission in
none (0.987 right, 0.013 undetermined, 7.45 asks). The probes are of the world you may ask: a law set whose other laws
or base are wrong where it probes may refute a right candidate (a candidate is refuted only where it is the applied law
and wrong, or where it does not hold, the set is wrong and its effect is what happened).

Abstention. `predict(state, action, disagree="abstain")` (or `LawSet(..., disagree="abstain")`) returns `Undetermined`
where the most specific matching laws — those no other matching law's condition strictly contains — predict different
outcomes: both passed their test, neither is an exception of the other, and the order between them would only be a
tie-break. It is counted in `abstained`; the default stays most specific first, the learning loop always predicts so,
and a compiled module dispatches most specific first. The search cost: `lawset.cost` (candidates proposed, tests,
accepted, probes, simulated) and `Step.tested` — report it next to accuracy.

What is not done:
- no grammar that cannot be enumerated (where the mismatch should matter most) has been measured;
- the sized rule, admission by probes and abstention are measured on synthetic worlds only: not yet on the game
  simulator or the file server stands, and a stand that tests tens of thousands of candidates needs a buffer of a few
  hundred rows per accepted law (a buffer of tens of rows needs `sized=False`, and then the fixed thresholds' false
  accepts are its price);
- retirement is by precision only; a weak law no observation contradicts stays;
- the store item's id is its content: a law retired and accepted again is the same item, asserted again;
- no wiring into a System's decisions or an Agenda gate (a law set predicts; using the prediction is yours)."""
from __future__ import annotations

from .. import warn_on_import
from ._compile import CompiledCheck, check_compiled, load_module
from ._core import (Admission, Atom, Buffer, CallableProposer, Candidate, Enumerate, Grammar, Law, LawSet, Proposer,
                    RandomProposer, Step, Transition, Undetermined, sized_level, sized_need)
from ._probes import Probe, discriminate, split
from ._rank import (FEATURES, BagOfAtomsRanker, FeatureRanker, RankedProposer, SyntheticTask, atom_features,
                    candidate_features, checks_to, synthetic_task)

warn_on_import(__name__)

__all__ = ["Admission", "Atom", "BagOfAtomsRanker", "Buffer", "CallableProposer", "Candidate", "CompiledCheck", "Enumerate",
           "FEATURES", "FeatureRanker", "Grammar", "Law", "LawSet", "Probe", "Proposer", "RandomProposer",
           "RankedProposer", "Step", "SyntheticTask", "Transition", "atom_features", "candidate_features",
           "check_compiled", "checks_to", "discriminate", "load_module", "sized_level", "sized_need", "split",
           "synthetic_task", "Undetermined"]
