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

The loop. Before every transition the set predicts (most specific matching law first, ties to the earlier law — or,
`ties="newer"`, the later — else the base predictor). A wrong prediction — a mismatch — asks the proposer for candidates "for this action, under this
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
In distribution only on the synthetic tasks. `JointRanker()` ranks candidate conditions by the joint statistic of the
whole condition over the logged rows — the Jaccard between the rows the whole condition matches and the rows where the
effect predicts right, equal Jaccard → fewer atoms — with no training: a sum over atoms cannot see two atoms that pick
out the effect only together, or a correlated twin of a true atom. With no logged rows every score is equal and the
order is the blind one — the statistic needs rows. On three synthetic settings of the library's own grammars (five seeds
× 60 tasks each; benchmarks/laws_rank_joint.py runs one seed) — the grammar above; deep conditions of 3–5 atoms of k = 5
with about 6,000 candidates per task and 5% label noise; correlated features (three feature pairs copied with p = 0.8) —
it needed 1.00 / 17 / 1.01 checks to the true condition (medians 1 / 1 / 1) against 85 / 2,182 / 126 blind, 27 / 1,310 /
47 for the raw Jaccard and 2.1 / 591 / 7.7 for the trained bag of atoms; a control scoring against the effect's rows
shuffled was worse than the blind order (112 / 2,434 / 166). On the deep setting the gain shows from about 20 logged
rows and is full by 300. The same law was accepted under every order in 894 of 900 tasks (by construction: the
acceptance scores every candidate; six exact ties); nothing was accepted in 180 of 300 deep noisy tasks at 300 rows —
acceptance needs more rows than ranking. The trained scorer it beats is the library's linear bag of atoms; in
distribution only.
Before the buffer has rows: `PriorRanker(seed).fit(grammar, sample, tasks=, rows=, sizes=)` scores a condition by its
own features — its number of atoms, its atoms' kinds (feature, comparison type read off the name: `atom_kind`), each
atom's prior usefulness (how often it is in the true condition when it is true in the mismatched state) and the
co-occurrence of its features in true conditions — a linear softmax over each synthetic task's candidates; it reads no
rows (`shuffle_labels=True` is the control). Unless its held-out checks on its own training tasks are 5% below both the
blind order's and a shuffled-label fit's, it records `learned=False` and scores as the blind order: a prior that learned
nothing scores as blind (the held-out check covers the training family only). `HandoverRanker(prior, JointRanker())` hands over gradually:
score = the joint Jaccard + n0 / (n0 + rows) × the prior's rank among the candidates in [0, 1] (n0 = 20): the prior's order
with no rows, the joint statistic's as rows grow; a prior with learned=False leaves the joint statistic alone;
`mode="switch"` is the fixed switch at `min_rows`; `last` says what scored (`RankedProposer.scored_by` too). The prior
learns what the task family favours: on a planted grammar whose sampler draws three features 30 times as often it needed
10.3 checks at 0 rows against 43.1 blind (mean over 40 tasks, one run; tests/test_laws_rank_prior.py); on a skewed family of deep conditions (the
second setting's grammar with a chosen skew: three features drawn 8 times and one comparison type 4 times as often;
5,112 candidates per task; five seeds, 300 tasks) 208 against 1,245 blind, and the blend 184 / 98.6 / 8.9 at 20 / 100 /
300 rows against 1,019 / 361 / 16.8 for the joint statistic (the fixed switch at 20 rows: the joint statistic's); on
the three settings above, drawn uniformly, the prior fell back in 2 of 5 seeds on the first and needed 0.93 / 0.91 /
0.83 of the blind checks, and on the deep one the blend at 300 rows needed 27.2 against 17.2 (111 of 300 tasks worse,
31 better, medians 2 and 1; benchmarks/laws_rank_joint.py, five seeds, one run).
Measured once on a live file server (8 mismatches; candidates contradicted by logged rows skipped free): every order
spent 29 probes in all, because 1–5 candidates per mismatch were left to probe and the true conditions were single atoms
— the shortest hypotheses, which the length-first order meets first. Ranking can pay only where many consistent
hypotheses remain after the free checks; the control first.

Knowledge. `to_knowledge(store, scope)` writes each law as a KnowledgeStore item of kind "rule" with body {"kind":
"law", "action", "if", "then"} (the sub-kind in the body, as the change pipeline's hazard rules), source "outcome",
evidence = support / hits / precision / net / the birth group / the ids of the held-out transitions it was accepted on;
a retired law is retire()d with its reason; writing again writes nothing new; the store's journal verifies and replays.

Measured (2026-10, two setups, one run each, fixed before the run): on a game simulator with written mechanics (20 fresh
worlds, 4,731 transitions) the laws closed 94% of the gap between a learned map and the written mechanics (0.892 → 0.988
against a ceiling of 0.994); random candidates from the same grammar with the same held-out test reached 0.977 — the
laws' own gain over them is 1.1 points, and the pre-registered bar for that control (random candidates at most +2 points
over the base) was missed by +8.4: the test does the work, the mismatch is a compact way to propose (61 laws against
103); an LLM proposer under the same test ended 0.95 points below enumeration. On a real MCP filesystem server with a
path policy the agent was not told, probes planned from each mismatch learned every denial class in one session:
refusals 6 → 0 in the second session (200 calls per session), no allowed call wrongly skipped, against 48 → 29 with
random probes at the same budget. benchmarks/laws_toy.py reproduces both findings in miniature (tests/test_laws_toy.py,
`pytest -m slow`).

Compiled laws. An atom may carry its code — `Atom(name, fn, feature=, code="s['front'] == 'tree'", defs=...)`, a
Python expression over `s` and `a` that equals `fn`, with the module-level source it needs. `law.to_code(grammar)` is a
pure function for one law; `lawset.to_module()` one module that imports nothing from solvi (the laws in the set's own
order, most specific first; `predict(s, a, base=None, apply=None)` with your base predictor as the fallback). A law
whose atoms have no code is refused, naming them; so is an action or effect that is not a literal. Compiled code is
accepted only after the differential check: `check_compiled(module, buffer, lawset)` runs the module against
`lawset.predict` (and the applied law's id) on every row of the buffer, and the bar is 0 differences. A grammar may grow
(`grammar.add(atom)`, e.g. a relational atom a proposer invents); a buffer evaluates the new atom on its older rows the
first time a condition names it.

Sized acceptance. Fixed thresholds do not see how many candidates were tried on one buffer (a setup of ours tried about
52,000 at support ≥ 5). By default (`sized=True`) a candidate passes only when its hits would be unlikely if it were no
better than its null: P[Binomial(support, null) ≥ hits] ≤ δ_N, N = the candidates scored on this buffer so far
(`buffer.tested`), δ_N = δ / (N·(N + 1)) — summing to below δ however long the stream runs — or δ / space when you
declare `space=` (Bonferroni). The null (`null=`): "base", the default — "no better than chance": the effect's base rate
among the buffer's rows of that action (the proposing group left out), floored at 0.05 and capped at min_precision — or
a fixed precision (`null=0.9`: the law must beat 0.9). min_precision stays a separate filter on the observed precision.
A fixed null close to 1 is the wrong question in a deterministic world: a law right on hundreds of rows can still fail
to beat 0.9 at tens of thousands of candidates. The need: a deterministic law needs ⌈ln(1/δ_N) / ln(1/null)⌉ rows, with δ_N = δ / (N·(N + 1)): at 52,000 candidates and δ = 0.1, 9 rows at a base rate of 3% (floored to 5%), 68 at 70%, and 228 against a fixed null of 0.9 (at 821 candidates: 6, 45, 150); errors ask for more. On buffers with no law
(2,000 rows; an effect independent of 40 sparse atoms; the 821 conditions of at most two atoms) the fixed thresholds
(support ≥ 5, precision ≥ 0.9) accepted a law in 100 of 100 buffers at an effect rate of 70%, the sized rule with the
base null in 0 of 100 (at 3%: 0 of 100 for both); a deterministic two-atom law at a base rate of 3% (1,000 rows) was
accepted in 100 of 100 buffers with the base null on a median of 29 rows, in 0 of 100 with a fixed null of 0.9
(tests/test_laws_sized_probes.py). A buffer too small for its null accepts nothing — the rule working, not failing: declare
`space=` or pass `sized=False` knowingly (the fixed thresholds; the measurements above ran so, and their false
accepts are the known price). Each Law records `tested`, `null`, `need`, `level` and `admitted` ("test", "probes",
"hand").

Admission by probes. Where the world (or an exact model of it) answers questions, `lawset.admit_by_probes(candidates,
ask, probes=, safe=, budget=)` asks where the surviving candidates disagree, CEGIS-style: candidates refuted by logged
rows (`buffer=`) go first; the survivors are grouped into classes by their predictions on the probe pool (the set without
a new law is a hypothesis too); `split` picks the safe probe that splits the classes most; the refuted are dropped; the
shortest candidate is accepted only when one class survives (identifiability) — else an `Admission` "undetermined"
with the classes, never a guess. On a synthetic world (a hidden two-atom law, 176 candidates, 300 trials) at 8 checks a
random held-out sample with the shortest survivor accepted a wrong law in 0.593 of the trials; the active admission in
none (0.897 right, 0.103 undetermined, 7.56 asks; at 10 checks 1.000 right). Where more than one law is missing the
"too narrow" refutation is not sound: the candidates it refuted that were never wrong where they applied are kept as
witnesses, and nothing is accepted while the set with all of them added still fits every row and predicts otherwise
than the surviving class where it would apply (x0 ∨ x1 with a base saying False: wrong admissions 29 → 6 of 100;
`assume_one_law=True` keeps the old rule). The probes are of the world you may ask: a law set whose other laws
or base are wrong where it probes may refute a right candidate (a candidate is refuted only where it is the applied law
and wrong, or where it does not hold, the set is wrong and its effect is what happened).

Interventions. `intervention(state_before, state_after, actions)` → one `Delta` per action: the prediction before and
after with the law behind each (`explain`), the change, and `covered` — False where the base predictor, not a law,
speaks on either side, so an effect the laws never tested is marked a guess.

Abstention. `predict(state, action, disagree="abstain")` (or `LawSet(..., disagree="abstain")`) returns `Undetermined`
where the most specific matching laws — those no other matching law's condition strictly contains — predict different
outcomes: both passed their test, neither is an exception of the other, and the order between them would only be a
tie-break. It is counted in `abstained`; the default stays most specific first, the learning loop always predicts so,
and a compiled module dispatches most specific first. The search cost: `lawset.cost` (candidates proposed, tests,
accepted, probes, simulated) and `Step.tested` — report it next to accuracy. `lawset.report(buffer, steps)` sums a
run up and states per action the rows the buffer has and the support the sized acceptance needs at this N
(`sized_need` against the action's least base null; `need_at_precision` for a law right on min_precision of its rows):
an action with fewer rows than that cannot get a law on this buffer — «too few rows», not «no law». For one candidate,
`lawset.why_not(candidate, buffer)` names the first test it fails, in `test`'s order (active, readmit, cap, net,
min_support, min_precision, exact, sized_need, sized_bound), and by how many rows — without touching the buffer's N.

Coarse outcomes and a backing-off base (opt-in). `LawSet(grammar, **coarse(lambda seen: seen["kind"]))`: a law predicts
the observation's class (an `OutcomeClass`), right when the observation falls in it — details no law can know (a random
amount) no longer make every row a mismatch; `report` counts outcomes by class. `BackoffBase(levels, min_rows=,
outcome=)` is a base predictor fitted on transitions: the most frequent outcome for the state's key at the most specific
level (`[("front", "wood"), ("front",), ()]`, the action always part of the key) with at least `min_rows` rows, backing
off to more general levels; `explain` names the level used. Fit it before the set uses it, on rows the laws are not
tested on.

History atoms (opt-in). `History(grammar, depth=1, actions=..., observed=..., same=True)` lifts a grammar to contexts —
the present state plus the last `depth` steps — with atoms "t-1:<atom>" (the state before), "t-1:action=<a>",
"t-1:observed=<v>" and "t-1:start" (the episode has just begun), and the effect `Same(1)` («what was observed one step
back»: one law for a value that persists instead of one per value). `LawSet(h.grammar, **h.kwargs(base=...))`; a
`Tracker` (`h.tracker()`) builds each step's `Context` along a stream and `reset()`s per episode; predict on
`tracker.context(state)`. The extra atoms enlarge the space the sized acceptance divides its error over.

Per-law confidence (opt-in). `LawConfidence(lawset, buffer)`: each active law's posterior precision on its held-out
rows, `exact` only with no miss on ≥ 60 rows, and at most one tail split — the atom where its misses concentrate, kept
when Fisher's exact test passes at alpha / atoms tested — with a shrunk confidence on each side. `of(state, action)` →
(value, law, confidence); `predict(state, action, below=t)` returns `Abstained` where the applied law's confidence is
below t. A tail no atom tracks gives per-law, not per-state, abstention. The confidence is an estimate — a Laplace
posterior mean plus one Fisher-exact split on an outside atom with m-estimate shrinkage, "exact" meaning no miss on
≥ 60 rows — not a bound: there is no risk guarantee, and the threshold t is yours to pick on your own data. Without
it the law set implicitly treats every law as exact.

Noisy probes (opt-in). `admit_by_probes(..., noise=ε, level=L)`: where an answer of the world may be wrong with a known
probability ε, one answer refutes nothing — every hypothesis keeps a likelihood, the next probe is the one with the
largest expected information gain under that noise (`split(score="eig", noise=ε)`, a repeat when that is best), and a
law is accepted only when one class of hypotheses holds at least L of the weight. On a synthetic single-law benchmark,
given the same number of answers as the best repetition scheme (a majority vote per probe or a per-probe sequential
test, each tuned under the same wrong-accept cap), it accepted the right law more often when answers were scarce and as
often with fewer answers when they were not; the default admission is often wrong under such noise. A known method:
noisy generalized binary search over classes of hypotheses, stopped at a posterior threshold.

Suspended laws (opt-in). `LawSet(..., on_fall="suspend", reactivate_min=m)`: a law whose running precision falls leaves
the active set but stays a candidate (`suspended`); it is re-checked on the next m fresh observations where it would
apply and comes back when they bear it out. `doubt_probe(probes)` names a probe where a suspended law and the set
disagree. The default still retires.

What is not done:
- no grammar that cannot be enumerated (where the mismatch should matter most) has been measured;
- the sized rule, admission by probes and abstention are measured on synthetic worlds only: not yet on the game
  simulator or the file server, and a setup that tests tens of thousands of candidates needs a buffer of a few
  hundred rows per accepted law (a buffer of tens of rows needs `sized=False`, and then the fixed thresholds' false
  accepts are its price);
- retirement is by precision only; a weak law no observation contradicts stays;
- the store item's id is its content: a law retired and accepted again is the same item, asserted again;
- no wiring into a System's decisions or an Agenda gate (a law set predicts; using the prediction is yours);
- noisy probes (`noise=`) assume symmetric flip noise with ε known — to the method and, through their tuning, to the
  repetition schemes it was compared with — and one missing law, on a synthetic single-law benchmark at three noise
  levels and two answer budgets; at the highest noise level with the small budget it did not hold the wrong-accept
  cap; a misspecified ε, state-dependent noise, numeric noise and several missing laws are not measured;
- suspension does not handle the "dead end": a law whose states the agent stops visiting is never re-checked and
  stays suspended;
- history atoms are measured at depth=1 only;
- `LawConfidence` is measured with ≥ 191 rows per law; at the default `exact_rows=60` a law right on 97 % of its rows
  is flagged exact about 1 time in 5 (0.97⁶⁰); it does not retire laws and gives no risk guarantee."""
from __future__ import annotations

from .. import warn_on_import
from ._coarse import BackoffBase, OutcomeClass, coarse
from ._compile import CompiledCheck, check_compiled, load_module
from ._confidence import Abstained, LawConfidence, LawStats, fisher_p
from ._history import Context, History, Past, Same, Tracker
from ._core import (Admission, Atom, Buffer, CallableProposer, Candidate, Delta, Enumerate, Grammar, Law, LawSet, Proposer,
                    RandomProposer, Step, Transition, Undetermined, Unknown, sized_level, sized_need)
from ._probes import Probe, discriminate, eig, split
from ._rank import (FEATURES, BagOfAtomsRanker, FeatureRanker, HandoverRanker, JointRanker, PriorRanker, RankedProposer,
                    SyntheticTask, atom_features, atom_kind, candidate_features, checks_to, synthetic_task)

warn_on_import(__name__)

__all__ = ["Abstained", "Admission", "Atom", "BackoffBase", "BagOfAtomsRanker", "Buffer", "CallableProposer", "Candidate",
           "CompiledCheck", "Context", "Delta", "Enumerate",
           "FEATURES", "FeatureRanker", "Grammar", "HandoverRanker", "History", "JointRanker", "Law", "LawConfidence", "LawSet", "LawStats", "PriorRanker", "Probe",
           "OutcomeClass", "Past", "Proposer", "RandomProposer",
           "RankedProposer", "Same", "Step", "SyntheticTask", "Transition", "atom_features", "atom_kind", "candidate_features",
           "check_compiled", "checks_to", "coarse", "discriminate", "eig", "fisher_p", "load_module", "sized_level", "sized_need", "split",
           "synthetic_task", "Tracker", "Undetermined", "Unknown"]
