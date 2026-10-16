# Experimental

`solvi.experimental` holds pieces that work and are tested but have not yet shown a measured gain, or a measured use,
of the kind the rest of solvi promises. Their API may change in any release. Each one either graduates or is removed by
**1.2** (two minor releases after 1.0).

```python
from solvi.experimental import STATUS
STATUS["lora"]       # {"since": "0.7", "what": "...", "missing": "...", "script": None, "deadline": "1.2"}
```

How you can tell you are using one:

- importing it warns once with a `solvi.ExperimentalWarning` (a `UserWarning`; silence it with
  `warnings.filterwarnings("ignore", category=solvi.ExperimentalWarning)`);
- a decision made by a System that uses one records it in the stored decision (`meta["experimental"]`, e.g.
  `["lora"]`), and `solvi report --overview` counts those decisions;
- nothing stable imports them, and `solvi` does not re-export them. One command loads one when you ask for it:
  `solvi hook` (hooks).

**Graduating** needs a bar fixed before the measurement, a script in `benchmarks/` that measures it, the bar met and
independently reviewed, no import of anything experimental, a conformance check, and a docs move; the old path then
keeps working for one release. **Removal**: a piece that has not graduated by its deadline, or that measures negative,
is removed and the negative result published.

Each section below gives the piece's `STATUS` entry: what it is, since when it exists, what it is missing to graduate,
the script that measures it (`changes`, `laws`, `skills`, `relations`, `charts`) and where the guide describes it.

## `lora`

A LoRA adapter on the decider for one question (`solvi.experimental.lora.adapt_lora(part, examples)`; was
`part.adapt_lora(...)`; needs `solvi[lora]`). Since 0.7.

- **Missing:** a decision on real use, a benchmark script in the repo, and wiring as the adapter rung of an update (solvi.experimental.changes); measured gains over fit come with overconfidence.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [A LoRA adapter per question](guide.md#a-lora-adapter-per-question-adapt_lora-experimental);
  API: [solvi.experimental.lora](api/lora.md).

## `compile`

A written specification compiled by an LLM into catalog parts, accepted when two drafts agree and the specification's
tests pass (with its sandbox, `solvi.experimental.compile.sandbox`). Since 0.9. Since 1.1 an optional certificate on fresh examples sized by the hypothesis space (`certificate=`, `fresh_examples_needed`, `skew_guard`).

- **Missing:** measured once on 15 classes of small pure functions over typed slots (a search over a bank of 137 operations, |H| up to 6.5·10⁸ programs of depth 2, N = 314–452 fresh examples; a small hosted LLM writer at N = 60, two drafts): no accepted function was wrong on 200 hidden examples per class (21 accepted), the search found 6 of 15 classes and the writer 15 of 15 — but neither proposer produced a false candidate, so whether the sized certificate refuses what a five-example certificate accepts was not shown on that setup; it is shown only on a planted constant in a smoke test; its necessity on a live stream of drafts is not shown; still missing: acceptance on larger specifications and a benchmark script in benchmarks/.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [A specification compiled into the catalog](guide.md#a-specification-compiled-into-the-catalog-solviexperimentalcompile);
  API: [solvi.experimental.compile](api/compile.md), [sandbox](api/sandbox.md).

## `hooks`

`solvi hook`: a coding agent's pre-edit rule checks and skill picker (Claude Code; Codex built from its documented
hook schema). Since 0.7.1.

- **Missing:** false deny and ask rates on real coding-agent sessions, with a script.
- **Script:** none yet. **Deadline:** 1.2.
- Guide: [solvi behind a coding agent's hooks](guide.md#solvi-behind-a-coding-agents-hooks);
  API: [solvi.experimental.hooks](api/hooks.md); runnable: `examples/22_coding_agent_hooks.py`.

## `charts`

Verified charts: every number quoted from the text, a deterministic SVG. Since 0.7.

- **Missing:** Measured on 50 paragraphs of US federal news releases against a hand reading written before any proposer ran (`benchmarks/charts_real_texts.py`, one run): the rule-based proposer drew a chart for 50 of 50 texts with every rendered value re-verified against the text (350 of 350 chart numbers), but only 33.3% of the drawn points agreed with the hand reading in label and unit (the bar was 95%: labels 39, units 82 of 99 points); the LLM proposer (gpt-oss-120b) drew a chart for 56% of the texts and agreed on 94.2% of the points, with 5 of 485 chart numbers not in the text (a sign the proposer added); a changed digit and swapped values were caught 300 of 300 times by both, a dropped point 74 (rules) and 109 (LLM) of 300; every run replays byte for byte (50 of 50). Still missing: a proposer whose labels and units match a reader's on real texts.
- **Script:** `benchmarks/charts_real_texts.py` (the LLM arm with `--llm MODEL`). **Deadline:** 1.2.
- Guide: [Verified charts](guide.md#verified-charts-a-specialist-that-checks-every-number);
  API: [solvi.experimental.charts](api/charts.md); runnable: `examples/21_verified_chart.py`.

## `changes`

One change pipeline for four sources (a document version, verified experience, a person's correction, a search
proposal): pluggable stages, a journal of kind "change", promote into the knowledge store / compiled versions / a
labelled update (`LearningTarget`: the update ladder and its gates, which were the learning loop's until 1.1), versions
and rollback (`solvi.experimental.changes.Pipeline(stages, storage, store=...)`). Since 1.1.

- **Missing:** on a credit policy the pipeline carried a document version (144 of 144 changed decisions traced to their clause), admitted 12 of 12 exact exceptions, held every widening one for a person and refused 8 of 8 exceptions that touched other decisions, 0 extraneous changes, 14 versions replayed; on a bank-request stream with a person's labels every one of 22 candidates passed every stage — the run with the gates equals the run without them — so the stages added nothing there — the module is journal, impact analysis, approval levels and rollback; still missing: narrows / widens proved over rule code outside a Python subset (calls, subscripts, loops: measured on probes instead), a regression-scenario stage, and a script for the bank-request stream (the credit numbers: benchmarks/changes_credit.py).
- **Refusal regression cannot be switched off.** Refusal is the first casualty of self-change, so for every change whose target can answer (a target with
  `answerer(change or None, ctx)` — `VersionsTarget`, `LearningTarget`, yours — or `Pipeline(answer_of=...)`) the
  pipeline runs `RefusalRegression`: the share refused on a held should-abstain set (`Pipeline(refusals=...)`, the
  `refusals=` kept by CleanCases / Confirmation, the stage's own) before and after; a change that answers an input
  refused before is refused, with the set and the counts in the reason (tolerance 0.0 and a count tolerance of 0 by
  default; raise `count_tolerance` for an answerer that flips a borderline input between runs). It is inserted before
  Approval when it is not among the stages, journaled for every such change, final under every approval level, and
  there is no argument that removes it. On the credit policy a threshold change that closes the review band ("points =
  3 → approve") is refused by it: it answers 137 of 137 applications the policy hands to a person; the 12 admitted
  corrections and both document versions pass. Without a declared set the change is held for a person ("no should-abstain set declared for a target that can
  answer"): every pipeline whose target can answer declares `refusals=` (directly, through CleanCases / Confirmation,
  or on the stage). The set follows the policy version — the credit benchmark declares it once version 2 is in force. An
  item change is checked only with `answer_of=`.
- **Slices and the original version.** `SliceGuard(measure, original=None, tol=0.02)` judges a change per slice on
  paired per-case losses (`measure(change, ctx)` → `{slice: (in_force_losses, candidate_losses)}` on the same cases),
  against the version in force and against the original one (pinned at the guard's first run, or given), so harm in a
  small slice under an overall gain and a chain of small changes each harmless on its own ("salami") are refused. A
  slice is refused above the tolerance or when its excess is significant (one-sided z, Bonferroni over every slice
  and reference — valid however the slices depend on each other; cases within a slice are taken as independent). A
  slice under `min_n` cases, or one the original has and the measurement lost, holds for a person by default. On a toy
  target eight steps of +0.012 on a 40-case slice (tolerance 0.03) all pass a guard that compares with the version in
  force only (+0.084 in all); with the original the chain stops at +0.036 (`tests/test_changes_slices.py`). The pin lives
  in the guard object, so a new process re-pins its own "original": `pin_file="slices_original.json"` keeps it in a
  file written once (read-only) at the first run in any process, with its SHA-256 in the verdict; `pinned=<sha256>`
  (kept next to `EvaluatorLock`'s pins) refuses a change when the file differs or is missing. With a new guard per change
  the same eight steps go 7 of 8 through without the file (+0.084) and stop at +0.036 with it.
- **Post-ship watch (`ShipWatch`, `pipe.watch(version, promise)`) and `Pipeline.revert(version)`.** Per declared slice
  the watch keeps a test martingale against that slice's promised failure rate on the promoted change's own (alone)
  outcomes; when one slice's evidence reaches slices/α it reverts the watched version, and only it
  (`pipe.revert(v)`; `rollback(v)` undoes every version after v). By construction a change that keeps every slice's
  promise is reverted with probability at most α, at any time (Bonferroni over slices). Measured on a synthetic simulation
  only (not in the repo; four slices at 60/25/10/5 % of traffic, pre-ship failure 5 %, promise 10 %, α 0.05, 4,000
  outcomes): a 5 % slice raised to 30 % failure was reverted in 60 of 60 rollouts; 0 of 60 clean rollouts were
  reverted, and 14 of 600 at exactly the promise (within α); a milder harm (that slice at 20 %, twice the promise)
  was reverted in 44 of 60 within the horizon. It is not faster than a mean-with-tolerance rule checked
  after every outcome — its gain is the bounded false-revert rate, not detection delay. `outcome="loss"` takes a
  numeric loss in [0, 1] per outcome instead of a failure (also `EProcess.update(..., loss=)` / `loss_refuted` in
  `experimental.skills`); not measured.
- **Script:** `benchmarks/changes_credit.py` (the credit policy; needs the German Credit applications:
  `benchmarks/tasks/fetch.sh credit` and `prepare.py`). **Deadline:** 1.2.
- Guide: [The change pipeline](guide.md#the-change-pipeline-solviexperimentalchanges); design and API: the module
  docstring of `solvi.experimental.changes`; the library pieces it uses are stable:
  sensitive fields and citations in [solvi check](guide.md#checking-a-catalog-solvi-check), `only=` in
  [solvi diff](guide.md#catalog-fingerprint-solvi-diff-and-shadow-mode).

## `laws`

Laws of an environment from prediction mismatches, judged by the environment, never by a model
(`solvi.experimental.laws`): a grammar of boolean atoms you write, candidates "for this action, under this condition,
the effect is this" from each wrong prediction, acceptance only by a held-out test on other worlds, sessions or paths,
most specific first, retirement when the held-out precision falls, a probe planner for live systems (`split`,
`discriminate`: read-only before write) and export into the knowledge store (`LawSet.to_knowledge`). The acceptance is
sized by how many candidates were tested on the buffer (`sized=True`, the default; the null is the effect's base rate, `null="base"`:
a deterministic law at a base rate of 3% needs about 9 rows at 52,000 candidates, at 70% about 68, against a fixed
null of 0.9 about 228; a buffer too small for its null accepts nothing — declare `space=` or pass `sized=False`
knowingly),
`LawSet.admit_by_probes` admits a law by asking the world where the candidates disagree and only when one class
survives, `predict(..., disagree="abstain")` abstains where laws that are not exceptions of one another disagree, and
`LawSet.cost` records the search cost. Since 1.1. Where more than one law may be missing,
`admit_by_probes(..., max_new=2)` (opt-in, 1.2; the default is unchanged) admits over a version space of law sets
with up to two new laws and splits their classes with discriminating probes — a known method (version spaces,
Mitchell 1982; generalized binary search over equivalence classes, Golovin et al. 2010) applied to pairs of laws. On
two synthetic two-law worlds (10 binary atoms, laws of at most two atoms, a pool of 200 states, 1,000 trials each,
budget 16) every accepted law was correct in 1,000 of 1,000 trials (95 % upper bound 0.3 %) against 5–6.5 % wrong for
the default, and both missing laws were found in about 42 % (one right law in the rest). It costs probes: 7.3–7.7
against 5.2–5.4 there, and 11.7 against 7.6 on a single-law benchmark, where at a budget of 8 it is undetermined in
97.5 % of trials — give it 12 or more. Three or more missing laws are not measured; `coverage` is not checked in this
mode; synthetic boolean worlds only.

`Buffer(window=N)`, `LawSet(recent=M, readmit="grown")`, `LawSet(memo=True)` and `LawSet.proposals` (opt-in, 1.2; the
defaults are unchanged) let a law set follow a stream whose laws change: the buffer keeps its last N rows, a law's
running precision counts only its last M observations, and a retired law is readmitted only after the window has turned
over. On synthetic streams whose laws change abruptly (two conjunctive laws over 12 atoms, one flipped and one replaced
at each of five switches, phases of 1,500–2,500 rows), `Buffer(window=150)` with `LawSet(recent=50, readmit="grown")`
re-learned the new laws in 605 rows on average (30 of 500 switches censored), where a periodic reset tuned under the
same rule took 1,080 (83 censored) and the library default did not re-learn within any phase (an oracle with known
switch points: 238); prequential accuracy 0.796 against 0.755 (reset) and 0.692 (default), oracle 0.813, with 0 wrong
laws among 1,221 accepted (reset 3 of 1,258); no arm accepted a law on streams without laws (a property of the sized
rule, shared by all arms). The window length was tuned to this phase length: another drift rate needs another window,
and an adaptive window is not built. Synthetic abrupt drift only — no gradual, incremental or recurring drift, no real
stream. The first measurement of this feature was mis-scored (a cache bug: 31 wrong laws that were in fact 1) and the
bar was then relaxed to at most 3 wrong laws; this run meets it at 0, as it meets the original bar of 0. The score memo
(`memo=True`) and the proposals count are built and unit-tested, not measured. A sliding window with forgetting and a
retirement on recent performance is a standard answer to concept drift (FLORA, Widmer and Kubat 1996; ADWIN, Bifet and
Gavaldà 2007; Gama et al. 2014); what is the library's is only how the sized rule's N behaves under a window (it never
decreases; a candidate counts again after a full turnover) and `readmit="grown"`.

- **Missing:** on a game simulator with written mechanics the laws closed 94% of the gap between a learned map and the written mechanics (20 fresh worlds, 4,731 transitions, one run; 0.892 → 0.988, ceiling 0.994), but random candidates under the same test came within 1.1 points — the measurement's pre-registered bar for that control (random candidates at most +2 points over the base) was missed by +8.4 — so the mismatch is a compact proposal heuristic and the test does the work; on a real MCP filesystem server with a path policy, probes planned from mismatches took refusals from 6 to 0 in one session (200 calls per session, one run; random probes: 48 to 29); still missing: a grammar too large to enumerate, retirement of weak laws nothing contradicts, wiring into a System's decisions, and a script for the two measurements (the toys: benchmarks/laws_toy.py); a ranker of candidates for expensive checks (RankedProposer, a bag-of-atoms model trained on tasks drawn from the grammar) cut the checks to the true condition from 69.7 in the blind order to 2.1 on synthetic tasks of one grammar, in distribution only; ranking by the joint statistic of the whole condition over the logged rows (JointRanker, no training; with no rows it is the blind order — the statistic needs rows) needed 1.00 / 17 / 1.01 checks (medians 1 / 1 / 1) on three synthetic settings of the library's own grammars (five seeds × 60 tasks) against 85 / 2,182 / 126 in the blind order (85 a different draw of tasks from the 69.7 above: five seeds against one) and 2.1 / 591 / 7.7 for the trained bag of atoms, a shuffled-rows control worse than the blind order (112 / 2,434 / 166), the gain on the deep setting showing from about 20 logged rows and full by 300, the same law accepted under every order in 894 of 900 tasks; before a buffer has rows, a prior order over the condition's own features learned from synthetic tasks of the task family (PriorRanker, which falls back to the blind order unless its held-out checks are 5% below the blind order's) needed 208 checks at 0 rows against 1,245 blind on a skewed family (deep conditions whose features are drawn unevenly by a chosen skew; 5,112 candidates per task; five seeds, 300 tasks, training and test tasks from the same sampler) and needed 0.83–0.93 of blind on the three uniform settings (on the first it fell back in 2 of 5 seeds); HandoverRanker blends the two — the joint Jaccard plus 20 / (20 + rows) times the prior's rank in [0, 1] — and on the skewed family needed 184 / 98.6 / 8.9 checks at 20 / 100 / 300 rows against 208 for the prior and 1,019 / 361 / 16.8 for the joint statistic, but on the deep uniform setting at 300 rows 27.2 against 17.2 (111 of 300 tasks worse, 31 better, medians 2 and 1; the pre-registered 5 % bar missed there; a smaller n0 not measured); measured once on a live file server (8 mismatches; candidates contradicted by logged rows skipped free): every order spent 29 probes in all, because 1–5 candidates per mismatch were left to probe and the true conditions were single atoms — the shortest hypotheses, which the length-first order meets first — so ranking can pay only where many consistent hypotheses remain after the free checks, the control first; the acceptance sized by the candidates tested on a buffer (on a synthetic buffer with no law the fixed thresholds accepted a law in 100 of 100 buffers, the sized rule in 0 of 100), admission by discriminating probes (on a synthetic world at 8 checks a random held-out sample accepted a wrong law in 0.593 of the trials, the probes in none) and abstention where laws that are not exceptions of one another disagree are measured on synthetic worlds only, not yet on the game simulator or the file server; the live probe-based learner can admit false laws on policies the grammar cannot express (observed once per control policy) — admission by probes or sized acceptance is the open item.
- **Script:** `benchmarks/laws_toy.py` (two deterministic toys that reproduce both findings in miniature; `pytest -m
  slow tests/test_laws_toy.py`); `benchmarks/laws_rank_joint.py` (the three ranking settings; one seed per run, `--seed`;
  the numbers above are five seeds; `--curve` for the rows curve). **Deadline:** 1.2.
- Guide: [Laws from an environment](best_practices.md#laws-from-an-environment-the-test-is-the-judge), [sized
  acceptance, probes, abstention](best_practices.md#size-the-acceptance-by-the-search-admit-by-probes-abstain-when-laws-disagree), [ranking
  candidates](best_practices.md#rules-over-relations-enumerate-with-a-library-rank-only-where-a-check-is-expensive); API:
  [solvi.experimental.laws](api/laws.md).

## `injection`

A detector of instruction-like text in tool outputs, fitted and thresholded on your own texts: a per-sentence logistic score over the fixed
detector's rule families and hashed word n-grams, fitted on labelled texts, its threshold set on ordinary texts to a
chosen false-flag rate. Since 1.2. Opt-in and shipped without weights: the fixed rules (`injection_spans`) stay Guard's
detector. Fit it on labelled outputs your own tools return and set its threshold on your ordinary ones
(`calibrate(max_flag_rate=)`); its gain held only on the injection family it was fitted on.

- **Missing:** measured once on injected tool outputs of one public benchmark (InjecAgent): on 248 held-out injections
  (attacker instructions and tools not used in fitting) a detector fitted on 279 other cases and 865 ordinary texts
  found 76% against the fixed rules' 23%, flagging fewer ordinary texts (1.8% against 4.2%); but a plain TF-IDF classifier
  did the same, half of the gain was one address repeated across that dataset (without it +0.19, 95% CI +0.06…+0.38),
  and on 140 AgentDojo-style injections the rules found 78%, the detector 53%; still missing: a gain on an injection
  family it was not fitted on, and wiring into Guard as its detector.
- **Script:** none yet. **Deadline:** 1.2.
- API: [solvi.experimental.injection](api/injection.md).

## `relations`

Relations over a state and rules found over them (`solvi.experimental.relations`): a state becomes named relations over
its nodes, stored as packed bits (a path tree: PARENT, SAME_DIR, IS_DIR, NAME=x; a grid seen from the player: EAST,
SOUTH, AT, FACE and the cells' materials and creatures); a relation-algebra enumerator builds programs bottom-up
(composition, converse, union, intersection, difference, a bounded complement, transitive closure, path counts) and
keeps one per behaviour — two programs are the same only when their full relations agree on every example; programs are
scored by the support and precision of a readout (does the subject relate to anything), noise-tolerant, and a
`Library` keeps accepted ones as terminals of later searches. `RelAlgProposer` hands the best programs to a law set as
atoms, and its unchanged held-out test judges them. A found rule runs as code: `compile_program` writes plain set code
(or index arrays for large graphs; dense answers refused) and `check_program` compares it with the interpreter pair for
pair; with `Atom(..., code=)`, `LawSet.to_module()` writes a law set as one module and `check_compiled` runs it against
`LawSet.predict` on every row (the bar: 0 differences). Composites by enumeration with a library
(`relations.library.find_rules`): enumerate small programs over the base relations, admit a program only when it passes
the held-out test on other worlds, and enumerate again with the admitted rules as new terminals, so a rule three levels
deep is a small program over earlier finds; a learned proposer has a fallback slot (`proposer=`), none ships.
Composition tables (`relations.CompositionTable`): which relation two relations compose to, learned by exact counting
from labelled chains (majority share and support thresholds), empty cells filled by residuals from chains of three, a
longer chain derived by composition over every bracketing and answered only when the bracketings agree — otherwise
refused with the reason; the cells used are the answer's reason and go into the knowledge store as rule items.
Forward chaining (`relations.forward_chain`): the pair facts of a scope and the composition rules in the knowledge
store derive new pair facts to a fixed point, each written through the store's gates with its two premises and the
rule as its justification and never ranked above the weaker premise; two derived relations of one pair are a
dispute (the table's refusal by disagreement, as a question for a person); a retracted premise takes its
consequences along and a promoted rule or an answered dispute turns them into facts without a re-run; `explain`
reads the derivation back. Rules first, a model second (`relations.RulesFirst`): the table answers what it derives and
no model is called; what it refuses goes to a fallback (a decider or an LLM) with the question rendered from the chains,
or stays a refusal; every answer names its path and replays from its record without the model. Residual synthesis
(`relations.synthesize_residual`, `synthesize(method="residual-first")`): a chain program found top-down from the target
through residuals, checked exactly and by the held-out test, the enumerator kept for what chains cannot express. Why not
derived (`relations.why_not`): the missing cell or fact behind a refusal, ranked by the paths it would unblock, as a
goal for the agenda. Guarded rules (`relations.GuardedRules`, `relations.chain_rules`): Datalog-lite over pair facts
and the story's unary facts — bodies with unary guards, chains through derived facts, «somewhere else», a negative
stratum — learned by counting on labelled pairs (a pair's label is a set of relations), confirmed on a check slice and
recounted on the joint fixed point after each round; a pair is answered with its active relations or refused. A text frontend (`relations.TextFrontend`, `relations.GraphCheck`; 1.1.1): five sentence patterns of kinship stories read into a relation graph by code (no model), every edge with its inverse by the relation's inverse table, gender facts with their source, a global check that drops a gender contradiction, two relations on one pair or a second father / mother / husband / wife and names it; a sentence no pattern reads gives no edge (a refusal), and an edge read wrongly is a wrong answer. Since 1.1.

- **Missing:** on a pre-registered battery of 28 random rules over three relations (the rule file was written in the author's earlier research line before any search ran and is not shipped — `relations_battery.py --rules` takes it; the bars were fixed from that line's reference run, so this is a port-fidelity check, one deterministic run) the enumerator found 22 within seven nodes, every find exact on larger graphs of other families, and compiled rules on a forest of 100,000 nodes were equal pair for pair; the index-array backend within 0.7–1.3× of hand code over a parent array (output conversion excluded); the plain set-code backend 2–3× slower than hand-written set code over the same pairs (benchmarks/relations_compile.py), so compiled relation programs are not faster than hand-written set code on graphs (2–3× slower); on five law sets learned on a game simulator with written mechanics (61–103 laws each; one with 64 relational atoms over a 3×3 window compiled as a bounded domain), `LawSet.to_module` + `check_compiled` agreed with `LawSet.predict` on every one of 308,510 logged transitions (training, validation and test worlds), no atom refused (a differential check on logged rows certifies the module on those rows, not equivalence in general — run it on your own rows); the compiled law part ran ≈ 14× (flat atoms) and ≈ 120–165× (64 relational atoms) faster than the library's own row-by-row interpreter on 100,000 rows (single runs) — a speedup over the interpreter, not over hand-written code; on two toys (a path tree with "denied under a directory named secret at depth 3 or more", three levels; kinship on forests, depth 1 to 6; five seeds each) the library found every level within 2,556 and 10,817 enumerated programs in all (10,817 the mean over seeds; 10,813–10,819 by seed), where the same search without it found neither top rule in 100,000, with 0 false accepts on larger fresh worlds, but the kinship ladder missed its bar of a tenth of that budget (1.08 times: the universal relation of the enumerator enlarges every level), and met it at 7,889 with the universal relation left out of the terminals (include_universal=False, an opt-in measured as an option, the bar unchanged); a rule the programs cannot express (x and y share a 4-clique) was never admitted on graphs of density 0.3 or 0.5, but on dense graphs (0.7) precision acceptance with held-out graphs of the same density admitted the plain edge relation, which held-out graphs of another density or exact acceptance rejected; on a synthetic five-level path-tree policy (denied if an ancestor directory is named secret and the depth is at least three; the four intermediate relations given as named, unordered targets) `find_rules` with the library found every level within 4,111–4,132 enumerated programs over five seeds; without the library the top level was not found within the cap of 1,000,000 programs, so the budget ratio is at most 0.0042 and its true value unknown; per level the library is not free (90 vs 71 and 2,951 vs 1,898 programs on two of the shared levels; cheaper only on one); two controls — a rule no program of the grammar can express, and one that needs a 56-node program — were never admitted (0 of 10 runs: five seeds, with and without the library); on the kinship ladder (nine rungs up to second cousin once removed) the library found all nine within 3,370–3,378 programs; without it eight of nine — the ninth needs a program of at least 17 nodes where the no-library search exhausted size 15 at ≈ 500,000 programs; 0 false accepts on held-out forests of the same generator at a larger size; relational atoms (an ancestor directory named secret) were measured once on an MCP filesystem server behind a written guard and on a game simulator with written mechanics: on the server they removed the refusals under a secret directory the flat grammar made, but one unrelated refusal remained, so the pre-registered bar of zero refusals was missed; on the game simulator +0.02 points; one layout, one run each; still missing: how deep the search reaches with the twenty-odd relations of a grid window, the library's intermediate concepts found rather than given as targets, a learned proposer for the fallback slot (none ships), and a way to drop proposed atoms no law uses and library entries that make other rules dearer; composition tables: on CLUTRR (Sinha et al., 2019), the clean split trained on chains of 2–3 relations and tested on 2–10; the input is the benchmark's relation graph — the chain of relations between the two people — not the story text; the text-to-graph step is not part of this measurement, a table learned by counting (majority 0.9, support 2, residual fill, all bracketings must agree; 111 cells from 9,074 training chains of length 2–3 in 0.07 s on CPU) gave 0 wrong among 1,087 answered of 1,146 test queries (38–190 per length; at lengths 2 and 3 only 2 and 7 distinct chains): precision among the answered 1.000 (0 of 1,087) at every length 2–10, coverage 0.949 overall, 0.975 at length 10 (116 of 119) and 0.916–0.993 at lengths 4–9; 59 refused (21 disagreeing bracketings, 38 underivable); one split, one run; the most frequent training relation per length answered 0.077 (0.134 at length 10); the benchmark is generated by closed-path composition rules, so an exact table can exist by construction; what is measured is whether counting with refusal recovers it without errors; still missing for them: the sized acceptance with a base-rate null admits an exact 2-vote cell only for a relation with base rate ≤ 0.03 — on a 10% slice it refused 12 of 15 exact support-2 cells the default admits; keep `support=` explicit for deterministic tables (a null of exactness is not built); residual fill after a different table admits different cells and was the source of that arm's extra wrong answers (residual votes are not gated like direct votes); one relation per cell, so a gender-dependent cell is refused, not split (no gender slot), and the opt-in identity relation, with no gender slot, gave 9 wrong answers at lengths 4–10; text frontend (TextFrontend, GraphCheck): on the same split's test file (1,146 queries, chains of 2–10; the table fitted on the whole training file, 111 cells), the graph read from the story text by five sentence patterns gave precision among the answered 0.960 (314 of 327; Wilson 95 % 0.933–0.977; 0.984 / 0.951 / 0.964 at lengths 2–3 / 4–7 / 8–10) at coverage 0.285 (0.526 at length 2, 0.185 at length 10), against 1.000 at 0.949 from the benchmark's relation graph; edges precision 0.957 / recall 0.744 against the benchmark's labels, which mark only the chain's edges and carry some edges reversed (of 229 false stated edges, 173 reversed in the labels, 47 on unlabelled pairs, 9 other); of the 13 wrong answers 9 rest on a reversed label and 4 on an unlabelled pair, none on labelled edges alone; the check dropped 70 stated edges, 50 of them false, and changed no wrong answer (4 right answers lost with them; precision 0.961 without it); 752 of 819 refusals had no path in the read graph; 0.09 ms a story on CPU; one run, the bars pre-registered before it (precision and coverage met; the check's bar — higher precision with it on chains of 4 and more — not met); the five patterns were written in the author's earlier research line while reading this benchmark's test stories, so this measures the shipped frontend on this split, not generalization to unseen phrasings; still missing for it: phrasings beyond the five patterns (a miss is a refusal; a model on the uncovered pairs raised edge recall to about 0.87–0.89 at lower precision in the earlier line and is not shipped — torch, non-commercial data), the owner of «his / her» where the sentence's subject is not the owner, a run with inferred genders only; forward chaining (forward_chain): on 1,816 held-out stories of the same split's training file (a fifth of it by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths, 111 cells), the store's reading of the query pair after chaining to the fixed point was compared with the table's derivation over the shortest paths: 1,675 equal (0.922), 11 disputed where the bracketings disagree (0.006), 130 with nothing derived where the table cannot derive (0.072), 0 differences; the fixed point took 0 rounds for 72 stories, 1 for 826 and 2 for 918; 2.6 s of chaining in all, 1.4 ms a story, on CPU; one run, the test file not read; on 120 random chains of 2–7 relations over 12 random tables whose bracketings can disagree, 0 differences (2 answers equal to the table's rest only on a disputed pair and stay hypotheses until a person answers it); still missing for it: rules other than the composition of two relations over pair facts (no general rules, negation or aggregation); a learned identity relation is chained as an ordinary relation, not as a neutral one; the stories of the training file need at most two rounds, so deeper fixed points are measured by the tests only; rules first, a model second (RulesFirst): on the frozen test split (1,146 queries, chains 2–10) the rules answered 1,087 with 0 wrong and refused 59; the LLM fallback gave a readable option on 50 and was right on 39 (8 empty at the token cap, 1 off the list): 0.983 at coverage 0.992 against 0.949 at 0.949 — the coverage bar of 1.0 missed; solvi-base 2 of 59; replay 0 failures; still missing for it: the same dispatcher over forward_chain, the text setting, gating the model's answer by its confidence, a retry on an empty reply; residual synthesis (synthesize_residual): on the kinship ladder without a library all nine rungs on five seeds (45 of 45; the top rung in 0.40–0.55 s) with 0 false accepts on check worlds, against 20 of 45 for the enumerator at 100,000 behaviours (benchmarks/relations_residual.py); still missing for it: union, intersection, closure and counts (only chains of atoms and converses with ∖ I; two masked segments side by side need a library entry), a use inside find_rules, and a measurement on rules of a real domain; why_not: tested on planted tables only (it names exactly a removed cell), not measured on the refusals of a real split; guarded rules (GuardedRules, chain_rules): on a planted world with an independent oracle (families, genders, homes; every ordered pair labelled) exact set match 1.000 on 4,992 test pairs, all 8 planted rules found, the store path equal to the in-memory fixed point on 1,568 pairs (benchmarks/relations_rules_toy.py); a first run on the held-out part of the benchmark's training file (NoRA, Das et al., NeurIPS 2025, the no-ambiguity configuration; two splits by a hash of the row index, about 9,250 fit / 2,000 check / 1,970 held-out query pairs each; one run per split), exact set match / precision among answered / coverage: the strict setting (majority 1.0, support 10 on the labelled pairs plus two confirmations on the check slice) 0.328 / 0.681 / 0.482 with 469 rules and 0.485 / 0.743 / 0.652 with 436 rules; the default counting setting (majority 0.9, support 2) 0.274 / 0.568 / 0.482 with 720 rules and 0.398 / 0.609 / 0.653 with 838 rules; the most frequent label set per entity kinds 0.335 and 0.341; (benchmarks/relations_rules_nora.py, --salt nora and nora-b). Measured once on the benchmark's three no-ambiguity test files, whole (NoRA, Das et al., NeurIPS 2025; D-na 3,345, OPEC-na 1,683 and BL-na 67,549 query pairs), with the rules learned on a third hash split of the training file (9,238 fit / 2,033 check; the strict setting — majority 1.0, support 10, two confirmations on the check slice — chosen on that split's held-out part, 1,973 pairs, before the test files were read; the bars pre-registered before the run); exact set match / precision among answered / coverage: BL-na (reasoning off the query path) 0.734 / 0.734 / 0.999; D-na (depth beyond the training depth) 0.477 / 0.961 / 0.496 — the rules refuse half of the queries, and the most frequent label set per entity kinds answers 0.491 there, more than the rules by exact match; OPEC-na 0.093 / 0.093 / 0.999; the held-out part of the training file 0.501 / 0.759 / 0.659. The default counting setting (majority 0.9, support 2) gave 0.593 / 0.603 / 0.984 on BL-na, 0.007 / 0.293 / 0.022 on D-na and 0.032 on OPEC-na. The benchmark paper reports, for its best model trained without ambiguity (an Edge Transformer, single-edge), 0.104 on BL-na, 0.822 on D-na and 0.110 on OPEC-na; models trained with ambiguity reach 0.785 on the ambiguous BL split, which was not measured here. The pre-registered bars (BL-na ≥ 0.50 and OPEC-na ≥ 0.20; D-na and held-out ≥ 0.80; precision among answered ≥ 0.95 on every split; the control ≤ 0.30) were not met — none of the four (BL-na's exact match 0.734 clears its 0.50 line, but its precision among answered, 0.734, is below the 0.95 required of every split). What the test shows: labels exist only on the query pairs, so a derived fact cannot be checked where it lands, and the looser admission admits rules right on the labelled pairs and wrong elsewhere (0.593 against 0.734 on BL-na); which chain rules the strict setting keeps depends on the split of the training file (held-out 0.328 / 0.485 / 0.501 on three splits; the test numbers are from the third, one run); the «elsewhere» relation (Not_living_in, about a third of the depth split's rows) is refused by every setting without the declaration, so the depth split is answered on half of its pairs (with the declaration, coverage 0.933 — below); the OPEC split needs a gender guard on the middle entity of a chain (maternal / paternal), which no template carries. Next, each with its own bar: a gender guard on the middle entity, and labels or a structural check on intermediate facts; a closed-world default per entity type beyond a declaration stays unbuilt (the declaration itself is measured on the test files below). The ground truth of this benchmark is computed from published rules, so an exact rule set exists by construction; what is measured is how much of it counting on query-pair labels recovers. The dataset is released under CC BY-NC 2.0: the script reads the files in place; no data and no learned rules ship with the library Since 1.1.1, with the closed world declared (functional={'living_in': 'person'}, the strict setting otherwise unchanged — the same rules), measured on the held-out part of the same two hash splits of the training file (one run per split): 0.337 / 0.684 / 0.492 and 0.674 / 0.798 / 0.844, exact set match / precision among answered / coverage; the rows holding Not_living_in answered exactly: 17 of 590 and 373 of 588 (0 of each without the declaration); answered wrongly on the other rows: 307 against 303 and 301 against 296 without it. The stratum answers only where living_in is known for the subject, and in this benchmark it is never given for a query subject — it is derived through the living_in chain rule, which the strict setting keeps on one split and not on the other; subjects with two derived places are left open. Measured once on the benchmark's three no-ambiguity test files with the declaration (the same strict rules, learned on the third hash split of the training file — 493 rules, unchanged; the bars pre-registered before the run): D-na 0.914 / 0.979 / 0.933 (exact set match / precision among answered / coverage) against 0.477 / 0.961 / 0.496 without it — 1,463 closed answers, all right, every one from a derived home (no query subject has a given one); BL-na 0.734 / 0.734 / 0.999 and OPEC-na 0.093 unchanged row for row; new wrong answers on the rows without «elsewhere» 0 of 70,943; the 170 «elsewhere» rows of D-na left unanswered are subjects with no known home (58) or with two derived ones (112 rows — the declaration is contradicted and the stratum stays silent). With the closed facts visible to the learning rounds (closed_world_in_rounds=True) the rule set changes (499 rules) and 110 right answers are lost on D-na, so the default stays off. The gain hangs on the living_in chain rule, which the strict setting keeps on this split and not on a neutral one (17 of 590 «elsewhere» rows on the held-out part there); the benchmark paper's best model reports 0.822 on D-na in a different setting — trained end to end without the world rules (the declaration is one of them), answering every pair, one relation per pair — so the number is above the paper's, not a like-for-like comparison; the declaration restates a constraint of the benchmark's own world rules (one home per person), so the gain is the caller's knowledge, not a learned rule. (benchmarks/relations_rules_nora.py --settings strict-closed) On the planted world with only the positive templates (no «elsewhere» rule and no negative stratum to learn) the declaration answers 340 test pairs, all right: exact 0.983 against 0.915 without it (benchmarks/relations_rules_toy.py).; still missing for them: labels exist only on the query pairs, so a derived fact cannot be checked where it lands, and counting at the looser setting admits rules right on the labelled pairs and wrong elsewhere (the joint recount after each round keeps the set from collapsing); the stricter setting does better here, but which chain rules it keeps depends on the split (0.328 against 0.485: whether living_in ← living_in_same_place ∘ living_in survives the strict share on derived facts); the «elsewhere» relation (Not_living_in, about a third of the rows) is refused by every setting without a declaration — a closed-world default per entity type was not built in 1.1.0 (since 1.1.1: functional=); residual votes only bring forward what the next round's fixed point shows anyway (identical rule sets on the planted world); two binary atoms per body, unary predicates only as given (none learned or invented), and a store path that writes every justification (slow on dense stories).
- **Script:** `benchmarks/relations_battery.py` (the battery; the rule file, not shipped, is passed with `--rules`),
  `benchmarks/relations_compile.py` (compiled rules on a forest of 100,000 nodes, on worlds of 1,000 nodes and on
  100,000 law rows; `pytest -m slow tests/test_relations_gate.py`) and `benchmarks/relations_library_toy.py` (a path
  tree and kinship on forests, with and without the library; `pytest -m slow tests/test_relations_library_toy.py`);
  composition tables: `tests/test_relations_table.py`; forward chaining: `benchmarks/relations_chain.py` (the
  training file of the split, passed with `--data`) and `tests/test_relations_chain.py`; residual synthesis:
  `benchmarks/relations_residual.py` and `tests/test_relations_residual.py`; why not derived:
  `tests/test_relations_why_not.py`; text frontend: `tests/test_relations_text.py`; guarded rules: `benchmarks/relations_rules_toy.py`,
  `benchmarks/relations_rules_nora.py` (the training file, passed with `--data`) and `tests/test_relations_rules.py`.
  **Deadline:** 1.2.
- Guide: [Relational rules](best_practices.md#relational-rules-enumerate-let-the-test-judge-then-compile),
  [rules over relations with a library](best_practices.md#rules-over-relations-enumerate-with-a-library-rank-only-where-a-check-is-expensive);
  [composition tables](best_practices.md#learn-composition-tables-by-exact-counting-derive-by-composition-refuse-what-the-table-cannot-derive),
  [forward chaining](best_practices.md#derive-to-a-fixed-point-let-disputes-refuse),
  [guarded rules](best_practices.md#rules-with-guards-and-a-negative-stratum-count-on-labelled-pairs-confirm-on-a-check-slice-recount-after-each-round),
  [rules first, a model second](best_practices.md#rules-first-a-model-second),
  [residuals first](best_practices.md#synthesise-chains-by-residuals-first-enumerate-second),
  [why not derived](best_practices.md#ask-why-a-rule-did-not-derive-and-learn-what-is-missing);
  [text frontend](best_practices.md#read-the-graph-with-code-where-the-phrasing-is-regular-refuse-the-rest);
  API: [solvi.experimental.relations](api/relations.md).

## `skills`

Learning in a world described by code (`solvi.experimental.skills`): System 2 plans on an exact model of the world (a
copy of the environment; the real one is never stepped by the planner), a plan counts only after the code re-simulated
it, verified plans are compiled into System 1 skills ("when these atoms hold, for this goal, do this sequence"; the
conditions keep the atoms every verified start shared), and a skill acts alone only while its outcomes keep an
anytime-valid certificate — before that System 2 re-verifies it at every use, and an alarm on its alone outcomes
retires it. `SkillLearner` runs the loop with a record per decision and per episode; `as_s2(learner)` makes the planner
System 2 of `solvi.Agent`; `SkillSet.to_knowledge` writes the skills into the knowledge store, and each episode
records its search cost. Since 1.1. Since 1.1.1, footprints with variables: `lift="lattice"` with a
`Vocabulary` of the world widens each new skill's footprint (a constant to a variable — «move D, do» —, a cell to «a
resource within 2 steps», atoms dropped), most general first, and keeps a wider candidate only when the plan,
instantiated on verification states that match it but not the original (the stream's recorded states and the world's
perturbations), reaches its goal on the exact copy; the skill records both footprints and the verification count, and
`replay` re-runs the cases.

- **Missing:** on a deterministic gridworld (20 fresh layouts, five seeds) System 1's share reached 0.81 in the last five episodes with every goal reached, as when System 2 plans every goal, and 1.5% of System 1's decisions were in failed runs under a 20% promise pooled over the skills, while replayed cases never transferred and plans compiled without verification reached no goal. On Crafter (version 1.8.3; written mechanics as the exact model; symbolic state; hand-written survival reflexes shared by every arm, 60% of all steps; the planner reads the copy's full map while the game shows an agent a 9×7 window), in one run of 10 streams × 30 episodes, no agent seed (the loop is deterministic given the world seeds). Achievements are unique achievements unlocked per episode (of 22). The skill set was certified at 5% / δ 0.1 (fifty clean uses, one e-process pooled over the stream's skills) in every stream by episode 1–2. In the last ten episodes System 1 acted alone 1,350 times with an observed outcome and failed twice (0.15%, Wilson upper bound 0.54%; 4 of 2,882 over the whole run); 57 further alone runs were paused by a reflex and not resumed, so they carry no outcome (counted as failures they would make 4.2%, Wilson 5.4%). The promise is the set's, not a single skill's; a skill's own alarm retires it (4 retirements, each later re-certified). Goals that are stochastic by the written rules stayed with System 2 and are outside the promise. With skills the planner was called 32 times per episode against 42 when it plans every goal (−23%), and an episode took 84 s against 107 s on the same machine (−21%), with achievements kept: 15.2 against 15.7 unique achievements unlocked per episode of 22 (ratio 0.966, bootstrap lower bound 0.934). System 1's share of the non-reflex steps rose from 7% in the first ten episodes to 23% in the last ten (9% of all steps); the pre-registered bar of 60% was missed. Routes to a resource and plans for unseen footprints stay System 2 by design (routes were 25% of non-reflex steps), and every certified skill was a one- or two-step gather sequence with a constant footprint: in an open world the planner did most of the work and skills took the repeatable part. Hierarchical skills are untested; conditions with variables were measured in 1.1.1 (below). An arm that executed and compiled random action sequences of the plan's length in place of verified plans unlocked 4.0 unique achievements per episode against 13.7 for the same agent with no memory at all; none of its 5,497 unverified skills was ever certified, so System 1 never acted. The gain is the verified plans', not the loop's bookkeeping, and the certificate held unverified skills out; what unverified memory does on its own was not measured. The observation-only learner (the agent's own action memory with the written gates on) unlocked 13.3 unique achievements per episode, no higher than the same agent without memory (13.7); the planner arms' lead over both is the planner's with the full map. The planner reads the whole map of the copy; a Crafter agent sees a 9×7 window from pixels. No number here is comparable with published Crafter scores (a geometric mean of success rates over 1M training steps). With footprints with variables (lift="lattice"; one run of 10 streams × 30 episodes in the same open-world setting, the share of non-reflex steps): System 1's share 0.343 / 0.486 / 0.385 by thirds against 0.071 / 0.175 / 0.233 with constant footprints (the pre-registered bar — at least 0.40 in the last third with a rise of 15 points — missed: 0.385 and a rise of 0.041 [−0.096; 0.173]); 886 of 1,032 lattice runs accepted (0.859, a verdict on a sample of up to 16 cases) and 99.4 % of System 1's steps by skills with variables; 10 failures in 1,976 alone decisions with an outcome (0.51 %, Wilson upper 0.93 %), none without a certificate; achievements 14.43 per episode against 15.72 when System 2 plans every goal (0.918, lower bound 0.861); planner calls per episode 11.8 against 28.1 with constant footprints and 43.2 planning always; the pooled certificate's 23 alarms retired 23 skills and episodes started without a certificate in force numbered 38 / 19 / 43 by thirds — with it in force System 1 took 0.50–0.58 in every third (a plausible cause of the last third's fall, not tested by a separate arm). Still missing: hierarchical skills, a certificate pool per skill or per goal (the pooled certificate's alarms retired skills with variables), a world where observation is expensive, a service with a specification, a condition rule that specializes after a failure, and a planner with a heuristic; the search-cost record is tested on a synthetic corridor only, and skills are not admitted by discriminating probes (laws are: LawSet.admit_by_probes).
- **Script:** `benchmarks/skills_toy.py` (the gridworld with four arms: skills, plan always, cases, scrambled; `pytest -m
  slow tests/test_skills_toy.py`). **Deadline:** 1.2.
- Guide: [Learning in a world described by code](best_practices.md#learning-in-a-world-described-by-code-plan-on-the-code);
  API: [solvi.experimental.skills](api/skills.md).

## `ood`

A watch next to `System.calibrate`'s default Platt scaling (`watch=ShiftWatch()`), packaging a known method: the
black-box shift test of Rabanser, Günnemann and Lipton (*Failing Loudly*, NeurIPS 2019) — a per-feature two-sample
Kolmogorov–Smirnov test with Bonferroni on a classifier's outputs — applied to three label-free features of each answer
(the raw confidence, the margin between the two best options, the options' normalized entropy). `shift(batch)` says
whether a batch's answer features differ from the calibration answers'; "out of domain" in `why` marks one answer far
from them, a max-softmax style score (Hendrycks and Gimpel, ICLR 2017) as a Mahalanobis distance above the reference's
own 1 − alpha quantile. Alpha is nominal (asymptotic Kolmogorov-Smirnov p-values, three Bonferroni tests). Since 1.2.

What it sees is the decider's confidence profile, not the content: it reports "the answers' features differ from the
reference", not "the inputs are out of domain". Intents the decider has never seen, phrased like the calibration ones,
mostly pass — for inputs from outside the calibration set use `OpenSetGate`
([solvi.core.guarantees.openset](api/openset.md)); for a fall of the share answered alone and a change of the answer
mix use `DriftMonitor`. A rephrasing of the same domain can be flagged like another domain. The per-answer note catches
few of another dataset's answers: it is a note on one answer, not a detector — test batches with `shift`, of about 200
answers (50-answer batches missed about a quarter of the shifts below). Confidence can rise under a shift as well as
fall, so a monitor that only watches for a drop misses it.

- **Missing:** measured once on intent classification with solvi-base (raw confidences, next to Platt calibration; alpha 0.01; 10 directed pairs of CLINC150, Banking77, HWU64 and Bitext customer support): with 200-answer batches (shift), 0.99–1.0 of the batches from another dataset flagged on every pair and at most 0.05 of in-domain batches (one domain at that bar, 0.03–0.05 over two draws of the windows; 0.01 or less for the others); with 50-answer batches, a mean of 0.76 and 0.30–0.48 on four pairs, +0.18 [0.05, 0.34] over a dev-tuned two-sided mean-confidence test at the watch's own in-domain rate of about 1 % (worse on one pair) and on par with it at 5 %. On half of the pairs the mean confidence rose, which a test for a fall of confidence cannot see. Held-out intents of the same dataset were mostly not flagged (CLINC150, Banking77: under 1 % of batches), while a rephrasing of the same domain (Banking77 and Bitext retail banking) was flagged like another dataset; the per-answer note marked on average 10 % of another dataset's answers. All four datasets are in or near the decider's training mix and their option sets differ (20 of 27–150 labels); still missing: a benchmark script in benchmarks/ and a decider that has not seen the data
- **Script:** none yet. **Deadline:** 1.2.
- API: [solvi.experimental.ood](api/ood.md). The note is a warning, not a guarantee: a regime whose answers look like the
  calibration answers but whose errors differ passes every check; only labels show it.

## `audit`

A check that is not run on every decision (an expensive second opinion, a person's look) can break without anyone
seeing it: the decisions it is bought for are the unsure ones, where it disagrees anyway. `AuditWatch` runs it on a
random share of decisions as well (`Dispatcher(audit=)` marks them) and follows its failure rate on that share with a
Bernoulli CUSUM whose threshold is simulated with the audit share, the horizon and the reference's own sampling error.
Since 1.2.

This is a known method, packaged: inspection of a random fraction of a running stream is Dodge's continuous sampling
plan (Ann. Math. Stat. 1943), a CUSUM on each inspected 0/1 outcome is the Bernoulli CUSUM of Reynolds and Stoumbos
(J. Qual. Tech. 1999), and `odds=2` is the log-likelihood-ratio tuning to an odds ratio of 2 of Steiner, Cook,
Farewell and Treasure (Biostatistics 2000). Labels seen only where a policy chose to look are the selective-labels
problem; a uniformly random slice beside the policy is the standard remedy.

Measured once on an intent-routing stream (100 replays, a check whose failure rate rises from 0.20 to 0.60 at one
point, alpha 0.05 per 600 decisions): a 10 % audit flagged the break in 96 of 100 within 300 decisions (median 118; a
20 % audit 98 of 100, median 75) with 4 of 100 false flags on unchanged streams. The same CUSUM on only the decisions
the policy bought the check for: 57 of 100 (median 176), 9 false flags. The drift monitor on the system's own answers
cannot see it by design, since the system did not change. The library's `NumericDrift` on the same audited values
flags sooner (24 decisions on average) but with 14 of 100 false flags: a Gaussian detector on 0/1 values, not
level-matched, and a level-matched comparison was not run. AuditWatch's merit here is that it held its false-alarm level (4 of 100 at alpha 0.05).
Condition: the reference was the check run on 800 random decisions, all checked; with only the audited share of a
reference (about 80 items) the threshold widens and detection is slower than measured here.
One task family (intent routing, a TF-IDF + logistic-regression router, a nearest-neighbour second opinion) and one
break size (odds × 6 while the CUSUM is tuned to × 2; a doubling will be slower). A break confined to the decisions
the policy never buys was not planted, so "sees what the bought check cannot" is untested. A flag says the check
disagrees with the system more often, not which of them broke.

- **Missing:** one task family and one break size measured (docs/experimental.md); a smaller break, a short
  reference and a break confined to the decisions the policy never checks are unmeasured
- **Script:** none yet. **Deadline:** 1.2.
- API: `solvi.experimental.audit` (`AuditWatch`).

## Graduated and removed

- **Graduated in 1.1:** `counterfactual` — counterfactual explanations are the stable `solvi.core.explain`
  ([guide](guide.md#explaining-a-decision-the-smallest-change-that-flips-it)), with the cost of a text change a
  parameter (`text_cost=`, `costs=`); `solvi.experimental.counterfactual` re-exports it, with this warning, until 1.2.
- **Graduated in 1.1:** `mcp` — the MCP proxy with a guard is `solvi.serve`'s (`run_proxy`, `Proxy`, `Upstream`;
  [guide](guide.md#an-mcp-proxy)); `solvi.experimental.mcp` re-exports it, with this warning, until 1.2.
- **Graduated in 1.1:** `specialist` — the specialist contract is the stable `solvi.core.specialist`, its runs stored
  as records of kind "specialist" in a TraceStorage ([guide](guide.md#specialists-a-model-proposes-code-checks-the-store-keeps-the-run));
  `solvi.experimental.specialist` re-exports it, with this warning, until 1.2.
- **Graduated in 1.1:** document versions (out of `compile`) — `Spec`, `Versions`, `decision_diff`, `split_clauses`,
  `build_catalog` and `CatalogVersion` are the stable `solvi.core.versions`; `solvi.experimental.compile` re-exports
  them, as the same objects, until 1.2.
- **1.1:** `oncalib` (recalibrating a guarantee on the fly from outcomes: it broke the promise — recalibrate explicitly,
  `System.guarantee(..., corrections=True)`, on a schedule or after a drift flag) and `learning` (the learning loop: no
  measurable gain on streams of one kind of decision; a labelled update is now a candidate change of
  `solvi.experimental.changes`, `Change(artifact="update")` with `LearningTarget`). The changelog's 1.1 "Removed" table
  says why and what to use instead.

## Tried and left out

Some ideas were measured and did not make it into solvi, experimental or not; the
[changelog](../CHANGELOG.md#tried-and-left-out) lists them with what was measured.
