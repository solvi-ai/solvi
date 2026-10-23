"""solvi.experimental — pieces that work and are tested but have not yet shown a measured gain (or a measured use) that
the rest of solvi promises. Their API may change in any release, and each one either graduates or is removed by its
deadline.

    from solvi.experimental import STATUS
    STATUS["lora"]       # {"since": "0.7", "missing": "...", "script": None, "deadline": "1.2"}

What marks them:

- importing one warns once per process (`solvi.ExperimentalWarning`, a UserWarning: silence it with
  `warnings.filterwarnings("ignore", category=solvi.ExperimentalWarning)`);
- a decision made by a System that uses one records it: `meta["experimental"]` in the stored decision (the names, e.g.
  `["lora"]`), and the system report counts them;
- nothing stable imports them, and the top-level `solvi` package does not re-export them. One command-line entry loads
  one on request: `solvi hook` (hooks).

Graduating needs a pre-registered bar, a script in `benchmarks/` that measures it, the bar met and reviewed, no import
of anything experimental, a conformance check and a docs move; the old path is then kept for one release. A piece that
does not graduate by its deadline (two minor releases after 1.0), or that measures negative, is removed and the negative
result published.

This package itself imports none of them: `import solvi.experimental` does not warn."""
from __future__ import annotations

DEADLINE = "1.2"

STATUS: dict[str, dict] = {
    "lora": {"since": "0.7", "what": "a LoRA adapter on the decider for one question (adapt_lora)", "missing": "a "
             "decision on real use, a benchmark script in the repo, and wiring as the adapter rung of an update "
             "(solvi.experimental.changes); measured gains over fit come with overconfidence", "script": None},
    "compile": {"since": "0.9", "what": "a written specification compiled by an LLM into catalog parts, accepted when "
                "two drafts agree and the specification's tests pass (with its sandbox)", "missing": "measured once on 15 classes of small pure functions over typed slots (a search over a bank of 137 operations, |H| up to 6.5·10⁸ programs of depth 2, N = 314–452 fresh examples; a small hosted LLM writer at N = 60, two drafts): no accepted function was wrong on 200 hidden examples per class (21 accepted), the search found 6 of 15 classes and the writer 15 of 15 — but neither proposer produced a false candidate, so whether the sized certificate refuses what a five-example certificate accepts was not shown on that setup; it is shown only on a planted constant in a smoke test; its necessity on a live stream of drafts is not shown; still missing: acceptance on larger specifications and a benchmark script in benchmarks/", "script": None},
    "hooks": {"since": "0.7.1", "what": "`solvi hook`: a coding agent's pre-edit rule checks and skill picker",
              "missing": "false deny and ask rates on real coding-agent sessions, with a script", "script": None},
    "charts": {"since": "0.7", "what": "verified charts: every number quoted from the text, a deterministic SVG",
               "missing": "Measured on 50 paragraphs of US federal news releases against a hand reading written before any "
               "proposer ran (`benchmarks/charts_real_texts.py`, one run): the rule-based proposer drew a chart for "
               "50 of 50 texts with every rendered value re-verified against the text (350 of 350 chart numbers), "
               "but only 33.3% of the drawn points agreed with the hand reading in label and unit (the bar was 95%: "
               "labels 39, units 82 of 99 points); the LLM proposer (gpt-oss-120b) drew a chart for 56% of the texts"
               " and agreed on 94.2% of the points, with 5 of 485 chart numbers not in the text (a sign the proposer"
               " added); a changed digit and swapped values were caught 300 of 300 times by both, a dropped point 74"
               " (rules) and 109 (LLM) of 300; every run replays byte for byte (50 of 50). Still missing: a proposer"
               " whose labels and units match a reader's on real texts",
               "script": "benchmarks/charts_real_texts.py"},
    "changes": {"since": "1.1", "what": "one change pipeline for four sources (a document version, verified "
                "experience, a person's correction, a search proposal): pluggable stages, a journal of kind \"change\", "
                "promote into the knowledge store / compiled versions / a labelled update, versions and rollback", "missing": "on a "
                "credit policy the pipeline carried a document version (144 of 144 changed decisions traced to their "
                "clause), admitted 12 of 12 exact exceptions, held every widening one for a person and refused 8 of 8 "
                "exceptions that touched other decisions, 0 extraneous changes, 14 versions replayed; on a bank-request "
                "stream with a person's labels every one of 22 candidates passed every stage — the run with the gates equals the run without them — so the stages added nothing there — the module is journal, impact analysis, approval levels and "
                "rollback; still missing: narrows / widens proved over rule code outside a Python subset (calls, "
                "subscripts, loops: measured on probes instead), a regression-scenario stage, and a script for the bank-request stream (the credit numbers: benchmarks/changes_credit.py)",
                "script": "benchmarks/changes_credit.py"},
    "laws": {"since": "1.1", "what": "laws of an environment from prediction mismatches: a grammar of atoms you "
             "write, candidates from each mismatch, acceptance only by a held-out test on other groups (worlds, "
             "sessions, paths), most specific first, retirement on falling precision, a probe planner for live "
             "systems, export into the knowledge store; acceptance sized by the candidates tested, admission by "
             "discriminating probes, abstention where laws disagree, a search-cost record",
             "missing": "on a game simulator with written mechanics "
             "the laws closed 94% of the gap between a learned map and the written mechanics (20 fresh worlds, 4,731 transitions, one run; 0.892 → 0.988, ceiling 0.994), but random candidates under the same test came within 1.1 points — the measurement's pre-registered bar for that control (random candidates at most +2 points over the base) was missed by +8.4 — so the mismatch is "
             "a compact proposal heuristic and the test does the work; on a real MCP filesystem server with a path "
             "policy, probes planned from mismatches took refusals from 6 to 0 in one session (200 calls per session, one run; random probes: 48 to 29); still missing: a grammar too large to enumerate, retirement of weak laws nothing contradicts, "
             "wiring into a System's decisions, and a script for the two measurements (the toys: benchmarks/laws_toy.py); a "
             "ranker of candidates for expensive checks (RankedProposer, a bag-of-atoms model trained on tasks drawn "
             "from the grammar) cut the checks to the true condition from 69.7 in the blind order to 2.1 on synthetic "
             "tasks of one grammar, in distribution only; ranking by the joint statistic of the whole condition over the "
             "logged rows (JointRanker, no training; with no rows it is the blind order — the statistic needs rows) "
             "needed 1.00 / 17 / 1.01 checks (medians 1 / 1 / 1) on three synthetic settings of the library's own "
             "grammars (five seeds × 60 tasks) against 85 / 2,182 / 126 in the blind order (85 a different draw of tasks from the 69.7 above: five seeds against one) and 2.1 / 591 / 7.7 for "
             "the trained bag of atoms, a shuffled-rows control worse than the blind order (112 / 2,434 / 166), the "
             "gain on the deep setting showing from about 20 logged rows and full by 300, the same law accepted "
             "under every order in 894 of 900 tasks; "
             "before a buffer has rows, a prior order over the condition's own features learned from synthetic "
             "tasks of the task family (PriorRanker, which falls back to the blind order unless its held-out "
             "checks are 5% below the blind order's) needed 208 checks at 0 rows against 1,245 blind on a skewed "
             "family (deep conditions whose features are drawn unevenly by a chosen skew; 5,112 candidates per task; "
             "five seeds, 300 tasks, training and test tasks from the same sampler) and needed 0.83–0.93 of blind on the three uniform settings (on the first it fell back in 2 of 5 seeds); HandoverRanker blends the two — the joint Jaccard "
             "plus 20 / (20 + rows) times the prior's rank in [0, 1] — and on the skewed family needed 184 / 98.6 / 8.9 "
             "checks at 20 / 100 / 300 rows against 208 for the prior and 1,019 / 361 / 16.8 for the joint "
             "statistic, but on the deep uniform setting at 300 rows 27.2 against 17.2 (111 of 300 tasks worse, 31 "
             "better, medians 2 and 1; the pre-registered 5 % bar missed there; a smaller n0 not measured); "
             "measured once on a live file "
             "server (8 mismatches; "
             "candidates contradicted by logged rows skipped free): every order spent 29 probes in all, because 1–5 "
             "candidates per mismatch were left to probe and the true conditions were single atoms — the shortest "
             "hypotheses, which the length-first order meets first — so ranking can pay only where many consistent "
             "hypotheses remain after the free checks, the control first; the acceptance sized by the candidates "
             "tested on a buffer (on a synthetic buffer with no law the fixed thresholds accepted a law in 100 of 100 "
             "buffers, the sized rule in 0 of 100), admission by discriminating probes (on a synthetic world at 8 "
             "checks a random held-out sample accepted a wrong law in 0.593 of the trials, the probes in none) and "
             "abstention where laws that are not exceptions of one another disagree are measured on synthetic worlds "
             "only, not yet on the game simulator or the file server; the live probe-based learner can admit false "
             "laws on policies the grammar cannot express (observed once per control policy) — admission by probes or "
             "sized acceptance is the open item", "script": "benchmarks/laws_toy.py"},
    "relations": {"since": "1.1", "what": "relations over a state and rules over them: packed-bit relations, the "
                  "relation algebra (composition, converse, closure, counts), a bottom-up enumerator with "
                  "observational equivalence on the full relations, a library of accepted programs, compilation into "
                  "set code or index arrays with a differential check, encoders (a grid window seen from the player, "
                  "a path tree) and a proposer of relational atoms for the laws' held-out test; composites by "
                  "enumeration with a library (find_rules admits only what passes a held-out test and builds the next "
                  "rules on the admitted ones) and a fallback slot for a learned proposer; composition tables of relations "
                  "learned by exact counting (residual fill, derivation over every bracketing, refusal of what the "
                  "table cannot derive); forward chaining to a fixed point over the composition rules in the knowledge store (derived facts with their premises and the rule as their justification, two derived relations of one pair a dispute); rules first, a model second (the table answers what it derives, a fallback model what it refuses, every answer replayable without the model); a text frontend for kinship stories (five sentence patterns read the story into a relation graph with inverse edges and gender facts, a global check drops contradictions, a pattern miss is a refusal); "
                  "top-down synthesis of chain programs through residuals of the target (synthesize_residual, "
                  "synthesize(method=\"residual-first\")); why a relation was not derived (why_not: the missing cell "
                  "or fact, ranked, as an agenda goal); guarded rules over pair facts with unary guards and a negative stratum, learned by counting on labelled pairs, applied to a fixed point (in memory or through the knowledge store)",
                  "missing": "on a pre-registered battery of 28 random rules over three relations (the rule file was written in the author's earlier research line before any search ran and is not shipped — `relations_battery.py --rules` takes it; the bars were fixed from that line's reference run, so this is a port-fidelity check, one deterministic run) the enumerator found 22 within "
                  "seven nodes, every find exact on larger graphs of other families, and compiled rules on a forest "
                  "of 100,000 nodes were equal pair for pair; the index-array backend within 0.7–1.3× of hand code "
                  "over a parent array (output conversion excluded); the plain set-code backend 2–3× slower than "
                  "hand-written set code over the same pairs (benchmarks/relations_compile.py), so compiled relation "
                  "programs are not faster than "
                  "hand-written set code on graphs (2–3× slower); on five law sets learned on a game simulator with "
                  "written mechanics (61–103 laws each; one with 64 relational atoms over a 3×3 window compiled as a "
                  "bounded domain), `LawSet.to_module` + `check_compiled` agreed with `LawSet.predict` on every one "
                  "of 308,510 logged transitions (training, validation and test worlds), no atom refused (a "
                  "differential check on logged rows certifies the module on those rows, not equivalence in general — "
                  "run it on your own rows); the compiled law part ran ≈ 14× (flat atoms) and ≈ 120–165× (64 "
                  "relational atoms) faster than the library's own row-by-row interpreter on 100,000 rows (single "
                  "runs) — a speedup over the interpreter, not over hand-written code; on two toys (a path tree with "
                  "\"denied under a directory named secret at depth 3 or more\", three levels; kinship on forests, "
                  "depth 1 to 6; five seeds each) the library found every level within 2,556 and 10,817 enumerated programs in all (10,817 the mean over seeds; 10,813–10,819 by seed), where the same search without it found neither top rule in 100,000, with 0 false "
                  "accepts on larger fresh worlds, but the kinship ladder missed its bar of a tenth of that budget "
                  "(1.08 times: the universal relation of the enumerator enlarges every level), and met it at 7,889 "
                  "with the universal relation left out of the terminals (include_universal=False, an opt-in measured "
                  "as an option, the bar unchanged); a rule the programs cannot express (x and y share a 4-clique) "
                  "was never admitted on graphs of density 0.3 or 0.5, but on dense graphs (0.7) precision acceptance "
                  "with held-out graphs of the same density admitted the plain edge relation, which held-out graphs "
                  "of another density or exact acceptance rejected; on a synthetic five-level path-tree policy "
                  "(denied if an ancestor directory is named secret and the depth is at least three; the four "
                  "intermediate relations given as named, unordered targets) `find_rules` with the library found "
                  "every level within 4,111–4,132 enumerated programs over five seeds; without the library the top "
                  "level was not found within the cap of 1,000,000 programs, so the budget ratio is at most 0.0042 "
                  "and its true value unknown; per level the library is not free (90 vs 71 and 2,951 vs 1,898 "
                  "programs on two of the shared levels; cheaper only on one); two controls — a rule no program of "
                  "the grammar can express, and one that needs a 56-node program — were never admitted (0 of 10 runs: five seeds, with and without the library); on the kinship ladder (nine rungs up to second cousin once removed) the library found all "
                  "nine within 3,370–3,378 programs; without it eight of nine — the ninth needs a program of at least "
                  "17 nodes where the no-library search exhausted size 15 at ≈ 500,000 programs; 0 false accepts on "
                  "held-out forests of the same generator at a larger size; relational atoms (an ancestor directory "
                  "named secret) were measured once on an MCP filesystem server behind a written guard and on a game "
                  "simulator with written mechanics: on the server they removed the refusals under a secret directory "
                  "the flat grammar made, but one unrelated refusal remained, so the pre-registered bar of zero "
                  "refusals was missed; on the game simulator +0.02 points; one layout, one run each; still missing: "
                  "how deep the search reaches with the twenty-odd relations of a grid window, the library's "
                  "intermediate concepts found rather than given as targets, a learned proposer for the fallback slot "
                  "(none ships), and a way to drop proposed atoms no law uses and library entries that make other "
                  "rules dearer; composition tables: on CLUTRR (Sinha et al., 2019), the clean split trained on "
                  "chains of 2–3 relations and tested on 2–10; the input is the benchmark's relation graph — the "
                  "chain of relations between the two people — not the story text; the text-to-graph step is not part "
                  "of this measurement, a table learned by counting (majority 0.9, support 2, residual fill, all "
                  "bracketings must agree; 111 cells from 9,074 training chains of length 2–3 in 0.07 s on CPU) gave "
                  "0 wrong among 1,087 answered of 1,146 test queries (38–190 per length; at lengths 2 and 3 only 2 "
                  "and 7 distinct chains): precision among the answered 1.000 (0 of 1,087) at every length 2–10, "
                  "coverage 0.949 overall, 0.975 at length 10 (116 of 119) and 0.916–0.993 at lengths 4–9; 59 refused "
                  "(21 disagreeing bracketings, 38 underivable); one split, one run; the most frequent training "
                  "relation per length answered 0.077 (0.134 at length 10); the benchmark is generated by closed-path "
                  "composition rules, so an exact table can exist by construction; what is measured is whether "
                  "counting with refusal recovers it without errors; still missing for them: the sized acceptance "
                  "with a base-rate null admits an exact 2-vote cell only for a relation with base rate ≤ 0.03 — on a "
                  "10% slice it refused 12 of 15 exact support-2 cells the default admits; keep `support=` explicit "
                  "for deterministic tables (a null of exactness is not built); residual fill after a different table "
                  "admits different cells and was the source of that arm's extra wrong answers (residual votes are "
                  "not gated like direct votes); one relation per cell, so a gender-dependent cell is refused, not "
                  "split (no gender slot), and the opt-in identity relation, with no gender slot, gave 9 wrong "
                  "answers at lengths 4–10; text frontend (TextFrontend, GraphCheck): on the same split's test file (1,146 queries, chains of 2–10; the table fitted on the whole training file, 111 cells), the graph read from the story text by five sentence patterns gave precision among the answered 0.960 (314 of 327; Wilson 95 % 0.933–0.977; 0.984 / 0.951 / 0.964 at lengths 2–3 / 4–7 / 8–10) at coverage 0.285 (0.526 at length 2, 0.185 at length 10), against 1.000 at 0.949 from the benchmark's relation graph; edges precision 0.957 / recall 0.744 against the benchmark's labels, which mark only the chain's edges and carry some edges reversed (of 229 false stated edges, 173 reversed in the labels, 47 on unlabelled pairs, 9 other); of the 13 wrong answers 9 rest on a reversed label and 4 on an unlabelled pair, none on labelled edges alone; the check dropped 70 stated edges, 50 of them false, and changed no wrong answer (4 right answers lost with them; precision 0.961 without it); 752 of 819 refusals had no path in the read graph; 0.09 ms a story on CPU; one run, the bars pre-registered before it (precision and coverage met; the check's bar — higher precision with it on chains of 4 and more — not met); the five patterns were written in the author's earlier research line while reading this benchmark's test stories, so this measures the shipped frontend on this split, not generalization to unseen phrasings; still missing for it: phrasings beyond the five patterns (a miss is a refusal; a model on the uncovered pairs raised edge recall to about 0.87–0.89 at lower precision in the earlier line and is not shipped — torch, non-commercial data), the owner of «his / her» where the sentence's subject is not the owner, a run with inferred genders only; "
                  "forward chaining (forward_chain): on 1,816 held-out stories of the same split's training file (a fifth of it by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths, 111 cells), the store's reading of the query pair after chaining to the fixed point was compared with the table's derivation over the shortest paths: 1,675 equal (0.922), 11 disputed where the bracketings disagree (0.006), 130 with nothing derived where the table cannot derive (0.072), 0 differences; the fixed point took 0 rounds for 72 stories, 1 for 826 and 2 for 918; 2.6 s of chaining in all, 1.4 ms a story, on CPU; one run, the test file not read; on 120 random chains of 2–7 relations over 12 random tables whose bracketings can disagree, 0 differences (2 answers equal to the table's rest only on a disputed pair and stay hypotheses until a person answers it); still missing for it: rules other than the composition of two relations over pair facts (no general rules, negation or aggregation); a learned identity relation is chained as an ordinary relation, not as a neutral one; the stories of the training file need at most two rounds, so deeper fixed points are measured by the tests only; rules first, a model second (RulesFirst): on the frozen test split (1,146 queries, chains 2–10) the rules answered 1,087 with 0 wrong and refused 59; the LLM fallback gave a readable option on 50 and was right on 39 (8 empty at the token cap, 1 off the list): 0.983 at coverage 0.992 against 0.949 at 0.949 — the coverage bar of 1.0 missed; solvi-base 2 of 59; replay 0 failures; still missing for it: the same dispatcher over forward_chain, the text setting, gating the model's answer by its confidence, a retry on an empty reply; residual synthesis (synthesize_residual): on the kinship ladder without a library all nine rungs on five seeds (45 of 45; the top rung in 0.40–0.55 s) with 0 false accepts on check worlds, against 20 of 45 for the enumerator at 100,000 behaviours (benchmarks/relations_residual.py); still missing for it: union, intersection, closure and counts (only chains of atoms and converses with ∖ I; two masked segments side by side need a library entry), a use inside find_rules, and a measurement on rules of a real domain; why_not: tested on planted tables only (it names exactly a removed cell), not measured on the refusals of a real split; guarded rules (GuardedRules, chain_rules): on a planted world with an independent oracle (families, genders, homes; every ordered pair labelled) exact set match 1.000 on 4,992 test pairs, all 8 planted rules found, the store path equal to the in-memory fixed point on 1,568 pairs (benchmarks/relations_rules_toy.py); a first run on the held-out part of the benchmark's training file (NoRA, Das et al., NeurIPS 2025, the no-ambiguity configuration; two splits by a hash of the row index, about 9,250 fit / 2,000 check / 1,970 held-out query pairs each; one run per split), exact set match / precision among answered / coverage: the strict setting (majority 1.0, support 10 on the labelled pairs plus two confirmations on the check slice) 0.328 / 0.681 / 0.482 with 469 rules and 0.485 / 0.743 / 0.652 with 436 rules; the default counting setting (majority 0.9, support 2) 0.274 / 0.568 / 0.482 with 720 rules and 0.398 / 0.609 / 0.653 with 838 rules; the most frequent label set per entity kinds 0.335 and 0.341; (benchmarks/relations_rules_nora.py, --salt nora and nora-b). Measured once on the benchmark's three no-ambiguity test files, whole (NoRA, Das et al., NeurIPS 2025; D-na 3,345, OPEC-na 1,683 and BL-na 67,549 query pairs), with the rules learned on a third hash split of the training file (9,238 fit / 2,033 check; the strict setting — majority 1.0, support 10, two confirmations on the check slice — chosen on that split's held-out part, 1,973 pairs, before the test files were read; the bars pre-registered before the run); exact set match / precision among answered / coverage: BL-na (reasoning off the query path) 0.734 / 0.734 / 0.999; D-na (depth beyond the training depth) 0.477 / 0.961 / 0.496 — the rules refuse half of the queries, and the most frequent label set per entity kinds answers 0.491 there, more than the rules by exact match; OPEC-na 0.093 / 0.093 / 0.999; the held-out part of the training file 0.501 / 0.759 / 0.659. The default counting setting (majority 0.9, support 2) gave 0.593 / 0.603 / 0.984 on BL-na, 0.007 / 0.293 / 0.022 on D-na and 0.032 on OPEC-na. The benchmark paper reports, for its best model trained without ambiguity (an Edge Transformer, single-edge), 0.104 on BL-na, 0.822 on D-na and 0.110 on OPEC-na; models trained with ambiguity reach 0.785 on the ambiguous BL split, which was not measured here. The pre-registered bars (BL-na ≥ 0.50 and OPEC-na ≥ 0.20; D-na and held-out ≥ 0.80; precision among answered ≥ 0.95 on every split; the control ≤ 0.30) were not met — none of the four (BL-na's exact match 0.734 clears its 0.50 line, but its precision among answered, 0.734, is below the 0.95 required of every split). What the test shows: labels exist only on the query pairs, so a derived fact cannot be checked where it lands, and the looser admission admits rules right on the labelled pairs and wrong elsewhere (0.593 against 0.734 on BL-na); which chain rules the strict setting keeps depends on the split of the training file (held-out 0.328 / 0.485 / 0.501 on three splits; the test numbers are from the third, one run); the «elsewhere» relation (Not_living_in, about a third of the depth split's rows) is refused by every setting without the declaration, so the depth split is answered on half of its pairs (with the declaration, coverage 0.933 — below); the OPEC split needs a gender guard on the middle entity of a chain (maternal / paternal), which no template carries. Next, each with its own bar: a gender guard on the middle entity, and labels or a structural check on intermediate facts; a closed-world default per entity type beyond a declaration stays unbuilt (the declaration itself is measured on the test files below). The ground truth of this benchmark is computed from published rules, so an exact rule set exists by construction; what is measured is how much of it counting on query-pair labels recovers. The dataset is released under CC BY-NC 2.0: the script reads the files in place; no data and no learned rules ship with the library Since 1.1.1, with the closed world declared (functional={'living_in': 'person'}, the strict setting otherwise unchanged — the same rules), measured on the held-out part of the same two hash splits of the training file (one run per split): 0.337 / 0.684 / 0.492 and 0.674 / 0.798 / 0.844, exact set match / precision among answered / coverage; the rows holding Not_living_in answered exactly: 17 of 590 and 373 of 588 (0 of each without the declaration); answered wrongly on the other rows: 307 against 303 and 301 against 296 without it. The stratum answers only where living_in is known for the subject, and in this benchmark it is never given for a query subject — it is derived through the living_in chain rule, which the strict setting keeps on one split and not on the other; subjects with two derived places are left open. Measured once on the benchmark's three no-ambiguity test files with the declaration (the same strict rules, learned on the third hash split of the training file — 493 rules, unchanged; the bars pre-registered before the run): D-na 0.914 / 0.979 / 0.933 (exact set match / precision among answered / coverage) against 0.477 / 0.961 / 0.496 without it — 1,463 closed answers, all right, every one from a derived home (no query subject has a given one); BL-na 0.734 / 0.734 / 0.999 and OPEC-na 0.093 unchanged row for row; new wrong answers on the rows without «elsewhere» 0 of 70,943; the 170 «elsewhere» rows of D-na left unanswered are subjects with no known home (58) or with two derived ones (112 rows — the declaration is contradicted and the stratum stays silent). With the closed facts visible to the learning rounds (closed_world_in_rounds=True) the rule set changes (499 rules) and 110 right answers are lost on D-na, so the default stays off. The gain hangs on the living_in chain rule, which the strict setting keeps on this split and not on a neutral one (17 of 590 «elsewhere» rows on the held-out part there); the benchmark paper's best model reports 0.822 on D-na in a different setting — trained end to end without the world rules (the declaration is one of them), answering every pair, one relation per pair — so the number is above the paper's, not a like-for-like comparison; the declaration restates a constraint of the benchmark's own world rules (one home per person), so the gain is the caller's knowledge, not a learned rule. (benchmarks/relations_rules_nora.py --settings strict-closed) On the planted world with only the positive templates (no «elsewhere» rule and no negative stratum to learn) the declaration answers 340 test pairs, all right: exact 0.983 against 0.915 without it (benchmarks/relations_rules_toy.py).; still missing for them: labels exist only on the query pairs, so a derived fact cannot be checked where it lands, and counting at the looser setting admits rules right on the labelled pairs and wrong elsewhere (the joint recount after each round keeps the set from collapsing); the stricter setting does better here, but which chain rules it keeps depends on the split (0.328 against 0.485: whether living_in ← living_in_same_place ∘ living_in survives the strict share on derived facts); the «elsewhere» relation (Not_living_in, about a third of the rows) is refused by every setting without a declaration — a closed-world default per entity type was not built in 1.1.0 (since 1.1.1: functional=); residual votes only bring forward what the next round's fixed point shows anyway (identical rule sets on the planted world); two binary atoms per body, unary predicates only as given (none learned or invented), and a store path that writes every justification (slow on dense stories)",
                  "script": "benchmarks/relations_battery.py"},
    "skills": {"since": "1.1", "what": "learning in a world described by code: System 2 plans on an exact model (a "
               "copy of the environment), only plans the code verified are compiled into System 1 skills with "
               "conditions over a grammar of atoms, each used alone only while its outcomes keep an anytime-valid "
               "certificate; a search-cost record per episode; the "
               "planner as System 2 of solvi.Agent; export into the knowledge store; footprints with variables "
               "generalized on the exact copy (1.1.1: lift='lattice', a wider footprint kept only when the plan "
               "instantiated on verification states reaches its goal on the copy)", "missing":
               "on a deterministic gridworld (20 fresh layouts, five seeds) System 1's share reached 0.81 in the "
               "last five episodes with every goal reached, as when System 2 plans every goal, and 1.5% of System "
               "1's decisions were in failed runs under a 20% promise pooled over the skills, while replayed cases "
               "never transferred and plans compiled without verification reached no goal. On Crafter (version "
               "1.8.3; written mechanics as the exact model; symbolic state; hand-written survival reflexes shared "
               "by every arm, 60% of all steps; the planner reads the copy's full map while the game shows an agent "
               "a 9×7 window), in one run of 10 streams × 30 episodes, no agent seed (the loop is deterministic "
               "given the world seeds). Achievements are unique achievements unlocked per episode (of 22). The skill "
               "set was certified at 5% / δ 0.1 (fifty clean uses, one e-process pooled over the stream's skills) in "
               "every stream by episode 1–2. In the last ten episodes System 1 acted alone 1,350 times with an "
               "observed outcome and failed twice (0.15%, Wilson upper bound 0.54%; 4 of 2,882 over the whole run); "
               "57 further alone runs were paused by a reflex and not resumed, so they carry no outcome (counted as "
               "failures they would make 4.2%, Wilson 5.4%). The promise is the set's, not a single skill's; a "
               "skill's own alarm retires it (4 retirements, each later re-certified). Goals that are stochastic by "
               "the written rules stayed with System 2 and are outside the promise. With skills the planner was "
               "called 32 times per episode against 42 when it plans every goal (−23%), and an episode took 84 s "
               "against 107 s on the same machine (−21%), with achievements kept: 15.2 against 15.7 unique "
               "achievements unlocked per episode of 22 (ratio 0.966, bootstrap lower bound 0.934). System 1's share "
               "of the non-reflex steps rose from 7% in the first ten episodes to 23% in the last ten (9% of all "
               "steps); the pre-registered bar of 60% was missed. Routes to a resource and plans for unseen "
               "footprints stay System 2 by design (routes were 25% of non-reflex steps), and every certified skill "
               "was a one- or two-step gather sequence with a constant footprint: in an open world the planner did "
               "most of the work and skills took the repeatable part. Hierarchical skills are untested; conditions with "
               "variables were measured in 1.1.1 (below). An arm that executed and compiled random action sequences of the plan's "
               "length in place of verified plans unlocked 4.0 unique achievements per episode against 13.7 for the "
               "same agent with no memory at all; none of its 5,497 unverified skills was ever certified, so System "
               "1 never acted. The gain is the verified plans', not the loop's bookkeeping, and the certificate held "
               "unverified skills out; what unverified memory does on its own was not measured. The observation-only "
               "learner (the agent's own action memory with the written gates on) unlocked 13.3 unique achievements "
               "per episode, no higher than the same agent without memory (13.7); the planner arms' lead over both "
               "is the planner's with the full map. The planner reads the whole map of the copy; a Crafter agent "
               "sees a 9×7 window from pixels. No number here is comparable with published Crafter scores (a "
               "geometric mean of success rates over 1M training steps). "
               "With footprints with variables (lift=\"lattice\"; one run of 10 streams × 30 episodes in the same open-world setting, "
               "the share of non-reflex steps): System 1's share 0.343 / 0.486 / 0.385 by thirds against 0.071 / 0.175 / 0.233 "
               "with constant footprints (the pre-registered bar — at least 0.40 in the last third with a rise of 15 points — missed: 0.385 and a rise of 0.041 [−0.096; 0.173]); 886 of 1,032 lattice runs accepted (0.859, a verdict on a sample of up to 16 "
               "cases) and 99.4 % of System 1's steps by skills with variables; 10 failures in 1,976 alone decisions with "
               "an outcome (0.51 %, Wilson upper 0.93 %), none without a certificate; achievements 14.43 per episode against 15.72 when System 2 plans every goal (0.918, lower bound 0.861); planner calls per episode 11.8 against 28.1 "
               "with constant footprints and 43.2 planning always; the pooled certificate's 23 alarms retired 23 skills and "
               "episodes started without a certificate in force numbered 38 / 19 / 43 by thirds — with it in force System "
               "1 took 0.50–0.58 in every third (a plausible cause of the last third's fall, not tested by a separate arm). Still missing: hierarchical skills, a certificate pool per "
               "skill or per goal (the pooled certificate's alarms retired skills with variables), "
               "a world where observation is expensive, a service with a specification, a "
               "condition rule that specializes after a failure, and a planner with a heuristic; the search-cost "
               "record is tested on a synthetic corridor only, and skills are not admitted by discriminating probes "
               "(laws are: LawSet.admit_by_probes)",
               "script": "benchmarks/skills_toy.py"},
    "injection": {"since": "1.2", "what": "a detector of instruction-like text in tool outputs, fitted and thresholded on your own texts: a "
                  "per-sentence logistic score over the fixed detector's rule families and hashed word n-grams, "
                  "fitted on labelled texts, its threshold set on ordinary texts to a chosen false-flag rate",
                  "missing": "measured once on injected tool outputs of one public benchmark (InjecAgent): on 248 "
                  "held-out injections (attacker instructions and tools not used in fitting) a detector fitted on 279 "
                  "other cases and 865 ordinary texts found 76% against the fixed rules' 23%, flagging fewer "
                  "ordinary texts (1.8% against 4.2%); but a plain TF-IDF classifier did the same, half of the gain "
                  "was one address repeated across that dataset (without it +0.19, 95% CI +0.06…+0.38), and on "
                  "140 AgentDojo-style injections the rules found 78%, the detector 53%; still missing: a gain "
                  "on an injection family it was not fitted on, and wiring into Guard as its detector", "script": None},
    "ood": {"since": "1.2",
            "what": "a watch next to System.calibrate's default Platt scaling (watch=), the known black-box shift "
            "test (Rabanser, Günnemann and Lipton, NeurIPS 2019) on the decider's answers: shift(batch), a "
            "per-feature two-sample Kolmogorov-Smirnov test of a batch's confidence, margin and options' "
            "entropy against the calibration answers, Bonferroni at alpha, says the answers' features differ "
            "from the reference; \"out of domain\" in why for one answer far from them (a max-softmax style "
            "score, Hendrycks and Gimpel 2017, as a Mahalanobis distance). It sees the decider's confidence "
            "profile, not the content: for inputs from outside the calibration set use OpenSetGate, for the "
            "share answered alone and the answer mix DriftMonitor",
            "missing": "measured once on intent classification with solvi-base (raw confidences, next to Platt "
            "calibration; alpha 0.01; 10 directed pairs of CLINC150, Banking77, HWU64 and Bitext customer "
            "support): with 200-answer batches (shift), 0.99–1.0 of the batches from another dataset flagged "
            "on every pair and at most 0.05 of in-domain batches (one domain at that bar, 0.03–0.05 over two "
            "draws of the windows; 0.01 or less for the others); with 50-answer batches, a mean of 0.76 and "
            "0.30–0.48 on four pairs, +0.18 [0.05, 0.34] over a dev-tuned two-sided mean-confidence test at "
            "the watch's own in-domain rate of about 1 % (worse on one pair) and on par with it at 5 %. On "
            "half of the pairs the mean confidence rose, which a test for a fall of confidence cannot see. "
            "Held-out intents of the same dataset were mostly not flagged (CLINC150, Banking77: under 1 % of "
            "batches), while a rephrasing of the same domain (Banking77 and Bitext retail banking) was "
            "flagged like another dataset; the per-answer note marked on average 10 % of another dataset's "
            "answers. All four datasets are in or near the decider's training mix and their option sets "
            "differ (20 of 27–150 labels); still missing: a benchmark script in benchmarks/ and a decider "
            "that has not seen the data",
            "script": None},
    "audit": {"since": "1.2",
              "what": "AuditWatch: a Bernoulli CUSUM on a check run on a random share of decisions "
              "(Dispatcher(audit=)), its threshold simulated with the share and the reference's sampling error, for a "
              "check that is otherwise bought only where the system is unsure (a known method: continuous sampling "
              "inspection with a proportion CUSUM)",
              "missing": "one task family and one break size measured (docs/experimental.md); a smaller break, a short "
              "reference and a break confined to the decisions the policy never checks are unmeasured",
              "script": None},
    "process": {"since": "1.4",
                "what": "a process of several solvi systems (stages 0 to 4 of solvi.Process): nodes, edges with "
                "predicates written as data, merges, loops with a mandatory max_iter and the declared process promise "
                "(union / end_to_end / per_node), as a JSON spec with a hash stable to key order, a structural diff, "
                "a Mermaid or text drawing, and from_spec checking every system's fingerprint; run / explain: each "
                "node's decision in its own store, chained by run_context into one run, with a \"process\" index "
                "record of the path; questions to a person through a queue (HumanChannel, MemoryQueue, "
                "CallbackChannel): a run waits \"pending\" and resume goes on as a new segment, the answer recorded "
                "with its reviewer; replay / replay_all of a run from its records across its segments, a person's "
                "answer a leaf recorded and not recomputed, the first divergence named by node and step; guarantee(): the "
                "path promise in the modes union (split equal, given or computed, each node's own promise at its "
                "share, loops counted), per_node (the end error is not bounded) and end_to_end (calibrate() on "
                "end-labelled runs, learn then test over a ThresholdFamily: shared_rank by default, per_node_fixed, "
                "or your own), and a PathWatch on the mix of paths",
                "missing": "the retraction across nodes with cancelled and re-asked tickets (a later stage); a run "
                "follows one path (merge \"all\" is not run yet, so the 100-node process of the promise stand "
                "cannot run); the pre-registered stands for scale (300 systems) and retraction are not run — ten "
                "systems exchanging facts (with 2% of the decisions asking a person) is a test, not the stand",
                "script": "benchmarks/process_e1.py"},
    "stream": {"since": "1.4",
               "what": "StreamPromise: an anytime-valid learn-then-test certificate on a threshold grid, earned on "
               "the stream (an e-process per grid point — a mixture of constant bets or the aGRAPA bet — "
               "fixed-sequence from strict to loose, the threshold only loosens, a missing label counted as a loss, "
               "δ spent over restarts by DeltaBudget), installed with System.guarantee(promise=); RiskDetector "
               "(a Shiryaev–Roberts e-detector on the deployed risk) and the calibration coin wired into the "
               "Dispatcher (audit=\"adaptive\", promise=, detector=, on_drift=\"recertify\", probation=), recorded "
               "in the chain and rebuilt by Dispatcher.replay_stream (stages 1–2 of 4); tickets: an agent as the "
               "oracle over MCP (stream.tickets, TicketDesk, solvi serve --mcp --agent-oracle) — blind, durable in "
               "the store, an expired ticket a recorded loss (stage 3a); the closed learning loop (stream.versions, "
               "LearningLoop over stream.learner's HashedTextLearner / FastHeadLearner): frozen versions, each with "
               "its own StreamPromise calibrated only on samples drawn after its freeze, δ from one DeltaBudget, "
               "promoted when certified, answering alone on ≥ min_share (5 %) and ≥ the served version's share + "
               "margin (1 point) on the same samples, every freeze / promote / discard / rollback in the chain and "
               "replayed; version 0 optionally the fitted System 1 with its own promise (initial=), a RiskDetector "
               "and probation on the served version, a resolved ticket's verdict as a label (ticket_verdict) "
               "(stage 3b); the adapter loop's policy (stream.policy: LoopPolicy, LoopState, Budget, DefaultPolicy) "
               "and gate arithmetic (stream.gates); the adapter loop core (stream.loop.AdapterLoop: policy hooks, "
               "novel-only promotion with the memorisation probe and the novel-risk bar, two alarm sources, "
               "pause / resume / train_now / promote / rollback; LearningLoop = AdapterLoop + DefaultPolicy at the "
               "lane's constants; stream.serving), unmeasured; no LoRA learner, runner, checkpoint or attach yet",
               "missing": "measured once; see the documented limits in docs/experimental.md: the "
               "RiskDetector alarmed far later than the confidence CUSUM on shifts to new intents and must not "
               "be the only alarm; the LearningLoop gained nothing with gold labels and is kept only as the core "
               "of a planned adapter loop; TicketDesk is unmeasured on a stream. The loop is not wired "
               "into the Dispatcher (no loop=: it drives its own coin and oracle); the expired-share stop "
               "(max_drop) and repeated tickets for the "
               "oracle's own noise are not built; a person's audit label does not yet replace the slow path's label",
               "script": None},
}
for _s in STATUS.values():
    _s["deadline"] = DEADLINE
del _s


_WARNED: set = set()        # the experimental modules that have warned in this process


def warn_on_import(module, why=None):
    """Called at the top of each experimental module: one ExperimentalWarning, located at the import that loaded it —
    once per module per process (a module imported again, by a package walk or after it left sys.modules, is quiet:
    long runs keep their logs clean)."""
    if module in _WARNED:
        return
    _WARNED.add(module)
    from .._deprecate import _warn_from_caller
    from ..core.catalog import ExperimentalWarning
    name = module.split(".")[2]
    _warn_from_caller(f"{module} is experimental{': ' + why if why else ''}; its API may change, and it is removed in "
                      f"{DEADLINE} unless it graduates (solvi.experimental.STATUS[{name!r}])", ExperimentalWarning, skip=2)


def mark(system, name):
    """Say that `system` decides with the experimental piece `name`: its decisions record it (meta["experimental"]).
    → the system."""
    system.experimental = sorted({*(getattr(system, "experimental", None) or ()), name})
    return system


__all__ = ["DEADLINE", "STATUS", "mark", "warn_on_import"]
