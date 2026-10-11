"""solvi.experimental — pieces that work and are tested but have not yet shown a measured gain (or a measured use) that
the rest of solvi promises. Their API may change in any release, and each one either graduates or is removed by its
deadline.

    from solvi.experimental import STATUS
    STATUS["lora"]       # {"since": "0.7", "missing": "...", "script": None, "deadline": "1.2"}

What marks them:

- importing one warns once (`solvi.ExperimentalWarning`, a UserWarning: silence it with
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
                "two drafts agree and the specification's tests pass (with its sandbox)", "missing": "acceptance on "
                "larger specifications and a benchmark script in benchmarks/", "script": None},
    "hooks": {"since": "0.7.1", "what": "`solvi hook`: a coding agent's pre-edit rule checks and skill picker",
              "missing": "false deny and ask rates on real coding-agent sessions, with a script", "script": None},
    "charts": {"since": "0.7", "what": "verified charts: every number quoted from the text, a deterministic SVG",
               "missing": "accuracy on real texts with the rule-based and the LLM proposer", "script": None},
    "changes": {"since": "1.1", "what": "one change pipeline for four sources (a document version, verified "
                "experience, a person's correction, a search proposal): pluggable stages, a journal of kind \"change\", "
                "promote into the knowledge store / compiled versions / a labelled update, versions and rollback", "missing": "on a "
                "credit policy the pipeline carried a document version (144 of 144 changed decisions traced to their "
                "clause), admitted 12 of 12 exact exceptions, held every widening one for a person and refused 8 of 8 "
                "exceptions that touched other decisions, 0 extraneous changes, 14 versions replayed; on a bank-request "
                "stream with a person's labels every one of 22 candidates passed every stage (`nogate` = `gate`), so "
                "the stages added nothing there — the module is journal, impact analysis, approval levels and "
                "rollback; still missing: the narrows / widens classifier over rule conditions, a regression-scenario "
                "stage, and a script for the bank-request stream (the credit numbers: benchmarks/changes_credit.py)",
                "script": "benchmarks/changes_credit.py"},
    "laws": {"since": "1.1", "what": "laws of an environment from prediction mismatches: a grammar of atoms you "
             "write, candidates from each mismatch, acceptance only by a held-out test on other groups (worlds, "
             "sessions, paths), most specific first, retirement on falling precision, a probe planner for live "
             "systems, export into the knowledge store; acceptance sized by the candidates tested, admission by "
             "discriminating probes, abstention where laws disagree, a search-cost record",
             "missing": "on a game simulator with written mechanics "
             "the laws closed 94% of the gap between a learned map and the "
             "written mechanics, but random candidates under the same test came within 1.1 points, so the mismatch is "
             "a compact proposal heuristic and the test does the work; on a real MCP filesystem server with a path "
             "policy, probes planned from mismatches took refusals from 6 to 0 in one session (random probes: 48 to "
             "29); still missing: a grammar too large to enumerate, retirement of weak laws nothing contradicts, "
             "wiring into a System's decisions, and a script for the two stands (the toys: benchmarks/laws_toy.py); a "
             "ranker of candidates for expensive checks (RankedProposer, a bag-of-atoms model trained on tasks drawn "
             "from the grammar) cut the checks to the true condition from 69.7 in the blind order to 2.1 on synthetic "
             "tasks of one grammar, in distribution only; ranking by the joint statistic of the whole condition over the "
             "logged rows (JointRanker, no training; with no rows it is the blind order — the statistic needs rows) "
             "needed 1.00 / 17 / 1.01 checks (medians 1 / 1 / 1) on three synthetic settings of the library's own "
             "grammars (five seeds × 60 tasks) against 85 / 2,182 / 126 in the blind order and 2.1 / 591 / 7.7 for "
             "the trained bag of atoms, a shuffled-rows control worse than the blind order (112 / 2,434 / 166), the "
             "gain on the deep setting showing from about 20 logged rows and full by 300, the same law accepted "
             "under every order in 894 of 900 tasks; measured once on a live file "
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
                  "table cannot derive); forward chaining to a fixed point over the composition rules in the knowledge store (derived facts with their premises and the rule as their justification, two derived relations of one pair a dispute); rules first, a model second (the table answers what it derives, a fallback model what it refuses, every answer replayable without the model); "
                  "top-down synthesis of chain programs through residuals of the target (synthesize_residual, "
                  "synthesize(method=\"residual-first\")); why a relation was not derived (why_not: the missing cell "
                  "or fact, ranked, as an agenda goal)",
                  "missing": "on a pre-registered battery of 28 random rules "
                  "over three relations the enumerator found 22 within "
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
                  "depth 1 to 6; five seeds each) the library found every level within 2,556 and 10,817 enumerated "
                  "programs in all, where the same search without it found neither top rule in 100,000, with 0 false "
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
                  "the grammar can express, and one that needs a 56-node program — were never admitted (0 of 20 "
                  "runs); on the kinship ladder (nine rungs up to second cousin once removed) the library found all "
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
                  "answers at lengths 4–10; the text setting (story → relation graph); "
                  "forward chaining (forward_chain): on 1,816 held-out stories of the same split's training file (a fifth of it by a hash of the id, chains of 2 and 3 relations; the table fitted on the other four fifths, 111 cells), the store's reading of the query pair after chaining to the fixed point was compared with the table's derivation over the shortest paths: 1,675 equal (0.922), 11 disputed where the bracketings disagree (0.006), 130 with nothing derived where the table cannot derive (0.072), 0 differences; the fixed point took 0 rounds for 72 stories, 1 for 826 and 2 for 918; 2.6 s of chaining in all, 1.4 ms a story, on CPU; one run, the test file not read; on 120 random chains of 2–7 relations over 12 random tables whose bracketings can disagree, 0 differences (2 answers equal to the table's rest only on a disputed pair and stay hypotheses until a person answers it); still missing for it: rules other than the composition of two relations over pair facts (no general rules, negation or aggregation); a learned identity relation is chained as an ordinary relation, not as a neutral one; the stories of the training file need at most two rounds, so deeper fixed points are measured by the tests only; rules first, a model second (RulesFirst): on the frozen test split (1,146 queries, chains 2–10) the rules answered 1,087 with 0 wrong and refused 59; the LLM fallback gave a readable option on 50 and was right on 39 (8 empty at the token cap, 1 off the list): 0.983 at coverage 0.992 against 0.949 at 0.949 — the coverage bar of 1.0 missed; solvi-base 2 of 59; replay 0 failures; still missing for it: the same dispatcher over forward_chain, the text setting, gating the model's answer by its confidence, a retry on an empty reply; residual synthesis (synthesize_residual): on the kinship ladder without a library all nine rungs on five seeds (45 of 45; the top rung in 0.40–0.55 s) with 0 false accepts on check worlds, against 20 of 45 for the enumerator at 100,000 behaviours (benchmarks/relations_residual.py); still missing for it: union, intersection, closure and counts (only chains of atoms and converses with ∖ I; two masked segments side by side need a library entry), a use inside find_rules, and a measurement on rules of a real domain; why_not: tested on planted tables only (it names exactly a removed cell), not measured on the refusals of a real split",
                  "script": "benchmarks/relations_battery.py"},
    "skills": {"since": "1.1", "what": "learning in a world described by code: System 2 plans on an exact model (a "
               "copy of the environment), only plans the code verified are compiled into System 1 skills with "
               "conditions over a grammar of atoms, each used alone only while its outcomes keep an anytime-valid "
               "certificate; a search-cost record per episode; the "
               "planner as System 2 of solvi.Agent; export into the knowledge store", "missing":
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
               "most of the work and skills took the repeatable part. Hierarchical skills and conditions with "
               "variables are untested. An arm that executed and compiled random action sequences of the plan's "
               "length in place of verified plans unlocked 4.0 unique achievements per episode against 13.7 for the "
               "same agent with no memory at all; none of its 5,497 unverified skills was ever certified, so System "
               "1 never acted. The gain is the verified plans', not the loop's bookkeeping, and the certificate held "
               "unverified skills out; what unverified memory does on its own was not measured. The observation-only "
               "learner (the agent's own action memory with the written gates on) unlocked 13.3 unique achievements "
               "per episode, no higher than the same agent without memory (13.7); the planner arms' lead over both "
               "is the planner's with the full map. The planner reads the whole map of the copy; a Crafter agent "
               "sees a 9×7 window from pixels. No number here is comparable with published Crafter scores (a "
               "geometric mean of success rates over 1M training steps). Still missing: hierarchical skills, "
               "footprints with variables, a world where observation is expensive, a service with a specification, a "
               "condition rule that specialises after a failure, and a planner with a heuristic; the search-cost "
               "record is tested on a synthetic corridor only, and skills are not admitted by discriminating probes "
               "(laws are: LawSet.admit_by_probes)",
               "script": "benchmarks/skills_toy.py"},
}
for _s in STATUS.values():
    _s["deadline"] = DEADLINE
del _s


def warn_on_import(module, why=None):
    """Called at the top of each experimental module: one ExperimentalWarning, located at the import that loaded it."""
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
