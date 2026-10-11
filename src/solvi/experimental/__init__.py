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
             "missing": "on a game simulator with written mechanics the "
             "laws closed 94% of the gap between a learned map and the written mechanics, but random candidates under "
             "the same test came within 1.1 points, so the mismatch is a compact proposal heuristic and the test does "
             "the work; on a real MCP filesystem server with a path policy, probes planned from mismatches took "
             "refusals from 6 to 0 in one session (random probes: 48 to 29); still missing: a grammar too large to "
             "enumerate, retirement of weak laws nothing contradicts, wiring into a System's decisions, and a "
             "script for the two stands (the toys: benchmarks/laws_toy.py); a ranker of candidates for expensive checks "
             "(RankedProposer, a bag-of-atoms model trained on tasks drawn from the grammar) cut the checks to the true "
             "condition from 69.7 in the blind order to 2.1 on synthetic tasks of one grammar, in distribution only "
             "and never on a live system"
             "; the acceptance sized by the candidates tested on a buffer (on a synthetic buffer with no law the "
             "fixed thresholds accepted a law in 100 of 100 buffers, the sized rule in 0 of 100), admission by "
             "discriminating probes (on a synthetic world at 8 checks a random held-out sample accepted a wrong "
             "law in 0.593 of the trials, the probes in none) and abstention where laws that are not exceptions of"
             " one another disagree are measured on synthetic worlds only, not yet on the game simulator or the "
             "file server", "script": "benchmarks/laws_toy.py"},
    "relations": {"since": "1.1", "what": "relations over a state and rules over them: packed-bit relations, the "
                  "relation algebra (composition, converse, closure, counts), a bottom-up enumerator with "
                  "observational equivalence on the full relations, a library of accepted programs, compilation into "
                  "set code or index arrays with a differential check, encoders (a grid window seen from the player, "
                  "a path tree) and a proposer of relational atoms for the laws' held-out test; composites by "
                  "enumeration with a library (find_rules admits only what passes a held-out test and builds the next "
                  "rules on the admitted ones) and a fallback slot for a learned proposer; composition tables of relations "
                  "learned by exact counting (residual fill, derivation over every bracketing, refusal of what the "
                  "table cannot derive)", "missing": "on a "
                  "pre-registered battery of 28 random rules over three relations the enumerator found 22 within "
                  "seven nodes, every find exact on larger graphs of other families, and compiled rules matched "
                  "hand-written set code pair for pair on a forest of 100,000 nodes within twice its time; on two "
                  "toys (a path tree with \"denied under a directory named secret at depth 3 or more\", three levels; "
                  "kinship on forests, depth 1 to 6; five seeds each) the library found every level within 2,556 and "
                  "10,817 enumerated programs in all, where the same search without it found neither top rule in "
                  "100,000, with 0 false accepts on larger fresh worlds, but the kinship ladder missed its bar of a "
                  "tenth of that budget (1.08 times: the universal relation of the enumerator enlarges every level), "
                  "and met it at 7,889 with the universal relation left out of the terminals "
                  "(include_universal=False, an opt-in measured as an option, the bar unchanged); a rule the programs "
                  "cannot express (x and y share a 4-clique) was never admitted on graphs of density 0.3 or 0.5, but "
                  "on dense graphs (0.7) precision acceptance with held-out graphs of the same density admitted the "
                  "plain edge relation, which held-out graphs of another density or exact acceptance rejected; still "
                  "missing: a stand where relational atoms beat the flat grammar on real data (an MCP filesystem "
                  "server with an ancestor rule, a game simulator with written mechanics), how deep the search "
                  "reaches with the twenty-odd relations of a grid window, the library's intermediate concepts found "
                  "rather than given as targets, a learned proposer for the fallback slot (none ships), and a way to "
                  "drop proposed atoms no law uses and library entries that make other rules dearer"
                  "; composition tables: on CLUTRR (Sinha et al., 2019), the clean split trained on chains of 2–3 "
                  "relations and tested on 2–10; the input is the benchmark's relation graph — the chain of relations "
                  "between the two people — not the story text; the text-to-graph step is not part of this measurement,"
                  " a table learned by counting (majority 0.9, support 2, residual fill, all bracketings must agree; "
                  "111 cells from 9,074 training chains of length 2–3 in 0.07 s on CPU) gave 0 wrong among 1,087 "
                  "answered of 1,146 test queries (38–190 per length; at lengths 2 and 3 only 2 and 7 distinct chains):"
                  " precision among the answered 1.000 (0 of 1,087) at every length 2–10, coverage 0.949 overall, 0.975"
                  " at length 10 (116 of 119) and 0.916–0.993 at lengths 4–9; 59 refused (21 disagreeing bracketings, "
                  "38 underivable); one split, one run; the most frequent training relation per length answered 0.077 "
                  "(0.134 at length 10); the benchmark is generated by closed-path composition rules, so an exact table"
                  " can exist by construction; what is measured is whether counting with refusal recovers it without "
                  "errors; still missing for them: the sized acceptance with a base-rate null admits an exact 2-vote "
                  "cell only for a relation with base rate ≤ 0.03 — on a 10% slice it refused 12 of 15 exact support-2 "
                  "cells the default admits; keep `support=` explicit for deterministic tables (a null of exactness is "
                  "not built); residual fill after a different table admits different cells and was the source of that "
                  "arm's extra wrong answers (residual votes are not gated like direct votes); one relation per cell, "
                  "so a gender-dependent cell is refused, not split (no gender slot), and the opt-in identity relation,"
                  " with no gender slot, gave 9 wrong answers at lengths 4–10; the text setting (story → relation "
                  "graph) and forward chaining over learned rules",
                  "script": "benchmarks/relations_battery.py"},
    "skills": {"since": "1.1", "what": "learning in a world described by code: System 2 plans on an exact model (a "
               "copy of the environment), only plans the code verified are compiled into System 1 skills with "
               "conditions over a grammar of atoms, each used alone only while its outcomes keep an anytime-valid "
               "certificate; a search-cost record per episode; the "
               "planner as System 2 of solvi.Agent; export into the knowledge store", "missing": "on "
               "a deterministic gridworld (20 fresh layouts, five seeds) System 1's share reached 0.81 in the last "
               "five episodes with every goal reached, as when System 2 plans every goal, and 1.5% of System 1's "
               "decisions were in failed runs under a 20% promise pooled over the skills, while replayed cases never "
               "transferred and plans compiled without verification reached no goal; still missing: a stand beyond "
               "the toy (a game simulator with written mechanics, a service with a specification), a 5% promise "
               "(fifty clean uses per certified unit), a condition rule that specialises after a failure, and a "
               "planner with a heuristic"
               "; the search-cost record is tested on a synthetic corridor only, and skills are not admitted by "
               "discriminating probes (laws are: LawSet.admit_by_probes)",
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
