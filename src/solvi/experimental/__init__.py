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
             "systems, export into the knowledge store", "missing": "on a game simulator with written mechanics the "
             "laws closed 94% of the gap between a learned map and the written mechanics, but random candidates under "
             "the same test came within 1.1 points, so the mismatch is a compact proposal heuristic and the test does "
             "the work; on a real MCP filesystem server with a path policy, probes planned from mismatches took "
             "refusals from 6 to 0 in one session (random probes: 48 to 29); still missing: a grammar too large to "
             "enumerate, retirement of weak laws nothing contradicts, wiring into a System's decisions, and a "
             "script for the two stands (the toys: benchmarks/laws_toy.py)", "script": "benchmarks/laws_toy.py"},
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
