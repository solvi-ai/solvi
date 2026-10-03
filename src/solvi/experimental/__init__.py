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
- nothing stable imports them, and the top-level `solvi` package does not re-export them. Two command-line entries load
  one on request: `solvi hook` (hooks) and `solvi serve --upstream` (mcp).

Graduating needs a pre-registered bar, a script in `benchmarks/` that measures it, the bar met and reviewed, no import
of anything experimental, a conformance check and a docs move; the old path is then kept for one release. A piece that
does not graduate by its deadline (two minor releases after 1.0), or that measures negative, is removed and the negative
result published.

This package itself imports none of them: `import solvi.experimental` does not warn."""
from __future__ import annotations

DEADLINE = "1.2"

STATUS: dict[str, dict] = {
    "learning": {"since": "0.7", "what": "the learning loop: trusted labels → a ladder of updates → gates → promote or "
                 "roll back", "missing": "a simulation on real data that decides the default gates; the gate's holdout "
                 "must come from the audit, and a rollback during a drift flag must hold", "script": None},
    "lora": {"since": "0.7", "what": "a LoRA adapter on the decider for one question (adapt_lora)", "missing": "a "
             "decision on real use, a benchmark script in the repo, and wiring as the learning loop's adapter step; "
             "measured gains over fit come with overconfidence", "script": None},
    "compile": {"since": "0.9", "what": "a written specification compiled by an LLM into catalog parts, accepted when "
                "two drafts agree and the specification's tests pass (with its sandbox)", "missing": "acceptance on "
                "larger specifications and a benchmark script in benchmarks/", "script": None},
    "hooks": {"since": "0.7.1", "what": "`solvi hook`: a coding agent's pre-edit rule checks and skill picker",
              "missing": "false deny and ask rates on real coding-agent sessions, with a script", "script": None},
    "specialist": {"since": "0.7", "what": "the propose → check → render specialist pattern", "missing": "a measured "
                   "specialist (charts has no accuracy number yet)", "script": None},
    "charts": {"since": "0.7", "what": "verified charts: every number quoted from the text, a deterministic SVG",
               "missing": "accuracy on real texts with the rule-based and the LLM proposer", "script": None},
    "mcp": {"since": "0.9", "what": "an MCP stdio proxy that puts the agent guard in front of `tools/call` "
            "(`solvi serve --guard --upstream`)", "missing": "a measured run through the proxy against a real MCP "
            "server", "script": None},
    "oncalib": {"since": "1.0", "what": "recalibrating a guarantee on the fly from outcomes", "missing": "a "
                "recalibration that keeps the guarantee's promise (it does not: see its module docs)", "script": None},
    "counterfactual": {"since": "0.7", "what": "the smallest change of a decision's inputs that changes its answer "
                       "(adverse-action reasons)", "missing": "one measured use, e.g. adverse-action reasons on the "
                       "credit gallery task", "script": None},
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
