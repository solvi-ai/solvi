"""The import structure of solvi, read from the source (ast), so that a cycle or a layer break fails here (LAYOUT §6;
the tiers are package prefixes of the 1.0 layout: solvi.core.<area>):

- no import cycle at module level (the imports a module runs when it is imported, nested in `if` / `try` included,
  `if TYPE_CHECKING:` and function bodies not): such a cycle makes the import order matter and breaks on a refactor;
- the execution layer (solvi.core: catalog, types, provenance, primitives, runtime, schema, textin) imports nothing
  above it, not even
  inside a function: what it needs of the modules above (the flow, the replay of their records) is defined in it;
- nothing below the server and agent layer (solvi.cli, solvi.serve, solvi.experimental.hooks / mcp, solvi.solutions.guard)
  imports from it;
- (rule 2) an import cycle — any import, inside a function too — stays inside one tier: the cross-tier cycle of 0.9
  (24 modules from decide to system and dispatch) was broken in 1.0 by the moves of LAYOUT §6, and the cycles left
  inside a tier are listed (INTRA_TIER_CYCLES), so a new one is a decision, not an accident;
- (rule 3) tier order: a low-level module imports only its own tier or the ones below it (kernel → parts → guarantees →
  records → system → deliberate → knowledge); a high-level module imports low and high; an experimental one anything;
  nothing stable imports an experimental module (except the allowlisted command-line entries). Every module is placed:
  a new one has to be put in a tier.

Inside a function, importing the package root itself (`from . import __version__`) is not counted: the root has finished
importing before any function of a submodule runs."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
BASE = {"solvi.core"} | {f"solvi.core.{m}" for m in ("catalog", "types", "provenance", "primitives", "runtime", "schema",
                                                     "textin")}
TOP = ("solvi.cli", "solvi.serve", "solvi.experimental.hooks", "solvi.experimental.mcp", "solvi.__main__", "solvi.solutions.guard")


def _modules():
    out = {}
    for p in (SRC / "solvi").rglob("*.py"):
        parts = list(p.relative_to(SRC).with_suffix("").parts)
        package = parts[-1] == "__init__"
        out[".".join(parts[:-1] if package else parts)] = (p, package)
    return out


def _type_checking(node):
    t = node.test
    return isinstance(node, ast.If) and (getattr(t, "id", None) == "TYPE_CHECKING" or getattr(t, "attr", None) == "TYPE_CHECKING")


def _imports(tree, module_level):
    """The import statements of a module: those it runs when imported (module_level), or all of them."""
    found = []

    def visit(body):
        for n in body:
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                found.append(n)
            elif module_level and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(n, ast.ClassDef):           # a class body runs at import; its methods do not
                    visit([x for x in n.body if not isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef))])
                continue
            elif module_level and isinstance(n, ast.If) and _type_checking(n):
                continue
            for f in ("body", "orelse", "finalbody", "handlers"):
                if isinstance(getattr(n, f, None), list):
                    visit(getattr(n, f))
    if module_level:
        visit(tree.body)
    else:
        found = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    return found


def graph(module_level):
    """{module: {solvi modules it imports}}. Importing a.b.c imports the packages a and a.b first (unless the importer
    is inside them: they are being imported already); a module that imports names from its own package depends on it."""
    mods = _modules()
    g = {m: set() for m in mods}
    for m, (path, package) in mods.items():
        for n in _imports(ast.parse(path.read_text()), module_level):
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            else:
                if n.level:
                    base = m.split(".") if package else m.split(".")[:-1]
                    base = base[:len(base) - (n.level - 1)]
                    t = ".".join(base + ([n.module] if n.module else []))
                else:
                    t = n.module or ""
                targets = [f"{t}.{a.name}" if f"{t}.{a.name}" in mods else t for a in n.names]
            for t in targets:
                while t not in mods and "." in t:
                    t = t.rsplit(".", 1)[0]
                if t not in mods:
                    continue
                parts = t.split(".")
                for i in range(1, len(parts) + 1):
                    x = ".".join(parts[:i])
                    implicit = i < len(parts)                # a package imported on the way to the named module
                    if x != m and x in mods and not (implicit and m.startswith(x + ".")):
                        g[m].add(x)
    if not module_level:                              # the root, imported inside a function: see the module docs
        for m in g:
            g[m].discard("solvi")
    return g


def cycles(g):
    """The strongly connected components with more than one module (Tarjan, iterative)."""
    index, low, on, stack, out, counter = {}, {}, set(), [], [], [0]
    for root in g:
        if root in index:
            continue
        work = [(root, iter(sorted(g[root])))]
        index[root] = low[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on.add(root)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in index:
                    index[w] = low[w] = counter[0]
                    counter[0] += 1
                    stack.append(w)
                    on.add(w)
                    work.append((w, iter(sorted(g[w]))))
                elif w in on:
                    low[v] = min(low[v], index[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == index[v]:
                comp = set()
                while True:
                    x = stack.pop()
                    on.discard(x)
                    comp.add(x)
                    if x == v:
                        break
                if len(comp) > 1:
                    out.append(comp)
    return out


def test_the_cycle_finder_finds_a_cycle():
    assert cycles({"a": {"b"}, "b": {"c"}, "c": {"a"}, "d": {"a"}}) == [{"a", "b", "c"}]
    assert cycles({"a": {"b"}, "b": set()}) == []


def test_no_import_cycle_at_module_level():
    found = cycles(graph(module_level=True))
    assert not found, f"module-level import cycle(s): {[sorted(c) for c in found]}"


def test_the_execution_layer_imports_nothing_above_it():
    g = graph(module_level=False)
    assert BASE <= set(g)
    leaves = {m for m, ts in g.items() if not ts}     # modules that import no other solvi module (_deprecate, costs)
    above = {f"{m} → {t}" for m in BASE for t in g[m] if t not in BASE | leaves}
    assert not above, f"the execution layer imports a module above it: {sorted(above)}"


def test_nothing_below_the_server_and_agent_layer_imports_it():
    def top(m):
        return m == "solvi" or any(m == p or m.startswith(p + ".") for p in TOP)
    g = graph(module_level=False)
    up = {f"{m} → {t}" for m, ts in g.items() if not top(m) for t in ts if top(t)}
    assert not up, f"a library module imports the server / agent layer: {sorted(up)}"


# --- the tier map (LAYOUT §2 / §6): a module belongs to the tier of its longest listed prefix (after "solvi.")
LOW_TIERS = {
    "kernel": "_deprecate _loader _migrate _rpc core",                  # solvi.core.* unless listed below
    "parts": "core.deciders core.extract core.plan core._inputs",
    "guarantees": "core.guarantees",
    "records": "core.store core.response",
    "system": "core.system",
    "deliberate": "core.slow core.dispatch",
    "knowledge": "core.knowledge",
}
HIGH = "__init__ __main__ _command solutions check cli models serve show testing"
EXPERIMENTAL = "experimental"                                           # solvi.experimental.*
# stable → experimental imports that are meant: the command-line entry for an experimental feature (LAYOUT §6:
# `solvi serve --upstream`). solvi.build takes a compiled specification and no longer compiles one (LAYOUT risk 3).
ALLOWED_EXPERIMENTAL = {("solvi.serve", "solvi.experimental.mcp")}
# stable modules that load an experimental module by its name, on request: `solvi hook`, and a calibration file that
# carries a LoRA adapter (the adapter's own loader)
ALLOWED_BY_NAME = {("solvi.cli", "solvi.experimental.hooks"), ("solvi.core.deciders.adapt", "solvi.experimental.lora")}
# the import cycles left in 1.0, each inside one tier
INTRA_TIER_CYCLES = [
    {"solvi.core.catalog", "solvi.core.primitives", "solvi.core.provenance", "solvi.core.runtime", "solvi.core.schema",
     "solvi.core.textin", "solvi.core.types"},
    {"solvi.core.store", "solvi.core.response", "solvi.core.store.report", "solvi.core.store.signature"},
    {"solvi.solutions.guard", "solvi.solutions.guard.confirm"},
]


def _level(m):
    """A module → ("low", tier index) / ("high", None) / ("experimental", None), by its longest listed prefix."""
    name = "__init__" if m == "solvi" else m.split(".", 1)[1]
    table = {x: ("experimental", None) for x in EXPERIMENTAL.split()}
    table.update({x: ("high", None) for x in HIGH.split()})
    for i, (_, mods) in enumerate(LOW_TIERS.items()):
        table.update({x: ("low", i) for x in mods.split()})
    parts = name.split(".")
    for k in range(len(parts), 0, -1):
        if ".".join(parts[:k]) in table:
            return table[".".join(parts[:k])]
    return None


def _tier(m):
    lvl, i = _level(m)
    return list(LOW_TIERS)[i] if lvl == "low" else lvl


def test_every_module_is_placed_in_a_tier():
    missing = sorted(m for m in graph(module_level=False) if _level(m) is None)
    assert not missing, f"put these modules in a tier (LOW_TIERS, HIGH or EXPERIMENTAL): {missing}"


def test_import_cycles_stay_inside_one_tier_and_are_the_known_ones():
    found = cycles(graph(module_level=False))
    across = [sorted(c) for c in found if len({_tier(m) for m in c}) > 1]
    assert not across, f"an import cycle across tiers: {across}"
    assert sorted(map(sorted, found)) == sorted(map(sorted, INTRA_TIER_CYCLES)), \
        f"the cycles inside a tier changed (update INTRA_TIER_CYCLES if it is meant): {sorted(map(sorted, found))}"


def test_tier_order_and_levels():
    bad = []
    for m, ts in graph(module_level=False).items():
        lm, im = _level(m)
        for t in ts:
            lt, it = _level(t)
            if lm == "low" and (lt != "low" or it > im):
                bad.append(f"{m} ({_tier(m)}) → {t} ({_tier(t)})")
            elif lm == "high" and lt == "experimental" and (m, t) not in ALLOWED_EXPERIMENTAL and not (
                    t == "solvi.experimental" and any(a == m for a, _ in ALLOWED_EXPERIMENTAL)):   # the package on the way
                bad.append(f"{m} (high) → {t} (experimental)")
    assert not bad, f"imports against the tier order: {sorted(bad)}"


def test_the_names_moved_by_the_import_cycle_lane_are_the_same_objects():
    """The cycle moves of 1.0 (LAYOUT §6) kept the names the old modules re-export: the same objects."""
    import solvi
    from solvi import _rpc, serve
    from solvi.cli import _calibrate as calibrate
    from solvi.core import deciders as decide, dispatch, response, store as storage, system
    from solvi.core import calibfile, calibration, chain, costs, runtime, sources
    assert system.Response is response.Response is solvi.Response and system._append is chain.append
    for name in ("TRUSTED_SOURCES", "VERIFIED", "VERIFIED_REFUSED", "UntrustedLabel", "check_source"):
        assert getattr(storage, name) is getattr(sources, name)
    assert decide.GroupBy is calibration.GroupBy and decide.group_name is calibration.group_name
    assert decide.plan_batches is runtime.plan_batches
    assert dispatch.price_of is costs.price_of and dispatch._usages is costs._usages
    for name in ("Limits", "RequestError", "NotFound", "BadRequest", "Busy", "too_deep", "parse_json", "internal_error",
                 "_readline"):
        assert getattr(serve, name) is getattr(_rpc, name)
    import pytest
    for name in ("add_parser", "cmd_calibrate", "examples_of", "find_part", "label_of", "read_rows"):
        with pytest.warns(solvi.SolviDeprecationWarning, match=f"solvi.calibfile.{name} moved in 1.0"):
            assert getattr(calibfile, name) is getattr(calibrate, name)


def test_stable_modules_name_an_experimental_module_only_where_allowed():
    """A string that is an experimental module's path ("solvi.experimental.hooks", to import it later) in a stable
    module: only the allowlisted command-line entry and adapter loader."""
    found = set()
    for m, (path, _) in _modules().items():
        if _level(m)[0] == "experimental" or m == "solvi._deprecate":          # _deprecate: the table of moves
            continue
        for n in ast.walk(ast.parse(path.read_text())):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.startswith("solvi.experimental.") \
                    and n.value[-1].isalnum() and n.value.replace(".", "").replace("_", "").replace(":", "").isalnum():
                found.add((m, n.value.split(":")[0]))
    assert found == ALLOWED_BY_NAME, sorted(found)


def test_experimental_modules_warn_on_import_and_say_their_status():
    import subprocess
    import sys

    from solvi import experimental
    mods = sorted(m for m in _modules() if m.startswith("solvi.experimental.") and m.count(".") == 2)
    names = {m.split(".")[2] for m in mods}
    assert names == set(experimental.STATUS), names ^ set(experimental.STATUS)
    for st in experimental.STATUS.values():
        assert {"since", "what", "missing", "script", "deadline"} <= set(st) and st["deadline"] == "1.2"
    code = ("import importlib, sys, warnings\nfrom solvi.core.catalog import ExperimentalWarning\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            "    for m in sys.argv[1:]:\n"
            "        importlib.import_module(m)\n"
            "said = ' '.join(str(x.message) for x in w if x.category is ExperimentalWarning)\n"
            "assert all(f'{m} is experimental' in said for m in sys.argv[1:]), said\n")
    run = subprocess.run([sys.executable, "-c", code, *mods], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    quiet = subprocess.run([sys.executable, "-W", "error", "-c", "import solvi, solvi.experimental"], capture_output=True)
    assert quiet.returncode == 0, quiet.stderr


# --- (rule 5) the public surfaces are pinned: a name added to or dropped from one is a decision, made here
SURFACES = {
    "solvi": ["build", "Guard", "Budget", "Catalog", "Question", "Answer", "System", "Response", "Quote", "Claim",
              "Decision", "Fail", "Unknown", "Span", "Maybe", "Rank", "Estimate", "Scale", "Bins",
              "SolviDeprecationWarning", "ExperimentalWarning"],          # LAYOUT §1; Agent, Knowledge come in L6
    "solvi.core": ["ExperimentalWarning", "NOT_STATED", "accept", "accepts", "Answer", "AnswerType", "bin_labels",
                   "Catalog", "check_evidence", "Claim", "cuts_number", "Decision", "evidence_rows", "find_quote",
                   "find_whole", "ground", "has_evidence", "locate", "NOT_STATED_KEY", "NotStated", "Part", "plain_json",
                   "PRIMITIVES", "Question", "question_data", "Quote", "Serial", "Unknown", "unknown_key", "unwrap",
                   "validated",
                   # the extension points (LAYOUT §3, lane L4): protocols, base classes and what they meet
                   "Scorer", "Decider", "Adapter", "Head", "Extractor", "Strategist", "DefaultStrategist", "Monitor",
                   "TraceStorage", "Response", "System", "Fail", "Proposer", "Space", "SlowPath", "AskPath",
                   "RefinePath", "SearchPath", "Thought", "Dispatcher", "Environment", "Outcome",
                   # knowledge (lane km10): "KnowledgeStore", "WriteGate", "ActionModel", "Vocabulary", "Agenda",
                   # "RiskPolicy" — pinned here by that lane
                   ],
    "solvi.models": ["cached", "cached_path", "decider", "DecideModel", "kind_of", "llm", "load", "ModelError",
                     "PUBLISHED", "pull", "resolve", "systemone"],
    "solvi.solutions.decisions": ["DecisionSystem", "build"],
    "solvi.solutions.guard": ["accepted_proposals", "accepts", "INTENTS", "model_from_json_schema", "arguments_from_user",
                              "arguments_grounded", "arguments_model", "arguments_valid", "AUTHORIZE_TASK",
                              "conversation", "Guard", "GuardDecision", "MATCHERS", "Message", "messages",
                              "no_injected_arguments", "no_instructions_in_tool_outputs", "proposal",
                              "request_authorizes", "same_url", "Session", "Tool", "ToolCall", "url_parts", "VERDICTS"],
    "solvi.experimental": ["DEADLINE", "STATUS", "mark", "warn_on_import"],
}


def test_the_public_surfaces_are_the_pinned_ones():
    import importlib
    import warnings
    for name, pinned in SURFACES.items():
        mod = importlib.import_module(name)
        assert sorted(mod.__all__) == sorted(pinned), (name, sorted(set(mod.__all__) ^ set(pinned)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for n in mod.__all__:
                assert getattr(mod, n) is not None, (name, n)
    import solvi
    assert solvi.build.__module__ == "solvi.solutions.decisions" and solvi.Guard.__module__ == "solvi.solutions.guard"


# --- (rule 6) the 1.0 shims resolve: every 0.9 module path and every name solvi no longer exports reach the same object,
# with a SolviDeprecationWarning (tests/test_layout_1_0.py imports each path). In 1.1 this flips: they are gone.
def test_the_names_solvi_no_longer_exports_resolve_with_a_warning():
    import importlib

    import pytest

    import solvi
    from solvi import _deprecate
    assert set(_deprecate.TOP_LEVEL_MOVED).isdisjoint(solvi.__all__)
    for name, home in _deprecate.TOP_LEVEL_MOVED.items():
        with pytest.warns(solvi.SolviDeprecationWarning, match=rf"solvi\.{name} moved in 1\.0: use {home}\.{name}"):
            obj = getattr(solvi, name)
        assert obj is getattr(importlib.import_module(home), name)


def test_every_old_path_names_a_module_that_exists_now():
    import importlib

    from solvi import _deprecate
    for old, new in _deprecate.old_paths().items():
        assert importlib.import_module(new).__name__ == new, (old, new)
