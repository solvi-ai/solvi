"""The import structure of solvi, read from the source (ast), so that a cycle or a layer break fails here (LAYOUT §6,
on a provisional tier map over the flat module names until the 1.0 package move):

- no import cycle at module level (the imports a module runs when it is imported, nested in `if` / `try` included,
  `if TYPE_CHECKING:` and function bodies not): such a cycle makes the import order matter and breaks on a refactor;
- the execution layer (core, typed, provenance, primitives, runtime, schema, textin) imports nothing above it, not even
  inside a function: what it needs of the modules above (the flow, the replay of their records) is defined in it;
- nothing below the server and agent layer (solvi.cli, solvi.serve, solvi.hooks, solvi.agents) imports from it;
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
TOP = ("solvi.cli", "solvi.serve", "solvi.hooks", "solvi.__main__", "solvi.agents")


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
    is inside them: they are being imported already)."""
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
                    if x != m and not m.startswith(x + ".") and x in mods:
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
        return any(m == p or m.startswith(p + ".") for p in TOP)
    g = graph(module_level=False)
    up = {f"{m} → {t}" for m, ts in g.items() if not top(m) for t in ts if top(t)}
    assert not up, f"a library module imports the server / agent layer: {sorted(up)}"


# --- the provisional tier map (LAYOUT §2 / §6) over the flat 1.0 module names; the package move (solvi.core.*) follows
LOW_TIERS = {
    "kernel": "_deprecate _loader _migrate core sets _rpc",
    "parts": "decide heads llm remote longdoc multi perturb rulelist systemone extract_long extract_multi strategist strategy "
             "inputs",
    "guarantees": "guarantee drift openset",
    "records": "storage response signature audit diff report sysreport",
    "system": "system",
    "deliberate": "agree generate refine search dispatch",
    "knowledge": "worldmap episode memory",
}
HIGH = "__init__ __main__ auto calibrate check cli command honesty models scaffold serve show testing agents"
EXPERIMENTAL = "compile sandbox hooks learning lora specialist charts agents.mcp oncalib"
# stable → experimental imports that are meant: the command-line entries for an experimental feature (LAYOUT §6), and
# auto's compiled slow path (LAYOUT risk 3: `slow=` will take a compiled System; until then it is listed here)
ALLOWED_EXPERIMENTAL = {("solvi.serve", "solvi.agents.mcp"), ("solvi.auto", "solvi.compile")}
# the import cycles left in 1.0, each inside one tier
INTRA_TIER_CYCLES = [
    {"solvi.core.catalog", "solvi.core.primitives", "solvi.core.provenance", "solvi.core.runtime", "solvi.core.schema",
     "solvi.core.textin", "solvi.core.types"},
    {"solvi.storage", "solvi.response", "solvi.report", "solvi.signature"},
    {"solvi.agents.guard", "solvi.agents.confirm"},
]


def _level(m):
    """A module → ("low", tier index) / ("high", None) / ("experimental", None), by its longest listed prefix."""
    name = "__init__" if m == "solvi" else m.split(".", 1)[1]
    table = {x: ("experimental", None) for x in EXPERIMENTAL.split()}
    table.update({x: ("high", None) for x in HIGH.split()})
    for i, (_, mods) in enumerate(LOW_TIERS.items()):
        table.update({x: ("low", i) for x in mods.split()})
    table["agents.mcp"] = ("experimental", None)
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
            elif lm == "high" and lt == "experimental" and (m, t) not in ALLOWED_EXPERIMENTAL:
                bad.append(f"{m} (high) → {t} (experimental)")
    assert not bad, f"imports against the tier order: {sorted(bad)}"


def test_the_names_moved_by_the_import_cycle_lane_are_the_same_objects():
    """The cycle moves of 1.0 (LAYOUT §6) kept the names the old modules re-export: the same objects."""
    import solvi
    from solvi import _rpc, calibrate, decide, dispatch, response, serve, storage, system
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
    for name in ("add_parser", "cmd_calibrate", "examples_of", "find_part", "label_of", "read_rows"):
        assert getattr(calibfile, name) is getattr(calibrate, name)
