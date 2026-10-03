"""The import structure of solvi, read from the source (ast), so that a cycle or a layer break fails here:

- no import cycle at module level (the imports a module runs when it is imported, nested in `if` / `try` included,
  `if TYPE_CHECKING:` and function bodies not): such a cycle makes the import order matter and breaks on a refactor;
- the execution layer (core, typed, provenance, primitives, runtime, schema, textin) imports nothing above it, not even
  inside a function: what it needs of the modules above (the flow, the replay of their records) is defined in it;
- nothing below the server and agent layer (solvi.cli, solvi.serve, solvi.hooks, solvi.agents) imports from it.

Inside a function, importing the package root itself (`from . import __version__`) is not counted: the root has finished
importing before any function of a submodule runs."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
BASE = {f"solvi.{m}" for m in ("core", "typed", "provenance", "primitives", "runtime", "schema", "textin")}
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
