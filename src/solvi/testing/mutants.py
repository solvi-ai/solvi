"""Catalog mutants: how many small faults in the catalog does a case suite catch?

`solvi test` says the cases pass; it does not say whether they would fail if the catalog were wrong. A mutant is the
catalog's task file with one small fault in it, of four kinds:

    flip       a boolean and/or swapped, a bare if/while/conditional test negated,
               or a comparison negated (<  ↔ >=, == ↔ !=, in ↔ not in, is ↔ is not)
    boundary   a comparison moved by its edge (< ↔ <=, > ↔ >=)
    constant   a signed threshold nudged up/down (comparison operands, UPPER_CASE module constants,
               including annotated assignments)
    branch     a rule branch swapped (if/else bodies, a if c else b → b if c else a)
    drop       a check dropped (a @cat.check / @cat.constraint function always answers True; a guard
               "if ...: raise" never raises; an assert always passes)

Each mutant is run against the cases with run_file(fast=True); it is "killed" when a case fails (or times out) and
"survived" when every case still passes — a fault the cases would let through. A mutant that cannot even be built (the
task does not load, or the System cannot be made) is "stillborn" and left out of the score:

    from solvi.testing import mutants
    rep = mutants.run("gallery/04_security_alert/cases.json")
    print(rep.score, rep.killed, rep.total)
    for m in rep.survivors():
        print(m.line, m.kind, m.description)

A survivor is either a gap in the cases (add one that tells the two apart) or an equivalent mutant (no input can tell
them apart: a constant nudged inside a range no realistic input reaches) — read it before writing a case.

Supported sites do not cover Python in general: match patterns, tuple/dict thresholds,
helper/call-form check registration, multi-statement raise guards and arbitrary returned
boolean expressions are not mutation operators. No static reachability is inferred;
use functions=["rule", "Rules.decide"] to exclude unrelated helpers explicitly.
Standard equality __main__ guards and their subtrees are never mutation sites.
Only identical resulting ASTs (and unchanged ASTs) are removed; semantically equivalent
but syntactically different mutants can still count separately.

The mutant runs from a copy of the task's directory (top-level files) in a temporary directory, so a "system" factory
next to the task (`{"system": "run.py:system"}`) loads the mutant too; the original files are never written. A task in
another directory than the cases file is named in the copied cases file by its mutant's path. A factory elsewhere that
loads the task by its own path would run the original: keep it next to the task."""
from __future__ import annotations

import ast
import copy
import json
import shutil
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

KINDS = ("flip", "boundary", "constant", "branch", "drop")
_FLIP = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt, ast.Eq: ast.NotEq, ast.NotEq: ast.Eq,
         ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_EDGE = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
_SYM = {ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">=", ast.Eq: "==", ast.NotEq: "!=", ast.In: "in",
        ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not"}
_CHECKS = ("check", "constraint")                    # the decorators of the functions "drop" makes always pass


@dataclass
class Mutant:
    """One fault: its kind, the line in the task file, what was changed, and after a run its status ("killed",
    "survived", "stillborn"), the first failing case and the time it took in ms."""
    kind: str
    line: int
    description: str
    source: str = field(repr=False, default="")
    status: str = ""
    killed_by: str | None = None
    ms: float = 0.0


@dataclass
class Report:
    """The mutants of one cases file after a run. score: killed / (killed + survived) — None when no mutant could be
    built; seconds: the whole run (the unmutated baseline included)."""
    path: Path
    task: Path
    mutants: list
    seconds: float = 0.0
    baseline_ms: float = 0.0

    def _n(self, st):
        return sum(m.status == st for m in self.mutants)

    @property
    def killed(self):
        return self._n("killed")

    @property
    def survived(self):
        return self._n("survived")

    @property
    def stillborn(self):
        return self._n("stillborn")

    @property
    def total(self):
        return self.killed + self.survived

    @property
    def score(self):
        return self.killed / self.total if self.total else None

    def survivors(self):
        return [m for m in self.mutants if m.status == "survived"]

    def by_kind(self):
        """{kind: [killed, survived, stillborn]}."""
        out = {}
        for m in self.mutants:
            row = out.setdefault(m.kind, [0, 0, 0])
            row[("killed", "survived", "stillborn").index(m.status)] += 1
        return out

    def to_dict(self):
        return {"path": str(self.path), "task": str(self.task), "score": self.score, "killed": self.killed,
                "survived": self.survived, "stillborn": self.stillborn, "seconds": round(self.seconds, 3),
                "by_kind": self.by_kind(),
                "survivors": [{"line": m.line, "kind": m.kind, "description": m.description} for m in self.survivors()]}


# --------------------------------------------------------------------------------------------------- generating
def _main_guard(node):
    if not isinstance(node, ast.If):
        return False
    for test in ast.walk(node.test):
        if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq):
            a, b = test.left, test.comparators[0]
            if any(isinstance(name, ast.Name) and name.id == "__name__" and
                   isinstance(value, ast.Constant) and value.value == "__main__"
                   for name, value in ((a, b), (b, a))):
                return True
    return False


def _scope_nodes(tree, functions=None):
    wanted = None if functions is None else set([functions] if isinstance(functions, str) else functions)
    if wanted is not None and (not wanted or any(not isinstance(f, str) or not f for f in wanted)):
        raise ValueError("functions must name at least one function")
    found, nodes = set(), []
    def walk(node, prefix="", active=wanted is None):
        if _main_guard(node):
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            qualified = prefix + node.name
            if not isinstance(node, ast.ClassDef) and wanted is not None and qualified in wanted:
                found.add(qualified)
                active = True
            prefix = qualified + "."
        if active:
            nodes.append(node)
        for child in ast.iter_child_nodes(node):
            walk(child, prefix, active)
    walk(tree)
    if wanted is not None and wanted - found:
        raise ValueError("unknown mutation function(s): " + ", ".join(sorted(wanted - found)))
    return nodes


def _sites(tree, functions=None):
    """Supported source sites; a main guard and its whole subtree are excluded."""
    out, upper = [], set()
    if functions is None:
        for node in tree.body:
            target = (node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else
                      node.target if isinstance(node, ast.AnnAssign) else None)
            if isinstance(target, ast.Name) and target.id.isupper() and _num(node.value):
                upper.add(id(node.value))
    nodes = _scope_nodes(tree, functions)
    for node in nodes:
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                if type(op) in _FLIP:
                    out.append(("flip", node, k))
                if type(op) in _EDGE:
                    out.append(("boundary", node, k))
            for c in [node.left, *node.comparators]:
                if _num(c):
                    upper.add(id(c))
        elif isinstance(node, ast.BoolOp):
            out.append(("flip", node, 0))
        if isinstance(node, (ast.If, ast.While, ast.IfExp)) and not isinstance(node.test, (ast.Compare, ast.Constant)):
            out.append(("flip", node, 0))
        if (isinstance(node, ast.If) and node.orelse) or isinstance(node, ast.IfExp):
            out.append(("branch", node, 0))
        elif isinstance(node, ast.If) and not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
            out.append(("drop", node, 0))
        elif isinstance(node, ast.Assert):
            out.append(("drop", node, 0))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(map(_check_decorator, node.decorator_list)):
            out.append(("drop", node, 0))
    for node in nodes:
        if id(node) in upper:
            out += [("constant", node, 0), ("constant", node, 1)]
    out.sort(key=lambda s: (getattr(s[1], "lineno", 0), getattr(s[1], "col_offset", 0), KINDS.index(s[0]), s[2]))
    return out


def _num(node):
    return (isinstance(node, ast.Constant) and type(node.value) in (int, float) or
            isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and
            isinstance(node.operand, ast.Constant) and type(node.operand.value) in (int, float))


def _check_decorator(d):
    f = d.func if isinstance(d, ast.Call) else d
    return isinstance(f, ast.Attribute) and f.attr in _CHECKS


def _nudge(v, up):
    if isinstance(v, int):
        return v + 1 if up else v - 1
    if v == 0:
        return 0.1 if up else -0.1
    return v * 1.1 if up else v * 0.9


def _apply(tree, kind, node, k):
    """Mutate `node` (a node of `tree`, a fresh copy) in place → a one-line description of the change."""
    if kind == "flip" and isinstance(node, ast.BoolOp):
        old = "and" if isinstance(node.op, ast.And) else "or"
        node.op = ast.Or() if old == "and" else ast.And()
        return f"{old} → {'or' if old == 'and' else 'and'}"
    if kind == "flip" and isinstance(node, (ast.If, ast.While, ast.IfExp)):
        node.test = ast.UnaryOp(ast.Not(), node.test)
        return "condition negated"
    if kind in ("flip", "boundary"):
        op = node.ops[k]
        new = (_FLIP if kind == "flip" else _EDGE)[type(op)]()
        node.ops[k] = new
        return f"{_SYM[type(op)]} → {_SYM[type(new)]}"
    if kind == "constant":
        old = ast.literal_eval(node)
        new = _nudge(old, up=(k == 0))
        if isinstance(node, ast.UnaryOp):
            node.op = ast.USub() if new < 0 else ast.UAdd()
            node.operand = ast.Constant(abs(new))
        else:
            node.value = new
        return f"{old!r} → {new!r}"
    if kind == "branch":
        if isinstance(node, ast.IfExp):
            node.body, node.orelse = node.orelse, node.body
            return "a if c else b → b if c else a"
        body, orelse = node.body, node.orelse
        node.body, node.orelse = orelse, body
        return "if / else bodies swapped"
    if isinstance(node, ast.Assert):
        node.test = ast.Constant(True)
        return "assert always passes"
    if isinstance(node, ast.If):                       # drop a guard
        cond = ast.unparse(node.test)
        node.test = ast.Constant(False)
        return f"guard 'if {cond[:60]}: raise' never raises"
    doc = node.body[:1] if node.body and isinstance(node.body[0], ast.Expr) \
        and isinstance(getattr(node.body[0], "value", None), ast.Constant) else []
    node.body = [*doc, ast.Return(ast.Constant(True))]
    return f"check {node.name}() always passes"


def generate(source, kinds=KINDS, *, functions=None):
    """One-fault sources in line order, excluding main guards and exact AST duplicates.

    functions optionally selects qualified names (e.g. ["rule", "Rules.decide"]),
    including their nested bodies, but excluding module constants and other helpers.
    This explicit source boundary is not System reachability or code coverage.
    Different ASTs can still be semantically equivalent and count separately.
    """
    if functions is not None and not isinstance(functions, str):
        functions = tuple(functions)
    tree = ast.parse(source)
    base = _sites(tree, functions)
    out, seen = [], {ast.dump(tree, include_attributes=False)}
    for i, (kind, node, k) in enumerate(base):
        if kind not in kinds:
            continue
        t2 = copy.deepcopy(tree)
        kind2, node2, _ = _sites(t2, functions)[i]
        desc = _apply(t2, kind2, node2, k)
        fingerprint = ast.dump(t2, include_attributes=False)
        if fingerprint not in seen:
            seen.add(fingerprint)
            out.append(Mutant(kind, getattr(node, "lineno", 0), desc, ast.unparse(ast.fix_missing_locations(t2))))
    return out


# --------------------------------------------------------------------------------------------------- running
def _inside(module, roots):
    file = getattr(module, "__file__", None)
    if not file:
        return False
    path = Path(file).resolve()
    return any(path.is_relative_to(root) for root in roots)


@contextmanager
def _imports(roots):
    """Temporarily own local imports, restoring the caller's modules afterwards."""
    roots = tuple(Path(root).resolve() for root in roots)
    saved = {name: mod for name, mod in list(sys.modules.items()) if _inside(mod, roots)}
    for name in saved:
        sys.modules.pop(name, None)
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        yield
    finally:
        for name, mod in list(sys.modules.items()):
            if _inside(mod, roots):
                sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.dont_write_bytecode = bytecode


def _mirror(cases_path, task_path, where):
    """Copy the top-level files of the cases file's directory (and the task's) into `where`/<dir name>/ → (the copied
    cases file, the path the mutant task goes to)."""
    src_dir = cases_path.parent
    dst = where / src_dir.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    cases_copy = dst / cases_path.name
    if task_path.parent.resolve() == src_dir.resolve():
        return cases_copy, dst / task_path.name
    mutant_task = where / "_task" / task_path.name
    mutant_task.parent.mkdir(parents=True, exist_ok=True)
    for f in task_path.parent.iterdir():
        if f.is_file():
            shutil.copy2(f, mutant_task.parent / f.name)
    data = json.loads(cases_path.read_text())
    data["task"] = str(mutant_task)
    if data.get("system"):
        file, _, fn = data["system"].partition(":")
        if not (dst / file).exists():
            data["system"] = f"{(src_dir / file).resolve()}:{fn}" if fn else str((src_dir / file).resolve())
    cases_copy.write_text(json.dumps(data))
    return cases_copy, mutant_task


def _verdict(fr):
    """A FileResult of a mutant → (status, the first failing case or the load error)."""
    if fr.error:
        return "stillborn", fr.error
    bad = next((c for c in fr.cases if not c.ok), None)
    return ("killed", bad.name) if bad else ("survived", None)


def _bounded(path, timeout, roots, stop_on_failure=False):
    """Bound the entire import/build/prepare/run lifecycle in a collected child process.

    Cases share this worker; never nest per-case workers, which could outlive it.
    """
    boot = ("import sys,json; req=json.loads(sys.stdin.read()); sys.path[:]=req['paths']; "
            "from solvi.testing.mutants import _worker; _worker(req)")
    with tempfile.TemporaryDirectory(prefix="solvi-mutant-result-") as directory:
        result = Path(directory) / "result.json"
        built = Path(directory) / "built"
        req = {"path": str(path), "paths": list(sys.path), "roots": [str(r) for r in roots],
               "result": str(result), "built": str(built), "stop_on_failure": stop_on_failure}
        # POSIX descendants inherit this worker's private process group. On Windows,
        # without a Job API, only the direct worker can be killed and collected.
        worker = subprocess.Popen([sys.executable, "-c", boot], stdin=subprocess.PIPE, text=True,  # noqa: S603 — fixed worker code, data on stdin
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                  start_new_session=(os.name == "posix"))
        try:
            worker.communicate(json.dumps(req), timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                worker.kill()
            worker.communicate()
            return "killed", f"timeout: lifecycle exceeded {timeout:g} s (worker killed and collected)"
        if not result.exists():
            status = "killed" if built.exists() else "stillborn"
            return status, f"worker exited without a result (exit {worker.returncode})"
        return tuple(json.loads(result.read_text()))


def _worker(req):
    from . import run_file
    built = Path(req["built"])
    try:
        with _imports(tuple(Path(r) for r in req["roots"])):
            verdict = _verdict(run_file(req["path"], timeout=None, fast=True, isolate=False,
                                        _stop_on_failure=req.get("stop_on_failure", False),
                                        _on_built=lambda: built.write_text("built")))
    except BaseException as exc:  # noqa: BLE001 — mutant SystemExit must be classified, not terminate its evaluator
        verdict = ("killed" if built.exists() else "stillborn", f"{type(exc).__name__}: {exc}")
    Path(req["result"]).write_text(json.dumps(verdict))


def run(path, kinds=KINDS, timeout=5.0, limit=None, isolate=False, progress=None, *, functions=None):
    """Mutants of the task a cases file tests, each run against the cases → Report.

    kinds: which faults to make (KINDS); timeout: finite positive wall-clock seconds for each whole
    lifecycle (Python startup, import, factory, prepare and all cases), including the baseline.
    An over-budget mutant is killed; an over-budget baseline is refused. Every lifecycle runs in a
    fresh process which is killed and collected on expiry. Cases share that process; isolate is
    retained for compatibility and now uses this same bounded worker in both modes.
    limit: positive integer, at most that many mutants, spread evenly over the file;
    functions: optional qualified function names, as in generate(); no reachability inference;
    progress: called with (i, n, mutant) after each one.
    The cases must pass on the catalog as it is (else ValueError: a failing suite kills every mutant)."""
    from . import _read
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError("limit must be a positive integer")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive lifecycle budget in seconds")
    path = Path(path).resolve()
    _, task_path, _ = _read(path)
    task_path = task_path.resolve()
    started = time.perf_counter()
    roots = (path.parent, task_path.parent)
    status, why = _bounded(path, timeout, roots)
    if status != "survived":
        raise ValueError(f"{path}: the cases fail on the catalog as it is ({why}); fix them before mutating")
    baseline_ms = (time.perf_counter() - started) * 1000
    ms = generate(task_path.read_text(), kinds, functions=functions)
    if limit is not None and len(ms) > limit:
        step = len(ms) / limit
        ms = [ms[int(i * step)] for i in range(limit)]
    extra = [str(path.parent.parent), str(task_path.parent.parent)]
    with tempfile.TemporaryDirectory(prefix="solvi-mutants-") as tmp:
        cases_copy, mutant_task = _mirror(path, task_path, Path(tmp))
        saved = list(sys.path)
        for p in extra:
            sys.path.append(p)                       # siblings a factory imports from its parent (gallery/_audit.py)
        paths = list(sys.path)
        try:
            for i, m in enumerate(ms):
                sys.path[:] = paths                  # what a factory inserted for the last mutant goes
                t0 = time.perf_counter()
                mutant_task.write_text(m.source)
                try:
                    m.status, m.killed_by = _bounded(cases_copy, timeout, (*roots, Path(tmp)), stop_on_failure=True)
                except Exception as e:  # noqa: BLE001 — a mutant that breaks the runner itself cannot be built
                    m.status, m.killed_by = "stillborn", f"{type(e).__name__}: {e}"
                m.ms = (time.perf_counter() - t0) * 1000
                if progress:
                    progress(i + 1, len(ms), m)
        finally:
            sys.path[:] = saved
    return Report(path, task_path, ms, time.perf_counter() - started, baseline_ms)


def main(argv=None):
    """`solvi mutants PATH...`: the mutation score of each cases file's suite and its surviving mutants."""
    import argparse
    from . import find
    ap = argparse.ArgumentParser(prog="solvi mutants", description="Make small faults in a catalog (flipped and moved "
                                 "comparisons, nudged constants, swapped branches, dropped checks) and report which "
                                 "ones the cases do not catch.")
    ap.add_argument("paths", nargs="+", help="cases files, or directories to search for cases.json / *.cases.json")
    ap.add_argument("--kinds", default=",".join(KINDS), help=f"comma-separated subset of {', '.join(KINDS)}")
    ap.add_argument("--timeout", type=float, default=5.0, metavar="S", help="wall-clock seconds per complete mutant lifecycle (default 5)")
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="at most N mutants per file")
    ap.add_argument("--isolate", action="store_true", help="compatibility flag; each mutant already runs in a bounded fresh process")
    ap.add_argument("--json", action="store_true", help="print the reports as JSON")
    a = ap.parse_args(argv)
    kinds = tuple(k.strip() for k in a.kinds.split(",") if k.strip())
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        print(f"solvi mutants: unknown kind(s) {bad}; known: {', '.join(KINDS)}", file=sys.stderr)
        return 2
    try:
        files = find(a.paths)
    except FileNotFoundError as e:
        print(f"solvi mutants: {e}", file=sys.stderr)
        return 2
    if not files:
        print("solvi mutants: no cases files found", file=sys.stderr)
        return 2
    reports, failed = [], 0
    for f in files:
        try:
            report = run(f, kinds, a.timeout, a.limit, a.isolate)
            reports.append(report)
            if not report.mutants:
                print(f"solvi mutants: {report.task}: no mutation sites selected; "
                      "imported catalogs are not mutated (check --kinds and --limit)", file=sys.stderr)
                failed += 1
        except (ValueError, FileNotFoundError) as e:
            print(f"solvi mutants: {e}", file=sys.stderr)
            failed += 1
    if a.json:
        print(json.dumps([r.to_dict() for r in reports], indent=1, ensure_ascii=False))
    else:
        for r in reports:
            sc = "n/a" if r.score is None else f"{r.score:.2f}"
            print(f"{r.path}: score {sc} ({r.killed} killed, {r.survived} survived, {r.stillborn} stillborn) "
                  f"in {r.seconds:.1f} s")
            for m in r.survivors():
                print(f"  survived  line {m.line:<4} {m.kind:<9} {m.description}")
    return 2 if failed else 0


__all__ = ["KINDS", "Mutant", "Report", "generate", "main", "run"]
