"""Catalog mutants: how many small faults in the catalog does a case suite catch?

`solvi test` says the cases pass; it does not say whether they would fail if the catalog were wrong. A mutant is the
catalog's task file with one small fault in it, of four kinds:

    flip       a comparison negated (<  ↔ >=, == ↔ !=, in ↔ not in, is ↔ is not)
    boundary   a comparison moved by its edge (< ↔ <=, > ↔ >=)
    constant   a threshold constant nudged up and down (a number in a comparison, an UPPER_CASE module constant)
    branch     a rule branch swapped (if/else bodies, a if c else b → b if c else a)
    drop       a check dropped (a @cat.check / @cat.constraint function always answers True; a guard
               "if ...: raise" never raises)

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

The mutant runs from a copy of the task's directory (top-level files) in a temporary directory, so a "system" factory
next to the task (`{"system": "run.py:system"}`) loads the mutant too; the original files are never written. A task in
another directory than the cases file is named in the copied cases file by its mutant's path. A factory elsewhere that
loads the task by its own path would run the original: keep it next to the task."""
from __future__ import annotations

import ast
import copy
import json
import shutil
import sys
import tempfile
import time
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
def _sites(tree):
    """Every place a mutant can be made → [(kind, node, k)] in source order; k picks the variant at that place."""
    out = []
    upper = set()
    for node in tree.body:                           # module constants: NAME = number, NAME all upper case
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id.isupper() and _num(node.value):
            upper.add(id(node.value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for k, op in enumerate(node.ops):
                if type(op) in _FLIP:
                    out.append(("flip", node, k))
                if type(op) in _EDGE:
                    out.append(("boundary", node, k))
            for c in [node.left, *node.comparators]:
                if _num(c):
                    upper.add(id(c))
        elif isinstance(node, ast.If) and node.orelse:
            out.append(("branch", node, 0))
        elif isinstance(node, ast.IfExp):
            out.append(("branch", node, 0))
        elif isinstance(node, ast.If) and not node.orelse and len(node.body) == 1 and isinstance(node.body[0], ast.Raise):
            out.append(("drop", node, 0))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(map(_check_decorator,
                                                                                    node.decorator_list)):
            out.append(("drop", node, 0))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and id(node) in upper:
            out += [("constant", node, 0), ("constant", node, 1)]
    out.sort(key=lambda s: (getattr(s[1], "lineno", 0), getattr(s[1], "col_offset", 0), KINDS.index(s[0]), s[2]))
    return out


def _num(node):
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


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
    if kind in ("flip", "boundary"):
        op = node.ops[k]
        new = (_FLIP if kind == "flip" else _EDGE)[type(op)]()
        node.ops[k] = new
        return f"{_SYM[type(op)]} → {_SYM[type(new)]}"
    if kind == "constant":
        old = node.value
        node.value = _nudge(old, up=(k == 0))
        return f"{old!r} → {node.value!r}"
    if kind == "branch":
        if isinstance(node, ast.IfExp):
            node.body, node.orelse = node.orelse, node.body
            return "a if c else b → b if c else a"
        body, orelse = node.body, node.orelse
        node.body, node.orelse = orelse, body
        return "if / else bodies swapped"
    if isinstance(node, ast.If):                       # drop a guard
        cond = ast.unparse(node.test)
        node.test = ast.Constant(False)
        return f"guard 'if {cond[:60]}: raise' never raises"
    doc = node.body[:1] if node.body and isinstance(node.body[0], ast.Expr) \
        and isinstance(getattr(node.body[0], "value", None), ast.Constant) else []
    node.body = [*doc, ast.Return(ast.Constant(True))]
    return f"check {node.name}() always passes"


def generate(source, kinds=KINDS):
    """The mutants of a task file's source → [Mutant] (with their sources), in line order."""
    tree = ast.parse(source)
    base = _sites(tree)
    out = []
    for i, (kind, node, k) in enumerate(base):
        if kind not in kinds:
            continue
        t2 = copy.deepcopy(tree)
        kind2, node2, _ = _sites(t2)[i]
        desc = _apply(t2, kind2, node2, k)
        out.append(Mutant(kind, getattr(node, "lineno", 0), desc, ast.unparse(ast.fix_missing_locations(t2))))
    return out


# --------------------------------------------------------------------------------------------------- running
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
    if fr.cases and all(c.status == "error" and any(p.startswith("prepare failed") for p in c.problems)
                        for c in fr.cases):
        return "stillborn", fr.cases[0].problems[0]
    bad = next((c for c in fr.cases if not c.ok), None)
    return ("killed", bad.name) if bad else ("survived", None)


def run(path, kinds=KINDS, timeout=5.0, limit=None, isolate=False, progress=None):
    """Mutants of the task a cases file tests, each run against the cases → Report.

    kinds: which faults to make (KINDS); timeout: seconds a case may take under a mutant (a mutant that makes a case
    run longer is killed: the case "timed out"); limit: at most that many mutants, spread evenly over the file;
    isolate: run each case in a fresh process (see run_file; a mutant that crashes the interpreter cannot stop the run,
    at a process start per case); progress: called with (i, n, mutant) after each one.
    The cases must pass on the catalog as it is (else ValueError: a failing suite kills every mutant)."""
    from . import _read, run_file
    path = Path(path).resolve()
    _, task_path, _ = _read(path)
    task_path = task_path.resolve()
    started = time.perf_counter()
    base = run_file(path, timeout=timeout, fast=True, isolate=isolate)
    if base.error or not base.ok:
        why = base.error or next(f"{c.name}: {c.problems[:1]}" for c in base.cases if not c.ok)
        raise ValueError(f"{path}: the cases fail on the catalog as it is ({why}); fix them before mutating")
    baseline_ms = (time.perf_counter() - started) * 1000
    ms = generate(task_path.read_text(), kinds)
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
                    fr = run_file(cases_copy, timeout=timeout, fast=True, isolate=isolate)
                    m.status, m.killed_by = _verdict(fr)
                except Exception as e:  # noqa: BLE001 — a mutant that breaks the runner itself cannot be built
                    m.status, m.killed_by = "stillborn", f"{type(e).__name__}: {e}"
                m.ms = (time.perf_counter() - t0) * 1000
                for name in [n for n in sys.modules if n.startswith("_solvi_task_")]:
                    mod = sys.modules[name]
                    if str(getattr(mod, "__file__", "")).startswith(tmp):
                        del sys.modules[name]    # a mutant's module does not outlive its run
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
    ap.add_argument("--timeout", type=float, default=5.0, metavar="S", help="seconds a case may take (default 5)")
    ap.add_argument("--limit", type=int, default=None, metavar="N", help="at most N mutants per file")
    ap.add_argument("--isolate", action="store_true", help="run each case in a fresh Python process")
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
    reports, failed = [], 0
    for f in files:
        try:
            reports.append(run(f, kinds, a.timeout, a.limit, a.isolate))
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
