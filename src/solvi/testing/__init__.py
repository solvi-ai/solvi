"""Decision regression tests from cases.json files — the format the gallery entries use — for `solvi test` and pytest.

A cases.json sits next to the task.py it tests (a module defining `cat` and `QUESTIONS`, or `system()`, and optionally
`prepare(state)`), or names it: {"task": "path/to/task.py", "cases": [...]}. When the System needs more than the catalog
(a head fitted or a rule list learned from examples), {"system": "run.py:system", "cases": [...]} names the function that
builds it (paths relative to the cases file). A case:

    {"name": "legal threat",
     "state": {...},                                       # init_state (prepare(state) runs first, if the task has one)
     "expected": {"priority": "high", "tags": ["refund_request"], "route": null},
     "status": {"priority": "forced"},                     # optional: ok / forced / abstain per question
     "safeguards": {"priority": ["hard_check"]},           # optional: the safeguard kinds that fire for a question (exactly)
     "ask": ["priority", "route"]}                         # optional: ask only these questions (default: all)

(A honesty set names its right answers "expected" too, so one file serves `solvi test` and `solvi honesty`; "gold", the
0.7 key, was removed in 0.9.)
An expected answer is written as in solvi's JSON: an option, a list for multi-label, a number, "<not stated>" for
solvi.Unknown; null or "abstain" means the question must abstain. Every response's trace must also replay.

    from solvi.testing import run_path
    for file in run_path("gallery"):
        print(file.path, file.passed, [c.problems for c in file.cases if not c.ok])

`fuzz(system, state)` asks with random, missing and wrong-typed fields and reports exceptions that escape solvi and answers
given with a confidence that is not a number in [0, 1]. See docs/testing.md."""
from __future__ import annotations

import copy
import json
import random
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from .honesty import gold_of, load_task, same, system_of
from .honesty import _plain

ABSTAIN_WORDS = ("abstain",)


# --------------------------------------------------------------------------------------------------- finding and loading
def is_cases_file(path):
    """Is this a cases file (cases.json, or <anything>.cases.json)?"""
    p = Path(path)
    return p.name == "cases.json" or p.name.endswith(".cases.json")


def find(paths):
    """Files and directories → the cases files among / under them, sorted."""
    out = []
    for p in [paths] if isinstance(paths, (str, Path)) else paths:
        p = Path(p)
        if p.is_dir():
            out += [f for f in sorted(p.rglob("*.json")) if is_cases_file(f) and ".venv" not in f.parts]
        elif p.exists():
            out.append(p)
        else:
            raise FileNotFoundError(p)
    return out


@dataclass
class Suite:
    """A cases file, loaded: its task module, the cases, and where the task came from."""
    path: Path
    cases: list
    task: object
    task_path: Path
    factory: str | None = None                       # "module.py:function" that builds the System (the "system" key)

    def system(self):
        """A fresh System for this task: the file's "system" factory, else task.system(), else System(task.cat,
        task.QUESTIONS)."""
        if self.factory:
            file, _, fn = self.factory.partition(":")
            import sys
            where = (self.path.parent / file).resolve()
            mod = load_task(where)
            sys.path.insert(0, str(where.parent))    # the factory may import its neighbours when it runs
            try:
                return getattr(mod, fn or "system")()
            finally:
                sys.path.remove(str(where.parent))
        return system_of(self.task)

    def state(self, case):
        """The case's init_state: a fresh copy, through the task's prepare() if it has one."""
        st = copy.deepcopy(case["state"])
        prep = getattr(self.task, "prepare", None)
        return prep(st) if callable(prep) else st


def load(path):
    """A cases file → Suite. The task is {"task": ...} in the file (relative to it), else task.py next to it."""
    path = Path(path)
    data = json.loads(path.read_text())
    if isinstance(data, list):
        cases, task, factory = data, None, None
    else:
        cases, task, factory = data.get("cases", []), data.get("task"), data.get("system")
    task_path = (path.parent / task) if task else path.parent / "task.py"
    if not task_path.exists():
        raise FileNotFoundError(f"{path}: no task module (put a task.py next to it, or name it: "
                                "{\"task\": ..., \"cases\": ...})")
    return Suite(path, cases, load_task(task_path.resolve()), task_path, factory)


# --------------------------------------------------------------------------------------------------- checking
@dataclass
class CaseResult:
    """One case of a cases.json file after `solvi test`: its name, the problems found (none: `ok`), the answers given
    {question: [answer, status]} and the time it took in ms."""
    name: str
    problems: list = field(default_factory=list)     # one line per mismatch (or the crash)
    answers: dict = field(default_factory=dict)      # question → [answer, status]
    ms: float = 0.0

    @property
    def ok(self):
        return not self.problems


@dataclass
class FileResult:
    """One cases.json file after `solvi test`: its CaseResults, `passed` (how many are ok), `ok` (all of them, and the
    file loaded), `error` (why the file or its task could not be loaded)."""
    path: Path
    cases: list = field(default_factory=list)
    error: str | None = None                         # the file or its task could not be loaded

    @property
    def passed(self):
        return sum(c.ok for c in self.cases)

    @property
    def ok(self):
        return self.error is None and all(c.ok for c in self.cases)


def _abstains(want, answer_type):
    return want is None or (isinstance(want, str) and want in ABSTAIN_WORDS and want not in answer_type.options)


def _show(v):
    return _plain(v)


def _store_kw(system, store):
    """store= for System.ask (an object with a plain ask(state), as fuzz also takes, gets nothing)."""
    import inspect
    try:
        return {"store": store} if "store" in inspect.signature(system.ask).parameters else {}
    except (TypeError, ValueError):
        return {}


CASE_KEYS = ("name", "state", "expected", "status", "safeguards", "ask", "note")


def run_case(system, case, state, store=False):
    """Ask one case and compare → CaseResult. Nothing is written to the system's own storage (a test input is not a
    decision) unless store=True. Exceptions from ask are reported as a crash, not raised. A case that cannot
    fail is a problem too: a key that is not one of CASE_KEYS (a misspelled "expcted"), a status or safeguards entry for
    a question that was not asked, a case that expects nothing."""
    name = case.get("name", "?")
    out = CaseResult(name)
    odd = sorted(k for k in case if k not in CASE_KEYS and k != "gold")
    if odd:
        out.problems.append(f"unknown key(s) {', '.join(map(repr, odd))} (a case has: {', '.join(CASE_KEYS)})")
    if "gold" in case:
        out.problems.append('"gold" is the 0.7 key of the right answers (removed in 0.9): name them "expected"')
    expected = case.get("expected")
    if not (expected or case.get("status") or case.get("safeguards") or isinstance(case.get("safeguards"), list)):
        out.problems.append("the case expects nothing: give \"expected\", \"status\" or \"safeguards\"")
    try:
        res = system.ask(state, case.get("ask"), **_store_kw(system, store))
    except Exception as e:  # noqa: BLE001
        out.problems.append(f"crash: {type(e).__name__}: {e}\n{traceback.format_exc(limit=-3)}")
        return out
    out.ms = res.ms
    out.answers = {q: [_show(r.answer), r.status] for q, r in res.results.items()}
    for q, want in (expected or {}).items():
        if q not in res.results:
            out.problems.append(f"{q}: not asked (unknown question, or left out by \"ask\")")
            continue
        r, at = res[q], system.questions[q].answer
        if _abstains(want, at):
            if r.status != "abstain":
                out.problems.append(f"{q}: expected an abstention, got {_show(r.answer)!r} [{r.status}]")
        elif r.status == "abstain":
            out.problems.append(f"{q}: expected {want!r}, abstained ({r.why})")
        elif not same(r.answer, gold_of(at, want)):
            out.problems.append(f"{q}: expected {want!r}, got {_show(r.answer)!r} [{r.status}]")
    for q, st in (case.get("status") or {}).items():
        if q not in res.results:
            out.problems.append(f"status of {q}: not asked (unknown question, or left out by \"ask\")")
        elif res[q].status != st:
            out.problems.append(f"{q}: expected status {st}, got {res[q].status} ({res[q].why})")
    fired = {}
    for e in res.safeguards or ():
        for q in e["questions"]:
            fired.setdefault(q, set()).add(e["kind"])
    want_sg = case.get("safeguards") or {}
    if isinstance(want_sg, list):                    # a list: the safeguards of the whole response
        got = sorted({e["kind"] for e in res.safeguards or ()})
        if got != sorted(set(want_sg)):
            out.problems.append(f"safeguards: expected {sorted(set(want_sg))}, fired {got}")
    else:
        for q, kinds in want_sg.items():
            if q not in res.results:
                out.problems.append(f"safeguards of {q}: not asked (unknown question, or left out by \"ask\")")
                continue
            got = sorted(fired.get(q, ()))
            if got != sorted(set(kinds)):
                out.problems.append(f"{q}: expected safeguards {sorted(set(kinds))}, fired {got}")
    rep = res.trace.replay(system)
    if not rep["ok"]:
        out.problems.append(f"trace replay failed: {rep['mismatches'][:3]}")
    return out


def run_file(path, fuzz_n=0, seed=0, store=False):
    """Run every case of a cases file → FileResult (with fuzz_n > 0, each case is also fuzzed; crashes are problems).
    store=True: the cases (never the fuzz mutations) are saved to the system's storage, when it has one."""
    path = Path(path)
    out = FileResult(path)
    try:
        suite = load(path)
        system = suite.system()
    except Exception as e:  # noqa: BLE001
        out.error = f"{type(e).__name__}: {e}"
        return out
    for i, case in enumerate(suite.cases):
        try:
            state = suite.state(case)
        except Exception as e:  # noqa: BLE001
            out.cases.append(CaseResult(case.get("name", f"#{i + 1}"), [f"prepare failed: {type(e).__name__}: {e}"]))
            continue
        r = run_case(system, case, state, store)
        if fuzz_n:
            r.problems += [f"fuzz {c}" for c in fuzz(system, state, fuzz_n, seed + i)]
        out.cases.append(r)
    return out


def run_path(paths, fuzz_n=0, seed=0, store=False):
    """Every cases file under the paths → [FileResult]."""
    return [run_file(f, fuzz_n, seed, store) for f in find(paths)]


# --------------------------------------------------------------------------------------------------- fuzzing
_ODD = [None, "", "x" * 3, "🙂 ünïcode", -1, 0, 10 ** 12, float("nan"), float("inf"), True, [], {}, [None], {"": None}]


def mutations(state, n=50, seed=0):
    """n mutated copies of a state → [(description, state)]: a field removed, set to None, to a value of another type
    (number ↔ text, a list, a dict, NaN, a huge number), text cut short or garbled, an unexpected extra field; one level
    into dicts and lists too. Deterministic for a seed."""
    rng = random.Random(seed)
    keys = list(state)
    out = []
    for _ in range(n):
        st = copy.deepcopy(state)
        op = rng.choice(["drop", "none", "odd", "odd", "text", "extra", "deep"]) if keys else "extra"
        k = rng.choice(keys) if keys else None
        if op == "drop":
            del st[k]
            what = f"removed {k!r}"
        elif op == "none":
            st[k] = None
            what = f"{k!r} = None"
        elif op == "odd":
            v = rng.choice(_ODD)
            st[k] = copy.deepcopy(v)
            what = f"{k!r} = {v!r}"
        elif op == "text":
            v = st[k]
            s = v if isinstance(v, str) else json.dumps(v, default=str)
            cut = s[:rng.randint(0, max(0, len(s) - 1))]
            st[k] = cut if rng.random() < 0.5 else "".join(rng.sample(s, len(s))) if s else "?"
            what = f"{k!r} as garbled text"
        elif op == "extra":
            st["__fuzz__"] = rng.choice(_ODD)
            what = "an unexpected extra field"
        else:
            v = st[k]
            if isinstance(v, dict) and v:
                kk = rng.choice(list(v))
                v[kk] = copy.deepcopy(rng.choice(_ODD))
                what = f"{k!r}[{kk!r}] = {v[kk]!r}"
            elif isinstance(v, list) and v:
                j = rng.randrange(len(v))
                v[j] = copy.deepcopy(rng.choice(_ODD))
                what = f"{k!r}[{j}] = {v[j]!r}"
            else:
                st[k] = [v]
                what = f"{k!r} wrapped in a list"
        out.append((what, st))
    return out


def fuzz(system, state, n=50, seed=0):
    """Ask with n mutated states (see mutations) → one line per problem: "crash — mutation: exception" for an exception
    that escaped System.ask, "invalid — mutation: ..." for an answer given with a confidence that is not a number in
    [0, 1] (NaN, inf). Abstentions are fine. Numeric warnings the mutations provoke are silenced."""
    import math
    import warnings
    out = []
    for what, st in mutations(state, n, seed):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = system.ask(st, **_store_kw(system, False))      # a mutated input is never a stored decision
        except Exception as e:  # noqa: BLE001
            out.append(f"crash — {what}: {type(e).__name__}: {e}")
            continue
        for q, r in res.results.items():
            c = r.confidence
            if r.status != "abstain" and not (isinstance(c, (int, float)) and math.isfinite(c) and 0 <= c <= 1):
                out.append(f"invalid — {what}: {q} answered {_show(r.answer)!r} with confidence {c!r}")
    return out


# --------------------------------------------------------------------------------------------------- the command
def main(argv=None):
    import argparse
    import sys
    ap = argparse.ArgumentParser(prog="solvi test", description="Run decision regression tests from cases.json files "
                                 "(next to a task.py): expected answers, statuses, safeguards, trace replay.")
    ap.add_argument("paths", nargs="+", help="cases files, or directories to search for cases.json / *.cases.json")
    ap.add_argument("--fuzz", type=int, default=0, metavar="N", help="also ask N mutated inputs per case; an exception "
                    "escaping solvi, or an answer with a NaN / out-of-range confidence, fails the case")
    ap.add_argument("--seed", type=int, default=0, help="fuzzing seed (default 0)")
    ap.add_argument("--store", action="store_true", help="save the cases' decisions to the system's own storage "
                                                         "(default: test inputs are not stored)")
    ap.add_argument("--json", action="store_true", help="print the results as JSON")
    ap.add_argument("-q", "--quiet", action="store_true", help="print failures and the summary only")
    a = ap.parse_args(argv)
    try:
        files = run_path(a.paths, a.fuzz, a.seed, a.store)
    except FileNotFoundError as e:
        print(f"solvi test: {e}", file=sys.stderr)
        return 2
    if not files:
        print("solvi test: no cases files found", file=sys.stderr)
        return 2
    n = sum(len(f.cases) for f in files)
    passed = sum(f.passed for f in files)
    bad_files = [f for f in files if f.error]
    if a.json:
        from ..core.schema import dumps
        print(dumps({"files": [{"path": str(f.path), "error": f.error, "cases": [
            {"name": c.name, "ok": c.ok, "problems": c.problems, "answers": c.answers} for c in f.cases]} for f in files],
            "cases": n, "passed": passed, "ok": passed == n and not bad_files}, indent=1, ensure_ascii=False, default=str))
    else:
        for f in files:
            if f.error:
                print(f"ERROR {f.path}: {f.error}")
                continue
            if not a.quiet or not f.ok:
                print(f"{f.path}: {f.passed}/{len(f.cases)}")
            for c in f.cases:
                if not c.ok:
                    print(f"  FAIL {c.name}")
                    for p in c.problems:
                        print("       " + p.replace("\n", "\n       ").rstrip())
                elif not a.quiet:
                    print(f"  ok   {c.name}")
        print(f"{passed}/{n} cases passed in {len(files)} file(s)"
              + (f", {len(bad_files)} could not be loaded" if bad_files else ""))
    return 0 if passed == n and not bad_files else 1


__all__ = ["CaseResult", "FileResult", "find", "fuzz", "is_cases_file", "load", "main", "mutations",
           "run_case", "run_file", "run_path", "Suite"]
