"""A change's code run on test inputs in an isolated process (new; opt-in): IsolatedTests, a stage over
solvi.experimental.compile.sandbox.run.

    st = IsolatedTests([{"id": "a", "input": [2, 3], "expected": 5}, ...], entry="add")
    Pipeline([KnownFailures(), Static(), st, ...], storage, ...)

Static only checks a change's code (the compiler's sandbox check); this stage runs it: a fresh `python -I` subprocess,
an empty environment, limits on memory, CPU and wall time, a per-input alarm, the same restricted builtins. Only the
inputs go to the child; the expected answers stay in this process and are compared here, strictly (the same type and
the same value: 5.0 is not 5, 1 is not True), so a candidate cannot read the answers it is tested on."""
from __future__ import annotations

import json
from typing import Any, Callable

from ._stages import StageVerdict

DRIVER = "solvi.experimental.changes._isolated:tests_main"


def tests_main(ns, payload, signal):
    """The sandbox driver (runs in the child): call the entry function on every input → {"outputs": [{"value"} |
    {"error"}]}. An input is the argument list (a list), the keyword arguments (a dict), or {"arg": x} for one argument."""
    fn = ns.get(payload["entry"])
    if not callable(fn):
        return {"outputs": [{"error": f"no function {payload['entry']!r} in the module"}] * len(payload["inputs"])}
    out = []
    for x in payload["inputs"]:
        signal.alarm(int(payload["item_s"]))
        try:
            v = fn(**x) if isinstance(x, dict) else fn(*x)
            json.dumps(v)                                         # a value that does not travel back is an error
            out.append({"value": v})
        except BaseException as e:  # noqa: BLE001 — anything the candidate raises (the alarm too) is its failure
            out.append({"error": f"{type(e).__name__}: {e}"[:300]})
        finally:
            signal.alarm(0)
    return {"outputs": out}


def strict_equal(a, b):
    """The same JSON type and value, recursively (1 is not True, 5.0 is not 5; NaN equals nothing)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    return a == b


def _case(c, i):
    """A case → (id, input as the child gets it, expected)."""
    if isinstance(c, dict):
        cid, x, want = str(c.get("id", f"#{i}")), c["input"], c["expected"]
    else:
        (x, want), cid = c, f"#{i}"
    if isinstance(x, tuple):
        x = list(x)
    if not isinstance(x, (list, dict)):
        raise TypeError(f"case {cid}: an input is the argument list or the keyword arguments, not {type(x).__name__}")
    return cid, x, json.loads(json.dumps(want))                   # compared as it would come back from the child


class IsolatedTests:
    """The candidate's code (`change.code`, or `source(change, ctx)`) run on test cases in an isolated process (new;
    opt-in).

    cases: [{"id"?, "input", "expected"}] or [(input, expected)] — an input is the entry function's argument list (a
    list or tuple) or its keyword arguments (a dict); values are JSON — or `cases(change, ctx)` → such a list. entry:
    the function of the module that is called. Only the inputs are sent to the child; the expected answers are compared
    in this process by `compare(got, expected)` → bool (default strict_equal: the same type and value).
    Limits: mem_mb, cpu_s and wall_s for the whole run, item_s per input (an input that runs longer raises in the
    child and fails). allowed: more modules the code may import (as for the sandbox's check).

    Refused when the code does not load (the sandbox's check refuses it, or it raises on import), the run crashes or
    exceeds a limit, or more than `max_failed` cases fail (a wrong answer, an exception, a value that is not JSON); a
    change with no code is refused ("no code to test"). A failure of the sandbox itself (the interpreter, the driver) is
    not a verdict on the change: the stage raises, and the pipeline ends the proposal with status "error". The
    verdict's numbers: n, passed, failed, and the first failing ids with what came back. runner: the sandbox's run
    (replaceable for tests). cost: the stage's cost for Pipeline(order="cheap_first")."""
    name = "isolated_tests"

    def __init__(self, cases, *, entry: str = "main", source: Callable | None = None, compare: Callable | None = None,
                 max_failed: int = 0, mem_mb: int = 1024, cpu_s: int = 60, wall_s: int = 120, item_s: int = 5,
                 allowed=(), runner: Callable | None = None, cost: float = 100.0):
        if int(max_failed) < 0:
            raise ValueError(f"max_failed is a count >= 0, not {max_failed!r}")
        self.cases, self.entry, self.source = cases, str(entry), source
        self.compare = compare or strict_equal
        self.max_failed = int(max_failed)
        self.mem_mb, self.cpu_s, self.wall_s, self.item_s = int(mem_mb), int(cpu_s), int(wall_s), int(item_s)
        self.allowed, self.cost = tuple(allowed), float(cost)
        self._runner = runner

    def _run(self, source, payload):
        if self._runner is not None:
            return self._runner(source, DRIVER, payload, mem_mb=self.mem_mb, cpu_s=self.cpu_s, wall_s=self.wall_s,
                                allowed=self.allowed)
        import warnings
        with warnings.catch_warnings():                       # this module's own import already warned
            warnings.simplefilter("ignore")
            from ..compile import sandbox
        return sandbox.run(source, DRIVER, payload, mem_mb=self.mem_mb, cpu_s=self.cpu_s, wall_s=self.wall_s,
                           allowed=self.allowed)

    def run(self, change, ctx):
        src = self.source(change, ctx) if self.source is not None else change.code
        if not src:
            return StageVerdict.refuse("no code to test")
        cases = list(self.cases(change, ctx) if callable(self.cases) else self.cases)
        if not cases:
            return StageVerdict.refuse("no test cases")
        rows = [_case(c, i) for i, c in enumerate(cases)]
        r = self._run(src, {"entry": self.entry, "inputs": [x for _, x, _ in rows], "item_s": self.item_s})
        if "sandbox_error" in r:
            raise RuntimeError(f"the sandbox failed (not the change's failure): {r['sandbox_error']}")
        n = len(rows)
        if "outputs" not in r or len(r["outputs"]) != n:
            what = r.get("load_error") or r.get("crash") or "no result"
            verb = "did not load" if "load_error" in r else "did not run"
            return StageVerdict.refuse(f"the code {verb}: {what}", {"n": n, "passed": 0, "failed": n})
        bad: list[Any] = []
        for (cid, _, want), o in zip(rows, r["outputs"]):
            if "error" in o:
                bad.append({"id": cid, "error": o["error"]})
            elif not self.compare(o["value"], want):
                bad.append({"id": cid, "got": o["value"]})
        m = {"n": n, "passed": n - len(bad), "failed": len(bad), "failures": bad[:10]}
        if len(bad) > self.max_failed:
            return StageVerdict.refuse(f"{len(bad)} of {n} test(s) failed in isolation (allowed {self.max_failed}): "
                                       f"{[b['id'] for b in bad[:5]]}", m)
        return StageVerdict.admit(f"{n - len(bad)} of {n} test(s) passed in isolation", m)


__all__ = ["DRIVER", "IsolatedTests", "strict_equal", "tests_main"]
