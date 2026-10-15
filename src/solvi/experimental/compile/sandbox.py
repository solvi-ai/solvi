"""A sandbox for code a model wrote: only pure functions over facts.

    from solvi.experimental.compile.sandbox import check, load, run
    check(source)                                   # [] when allowed, else the reasons it is refused
    ns = load(source)                               # the module's namespace, in this process (refuses what check refuses)
    out = run(source, "mypkg.driver:main", payload) # the module run in a subprocess by a trusted driver → its JSON

`check` and `load` are the stable solvi.core.versions._sandbox's since 1.1 (a catalog version's module is loaded the same
way): see there for what they refuse. `run` executes the module in a fresh `python -I` subprocess: an empty
environment, a scratch working directory, limits on memory (RLIMIT_AS; RLIMIT_DATA on macOS), CPU time and wall-clock time, the same
restricted builtins. The driver — your code, never the model's — is imported before the module runs; it gets the
module's namespace, the payload and the `signal` module (for a per-item alarm) and returns JSON. A module that does not
load, a crash, a run over its limits come back as `{"load_error": ...}` or `{"crash": ...}`; a failure of the sandbox
itself (the interpreter, the driver, a limit) as `{"sandbox_error": ...}` — never as the module's.

What it does not do: the limits hold in `run` only; once loaded into your process a module is held by the checks alone.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

from ...core.versions._sandbox import (ALLOWED, FORBIDDEN, INTERNAL, Refused, _dunder, _guarded_import,  # noqa: F401
                                       check, load, safe_builtins, source_name)


RUNNER = r'''
import json, signal, sys, traceback
mem, cpu = int(sys.argv[1]), int(sys.argv[2])
sys.path[:0] = json.loads(sys.argv[3])
src_path, driver, payload_path, out_path = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
allowed = json.loads(sys.argv[8]) if len(sys.argv) > 8 else []
try:                                   # the sandbox itself: a failure here is not the module's
    mod, fn = driver.split(":")
    import importlib
    from solvi.experimental.compile.sandbox import load, set_limits
    notes = set_limits(mem, cpu)
    drv = importlib.import_module(mod)
    source = open(src_path).read()
    payload = json.load(open(payload_path))
except BaseException as e:
    json.dump({"sandbox_error": "".join(traceback.format_exception_only(type(e), e)).strip()[-1500:]}, open(out_path, "w"))
    sys.exit(0)
try:
    ns = load(source, getattr(drv, "SANDBOX_EXTRA", None), allowed)
except BaseException as e:
    json.dump({"load_error": "".join(traceback.format_exception_only(type(e), e)).strip()[-1500:]}, open(out_path, "w"))
    sys.exit(0)
def alarm(sig, frm):
    raise TimeoutError("time limit for one item")
signal.signal(signal.SIGALRM, alarm)
res = getattr(drv, fn)(ns, payload, signal)
if notes and isinstance(res, dict):
    res = {**res, "sandbox_notes": notes}
json.dump(res, open(out_path, "w"), default=str)
'''


class SandboxWarning(UserWarning):
    """The sandbox ran without one of its limits (the platform refused it): the module's run is not held by it."""


def set_limits(mem: int, cpu: int, platform: str | None = None) -> list:
    """Set the child's limits: memory `mem` bytes, CPU time `cpu` seconds → notes on what could not be set. Memory is
    RLIMIT_AS (the address space) on Linux; on macOS RLIMIT_DATA, because there RLIMIT_AS below the address space the
    interpreter has already reserved raises ValueError("current limit exceeds maximum limit") — every limit under tens
    of gigabytes would. A memory limit the platform refuses is skipped and noted (the run goes on: not the module's
    fault); the CPU limit is required."""
    import resource
    platform = sys.platform if platform is None else platform
    notes = []
    which = "RLIMIT_DATA" if platform == "darwin" else "RLIMIT_AS"
    try:
        resource.setrlimit(getattr(resource, which), (mem, mem))
    except (ValueError, OSError) as e:
        notes.append(f"the memory limit ({mem // 2 ** 20} MB, {which}) was not set on {platform}: {e}; the run "
                     "went on without it")
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    return notes


def run(source: str, driver: str, payload, *, mem_mb: int = 1024, cpu_s: int = 300, wall_s: int = 600,
        path=(), allowed=()) -> dict:
    """Run the module in a subprocess under `driver` ("package.module:function": trusted code, importable in the child
    from solvi's own folder or the folders in `path`; it gets (namespace, payload, signal) and returns JSON) → the
    driver's result, `{"load_error": reason}` when the module is refused or does not load, `{"crash": reason}` (a
    crash, a limit, no result), or `{"sandbox_error": reason}` when the sandbox itself failed — the interpreter did not
    start, the driver did not import, a required limit could not be set: not the module's failure. A memory limit
    the platform refuses is skipped with a SandboxWarning (see set_limits). The driver module's `SANDBOX_EXTRA` dict,
    if any, is put into the namespace. allowed: more modules the module may import, beyond ALLOWED (as for check)."""
    import solvi
    from ...core.versions._sandbox import _allowed
    extra = sorted(_allowed(allowed) - ALLOWED)          # refuses a forbidden name before anything runs
    paths = [str(Path(solvi.__file__).resolve().parents[1])] + [str(p) for p in path]
    with tempfile.TemporaryDirectory(prefix="solvi_box_") as d:
        d = Path(d)
        (d / "module.py").write_text(source)
        (d / "runner.py").write_text(RUNNER)
        (d / "payload.json").write_text(json.dumps(payload, default=str))
        out = d / "out.json"
        cmd = [sys.executable, "-I", str(d / "runner.py"), str(mem_mb * 2 ** 20), str(cpu_s), json.dumps(paths),
               str(d / "module.py"), driver, str(d / "payload.json"), str(out), json.dumps(extra)]
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=wall_s)  # noqa: S603 — our runner, a fixed argv
        except subprocess.TimeoutExpired:
            return {"crash": f"the run exceeded {wall_s} s"}
        except OSError as e:                       # the interpreter could not start: the sandbox's failure
            return {"sandbox_error": f"the sandbox's interpreter did not start: {e}"}
        if not out.exists():
            return {"crash": f"exit {p.returncode}: " + (p.stderr or p.stdout)[-1500:]}
        res = json.loads(out.read_text())
        if isinstance(res, dict):
            for note in res.pop("sandbox_notes", None) or ():
                warnings.warn(note, SandboxWarning, stacklevel=2)
        return res


__all__ = ["ALLOWED", "FORBIDDEN", "Refused", "SandboxWarning", "check", "load", "run", "safe_builtins", "set_limits", "source_name"]
