"""A sandbox for code a model wrote: only pure functions over facts.

    from solvi.experimental.compile.sandbox import check, load, run
    check(source)                                   # [] when allowed, else the reasons it is refused
    ns = load(source)                               # the module's namespace, in this process (refuses what check refuses)
    out = run(source, "mypkg.driver:main", payload) # the module run in a subprocess by a trusted driver → its JSON

`check` and `load` are the stable solvi.core.versions._sandbox's since 1.1 (a catalog version's module is loaded the same
way): see there for what they refuse. `run` executes the module in a fresh `python -I` subprocess: an empty
environment, a scratch working directory, limits on memory (RLIMIT_AS), CPU time and wall-clock time, the same
restricted builtins. The driver — your code, never the model's — is imported before the module runs; it gets the
module's namespace, the payload and the `signal` module (for a per-item alarm) and returns JSON. A module that does not
load, a crash, a run over its limits come back as `{"load_error": ...}` or `{"crash": ...}`.

What it does not do: the limits hold in `run` only; once loaded into your process a module is held by the checks alone.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from ...core.versions._sandbox import (ALLOWED, FORBIDDEN, INTERNAL, Refused, _dunder, _guarded_import,  # noqa: F401
                                       check, load, safe_builtins, source_name)


RUNNER = r'''
import json, resource, signal, sys, traceback
mem, cpu = int(sys.argv[1]), int(sys.argv[2])
resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
sys.path[:0] = json.loads(sys.argv[3])
src_path, driver, payload_path, out_path = sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
mod, fn = driver.split(":")
import importlib
drv = importlib.import_module(mod)
from solvi.experimental.compile.sandbox import load
source = open(src_path).read()
payload = json.load(open(payload_path))
try:
    ns = load(source, getattr(drv, "SANDBOX_EXTRA", None))
except BaseException as e:
    json.dump({"load_error": "".join(traceback.format_exception_only(type(e), e)).strip()[-1500:]}, open(out_path, "w"))
    sys.exit(0)
def alarm(sig, frm):
    raise TimeoutError("time limit for one item")
signal.signal(signal.SIGALRM, alarm)
res = getattr(drv, fn)(ns, payload, signal)
json.dump(res, open(out_path, "w"), default=str)
'''


def run(source: str, driver: str, payload, *, mem_mb: int = 1024, cpu_s: int = 300, wall_s: int = 600,
        path=()) -> dict:
    """Run the module in a subprocess under `driver` ("package.module:function": trusted code, importable in the child
    from solvi's own folder or the folders in `path`; it gets (namespace, payload, signal) and returns JSON) → the
    driver's result, `{"load_error": reason}` when the module is refused or does not load, or `{"crash": reason}` (a
    crash, a limit, no result). The driver module's `SANDBOX_EXTRA` dict, if any, is put into the namespace."""
    import solvi
    paths = [str(Path(solvi.__file__).resolve().parents[1])] + [str(p) for p in path]
    with tempfile.TemporaryDirectory(prefix="solvi_box_") as d:
        d = Path(d)
        (d / "module.py").write_text(source)
        (d / "runner.py").write_text(RUNNER)
        (d / "payload.json").write_text(json.dumps(payload, default=str))
        out = d / "out.json"
        cmd = [sys.executable, "-I", str(d / "runner.py"), str(mem_mb * 2 ** 20), str(cpu_s), json.dumps(paths),
               str(d / "module.py"), driver, str(d / "payload.json"), str(out)]
        env = {"PATH": "/usr/bin:/bin", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            p = subprocess.run(cmd, cwd=d, env=env, capture_output=True, text=True, timeout=wall_s)  # noqa: S603 — our runner, a fixed argv
        except subprocess.TimeoutExpired:
            return {"crash": f"the run exceeded {wall_s} s"}
        if not out.exists():
            return {"crash": f"exit {p.returncode}: " + (p.stderr or p.stdout)[-1500:]}
        return json.loads(out.read_text())


__all__ = ["ALLOWED", "FORBIDDEN", "Refused", "check", "load", "run", "safe_builtins", "source_name"]
