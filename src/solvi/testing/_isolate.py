"""run_file(isolate=True): each case in a fresh Python process, with an optional memory cap.

The parent never imports the task: it reads the cases from the JSON and starts, per case, `python -c BOOT` with the
interpreter's own sys.path. The child loads the suite, builds the System, prepares the case's state, says "ready" and
runs the case (run_case, with reruns, then fuzz if asked), writing one JSON line per stage on a private copy of its
stdout (the code under test prints to stderr). The parent times the case from "ready": a case past its timeout is
killed — also inside a C call, and on systems without SIGALRM — and reported as "timeout". A process that dies without
a result (a crash of the interpreter, os._exit) is an "error" with the end of its stderr; the next case runs anyway.

max_mem caps the child's address space with RLIMIT_AS (the whole process: Python, solvi, the task and the case). An
allocation past it raises MemoryError in the case, reported with status "memory". Where RLIMIT_AS cannot be set or is
not enforced (macOS, Windows), the cap is checked after the case instead, by the process's peak resident memory, and
run_file warns once that it was not enforced."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time

BOOT = ("import sys, json; req = json.loads(sys.stdin.readline()); sys.path[:0] = req['sys_path']; "
        "from solvi.testing import _isolated_child; _isolated_child(req)")


def parse_mem(v):
    """max_mem as bytes: an int, or a text with a unit ("512M", "2G", "1.5GB"; K / M / G / T, powers of 1024)."""
    if v is None:
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        n = int(v)
    else:
        t = str(v).strip().upper().removesuffix("B").removesuffix("I")
        mult = {"K": 2 ** 10, "M": 2 ** 20, "G": 2 ** 30, "T": 2 ** 40}.get(t[-1:], 1)
        try:
            n = int(float(t[:-1] if mult > 1 else t) * mult)
        except ValueError:
            raise ValueError(f"max_mem={v!r}: give bytes or a size such as '512M' or '2G'") from None
    if n <= 0:
        raise ValueError(f"max_mem={v!r} must be positive")
    return n


def cap(max_mem):
    """Set RLIMIT_AS in this process → None when it is enforced, else why not (the cap is then checked after the case
    by the peak resident memory)."""
    if max_mem is None:
        return None
    if sys.platform == "darwin":
        return "macOS does not enforce RLIMIT_AS"
    try:
        import resource
    except ImportError:
        return f"no resource limits on {sys.platform}"
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY and hard < max_mem:
            max_mem = hard
        resource.setrlimit(resource.RLIMIT_AS, (max_mem, hard))
    except (ValueError, OSError) as e:
        return f"RLIMIT_AS could not be set ({e})"
    return None


def peak_rss():
    """The process's peak resident memory in bytes, or None where it cannot be read."""
    try:
        import resource
    except ImportError:
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


def _as_memory(r, max_mem):
    """A case that failed through a MemoryError — raised by ask, or by a step whose failure made an answer abstain or
    change — has status "memory" instead of "error" / "regression"."""
    if max_mem and r.status in ("error", "regression", "nondeterministic") and \
            any("MemoryError" in p for p in r.problems):
        r.status = "memory"
        r.problems.insert(0, f"memory: the case ran out of max_mem={max_mem} bytes")
    return r


def _over(r, max_mem, peak, note):
    """The after-the-case check where the cap was not enforced: a peak over max_mem is status "memory"."""
    if note and max_mem and peak and peak > max_mem and r.status not in ("timeout", "memory"):
        r.status = "memory"
        r.problems.insert(0, f"memory: peak {peak} bytes over max_mem={max_mem} ({note}: checked after the case)")
    return r


def _dump(r):
    return {"name": r.name, "problems": r.problems, "answers": r.answers, "ms": r.ms, "status": r.status}


def child(req, t):
    """The child's side: one case of one file (see the module docstring); t is the solvi.testing module."""
    proto = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)                                    # the code under test prints to stderr, not into the protocol
    sys.stdout = sys.stderr
    note = cap(req["max_mem"])

    def say(kind, **kw):
        proto.write(json.dumps({"kind": kind, "note": note, "peak": peak_rss(), **kw}, default=str) + "\n")
        proto.flush()
    CaseResult, fuzz, load, run_case = t.CaseResult, t.fuzz, t.load, t.run_case
    try:
        suite = load(req["path"])
        system = suite.system()
    except BaseException as e:  # noqa: BLE001 — reported to the parent as the file's error
        say("load", error=f"{type(e).__name__}: {e}")
        return
    i = req["index"]
    case = suite.cases[i]
    try:
        state = suite.state(case)
    except BaseException as e:  # noqa: BLE001
        r = _as_memory(CaseResult(case.get("name", f"#{i + 1}"), [f"prepare failed: {type(e).__name__}: {e}"],
                                  status="error"), req["max_mem"])
        say("result", case=_dump(r))
        return
    say("ready")
    r = _as_memory(run_case(system, case, state, req["store"], None, req["reruns"], req.get("fast", False)), req["max_mem"])
    r = _over(r, req["max_mem"], peak_rss(), note)
    say("result", case=_dump(r))
    if req["fuzz_n"]:
        try:
            found = [f"fuzz {c}" for c in fuzz(system, state, req["fuzz_n"], req["seed"] + i)]
        except MemoryError:
            found = [f"fuzz: ran out of max_mem={req['max_mem']} bytes"]
        say("fuzz", found=found)


def _reader(stream, q):
    for line in stream:
        try:
            q.put(json.loads(line))
        except ValueError:
            continue
    q.put(None)


def run_isolated(path, cases, timeout, reruns, store, fuzz_n, seed, max_mem, CaseResult, fast=False):
    """The parent's side: every case of a cases file in its own process → ([CaseResult], file error or None, notes)."""
    import tempfile
    out, notes = [], set()
    for i, case in enumerate(cases):
        name = case.get("name", f"#{i + 1}") if isinstance(case, dict) else f"#{i + 1}"
        req = {"sys_path": list(sys.path), "path": str(path), "index": i, "store": store, "reruns": reruns,
               "fuzz_n": fuzz_n, "seed": seed, "max_mem": max_mem, "fast": fast}
        with tempfile.TemporaryFile() as err:
            started = time.perf_counter()
            proc = subprocess.Popen([sys.executable, "-c", BOOT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,  # noqa: S603 — our runner, a fixed argv
                                    stderr=err, text=True)
            proc.stdin.write(json.dumps(req) + "\n")
            proc.stdin.close()
            q = queue.Queue()
            threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
            msg, r, killed = q.get(), None, False
            if msg is not None and msg["kind"] == "load":
                proc.wait()
                return out, msg["error"], notes
            if msg is not None and msg["kind"] == "ready":
                started = time.perf_counter()
                try:
                    msg = q.get(timeout=timeout or None)
                except queue.Empty:
                    proc.kill()
                    killed, msg = True, None
            if msg is not None and msg["kind"] == "result":
                notes.add(msg["note"])
                r = CaseResult(**msg["case"])
                if fuzz_n:
                    f = q.get()
                    if f is not None and f["kind"] == "fuzz":
                        r.problems += f["found"]
                        if f["found"] and r.status == "ok":
                            r.status = "error"
            proc.wait()
            if r is None and killed:
                r = CaseResult(name, [f"timeout: ran longer than {timeout:g} s (the process was killed)"],
                               status="timeout")
                r.ms = (time.perf_counter() - started) * 1000
            elif r is None:
                err.seek(0)
                tail = err.read().decode("utf-8", "replace").strip().splitlines()[-5:]
                status = "memory" if max_mem and any("MemoryError" in t for t in tail) else "error"
                r = CaseResult(name, [f"crash: the case's process exited with code {proc.returncode}"
                                      + ("" if not tail else "\n" + "\n".join(tail))], status=status)
        out.append(r)
    return out, None, notes - {None}


__all__ = ["cap", "child", "parse_mem", "peak_rss", "run_isolated"]
