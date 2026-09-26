"""Sandboxed runner for the Playground tab.

The app starts this file as a separate Python process (see `run_sandboxed`), with a hard wall-clock timeout, resource limits,
a throw-away working directory and a minimal environment. The child reads one JSON job from stdin:

    {"code": "<catalog module>", "state": {...}, "selected": [...], "known": [...],
     "tamper": {"step": 3, "value": "42", "rehash": false} | null}

executes the catalog module (it must define `cat` and `QUESTIONS`, optionally `prepare(state)`), asks the questions and prints
one JSON result to stdout. Nothing from the user's code ever runs inside the web server process.
"""
from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile

TIMEOUT_S = 5
CPU_S = 5
MEMORY_BYTES = 1024 * 1024 * 1024        # address space ~1 GB
FILE_BYTES = 16 * 1024 * 1024            # largest file the code may write
OUTPUT_LIMIT = 2_000_000                 # characters of stdout we accept back
LIMITS_NOTE = (f"Sandbox: your code runs in a separate process with a {TIMEOUT_S} s timeout, {CPU_S} s CPU, ~1 GB memory, "
               "a temporary working directory and no environment variables. solvi and the standard library are available.")


# ------------------------------------------------------------------ parent side (called by app.py)
def _limit_child():
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_S, CPU_S + 1))
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_BYTES, MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_BYTES, FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _solvi_path():
    """Directory that holds the solvi package the app imported (installed wheel or ../../src for local runs)."""
    try:
        import solvi
        return os.path.dirname(os.path.dirname(os.path.abspath(solvi.__file__)))
    except ImportError:
        return ""


def run_sandboxed(job: dict, timeout: float = TIMEOUT_S) -> dict:
    """Run one job in a fresh sandboxed process and return its JSON result (or an error dict)."""
    workdir = tempfile.mkdtemp(prefix="solvi-play-")
    env = {"PATH": "/usr/bin:/bin", "HOME": workdir, "TMPDIR": workdir, "LANG": "C.UTF-8", "PYTHONIOENCODING": "utf-8",
           "PYTHONDONTWRITEBYTECODE": "1", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    paths = [p for p in [_solvi_path(), os.environ.get("PYTHONPATH", "")] if p]
    if paths:
        env["PYTHONPATH"] = os.pathsep.join(paths)
    try:
        proc = subprocess.Popen([sys.executable, os.path.abspath(__file__), "--child"], stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=workdir, env=env, text=True,
                                preexec_fn=_limit_child, start_new_session=True)
        try:
            out, err = proc.communicate(json.dumps(job, default=str), timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.communicate()
            return {"ok": False, "error": f"Timed out after {timeout:g} s: the code did not finish (an infinite loop?). "
                                          "The sandbox process was killed."}
        if proc.returncode != 0 and not out.strip():
            why = {-signal.SIGXCPU: "CPU time limit exceeded", -signal.SIGKILL: "killed (CPU or memory limit)",
                   -signal.SIGSEGV: "crashed (memory limit?)"}.get(proc.returncode, f"exit code {proc.returncode}")
            return {"ok": False, "error": f"The sandbox process stopped: {why}.", "traceback": err[-3000:]}
        try:
            return json.loads(out[-OUTPUT_LIMIT:] if len(out) <= OUTPUT_LIMIT else out[:OUTPUT_LIMIT])
        except json.JSONDecodeError:
            return {"ok": False, "error": "The sandbox returned no result (output too large or the code printed to stdout "
                                          "in a way that broke the result).", "traceback": (err or out)[-3000:]}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ------------------------------------------------------------------ shared: Response -> plain JSON-able dict
def serialize(res, cat, state, questions, show_text=""):
    """Everything the UI shows about one System.ask() response (used by the sandbox child and by the in-process tabs)."""
    from solvi.runtime import MISSING
    rep = res.trace.replay(cat)
    out = {"ok": True, "ms": res.ms, "show": show_text, "question_text": {q.name: q.text for q in questions}}
    out["answers"] = [{"question": q, "answer": r.answer, "confidence": round(float(r.confidence), 4), "status": r.status,
                       "why": r.why, "probs": {k: round(float(v), 4) for k, v in (r.probs or {}).items()}}
                      for q, r in res.results.items()]
    out["flow"] = [{"step": i, "kind": s.part.kind, "name": s.part.name, "inputs": list(s.part.inputs),
                    "reasons": list(s.reasons), "hard": bool(getattr(s.part, "hard", False)), "doc": s.part.doc}
                   for i, s in enumerate(res.flow.steps, 1)]
    out["skipped"] = [{"name": k, "kind": cat.parts[k].kind, "why": v, "inputs": list(cat.parts[k].inputs)}
                      for k, v in sorted(res.flow.skipped.items())]
    out["unresolved"] = res.flow.unresolved
    out["run_skipped"] = [[n, why] for n, why in (getattr(res.trace, "skipped", None) or [])]   # early exit at run time
    out["per_question"] = res.flow.per_question
    recs = []
    for r in res.trace.records:
        quote_text = None
        if r.quote:
            s, e, src = r.quote
            doc = state.get(src)
            if isinstance(doc, str):
                quote_text = doc[s:e]
        recs.append({"step": r.step, "kind": r.kind, "name": r.name, "value": "—" if r.value is MISSING else _short(r.value),
                     "quote": list(r.quote) if r.quote else None, "quote_text": quote_text,
                     "confidence": round(float(r.confidence), 4), "error": r.error, "inputs": dict(r.inputs),
                     "prev": r.prev, "hash": r.hash})
    out["records"] = recs
    out["init_hash"] = res.trace.init_hash
    out["replay"] = {"ok": rep["ok"], "steps": rep["steps"], "mismatches": [list(m) for m in rep["mismatches"]]}
    return out


# ------------------------------------------------------------------ child side
def _short(v, n=240):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _parse_value(text, original):
    """Turn the tamper textbox into a Python value of a sensible type."""
    import ast
    from datetime import date
    text = text.strip()
    if isinstance(original, date):
        try:
            return date.fromisoformat(text.strip("'\""))
        except ValueError:
            pass
    for parse in (ast.literal_eval, json.loads):
        try:
            return parse(text)
        except Exception:  # noqa: BLE001
            pass
    return text


def _child():
    import copy
    import io
    import time
    import traceback
    import types
    from contextlib import redirect_stdout

    job = json.loads(sys.stdin.read())
    real_stdout = sys.stdout
    captured = io.StringIO()
    stage = "loading your catalog code"
    try:
        with redirect_stdout(captured):
            from solvi import System
            from solvi.runtime import MISSING, vhash
            from solvi.show import show

            mod = types.ModuleType("catalog")
            mod.__file__ = "<catalog>"
            sys.modules["catalog"] = mod
            exec(compile(job["code"], "<catalog>", "exec"), mod.__dict__)
            cat = getattr(mod, "cat", None)
            questions = getattr(mod, "QUESTIONS", None)
            if cat is None or questions is None:
                raise NameError("the catalog code must define `cat = Catalog()` and `QUESTIONS = [...]`")
            names = [q.name for q in questions]
            known, selected = set(job.get("known") or []), set(job.get("selected") or [])
            ask = [n for n in names if n in selected or n not in known] or names

            stage = "preparing init_state"
            state = job.get("state") or {}
            if not isinstance(state, dict):
                raise TypeError("init_state must be a JSON object")
            if callable(getattr(mod, "prepare", None)):
                state = mod.prepare(dict(state))

            stage = "asking the questions"
            system = System(cat, questions)
            t0 = time.perf_counter()
            res = system.ask(state, ask)
            wall_ms = (time.perf_counter() - t0) * 1000

            stage = "replaying the trace"
            buf = io.StringIO()
            with redirect_stdout(buf):
                show(res, cat)

            out = serialize(res, cat, state, questions, show_text=buf.getvalue())
            out.update({"questions": names, "asked": ask, "wall_ms": wall_ms})

            tamper = job.get("tamper")
            if tamper:
                stage = "tampering with the trace"
                t = copy.deepcopy(res.trace)
                step = int(tamper["step"])
                rec = next((r for r in t.records if r.step == step), None)
                if rec is None:
                    raise ValueError(f"there is no step {step} in this trace")
                old = rec.value
                rec.value = _parse_value(str(tamper.get("value", "")), old)
                if tamper.get("rehash"):                 # the attacker also recomputes the hash of this and every later record
                    prev = rec.prev
                    for x in t.records[t.records.index(rec):]:
                        x.prev = prev
                        x.hash = vhash(x.body())
                        prev = x.hash
                trep = t.replay(cat)
                out["tamper"] = {"step": step, "name": rec.name, "old": "—" if old is MISSING else _short(old),
                                 "new": _short(rec.value), "rehash": bool(tamper.get("rehash")), "ok": trep["ok"],
                                 "mismatches": [list(m) for m in trep["mismatches"]]}
        out["stdout"] = captured.getvalue()[-2000:]
    except BaseException as e:  # noqa: BLE001 — report everything, including SystemExit from user code
        tb = traceback.format_exc()
        out = {"ok": False, "error": f"Error while {stage}: {type(e).__name__}: {e}"[:600],
               "traceback": "\n".join(tb.strip().splitlines()[-14:]), "stdout": captured.getvalue()[-2000:]}
    real_stdout.write(json.dumps(out, default=str))
    real_stdout.flush()


if __name__ == "__main__" and "--child" in sys.argv:
    _limit_child()          # set again from inside, in case the parent could not (non-POSIX preexec)
    _child()
