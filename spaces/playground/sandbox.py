"""In-browser runner for the Playground tab (replaces the server Space's subprocess sandbox, runner.py).

This Space runs on Gradio-Lite: Python (Pyodide) runs inside the visitor's own browser, in a web worker. There is no server to
protect, so the visitor's catalog code is executed in-process with `exec` in a fresh module namespace. A time guard
(`sys.settrace`, only on frames of the visitor's code) stops a runaway loop after TIME_LIMIT_S seconds, so a mistake does not
hang the Python worker forever. It cannot interrupt a single long-running C call (e.g. `sum(range(10**12))`), and there is no
memory limit: the browser tab is the limit.
"""
from __future__ import annotations

import ast
import copy
import io
import json
import sys
import time
import traceback
import types
from contextlib import redirect_stdout

TIME_LIMIT_S = 5.0
USER_FILE = "<catalog>"
LIMITS_NOTE = ("**Your code runs in your browser (Pyodide); nothing is sent to a server.** It is executed in-process with a "
               f"{TIME_LIMIT_S:g} s time guard on your own functions (a step counter via `sys.settrace`), so an infinite loop "
               "is stopped instead of freezing the page. solvi, numpy, scipy and the standard library are available. The "
               "guard cannot interrupt one long C call (e.g. `sum(range(10**12))`); if that happens, reload the page.")


class TimeLimit(BaseException):
    """Raised inside the visitor's code when the time guard fires. A BaseException, so solvi's `except Exception` around each
    part does not swallow it."""


class TimeGuard:
    """Context manager: trace only frames whose code comes from the visitor's module and check the clock every few hundred
    lines (and on every call). Frames of solvi, numpy and the standard library are not traced, so the overhead stays small."""

    def __init__(self, seconds=TIME_LIMIT_S, filename=USER_FILE):
        self.seconds, self.filename = seconds, filename
        self.fired, self.lines = False, 0

    def _check(self):
        if time.perf_counter() > self.deadline:
            self.fired = True
            raise TimeLimit(f"stopped after {self.seconds:g} s")

    def _local(self, frame, event, arg):
        if event == "line":
            self.lines += 1
            if self.fired or (self.lines & 255) == 0:
                self._check()
        return self._local

    def _global(self, frame, event, arg):
        if frame.f_code.co_filename == self.filename:
            self._check()
            return self._local
        return None

    def __enter__(self):
        self.deadline = time.perf_counter() + self.seconds
        self._old = sys.gettrace()
        sys.settrace(self._global)
        return self

    def __exit__(self, *exc):
        sys.settrace(self._old)
        return False


# ------------------------------------------------------------------ Response -> plain JSON-able dict (shared with the other tabs)
def _short(v, n=240):
    s = repr(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def serialize(res, cat, state, questions, show_text=""):
    """Everything the UI shows about one System.ask() response."""
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


def _parse_value(text, original):
    """Turn the tamper textbox into a Python value of a sensible type."""
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


# ------------------------------------------------------------------ one job
def run_job(job: dict) -> dict:
    """Run one Playground job in-process and return the same result dict the server Space's sandbox returned:

        {"code": "<catalog module>", "state": {...}, "selected": [...], "known": [...],
         "tamper": {"step": 3, "value": "42", "rehash": false} | None}
    """
    captured = io.StringIO()
    stage = "loading your catalog code"
    guard = TimeGuard()
    try:
        from solvi import System
        from solvi.runtime import MISSING, vhash
        from solvi.show import show

        with redirect_stdout(captured), guard:
            mod = types.ModuleType("catalog")            # a fresh namespace for every run
            mod.__file__ = USER_FILE
            sys.modules["catalog"] = mod
            exec(compile(job["code"], USER_FILE, "exec"), mod.__dict__)
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
            if guard.fired:                              # a part hit the guard and solvi recorded it as an error
                raise TimeLimit(f"stopped after {guard.seconds:g} s")

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
            if guard.fired:
                raise TimeLimit(f"stopped after {guard.seconds:g} s")
        out["stdout"] = captured.getvalue()[-2000:]
        return out
    except TimeLimit:
        return {"ok": False, "error": f"Time limit: your code ran for more than {guard.seconds:g} s while {stage} "
                                      "(an infinite loop?) and was stopped by the time guard.",
                "stdout": captured.getvalue()[-2000:]}
    except BaseException as e:  # noqa: BLE001 — report everything, including SystemExit from user code
        tb = traceback.format_exc()
        return {"ok": False, "error": f"Error while {stage}: {type(e).__name__}: {e}"[:600],
                "traceback": "\n".join(tb.strip().splitlines()[-14:]), "stdout": captured.getvalue()[-2000:]}
