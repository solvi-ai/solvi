"""In-browser runner for the Playground tab (replaces the server Space's subprocess sandbox, runner.py).

This Space runs on Gradio-Lite: Python (Pyodide) runs inside the visitor's own browser, in a web worker. There is no server to
protect, so the visitor's catalog code is executed in-process with `exec` in a fresh module namespace. A time guard
(`sys.settrace`, only on frames of the visitor's code) stops a runaway loop after TIME_LIMIT_S seconds, so a mistake does not
hang the Python worker forever. It cannot interrupt a single long-running C call (e.g. `sum(range(10**12))`), and there is no
memory limit: the browser tab is the limit.

The catalog module and its System are kept while the code is unchanged (see _load), so System.stats counts over the
System's lifetime and an optional setup(system) hook runs once.
"""
from __future__ import annotations

import ast
import copy
import hashlib
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


def _plain(v, n=240):
    """Anything → JSON-able: containers recursively, strings cut to n characters, other objects as a short repr."""
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return round(v, 6)
    if isinstance(v, str):
        return v if len(v) <= n else v[: n - 1] + "…"
    if isinstance(v, dict):
        return {str(k): _plain(x, n) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_plain(x, n) for x in v]
    return _short(v, n)


def _is_unknown(v):
    try:
        from solvi import Unknown
    except ImportError:                                  # solvi < 0.5
        return False
    return v is Unknown


def _answer(v):
    """An answer for the UI: "not stated" for solvi.Unknown, tuples as lists (multi-label, rankings)."""
    if _is_unknown(v):
        return "not stated"
    if isinstance(v, tuple):
        return [_answer(x) for x in v]
    return v if v is None or isinstance(v, (bool, int, float, str)) else _plain(v)


def _key(k):
    return "<not stated>" if _is_unknown(k) else str(k)


def audit_dict(res, state, context=48):
    """res.audit().to_dict() made JSON-able for the UI: given values as short reprs, and every quote with the text around it
    (before / quoted / after) so the panel can highlight the span in its document."""
    a = res.audit().to_dict()
    for q, d in a["answers"].items():
        for g in d["given"]:
            g["value"] = _short(g["value"], 80)
        for c in d["computed"] + d["decided"] + d["learned"]:
            c["value"] = None if c.get("value") is None else _short(c["value"], 120)
        for x in d["quoted"]:
            doc = state.get(x.get("source"))
            if isinstance(doc, str) and x.get("error") is None:
                s, e = x["start"], x["end"]
                x["context"] = [("…" if s > context else "") + doc[max(0, s - context):s], doc[s:e],
                                doc[e:e + context] + ("…" if e + context < len(doc) else "")]
            x["value"] = None if x.get("value") is None else _short(x["value"], 120)
        d["answer"] = _answer(d["answer"])
        d["not_run"] = [list(t) for t in d["not_run"]]
    return _plain(a, 400)


def serialize(res, cat, state, questions, show_text="", system=None):
    """Everything the UI shows about one System.ask() response. Pass the System to replay answer heads too and to include
    its lifetime safeguard stats."""
    from solvi.runtime import MISSING
    rep = res.trace.replay(system if system is not None else cat)
    out = {"ok": True, "ms": res.ms, "show": show_text, "question_text": {q.name: q.text for q in questions}}
    out["answers"] = [{"question": q, "answer": _answer(r.answer),
                       "confidence": round(float(r.confidence), 4), "status": r.status, "why": r.why,
                       "probs": {_key(k): round(float(v), 4) for k, v in (r.probs or {}).items()},
                       "provenance": getattr(r, "provenance", None), "guard": getattr(r, "guard", None),
                       "kind": getattr(r, "kind", None),
                       "evidence": [{"value": e.value, "start": e.start, "end": e.end, "source": e.source}
                                    for e in (getattr(r, "evidence", None) or [])],
                       "extra": _plain(getattr(r, "extra", None))}
                      for q, r in res.results.items()]
    ov = getattr(res, "overall", None)
    out["overall"] = _plain({k: v for k, v in ov.items() if k != "weakest"}) if isinstance(ov, dict) else None
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
        m = getattr(r, "model", None)
        recs.append({"step": r.step, "kind": r.kind, "name": r.name, "value": "—" if r.value is MISSING else _short(r.value),
                     "quote": list(r.quote) if r.quote else None, "quote_text": quote_text,
                     "confidence": round(float(r.confidence), 4), "error": r.error, "inputs": dict(r.inputs),
                     "prev": r.prev, "hash": r.hash, "origin": getattr(r, "origin", None),
                     "model": f"{m['type']} {m['id']} #{m['fp'][:8]}" if m else None,
                     "producer": getattr(r, "producer", None)})
    out["records"] = recs
    out["init_hash"] = res.trace.init_hash
    out["replay"] = {"ok": rep["ok"], "steps": rep["steps"], "mismatches": [list(m) for m in rep["mismatches"]],
                     "models": [list(m) for m in rep.get("models", [])]}
    out["feasible"] = bool(getattr(res, "feasible", True))
    out["violations"] = list(getattr(res, "violations", None) or [])
    out["audit"] = audit_dict(res, state)
    out["report_md"] = None
    if hasattr(res, "report"):                           # solvi 0.7: a report for people (no model is called)
        try:
            out["report_md"] = res.report(format="md", system=system)
        except Exception as e:  # noqa: BLE001 — the report is an extra; the rest of the panel must still render
            out["report_md"] = f"**The report could not be built:** `{type(e).__name__}: {e}`"
    if system is not None:
        out["stats"] = dict(system.stats)
    return out


# ------------------------------------------------------------------ "the model was replaced after the decision"
class _Replaced:
    """Stands for a retrained or swapped model: the same object, with a different fingerprint. Replay compares the fingerprint
    recorded in the trace with the catalog's current model and must report "model changed since this decision"."""

    def __init__(self, model):
        self._model = model

    def fingerprint(self):
        from solvi.provenance import digest, fingerprint
        return digest("replaced after the decision", fingerprint(self._model))

    def __getattr__(self, k):
        return getattr(self._model, k)


def replace_models(cat, system):
    """Swap every model behind a catalog part (alternative producers and model-backed rules included) and every answer head of
    the system for a _Replaced stand-in. → (names, undo)."""
    swapped, seen, names = [], set(), []
    for p in list(cat.parts.values()) + list(cat.rules.values()):
        for x in [p] + list(getattr(p, "alternatives", None) or []):
            if id(x) not in seen and getattr(x, "model", None) is not None:
                seen.add(id(x))
                swapped.append((x, x.model))
                x.model = _Replaced(x.model)
                names.append(x.name)
    heads = dict(system.heads)
    for q, h in heads.items():
        system.heads[q] = _Replaced(h)
        names.append(f"answer head of {q}")

    def undo():
        for x, m in swapped:
            x.model = m
        system.heads.update(heads)
    return names, undo


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
_LOADED = {}     # sha256 of the catalog code -> (module, cat, questions, System): the last successfully loaded catalog


def _load(code):
    """Exec the visitor's catalog module and build its System, once per code version: while the code is unchanged the same
    System answers every run, so System.stats counts over its lifetime (and an optional setup(system), e.g. fit_fast, runs
    only once). Called inside the time guard."""
    from solvi import System
    key = hashlib.sha256(code.encode()).hexdigest()
    if key in _LOADED:
        mod = _LOADED[key][0]
        sys.modules["catalog"] = mod
        return _LOADED[key]
    mod = types.ModuleType("catalog")                    # a fresh namespace for every new version of the code
    mod.__file__ = USER_FILE
    sys.modules["catalog"] = mod
    exec(compile(code, USER_FILE, "exec"), mod.__dict__)
    cat = getattr(mod, "cat", None)
    questions = getattr(mod, "QUESTIONS", None)
    if cat is None or questions is None:
        raise NameError("the catalog code must define `cat = Catalog()` and `QUESTIONS = [...]`")
    system = System(cat, questions)
    if callable(getattr(mod, "setup", None)):
        mod.setup(system)
    _LOADED.clear()
    _LOADED[key] = (mod, cat, questions, system)
    return _LOADED[key]


def run_job(job: dict) -> dict:
    """Run one Playground job in-process and return the same result dict the server Space's sandbox returned:

        {"code": "<catalog module>", "state": {...}, "selected": [...], "known": [...],
         "tamper": {"step": 3, "value": "42", "rehash": false} | {"mode": "model"} | None}

    The module must define `cat` and `QUESTIONS`; optional `prepare(state)` converts init_state before each ask, optional
    `setup(system)` runs once after the System is built (fit a head, learn a rule)."""
    captured = io.StringIO()
    stage = "loading your catalog code"
    guard = TimeGuard()
    try:
        from solvi.runtime import MISSING, vhash
        from solvi.show import show

        with redirect_stdout(captured), guard:
            mod, cat, questions, system = _load(job["code"])
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
            t0 = time.perf_counter()
            res = system.ask(state, ask)
            wall_ms = (time.perf_counter() - t0) * 1000
            if guard.fired:                              # a part hit the guard and solvi recorded it as an error
                raise TimeLimit(f"stopped after {guard.seconds:g} s")

            stage = "replaying the trace"
            buf = io.StringIO()
            with redirect_stdout(buf):
                show(res, cat)
            out = serialize(res, cat, state, questions, show_text=buf.getvalue(), system=system)
            out.update({"questions": names, "asked": ask, "wall_ms": wall_ms})

            tamper = job.get("tamper")
            if tamper and tamper.get("mode") == "model":
                stage = "replacing the models"
                replaced, undo = replace_models(cat, system)
                try:
                    trep = res.trace.replay(system)
                finally:
                    undo()
                out["tamper"] = {"mode": "model", "replaced": replaced, "ok": trep["ok"],
                                 "mismatches": [list(m) for m in trep["mismatches"]],
                                 "models": [list(m) for m in trep.get("models", [])]}
            elif tamper:
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
                trep = t.replay(system)
                out["tamper"] = {"mode": "value", "step": step, "name": rec.name,
                                 "old": "—" if old is MISSING else _short(old), "new": _short(rec.value),
                                 "rehash": bool(tamper.get("rehash")), "ok": trep["ok"],
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
