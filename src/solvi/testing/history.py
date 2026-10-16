"""Verdicts across runs: on a live environment a single pass or fail says little.

    from solvi.testing import run_path
    from solvi.testing.history import KnownBugs, VerdictHistory, run_status

    results = run_path("cases")
    print(run_status(results))                 # "degraded" when many cases of this run are blocked at once
    history = VerdictHistory("verdicts.jsonl") # kept in a store (any TraceStorage, or a path): one record per run
    history.add(results)
    history.classify("cases/a.cases.json::legal threat")   # "stable" / "flaky" / "regression" / "fixed" / "new"
    history.gate(results)                      # the failures that count: the flaky cases are quarantined
    bugs = KnownBugs("verdicts.jsonl")
    bugs.register("BUG-12", r"expected 'high', got 'low'")
    bugs.triage(results)                       # failing cases → a known bug id, or new; open bugs not seen for N runs

A case is named by a key: "<cases file>::<case name>" for the results of run_file / run_path, or any string you give
(add({key: status})). A verdict is the case's status in one run (solvi.testing.STATUSES) and the attempt it passed on.
Nothing here re-runs anything: the runs are yours (run_file(reruns=), a retry loop, a CI job), these keep their outcomes."""
from __future__ import annotations

import re
import time as _time

PASS = "ok"
BLOCKED = ("error", "timeout")                   # statuses that say the case could not run, not that it decided wrong
CLASSES = ("new", "stable", "flaky", "regression", "fixed")


def _storage(storage):
    if storage is None:
        return None
    from ..core.store import open_storage
    return open_storage(storage)


def _append(storage, kind, body):
    from ..core.store import FORMAT
    return storage._append({"v": FORMAT, "kind": kind, **body})


def verdicts_of(results):
    """Results → {key: {"status", "attempt", "problems"}}. results: a FileResult, a list of them (run_path), or a dict
    {key: a status string, a CaseResult, or {"status", "attempt"=1, "problems"=[]}}."""
    from . import CaseResult, FileResult
    if isinstance(results, FileResult):
        results = [results]
    out = {}
    if isinstance(results, dict):
        for k, v in results.items():
            if isinstance(v, str):
                out[str(k)] = {"status": v, "attempt": 1, "problems": []}
            elif isinstance(v, CaseResult):
                out[str(k)] = {"status": v.status, "attempt": 1, "problems": list(v.problems)}
            else:
                out[str(k)] = {"status": v["status"], "attempt": int(v.get("attempt", 1)),
                               "problems": list(v.get("problems") or [])}
        return out
    for f in results:
        if f.error is not None:
            out[f"{f.path}"] = {"status": "error", "attempt": 1, "problems": [f.error]}
        for c in f.cases:
            out[f"{f.path}::{c.name}"] = {"status": c.status, "attempt": 1, "problems": list(c.problems)}
    return out


def run_status(results, blocked=BLOCKED, share=0.5, min_blocked=3, pattern=None):
    """One status for a whole run → {"status": "passed" | "failed" | "degraded", "total", "passed", "failed": did not pass and
    not blocked, "blocked", "why"}. A case is blocked when its status is one of `blocked` (an error or a timeout: it could not run) or a problem
    of it matches `pattern` (a regex: "ConnectionError|503"). The run is "degraded" — the environment, not the
    decisions — when at least `min_blocked` cases and at least `share` of them are blocked: one run-level status
    instead of many independent failures. Otherwise "failed" when a case did not pass, else "passed"."""
    vs = verdicts_of(results)
    rx = re.compile(pattern) if isinstance(pattern, str) else pattern
    n_blocked = sum(1 for v in vs.values() if v["status"] in blocked or
                    (rx is not None and v["status"] != PASS and any(rx.search(p) for p in v["problems"])))
    n_pass = sum(1 for v in vs.values() if v["status"] == PASS)
    total = len(vs)
    if total and n_blocked >= min_blocked and n_blocked >= share * total:
        status, why = "degraded", f"{n_blocked} of {total} cases blocked (an error, a timeout or a matching problem)"
    elif n_pass < total:
        status, why = "failed", f"{total - n_pass} of {total} cases did not pass"
    else:
        status, why = "passed", f"all {total} cases passed"
    return {"status": status, "total": total, "passed": n_pass, "failed": total - n_pass - n_blocked, "blocked": n_blocked,
            "why": why}


class VerdictHistory:
    """The verdicts of the same cases over runs, with a class per case over the last `window` runs it was in:
    "new" (no verdict yet), "stable" (the same outcome every time — passing or failing, see last()), "regression"
    (passed, then fails since one change), "fixed" (failed, then passes since one change), "flaky" (the outcome changed
    more than once, a "nondeterministic" status, or a pass on a later attempt). A degraded run (run_status) can be left
    out: add(results, skip_degraded=True). storage: None (in memory), a path or a TraceStorage — one "test_verdicts"
    record per run, read back when the history is opened."""

    KIND = "test_verdicts"

    def __init__(self, storage=None, window=10):
        self.storage = _storage(storage)
        self.window = int(window)
        self.runs = []                                # [{"run", "time", "cases": {key: verdict}}]
        if self.storage is not None:
            for s in self.storage.iter(self.KIND):
                self.runs.append({"run": s.data.get("run_id"), "time": s.time, "cases": s.data.get("cases") or {}})

    def add(self, results, run=None, *, skip_degraded=False, **status_kw):
        """Add one run's verdicts (see verdicts_of) → the run's run_status. run: its name (default: its position).
        skip_degraded=True: a degraded run is not added (its failures say nothing about the cases)."""
        vs = verdicts_of(results)
        st = run_status(vs, **status_kw)
        if skip_degraded and st["status"] == "degraded":
            return st
        run = f"#{len(self.runs) + 1}" if run is None else str(run)
        if self.storage is not None:
            rec = _append(self.storage, self.KIND, {"run_id": run, "cases": vs, "run_status": st["status"]})
            t = rec.get("time", _time.time())
        else:
            t = _time.time()
        self.runs.append({"run": run, "time": t, "cases": vs})
        return st

    def verdicts(self, key):
        """The case's verdicts, oldest first → [{"run", "status", "attempt"}] (the runs it was in)."""
        return [{"run": r["run"], "status": r["cases"][key]["status"], "attempt": r["cases"][key].get("attempt", 1)}
                for r in self.runs if key in r["cases"]]

    def keys(self):
        return list(dict.fromkeys(k for r in self.runs for k in r["cases"]))

    def last(self, key):
        v = self.verdicts(key)
        return v[-1]["status"] if v else None

    def classify(self, key):
        """The case's class over its last `window` verdicts (see the class docstring)."""
        vs = self.verdicts(key)[-self.window:]
        if not vs:
            return "new"
        if any(v["status"] == "nondeterministic" or (v["status"] == PASS and v["attempt"] > 1) for v in vs):
            return "flaky"
        ok = [v["status"] == PASS for v in vs]
        flips = sum(a != b for a, b in zip(ok, ok[1:]))
        if flips == 0:
            return "stable"
        if flips == 1:
            return "fixed" if ok[-1] else "regression"
        return "flaky"

    def report(self):
        """key → {"class", "last", "runs", "passed", "later_attempt": passes on a later attempt}."""
        out = {}
        for k in self.keys():
            vs = self.verdicts(k)[-self.window:]
            out[k] = {"class": self.classify(k), "last": vs[-1]["status"], "runs": len(vs),
                      "passed": sum(v["status"] == PASS for v in vs),
                      "later_attempt": sum(v["status"] == PASS and v["attempt"] > 1 for v in vs)}
        return out

    def quarantined(self):
        """The flaky cases (their failures do not fail a gate), sorted."""
        return sorted(k for k in self.keys() if self.classify(k) == "flaky")

    def stable_share(self):
        """The share of cases whose outcome was the same in each of their last `window` runs (None: no case yet)."""
        ks = self.keys()
        return None if not ks else sum(self.classify(k) == "stable" for k in ks) / len(ks)

    def gate(self, results):
        """This run's failures against the history → {"ok", "failed": the failing cases that count, "quarantined": the
        failing cases that are flaky}. Call it before add() to judge a run by the history before it."""
        bad = sorted(k for k, v in verdicts_of(results).items() if v["status"] != PASS)
        q = set(self.quarantined())
        failed = [k for k in bad if k not in q]
        return {"ok": not failed, "failed": failed, "quarantined": [k for k in bad if k in q]}


class KnownBugs:
    """Failing checks matched to known bugs: register(bug_id, pattern) with a regex searched in a failing case's problem
    text (and, optionally, a key regex for the cases it applies to); triage(results) names the known bug of each failing
    case, lists the failures no bug explains, a fixed bug seen again, and the open bugs not seen for `quiet` runs
    ("check that it is fixed"). storage: None, a path or a TraceStorage — the registry's events ("known_bug" records)
    are kept and replayed there."""

    KIND = "known_bug"

    def __init__(self, storage=None, quiet=5):
        self.storage = _storage(storage)
        self.quiet = int(quiet)
        self.bugs = {}                                # id → {"pattern", "key", "status", "note", "last_seen", "seen"}
        self.run = 0                                  # triage calls so far
        if self.storage is not None:
            for s in self.storage.iter(self.KIND):
                self._apply(s.data)

    def _event(self, op, **data):
        body = {"op": op, **data}
        if self.storage is not None:
            _append(self.storage, self.KIND, body)
        self._apply(body)

    def _apply(self, e):
        op, b = e.get("op"), e.get("bug")
        if op == "register":
            self.bugs[b] = {"pattern": e["pattern"], "key": e.get("key"), "status": e.get("status", "open"),
                            "note": e.get("note"), "last_seen": None, "seen": 0, "since": self.run}
        elif op in ("fixed", "open") and b in self.bugs:
            self.bugs[b]["status"] = op
        elif op == "run":
            self.run = e["n"]
            for bug in e.get("seen") or ():
                if bug in self.bugs:
                    self.bugs[bug]["last_seen"] = e["n"]
                    self.bugs[bug]["seen"] += 1

    def register(self, bug, pattern, key=None, status="open", note=None):
        """A known bug: `pattern` (a regex) found in a failing case's problems; `key` (a regex) limits it to some cases."""
        if status not in ("open", "fixed"):
            raise ValueError('status is "open" or "fixed"')
        re.compile(pattern)
        if key is not None:
            re.compile(key)
        self._event("register", bug=str(bug), pattern=pattern, key=key, status=status, note=note)

    def fixed(self, bug):
        """Mark a bug fixed: seen again, triage reports it under "reappeared"."""
        self._event("fixed", bug=bug)

    def reopen(self, bug):
        self._event("open", bug=bug)

    def match(self, text, key=None):
        """The first registered bug whose pattern is in `text` (and whose key pattern matches `key`, when it has one) →
        its id, or None."""
        for b, d in self.bugs.items():
            if d["key"] is not None and (key is None or not re.search(d["key"], key)):
                continue
            if re.search(d["pattern"], text):
                return b
        return None

    def triage(self, results):
        """One run's failing cases → {"known": {key: bug}, "new": [keys no bug explains], "reappeared": [fixed bugs seen
        again], "not_seen": [open bugs not seen for `quiet` runs]}. Counts as a run of the registry."""
        known, new = {}, []
        for k, v in verdicts_of(results).items():
            if v["status"] == PASS:
                continue
            b = self.match("\n".join(v["problems"]) or v["status"], k)
            if b is None:
                new.append(k)
            else:
                known[k] = b
        seen = sorted(set(known.values()))
        self._event("run", n=self.run + 1, seen=seen)
        return {"known": known, "new": new, "reappeared": [b for b in seen if self.bugs[b]["status"] == "fixed"],
                "not_seen": self.not_seen()}

    def not_seen(self, runs=None):
        """The open bugs not seen in the last `runs` runs (default: quiet) — check that they are fixed."""
        n = self.quiet if runs is None else int(runs)
        return sorted(b for b, d in self.bugs.items() if d["status"] == "open" and
                      self.run - (d["last_seen"] if d["last_seen"] is not None else d["since"]) >= n)


__all__ = ["BLOCKED", "CLASSES", "KnownBugs", "run_status", "VerdictHistory", "verdicts_of"]
