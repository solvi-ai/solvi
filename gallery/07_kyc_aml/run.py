"""Run every KYC / AML case: answers, the strategist's flow, what the early exit skipped, trace replay, timing; then show the
trace as audit evidence (a record altered after the decision is caught). Exits non-zero if any answer differs from cases.json.

Run:  uv run python gallery/07_kyc_aml/run.py"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("kyc_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)


def state_of(raw):
    return task.prepare(json.loads(json.dumps(raw)))           # a fresh copy, as the playground passes it


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def short(s, n=150):
    return s if len(s) <= n else s[:n - 1] + "…"


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    cases = json.loads((HERE / "cases.json").read_text())
    tally = Tally()
    failures, times, external_run, external_skipped = [], [], 0, 0
    for i, case in enumerate(cases, 1):
        res = system.ask(state_of(case["state"]))
        times.append(res.ms)
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        for q, r in res.results.items():
            print(f"    {q:9s} {got(r)!s:8s} {r.status:8s} {r.confidence:.2f}  {short(r.why)}")
        scr = res.values["sanctions_screen"]
        print(f"    screening: {scr['listed']!r} name score {scr['score']}, birth date match {scr['dob_match']} -> hit {scr['hit']}")
        ran = [x.name for x in res.trace.records if x.kind != "rule"]
        skipped = [n for n, _ in res.trace.skipped]
        external_run += sum(n in task.EXTERNAL for n in ran)
        external_skipped += sum(n in task.EXTERNAL for n in skipped)
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad]
        print(f"    {len(ran)} steps run, {len(skipped)} skipped" + (f" ({', '.join(skipped)})" if skipped else "")
              + f" · replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)))
        if not rep["ok"]:
            failures.append(f"{case['name']}: replay failed {rep['mismatches']}")
        print()

    print("── the strategist's plan (same for every request with these keys)")
    res = system.ask(state_of(cases[2]["state"]))
    print("\n".join("   " + line for line in str(res.flow).splitlines()))
    print(f"   at run time the hard check failed first, so {len(res.trace.skipped)} steps were skipped: "
          + ", ".join(n for n, _ in res.trace.skipped))
    print(f"   paid lookups over all cases: {external_run} made, {external_skipped} skipped by the early exit\n")

    print("── the trace as audit evidence")
    res = system.ask(state_of(cases[1]["state"]))
    rec = next(r for r in res.trace.records if r.name == "structuring")
    print(f"   recorded: structuring = {rec.value}")
    t = copy.deepcopy(res.trace)
    next(r for r in t.records if r.name == "structuring").value = {**rec.value, "count": 2}
    print(f"   someone edits the record to count 2 after the fact -> replay: {t.replay(task.cat)['mismatches'][:2]}")
    print(f"   untouched trace -> replay ok: {res.trace.replay(task.cat)['ok']}, chain head {res.trace.records[-1].hash}\n")

    print(tally)
    res = system.ask(state_of(cases[1]["state"]))
    print(f"\nthe audit of '{cases[1]['name']}' (res.audit('file_sar')):\n" + str(res.audit('file_sar')) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
