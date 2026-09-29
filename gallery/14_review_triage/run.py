"""Run every change: what the diff touches, the seven risk answers (code or decider, and who answered), the reasons and
the review route, the audit, trace replay. Then the promise: act_guard per decider question on 400 labelled synthetic
changes, and how many risky changes went to quick review among 2000 fresh ones. Exits non-zero if any answer differs
from cases.json.

Run:  uv run python gallery/14_review_triage/run.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("review_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def short(res, q):
    """one risk answer: yes / no / ? (not sure), and D when a decider answered it"""
    r = res[q]
    mark = "?" if r.status == "abstain" else r.answer
    return f"{q}={mark}" + ("·D" if q in task.DECIDED else "")


def on_fresh(system, n=2000, seed=1):
    """quick share and P(risky and quick) on fresh synthetic changes (another seed)"""
    quick = risky_quick = 0
    for state, labels in task.synthetic_changes(n, seed=seed):
        res = system.ask(state)
        q = res["review"].answer == "quick"
        risky = any(labels.values()) or any(res.values[f] is True for f in task.FLAGS[4:])
        quick += q
        risky_quick += q and risky
    return quick / n, risky_quick / n


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    cases = json.loads((HERE / "cases.json").read_text())
    tally = Tally()
    failures, times = [], []
    for i, case in enumerate(cases, 1):
        res = system.ask(case["state"])
        times.append(res.ms)
        v = res.values
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        print(f"    {case['state']['title']!r}: {len(v['files'])} file(s), {v['changed_lines']} changed lines, "
              f"areas {v['areas']}, tests {v['test_files'] or 'none'}")
        print("    " + "  ".join(short(res, q) for q in task.FLAGS))
        for q in task.DECIDED:
            rec = next(r for r in res.trace.records if r.name == q)
            if rec.producer.endswith("_to_person"):
                print(f"    {q}: to a person — {rec.tried[0][1]}")
        for reason in v.get("risk_reasons", []):
            print(f"    reason: {reason}")
        r = res["review"]
        print(f"    review = {got(r)} ({r.status}, {r.confidence:.2f})")
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print(tally)
    print(f"\nact_guard at risk {task.RISK:g} / {len(task.DECIDED)} per decider question, on {len(task.EXAMPLES)} labelled "
          "changes (synthetic, seeded):")
    for q, info in task.CALIBRATION.items():
        print(f"  {q:22s} threshold {info['threshold']:.3f}: answered alone {info['answered']:.0%}, error among them "
              f"{info['error']:.1%}, risk {info['risk']:.3f}")
    quick, risky_quick = on_fresh(system)
    print(f"2000 fresh synthetic changes: {quick:.0%} to quick review; risky and sent to quick review: {risky_quick:.2%} "
          f"(the promise: at most {task.RISK:.0%})")
    res = system.ask(cases[11]["state"])
    print(f"\nthe audit of '{cases[11]['name']}':\n" + str(res.audit(["touches_auth", "review"])) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
