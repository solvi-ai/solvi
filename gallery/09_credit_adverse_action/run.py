"""Run every credit case: decision, principal reason and the full adverse-action reason list, trace replay, timing. Before
that, learn "refer to a senior underwriter?" from 200 synthetic past files with fit_fast (milliseconds); afterwards an
underwriter reviews new files one by one and each verdict is absorbed instantly by teach (time shown).
Exits non-zero if any answer differs from cases.json.

Run:  uv run python gallery/09_credit_adverse_action/run.py"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("credit_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)
Q = "refer_to_underwriter"


def state_of(raw):
    return json.loads(json.dumps(raw))


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def short(s, n=150):
    return s if len(s) <= n else s[:n - 1] + "…"


def past_file(rng, i):
    """A synthetic past application and whether a senior underwriter took it. The habit is not written anywhere: large
    loans against a short job, self-employed with a thin file, anything from 45 000; 4% of the labels are inconsistent."""
    income = round(rng.lognormvariate(11.0, 0.4), -2)
    s = {"application_id": f"H-{i}", "age": rng.randint(19, 70), "residency_status": "citizen", "annual_income": income,
         "monthly_debt_payments": round(rng.uniform(0, 0.35) * income / 12), "requested_amount": round(rng.uniform(2, 60)) * 1000,
         "term_months": rng.choice([24, 36, 48, 60]), "annual_rate": round(rng.uniform(0.06, 0.18), 3),
         "credit_history_months": rng.randint(3, 300), "credit_score": max(480, min(850, round(rng.gauss(690, 60)))),
         "delinquencies_24m": rng.choice([0, 0, 0, 0, 1, 1, 2, 3]), "utilization": round(rng.random(), 2),
         "inquiries_6m": rng.randint(0, 7), "employment_months": rng.randint(0, 240), "self_employed": rng.random() < 0.2}
    lti = s["requested_amount"] / income
    y = (lti > 0.4 and s["employment_months"] < 18) or (s["self_employed"] and s["credit_history_months"] < 36) \
        or s["requested_amount"] >= 45000
    if rng.random() < 0.04:
        y = not y
    return s, "yes" if y else "no"


def accuracy(system, data):
    return sum(system.ask(s, [Q])[Q].answer == y for s, y in data) / len(data)


def _refit(data):
    s = System(task.cat, task.QUESTIONS)
    s.fit_fast(Q, data, features=task.UNDERWRITER_FACTS)
    return s


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    rng = random.Random(3)
    history = [past_file(rng, i) for i in range(200)]
    test = [past_file(rng, 1000 + i) for i in range(400)]
    stream = [past_file(rng, 5000 + i) for i in range(300)]
    head = system.fit_fast(Q, history, features=task.UNDERWRITER_FACTS)
    acc0 = accuracy(system, test)
    print(f"'{Q}' learned from {len(history)} past files in {head.fit_ms:.1f} ms "
          f"(exact leave-one-out {head.loo_acc:.3f}); accuracy on {len(test)} new files {acc0:.3f}\n")

    cases = json.loads((HERE / "cases.json").read_text())
    tally = Tally()
    failures, times, before = [], [], {}
    for i, case in enumerate(cases, 1):
        res = system.ask(state_of(case["state"]))
        times.append(res.ms)
        before[case["name"]] = {q: got(r) for q, r in res.results.items()}
        v = res.values
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        for q, r in res.results.items():
            print(f"    {q:20s} {got(r)!s:44s} {r.status:7s} {r.confidence:.2f}")
        if "points" in v:
            print(f"    points {v['points']}/100, DTI {v['dti']}, knock-outs {v['knockouts'] or 'none'}")
            print("    adverse-action reasons: " + ("none required (approved)" if res["decision"].answer == "approve"
                                                     else str(v["adverse_action_reasons"])))
        else:
            print(f"    {res['decision'].why}; scoring skipped ({len(res.trace.skipped)} steps not run)")
        print(f"    underwriter head: {short(res[Q].why, 120)}")
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print(f"── an underwriter reviews {len(stream)} new files; each verdict is taught to the head at once")
    ms, corrected = [], 0
    for s, y in stream:
        corrected += system.ask(s, [Q])[Q].answer != y
        ms.append(system.teach(Q, s, y))                         # instant: a rank-one update, nothing retrained
    ms.sort()
    print(f"   {len(ms)} verdicts ({corrected} of them corrections), each absorbed in {ms[len(ms) // 2]:.3f} ms "
          f"(median, max {ms[-1]:.3f} ms)")
    print(f"   accuracy on the same {len(test)} new files: {acc0:.3f} -> {accuracy(system, test):.3f} "
          f"(refitting from scratch on all {len(history) + len(stream)} files gives {accuracy(_refit(history + stream), test):.3f})")
    moved = {q: [] for q in system.questions}
    for case in cases:
        after = system.ask(state_of(case["state"]))
        for q, r in after.results.items():
            if got(r) != before[case["name"]][q]:
                moved[q].append(f"{case['name']} {before[case['name']][q]} -> {got(r)} (p {r.probs.get(r.answer, 0):.2f})")
    print("   answers on the cases changed by these verdicts: " + "; ".join(f"{q} {len(m)}" + (f" [{', '.join(m)}]" if m else "")
                                                                        for q, m in moved.items()))
    print("   (the learned answer may move; decisions and reasons come from rules and hard checks, which teach does not touch)\n")

    print(tally)
    res = system.ask(state_of(cases[8]["state"]))
    print(f"\nthe audit of '{cases[8]['name']}' (res.audit('refer_to_underwriter')):\n" + str(res.audit('refer_to_underwriter')) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
