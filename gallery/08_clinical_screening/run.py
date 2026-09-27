"""Run every clinical screening case (DEMO ONLY, NOT MEDICAL ADVICE): answers, NEWS2 points per parameter, what was missing,
trace replay, timing. Then measure an answer-only baseline: a ridge answerer (solvi's fit_fast on the raw vital signs, no
table, no rules) trained on synthetic patients labeled by the exact NEWS2 rules. Exits non-zero if any answer differs from
cases.json.

Run:  uv run python gallery/08_clinical_screening/run.py"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

from solvi import Answer, Catalog, Question, System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("clinical_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)

RAW = ["resp_rate", "spo2", "spo2_scale", "on_oxygen", "systolic_bp", "pulse", "consciousness", "temperature"]


def state_of(raw):
    return task.prepare(json.loads(json.dumps(raw)))


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def short(s, n=150):
    return s if len(s) <= n else s[:n - 1] + "…"


def patient(rng):
    """A synthetic ward patient: each vital normal (70%) or anywhere in a wide abnormal range (30%)."""
    def ab():
        return rng.random() < 0.3

    return {"resp_rate": rng.randint(6, 36) if ab() else round(rng.gauss(16, 2)),
            "spo2": rng.randint(78, 95) if ab() else rng.randint(94, 100),
            "spo2_scale": 2 if rng.random() < 0.15 else 1, "on_oxygen": rng.random() < 0.25,
            "systolic_bp": rng.randint(70, 230) if ab() else round(rng.gauss(125, 15)),
            "pulse": rng.randint(35, 150) if ab() else round(rng.gauss(80, 12)),
            "consciousness": rng.choice("CVPU") if rng.random() < 0.15 else "A",
            "temperature": round(rng.uniform(34.0, 40.5), 1) if ab() else round(rng.gauss(37.0, 0.4), 1)}


def hard_emergency(s):
    return s["spo2"] < 85 or (s["systolic_bp"] < 90 and s["consciousness"] != "A")


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    cases = json.loads((HERE / "cases.json").read_text())
    failures, times, tally = [], [], Tally()
    for i, case in enumerate(cases, 1):
        state = state_of(case["state"])
        res = system.ask(state)
        times.append(res.ms)
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        for q, r in res.results.items():
            print(f"    {q:13s} {got(r)!s:13s} {r.status:8s} {r.confidence:.2f}  {short(r.why)}")
        print("    " + line(tally.add(check(res, system))))            # asserts the audit's invariants
        n2 = res.values.get("news2")
        missing = sorted(set(RAW) - set(state))
        print(f"    NEWS2 {n2['total']} points {n2['by_parameter']}" if n2 else "    NEWS2 not computed", end="")
        print(f" · qSOFA {res.values.get('qsofa', '—')}" + (f" · missing: {', '.join(missing)}" if missing else ""))
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    # ---- an answer-only baseline, measured: typed answer + probability from the raw vitals, no table, no rules
    rng = random.Random(7)
    pool = [patient(rng) for _ in range(6000)]
    data = [(s, system.ask(dict(s), ["escalation"])["escalation"].answer) for s in pool]     # the exact rules = ground truth
    train, test = data[:4000], data[4000:]
    exact = sum(system.ask(dict(s), ["escalation"])["escalation"].answer == y for s, y in test)
    print("── answer-only baseline (ridge on the 8 raw vitals, trained on 4 000 synthetic patients, tested on 2 000)")
    print(f"   solvi's table + rules: {exact}/{len(test)} by construction")
    hard = [s for s, _ in test if hard_emergency(s)]
    for kind in ("ordinal", "choice"):      # the ordinal head answers with the median level; a choice head with the most likely
        q = Question("escalation", "Escalation level", getattr(Answer, kind)(task.ESCALATION))
        base = System(Catalog(), [q])
        head = base.fit_fast("escalation", train, features=RAW)
        pred = [base.ask(dict(s), ["escalation"])["escalation"].answer for s, _ in test]
        acc = sum(p == y for p, (_, y) in zip(pred, test)) / len(test)
        emerg = [p for p, (_, y) in zip(pred, test) if y == "emergency"]
        hard_missed = sum(p != "emergency" for p, (s, _) in zip(pred, test) if hard_emergency(s))
        print(f"   as {kind:7s}: fitted in {head.fit_ms:.0f} ms; accuracy {acc:.3f}; emergencies missed "
              f"{sum(p != 'emergency' for p in emerg)} of {len(emerg)}; hard-rule emergencies (SpO2 < 85 or shock) missed "
              f"{hard_missed} of {len(hard)}; routine sent to emergency "
              f"{sum(p == 'emergency' and y == 'routine' for p, (_, y) in zip(pred, test))}")
    print()

    print(tally)
    hyp = next(c for c in cases if c["name"] == "hypoxic_hard_check")
    print(f"\nthe audit of '{hyp['name']}' (res.audit('escalation')):\n" + str(system.ask(state_of(hyp["state"])).audit("escalation")) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
