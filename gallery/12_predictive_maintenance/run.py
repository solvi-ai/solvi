"""Predictive maintenance: learn a readable "likely fault" rule list from 150 labeled past incidents (learn_rule), check it on
300 new ones, then run every case: health, fault, the computed trends, z-scores and time to the alert level, trace replay,
timing. Exits non-zero if any answer differs from cases.json.

Run:  uv run python gallery/12_predictive_maintenance/run.py"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
import time
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pdm_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)

BASELINE = {"temp_c": {"mean": 62.0, "sd": 2.0}, "vib_rms": {"mean": 1.8, "sd": 0.25}, "current_a": {"mean": 41.0, "sd": 1.2}}
FAULTS = ["none", "bearing", "imbalance", "electrical", "cooling"]


def simulate(rng, fault, hours=48):
    """48 hourly readings of a pump with the given fault developing (synthetic physics, noisy). Returns the raw JSON state."""
    onset = rng.randint(4, 20) if fault == "imbalance" else rng.randint(8, 30)
    vib_amp = {"bearing": rng.uniform(1.2, 3.2), "imbalance": rng.uniform(1.0, 2.4)}.get(fault, 0.0)
    heat = {"bearing": rng.uniform(2, 6), "electrical": rng.uniform(3, 9), "cooling": rng.uniform(8, 20)}.get(fault, 0.0)
    amps = rng.uniform(3.5, 8.0) if fault == "electrical" else 0.0
    temp, vib, cur = [], [], []
    for i in range(hours):
        r = max(0.0, (i - onset) / (hours - 1 - onset))            # 0 before onset, ramps to 1 at the last reading
        wear = r ** 1.5 if fault == "bearing" else float(i >= onset)    # bearings wear progressively, imbalance is a step
        temp.append(round(62.0 + rng.gauss(0, 0.9) + heat * r, 1))
        vib.append(round(1.8 + rng.gauss(0, 0.12) + vib_amp * wear, 2))
        cur.append(round(41.0 + rng.gauss(0, 0.6) + amps * r, 1))
    return {"machine": {"id": f"P-{rng.randint(100, 999)}", "type": "centrifugal pump, 55 kW", "iso_class": "II"},
            "readings": {"hours": list(range(-hours + 1, 1)), "temp_c": temp, "vib_rms": vib, "current_a": cur},
            "baseline": BASELINE}


def incidents(rng, n, noise=0.05):
    """labeled past incidents: the fault found by the technician (5% of labels wrong, as in real maintenance logs)"""
    out = []
    for _ in range(n):
        f = rng.choice(FAULTS)
        label = rng.choice([x for x in FAULTS if x != f]) if rng.random() < noise else f
        out.append((task.prepare(simulate(rng, f)), label, f))
    return out


def state_of(raw):
    return task.prepare(json.loads(json.dumps(raw)))


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    rng = random.Random(12)
    past = incidents(rng, 150)
    test = incidents(rng, 300, noise=0.0)
    t0 = time.perf_counter()
    rules = system.learn_rule("fault", [(s, y) for s, y, _ in past], task.FAULT_FACTS)
    t_learn = (time.perf_counter() - t0) * 1000
    acc = sum(system.ask(s, ["fault"])["fault"].answer == f for s, _, f in test) / len(test)
    print(f"'fault' learned from {len(past)} labeled incidents in {t_learn:.0f} ms: {len(rules.rules)} rules\n{rules}")
    print(f"accuracy on {len(test)} new incidents (true fault): {acc:.3f}\n")

    cases = json.loads((HERE / "cases.json").read_text())
    tally = Tally()
    failures, times = [], []
    for i, case in enumerate(cases, 1):
        state = state_of(case["state"])
        res = system.ask(state)
        times.append(res.ms)
        v = res.values
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        for q, r in res.results.items():
            print(f"    {q:7s} {got(r)!s:9s} {r.status:8s} {r.confidence:.2f}  {r.why}")
        fired = system.learned_rules["fault"].predict(v)[1] if all(f in v for f in task.FAULT_FACTS) else None
        facts = [f"vib {v['vib_latest']} mm/s (z {v['vib_z']}, slope {v['vib_slope'] * 24:+.2f}/day)" if "vib_z" in v else None,
                 f"temp {v['temp_latest']} °C (z {v['temp_z']}, slope {v['temp_slope'] * 24:+.1f}/day)" if "temp_z" in v else None,
                 f"current {v['current_latest']} A (z {v['current_z']})" if "current_z" in v else None,
                 f"vib alert in {v['hours_to_vib_alert']} h" if v.get("hours_to_vib_alert") is not None else None]
        missing = [s for s in task.SIGNALS if s not in state]
        print("    " + " · ".join(x for x in facts if x) + (f" · offline: {', '.join(missing)}" if missing else ""))
        if fired:
            print(f"    fault rule fired: if {fired['if']} -> {fired['then']}")
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print(tally)
    res = system.ask(state_of(cases[1]["state"]))
    print(f"\nthe audit of '{cases[1]['name']}' (res.audit('fault')):\n" + str(res.audit('fault')) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
