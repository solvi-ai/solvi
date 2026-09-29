"""Run every prompt: which producer picked (the user's own naming, cited / the decider / a person), the decider's
probabilities and why it escalated, the skill and the outcome, the audit, trace replay. Then act_guard's promise on the
labelled prompts and the three kinds of answer — a pick, a confident "none", an abstention — on fresh ones, with the
wrong picks counted apart. Exits non-zero if any answer differs from cases.json.

Run:  uv run python gallery/15_skill_picker/run.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("skill_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def how_picked(res):
    rec = next(r for r in res.trace.records if r.name == "picked")
    if rec.producer == "named_skill":
        s, e, src = rec.quote
        return f"named by the user: prompt[{s}:{e}] = {res.trace.init[src][s:e]!r}"
    lines = []
    for p, why in rec.tried or []:
        if p == "skill_decider" and why != "accepted":
            lines.append(f"decider escalated: {why}")
    if rec.producer == "skill_decider":
        top = sorted(rec.probs.items(), key=lambda kv: -kv[1])[:2]
        lines.append("decider: " + ", ".join(f"{k} {v:.2f}" for k, v in top))
    return "; ".join(lines) or rec.producer


def on_fresh(system, n=1000, seed=3):
    """right picks, right "none", abstentions and wrong answers on fresh synthetic prompts (another seed)"""
    c = Counter()
    for prompt, label in task.synthetic_prompts(n, seed=seed):
        r = system.ask({"prompt": prompt})["skill"]
        if r.status == "abstain":
            c["abstained"] += 1
        elif r.answer == label:
            c["right none" if label == task.NONE else "right pick"] += 1
        else:
            c["none instead of a skill" if r.answer == task.NONE else "wrong pick"] += 1
    return c


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    cases = json.loads((HERE / "cases.json").read_text())
    tally = Tally()
    failures, times = [], []
    for i, case in enumerate(cases, 1):
        res = system.ask(case["state"])
        times.append(res.ms)
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        print(f"    prompt: {case['state']['prompt']!r}")
        print(f"    {how_picked(res)}")
        for q, r in res.results.items():
            print(f"    {q:8s} {got(r)!s:18s} {r.status:7s} {r.confidence:.2f}" + ("  " + r.why if r.status != "ok" else ""))
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print(tally)
    info = task.CALIBRATION
    print(f"\nact_guard, risk {task.RISK:g}, on {info['n']} labelled prompts (synthetic, seeded): threshold {info['threshold']:.3f} "
          f"on the {info['signal']}, answered alone {info['answered']:.0%}, risk {info['risk']:.3f}; conformal candidates "
          f"at {task.CANDIDATES['coverage']:.0%}: {task.CANDIDATES['mean_size']:.2f} per prompt on average")
    c = on_fresh(system)
    print("1000 fresh synthetic prompts: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    res = system.ask(cases[6]["state"])
    print(f"\nthe audit of '{cases[6]['name']}':\n" + str(res.audit("skill")) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
