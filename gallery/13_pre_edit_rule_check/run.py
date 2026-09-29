"""Run every proposed edit: the rules in scope for its path, what code found (with lines), what the decider said per fuzzy
rule and which producer answered (out of scope / the decider / a person), the decision, the audit, trace replay. Then
what act_guard promised on the labelled examples and what happened on fresh ones. Exits non-zero if any answer differs
from cases.json.

Run:  uv run python gallery/13_pre_edit_rule_check/run.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("pre_edit_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)


def got(r):
    """the answer as cases.json writes it: "abstain", an option, a list for multi-label"""
    return "abstain" if r.status == "abstain" else list(r.answer) if isinstance(r.answer, tuple) else r.answer


def producer(res, fact):
    """which producer answered a fuzzy rule, and what the ones before it did"""
    rec = next((r for r in res.trace.records if r.name == fact), None)
    if rec is None:
        return f"{fact}: not asked"
    tried = "; ".join(f"{p.split('_')[-1] if p.endswith(('_scope', '_person')) else 'decider'}: {why}"
                      for p, why in (rec.tried or [])[:-1])
    return f"{fact} = {rec.value!r} (by {rec.producer})" + (f"  [before it: {tried}]" if tried else "")


def code_findings(v):
    """when a hard check decided early, `findings` was not computed: what code found, from the facts it computed"""
    out = [f"no-secrets: line {n}: {kind} {start!r}" for n, kind, start in v.get("secret_lines", [])]
    out += [f"browser-storage: line {n}: reads {what}" for n, what in v.get("browser_storage_lines", [])]
    return out + [f"reversible-migration: line {n}: {why}" for n, why in v.get("migration_problems", [])]


def on_fresh(part, rule, n=1000, seed=11):
    """answered share and P(answered alone and wrong) on fresh synthetic examples (another seed)"""
    test = task.labelled(rule, n, seed=seed)
    ds = [part(x) for x, _ in test]
    auto = [d.escalate is None for d in ds]
    wrong = [a and d.value != y for a, d, (_, y) in zip(auto, ds, test)]
    return sum(auto) / n, sum(wrong) / n


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
        print(f"    {case['state']['path']}: {len(v.get('added_lines', []))} lines added; rules in scope: "
              f"{', '.join(task.rules_in_scope(case['state']['path']))}")
        for f in v["findings"] if "findings" in v else code_findings(v):
            print(f"    finding: {f}")
        for fact in ("require_role", "no_pii_in_logs"):
            if fact in v:
                print(f"    {producer(res, fact)}")
        for q, r in res.results.items():
            print(f"    {q:14s} {got(r)!s:36s} {r.status:7s} {r.confidence:.2f}" + ("  " + r.why if r.status != "ok" else ""))
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records) · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print(tally)
    print(f"\nact_guard, risk {task.RISK:g}, on {task.CALIBRATION['require-role']['n']} labelled examples per fuzzy rule "
          "(synthetic, seeded):")
    for rule, part in (("require-role", task.require_role_decider), ("no-pii-in-logs", task.no_pii_in_logs_decider)):
        info = task.CALIBRATION[rule]
        a, r = on_fresh(part, rule)
        print(f"  {rule:15s} threshold {info['threshold']:.3f} on the {info['signal']}: answered alone {info['answered']:.0%}, "
              f"risk {info['risk']:.3f}; on 1000 fresh examples: answered {a:.0%}, risk {r:.3f}")
    res = system.ask(cases[2]["state"])
    print(f"\nthe audit of '{cases[2]['name']}':\n" + str(res.audit("decision")) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
