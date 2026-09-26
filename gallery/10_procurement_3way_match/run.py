"""Run every 3-way-match case: answers, the line-by-line match, the amount in base currency, what the early exit skipped, trace
replay, timing. Exits non-zero if any answer differs from cases.json.

Run:  uv run python gallery/10_procurement_3way_match/run.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from solvi import System

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("p2p_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)


def state_of(raw):
    return json.loads(json.dumps(raw))


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def short(s, n=140):
    return s if len(s) <= n else s[:n - 1] + "…"


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    cases = json.loads((HERE / "cases.json").read_text())
    failures, times = [], []
    for i, case in enumerate(cases, 1):
        state = state_of(case["state"])
        res = system.ask(state)
        times.append(res.ms)
        v, inv = res.values, state["invoice"]
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        for q, r in res.results.items():
            print(f"    {q:17s} {got(r)!s:8s} {r.status:8s} {r.confidence:.2f}  {short(r.why)}")
        amount = f"{inv['total']:,.2f} {inv['currency']}"
        if "invoice_total_base" in v:
            amount += f" = {v['invoice_total_base']:,.2f} USD (limit {state['approval_limits'][state['approver']['role']]:,} for {state['approver']['role']})"
        else:
            amount += " -> no rate to the base currency"
        print(f"    amount {amount}")
        if "line_match" in v:
            print("    lines: " + "; ".join(f"{m['line']} {m['sku']} qty {m.get('qty_invoiced', '?')}/{m.get('qty_received', '?')} "
                                             f"price {m.get('price_var_pct', 0):+.2f}%" + (" ✗" if "problem" in m else " ✓")
                                             for m in v["line_match"]))
        skipped = [n for n, _ in res.trace.skipped]
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    {len(res.trace.records)} steps run, {len(skipped)} skipped" + (f" ({', '.join(skipped)})" if skipped else "")
              + f" · replay {'ok' if rep['ok'] else 'FAILED'} · {res.ms:.2f} ms · "
              + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print("── the strategist's plan for these keys (hard checks and what they read come first)")
    print("\n".join("   " + line for line in str(system.ask(state_of(cases[0]["state"])).flow).splitlines()))
    times.sort()
    print(f"\n{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
