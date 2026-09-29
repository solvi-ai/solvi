"""Run every "charged twice" ticket: the customer's claim cited from the ticket (quote + offsets), what the ledger shows, the
refund decision and reply, trace replay, timing. Exits non-zero if any answer differs from cases.json.

Run:  uv run python gallery/11_refund_double_charge/run.py"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("refund_task", HERE / "task.py")
task = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(task)


def state_of(raw):
    return task.prepare(json.loads(json.dumps(raw)))


def got(r):
    return "abstain" if r.status == "abstain" else r.answer


def cited(res, name):
    rec = next((r for r in res.trace.records if r.name == name), None)
    if rec is None or rec.quote is None:
        return f"{name}: — ({rec.error if rec else 'not run'})"
    s, e, src = rec.quote
    return f"{name} = {rec.value!r} " + (f"from ticket[{s}:{e}] {res.trace.init[src][s:e]!r}" if e > s else "(no such phrase)")


if __name__ == "__main__":
    system = System(task.cat, task.QUESTIONS)
    cases = json.loads((HERE / "cases.json").read_text())
    tally = Tally()
    failures, times, claim_vs_ledger, unclear = [], [], 0, 0
    for i, case in enumerate(cases, 1):
        res = system.ask(state_of(case["state"]))
        times.append(res.ms)
        v = res.values
        print(f"[{i}/{len(cases)}] {case['name']}  ({case['note']})")
        print(f"    ticket: {case['state']['ticket']!r}")
        print(f"    claim (cited): {cited(res, 'double_charge_claim')}; {cited(res, 'claimed_amount')}")
        pairs = v.get("duplicate_pairs", [])
        print("    ledger: " + ("; ".join(f"{p['first']} + {p['duplicate']} {p['merchant']} {p['amount']:.2f}, {p['minutes_apart']} min apart"
                                   + (" (already refunded)" if p["refunded"] else "") for p in pairs) or "no double charge")
              + (f" · refund {v['refund_amount']:.2f}, free balance {v['free_balance']:.2f}" if "free_balance" in v else ""))
        for q, r in res.results.items():
            print(f"    {q:22s} {got(r)!s:22s} {r.status:7s} {r.confidence:.2f}" + ("  " + r.why if r.status != "ok" else ""))
        if res["customer_claims_double"].status == "abstain":
            unclear += 1
        else:
            claim_vs_ledger += res["customer_claims_double"].answer != res["is_double_charge"].answer
        print("    " + audit_line(tally.add(check(res, system))))      # asserts the audit's invariants
        rep = res.trace.replay(task.cat)
        bad = [f"{q}: expected {want}, got {got(res[q])}" for q, want in case["expected"].items() if got(res[q]) != want]
        failures += [f"{case['name']}: {b}" for b in bad] + ([] if rep["ok"] else [f"{case['name']}: replay {rep['mismatches']}"])
        print(f"    replay {'ok' if rep['ok'] else 'FAILED'} ({rep['steps']} records, quotes checked against the ticket) · "
              f"{res.ms:.2f} ms · " + ("expected ✓" if not bad else "MISMATCH: " + "; ".join(bad)) + "\n")

    print(f"the ticket and the ledger disagree in {claim_vs_ledger} of {len(cases)} cases (the claim is unclear in {unclear} more); "
          "every refund decision followed the ledger")
    print(tally)
    res = system.ask(state_of(cases[1]["state"]))
    print(f"\nthe audit of '{cases[1]['name']}' (the claim and the ledger):\n" + str(res.audit(['customer_claims_double', 'is_double_charge'])) + "\n")
    times.sort()
    print(f"{len(cases) - len({f.split(':')[0] for f in failures})}/{len(cases)} cases as expected; "
          f"decision time median {times[len(times) // 2]:.2f} ms, max {times[-1]:.2f} ms")
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures))
        sys.exit(1)
