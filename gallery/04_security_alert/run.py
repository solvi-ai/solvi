"""Security alert: run every scenario in cases.json — answers, computed numbers, the strategist's
flow, expected answers, trace replay and timing. Exits non-zero on a mismatch.

Run from the repo root:  uv run python gallery/04_security_alert/run.py"""
import copy
import json
import sys
import types
from pathlib import Path

from solvi import System
from solvi.runtime import vhash

HERE = Path(__file__).resolve().parent
SHOW_WHY = []                                           # print the reason for these answers even when they are plain "ok"
# computed facts printed for each case
NUMBERS = ["distance_km", "hours_between", "travel_kmh", "impossible_travel", "new_device", "failed_spike_pct"]


def load_task():
    mod = types.ModuleType("task")                      # the way the playground loads a preset: exec in a fresh namespace
    exec(compile((HERE / "task.py").read_text(), str(HERE / "task.py"), "exec"), mod.__dict__)
    return mod


def prepared(task, raw):
    s = copy.deepcopy(raw)
    return task.prepare(s) if callable(getattr(task, "prepare", None)) else s


def fmt(r):
    return "— (abstain)" if r.status == "abstain" else f"{r.answer} (forced)" if r.status == "forced" else r.answer


def cited(res, state):
    out = []
    for r in res.trace.records:
        if r.quote and r.quote[1] > r.quote[0]:
            s, e, src = r.quote
            out.append(f"{r.name}={r.value} «{state[src][s:e]}» [{s}:{e}]")
    return out


def run_cases(task, cases):
    system = System(task.cat, task.QUESTIONS)
    bad, times, replay_ok = [], [], 0
    for i, case in enumerate(cases, 1):
        state = prepared(task, case["state"])
        res = system.ask(state)
        rep = res.trace.replay(task.cat)
        replay_ok += rep["ok"]
        times.append(res.ms)
        print(f"[{i}] {case['name']:40s} {res.ms:6.2f} ms   replay {'ok' if rep['ok'] else 'FAILED'}")
        print("    " + "  ".join(f"{q}={fmt(r)}" for q, r in res.results.items()))
        for line in cited(res, state):
            print("    cited: " + line)
        for q, r in res.results.items():
            if r.status != "ok" or q in SHOW_WHY:
                print(f"    {q} {r.status}: {r.why}")
        v = res.values
        shown = [f"{k}={v[k]}" for k in NUMBERS if k in v]
        if shown:
            print("    computed: " + ", ".join(shown))
        if res.trace.skipped:
            print(f"    early exit: {len(res.trace.skipped)} steps skipped")
        for q, want in case["expected"].items():
            r = res[q]
            want_status = case.get("status", {}).get(q)
            if r.answer != want or (want_status and r.status != want_status):
                bad.append((case["name"], q, r.answer, r.status, want, want_status))
                print(f"    MISMATCH {q}: got {r.answer} [{r.status}], expected {want} [{want_status or 'any'}]")
    times.sort()
    print(f"\n{len(cases) - len({b[0] for b in bad})}/{len(cases)} cases match, replay ok on {replay_ok}/{len(cases)} traces, "
          f"median decision {times[len(times) // 2]:.2f} ms")
    return bad


if __name__ == "__main__":
    task = load_task()

    cases = json.loads((HERE / "cases.json").read_text())
    bad = run_cases(task, cases)

    res = System(task.cat, task.QUESTIONS).ask(prepared(task, json.loads((HERE / "state.json").read_text())))
    print("\nthe strategist's flow for state.json:\n" + str(res.flow))
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[0]["state"]))
    print(f"\nearly exit on '{cases[0]['name']}': " + (", ".join(f"{n} ({w})" for n, w in res.trace.skipped) or "nothing skipped"))

    # an insider edits the stored trace to make the Lagos login look like a normal commute, and re-hashes the chain
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[1]["state"]))
    t = copy.deepcopy(res.trace)
    i = next(k for k, r in enumerate(t.records) if r.name == "travel_kmh")
    t.records[i].value = 80.0
    prev = t.records[i].prev
    for r in t.records[i:]:
        r.prev, r.hash = prev, vhash(dict(r.body(), prev=prev))
        prev = r.hash
    print(f"\ntampered trace of '{cases[1]['name']}' (travel_kmh 2309.2 -> 80.0, hashes recomputed): {t.replay(task.cat)}")
    sys.exit(1 if bad else 0)
