"""Support triage: learn the intent rules, then run every scenario in cases.json — answers, cited evidence, the strategist's
flow, expected answers, trace replay and timing. Exits non-zero on a mismatch.

Run from the repo root:  uv run python gallery/01_support_triage/run.py"""
import copy
import json
import sys
import time
import types
from pathlib import Path

from solvi import System
from solvi.rules import RuleList

HERE = Path(__file__).resolve().parent


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
            if r.status != "ok":
                print(f"    {q} {r.status}: {r.why}")
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

    # the intent rule list: learned again here with System.learn_rule to time it, then checked on 500 fresh synthetic tickets
    fresh_task = load_task()                            # learn_rule installs the list as the rule, so use a separate copy
    t0 = time.perf_counter()
    rules = System(fresh_task.cat, fresh_task.QUESTIONS).learn_rule("intent", task.TRAIN, ["cue_words"], min_support=3,
                                                                    min_precision=0.8)
    ms_learn = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    RuleList(["cue_words"], min_support=3, min_precision=0.8).fit([{"cue_words": task.cue_words(s["message"])} for s, _ in task.TRAIN],
                                                                  [y for _, y in task.TRAIN])
    ms_direct = (time.perf_counter() - t0) * 1000
    assert str(rules) == str(task.INTENT_RULES)
    fresh = task.make_tickets(500, seed=99)
    acc = sum(rules.predict({"cue_words": task.cue_words(s["message"])})[0] == y for s, y in fresh) / len(fresh)
    print(f"intent rules learned from {len(task.TRAIN)} labeled tickets: {len(rules.rules)} lines in {ms_learn:.0f} ms with "
          f"System.learn_rule ({ms_direct:.0f} ms fitting RuleList on cue_words alone, as task.py does); "
          f"accuracy on 500 fresh synthetic tickets {acc:.3f}")
    print("\n".join(str(rules).splitlines()[:8]) + f"\n   ... {len(rules.rules) - 8} more lines, then the default\n")

    cases = json.loads((HERE / "cases.json").read_text())
    bad = run_cases(task, cases)

    res = System(task.cat, task.QUESTIONS).ask(prepared(task, json.loads((HERE / "state.json").read_text())))
    print("\nthe strategist's flow for state.json:\n" + str(res.flow))
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[3]["state"]))
    print(f"\nearly exit on '{cases[3]['name']}': " + (", ".join(f"{n} ({w})" for n, w in res.trace.skipped) or "nothing skipped"))
    sys.exit(1 if bad else 0)
