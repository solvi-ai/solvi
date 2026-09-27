"""Email routing: learn the routing table, then run every scenario in cases.json — answers, cited evidence, the strategist's
flow, expected answers, trace replay and timing. Exits non-zero on a mismatch.

Run from the repo root:  uv run python gallery/02_email_routing/run.py"""
import copy
import json
import sys
import time
import types
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402
from solvi.rules import RuleList

HERE = Path(__file__).resolve().parent
SHOW_WHY = ["team"]                                     # print the reason for these answers even when they are plain "ok"


def load_task():
    mod = types.ModuleType("task")                      # the way the playground loads a preset: exec in a fresh namespace
    exec(compile((HERE / "task.py").read_text(), str(HERE / "task.py"), "exec"), mod.__dict__)
    return mod


def prepared(task, raw):
    s = copy.deepcopy(raw)
    return task.prepare(s) if callable(getattr(task, "prepare", None)) else s


def plain(a):
    return list(a) if isinstance(a, tuple) else a        # a multi-label answer is a tuple; cases.json has lists


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
    bad, times, replay_ok, tally = [], [], 0, Tally()
    for i, case in enumerate(cases, 1):
        state = prepared(task, case["state"])
        res = system.ask(state)
        rep = res.trace.replay(task.cat)
        replay_ok += rep["ok"]
        times.append(res.ms)
        print(f"[{i}] {case['name']:40s} {res.ms:6.2f} ms   replay {'ok' if rep['ok'] else 'FAILED'}")
        print("    " + "  ".join(f"{q}={fmt(r)}" for q, r in res.results.items()))
        print("    " + audit_line(tally.add(check(res, system))))            # asserts the audit's invariants
        for line in cited(res, state):
            print("    cited: " + line)
        for q, r in res.results.items():
            if r.status != "ok" or q in SHOW_WHY:
                print(f"    {q} {r.status}: {r.why}")
        for q, want in case["expected"].items():
            r = res[q]
            want_status = case.get("status", {}).get(q)
            if plain(r.answer) != want or (want_status and r.status != want_status):
                bad.append((case["name"], q, r.answer, r.status, want, want_status))
                print(f"    MISMATCH {q}: got {r.answer} [{r.status}], expected {want} [{want_status or 'any'}]")
    times.sort()
    print(f"\n{len(cases) - len({b[0] for b in bad})}/{len(cases)} cases match, replay ok on {replay_ok}/{len(cases)} traces, "
          f"median decision {times[len(times) // 2]:.2f} ms")
    print(tally)
    return bad


if __name__ == "__main__":
    task = load_task()

    # the routing table: learned again here with System.learn_rule to time it, then checked on 500 fresh synthetic emails
    fresh_task = load_task()                            # learn_rule installs the list as the rule, so use a separate copy
    t0 = time.perf_counter()
    table = System(fresh_task.cat, fresh_task.QUESTIONS).learn_rule("team", task.TRAIN, ["words"], min_support=3, min_precision=0.8)
    ms_learn = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    RuleList(["words"], min_support=3, min_precision=0.8).fit([{"words": task.words(s["subject"], s["body"])} for s, _ in task.TRAIN],
                                                              [t for _, t in task.TRAIN])
    ms_direct = (time.perf_counter() - t0) * 1000
    assert str(table) == str(task.ROUTING)
    print(f"routing table learned from {len(task.TRAIN)} labeled emails in {ms_learn:.0f} ms with System.learn_rule "
          f"({ms_direct:.0f} ms fitting RuleList on the one fact, as task.py does):\n{table}")
    system = System(task.cat, task.QUESTIONS)
    fresh = [(system.ask(s)["team"], t) for s, t in task.make_emails(500, seed=11)]
    ok = sum(r.answer == t for r, t in fresh)
    abstain = sum(r.status == "abstain" for r, _ in fresh)
    print(f"on 500 fresh synthetic emails: {ok} routed correctly, {abstain} abstained (sent to a person), "
          f"{len(fresh) - ok - abstain} routed wrong\n")

    cases = json.loads((HERE / "cases.json").read_text())
    bad = run_cases(task, cases)

    res = System(task.cat, task.QUESTIONS).ask(prepared(task, json.loads((HERE / "state.json").read_text())))
    print("\nthe strategist's flow for state.json:\n" + str(res.flow))
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[7]["state"]))
    print(f"\nearly exit on '{cases[7]['name']}': " + (", ".join(f"{n} ({w})" for n, w in res.trace.skipped) or "nothing skipped"))
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[5]["state"]))
    print(f"\nthe audit of '{cases[5]['name']}' (res.audit('team')):\n" + str(res.audit('team')))
    sys.exit(1 if bad else 0)
