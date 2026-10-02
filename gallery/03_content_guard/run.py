"""Content guard: run every scenario in cases.json — answers, cited spans, the strategist's
flow, expected answers, trace replay and timing. Exits non-zero on a mismatch.

Run from the repo root:  uv run python gallery/03_content_guard/run.py"""
import copy
import json
import sys
import types
from pathlib import Path

from solvi import System

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # gallery/_audit.py: the audit check shared by the runners
from _audit import Tally, check, line as audit_line  # noqa: E402

HERE = Path(__file__).resolve().parent
SHOW_WHY = ["verdict"]
LABELLED = [({"surface": "support_chat", "text": "My order never arrived, can you check?"}, "allow"),
            ({"surface": "public_post", "text": "Great ride today, 80 km along the coast."}, "allow"),
            ({"surface": "public_post", "text": "Call me on 0176 555 01234 or mail ann@web.de"}, "review"),
            ({"surface": "support_chat", "text": "you idiot, shut up and fix it"}, "review"),
            ({"surface": "llm_prompt", "text": "Summarize: the meeting moved to Friday."}, "allow"),
            ({"surface": "support_chat", "text": "Where do I change my password?"}, "allow")]                                     # print the reason for these answers even when they are plain "ok"


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
        if res["verdict"].status == "forced":           # what the risk points alone would have said (the hard check won)
            pts = system.facts_for(state)["risk_points"]
            print(f"    the score alone would say: {task.verdict(pts)} (risk_points = {pts})")
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

    cases = json.loads((HERE / "cases.json").read_text())
    bad = run_cases(task, cases)

    res = System(task.cat, task.QUESTIONS).ask(prepared(task, json.loads((HERE / "state.json").read_text())))
    print("\nthe strategist's flow for state.json:\n" + str(res.flow))
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[0]["state"]))
    print(f"\nearly exit on '{cases[0]['name']}': " + (", ".join(f"{n} ({w})" for n, w in res.trace.skipped) or "nothing skipped"))
    res = System(task.cat, task.QUESTIONS).ask(prepared(task, cases[0]["state"]))
    print(f"\nthe audit of '{cases[0]['name']}' (res.audit('verdict')):\n" + str(res.audit('verdict')))

    # the constraints at work: replace the verdict rule by a head learned from 6 labelled texts (fit). Hard checks still
    # force "block"; where the head contradicts the harm answer, solvi picks the most probable verdict that satisfies every
    # constraint and says so. What no constraint covers stays the head's own answer (a constraint is not the whole policy).
    weak = load_task()
    del weak.cat.rules["verdict"]
    for label, feats, must_repair in (("a weak head that sees only the surface", ["surface"], True),
                                      ("a head that also reads the risk score", ["risk_score", "surface"], False)):
        learned = System(weak.cat, weak.QUESTIONS)
        learned.fit("verdict", LABELLED, features=feats)
        print(f"\na learned verdict (fit on 6 labelled texts) instead of the rule — {label}:")
        repaired, demo_tally, agree = None, Tally(), 0
        for case in cases:
            r = learned.ask(prepared(weak, case["state"]))
            demo_tally.add(check(r, learned))
            v = r["verdict"]
            agree += v.answer == case["expected"]["verdict"]
            mark = "  <- repaired by a constraint" if v.repaired else ""
            print(f"    {case['name']:36s} verdict={fmt(v)!s:15s} (rule: {case['expected']['verdict']})  harm={list(r['harm'].answer)}{mark}")
            if v.repaired and repaired is None:
                repaired = r
        print(demo_tally, f"· agrees with the rule on {agree}/{len(cases)}")
        if must_repair and repaired is None:
            bad.append(("learned verdict", "verdict", None, None, "a constraint repair", None))
        elif repaired is not None:
            print("\n" + str(repaired.audit("verdict")))
    sys.exit(1 if bad else 0)
