"""Print a response: answers, flow, computed_state, replay — so you can see how the system reached each answer."""
from __future__ import annotations


def show(res, catalog=None, flow=True, state=True):
    print("── answers " + "─" * 60)
    for q, r in res.results.items():
        a = "—" if r.answer is None else r.answer
        extra = "" if r.status == "ok" else f"  [{r.status}]"
        print(f"  {q:16s} {a!s:12s} confidence {r.confidence:.2f}{extra}")
        print(f"  {'':16s} why: {r.why}")
    if flow:
        print("── flow (chosen by the strategist) " + "─" * 35)
        for line in str(res.flow).splitlines():
            print("  " + line)
    if flow and getattr(res.trace, "skipped", None):
        print("  skipped at run time: " + ", ".join(f"{n} ({why})" for n, why in res.trace.skipped))
    if state:
        print("── computed_state " + "─" * 52)
        for line in res.computed_state.splitlines():
            print("  " + line)
    if catalog is not None:
        rep = res.trace.replay(catalog)
        print("── trace replay " + "─" * 54)
        print(f"  steps {rep['steps']}, mismatches {len(rep['mismatches'])}" + ("" if rep["ok"] else f": {rep['mismatches'][:3]}"))
    print(f"── time {res.ms:.2f} ms")
