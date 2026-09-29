"""Side by side on cases.json: solvi (task.py) vs Laya, an answer-only model (typed answer + probability, no citation, no rules).

Laya has no preset for canary analysis, so both questions are written in its format with our policy spelled out in the
instructions; the state is the case's JSON as is (counts, histograms, SLO, thresholds).
A yes/no answer is P(true) >= 0.5.

Needs the laya package and its model (GPU optional). From the repository root:
  uv run --with laya==0.3.20 --with torch --with transformers python gallery/06_release_rollout/compare_laya.py"""
import copy
import json
import os
import time
import types
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
import laya  # noqa: E402
import torch  # noqa: E402

from solvi import System  # noqa: E402

HERE = Path(__file__).resolve().parent
mod = types.ModuleType("task")
exec(compile((HERE / "task.py").read_text(), "task.py", "exec"), mod.__dict__)
cases = json.loads((HERE / "cases.json").read_text())

Q = {"decision": {"type": "choice", "instructions": "A canary release is compared with the baseline (`baseline`, `canary`: request "
                  "and error counts, cumulative latency histograms in ms). Promote it, hold it, or roll it back?", "criteria": {
                      "roll_back": "the canary error rate is above max_error_pct, or errors rose more than 50% and significantly, or the "
                                   "canary p95 latency is above slo.p95_ms, or the error budget burns 10x too fast",
                      "hold": "fewer canary requests than min_canary_requests, or errors rose more than 20%, or p95 latency rose "
                              "more than 15%, or the error budget burns faster than allowed",
                      "promote": "none of the above"}},
     "page_oncall": {"type": "noul", "instructions": "Should the on-call engineer be paged: the canary error rate is above "
                     "max_error_pct, or it spends the error budget (100 - slo.availability_pct) at least 3 times faster than allowed?"}}
agent = laya.load("convaiinnovations/laya", device="cuda" if torch.cuda.is_available() else "cpu")
system = System(mod.cat, mod.QUESTIONS)
agent.predict(cases[0]["state"], Q)


def laya_answer(a):
    if a["type"] == "noul":
        return ("yes" if a["noul"] >= 0.5 else "no"), a["noul"] if a["noul"] >= 0.5 else 1 - a["noul"]
    return a["choice"], a["probabilities"][a["choice"]]


score = {"solvi": 0, "laya": 0, "n": 0}
lt, st = [], []
for c in cases:
    s = copy.deepcopy(c["state"])
    t0 = time.perf_counter()
    res = system.ask(s)
    st.append((time.perf_counter() - t0) * 1000)
    ls = c["state"]
    t0 = time.perf_counter()
    la = agent.predict(ls, Q)["answers"]
    lt.append((time.perf_counter() - t0) * 1000)
    print(f"\n{c['name']}")
    for q, want in c["expected"].items():
        r = res[q]
        sv = "abstain" if r.status == "abstain" else r.answer
        lv, lp = laya_answer(la[q])
        want_s = "abstain" if want is None else want
        shown = lv
        score["n"] += 1
        score["solvi"] += sv == want_s
        score["laya"] += lv == want_s
        print(f"  {q:17s} expected {want_s:15s} solvi {sv:15s}{'' if sv == want_s else ' ✗'}  laya {shown} ({lp:.2f})"
              f"{'' if lv == want_s else ' ✗'}")
lt.sort(), st.sort()
print(f"\ncorrect answers: solvi {score['solvi']}/{score['n']}, laya {score['laya']}/{score['n']}")
print(f"median time per decision: solvi {st[len(st) // 2]:.2f} ms (CPU), laya {lt[len(lt) // 2]:.1f} ms "
      f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
