"""Side by side on cases.json: solvi (task.py) vs Laya, an answer-only model (typed answer + probability, no citation, no rules).

Laya gets its own support-triage preset (laya.triage_questions: intent, is_urgent, refund_requested) plus two questions we add in
its format with our business rules written into the instructions (priority, route), and the customer tier / waiting time in
the state. A yes/no answer is P(true) >= 0.5.

Needs the laya package and its model (GPU optional). From the new_kelly repo (its env has torch):
  cd exps_v2 && uv run --with laya==0.3.20 --with transformers --with-editable ../solvi \\
      python ../solvi/gallery/01_support_triage/compare_laya.py"""
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

Q = laya.triage_questions()
Q = {"intent": Q["intent"], "urgent": Q["is_urgent"], "refund_requested": Q["refund_requested"]}
Q["priority"] = {"type": "choice", "instructions": "What priority should the ticket get?", "criteria": {
    "high": "a legal threat, a chargeback threat, a VIP customer who has waited 4 hours or more, or an urgent problem",
    "normal": "an ordinary request with no special risk",
    "low": "a general question with no time pressure"}}
Q["route"] = {"type": "choice", "instructions": "Which queue should handle the ticket?", "criteria": {
    "billing": "refunds, invoices, charges", "tech_support": "bugs, errors, outages", "retention": "cancellations and downgrades",
    "legal": "any threat of lawyers, courts or regulators", "general": "anything else"}}

agent = laya.load("convaiinnovations/laya", device="cuda" if torch.cuda.is_available() else "cpu")
system = System(mod.cat, mod.QUESTIONS)
agent.predict({"message": "warm up"}, Q)


def laya_answer(a):
    if a["type"] == "noul":
        return ("yes" if a["noul"] >= 0.5 else "no"), a["noul"] if a["noul"] >= 0.5 else 1 - a["noul"]
    return a["choice"], a["probabilities"][a["choice"]]


score = {"solvi": 0, "laya": 0, "n": 0}
lt, st = [], []
for c in cases:
    s = mod.prepare(copy.deepcopy(c["state"]))
    t0 = time.perf_counter()
    res = system.ask(s)
    st.append((time.perf_counter() - t0) * 1000)
    ls = {"message": c["state"]["message"]}
    if "tier" in c["state"]:
        ls.update(customer_tier=s["tier"], hours_waiting=round((s["now"] - s["received_at"]).total_seconds() / 3600, 1))
    t0 = time.perf_counter()
    la = agent.predict(ls, Q)["answers"]
    lt.append((time.perf_counter() - t0) * 1000)
    print(f"\n{c['name']}")
    for q, want in c["expected"].items():
        if q not in la:                  # tags: a solvi 0.4 multi-label answer with no Laya counterpart, not scored
            continue
        r = res[q]
        sv = "abstain" if r.status == "abstain" else r.answer
        lv, lp = laya_answer(la[q])
        want_s = "abstain" if want is None else want
        score["n"] += 1
        score["solvi"] += sv == want_s
        score["laya"] += lv == want_s
        print(f"  {q:17s} expected {want_s:15s} solvi {sv:15s}{'' if sv == want_s else ' ✗'}  laya {lv} ({lp:.2f})"
              f"{'' if lv == want_s else ' ✗'}")
lt.sort(), st.sort()
print(f"\ncorrect answers: solvi {score['solvi']}/{score['n']}, laya {score['laya']}/{score['n']}")
print(f"median time per ticket: solvi {st[len(st) // 2]:.2f} ms (CPU), laya {lt[len(lt) // 2]:.1f} ms "
      f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
