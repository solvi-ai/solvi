"""Side by side on cases.json: solvi (task.py) vs Laya, an answer-only model (typed answer + probability, no citation, no rules).

Laya gets its own email preset (laya.email_questions: category with billing / technical / sales / security / hr / other, and
is_phishing) plus a needs_human question we add in its format with our rules written into the instructions. Where solvi
abstains on team, Laya's "other" counts as the matching answer. A yes/no answer is P(true) >= 0.5.

Needs the laya package and its model (GPU optional). From the repository root:
  uv run --with laya==0.3.20 --with torch --with transformers python gallery/02_email_routing/compare_laya.py"""
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

Q = laya.email_questions()               # category: billing, technical, sales, security, hr, other
Q = {"team": Q["category"], "is_phishing": Q["is_phishing"]}
Q["needs_human"] = {"type": "noul", "instructions": "Does a person need to check this email before it is routed: the sender imitates "
                    "our domain acme.io, it asks to change bank details from outside the company, or the right team is unclear?"}
agent = laya.load("convaiinnovations/laya", device="cuda" if torch.cuda.is_available() else "cpu")
system = System(mod.cat, mod.QUESTIONS)
agent.predict(laya.email_state("hello", "warm up"), Q)


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
    ls = laya.email_state(c["state"]["subject"], c["state"]["body"], c["state"]["sender"])
    t0 = time.perf_counter()
    la = agent.predict(ls, Q)["answers"]
    lt.append((time.perf_counter() - t0) * 1000)
    print(f"\n{c['name']}   (laya is_phishing {la['is_phishing']['noul']:.2f})")
    for q, want in c["expected"].items():
        r = res[q]
        sv = "abstain" if r.status == "abstain" else r.answer
        lv, lp = laya_answer(la[q])
        want_s = "abstain" if want is None else want
        shown = lv
        lv = "abstain" if (q == "team" and lv == "other") else lv
        score["n"] += 1
        score["solvi"] += sv == want_s
        score["laya"] += lv == want_s
        print(f"  {q:17s} expected {want_s:15s} solvi {sv:15s}{'' if sv == want_s else ' ✗'}  laya {shown} ({lp:.2f})"
              f"{'' if lv == want_s else ' ✗'}")
lt.sort(), st.sort()
print(f"\ncorrect answers: solvi {score['solvi']}/{score['n']}, laya {score['laya']}/{score['n']}")
print(f"median time per email: solvi {st[len(st) // 2]:.2f} ms (CPU), laya {lt[len(lt) // 2]:.1f} ms "
      f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
