"""Side by side on cases.json: solvi (task.py) vs Laya, an answer-only model (typed answer + probability, no citation, no rules).

Laya has no preset for login alerts, so both questions are written in its format with our rules spelled out in the
instructions (thresholds, denylist); the state is the case's JSON as is (event, history, failed-login counts, is_admin).
A yes/no answer is P(true) >= 0.5.

Needs the laya package and its model (GPU optional). From the repository root:
  uv run --with laya==0.3.20 --with torch --with transformers python gallery/04_security_alert/compare_laya.py"""
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

Q = {"suspicious": {"type": "noul", "instructions": "Is the login in `event` suspicious? It is suspicious if the distance from the last "
                    "successful login in `history` could not be travelled in the time between them (faster than 900 km/h over more "
                    "than 500 km), or if the device is new and failed logins in the last hour are at least 150% above the 30-day "
                    "hourly average, or if the IP is on the denylist 198.51.100.23, 198.51.100.77, 203.0.113.200."},
     "action": {"type": "choice", "instructions": "What should happen to the session in `event`?", "criteria": {
         "lock_account": "impossible travel from a new device, impossible travel on an admin account, or a denylisted IP",
         "require_mfa": "impossible travel from a known device, or a new device with a failed-login spike or without passed MFA",
         "allow": "none of the above"}}}
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
    s = mod.prepare(copy.deepcopy(c["state"]))
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
print(f"median time per login: solvi {st[len(st) // 2]:.2f} ms (CPU), laya {lt[len(lt) // 2]:.1f} ms "
      f"({'GPU ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
