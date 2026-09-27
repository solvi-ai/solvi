"""B5 check (realms/lookahead.py): every adaptive key decision decided by the planner has a `why` naming the proposer's
candidates with predicted values, the laws that removed candidates (or "none") and, when the lookahead ran, every evaluated
candidate's lookahead score; a failed hard check forces the answer and the planner is never consulted.

    uv run python spaces/realms/results/lookahead/b5_check.py SEED TURNS   → results/lookahead/b5_check_seed<SEED>.json
"""
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from realms.adaptive import LEARNED  # noqa: E402
from realms.engine import Game  # noqa: E402

seed, turns = int(sys.argv[1]), int(sys.argv[2])
g = Game(seed=seed, keep_last=False, variant="lookahead")
g.m.replay_rate = 0
stats = collections.Counter()
bad = []
HC = {"build": {"treasury_ok": "gold", "not_under_siege": "archer"}, "order_military": {"keeps_capital_defender": "fortify"},
      "stance": {"war_needs_strength": "peace"}}
orig_ask, orig_decide = g.ask, g.planner.decide
pending = {}


def ask(fid, q, state, key, title):
    res = orig_ask(fid, q, state, key, title)
    if g.factions[fid]["pers"] != "adaptive" or q not in LEARNED:
        return res
    r = res[q]
    fired = [rec.name for rec in res.trace.records if rec.name in HC[q] and rec.value is False]
    stats[(q, r.status)] += 1
    if fired:
        stats[(q, "hard_check_fired")] += 1
        if r.status != "forced" or r.answer != HC[q][fired[0]]:
            bad.append(("veto not applied", q, fired, r.answer, r.status))
        pending["forced"] = (q, r.answer)
    return res


def decide(game, q, fid, ent, res, aff):
    if res[q].status != "ok":
        bad.append(("planner consulted on a forced/abstained answer", q, res[q].status))
    n0 = g.planner.tot["lookaheads"]
    pick = orig_decide(game, q, fid, ent, res, aff)
    why = res[q].why
    ran = g.planner.tot["lookaheads"] > n0
    stats[(q, "decided")] += 1
    stats[(q, "lookahead")] += int(ran)
    m = re.match(r"proposer: ((?:\w+ [+-]\d+\.\d\d)(?:, \w+ [+-]\d+\.\d\d)*); laws removed: ([^;]+)", why)
    ok = m is not None
    if ok and ran:
        la = re.search(r"lookahead \([^)]*\): ((?:\w+ [+-]\d+\.\d\d)(?:, \w+ [+-]\d+\.\d\d)*) → (\w+)", why)
        ok = la is not None and la.group(2) == pick
        if ok:
            cands = [c.split()[0] for c in la.group(1).split(", ")]
            prop = [c.split()[0] for c in m.group(1).split(", ")]
            ok = pick in cands and all(c in prop for c in cands)
    elif ok:                                       # confident, budget spent, or a single candidate: names the pick
        ok = f"→ {pick}" in why or f"→ proposer's {pick}" in why
    stats[(q, "why_ok")] += int(ok)
    if not ok:
        bad.append(("why incomplete", q, why[:300]))
    if len(stats) and stats[(q, "decided")] <= 2:
        stats[f"example|{q}|{'lookahead' if ran else 'no lookahead'}"] = why[:400]
    return pick


g.ask = ask
g.planner.decide = decide
g.play(turns)
out = {"seed": seed, "turns": turns,
       "stats": {(f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k): v for k, v in sorted(stats.items(), key=str)},
       "violations": len(bad), "examples": bad[:10]}
print(json.dumps(out, indent=1, ensure_ascii=False))
with open(os.path.join(HERE, f"b5_check_seed{seed}.json"), "w") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
