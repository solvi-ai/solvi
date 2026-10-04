"""Protection vs justified risk, and a learned gate with and without its bound — on a toy dungeon with the protocol of
the research program's NetHack runs (not NetHack, and not its numbers).

    uv run python benchmarks/knowledge/risk_dungeon.py                 # 20 streams × 30 games, a few seconds
    → benchmarks/knowledge/results/risk_dungeon.json

The game (seeded; no model, no network). A hero goes down a dungeon of 12 levels, at most 120 events per game. Each
event on a level is one of: a monster (fight it or not; a monster that blocks the way costs 15 turns of food if
avoided), a corpse (eat it or not), the stairs (go down or stay and explore: 5 turns, maybe a monster). Fights give
experience (a level per 3 kills). Food runs out after 400 turns without eating: death by hunger. Hidden rules: going down
to depth d at experience level x dies with probability 0.04 × max(0, d − x − 1); a floating eye kills in melee with
probability 0.35, a gas spore 0.25, other monsters 0.02; a kobold corpse sickens (a death with probability 0.15), a
cockatrice corpse kills at once (a written rule: hard), other corpses are food. Progression = the deepest level reached.

Knowledge (every arm but "none"): a written spec (cockatrice corpses: hard; floating-eye melee: a rate 0.35) and a
memory of outcomes per (action, kind) — refused when a bad outcome was seen, with its rate and support — plus a learned
depth gate (solvi.core.knowledge.LearnedGate: the median depth of recent deaths at the hero's level − 1). Arms:
  none            no knowledge: fights, eats and descends whatever comes
  protect         Protect(): every refusal is final; the gate bounded (expiry 10 games, floor = level + 1)
  risk            RiskBudget(max_risk_per_episode=1.0, min_gain_ratio=1.0, min_support=3); the same bounded gate
  protect_unbounded  Protect() with a gate that never expires and has no floor (how a learned gate tightens itself)
Reported per arm: mean progression over all games and over the last third of each stream, deaths by cause, risky
takes per game, the learned gate's limit at level 3 at the end of each stream."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solvi.core.knowledge import LearnedGate, Prediction, Protect, RiskBudget  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"
MONSTERS = {"jackal": 0.02, "newt": 0.02, "orc": 0.02, "floating eye": 0.35, "gas spore": 0.25}
CORPSES = {"jackal": 0.0, "newt": 0.0, "kobold": 0.15, "cockatrice": 1.0}
HARD_EAT = {"cockatrice"}
GAIN = {"fight_blocking": 0.5, "fight": 0.05, "eat_hungry": 0.4, "eat": 0.02, "descend": 0.3}


class Memory:
    """Outcomes per (action, kind): seen bad → refuse with the rate (Laplace) and support."""

    def __init__(self):
        self.n, self.bad = Counter(), Counter()

    def predict(self, action, kind, spec=None):
        k = (action, kind)
        preds = []
        if spec is not None:
            preds.append(spec)
        if self.bad[k]:
            rate = (self.bad[k] + 1) / (self.n[k] + 2)
            preds.append(Prediction("refuse", rate, self.n[k], f"memory: {action} {kind} went wrong {self.bad[k]} / "
                                    f"{self.n[k]}"))
        refusals = [p for p in preds if p.verdict == "refuse"]
        if not refusals:
            return Prediction("accept", 0.0, self.n[k], "no bad outcome known")
        worst = max(refusals, key=lambda p: (p.hard, p.risk))
        return Prediction("refuse", worst.risk, min(p.support for p in refusals), "; ".join(p.reason for p in refusals),
                          hard=any(p.hard for p in refusals))

    def record(self, action, kind, bad):
        self.n[(action, kind)] += 1
        self.bad[(action, kind)] += bool(bad)


def spec_for(action, kind):
    if action == "eat" and kind in HARD_EAT:
        return Prediction("refuse", 1.0, -1, "spec: never eat a cockatrice", hard=True)
    if action == "fight" and kind == "floating eye":
        return Prediction("refuse", 0.35, -1, "spec: a floating eye paralyses")
    return None


def play(arm, rng, memory, gate, policy):
    depth = xl = 1
    kills, food, deepest, takes = 0, 400, 1, 0
    if policy is not None:
        policy.new_episode()
    arrivals = {}
    for _ in range(120):
        food -= 5
        if food <= 0:
            return deepest, "hunger", takes, arrivals
        u = rng.random()
        if u < 0.45:                                            # a monster
            kind = rng.choice(sorted(MONSTERS))
            blocking = rng.random() < 0.4
            take = True
            if arm != "none":
                p = memory.predict("fight", kind, spec_for("fight", kind))
                if p.verdict == "refuse":
                    d = policy.decide("fight", p, GAIN["fight_blocking" if blocking else "fight"], key=("fight", kind))
                    take = d.choice == "take"
                    takes += take
            if not take:
                food -= 15 if blocking else 0
                continue
            bad = rng.random() < MONSTERS[kind] * (1.5 if depth > xl + 2 else 1.0)
            memory.record("fight", kind, bad)
            if bad:
                return deepest, f"melee {kind}", takes, arrivals
            kills += 1
            xl = 1 + kills // 3
        elif u < 0.65:                                          # a corpse
            kind = rng.choice(sorted(CORPSES))
            hungry = food < 150
            take = True
            if arm != "none":
                p = memory.predict("eat", kind, spec_for("eat", kind))
                if p.verdict == "refuse":
                    d = policy.decide("eat", p, GAIN["eat_hungry" if hungry else "eat"], key=("eat", kind))
                    take = d.choice == "take"
                    takes += take
            if not take:
                continue
            bad = rng.random() < CORPSES[kind]
            memory.record("eat", kind, bad)
            if bad:
                return deepest, f"corpse {kind}", takes, arrivals
            food = 400
        else:                                                   # the stairs
            nxt = depth + 1
            if nxt > 12:
                continue
            take = True
            if arm != "none":
                p = gate.predict(xl, nxt)
                if p.verdict == "refuse":
                    d = policy.decide("descend", p, GAIN["descend"], key=("descend", nxt))
                    take = d.choice == "take"
                    takes += take and d.reason != "already charged this episode"
            if not take:
                continue
            depth = nxt
            deepest = max(deepest, depth)
            arrivals.setdefault(depth, xl)
            if rng.random() < 0.04 * max(0, depth - xl - 1):
                return deepest, "descent", takes, arrivals
    return deepest, "alive", takes, arrivals


def stream(arm, s, games):
    rng = random.Random(7919 * s + 17)
    memory = Memory()
    if arm == "protect_unbounded":
        gate = LearnedGate("depth", expiry=10 ** 6, floor=0, min_support=3, margin=1)
    else:
        gate = LearnedGate("depth", expiry=10, floor=lambda x: x + 1, min_support=3, margin=1)
    policy = RiskBudget(1.0, 1.0, 3) if arm == "risk" else Protect()
    rows = []
    for _ in range(games):
        deepest, cause, takes, arrivals = play(arm, rng, memory, gate, policy)
        if cause not in ("alive", "hunger"):
            gate.failed(max(arrivals.values(), default=1) if arrivals else 1, deepest)
        for d, x in arrivals.items():
            gate.arrived(x, d, failed=(cause == "descent" and d == deepest))
        gate.end_episode()
        rows.append({"deepest": deepest, "cause": cause, "takes": takes})
    return rows, gate.limit(3)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=20)
    ap.add_argument("--games", type=int, default=30)
    a = ap.parse_args(argv)
    t0 = time.time()
    res = {}
    for arm in ("none", "protect", "risk", "protect_unbounded"):
        per = [stream(arm, s, a.games) for s in range(a.streams)]
        rows = [r for rs, _ in per for r in rs]
        last = [r for rs, _ in per for r in rs[a.games - a.games // 3:]]
        first = [r for rs, _ in per for r in rs[:a.games // 3]]
        causes = Counter(r["cause"] for r in rows)
        res[arm] = {"progression_mean": sum(r["deepest"] for r in rows) / len(rows),
                    "progression_first_third": sum(r["deepest"] for r in first) / len(first),
                    "progression_last_third": sum(r["deepest"] for r in last) / len(last),
                    "deaths": dict(sorted(causes.items())),
                    "targeted_deaths": sum(v for k, v in causes.items() if k.startswith(("melee floating", "melee gas",
                                                                                          "corpse"))),
                    "risky_takes_per_game": sum(r["takes"] for r in rows) / len(rows),
                    "gate_limit_at_level_3_end": [lim for _, lim in per]}
        print(f"{arm}: progression {res[arm]['progression_mean']:.2f} (first third "
              f"{res[arm]['progression_first_third']:.2f}, last {res[arm]['progression_last_third']:.2f}), targeted deaths "
              f"{res[arm]['targeted_deaths']}, hunger {causes['hunger']}, takes/game {res[arm]['risky_takes_per_game']:.2f}",
              flush=True)
    res["config"] = {"streams": a.streams, "games": a.games, "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(exist_ok=True)
    (OUT / "risk_dungeon.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
