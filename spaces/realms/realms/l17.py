"""The adaptive faction, variant 3 (L17): "the model proposes, deterministic code verifies and decides, everything in the trace".

A small learned policy (one MLP per question: build, order_military, stance; ~23 000 weights in all, realms/l17_net.json) picks
the answer among the options that the LAWS allow; an independent CHECK re-verifies the final answer on every decision. Pure
Python + numpy (runs in Pyodide as is; no torch, no onnxruntime). The net was trained outside the Space (exps_v2 L17:
approximate policy iteration with rollout labels — every allowed option of a sampled decision played out 3 × 30 turns on a copy
of the game, the net learns the argmax; two iterations, 278 000 labels, seeds 100-399). Numbers: see the README.

How a key decision is made (build per city, order per warrior/archer, war/peace per neighbour):
 1. solvi System.ask (the adaptive catalog): a failed hard check (treasury_ok, not_under_siege, keeps_capital_defender,
    war_needs_strength) forces the answer; the policy is never consulted then.
 2. LAWS filter the options, in this order:
      mask            affordable builds; attack needs a target, defend a threatened own city, disband surplus troops when broke
      upkeep_ok       no build whose upkeep would push next turn's treasury below 0
      war_min_length  a war cannot end before 8 turns
      capital_guard   the capital is threatened (enemy strength within 3 tiles >= its defence): the capital builds archer /
                      warrior / walls when it can, its defenders (within 1 tile) only fortify or defend, no new war
      no_starve       a city whose food surplus is negative builds a farm when it can (and the capital is not threatened)
    A law never empties the set (fallback: gold / fortify / peace).
 3. POLICY: the net scores the remaining options (inputs: the question's facts, the balanced rule's score per option, 27
    global features of the faction's position); the best one wins.
    Mode "l17_la" (optional, small budget): when the net is unsure (margin < 0.15) or on a 4% random audit, the top 3 are
    verified by a lookahead (2 rollouts × 15 turns on a copy of the game, realms/lookahead.value()), paid from a budget.
 4. CHECK: audit() re-verifies the final answer against the laws (separate code); violations are counted (0 expected).
 5. The decision's `why` names the options with the net's scores, the laws that removed options, the lookahead scores when
    it ran, and the pick.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import deque

import numpy as np
from solvi import System

from . import econ
from .adaptive import SPECS
from .brains import QUESTIONS, _rng_from, _rng_to, make_catalog
from .world import cheb

KINDS = ("l17", "l17_la")
OPTS = {q: list(SPECS[q]["options"]) for q in ("build", "order_military", "stance")}
CONTACT = 14
INIT_KEYS = {
    "build": ["pop", "food_store", "worked", "buildings", "gold", "wood", "stone", "income", "upkeep", "enemy_near",
              "garrison", "is_capital", "n_cities", "fleet", "unit_cap", "settle_value", "trade_partners", "at_war",
              "unimproved", "power", "weakest", "unit_room"],
    "order_military": ["unit", "strength", "targets", "threats", "at_capital", "capital_defenders", "fortified", "at_war",
                       "surplus_troops", "income", "upkeep"],
    "stance": ["my_strength", "their_strength", "border", "other_wars", "grievance", "trade_links", "at_war", "war_len",
               "my_cities", "their_cities"],
}
RULE_SCORES = {"build": "build_scores", "order_military": "military_scores", "stance": "stance_scores"}
CAP_THREAT = 1.0
NET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "l17_net.json")


# ---------------------------------------------------------------------------------------------------------------- features
def cap_threat(g, fid, cap=None):
    """(threat, defence) at the capital: enemy (at war) strength within 3 tiles / (own military strength within 1 tile
    × 2 with walls + 1)."""
    f = g.factions[fid]
    if cap is None:
        cap = g.cities.get(f["capital"]) if f["capital"] is not None else None
    if cap is None or cap["fid"] != fid:
        return 0.0, 0
    p = cap["pos"]
    enemy = dfn = 0
    for u in g.units.values():
        if u["type"] in econ.MILITARY:
            if u["fid"] == fid:
                if cheb(u["pos"], p) <= 1:
                    dfn += econ.STRENGTH[u["type"]]
            elif cheb(u["pos"], p) <= 3 and g.at_war(fid, u["fid"]):
                enemy += econ.STRENGTH[u["type"]]
    return enemy / (dfn * (2 if "walls" in cap["buildings"] else 1) + 1), dfn


def gfeat(g, fid):
    """27 global features of the faction's position; computed once per turn and faction (cached on the game object)."""
    c = getattr(g, "_l17_gf", None)
    if c is not None and c[0] == g.turn and c[1] == fid:
        return c[2]
    sc = {}
    for cc in g.cities.values():
        sc[cc["fid"]] = sc.get(cc["fid"], 0) + cc["pop"] + 3
    own = sc.get(fid, 0)
    others = [v for k, v in sc.items() if k != fid]
    mo = sum(others) / len(others) if others else 0.0
    mx = max(others) if others else 0.0
    st, fleet = {}, {k: 0 for k in econ.UNITS}
    for u in g.units.values():
        if u["type"] in econ.MILITARY:
            st[u["fid"]] = st.get(u["fid"], 0) + econ.STRENGTH[u["type"]]
        if u["fid"] == fid:
            fleet[u["type"]] += 1
    mine = [cc for cc in g.cities.values() if cc["fid"] == fid]
    f = g.factions[fid]
    alive = g.alive()
    contact = []
    for o in alive:
        if o == fid:
            continue
        theirs = [cc["pos"] for cc in g.cities.values() if cc["fid"] == o]
        if mine and theirs and min(cheb(a["pos"], b) for a in mine for b in theirs) <= CONTACT:
            contact.append(o)
    enemy_st = max([st.get(o, 0) for o in contact] or [0])
    strongest = max(contact, key=lambda o: (st.get(o, 0), -o)) if contact else None
    wars = [o for o in alive if o != fid and g.at_war(fid, o)]
    cap = g.cities.get(f["capital"]) if f["capital"] is not None else None
    threat, cdef = cap_threat(g, fid, cap)
    war_pers = [g.factions[o]["pers"] for o in wars]
    v = [math.log1p(own), math.log1p(mo), math.log1p(mx), math.log((own + 1) / (mo + 1)), len(mine) / 10,
         sum(cc["pop"] for cc in mine) / 50, math.log1p(st.get(fid, 0)), math.log1p(enemy_st), len(wars), min(threat, 5.0),
         cdef / 5, math.log1p(max(0, f["gold"])), math.log1p(max(0, f["wood"])), math.log1p(max(0, f["stone"])),
         f["income"] / 20, g.upkeep(fid) / 20, fleet["settler"], fleet["worker"] / 5,
         (fleet["warrior"] + fleet["archer"]) / 10, fleet["caravan"], math.log1p(max(0, g.turn - f["born"])), len(alive) / 5,
         float(any(g.factions[o]["pers"] == "warmonger" for o in contact)), float("warmonger" in war_pers),
         float(strongest is not None and strongest in wars), float(cap is not None and cap["fid"] == fid),
         (g.turn % 5) / 5]
    arr = np.asarray(v, dtype=np.float64)
    g._l17_gf = (g.turn, fid, arr, threat)
    return arr


def threat_now(g, fid):
    gfeat(g, fid)
    return g._l17_gf[3]


def _num(v):
    if v is None:
        return 0.0
    if isinstance(v, bool):
        return 1.0 if v else -1.0
    return float(v)


def dec_features(q, row, rule_scores, g, fid):
    feats = [_num(row.get(f)) for f in SPECS[q]["features"]]
    rs = [float(rule_scores.get(o, 0.0)) if rule_scores else 0.0 for o in OPTS[q]]
    return np.concatenate([np.asarray(feats + rs, dtype=np.float64), gfeat(g, fid)])


# ---------------------------------------------------------------------------------------------------------------- the net
class PolicyNet:
    """Per question: standardize → Linear-ReLU-Linear-ReLU-Linear → one score per option (numpy)."""

    def __init__(self, blob):
        self.meta = blob.get("meta", {})
        self.q = {}
        for q, d in blob["questions"].items():
            self.q[q] = {k: np.asarray(v, dtype=np.float64) for k, v in d.items() if k != "options"}
            self.q[q]["options"] = d["options"]

    @classmethod
    def load(cls, path=NET_PATH):
        with open(path) as fh:
            return cls(json.load(fh))

    def scores(self, q, x):
        p = self.q[q]
        h = np.clip((x - p["mu"]) / p["sd"], -6, 6)
        h = np.maximum(h @ p["W1"] + p["b1"], 0.0)
        h = np.maximum(h @ p["W2"] + p["b2"], 0.0)
        return h @ p["W3"] + p["b3"]


_NET = {}


def default_net():
    if "n" not in _NET:
        _NET["n"] = PolicyNet.load()
    return _NET["n"]


# ---------------------------------------------------------------------------------------------------------------- laws
def mask_allowed(q, row, aff):
    if q == "build":
        return [o for o in OPTS[q] if o in (aff or [])] or list(aff or ["gold"])
    if q == "order_military":
        m = row.get("military_allowed") or []
        return [o for o in OPTS[q] if o in m] or list(OPTS[q])
    return list(OPTS[q])


def laws(g, q, fid, ent, row, aff):
    """→ (candidates, removed {option: law})."""
    removed = {}
    allowed = mask_allowed(q, row, aff)
    for o in OPTS[q]:
        if o not in allowed:
            removed[o] = "mask"
    cands = list(allowed)

    def drop(keep, law):
        nonlocal cands
        for o in cands:
            if o not in keep:
                removed[o] = law
        cands = [o for o in cands if o in keep]

    threat = threat_now(g, fid)
    cap_id = g.factions[fid]["capital"]
    if q == "build":
        tr = row.get("treasury_after_upkeep")
        if tr is not None:
            ok = [o for o in cands if tr - econ.UPKEEP.get(o, 0) - econ.COST[o][3] >= 0]
            drop(ok or ["gold"], "upkeep_ok")
            if not cands:
                cands = ["gold"]
        guarded = False
        if ent == cap_id and threat >= CAP_THREAT:
            mil = [o for o in cands if o in ("archer", "warrior", "walls")]
            if mil:
                drop(mil, "capital_guard")
                guarded = True
        fs = row.get("food_surplus")
        if not guarded and fs is not None and fs < 0 and "farm" in cands:
            drop(["farm"], "no_starve")
    elif q == "order_military":
        u = g.units.get(ent)
        cap = g.cities.get(cap_id) if cap_id is not None else None
        if u is not None and cap is not None and cap["fid"] == fid and threat >= CAP_THREAT and cheb(u["pos"], cap["pos"]) <= 1:
            keep = [o for o in cands if o in ("fortify", "defend")]
            drop(keep or ["fortify"], "capital_guard")
            if not cands:
                cands = ["fortify"]
    else:
        if row.get("at_war") and row.get("war_len", 0) < 8 and "peace" in cands:
            drop(["war"], "war_min_length")
        if not row.get("at_war") and threat >= CAP_THREAT and "war" in cands:
            drop(["peace"], "capital_guard")
    if not cands:
        cands = [{"build": "gold", "order_military": "fortify", "stance": "peace"}[q]]
    return cands, removed


def audit(g, q, fid, ent, row, aff, pick, threat):
    """Independent re-check of the final answer (separate code from laws()): → violated laws ([] = clean)."""
    bad = []
    if q == "build":
        if aff and pick not in aff:
            bad.append("affordable")
        tr = row.get("treasury_after_upkeep")
        if tr is not None and pick != "gold" and tr - econ.UPKEEP.get(pick, 0) - econ.COST[pick][3] < 0:
            bad.append("upkeep_ok")
        if ent == g.factions[fid]["capital"] and threat >= CAP_THREAT and pick not in ("archer", "warrior", "walls"):
            if any(o in (aff or []) and (tr is None or tr - econ.UPKEEP.get(o, 0) >= 0) for o in ("archer", "warrior", "walls")):
                bad.append("capital_guard")
    elif q == "order_military":
        m = row.get("military_allowed") or []
        if m and pick not in m:
            bad.append("mask")
        cap_id = g.factions[fid]["capital"]
        cap = g.cities.get(cap_id) if cap_id is not None else None
        u = g.units.get(ent)
        if u is not None and cap is not None and cap["fid"] == fid and threat >= CAP_THREAT and \
                max(abs(u["pos"] % 28 - cap["pos"] % 28), abs(u["pos"] // 28 - cap["pos"] // 28)) <= 1 and \
                pick not in ("fortify", "defend"):
            bad.append("capital_guard")
        if row.get("at_capital") and (row.get("capital_defenders") or 0) <= 1 and pick not in ("fortify", "defend"):
            bad.append("keeps_capital_defender")
    else:
        if pick == "war" and (row.get("strength_ratio") or 0) < econ.WAR_MIN_RATIO:
            bad.append("war_needs_strength")
        if pick == "war" and not row.get("at_war") and threat >= CAP_THREAT:
            bad.append("capital_guard")
    return bad


# ---------------------------------------------------------------------------------------------------------------- planner
_RULE = {}


def rule_system():
    """The balanced personality's rules (a per-option input of the net)."""
    if "s" not in _RULE:
        s = System(make_catalog("adaptive"), QUESTIONS)
        s.learn = False
        _RULE["s"] = s
    return _RULE["s"]


class L17Planner:
    """The engine's hook (game.planner.decide / tick / stats / to_dict). mode "net" (browser default) or "net_la" (net +
    budgeted verified lookahead). record=False: the lean copy used inside the lookahead's rollouts."""

    LA_K, LA_R, LA_H = 3, 2, 15
    ENT_REF, LA_P_AUDIT, LA_MARGIN = 100, 0.04, 0.15

    def __init__(self, mode="net", net=None, seed=0, record=True, la_rate=30, la_cap=120):
        from .lookahead import FastAsk
        self.mode, self.net = mode, net or default_net()
        self.rng = random.Random(f"l17-{seed}")
        self.fast = FastAsk()
        self.record = record
        self.la_rate, self.la_cap = la_rate, la_cap
        self.bucket = float(la_cap)
        self.recent = deque(maxlen=60)       # last decisions: [turn, q, fid, ent, mode, why, violations] (not saved)
        self.child = None
        self.tot = {"decisions": 0, "violations": 0, "law_removed": {}, "lookaheads": 0}
        self._reset_window()

    def _reset_window(self):
        self.ms, self.n, self.viol, self.by_mode, self.laws, self.why_ok = [], 0, 0, {}, {}, 0

    @classmethod
    def create(cls, game, variant):
        return cls("net_la" if variant == "l17_la" else "net", seed=game.seed)

    def rollout_planner(self):
        if self.child is None:
            p = L17Planner("net", self.net, record=False)
            p.fast = self.fast
            self.child = p
        return self.child

    def rule_scores(self, q, row):
        state = {k: row[k] for k in INIT_KEYS[q] if k in row}
        try:
            return self.fast.ask(rule_system(), "rules", q, state).values.get(RULE_SCORES[q]) or {}
        except Exception:  # noqa: BLE001
            return {}

    def policy(self, g, q, fid, row, cands):
        rs = self.rule_scores(q, row)
        s = self.net.scores(q, dec_features(q, row, rs, g, fid))
        sc = {o: float(s[OPTS[q].index(o)]) for o in cands}
        return max(cands, key=lambda o: (sc[o], -OPTS[q].index(o))), sc

    def decide(self, g, q, fid, ent, res, aff):
        t0 = time.perf_counter()
        r = res[q]
        row = res.values
        cands, removed = laws(g, q, fid, ent, row, aff)
        pick, sc = self.policy(g, q, fid, row, cands)
        if not self.record:
            r.answer = pick
            return pick
        mode, la_scores = "net", None
        if self.mode == "net_la" and len(cands) > 1:
            pick, la_scores, mode = self._maybe_lookahead(g, q, fid, ent, res, cands, sc, pick)
        bad = audit(g, q, fid, ent, row, aff, pick, threat_now(g, fid))
        self.n += 1
        self.tot["decisions"] += 1
        if bad:
            self.viol += 1
            self.tot["violations"] += 1
        for o, law in removed.items():
            if law != "mask":
                self.laws[law] = self.laws.get(law, 0) + 1
                self.tot["law_removed"][law] = self.tot["law_removed"].get(law, 0) + 1
        self.by_mode[mode] = self.by_mode.get(mode, 0) + 1
        why = self.why(q, cands, sc, removed, la_scores, pick, bad)
        self.why_ok += int(why.startswith("L17") and "laws removed:" in why and f"→ {pick}" in why)
        r.answer, r.why = pick, why
        self.recent.append([g.turn, q, fid, ent, mode, why, bad])
        self.ms.append((time.perf_counter() - t0) * 1000)
        return pick

    def why(self, q, cands, sc, removed, la_scores, pick, bad):
        order = sorted(cands, key=lambda o: (-sc[o], OPTS[q].index(o)))
        s = "L17 policy net: " + ", ".join(f"{o} {sc[o]:+.2f}" for o in order[:4])
        laws_ = ", ".join(f"{o} ({w})" for o, w in removed.items() if w != "mask") or "none"
        masked = ", ".join(o for o, w in removed.items() if w == "mask")
        s += f"; laws removed: {laws_}" + (f"; not allowed: {masked}" if masked else "")
        if la_scores is not None:
            s += f"; verified lookahead ({self.LA_R}×{self.LA_H} turns): " + \
                 ", ".join(f"{o} {v:+.2f}" for o, v in la_scores.items())
        s += f"; check: {'VIOLATION ' + ', '.join(bad) if bad else 'ok'}"
        return s + f" → {pick}"

    # --- the verified lookahead (mode net_la)
    def tick(self, g):
        self.bucket = min(self.la_cap, self.bucket + self.la_rate)

    def _maybe_lookahead(self, g, q, fid, ent, res, cands, sc, pick):
        order = sorted(cands, key=lambda o: (-sc[o], OPTS[q].index(o)))
        margin = sc[order[0]] - sc[order[1]]
        audit_ = self.rng.random() < self.LA_P_AUDIT
        eligible = q != "order_military" or "attack" in cands or "defend" in cands
        if not eligible or not (audit_ or margin < self.LA_MARGIN):
            return pick, None, "net"
        top = order[:self.LA_K]
        cost = len(top) * self.LA_R * self.LA_H * (len(g.units) + len(g.cities)) / self.ENT_REF
        if self.bucket < min(cost, self.la_cap):
            return pick, None, "net_budget"
        self.bucket -= cost
        vals = rollout_values(g, q, fid, ent, res, top, self.LA_R, self.LA_H, self.rollout_planner(),
                              f"la-{g.seed}-{g.turn}-{self.tot['lookaheads']}")
        self.tot["lookaheads"] += 1
        best = max(top, key=lambda o: (vals[o], -top.index(o)))
        return best, vals, "net_la_audit" if audit_ else "net_la"

    # --- sim.py checkpoint fields and exact save / load
    def stats(self, game):
        xs = sorted(self.ms)
        p = (lambda k: round(xs[min(len(xs) - 1, int(k * len(xs)))], 4) if xs else 0.0)
        out = {"l17_n": self.n, "l17_viol": self.viol, "l17_ms_med": p(0.5), "l17_ms_p99": p(0.99),
               "l17_modes": dict(self.by_mode), "l17_laws": dict(self.laws), "l17_why_ok": self.why_ok,
               "l17_tot": json.loads(json.dumps(self.tot)), "l17_bucket": round(self.bucket, 1)}
        self._reset_window()
        return out

    def to_dict(self, game):
        return {"kind": "l17", "mode": self.mode, "rng": _rng_to(self.rng), "bucket": self.bucket, "tot": self.tot,
                "la_rate": self.la_rate}

    @classmethod
    def from_dict(cls, game, d):
        p = cls(d["mode"], seed=game.seed, la_rate=d.get("la_rate", 30))
        p.rng, p.bucket, p.tot = _rng_from(d["rng"]), d["bucket"], d["tot"]
        return p


# ---------------------------------------------------------------------------------------------------------------- rollouts
def rollout_values(g, q, fid, ent, res, cands, R, H, planner, tag):
    """Mean realms/lookahead.value() of R rollouts of H turns per candidate (common random numbers); the unit acts on its
    candidate order at once, as in the real game. The cyclic GC is paused meanwhile (results unaffected)."""
    import gc

    from .engine import Metrics
    from .lookahead import _same_state, sandbox, value
    was = gc.isenabled()
    gc.disable()
    try:
        met = Metrics(replay_rate=0)
        vs = {o: [] for o in cands}
        r = res[q]
        for k in range(R):
            live = {}
            for o in cands:
                s = sandbox(g, f"{tag}-{k}", planner.fast, met)
                s.planner = planner
                if q == "build":
                    s._start_build(s.factions[fid], s.cities[ent], o)
                elif q == "order_military":
                    u = s.units.get(ent)
                    if u is not None:
                        s._set_military_goal(fid, u, o, res.values)
                        s._act_military(fid, u)
                else:
                    s._apply_stance(fid, ent, o, r, res)
                i = s._order.index(fid) if fid in s._order else len(s._order)
                for f2 in s._order[i + 1:]:
                    if s.factions[f2]["alive"]:
                        s._faction_turn(f2)
                s._world_turn()
                live[o] = s
            alias = {}
            for t in range(H - 1):
                for o, s in live.items():
                    s.play_turn()
                if t < 2 and len(live) > 1:
                    keep = {}
                    for o, s in live.items():
                        twin = next((p for p, s2 in keep.items() if _same_state(s, s2)), None)
                        if twin is None:
                            keep[o] = s
                        else:
                            alias[o] = twin
                    live = keep
            val = {o: value(s, fid) for o, s in live.items()}
            for o in cands:
                a = alias.get(o, o)
                while a in alias:
                    a = alias[a]
                vs[o].append(val[a])
        return {o: round(sum(v) / len(v), 3) for o, v in vs.items()}
    finally:
        if was:
            gc.enable()
