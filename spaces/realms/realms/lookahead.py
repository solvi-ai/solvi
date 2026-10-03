"""The adaptive faction, variant 2: "the network proposes, the law verifies".

PRE-REGISTERED SUCCESS CRITERIA (written 2026-09-26 before any run of this variant; not to be edited afterwards)
-------------------------------------------------------------------------------------------------------------------
Evaluation: seeds 1, 2, 3, 4 × 20 000 turns each (sim.py --variant lookahead), same maps, the other four factions
unchanged (builder, expansionist, warmonger, trader keep their rules). "share" = the adaptive faction's score / mean score
of the other factions (Game._score_share, score = population + 3 × cities), one value per 500-turn checkpoint, averaged
over the first or the last 20% of the checkpoints. The baseline arm is the value-head variant (realms/adaptive.py) on the
same seeds and turns: results/endless_<seed>.json (its runs from the A-series evaluation; the engine refactor that added
the planner hook is checked to leave that variant's game bit-for-bit unchanged).

  B1  beats the rules        share over the last 20% >= 1.0 in at least 3 of the 4 seeds.
  B2  beats the value heads  share over the last 20% > the value-head variant's share over the last 20% on the same
                             seed, in at least 3 of the 4 seeds.
  B3  distillation works     (a) the proposer's agreement with the lookahead, measured on the random audit sample (an
                             eligible decision is audited with a fixed probability whatever the proposer's confidence),
                             is higher over the last 20% of turns than over the first 20%, pooled over the 4 seeds; and
                             (b) rollouts per eligible decision fall by >= 30% from the first 20% to the last 20% of
                             turns, pooled over the 4 seeds (and in at least 3 of the 4 seeds individually).
  B4  stays healthy          all 11 health invariants of realms/health.py hold on every evaluation run; in addition the
                             adaptive key-decision time including the lookahead (p99 per checkpoint) and the turn time
                             stay flat: mean over the last 10% of checkpoints <= 2 × the first 10%; saved state < 1 MB.
  B5  explainable            every adaptive key decision (build, order_military, stance) that the planner decides has a
                             `why` naming: the proposer's candidates with their predicted values, the laws/hard checks
                             that removed candidates (or "none"), and — when the lookahead ran — every surviving
                             candidate's lookahead score; a failed hard check still forces the answer (no lookahead can
                             override it). Audited on 2 000 turns of seed 1: 0 violations.
-------------------------------------------------------------------------------------------------------------------

How a key decision is made (build per city, order per warrior/archer, war/peace per neighbour):

 1. solvi System.ask (the adaptive catalog): hard checks first; a failed one forces the answer and nothing below runs.
 2. PROPOSER: a value head per question (same features and form as realms/adaptive.py: ridge per option, UCB), but its
    targets are the lookahead's scores (distillation), not the faction score 20 turns later. It ranks the options.
 3. LAWS filter the candidates: the question's mask (affordable builds; attack needs a target, defend a threatened city,
    disband surplus troops when broke), "upkeep_ok" (a build whose upkeep would push next turn's treasury below 0 is
    dropped) and "war_min_length" (a war cannot end before 8 turns, so "peace" in a young war is the same as "war").
 4. GATE: the lookahead runs when the proposer is unsure — the margin between its two best candidates is smaller than
    KAPPA × the uncertainty of that margin (ridge posterior width × the running residual of its past predictions) and
    that width is above WIDTH_MIN — or when the decision is drawn for the random audit (P_AUDIT). Rollouts are paid from
    a budget refilled every turn (RATE rollout-turns of a reference-size world, capped at CAP); a lookahead (audit or
    not) runs only when the budget covers it, so the time spent per turn is bounded.
 5. LOOKAHEAD (verified): for each of the top-K candidates, R rollouts of H turns on a copy of the state (every faction
    plays its normal policy, the adaptive one its proposer; fixed RNG seed per rollout, the same seeds for every
    candidate), scored by value() below. The best mean score wins.
 6. DISTILLATION: every evaluated candidate's score (relative to the mean of the candidates) is a training target for the
    proposer; with more data its uncertainty shrinks and the gate asks for fewer rollouts.

Military decisions are eligible for the lookahead only in contact with an enemy (attack or defend among the allowed
orders); in peace time the proposer decides alone.
"""
from __future__ import annotations

import gc
import random
import time
from collections import deque

import numpy as np
from solvi.core.plan.strategist import plan
from solvi.core.system import governs

from . import econ
from .adaptive import LAM, LEARNED, ValueHead
from .brains import _rng_from, _rng_to
from .engine import Game, Metrics
from .world import RAD2, cheb

K = 3                 # candidates evaluated by the lookahead
H = 10                # rollout horizon (turns)
R = 2                 # rollouts per candidate (common random numbers across candidates)
RATE = 30             # budget added every turn, in rollout-turns of a world with ENT_REF units + cities
CAP = 120             # budget cap (same units): at most ~2 lookaheads in a burst
ENT_REF = 100         # a rollout-turn costs (units + cities) / ENT_REF budget units (bigger worlds: slower rollouts)
P_AUDIT = 0.04        # random audit: an eligible decision gets the lookahead whatever the proposer's confidence
KAPPA = 1.0           # gate: margin < KAPPA × width
WIDTH_MIN = 0.08      # gate: a width below this is trusted even with a small margin (the options are about equal)
VSCALE = 3.0          # lookahead score units per target unit
PRIOR_W = 0.05        # the bootstrap prior (balanced rule) enters the proposer with this weight
REFRESH_P = 10        # turns between re-solving the proposer's weights
DECAY_P = 0.999       # per refresh
RES_EMA = 0.02        # residual variance: exponential moving average


# ======================================================================================================================
# Fast ask for rollouts: the same flow as solvi System.ask (plan cached per question and state keys), without hashing or
# the trace. Hard checks first in catalog order; the rule, or the head's argmax. Verified against System.ask (tests below).
# ======================================================================================================================
class _Res:
    __slots__ = ("answer", "status", "probs", "why", "confidence")

    def __init__(self, answer, status, probs=None, why=""):
        self.answer, self.status, self.probs, self.why, self.confidence = answer, status, probs or {}, why, 1.0


class _Resp:
    __slots__ = ("results", "values")

    def __init__(self, q, r, values):
        self.results, self.values = {q: r}, values

    def __getitem__(self, q):
        return self.results[q]


class FastAsk:
    def __init__(self):
        self.flows = {}

    def _flow(self, system, pers, question, keys):
        k = (pers, question, keys)
        fl = self.flows.get(k)
        if fl is not None:
            return fl
        cat, q = system.catalog, system.questions[question]
        flow = plan(cat, [q], keys, system.heads)
        facts = set(flow.per_question.get(question, []))
        prod = {st.part.name: st.part for st in flow.steps if st.part.kind != "rule"}
        hard = [n for n in cat.parts if n in facts and cat.parts[n].kind == "check" and cat.parts[n].hard
                and governs(cat.parts[n], q)]
        need, stack = set(), list(hard)
        while stack:
            n = stack.pop()
            if n in need or n not in prod:
                continue
            need.add(n)
            stack.extend(prod[n].inputs)
        consumed = {x for p in prod.values() for x in p.inputs}
        rule = cat.rules.get(question)
        if rule is not None:
            consumed |= set(rule.inputs)
        steps = [st.part for st in flow.steps if st.part.kind != "rule"]
        pre = [(p.name, p.func, tuple(p.inputs)) for p in steps if p.name in need]
        rest = [(p.name, p.func, tuple(p.inputs)) for p in steps if p.name not in need
                and not (p.kind == "check" and p.name not in consumed)]
        hard = [(n, cat.parts[n].then.get(question)) for n in hard]
        fl = (self._compile(pre), hard, self._compile(rest), (rule.func, tuple(rule.inputs)) if rule is not None else None, q,
              bool(flow.unresolved.get(question)))
        self.flows[k] = fl
        return fl

    @staticmethod
    def _compile(steps):
        """The steps as one generated function (keyword calls, a missing input skips the step, a failed step leaves its
        fact missing — the same semantics as solvi's executor, without its per-step bookkeeping)."""
        env, lines = {}, ["def run(vals):"]
        for k, (name, func, ins) in enumerate(steps):
            env[f"F{k}"] = func
            cond = " and ".join(f"{x!r} in vals" for x in ins) or "True"
            call = ", ".join(f"{x}=vals[{x!r}]" for x in ins)
            lines += [f"    if {cond}:", "        try:", f"            vals[{name!r}] = F{k}({call})",
                      "        except Exception:", "            pass"]
        if len(lines) == 1:
            lines.append("    pass")
        exec("\n".join(lines), env)  # noqa: S102 — code generated from the catalog's own part names
        return env["run"]

    def ask(self, system, pers, question, state):
        pre, hard, rest, rule, q, unresolved = self._flow(system, pers, question, frozenset(state))
        vals = dict(state)
        pre(vals)
        for name, then in hard:
            if vals.get(name, None) is False:
                if then is None:
                    return _Resp(question, _Res(None, "abstain"), vals)
                return _Resp(question, _Res(q.answer.normalize(then), "forced", why=f"hard check {name} is false"), vals)
        if unresolved:
            return _Resp(question, _Res(None, "abstain"), vals)
        rest(vals)
        if rule is not None:
            func, ins = rule
            try:
                v = func(**{x: vals[x] for x in ins})
                return _Resp(question, _Res(q.answer.normalize(v), "ok"), vals)
            except Exception:  # noqa: BLE001
                return _Resp(question, _Res(None, "abstain"), vals)
        head = system.heads.get(question)
        if head is None or any(f not in vals for f in head.features):
            return _Resp(question, _Res(None, "abstain"), vals)
        p = head.predict(vals)
        return _Resp(question, _Res(max(p, key=p.get), "ok", probs=p), vals)


# ======================================================================================================================
# The sandbox: a cheap copy of the game that plays on with the fast ask, no learning, no planner (no nested lookahead)
# ======================================================================================================================
class _FrozenLearner:
    frozen, eps, curve = True, 0.0, ()

    def snapshot(self, *a):
        pass

    def explore(self, options):
        return None

    def observe(self, *a, **k):
        pass

    def mature(self, *a, **k):
        pass


_FROZEN = _FrozenLearner()


class Sandbox(Game):
    def ask(self, fid, question, state, key, title):
        pers = self.factions[fid]["pers"]
        return self._fast.ask(self.systems[pers][1], pers, question, state)


def sandbox(g, rng_seed, fast, metrics):
    s = Sandbox.__new__(Sandbox)
    s.seed, s.turn, s.rng = g.seed, g.turn, random.Random(rng_seed)
    s.terrain = g.terrain
    s.dep, s.amt, s.improved, s.owner, s.worked_n = list(g.dep), list(g.amt), list(g.improved), list(g.owner), list(g.worked_n)
    s.cities = {k: {**c, "buildings": list(c["buildings"])} for k, c in g.cities.items()}
    s.units = {k: {**u, "goal": list(u["goal"]) if u["goal"] else u["goal"], "ban": list(u["ban"])} for k, u in g.units.items()}
    s.factions = {k: {**f, "grievance": dict(f["grievance"]), "trade": dict(f["trade"])} for k, f in g.factions.items()}
    s.wars, s.wants, s.truce = dict(g.wars), dict(g.wants), dict(g.truce)
    s.effects = [list(e) for e in g.effects]
    s.log, s.history = deque(maxlen=20), deque(maxlen=5)
    s.next_id, s.spawned, s.eliminated, s.counters = g.next_id, g.spawned, g.eliminated, dict(g.counters)
    s.human, s.human_orders = None, {}
    s.m, s.keep_last, s.last = metrics, False, {}
    s.dist, s.systems = g.dist, g.systems
    s._site_turn, s.site_parts, s.site_rank = g._site_turn, g.site_parts, g.site_rank
    s.learner, s.planner = _FROZEN, None
    s._order = list(getattr(g, "_order", g.alive()))
    s._enemy_tiles = set(getattr(g, "_enemy_tiles", ()))
    s._fast = fast
    return s


_STATE = ("turn", "dep", "amt", "improved", "owner", "worked_n", "cities", "units", "factions", "wars", "wants", "truce",
          "effects", "next_id", "spawned", "eliminated", "_site_turn", "site_rank")


def _same_state(a, b):
    """Every part of a sandbox that the engine or value() reads (logs, counters and history are write-only)."""
    return all(getattr(a, k) == getattr(b, k) for k in _STATE) and a.rng.getstate() == b.rng.getstate()


def value(s, fid):
    """The lookahead's objective for faction fid at the end of a rollout (higher is better):
    own score (population + 3 × cities, the share criterion's quantity) − 0.5 × the mean score of the other factions
    + 0.1 × territory (tiles owned) + 0.05 × material (production cost of own units and buildings, and work in progress)
    + 2.5 per own settler on the road (or its share of a settler in production: the city it will found lies beyond H)
    + 0.02 × stock (gold + wood + stone) − 2 × capital danger (enemy strength within 3 tiles of the capital per unit of
    its defence, above 1). A dead faction scores −20 − 0.5 × the others' mean."""
    sc, ids = {}, set()
    for c in s.cities.values():
        sc[c["fid"]] = sc.get(c["fid"], 0) + c["pop"] + 3
        if c["fid"] == fid:
            ids.add(c["id"])
    others = [v for k, v in sc.items() if k != fid]
    mo = sum(others) / len(others) if others else 0.0
    f = s.factions[fid]
    if not f["alive"] or not ids:
        return -20.0 - 0.5 * mo
    own = sc[fid]
    terr = sum(1 for o in s.owner if o in ids)
    mat = settlers = 0.0
    for u in s.units.values():
        if u["fid"] == fid:
            mat += econ.COST[u["type"]][0]
            settlers += u["type"] == "settler"
    for c in s.cities.values():
        if c["fid"] == fid:
            mat += sum(econ.COST[b][0] for b in c["buildings"]) + c["prog"]
            if c["build"] == "settler":
                settlers += min(1.0, c["prog"] / econ.COST["settler"][0])
    stock = f["gold"] + f["wood"] + f["stone"]
    danger = 0.0                                   # the capital: enemy strength within 3 tiles per unit of its defence
    cap = s.cities.get(f["capital"]) if f["capital"] is not None else None
    if cap is not None and cap["fid"] == fid:
        p = cap["pos"]
        enemy = dfn = 0
        for u in s.units.values():
            if u["type"] in econ.MILITARY:
                if u["fid"] == fid and cheb(u["pos"], p) <= 1:
                    dfn += econ.STRENGTH[u["type"]]
                elif u["fid"] != fid and cheb(u["pos"], p) <= 3 and s.at_war(fid, u["fid"]):
                    enemy += econ.STRENGTH[u["type"]]
        danger = max(0.0, enemy / (dfn * (2 if "walls" in cap["buildings"] else 1) + 1) - 1.0)
    return own - 0.5 * mo + 0.1 * terr + 0.05 * mat + 2.5 * settlers + 0.02 * stock - 2.0 * danger


# ======================================================================================================================
# The proposer and the planner
# ======================================================================================================================
class ProposerHead(ValueHead):
    """realms/adaptive.ValueHead trained on lookahead scores (distillation); slower forgetting, own refresh."""

    def refresh(self):
        self.solve()
        d = self.A.shape[1]
        self.A = DECAY_P * self.A + (1 - DECAY_P) * LAM * np.eye(d)
        self.b = DECAY_P * self.b

    def width(self, x, ks):
        return [float(max(x @ self.Ainv[k] @ x, 0.0)) for k in ks]


def _window():
    return {"decisions": 0, "lookaheads": 0, "rollouts": 0, "audits": 0, "audit_agree": 0, "agree": 0, "budget_skips": 0,
            "by_q": {}}


LAWS = {"build": "mask: affordable; upkeep_ok: next turn's treasury stays >= 0 with the option's upkeep",
        "order_military": "mask: attack needs a target, defend a threatened own city, disband surplus troops when broke",
        "stance": "war_min_length: a war cannot end before 8 turns (peace = war then)"}


class Planner:
    def __init__(self, game, init=True):
        self.fast = FastAsk()
        self.metrics = Metrics(replay_rate=0, seed=game.seed)
        self.recent = deque(maxlen=60)            # last decisions' why (for the app and the B5 audit; not saved)
        self.dedup = 0                            # rollouts skipped because their state equalled another candidate's
        if not init:
            return
        system = game.systems["adaptive"][1]
        for q in LEARNED:                          # start from the value heads' bootstrap (balanced rule), down-weighted
            h = ProposerHead.from_dict(system.heads[q].to_dict())
            d = h.A.shape[1]
            h.A = LAM * np.eye(d) + PRIOR_W * (h.A - LAM * np.eye(d))
            h.b = PRIOR_W * h.b
            h.n = 0
            h.solve()
            system.heads[q] = h
        self.rng = random.Random(f"planner-{game.seed}")
        self.bucket = float(CAP)
        self.res2 = {q: 1.0 for q in LEARNED}      # running residual variance of the proposer (target units)
        self.tot = {"decisions": 0, "lookaheads": 0, "rollouts": 0, "audits": 0, "audit_agree": 0, "agree": 0}

    # --- the decision
    def candidates(self, game, q, fid, ent, row, aff, head):
        allowed = head.allowed(row)
        removed = {}
        if q == "build":
            cands = [o for o in allowed if o in aff] or list(aff)
            tr = row.get("treasury_after_upkeep")
            if tr is not None:
                keep = [o for o in cands if tr - econ.UPKEEP.get(o, 0) - econ.COST[o][3] >= 0]
                for o in cands:
                    if o not in keep:
                        removed[o] = "upkeep_ok"
                cands = keep or ["gold"]
        elif q == "order_military":
            cands = list(allowed)
        else:
            cands = list(allowed)
            if row.get("at_war") and row.get("war_len", 0) < 8 and "peace" in cands:
                cands.remove("peace")
                removed["peace"] = "war_min_length"
        for o in head.options:
            if o not in cands and o not in removed:
                removed[o] = "mask"
        return cands, removed

    def decide(self, game, q, fid, ent, res, aff):
        t0 = time.perf_counter()
        r = res[q]
        head = game.systems["adaptive"][1].heads[q]
        row = res.values
        cands, removed = self.candidates(game, q, fid, ent, row, aff, head)
        x = head._x(row)
        sc = head.scores(row)
        pred = {o: float(sc[head.options.index(o)]) for o in cands}
        order = sorted(cands, key=lambda o: (-pred[o], head.options.index(o)))
        top = order[:K]
        eligible = len(cands) > 1 and (q != "order_military" or "attack" in cands or "defend" in cands)
        m = game.m
        if m.la is None:
            m.la = _window()
        la = m.la
        pick, scores, mode = order[0], None, "only"
        width = margin = 0.0
        if eligible:
            la["decisions"] += 1
            self.tot["decisions"] += 1
            k1, k2 = head.options.index(order[0]), head.options.index(order[1])
            u1, u2 = head.width(x, [k1, k2])
            width = float(np.sqrt(self.res2[q] * (u1 + u2)))
            margin = pred[order[0]] - pred[order[1]]
            audit = self.rng.random() < P_AUDIT
            unsure = width > WIDTH_MIN and margin < KAPPA * width
            cost = len(top) * R * H * (len(game.units) + len(game.cities)) / ENT_REF
            mode = "confident"
            if (audit or unsure) and self.bucket >= min(cost, CAP):    # a debt is paid back before the next one
                scores = self.lookahead(game, q, fid, ent, res, top)
                pick = max(top, key=lambda o: (scores[o], -top.index(o)))
                self.bucket -= cost
                self.distill(head, q, x, top, scores, pred)
                agree = pick == order[0]
                mode = "audit" if audit else "unsure"
                for d in (la, self.tot):
                    d["lookaheads"] += 1
                    d["rollouts"] += len(top) * R
                    d["agree"] += int(agree)
                    if audit:
                        d["audits"] += 1
                        d["audit_agree"] += int(agree)
                la["by_q"][q] = la["by_q"].get(q, 0) + 1
            elif unsure or audit:
                la["budget_skips"] += 1
                mode = "budget"
        why = self.why(q, mode, order, pred, removed, scores, pick, margin, width, r.why)
        r.answer, r.why = pick, why
        self.recent.append([game.turn, q, fid, ent, mode, why])
        m.la_ms.append((time.perf_counter() - t0) * 1000)
        return pick

    def lookahead(self, game, q, fid, ent, res, top):
        """Mean value() of R rollouts of H turns per candidate (the same R seeds for every candidate). The cyclic garbage
        collector is paused meanwhile: rollouts allocate many short-lived, acyclic containers freed by reference counting,
        and the collector's passes over them were pure overhead (results are unaffected)."""
        was = gc.isenabled()
        gc.disable()
        try:
            return self._lookahead(game, q, fid, ent, res, top)
        finally:
            if was:
                gc.enable()

    def _lookahead(self, game, q, fid, ent, res, top):
        """Candidates are rolled out in lockstep per seed. After the first two turns, a rollout whose whole game state
        equals an earlier candidate's (same seed) is dropped and takes that candidate's value: the game is deterministic
        given its state and RNG, so the value is exactly the same (e.g. a unit next to an enemy re-plans at once, which
        erases the difference between its candidate orders). This saves time without changing any result."""
        n = self.tot["lookaheads"]
        vs = {o: [] for o in top}
        for k in range(R):
            live = {}
            for o in top:
                s = sandbox(game, f"{game.seed}-{game.turn}-{n}-{k}", self.fast, self.metrics)
                if q == "build":
                    s._start_build(s.factions[fid], s.cities[ent], o)
                elif q == "order_military":
                    s._set_military_goal(fid, s.units[ent], o, res.values)
                else:
                    s._apply_stance(fid, ent, o, res[q], res)
                i = s._order.index(fid)
                for f2 in s._order[i + 1:]:              # the rest of this turn, then H - 1 full turns
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
            for o in top:
                r = alias.get(o, o)
                while r in alias:
                    r = alias[r]
                vs[o].append(val[r])
            self.dedup += len(alias)
        self.metrics.reset()
        return {o: round(sum(v) / len(v), 3) for o, v in vs.items()}

    def distill(self, head, q, x, top, scores, pred):
        mv = sum(scores.values()) / len(scores)
        mp = sum(pred[o] for o in top) / len(top)
        err = 0.0
        for o in top:
            y = max(-3.0, min(3.0, (scores[o] - mv) / VSCALE))
            err += (y - (pred[o] - mp)) ** 2
            head.observe(x, o, y)
        self.res2[q] = (1 - RES_EMA) * self.res2[q] + RES_EMA * err / len(top)

    @staticmethod
    def why(q, mode, order, pred, removed, scores, pick, margin, width, head_why):
        cand = ", ".join(f"{o} {pred[o]:+.2f}" for o in order[:K + 1])
        laws = ", ".join(f"{o} ({w})" for o, w in removed.items() if w != "mask") or "none"
        masked = ", ".join(o for o, w in removed.items() if w == "mask")
        s = f"proposer: {cand}; laws removed: {laws}" + (f"; not allowed: {masked}" if masked else "")
        if scores is not None:
            s += (f"; lookahead ({'audit' if mode == 'audit' else 'unsure: margin ' + f'{margin:.2f} < width {width:.2f}'}, "
                  f"{R}×{H} turns): " + ", ".join(f"{o} {v:+.2f}" for o, v in scores.items()) + f" → {pick}")
        elif mode == "confident":
            s += f"; proposer confident (margin {margin:.2f}, width {width:.2f}) → {pick}"
        elif mode == "budget":
            s += f"; unsure but rollout budget spent → proposer's {pick}"
        else:
            s += f"; single candidate → {pick}"
        return s + f" | head: {head_why}"

    def tick(self, game):
        self.bucket = min(CAP, self.bucket + RATE)
        if game.turn % REFRESH_P == 0:
            for q in LEARNED:
                game.systems["adaptive"][1].heads[q].refresh()

    def stats(self, game):
        """Checkpoint fields for sim.py (window counters of game.m.la, cumulative totals, decision time with lookahead)."""
        la = game.m.la or _window()
        xs = sorted(game.m.la_ms)
        p = (lambda q: round(xs[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else 0.0)
        return {"la": dict(la), "la_ms_med": p(0.5), "la_ms_p99": p(0.99), "la_n": len(xs), "la_tot": dict(self.tot),
                "bucket": round(self.bucket, 1), "res2": {q: round(v, 4) for q, v in self.res2.items()}}

    # --- exact save / load (the proposer heads are saved by the learner as system.heads; they are re-typed here)
    def to_dict(self, game):
        return {"rng": _rng_to(self.rng), "bucket": self.bucket, "res2": self.res2, "tot": self.tot}

    @classmethod
    def from_dict(cls, game, d):
        p = cls(game, init=False)
        p.rng, p.bucket, p.res2, p.tot = _rng_from(d["rng"]), d["bucket"], d["res2"], d["tot"]
        system = game.systems["adaptive"][1]
        for q in LEARNED:
            system.heads[q] = ProposerHead.from_dict(system.heads[q].to_dict())
        return p
