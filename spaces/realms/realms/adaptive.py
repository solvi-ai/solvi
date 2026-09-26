"""The adaptive faction's learner.

PRE-REGISTERED SUCCESS CRITERIA (written 2026-09-26 before any change to the learner; not to be edited afterwards)
-------------------------------------------------------------------------------------------------------------------
Evaluation: seeds 1, 2, 3, 4 × 20 000 turns each (sim.py), same map/seed as before, the other factions unchanged
(builder, expansionist, warmonger, trader keep their rules and weights). "share" = the adaptive faction's score / mean
score of the other factions (Game._score_share: score = population + 3 × cities), one value per 500-turn checkpoint.

  A1  beats the rules      over seeds 1–4 × 20 000 turns, the adaptive faction's share averaged over the last 20% of
                           turns is >= 1.0 in at least 3 of the 4 seeds.
  A2  improves with time   share averaged over the last 20% of turns > share averaged over the first 20% of turns,
                           in all 4 seeds.
  A3  stays healthy        all 11 health invariants of realms/health.py still hold on every evaluation run (I1 no
                           exceptions, I2/I3 decision and turn time flat within 2×, I4 state bounded, I5 memory
                           bounded, I6 >= 2 factions alive, I7 economy not degenerate, I8 treasury never negative,
                           I9 trace replay OK, I10 no abstentions, I11 save/load exact).
  A4  stays explainable    every adaptive decision still comes out of solvi System.ask with a `why` that names the
                           learned contributions, and the hard checks still veto it (a failed hard check forces the
                           answer whatever the learner prefers).

Plus a check that it is learning rather than hard-coded: an ablation with learning frozen after the bootstrap should do
worse (reported, not a pass/fail criterion).
-------------------------------------------------------------------------------------------------------------------
"""
from __future__ import annotations

import math
import random
import time
from collections import deque

import numpy as np
from solvi import System
from solvi.fast import FastHead, VecFeaturizer

from . import econ
from .brains import (DERIVED, MIL_OPTS, STANCE_OPTS, _arr, _rng_from, _rng_to, _unarr, make_catalog, random_city_state)

# ======================================================================================================================
# What is learned
# ======================================================================================================================
# One value head per question: for every answer option a ridge regression of the reward on the question's facts.
# reward = log(score + 5) HORIZON turns after the decision − log(score + 5) at the decision (score = population + 3 × cities,
# the quantity the share criterion is measured on), for the faction that made the decision.
SPECS = {
    "build": dict(options=econ.BUILD_OPTIONS, mask="affordable", horizon=20, keep_others=0.35,
                  features=["pop", "food_surplus", "growth_need", "infra_best", "military_need", "expand_value",
                            "trade_value", "gold_need", "threat_ratio", "net_income", "n_cities", "garrison", "at_war",
                            "is_capital", "power", "weakest", "unit_room"]),
    "order_military": dict(options=MIL_OPTS, mask="military_allowed", horizon=20, keep_others=0.5,
                           features=["attack_odds", "target_is_city", "target_dist", "defend_need", "at_capital",
                                     "capital_defenders", "fortified", "at_war", "surplus_troops", "net_income",
                                     "strength"]),
    "stance": dict(options=STANCE_OPTS, mask=None, horizon=25, keep_others=1.0,
                   features=["strength_ratio", "tension", "at_war", "war_len", "other_wars", "my_cities", "their_cities",
                             "border", "grievance", "trade_links"]),
}
LEARNED = tuple(SPECS)
REFRESH = 25          # turns between re-solving the ridge weights from the running sufficient statistics
DECAY = 0.995         # per refresh: old evidence fades (effective memory ~5 000 turns), the ridge prior is restored
LAM = 1.0             # ridge strength
ALPHA = 0.005         # optimism: value + ALPHA × sqrt(xᵀ A⁻¹ x) (upper confidence bound)
EPS = 0.01            # small ε exploration (build only)
PRIOR_N = 200         # bootstrap: synthetic states per question labelled by the balanced rule
PRIOR_R = 0.05        # bootstrap: the rule's choice is worth PRIOR_R, other allowed options 0
TEACH_R = 0.2         # System.teach(question, state, answer) = "this answer was worth TEACH_R here"
PENDING_MAX = {"build": 700, "order_military": 500, "stance": 250}


def score_of(game):
    sc = {}
    for c in game.cities.values():
        sc[c["fid"]] = sc.get(c["fid"], 0) + c["pop"] + 3
    return sc


class ValueHead(FastHead):
    """A solvi FastHead whose per-option scores are learned action values (disjoint linear UCB).

    It plugs into System.ask like any fitted head: the strategist plans the facts in `features` (and the mask fact), hard
    checks run first and override it, `predict` gives the probabilities, `contributions` the `why`. Per option it keeps the
    ridge statistics A = λI + Σ xxᵀ and b = Σ r·x; weights and A⁻¹ are re-solved every REFRESH turns (fixed in between, so a
    decision and its trace replay see the same head)."""

    def __init__(self, options, features, mask=None):
        super().__init__(options, lam=LAM, pairs=False)
        self.feats, self.mask = list(features), mask
        self.features = self.feats + ([mask] if mask else [])
        self.frozen = False
        self.teaches = 0

    # --- featurization: one standardized value per number, ±1 per boolean (solvi VecFeaturizer.compact), plus a bias
    def setup(self, rows):
        self.fz = VecFeaturizer().fit(rows, self.feats)
        d = len(self.fz.compact(rows[0], self.feats)) + 1
        k = len(self.options)
        self.A = np.array([np.eye(d) * LAM for _ in range(k)])
        self.b = np.zeros((k, d))
        self.n = 0
        self.solve()
        return self

    def _x(self, row):
        return np.append(np.clip(self.fz.compact(row, self.feats), -5, 5), 1.0)

    def vec(self, row):
        return [round(float(v), 4) for v in self._x(row)]

    def allowed(self, row):
        if not self.mask:
            return list(self.options)
        m = row.get(self.mask) or []
        return [o for o in self.options if o in m] or list(self.options)

    # --- learning
    def observe(self, x, a, r, w=1.0):
        x = np.asarray(x, float)
        k = self.options.index(a)
        self.A[k] += w * np.outer(x, x)
        self.b[k] += w * r * x
        self.n += 1

    def solve(self):
        self.Ainv = np.linalg.inv(self.A)
        self.W = np.einsum("kij,kj->ki", self.Ainv, self.b)

    def refresh(self):
        self.solve()
        d = self.A.shape[1]
        self.A = DECAY * self.A + (1 - DECAY) * LAM * np.eye(d)
        self.b = DECAY * self.b

    def update(self, row, answer):
        """System.teach: a human says `answer` is right here → observed with reward TEACH_R, used at the next refresh."""
        t0 = time.perf_counter()
        self.observe(self._x(row), answer, TEACH_R)
        self.teaches += 1
        return (time.perf_counter() - t0) * 1000

    # --- answering
    def values(self, row):
        x = self._x(row)
        q = self.W @ x
        bonus = ALPHA * np.sqrt(np.maximum(np.einsum("i,kij,j->k", x, self.Ainv, x), 0.0))
        return x, q, bonus

    def scores(self, row):
        _, q, bonus = self.values(row)
        ok = set(self.allowed(row))
        return np.array([q[k] + bonus[k] if o in ok else -1e9 for k, o in enumerate(self.options)])

    def predict(self, row):
        s = self.scores(row)
        m = s.max()
        e = np.where(s > -1e8, np.exp((s - m) / 0.02), 0.0)
        p = e / e.sum()
        return {o: float(v) for o, v in zip(self.options, p)}

    def contributions(self, row):
        """Per fact: how much it pushed the chosen option's learned value above the mean of the other allowed options."""
        x, q, bonus = self.values(row)
        s = self.scores(row)
        a = int(np.argmax(s))
        ok = [k for k, o in enumerate(self.options) if o in set(self.allowed(row)) and k != a]
        base = self.W[ok].mean(0) if ok else np.zeros_like(self.W[a])
        diff = x * (self.W[a] - base)
        out, j = {}, 0
        for f in self.feats:
            n = len(self.fz.compact({f: row.get(f)}, [f])) if f in self.fz.spec else 0
            out[f] = float(diff[j:j + n].sum())
            j += n
        return out

    # --- exact save / load
    def to_dict(self):
        spec = {f: [s[0]] + [x.tolist() if isinstance(x, np.ndarray) else x for x in s[1:]] for f, s in self.fz.spec.items()}
        return {"options": self.options, "feats": self.feats, "mask": self.mask, "n": self.n, "teaches": self.teaches,
                "frozen": self.frozen, "spec": spec, "A": _arr(self.A), "b": _arr(self.b), "W": _arr(self.W),
                "Ainv": _arr(self.Ainv)}

    @classmethod
    def from_dict(cls, d):
        h = cls(d["options"], d["feats"], d["mask"])
        h.n, h.teaches, h.frozen = d["n"], d["teaches"], d["frozen"]
        h.fz = VecFeaturizer()
        h.fz.spec = {f: tuple(s) for f, s in d["spec"].items()}
        h.A, h.b, h.W, h.Ainv = _unarr(d["A"]), _unarr(d["b"]), _unarr(d["W"]), _unarr(d["Ainv"])
        return h


# ======================================================================================================================
# Bootstrap: synthetic decisions labelled by the balanced rule (the prior every head starts from)
# ======================================================================================================================
def _build_state(rng):
    st = random_city_state(rng)
    st.update(power=round(rng.uniform(0.2, 3.0), 2), weakest=round(rng.uniform(0.2, 5.0), 2),
              unit_room=rng.randint(0, 8))
    return st


def _military_state(rng):
    targets = [[rng.randrange(500), rng.randint(1, 10), round(rng.uniform(0.5, 8), 2), rng.random() < 0.4]
               for _ in range(rng.choice([0, 0, 1, 2, 3]))]
    threats = [[rng.randrange(500), rng.randint(0, 8), round(rng.uniform(0.2, 4), 2)] for _ in range(rng.choice([0, 0, 1, 2]))]
    at_cap = rng.random() < 0.3
    return {"unit": rng.choice(["warrior", "archer"]), "strength": rng.choice([2, 3]), "targets": targets,
            "threats": threats, "at_capital": at_cap, "capital_defenders": rng.randint(1, 4) if at_cap else rng.randint(0, 4),
            "fortified": rng.random() < 0.5, "at_war": bool(targets) or rng.random() < 0.2,
            "surplus_troops": rng.choice([0, 0, 1, 3, 6]), "income": rng.randint(2, 30), "upkeep": rng.randint(0, 30)}


def _stance_state(rng):
    war = rng.random() < 0.3
    return {"my_strength": rng.randint(1, 60), "their_strength": rng.randint(1, 60), "border": rng.randint(2, 14),
            "other_wars": rng.choice([0, 0, 1, 2]), "grievance": round(rng.choice([0, 0, 0.5, 1.5, 3.0]), 2),
            "trade_links": rng.choice([0, 0, 1, 3]), "at_war": war, "war_len": rng.randint(0, 40) if war else 0,
            "my_cities": rng.randint(1, 12), "their_cities": rng.randint(1, 12)}


GENERATORS = {"build": _build_state, "order_military": _military_state, "stance": _stance_state}


def row_of(question, values):
    """The head's input row from a decision's computed facts (any faction's): derived facts filled in the same way the
    adaptive catalog computes them."""
    row = values
    for f in SPECS[question]["features"] + ([SPECS[question]["mask"]] if SPECS[question]["mask"] else []):
        if f not in row and f in DERIVED:
            fn, ins = DERIVED[f]
            if all(x in row for x in ins):
                if row is values:
                    row = dict(values)
                row[f] = fn(*[row[x] for x in ins])
    return row


# ======================================================================================================================
# The learner
# ======================================================================================================================
class Learner:
    """Contextual-bandit learner of the adaptive faction (build, order_military, stance).

    Data: every non-forced decision of EVERY faction (the adaptive one's own, and a sample of the others' — observational
    data: it sees what the warmonger's archers and wars bring) is kept in a bounded ring with its feature vector; HORIZON turns
    later its reward is the deciding faction's log-score gain. Rewards update per-option ridge statistics; weights are
    re-solved every REFRESH turns with decay (non-stationary world, bounded numbers). Decisions: argmax over the allowed
    options of value + UCB bonus, through System.ask (hard checks first, `why` = the learned contributions)."""

    def __init__(self, seed: int):
        self.rng = random.Random(f"learner-{seed}")
        self.pending = {q: deque(maxlen=PENDING_MAX[q]) for q in LEARNED}   # [turn, fid, adaptive?, action, x]
        self.scores = deque(maxlen=max(s["horizon"] for s in SPECS.values()) + 2)  # [turn, {fid: score}]
        self.recent = deque(maxlen=400)       # [turn, adaptive?, question index, reward]
        self.curve = deque(maxlen=400)        # [turn, adaptive mean reward, others' mean reward, observed, share]
        self.observed = self.own = self.explored = self.refits = 0
        self.frozen = False
        self.eps = EPS
        self.base = {}                        # "question:fid" -> [running mean reward of that faction's decisions, turn]

    # --- compatibility names used by health.record / the app
    @property
    def teaches(self):
        return self.observed

    def bootstrap(self, system: System):
        """Prior: PRIOR_N synthetic states per question, the balanced rule's answer worth PRIOR_R."""
        rule_cat = make_catalog("adaptive")                       # the balanced rules (no learned heads)
        rule_sys = System(rule_cat, system.questions.values())
        rng = random.Random(1234)
        for q, spec in SPECS.items():
            rows, labels = [], []
            while len(rows) < PRIOR_N:
                st = GENERATORS[q](rng)
                r = rule_sys.ask(st, [q])[q]
                if r.status != "ok":
                    continue
                rows.append(row_of(q, system.facts_for(st)))
                labels.append(r.answer)
            h = ValueHead(spec["options"], spec["features"], spec["mask"]).setup(rows)
            for row, lab in zip(rows, labels):
                x = h._x(row)
                for o in h.allowed(row):
                    h.observe(x, o, PRIOR_R if o == lab else 0.0)
            h.n = 0
            h.solve()
            system.heads[q] = h

    # --- during the game
    def snapshot(self, turn, scores):
        self.scores.append([turn, {str(k): v for k, v in scores.items()}])

    def explore(self, options):
        if not self.frozen and self.rng.random() < self.eps and options:
            self.explored += 1
            return self.rng.choice(options)
        return None

    def observe(self, system, question, turn, fid, adaptive, values, action):
        if self.frozen:
            return
        spec = SPECS[question]
        if not adaptive and self.rng.random() >= spec["keep_others"]:
            return
        h = system.heads[question]
        row = row_of(question, values)
        if any(f not in row for f in spec["features"]) or action not in h.options:
            return
        self.pending[question].append([turn, fid, adaptive, action, h.vec(row)])

    def mature(self, system, turn, share):
        if self.frozen:
            return
        snap = {t: s for t, s in self.scores}
        now = snap.get(turn)
        if now is None:
            return
        for qi, (q, spec) in enumerate(SPECS.items()):
            h, pend, H = system.heads[q], self.pending[q], spec["horizon"]
            while pend and pend[0][0] + H <= turn:
                t, fid, adaptive, a, x = pend.popleft()
                then = snap.get(t)
                if then is None:
                    continue
                r = math.log(now.get(str(fid), 0) + 5) - math.log(then.get(str(fid), 0) + 5)
                key = f"{qi}:{fid}"                   # advantage: minus that faction's running mean reward
                b0 = self.base.get(key, [r, turn])[0]
                self.base[key] = [round(b0 + 0.05 * (r - b0), 6), turn]
                r = r - b0
                h.observe(x, a, r)
                self.observed += 1
                self.own += int(adaptive)
                self.recent.append([turn, adaptive, qi, round(r, 4)])
        if turn % 500 == 0:
            self.base = {k: v for k, v in self.base.items() if v[1] > turn - 500}
        if turn % REFRESH == 0:
            for q in LEARNED:
                system.heads[q].refresh()
            self.refits += 1
        if turn % 50 == 0:
            rec = [r for r in self.recent if r[0] > turn - 250 and r[2] == 0]
            ad = [r[3] for r in rec if r[1]]
            ot = [r[3] for r in rec if not r[1]]
            self.curve.append([turn, round(sum(ad) / len(ad), 4) if ad else None, round(sum(ot) / len(ot), 4) if ot else None,
                               self.observed, round(share, 3)])

    # --- exact save / load
    def to_dict(self, system):
        return {"rng": _rng_to(self.rng), "pending": {q: list(p) for q, p in self.pending.items()},
                "scores": list(self.scores), "recent": list(self.recent), "curve": list(self.curve),
                "observed": self.observed, "own": self.own, "explored": self.explored, "refits": self.refits,
                "frozen": self.frozen, "eps": self.eps, "base": self.base,
                "heads": {q: system.heads[q].to_dict() for q in LEARNED}}

    def load(self, d, system):
        self.rng = _rng_from(d["rng"])
        self.pending = {q: deque(d["pending"][q], maxlen=PENDING_MAX[q]) for q in LEARNED}
        self.scores = deque(d["scores"], maxlen=self.scores.maxlen)
        self.recent = deque(d["recent"], maxlen=400)
        self.curve = deque(d["curve"], maxlen=400)
        self.observed, self.own, self.explored, self.refits = d["observed"], d["own"], d["explored"], d["refits"]
        self.frozen, self.eps, self.base = d["frozen"], d["eps"], d["base"]
        for q in LEARNED:
            system.heads[q] = ValueHead.from_dict(d["heads"][q])
