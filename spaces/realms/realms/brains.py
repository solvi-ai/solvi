"""The factions' brains: one solvi catalog per personality (same functions, personality in the weights) and six questions.

Questions (each asked with a small per-decision init_state; the strategist plans only the parts that question needs):
  build           per city, when its queue is empty      worker settler warrior archer caravan farm lumber_mill mine walls market gold
  order_military  per warrior/archer                     attack defend fortify move disband
  order_settler   per settler                            settle move fortify
  order_worker    per worker                             work move fortify disband
  order_caravan   per caravan                            trade move disband
  stance          per neighbour, every 5 turns           war peace

HARD checks (a failed one forces the answer, no rule or model can override it):
  treasury_ok               build → "gold"            the treasury must stay >= 0 after next turn's upkeep
  not_under_siege           build → "archer"          enemy strength near the city > 1.5 × its defence: skip economic planning
  worker_upkeep_ok          order_worker → "disband"  a worker is disbanded when the treasury cannot carry it
  keeps_capital_defender    order_military → "fortify" the capital always keeps a defender
  settles_away_from_enemies order_settler → "move"    never found a city within 4 tiles of an enemy city (touching territory)
  war_needs_strength        stance → "peace"          never declare (or keep) war below 80% of the enemy's strength

Hard checks are planned first: when treasury_ok or not_under_siege fails, the whole economic plan of that city (yields, growth,
affordable options, needs, scores) is skipped (res.trace.skipped).

The adaptive faction has no rule for build, order_military or stance: each is answered by a learned value head (realms/adaptive.py).
For it, keeps_capital_defender and war_needs_strength look only at the state (the capital's last defender always stays; no
war below 80% strength), so they veto whatever the head proposes rather than a rule's preference."""
from __future__ import annotations

import base64
import random

import numpy as np
from solvi import Answer, Catalog, Question, System

from . import econ

MIL_OPTS = ["attack", "defend", "fortify", "move", "disband"]
SETTLER_OPTS = ["settle", "move", "fortify"]
WORKER_OPTS = ["work", "move", "fortify", "disband"]
CARAVAN_OPTS = ["trade", "move", "disband"]
STANCE_OPTS = ["war", "peace"]

QUESTIONS = [
    Question("build", "What should this city build next?", Answer.choice(econ.BUILD_OPTIONS),
             checkpoints=["treasury_ok", "not_under_siege"]),
    Question("order_military", "What should this warrior/archer do?", Answer.choice(MIL_OPTS),
             checkpoints=["keeps_capital_defender"]),
    Question("order_settler", "Where should this settler go?", Answer.choice(SETTLER_OPTS),
             checkpoints=["settles_away_from_enemies"]),
    Question("order_worker", "What should this worker do?", Answer.choice(WORKER_OPTS), checkpoints=["worker_upkeep_ok"]),
    Question("order_caravan", "What should this caravan do?", Answer.choice(CARAVAN_OPTS)),
    Question("stance", "War or peace with this neighbour?", Answer.choice(STANCE_OPTS), checkpoints=["war_needs_strength"]),
]

#                 growth infra  mil   expand trade  gold  aggr   site: food  gold  far
PERSONALITIES = {
    "builder":      dict(growth=1.5, infra=1.6, military=0.8, expand=0.9, trade=0.8, gold=1.0, aggression=0.35,
                         site_food=1.5, site_gold=0.8, site_far=0.8),
    "expansionist": dict(growth=1.0, infra=0.7, military=0.9, expand=2.0, trade=0.6, gold=0.9, aggression=0.6,
                         site_food=1.0, site_gold=1.0, site_far=0.35),
    "warmonger":    dict(growth=0.8, infra=0.6, military=1.9, expand=1.0, trade=0.4, gold=0.8, aggression=1.7,
                         site_food=1.0, site_gold=1.0, site_far=0.6),
    "trader":       dict(growth=1.0, infra=1.0, military=0.8, expand=0.9, trade=2.0, gold=1.4, aggression=0.35,
                         site_food=1.0, site_gold=1.6, site_far=0.7),
    "adaptive":     dict(growth=1.0, infra=1.0, military=1.0, expand=1.0, trade=1.0, gold=1.0, aggression=0.6,
                         site_food=1.2, site_gold=1.0, site_far=0.6),
}
BALANCED = PERSONALITIES["adaptive"]


def argmax(scores, options):
    """Highest score; ties go to the earlier option (deterministic)."""
    best, bv = None, None
    for o in options:
        if o in scores and (bv is None or scores[o] > bv):
            best, bv = o, scores[o]
    return best


def score_build(w, affordable, growth_need, infra_value, military_need, expand_value, trade_value, gold_need):
    """Personality-weighted value of every option this city can start now."""
    s = {}
    for o in affordable:
        if o == "worker":
            v = w["infra"] * infra_value["worker"]
        elif o == "settler":
            v = w["expand"] * expand_value
        elif o == "warrior":
            v = w["military"] * military_need * 0.8
        elif o == "archer":
            v = w["military"] * military_need
        elif o == "caravan":
            v = w["trade"] * trade_value
        elif o == "farm":
            v = w["growth"] * growth_need * infra_value["farm"]
        elif o == "walls":
            v = w["military"] * military_need * 0.6
        elif o == "market":
            v = (w["infra"] + w["trade"]) / 2 * infra_value["market"]
        elif o == "gold":
            v = w["gold"] * gold_need * 0.8 + 0.15
        else:
            v = w["infra"] * infra_value[o]
        s[o] = round(v, 3)
    return s


# ---------------------------------------------------------------------------------------------------------------------
# Derived scalar facts the adaptive faction's value heads read. Plain functions, so the learner can compute them from any
# faction's decision (observational data) exactly as the adaptive catalog computes them for its own decisions.
def d_target_is_city(target_pick):
    return bool(target_pick and target_pick[3])


def d_target_dist(target_pick):
    return target_pick[1] if target_pick else 11


def d_defend_need(defense_call):
    if not defense_call or defense_call[1] <= 0:
        return 0.0
    return round(defense_call[2] / (1 + 0.1 * defense_call[1]), 3)


def d_military_allowed(target_pick, defense_call, surplus_troops, net_income):
    """Orders that make sense for this unit now (the learned head only ranks these): attack needs a target, defend a
    threatened own city, disband surplus troops that the income cannot carry."""
    out = ["fortify", "move"]
    if target_pick:
        out.insert(0, "attack")
    if defense_call and defense_call[1] > 0:
        out.insert(1 if target_pick else 0, "defend")
    if surplus_troops > 0 and net_income < 0:
        out.append("disband")
    return out


def d_infra_best(infra_value):
    return max(infra_value.values())


DERIVED = {"target_is_city": (d_target_is_city, ["target_pick"]), "target_dist": (d_target_dist, ["target_pick"]),
           "defend_need": (d_defend_need, ["defense_call"]),
           "military_allowed": (d_military_allowed, ["target_pick", "defense_call", "surplus_troops", "net_income"]),
           "infra_best": (d_infra_best, ["infra_value"])}


def make_catalog(pers: str, learned_build: bool = False, learned: tuple = ()) -> Catalog:
    """learned: questions answered by a learned head instead of a rule (the adaptive faction: build, order_military,
    stance). For those, the hard checks look only at the state, never at a rule's preference, so they veto whatever the
    head proposes."""
    w = PERSONALITIES[pers]
    cat = Catalog()
    learned = set(learned) | ({"build"} if learned_build else set())

    # ------------------------------------------------------------------ city: build
    @cat.fn
    def net_income(income, upkeep):
        "gold per turn after unit and building upkeep"
        return income - upkeep

    @cat.fn
    def treasury_after_upkeep(gold, net_income):
        return gold + net_income

    @cat.fn
    def threat_ratio(enemy_near, garrison, buildings):
        "enemy strength within 3 tiles / (garrison × 2 with walls + 1)"
        return round(enemy_near / (garrison * (2 if "walls" in buildings else 1) + 1), 3)

    @cat.check(hard=True, then={"build": "gold"})
    def treasury_ok(treasury_after_upkeep):
        "HARD: the treasury stays >= 0 after next turn's upkeep, else the city makes gold"
        return treasury_after_upkeep >= 0

    @cat.check(hard=True, then={"build": "archer"})
    def not_under_siege(threat_ratio):
        "HARD: a besieged city (threat > 1.5) builds defenders and skips economic planning"
        return threat_ratio < econ.SIEGE_RATIO

    @cat.fn
    def yields(worked, buildings):
        "food, wood, stone, gold of the worked tiles with buildings"
        return econ.city_yields(worked, buildings)

    @cat.fn
    def food_surplus(yields, pop):
        return yields[0] - 2 * pop

    @cat.fn
    def growth_turns(food_surplus, food_store, pop):
        need = econ.growth_threshold(pop) - food_store
        if need <= 0:
            return 0
        return 99 if food_surplus <= 0 else -(-need // food_surplus)

    @cat.fn
    def affordable(gold, wood, stone, pop, fleet, unit_cap, buildings):
        "options the faction can pay for now (resources up front, unit cap, one of each building)"
        return econ.affordable(gold, wood, stone, pop, fleet, unit_cap, buildings)

    @cat.fn
    def growth_need(food_surplus, growth_turns, pop):
        if pop >= econ.MAX_POP:
            return 0.0
        return round(max(0.0, 1.0 - food_surplus / 4) + min(growth_turns, 20) / 20, 3)

    @cat.fn
    def infra_value(yields, pop, buildings, unimproved, fleet):
        b = buildings
        return {"farm": 0.0 if "farm" in b else round(0.6 + 0.1 * pop, 3),
                "lumber_mill": 0.0 if "lumber_mill" in b else round(0.3 + 0.2 * yields[1], 3),
                "mine": 0.0 if "mine" in b else round(0.3 + 0.2 * yields[2] + 0.15 * yields[3], 3),
                "market": 0.0 if "market" in b else round(0.2 + 0.15 * yields[3] + 0.05 * pop, 3),
                "worker": round(min(unimproved, 4) * 0.25 / (1 + fleet.get("worker", 0)), 3)}

    @cat.fn
    def infra_best(infra_value):
        return d_infra_best(infra_value)

    @cat.fn
    def military_need(threat_ratio, at_war, garrison, is_capital):
        want = 2 if is_capital else 1
        return round(threat_ratio * 1.2 + (0.8 if at_war else 0.0) + (0.6 if garrison < want * 2 else 0.0), 3)

    @cat.fn
    def expand_value(settle_value, n_cities, pop, fleet):
        if settle_value <= 0 or pop < 3:
            return 0.0
        return round(settle_value / 10 / (1 + 0.25 * n_cities) * (0.2 if fleet.get("settler", 0) else 1.0), 3)

    @cat.fn
    def trade_value(trade_partners, fleet):
        if trade_partners == 0:
            return 0.0
        return round(max(0.0, 0.4 + 0.15 * min(trade_partners, 4) - 0.35 * fleet.get("caravan", 0)), 3)

    @cat.fn
    def gold_need(net_income, gold):
        return round(max(0.0, 0.8 - net_income * 0.15) + (0.5 if gold < 20 else 0.0), 3)

    @cat.fn
    def build_scores(affordable, growth_need, infra_value, military_need, expand_value, trade_value, gold_need):
        "personality-weighted value of each affordable option"
        return score_build(w, affordable, growth_need, infra_value, military_need, expand_value, trade_value, gold_need)

    if "build" not in learned:
        @cat.rule("build")
        def build(build_scores):
            return argmax(build_scores, econ.BUILD_OPTIONS) or "gold"

    # ------------------------------------------------------------------ military units
    @cat.fn
    def target_pick(targets, strength):
        "best enemy target: [tile, distance, win odds, is_city] (targets: [tile, dist, defence, is_city])"
        best, bv = None, -1.0
        for t, d, dfn, is_city in targets:
            odds = strength / (strength + dfn) if strength + dfn > 0 else 0.0
            v = odds * (1.6 if is_city else 1.0) / (1 + 0.15 * d)
            if v > bv:
                best, bv = [t, d, round(odds, 3), is_city], v
        return best

    @cat.fn
    def attack_odds(target_pick):
        return target_pick[2] if target_pick else 0.0

    @cat.check
    def good_odds(attack_odds):
        "soft: an attack has at least even odds"
        return attack_odds >= 0.5

    @cat.fn
    def defense_call(threats):
        "most threatened own city nearby: [tile, distance, threat ratio] or None"
        best = None
        for t, d, r in threats:
            if best is None or r / (1 + 0.1 * d) > best[2] / (1 + 0.1 * best[1]):
                best = [t, d, r]
        return best

    @cat.fn
    def last_capital_defender(at_capital, capital_defenders):
        return at_capital and capital_defenders <= 1

    @cat.fn
    def military_scores(target_pick, attack_odds, defense_call, at_capital, fortified, at_war, surplus_troops, net_income):
        s = {"attack": 0.0, "defend": 0.0, "fortify": 0.3 + (0.45 if at_capital else 0.0) + (0.15 if fortified else 0.0),
             "move": 0.2 + (0.3 * w["aggression"] if at_war and not target_pick else 0.0), "disband": 0.0}
        if target_pick:
            s["attack"] = w["aggression"] * attack_odds * (1.5 if target_pick[3] else 1.0) / (1 + 0.12 * target_pick[1])
        if defense_call and defense_call[1] > 0:
            s["defend"] = 1.1 * defense_call[2] / (1 + 0.1 * defense_call[1])
        if net_income < 0 and surplus_troops > 0:
            s["disband"] = 0.9
        return {k: round(v, 3) for k, v in s.items()}

    @cat.fn
    def target_is_city(target_pick):
        "the picked target is an enemy city"
        return d_target_is_city(target_pick)

    @cat.fn
    def target_dist(target_pick):
        "distance to the picked target (11: none)"
        return d_target_dist(target_pick)

    @cat.fn
    def defend_need(defense_call):
        "threat ratio of the most threatened own city nearby, discounted by distance (0: none)"
        return d_defend_need(defense_call)

    @cat.fn
    def military_allowed(target_pick, defense_call, surplus_troops, net_income):
        "orders that make sense now: attack needs a target, defend a threatened city, disband surplus troops when broke"
        return d_military_allowed(target_pick, defense_call, surplus_troops, net_income)

    if "order_military" in learned:
        @cat.check(hard=True, then={"order_military": "fortify"})
        def keeps_capital_defender(last_capital_defender):
            "HARD: the capital's last defender never leaves it (whatever the learned head prefers)"
            return not last_capital_defender
    else:
        @cat.check(hard=True, then={"order_military": "fortify"})
        def keeps_capital_defender(last_capital_defender, military_scores):
            "HARD: the capital's last defender never leaves it"
            return not last_capital_defender or argmax(military_scores, MIL_OPTS) in ("fortify", "defend")

        @cat.rule("order_military")
        def order_military(military_scores):
            return argmax(military_scores, MIL_OPTS)

    # ------------------------------------------------------------------ settlers
    @cat.fn
    def site_scores(sites):
        "settle sites [tile, food, wood, stone, gold, distance, enemy city distance] scored by this personality"
        out = []
        for t, f, wd, st, g, d, ed in sites:
            v = w["site_food"] * f + wd + st + w["site_gold"] * g - w["site_far"] * 0.6 * d - (4.0 if ed < 7 else 0.0)
            out.append([t, round(v, 2), d])
        return out

    @cat.fn
    def best_site(site_scores):
        best = None
        for s in site_scores:
            if best is None or s[1] > best[1]:
                best = s
        return best

    @cat.fn
    def settler_scores(best_site, site_scores, wander):
        here = next((s for s in site_scores if s[2] == 0), None)
        s = {"settle": 0.0, "move": 0.6 if best_site else 0.0, "fortify": 0.1}
        if here is not None:
            s["settle"] = 1.0 if best_site is None or here[1] >= best_site[1] - 1.5 - 0.3 * wander else 0.3
        return s

    @cat.check(hard=True, then={"order_settler": "move"})
    def settles_away_from_enemies(settler_scores, here_enemy_dist):
        "HARD: never found a city within 4 tiles of an enemy city"
        return argmax(settler_scores, SETTLER_OPTS) != "settle" or here_enemy_dist >= econ.SETTLE_ENEMY_MIN

    @cat.rule("order_settler")
    def order_settler(settler_scores):
        return argmax(settler_scores, SETTLER_OPTS)

    # ------------------------------------------------------------------ workers
    @cat.check(hard=True, then={"order_worker": "disband"})
    def worker_upkeep_ok(treasury_next):
        "HARD: disband a worker when next turn's treasury would be negative"
        return treasury_next >= 0

    @cat.fn
    def worker_scores(job, idle):
        "job: [tile, distance, gain] of the nearest unimproved worked tile, or None"
        if job is None:
            return {"work": 0.0, "move": 0.0, "fortify": 0.2, "disband": 0.5 if idle > 8 else 0.0}
        return {"work": 1.0 if job[1] == 0 else 0.0, "move": 0.8 if job[1] > 0 else 0.0, "fortify": 0.1, "disband": 0.0}

    @cat.rule("order_worker")
    def order_worker(worker_scores):
        return argmax(worker_scores, WORKER_OPTS)

    # ------------------------------------------------------------------ caravans
    @cat.fn
    def route_pick(routes):
        "best trade route [tile, distance, value, at_war] by value per distance"
        best, bv = None, -1.0
        for r in routes:
            v = r[2] / (1 + 0.1 * r[1]) * w["trade"]
            if not r[3] and v > bv:
                best, bv = r, v
        return best

    @cat.check
    def route_safe(route_pick):
        "soft: the chosen partner is not at war with us"
        return route_pick is None or not route_pick[3]

    @cat.fn
    def caravan_scores(route_pick):
        if route_pick is None:
            return {"trade": 0.0, "move": 0.0, "disband": 0.4}
        return {"trade": 1.0 if route_pick[1] <= 1 else 0.0, "move": 0.8, "disband": 0.0}

    @cat.rule("order_caravan")
    def order_caravan(caravan_scores):
        return argmax(caravan_scores, CARAVAN_OPTS)

    # ------------------------------------------------------------------ diplomacy
    @cat.fn
    def strength_ratio(my_strength, their_strength):
        "our military strength / theirs"
        return round(my_strength / max(their_strength, 0.5), 3)

    @cat.fn
    def tension(border, grievance, trade_links):
        "close borders and past attacks raise tension, trade lowers it"
        return round(max(0, 7 - border) * 0.2 + grievance * 0.35 - trade_links * 0.25, 3)

    @cat.fn
    def stance_scores(strength_ratio, tension, at_war, war_len, other_wars, my_cities, their_cities):
        "war: aggression × (strength, tension, land hunger), minus other open wars and war weariness"
        hunger = 0.25 if their_cities > my_cities else 0.0
        war = w["aggression"] * (0.35 * min(strength_ratio, 3.0) + tension + hunger) - 0.45 * other_wars
        if at_war:
            war += 0.3 if war_len < 8 else -0.02 * (war_len - 8)     # wars last a while, then weariness sets in
        peace = 1.2 + 0.2 * w["trade"]
        return {"war": round(war, 3), "peace": round(peace, 3)}

    if "stance" in learned:
        @cat.check(hard=True, then={"stance": "peace"})
        def war_needs_strength(strength_ratio):
            "HARD: never declare or keep a war below 80% of the enemy's strength (whatever the learned head prefers)"
            return strength_ratio >= econ.WAR_MIN_RATIO
    else:
        @cat.check(hard=True, then={"stance": "peace"})
        def war_needs_strength(stance_scores, strength_ratio):
            "HARD: never declare or keep a war below 80% of the enemy's strength"
            return argmax(stance_scores, STANCE_OPTS) != "war" or strength_ratio >= econ.WAR_MIN_RATIO

        @cat.rule("stance")
        def stance(stance_scores):
            return argmax(stance_scores, STANCE_OPTS)

    return cat


LEARNED_QUESTIONS = ("build", "order_military", "stance")


def make_system(pers: str) -> tuple[Catalog, System]:
    cat = make_catalog(pers, learned=LEARNED_QUESTIONS if pers == "adaptive" else ())
    system = System(cat, QUESTIONS)
    system.learn = False        # solvi >= 0.3.x learns a check order / costs after every ask; realms does not use them,
    return cat, system          # and that refit made rare asks slow (decision p99 3 → 40-90 ms in long runs)


def random_city_state(rng: random.Random) -> dict:
    """A plausible city build request, for bootstrapping the adaptive head before it has any experience."""
    pop = rng.randint(1, 9)
    worked = [[rng.choice([0, 1, 2, 2, 3]), rng.choice([0, 0, 1, 2]), rng.choice([0, 0, 1, 2]), rng.choice([0, 1, 1, 3])]
              for _ in range(pop + 1)]
    b = sorted(x for x in econ.BUILDINGS if rng.random() < 0.3)
    fleet = {"worker": rng.randint(0, 2), "settler": rng.randint(0, 1), "warrior": rng.randint(0, 3),
             "archer": rng.randint(0, 2), "caravan": rng.randint(0, 1)}
    return {"pop": pop, "food_store": rng.randint(0, 20), "worked": worked, "buildings": b,
            "gold": rng.randint(0, 150), "wood": rng.randint(0, 40), "stone": rng.randint(0, 30),
            "income": rng.randint(2, 20), "upkeep": rng.randint(0, 12), "enemy_near": rng.choice([0, 0, 0, 2, 4, 6]),
            "garrison": rng.choice([0, 2, 2, 3, 5]), "is_capital": rng.random() < 0.3, "n_cities": rng.randint(1, 8),
            "fleet": fleet, "unit_cap": rng.choice([6, 8, 12]), "settle_value": rng.choice([0.0, 8.0, 12.0, 16.0]),
            "trade_partners": rng.randint(0, 4), "at_war": rng.random() < 0.3, "unimproved": rng.randint(0, 5)}


def _rng_to(r):
    a, b, c = r.getstate()
    return [a, list(b), c]


def _rng_from(s):
    r = random.Random()
    r.setstate((s[0], tuple(s[1]), s[2]))
    return r


def _arr(a):
    return {"shape": list(a.shape), "b64": base64.b64encode(np.ascontiguousarray(a, dtype=np.float64).tobytes()).decode()}


def _unarr(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype=np.float64).reshape(d["shape"]).copy()


def head_to_dict(h):
    """Exact serialization of a solvi FastHead (so a saved game continues bit-for-bit)."""
    if h is None:
        return None
    spec = {f: [s[0]] + [x if not isinstance(x, np.ndarray) else x.tolist() for x in s[1:]] for f, s in h.fz.spec.items()}
    return {"options": h.options, "lam": h.lam, "pairs": h.pairs, "features": h.features, "n": h.n, "loo_acc": h.loo_acc,
            "spec": spec, "Ainv": _arr(h.Ainv), "B": _arr(h.B), "W": _arr(h.W)}


def head_from_dict(d):
    from solvi.fast import FastHead, VecFeaturizer
    h = FastHead(d["options"], lam=d["lam"], pairs=d["pairs"])
    h.features, h.n, h.loo_acc = d["features"], d["n"], d["loo_acc"]
    h.fz = VecFeaturizer()
    h.fz.spec = {f: tuple(s) for f, s in d["spec"].items()}
    h.Ainv, h.B, h.W = _unarr(d["Ainv"]), _unarr(d["B"]), _unarr(d["W"])
    return h
