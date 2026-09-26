"""The endless game. Every faction decision goes through its solvi System; the engine only applies the answers.

Endless by design: there is no win condition. A faction that loses its last city is replaced (a new faction lands on free
land, or a far-away city of the largest empire rebels), deposits deplete when worked and regenerate when not, random events
(drought, plague, gold rush, discovery) keep the world moving, and every log is a ring buffer, so the saved state stays
bounded however long the game runs."""
from __future__ import annotations

import hashlib
import json
import random
import time
from collections import deque

from . import econ
from .brains import (FEATURES, HORIZON, PERSONALITIES, QUESTIONS, REFIT_EVERY, Learner, _rng_from, _rng_to, argmax,
                     make_system)
from .world import (DEP_BONUS, DEP_MAX, DEPOSITS, FOREST, HILLS, IMPROVE, MOUNT, N, NEI8, RAD2, TERRAIN, W, WATER,
                    BASE_YIELD, DistCache, cheb, gen_map, passable)

MIN_FACTIONS, MAX_FACTIONS = 3, 5
STANCE_EVERY = 5
CONTACT = 14                    # factions whose cities are this close decide war/peace about each other
FACTION_NAMES = ["Aurelia", "Borealis", "Cyrene", "Dravia", "Elyra", "Fenmark", "Galdor", "Halvor", "Istra", "Jorvik",
                 "Kaldra", "Lumen", "Morrow", "Nysa", "Orin", "Pella", "Quill", "Rhosk", "Sable", "Tarn", "Umber", "Vesk",
                 "Wyn", "Xandor", "Yarrow", "Zeph"]
COLORS = ["#e6194b", "#3c7dd9", "#2ca02c", "#f58231", "#9b59b6", "#17becf", "#d4a017", "#e377c2", "#8c564b", "#7f7f7f"]
SYL_A = ["Ar", "Bel", "Cor", "Dun", "El", "Fal", "Gor", "Hal", "Ith", "Kor", "Lin", "Mar", "Nor", "Os", "Pel", "Qua",
         "Ros", "Sil", "Tor", "Ul", "Val", "Wen", "Yl", "Zan"]
SYL_B = ["ad", "bury", "dor", "ford", "gard", "heim", "is", "mere", "mont", "nor", "polis", "rest", "stad", "ton", "vale",
         "wick", "ia", "on", "um", "ay"]
REPLAN = {"warrior": 3, "archer": 3, "settler": 2, "worker": 6, "caravan": 6}
QUESTION_OF = {"warrior": "order_military", "archer": "order_military", "settler": "order_settler",
               "worker": "order_worker", "caravan": "order_caravan"}


def city_name(cid):
    return SYL_A[(cid * 7) % len(SYL_A)] + SYL_B[(cid * 11 + cid // len(SYL_A)) % len(SYL_B)]


def faction_name(fid):
    n = FACTION_NAMES[fid % len(FACTION_NAMES)]
    k = fid // len(FACTION_NAMES)
    return n if k == 0 else f"{n} {['', 'II', 'III', 'IV', 'V', 'VI'][min(k, 5)] or k + 1}"


class Metrics:
    """Per-window health counters (not part of the saved game)."""

    def __init__(self, replay_rate=0.02, seed=0):
        self.replay_rate = replay_rate
        self.rng = random.Random(f"replay-{seed}")
        self.reset()
        self.total_decisions = 0

    def reset(self):
        self.dec_ms, self.turn_ms = [], []
        self.n = self.forced = self.abstain = self.early = self.part_errors = 0
        self.replay_n = self.replay_ok = self.masked = self.fallback = 0
        self.by_q = {}
        self.forced_by = {}
        self.skipped_steps = 0


class Game:
    def __init__(self, seed=0, n_factions=4, adaptive=True, metrics=None, keep_last=True):
        self.seed = seed
        self.turn = 0
        self.rng = random.Random(seed)
        self.terrain, self.dep, self.amt = gen_map(seed)
        self.improved = [0] * N
        self.owner = [-1] * N            # city id owning the tile
        self.worked_n = [0] * N          # turns a deposit has been worked since its last depletion step
        self.cities, self.units, self.factions = {}, {}, {}
        self.wars = {}                   # "a-b" (a < b) -> turn the war started
        self.wants = {}                  # "a>b" -> a's latest stance answer about b
        self.truce = {}                  # "a-b" -> turn until which war cannot be declared
        self.effects = []                # active droughts: [kind, center tile, until turn]
        self.log = deque(maxlen=200)     # [turn, kind, text]
        self.next_id = 1
        self.spawned = 0
        self.eliminated = 0
        self.counters = {"battles": 0, "captures": 0, "founded": 0, "rebellions": 0, "events": 0, "disbanded_broke": 0,
                         "bailouts": 0, "negative_treasury": 0, "teach": 0}
        self.rewards = deque(maxlen=600)  # [turn, fid, adaptive?, reward]
        self.pending = deque(maxlen=600)  # [turn, city id, fid, value before]
        self.history = deque(maxlen=300)  # [turn, factions alive, population, gold, decisions ms p50, vetoes/100, ...]
        self.human = None                # faction id the human controls (its builds and stances), or None
        self.human_orders = {}           # city id -> build chosen by the human
        self.m = metrics or Metrics(seed=seed)
        self.keep_last = keep_last
        self.last = {}                   # entity key -> (Response, question, title) for the decision cards
        self._init_caches()
        self.learner = Learner(seed)
        self.learner.bootstrap(self.systems["adaptive"][1])
        pers = ["adaptive", "builder", "expansionist", "warmonger", "trader"] if adaptive else \
            ["builder", "expansionist", "warmonger", "trader"]
        for k in range(n_factions):
            self.spawn_faction(pers[k % len(pers)], initial=True)

    # ------------------------------------------------------------------------------------------------ setup / caches
    def _init_caches(self):
        self.dist = DistCache(self.terrain)
        self.systems = {p: make_system(p) for p in PERSONALITIES}
        self._site_turn = -999
        self._refresh_sites()

    def _refresh_sites(self):
        """Settle value of every land tile (sum of the yields around it), refreshed every 25 turns."""
        self.site_parts, vals = [None] * N, []
        for i in range(N):
            if not passable(self.terrain, i):
                continue
            f = w = s = g = 0
            for j in RAD2[i]:
                y = self.tile_yield(j)
                f, w, s, g = f + y[0], w + y[1], s + y[2], g + y[3]
            self.site_parts[i] = (f, w, s, g)
            vals.append((round(1.2 * f + w + s + 1.1 * g, 2), i))
        vals.sort(key=lambda t: (-t[0], t[1]))
        self.site_rank = vals[:120]
        self._site_turn = self.turn

    def new_id(self):
        self.next_id += 1
        return self.next_id

    def note(self, kind, text):
        self.log.append([self.turn, kind, text])

    # ------------------------------------------------------------------------------------------------ map helpers
    def tile_yield(self, i):
        f, w, s, g = BASE_YIELD[self.terrain[i]]
        d = self.dep[i]
        if d and self.amt[i] > 0:
            b = DEP_BONUS[d]
            f, w, s, g = f + b[0], w + b[1], s + b[2], g + b[3]
        if self.improved[i]:
            b = IMPROVE.get(self.terrain[i], (0, 0, 0, 0))
            f, w, s, g = f + b[0], w + b[1], s + b[2], g + b[3]
        for kind, center, until in self.effects:
            if kind == "drought" and cheb(i, center) <= 4:
                f = max(0, f - 1)
        return (f, w, s, g)

    def city_tiles(self, c):
        cid = c["id"]
        return [i for i in RAD2[c["pos"]] if self.owner[i] == cid]

    def worked(self, c):
        """The city tile (+1 food, +1 gold) plus the pop best tiles the city owns."""
        pos = c["pos"]
        y = self.tile_yield(pos)
        center = [y[0] + 1, y[1], y[2], y[3] + 1]
        cand = []
        for i in self.city_tiles(c):
            if i == pos:
                continue
            t = self.tile_yield(i)
            cand.append((1.5 * t[0] + t[1] + t[2] + 1.2 * t[3], -i, i, t))
        cand.sort(reverse=True)
        best = cand[: c["pop"]]
        return [center] + [list(t) for _, _, _, t in best], [i for _, _, i, _ in best]

    def fcities(self, fid):
        return [c for c in self.cities.values() if c["fid"] == fid]

    def funits(self, fid):
        return [u for u in self.units.values() if u["fid"] == fid]

    def alive(self):
        return sorted(f for f, d in self.factions.items() if d["alive"])

    def at_war(self, a, b):
        return f"{min(a, b)}-{max(a, b)}" in self.wars

    def units_at(self, i):
        return [u for u in self.units.values() if u["pos"] == i]

    def city_at(self, i):
        cid = self.owner[i]
        if cid >= 0:
            c = self.cities.get(cid)
            if c is not None and c["pos"] == i:
                return c
        return None

    def strength(self, fid):
        return sum(econ.STRENGTH[u["type"]] for u in self.units.values() if u["fid"] == fid) + \
            sum(1 for c in self.cities.values() if c["fid"] == fid)

    def upkeep(self, fid):
        return sum(econ.UPKEEP[u["type"]] for u in self.units.values() if u["fid"] == fid) + \
            sum(econ.building_upkeep(c["buildings"]) for c in self.cities.values() if c["fid"] == fid)

    def unit_cap(self, fid):
        return econ.unit_cap(len(self.fcities(fid)))

    def city_value(self, c):
        """What a build decision is judged by: population, yields, garrison."""
        ys = econ.city_yields(self.worked(c)[0], c["buildings"])
        gar = sum(1 for u in self.units.values() if u["pos"] == c["pos"] and u["type"] in econ.MILITARY)
        return 2 * c["pop"] + sum(ys) + 2 * min(gar, 2)

    # ------------------------------------------------------------------------------------------------ factions
    def spawn_faction(self, pers=None, initial=False, from_city=None):
        """A new faction on the best free site (or, for rebels, from an existing city). Returns the fid or None."""
        taken = {d["pers"] for d in self.factions.values() if d["alive"]}
        if pers is None:
            if "adaptive" not in taken:
                pers = "adaptive"          # the learner survives its faction: the next newcomer inherits it
            else:
                opts = [p for p in ["builder", "expansionist", "warmonger", "trader"] if p not in taken] or \
                    ["builder", "expansionist", "warmonger", "trader"]
                pers = self.rng.choice(opts)
        pos = None
        if from_city is None:
            pos = self._free_site(min_dist=6 if initial else 5)
            if pos is None:
                return None
        fid = len(self.factions)
        self.factions[fid] = {"fid": fid, "name": faction_name(fid), "pers": pers, "color": COLORS[fid % len(COLORS)],
                              "gold": 30, "wood": 12, "stone": 6, "income": 0, "alive": True, "born": self.turn,
                              "died": None, "capital": None, "grievance": {}, "trade": {}, "peak_cities": 0}
        self.spawned += 1
        if from_city is not None:
            c = from_city
            old = c["fid"]
            for u in list(self.units.values()):
                if u["pos"] == c["pos"] and u["fid"] == old:
                    u["fid"] = fid
                    u["goal"] = None
            self._transfer_city(c, fid)
            self.wars[f"{min(old, fid)}-{max(old, fid)}"] = self.turn
            self.add_unit(fid, "archer", c["pos"], fortified=True)
            self.add_unit(fid, "archer", c["pos"], fortified=True)
            self.counters["rebellions"] += 1
            self.note("rebels", f"{c['name']} rebels against {self.factions[old]['name']}: the {pers} "
                                f"{self.factions[fid]['name']} rise (war declared)")
            self.factions[fid]["gold"] = 40
        else:
            c = self.found_city(fid, pos, pop=2)
            self.add_unit(fid, "warrior", pos, fortified=True)
            self.add_unit(fid, "worker", pos)
            if not initial:
                self.add_unit(fid, "warrior", pos, fortified=True)
                self.note("spawn", f"the {pers} {self.factions[fid]['name']} land at {c['name']}")
        return fid

    def _free_site(self, min_dist):
        best = None
        for v, i in self.site_rank:
            if self.owner[i] != -1 or self.units_at(i):
                continue
            if any(cheb(i, c["pos"]) < min_dist for c in self.cities.values()):
                continue
            if self.dist.dist(i, self.site_rank[0][1]) >= 999 and self.cities:
                continue                                           # keep everyone on the main continent
            if best is None or v > best[0]:
                best = (v, i)
        return None if best is None else best[1]

    def found_city(self, fid, pos, pop=1):
        cid = self.new_id()
        c = {"id": cid, "fid": fid, "name": city_name(cid), "pos": pos, "pop": pop, "food": 0, "prog": 0, "build": None,
             "buildings": [], "founded": self.turn, "siege": 0}
        self.cities[cid] = c
        for i in RAD2[pos]:
            if self.owner[i] == -1 or i == pos:
                self.owner[i] = cid
        f = self.factions[fid]
        if f["capital"] is None or f["capital"] not in self.cities:
            f["capital"] = cid
        f["peak_cities"] = max(f["peak_cities"], len(self.fcities(fid)))
        self.counters["founded"] += 1
        return c

    def add_unit(self, fid, kind, pos, fortified=False):
        uid = self.new_id()
        self.units[uid] = {"id": uid, "fid": fid, "type": kind, "pos": pos, "fort": fortified, "goal": None, "age": 0,
                           "idle": 0, "work": 0, "ban": [], "wander": 0}
        return self.units[uid]

    def _transfer_city(self, c, new_fid):
        old = c["fid"]
        c["fid"] = new_fid
        c["pop"] = max(1, c["pop"] // 2)
        c["build"], c["prog"] = None, 0
        c["buildings"] = [b for b in c["buildings"] if b != "walls"]
        of, nf = self.factions[old], self.factions[new_fid]
        if of["capital"] == c["id"]:
            rest = self.fcities(old)
            of["capital"] = max(rest, key=lambda x: (x["pop"], -x["id"]))["id"] if rest else None
        if nf["capital"] is None or nf["capital"] not in self.cities or self.cities[nf["capital"]]["fid"] != new_fid:
            nf["capital"] = c["id"]
        nf["peak_cities"] = max(nf["peak_cities"], len(self.fcities(new_fid)))
        if not self.fcities(old):
            self._eliminate(old, by=new_fid)

    def _eliminate(self, fid, by=None):
        f = self.factions[fid]
        f["alive"], f["died"] = False, self.turn
        for u in [u for u in self.units.values() if u["fid"] == fid]:
            del self.units[u["id"]]
        for k in [k for k in self.wars if fid in map(int, k.split("-"))]:
            del self.wars[k]
        for k in [k for k in self.wants if fid in map(int, k.split(">"))]:
            del self.wants[k]
        for k in [k for k in self.truce if fid in map(int, k.split("-"))]:
            del self.truce[k]
        for d in self.factions.values():
            d["grievance"].pop(str(fid), None)
            d["trade"].pop(str(fid), None)
        self.eliminated += 1
        if self.human == fid:
            self.human = None
        self.note("fall", f"{f['name']} ({f['pers']}) is eliminated" + (f" by {self.factions[by]['name']}" if by is not None else ""))

    # ------------------------------------------------------------------------------------------------ solvi calls
    def ask(self, fid, question, state, key, title):
        pers = self.factions[fid]["pers"]
        cat, system = self.systems[pers]
        t0 = time.perf_counter()
        res = system.ask(state, [question])
        wall = (time.perf_counter() - t0) * 1000
        r = res[question]
        m = self.m
        m.n += 1
        m.total_decisions += 1
        m.dec_ms.append(wall)
        m.by_q[question] = m.by_q.get(question, 0) + 1
        if r.status == "forced":
            m.forced += 1
            fired = r.why.replace("hard check ", "").replace(" is false", "")
            m.forced_by[fired] = m.forced_by.get(fired, 0) + 1
        elif r.status == "abstain":
            m.abstain += 1
        if res.trace.skipped:
            m.early += 1
            m.skipped_steps += len(res.trace.skipped)
        for rec in res.trace.records:
            if rec.error and not rec.error.startswith("missing inputs"):
                m.part_errors += 1
        if m.replay_rate and m.rng.random() < m.replay_rate:
            rep = res.trace.replay(cat)
            m.replay_n += 1
            m.replay_ok += int(rep["ok"])
        if self.keep_last:
            self.last[key] = (res, question, title, pers, self.turn)
        return res

    # ------------------------------------------------------------------------------------------------ the turn
    def play(self, n=1):
        for _ in range(n):
            t0 = time.perf_counter()
            self.play_turn()
            self.m.turn_ms.append((time.perf_counter() - t0) * 1000)

    def play_turn(self):
        self.turn += 1
        if self.turn - self._site_turn >= 25:
            self._refresh_sites()
        for fid in self.alive():
            if self.factions[fid]["alive"]:
                self._faction_turn(fid)
        self._world_turn()

    def _threats(self, fid):
        """Enemy (at war) military units: [(tile, strength)]."""
        return [(u["pos"], econ.STRENGTH[u["type"]]) for u in self.units.values()
                if u["fid"] != fid and u["type"] in econ.MILITARY and self.at_war(fid, u["fid"])]

    def _faction_turn(self, fid):
        f = self.factions[fid]
        enemies = self._threats(fid)
        cities = sorted(self.fcities(fid), key=lambda c: c["id"])
        cap = f["capital"]
        mil_at = {}
        for u in self.units.values():
            if u["fid"] == fid and u["type"] in econ.MILITARY:
                mil_at[u["pos"]] = mil_at.get(u["pos"], 0) + econ.STRENGTH[u["type"]]
        income = wood = stone = 0
        city_info = {}
        for c in cities:
            wk, tiles = self.worked(c)
            for i in tiles:
                if self.dep[i]:
                    self.worked_n[i] += 1
            ys = econ.city_yields(wk, c["buildings"])
            c["food"] += ys[0] - 2 * c["pop"]
            if c["food"] >= econ.growth_threshold(c["pop"]):
                if c["pop"] < econ.MAX_POP and c["pop"] < len(self.city_tiles(c)):
                    c["pop"] += 1
                c["food"] = 0
            elif c["food"] < 0:
                c["pop"] = max(1, c["pop"] - 1)
                c["food"] = 0
            income += ys[3] + econ.city_tax(c["pop"])
            wood += ys[1]
            stone += ys[2]
            near = sum(s for p, s in enemies if cheb(p, c["pos"]) <= 3)
            gar = sum(v for p, v in mil_at.items() if cheb(p, c["pos"]) <= 1)
            city_info[c["id"]] = (wk, near, gar)
        f["gold"] += income
        f["wood"] += wood
        f["stone"] += stone
        f["income"] = income
        fleet = {k: 0 for k in econ.UNITS}
        for u in self.units.values():
            if u["fid"] == fid:
                fleet[u["type"]] += 1
        upkeep = self.upkeep(fid)
        at_war = any(self.at_war(fid, o) for o in self.alive() if o != fid)
        lazy = {}
        # ---- cities: production and build decisions
        for c in cities:
            if c["id"] not in self.cities or self.cities[c["id"]]["fid"] != fid:
                continue
            wk, near, gar = city_info[c["id"]]
            prod = econ.city_prod(c["pop"], c["buildings"])
            threat = near / (gar * (2 if "walls" in c["buildings"] else 1) + 1)
            besieged = threat >= econ.SIEGE_RATIO
            if besieged and c["build"] not in econ.MILITARY and not c["siege"]:
                c["build"] = None                                   # re-plan when a siege starts
            c["siege"] = int(besieged)
            if c["build"] is None:
                if not lazy:                                        # computed once per faction turn, only when needed
                    lazy["s"], lazy["t"] = self._settle_value(fid), len(self._trade_partners(fid))
                state = {"pop": c["pop"], "food_store": c["food"], "worked": wk, "buildings": sorted(c["buildings"]),
                         "gold": f["gold"], "wood": f["wood"], "stone": f["stone"], "income": income,
                         "upkeep": upkeep, "enemy_near": near, "garrison": gar, "is_capital": c["id"] == cap,
                         "n_cities": len(cities), "fleet": dict(fleet), "unit_cap": self.unit_cap(fid),
                         "settle_value": lazy["s"], "trade_partners": lazy["t"],
                         "at_war": at_war,
                         "unimproved": self._unimproved(c)}
                choice = self._decide_build(fid, c, state)
                if choice in econ.UNITS:
                    fleet[choice] += 1
                    upkeep += econ.UPKEEP[choice]
                elif choice in econ.BUILDINGS:
                    upkeep += econ.UPKEEP[choice]
            if c["build"] == "gold":
                f["gold"] += prod
                c["build"], c["prog"] = None, 0
                continue
            c["prog"] += prod
            gcap = 100 + 40 * len(cities)
            if c["build"] and f["gold"] > 0.7 * gcap and c["prog"] < econ.COST[c["build"]][0]:
                buy = min(econ.COST[c["build"]][0] - c["prog"], int((f["gold"] - 0.3 * gcap) // 2))
                if buy > 0:                                         # a rich treasury rushes production (2 gold / point)
                    f["gold"] -= 2 * buy
                    c["prog"] += buy
            if c["build"] and c["prog"] >= econ.COST[c["build"]][0]:
                item = c["build"]
                c["prog"] -= econ.COST[item][0]
                c["build"] = None
                if item in econ.UNITS:
                    self.add_unit(fid, item, c["pos"], fortified=item in econ.MILITARY)
                    if item == "settler":
                        c["pop"] = max(1, c["pop"] - 1)
                else:
                    c["buildings"].append(item)
                c["prog"] = min(c["prog"], 5)
        # ---- units
        self._enemy_tiles = {x["pos"] for x in self.units.values() if x["fid"] != fid and self.at_war(fid, x["fid"])}
        for u in sorted(self.funits(fid), key=lambda u: u["id"]):
            if u["id"] in self.units and self.units[u["id"]]["fid"] == fid and self.factions[fid]["alive"]:
                self._unit_turn(fid, u)
            if not self.factions[fid]["alive"]:
                return
        # ---- diplomacy
        if (self.turn + fid) % STANCE_EVERY == 0:
            self._diplomacy(fid)
        # ---- upkeep: the treasury never ends a turn below zero
        f = self.factions[fid]
        up = self.upkeep(fid)
        if f["gold"] - up < 0:
            capital = self.cities.get(f["capital"]) if f["capital"] is not None else None
            for u in sorted(self.funits(fid), key=lambda u: (-econ.UPKEEP[u["type"]], -u["id"])):
                if f["gold"] - up >= 0:
                    break
                if econ.UPKEEP[u["type"]] == 0:
                    continue
                if capital and u["pos"] == capital["pos"] and u["type"] in econ.MILITARY and \
                        sum(1 for x in self.units.values() if x["pos"] == capital["pos"] and x["fid"] == fid
                            and x["type"] in econ.MILITARY) <= 1:
                    continue
                up -= econ.UPKEEP[u["type"]]
                del self.units[u["id"]]
                self.counters["disbanded_broke"] += 1
            if f["gold"] - up < 0:
                self.counters["bailouts"] += 1
                up = f["gold"]                                    # debts forgiven (counted; should stay rare)
        f["gold"] -= up
        if f["gold"] < 0:
            self.counters["negative_treasury"] += 1
        n = len(self.fcities(fid))
        for res, cap_ in (("gold", 100 + 40 * n), ("wood", 40 + 15 * n), ("stone", 30 + 12 * n)):
            if f[res] > cap_:
                f[res] -= max(1, (f[res] - cap_) // 4)             # storage: a quarter of the surplus spoils each turn

    def _unimproved(self, c):
        _, tiles = self.worked(c)
        return sum(1 for i in tiles if not self.improved[i] and self.terrain[i] in IMPROVE)

    def _settle_value(self, fid):
        """Value of the best free site within reach of this faction (0 when there is none)."""
        cities = self.fcities(fid)
        if not cities:
            return 0.0
        for v, i in self.site_rank:
            if self.owner[i] != -1:
                continue
            if any(cheb(i, c["pos"]) < 3 for c in self.cities.values()):
                continue
            d = min(cheb(i, c["pos"]) for c in cities)
            if d <= 10:
                return round(max(0.0, v - 0.5 * d), 2)
        return 0.0

    def _trade_partners(self, fid):
        """Cities a caravan could reach: own cities 5+ tiles away from another own city, foreign cities at peace."""
        mine = self.fcities(fid)
        out = []
        for c in self.cities.values():
            if c["fid"] == fid:
                if len(mine) > 1:
                    out.append(c)
            elif not self.at_war(fid, c["fid"]) and any(cheb(c["pos"], m["pos"]) <= 12 for m in mine):
                out.append(c)
        return out

    def _decide_build(self, fid, c, state):
        f = self.factions[fid]
        key = f"city:{c['id']}"
        res = self.ask(fid, "build", state, key, f"{c['name']} ({f['name']}): what to build")
        r = res["build"]
        choice = r.answer
        aff = econ.affordable(f["gold"], f["wood"], f["stone"], c["pop"], state["fleet"], state["unit_cap"], c["buildings"])
        if f["pers"] == "adaptive" and r.status == "ok":
            if self.learner.rng.random() < self.learner.eps:
                choice = self.learner.rng.choice(aff)
                self.learner.explored += 1
            elif choice not in aff:
                self.m.masked += 1
                choice = max(aff, key=lambda o: (r.probs.get(o, 0.0), -econ.BUILD_OPTIONS.index(o)))
        if self.human == fid and c["id"] in self.human_orders:
            h = self.human_orders.pop(c["id"])
            if h in aff and r.status != "forced":
                choice = h
        if choice is None or choice not in aff:
            self.m.fallback += 1                                   # forced archer unaffordable, or an abstention
            choice = "warrior" if (r.answer == "archer" and "warrior" in aff) else "gold"
        pw, ww, sw, gw = econ.COST[choice]
        f["wood"] -= ww
        f["stone"] -= sw
        f["gold"] -= gw
        c["build"] = choice
        if choice != "gold":
            self.pending.append([self.turn, c["id"], fid, self.city_value(c)])
            if f["pers"] == "adaptive" and r.status == "ok":
                self.learner.pending.append([self.turn, c["id"], state, choice, self.city_value(c)])
        return choice

    # ------------------------------------------------------------------------------------------------ units
    def _unit_turn(self, fid, u):
        u["age"] += 1
        g = u["goal"]
        if u["type"] in econ.MILITARY:
            et = self._enemy_tiles
            enemy_adj = bool(et) and any(n in et for n in NEI8[u["pos"]])
            if g is None or u["age"] >= (6 if u["fort"] and not self.wars else REPLAN[u["type"]]) or enemy_adj or \
                    (g[1] is not None and g[1] == u["pos"] and g[0] != "fortify"):
                self._decide_military(fid, u)
            self._act_military(fid, u)
        elif u["type"] == "settler":
            if g is None or u["age"] >= REPLAN["settler"] or g[1] == u["pos"]:
                self._decide_settler(fid, u)
            self._act_settler(fid, u)
        elif u["type"] == "worker":
            if g is None or u["age"] >= REPLAN["worker"] or (g[0] == "move" and g[1] == u["pos"]):
                self._decide_worker(fid, u)
            self._act_worker(fid, u)
        else:
            if g is None or u["age"] >= REPLAN["caravan"] or cheb(u["pos"], g[1] if g[1] is not None else u["pos"]) <= 1:
                self._decide_caravan(fid, u)
            self._act_caravan(fid, u)

    def _decide_military(self, fid, u):
        f = self.factions[fid]
        cap = self.cities.get(f["capital"]) if f["capital"] is not None else None
        pos = u["pos"]
        targets = []
        if self.wars:
            for x in self.units.values():
                if x["fid"] != fid and self.at_war(fid, x["fid"]) and cheb(x["pos"], pos) <= 8:
                    if not self.city_at(x["pos"]):
                        targets.append([x["pos"], cheb(x["pos"], pos), self._defence(x["pos"], fid)[0], False])
            for c in self.cities.values():
                if c["fid"] != fid and self.at_war(fid, c["fid"]) and cheb(c["pos"], pos) <= 10:
                    targets.append([c["pos"], cheb(c["pos"], pos), self._defence(c["pos"], fid)[0], True])
        targets.sort(key=lambda t: (t[1], t[0]))
        targets = [[t, d, round(v, 2), ic] for t, d, v, ic in targets[:5]]
        enemies = self._threats(fid)
        threats = []
        for c in self.fcities(fid):
            near = sum(s for p, s in enemies if cheb(p, c["pos"]) <= 3)
            if near:
                gar = sum(econ.STRENGTH[x["type"]] for x in self.units.values()
                          if x["fid"] == fid and x["type"] in econ.MILITARY and cheb(x["pos"], c["pos"]) <= 1)
                d = cheb(c["pos"], pos)
                if d <= 8:
                    threats.append([c["pos"], d, round(near / (gar + 1), 2)])
        threats = sorted(threats, key=lambda t: (t[1], t[0]))[:3]
        cap_def = sum(1 for x in self.units.values() if cap and x["fid"] == fid and x["pos"] == cap["pos"]
                      and x["type"] in econ.MILITARY)
        mil = sum(1 for x in self.units.values() if x["fid"] == fid and x["type"] in econ.MILITARY)
        state = {"unit": u["type"], "strength": econ.STRENGTH[u["type"]], "targets": targets, "threats": threats,
                 "at_capital": bool(cap and pos == cap["pos"]), "capital_defenders": cap_def, "fortified": u["fort"],
                 "at_war": any(self.at_war(fid, o) for o in self.alive() if o != fid),
                 "surplus_troops": max(0, mil - 2 * len(self.fcities(fid))),
                 "income": f["income"], "upkeep": self.upkeep(fid)}
        res = self.ask(fid, "order_military", state, f"unit:{u['id']}", f"{u['type']} #{u['id']} ({f['name']})")
        a = res["order_military"].answer or "fortify"
        v = res.values
        tgt = None
        if a == "attack" and v.get("target_pick"):
            tgt = v["target_pick"][0]
        elif a == "defend" and v.get("defense_call"):
            tgt = v["defense_call"][0]
        elif a == "move":
            tgt = self._frontier(fid, pos)
        elif a == "fortify":
            tgt = pos
        u["goal"] = [a, tgt]
        u["age"] = 0

    def _frontier(self, fid, pos):
        """Where a patrol goes: the nearest enemy city at war; else 3 tiles short of the nearest foreign city."""
        best = None
        for c in self.cities.values():
            if c["fid"] == fid:
                continue
            d = cheb(c["pos"], pos) - (0 if self.at_war(fid, c["fid"]) else 100)
            if best is None or d < best[0] or (d == best[0] and c["pos"] < best[1]):
                best = (d, c["pos"], c["fid"])
        if best is None:
            own = self.fcities(fid)
            return self.rng.choice(own)["pos"] if own else pos
        if self.at_war(fid, best[2]):
            return best[1]
        target, p = best[1], pos
        for _ in range(12):
            if cheb(p, target) <= 3:
                break
            nxt = self.dist.step(p, target)
            if nxt == p:
                break
            p = nxt
        return p

    def _defence(self, tile, attacker):
        """(defence value, defending unit or None) of a tile against `attacker`'s faction."""
        best, bu = 0.0, None
        c = self.city_at(tile)
        walls = c is not None and "walls" in c["buildings"]
        for x in self.units.values():
            if x["pos"] != tile or x["type"] not in econ.MILITARY:
                continue
            v = econ.STRENGTH[x["type"]] * (1.5 if x["fort"] else 1.0) * (1.25 if self.terrain[tile] in (FOREST, HILLS) else 1.0)
            if c is not None:
                v *= 2.0 if walls else 1.25
            if v > best:
                best, bu = v, x
        if bu is None and c is not None:
            best = 1.0 + (1.0 if walls else 0.0) + c["pop"] * 0.1       # militia
        return best, bu

    def _act_military(self, fid, u):
        a, tgt = u["goal"] if u["goal"] else ("fortify", u["pos"])
        if a == "disband":
            del self.units[u["id"]]
            return
        if a == "fortify" or tgt is None or tgt == u["pos"]:
            u["fort"] = True
            return
        u["fort"] = False
        nxt = self.dist.step(u["pos"], tgt)
        if nxt == u["pos"]:
            return
        occupants = [x for x in self.units.values() if x["pos"] == nxt and x["fid"] != fid]
        city = self.city_at(nxt)
        foreign_city = city is not None and city["fid"] != fid
        hostile = any(self.at_war(fid, x["fid"]) for x in occupants) or (foreign_city and self.at_war(fid, city["fid"]))
        if hostile:
            self._battle(fid, u, nxt)
        elif not occupants and not foreign_city:
            u["pos"] = nxt

    def _battle(self, fid, u, tile):
        dv, du = self._defence(tile, fid)
        city = self.city_at(tile)
        a = econ.STRENGTH[u["type"]]
        self.counters["battles"] += 1
        victim = du["fid"] if du else (city["fid"] if city else None)
        if victim is None:                                      # only civilians on the tile
            victim = next(x["fid"] for x in self.units.values() if x["pos"] == tile and x["fid"] != fid)
        g = self.factions[victim]["grievance"]
        g[str(fid)] = g.get(str(fid), 0) + 1
        if du is None and city is None:
            for x in [x for x in self.units.values() if x["pos"] == tile and x["fid"] != fid]:
                del self.units[x["id"]]
            u["pos"] = tile
            return
        if self.rng.random() < a / (a + dv):
            if du is not None:
                del self.units[du["id"]]
            left = [x for x in self.units.values() if x["pos"] == tile and x["fid"] != fid and x["type"] in econ.MILITARY]
            if not left:
                for x in [x for x in self.units.values() if x["pos"] == tile and x["fid"] != fid]:
                    del self.units[x["id"]]
                if city is not None and du is None:
                    old = self.factions[city["fid"]]["name"]
                    self.counters["captures"] += 1
                    self.note("capture", f"{self.factions[fid]['name']} capture {city['name']} from {old}")
                    self._transfer_city(city, fid)
                    u["pos"] = tile
                elif city is None:
                    u["pos"] = tile
        else:
            del self.units[u["id"]]

    def _decide_settler(self, fid, u):
        f = self.factions[fid]
        pos = u["pos"]
        field = self.dist.field(pos)
        sites = []
        for v, i in self.site_rank:
            if len(sites) >= 5:
                break
            if i in u["ban"] or not self._legal_site(fid, i) or field[i] >= 999 or field[i] > 14:
                continue
            fp = self.site_parts[i]
            sites.append([i, fp[0], fp[1], fp[2], fp[3], field[i], self._enemy_city_dist(fid, i)])
        if pos not in [s[0] for s in sites] and self._legal_site(fid, pos) and pos not in u["ban"]:
            fp = self.site_parts[pos]
            sites.append([pos, fp[0], fp[1], fp[2], fp[3], 0, self._enemy_city_dist(fid, pos)])
        state = {"sites": sites, "here_enemy_dist": self._enemy_city_dist(fid, pos), "wander": u["wander"]}
        res = self.ask(fid, "order_settler", state, f"unit:{u['id']}", f"settler #{u['id']} ({f['name']})")
        r = res["order_settler"]
        a = r.answer or "fortify"
        best = res.values.get("best_site")
        if r.status == "forced" and pos not in u["ban"]:
            u["ban"] = (u["ban"] + [pos])[-4:]                 # vetoed here: never try this tile again
        tgt = pos if a == "settle" else (best[0] if best and best[0] != pos else None)
        u["goal"] = [a, tgt]
        u["age"] = 0

    def _enemy_city_dist(self, fid, i):
        return min([cheb(i, c["pos"]) for c in self.cities.values() if c["fid"] != fid] or [99])

    def _legal_site(self, fid, i):
        if not passable(self.terrain, i):
            return False
        o = self.owner[i]
        if o != -1 and (o not in self.cities or self.cities[o]["fid"] != fid):
            return False
        return all(cheb(i, c["pos"]) >= 3 for c in self.cities.values())

    def _act_settler(self, fid, u):
        a, tgt = u["goal"]
        u["wander"] += 1
        if a == "settle" and self._legal_site(fid, u["pos"]):
            c = self.found_city(fid, u["pos"], pop=1)
            del self.units[u["id"]]
            self.note("found", f"{self.factions[fid]['name']} found {c['name']}")
            return
        if u["wander"] > 40:
            del self.units[u["id"]]                           # gave up: nowhere to settle
            return
        if a == "move" and tgt is not None:
            nxt = self.dist.step(u["pos"], tgt)
            if not any(x["fid"] != fid for x in self.units_at(nxt)) and not (self.city_at(nxt) and self.city_at(nxt)["fid"] != fid):
                u["pos"] = nxt

    def _decide_worker(self, fid, u):
        f = self.factions[fid]
        job, best = None, None
        for c in self.fcities(fid):
            _, tiles = self.worked(c)
            for i in tiles:
                if not self.improved[i] and self.terrain[i] in IMPROVE:
                    d = cheb(u["pos"], i)
                    if best is None or (d, i) < best:
                        best = (d, i)
        if best is not None:
            job = [best[1], best[0], 1]
        state = {"job": job, "idle": u["idle"], "treasury_next": f["gold"] + f["income"] - self.upkeep(fid)}
        res = self.ask(fid, "order_worker", state, f"unit:{u['id']}", f"worker #{u['id']} ({f['name']})")
        a = res["order_worker"].answer or "fortify"
        u["goal"] = [a, job[0] if job else None]
        u["age"] = 0

    def _act_worker(self, fid, u):
        a, tgt = u["goal"]
        if a == "disband":
            del self.units[u["id"]]
            return
        if a == "fortify" or tgt is None:
            u["idle"] += 1
            return
        u["idle"] = 0
        if u["pos"] != tgt:
            nxt = self.dist.step(u["pos"], tgt)
            if not any(x["fid"] != fid for x in self.units_at(nxt)) and not (self.city_at(nxt) and self.city_at(nxt)["fid"] != fid):
                u["pos"] = nxt
            return
        u["work"] += 1
        if u["work"] >= 3:
            self.improved[tgt] = 1
            u["work"] = 0
            u["goal"] = None

    def _decide_caravan(self, fid, u):
        f = self.factions[fid]
        routes = []
        home = min(self.fcities(fid), key=lambda c: cheb(c["pos"], u["pos"]), default=None)
        for c in self._trade_partners(fid):
            if home is not None and c["id"] == home["id"] and c["fid"] == fid:
                continue
            d = cheb(c["pos"], u["pos"])
            val = 4 + cheb(c["pos"], home["pos"]) // 2 + (4 if c["fid"] != fid else 0) + c["pop"] // 2 if home else 4
            routes.append([c["pos"], d, val, self.at_war(fid, c["fid"])])
        routes = sorted(routes, key=lambda r: (r[1], r[0]))[:4]
        res = self.ask(fid, "order_caravan", {"routes": routes}, f"unit:{u['id']}", f"caravan #{u['id']} ({f['name']})")
        a = res["order_caravan"].answer or "disband"
        rp = res.values.get("route_pick")
        u["goal"] = [a, rp[0] if rp else None, rp[2] if rp else 0]
        u["age"] = 0

    def _act_caravan(self, fid, u):
        a, tgt, val = u["goal"]
        if a == "disband" or tgt is None:
            del self.units[u["id"]]
            return
        if a == "trade" or cheb(u["pos"], tgt) <= 1:
            c = self.city_at(tgt)
            f = self.factions[fid]
            if c is not None and not self.at_war(fid, c["fid"]):
                f["gold"] += val
                if c["fid"] != fid:
                    other = self.factions[c["fid"]]
                    other["gold"] += val // 2
                    for a_, b_ in ((f, c["fid"]), (other, fid)):
                        a_["trade"][str(b_)] = a_["trade"].get(str(b_), 0) + 1
                self.note("trade", f"a {f['name']} caravan trades at {c['name']} (+{val} gold)")
            del self.units[u["id"]]
            return
        nxt = self.dist.step(u["pos"], tgt)
        if not any(x["fid"] != fid for x in self.units_at(nxt)) and not self.city_at(nxt):
            u["pos"] = nxt

    # ------------------------------------------------------------------------------------------------ diplomacy
    def _diplomacy(self, fid):
        f = self.factions[fid]
        mine = self.fcities(fid)
        my_str = self.strength(fid)
        for o in self.alive():
            if o == fid:
                continue
            theirs = self.fcities(o)
            if not mine or not theirs:
                continue
            border = min(cheb(a["pos"], b["pos"]) for a in mine for b in theirs)
            if border > CONTACT:
                continue
            key = f"{min(fid, o)}-{max(fid, o)}"
            war = key in self.wars
            if not war and self.truce.get(key, -1) > self.turn:
                continue                                           # truce: nothing to decide yet
            others = sum(1 for k in self.wars if k != key and fid in map(int, k.split("-")))
            state = {"my_strength": my_str, "their_strength": self.strength(o), "border": border, "other_wars": others,
                     "grievance": round(f["grievance"].get(str(o), 0), 2), "trade_links": min(f["trade"].get(str(o), 0), 5),
                     "at_war": war, "war_len": self.turn - self.wars[key] if war else 0,
                     "my_cities": len(mine), "their_cities": len(theirs)}
            of = self.factions[o]
            res = self.ask(fid, "stance", state, f"stance:{fid}>{o}", f"{f['name']} → {of['name']}: war or peace")
            r = res["stance"]
            ans = r.answer or "peace"
            if self.human == fid and str(o) in self.human_orders.get("stance", {}):
                ans = self.human_orders["stance"][str(o)]
            self.wants[f"{fid}>{o}"] = ans
            if ans == "war" and not war:
                self.wars[key] = self.turn
                why = r.why
                self.note("war", f"{f['name']} declare war on {of['name']} (strength ratio "
                                 f"{res.values.get('strength_ratio')}, tension {res.values.get('tension')}) — {why[:80]}")
            elif ans == "peace" and war and self.wants.get(f"{o}>{fid}") == "peace" and self.turn - self.wars[key] >= 8:
                del self.wars[key]
                self.truce[key] = self.turn + econ.TRUCE
                self.note("peace", f"{f['name']} and {of['name']} make peace (truce for {econ.TRUCE} turns)")
            if r.status == "forced" and war:
                self.note("veto", f"{f['name']}: hard check war_needs_strength vetoes continuing the war on {of['name']} "
                                  f"(strength ratio {res.values.get('strength_ratio')} < {econ.WAR_MIN_RATIO})")

    # ------------------------------------------------------------------------------------------------ world
    def _world_turn(self):
        t = self.turn
        # deposits: worked ones deplete, unworked ones regenerate
        if t % 15 == 0:
            worked_now = set()
            for c in self.cities.values():
                worked_now.update(self.worked(c)[1])
            for i in range(N):
                if not self.dep[i]:
                    continue
                if self.worked_n[i] >= 12:
                    self.worked_n[i] = 0
                    self.amt[i] = max(0, self.amt[i] - 1)
                elif i not in worked_now and self.amt[i] < DEP_MAX:
                    self.amt[i] += 1
        self.effects = [e for e in self.effects if e[2] > t]
        # random events
        if self.rng.random() < 0.04:
            self._event()
        # grievances fade
        if t % 10 == 0:
            for f in self.factions.values():
                for k in list(f["grievance"]):
                    f["grievance"][k] = round(f["grievance"][k] * 0.7, 2)
                    if f["grievance"][k] < 0.1:
                        del f["grievance"][k]
                for k in list(f["trade"]):
                    f["trade"][k] = max(0, f["trade"][k] - 1)
                    if not f["trade"][k]:
                        del f["trade"][k]
        # endless: keep at least MIN_FACTIONS alive; big empires may see rebellions
        alive = self.alive()
        if len(alive) < MIN_FACTIONS:
            if self.spawn_faction() is None:
                self._rebellion(force=True)
        elif len(alive) < MAX_FACTIONS and self.rng.random() < 0.02:
            self._rebellion()
        # learning: judge build decisions HORIZON turns later
        self._mature()
        if self.turn % REFIT_EVERY == 0:
            self._refit()
        # bounded UI memory: forget cards of entities that no longer exist
        if self.keep_last and t % 10 == 0:
            for k in list(self.last):
                kind, _, rest = k.partition(":")
                gone = (kind == "city" and int(rest) not in self.cities) or (kind == "unit" and int(rest) not in self.units) \
                    or (kind == "stance" and not all(self.factions[int(x)]["alive"] for x in rest.split(">")))
                if gone or t - self.last[k][4] > 200:
                    del self.last[k]
        if t % 10 == 0:
            self._sample_history()

    def _event(self):
        kind = self.rng.choice(["drought", "plague", "gold_rush", "discovery", "harvest"])
        self.counters["events"] += 1
        if kind == "drought":
            center = self.rng.randrange(N)
            self.effects.append(["drought", center, self.turn + 12])
            self.effects = self.effects[-6:]
            self.note("event", f"drought around ({center % W},{center // W}) for 12 turns: food -1")
        elif kind == "plague" and self.cities:
            c = self.cities[self.rng.choice(sorted(self.cities))]
            lost = max(1, c["pop"] * 3 // 10)
            c["pop"] = max(1, c["pop"] - lost)
            self.note("event", f"plague in {c['name']}: -{lost} population")
        elif kind == "gold_rush":
            land = [i for i in range(N) if passable(self.terrain, i) and not self.dep[i]]
            if land:
                i = self.rng.choice(land)
                self.dep[i], self.amt[i] = "gold", DEP_MAX
                o = self.owner[i]
                who = ""
                if o >= 0 and o in self.cities:
                    self.factions[self.cities[o]["fid"]]["gold"] += 20
                    who = f" in {self.cities[o]['name']}'s land (+20 gold)"
                self.note("event", f"gold rush at ({i % W},{i // W}){who}")
        elif kind == "discovery":
            land = [i for i in range(N) if self.terrain[i] != WATER and not self.dep[i]]
            if land:
                i = self.rng.choice(land)
                d = self.rng.choice(DEPOSITS)
                self.dep[i], self.amt[i] = d, DEP_MAX
                self.note("event", f"new {d} deposit found at ({i % W},{i // W}) on {TERRAIN[self.terrain[i]]}")
        elif kind == "harvest" and self.cities:
            c = self.cities[self.rng.choice(sorted(self.cities))]
            c["food"] += 10
            self.note("event", f"bountiful harvest in {c['name']}: +10 food")
        # a depleted deposit sometimes vanishes: the map keeps changing
        dead = [i for i in range(N) if self.dep[i] and self.amt[i] == 0]
        if dead and self.rng.random() < 0.3:
            i = self.rng.choice(dead)
            self.dep[i] = ""

    def _rebellion(self, force=False):
        counts = {}
        for c in self.cities.values():
            counts[c["fid"]] = counts.get(c["fid"], 0) + 1
        if not counts:
            return
        big = max(counts, key=lambda k: (counts[k], -k))
        total = sum(counts.values())
        if not force and (counts[big] < 6 or counts[big] < 0.45 * total):
            return
        f = self.factions[big]
        cap = self.cities.get(f["capital"])
        cand = [c for c in self.fcities(big) if c["id"] != f["capital"]]
        if not cand or cap is None:
            return
        c = max(cand, key=lambda c: (cheb(c["pos"], cap["pos"]), c["id"]))
        self.spawn_faction(from_city=c)

    def _mature(self):
        t = self.turn
        L = self.learner
        while self.pending and self.pending[0][0] + HORIZON <= t:
            turn, cid, fid, before = self.pending.popleft()
            c = self.cities.get(cid)
            reward = -25 if c is None or c["fid"] != fid else self.city_value(c) - before
            self.rewards.append([t, fid, self.factions[fid]["pers"] == "adaptive", reward])
        while L.pending and L.pending[0][0] + HORIZON <= t:
            turn, cid, state, choice, before = L.pending.popleft()
            fid_ok = cid in self.cities and self.factions[self.cities[cid]["fid"]]["pers"] == "adaptive"
            reward = self.city_value(self.cities[cid]) - before if fid_ok else -25
            thr = L.threshold()
            L.rewards.append(reward)
            if reward >= thr and reward > 0:
                self.systems["adaptive"][1].teach("build", state, choice)
                L.examples.append([state, choice])
                L.teaches += 1
                self.counters["teach"] += 1
        if t % 50 == 0:
            recent = [r for r in self.rewards if r[0] > t - 250]
            ad = [r[3] for r in recent if r[2]]
            ot = [r[3] for r in recent if not r[2]]
            L.curve.append([t, round(sum(ad) / len(ad), 3) if ad else None, round(sum(ot) / len(ot), 3) if ot else None,
                            L.teaches, round(self._score_share(), 3)])
            L.eps = max(0.03, 0.12 * (1 - t / 20000))

    def _score_share(self):
        """Adaptive faction's score / mean score of the others (score = population + 3 × cities)."""
        sc = {}
        for c in self.cities.values():
            sc[c["fid"]] = sc.get(c["fid"], 0) + c["pop"] + 3
        ad = [v for k, v in sc.items() if self.factions[k]["pers"] == "adaptive"]
        ot = [v for k, v in sc.items() if self.factions[k]["pers"] != "adaptive"]
        if not ot:
            return 0.0
        return (ad[0] if ad else 0) / (sum(ot) / len(ot))

    def _refit(self):
        ex = [(s, a) for s, a in self.learner.examples]
        if len(ex) >= 40 and len({a for _, a in ex}) >= 2:
            self.systems["adaptive"][1].fit_fast("build", ex, features=FEATURES)
            self.learner.refits += 1

    def _sample_history(self):
        pops = sum(c["pop"] for c in self.cities.values())
        fs = self.alive()
        self.history.append([self.turn, len(fs), pops, len(self.cities), len(self.units),
                             sum(self.factions[f]["gold"] for f in fs), len(self.wars)])

    # ------------------------------------------------------------------------------------------------ save / load
    def to_dict(self):
        return {"version": 1, "seed": self.seed, "turn": self.turn, "rng": _rng_to(self.rng), "terrain": self.terrain,
                "dep": self.dep, "amt": self.amt, "improved": self.improved, "owner": self.owner, "worked_n": self.worked_n,
                "cities": list(self.cities.values()), "units": list(self.units.values()),
                "factions": list(self.factions.values()), "wars": self.wars, "wants": self.wants, "effects": self.effects,
                "truce": self.truce,
                "log": list(self.log), "next_id": self.next_id, "spawned": self.spawned, "eliminated": self.eliminated,
                "counters": self.counters, "rewards": list(self.rewards), "pending": list(self.pending),
                "history": list(self.history), "human": self.human, "site_turn": self._site_turn,
                "site_rank": self.site_rank, "site_parts": self.site_parts,
                "human_orders": {str(k): v for k, v in self.human_orders.items()},
                "learner": self.learner.to_dict(self.systems["adaptive"][1])}

    def to_json(self):
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, d, metrics=None, keep_last=True):
        g = cls.__new__(cls)
        g.seed, g.turn = d["seed"], d["turn"]
        g.rng = _rng_from(d["rng"])
        g.terrain, g.dep, g.amt, g.improved = d["terrain"], d["dep"], d["amt"], d["improved"]
        g.owner, g.worked_n = d["owner"], d["worked_n"]
        g.cities = {c["id"]: c for c in d["cities"]}
        g.units = {u["id"]: u for u in d["units"]}
        g.factions = {f["fid"]: f for f in d["factions"]}
        g.wars, g.wants, g.effects, g.truce = d["wars"], d["wants"], d["effects"], d["truce"]
        g.log = deque(d["log"], maxlen=200)
        g.next_id, g.spawned, g.eliminated, g.counters = d["next_id"], d["spawned"], d["eliminated"], d["counters"]
        g.rewards = deque(d["rewards"], maxlen=600)
        g.pending = deque(d["pending"], maxlen=600)
        g.history = deque(d["history"], maxlen=300)
        g.human = d.get("human")
        ho = d.get("human_orders", {})
        g.human_orders = {(int(k) if k.isdigit() else k): v for k, v in ho.items()}
        g.m = metrics or Metrics(seed=g.seed)
        g.keep_last = keep_last
        g.last = {}
        g._init_caches()
        g._site_turn = d["site_turn"]
        g.site_rank = [tuple(x) for x in d["site_rank"]]
        g.site_parts = [tuple(x) if x is not None else None for x in d["site_parts"]]
        g.learner = Learner(g.seed)
        g.learner.load(d["learner"], g.systems["adaptive"][1])
        return g

    @classmethod
    def from_json(cls, s, **kw):
        return cls.from_dict(json.loads(s), **kw)

    def state_hash(self):
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]

    # ------------------------------------------------------------------------------------------------ human player
    def human_set_build(self, cid, option):
        """The human (playing this city's faction) replaces the city's build; returns an error text or ''."""
        c = self.cities.get(cid)
        if c is None or c["fid"] != self.human:
            return "that city is not yours"
        f = self.factions[c["fid"]]
        fleet = {k: 0 for k in econ.UNITS}
        for u in self.funits(c["fid"]):
            fleet[u["type"]] += 1
        aff = econ.affordable(f["gold"], f["wood"], f["stone"], c["pop"], fleet, self.unit_cap(c["fid"]), c["buildings"])
        if option not in aff:
            return f"{option} is not affordable now (affordable: {', '.join(aff)})"
        _, ww, sw, gw = econ.COST[option]
        f["wood"] -= ww
        f["stone"] -= sw
        f["gold"] -= gw
        c["build"] = option
        self.note("human", f"you order {c['name']} to build {option}")
        return ""

    # ------------------------------------------------------------------------------------------------ summaries
    def summary(self):
        out = []
        for fid in self.alive():
            f = self.factions[fid]
            cs = self.fcities(fid)
            out.append({"fid": fid, "name": f["name"], "pers": f["pers"], "color": f["color"], "cities": len(cs),
                        "pop": sum(c["pop"] for c in cs), "units": len(self.funits(fid)), "gold": f["gold"],
                        "wood": f["wood"], "stone": f["stone"], "strength": self.strength(fid),
                        "wars": [self.factions[o]["name"] for o in self.alive() if o != fid and self.at_war(fid, o)]})
        return out


__all__ = ["Game", "Metrics", "QUESTIONS", "argmax", "MOUNT"]
