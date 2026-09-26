"""Game rules shared by the engine and the solvi catalogs: costs, upkeep, strength, yields. Pure functions, no state."""
from __future__ import annotations

UNITS = ["worker", "settler", "warrior", "archer", "caravan"]
BUILDINGS = ["farm", "lumber_mill", "mine", "walls", "market"]
BUILD_OPTIONS = UNITS + BUILDINGS + ["gold"]          # "gold": turn this city's production into gold for a turn

# item: (production, wood, stone, gold) paid when the item is started
COST = {"worker": (8, 0, 0, 0), "settler": (14, 5, 0, 0), "warrior": (8, 3, 0, 0), "archer": (11, 4, 2, 0),
        "caravan": (10, 0, 0, 6), "farm": (12, 5, 0, 0), "lumber_mill": (14, 0, 4, 0), "mine": (16, 6, 0, 0),
        "walls": (18, 0, 10, 0), "market": (20, 6, 6, 0), "gold": (1, 0, 0, 0)}
UPKEEP = {"worker": 1, "settler": 1, "warrior": 1, "archer": 2, "caravan": 0,
          "farm": 0, "lumber_mill": 1, "mine": 1, "walls": 1, "market": 0}
STRENGTH = {"worker": 0, "settler": 0, "warrior": 2, "archer": 3, "caravan": 0}
MILITARY = ("warrior", "archer")
CAPPED = ("worker", "warrior", "archer")   # settlers and caravans are consumed on arrival and have their own limits
TRUCE = 20                    # turns after a peace during which neither side can declare war again
WAR_MIN_RATIO = 0.8           # never declare (or keep) war below 80% of the enemy's strength
SIEGE_RATIO = 1.5             # enemy strength near a city / its defence above this = under siege
SETTLE_ENEMY_MIN = 5          # never settle closer than this (Chebyshev) to an enemy city: territories would touch
MAX_POP = 12


def city_yields(worked, buildings):
    """worked: [[food, wood, stone, gold], ...] (the city tile included) → [food, wood, stone, gold] with buildings."""
    f = w = s = g = 0
    for t in worked:
        f += t[0]
        w += t[1]
        s += t[2]
        g += t[3]
    if "farm" in buildings:
        f += 2
    if "lumber_mill" in buildings:
        w += 2
    if "mine" in buildings:
        s += 2
        g += 1
    if "market" in buildings:
        g += g // 2 + 1
    return [f, w, s, g]


def city_prod(pop, buildings):
    return 1 + pop // 2 + ("lumber_mill" in buildings) + ("mine" in buildings)


def city_tax(pop):
    return pop // 2


def growth_threshold(pop):
    return 8 + 4 * pop


def building_upkeep(buildings):
    return sum(UPKEEP.get(b, 0) for b in buildings)


def unit_cap(n_cities):
    return min(30, 4 + 2 * n_cities)


def affordable(gold, wood, stone, pop, fleet, cap, buildings):
    """Options this city can start now: resources paid up front, unit cap (workers + troops), at most 2 settlers and 3
    caravans on the road, one of each building, a settler needs pop 2."""
    out = []
    capped = sum(fleet.get(k, 0) for k in CAPPED)
    for o in BUILD_OPTIONS:
        _, w, s, g = COST[o]
        if w > wood or s > stone or g > gold:
            continue
        if o in CAPPED and capped >= cap:
            continue
        if (o == "settler" and fleet.get("settler", 0) >= 2) or (o == "caravan" and fleet.get("caravan", 0) >= 3):
            continue
        if o in BUILDINGS and o in buildings:
            continue
        if o == "settler" and pop < 2:
            continue
        out.append(o)
    return out
