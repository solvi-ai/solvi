"""How data/world.json was made: the place graph of a recorded Pokémon Red playthrough. Not needed to run anything.

    python tools/extract_world.py --run RUN_DIR --harness HARNESS_DIR --rom pokemon_red.gb --out data/world.json

Inputs (none of them is in this repository): a playthrough recorded by a rules-mode game harness — its decisions.jsonl
(each overworld decision with the map, the position and the options on offer, such as "Leave north to ROUTE_1" or
"Go to OAKS_LAB") and story.jsonl (when each goal of the story was reached); the harness's tables of map names and
story goals; and your own ROM, read only for the door coordinates of the game's map tables (nothing of it is copied).

What it writes: places (a map, or a part of a map when exits seen together split it), exits as a player sees them
("Leave north", "Door at (x,y)") with where they led, the stage of the story from which each exit is on offer and can
be passed, two scripted moves, and the first fifteen goals of the story. Exits into places the recording never entered
are kept only when the ROM's tables show the place to be a dead end (every door of it leads back)."""
import argparse
import ast
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="the recorded run folder (decisions.jsonl, story.jsonl)")
ap.add_argument("--harness", required=True, help="the game harness folder (its tables of map names and story goals)")
ap.add_argument("--rom", required=True, help="your Pokémon Red ROM (read for door coordinates only)")
ap.add_argument("--out", required=True)
A = ap.parse_args()
sys.path.insert(0, A.harness)
from solvimon.data import maps as map_table  # noqa: E402
from solvimon.milestones import M  # noqa: E402

RUN, ROM, OUT = Path(A.run), Path(A.rom), Path(A.out)

EXIT_RE = re.compile(r"^Leave (north|south|east|west) to ([A-Z0-9_]+)(?: \(([xy]) (\d+)-(\d+)\))?$")
WARP_RE = re.compile(r"^Go to ([A-Z0-9_]+)(?: \((?:west|east|north|south|middle) door (\d+),(\d+)\))?$")

# --- ROM: warps per map (x, y, dest_map) at the pokered addresses for this ROM
b = ROM.read_bytes()
NAMES = {k: v["name"] for k, v in map_table().items()}
ID = {v: k for k, v in NAMES.items()}


def u16(a):
    return b[a] | (b[a + 1] << 8)


def flat(bank, ptr):
    return ptr if ptr < 0x4000 else bank * 0x4000 + (ptr - 0x4000)


def rom_map(mid):
    bank = b[0xC23D + mid]
    h = flat(bank, u16(0x01AE + mid * 2))
    flags = b[h + 9]
    h += 10
    conns = []
    for bit, d in ((3, "north"), (2, "south"), (1, "west"), (0, "east")):
        if flags & (1 << bit):
            conns.append((d, b[h]))
            h += 11
    o = flat(bank, u16(h)) + 1
    n = b[o]
    o += 1
    warps = []
    for _ in range(n):
        warps.append((b[o + 1], b[o], b[o + 3], b[o + 2]))     # x, y, dest map, dest warp index
        o += 4
    return conns, warps


ms_at = []                                              # (decision n at which milestone i completed)
for line in open(RUN / "story.jsonl"):
    d = json.loads(line)
    if d["kind"] == "milestone":
        ms_at.append(d["decisions"])


def milestone_of(n):
    """index of the milestone being worked on at decision n"""
    i = 0
    while i < len(ms_at) and n > ms_at[i]:
        i += 1
    return i


recs = []
kinds = Counter()
for line in open(RUN / "decisions.jsonl"):
    d = json.loads(line)
    kinds[d["kind"]] += 1
    if d["kind"] != "overworld":
        continue
    opts = d["options"]
    if isinstance(opts, str):
        opts = ast.literal_eval(opts)
    ctx = d.get("context") or {}
    recs.append({"n": d["n"], "map": ctx.get("map"), "x": ctx.get("x"), "y": ctx.get("y"),
                 "keys": [o["key"] for o in opts], "answer": d["answer"]})

warps_of = {}


def hidden(mapn, key):
    """(hidden key, destination map) or None"""
    m = EXIT_RE.match(key)
    if m:
        d, to, ax, lo, hi = m.groups()
        return (f"Leave {d}" + (f" ({ax} {lo}-{hi})" if ax else ""), to)
    m = WARP_RE.match(key)
    if m:
        to, x, y = m.groups()
        if x is None:
            mid = ID.get(mapn)
            if mid not in warps_of:
                warps_of[mid] = rom_map(mid)[1]
            cand = sorted((wx, wy) for wx, wy, dm, _ in warps_of[mid] if NAMES.get(dm) == to)
            if not cand:                                 # LAST_MAP warps (0xFF): the recording knows where they led
                cand = sorted((wx, wy) for wx, wy, dm, _ in warps_of[mid] if dm == 0xFF)
            if len(cand) != 1 and not (len(cand) == 2 and abs(cand[0][0] - cand[1][0]) + abs(cand[0][1] - cand[1][1]) == 1):
                return (f"Door to the unknown {to}", to) if not cand else (f"Door at ({cand[0][0]},{cand[0][1]})", to)
            x, y = cand[0]
        return (f"Door at ({x},{y})", to)
    return None


# --- parts of maps: exits seen together in one decision are in one part
parent = {}


def find(a):
    while parent.setdefault(a, a) != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, c):
    parent[find(a)] = find(c)


seen = []                                               # per record: [(hk, dest)]
for r in recs:
    ex = [h for h in (hidden(r["map"], k) for k in r["keys"]) if h]
    seen.append(ex)
    ks = [(r["map"], hk) for hk, _ in ex]
    for k in ks:
        find(k)
    for k in ks[1:]:
        union(ks[0], k)
    if not ks:
        find((r["map"], "__nokeys__"))

comp = defaultdict(set)
for k in list(parent):
    comp[(k[0], find(k))].add(k[1])
parts = defaultdict(list)                               # map → [root]
for (mapn, root) in comp:
    parts[mapn].append(root)


def part_xy(mapn, root):
    xs = [(r["x"], r["y"]) for r, ex in zip(recs, seen) if r["map"] == mapn and ex and find((mapn, ex[0][0])) == root]
    return (sum(x for x, _ in xs) / max(1, len(xs)), sum(y for _, y in xs) / max(1, len(xs)))


place_name = {}
for mapn, roots in parts.items():
    roots = [rt for rt in roots if rt[1] != "__nokeys__"] or roots
    if len(roots) == 1:
        place_name[(mapn, roots[0])] = mapn
        continue
    cs = {rt: part_xy(mapn, rt) for rt in roots}
    spanx = max(c[0] for c in cs.values()) - min(c[0] for c in cs.values())
    spany = max(c[1] for c in cs.values()) - min(c[1] for c in cs.values())
    axis = 0 if spanx >= spany else 1
    order = sorted(roots, key=lambda rt: cs[rt][axis])
    words = (["west", "east"] if axis == 0 else ["north", "south"]) if len(order) == 2 else \
        [f"part {i + 1}" for i in range(len(order))]
    for w, rt in zip(words, order):
        place_name[(mapn, rt)] = f"{mapn} ({w})"


def place_of(mapn, ex):
    if ex:
        return place_name[(mapn, find((mapn, ex[0][0])))]
    roots = parts.get(mapn, [])
    named = {place_name.get((mapn, rt)) for rt in roots if (mapn, rt) in place_name}
    return next(iter(named)) if len(named) == 1 else None


# --- walk the recording: places, exits with the milestone they were first on offer, observed transitions
world = {}
observed = {}                                           # (place, hk) → destination place
taken_at = {}                                           # (place, hk) → the stage it was first taken
prev = None
for r, ex in zip(recs, seen):
    p = place_of(r["map"], ex)
    if p is None:
        prev = None
        continue
    mi = milestone_of(r["n"])
    w = world.setdefault(p, {"map": r["map"], "first": mi, "exits": {}, "services": set()})
    w.setdefault("stages", set()).add(mi)
    for hk, to in ex:
        e = w["exits"].setdefault(hk, {"map": to, "open": mi, "stages": set()})
        e["open"] = min(e["open"], mi)
        e["stages"].add(mi)
    for k in r["keys"]:
        if k == "Talk to the nurse":
            w["services"].add("center")
        elif k == "Talk to the shop clerk":
            w["services"].add("mart")
        elif k == "Walk in the tall grass":
            w["services"].add("grass")
    if prev is not None and prev[0] != p:
        pp, pans, _ = prev
        h = hidden(world[pp]["map"], pans) if pans else None
        if h and h[1] == r["map"]:
            observed[(pp, h[0])] = p
            taken_at.setdefault((pp, h[0]), prev[2])
            w["first"] = min(w["first"], prev[2])      # the player arrived during the decision before
    prev = (p, r["answer"], mi)

# destinations: observed, else the single part of the destination map, else the part nearest to where the ROM says
# the exit arrives; a building the recording never entered is added when the ROM says it is a dead end (every door of
# it leads back)
def positions(place):
    return [(r["x"], r["y"]) for r, ex in zip(recs, seen) if ex and place_of(r["map"], ex) == place]


def door_xy(mapn, hk):
    m = re.match(r"Door at \((\d+),(\d+)\)", hk)
    return (int(m.group(1)), int(m.group(2))) if m else None


def resolve(src_map, hk, dst_map, cands):
    mid, did = ID[src_map], ID[dst_map]
    if hk.startswith("Leave "):
        d = hk.split()[1]
        key = {"east": lambda c: min(x for x, _ in positions(c)), "west": lambda c: -max(x for x, _ in positions(c)),
               "south": lambda c: min(y for _, y in positions(c)), "north": lambda c: -max(y for _, y in positions(c))}[d]
        return min(cands, key=key)
    xy = door_xy(src_map, hk)
    ws = [w for w in rom_map(mid)[1] if (w[0], w[1]) == xy]
    if not ws:
        return None
    dw = rom_map(did)[1]
    if ws[0][3] >= len(dw):
        return None
    tx, ty = dw[ws[0][3]][:2]
    return min(cands, key=lambda c: min(abs(x - tx) + abs(y - ty) for x, y in positions(c)))


added = {}


def dead_end(src_map, hk, dst_map, src_place):
    xy = door_xy(src_map, hk)
    if xy is None:
        return None
    conns, ws = rom_map(ID[dst_map])
    if conns or not ws or any(dm not in (0xFF, ID[src_map]) for _, _, dm, _ in ws):
        return None
    src_ws = [w for w in rom_map(ID[src_map])[1] if (w[0], w[1]) == xy]
    if not src_ws or src_ws[0][3] >= len(ws):
        return None
    bx, by = ws[src_ws[0][3]][:2]
    added[dst_map] = {"map": dst_map, "first": None, "services": set(), "dead_end": True,
                      "exits": {f"Door at ({bx},{by})": {"map": src_map, "open": 0, "to": src_place, "observed": False}}}
    return dst_map


dropped = []
for p, w in world.items():
    for hk, e in list(w["exits"].items()):
        to = observed.get((p, hk))
        if to is None:
            cands = sorted({q for q, ww in world.items() if ww["map"] == e["map"]})
            to = cands[0] if len(cands) == 1 else resolve(w["map"], hk, e["map"], cands) if cands else None
        if to is None and e["map"] in ID and "unknown" not in hk:
            to = dead_end(w["map"], hk, e["map"], p)
        if to is None:
            dropped.append((p, hk, e["map"]))
            del w["exits"][hk]
            continue
        e["to"] = to
        e["observed"] = (p, hk) in observed

for k, v in added.items():
    if k not in world:
        world[k] = v

# gates: an exit on offer at the place's first visit is open from the start; one that appeared at a later visit opens
# at that stage; one missing at a later visit closes there
for p, w in world.items():
    for hk, e in w["exits"].items():
        f = w.get("first")
        st = e.get("stages") or set()
        if f is not None and e["open"] <= f:
            e["open"] = 0
        dst = world.get(e.get("to")) or {}
        e["pass"] = 0 if dst.get("dead_end") else min(x for x in (dst.get("first"), taken_at.get((p, hk))) if x is not None) \
            if (dst.get("first") is not None or (p, hk) in taken_at) else 0
        later = sorted(x for x in w.get("stages", set()) if st and x > max(st) and x not in st)
        if later:
            e["close"] = later[0]

# scripted moves: an exit taken in the recording that put the player somewhere else, not after a blackout
EVENTS = [{"stage": 0, "place": "PALLET_TOWN", "exit": "Leave north", "to": "OAKS_LAB",
           "say": "Professor Oak stops you in the tall grass and walks you to his lab.", "done": "meet_oak"},
          {"stage": 11, "place": "SS_ANNE_1F", "exit": hidden("SS_ANNE_1F", "Go to VERMILION_DOCK")[0], "to": "VERMILION_CITY",
           "say": "The S.S. Anne sets sail as you leave; you watch it go from the dock and walk into the city.", "done": None}]

story = []
for i, m in enumerate(M[:15]):                          # Pallet Town to the fourth badge (no lifts on the way)
    targets = []
    for need in m.get("needs", []):
        targets.append(need["maps"])
    targets.append(m["maps"])
    story.append({"id": m["id"], "goal": m["goal"], "targets": [t for ts in targets for t in ts], "stage": i})

out = {"format": "pokeworld v1",
       "source": "a recorded playthrough of Pokemon Red (place names, exits as the player sees them, where they led); "
                 "no ROM bytes, no graphics",
       "start": "REDS_HOUSE_2F", "story": story, "events": EVENTS, "map_ids": {w["map"]: ID[w["map"]] for w in world.values()},
       "places": {p: {"map": w["map"], "first": w["first"], "services": sorted(w["services"]),
                      "dead_end": bool(w.get("dead_end")),
                      "exits": {k: {"to": e["to"], "open": e["open"], "pass": max(e["open"], e["pass"]),
                                    **({"close": e["close"]} if "close" in e else {}), "observed": e["observed"]}
                                for k, e in sorted(w["exits"].items()) if "unknown" not in k}}
                  for p, w in sorted(world.items())}}
OUT.write_text(json.dumps(out, indent=0))
print(len(world), "places;", sum(len(w["exits"]) for w in world.values()), "exits;", len(dropped), "exits left out")
