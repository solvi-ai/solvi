"""The environment: the world of Pokémon Red as places and exits, recorded from a real playthrough.

A place is a map (PALLET_TOWN, MT_MOON_1F), or a part of a map that walls split (ROUTE_4 (west) and ROUTE_4 (east), with
Mt. Moon between them). An exit is what a player sees on the screen — "Leave north", "Door at (12,11)" — and where it
leads is found out by taking it. `data/world.json` holds, for every place a recorded playthrough reached:

- the exits on offer there and where each led (map names; door coordinates are tile positions);
- from which stage of the story each exit is on offer, and from which stage it can be passed (before that a try leaves
  the player where they are: water without Surf, a guard, a ledge) — taken from when the recorded player first had it
  on offer and first reached the place behind it;
- two scripted moves of the story (Professor Oak walks you to his lab; the S.S. Anne leaves);
- the story: the first fifteen goals of the game (Pallet Town to the fourth badge), each naming the places it needs.

No ROM bytes, no graphics, no game text beyond place names: coordinates, map names and the goals in our own words.
Places the recording never entered are left out, except buildings the ROM's map tables show to be dead ends (every
door of them leads back); exits into places left out are not on offer.

With your own ROM (`World.check_rom(path)`), every recorded exit is checked against the ROM's map tables (its
connections and doors) before a run; nothing of the ROM is copied or kept."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
ROM_SHA1 = "ea9bcae617fdf159b045185467ae58b2e4a48b9a"      # Pokémon Red (UE): the version the playthrough was recorded on
_HEADER_BANKS, _HEADER_POINTERS = 0xC23D, 0x01AE           # MapHeaderBanks, MapHeaderPointers in that version
_DOOR = re.compile(r"^Door at \((\d+),(\d+)\)$")
_EDGE = re.compile(r"^Leave (north|south|east|west)\b")


def place_map(place):
    """The map a place is on: "ROUTE_4 (west)" → "ROUTE_4"."""
    return place.split(" (")[0]


@dataclass
class Step:
    """What taking an exit did: where the player is now, whether the try was blocked (they stayed), and the scripted
    event that moved them instead, if any ({"say", "done", ...})."""
    to: str
    blocked: bool = False
    event: dict | None = None


class World:
    """The recorded world (see the module docs). on_offer(place, stage) → the exits a player sees there at that stage of
    the story; take(place, exit, stage, fired) → a Step."""

    def __init__(self, data):
        if data.get("format") != "pokeworld v1":
            raise ValueError("not a pokeworld v1 world file")
        self.data = data
        self.places = data["places"]
        self.story = data["story"]
        self.events = data["events"]
        self.start = data["start"]
        self.map_ids = data["map_ids"]

    @classmethod
    def load(cls, path=None):
        return cls(json.loads(Path(path or DATA / "world.json").read_text(encoding="utf-8")))

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()[:16]

    def on_offer(self, place, stage):
        """The exits on offer in `place` at `stage`, sorted (one per line in a decision's input)."""
        out = []
        for key, e in self.places[place]["exits"].items():
            if e["open"] <= stage and ("close" not in e or stage < e["close"]):
                out.append(key)
        return sorted(out)

    def services(self, place):
        return list(self.places[place].get("services", []))

    def take(self, place, key, stage, fired):
        """Take exit `key` in `place` at `stage`. fired: the set of scripted events already seen (updated here)."""
        if key not in self.on_offer(place, stage):
            raise ValueError(f"{key!r} is not on offer in {place} at stage {stage}")
        for i, ev in enumerate(self.events):
            if ev["stage"] == stage and ev["place"] == place and ev["exit"] == key and i not in fired:
                fired.add(i)
                return Step(ev["to"], event=ev)
        e = self.places[place]["exits"][key]
        if stage < e["pass"]:
            return Step(place, blocked=True)
        return Step(e["to"])

    def targets(self, maps):
        """The places on the maps an objective names."""
        maps = set(maps)
        return sorted(p for p in self.places if place_map(p) in maps)

    # --- checking the recording against a ROM
    def check_rom(self, path):
        """Read YOUR ROM's map tables (connections and doors of every map in the world) and compare every recorded exit
        with them → a list of mismatches (empty: the recording is what the ROM says). Only the version the playthrough
        was recorded on is read (its SHA-1 is checked); nothing is copied or kept."""
        b = Path(path).read_bytes()
        sha = hashlib.sha1(b).hexdigest()       # noqa: S324 — the identity of a ROM dump, not a security hash
        if sha != ROM_SHA1:
            raise ValueError(f"this ROM's SHA-1 is {sha}; the recording was made on Pokémon Red (UE), SHA-1 {ROM_SHA1}")

        def u16(a):
            return b[a] | (b[a + 1] << 8)

        def flat(bank, ptr):
            return ptr if ptr < 0x4000 else bank * 0x4000 + (ptr - 0x4000)

        def tables(mid):
            bank = b[_HEADER_BANKS + mid]
            h = flat(bank, u16(_HEADER_POINTERS + mid * 2))
            flags, h = b[h + 9], h + 10
            conns = {}
            for bit, d in ((3, "north"), (2, "south"), (1, "west"), (0, "east")):
                if flags & (1 << bit):
                    conns[d] = b[h]
                    h += 11
            o = flat(bank, u16(h)) + 1
            n, o = b[o], o + 1
            doors = {}
            for i in range(n):
                doors.setdefault((b[o + 4 * i + 1], b[o + 4 * i]), b[o + 4 * i + 3])
            return conns, doors

        bad = []
        cache = {}
        for place, p in sorted(self.places.items()):
            mid = self.map_ids[p["map"]]
            conns, doors = cache.setdefault(mid, tables(mid))
            for key, e in sorted(p["exits"].items()):
                want = self.map_ids[place_map(e["to"])]
                m, d = _DOOR.match(key), _EDGE.match(key)
                if m:
                    got = doors.get((int(m.group(1)), int(m.group(2))))
                    if got is None:
                        bad.append(f"{place}: no door at {m.group(1)},{m.group(2)} in the ROM")
                    elif got not in (want, 0xFF):            # 0xFF: the door leads back to where you came from
                        bad.append(f"{place}: {key} leads to map {got} in the ROM, the recording says {e['to']}")
                elif d:
                    got = conns.get(d.group(1))
                    if got != want:
                        bad.append(f"{place}: {key} — the ROM connects {d.group(1)} to map {got}, the recording says {e['to']}")
                else:
                    bad.append(f"{place}: {key} is not an exit the ROM's tables describe")
        return bad


def place_type(place):
    """What kind of place a name says it is (the agent knows the names of places it has been to, and the names a goal
    mentions)."""
    n = place_map(place)
    if "POKECENTER" in n:
        return "center"
    if "MART" in n and "MARTS" not in n:
        return "mart"
    if n.endswith("_GYM"):
        return "gym"
    if n.endswith(("_CITY", "_TOWN", "_ISLAND", "_PLATEAU")):
        return "town"
    if n.startswith("ROUTE_"):
        return "gate" if "GATE" in n else "route"
    if "FOREST" in n:
        return "gate" if "GATE" in n else "forest"
    if any(c in n for c in ("MT_MOON", "ROCK_TUNNEL", "CAVE", "UNDERGROUND", "VICTORY_ROAD", "SEAFOAM")):
        return "cave"
    if n.startswith("SS_ANNE") or "DOCK" in n:
        return "ship"
    return "building"
