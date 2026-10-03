"""System 1 and System 2 on a game: solvi walks the world of Pokémon Red twice, and the second time almost every
decision is a fast one.

The game is the world map of Pokémon Red — 190 places (towns, routes, buildings, cave floors) and 452 exits, recorded
from a real playthrough as place names, exits as a player sees them ("Leave north", "Door at (12,11)") and where they
led; no ROM bytes, no graphics. The story is the game's first fifteen goals, Pallet Town to the fourth badge; each goal
names the place it needs ("Defeat Brock, the Pewter City Gym Leader" → PEWTER_GYM), never the way there. One decision
is "which exit do I take here?".

- **System 1** is a catalog of two rules: take the exit of the route it remembers to the goal, or the only exit there
  is. No search. It abstains when it remembers no route, when the remembered exit is not on offer, or when the last
  step surprised it (it led somewhere else, or nowhere).
- **System 2** is a search over the world map the player builds as it goes (`solvi.worldmap.WorldMap`): the known way
  to the goal when the map has one, else every unexplored exit within reach, scored by expected value (the kind of
  place the goal needs behind it, the goal's direction, the steps to get there). An LLM can add a hint (off by
  default: `--llm URL MODEL`, key in SOLVI_LLM_KEY).
- **The dispatcher** (`solvi.dispatch.Dispatcher`) asks System 1 first and System 2 when System 1 abstains; every
  decision is one hash-chained record that replays without the game.
- **Consolidation**, after each goal, compiles System 1's routes from the map's confirmed claims (a route to every place
  a goal has named and the player has reached): what System 2 found by deliberating becomes what System 1 does at once.

Run 1 starts with an empty memory, run 2 with what run 1 left.

    uv run python examples/23_pokemon_world_map.py                # replay the bundled recording (no game needed)
    uv run python examples/23_pokemon_world_map.py --live         # play both runs again on the recorded world
    uv run python examples/23_pokemon_world_map.py --rom PATH     # check the recording against your own ROM, then play

The code of the player is in spaces/pokemon/pokeworld/ (agent.py: the two systems, the memory, consolidation;
world.py: the recorded world; record.py: recording, replay, numbers); the Space in spaces/pokemon/ shows the replays,
the world map and the report. Pokémon is a trademark of Nintendo, Creatures and GAME FREAK; this is not affiliated
with them, and no part of the game is included."""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "spaces" / "pokemon"))

from pokeworld import record as rec  # noqa: E402
from pokeworld.agent import dispatcher, system1  # noqa: E402
from pokeworld.world import World  # noqa: E402


def table(nums, story):
    print(f"  {'goal':<16} {'run 1: moves':>12} {'S1':>4} {'S2':>4}   {'run 2: moves':>12} {'S1':>4} {'S2':>4}   shortest")
    for row in nums["objectives"]:
        a, b = row["run1"], row["run2"]
        print(f"  {row['objective']:<16} {a['moves']:>12} {a['s1']:>4} {a['s2']:>4}   {b['moves']:>12} {b['s1']:>4} "
              f"{b['s2']:>4}   {row['shortest']:>8}")
    r1, r2 = nums["runs"]["run1"], nums["runs"]["run2"]
    short = sum(r["shortest"] for r in nums["objectives"])
    print(f"  {'all goals':<16} {r1['decisions']:>12} {r1['s1']:>4} {r1['s2']:>4}   {r2['decisions']:>12} {r2['s1']:>4} "
          f"{r2['s2']:>4}   {short:>8}")
    for name, r in (("run 1", r1), ("run 2", r2)):
        print(f"  {name}: System 1 {r['s1']} decisions, {r['ms']['s1']:.0f} ms ({r['ms_per_decision']['s1'] or 0:.2f} ms "
              f"each); System 2 {r['s2']} decisions, {r['ms']['s2']:.0f} ms ({r['ms_per_decision']['s2'] or 0:.2f} ms each); "
              f"{r['blocked']} tries that led nowhere, {r['surprises']} surprises, {r['events']} story events")


def examples(out):
    """One decision of each kind, read back from the stored records."""
    d1 = dispatcher(storage=rec.open_store(out, "run1")).stored()
    d2 = dispatcher(storage=rec.open_store(out, "run2")).stored()
    m1, m2 = rec._moves(Path(out) / "run1.moves.jsonl"), rec._moves(Path(out) / "run2.moves.jsonl")
    s2 = next(i for i, m in enumerate(m1) if m["by"] == "s2" and m["objective"] == "mt_moon")
    run = d1[s2].s2.record
    kept = ", ".join(f"{c['place']} | {c['exit']} ({v:.3f})" for c, v in run.kept[:1])
    v = d1[s2].s1["exit"]
    print(f"  System 2, run 1 #{m1[s2]['n']} at {m1[s2]['here']} (goal: {m1[s2]['objective']}): System 1 abstained "
          f"({d1[s2].s1.values.get('stands_back')}); System 2 asked {run.asked} plans through its checks, best {kept}; "
          f"took {m1[s2]['exit']!r} → {m1[s2]['to']}  [System 1 status: {v.status}]")
    f = next(i for i, m in enumerate(m2) if m["by"] == "s1" and m["objective"] == "mt_moon")
    print(f"  System 1, run 2 #{m2[f]['n']} at {m2[f]['here']}: {m2[f]['exit']!r} → {m2[f]['to']} — "
          f"{m2[f]['why']}, {d2[f].cost['total'].ms:.2f} ms, no search")
    for i, m in enumerate(m2):
        if m["surprise"]:
            nxt = m2[i + 1]
            print(f"  Surprise, run 2 #{m['n']}: {m['surprise']}")
            print(f"     → #{nxt['n']} at {nxt['here']}: System 1 stands back ({nxt['why'][:60]}...), "
                  f"System 2 decides: {nxt['exit']!r} → {nxt['to']}")
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--live", action="store_true", help="play both runs again on the recorded world")
    ap.add_argument("--rom", help="your own Pokémon Red ROM: the recording is checked against its map tables, then played")
    ap.add_argument("--out", help="where a live run writes its recording (default: a temporary folder)")
    ap.add_argument("--llm", nargs=2, metavar=("URL", "MODEL"), help="System 2 asks this LLM for a hint (key: SOLVI_LLM_KEY)")
    a = ap.parse_args()

    world = World.load()
    print("solvi plays the world map of Pokémon Red: System 1 + System 2\n")
    print(f"World: {len(world.places)} places, {sum(len(p['exits']) for p in world.places.values())} exits, recorded from a "
          f"real playthrough (place names and exits; no ROM bytes, no graphics).")
    print(f"Story: {len(world.story)} goals, from '{world.story[0]['id']}' to '{world.story[-1]['id']}'.\n")

    llm = None
    if a.llm:
        from solvi.llm import llm as make_llm
        llm = make_llm(a.llm[0], a.llm[1], api_key=os.environ.get("SOLVI_LLM_KEY"))
    out = rec.RUNS
    if a.rom:
        bad = world.check_rom(a.rom)
        print(f"Your ROM: every recorded exit checked against its map tables — {len(bad)} difference(s).")
        for b in bad[:10]:
            print("   ", b)
        if bad:
            sys.exit(1)
    if a.rom or a.live or a.llm:
        out = Path(a.out or tempfile.mkdtemp(prefix="pokeworld-"))
        print(f"Playing both runs live (recording to {out}) ...")
        rec.record(out, world, llm=llm)
        if llm is None:
            for name in rec.NAMES:
                diff = rec.same_as_recorded(rec._moves(out / f"{name}.moves.jsonl"), run=name)
                print(f"  {name}: {'the same decisions as the bundled recording' if diff is None else f'differs: {diff}'}")
        print()
    else:
        print("Replaying the bundled recording (spaces/pokemon/data/runs; --live plays it again).\n")

    nums = rec.numbers(out, world)
    print("Decisions per goal (a move = one decision: which exit to take; 'shortest' = the fewest moves with the whole map "
          "known):")
    table(nums, world.story)
    r1, r2 = nums["runs"]["run1"], nums["runs"]["run2"]
    print(f"\n  → run 2 needs {r2['s2']} slow decisions instead of {r1['s2']}, and {r2['decisions']} moves instead of "
          f"{r1['decisions']}.\n")

    print("What the decisions look like:")
    examples(out)
    mem = __import__("json").loads((Path(out) / "memory.json").read_text(encoding="utf-8"))
    c = mem["consolidations"]
    k = len(world.story)
    print(f"\nConsolidation after each goal: System 1's routes {c[k - 1]['routes']} after run 1 (from "
          f"{c[0]['routes']} after the first goal); run 2 added {sum(x['added'] for x in c[k:])} and changed "
          f"{sum(x['changed'] for x in c[k:])}.")

    chk = rec.replay(out, llm=llm)
    print(f"\nReplay without the game: {chk['replayed']} of {chk['decisions']} stored decisions replay "
          f"(System 1's trace, the dispatch, System 2's search and its winner); world maps verify: "
          f"{all(m['verify'] for m in chk['maps'].values())} ({chk['maps']['run2']['journal']} journal entries); move "
          f"logs agree: {chk['logs']}; numbers agree: {chk['numbers']}.")
    for m in chk["mismatches"][:5]:
        print("   ", m)

    print("\nSystem report of run 2 (System.report over its stored decisions):\n")
    print(system1().report(store=rec.open_store(out, "run2")))
    if not chk["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
