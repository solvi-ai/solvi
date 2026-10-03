---
title: solvi plays the Pokémon Red world map
emoji: 🗺️
colorFrom: blue
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: System 1 and System 2 on a game, replayed in your browser
---

# solvi plays the world map of Pokémon Red (runs in your browser)

A player built with [solvi](https://github.com/solvi-ai/solvi) walks the world of Pokémon Red twice — the game's first
fifteen goals, Pallet Town to the fourth badge. One decision is "which exit do I take here?". Each goal names the place
it needs ("Defeat Brock, the Pewter City Gym Leader"), never the way there. Run 1 starts with an empty memory; run 2
starts with what run 1 left. The page replays the two recorded runs with
[Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (Pyodide): no server, and no ROM or emulator anywhere.

| | run 1 (empty memory) | run 2 (run 1's memory) |
|---|---|---|
| decisions (moves) | 183 | 62 — the fewest possible for these goals |
| by System 1 (rules, no search) | 36 | 60 |
| by System 2 (search over the world map) | 147 | 2 — both right after a surprise |
| time inside the decisions | System 1 ≈ 0.3 ms each, System 2 ≈ 2 ms each (recorded on a laptop CPU) | |

- **System 1** is a catalog of two rules: take the exit of the route it remembers to the goal, or the only exit there
  is. It abstains when it remembers no route, when the remembered exit is not on offer, or when the last step surprised
  it (Professor Oak walks you to his lab; the S.S. Anne leaves the dock).
- **System 2** is a search (`solvi.core.slow.search` through `solvi.core.dispatch.SlowPath`) over the world map the player writes as
  it goes (`solvi.core.knowledge.worldmap.WorldMap`): the known way to the goal when the map has one, else every unexplored exit within
  reach, scored by expected value; a hard check keeps only plans whose first step is on offer. An LLM hint is possible
  (`llm=`), and off.
- **The dispatcher** (`solvi.core.dispatch.Dispatcher`) asks System 1 first and System 2 when it abstains. Every decision is
  one hash-chained record; the "Check" tab replays all 245 in the browser, and shows `System.report` for each run.
- **Consolidation**, after each goal, compiles System 1's routes from the map's confirmed claims.

## The data

`data/world.json` is the world as a real playthrough saw it: 190 places (a map, or a part of one that walls split),
447 exits as a player sees them ("Leave north", "Door at (12,11)") and where they led, the stage of the story from
which each is on offer and can be passed, two scripted moves and the fifteen goals. Place names, tile coordinates and
goal texts in our own words: no ROM bytes, no graphics, no game text. `tools/extract_world.py` is how it was made from
a harness recording (not needed to run anything). Simplifications: a place counts as reachable from the stage at which
the recorded player first reached it; buildings the recording never entered are included only where the game's map
tables show them to be dead ends; walking, battles and menus inside a place are not part of this showcase.

`data/runs/` is the recording the page replays: the stored decisions (`run1.decisions.jsonl.gz`,
`run2.decisions.jsonl.gz`: solvi JSONL stores, gzipped), the move logs, the world map after each run (its journal),
System 1's routes and every consolidation (`memory.json`) and the numbers (`summary.json`). Recompute them with the
example in the solvi repository:

```bash
uv run python examples/23_pokemon_world_map.py            # replay the bundled recording, print the numbers
uv run python examples/23_pokemon_world_map.py --live     # play both runs again: the same decisions
uv run python examples/23_pokemon_world_map.py --rom ~/roms/pokemon_red.gb   # check the world against your own ROM first
cd spaces/pokemon && python -m pokeworld.record           # re-record data/runs (after changing the player)
```

## Layout

- `index.html`: Gradio-Lite 5.45 (with the same package "time machine" as the other solvi Spaces), the requirement
  `solvi==0.9.0` and every file below mounted by URL.
- `app.py`: the UI — Replay (the map the player knows after each decision, who decided and why, System 2's plans with
  their values), World map, Numbers, Check and report, About.
- `pokeworld/world.py` (the recorded world, the ROM check), `pokeworld/agent.py` (System 1, System 2, the dispatcher,
  memory and consolidation, a run), `pokeworld/record.py` (record, replay, numbers), `pokeworld/render.py` (SVG).

## Run locally

```bash
cd spaces/pokemon
PYTHONPATH=../../src python app.py       # with gradio installed; or serve the folder (python -m http.server) for the
                                          # in-browser version, which needs solvi 0.9.0 on PyPI
```

Pokémon is a trademark of Nintendo, Creatures and GAME FREAK. This project is not affiliated with them and includes no
part of the game.
