---
title: solvi realms
emoji: 🏰
colorFrom: purple
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Endless strategy game with solvi AI, in your browser
---

# solvi realms (runs in your browser)

An endless turn-based strategy game (a small 4X: map, resources, cities, units) whose AI factions are
[solvi](https://github.com/solvi-ai/solvi) decision systems. There is no win condition. The point is to watch the game
run indefinitely and show that it stays healthy. Everything runs in the visitor's browser with
[Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (Pyodide): no server, no threads, no network after load.

## The game

- **Map**: a seeded 28×18 grid (8 neighbours) of plains, forest, hills, mountains, water and desert, with food, wood,
  stone and gold deposits. A worked deposit slowly depletes. An unworked one regenerates.
- **Factions**: builder, expansionist, warmonger, trader and one **adaptive** faction that learns while it plays. Cities
  work the tiles around them, grow, and build units (worker, settler, warrior, archer, caravan) and buildings (farm, lumber
  mill, mine, walls, market), or turn their production into gold. Upkeep is paid in gold, population eats food. Combat is
  simple and seeded (win odds = attack / (attack + defence), with fortification, terrain, walls and militia). Diplomacy is
  minimal: war and peace, each with its reason. A peace brings a 20-turn truce.
- **Endless**: when fewer than 3 factions are alive, a new one lands on free land. If no land is free, a far-away city of
  the largest empire rebels. Big empires also see random rebellions. Droughts, plagues, gold rushes, discoveries and
  harvests keep the world moving. Stockpiles above a storage capacity spoil, and a rich treasury rushes production. Every
  log and buffer is a ring buffer, so the saved state stays bounded.

## The factions are solvi systems

One catalog of small Python functions per personality: the functions are shared, and the personality lives in the
weights. Six questions:

| question | asked | answers |
|---|---|---|
| `build` | per city, when its queue is empty | worker, settler, warrior, archer, caravan, farm, lumber_mill, mine, walls, market, gold |
| `order_military` | per warrior/archer, every few turns | attack, defend, fortify, move, disband |
| `order_settler` | per settler | settle, move, fortify |
| `order_worker` | per worker | work, move, fortify, disband |
| `order_caravan` | per caravan | trade, move, disband |
| `stance` | per neighbour, every 5 turns | war, peace |

Computed facts include yields, food surplus, growth time, treasury after upkeep, a threat ratio (enemy strength near the
city against its defence), affordable options, scored settle sites, attack odds, the most threatened own city, strength
ratio and border tension. **Hard checks** (a failed one forces the answer, whatever the rule says):

| hard check | forces | rule |
|---|---|---|
| `treasury_ok` | build → gold | the treasury stays >= 0 after next turn's upkeep |
| `not_under_siege` | build → archer | a besieged city skips economic planning |
| `keeps_capital_defender` | order_military → fortify | the capital always keeps a defender |
| `settles_away_from_enemies` | order_settler → move | never found a city within 4 tiles of an enemy city |
| `worker_upkeep_ok` | order_worker → disband | a worker goes when the treasury cannot carry it |
| `war_needs_strength` | stance → peace | never declare or keep a war below 80% of the enemy's strength |

The **strategist's plan** matters. Each question's flow takes only the parts it needs: a worker's order runs 3 of the
43 catalog parts, a city's build 17. Hard checks run first. When a city is under siege or broke, **early exit** skips the
whole economic plan (yields, growth, affordable options, needs, scores), and the decision card lists the skipped steps.

**The adaptive faction** has no rule for `build`. Its answer head is a `System.fit_fast` ridge head, bootstrapped on 80
synthetic city states labeled by a balanced rule. Every build decision is judged 12 turns later (city value = 2 × pop +
yields + garrison; a lost city counts −25). Decisions whose reward reaches the rolling 60th percentile are taught with
`System.teach`, a rank-one update of about 0.2 ms. It explores a random affordable option with probability 12%, decaying
to 3%, and refits from the last 400 good examples every 1 000 turns.

## In the browser

- The map is SVG. Click a city or unit to see its last solvi decision: the answer, `why`, the strategist's plan (steps
  and the reason each was taken), which hard checks passed or fired, the steps skipped by early exit, every computed fact,
  the `init_state` the engine passed in, and a live trace replay.
- Controls: next turn, play 10 or 100, **▶ Endless** (a `gr.Timer` that plays 1, 5 or 20 turns per tick), pause, new
  world by seed.
- **Health panel**: the same invariants as the headless test (below), checked live every 25 turns, with sparklines for
  decision time, turn time, factions alive, population, vetoes, state size and the adaptive faction's score.
- The event log records wars and peace with their reasons, captures, rebellions, new factions, events and vetoes.
- Take over a faction (you choose its builds and war/peace while solvi keeps advising). Save or load the game as JSON.

In Chromium (Pyodide): next turn 113 ms; 100 turns in 1.6 s (~15 ms/turn); endless auto-play ~25 turns/s. Live health at turn 591: 0 exceptions, decision p99 ×0.45 of the start, state 200→229 KB, min 3 factions alive, 376/376 replays OK, 0 abstentions in 20 477 decisions.

## Endless-game test

`sim.py` runs the game headless (plain CPython, no Gradio) and records a checkpoint every 500 turns. The invariants were
fixed before the runs (`realms/health.py`):

| id | invariant | test |
|---|---|---|
| I1 | no exceptions | engine exceptions = 0 and no solvi part raised |
| I2 | decision time flat | p99 decision time, last 10% of checkpoints <= 2 × first 10% |
| I3 | turn time flat | p99 turn time, same test |
| I4 | state bounded | serialized state: max over the last half <= 1.5 × max over the first half, and < 1 MB |
| I5 | memory bounded | tracemalloc (or process RSS): max over the last half <= 1.5 × max over the first half |
| I6 | >= 2 factions alive | at every turn |
| I7 | economy not degenerate | population, gold, wood, stone, deposit richness: over the last half no collapse (min > 5% of the mean) and no runaway (linear trend < 50% of the mean) |
| I8 | treasury never negative | no faction ends a turn below 0 gold |
| I9 | trace replay OK | 2% of all decisions, sampled at random, replayed with `trace.replay` |
| I10 | no abstentions | every question answered (ok or forced) |
| I11 | save/load exact | every 5 000 turns: save, reload, play 20 turns on the copy and on the original, compare state hashes |

Four long endless runs: 100 000 turns × 2 seeds, 20 000 turns × 2 seeds with memory tracing (tracemalloc). **All eleven invariants held on every run**: 0 exceptions, decision time flat (p99 first vs last 10% of turns within 2×), saved state bounded (max 358 KB), memory bounded, never fewer than 3 factions alive, no degenerate economy, and every sampled trace replay OK (241 314 of 241 314).

| seed | turns | wall time | decision ms p50 first/last 10% | p99 first/last | max state | min alive | factions spawned | decisions | replay OK | held | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100,000 | 45 min | 0.41 / 0.57 | 0.87 / 0.89 | 358 KB | 3 | 37 | 4,820,021 | 96445/96445 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 2 | 100,000 | 47 min | 0.32 / 0.32 | 0.85 / 0.84 | 335 KB | 4 | 9 | 5,307,819 | 106113/106113 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 3 | 20,000 | 24 min | 0.84 / 0.86 | 1.88 / 1.81 | 324 KB | 3 | 20 | 1,083,497 | 21617/21617 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 4 | 20,000 | 20 min | 0.85 / 1.25 | 1.83 / 1.88 | 342 KB | 3 | 212 | 859,430 | 17139/17139 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |

![health](results/health.png)
![learning](results/learning.png)

Honest notes: the adaptive faction (learning with `fit_fast` + `teach`) does not beat the rule-based factions — its score is about 0.1–0.3 of their mean; the warmonger tends to dominate; seed 4 saw many rebellions (212 factions spawned) and its median decision time rose from 0.85 to 1.25 ms, still under the 2× limit.

![health](results/health.png)

![learning](results/learning.png)

Raw checkpoints: `results/endless_<seed>.json`. Charts: `results/health.{png,svg}`, `results/learning.{png,svg}`.

## Layout

- `index.html`: loads `@gradio/lite@5.45.0` from jsDelivr, requires `solvi>=0.2.1`, and mounts `app.py` and `realms/*.py`.
  It keeps the arcade's PyPI "time machine" (the simple index is filtered to files uploaded before the Gradio-Lite release,
  solvi exempt), without which micropip cannot resolve gradio 5.45.
- `app.py`: the Gradio 5 UI.
- `realms/world.py` (map, distance fields), `realms/econ.py` (rules), `realms/brains.py` (catalogs, questions, hard
  checks, learner), `realms/engine.py` (turn loop, combat, events, spawning, save/load), `realms/health.py` (invariants,
  checkpoints), `realms/render.py` (SVG map, cards, sparklines).
- `sim.py`: the headless long-run test and the charts.

## Run locally

```bash
cd solvi/spaces/realms
python -m http.server 8080                 # the browser app: open http://localhost:8080
# headless (from the solvi repo, which has solvi installed):
OPENBLAS_NUM_THREADS=1 uv run python spaces/realms/sim.py --seed 1 --turns 100000 --no-tracemalloc
uv run --with matplotlib python spaces/realms/sim.py --charts
uv run python spaces/realms/sim.py --table
```
