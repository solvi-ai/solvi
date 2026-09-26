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
47 catalog parts, a city's build 17. Hard checks run first. When a city is under siege or broke, **early exit** skips the
whole economic plan (yields, growth, affordable options, needs, scores), and the decision card lists the skipped steps.

**The adaptive faction** has no rule for `build`, `order_military` or `stance`. Each of the three is answered by a
learned **value head** (`realms/adaptive.py`): a solvi `FastHead` subclass that keeps, per answer option, a ridge
regression of the reward on the question's computed facts (city: growth, infrastructure, military need, threat, income,
cities, own strength against the others and against the weakest neighbour, free unit slots; unit: attack odds, target,
defence need, capital, war; stance: strength ratio, tension, war length, other wars, cities, border, grievance, trade).

- **Decision**: `System.ask` as for every faction. Hard checks run first and force the answer; otherwise the head picks
  the allowed option (affordable builds; attack only with a target, defend only for a threatened city, disband only
  surplus troops the income cannot carry) with the highest value + a small UCB bonus `0.005·sqrt(xᵀA⁻¹x)`. The `why`
  lists the facts that pushed the chosen option above the others (per-fact contributions of the learned weights), e.g.
  `stance = peace — their_cities = 1 (-0.06), at_war = True (+0.03), other_wars = 0 (+0.02), my_cities = 4 (-0.02)`.
  1% ε-exploration on builds.
- **Hard checks for the learned questions look only at the state** (the capital's last defender always stays; no war
  below 80% strength), so they veto whatever the head prefers, not a rule's preference.
- **Data**: every non-forced decision of **every** faction (its own, plus a sample of the others' — observational data)
  is kept in a bounded ring as its feature vector. 20 turns later (25 for stance) its reward is the deciding faction's
  log-score gain (score = population + 3 × cities) minus that faction's running mean gain (an advantage, so a faction
  that grows anyway does not make all its choices look good). Rewards update the ridge statistics; weights are re-solved
  every 25 turns with a slow decay (memory ~5 000 turns).
- **Bootstrap**: 200 synthetic states per question labelled by the balanced rule (its choice worth +0.05). The
  ablation "frozen after bootstrap" keeps exactly this prior for the whole game.

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

Five endless runs with the learned adaptive faction: 20 000 turns × seeds 1–4 with memory tracing (tracemalloc, the
pre-registered evaluation below) and one 100 000-turn run (seed 5, process RSS). **All eleven invariants held on every run**: 0 exceptions, decision and turn time flat (p99 first vs last 10% within 2×; the runs shared the machine with up to five other simulations, so absolute times are inflated early on), saved state bounded (max 295 KB), memory bounded, never fewer than 3 factions alive, no degenerate economy, every save/load round-trip identical, and every sampled trace replay OK (223 155 of 223 155). On the 100 000-turn run the adaptive faction's share went from 0.58 (first 20%) to 0.87 (last 20%), peak 1.14.

| seed | turns | wall time | decision ms p50 first/last 10% | p99 first/last | max state | min alive | factions spawned | decisions | replay OK | held | failed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 20,000 | 42 min | 1.53 / 0.99 | 3.17 / 2.34 | 282 KB | 3 | 21 | 1,187,689 | 23749/23749 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 2 | 20,000 | 36 min | 1.20 / 1.54 | 3.14 / 2.98 | 264 KB | 3 | 25 | 996,110 | 19759/19759 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 3 | 20,000 | 37 min | 1.11 / 1.33 | 2.95 / 3.11 | 276 KB | 3 | 25 | 1,097,239 | 21881/21881 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 4 | 20,000 | 39 min | 1.12 / 1.14 | 3.00 / 2.80 | 273 KB | 3 | 87 | 1,126,624 | 22506/22506 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |
| 5 | 100,000 | 81 min | 0.67 / 0.44 | 1.58 / 0.91 | 295 KB | 3 | 16 | 6,792,282 | 135260/135260 | I1, I2, I3, I4, I5, I6, I7, I8, I9, I10, I11 | — |

![health](results/health.png)

### Does the adaptive faction learn to beat the rules? (pre-registered test)

The criteria were written into `realms/adaptive.py` before the learner was changed and were not edited afterwards.
"share" = the adaptive faction's score / the mean score of the other factions (score = population + 3 × cities),
averaged over the first or last 20% of the 500-turn checkpoints; the other four factions and the maps are unchanged.

| id | criterion | result |
|---|---|---|
| A1 | share over the last 20% ≥ 1.0 in at least 3 of seeds 1–4 (20 000 turns) | **FAIL** — 0 of 4 (0.21, 0.48, 0.26, 0.12) |
| A2 | share last 20% > first 20% in all 4 seeds | **FAIL** — 2 of 4 (seeds 2 and 3 improve; 1 and 4 do not) |
| A3 | all 11 health invariants hold | **PASS** — all 11 on all 4 runs, and on the 100 000-turn run |
| A4 | every adaptive decision has a solvi `why` with the learned contributions; hard checks still veto it | **PASS** — audited on 2 000 turns: 52 377 answered decisions, every `why` names learned facts with signed contributions; 2 673 forced by a hard check, 2 140 of them against the head's own preference; 0 violations (`results/a4_check_seed1.json`) |

| seed | share first 20% | share last 20% | A1 ≥ 1.0 | A2 last > first | A3 invariants | frozen after bootstrap (first → last 20%) | previous fit_fast learner (first → last 20%) |
|---|---|---|---|---|---|---|---|
| 1 | 3.05 | **0.21** | no | no | all 11 held | 0.27 → 0.23 | 0.24 → 0.15 |
| 2 | 0.14 | **0.48** | no | yes | all 11 held | 0.96 → 0.61 | 0.24 → 0.12 |
| 3 | 0.11 | **0.26** | no | yes | all 11 held | 0.27 → 0.14 | 0.20 → 0.15 |
| 4 | 0.18 | **0.12** | no | no | all 11 held | 0.66 → 0.14 | 0.12 → 0.10 |

![learning](results/learning.png)

What the numbers say, plainly:

- **Better than before, not better than the rules.** The previous learner (a `fit_fast` build head taught from its
  own top-40% outcomes) sat at 0.10–0.24 share on every seed. The value heads reach 0.12–0.48 at the end of the four
  evaluation runs, and on seed 1 the adaptive faction led the map for most of the first ~5 500 turns (share 1–12) before
  wars reduced it to one city (about 0.2 from then on). It does not beat the rule-based factions on any of the four seeds.
- **Learning is not shown to beat the frozen bootstrap.** With the heads frozen after the bootstrap the shares at the
  end are 0.23, 0.61, 0.14, 0.17 — about the same as with learning (0.21, 0.48, 0.26, 0.12). On eight other seeds
  (5–12, 5 000 turns, used while developing) the learner improved from the first to the last 20% in 9 of 16 runs, the
  frozen heads in 2 of 8, but the final share was ≥ 1 in 5 of 16 learner runs and 3 of 8 frozen runs. Most of the jump
  over the old learner comes from handing unit orders and war/peace to value heads at all, not from what they learn
  in-game.
- **Outcomes are chaotic.** The same learner with a different exploration seed moved the median final share on seeds
  5–12 from 1.53 to 0.46. Every learner variant tried (own data only, advantage baseline, pairwise stance reward, danger
  features, more or less optimism, faster or slower forgetting) landed in the same band: roughly a third of worlds
  dominated (share 2–15), the rest stuck at 0.1–0.5 — usually after the warmonger conquers the adaptive faction's
  core and the respawned one-city faction cannot grow in a full world. A first full evaluation attempt with an earlier
  variant (no advantage baseline, more optimism, faster forgetting) gave seed 1: 2.07 → 14.19 and seed 2: 1.63 → 0.10 (collapse at turn
  14 500); it was stopped when seed 2 failed, and is kept in `results/attempt1/`.
- **Why it stays hard**: the reward is the whole faction's score 20 turns later, so one build or one order barely
  moves it; the signal that walls and archers prevent a conquest arrives rarely and late. Faction-level credit
  assignment is the weak point, not speed or memory (decisions stay ~1 ms, the saved state < 300 KB).

Raw checkpoints: `results/endless_<seed>.json` (evaluation and 100 000-turn run), `results/ablation/frozen_<seed>.json`
(frozen ablation), `results/baseline_fitfast/` (the previous learner's runs), `results/attempt1/`. Charts:
`results/health.{png,svg}`, `results/learning.{png,svg}` (solid: learner, dashed: frozen, dotted: previous learner).

## Layout

- `index.html`: loads `@gradio/lite@5.45.0` from jsDelivr, requires `solvi>=0.2.1`, and mounts `app.py` and `realms/*.py`.
  It keeps the arcade's PyPI "time machine" (the simple index is filtered to files uploaded before the Gradio-Lite release,
  solvi exempt), without which micropip cannot resolve gradio 5.45.
- `app.py`: the Gradio 5 UI.
- `realms/world.py` (map, distance fields), `realms/econ.py` (rules), `realms/brains.py` (catalogs, questions, hard
  checks), `realms/adaptive.py` (the adaptive faction's value heads, pre-registered criteria), `realms/engine.py` (turn loop, combat, events, spawning, save/load), `realms/health.py` (invariants,
  checkpoints), `realms/render.py` (SVG map, cards, sparklines).
- `sim.py`: the headless long-run test and the charts.

## Run locally

```bash
cd solvi/spaces/realms
python -m http.server 8080                 # the browser app: open http://localhost:8080
# headless (from the solvi repo, which has solvi installed):
OPENBLAS_NUM_THREADS=1 uv run python spaces/realms/sim.py --seed 1 --turns 100000 --no-tracemalloc
uv run --with matplotlib python spaces/realms/sim.py --charts
uv run python spaces/realms/sim.py --table          # health table + criteria A1-A3 + ablation
# ablation: the adaptive heads frozen after the bootstrap
uv run python spaces/realms/sim.py --seed 1 --turns 20000 --frozen --out spaces/realms/results/ablation/frozen_1.json
```
