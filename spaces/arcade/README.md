---
title: solvi arcade
emoji: 🕹️
colorFrom: purple
colorTo: yellow
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Game agents from Python rules that explain every move
---

# solvi arcade

Game agents built with [solvi](https://github.com/solvi-ai/solvi) from small Python functions, a few rules and hard
checks. They play well, and every move comes with its reason: the rule's inputs, the facts computed this turn, and which
hard check fired, if any. Runs on CPU only, with no models.

| Tab | What you see |
|---|---|
| ❌⭕ Tic-tac-toe | The catalog from `examples/05_tic_tac_toe.py`, with an extra `reason` rule that names the move (win, block, fork, forcing, center ...). It keeps a "you can't beat it" counter and has an "explain the position" view. |
| 👻 Ghost maze | Pac is a catalog of BFS functions plus one scoring rule. The hard check `greedy_is_safe` blocks any step into a ghost's reach while a safer move exists. You can turn the check off, take over with the arrow keys, and see a 100-game benchmark. |
| 🃏 Card fusion lab | Fuse two weapon cards with BLEND or DOMINANT. A balance catalog clamps slow, lifesteal and cooldown (hard checks) and rescales damage when the power is out of band (soft check). |
| 🛠️ Build your own bot | Edit the maze agent's scoring function, run 20 seeded games in a resource-limited subprocess and compare it with the default bot. |

## Maze benchmark (100 seeded games, same scoring rule and ghosts)

| agent | games cleared | avg score | dots eaten | lives lost (of 3) |
|---|---|---|---|---|
| hard safety check ON | 59% | 764 | 94% | 1.95 |
| hard safety check OFF | 4% | 531 | 76% | 2.96 |

The Space recomputes these numbers at startup, in about 7 s on one CPU core.

## Layout

- `app.py`: the Gradio 6 UI (theme and CSS are passed to `demo.launch()`).
- `games/tictactoe.py`, `games/maze.py`, `games/fusion.py`: pure game logic and solvi catalogs, importable and testable
  without Gradio.
- `games/explain.py`: renders a solvi `Response` as a "why" card.
- `games/sandbox.py` and `games/_bot_runner.py`: run user bot code in a subprocess with a 5 s timeout and
  `resource.setrlimit` limits (CPU, memory, file size, no new processes). This limits runaway code. It is not a hardened
  security sandbox.
- `games/_ttt_catalog.py`: a vendored copy of the tic-tac-toe catalog, used when `../../examples` is not present (as on
  Hugging Face).

## Run locally

```bash
cd solvi/spaces/arcade
PYTHONPATH=../../src uv run --with "gradio>=6" python app.py
```
