---
title: solvi arcade
emoji: 🕹️
colorFrom: purple
colorTo: yellow
sdk: static
app_file: index.html
pinned: false
license: apache-2.0
short_description: Explainable game agents that run in your browser
---

# solvi arcade (runs in your browser)

Game agents built with [solvi](https://github.com/solvi-ai/solvi) from small Python functions, a few rules and hard
checks. They play well, and every move comes with its reason: the rule's inputs, the facts computed this turn, and which
hard check fired, if any. No models, and **no server**: the page runs Python in the visitor's browser with
[Gradio-Lite](https://www.gradio.app/guides/gradio-lite) (Pyodide, Python compiled to WebAssembly). solvi is pure Python
on numpy/scipy, so it installs from PyPI straight into the browser.

The first visit downloads the Python runtime with numpy, scipy, Gradio and solvi (about 20–30 MB, once; later visits use
the browser cache). Write your own decision task in the [solvi playground](https://huggingface.co/spaces/solvi-ai/playground).

| Tab | What you see |
|---|---|
| ❌⭕ Tic-tac-toe | The catalog from `examples/05_tic_tac_toe.py` (vendored), with an extra `reason` rule that names the move (win, block, fork, forcing, center ...). A "you can't beat it" counter and an "explain the position" view. |
| 👻 Ghost maze | Pac is a catalog of BFS functions plus one scoring rule. The hard check `greedy_is_safe` blocks any step into a ghost's reach while a safer move exists. Turn the check off, take over with the arrows, run the benchmark in the browser. |
| 🃏 Card fusion lab | Fuse two weapon cards with BLEND or DOMINANT. A balance catalog clamps slow, lifesteal and cooldown (hard checks) and rescales damage when the power is out of band (soft check). |
| 🛠️ Build your own bot | Edit the maze agent's scoring function and run 20 seeded games against the default bot, in your own browser. |

## Maze benchmark (100 seeded games, same scoring rule and ghosts)

| agent | games cleared | avg score | dots eaten | lives lost (of 3) |
|---|---|---|---|---|
| hard safety check ON | 59% | 764 | 94% | 1.95 |
| hard safety check OFF | 4% | 531 | 76% | 2.96 |

Native Python computes this in about 7 s on one CPU core; the browser, on a button click, in about 20 s, with the same
numbers.

## Layout

- `index.html`: loads `@gradio/lite@5.45.0` from jsDelivr, lists the requirement (`solvi>=0.1.1`) and mounts `app.py`
  and `games/*.py` by URL.
- `app.py`: the Gradio 5 UI (theme and CSS in `gr.Blocks(...)`).
- `games/tictactoe.py`, `games/maze.py`, `games/fusion.py`: pure game logic and solvi catalogs, the same as in the
  server version of this Space. `games/_ttt_catalog.py` is a vendored copy of the tic-tac-toe example catalog.
- `games/explain.py`: renders a solvi `Response` as a "why" card.
- `games/sandbox.py`: runs the visitor's bot code in-process, in a fresh namespace, under a `sys.settrace` guard: a call
  that runs more than 200,000 lines of the bot's code (an infinite loop), or a run longer than 30 s, is stopped. It
  protects against runaway Python loops, not against a loop inside C code, and it is not a security sandbox (the code
  runs in the visitor's own browser tab).

### Differences from the server version (`../arcade`)

- No threads or subprocesses in Pyodide: the maze benchmark runs on a button click (20 or 100 games) instead of at
  startup, and the bot runs in-process with the guard above instead of a resource-limited subprocess.
- The tic-tac-toe counter is per tab (there is no shared server to count everyone's games).
- "Play 50 ticks" uses an async generator with `asyncio.sleep` (a blocking `time.sleep` would stall the worker).

### Workaround in `index.html`

Gradio-Lite 5.45.0 (the latest release) installs gradio 5.45 with micropip, whose resolver is greedy. With today's PyPI
it picks `huggingface-hub` 1.x/2.x for `gradio_client` and then fails on gradio's `huggingface-hub<1.0` pin (also
`anyio<5`, ...), so the app does not start. `index.html` wraps the web worker and filters PyPI's simple index to files
uploaded before the Gradio-Lite release (2025-09-10), except `solvi`. If a future Gradio-Lite release fixes this, the
wrapper script can be removed.

## Run locally

```bash
cd solvi/spaces/arcade-lite
python -m http.server 8080      # then open http://localhost:8080
```
