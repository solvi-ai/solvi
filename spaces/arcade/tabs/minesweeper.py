"""💣 Minesweeper tab: human play, hints and a solvi solver bot that explains every move (logic in games/minesweeper.py).

app.py calls `build()` inside `with gr.Tab("💣 Minesweeper"):` and appends `CSS` to the page's CSS. The board is a grid of
gr.Buttons (16×16 created once; for 9×9 the extra rows and columns are hidden), so it needs no JavaScript. Runs in Pyodide:
"play to the end" is an async generator with asyncio.sleep, the benchmark runs on a button click."""
from __future__ import annotations

import asyncio
import html
import time

import gradio as gr

from games import minesweeper as ms
from games.explain import why_html

N = ms.MAX_R                     # the grid is N×N buttons; smaller boards hide the rest
DEFAULT_SIZE = next(iter(ms.SIZES))
PLANNERS = {"exact (enumerate the frontier)": "exact", "quick (local ratios, no enumeration)": "quick"}
BENCH = {"9×9 · 100 games": ("9×9 · 10 mines", 100), "16×16 · 30 games": ("16×16 · 40 mines", 30)}
# measured with native Python (CPython 3.10, one core), seeds 0..n-1 — the same code as in the browser
NATIVE = [("9×9 · 10 mines", 100, "exact", True, 0.97, 0.992, 0.17, 0.0),
          ("9×9 · 10 mines", 100, "quick", True, 0.95, 0.990, 0.20, 9.4),
          ("9×9 · 10 mines", 100, "quick", False, 0.71, 0.769, 2.83, 0.0),
          ("16×16 · 40 mines", 100, "exact", True, 0.84, 0.990, 0.77, 0.0),
          ("16×16 · 40 mines", 100, "quick", True, 0.82, 0.987, 0.93, 49.0),
          ("16×16 · 40 mines", 100, "quick", False, 0.05, 0.548, 5.95, 0.0)]

CSS = """
.ms-grid {gap: 3px !important; width: fit-content !important; max-width: 100%; overflow-x: auto; margin: 4px auto;
  padding: 8px !important; border-radius: 12px; border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill);}
.ms-row {gap: 3px !important; flex-wrap: nowrap !important;}
.ms-grid button.ms-cell {min-width: 0 !important; width: 36px !important; height: 36px !important; flex: 0 0 36px !important;
  padding: 0 !important; border-radius: 6px !important; font-size: 1.05rem !important; font-weight: 800 !important;
  box-shadow: none; line-height: 1;}
.ms-grid button.ms-s16 {width: 26px !important; height: 26px !important; flex-basis: 26px !important; font-size: 0.8rem !important;
  border-radius: 4px !important;}
.ms-grid button.ms-hid {background: linear-gradient(145deg, #a78bfa, #7c3aed) !important; border: none !important; color: #fff !important;}
.ms-grid button.ms-hid:hover {filter: brightness(1.15);}
.ms-grid button.ms-flag {background: linear-gradient(145deg, #fde68a, #f59e0b) !important; border: none !important;}
.ms-grid button.ms-rev {background: var(--background-fill-secondary) !important; border: 1px solid var(--border-color-primary) !important;
  cursor: default;}
.ms-grid button.ms-n1 {color: #2563eb !important;} .ms-grid button.ms-n2 {color: #16a34a !important;}
.ms-grid button.ms-n3 {color: #dc2626 !important;} .ms-grid button.ms-n4 {color: #7c3aed !important;}
.ms-grid button.ms-n5 {color: #b45309 !important;} .ms-grid button.ms-n6 {color: #0891b2 !important;}
.ms-grid button.ms-n7 {color: #db2777 !important;} .ms-grid button.ms-n8 {color: #6b7280 !important;}
.ms-grid button.ms-mine {background: #fca5a5 !important; border: none !important;}
.ms-grid button.ms-boom {background: #dc2626 !important; border: none !important;}
.ms-grid button.ms-ev {box-shadow: inset 0 0 0 3px #f59e0b !important;}
.ms-grid button.ms-mu {box-shadow: inset 0 0 0 3px #ef4444 !important;}
.ms-grid button.ms-alt {box-shadow: inset 0 0 0 3px #16a34a !important;}
.ms-grid button.ms-tgt {box-shadow: 0 0 0 3px #16a34a, 0 0 12px #16a34a !important; position: relative; z-index: 1;}
.ms-grid button.ms-tgt-guess {box-shadow: 0 0 0 3px #eab308, 0 0 12px #eab308 !important; position: relative; z-index: 1;}
.ms-grid button.ms-tgt-flag {box-shadow: 0 0 0 3px #ef4444, 0 0 12px #ef4444 !important; position: relative; z-index: 1;}
.ms-legend {font-size: 0.8rem; opacity: 0.8; text-align: center;}
.ms-legend span {display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin: 0 3px 0 8px; vertical-align: -1px;}
.ms-reason {font-size: 0.98rem; line-height: 1.45; margin-bottom: 6px;}
"""

LEGEND = ("<div class='ms-legend'><span style='box-shadow: inset 0 0 0 3px #f59e0b'></span>numbers used as evidence"
          "<span style='box-shadow: inset 0 0 0 3px #ef4444'></span>mines it relies on"
          "<span style='box-shadow: 0 0 0 3px #16a34a'></span>certain move"
          "<span style='box-shadow: 0 0 0 3px #eab308'></span>guess</div>")


# --------------------------------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------------------------------

def _cells(game, hl=None):
    hl = hl or {}
    ev, mu = set(map(tuple, hl.get("evidence", []))), set(map(tuple, hl.get("mines", [])))
    tgt = tuple(hl["target"]) if hl.get("target") else None
    alt = tuple(hl["alt"]) if hl.get("alt") else None
    tcls = {"certain": "ms-tgt", "guess": "ms-tgt-guess", "flag": "ms-tgt-flag"}.get(hl.get("kind"), "ms-tgt")
    out = []
    for r in range(N):
        for c in range(N):
            if r >= game.rows or c >= game.cols:
                out.append(gr.Button(visible=False))
                continue
            x = (r, c)
            cls = ["ms-cell", f"ms-s{game.cols}"]
            if x in game.revealed:
                n = game.number(x)
                label = str(n) if n else " "
                cls += ["ms-rev", f"ms-n{n}"]
            elif game.over and x == game.boom:
                label, cls = "💥", cls + ["ms-boom"]
            elif game.over and x in game.mines and not game.won:
                label = "💣" if x not in game.flagged else "🚩"
                cls.append("ms-mine")
            elif x in game.flagged:
                label = "✖" if game.over and x not in game.mines else "🚩"
                cls.append("ms-flag")
            else:
                label, cls = " ", cls + ["ms-hid"]
            if x == tgt:
                cls.append(tcls)
            elif x in mu:
                cls.append("ms-mu")
            elif x in ev:
                cls.append("ms-ev")
            elif x == alt:
                cls.append("ms-alt")
            out.append(gr.Button(value=label, elem_classes=cls, visible=True))
    return out


def _rows(game):
    return [gr.Row(visible=r < game.rows) for r in range(N)]


def _status(game):
    if game.over:
        head = "### 🎉 Cleared!" if game.won else "### 💥 Boom. Press New game."
    elif not game.placed:
        head = "### Click any cell: the first click is always safe"
    else:
        head = "### Your move"
    return (f"{head}\n**Mines** {game.n_mines} · 🚩 {len(game.flagged)} · **safe cells left** {game.safe_left} · seed "
            f"{game.seed} · bot reveals: **{game.bot_certain} certain**, {game.bot_guesses} guesses, "
            f"{game.bot_forced} hard-check vetoes")


def _card(text, cls="", extra=""):
    return f"<div class='sv-card {cls}'><div class='ms-reason'>{text}</div>{extra}</div>"


def _bot_card(resp, act, cell, info, planner, check, prefix):
    cat = ms.SYSTEMS[check][0]
    text = html.escape(ms.explain(act, cell, info, planner))
    badge = ("<span class='sv-badge sv-forced'>hard check fired</span>" if info.get("forced") else
             f"<span class='sv-badge {'sv-ok' if info['kind'] == 'certain' else 'sv-abstain'}'>{info['kind']}</span>")
    rm = resp.values.get("risk_map") or {}
    cm = resp.values.get("certain_moves") or []
    summary = (f"<div class='sv-dim' style='font-size:0.82rem'>facts this turn: "
               f"{len(resp.values.get('constraints') or [])} numbers on the frontier · "
               f"{sum(m['act'] == 'reveal' for m in cm)} certain-safe and {sum(m['act'] == 'flag' for m in cm)} certain-mine "
               f"cells · frontier groups to enumerate: {rm.get('groups') or 'none'} · probabilities "
               f"{'exact' if rm.get('exact') else 'estimated'} · {resp.ms:.0f} ms</div>")
    return (f"<div class='sv-card'><div class='ms-reason'><b>{prefix}</b> {badge}<br>{text}</div>{summary}</div>" +
            why_html(resp, cat, ["move", "certainty"] + (["certain_move"] if info.get("forced") else []), show_facts=False,
                     title="solvi's answers (rule inputs, hard check)"))


def _hl(cell, info, act):
    kind = "flag" if act == "flag" else info.get("kind", "certain")
    return {"target": cell, "evidence": info.get("evidence", []), "mines": info.get("mines_used", []), "kind": kind}


def _out(game, hl, why, log=None):
    lines = "\n".join(reversed(game.log[-60:]))
    return (*_cells(game, hl), *_rows(game), _status(game), why, lines, game)


INTRO_CARD = _card("Play by clicking cells (switch the click to 🚩 to mark mines), or let the solvi bot play. "
                   "Every bot move comes with its reason in plain words, and the cells it used are outlined on the board.",
                   "sv-dim")


# --------------------------------------------------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------------------------------------------------

def new(size, seed):
    game = ms.new_game(size, int(seed or 0))
    game.log.append(f"new game {size}, seed {game.seed}")
    return _out(game, None, INTRO_CARD)


def click(i, game, mode, planner_label):
    r, c = divmod(i, N)
    if game is None or r >= game.rows or c >= game.cols or game.over:
        return _out(game, None, _card("The game is over. Press New game.") if game and game.over else INTRO_CARD)
    cell = (r, c)
    if mode.startswith("Flag"):
        ms.toggle_flag(game, cell)
        game.log.append(f"you {'flag' if cell in game.flagged else 'unflag'} {ms.name(cell)}")
        return _out(game, None, _card(f"You {'flagged' if cell in game.flagged else 'unflagged'} {ms.name(cell)}. "
                                      "(The bot does not trust flags: it deduces mines itself.)", "sv-dim"))
    if cell in game.revealed:
        return _out(game, None, _card(f"{ms.name(cell)} is already open.", "sv-dim"))
    text, hl = ms.assess(game, cell, PLANNERS[planner_label])
    ev = ms.reveal(game, cell)
    game.log.append(f"you reveal {ms.name(cell)}" + {"boom": "  💥", "won": "  🎉 cleared"}.get(ev, ""))
    if text is None:
        text = f"You revealed {ms.name(cell)}" + (": the first click is always safe." if ev != "boom" else ".")
    if ev == "boom":
        text = "💥 " + text
    return _out(game, hl, _card(html.escape(text)))


def hint(game, planner_label, check):
    if game.over:
        return _out(game, None, _card("The game is over. Press New game."))
    planner = PLANNERS[planner_label]
    resp, act, cell, info = ms.bot_decide(game, planner, check)
    return _out(game, _hl(cell, info, act), _bot_card(resp, act, cell, info, planner, check, "💡 Hint:"))


def _bot_move(game, planner, check):
    resp, act, cell, info, ev = ms.bot_step(game, planner, check)
    game.log.append(f"bot {act} {ms.name(cell)} — {'VETO→' if info.get('forced') else ''}{info['kind']}"
                    + {"boom": "  💥", "won": "  🎉 cleared"}.get(ev, ""))
    return resp, act, cell, info


def step(game, planner_label, check):
    if game.over:
        return _out(game, None, _card("The game is over. Press New game."))
    planner = PLANNERS[planner_label]
    resp, act, cell, info = _bot_move(game, planner, check)
    return _out(game, _hl(cell, info, act), _bot_card(resp, act, cell, info, planner, check, "🤖 Bot:"))


async def play(game, planner_label, check):
    planner = PLANNERS[planner_label]
    last = None
    for _ in range(400):
        if game.over:
            break
        resp, act, cell, info = _bot_move(game, planner, check)
        last = (resp, act, cell, info)
        yield _out(game, _hl(cell, info, act), _card(html.escape(ms.explain(act, cell, info, planner))))
        await asyncio.sleep(0.03)            # time.sleep would block the browser's Python worker
    if last:
        yield _out(game, _hl(last[2], last[3], last[1]), _bot_card(*last, planner, check, "🤖 Bot, last move:"))
    else:
        yield _out(game, None, _card("The game is over. Press New game."))


def _native_md():
    rows = []
    for size, n, pl, ck, win, cert, gpg, fpg in NATIVE:
        if win is None:
            continue
        rows.append(f"| {size} | {pl}{' + hard check' if ck else ', check OFF'} | {n} | **{win * 100:.0f}%** | "
                    f"{cert * 100:.1f}% | {gpg:.2f} | {fpg:.1f} |")
    return ("#### Benchmark (native Python, seeds 0..n−1)\n| board | bot | games | win rate | reveals that were certain | "
            "guesses per game | check vetoes per game |\n|---|---|---|---|---|---|---|\n" + "\n".join(rows) +
            "\n\n<span class='sv-dim'>The first click is always safe and is not counted. The quick bot guesses from "
            "local ratios; with the hard check ON it is forced back to a certain move whenever one exists, and wins almost "
            "as often as the exact bot. Press the button to recompute with the settings above, in your browser.</span>")


def bench(which, planner_label, check):
    size, n = BENCH[which]
    planner = PLANNERS[planner_label]
    t0 = time.time()
    s = ms.summarize([ms.run_game(i, size, planner, check) for i in range(n)])
    dt = time.time() - t0
    return (f"#### Benchmark: {n} seeded games ({size}), computed in your browser in {dt:.1f} s\n"
            "| bot | win rate | reveals that were certain | guesses per game | check vetoes per game | wrong certain moves |\n"
            "|---|---|---|---|---|---|\n"
            f"| {planner}{' + hard check' if check else ', check OFF'} | **{s['win_rate'] * 100:.0f}%** | "
            f"{s['certain_pct'] * 100:.1f}% | {s['guesses_per_game']:.2f} | {s['forced_per_game']:.1f} | "
            f"{s['wrong_certain']} |\n\n<span class='sv-dim'>Seeds 0 to {n - 1}. \"Wrong certain moves\" counts moves "
            f"the bot called certain that were wrong: it should always be 0.</span>\n\n" + _native_md())


# --------------------------------------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------------------------------------

def build():
    gr.Markdown(
        "The bot is a solvi catalog over the **visible** board: `constraints` (numbers next to hidden cells), "
        "`deductions` (single numbers, then pairs of overlapping numbers), `risk_map` (exact mine chances: the frontier "
        "is split into independent groups and every mine arrangement is counted), `certain_moves`, `planned_move`, and a "
        "**hard check** `no_guess_when_certain`: *never guess while a certain-safe cell is known*. Each move answers "
        "`move` and `certainty`, with the reason in plain words and the evidence outlined on the board.")
    game0 = ms.new_game(DEFAULT_SIZE, 0)
    game0.log.append(f"new game {DEFAULT_SIZE}, seed 0")
    state = gr.State(game0)
    buttons, rows = [], []
    with gr.Row():
        with gr.Column(scale=6, min_width=380):
            with gr.Row(equal_height=True):
                size = gr.Radio(list(ms.SIZES), value=DEFAULT_SIZE, label="board", scale=3)
                seed = gr.Number(value=0, precision=0, label="seed", minimum=0, maximum=10**6, scale=1, min_width=80)
                new_btn = gr.Button("New game", variant="primary", scale=1, min_width=90)
            status = gr.Markdown(_status(game0))
            mode = gr.Radio(["Reveal", "Flag 🚩"], value="Reveal", label="a click on the board will", container=False)
            with gr.Column(elem_classes="ms-grid"):
                for r in range(N):
                    with gr.Row(elem_classes="ms-row", visible=r < game0.rows) as row:
                        for c in range(N):
                            buttons.append(gr.Button(value=" ",
                                                     elem_classes=["ms-cell", f"ms-s{game0.cols}", "ms-hid"],
                                                     visible=r < game0.rows and c < game0.cols, min_width=0, scale=0))
                    rows.append(row)
            gr.HTML(LEGEND)
            with gr.Row():
                hint_btn = gr.Button("💡 Hint")
                step_btn = gr.Button("🤖 Bot: one step")
                play_btn = gr.Button("▶ Bot: play to the end", variant="primary")
        with gr.Column(scale=5):
            why = gr.HTML(INTRO_CARD)
            with gr.Row():
                planner = gr.Radio(list(PLANNERS), value=next(iter(PLANNERS)), label="bot's risk estimate", scale=3)
                check = gr.Checkbox(value=True, label="hard check: never guess while a certain move exists", scale=2)
            log = gr.Textbox(label="move log (newest first)", lines=8, max_lines=8, interactive=False,
                             value="\n".join(game0.log))
    with gr.Row(equal_height=True):
        bench_size = gr.Radio(list(BENCH), value=next(iter(BENCH)), label="benchmark", scale=3)
        bench_btn = gr.Button("📊 Run the benchmark in your browser", scale=2)
    bench_md = gr.Markdown(_native_md())

    outs = [*buttons, *rows, status, why, log, state]
    new_btn.click(new, [size, seed], outs)
    size.change(new, [size, seed], outs)
    for i, b in enumerate(buttons):
        b.click(lambda g, m, p, i=i: click(i, g, m, p), [state, mode, planner], outs)
    hint_btn.click(hint, [state, planner, check], outs)
    step_btn.click(step, [state, planner, check], outs)
    play_btn.click(play, [state, planner, check], outs)
    bench_btn.click(lambda: "#### Benchmark: running in your browser …", None, bench_md).then(
        bench, [bench_size, planner, check], bench_md)
