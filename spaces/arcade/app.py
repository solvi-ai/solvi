"""solvi arcade — game agents built from small Python functions, rules and hard checks, explaining every move.

Local run (from solvi/spaces/arcade):  PYTHONPATH=../../src python app.py"""
from __future__ import annotations

import html
import random
import threading
import time

import gradio as gr
from games import fusion, maze, sandbox, tictactoe
from games.explain import why_html

REPO = "https://github.com/solvi-ai/solvi"

# ====================================================================================================================
# Tic-tac-toe
# ====================================================================================================================

CELL_NAMES = ["top-left", "top", "top-right", "left", "center", "right", "bottom-left", "bottom", "bottom-right"]
EMPTY = " "
_TTT_GLOBAL = {"games": 0, "agent_losses": 0, "agent_wins": 0, "draws": 0}
_TTT_LOCK = threading.Lock()


def _ttt_counter(stats):
    with _TTT_LOCK:
        g = dict(_TTT_GLOBAL)
    return (f"### You can't beat it\n"
            f"**You:** {stats['games']} games · agent won {stats['agent_wins']} · draws {stats['draws']} · "
            f"**agent lost {stats['agent_losses']}**  \n"
            f"**Everyone since this Space started:** {g['games']} games · agent lost **{g['agent_losses']}**")


def _ttt_cells(game):
    out = []
    for i, ch in enumerate(game.board):
        label = EMPTY if ch == "." else ("❌" if ch == "X" else "⭕")
        out.append(gr.Button(value=label, variant="primary" if i == game.last_agent_cell else "secondary"))
    return out


def _ttt_status(game):
    if game.over:
        return {"agent": "### solvi wins. Try again?", "human": "### You beat solvi! (please report a bug)",
                "draw": "### Draw. That is the best anyone can do."}[game.result]
    return f"### Your move ({'❌' if game.human == 'X' else '⭕'})"


def _ttt_why(game, resp):
    if resp is None:
        return "<div class='sv-card sv-dim'>solvi has not moved yet. It explains each of its moves here.</div>"
    r = resp["move"]
    if r.answer in (None, "none"):
        return why_html(resp, tictactoe.cat, ["move", "reason"], title="solvi could not move")
    c = int(r.answer)
    reason = resp["reason"].answer
    title = f"solvi plays {CELL_NAMES[c]} (cell {c}) — {tictactoe.REASONS[reason]}"
    return why_html(resp, tictactoe.cat, ["move", "reason"], title=title)


def _ttt_position(game, explain):
    if not explain:
        return gr.HTML(visible=False)
    me = game.turn
    resp = tictactoe.explain_position(game.board, me)
    side = "you" if me == game.human else "solvi"
    title = (f"Position seen from the side to move ({'❌' if me == 'X' else '⭕'}, {side}) — "
             f"every fact the catalog computes, and the move solvi would play here")
    if game.over:
        title = "Game over: the hard check board_valid refuses to pick a move on a finished board"
    return gr.HTML(why_html(resp, tictactoe.cat, ["move", "reason", "outcome"], title=title), visible=True)


def _ttt_count(game, stats):
    if not game.over or game.counted:
        return
    game.counted = True
    key = {"agent": "agent_wins", "human": "agent_losses", "draw": "draws"}[game.result]
    stats["games"] += 1
    stats[key] += 1
    with _TTT_LOCK:
        _TTT_GLOBAL["games"] += 1
        _TTT_GLOBAL[key] += 1


def _ttt_outputs(game, resp, stats, explain, status=None):
    return (*_ttt_cells(game), status or _ttt_status(game), _ttt_why(game, resp), _ttt_position(game, explain),
            _ttt_counter(stats), game, resp, stats)


def ttt_new(side, stats, explain):
    game, resp = tictactoe.new_game("X" if side.startswith("X") else "O")
    return _ttt_outputs(game, resp, stats, explain)


def ttt_click(i, game, resp, stats, explain):
    err = tictactoe.human_move(game, i)
    if err:
        return _ttt_outputs(game, resp, stats, explain, status=f"### {err}")
    if not game.over:
        resp = tictactoe.agent_move(game)
    _ttt_count(game, stats)
    return _ttt_outputs(game, resp, stats, explain)


def ttt_explain(game, explain):
    return _ttt_position(game, explain)


# ====================================================================================================================
# Ghost maze
# ====================================================================================================================

CAT_ON, SYS_ON = maze.build_system(True)
CAT_OFF, SYS_OFF = maze.build_system(False)
PAC_ANGLE = {"UP": 0, "RIGHT": 90, "DOWN": 180, "LEFT": 270, None: 90}
_BENCH: dict = {}


def _run_benchmark():
    t0 = time.time()
    _BENCH["result"] = maze.benchmark(100)
    _BENCH["seconds"] = time.time() - t0


_BENCH_THREAD = threading.Thread(target=_run_benchmark, daemon=True)
_BENCH_THREAD.start()


def bench_markdown():
    _BENCH_THREAD.join()
    b = _BENCH["result"]
    rows = []
    for key, label in (("with_check", "hard safety check **ON**"), ("without_check", "hard safety check OFF")):
        s = b[key]
        rows.append(f"| {label} | **{s['win_rate'] * 100:.0f}%** | {s['avg_score']:.0f} | {s['avg_dots_pct'] * 100:.0f}% | "
                    f"{s['avg_deaths']:.2f} | {s['avg_forced']:.1f} |")
    return ("#### Benchmark: 100 seeded games, same scoring rule, same ghosts\n"
            "| agent | games cleared | avg score | dots eaten | lives lost (of 3) | vetoes per game |\n"
            "|---|---|---|---|---|---|\n" + "\n".join(rows) +
            f"\n\n<span class='sv-dim'>Computed when this Space started ({_BENCH['seconds']:.1f} s on this CPU). "
            f"A game is cleared when Pac eats all {maze.TOTAL_DOTS} dots before losing 3 lives "
            f"(limit {maze.MAX_TICKS} ticks). The only difference between the rows is one hard check.</span>")


def board_html(game, resp=None, show_danger=False):
    ghosts = {}
    for gh in game.ghosts:
        ghosts.setdefault(gh.pos, gh)
    danger = set(map(tuple, resp.values.get("ghost_reach", []))) if (resp is not None and show_danger) else set()
    cells = []
    for r in range(maze.H):
        for c in range(maze.W):
            cell = (r, c)
            if cell in maze.WALLS:
                cells.append("<div class='mz c-wall'></div>")
                continue
            cls = "mz c-open" + (" c-danger" if cell in danger else "")
            inner = ""
            if cell == game.pac:
                a = PAC_ANGLE[game.last_move] - 35
                inner = (f"<div class='mz-pac' style='background:conic-gradient(from {a}deg, transparent 0deg 70deg, "
                         f"#ffd400 70deg 360deg)'></div>")
            elif cell in ghosts:
                gh = ghosts[cell]
                inner = f"<div class='mz-ghost' style='background:{gh.color}' title='{gh.name}'>👀</div>"
            elif cell in game.dots:
                inner = "<div class='mz-dot'></div>"
            cells.append(f"<div class='{cls}'>{inner}</div>")
    banner = ""
    if game.over:
        banner = ("<div class='mz-banner mz-win'>CLEARED! 🎉</div>" if game.won
                  else "<div class='mz-banner mz-lose'>GAME OVER</div>")
    return (f"<div class='mz-wrap'>{banner}<div class='mz-grid' style='grid-template-columns:repeat({maze.W}, 1fr)'>"
            f"{''.join(cells)}</div></div>")


def maze_stats(game):
    eaten = maze.TOTAL_DOTS - len(game.dots)
    return (f"**Score** {game.score} &nbsp;·&nbsp; **Lives** {'❤️' * game.lives or '—'} &nbsp;·&nbsp; "
            f"**Dots** {eaten}/{maze.TOTAL_DOTS} &nbsp;·&nbsp; **Tick** {game.tick} &nbsp;·&nbsp; "
            f"**hard-check vetoes** {game.forced} &nbsp;·&nbsp; seed {game.seed}")


def _maze_why(resp, move, note, safety, who="solvi"):
    if resp is None:
        return "<div class='sv-card sv-dim'>Press Step or Play to let the solvi agent move. Each tick is explained here.</div>"
    cat = CAT_ON if safety else CAT_OFF
    qs = ["move", "fallback"] if safety else ["move"]
    if who == "you":
        title = f"You played {move}. solvi would have played {note}."
    else:
        title = "solvi: " + maze.describe(move, resp, note)
    return why_html(resp, cat, qs, title=title)


def _maze_outputs(game, resp, move, note, safety, show_danger, who="solvi"):
    log = "\n".join(reversed(game.log[-40:]))
    return board_html(game, resp, show_danger), maze_stats(game), log, _maze_why(resp, move, note, safety, who), game


def _event_suffix(ev):
    return {"dot": "", "caught": "  ✖ caught by a ghost!", "won": "  ★ all dots eaten!", "over": "  ✖ game over"}.get(ev, "")


def _agent_tick(game, safety):
    system = SYS_ON if safety else SYS_OFF
    mv, resp, note = maze.agent_decide(system, game, safety)
    line = f"t{game.tick + 1:03d} " + maze.describe(mv, resp, note)
    ev = maze.step(game, mv, note)
    game.log.append(line + _event_suffix(ev))
    return mv, resp, note


def maze_step(game, safety, show_danger):
    if game.over:
        return _maze_outputs(game, None, None, None, safety, show_danger)
    mv, resp, note = _agent_tick(game, safety)
    return _maze_outputs(game, resp, mv, note, safety, show_danger)


def maze_play(game, safety, show_danger):
    resp = mv = note = None
    for _ in range(50):
        if game.over:
            break
        mv, resp, note = _agent_tick(game, safety)
        yield _maze_outputs(game, resp, mv, note, safety, show_danger)
        time.sleep(0.07)
    yield _maze_outputs(game, resp, mv, note, safety, show_danger)


def maze_reset(seed, safety, show_danger):
    game = maze.new_game(int(seed or 0))
    game.log.append(f"new game, seed {game.seed}: {maze.TOTAL_DOTS} dots, 3 ghosts, 3 lives")
    return _maze_outputs(game, None, None, None, safety, show_danger)


def maze_human(move, game, safety, show_danger):
    if game.over:
        return _maze_outputs(game, None, None, None, safety, show_danger)
    system = SYS_ON if safety else SYS_OFF
    smv, resp, note = maze.agent_decide(system, game, safety)
    v = resp.values
    line = (f"t{game.tick + 1:03d} you {move} (ghost {v['ghost_dist'].get(move, 'wall')} away) — "
            f"solvi would play {smv}" + (" (hard check vetoed its first choice)" if note == "forced" else ""))
    ev = maze.step(game, move)
    game.log.append(line + _event_suffix(ev))
    return _maze_outputs(game, resp, move, smv, safety, show_danger, who="you")


def maze_redraw(game, safety, show_danger):
    system = SYS_ON if safety else SYS_OFF
    resp = None if game.over else system.ask(game.state(), ["move", "fallback"] if safety else ["move"])
    return board_html(game, resp, show_danger)


# ====================================================================================================================
# Card fusion lab
# ====================================================================================================================

CARD_COLORS = {"Whip": "#b5651d", "Magic Wand": "#7b61ff", "Frost Orb": "#2fa4e7", "Fire Staff": "#ff5a36",
               "Vampire Fang": "#b3123f", "Thunder Hammer": "#e0b000", "Throwing Knives": "#6f7f8f", "Poison Cloud": "#3fa34d"}
BAR_MAX = {"damage": 40, "fire_rate": 3.4, "area": 6, "speed": 18, "pierce": 10}
BAR_LABEL = {"damage": "damage", "fire_rate": "fire rate /s", "area": "area", "speed": "speed", "pierce": "pierce"}


def card_html(c, colors, tag="", marks=None, small=False):
    marks = marks or {}
    bars = []
    for k in fusion.NUMERIC:
        v = getattr(c, k)
        pct = max(4, min(100, 100 * v / BAR_MAX[k]))
        mark = f" <span class='fx-mark'>{html.escape(marks[k])}</span>" if k in marks else ""
        txt = f"{v:.0f}" if k == "pierce" else f"{v:.2f}" if v < 10 else f"{v:.1f}"
        bars.append(f"<div class='fx-row'><span class='fx-k'>{BAR_LABEL[k]}</span><div class='fx-bar'>"
                    f"<div style='width:{pct:.0f}%'></div></div><span class='fx-v'>{txt}{mark}</span></div>")
    chips = []
    for k, icon, label in (("slow", "❄️", "slow"), ("burn", "🔥", "burn"), ("lifesteal", "🩸", "lifesteal")):
        v = getattr(c, k)
        if v > 1e-9:
            mark = f" <span class='fx-mark'>{html.escape(marks[k])}</span>" if k in marks else ""
            chips.append(f"<span class='fx-chip'>{icon} {label} {'+' if k == 'burn' else ''}{v * 100:.0f}%{mark}</span>")
    cd_mark = f" <span class='fx-mark'>{html.escape(marks['cooldown'])}</span>" if "cooldown" in marks else ""
    chips.append(f"<span class='fx-chip'>⏱ cooldown {c.cooldown:.2f}s{cd_mark}</span>")
    grad = f"linear-gradient(135deg, {colors[0]}, {colors[-1]})"
    sig = f"<div class='fx-sig'>style trait: {html.escape(c.signature)}</div>" if c.signature else ""
    return (f"<div class='fx-card{' fx-small' if small else ''}' style='--fx-grad:{grad}'>"
            f"<div class='fx-head'><span class='fx-icon'>{c.icon}</span><div><div class='fx-name'>{html.escape(c.name)}</div>"
            f"<div class='fx-shape'>{html.escape(c.shape)}{(' · ' + html.escape(tag)) if tag else ''}</div></div></div>"
            f"{''.join(bars)}<div class='fx-chips'>{''.join(chips)}</div>{sig}"
            f"<div class='fx-power'>⚡ power {fusion.card_power(c):.1f}</div></div>")


def gallery_html():
    return "<div class='fx-gallery'>" + "".join(card_html(c, [CARD_COLORS[c.name]] * 2, small=True) for c in fusion.BASE) + "</div>"


def do_fuse(a_name, b_name, law):
    a, b = fusion.BY_NAME[a_name], fusion.BY_NAME[b_name]
    raw = fusion.fuse(a, b, law)
    final, resp, changes = fusion.balance(raw)
    colors = [CARD_COLORS[a_name], CARD_COLORS[b_name]]
    marks = {}
    for q, key, txt in (("slow", "slow", "clamped"), ("lifesteal", "lifesteal", "clamped"), ("cooldown", "cooldown", "clamped")):
        if resp[q].answer == "clamp":
            marks[key] = txt
    if resp["power"].answer != "in_band":
        marks["damage"] = f"×{resp.values['damage_scale']:.2f}"
    law_txt = ("BLEND — order does not matter: shape of the stronger parent, numbers averaged in log space, effects stack"
               if law == "BLEND" else
               f"DOMINANT — {a.name} in the style of {b.name}: shape and 75% of the numbers from {a.name}, "
               f"{b.name}'s effects stack on top, plus its trait ({b.signature})")
    cards = (f"<div class='fx-law'>{html.escape(law_txt)}</div><div class='fx-flow'>"
             f"{card_html(a, [colors[0]] * 2, 'parent A', small=True)}<div class='fx-op'>{'＋' if law == 'BLEND' else '◀'}</div>"
             f"{card_html(b, [colors[1]] * 2, 'parent B', small=True)}<div class='fx-op'>＝</div>"
             f"{card_html(raw, colors, 'raw fusion', small=True)}<div class='fx-op'>⚖️</div>"
             f"{card_html(final, colors, 'balanced by solvi', marks)}</div>")
    ch = ("".join(f"<li>{html.escape(x)}</li>" for x in changes) if changes
          else "<li>nothing to fix: every hard check passed and the power is inside the band</li>")
    verdict = "ships unchanged ✅" if resp["ship_as_is"].answer == "yes" else "needed fixes 🔧"
    report = (f"<div class='sv-card'><div class='sv-title'>Balance report: {html.escape(raw.name)} {verdict}</div>"
              f"<ul class='fx-changes'>{ch}</ul></div>" +
              why_html(resp, fusion.cat, [q.name for q in fusion.QUESTIONS], title="solvi answers (rule inputs, hard checks, facts)"))
    return cards, report


def surprise():
    rng = random.Random()
    a, b = rng.sample([c.name for c in fusion.BASE], 2)
    law = rng.choice(["BLEND", "DOMINANT"])
    return (a, b, law, *do_fuse(a, b, law))


# ====================================================================================================================
# Build your own bot
# ====================================================================================================================

_DEFAULT_RUNS: dict = {}


def _default_run(safety):
    if safety not in _DEFAULT_RUNS:
        _, system = maze.build_system(safety)
        _DEFAULT_RUNS[safety] = maze.summarize([maze.run_game(system, s, safety) for s in range(sandbox.N_GAMES)])
    return _DEFAULT_RUNS[safety]


def run_user_bot(code, safety):
    t0 = time.time()
    res = sandbox.run_bot(code, safety=safety)
    dt = time.time() - t0
    base = _default_run(safety)
    if not res["ok"]:
        return (f"<div class='sv-err'><b>Your bot did not run ({html.escape(res['kind'])}).</b>"
                f"<pre>{html.escape(res['error'])}</pre></div>", "")
    you = res["summary"]

    def row(name, s):
        return (f"| {name} | **{s['win_rate'] * 100:.0f}%** | {s['avg_score']:.0f} | {s['avg_dots_pct'] * 100:.0f}% | "
                f"{s['avg_deaths']:.2f} | {s['avg_forced']:.1f} |")
    diff = you["avg_score"] - base["avg_score"]
    verdict = ("🏆 Your bot beats the default!" if diff > 0 else "🤝 Same score as the default." if diff == 0
               else "The default bot is still ahead. Tweak and try again.")
    warn = ""
    if res.get("abstains"):
        warn = (f"<div class='sv-err'><b>Your function failed or returned an invalid move on {res['abstains']} ticks</b> "
                f"(solvi abstained and Pac took the first legal move). First problem:"
                f"<pre>{html.escape(str(res.get('first_error')))}</pre></div>")
    md = (f"### {verdict}\n"
          f"{sandbox.N_GAMES} seeded games, hard safety check {'ON' if safety else 'OFF'}, ran in {dt:.1f} s (limit 5 s)\n\n"
          "| bot | games cleared | avg score | dots eaten | lives lost | vetoes per game |\n|---|---|---|---|---|---|\n"
          f"{row('your bot', you)}\n{row('default bot', base)}\n\n"
          "Per game (seed: score, ✓ = cleared): " +
          " · ".join(f"{i}: {g['score']}{' ✓' if g['won'] else ''}" for i, g in enumerate(res["games"])))
    return warn, md


# ====================================================================================================================
# UI
# ====================================================================================================================

CSS = """
.gradio-container {max-width: 1180px !important; margin: auto;}
#hero h1 {font-size: 2.1rem; margin-bottom: 0;}
.sv-card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px; margin: 6px 0;
  background: var(--block-background-fill); font-size: 0.92rem; overflow-x: auto;}
.sv-title {font-weight: 700; font-size: 1.02rem; margin-bottom: 8px;}
.sv-ans {margin: 4px 0 8px;}
.sv-why {font-family: var(--font-mono); font-size: 0.82rem; opacity: 0.85; margin-top: 2px; word-break: break-word;}
.sv-badge {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: 0.76rem; font-weight: 600; margin: 2px 2px;}
.sv-ok {background: #16a34a22; color: #16a34a; border: 1px solid #16a34a66;}
.sv-forced {background: #dc262622; color: #ef4444; border: 1px solid #dc262688;}
.sv-abstain {background: #ca8a0422; color: #ca8a04; border: 1px solid #ca8a0466;}
.sv-dim {opacity: 0.7;}
.sv-hc {margin: 6px 0;}
.sv-facts {width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 6px;}
.sv-facts td {border-top: 1px solid var(--border-color-primary); padding: 3px 6px; vertical-align: top;}
.sv-facts td:first-child {font-family: var(--font-mono); white-space: nowrap;}
.sv-err {border: 1px solid #dc2626aa; background: #dc262614; border-radius: 10px; padding: 10px 12px; margin: 6px 0;}
.sv-err pre {white-space: pre-wrap; font-size: 0.8rem; margin: 6px 0 0;}
.ttt-grid {max-width: 330px; gap: 6px !important;}
.ttt-grid button {min-height: 92px !important; font-size: 2.4rem !important; border-radius: 14px !important;}
.mz-wrap {position: relative; max-width: 440px; margin: 0 auto;}
.mz-grid {display: grid; gap: 0; background: #05061a; padding: 6px; border-radius: 12px; border: 2px solid #2233aa;}
.mz {aspect-ratio: 1; position: relative; display: flex; align-items: center; justify-content: center;}
.c-wall {background: #1b2a9e; border-radius: 4px; box-shadow: inset 0 0 0 2px #3b52ff55; margin: 1px;}
.c-danger {background: #ff2d2d33;}
.mz-dot {width: 22%; height: 22%; border-radius: 50%; background: #ffcfa0;}
.mz-pac {width: 82%; height: 82%; border-radius: 50%;}
.mz-ghost {width: 82%; height: 82%; border-radius: 45% 45% 12% 12%; display: flex; align-items: center; justify-content: center;
  font-size: 0.7rem; line-height: 1;}
.mz-banner {position: absolute; top: 42%; left: 0; right: 0; text-align: center; font-weight: 800; font-size: 2rem; z-index: 2;
  letter-spacing: 2px; text-shadow: 0 2px 8px #000;}
.mz-win {color: #7CFF7C;} .mz-lose {color: #ff5a5a;}
.arrows button {font-size: 1.4rem !important; min-width: 56px !important;}
.fx-gallery {display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 10px;}
.fx-flow {display: flex; flex-wrap: wrap; align-items: center; gap: 8px;}
.fx-op {font-size: 1.6rem; opacity: 0.8;}
.fx-law {font-size: 0.88rem; opacity: 0.85; margin-bottom: 8px;}
.fx-card {background: var(--fx-grad); color: #fff; border-radius: 16px; padding: 12px; width: 250px; box-sizing: border-box;
  box-shadow: 0 6px 18px #0004; text-shadow: 0 1px 2px #0006;}
.fx-small {width: 185px; padding: 9px; font-size: 0.85rem;}
.fx-gallery .fx-small {width: auto;}
.fx-head {display: flex; gap: 8px; align-items: center; margin-bottom: 6px;}
.fx-icon {font-size: 2rem; background: #ffffff30; border-radius: 12px; padding: 2px 6px;}
.fx-small .fx-icon {font-size: 1.5rem;}
.fx-icon {white-space: nowrap;}
.fx-small .fx-row {grid-template-columns: 60px 1fr 44px;}
.fx-name {font-weight: 800; font-size: 1.08rem;}
.fx-shape {font-size: 0.72rem; opacity: 0.9; text-transform: uppercase; letter-spacing: 1px;}
.fx-row {display: grid; grid-template-columns: 74px 1fr 64px; align-items: center; gap: 6px; font-size: 0.76rem; margin: 2px 0;}
.fx-bar {height: 7px; background: #00000040; border-radius: 4px; overflow: hidden;}
.fx-bar div {height: 100%; background: #fff; border-radius: 4px;}
.fx-v {text-align: right; font-variant-numeric: tabular-nums;}
.fx-chips {display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px;}
.fx-chip {background: #00000038; border-radius: 999px; padding: 1px 7px; font-size: 0.72rem;}
.fx-mark {background: #fff; color: #b00020; border-radius: 6px; padding: 0 4px; font-weight: 700; text-shadow: none; font-size: 0.68rem;}
.fx-power {margin-top: 7px; font-weight: 800;}
.fx-sig {font-size: 0.7rem; margin-top: 5px; opacity: 0.9;}
.fx-changes {margin: 0 0 4px 18px;}
@media (max-width: 640px) { .fx-card {width: 100%;} .fx-small {width: 46%;} .fx-op {display: none;} }
"""

INTRO = f"""
# 🕹️ solvi arcade
**Game agents made of small Python functions, a few rules and hard checks — strong, and every move is explained.**
No neural network, no GPU: each agent is a [solvi]({REPO}) catalog. For every move it shows the rule's inputs,
the facts it computed this turn and which hard check fired.
"""

with gr.Blocks(title="solvi arcade") as demo:
    gr.Markdown(INTRO, elem_id="hero")
    with gr.Tabs():
        # ---------------------------------------------------------------------------------------------------- TTT
        with gr.Tab("❌⭕ Tic-tac-toe"):
            gr.Markdown("Play against a solvi agent whose whole brain is seven small functions (win, block, fork, forcing move ...) "
                        f"and one rule, from `{tictactoe.SOURCE}`. It answers every move with its reason. "
                        "It has never lost.")
            ttt_game, ttt_resp = gr.State(), gr.State()
            ttt_stats = gr.State({"games": 0, "agent_losses": 0, "agent_wins": 0, "draws": 0})
            with gr.Row():
                with gr.Column(scale=4, min_width=300):
                    with gr.Row():
                        ttt_side = gr.Radio(["X (you start)", "O (solvi starts)"], value="X (you start)", label="You play",
                                            scale=3)
                        ttt_new_btn = gr.Button("New game", variant="primary", scale=1)
                    ttt_status = gr.Markdown("### Your move (❌)")
                    ttt_buttons = []
                    with gr.Column(elem_classes="ttt-grid"):
                        for _ in range(3):
                            with gr.Row(equal_height=True):
                                for _ in range(3):
                                    ttt_buttons.append(gr.Button(EMPTY, min_width=80))
                    ttt_counter = gr.Markdown()
                with gr.Column(scale=6):
                    ttt_why = gr.HTML()
                    ttt_explain_cb = gr.Checkbox(label="Explain the position (all computed facts for the side to move)")
                    ttt_pos = gr.HTML(visible=False)
            ttt_outs = [*ttt_buttons, ttt_status, ttt_why, ttt_pos, ttt_counter, ttt_game, ttt_resp, ttt_stats]
            ttt_new_btn.click(ttt_new, [ttt_side, ttt_stats, ttt_explain_cb], ttt_outs)
            ttt_side.change(ttt_new, [ttt_side, ttt_stats, ttt_explain_cb], ttt_outs)
            for i, b in enumerate(ttt_buttons):
                b.click(lambda g, rs, s, e, i=i: ttt_click(i, g, rs, s, e), [ttt_game, ttt_resp, ttt_stats, ttt_explain_cb],
                        ttt_outs)
            ttt_explain_cb.change(ttt_explain, [ttt_game, ttt_explain_cb], ttt_pos)
            demo.load(ttt_new, [ttt_side, ttt_stats, ttt_explain_cb], ttt_outs)

        # ---------------------------------------------------------------------------------------------------- maze
        with gr.Tab("👻 Ghost maze"):
            gr.Markdown(
                "Pac is a solvi catalog: `legal_moves`, BFS `dot_dist` and `ghost_dist` per move, `ghost_reach`, `danger`, "
                "`safe_moves`, one scoring function `greedy_move`, and a **hard check** `greedy_is_safe`: *never step onto a "
                "cell a ghost can reach this tick when a safer move exists*. When it fires, the answer is forced to EVADE and "
                "Pac takes the best safe move. Turn the check off and watch Pac die. Or take over with the arrows.")
            mz_game = gr.State()
            with gr.Row():
                with gr.Column(scale=5, min_width=320):
                    mz_board = gr.HTML()
                    mz_stats = gr.Markdown()
                    with gr.Row():
                        mz_step = gr.Button("Step", variant="secondary")
                        mz_play = gr.Button("▶ Play 50 ticks", variant="primary")
                        mz_reset = gr.Button("Reset")
                    with gr.Row():
                        mz_safety = gr.Checkbox(value=True, label="hard safety check ON")
                        mz_danger = gr.Checkbox(value=False, label="show ghost reach (red)")
                        mz_seed = gr.Number(value=0, precision=0, label="seed", minimum=0, maximum=10**6)
                    gr.Markdown("**Take over** (one tick per press; solvi's choice is shown for comparison):")
                    with gr.Row(elem_classes="arrows"):
                        mz_left = gr.Button("←")
                        mz_up = gr.Button("↑")
                        mz_down = gr.Button("↓")
                        mz_right = gr.Button("→")
                with gr.Column(scale=6):
                    mz_log = gr.Textbox(label="tick log (newest first)", lines=9, max_lines=9, interactive=False)
                    mz_why = gr.HTML()
            mz_bench = gr.Markdown("#### Benchmark: running 100 seeded games with and without the check …")
            mz_outs = [mz_board, mz_stats, mz_log, mz_why, mz_game]
            mz_step.click(maze_step, [mz_game, mz_safety, mz_danger], mz_outs)
            mz_play.click(maze_play, [mz_game, mz_safety, mz_danger], mz_outs)
            mz_reset.click(maze_reset, [mz_seed, mz_safety, mz_danger], mz_outs)
            for btn, mv in ((mz_left, "LEFT"), (mz_up, "UP"), (mz_down, "DOWN"), (mz_right, "RIGHT")):
                btn.click(lambda g, s, d, mv=mv: maze_human(mv, g, s, d), [mz_game, mz_safety, mz_danger], mz_outs)
            mz_danger.change(maze_redraw, [mz_game, mz_safety, mz_danger], mz_board)
            demo.load(maze_reset, [mz_seed, mz_safety, mz_danger], mz_outs)
            demo.load(bench_markdown, None, mz_bench)

        # ---------------------------------------------------------------------------------------------------- fusion
        with gr.Tab("🃏 Card fusion lab"):
            gr.Markdown(
                "Fuse two weapon cards. The fused card then goes through a solvi balance catalog: it computes the power "
                "(dps × targets × effects), **hard checks** slow ≤ 0.5, lifesteal ≤ 0.4, cooldown ≥ 0.3 s (a violation is "
                "clamped and reported) and a soft check that power stays within [0.5 × median, 1.5 × max] of the base cards "
                "(outside → damage is rescaled, and the report says by how much).")
            with gr.Accordion("The 8 base cards", open=False):
                gr.HTML(gallery_html())
            names = [c.name for c in fusion.BASE]
            with gr.Row():
                fx_a = gr.Dropdown(names, value="Whip", label="Card A")
                fx_b = gr.Dropdown(names, value="Frost Orb", label="Card B")
                fx_law = gr.Radio(["BLEND", "DOMINANT"], value="BLEND", label="Fusion law",
                                  info="BLEND: A+B = B+A · DOMINANT: A in the style of B")
            with gr.Row():
                fx_go = gr.Button("✨ Fuse", variant="primary")
                fx_swap = gr.Button("⇄ Swap A and B")
                fx_rand = gr.Button("🎲 Surprise me")
            fx_cards = gr.HTML()
            fx_report = gr.HTML()
            gr.Markdown("Try **Whip ◀ Vampire Fang** (DOMINANT: lifesteal over the cap), **Magic Wand ◀ Throwing Knives** "
                        "(cooldown too short), **Thunder Hammer + Frost Orb** (BLEND: slow over the cap), or **Thunder Hammer ◀ "
                        "Fire Staff** (too strong, damage rescaled).")
            fx_go.click(do_fuse, [fx_a, fx_b, fx_law], [fx_cards, fx_report])
            fx_swap.click(lambda a, b: (b, a), [fx_a, fx_b], [fx_a, fx_b]).then(do_fuse, [fx_a, fx_b, fx_law], [fx_cards, fx_report])
            fx_rand.click(surprise, None, [fx_a, fx_b, fx_law, fx_cards, fx_report])
            demo.load(do_fuse, [fx_a, fx_b, fx_law], [fx_cards, fx_report])

        # ---------------------------------------------------------------------------------------------------- bot
        with gr.Tab("🛠️ Build your own bot"):
            gr.Markdown(
                "This is the maze agent's scoring function — the one piece of Pac's catalog you can rewrite. Change the "
                f"weights or the logic, then run {sandbox.N_GAMES} seeded games against the default bot. solvi plans the flow "
                "from your function's argument names, so you can use any fact listed in the comments. "
                "Ideas: set `W_GHOST = 0` and turn the safety check off; use `danger` or `safe_moves` yourself; prefer moves "
                "that keep more `legal_moves` open.")
            with gr.Row():
                with gr.Column(scale=7):
                    bot_code = gr.Code(value=maze.DEFAULT_BOT_CODE, language="python", lines=26, max_lines=40,
                                       label="greedy_move (Python)", interactive=True)
                with gr.Column(scale=5):
                    bot_safety = gr.Checkbox(value=True, label="keep the hard safety check greedy_is_safe")
                    with gr.Row():
                        bot_run = gr.Button(f"▶ Run {sandbox.N_GAMES} games", variant="primary")
                        bot_reset = gr.Button("Reset code")
                    bot_err = gr.HTML()
                    bot_out = gr.Markdown()
                    gr.Markdown("<span class='sv-dim'>Your code runs in a separate process with a 5 s timeout, CPU, memory, "
                                "file-size and process limits, in a temporary directory and without environment variables. "
                                "It is resource-limited, not a hardened security sandbox.</span>")
            bot_run.click(run_user_bot, [bot_code, bot_safety], [bot_err, bot_out])
            bot_reset.click(lambda: maze.DEFAULT_BOT_CODE, None, bot_code)

    gr.Markdown(f"<span class='sv-dim'>Built with [solvi]({REPO}) — decision systems from a catalog of Python functions and "
                "checks. CPU only, no models. Every answer here is a solvi `Response`: `r[q].answer / .why / .status`, "
                "the planned `flow` and a hash-chained `trace`.</span>")


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="violet", secondary_hue="amber"), css=CSS)
