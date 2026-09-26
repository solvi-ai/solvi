"""🏟️ Bot arena tab: a tournament of maze bots on the same seeded games (logic in games/arena.py).

app.py calls `build()` inside `with gr.Tab("🏟️ Bot arena"):` and adds `CSS` to the page's CSS. Runs in Pyodide: no
threads; the tournament is an async generator that yields a progress bar after each game."""
from __future__ import annotations

import asyncio
import html
import time

import gradio as gr
import pandas as pd

from games import arena, maze

SIZES = [5, 10, 20]
SORT_KEYS = {"avg score": "avg_score", "levels cleared": "cleared", "lives lost (fewest)": "lives_lost",
             "decision time (fastest)": "ms_per_decision"}
COLUMNS = ["#", "bot", "hard check", "avg score", "cleared", "lives lost", "ms / decision", "safety", "status"]
DTYPES = ["number", "str", "str", "number", "number", "number", "number", "str", "str"]
USER_DEFAULTS = [("the default rule, braver", True, False), ("danger-aware", False, False),
                 ("infinite loop (see the guard)", False, False)]
PAC_ANGLE = {"UP": 0, "RIGHT": 90, "DOWN": 180, "LEFT": 270, None: 90}
OWN_COLOR, ENF_COLOR = "#7c3aed", "#f59e0b"

CSS = """
.ar-prog {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 12px; font-size: 0.9rem;}
.ar-prog .ar-track {height: 8px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; margin-top: 6px;}
.ar-prog .ar-fill {height: 100%; background: var(--color-accent); border-radius: 4px; transition: width 0.2s;}
.ar-roster {display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 8px;}
.ar-bot {border: 1px solid var(--border-color-primary); border-radius: 10px; padding: 8px 10px; font-size: 0.85rem;
  background: var(--block-background-fill);}
.ar-bot b {font-size: 0.95rem;}
.ar-tag {display: inline-block; border-radius: 999px; padding: 0 8px; font-size: 0.72rem; font-weight: 600; margin-left: 4px;}
.ar-on {background: #16a34a22; color: #16a34a; border: 1px solid #16a34a66;}
.ar-off {background: #6b728022; color: var(--body-text-color-subdued); border: 1px solid #6b728066;}
.ar-chart {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 12px; font-size: 0.86rem;
  background: var(--block-background-fill); overflow-x: auto;}
.ar-row {display: grid; grid-template-columns: minmax(90px, 150px) 1fr; gap: 8px; align-items: center; margin: 5px 0;}
.ar-name {white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.ar-bars {display: flex; flex-direction: column; gap: 2px;}
.ar-bar {height: 14px; border-radius: 4px; display: flex; align-items: center; justify-content: flex-end; padding-right: 5px;
  color: #fff; font-size: 0.7rem; font-weight: 700; min-width: 30px; box-sizing: border-box;}
.ar-bar.ar-thin {height: 9px; font-size: 0; opacity: 0.85;}
.ar-legend {display: flex; gap: 14px; flex-wrap: wrap; font-size: 0.78rem; opacity: 0.85; margin: 2px 0 6px;}
.ar-legend i {display: inline-block; width: 12px; height: 10px; border-radius: 3px; margin-right: 4px; vertical-align: middle;}
.ar-h2h {border-collapse: collapse; font-size: 0.78rem; margin-top: 10px;}
.ar-h2h th, .ar-h2h td {border: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: center; white-space: nowrap;}
.ar-h2h th:first-child {text-align: left;}
.ar-win {background: #16a34a2e;} .ar-lose {background: #dc26262a;} .ar-tie {background: #6b72801f;}
.ar-card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 10px 12px; margin: 4px 0;
  background: var(--block-background-fill); font-size: 0.88rem; overflow-x: auto;}
.ar-card h4 {margin: 0 0 6px;}
.ar-dim {opacity: 0.7;}
.ar-err {border: 1px solid #dc2626aa; background: #dc262614; border-radius: 10px; padding: 8px 10px; margin: 6px 0;}
.ar-err pre {white-space: pre-wrap; font-size: 0.78rem; margin: 4px 0 0;}
.ar-deaths {border-collapse: collapse; width: 100%; font-size: 0.8rem; margin-top: 6px;}
.ar-deaths td, .ar-deaths th {border-top: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: left; vertical-align: top;}
.ar-yes {color: #16a34a; font-weight: 700;} .ar-no {color: #ef4444; font-weight: 700;}
.ar-mz-wrap {position: relative; max-width: 380px; margin: 0 auto;}
.ar-mz-grid {display: grid; gap: 0; background: #05061a; padding: 5px; border-radius: 12px; border: 2px solid #2233aa;}
.ar-mz {aspect-ratio: 1; position: relative; display: flex; align-items: center; justify-content: center;}
.ar-wall {background: #1b2a9e; border-radius: 4px; box-shadow: inset 0 0 0 2px #3b52ff55; margin: 1px;}
.ar-reach {background: #ff2d2d30;}
.ar-next {box-shadow: inset 0 0 0 2px #7CFF7C;}
.ar-next.ar-reach {box-shadow: inset 0 0 0 2px #ff5a5a;}
.ar-dot {width: 22%; height: 22%; border-radius: 50%; background: #ffcfa0;}
.ar-pac {width: 82%; height: 82%; border-radius: 50%;}
.ar-ghost {width: 82%; height: 82%; border-radius: 45% 45% 12% 12%; display: flex; align-items: center; justify-content: center;
  font-size: 0.62rem; line-height: 1;}
.ar-banner {position: absolute; top: 42%; left: 0; right: 0; text-align: center; font-weight: 800; font-size: 1.7rem; z-index: 2;
  letter-spacing: 2px; text-shadow: 0 2px 8px #000;}
.ar-banner.ar-w {color: #7CFF7C;} .ar-banner.ar-l {color: #ff5a5a;}
.ar-lines {font-family: var(--font-mono); font-size: 0.78rem; line-height: 1.45;}
.ar-lines div {padding: 1px 6px; border-radius: 5px; word-break: break-word;}
.ar-lines .ar-cur {background: var(--color-accent-soft, #7c3aed22); font-weight: 700;}
.ar-lines .ar-dead {color: #ef4444;}
.ar-hof ol {margin: 4px 0 0 20px; padding: 0;}
@media (max-width: 640px) { .ar-row {grid-template-columns: 90px 1fr;} }
"""

INTRO = f"""### 🏟️ Bot arena
Six maze bots with different personalities and up to three of yours play **the same {arena.N_GAMES} seeded games**
(same maze, same ghost dice). Every bot is a solvi catalog — the maze facts (`dot_dist`, `ghost_dist`, `danger`,
`safe_moves` …) plus the bot's own scoring function `greedy_move`. The hard check `greedy_is_safe` is a separate part of
the catalog, **outside the bot's own logic**: switch on *enforce the hard safety check for every bot* and watch it lift the
reckless bots without changing a line of their code."""


# --------------------------------------------------------------------------------------------------------------------
# HTML pieces
# --------------------------------------------------------------------------------------------------------------------

def roster_html():
    cards = []
    for e in arena.ROSTER:
        tag = "<span class='ar-tag ar-on'>check ON</span>" if e.own_check else "<span class='ar-tag ar-off'>no check</span>"
        cards.append(f"<div class='ar-bot'>{e.icon} <b>{html.escape(e.name)}</b>{tag}"
                     f"<div class='ar-dim'>{html.escape(e.blurb)}</div></div>")
    return f"<div class='ar-roster'>{''.join(cards)}</div>"


def progress_html(done, total, label, secs, final=False):
    pct = 100 * done / max(total, 1)
    head = (f"✅ <b>Done:</b> {total} games in <b>{secs:.1f} s</b>, in your browser. {html.escape(label)}" if final else
            f"⏳ <b>{done}/{total}</b> games · {html.escape(label)} · {secs:.1f} s")
    return f"<div class='ar-prog'>{head}<div class='ar-track'><div class='ar-fill' style='width:{pct:.0f}%'></div></div></div>"


def _check_label(r, enforce):
    if r.check_on and not r.entry.own_check:
        return "ON (enforced)"
    return "ON" if r.check_on else "off"


def _status(r):
    if r.status == "crashed":
        at = f" (seed {r.crashed_seed})" if r.crashed_seed is not None else " (at load)"
        return "💥 crashed" + at
    if r.status == "error":
        return "⚠️ error"
    return "ok" if not r.summary()["abstains"] else f"ok, {r.summary()['abstains']} failed ticks"


def leaderboard_df(data, sort_label="avg score"):
    if not data:
        return pd.DataFrame([], columns=COLUMNS)
    rows = []
    for i, r in enumerate(arena.ranked(data["results"], SORT_KEYS.get(sort_label, "avg_score")), 1):
        s = r.summary()
        safety = ("—" if not r.games else f"🛡️ {s['vetoes']} vetoed" if r.check_on else f"⚠️ {s['unsafe']} unsafe taken")
        rows.append([i, f"{r.entry.icon} {r.name}", _check_label(r, data["enforce"]), round(s["avg_score"], 1),
                     s["cleared"], s["lives_lost"], round(s["ms_per_decision"], 3), safety,
                     _status(r)])
    return pd.DataFrame(rows, columns=COLUMNS)


def chart_html(data, cache):
    if not data:
        return "<div class='ar-chart ar-dim'>Run the tournament to see the chart.</div>"
    seeds = tuple(data["seeds"])
    res = arena.ranked(data["results"])
    rows, top = [], 1.0
    pairs = []
    for r in res:
        own = cache.get((r.entry.ident(), r.entry.own_check, seeds))
        enf = cache.get((r.entry.ident(), True, seeds))
        cur = r
        pairs.append((r, own, enf, cur))
        for x in (own, enf, cur):
            if x is not None and x.games:
                top = max(top, x.summary()["avg_score"])
    any_both = False
    for r, own, enf, cur in pairs:
        bars = []
        if r.status != "ok":
            bars.append(f"<span class='ar-dim'>{html.escape(_status(r))}</span>")
        else:
            both = own is not None and enf is not None and not r.entry.own_check
            any_both |= both
            for x, color, label in ((own, OWN_COLOR, "own rules"), (enf, ENF_COLOR, "check enforced")):
                if x is None or (x is enf and r.entry.own_check):
                    continue
                v = x.summary()["avg_score"]
                thin = " ar-thin" if both and x is not cur else ""
                bars.append(f"<div class='ar-bar{thin}' title='{label}: {v:.0f}' "
                            f"style='width:{max(4, 100 * v / top):.1f}%;background:{color}'>{v:.0f}</div>")
            if not bars:
                v = r.summary()["avg_score"]
                bars.append(f"<div class='ar-bar' style='width:{max(4, 100 * v / top):.1f}%;background:{OWN_COLOR}'>{v:.0f}</div>")
        rows.append(f"<div class='ar-row'><div class='ar-name'>{r.entry.icon} {html.escape(r.name)}</div>"
                    f"<div class='ar-bars'>{''.join(bars)}</div></div>")
    legend = (f"<div class='ar-legend'><span><i style='background:{OWN_COLOR}'></i>own rules</span>"
              f"<span><i style='background:{ENF_COLOR}'></i>hard check enforced</span>"
              + ("" if any_both else "<span class='ar-dim'>toggle <b>enforce</b> and run again to see both bars</span>")
              + "</div>")
    # head-to-head matrix: wins on the same seeds
    ok = [r for r in res if r.status == "ok" and r.games]
    h2h = arena.head_to_head(ok)
    head = "".join(f"<th title='{html.escape(r.name)}'>{r.entry.icon}</th>" for r in ok)
    body = []
    for a in ok:
        cells = []
        for b in ok:
            if a is b:
                cells.append("<td>—</td>")
                continue
            w, lz, t = h2h[(a.name, b.name)]
            cls = "ar-win" if w > lz else "ar-lose" if w < lz else "ar-tie"
            cells.append(f"<td class='{cls}' title='{html.escape(a.name)} vs {html.escape(b.name)}: {w} wins, {lz} losses, "
                         f"{t} ties'>{w}–{lz}</td>")
        body.append(f"<tr><th>{a.entry.icon} {html.escape(a.name)}</th>{''.join(cells)}</tr>")
    matrix = (f"<table class='ar-h2h'><tr><th>head-to-head (row beats column on N seeds)</th>{head}</tr>{''.join(body)}</table>"
              if len(ok) > 1 else "")
    return (f"<div class='ar-chart'><b>Average score per game</b> ({len(seeds)} games, seeds {seeds[0]}–{seeds[-1]})"
            f"{legend}{''.join(rows)}{matrix}</div>")


def _find(data, name):
    for r in (data or {}).get("results", []):
        if r.name == name:
            return r
    return None


def why_lost_html(data, name, cache):
    r = _find(data, name)
    if r is None:
        return "<div class='ar-card ar-dim'>Run the tournament, then pick a bot.</div>"
    ranking = arena.ranked(data["results"])
    rank = ranking.index(r) + 1
    out = [f"<h4>{r.entry.icon} {html.escape(r.name)} — rank {rank} of {len(ranking)}</h4>"]
    if r.status != "ok":
        where = (f"in the game with seed {r.crashed_seed}, after {len(r.games)} finished games" if r.crashed_seed is not None
                 else "while loading the code")
        kind = "stopped by the guard" if r.status == "crashed" else "did not run"
        out.append(f"<div class='ar-err'><b>Your bot {kind}</b> {where}. The rest of the tournament went on without it."
                   f"<pre>{html.escape(r.error)}</pre></div>")
        if not r.games:
            return f"<div class='ar-card'>{''.join(out)}</div>"
    s = r.summary()
    d = arena.death_report(r)
    n = len(r.games)
    best = ranking[0].summary()["avg_score"] if ranking[0].games else 0
    out.append(f"<p>Average score <b>{s['avg_score']:.0f}</b> (leader {best:.0f}); cleared <b>{s['cleared']}</b> of {n} games; "
               f"lost <b>{s['lives_lost']}</b> lives of {n * maze.LIVES}."
               + (f" {d['timeouts']} lost games ran out of time ({maze.MAX_TICKS} ticks) with lives left: the bot wandered "
                  f"instead of eating." if d["timeouts"] else "") + "</p>")
    if d["deaths"]:
        dist = d["avg_ghost_dist"]
        out.append(f"<p>Of the {len(d['deaths'])} lives lost: <span class='ar-yes'>{d['preventable']}</span> times the fatal "
                   "move stepped into a ghost's reach <b>while a safe move existed</b> — the hard check greedy_is_safe "
                   f"would have refused it; <span class='ar-no'>{d['trapped']}</span> times Pac was trapped (every move "
                   "within a ghost's reach), so no check could help: the plan led into a dead end. "
                   f"When Pac chose the fatal move, the nearest ghost was on average <b>{dist:.1f}</b> cells away.</p>")
    if r.check_on:
        out.append(f"<p>The check was ON for this bot and vetoed <b>{d['vetoes']}</b> of its moves "
                   f"({d['vetoes'] / max(1, sum(g.ticks for g in r.games)) * 100:.1f}% of all ticks).</p>")
    else:
        enf = cache.get((r.entry.ident(), True, tuple(data["seeds"])))
        tail = (f" With the check enforced, the same code scores <b>{enf.summary()['avg_score']:.0f}</b> "
                f"({enf.summary()['avg_score'] - s['avg_score']:+.0f})." if enf is not None and enf.games else
                " Toggle <b>enforce the hard safety check</b> and run again to see what the check would change.")
        out.append(f"<p>No check: the bot took <b>{d['unsafe']}</b> unsafe moves while a safe one existed.{tail}</p>")
    if d["abstains"]:
        out.append(f"<div class='ar-err'>The bot's function failed or returned an invalid move on <b>{d['abstains']}</b> "
                   "ticks; solvi abstained and Pac took the first legal move. Check the replay's why lines.</div>")
    if d["deaths"]:
        rows = []
        for x in d["deaths"][:15]:
            verdict = ("<span class='ar-yes'>yes</span> — safe: " + "/".join(x["safe_moves"]) if x["preventable"] else
                       "<span class='ar-no'>no</span> — trapped" if not x["safe_moves"] else "no — the move was safe")
            wanted = (f" (wanted {x['wanted']}, vetoed)" if x["note"] == "forced" else "")
            rows.append(f"<tr><td>{x['seed']}</td><td>{x['tick']}</td><td>{x['move']}{wanted}</td>"
                        f"<td>{x['ghost_dist_before']}</td><td>{verdict}</td></tr>")
        more = f"<div class='ar-dim'>… and {len(d['deaths']) - 15} more</div>" if len(d["deaths"]) > 15 else ""
        out.append("<table class='ar-deaths'><tr><th>seed</th><th>tick</th><th>fatal move</th><th>ghost dist.</th>"
                   f"<th>would the check have stopped it?</th></tr>{''.join(rows)}</table>{more}"
                   "<div class='ar-dim'>Open any of these in the replay below: pick the bot and the seed, then ⏭ next death.</div>")
    else:
        out.append("<p>No lives lost. 🎉</p>")
    return f"<div class='ar-card'>{''.join(out)}</div>"


def board_html(g, next_move=None):
    ghosts = {}
    for gh in g.ghosts:
        ghosts.setdefault(gh.pos, gh)
    reach = set()
    if not g.over:
        reach = {gh.pos for gh in g.ghosts} | {n for gh in g.ghosts for _, n in maze.open_neighbors(gh.pos)}
    nxt = maze.step_cell(g.pac, next_move) if next_move in maze.DIRS else None
    cells = []
    for r in range(maze.H):
        for c in range(maze.W):
            cell = (r, c)
            if cell in maze.WALLS:
                cells.append("<div class='ar-mz ar-wall'></div>")
                continue
            cls = "ar-mz" + (" ar-reach" if cell in reach else "") + (" ar-next" if cell == nxt else "")
            inner = ""
            if cell == g.pac:
                a = PAC_ANGLE[g.last_move] - 35
                inner = (f"<div class='ar-pac' style='background:conic-gradient(from {a}deg, transparent 0deg 70deg, "
                         f"#ffd400 70deg 360deg)'></div>")
            elif cell in ghosts:
                gh = ghosts[cell]
                inner = f"<div class='ar-ghost' style='background:{gh.color}' title='{gh.name}'>👀</div>"
            elif cell in g.dots:
                inner = "<div class='ar-dot'></div>"
            cells.append(f"<div class='{cls}'>{inner}</div>")
    banner = ""
    if g.over:
        banner = ("<div class='ar-banner ar-w'>CLEARED!</div>" if g.won else "<div class='ar-banner ar-l'>GAME OVER</div>")
    stats = (f"<div class='ar-dim' style='text-align:center;font-size:0.82rem;margin-top:4px'>tick {g.tick} · score {g.score}"
             f" · lives {'❤️' * g.lives or '—'} · dots {maze.TOTAL_DOTS - len(g.dots)}/{maze.TOTAL_DOTS}"
             " · red = ghost reach, outline = next move</div>")
    return (f"<div class='ar-mz-wrap'>{banner}<div class='ar-mz-grid' style='grid-template-columns:repeat({maze.W}, 1fr)'>"
            f"{''.join(cells)}</div>{stats}</div>")


def _game(data, bot, seed_label):
    r = _find(data, bot)
    if r is None or not r.games:
        return r, None
    seed = _seed_of(seed_label)
    for g in r.games:
        if g.seed == seed:
            return r, g
    return r, r.games[0]


def _seed_of(label):
    try:
        return int(str(label).split()[1])
    except (IndexError, ValueError):
        return None


def _game_labels(r):
    if r is None:
        return []
    return [f"seed {g.seed} · {g.score} pts{' ✓' if g.won else ''} · {g.deaths} lives lost" for g in r.games]


def replay_view(data, bot, seed_label, t):
    r, g = _game(data, bot, seed_label)
    if g is None:
        msg = "Run the tournament first." if r is None else "This bot has no finished game to replay."
        return f"<div class='ar-card ar-dim'>{msg}</div>", ""
    t = max(0, min(int(t or 0), g.ticks))
    st = arena.state_at(g.seed, g.moves, g.notes, t)
    board = board_html(st, g.moves[t] if t < g.ticks else None)
    lo, hi = max(0, t - 4), min(g.ticks, t + 5)
    lines = []
    for i in range(lo, hi):
        ln = g.lines[i]
        cls = "ar-cur" if i == t else ""
        if "✖ caught" in ln:
            cls += " ar-dead"
        lines.append(f"<div class='{cls}'>t{i + 1:03d} {html.escape(ln)}</div>")
    if t >= g.ticks:
        lines.append(f"<div class='ar-cur'>end: {'cleared' if g.won else 'game over'}, score {g.score}</div>")
    head = (f"<div class='ar-dim' style='margin-bottom:4px'>{html.escape(r.entry.icon + ' ' + r.name)}, seed {g.seed}: "
            f"the highlighted line is the move Pac makes <b>from the position shown</b> (tick {t + 1} of {g.ticks}).</div>")
    return board, f"<div class='ar-card'>{head}<div class='ar-lines'>{''.join(lines)}</div></div>"


def hof_html(hof):
    entries = (hof or {}).get("entries") or []
    if not entries:
        return ("<div class='ar-card ar-hof'><b>🏆 Hall of fame</b> <span class='ar-dim'>(kept in this browser only)</span>"
                "<div class='ar-dim'>No bot of yours has finished a tournament yet.</div></div>")
    medals = ["🥇", "🥈", "🥉", "4.", "5."]
    items = "".join(f"<li style='list-style:none'>{medals[i] if i < len(medals) else ''} <b>{html.escape(e['name'])}</b> — "
                    f"avg <b>{e['avg_score']:.0f}</b>, cleared {e['cleared']}/{e['games']}, check {e['check']}"
                    f"{' (enforced)' if e.get('enforced') and e['check'] == 'on' else ''}, seeds {html.escape(e['seeds'])} "
                    f"<span class='ar-dim'>{html.escape(e['when'])}</span></li>" for i, e in enumerate(entries))
    return (f"<div class='ar-card ar-hof'><b>🏆 Hall of fame</b> <span class='ar-dim'>(your best bots, kept in this "
            f"browser only)</span><ol>{items}</ol></div>")


# --------------------------------------------------------------------------------------------------------------------
# handlers
# --------------------------------------------------------------------------------------------------------------------

def _entries(slots):
    out = list(arena.ROSTER)
    for i in range(arena.MAX_USER_BOTS):
        name, enabled, check, code = slots[4 * i: 4 * i + 4]
        if enabled and (code or "").strip():
            out.append(arena.user_entry(i + 1, name, code, check))
    names = set()
    for e in out:                                   # unique names: the UI finds bots by name
        base, k = e.name, 2
        while e.name in names:
            e.name = f"{base} ({k})"
            k += 1
        names.add(e.name)
    return out


def _default_focus(results):
    users = [r for r in results if r.entry.kind == "user"]
    if users:
        return users[0].name
    ok = [r for r in arena.ranked(results) if r.status == "ok"]
    return ok[-1].name if ok else results[0].name


async def run_tournament(n_games, first_seed, enforce, sort_label, cache, hof, *slots, _c=None):
    c = _c
    seeds = list(range(int(first_seed or 0), int(first_seed or 0) + int(n_games or arena.N_GAMES)))
    cache = dict(cache or {})
    t = arena.Tournament(_entries(slots), seeds, enforce, cache)
    t0 = last = time.perf_counter()
    yield {c["progress"]: progress_html(0, t.total(), "starting …", 0.0)}
    await asyncio.sleep(0.01)
    cached = 0
    for p in t.steps():
        cached += bool(p.get("cached"))
        now = time.perf_counter()
        if now - last > 0.15 or p["done"] == p["total"]:
            last = now
            label = p["bot"] + (f" · seed {p['seed']}" if "seed" in p else "") + (" (cached)" if p.get("cached") else "")
            if p.get("status") in ("crashed", "error"):
                label += " · 💥 stopped, the tournament goes on"
            yield {c["progress"]: progress_html(p["done"], p["total"], label, now - t0)}
        await asyncio.sleep(0)
    secs = time.perf_counter() - t0
    data = {"results": t.results, "seeds": seeds, "enforce": bool(enforce), "seconds": secs}
    note = f"{cached} bots reused from the previous run." if cached else ""
    crashed = [r.name for r in t.results if r.status != "ok"]
    if crashed:
        note += " Stopped: " + ", ".join(crashed) + " (see “Why did my bot lose?”)."
    hof = arena.hall_of_fame_update(hof, t.results, seeds, enforce)
    names = [r.name for r in arena.ranked(t.results)]
    focus = _default_focus(t.results)
    rp_bot = next((r.name for r in arena.ranked(t.results) if r.games), names[0])
    labels = _game_labels(_find(data, rp_bot))
    board, lines = replay_view(data, rp_bot, labels[0] if labels else None, 0)
    yield {c["progress"]: progress_html(t.total(), t.total(), note, secs, final=True),
           c["data"]: data, c["cache"]: cache, c["board"]: leaderboard_df(data, sort_label),
           c["chart"]: chart_html(data, cache),
           c["why_bot"]: gr.update(choices=names, value=focus), c["why"]: why_lost_html(data, focus, cache),
           c["rp_bot"]: gr.update(choices=names, value=rp_bot),
           c["rp_game"]: gr.update(choices=labels, value=labels[0] if labels else None),
           c["rp_tick"]: gr.update(value=0, maximum=maze.MAX_TICKS),
           c["rp_board"]: board, c["rp_lines"]: lines, c["hof"]: hof, c["hof_html"]: hof_html(hof)}


def pick_bot(data, bot):
    labels = _game_labels(_find(data, bot))
    board, lines = replay_view(data, bot, labels[0] if labels else None, 0)
    return gr.update(choices=labels, value=labels[0] if labels else None), 0, board, lines


def pick_game(data, bot, seed_label):
    board, lines = replay_view(data, bot, seed_label, 0)
    return 0, board, lines


def move_tick(data, bot, seed_label, t, how):
    _, g = _game(data, bot, seed_label)
    if g is None:
        return (0, *replay_view(data, bot, seed_label, 0))
    t = int(t or 0)
    if how == "start":
        t = 0
    elif how == "prev":
        t -= 1
    elif how == "next":
        t += 1
    elif how == "death":
        later = [d["tick"] - 1 for d in g.death_info if d["tick"] - 1 > t]
        t = later[0] if later else ([d["tick"] - 1 for d in g.death_info] or [g.ticks])[0]
    elif how == "end":
        t = g.ticks
    t = max(0, min(t, g.ticks))
    return (t, *replay_view(data, bot, seed_label, t))


async def play_ticks(data, bot, seed_label, t):
    _, g = _game(data, bot, seed_label)
    if g is None:
        yield (0, *replay_view(data, bot, seed_label, 0))
        return
    t = int(t or 0)
    for _ in range(40):
        if t >= g.ticks:
            break
        t += 1
        yield (t, *replay_view(data, bot, seed_label, t))
        await asyncio.sleep(0.08)


# --------------------------------------------------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------------------------------------------------

def build():
    gr.Markdown(INTRO)
    data = gr.State(None)
    cache = gr.State({})
    hof = gr.BrowserState({"entries": []}, storage_key="solvi_arena_hall_of_fame", secret="solvi-arena-hof-v1")
    with gr.Row(equal_height=True):
        n_games = gr.Radio(SIZES, value=arena.N_GAMES, label="games per bot", scale=2)
        first_seed = gr.Number(value=0, precision=0, minimum=0, maximum=10**6, label="first seed", scale=1, min_width=90)
        enforce = gr.Checkbox(value=False, label="enforce the hard safety check for every bot", scale=3)
        run = gr.Button("🏟️ Run the tournament", variant="primary", scale=2)
    with gr.Accordion("The roster", open=False):
        gr.HTML(roster_html())
    slots = []
    with gr.Accordion("Your bots (up to 3) — same function as in “Build your own bot”", open=True):
        gr.Markdown("<span class='ar-dim'>Write `greedy_move(...)`: its argument names are facts the maze catalog computes "
                    "(`legal_moves, dot_dist, ghost_dist, danger, safe_moves, next_cells, pac, ghosts, dots, last_move, "
                    "dots_left`); it returns `\"UP\"`, `\"DOWN\"`, `\"LEFT\"` or `\"RIGHT\"`. It runs in your browser under a "
                    "guard: a call that runs too long (an infinite loop) is stopped, and the bot is marked as crashed.</span>")
        with gr.Tabs():
            for i, (tpl, on, chk) in enumerate(USER_DEFAULTS, 1):
                with gr.Tab(f"Your bot {i}"):
                    with gr.Row(equal_height=True):
                        name = gr.Textbox(value=f"Your bot {i}", label="name", max_length=24, scale=2)
                        enabled = gr.Checkbox(value=on, label="enter the tournament", scale=1)
                        check = gr.Checkbox(value=chk, label="keep greedy_is_safe on", scale=1)
                    with gr.Row(equal_height=True):
                        tpl_dd = gr.Dropdown(list(arena.USER_TEMPLATES), value=tpl, label="template", scale=3)
                        tpl_btn = gr.Button("Load template", scale=1)
                    code = gr.Code(value=arena.USER_TEMPLATES[tpl], language="python", lines=12, max_lines=30,
                                   label="greedy_move (Python)", interactive=True)
                    tpl_btn.click(lambda k: arena.USER_TEMPLATES.get(k, ""), tpl_dd, code)
                    slots += [name, enabled, check, code]
    progress = gr.HTML(progress_html(0, 1, "press Run: 6 built-in bots + your enabled bots, "
                                           f"{arena.N_GAMES} games each", 0.0).replace("⏳", "⏸"))
    sort_by = gr.Radio(list(SORT_KEYS), value="avg score", label="rank by")
    board = gr.Dataframe(value=leaderboard_df(None), headers=COLUMNS, datatype=DTYPES, interactive=False,
                         label="Leaderboard (or click a column header to sort)", wrap=False, max_height=560,
                         column_widths=["44px", "170px", "120px", "96px", "80px", "90px", "110px", "150px", "150px"])
    chart = gr.HTML(chart_html(None, {}))
    with gr.Accordion("🔍 Why did my bot lose?", open=True):
        why_bot = gr.Dropdown([], label="bot", allow_custom_value=True)
        why = gr.HTML(why_lost_html(None, None, {}))
    with gr.Accordion("🎞️ Replay a match", open=True):
        with gr.Row(equal_height=True):
            rp_bot = gr.Dropdown([], label="bot", allow_custom_value=True, scale=2)
            rp_game = gr.Dropdown([], label="game", allow_custom_value=True, scale=3)
        rp_tick = gr.Slider(0, maze.MAX_TICKS, value=0, step=1, label="tick")
        with gr.Row():
            b_start = gr.Button("⏮", min_width=50)
            b_prev = gr.Button("◀ tick", min_width=70)
            b_next = gr.Button("tick ▶", min_width=70)
            b_play = gr.Button("▶ play 40", variant="primary", min_width=90)
            b_death = gr.Button("⏭ next death", min_width=110)
            b_end = gr.Button("⏭ end", min_width=70)
        with gr.Row():
            with gr.Column(scale=5, min_width=280):
                rp_board = gr.HTML()
            with gr.Column(scale=6):
                rp_lines = gr.HTML()
    hof_view = gr.HTML(hof_html(None))
    clear_hof = gr.Button("Clear the hall of fame", size="sm")

    comps = {"progress": progress, "data": data, "cache": cache, "board": board, "chart": chart, "why_bot": why_bot,
             "why": why, "rp_bot": rp_bot, "rp_game": rp_game, "rp_tick": rp_tick, "rp_board": rp_board,
             "rp_lines": rp_lines, "hof": hof, "hof_html": hof_view}

    async def _run(*args):
        async for out in run_tournament(*args, _c=comps):
            yield out

    run.click(_run, [n_games, first_seed, enforce, sort_by, cache, hof, *slots], list(comps.values()),
              concurrency_id="arena_run", concurrency_limit=1)
    sort_by.change(leaderboard_df, [data, sort_by], board)
    why_bot.change(why_lost_html, [data, why_bot, cache], why)
    rp_bot.input(pick_bot, [data, rp_bot], [rp_game, rp_tick, rp_board, rp_lines])
    rp_game.input(pick_game, [data, rp_bot, rp_game], [rp_tick, rp_board, rp_lines])
    rp_tick.input(lambda d, b, s, t: replay_view(d, b, s, t), [data, rp_bot, rp_game, rp_tick], [rp_board, rp_lines],
                  trigger_mode="always_last")
    for btn, how in ((b_start, "start"), (b_prev, "prev"), (b_next, "next"), (b_death, "death"), (b_end, "end")):
        btn.click(lambda d, b, s, t, how=how: move_tick(d, b, s, t, how), [data, rp_bot, rp_game, rp_tick],
                  [rp_tick, rp_board, rp_lines])
    b_play.click(play_ticks, [data, rp_bot, rp_game, rp_tick], [rp_tick, rp_board, rp_lines])
    clear_hof.click(lambda: ({"entries": []}, hof_html(None)), None, [hof, hof_view])
    root = gr.context.Context.root_block           # the page's Blocks: show the stored hall of fame on load
    if root is not None:
        root.load(hof_html, hof, hof_view)
    return comps
