"""🎭 Mafia detective tab: watch the solvi detective bot play, or play the detective with the bot as your advisor (logic in
games/mafia.py).

app.py calls `build()` inside `with gr.Tab("🎭 Mafia detective"):` and adds `CSS` to the page's CSS. No threads, no network:
the benchmark runs on a button click."""
from __future__ import annotations

import html
import time

import gradio as gr

from games import mafia as mf
from games.explain import why_html

CSS = """
.mf-board {display: grid; grid-template-columns: repeat(auto-fill, minmax(128px, 1fr)); gap: 8px;}
.mf-p {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 8px 10px; font-size: 0.82rem;
  background: var(--block-background-fill); position: relative;}
.mf-p.mf-dead {opacity: 0.5;}
.mf-p.mf-me {border: 2px solid var(--color-accent);}
.mf-av {font-size: 1.7rem; line-height: 1.1;}
.mf-name {font-weight: 700; font-size: 0.95rem;}
.mf-role {display: inline-block; border-radius: 999px; padding: 0 7px; font-size: 0.7rem; font-weight: 600; margin: 2px 0;
  border: 1px solid var(--border-color-primary);}
.mf-maf {background: #dc262622; color: #ef4444; border-color: #dc262688;}
.mf-town {background: #16a34a22; color: #16a34a; border-color: #16a34a66;}
.mf-track {height: 7px; border-radius: 4px; background: var(--border-color-primary); overflow: hidden; margin-top: 4px;}
.mf-fill {height: 100%; border-radius: 4px; background: #ef4444;}
.mf-log {font-size: 0.85rem; max-height: 420px; overflow-y: auto;}
.mf-log h4 {margin: 8px 0 4px; font-size: 0.9rem;}
.mf-log ul {margin: 0; padding-left: 18px;}
.mf-log li {margin: 1px 0;}
.mf-ev {font-size: 0.8rem; margin: 2px 0 2px 14px;}
.mf-w {font-family: var(--font-mono); font-size: 0.75rem; display: inline-block; min-width: 44px;}
.mf-sus {margin: 6px 0 8px;}
.mf-banner {font-size: 1.25rem; font-weight: 800; margin: 4px 0;}
"""

MODES = ["👀 Watch the bot play", "🕵️ You are the detective (the bot advises)"]
ICON = {"kill": "🔪", "saved": "💉", "accuse": "👉", "defend": "🛡️", "claim": "🗣️", "det_claim": "🕵️", "counter_claim": "🎭",
        "vote": "🗳️", "out": "⚖️", "tie": "🤝", "notebook": "📓"}


# ------------------------------------------------------------------------------------------------------------ views

def board_html(g, resp=None, spoil=False):
    sus = resp.values.get("suspicion", {}) if resp is not None else {}
    cards = []
    for p in g.players:
        dead = p not in g.alive
        me = p == g.det
        badges = []
        if dead:
            r = g.flipped[p]
            badges.append(f"<span class='mf-role {'mf-maf' if r == 'mafia' else 'mf-town'}'>✝ {html.escape(r)}</span>")
        elif me:
            badges.append("<span class='mf-role'>detective</span>")
        if p in g.checks:
            r = g.checks[p][1]
            badges.append(f"<span class='mf-role {'mf-maf' if r == 'mafia' else 'mf-town'}'>🔍 {r} (night {g.checks[p][0]})</span>")
        if (spoil or g.phase == "over") and not dead and not me and p not in g.checks:
            r = g.roles[p]
            badges.append(f"<span class='mf-role {'mf-maf' if r == 'mafia' else ''}'>🙈 {r}</span>")
        bar = ""
        if not dead and not me and p in sus:
            bar = (f"<div class='mf-track' title='bot suspicion'><div class='mf-fill' style='width:{sus[p] * 100:.0f}%'>"
                   f"</div></div><div class='sv-dim'>suspicion {sus[p] * 100:.0f}%</div>")
        cls = "mf-p" + (" mf-dead" if dead else "") + (" mf-me" if me else "")
        cards.append(f"<div class='{cls}'><div class='mf-av'>{mf.AVATAR.get(p, '🙂')}</div><div class='mf-name'>"
                     f"{html.escape(p)}</div>{''.join(badges)}{bar}</div>")
    return f"<div class='mf-board'>{''.join(cards)}</div>"


def status_md(g):
    alive_m = len(g.mafia_alive())
    if g.phase == "over":
        who = "🏆 The town wins: both mafia are out." if g.winner == "town" else "💀 The mafia win."
        return f"<div class='mf-banner'>{who}</div>Mafia were {' and '.join(g.mafia)}. Seed {g.seed}."
    ph = {"night": f"🌙 Night {g.day}", "day": f"☀️ Day {g.day}: discussion", "vote": f"☀️ Day {g.day}: the vote"}[g.phase]
    det = "alive" if g.det in g.alive else "dead"
    return (f"<div class='mf-banner'>{ph}</div>{len(g.alive)} alive · mafia left {alive_m} (hidden) · "
            f"the detective is {det}{' and has claimed' if g.det_claimed else ''} · seed {g.seed}")


def log_html(g):
    by = {}
    for e in g.events:
        by.setdefault(e.day, []).append(e)
    parts = []
    for d in sorted(by, reverse=True):
        items = "".join(f"<li>{ICON.get(e.kind, '•')} {html.escape(e.text())}</li>" for e in by[d])
        parts.append(f"<h4>Night {d} → Day {d}</h4><ul>{items}</ul>")
    return f"<div class='sv-card mf-log'>{''.join(parts) or '<span class=sv-dim>Nothing has happened yet.</span>'}</div>"


def analysis_html(g, resp, who="Dex"):
    if resp is None:
        return "<div class='sv-card sv-dim'>The detective's analysis appears here after each day's discussion.</div>"
    v = resp.values
    qs = [q for q in ("accuse", "vote", "reveal", "investigate") if q in resp.results]
    if "accuse" not in resp.results:                       # a night decision
        t = resp["investigate"].answer
        title = f"{who} checks {t} tonight: the most suspicious player not checked yet"
        return why_html(resp, mf.cat, qs, show_facts=False, title=title) + _suspects(v, 3)
    top2 = v.get("top2", [])
    title = (f"{who}'s answer, day {g.day}: the mafia are most likely " + (" & ".join(top2) or "unknown") +
             f" · vote {resp['vote'].answer}" + (" · claims detective" if resp["reveal"].answer == "yes" else ""))
    note = ""
    beh = v.get("behavior_score", {})
    cleared = v.get("cleared", [])
    ranked = sorted(beh, key=lambda p: -beh[p])
    hidden = [p for p in ranked[:2] if p in cleared]
    if hidden:
        note = ("<div class='sv-dim'>On behaviour alone " + ", ".join(
            f"{html.escape(p)} would rank #{ranked.index(p) + 1}" for p in hidden) +
                ", but the detective verified them as innocent: they are never accused (hard check never_accuse_cleared).</div>")
    return why_html(resp, mf.cat, qs, show_facts=False, title=title) + _suspects(v, 4) + note + (
        f"<details><summary>the strategist's plan (r.flow)</summary><pre style='white-space:pre-wrap;font-size:0.75rem'>"
        f"{html.escape(str(resp.flow))}</pre></details>")


def _suspects(v, k):
    sus, ev = v.get("suspicion", {}), v.get("evidence", {})
    rows = []
    for p in sorted(sus, key=lambda q: -sus[q])[:k]:
        items = sorted(ev.get(p, []), key=lambda t: -abs(t[1]))[:6]
        lines = "".join(f"<div class='mf-ev'><span class='mf-w'>{w:+.2f}</span> {html.escape(txt)} "
                        f"<span class='sv-dim'>({html.escape(mf.LABEL[kind])})</span></div>" for kind, w, txt in items)
        if p in v.get("checks", {}):
            lines = f"<div class='mf-ev'>🔍 checked: {v['checks'][p][1]}</div>" + lines
        rows.append(f"<div class='mf-sus'><b>{html.escape(p)}</b> — {sus[p] * 100:.0f}% mafia{lines or '<div class=mf-ev>no evidence yet</div>'}</div>")
    return f"<div class='sv-card'><div class='sv-title'>Evidence chain (weights = log-likelihood ratios)</div>{''.join(rows)}</div>"


# ------------------------------------------------------------------------------------------------------------ logic

def _advance_watch(g):
    """One phase with the bot as the detective. Returns the bot's Response (or None)."""
    resp = None
    if g.phase == "night":
        t = None
        if g.det in g.alive:
            t, resp = mf.bot_turn_night(g)
        mf.night(g, t)
    elif g.phase in ("day", "vote"):
        if g.phase == "day":
            mf.talk(g)
        if g.det in g.alive:
            v, rev, resp = mf.bot_turn_day(g)
            mf.detective_says(g, v, rev)
            mf.vote(g, v)
        else:
            mf.vote(g, None)
    return resp


def _human_prepare(g):
    """In player mode: run the talk if needed and get the advisor's suggestion for the current phase."""
    if g.phase == "day":
        mf.talk(g)
    if g.det not in g.alive or g.phase == "over":
        return None
    return mf.bot_night(g)[1] if g.phase == "night" else mf.bot_day(g)


def _controls(g, resp, human):
    alive_others = [p for p in g.alive if p != g.det]
    you_alive = g.det in g.alive and g.phase != "over"
    night_vis = human and you_alive and g.phase == "night"
    vote_vis = human and you_alive and g.phase == "vote"
    inv_choices = [p for p in alive_others if p not in g.checks] or alive_others
    inv_val = resp["investigate"].answer if (night_vis and resp is not None and "investigate" in resp.results) else None
    vote_val = resp["vote"].answer if (vote_vis and resp is not None and "vote" in resp.results) else None
    rev_val = bool(vote_vis and resp is not None and "reveal" in resp.results and resp["reveal"].answer == "yes")
    watch_vis = not human or (g.det not in g.alive and g.phase != "over")
    return (gr.update(visible=night_vis), gr.update(choices=inv_choices, value=inv_val if inv_val in inv_choices else None),
            gr.update(visible=vote_vis), gr.update(choices=alive_others, value=vote_val if vote_val in alive_others else None),
            gr.update(value=rev_val and not g.det_claimed, interactive=not g.det_claimed),
            gr.update(visible=watch_vis), gr.update(visible=watch_vis))


def _outs(g, resp, human, spoil, msg=""):
    who = "Advisor" if human else "Dex"
    return (status_md(g) + (f"<div class='sv-dim'>{msg}</div>" if msg else ""), board_html(g, resp, spoil), log_html(g),
            analysis_html(g, resp, who), g, resp, *_controls(g, resp, human))


def new_game(mode, seed, spoil):
    human = mode == MODES[1]
    g = mf.new_game(int(seed or 0), det="You" if human else "Dex")
    resp = _human_prepare(g) if human else None
    msg = ("You are the detective. Tonight, pick whom to check (the advisor pre-selects its choice)." if human else
           "Press ▶ Next phase: night, then day. The bot explains every decision.")
    return _outs(g, resp, human, spoil, msg)


def watch_next(g, mode, spoil, last=None):
    if g is None:
        g = mf.new_game(0, det="Dex")
    human = g.det == "You"
    if g.phase == "over":
        return _outs(g, None, human, spoil)
    resp = _advance_watch(g) or (last if g.phase == "over" else None)
    return _outs(g, resp, human, spoil)


def watch_all(g, mode, spoil):
    if g is None:
        g = mf.new_game(0, det="Dex")
    human = g.det == "You"
    resp = None
    while g.phase != "over":
        r = _advance_watch(g)
        resp = r or resp
    return _outs(g, resp, human, spoil)


def human_check(target, g, spoil):
    if g is None or g.phase != "night" or g.det not in g.alive:
        return _outs(g, None, True, spoil) if g else new_game(MODES[1], 0, spoil)
    if not target:
        return _outs(g, mf.bot_night(g)[1], True, spoil, "Pick a player to check first.")
    g.det_suspect = None
    mf.night(g, target)
    res = g.checks.get(target)
    msg = f"Your check: {target} is {'MAFIA 🔴' if res and res[1] == 'mafia' else 'town 🟢'}." if res else ""
    resp = _human_prepare(g)
    if g.phase == "vote" and g.det in g.alive:
        msg += " Now the town has talked; choose your vote (the advisor pre-selects its suggestion)."
    elif g.det not in g.alive:
        msg += " You were killed in the night. Your notebook is public; press ⏩ to watch the rest."
    return _outs(g, resp, True, spoil, msg)


def human_vote(target, reveal, g, spoil):
    if g is None or g.phase != "vote" or g.det not in g.alive:
        return _outs(g, None, True, spoil) if g else new_game(MODES[1], 0, spoil)
    if not target:
        return _outs(g, mf.bot_day(g), True, spoil, "Pick whom to vote against.")
    warn = ""
    if target in g.checks and g.checks[target][1] == "town":
        warn = (f"⚠️ You verified {target} as innocent — the bot's hard check never_accuse_cleared would never allow this "
                "vote. ")
    g.det_suspect = target
    mf.detective_says(g, target, bool(reveal))
    mf.vote(g, target)
    resp = _human_prepare(g)
    return _outs(g, resp, True, spoil, warn + ("Night falls: pick whom to check." if g.phase == "night" and
                                               g.det in g.alive else ""))


def bench(n):
    n = int(n)
    t0 = time.time()
    b = mf.benchmark(n)
    dt = time.time() - t0
    rows = "\n".join(f"| {name} | **{s['win_rate'] * 100:.1f}%** | {s['avg_days']:.2f} |" for name, s in b.items())
    return (f"#### Town win rate over {n} seeded games (the same role deals for every detective), {dt:.1f} s in your browser\n"
            "| detective | town wins | days per game |\n|---|---|---|\n" + rows +
            "\n\n<span class='sv-dim'>Baselines investigate at random, never claim and vote randomly or for the most "
            "accused player of the day. *Checks only* is the bot without the behaviour evidence (it still checks, "
            "reveals and follows its checks). Native Python over seeds 0–199: bot 61.5%, checks only 60.5%, most accused "
            "28.0%, random 30.0%; over seeds 0–999: bot 65.6%, checks only 55.3%, most accused 27.9%, random "
            "29.0%.</span>")


def build():
    gr.Markdown(
        "Eight players: the detective and seven NPCs with hidden roles — 2 **mafia**, 1 doctor, 4 villagers. At night the "
        "mafia kill, the doctor protects, the detective checks one player. By day the NPCs accuse, defend and claim roles, "
        "then everyone votes. NPCs follow noisy, seeded rules: the mafia bandwagon, hit back at whoever accuses their "
        "partner, defend each other and lie under pressure — but villagers do some of that too. The solvi detective turns "
        "the public record into concrete **evidence events**, weighs them (log-likelihood ratios measured on 3000 simulated "
        "games) and answers *who are the mafia (top 2)*, citing the events. **Hard check `never_accuse_cleared`:** it never "
        "accuses or votes for a player it verified as innocent.")
    game, resp = gr.State(None), gr.State(None)
    with gr.Row():
        mode = gr.Radio(MODES, value=MODES[0], label="Mode", scale=3)
        seed = gr.Number(value=0, precision=0, minimum=0, maximum=10**6, label="seed", scale=1)
        spoil = gr.Checkbox(value=False, label="show hidden roles (spoiler)", scale=1)
        new = gr.Button("🎲 New game", variant="primary", scale=1)
    with gr.Row():
        with gr.Column(scale=6, min_width=320):
            g0 = mf.new_game(0)
            status = gr.HTML(status_md(g0) + "<div class='sv-dim'>Press ▶ Next phase to watch the bot, or pick the "
                             "detective mode and press New game.</div>")
            board = gr.HTML(board_html(g0))
            with gr.Row():
                nxt = gr.Button("▶ Next phase", variant="primary")
                allb = gr.Button("⏩ Play to the end")
            with gr.Group(visible=False) as night_box:
                inv = gr.Dropdown([], label="🌙 Whom do you check tonight?")
                inv_btn = gr.Button("🔍 Check and sleep", variant="primary")
            with gr.Group(visible=False) as vote_box:
                vote_dd = gr.Dropdown([], label="🗳️ Whom do you vote against?")
                reveal = gr.Checkbox(label="Claim detective and reveal my checks before the vote")
                vote_btn = gr.Button("🗳️ Vote", variant="primary")
            log = gr.HTML()
        with gr.Column(scale=6):
            analysis = gr.HTML()
    with gr.Row(equal_height=True):
        n_games = gr.Radio([50, 200], value=200, label="benchmark games", scale=2)
        bench_btn = gr.Button("📊 Benchmark: bot vs baselines", scale=2)
    bench_md = gr.Markdown()
    outs = [status, board, log, analysis, game, resp, night_box, inv, vote_box, vote_dd, reveal, nxt, allb]
    new.click(new_game, [mode, seed, spoil], outs)
    mode.change(new_game, [mode, seed, spoil], outs)
    nxt.click(watch_next, [game, mode, spoil, resp], outs)
    allb.click(watch_all, [game, mode, spoil], outs)
    inv_btn.click(human_check, [inv, game, spoil], outs)
    vote_btn.click(human_vote, [vote_dd, reveal, game, spoil], outs)
    spoil.change(lambda g, r, s: board_html(g, r, s) if g else "", [game, resp, spoil], board)
    bench_btn.click(lambda: "#### Running in your browser …", None, bench_md).then(bench, n_games, bench_md)
