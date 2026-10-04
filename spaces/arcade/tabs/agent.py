"""🧭 Agent and knowledge (1.0) tab: solvi's environment agent on the toy crafting world of examples/25 (logic in
games/crafting.py). Part 1: run 1 vs run 2 in the same world (System 2 explores, System 1 walks what it learned), a new
world, the knowledge report (items, sources, retractions) and replay. Part 2: protection vs justified risk.

app.py calls `build()` inside `with gr.Tab(...)` and appends `CSS` to the page's CSS. The agent lives in a gr.State, so
every visitor has their own memory."""
from __future__ import annotations

import html
import math
import time

import gradio as gr

from games import crafting as cr

CSS = """
.ag-map svg {max-width: 360px; width: 100%; height: auto;}
.ag-table {width: 100%; border-collapse: collapse; font-size: 0.82rem;}
.ag-table th, .ag-table td {border-top: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: left; vertical-align: top;}
.ag-table td.num, .ag-table th.num {text-align: right; font-variant-numeric: tabular-nums;}
.ag-s1 {color: #059669; font-weight: 700;} .ag-s2 {color: #d97706; font-weight: 700;}
.ag-no {color: #dc2626;}
.ag-log {max-height: 300px; overflow-y: auto;}
.ag-kpi {display: flex; flex-wrap: wrap; gap: 8px; margin: 6px 0;}
.ag-kpi div {padding: 6px 10px; border-radius: 8px; background: rgba(124,58,237,.08); font-size: 0.8rem;}
.ag-kpi b {display: block; font-size: 1.15rem; font-variant-numeric: tabular-nums;}
"""

INTRO = (
    "An **environment agent** (`solvi.Agent`, new in solvi 1.0) in a small crafting world: 8 places joined by exits, "
    "each with a resource (🌳 tree, 🪨 stone, ⛓️ iron, 💧 water), and six goals — collect wood, place a table, make a "
    "pickaxe, collect stone, make a stone pickaxe, collect iron. Every step is one recorded decision: **System 1** takes "
    "an action its knowledge (`solvi.Knowledge`) predicts will work and that advances a goal; **System 2** searches when "
    "System 1 has nothing it is sure of. After each step the knowledge learns what each action needs, which action "
    "finished each goal, and where each exit leads. No model; each run takes well under a second.")


def _kpis(pairs):
    return "<div class='ag-kpi'>" + "".join(f"<div><b>{html.escape(str(v))}</b>{html.escape(k)}</div>"
                                            for k, v in pairs) + "</div>"


def map_svg(env, rows):
    """The world as a ring of places with their exits; a place's ring is thicker the more the last run visited it."""
    places = env.places
    n = len(places)
    pos = {p: (180 + 125 * math.cos(2 * math.pi * i / n - math.pi / 2), 160 + 125 * math.sin(2 * math.pi * i / n - math.pi / 2))
           for i, p in enumerate(places)}
    visits = {p: 0 for p in places}
    for r in rows:
        visits[r[1]] += 1
    out = ["<svg viewBox='0 0 360 320' xmlns='http://www.w3.org/2000/svg' role='img' aria-label='the world map'>"]
    seen = set()
    for a, ex in env.exits.items():
        for b in ex.values():
            if (b, a) in seen or a == b:
                continue
            seen.add((a, b))
            (x1, y1), (x2, y2) = pos[a], pos[b]
            out.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='#94a3b8' "
                       "stroke-width='1.5'/>")
    if getattr(env, "bridge", False):
        iron = next(p for p in places if env.resource[p] == "iron")
        (x1, y1), (x2, y2) = pos[env.start], pos[iron]
        out.append(f"<line x1='{x1:.0f}' y1='{y1:.0f}' x2='{x2:.0f}' y2='{y2:.0f}' stroke='#dc2626' "
                   "stroke-width='2.5' stroke-dasharray='6 4'/>")
    for p in places:
        x, y = pos[p]
        w = 1.5 + min(visits[p], 12) * 0.5
        color = "#7c3aed" if p == env.start else "#64748b"
        out.append(f"<circle cx='{x:.0f}' cy='{y:.0f}' r='22' fill='rgba(124,58,237,.06)' stroke='{color}' "
                   f"stroke-width='{w:.1f}'/>")
        out.append(f"<text x='{x:.0f}' y='{y + 6:.0f}' text-anchor='middle' font-size='18'>{cr.ICON[env.resource[p]]}</text>")
        out.append(f"<text x='{x:.0f}' y='{y - 26:.0f}' text-anchor='middle' font-size='11' fill='currentColor'>"
                   f"{p}{' start' if p == env.start else ''}{f' ×{visits[p]}' if visits[p] else ''}</text>")
    out.append("</svg>")
    return "<div class='ag-map'>" + "".join(out) + "</div>"


def runs_table(runs):
    head = ("<table class='ag-table'><tr><th>run</th><th>world</th><th class='num'>steps</th><th class='num'>goals</th>"
            "<th class='num'>System 1</th><th class='num'>System 2</th><th class='num'>refused</th><th class='num'>ms</th></tr>")
    body = "".join(f"<tr><td>{html.escape(r['name'])}</td><td>{r['seed']}</td><td class='num'>{r['steps']}</td>"
                   f"<td class='num'>{r['goals']} of 6</td><td class='num ag-s1'>{r['s1']}</td>"
                   f"<td class='num ag-s2'>{r['s2']}</td><td class='num'>{r['refused']}</td><td class='num'>{r['ms']:.0f}</td></tr>"
                   for r in runs)
    return head + body + "</table>"


def log_table(rows):
    head = ("<div class='ag-log'><table class='ag-table'><tr><th class='num'>#</th><th>at</th><th>action</th><th>by</th>"
            "<th>accepted</th><th>why it was chosen</th></tr>")
    body = "".join(f"<tr><td class='num'>{i}</td><td>{at}</td><td>{html.escape(a)}</td>"
                   f"<td class='ag-{by}'>{'System 1' if by == 's1' else 'System 2' if by == 's2' else html.escape(str(by))}</td>"
                   f"<td class='{'' if ok else 'ag-no'}'>{'yes' if ok else 'no' + (f' ({html.escape(eff)})' if eff else '')}</td>"
                   f"<td>{html.escape(why)}</td></tr>" for i, at, a, by, ok, why, eff in rows)
    return head + body + "</table></div>"


def knowledge_md(agent, retraction=None):
    rep = agent.knowledge.report()
    st = rep["store"]
    acts = rep.get("actions") or {}
    learned = []
    for a in cr.OPS:
        s = acts.get(a) or {}
        allowed = s.get("allowed") or {}
        need = [f"{k} = {v[0]}" for k, v in allowed.items() if len(v) == 1 and k != "table_here" or
                (k == "table_here" and v == ["True"])]
        learned.append(f"| `{a}` | {', '.join(need) or '—'} | {s.get('transitions', 0)} | {len(s.get('refusals') or [])} |")
    rp = agent.replay()
    status = st["status"]
    md = [f"**Knowledge report** (`agent.knowledge.report()`): **{st['items']} items** in a hash-chained journal of "
          f"{st['journal']} entries (fingerprint `{st['fingerprint']}`, verifies: **{agent.knowledge.store.verify()}**). "
          f"By source: {', '.join(f'{k} {v}' for k, v in sorted(st['by_source'].items()))}. "
          f"Status: {', '.join(f'{k} {v}' for k, v in sorted(status.items()))} — a later observation refutes a map fact "
          "it contradicts; a retracted item stays in the journal. "
          f"Map facts per world: {', '.join(f'{k}: {v}' for k, v in rep.get('maps', {}).items()) or '—'}. "
          f"Skills: {len(rep.get('skills') or {})} of 6 goals.",
          "",
          f"**Every decision replays** (`agent.replay()`, from the stored facts, without the world): "
          f"**{rp['ok']} of {rp['decisions']}**.",
          "",
          "| action | conditions that held at every success | transitions seen | refusal patterns |",
          "|---|---|---|---|", *learned]
    if retraction:
        r = retraction
        md += ["", "**Retraction** (`km.tell(..., source=\"person\")`, then `km.retract(id)`): a guide told the agent "
               "“in world 7 the iron is at p3”, and a second fact derived from it. "
               f"The agent's own guess offered as a fact (`source=\"s1\"`) was **{'refused' if not r['own'] else 'stored'}** "
               "by the source check — a system never learns from its own answers. Retracting the guide's fact took back "
               f"**{len(r['status'])} items** ({', '.join(f'{b} → {a}' for b, a in r['status'].values())}): the derived "
               f"one went with it. The store now has the fingerprint `{r['after']['fingerprint']}` — the same as the "
               f"store rebuilt from the journal without the retracted fact (`{r['rebuilt']}`): "
               f"**{'exact' if r['after']['fingerprint'] == r['rebuilt'] else 'NOT exact'}**; the journal verifies: "
               f"**{r['verify']}**."]
    return "\n".join(md)


# ------------------------------------------------------------------------------------------------- handlers
def _run(agent, runs, seed, name):
    t0 = time.perf_counter()
    ep, rows = cr.run(agent, int(seed))
    runs = list(runs or []) + [{"name": name, "seed": int(seed), "steps": ep["steps"], "goals": len(ep["goals_done"]),
                                "s1": ep["s1"], "s2": ep["s2"], "refused": ep["refused"],
                                "ms": (time.perf_counter() - t0) * 1000}]
    status = (f"### {name}: {ep['steps']} steps, {len(ep['goals_done'])} of 6 goals — System 1 {ep['s1']}, "
              f"System 2 {ep['s2']}, refused {ep['refused']}")
    return agent, runs, status, runs_table(runs), map_svg(agent.env, rows), log_table(rows), knowledge_md(agent)


def first_run(seed):
    agent = cr.new_agent()
    return _run(agent, [], seed, f"run 1 in world {int(seed)} (empty memory)")


def again(agent, runs, seed):
    if agent is None:
        return first_run(seed)
    n = sum(r["seed"] == int(seed) for r in runs or []) + 1
    return _run(agent, runs, seed, f"run {n} in world {int(seed)} (same memory)")


def new_world(agent, runs, seed):
    if agent is None:
        agent = cr.new_agent()
    return _run(agent, runs, seed, f"world {int(seed)} (new world, same memory)")


def retract(agent, runs):
    if agent is None:
        return gr.update(), "Run the agent first."
    return gr.update(), knowledge_md(agent, cr.retraction_demo(agent))


def risk(episodes):
    t0 = time.perf_counter()
    out = cr.risk_arms(int(episodes))
    ms = (time.perf_counter() - t0) * 1000
    n = int(episodes)
    rows = "".join(
        f"<tr><td>{'protection (default)' if k == 'protect' else 'RiskBudget (justified risk)'}</td>"
        f"<td class='num'>{v['iron']} of {n}</td><td class='num'>{v['falls']}</td><td class='num'>{v['takes']}</td>"
        f"<td class='num'>{v['goals']:.2f}</td><td class='num'>{v['gate_blocked']}</td>"
        f"<td class='num'>{v['gate_broken']}</td><td class='num'>{v['replay']['ok']} of {v['replay']['decisions']}</td></tr>"
        for k, v in out.items())
    table = ("<table class='ag-table'><tr><th>agent</th><th class='num'>iron in</th><th class='num'>fell</th>"
             "<th class='num'>risky crossings</th><th class='num'>goals per episode</th>"
             "<th class='num'>gate blocked the bridge</th><th class='num'>crossed past the gate</th>"
             "<th class='num'>decisions replay</th></tr>" + rows + "</table>")
    held = all(v["gate_broken"] == 0 for v in out.values())
    md = (f"### {n} episodes per agent in {ms / 1000:.1f} s — the gate held in both: **{'yes' if held else 'NO'}**\n"
          f"With protection the agent fell {out['protect']['falls']} time(s), learned “the bridge: refuse”, and never "
          f"crossed again: iron in {out['protect']['iron']} of {n}. With `RiskBudget` it crossed "
          f"{out['risk']['takes']} times against the prediction, because the expected gain outweighed the estimated "
          f"risk, fell {out['risk']['falls']} times and got the iron in {out['risk']['iron']} of {n}. The written "
          "gate “no bridge without the stone pickaxe” is a hard check that no risk policy trades: every time it "
          "blocked the bridge, the bridge was not taken.")
    return md, table


def build():
    gr.Markdown(INTRO)
    agent, runs = gr.State(), gr.State([])
    gr.Markdown("#### 1 · Memory: the same world met again needs fewer slow decisions")
    with gr.Row():
        with gr.Column(scale=4, min_width=300):
            with gr.Row():
                seed = gr.Number(7, precision=0, minimum=0, maximum=10**6, label="world (seed)", scale=1, min_width=90)
                seed_new = gr.Number(8, precision=0, minimum=0, maximum=10**6, label="new world (seed)", scale=1,
                                     min_width=90)
            with gr.Row():
                b1 = gr.Button("▶ Run 1 (empty memory)", variant="primary")
                b2 = gr.Button("▶ Run again, same world")
            with gr.Row():
                b3 = gr.Button("🗺️ A new world, same memory")
                b4 = gr.Button("↩ Retract a person's fact")
            m_map = gr.HTML()
        with gr.Column(scale=7):
            m_status = gr.Markdown("Press **Run 1**: the agent starts with an empty memory and explores.")
            m_runs = gr.HTML()
            with gr.Accordion("Step log of the last run: who decided, and why", open=False):
                m_log = gr.HTML()
    m_know = gr.Markdown()
    outs = [agent, runs, m_status, m_runs, m_map, m_log, m_know]
    b1.click(first_run, seed, outs)
    b2.click(again, [agent, runs, seed], outs)
    b3.click(new_world, [agent, runs, seed_new], outs)
    b4.click(retract, [agent, runs], [m_status, m_know])

    gr.Markdown("#### 2 · Protection vs justified risk\n"
                "Another world: the iron lies only across a rope bridge from the start (red dashes) that breaks one "
                "time in three — the episode ends. An action model of the page's own (`OutcomeRates`: the outcome rate "
                "of each action) predicts “refuse” after a fall, with its rate and the number of crossings behind it. "
                "With **protection** (the default) a predicted refusal is never taken; with "
                "`RiskBudget(max_risk_per_episode=1.0, min_gain_ratio=1.0, min_support=1)` the agent crosses when the "
                "expected gain outweighs the estimated risk, within the episode's budget. A **written gate** stays hard "
                "in both: `km.agenda.gate(\"bridge_needs_the_tool\", lambda s: s[\"stone_pickaxe\"], blocks=[\"bridge\"])`.")
    with gr.Row():
        episodes = gr.Radio([10, 20, 30], value=10, label="episodes per agent", scale=3)
        rb = gr.Button("▶ Run both agents", variant="primary", scale=1)
    r_md = gr.Markdown()
    r_table = gr.HTML()
    rb.click(risk, episodes, [r_md, r_table])
