"""solvi realms (browser edition): an endless turn-based strategy game whose factions are solvi decision systems.
Runs in the visitor's browser with Gradio-Lite (Pyodide): index.html mounts this file and realms/*.py.

No threads, subprocesses or network. "Endless" auto-play is a gr.Timer: every tick plays a chunk of turns and redraws.

Local run with a normal Python (from solvi/spaces/realms):  PYTHONPATH=../../src python app.py"""
from __future__ import annotations

import json
import os
import tempfile
import time

import gradio as gr
from realms import econ, health
from realms.engine import Game
from realms.render import card_html, factions_html, health_html, log_html, map_svg

REPO = "https://github.com/solvi-ai/solvi"
RECORD_EVERY = 25          # turns per health checkpoint in the browser
MAX_RECS = 240             # ring buffer of checkpoints shown in the health panel
SPEEDS = {"1 turn / tick": 1, "5 turns / tick": 5, "20 turns / tick": 20}
VARIANTS = {"learning value heads": "value", "L17 policy net + laws": "l17"}
LA_RATE_BROWSER = 10       # L17 + verified lookahead in the browser: a third of the headless budget (rollout-turns per turn)


def variant_of(label, la):
    v = VARIANTS.get(label, "value")
    return "l17_la" if v == "l17" and la else v


def new_ui(seed, label="learning value heads", la=False):
    v = variant_of(label, la)
    g = Game(seed=int(seed or 0), variant=v)
    if v == "l17_la":
        g.planner.la_rate = LA_RATE_BROWSER         # smaller lookahead budget than headless (saved with the game)
    return {"g": g, "recs": [], "alive_min": 99, "sel": None, "perf": "", "since": 0, "t_turns": [], "neg": 0}


def is_l17(p):
    return p is not None and getattr(p, "tot", None) is not None and "violations" in p.tot    # realms/l17.py


def play(ui, n):
    g = ui["g"]
    t0 = time.perf_counter()
    for _ in range(int(n)):
        g.play(1)
        ui["alive_min"] = min(ui["alive_min"], len(g.alive()))
        ui["neg"] += sum(1 for f in g.alive() if g.factions[f]["gold"] < 0)
        ui["since"] += 1
        if ui["since"] >= RECORD_EVERY:
            ui["recs"].append(health.record(g, ui["alive_min"], negative_treasury=ui["neg"]))
            ui["recs"] = ui["recs"][-MAX_RECS:]
            if is_l17(g.planner):
                ui["l17w"] = g.planner.stats(g)     # closes the planner's timing window (keeps it bounded)
            ui["alive_min"], ui["since"], ui["neg"] = 99, 0, 0
            g.m.reset()
    dt = (time.perf_counter() - t0) * 1000
    if n:
        ui["t_turns"] = (ui["t_turns"] + [dt / n])[-50:]
        ui["perf"] = (f"last chunk: <b>{int(n)} turn{'s' if n > 1 else ''} in {dt:.0f} ms</b> "
                      f"({dt / n:.1f} ms/turn in this browser; mean of recent chunks "
                      f"{sum(ui['t_turns']) / len(ui['t_turns']):.1f} ms/turn) · {g.m.total_decisions:,} solvi decisions so far")
    return ui


def turn_md(ui):
    g = ui["g"]
    alive = g.alive()
    return (f"### Turn {g.turn:,} · {len(alive)} factions alive · {g.spawned} ever spawned · {len(g.cities)} cities · "
            f"{sum(c['pop'] for c in g.cities.values())} population · {len(g.units)} units · {len(g.wars)} wars · seed {g.seed}")


def learn_md(g, ui=None):
    p = g.planner
    if is_l17(p):
        t = p.tot
        laws = ", ".join(f"{k} {v:,}" for k, v in sorted(t["law_removed"].items())) or "none yet"
        w = (ui or {}).get("l17w")
        xs = sorted(p.ms)
        if w and w["l17_n"]:
            ms = f"{w['l17_ms_med']:.2f} ms median, {w['l17_ms_p99']:.2f} ms p99 (last {RECORD_EVERY} turns)"
        else:
            ms = f"{xs[len(xs) // 2]:.2f} ms median, {xs[int(0.99 * (len(xs) - 1))]:.2f} ms p99" if xs else "—"
        la = (f" Verified lookahead (small budget, {p.la_rate} rollout-turns per turn): {t['lookaheads']:,} so far."
              if p.mode == "net_la" else "")
        return (f"**Adaptive faction = L17 policy net** (a small MLP per question, trained offline by policy iteration on "
                f"rollouts; the model proposes, deterministic code decides): laws filter the options before the net "
                f"(capital_guard, no_starve, upkeep_ok, war_min_length), an independent check re-verifies every answer "
                f"after it, hard checks still force. {t['decisions']:,} decisions, **{t['violations']} check violations**; "
                f"options removed by a law: {laws}. Decision time in this window: {ms}.{la} Click a unit or city of the "
                f"adaptive faction: its `why` lists the options with the net's scores, the laws and the check.")
    L = g.learner
    hs = g.systems["adaptive"][1].heads
    pts = [c for c in L.curve if c[1] is not None and c[2] is not None][-6:]
    cur = " · ".join(f"t{t}: {a:+.3f} vs {o:+.3f}" for t, a, o, _, _ in pts) or "not enough judged decisions yet"
    return (f"**Adaptive faction** (no rule for build, unit orders or war/peace: learned value heads, one ridge regression "
            f"per answer, chosen by value + UCB bonus; hard checks still veto): {L.observed:,} decisions judged "
            f"({L.own:,} its own, the rest observed from the other factions), {L.refits} refreshes, {L.explored} "
            f"ε-explorations{' — FROZEN (ablation)' if L.frozen else ''}. Heads: "
            + ", ".join(f"{q} {h.n:,}" for q, h in hs.items() if hasattr(h, "refresh")) +
            f". Build reward (log-score gain 20 turns later), adaptive vs others: {cur}")


def outputs(ui, refresh_choices=False):
    g = ui["g"]
    checks = health.check(ui["recs"])
    perf = ui["perf"] or "press a button to play"
    res = [ui, map_svg(g, ui["sel"]), turn_md(ui), factions_html(g), log_html(g),
           health_html(ui["recs"], checks, perf), card_html(g, ui["sel"]), learn_md(g, ui)]
    if refresh_choices:
        res.append(gr.update(choices=faction_choices(g), value=human_label(g)))
        res.append(gr.update(choices=other_choices(g)))
    else:
        res += [gr.update(), gr.update()]
    return res


def faction_choices(g):
    return ["nobody (AI only)"] + [f"{g.factions[f]['name']} ({g.factions[f]['pers']})" for f in g.alive()]


def human_label(g):
    if g.human is None or not g.factions[g.human]["alive"]:
        return "nobody (AI only)"
    f = g.factions[g.human]
    return f"{f['name']} ({f['pers']})"


def other_choices(g):
    return [g.factions[f]["name"] for f in g.alive() if f != g.human]


def on_new(seed, label="learning value heads", la=False):
    return outputs(new_ui(seed, label, la), refresh_choices=True)


def on_play(ui, n):
    if ui is None:
        ui = new_ui(0)
    return outputs(play(ui, n), refresh_choices=True)


def on_tick(ui, speed):
    if ui is None:
        return [gr.update()] * 10
    return outputs(play(ui, SPEEDS.get(speed, 5)))


def on_pick(ui, key):
    if ui is None:
        return gr.update(), gr.update(), gr.update()
    key = (key or "").strip()
    ui["sel"] = key or None
    return ui, map_svg(ui["g"], ui["sel"]), card_html(ui["g"], ui["sel"])


def on_human(ui, label):
    g = ui["g"]
    g.human = None
    for f in g.alive():
        if label == f"{g.factions[f]['name']} ({g.factions[f]['pers']})":
            g.human = f
            g.human_orders = {}
            g.note("human", f"you take over {g.factions[f]['name']}: you choose its builds and stances, solvi advises")
    return outputs(ui, refresh_choices=True)


def on_build(ui, option):
    g = ui["g"]
    sel = ui["sel"] or ""
    if g.human is None:
        msg = "Take over a faction first."
    elif not sel.startswith("city:"):
        msg = "Select one of your cities on the map first."
    else:
        msg = g.human_set_build(int(sel.split(":")[1]), option) or f"ordered: {option}"
    return [*outputs(ui), msg]


def on_stance(ui, other, stance):
    g = ui["g"]
    if g.human is None:
        msg = "Take over a faction first."
    else:
        target = next((f for f in g.alive() if g.factions[f]["name"] == other), None)
        if target is None:
            msg = "Pick a neighbour."
        else:
            g.human_orders.setdefault("stance", {})[str(target)] = stance
            msg = f"stance toward {other}: {stance} (applied on your next diplomacy turn, every 5 turns)"
    return [*outputs(ui), msg]


def on_save(ui):
    g = ui["g"]
    path = os.path.join(tempfile.gettempdir(), f"realms_seed{g.seed}_turn{g.turn}.json")
    with open(path, "w") as fh:
        fh.write(g.to_json())
    return path, f"saved turn {g.turn} ({os.path.getsize(path) / 1024:.0f} KB)"


def on_load(ui, file):
    if file is None:
        return [*outputs(ui), "choose a saved .json file"]
    path = file if isinstance(file, str) else getattr(file, "name", None) or file.get("path")
    with open(path) as fh:
        g = Game.from_dict(json.load(fh))
    ui = {"g": g, "recs": [], "alive_min": 99, "sel": None, "perf": f"loaded turn {g.turn}", "since": 0, "t_turns": [],
          "neg": 0}
    return [*outputs(ui, refresh_choices=True), f"loaded turn {g.turn}"]


CSS = """
.gradio-container {max-width: 1320px !important; margin: auto;}
#hero h1 {font-size: 2.0rem; margin-bottom: 0;}
.rm-mapwrap {width: 100%; overflow-x: auto;}
.rm-map {width: 100%; min-width: 560px; height: auto; border-radius: 10px; border: 1px solid var(--border-color-primary);
  display: block; cursor: pointer;}
.rm-click:hover {filter: brightness(1.15);}
.rm-tab {width: 100%; border-collapse: collapse; font-size: 0.82rem;}
.rm-tab td, .rm-tab th {border-top: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: left; vertical-align: top;}
.rm-sw {display: inline-block; width: 11px; height: 11px; border-radius: 3px; margin-right: 5px; vertical-align: -1px;}
.rm-badge {font-size: 0.7rem; background: #9333ea22; color: #9333ea; border: 1px solid #9333ea66; border-radius: 999px;
  padding: 0 6px; margin-left: 4px;}
.rm-you {background: #2563eb22; color: #2563eb; border-color: #2563eb66;}
.rm-log {max-height: 330px; overflow-y: auto; font-size: 0.8rem;}
.rm-ev {padding: 2px 0; border-bottom: 1px dashed var(--border-color-primary);}
.rm-t {opacity: 0.6; font-family: var(--font-mono);} .rm-k {font-weight: 700;}
.rm-sparks {display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; margin: 6px 0;}
.rm-spark {border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 4px 8px;}
.rm-st {font-size: 0.75rem;} .rm-sr {font-size: 0.68rem; opacity: 0.6;}
.rm-perf {font-size: 0.82rem; margin: 2px 0 4px;}
.rm-inv td:nth-child(4) {font-size: 0.72rem;}
.rm-skip {margin: 6px 0; font-size: 0.82rem; background: #f59e0b18; border: 1px solid #f59e0b55; border-radius: 8px; padding: 4px 8px;}
.rm-ent {font-size: 0.85rem; margin-bottom: 6px;}
.sv-card {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px; margin: 6px 0;
  background: var(--block-background-fill); font-size: 0.9rem; overflow-x: auto;}
.sv-title {font-weight: 700; font-size: 1.0rem; margin-bottom: 8px;}
.sv-ans {margin: 4px 0 8px;}
.sv-why {font-family: var(--font-mono); font-size: 0.8rem; opacity: 0.85; margin-top: 2px; word-break: break-word;}
.sv-badge {display: inline-block; border-radius: 999px; padding: 1px 9px; font-size: 0.74rem; font-weight: 600; margin: 2px 2px;}
.sv-ok {background: #16a34a22; color: #16a34a; border: 1px solid #16a34a66;}
.sv-forced {background: #dc262622; color: #ef4444; border: 1px solid #dc262688;}
.sv-abstain {background: #ca8a0422; color: #ca8a04; border: 1px solid #ca8a0466;}
.sv-dim {opacity: 0.7;}
.sv-hc {margin: 6px 0;}
.sv-facts {width: 100%; border-collapse: collapse; font-size: 0.76rem; margin-top: 6px;}
.sv-facts td {border-top: 1px solid var(--border-color-primary); padding: 2px 6px; vertical-align: top;}
.sv-facts td:first-child {font-family: var(--font-mono); white-space: nowrap;}
"""

INTRO = f"""
# 🏰 solvi realms
**An endless strategy game whose AI factions are [solvi]({REPO}) decision systems — running entirely in your browser.**
Four factions (builder, expansionist, warmonger, trader) plus one that **learns while it plays** share one catalog of small
Python functions; personality lives in the rule weights. Every build, order and war/peace choice is a solvi answer with its
reason, the strategist's plan and hard checks (the capital keeps a defender, no negative treasury, no settling next to an
enemy, no war below 80% of the enemy's strength). There is no win condition: fallen factions are replaced, deposits
regenerate, droughts, plagues and gold rushes keep coming. Press **▶ Endless** and watch the health panel.
"""

LEGEND = ("<span class='sv-dim'>Map: colored squares are cities (number = population, ★ capital, red dashes = under siege), "
          "dots are units (W warrior, A archer, S settler, K worker, C caravan), small circles are deposits "
          "(yellow food, green wood, grey stone, orange gold; fainter = depleted), orange dashed boxes are droughts. "
          "<b>Click a city or unit</b> for its last solvi decision.</span>")

THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="amber",
                       font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"])

with gr.Blocks(title="solvi realms", theme=THEME, css=CSS) as demo:
    gr.Markdown(INTRO, elem_id="hero")
    ui_state = gr.State(None)
    timer = gr.Timer(0.4, active=False)
    turn_line = gr.Markdown("### Starting …")
    with gr.Row():
        with gr.Column(scale=7, min_width=560):
            map_html = gr.HTML()
            gr.HTML(LEGEND)
            with gr.Row():
                btn_next = gr.Button("Next turn", variant="secondary")
                btn_10 = gr.Button("Play 10", variant="secondary")
                btn_100 = gr.Button("Play 100")
                btn_go = gr.Button("▶ Endless", variant="primary")
                btn_pause = gr.Button("⏸ Pause")
            with gr.Row():
                speed = gr.Radio(list(SPEEDS), value="5 turns / tick", label="endless speed", scale=3)
                seed = gr.Number(value=1, precision=0, label="seed", minimum=0, maximum=10**6, scale=1)
                btn_new = gr.Button("New world", scale=1)
            with gr.Row():
                variant = gr.Radio(list(VARIANTS), value="learning value heads", label="adaptive faction (applies to a new world)",
                                   scale=3)
                la_box = gr.Checkbox(False, label="L17 + verified lookahead (small budget, slower)", scale=2)
            factions = gr.HTML()
            learn = gr.Markdown()
        with gr.Column(scale=5, min_width=360):
            pick = gr.Textbox(label="selected (click the map, or type city:ID / unit:ID)", elem_id="pick", max_lines=1)
            card = gr.HTML()
            with gr.Accordion("Event log (newest first)", open=True):
                log = gr.HTML()
    with gr.Accordion("Health of the endless game (one checkpoint every 25 turns, the same invariants as sim.py)", open=True):
        health_panel = gr.HTML()
    with gr.Accordion("Take over a faction (you choose its builds and war/peace; solvi keeps advising)", open=False):
        with gr.Row():
            human = gr.Dropdown(["nobody (AI only)"], value="nobody (AI only)", label="you play", scale=2)
            build_opt = gr.Dropdown(econ.BUILD_OPTIONS, value="settler", label="build in the selected city", scale=2)
            btn_build = gr.Button("Order build", scale=1)
        with gr.Row():
            other = gr.Dropdown([], label="neighbour", scale=2)
            stance = gr.Radio(["war", "peace"], value="peace", label="stance", scale=2)
            btn_stance = gr.Button("Set stance", scale=1)
        human_msg = gr.Markdown()
    with gr.Accordion("Save / load (JSON)", open=False):
        with gr.Row():
            btn_save = gr.Button("Save game")
            save_file = gr.File(label="saved game", interactive=False)
            load_file = gr.File(label="load a saved game", file_types=[".json"], type="filepath")
        save_msg = gr.Markdown()

    OUTS = [ui_state, map_html, turn_line, factions, log, health_panel, card, learn, human, other]
    demo.load(on_new, seed, OUTS)
    btn_new.click(on_new, [seed, variant, la_box], OUTS)
    btn_next.click(lambda u: on_play(u, 1), ui_state, OUTS)
    btn_10.click(lambda u: on_play(u, 10), ui_state, OUTS)
    btn_100.click(lambda u: on_play(u, 100), ui_state, OUTS)
    btn_go.click(lambda: gr.Timer(active=True), None, timer)
    btn_pause.click(lambda: gr.Timer(active=False), None, timer)
    timer.tick(on_tick, [ui_state, speed], OUTS)
    pick.input(on_pick, [ui_state, pick], [ui_state, map_html, card])
    human.input(on_human, [ui_state, human], OUTS)
    btn_build.click(on_build, [ui_state, build_opt], OUTS + [human_msg])
    btn_stance.click(on_stance, [ui_state, other, stance], OUTS + [human_msg])
    btn_save.click(on_save, ui_state, [save_file, save_msg])
    load_file.upload(on_load, [ui_state, load_file], OUTS + [save_msg])

    gr.Markdown(f"<span class='sv-dim'>Built with [solvi]({REPO}) (`pip install solvi`). No server: this page runs Python in "
                "your browser with Gradio-Lite and Pyodide. Every decision is a solvi `Response`: `r[q].answer / .why / "
                ".status`, the strategist's `flow`, and a hash-chained `trace` that the card replays. The long headless "
                "runs (100 000 turns) are in the Space's README.</span>")

demo.launch()
