"""solvi plays the world map of Pokémon Red — System 1 and System 2, replayed in the browser.

A static Space (Gradio-Lite: Python in the visitor's browser). It shows two recorded runs of the same story: run 1
with an empty memory, run 2 with what run 1 left. No ROM and no emulator: the world is a recording of place names and
exits (data/world.json), the runs are solvi decision records (data/runs/), and everything shown is read or recomputed
from them here.

Local run with a normal Python (from spaces/pokemon):  PYTHONPATH=../../src python app.py"""
from __future__ import annotations

import html
import json
from functools import lru_cache

import gradio as gr
from pokeworld import record as rec
from pokeworld.agent import dispatcher, system2
from pokeworld.render import COLORS, layout, map_svg, strip_svg
from pokeworld.world import World
from solvi.core.knowledge.worldmap import WorldMap

REPO = "https://github.com/solvi-ai/solvi"
RUNS = rec.RUNS
WORLD = World.load()
POS = layout(WORLD)
GOALS = {o["id"]: o for o in WORLD.story}
MOVES = {r: rec._moves(RUNS / f"{r}.moves.jsonl") for r in rec.NAMES}
SUMMARY = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
RUN_LABEL = {"run1": "Run 1 — empty memory", "run2": "Run 2 — with what run 1 left"}
BY = {"s1": "System 1 (fast)", "s2": "System 2 (slow)", "human": "a person"}


@lru_cache(maxsize=2)
def journal(run):
    """The run's world map file and, per decision, how many journal entries the map had after it."""
    m = WorldMap(RUNS / f"{run}.map.json")
    upto, cut = {}, 0
    for i, r in enumerate(m.journal):
        step = str(r.get("step") or "")
        name, _, num = step.partition("#")
        if name == run and num.isdigit():
            upto[int(num)] = i + 1
        elif name < run:
            cut = i + 1
    last = cut
    for n in range(0, len(MOVES[run]) + 1):
        last = upto.get(n, last)
        upto[n] = last
    return m, upto


@lru_cache(maxsize=2)
def stored(run):
    return dispatcher(storage=rec.open_store(RUNS, run)).stored()


def plans_table(d):
    """System 2's candidates at a stored decision, each scored again by its catalog (the stored search kept the best)."""
    state = dict(d.s1.trace.init)
    s2 = system2()
    rows = []
    for p in state["view"]["plans"]:
        r = s2.ask({**state, "plan": p}, store=False)
        v = r.values.get("value")
        ok = r["exit"].status != "abstain"
        rows.append((v if isinstance(v, (int, float)) else -1, p, ok))
    rows.sort(key=lambda x: -x[0])
    best = d.s2.record.best if d.s2 is not None else None
    out = ["<table class='pk-t'><tr><th>plan</th><th>steps to it</th><th>value</th><th></th></tr>"]
    for v, p, ok in rows[:8]:
        mark = "◀ chosen" if best and p == best else ("" if ok else "first step not on offer")
        what = "known way to the goal" if p["kind"] == "route" else f"explore {html.escape(p['place'])} | {html.escape(p['exit'])}"
        out.append(f"<tr><td>{what}</td><td>{p['hops']}</td><td>{v:.3f}</td><td>{mark}</td></tr>")
    if len(rows) > 8:
        out.append(f"<tr><td colspan=4 class='pk-dim'>… {len(rows) - 8} more</td></tr>")
    out.append("</table>")
    return "".join(out)


def card(run, k):
    moves = MOVES[run]
    m = moves[k]
    g = GOALS[m["objective"]]
    offer = WORLD.on_offer(m["here"], m["stage"])
    color = COLORS.get(m["by"], "#999")
    lines = [f"<div class='pk-card'><div class='pk-head'><span class='pk-badge' style='background:{color}'>"
             f"{BY[m['by']]}</span> decision #{m['n']} of {len(moves)} · goal <b>{html.escape(m['objective'])}</b></div>",
             f"<div class='pk-dim'>{html.escape(g['goal'])}</div>",
             f"<p>At <b>{html.escape(m['here'])}</b>, exits on offer: "
             + ", ".join(f"<code>{html.escape(x)}</code>" for x in offer) + "</p>",
             f"<p>Took <code>{html.escape(str(m['exit']))}</code> → <b>{html.escape(str(m['to']))}</b>"
             + (" (it did not lead anywhere: tried again later in the story)" if m["to"] == m["here"] else "") + "</p>",
             f"<p><b>Why:</b> {html.escape(m['why'])} · {m['ms']:.2f} ms</p>"]
    if m["event"]:
        lines.append(f"<p class='pk-ev'>Story event: {html.escape(m['event'])}</p>")
    if m["surprise"]:
        lines.append(f"<p class='pk-sur'>Surprise: {html.escape(m['surprise'])} — the next decision goes to System 2.</p>")
    if m["by"] == "s2":
        d = stored(run)[k]
        lines.append(f"<p><b>System 1 stood back:</b> {html.escape(str(d.s1.values.get('stands_back')))}. "
                     f"<b>System 2</b> searched {d.s2.record.asked} plan(s) through its checks:</p>")
        lines.append(plans_table(d))
    elif m["by"] == "s1":
        lines.append("<p class='pk-dim'>No search: a route compiled by consolidation (or the only exit there is).</p>")
    lines.append("</div>")
    return "".join(lines)


def view(run, n):
    moves = MOVES[run]
    k = max(0, min(len(moves) - 1, int(n) - 1))
    m = moves[k]
    wm, upto = journal(run)
    known = wm.rebuild(upto=upto[m["n"]])
    trail = [x["here"] for x in moves[max(0, k - 7):k + 1]] + [m["to"]]
    targets = WORLD.targets(GOALS[m["objective"]]["targets"])
    svg = map_svg(WORLD, POS, known, here=m["to"], targets=targets, trail=trail,
                  title=f"{RUN_LABEL[run]} · the world map the player knows after decision #{m['n']}")
    return svg, card(run, k), strip_svg(moves, at=k)


def numbers_html():
    nums = rec.numbers()
    r1, r2 = nums["runs"]["run1"], nums["runs"]["run2"]
    rows = ["<table class='pk-t'><tr><th>goal</th><th>run 1 moves</th><th>System 1</th><th>System 2</th>"
            "<th>run 2 moves</th><th>System 1</th><th>System 2</th><th>shortest</th></tr>"]
    for o in nums["objectives"]:
        a, b = o["run1"], o["run2"]
        rows.append(f"<tr><td>{o['objective']}</td><td>{a['moves']}</td><td>{a['s1']}</td><td>{a['s2']}</td>"
                    f"<td>{b['moves']}</td><td>{b['s1']}</td><td>{b['s2']}</td><td>{o['shortest']}</td></tr>")
    short_total = sum(o["shortest"] for o in nums["objectives"])
    rows.append(f"<tr class='pk-tot'><td>all goals</td><td>{r1['decisions']}</td><td>{r1['s1']}</td><td>{r1['s2']}</td>"
                f"<td>{r2['decisions']}</td><td>{r2['s1']}</td><td>{r2['s2']}</td><td>{short_total}</td></tr></table>")
    t = (f"<p>Time inside the decisions (as recorded): run 1 — System 1 {r1['ms']['s1']:.0f} ms over {r1['s1']} decisions, "
         f"System 2 {r1['ms']['s2']:.0f} ms over {r1['s2']}; run 2 — System 1 {r2['ms']['s1']:.0f} ms over {r2['s1']}, "
         f"System 2 {r2['ms']['s2']:.0f} ms over {r2['s2']}. Run 1 tried {r1['blocked']} exits that did not lead anywhere "
         f"yet (water, a guard: they open later in the story); run 2 was surprised {r2['surprises']} times (story events "
         f"its routes did not know), and each surprise went to System 2.</p>")
    strips = "".join(f"<p><b>{RUN_LABEL[r]}</b> — each bar a decision (tall amber: System 2, short blue: System 1, "
                     f"lines: a new goal)</p>{strip_svg(MOVES[r])}" for r in rec.NAMES)
    head = (f"<h3>Run 2 made {r2['s2']} slow decisions instead of {r1['s2']}, and {r2['decisions']} moves instead of "
            f"{r1['decisions']} — the fewest possible ({short_total}).</h3>")
    return head + "".join(rows) + t + strips


def whole_map(r):
    wm, _ = journal(r)
    st = wm.stats()
    return map_svg(WORLD, POS, wm, title=f"After {RUN_LABEL[r].lower()}: {st['states']} places visited, "
                                         f"{st['confirmed']} exits walked, {st['unchecked']} not taken yet, "
                                         f"journal {st['journal']} entries (hash chain verified: {wm.verify()})")


def check():
    r = rec.replay()
    maps = ", ".join(f"{k}: journal {v['journal']} entries, verified {v['verify']}" for k, v in r["maps"].items())
    txt = (f"Replayed {r['replayed']} of {r['decisions']} stored decisions without the game (System 1's trace, the "
           f"dispatch, System 2's search and its winner's trace). World maps: {maps}. Move logs agree with the records: "
           f"{r['logs']}. The numbers agree with the logs: {r['numbers']}.")
    if r["mismatches"]:
        txt += "\nMismatches: " + "; ".join(map(str, r["mismatches"][:5]))
    return txt


def report(run):
    from pokeworld.agent import system1
    return str(system1().report(store=rec.open_store(RUNS, run)))


ABOUT = f"""
**What this is.** A player built with [solvi]({REPO}) walks the world of Pokémon Red twice — the game's first fifteen
goals, from Pallet Town to the fourth badge. One decision is "which exit do I take here?". Each goal names the place it
needs, never the way there.

- **System 1** is a catalog of two rules: take the exit of the route it remembers to the goal, or the only exit there
  is. No search; a fraction of a millisecond. It stands back when it remembers no route, when the remembered exit is not
  on offer, or when the last step surprised it.
- **System 2** is a search over the player's world map (`solvi.core.knowledge.worldmap`): the known way to the goal when the map has
  one, else every unexplored exit within reach, scored by expected value. A hard check keeps only plans whose first step
  is on offer. (An LLM can add a hint; it is off here.)
- **The dispatcher** (`solvi.core.dispatch`) asks System 1 first and System 2 when System 1 abstains. Every decision is a
  hash-chained record that replays without the game (tab "Check").
- **Consolidation**, after each goal, compiles System 1's routes from the map's confirmed claims: what System 2 found by
  deliberating becomes what System 1 does at once. Run 2 starts with run 1's memory.

**The world** is a recording of a real playthrough: place names, exits as a player sees them ("Leave north",
"Door at (12,11)") and where they led — 190 places, 447 exits. No ROM bytes and no graphics. Simplifications: a place
counts as reachable from the stage of the story at which the recorded player first reached it (before that, a try
leads nowhere); buildings the recording never entered are included only where the game's map tables show them to be
dead ends; walking, battles and menus inside a place are not part of this showcase.

**Numbers** are counted from the bundled records (tab "Numbers"); the example `examples/23_pokemon_world_map.py`
in the solvi repository prints the same numbers and can play the runs again, or check the recording against your own
ROM file. Pokémon is a trademark of Nintendo, Creatures and GAME FREAK; this project is not affiliated with them.
"""

CSS = """
.pk-card{border:1px solid #e2e8f0;border-radius:12px;padding:12px 14px;line-height:1.45}
.pk-head{font-size:1.02rem;margin-bottom:4px}.pk-badge{color:#fff;border-radius:8px;padding:2px 8px;font-weight:600}
.pk-dim{color:#64748b;font-size:.9rem}.pk-ev{color:#7c3aed}.pk-sur{color:#dc2626;font-weight:600}
.pk-t{border-collapse:collapse;font-size:.9rem}.pk-t td,.pk-t th{border-bottom:1px solid #e2e8f0;padding:3px 8px;text-align:left}
.pk-tot td{font-weight:700}
"""


GR6 = int(gr.__version__.split(".")[0]) >= 6       # Gradio 6 takes css= in launch(); Gradio-Lite 5.45 in Blocks()


def build():
    with gr.Blocks(title="solvi plays the world map of Pokémon Red", **({} if GR6 else {"css": CSS})) as demo:
        gr.Markdown("# solvi plays the world map of Pokémon Red — System 1 and System 2", elem_id="hero")
        gr.Markdown(f"Two recorded runs of the same story. Run 1 starts with an empty memory and deliberates "
                    f"({SUMMARY['runs']['run1']['s2']} slow decisions); run 2 starts with what run 1 consolidated and "
                    f"needs {SUMMARY['runs']['run2']['s2']}. Everything below is read from the records in your browser.")
        with gr.Tab("Replay"):
            with gr.Row():
                run = gr.Radio(list(rec.NAMES), value="run1", label="run",
                               info="run1: empty memory · run2: with what run 1 left")
                n = gr.Slider(1, len(MOVES["run1"]), value=1, step=1, label="decision")
            with gr.Row():
                prev_b, next_b, jump = gr.Button("◀ previous"), gr.Button("next ▶"), gr.Button("next System 2 decision ⏭")
            first = view("run1", 1)
            strip = gr.HTML(first[2])
            with gr.Row():
                with gr.Column(scale=3):
                    pic = gr.HTML(first[0])
                with gr.Column(scale=2):
                    info = gr.HTML(first[1])

            def on_run(r):
                return gr.Slider(1, len(MOVES[r]), value=1, step=1, label="decision"), *view(r, 1)

            def step(r, k, d):
                k = max(1, min(len(MOVES[r]), int(k) + d))
                return k, *view(r, k)

            def to_s2(r, k):
                ms = MOVES[r]
                nxt = next((i + 1 for i in range(int(k), len(ms)) if ms[i]["by"] != "s1"), int(k))
                return nxt, *view(r, nxt)
            run.change(on_run, run, [n, pic, info, strip])
            n.change(lambda r, k: view(r, k), [run, n], [pic, info, strip])
            prev_b.click(lambda r, k: step(r, k, -1), [run, n], [n, pic, info, strip])
            next_b.click(lambda r, k: step(r, k, 1), [run, n], [n, pic, info, strip])
            jump.click(to_s2, [run, n], [n, pic, info, strip])
        with gr.Tab("World map"):
            which = gr.Radio(list(rec.NAMES), value="run2", label="the world map after")
            whole = gr.HTML(whole_map("run2"))
            which.change(whole_map, which, whole)
        with gr.Tab("Numbers"):
            gr.HTML(numbers_html())
        with gr.Tab("Check and report"):
            b = gr.Button("Replay every stored decision", variant="primary")
            out = gr.Textbox(label="replay", lines=4)
            b.click(check, None, out)
            rr = gr.Radio(list(rec.NAMES), value="run2", label="System.report of")
            rep = gr.Textbox(report("run2"), label="system report", lines=22)
            rr.change(report, rr, rep)
        with gr.Tab("About"):
            gr.Markdown(ABOUT)
    return demo


demo = build()
demo.launch(**({"css": CSS} if GR6 else {}))       # Gradio-Lite mounts the app on launch(), as in the other Spaces
