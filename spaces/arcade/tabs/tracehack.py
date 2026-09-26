"""🕵️ Hack the trace tab: solvi makes a loan decision and publishes its hash-chained trace; the player edits it to flip the
answer, and the verifier shows which defence catches the forgery (logic in games/tracehack.py).

app.py calls `build()` inside `with gr.Tab("🕵️ Hack the trace"):` and appends `CSS` to the page's CSS. The attempts /
successes counter lives in the visitor's browser (gr.BrowserState → localStorage)."""
from __future__ import annotations

import html
import json

import gradio as gr

from games import tracehack as th
from games.explain import why_html

DEFAULT_APPLICANT = next(iter(th.APPLICANTS))
DEFAULT_LEVEL = next(iter(th.LEVELS))

CSS = """
.th-receipt {font-family: var(--font-mono); font-size: 0.8rem; line-height: 1.5; word-break: break-all;}
.th-verdict {font-size: 1.05rem; font-weight: 700; margin-bottom: 6px;}
.th-checks {list-style: none; padding: 0; margin: 6px 0;}
.th-checks li {margin: 3px 0; font-size: 0.88rem;}
.th-checks li .th-msg {display: block; font-family: var(--font-mono); font-size: 0.78rem; opacity: 0.9; margin-left: 22px;}
.th-table {width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 6px;}
.th-table th, .th-table td {border-top: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: left; vertical-align: top;}
.th-table td.th-mono {font-family: var(--font-mono);}
.th-table tr.th-bad td {background: #dc262618;}
.th-table td.th-edited {background: #f59e0b30; font-weight: 700;}
.th-counter {font-size: 0.95rem;}
"""

WHY = """
**Why this matters.** When an AI system decides something about a person (a loan, a claim, a diagnosis), someone will
later ask *"why, and is that really what the system computed?"* A log can be edited. A solvi trace is harder to fake:
every step records the hashes of its inputs and of its value, and points to the hash of the step before it, starting
from the hash of the input. `trace.replay(catalog)` re-executes every step from its recorded inputs, so it catches a
changed value even when all hashes were recomputed. What replay cannot know is *which* input was the real one; for that
the system publishes a short receipt (the input hash and the final hash) at decision time, to the applicant and to an
append-only log. Together: the trace explains the decision, replay proves the explanation is the computation, the
receipt proves it is the same computation.

| level | what you do | what catches it |
|---|---|---|
| 1 | edit a value | the record's own hash; replay |
| 2 | edit values and rehash those records | the hash chain (next record's `prev`); replay |
| 3 | edit values and rebuild the whole chain | replay: the value ≠ the value recomputed from its inputs |
| 4 | change an input, re-run honestly, rebuild the chain | the published receipt (a different input hash) |
| bonus | delete a step so replay cannot recompute its reader | the planned flow (and the receipt) |
"""


def _receipt_html(receipt, applicant):
    return (f"<div class='sv-card'><div class='sv-title'>📜 Published receipt (at decision time, for {html.escape(applicant)})"
            f"</div><div class='th-receipt'>answer: <b>{receipt['answer']}</b><br>input hash: {receipt['init_hash']}<br>"
            f"final hash: {receipt['head']}<br>steps: {receipt['steps']}</div>"
            "<div class='sv-dim' style='font-size:0.8rem;margin-top:4px'>The goal: make the trace below say "
            "<b>approve</b> and pass every check.</div></div>")


def _level_md(level):
    task, defence = th.LEVELS[level]
    return f"**Your task.** {task}\n\n<details><summary>Which defence catches it?</summary>\n\n{defence}\n</details>"


def _counter_md(counts):
    counts = counts or {}
    return (f"<div class='th-counter'>🕵️ In this browser: <b>{counts.get('attempts', 0)}</b> forgery attempts · "
            f"<b>{counts.get('successes', 0)}</b> successes</div>")


def _table(d, original, bad_steps):
    orig = {r.get("step"): r for r in original["records"]}
    rows = []
    for r in d["records"]:
        o = orig.get(r.get("step"), {})
        bad = r.get("step") in bad_steps
        cells = []
        for key, mono in (("step", False), ("kind", False), ("name", True), ("value", True), ("prev", True), ("hash", True)):
            v = r.get(key)
            txt = json.dumps(v, ensure_ascii=False) if key == "value" else str(v)
            if key in ("prev", "hash"):
                txt = txt[:10] + "…"
            edited = o.get(key) != v
            cls = " ".join(x for x in ("th-mono" if mono else "", "th-edited" if edited else "") if x)
            cells.append(f"<td class='{cls}'>{html.escape(txt)}</td>")
        status = "❌ " + html.escape("; ".join(m.split(": ", 1)[-1] for m in bad_steps[r.get("step")])) if bad else "✅"
        rows.append(f"<tr class='{'th-bad' if bad else ''}'>{''.join(cells)}<td>{status}</td></tr>")
    init_edit = d["init"] != original["init"]
    head = ("<tr><th>step</th><th>kind</th><th>name</th><th>value</th><th>prev</th><th>hash</th><th>replay</th></tr>")
    note = ("<div class='sv-dim' style='font-size:0.8rem'>amber = changed from the published trace"
            + (" · <b>the input (init) was changed too</b>" if init_edit else "") + "</div>")
    return f"<table class='th-table'>{head}{''.join(rows)}</table>{note}"


def _verdict_html(v, original):
    if v.get("parse_error"):
        return f"<div class='sv-err'><b>Cannot read the trace:</b> {html.escape(v['parse_error'])}</div>"
    d = v["doc"]
    if not v["tampered"]:
        head = "✅ The published trace is untouched: every check passes, the answer is <b>reject</b>. Now try to forge it."
    elif v["ok"] and v["changed"]:
        head = f"🏆 Every check passes and the answer is now <b>{html.escape(str(v['answer']))}</b>. You beat it (please report how)!"
    elif v["ok"]:
        head = "✅ Every check passes, but the answer is still <b>reject</b>: nothing was forged."
    else:
        label, msg = v["first"]
        head = (f"🚨 Forgery caught by <b>{label}</b>: {html.escape(msg)}"
                + ("" if v["changed"] else " <span class='sv-dim'>(and the answer did not even change)</span>"))
    items = []
    for key, label, desc in th.CHECKS:
        msgs = v["checks"][key]
        mark = "❌" if msgs else "✅"
        src = "solvi replay" if key in ("record", "chain", "inputs", "recompute") else "arcade verifier"
        items.append(f"<li>{mark} <b>{label}</b> <span class='sv-dim'>— {desc} ({src})</span>" +
                     "".join(f"<span class='th-msg'>{html.escape(m)}</span>" for m in msgs[:4]) + "</li>")
    replay = ("<span class='sv-badge sv-ok'>trace.replay(catalog): no mismatches</span>" if v["replay_ok"] else
              "<span class='sv-badge sv-forced'>trace.replay(catalog): mismatches</span>")
    return (f"<div class='sv-card'><div class='th-verdict'>{head}</div>{replay} "
            f"<span class='sv-dim'>the trace claims: <b>{html.escape(str(v['answer']))}</b></span>"
            f"<ul class='th-checks'>{''.join(items)}</ul>{_table(d, original, v['bad_steps'])}</div>")


# --------------------------------------------------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------------------------------------------------

def new_decision(applicant):
    resp, d, receipt = th.decide(applicant)
    title = f"solvi's decision for {th.APPLICANTS[applicant]['applicant']}: {resp['decision'].answer}"
    card = why_html(resp, th.cat, ["decision"], title=title)
    first = _verdict_html(th.verify(th.dumps(d), d, receipt), d)
    return card, _receipt_html(receipt, th.APPLICANTS[applicant]["applicant"]), th.dumps(d), first, d, receipt


def do_verify(text, original, receipt, counts):
    counts = dict(counts or {"attempts": 0, "successes": 0})
    v = th.verify(text, original, receipt)
    if not v.get("parse_error") and v["tampered"]:
        counts["attempts"] = counts.get("attempts", 0) + 1
        if v["ok"] and v["changed"]:
            counts["successes"] = counts.get("successes", 0) + 1
    return _verdict_html(v, original), counts, _counter_md(counts)


def _helper(fn):
    def run(text, *args):
        try:
            return fn(text, *args), gr.update()
        except th.TraceFormatError as e:
            return gr.update(), f"<div class='sv-err'><b>The helper could not run:</b> {html.escape(str(e))}</div>"
    return run


def build():
    gr.Markdown(
        "solvi decides a loan and publishes its **hash-chained trace**: every step with the hashes of its inputs, its "
        "value, the hash of the step before (`prev`) and its own `hash`. **Your challenge:** edit the trace so it says "
        "`approve` and still passes `trace.replay(catalog)`. You may change anything, and the forger's helpers below "
        "rebuild hashes for you. Pick a level for a hint, or press *Show me the attack* to see it done and caught.")
    resp0, d0, receipt0 = th.decide(DEFAULT_APPLICANT)
    original = gr.State(d0)
    receipt = gr.State(receipt0)
    counts = gr.BrowserState({"attempts": 0, "successes": 0}, storage_key="solvi_arcade_tracehack_counter",
                             secret="solvi-arcade-tracehack-v1")
    with gr.Row():
        with gr.Column(scale=5, min_width=320):
            with gr.Row(equal_height=True):
                applicant = gr.Dropdown(list(th.APPLICANTS), value=DEFAULT_APPLICANT, label="applicant", scale=3)
                new_btn = gr.Button("New decision", scale=1, min_width=110)
            decision = gr.HTML(why_html(resp0, th.cat, ["decision"],
                                        title=f"solvi's decision for {th.APPLICANTS[DEFAULT_APPLICANT]['applicant']}: "
                                              f"{resp0['decision'].answer}"))
            receipt_box = gr.HTML(_receipt_html(receipt0, th.APPLICANTS[DEFAULT_APPLICANT]["applicant"]))
            level = gr.Radio(list(th.LEVELS), value=DEFAULT_LEVEL, label="level")
            level_md = gr.Markdown(_level_md(DEFAULT_LEVEL))
            attack_btn = gr.Button("🎬 Show me the attack (applies it to the trace)")
            counter = gr.HTML(_counter_md(None))
        with gr.Column(scale=7):
            editor = gr.Code(value=th.dumps(d0), language="json", lines=18, max_lines=30, interactive=True,
                             label="the published trace (edit anything)")
            with gr.Row(equal_height=True):
                step_no = gr.Number(value=5, precision=0, minimum=1, maximum=20, label="step N", scale=1, min_width=80)
                rehash_btn = gr.Button("#️⃣ Rehash step N", scale=2)
                chain_btn = gr.Button("🔗 Recompute the whole hash chain", scale=3)
            with gr.Row():
                rerun_btn = gr.Button("🔁 Re-run the steps from the inputs")
                reset_btn = gr.Button("↩ Reset to the published trace")
            verify_btn = gr.Button("🔍 Verify: replay the trace", variant="primary")
            verdict = gr.HTML(_verdict_html(th.verify(th.dumps(d0), d0, receipt0), d0))
    with gr.Accordion("Why this matters, and which defence catches what", open=False):
        gr.Markdown(WHY)

    new_btn.click(new_decision, applicant, [decision, receipt_box, editor, verdict, original, receipt])
    applicant.change(new_decision, applicant, [decision, receipt_box, editor, verdict, original, receipt])
    level.change(_level_md, level, level_md)
    attack_btn.click(lambda lv, o: th.attack(lv, o), [level, original], editor).then(
        do_verify, [editor, original, receipt, counts], [verdict, counts, counter])
    rehash_btn.click(_helper(lambda t, n: th.rehash_step(t, int(n or 0))), [editor, step_no], [editor, verdict])
    chain_btn.click(_helper(th.rechain), editor, [editor, verdict])
    rerun_btn.click(_helper(th.rerun), editor, [editor, verdict])
    reset_btn.click(lambda o: th.dumps(o), original, editor)
    verify_btn.click(do_verify, [editor, original, receipt, counts], [verdict, counts, counter])
    gr.on(triggers=None, fn=_counter_md, inputs=[counts], outputs=counter)   # on page load (reads localStorage) and on change
