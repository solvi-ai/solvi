"""🔮 20 questions tab: think of an animal, solvi asks the most informative question and guesses (logic in games/akinator.py).

app.py calls `build()` inside `with gr.Tab("🔮 20 questions"):` and adds `CSS` to the page's CSS. No threads, no network."""
from __future__ import annotations

import html

import gradio as gr

from games import akinator as ak
from games.explain import why_html

CSS = """
.ak-card {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 16px 18px; margin: 4px 0 8px;
  background: var(--block-background-fill);}
.ak-qn {font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.7;}
.ak-q {font-size: 1.45rem; font-weight: 700; margin: 4px 0 6px; line-height: 1.3;}
.ak-emoji {font-size: 3rem; line-height: 1.1;}
.ak-sub {font-size: 0.88rem; opacity: 0.8;}
.ak-bars {display: flex; flex-direction: column; gap: 4px;}
.ak-bar {display: grid; grid-template-columns: 26px minmax(80px, 140px) 1fr 48px; gap: 6px; align-items: center; font-size: 0.86rem;}
.ak-track {height: 10px; border-radius: 5px; background: var(--border-color-primary); overflow: hidden;}
.ak-fill {height: 100%; border-radius: 5px; background: var(--color-accent);}
.ak-path {margin: 0; padding-left: 20px; font-size: 0.86rem;}
.ak-path li {margin: 3px 0;}
.ak-ans {display: inline-block; border-radius: 999px; padding: 0 8px; font-size: 0.76rem; font-weight: 600;
  border: 1px solid var(--border-color-primary);}
.ak-alt {width: 100%; border-collapse: collapse; font-size: 0.8rem; margin-top: 6px;}
.ak-alt td, .ak-alt th {border-top: 1px solid var(--border-color-primary); padding: 3px 6px; text-align: left;}
.ak-answers button {min-width: 90px !important;}
"""

ANSWER_BUTTONS = [("✅ Yes", "yes"), ("👍 Probably", "probably"), ("🤷 Don't know", "don't know"),
                  ("👎 Probably not", "probably not"), ("❌ No", "no")]


def _emoji(kb, name):
    return kb.emoji[kb.names.index(name)] if name in kb.names else "✨"


def card_html(g):
    kb = g.kb
    if g.over:
        if g.result == "won":
            return ("<div class='ak-card'><div class='ak-qn'>solved</div>"
                    f"<div class='ak-emoji'>{_emoji(kb, g.solved)}</div><div class='ak-q'>🎉 {html.escape(g.solved)}! "
                    f"I got it in {g.q_count} questions.</div><div class='ak-sub'>Press <b>New game</b> and think of another animal.</div></div>")
        return ("<div class='ak-card'><div class='ak-qn'>I give up</div><div class='ak-q'>😵 You win! Which animal was it?</div>"
                "<div class='ak-sub'>Tell me below and I learn it instantly: a direct update of my table, no retraining.</div></div>")
    kind, what = g.pending
    if kind == "ask":
        return (f"<div class='ak-card'><div class='ak-qn'>question {g.q_count + 1} of {g.max_questions}</div>"
                f"<div class='ak-q'>{html.escape(ak.ATTR_TEXT[what])}</div>"
                f"<div class='ak-sub'>{html.escape(ak.explain_question(g.resp))}</div></div>")
    p = g.resp.values["top_prob"]
    why = ("the questions ran out" if g.resp.values["questions_left"] <= 0 else
           "no question can split the candidates any more" if g.resp.values["best_gain"] < ak.MIN_GAIN else
           f"confidence {p * 100:.0f}% ≥ {ak.GUESS_AT * 100:.0f}%")
    tries = f" (guess {len(g.wrong) + 1} of {ak.MAX_GUESSES})" if g.wrong else ""
    return (f"<div class='ak-card'><div class='ak-qn'>my guess after {g.q_count} questions{tries}</div>"
            f"<div class='ak-emoji'>{_emoji(g.kb, what)}</div><div class='ak-q'>Is it a {html.escape(what)}?</div>"
            f"<div class='ak-sub'>{p * 100:.0f}% sure — I guess now because {why}.</div></div>")


def shortlist_html(g):
    v = g.resp.values if g.resp is not None else None
    if v is None or "shortlist" not in v:
        return ""
    rows = []
    for name, p in v["shortlist"]:
        rows.append(f"<div class='ak-bar'><span>{_emoji(g.kb, name)}</span><span>{html.escape(name)}</span>"
                    f"<div class='ak-track'><div class='ak-fill' style='width:{max(1.0, p * 100):.1f}%'></div></div>"
                    f"<span>{p * 100:.1f}%</span></div>")
    return (f"<div class='sv-card'><div class='sv-title'>Shortlist — {v['live_candidates']} live candidates of "
            f"{len(g.kb.names)} (at least 1% as likely as the leader)</div><div class='ak-bars'>{''.join(rows)}</div></div>")


def path_html(g):
    if not g.path:
        return "<div class='sv-card sv-dim'>The path of questions appears here.</div>"
    items = []
    for i, s in enumerate(g.path, 1):
        y, n, u = s["split"]
        items.append(f"<li><b>{html.escape(s['text'])}</b> <span class='ak-ans'>{html.escape(s['answer'])}</span> "
                     f"<span class='sv-dim'>split {y}/{n}{f'/{u}?' if u else ''}, expected {s['bits']:.2f} bits · "
                     f"candidates {s['live_before']} → {s['live_after']}</span></li>")
    wrong = ""
    if g.wrong:
        wrong = "<div class='sv-dim'>wrong guesses: " + ", ".join(html.escape(g.kb.names[i]) for i in g.wrong) + "</div>"
    return f"<div class='sv-card'><div class='sv-title'>Path of questions</div><ol class='ak-path'>{''.join(items)}</ol>{wrong}</div>"


def why_card(g, note=""):
    r = g.resp
    if r is None:
        return ""
    v = r.values
    alt = "".join(f"<tr><td>{html.escape(ak.ATTR_TEXT[x['attr']])}</td><td>{x['bits']:.3f}</td>"
                  f"<td>{x['n_yes']} / {x['n_no']} / {x['n_unsure']}</td></tr>"
                  for x in v["question_gains"][:5])
    title = "Why this move"
    if r["move"].status == "forced":
        title = (f"🛑 hard check confident_enough fired: you asked me to guess, but the top candidate is only "
                 f"{v['top_prob'] * 100:.0f}% (< {ak.GUESS_AT * 100:.0f}%) with {v['questions_left']} questions left, "
                 "so I must ask another question")
    body = why_html(r, ak.cat, ["move", "ask_about", "guess"], show_facts=False, title=title)
    extra = (f"<div class='sv-card'><div class='sv-title'>Best questions right now (expected information gain)</div>"
             f"<table class='ak-alt'><tr><th>question</th><th>bits</th><th>yes / no / unsure among live</th></tr>{alt}</table>"
             f"<div class='sv-dim' style='margin-top:6px'>facts this turn: live_candidates = {v['live_candidates']}, "
             f"top_prob = {v['top_prob']}, best_gain = {v['best_gain']}, questions_left = {v['questions_left']} · "
             f"{r.ms:.1f} ms</div>"
             f"<details><summary>the strategist's plan (r.flow)</summary><pre style='white-space:pre-wrap;font-size:0.75rem'>"
             f"{html.escape(str(r.flow))}</pre></details></div>")
    return (note or "") + body + extra


def _outs(g, note="", teach_msg=None):
    asking = not g.over and g.pending and g.pending[0] == "ask"
    guessing = not g.over and g.pending and g.pending[0] == "guess"
    teach_open = g.over and g.result == "lost"
    return (card_html(g), gr.update(visible=bool(asking)), gr.update(visible=bool(guessing)),
            gr.update(visible=bool(teach_open)), shortlist_html(g), why_card(g, note), path_html(g), g,
            teach_msg if teach_msg is not None else gr.update(),
            gr.update(choices=sorted(g.kb.names), value=None))


def _ensure(g, kb):
    return g if g is not None else ak.new_game(kb.copy())


def do_new(g, kb):
    return _outs(ak.new_game(kb.copy()), teach_msg="")


def do_answer(label, g, kb):
    g = _ensure(g, kb)
    if not g.over and g.pending and g.pending[0] == "ask":
        ak.answer(g, label)
    return _outs(g)


def do_guess_now(g, kb):
    g = _ensure(g, kb)
    if not g.over and g.pending and g.pending[0] == "ask":
        ak.next_move(g, wants_guess=True)
    return _outs(g)


def do_verdict(correct, g, kb):
    g = _ensure(g, kb)
    note = ""
    if not correct and g.pending and g.pending[0] == "guess":
        note = f"<div class='sv-card sv-dim'>Not a {html.escape(g.pending[1])}: it drops out, and I continue.</div>"
    ak.guess_result(g, correct)
    return _outs(g, note)


def do_teach(name, g, kb):
    g = _ensure(g, kb)
    name = (name or "").strip()
    if not name:
        return (*_outs(g, teach_msg="Type or pick the animal first."), kb)
    kind, ms = ak.teach(kb, name, g.answers)           # the session's table: every later game uses it
    g.kb = kb
    msg = (f"**Learned in {ms:.2f} ms** — a direct update of the knowledge base, nothing retrained: " +
           (f"*{name}* is a new row ({len(g.answers)} answered questions; the rest stay unknown). "
            if kind == "new" else f"*{name}* got one vote per answered question, so its table values moved towards "
                                  "your answers. ") +
           f"The table now has {len(kb.names)} animals. Press **New game** and think of it again.")
    out = list(_outs(g, teach_msg=msg))
    out[3] = gr.update(visible=True)
    return (*out, kb)


def build():
    g0 = ak.new_game()
    gr.Markdown(
        f"Think of an animal ({len(g0.kb.names)} in my table, {len(ak.ATTR_KEYS)} yes/no facts each). Each turn a solvi "
        "catalog computes the probability of every animal from your answers, the **expected information gain** of every "
        "question not asked yet, and picks the best one. Answers are soft evidence: one wrong answer lowers an animal's "
        f"weight instead of removing it. **Hard check `confident_enough`:** I never guess below {ak.GUESS_AT * 100:.0f}% "
        "confidence unless the questions ran out — press *Guess now!* early to watch it refuse. If I lose, teach me the "
        "animal and I learn it in a fraction of a millisecond.")
    game = gr.State(None)
    kb = gr.State(g0.kb.copy())
    with gr.Row():
        with gr.Column(scale=5, min_width=300):
            card = gr.HTML(card_html(g0))
            with gr.Row(elem_classes="ak-answers") as ans_row:
                ans_btns = [gr.Button(lbl, variant="primary" if val == "yes" else "secondary", min_width=90)
                            for lbl, val in ANSWER_BUTTONS]
            with gr.Row(visible=False) as guess_row:
                right = gr.Button("🎉 Yes, that's it!", variant="primary")
                wrong = gr.Button("🙅 No, keep going")
            with gr.Group(visible=False) as teach_box:
                teach_dd = gr.Dropdown(sorted(g0.kb.names), label="Which animal was it? Pick one or type a new name",
                                       allow_custom_value=True, value=None)
                teach_btn = gr.Button("🧠 Learn it", variant="primary")
            teach_msg = gr.Markdown()
            with gr.Row():
                guess_now = gr.Button("🎯 Guess now!")
                new = gr.Button("🔄 New game")
            gr.Markdown(f"<span class='sv-dim'>Self-play benchmark (native Python, every animal once): truthful answers → "
                        "right on the first guess 145/145, 10.0 questions on average; 10% of answers flipped → right within "
                        "3 guesses 93%, first guess 81%, 14.5 questions.</span>")
        with gr.Column(scale=6):
            shortlist = gr.HTML(shortlist_html(g0))
            path = gr.HTML(path_html(g0))
            why = gr.HTML(why_card(g0))
    outs = [card, ans_row, guess_row, teach_box, shortlist, why, path, game, teach_msg, teach_dd]
    for b, (_, val) in zip(ans_btns, ANSWER_BUTTONS):
        b.click(lambda g, k, val=val: do_answer(val, g, k), [game, kb], outs)
    guess_now.click(do_guess_now, [game, kb], outs)
    right.click(lambda g, k: do_verdict(True, g, k), [game, kb], outs)
    wrong.click(lambda g, k: do_verdict(False, g, k), [game, kb], outs)
    new.click(do_new, [game, kb], outs)
    teach_btn.click(do_teach, [teach_dd, game, kb], outs + [kb])
