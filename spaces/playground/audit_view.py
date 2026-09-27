"""The Audit panel: sandbox.serialize()'s "audit" (Response.audit().to_dict(), solvi 0.4) rendered as HTML.

Per answer: the chain given → computed → quoted (offsets, the span highlighted in its text) → decided (model, probabilities) →
learned (head, fingerprint) → checks / rule / constraints → answer; the safeguards that fired; and the share of the support
that is deterministic vs from models (a small bar). Plus feasibility of the answers under the constraints and, when the
System is known, its lifetime safeguard counts (System.stats). Pure Python, no gradio: the styles live in app.CSS."""
from __future__ import annotations

LABEL = {"grounding": "grounding rejected", "outside_options": "outside the options", "low_confidence": "low confidence",
         "validator": "rejected by validate", "hard_check": "hard check decided", "constraint_repair": "constraint repair",
         "fallback": "fallback producer"}
STAT_ROWS = [("asks", "asks"), ("answers", "answers"), ("abstained", "abstained"), ("model_outputs", "model outputs"),
             ("grounding_rejected", "grounding rejected"), ("outside_options", "outside the options"), ("rule_abstained", "rule abstained"),
             ("low_confidence", "low confidence"), ("validator_rejected", "rejected by validate"),
             ("forced_by_hard_check", "hard check decided"), ("constraint_repairs", "constraint repair"),
             ("fallbacks", "fallback producer")]


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def fmt_answer(a):
    if a is None:
        return "—"
    if isinstance(a, (list, tuple)):
        return ", ".join(map(str, a)) if a else "∅ (none)"
    return str(a)


def _probs(p, top=4):
    if not p:
        return ""
    items = sorted(p.items(), key=lambda t: -t[1])[:top]
    return '<span class="aud-probs">' + "".join(
        f'<span class="aud-p"><span class="aud-pbar" style="width:{max(2, 44 * float(v)):.0f}px"></span>'
        f'{esc(k)} {float(v):.2f}</span>' for k, v in items) + "</span>"


def _model(m):
    return f' <span class="aud-model" title="model type, id and fingerprint">{esc(m)}</span>' if m else ""


def _row(kind, label, body):
    return f'<li class="aud-row k-{kind}"><span class="aud-tag">{esc(label)}</span><span class="aud-body">{body}</span></li>'


def _rejected(d):
    return f'<b>{esc(d["name"])}</b> <span class="aud-bad">REJECTED</span> — {esc(d["error"])}'


def share_bar(d):
    n = sum(d.get("counts", {}).values())
    det = d.get("deterministic", n)
    pct = 100.0 if n == 0 else 100.0 * det / n
    parts = [f"{v} {k.replace('_', ' ')}" for k, v in d.get("counts", {}).items() if v]
    fuzzy = n - det
    return (f'<div class="aud-share" title="{esc(", ".join(parts))}"><div class="aud-sbar">'
            f'<span class="aud-det" style="width:{pct:.1f}%"></span><span class="aud-fuz" style="width:{100 - pct:.1f}%"></span>'
            f'</div><span class="aud-stxt"><b>{pct:.0f}% deterministic</b> · {det} deterministic'
            f'{f", {fuzzy} from models" if fuzzy else ""} ({esc(", ".join(parts) or "no items")})</span></div>')


def answer_card(d):
    status = d["status"]
    who = (f' ← {esc(d["provenance"])}' if d.get("provenance") else "") + (f' by {esc(d["source"])}' if d.get("source") else "")
    rows = []
    if d["given"]:
        rows.append(_row("given", "given", "; ".join(f'<b>{esc(g["name"])}</b> = <code>{esc(g["value"])}</code>'
                                                   for g in d["given"])))
    for c in d["computed"]:
        rows.append(_row("computed", "computed", _rejected(c) if c.get("error") else
                         f'<b>{esc(c["name"])}</b> = <code>{esc(c["value"])}</code>'))
    for q in d["quoted"]:
        label = "quoted by model" if q.get("model") else "quoted"
        if q.get("error"):
            rows.append(_row("rejected", label, _rejected(q) + _model(q.get("model"))))
            continue
        if q["start"] == q["end"]:                     # an empty quote: the extractor found nothing to cite
            rows.append(_row("quoted", label, f'<b>{esc(q["name"])}</b> = <code>{esc(q["value"])}</code> '
                                              f'<span class="aud-off">nothing found (empty quote)</span>{_model(q.get("model"))}'))
            continue
        how = {True: "literal", False: "derived from the text", None: "converted from the text"}[q.get("match")]
        conf = f' · confidence {q["confidence"]:.2f}' if q.get("confidence", 1) < 1 else ""
        body = (f'<b>{esc(q["name"])}</b> = <code>{esc(q["value"])}</code> '
                f'<span class="aud-off">{esc(q["source"])}[{q["start"]}:{q["end"]}]</span> '
                f'<span class="aud-ok">{how}</span>{conf}{_model(q.get("model"))}')
        if q.get("context"):
            b, m, a = q["context"]
            body += f'<div class="aud-snip">{esc(b)}<mark>{esc(m)}</mark>{esc(a)}</div>'
        rows.append(_row("quoted", label, body))
    for x in d["decided"]:
        rows.append(_row("rejected" if x.get("error") else "decided", "decided",
                         (_rejected(x) if x.get("error") else f'<b>{esc(x["name"])}</b> = <code>{esc(x["value"])}</code> '
                          + _probs(x.get("probs"))) + _model(x.get("model"))))
    for x in d["learned"]:
        rows.append(_row("learned", x.get("provenance") or "learned",
                         (_rejected(x) if x.get("error") else f'<b>{esc(x["name"])}</b> = <code>{esc(x["value"])}</code> '
                          + _probs(x.get("probs"))) + _model(x.get("model"))))
    for c in d["checks"]:
        tag = ("hard" if c["hard"] else "soft") + (", decides the answer" if c.get("decides") else "")
        val = "—" if c.get("value") is None else c["value"]
        bad = c.get("value") is False
        rows.append(_row("check", "check", f'<b>{esc(c["name"])}</b> = <span class="{"aud-bad" if bad else "aud-ok"}">'
                                           f'{esc(val)}</span> ({esc(tag)})' + (f' — {esc(c["error"])}' if c.get("error") else "")))
    if d.get("rule"):
        r = d["rule"]
        rows.append(_row("rule", "rule", f'<b>{esc(r["name"])}</b> ({esc(r.get("provenance") or "computed")})'
                         + _model(r.get("model")) + (f' — not computed: {esc(r["error"])}' if r.get("error") else "")))
    for c in d["constraints"]:
        rows.append(_row("constraint", "constraint", f'<b>{esc(c["name"])}</b> ' + (
            '<span class="aud-ok">satisfied</span>' if c["satisfied"] else '<span class="aud-bad">BROKEN</span>')))
    for n, why in d.get("not_run") or []:
        rows.append(_row("notrun", "not run", f'<b>{esc(n)}</b> ({esc(why)})'))
    rows.append(_row("answer", "→ answer", f'<b>{esc(fmt_answer(d["answer"]))}</b> — {esc(d["why"])}'))
    sg = d.get("safeguards") or []
    if sg:
        kinds = {}
        for e in sg:
            kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
        chips = "".join(f'<span class="aud-chip">{esc(LABEL.get(k, k))} ×{v}</span>' for k, v in kinds.items())
        lst = "".join(f'<li><b>{esc(LABEL.get(e["kind"], e["kind"]))}</b>: {esc(e["fact"])} — {esc(e["detail"])}</li>'
                      for e in sg)
        guards = f'<div class="aud-guards"><span class="aud-glabel">safeguards fired</span>{chips}<ul>{lst}</ul></div>'
    else:
        guards = '<div class="aud-guards"><span class="aud-glabel">safeguards</span><span class="aud-none">none fired</span></div>'
    return (f'<details class="aud-card" open><summary><span class="aud-q">{esc(d["question"])}</span> = '
            f'<span class="aud-ans">{esc(fmt_answer(d["answer"]))}</span> <span class="aud-st st-{esc(status)}">{esc(status)}</span>'
            f' <span class="aud-conf">confidence {d["confidence"]:.2f}{who}</span></summary>'
            f'{share_bar(d)}<ol class="aud-chain">{"".join(rows)}</ol>{guards}</details>')


def stats_html(stats):
    cells = "".join(f'<div class="aud-stat{" hot" if (stats.get(k) and i >= 4) else ""}"><b>{stats.get(k, 0)}</b>'
                    f'<span>{esc(lab)}</span></div>' for i, (k, lab) in enumerate(STAT_ROWS))
    return ('<div class="aud-stats"><div class="aud-stitle">System.stats: lifetime counts of this System '
            '(it lives while the code is unchanged; editing the code starts a new one)</div>'
            f'<div class="aud-sgrid">{cells}</div></div>')


def audit_html(out, with_stats=True):
    a = out.get("audit")
    if not a:
        return ""
    feas = out.get("feasible", True)
    viol = out.get("violations") or []
    n_sg = len(a.get("safeguards") or [])
    head = (f'<div class="aud-top"><span class="aud-chip {"ok" if feas else "bad"}">feasible: {"yes" if feas else "NO"}</span>'
            + (f'<span class="aud-chip bad">violated: {esc(", ".join(viol))}</span>' if viol else "")
            + f'<span class="aud-chip">{a.get("model_outputs", 0)} model output(s)</span>'
            + f'<span class="aud-chip{" warn" if n_sg else ""}">{n_sg} safeguard event(s)</span>'
            + '<span class="aud-legend">res.audit(): what each answer rests on. Deterministic = given, computed, quoted by '
              'plain code; from models = quoted by a model, decided, learned.</span></div>')
    cards = "".join(answer_card(d) for d in a["answers"].values())
    st = stats_html(out["stats"]) if with_stats and out.get("stats") else ""
    return f'<div class="aud">{head}{cards}{st}</div>'
