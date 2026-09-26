"""solvi playground, browser edition: a static Hugging Face Space on Gradio-Lite (Gradio 5 in Pyodide).

Everything, including the visitor's catalog code, runs inside the visitor's browser; there is no server.
Local run: serve this folder (python -m http.server 8000) and open http://localhost:8000/index.html.
It also runs as a normal Gradio 5 app: pip install "gradio>=5,<6" solvi && python app.py"""
from __future__ import annotations

import json
import re
import sys
import time
from datetime import date
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                     # Gradio-Lite may run the entrypoint without __file__
    HERE = Path.cwd()
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gradio as gr
import pandas as pd

import demos
import strategy_demo as sd
from sandbox import LIMITS_NOTE, run_job, serialize
from solvi import System
from solvi.strategist import plan

IN_BROWSER = sys.platform == "emscripten"

# ====================================================================== presets
PRESET_DIR = HERE / "presets"
PRESET_TITLES = {
    "01_leave_request": "Leave request (HR)",
    "02_refund_email": "Refund e-mail with a hard 30-day check",
    "03_tic_tac_toe": "Tic-tac-toe move",
    "04_loan_preapproval": "Loan pre-approval",
    "05_pizza_order": "Pizza order sanity check",
    "06_trip_packing": "Trip packing advisor",
    "07_smart_thermostat": "Smart thermostat",
    "08_plant_watering": "Plant watering",
    "09_blank_template": "Blank template (start here)",
}
PRESETS = {}
for stem, title in PRESET_TITLES.items():
    PRESETS[title] = ((PRESET_DIR / f"{stem}.py").read_text(), (PRESET_DIR / f"{stem}.json").read_text())
FIRST = next(iter(PRESETS))


def question_names(code: str):
    """Question names found in the code text (for the checkboxes before the first run)."""
    return re.findall(r"Question\(\s*[\"'](\w+)[\"']", code)


# ====================================================================== rendering: result dict -> UI values
def fmt_status(s):
    return {"ok": "ok", "forced": "forced (hard check)", "abstain": "abstain"}.get(s, s)


def answers_df(out):
    rows = [[a["question"], "—" if a["answer"] is None else a["answer"], f"{a['confidence']:.2f}", fmt_status(a["status"]),
             a["why"]] for a in out.get("answers", [])]
    return pd.DataFrame(rows, columns=["question", "answer", "confidence", "status", "why"])


def flow_df(out):
    rows = []
    skipped_run = dict(out.get("run_skipped") or [])
    for s in out.get("flow", []):
        kind = s["kind"] + (" (hard)" if s["kind"] == "check" and s["hard"] else "")
        ran = "skipped: " + skipped_run[s["name"]] if s["name"] in skipped_run else "ran"
        rows.append([s["step"], kind, s["name"], ", ".join(s["inputs"]) or "—", "; ".join(s["reasons"]), ran])
    return pd.DataFrame(rows, columns=["#", "kind", "part", "reads", "why it was taken", "at run time"])


def skipped_df(out):
    why = {"no inputs": "cannot run: its inputs are not in init_state and nothing computes them",
           "not needed for questions": "could run, but no asked question needs it"}
    rows = [[s["name"], s["kind"], why.get(s["why"], s["why"]), ", ".join(s["inputs"]) or "—"] for s in out.get("skipped", [])]
    for q, facts in (out.get("unresolved") or {}).items():
        for f in facts:
            rows.append([f, "missing fact", f"question '{q}' needs it, but nothing computes it from init_state", "—"])
    return pd.DataFrame(rows, columns=["part", "kind", "why it was NOT taken", "reads"])


def provenance(r):
    if r["kind"] == "extract" and r["quote"]:
        s, e, src = r["quote"]
        txt = r["quote_text"]
        p = f'quote {src}[{s}:{e}]' + (f' "{txt if len(txt) < 60 else txt[:57] + "…"}"' if txt else "")
        return p + (f"  (confidence {r['confidence']:.2f})" if r["confidence"] < 1 else "")
    if r["inputs"]:
        return "computed from " + ", ".join(r["inputs"])
    return "—"


def state_df(out):
    rows = [[r["step"], r["kind"], r["name"], r["value"], provenance(r), r["error"] or ""]
            for r in out.get("records", []) if r["kind"] != "rule"]
    return pd.DataFrame(rows, columns=["#", "kind", "fact", "value", "provenance", "error"])


def replay_line(out):
    rep = out.get("replay")
    if not rep:
        return ""
    if rep["ok"]:
        n = rep["steps"]
        return (f"**Trace replay: OK.** {'The only step was' if n == 1 else f'All {n} steps were'} recomputed from "
                "init_state; values, quotes and the hash chain match.")
    bad = "; ".join(f"step {s} `{n}`: {why}" for s, n, why in rep["mismatches"][:5])
    return f"**Trace replay: {len(rep['mismatches'])} mismatch(es).** {bad}"


def summary_md(out, roundtrip_ms=None):
    n = len(out.get("answers", []))
    forced = sum(a["status"] == "forced" for a in out["answers"])
    abst = sum(a["status"] == "abstain" for a in out["answers"])
    t = f"**{n} question{'s' if n != 1 else ''}** answered in **{out['ms']:.2f} ms** (solvi decision time)"
    if roundtrip_ms is not None:
        t += f", {roundtrip_ms:.0f} ms in total with the replay"
    extra = []
    if forced:
        extra.append(f"{forced} forced by a hard check")
    if abst:
        extra.append(f"{abst} abstained")
    t += (" · " + ", ".join(extra) if extra else "") + f" · {len(out.get('flow', []))} steps in the flow"
    return t + "\n\n" + replay_line(out)


def chain_text(out):
    lines = [f"init_state hash  {out.get('init_hash', '')}"]
    for r in out.get("records", []):
        lines.append(f"{r['step']:>3}  {r['kind']:<7} {r['name']:<24} prev {r['prev'][:10]}  hash {r['hash'][:10]}  "
                     f"value {r['value'][:40]}")
    return "\n".join(lines)


def tamper_choices(out):
    ch = [(f"{r['step']} · {r['kind']} · {r['name']} = {r['value'][:40]}", r["step"]) for r in out.get("records", [])]
    default = next((r["step"] for r in out.get("records", []) if r["kind"] in ("fn", "extract") and not r["error"]),
                   ch[0][1] if ch else None)
    return ch, default


# ====================================================================== tab 1: playground
EMPTY_DF = {k: pd.DataFrame() for k in ("answers", "flow", "skipped", "state")}


def load_preset(title):
    code, state = PRESETS.get(title, PRESETS[FIRST])
    names = question_names(code)
    return code, state, gr.update(choices=names, value=names), names


def _job(code, state_json, selected, known, tamper=None):
    try:
        state = json.loads(state_json or "{}")
    except json.JSONDecodeError as e:
        return None, f"init_state is not valid JSON: {e}"
    if not isinstance(state, dict):
        return None, "init_state must be a JSON object ({...})"
    return {"code": code or "", "state": state, "selected": list(selected or []), "known": list(known or []),
            "tamper": tamper}, None


def run_playground(code, state_json, selected, known):
    """Run button: execute the catalog in-process (in the visitor's browser) and render everything."""
    job, err = _job(code, state_json, selected, known)
    t0 = time.perf_counter()
    out = run_job(job) if job else {"ok": False, "error": err}
    rt = (time.perf_counter() - t0) * 1000
    if not out.get("ok"):
        msg = f"**Could not run.** {out.get('error', 'unknown error')}"
        tb = out.get("traceback") or ""
        return (msg, gr.update(value=tb, visible=bool(tb)), EMPTY_DF["answers"], EMPTY_DF["flow"], EMPTY_DF["skipped"],
                EMPTY_DF["state"], "", gr.update(), known, gr.update(choices=[], value=None), "")
    names = out["questions"]
    ch, default = tamper_choices(out)
    stdout = out.get("stdout") or ""
    raw = out["show"] + (f"\n── your code printed ──\n{stdout}" if stdout.strip() else "")
    return (summary_md(out, rt), gr.update(value="", visible=False), answers_df(out), flow_df(out), skipped_df(out),
            state_df(out), raw, gr.update(choices=names, value=out["asked"]), names,
            gr.update(choices=ch, value=default), chain_text(out))


def tamper_playground(code, state_json, selected, known, step, value, rehash):
    """Change one recorded value in a copy of the trace, then replay it: the chain must catch and locate the change."""
    if step is None:
        return "Run the system first, then pick a step to tamper with."
    job, err = _job(code, state_json, selected, known, {"step": int(step), "value": value or "", "rehash": bool(rehash)})
    out = run_job(job) if job else {"ok": False, "error": err}
    if not out.get("ok"):
        return f"**Could not run.** {out.get('error')}"
    t = out["tamper"]
    head = (f"Changed step **{t['step']}** `{t['name']}` from `{t['old']}` to `{t['new']}`"
            + (", and recomputed the hash of that record and of every record after it (a careful attacker)." if t["rehash"]
               else " without touching the hashes (a naive edit)."))
    if t["ok"]:
        return head + "\n\n**Replay found nothing:** the new value equals the recorded one, so nothing actually changed."
    first = t["mismatches"][0]
    where = ("the exact step you changed" if first[0] == t["step"] else f"step {first[0]}")
    lines = "\n".join(f"- step {s} `{n}`: {why}" for s, n, why in t["mismatches"][:12])
    more = f"\n- … and {len(t['mismatches']) - 12} more" if len(t["mismatches"]) > 12 else ""
    return (f"{head}\n\n**Caught.** Replay reports {len(t['mismatches'])} mismatch(es); the first one is at "
            f"step {first[0]} `{first[1]}`, {where}. Later steps are flagged because their recorded inputs no longer match."
            f"\n\n{lines}{more}")


# ====================================================================== tab: strategy (insurance claim desk, example 09)
CLAIM_SCENARIOS = {
    "Normal claim": {},
    "Policy expired": {"incident_date": "2027-02-01", "reported_date": "2027-02-02"},
    "Sanctioned claimant": {"claimant_name": "Ivan Shady"},
    "Late report": {"incident_date": "2026-06-01", "reported_date": "2026-08-14"},
    "Big claim": {"claimant_id": "C-17", "amount_claimed": 180_000.0, "peril": "fire"},
}
PERILS = ["fire", "theft", "flood", "storm", "water damage"]
Q_NAMES = [q.name for q in sd.QUESTIONS]


def claim_fields(scenario):
    """Values for the claim form: example 09's CLAIM with the scenario's changes."""
    c = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in sd.CLAIM.items()}
    c.update(CLAIM_SCENARIOS.get(scenario, {}))
    return [c["policy_id"], c["claim_type"], c["claimant_id"], c["claimant_name"], c["policy_start"], c["policy_end"],
            c["incident_date"], c["reported_date"], c["peril"], c["policy_perils"], c["amount_claimed"], c["deductible"],
            c["coverage_limit"], ", ".join(c["diagnosis_codes"]), ", ".join(f"{x:g}" for x in c["invoice_lines"]),
            c["hospital_days"]]


def _claim_state(policy_id, claim_type, claimant_id, claimant_name, policy_start, policy_end, incident, reported, peril,
                 perils, amount, deductible, limit, codes, lines, hospital_days):
    return {"policy_id": (policy_id or "").strip(), "claim_type": claim_type, "claimant_id": (claimant_id or "").strip(),
            "claimant_name": (claimant_name or "").strip(), "policy_start": _d(policy_start, "policy start"),
            "policy_end": _d(policy_end, "policy end"), "policy_perils": list(perils or []),
            "incident_date": _d(incident, "incident date"), "reported_date": _d(reported, "reported date"), "peril": peril,
            "amount_claimed": float(amount or 0), "deductible": float(deductible or 0), "coverage_limit": float(limit or 0),
            "diagnosis_codes": [c.strip() for c in (codes or "").split(",") if c.strip()],
            "invoice_lines": [float(x) for x in re.split(r"[,\s]+", lines or "") if x.strip()],
            "hospital_days": int(hospital_days or 0)}


NODE_STYLE = {   # class → (fill, stroke, text color, dash)
    "fn": ("#dbeafe", "#2563eb", "#1e3a8a", ""),
    "check": ("#fef3c7", "#d97706", "#78350f", ""),
    "hard": ("#fee2e2", "#dc2626", "#7f1d1d", ""),
    "extract": ("#dcfce7", "#16a34a", "#14532d", ""),
    "rule": ("#ede9fe", "#7c3aed", "#3b0764", ""),
    "off": ("#f1f5f9", "#94a3b8", "#64748b", "4 3"),
    "early": ("#fff1f2", "#e11d48", "#9f1239", "6 3"),
}


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def strategy_svg(cat, flow, trace, slow):
    """The generated strategy as a layered graph (inline SVG): every catalog part, colored by kind when taken, grey and dashed
    when not taken (with the reason), struck through when planned but skipped at run time (early exit), slow parts marked
    with their latency. Columns = dependency depth; rules on the right; numbers = execution order."""
    step_no = {s.part.name: i for i, s in enumerate(flow.steps, 1)}
    reasons = {s.part.name: "; ".join(s.reasons) for s in flow.steps}
    run_skip = dict(trace.skipped or [])
    parts = list(cat.parts.values())
    rules = list(cat.rules.values())
    depth = {}

    def d(name):
        if name not in depth:
            p = cat.parts[name]
            depth[name] = 1 + max([d(x) for x in p.inputs if x in cat.parts] or [-1])
        return depth[name]
    for p in parts:
        d(p.name)
    last = max(depth.values(), default=0) + 1
    cols = {}
    for p in parts:
        cols.setdefault(depth[p.name], []).append(p)
    cols[last] = rules
    W, H, GX, GY, PAD = 178, 44, 56, 14, 10
    pos = {}
    for c in sorted(cols):                       # order each column by the mean row of its inputs (fewer crossings)
        def key(p):
            ys = [pos[x][1] for x in p.inputs if x in pos]
            return (sum(ys) / len(ys) if ys else -1, step_no.get(p.name, 999), p.name)
        for r, p in enumerate(sorted(cols[c], key=key)):
            pos[p.name] = (PAD + c * (W + GX), PAD + r * (H + GY))
    width = PAD * 2 + (last + 1) * W + last * GX
    height = PAD * 2 + max(len(v) for v in cols.values()) * (H + GY) - GY
    edges, nodes = [], []
    for p in parts + rules:
        for x in p.inputs:
            if x in pos:
                (x1, y1), (x2, y2) = pos[x], pos[p.name]
                x1, y1, y2 = x1 + W, y1 + H / 2, y2 + H / 2
                taken = x in step_no and p.name in step_no and x not in run_skip and p.name not in run_skip
                m = (x1 + x2) / 2
                look = 'stroke-width="1.6"' if taken else 'stroke-width="1" stroke-dasharray="3 3" opacity="0.55"'
                edges.append(f'<path d="M{x1},{y1} C{m},{y1} {m},{y2} {x2},{y2}" fill="none" stroke="#94a3b8" {look} '
                             'marker-end="url(#arr)"/>')
    for p in parts + rules:
        x, y = pos[p.name]
        title = p.name if p.kind != "rule" else "answer: " + p.question
        sub = "hard check" if p.kind == "check" and p.hard else p.kind
        if p.name in run_skip:
            cls, note = "early", "skipped: " + run_skip[p.name].replace("not needed: hard check ", "").replace(
                "not needed: ", "")
        elif p.name in step_no:
            cls, note = ("hard" if p.kind == "check" and p.hard else p.kind), ""
        else:
            cls = "off"
            note = "not asked" if p.kind == "rule" else "not taken: " + flow.skipped.get(p.name, "not needed").replace(
                " for questions", "")
        fill, stroke, color, dash = NODE_STYLE[cls]
        head = f"{step_no[p.name]}. {title}" if p.name in step_no else title
        head = head if len(head) <= 26 else head[:25] + "…"
        line2 = note or sub
        line2 = line2 if len(line2) <= 34 else line2[:33] + "…"
        tip = f"{title} ({sub})" + (f"\n{p.doc}" if p.doc else "") + (f"\ntaken: {reasons[p.name]}" if p.name in reasons else "") \
            + (f"\n{note}" if note else "")
        deco = ' text-decoration="line-through"' if cls == "early" else ""
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        nodes.append(
            f'<g><title>{_esc(tip)}</title>'
            f'<rect x="{x}" y="{y}" width="{W}" height="{H}" rx="7" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{2 if cls in ("hard", "rule") else 1.2}"{dash_attr}/>'
            f'<text x="{x + 8}" y="{y + 18}" font-size="12.5" font-weight="600" fill="{color}"{deco}>{_esc(head)}</text>'
            f'<text x="{x + 8}" y="{y + 34}" font-size="10.5" fill="{color}" opacity="0.9">{_esc(line2)}</text>')
        if p.name in slow:
            nodes.append(f'<rect x="{x + W - 62}" y="{y - 7}" width="58" height="15" rx="7" fill="#f97316"/>'
                         f'<text x="{x + W - 33}" y="{y + 4}" font-size="9.5" font-weight="700" fill="#fff" '
                         f'text-anchor="middle">{int(slow[p.name] * 1000)} ms</text>')
        nodes.append("</g>")
    return (f'<div class="dag"><svg viewBox="0 0 {width} {height + 8}" width="100%" role="img" '
            f'aria-label="strategy graph" style="font-family: ui-sans-serif, system-ui, sans-serif; max-width:{width}px">'
            '<defs><marker id="arr" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
            '<path d="M0,0 L8,4 L0,8 z" fill="#94a3b8"/></marker></defs>'
            + "".join(edges) + "".join(nodes) + "</svg></div>")


def timing_html(rows):
    """rows: [(label, ms, css color)] → simple horizontal bars."""
    top = max(ms for _, ms, _ in rows) or 1
    out = ['<div class="bars">']
    for label, ms, color in rows:
        w = max(0.6, 100 * ms / top)
        out.append(f'<div class="bar-row"><div class="bar-label">{label}</div><div class="bar-track">'
                   f'<div class="bar" style="width:{w:.1f}%;background:{color}"></div></div>'
                   f'<div class="bar-ms">{ms:,.0f} ms</div></div>')
    return "\n".join(out + ["</div>"])


def _timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, (time.perf_counter() - t0) * 1000


def run_strategy(asked, *fields):
    names = [n for n in Q_NAMES if n in (asked or [])]
    if not names:
        return ("Pick at least one question.", "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame())
    try:
        state = _claim_state(*fields)
    except (ValueError, TypeError) as e:
        return (f"**Check the inputs.** {e}", "", pd.DataFrame(), pd.DataFrame(), "", pd.DataFrame())
    qs = [q for q in sd.QUESTIONS if q.name in names]
    system = System(sd.cat, sd.QUESTIONS)
    flow, t_plan = _timed(lambda: plan(sd.cat, qs, state.keys()))
    sd.reset_simulated()
    res, t_seq = _timed(lambda: system.ask(state, names, workers=1))
    sim_seq = sd.SIMULATED_MS[0]
    sd.reset_simulated()
    _, t_all = _timed(lambda: sd.run_everything(state))
    sim_all = sd.SIMULATED_MS[0]
    out = serialize(res, sd.cat, state, sd.QUESTIONS)
    total = len(sd.cat.parts) + len(sd.cat.rules)
    slow_run = [r.name for r in res.trace.records if r.name in sd.SLOW]
    slow_skip = [n for n in sd.SLOW if n not in slow_run]
    early = res.trace.skipped or []
    md = (f"**Strategy for {len(names)} question{'s' if len(names) > 1 else ''}: {len(flow.steps)} of {total} catalog parts, "
          f"planned in {t_plan:.2f} ms.** "
          + (f"Early exit: a hard check failed, so {len(early)} planned step{'s' if len(early) > 1 else ''} did not run. "
             if early else "")
          + f"Slow parts run: {', '.join(slow_run) or 'none'}"
          + (f"; avoided: {', '.join(slow_skip)}." if slow_skip else ".")
          + f"\n\n{replay_line(out)}")
    if sd.REAL_SLEEP:
        rows = [("Plain script, computes everything", t_all, "#94a3b8"), ("solvi, one step at a time", t_seq, "#818cf8")]
        how = ("Measured live in your browser; the slow parts really wait (<code>time.sleep</code> works here).")
    else:
        rows = [("Plain script, computes everything", t_all + sim_all, "#94a3b8"),
                ("solvi, one step at a time", t_seq + sim_seq, "#818cf8")]
        how = (f"<code>time.sleep</code> does not wait in this browser runtime, so the slow parts are simulated: each bar is "
               f"the compute time measured live plus the <b>simulated service time</b> of the slow parts that actually ran "
               f"(plain script {t_all:.1f} ms + {sim_all:,.0f} ms simulated; solvi {t_seq:.1f} ms + {sim_seq:,.0f} ms "
               "simulated).")
    bars = timing_html(rows)
    fast = rows[0][1] / max(rows[1][1], 1e-6)
    bars += (f'<div class="note">{how} solvi runs only the planned parts: {fast:.1f}x faster than the plain script for '
             "these questions.</div>")
    bars += ('<div class="note" style="margin-top:10px"><b>Parallel (workers=8) is not available here:</b> Pyodide has no '
             "threads, so solvi runs one step at a time in the browser. Native Python, all five questions "
             "(solvi README, “Speed”):</div>"
             + timing_html([("Plain script (native)", 1122, "#cbd5e1"), ("solvi, one by one (native)", 1025, "#c7d2fe"),
                            ("solvi, workers=8 (native)", 463, "#6366f1")]))
    return (md, strategy_svg(sd.cat, flow, res.trace, sd.SLOW), flow_df(out), skipped_df(out), bars,
            answers_df(out))


def run_scale():
    """The strategist on random layered catalogs (benchmarks/strategist_scale.py): 100, 1 000 and 10 000 parts."""
    import statistics
    rows = []
    for n in (100, 1000, 10000):
        cat, qs, state = sd.make_catalog(n)
        system = System(cat, qs)
        reps = 3 if n <= 1000 else 1
        t_plan = statistics.median(_timed(lambda: plan(cat, qs, state.keys()))[1] for _ in range(reps))
        t_ask = statistics.median(_timed(lambda: system.ask(state))[1] for _ in range(reps))
        steps = len(system.ask(state).flow.steps)
        vals = dict(state)

        def run_all():
            for p in cat.parts.values():
                vals[p.name] = p.func(**{x: vals[x] for x in p.inputs})
        t_all = statistics.median(_timed(run_all)[1] for _ in range(reps))
        total = len(cat.parts) + len(cat.rules)
        rows.append([f"{total:,}", f"{t_plan:.2f}", f"{t_ask:.2f}", steps, f"{steps / total:.1%}", f"{t_all:.2f}"])
    return pd.DataFrame(rows, columns=["catalog parts", "plan ms", "ask ms (plan + run)", "flow steps", "share of catalog run",
                                       "run-everything ms"])


# ====================================================================== tab 2: business decisions
def _d(s, field):
    s = (s if isinstance(s, str) else str(s or "")).strip()[:10]
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise ValueError(f"{field}: '{s}' is not a date (YYYY-MM-DD)") from None


def _rows(df):
    if df is None:
        return []
    if isinstance(df, pd.DataFrame):
        df = df.values.tolist()
    return [[("" if c is None else str(c)).strip() for c in row] for row in df
            if any(str(c).strip() for c in row if c is not None)]


def _business_outputs(out):
    return summary_md(out), answers_df(out), flow_df(out), skipped_df(out), state_df(out)


def _business_error(e):
    return (f"**Check the inputs.** {e}", pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())


def run_leave(employee, start, end, today, balance, team_size, leaves):
    try:
        state = {"employee": (employee or "").strip().lower(), "start": _d(start, "first day"), "end": _d(end, "last day"),
                 "today": _d(today, "today"), "balance": int(balance or 0), "team_size": int(team_size or 1),
                 "team_leaves": [(r[0].lower(), _d(r[1], f"leave of {r[0]}"), _d(r[2], f"leave of {r[0]}"))
                                 for r in _rows(leaves)],
                 "blackout": demos.LEAVE_BLACKOUT}
        res = demos.LEAVE_SYSTEM.ask(state)
        return _business_outputs(serialize(res, demos.LEAVE.cat, state, demos.LEAVE.QUESTIONS))
    except (ValueError, TypeError, IndexError) as e:
        return _business_error(e)


def highlight(doc, out):
    spans = sorted((r["quote"][0], r["quote"][1], r["name"]) for r in out.get("records", [])
                   if r["quote"] and r["quote"][2] == "doc" and r["quote"][1] > r["quote"][0])
    segs, pos = [], 0
    for s, e, name in spans:
        if s < pos:
            continue
        if s > pos:
            segs.append((doc[pos:s], None))
        segs.append((doc[s:e], name))
        pos = e
    if pos < len(doc):
        segs.append((doc[pos:], None))
    return segs


def run_invoice(text, vendors, prior_ids, payments, today):
    try:
        pays = []
        for r in _rows(payments):
            pays.append({"vendor": r[0], "amount": float(r[1].replace(",", ""))})
        state = {"doc": text or "", "today": _d(today, "today"),
                 "vendors_db": [v.strip() for v in (vendors or "").split(",") if v.strip()],
                 "prior_invoice_ids": [v.strip() for v in re.split(r"[,\s]+", prior_ids or "") if v.strip()],
                 "payments_db": pays}
        res = demos.INV_SYSTEM.ask(state)
        out = serialize(res, demos.inv, state, demos.INV_QUESTIONS)
        return (*_business_outputs(out), highlight(state["doc"], out))
    except (ValueError, TypeError, IndexError) as e:
        return (*_business_error(e), [])


# ====================================================================== tab 3: learn from examples
def seed_df():
    return pd.DataFrame([[s["address"], z] for s, z in demos.SEED_ADDRESSES], columns=["address", "zone"])


def _examples(df):
    ex, dropped = [], 0
    for r in _rows(df):
        if len(r) >= 2 and r[0] and r[1] in demos.ZONES:
            ex.append(({"address": r[0]}, r[1]))
        else:
            dropped += 1
    return ex, dropped


def _learned(examples, min_support, min_precision):
    cat, system = demos.zone_system()
    rules = system.learn_rule("zone", examples, ["address"], min_support=int(min_support),
                              min_precision=float(min_precision))
    return cat, system, rules


def learn(df, min_support, min_precision):
    ex, dropped = _examples(df)
    if len(ex) < 5:
        return "Need at least 5 labeled rows (zone must be one of " + ", ".join(demos.ZONES) + ").", "", None
    t0 = time.perf_counter()
    cat, system, rules = _learned(ex, min_support, min_precision)
    fit_ms = (time.perf_counter() - t0) * 1000
    hits = sum(system.ask(s)["zone"].answer == z for s, z in demos.TEST_ADDRESSES)
    md = (f"Learned **{len(rules.rules)} rules** from **{len(ex)}** labeled addresses in {fit_ms:.0f} ms"
          + (f" ({dropped} rows skipped: empty or unknown zone)" if dropped else "")
          + f".\n\nAccuracy on **{len(demos.TEST_ADDRESSES)} fresh generated addresses: {hits / len(demos.TEST_ADDRESSES):.1%}**"
          " (never seen during learning).")
    return md, "\n".join(line.strip() for line in str(rules).splitlines()), {"examples": ex, "min_support": min_support, "min_precision": min_precision}


def try_address(addr, learned, df, min_support, min_precision):
    if not (addr or "").strip():
        return "Type an address.", pd.DataFrame(), pd.DataFrame(), ""
    if not learned:                                   # not learned yet: learn from the current table first
        ex, _ = _examples(df)
        if len(ex) < 5:
            return "Add labeled rows and press **Learn rules** first.", pd.DataFrame(), pd.DataFrame(), ""
        learned = {"examples": ex, "min_support": min_support, "min_precision": min_precision}
    cat, system, rules = _learned(learned["examples"], learned["min_support"], learned["min_precision"])
    state = {"address": addr.strip()}
    res = system.ask(state)
    out = serialize(res, cat, state, list(system.questions.values()))
    answer, fired = rules.predict(state)
    r = res["zone"]
    rule_txt = (f"rule {rules.rules.index(fired) + 1}: **if {fired['if']} → {fired['then']}** "
                f"({fired['support']}/{fired['covered']} training examples)") if fired else \
        f"no rule matched, so the default answer **{rules.default}** was used"
    md = (f"### {answer}\n\nFired: {rule_txt}.\n\nThe learned rule list is installed in the catalog as the ordinary rule "
          f"`answer:zone` (a `{r.status}` answer, confidence {r.confidence:.2f}); below is the flow the strategist planned "
          "for it, like for any hand-written rule.\n\n" + replay_line(out))
    return md, flow_df(out), skipped_df(out), out["show"]


# ====================================================================== UI
CSS = """
.mono textarea, .mono pre, .mono code, .mono .cm-content { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace !important; font-size: 12.5px !important; }
#header h1 { margin-bottom: 0; }
#header p { margin-top: 4px; opacity: .8; }
.note { font-size: 12.5px; opacity: .75; }
.bars { display: flex; flex-direction: column; gap: 8px; margin: 4px 0 8px; }
.bar-row { display: grid; grid-template-columns: minmax(120px, 38%) 1fr 70px; align-items: center; gap: 8px; font-size: 13px; }
.bar-track { background: var(--block-background-fill, #f1f5f9); border-radius: 4px; height: 18px; }
.bar { height: 18px; border-radius: 4px; }
.dag { overflow-x: auto; }
.dag svg { min-width: 860px; }
.bar-ms { text-align: right; font-variant-numeric: tabular-nums; font-family: ui-monospace, monospace; }
"""
PITCH = ("Write a decision task as small Python functions, checks and rules; ask typed questions. The strategist plans only the "
         "steps those questions need and answers in about a millisecond, with reasons and a hash-chained trace you can re-check.")
ABOUT = """
**solvi** builds decision systems from a catalog of plain Python functions, checks and rules plus typed questions (yes/no or a choice).
For each request a **strategist** plans which parts to run; everything else in the catalog is skipped, and the flow says why.
Every answer has a **confidence**, a **reason you can check** (rule inputs, a formula, or a quote with character offsets), and a **hash-chained trace** that `trace.replay(catalog)` re-executes to confirm the answer or pinpoint an altered step.
It **abstains** instead of guessing, and a failed **hard check** always overrides the rule or model.
Questions without a rule can be learned from ~100 labeled examples: a small answer head (`fit`) or a readable rule list (`learn_rule`).
Benchmarks (from the README): SROIE receipts **97.8%** vs 92.6% baseline; CORD receipts **98.5%**, with 99.7% of questions answered at >= 99% precision; CUAD contracts **94.8%** vs 77.5%.
Calibrated confidence (ECE 0.008 to 0.027); 100% of document answers backed by a quote or abstained; ~**0.4 ms** per decision without a model.
Install: `pip install solvi` (core, numpy/scipy only) or `pip install "solvi[model]"` for ModernBERT document extractors.

**This Space runs entirely in your browser.** It is a static page: [Gradio-Lite](https://www.gradio.app/guides/gradio-lite) loads Python (Pyodide) into the tab and installs solvi from PyPI there, so every decision, including the code you type in the Playground, is computed on your machine and nothing is sent to a server. solvi is pure Python on numpy/scipy, which is why this works. In the browser there are no threads (solvi then runs steps one by one), and Python is roughly 1.5 to 3 times slower than native.

Code, docs and examples: [github.com/solvi-ai/solvi](https://github.com/solvi-ai/solvi) (Apache-2.0) · [PyPI: solvi](https://pypi.org/project/solvi/).
Another Space: [solvi arcade](https://huggingface.co/spaces/solvi-ai/arcade).
"""

# Soft's default fonts are served from the Gradio server's /static folder, which does not exist in Gradio-Lite: use Google Fonts.
THEME = gr.themes.Soft(primary_hue="indigo", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Montserrat"), "ui-sans-serif", "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"])
ANS_W = ["17%", "14%", "11%", "14%", "44%"]

with gr.Blocks(title="solvi playground", theme=THEME, css=CSS) as demo:
    gr.Markdown(f"# solvi playground\n{PITCH}\n\n" + (
        "**Runs entirely in your browser** (Python via Pyodide): no server, nothing you type leaves this tab." if IN_BROWSER
        else "Running as a normal Gradio server app."), elem_id="header")

    with gr.Tab("Playground"):
        known = gr.State(question_names(PRESETS[FIRST][0]))
        with gr.Row():
            with gr.Column(scale=5):
                preset = gr.Dropdown(list(PRESETS), value=FIRST, label="Preset")
                code = gr.Code(PRESETS[FIRST][0], language="python", lines=26, max_lines=40,
                               label="Catalog module: defines cat, QUESTIONS and optionally prepare(state)")
                init_json = gr.Code(PRESETS[FIRST][1], language="json", lines=8, max_lines=20, label="init_state (JSON)")
                qs = gr.CheckboxGroup(question_names(PRESETS[FIRST][0]), value=question_names(PRESETS[FIRST][0]),
                                      label="Questions to ask", info="New questions in your code are asked automatically.")
                run_btn = gr.Button("Run", variant="primary")
                gr.Markdown(LIMITS_NOTE, elem_classes="note")
            with gr.Column(scale=6):
                summary = gr.Markdown("Press **Run**.")
                err_box = gr.Code(language=None, label="Traceback (last lines)", visible=False, elem_classes="mono")
                ans = gr.Dataframe(label="Answers", wrap=True, interactive=False, column_widths=ANS_W)
                flow = gr.Dataframe(label="Flow chosen by the strategist (execution order)", wrap=True, interactive=False)
                skipped = gr.Dataframe(label="Parts NOT taken, and why", wrap=True, interactive=False)
                cstate = gr.Dataframe(label="computed_state with provenance", wrap=True, interactive=False)
                with gr.Accordion("Raw output of solvi.show(res, cat)", open=False):
                    raw = gr.Code(language=None, interactive=False, elem_classes="mono", lines=12)
                with gr.Accordion("Tamper with the trace", open=False):
                    gr.Markdown("Every step is hashed and chained to the previous one. Change one recorded value in a copy of "
                                "the trace and replay it: replay recomputes each step from init_state and names the step "
                                "that no longer matches, even if the attacker also recomputes all hashes.",
                                elem_classes="note")
                    chain = gr.Code(language=None, interactive=False, elem_classes="mono", lines=8, label="Hash chain")
                    with gr.Row():
                        t_step = gr.Dropdown([], label="Step to change", scale=3)
                        t_val = gr.Textbox("999", label="Replacement value (Python literal)", scale=2)
                    t_rehash = gr.Checkbox(False, label="Attacker also recomputes the hashes of the chain")
                    t_btn = gr.Button("Tamper and replay")
                    t_out = gr.Markdown()

        preset.change(load_preset, preset, [code, init_json, qs, known])
        play_outputs = [summary, err_box, ans, flow, skipped, cstate, raw, qs, known, t_step, chain]
        run_btn.click(run_playground, [code, init_json, qs, known], play_outputs)
        t_btn.click(tamper_playground, [code, init_json, qs, known, t_step, t_val, t_rehash], t_out)

    with gr.Tab("Strategy"):
        gr.Markdown(f"An insurance claim desk with **{len(sd.cat.parts)} parts and {len(sd.cat.rules)} rules** "
                    "(examples/09_strategy_at_scale.py); six parts are slow on purpose, like real API calls "
                    f"({', '.join(f'{k} {int(v * 1000)} ms' for k, v in sd.SLOW.items())}). For each set of questions the "
                    "strategist writes a different plan: only what those questions need. Hard checks run first; when one "
                    "fails, the expensive rest is skipped. (Natively, independent steps can also run in parallel threads; "
                    "the browser has no threads, see the note under Timing.)", elem_classes="note")
        with gr.Row():
            with gr.Column(scale=4):
                s_scen = gr.Radio(list(CLAIM_SCENARIOS), value="Normal claim", label="Scenario")
                s_qs = gr.CheckboxGroup([(f"{q.name}: {q.text}", q.name) for q in sd.QUESTIONS], value=["decision", "payout_band"],
                                        label="Questions to ask")
                v0 = claim_fields("Normal claim")
                with gr.Accordion("Claim fields", open=False):
                    with gr.Row():
                        s_pol = gr.Textbox(v0[0], label="policy_id")
                        s_type = gr.Dropdown(["property", "medical", "auto"], value=v0[1], label="claim_type")
                    with gr.Row():
                        s_cid = gr.Textbox(v0[2], label="claimant_id", info="C-17 has 4 prior claims and a lawsuit")
                        s_name = gr.Textbox(v0[3], label="claimant_name", info="'Ivan Shady' is on the sanctions list")
                    with gr.Row():
                        s_ps = gr.DateTime(v0[4], include_time=False, type="string", label="policy_start")
                        s_pe = gr.DateTime(v0[5], include_time=False, type="string", label="policy_end")
                    with gr.Row():
                        s_inc = gr.DateTime(v0[6], include_time=False, type="string", label="incident_date")
                        s_rep = gr.DateTime(v0[7], include_time=False, type="string", label="reported_date")
                    with gr.Row():
                        s_peril = gr.Dropdown(PERILS, value=v0[8], label="peril")
                        s_perils = gr.CheckboxGroup(PERILS, value=v0[9], label="policy_perils (covered)")
                    with gr.Row():
                        s_amt = gr.Number(v0[10], label="amount_claimed")
                        s_ded = gr.Number(v0[11], label="deductible")
                        s_lim = gr.Number(v0[12], label="coverage_limit")
                    with gr.Row():
                        s_codes = gr.Textbox(v0[13], label="diagnosis_codes")
                        s_lines = gr.Textbox(v0[14], label="invoice_lines")
                        s_days = gr.Number(v0[15], label="hospital_days", precision=0)
                s_run = gr.Button("Plan and run", variant="primary")
            with gr.Column(scale=7):
                s_md = gr.Markdown("Press **Plan and run**.")
                s_ans = gr.Dataframe(label="Answers", wrap=True, interactive=False, column_widths=ANS_W)
                gr.Markdown("### Timing")
                s_bars = gr.HTML()
        gr.Markdown("### The generated strategy")
        s_graph = gr.HTML()
        gr.Markdown("Colors: blue = function, amber = check, red = hard check, purple = answer rule. Grey dashed = in the "
                    "catalog but NOT taken (reason inside). Red dashed and struck through = planned, but skipped at run time "
                    "because a hard check had already settled the answer. Orange tag = slow part and its latency. Numbers = "
                    "execution order; hover a box for details.", elem_classes="note")
        with gr.Row():
            s_plan = gr.Dataframe(label="The plan, step by step", wrap=True, interactive=False, scale=3)
            s_skip = gr.Dataframe(label="Not taken, and why", wrap=True, interactive=False, scale=2)
        with gr.Accordion("Scale: the strategist on catalogs of 100 / 1 000 / 10 000 parts", open=False):
            gr.Markdown("Random layered catalogs (benchmarks/strategist_scale.py): functions over 1-3 earlier facts, checks, "
                        "5 questions answered by rules over deep facts. Takes several seconds in the browser (building the "
                        "10 000-part catalog alone runs 10 000 small `exec` calls).", elem_classes="note")
            s_scale_btn = gr.Button("Run the scale test")
            s_scale = gr.Dataframe(interactive=False, show_label=False)
            gr.Markdown("Parts here are one-line arithmetic, so computing everything is cheap; the point is that planning "
                        "stays in milliseconds and the flow touches a small, shrinking share of a growing catalog. With real "
                        "parts (APIs, models) the share not run is time saved.", elem_classes="note")
        s_fields = [s_pol, s_type, s_cid, s_name, s_ps, s_pe, s_inc, s_rep, s_peril, s_perils, s_amt, s_ded, s_lim, s_codes,
                    s_lines, s_days]
        s_out = [s_md, s_graph, s_plan, s_skip, s_bars, s_ans]
        s_scen.change(claim_fields, s_scen, s_fields).then(run_strategy, [s_qs, *s_fields], s_out)
        s_run.click(run_strategy, [s_qs, *s_fields], s_out)
        s_qs.change(run_strategy, [s_qs, *s_fields], s_out)
        s_scale_btn.click(run_scale, None, s_scale)

    with gr.Tab("Business decisions"):
        gr.Markdown("No code: fill in a form, and the answers update as you type. Same engine, fixed catalogs.",
                    elem_classes="note")
        with gr.Tab("Leave request"):
            with gr.Row():
                with gr.Column(scale=4):
                    with gr.Row():
                        l_emp = gr.Textbox("anna", label="Employee")
                        l_today = gr.DateTime("2026-09-25", include_time=False, type="string", label="Today")
                    with gr.Row():
                        l_start = gr.DateTime("2026-10-19", include_time=False, type="string", label="First day")
                        l_end = gr.DateTime("2026-10-30", include_time=False, type="string", label="Last day")
                    with gr.Row():
                        l_bal = gr.Number(14, label="Leave balance (days)", precision=0)
                        l_team = gr.Number(6, label="Team size", precision=0)
                    l_leaves = gr.Dataframe([["boris", "2026-10-26", "2026-11-06"], ["vera", "2026-12-01", "2026-12-10"]],
                                            headers=["colleague", "from", "to"], type="array", interactive=True,
                                            label="Colleagues on leave (YYYY-MM-DD)", row_count=(2, "dynamic"))
                    gr.Markdown("Blackout (year-end close): 2026-12-20 to 2026-12-31. Rules: 5+ days need 14 days notice; "
                                "less than half the team away; the balance may not go negative (hard).",
                                elem_classes="note")
                with gr.Column(scale=6):
                    l_sum = gr.Markdown()
                    l_ans = gr.Dataframe(label="Answers", wrap=True, interactive=False, column_widths=ANS_W)
                    l_flow = gr.Dataframe(label="Flow", wrap=True, interactive=False)
                    l_skip = gr.Dataframe(label="Not taken", wrap=True, interactive=False)
                    l_state = gr.Dataframe(label="computed_state", wrap=True, interactive=False)
            l_in = [l_emp, l_start, l_end, l_today, l_bal, l_team, l_leaves]
            l_out = [l_sum, l_ans, l_flow, l_skip, l_state]
            for c in l_in:
                c.change(run_leave, l_in, l_out)

        with gr.Tab("Invoice approval"):
            with gr.Row():
                with gr.Column(scale=4):
                    i_text = gr.Textbox(demos.INVOICE_TEXT, lines=12, label="Invoice text", elem_classes="mono")
                    i_vendors = gr.Textbox(", ".join(demos.VENDORS), label="Known vendors (comma separated)")
                    i_prior = gr.Textbox("INV-1999, INV-2003", label="Already paid invoice ids",
                                         info="Add INV-2041 to see the duplicate check reject it.")
                    i_pay = gr.Dataframe([[p["vendor"], p["amount"]] for p in demos.PAYMENTS_DB], headers=["vendor", "amount"],
                                         type="array", interactive=True, label="Past payments (vendor DB)",
                                         row_count=(4, "dynamic"))
                    i_today = gr.DateTime("2026-09-25", include_time=False, type="string", label="Today")
                    gr.Markdown("Fields are found with regular expressions (examples/03_invoices.py); each keeps a quote. "
                                "Try: change the total to 600.00, the supplier to Shady Traders, or the due date. "
                                f"The risk question has no rule: a head learned from 300 synthetic invoices uses "
                                f"{', '.join(demos.RISK_HEAD.features)}.", elem_classes="note")
                with gr.Column(scale=6):
                    i_sum = gr.Markdown()
                    i_hl = gr.HighlightedText(label="Extracted quotes in the text", show_legend=False,
                                              show_inline_category=True, elem_classes="mono")
                    i_ans = gr.Dataframe(label="Answers", wrap=True, interactive=False, column_widths=ANS_W)
                    i_flow = gr.Dataframe(label="Flow", wrap=True, interactive=False)
                    i_skip = gr.Dataframe(label="Not taken", wrap=True, interactive=False)
                    i_state = gr.Dataframe(label="computed_state", wrap=True, interactive=False)
            i_in = [i_text, i_vendors, i_prior, i_pay, i_today]
            i_out = [i_sum, i_ans, i_flow, i_skip, i_state, i_hl]
            for c in i_in:
                c.change(run_invoice, i_in, i_out)

    with gr.Tab("Learn from examples"):
        gr.Markdown("Route parcels to delivery zones from free-form addresses. Nobody writes the rule: `System.learn_rule` "
                    "learns a short, readable if-then list from the labeled rows (examples/06_learned_rules.py). Edit, add "
                    "or mislabel rows, then learn again.", elem_classes="note")
        learned = gr.State(None)
        with gr.Row():
            with gr.Column(scale=5):
                z_df = gr.Dataframe(seed_df(), headers=["address", "zone"], type="array", interactive=True,
                                    row_count=(200, "dynamic"), max_height=420,
                                    label=f"Labeled addresses (zone: {', '.join(demos.ZONES)})")
                with gr.Row():
                    z_sup = gr.Slider(1, 20, 3, step=1, label="min support")
                    z_prec = gr.Slider(0.5, 0.99, 0.8, step=0.01, label="min precision")
                with gr.Row():
                    z_learn = gr.Button("Learn rules", variant="primary")
                    z_reset = gr.Button("Reset to 200 generated rows")
            with gr.Column(scale=6):
                z_md = gr.Markdown("Press **Learn rules**.")
                z_rules = gr.Code(language=None, interactive=False, elem_classes="mono", lines=10, label="Learned rule list")
                z_addr = gr.Textbox("8 Station Rd, Crossfield", label="Try your own address",
                                    info="Try '17 Harbour Rd, Portsea': 200 rows may hold too few Portsea examples for a "
                                         "rule. Add a few labeled Portsea rows and learn again.")
                z_try = gr.Button("Which zone?")
                z_out = gr.Markdown()
                z_flow = gr.Dataframe(label="Flow: the learned rule is an ordinary rule step", wrap=True, interactive=False)
                z_skip = gr.Dataframe(label="Not taken", wrap=True, interactive=False)
                with gr.Accordion("Raw output of solvi.show", open=False):
                    z_raw = gr.Code(language=None, interactive=False, elem_classes="mono", lines=10)
        z_learn.click(learn, [z_df, z_sup, z_prec], [z_md, z_rules, learned]).then(
            try_address, [z_addr, learned, z_df, z_sup, z_prec], [z_out, z_flow, z_skip, z_raw])
        z_reset.click(lambda: (seed_df(), None, "Press **Learn rules**.", ""), None, [z_df, learned, z_md, z_rules])
        try_in = [z_addr, learned, z_df, z_sup, z_prec]
        z_try.click(try_address, try_in, [z_out, z_flow, z_skip, z_raw])
        z_addr.submit(try_address, try_in, [z_out, z_flow, z_skip, z_raw])

    with gr.Tab("About"):
        gr.Markdown(ABOUT)

    demo.load(run_playground, [code, init_json, qs, known], play_outputs)
    demo.load(run_strategy, [s_qs, *s_fields], s_out)
    demo.load(run_leave, l_in, l_out)
    demo.load(run_invoice, i_in, i_out)

demo.launch()
