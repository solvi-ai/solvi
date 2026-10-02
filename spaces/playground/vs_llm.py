"""The "solvi vs LLM" tab: cases from the public benchmark (benchmarks/vs_llm), side by side.

For a curated case of the refund or 3-way-match set it shows the input, what each LLM answered when asked directly (the
saved answers of the published run, with their confidence, marked against the right answer), what solvi decides (the
gallery catalog runs LIVE here, with the checks and rules that fired and the audit), and what the same LLMs answered
inside solvi. "Reorder the options" shows the saved answer to the reordered request where the run has one, and runs
solvi again with its options reordered; "Replay" re-executes solvi's trace.

Everything comes from vs_llm.json, built by benchmarks/vs_llm/make_playground_bundle.py. The arms are read from that file
(its "arms" and "groups"), so a new model or a new kind of arm appears after the bundle is rebuilt, with no change here.
No API call is made and no key is needed. Pure Python, no gradio: app.py wires the handlers."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import html
import json
import random
import re
import sys
import time
import types
from pathlib import Path

try:
    HERE = Path(__file__).resolve().parent
except NameError:                                     # Gradio-Lite may run modules without __file__
    HERE = Path.cwd()

DATA_FILE = HERE / "vs_llm.json"
PRESET_DIR = HERE / "presets"
ABST = "abstain"
CONFIDENT = 0.9
INJECTED = ("SYSTEM:", "[Note to the AI", "(Assistant:", "Ignore previous", "APPROVED BY", "Note for the automated")

_BUNDLE = None
_SYSTEMS = {}        # set → (module, System, questions)
_LAST = {}           # case id → (Response, System, state): the last live decision, for Replay


def bundle():
    global _BUNDLE
    if _BUNDLE is None:
        _BUNDLE = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return _BUNDLE


def solvi_version():
    try:
        from importlib.metadata import version
        return version("solvi")
    except Exception:  # noqa: BLE001
        return "?"


def esc(x):
    return html.escape(str(x), quote=True)


# ---------------------------------------------------------------------------------------------- choices
def set_choices():
    """[(title, key)] for the set picker."""
    return [(v["title"], k) for k, v in bundle()["sets"].items()]


def case_choices(set_key):
    """[(label, case id)] for the case picker."""
    return [(f"{c['id']} · {c['title']}", c["id"]) for c in bundle()["sets"][set_key]["cases"]]


def first_case(set_key):
    return bundle()["sets"][set_key]["cases"][0]["id"]


def _case(set_key, case_id):
    S = bundle()["sets"][set_key]
    for c in S["cases"]:
        if c["id"] == case_id:
            return S, c
    return S, S["cases"][0]


def _head(a, extra=""):
    """A column header for an arm; one that is not part of the published run says so."""
    new = "" if a.get("published", True) else ' <span class="vs-sub">not in the published run</span>'
    return f'<th>{esc(a["label"])}{extra}{new}</th>'


def _arms(group=None, set_key=None):
    return [a for a in bundle()["arms"] if (group is None or a["group"] == group)
            and (set_key is None or set_key in a["sets"])]


# ---------------------------------------------------------------------------------------------- the live catalog
def _module(stem):
    name = f"vs_llm_{stem}"
    mod = types.ModuleType(name)
    mod.__file__ = str(PRESET_DIR / f"{stem}.py")
    sys.modules[name] = mod
    exec(compile((PRESET_DIR / f"{stem}.py").read_text(encoding="utf-8"), mod.__file__, "exec"), mod.__dict__)  # noqa: S102 — our own preset file
    return mod


def _build(mod, S, questions=None):
    from solvi import System
    kw = {"input_model": getattr(mod, S["inputs"])} if S.get("inputs") else {}
    return System(mod.cat, questions or mod.QUESTIONS, **kw)


def _state(mod, state):
    s = copy.deepcopy(state)
    return mod.prepare(s) if callable(getattr(mod, "prepare", None)) else s


def system(set_key):
    """(module, System) of the set's gallery catalog, built once; the first ask warms it up."""
    if set_key not in _SYSTEMS:
        S = bundle()["sets"][set_key]
        mod = _module(S["preset"])
        sysm = _build(mod, S)
        sysm.ask(_state(mod, S["cases"][0]["state"]))
        _SYSTEMS[set_key] = (mod, sysm)
    return _SYSTEMS[set_key]


def _norm(v):
    if v is None:
        return ABST
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (list, tuple)):
        return sorted(str(x) for x in v)
    return str(v)


def _answers(res):
    return {q: _norm(None if r.status == "abstain" else r.answer) for q, r in res.results.items()}


def decide(set_key, case_id, sysm=None, state=None):
    """Ask the live catalog → (Response, answers, ms, state)."""
    S, c = _case(set_key, case_id)
    mod, base = system(set_key)
    sysm = sysm or base
    st = _state(mod, c["state"] if state is None else state)
    t0 = time.perf_counter()
    res = sysm.ask(st)
    ms = (time.perf_counter() - t0) * 1000
    return res, _answers(res), ms, st


# ---------------------------------------------------------------------------------------------- small pieces
def _same(a, g):
    if isinstance(a, list) and isinstance(g, list):
        return sorted(a) == sorted(g)
    return a == g


def _fmt(v):
    if isinstance(v, list):
        return ", ".join(map(str, v)) or "none"
    return "not stated" if v == "<not stated>" else str(v)


def _money(x):
    try:
        return f"{float(x):,.2f}"
    except (TypeError, ValueError):
        return esc(x)


def _mark(ok):
    return '<span class="vs-mk vs-ok" title="right">✓</span>' if ok else '<span class="vs-mk vs-bad" title="wrong">✗</span>'


def _qn(name):
    """A question name that may break after its underscores."""
    return esc(name).replace("_", "_<wbr>")


def _chip(text, kind=""):
    return f'<span class="vs-chip {kind}">{esc(text)}</span>'


def _short(v, n=70):
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"


def _secs(ms):
    return f"{ms / 1000:.1f} s" if ms >= 1000 else f"{ms:.0f} ms"


def _ticket(text):
    """The ticket with an injected instruction highlighted (line by line)."""
    out = []
    for line in str(text).split("\n"):
        e = esc(line)
        if any(line.lstrip().startswith(p) for p in INJECTED):
            e = f'<mark class="vs-inj" title="an instruction injected into the input">{e}</mark>'
        out.append(e)
    return '<div class="vs-ticket">' + "\n".join(out) + "</div>"


# ---------------------------------------------------------------------------------------------- the input, compactly
def _facts_R(st):
    rows = sorted(st.get("ledger") or [], key=lambda e: e.get("ts", ""))
    led = "".join(f'<tr><td class="vs-mono">{esc(str(e.get("ts", ""))[5:].replace("T", " "))}</td>'
                  f'<td class="vs-mono">{esc(e.get("id"))}</td><td>{esc(e.get("merchant"))}</td>'
                  f'<td class="vs-num">{_money(e.get("amount"))}</td>'
                  f'<td>{esc(e.get("type"))} · {esc(e.get("status"))}</td></tr>' for e in rows)
    year = str((rows[0] if rows else {}).get("ts", ""))[:4]
    pa = st.get("payout_account")
    if isinstance(pa, dict):
        free = pa.get("balance", 0) - pa.get("reserved", 0)
        payout = (f'balance {_money(pa.get("balance"))} − reserved {_money(pa.get("reserved"))} = free '
                  f'<b>{_money(free)}</b>')
    else:
        payout = '<span class="vs-bad">missing from the input</span>'
    return (f'<div class="vs-lab">Ticket</div>{_ticket(st.get("ticket", ""))}'
            f'<div class="vs-lab">Ledger, sorted by time</div><div class="vs-scroll"><table class="vs-t">'
            f'<tr><th>time ({esc(year)})</th><th>id</th><th>merchant</th><th class="vs-num">amount</th>'
            f'<th>type · status</th></tr>{led}</table></div><div class="vs-lab">Payout account</div><div>{payout}</div>')


def _facts_P(st):
    po, inv, grn = st.get("po") or {}, st.get("invoice") or {}, st.get("receipt") or {}
    sups = st.get("suppliers") or {}
    sup = sups.get(inv.get("supplier_id"), {})
    appr = st.get("approver") or {}
    lim = (st.get("approval_limits") or {}).get(appr.get("role"))
    got = {x.get("line"): x.get("qty_received") for x in grn.get("lines") or []}
    pol = {x.get("line"): x for x in po.get("lines") or []}
    lines = ""
    for x in inv.get("lines") or []:
        p = pol.get(x.get("line")) or {}
        lines += (f'<tr><td>{esc(x.get("line"))}</td><td class="vs-mono">{esc(x.get("sku"))}</td>'
                  f'<td class="vs-num">{esc(p.get("qty", "—"))} × {esc(p.get("unit_price", "—"))}</td>'
                  f'<td class="vs-num">{esc(got.get(x.get("line"), "—"))}</td>'
                  f'<td class="vs-num">{esc(x.get("qty"))} × {esc(x.get("unit_price"))}</td></tr>')
    feed = st.get("fx_feed") or {}
    rates = " · ".join(f"{esc(k)} {esc(v)}" for k, v in (st.get("fx_rates") or {}).items() if k != "USD")
    frates = " · ".join(f"{esc(k)} {esc(v)}" for k, v in (feed.get("rates") or {}).items() if k != "USD")
    paid = "; ".join(f'{esc(p.get("invoice_number"))} from {esc(p.get("supplier_id"))} ({_money(p.get("amount"))} '
                     f'{esc(p.get("currency"))})' for p in st.get("paid_invoices") or [])
    notes = inv.get("notes")
    head = (_kv("Invoice", f'<b>{esc(inv.get("invoice_number"))}</b> for <b>{esc(inv.get("po_number"))}</b>, '
                           f'{esc(inv.get("date"))}, total <b>{_money(inv.get("total"))} {esc(inv.get("currency"))}</b>')
            + _kv("PO", f'<b>{esc(po.get("po_number"))}</b> in {esc(po.get("currency"))}, supplier {esc(po.get("supplier_id"))}')
            + _kv("Supplier", f'{esc(inv.get("supplier_id"))} {esc(sup.get("name", "(not in the list)"))}: '
                              f'<b>{esc(sup.get("status", "unknown"))}</b>')
            + _kv("Approver", f'{esc(appr.get("name"))}, {esc(appr.get("role"))}: limit <b>{_money(lim)} USD</b>'))
    if notes:
        head += _kv("Notes", f'<mark class="vs-inj" title="an instruction injected into the input">{esc(notes)}</mark>')
    return (head + '<div class="vs-lab">Lines: PO qty × price · received · invoiced qty × price</div>'
            f'<div class="vs-scroll"><table class="vs-t"><tr><th>line</th><th>sku</th><th class="vs-num">PO</th>'
            f'<th class="vs-num">received</th><th class="vs-num">invoice</th></tr>{lines}</table></div>'
            + _kv("Rates to USD", rates or "none")
            + _kv("Daily feed", f'as of {esc(feed.get("as_of", "—"))}: {frates or "none"}')
            + _kv("Paid invoices", paid or "none"))


def _kv(label, value_html):
    return f'<div class="vs-kv"><span>{esc(label)}</span><span>{value_html}</span></div>'


def facts_html(set_key, case_id):
    S, c = _case(set_key, case_id)
    st = c["state"]
    body = {"R": _facts_R, "P": _facts_P}.get(set_key)
    inner = body(st) if body else f'<pre class="vs-ticket">{esc(json.dumps(st, indent=1, ensure_ascii=False))}</pre>'
    tags = "".join(_chip(t) for t in c.get("tags") or [])
    return (f'<div class="vs-case"><div class="vs-h"><b>{esc(c["id"])}</b> · {esc(c["title"])} {tags}</div>'
            f'<div class="vs-why">{esc(c["why"])}</div>{inner}</div>')


# ---------------------------------------------------------------------------------------------- side by side
def _cell_llm(x, gold):
    """A saved direct answer [answer, confidence] (None: no usable reply) → (html, right?)."""
    if x is None:
        return '<td class="vs-c vs-wrong"><i>no usable reply</i> ' + _mark(False) + "</td>", False
    a, c = x[0], x[1]
    ok = _same(a, gold)
    conf = f' <span class="vs-conf">{c:.2f}</span>' if isinstance(c, (int, float)) else ""
    loud = ("" if ok or a == ABST or not isinstance(c, (int, float)) or c < CONFIDENT
            else ' <span class="vs-chip bad" title="wrong with confidence of 0.9 or more">confident</span>')
    return (f'<td class="vs-c {"vs-right" if ok else "vs-wrong"}"><b>{esc(_fmt(a))}</b>{conf} {_mark(ok)}{loud}</td>', ok)


def _cell_solvi(r, a, gold):
    ok = _same(a, gold)
    st = r.status
    tag = _chip("hard check", "warn") if st == "forced" else ""
    return f'<td class="vs-c vs-live {"vs-right" if ok else "vs-wrong"}"><b>{esc(_fmt(a))}</b> {tag} {_mark(ok)}</td>', ok


def compare_html(set_key, case_id, res, ans, ms):
    S, c = _case(set_key, case_id)
    arms = [a for a in _arms("llm", set_key) if a["key"] in c["arms"]]
    qs = [q["name"] for q in S["questions"]]
    head = ('<tr><th>question</th><th>right answer</th><th class="vs-live">solvi <span class="vs-sub">live, '
            f'{esc(solvi_version())}</span></th>' + "".join(_head(a) for a in arms) + "</tr>")
    body, right = "", {"solvi": 0, **{a["key"]: 0 for a in arms}}
    for q in qs:
        g = c["gold"].get(q)
        cell, ok = _cell_solvi(res[q], ans[q], g)
        right["solvi"] += ok
        row = f'<tr><td class="vs-q">{_qn(q)}</td><td class="vs-gold">{esc(_fmt(g))}</td>{cell}'
        for a in arms:
            rec = c["arms"][a["key"]]
            x = (rec.get("answers") or {}).get(q) if rec.get("answers") is not None else None
            h, ok = _cell_llm(x, g)
            right[a["key"]] += ok
            row += h
        body += row + "</tr>"
    n = len(qs)
    foot = (f'<tr class="vs-foot"><td>right</td><td></td><td class="vs-live">{right["solvi"]} of {n}</td>'
            + "".join(f'<td>{right[a["key"]]} of {n}</td>' for a in arms) + "</tr>")
    foot += (f'<tr class="vs-foot"><td>time</td><td></td><td class="vs-live">{ms:.1f} ms, here</td>'
             + "".join(f'<td>{_secs(c["arms"][a["key"]]["ms"])} (one request)</td>' if c["arms"][a["key"]].get("ms")
                       else "<td></td>" for a in arms) + "</tr>")
    foot += ('<tr class="vs-foot"><td>cost</td><td></td><td class="vs-live">$0</td>'
             + "".join(f'<td>${c["arms"][a["key"]]["usd"]:.4f}</td>' if c["arms"][a["key"]].get("usd") is not None
                       else "<td></td>" for a in arms) + "</tr>")
    note = ""
    pub = c.get("published_solvi") or {}
    diff = [q for q in qs if q in pub and not _same(pub[q], ans[q])]
    if diff:
        note = ('<div class="vs-note">In the published run, solvi answered '
                + ", ".join(f"<b>{esc(q)}</b> = {esc(_fmt(pub[q]))}" for q in diff)
                + " here: the reader of the customer's claim before 0.7.0 missed paraphrases and negations. 0.7.0 "
                  "rewrote it, and that is the catalog running above (see the note under the summary table).</div>")
    g = bundle()["groups"].get("llm", {})
    return (f'<div class="vs-scroll"><table class="vs-t vs-cmp">{head}{body}{foot}</table></div>{note}'
            f'<div class="vs-note">{esc(g.get("note", ""))} ✗ with <span class="vs-chip bad">confident</span>: wrong with '
            'confidence of 0.9 or more.</div>')


def _checks_line(d):
    out = []
    for ch in d.get("checks") or []:
        v = ch.get("value")
        ok = v is True
        label = ch["name"] + (" (hard)" if ch.get("hard") else "")
        cls = "ok" if ok else ("bad" if v is False else "")
        sym = "✓" if ok else ("✗" if v is False else "—")
        out.append(f'<span class="vs-chip {cls}" title="{"hard check" if ch.get("hard") else "check"} = {esc(v)}">'
                   f'{sym} {esc(label)}</span>')
    return " ".join(out)


def solvi_html(set_key, case_id, res, sysm, st, ms):
    """What solvi did: per answer, its status, the rule or hard check that decided it and the checks it rests on; then
    the values it computed."""
    from sandbox import audit_dict
    aud = audit_dict(res, st)
    S, c = _case(set_key, case_id)
    cards = ""
    for q in [x["name"] for x in S["questions"]]:
        r = res[q]
        d = aud["answers"].get(q, {})
        a = _norm(None if r.status == "abstain" else r.answer)
        status = {"forced": _chip("decided by a hard check", "warn"), "abstain": _chip("abstains")}.get(r.status, "")
        rule = (d.get("rule") or {}).get("name")
        src = (f'rule <code>{esc(rule)}</code>' if r.status == "ok" and rule else "")
        cards += (f'<div class="vs-ans"><div><span class="vs-q">{esc(q)}</span> = <b>{esc(_fmt(a))}</b> {status}</div>'
                  f'<div class="vs-whyline">{src + ": " if src else ""}{esc(_short(r.why, 260))}</div>'
                  f'<div class="vs-checks">{_checks_line(d)}</div></div>')
    comp = []
    for rec in res.trace.records:
        if rec.kind in ("fn", "extract"):
            if rec.error:
                comp.append(f'<span class="vs-chip" title="{esc(rec.error)}">{esc(rec.name)}: not computed '
                            f'({esc(_short(rec.error, 50))})</span>')
            else:
                comp.append(f'<code>{esc(rec.name)} = {esc(_short(rec.value, 60))}</code>')
    skipped = [n for n, _ in (getattr(res.trace, "skipped", None) or [])]
    top = (f'<div class="vs-top">solvi {esc(solvi_version())} · {esc(c["id"])} · {len(res.results)} answers in <b>{ms:.1f} ms</b> · '
           f'{len(res.trace.records)} trace steps · no model, no API call</div>')
    tail = f'<div class="vs-lab">Computed along the way</div><div class="vs-comp">{" ".join(comp) or "nothing"}</div>'
    if skipped:
        tail += (f'<div class="vs-note">Not run: {esc(", ".join(skipped))} (a hard check had already decided the '
                 'answers that need them).</div>')
    return f'<div class="vs-solvi">{top}{cards}{tail}</div>'


def _reason(why):
    """A short label for why an inside-solvi answer went to a person (the full text is the cell's tooltip)."""
    m = re.match(r"confidence ([\d.]+) < ([\d.]+)", why or "")
    if m:
        if m.group(1) == m.group(2):                  # equal after rounding to two decimals
            return "confidence just below its calibrated threshold"
        return f"confidence {m.group(1)} below its calibrated threshold {m.group(2)}"
    if "invalid LLM output" in (why or ""):
        return "its output was rejected" + (": a quote not in the text" if "not in the text" in why else "")
    if (why or "").startswith("rule returned Unknown"):
        return "not stated"
    return _short(why or "", 60)


def _cell_inside(rec, q, gold):
    """A saved inside-solvi answer → (html, kind) with kind in alone_ok, alone_bad, person, abstain_ok, missing."""
    ans = (rec.get("answers") or {}).get(q)
    if ans is None:
        return '<td class="vs-c"><i>no answer</i></td>', "missing"
    a, c = ans[0], ans[1]
    why = (rec.get("why") or {}).get(q, "")
    prop = (rec.get("proposed") or {}).get(q)
    if a == ABST:
        if _same(a, gold):
            return f'<td class="vs-c vs-right" title="{esc(why)}"><b>abstains</b> {_mark(True)}</td>', "abstain_ok"
        said = ""
        if prop and prop[2] is not None:
            said = f'<div class="vs-sub">the model said {esc(_fmt(prop[2]))}' + (
                f" ({prop[1]:.2f})" if isinstance(prop[1], (int, float)) and prop[1] > 0 else "") + "</div>"
        reason = f'<div class="vs-sub">{esc(_reason(why))}</div>' if why else ""
        return f'<td class="vs-c vs-person" title="{esc(why)}"><b>→ a person</b>{said}{reason}</td>', "person"
    ok = _same(a, gold)
    forced = q in (rec.get("forced") or [])
    tag = _chip("hard check", "warn") if forced else '<span class="vs-sub">alone</span>'
    conf = f' <span class="vs-conf">{c:.2f}</span>' if isinstance(c, (int, float)) and not forced else ""
    return (f'<td class="vs-c {"vs-right" if ok else "vs-wrong"}" title="{esc(why)}"><b>{esc(_fmt(a))}</b>{conf} '
            f'{_mark(ok)} {tag}</td>', "alone_ok" if ok else "alone_bad")


def inside_html(set_key, case_id):
    S, c = _case(set_key, case_id)
    out = ""
    for gid, g in bundle()["groups"].items():
        if g.get("scored") != "escalation":
            continue
        arms = [a for a in _arms(gid, set_key) if a["key"] in c["arms"]]
        missing = [a for a in _arms(gid, set_key) if a["key"] not in c["arms"]]
        if not arms:
            if missing:
                out += (f'<div class="vs-gh">{esc(g["title"])}</div><div class="vs-note">No saved answers for this case '
                        f'({esc(", ".join(a["label"] for a in missing))} ran on a subset of the test cases).</div>')
            continue
        qs = [q["name"] for q in S["questions"]]
        head = "<tr><th>question</th><th>right answer</th>" + "".join(_head(a) for a in arms) + "</tr>"
        body, counts = "", {a["key"]: {} for a in arms}
        for q in qs:
            g_ = c["gold"].get(q)
            row = f'<tr><td class="vs-q">{_qn(q)}</td><td class="vs-gold">{esc(_fmt(g_))}</td>'
            for a in arms:
                h, kind = _cell_inside(c["arms"][a["key"]], q, g_)
                counts[a["key"]][kind] = counts[a["key"]].get(kind, 0) + 1
                row += h
            body += row + "</tr>"

        def summ(k):
            n = counts[k]
            alone = n.get("alone_ok", 0) + n.get("alone_bad", 0)
            s = f"{alone} alone ({n.get('alone_bad', 0)} wrong)"
            if n.get("person"):
                s += f", {n['person']} to a person"
            if n.get("abstain_ok"):
                s += f", {n['abstain_ok']} rightly abstained"
            return s
        foot = ('<tr class="vs-foot"><td>outcome</td><td></td>' + "".join(f"<td>{esc(summ(a['key']))}</td>" for a in arms)
                + "</tr>")
        cost = ""
        pq = [a for a in arms if a.get("per_question")]
        if pq:
            cost = ('<tr class="vs-foot"><td>per question</td><td></td>' + "".join(
                (f'<td>{_secs(a["per_question"]["ms_median"])}, '
                 f'${a["per_question"]["usd_per_1k"] / 1000:.5f} (run median / average)</td>') if a.get("per_question")
                else "<td></td>" for a in arms) + "</tr>")
        more = (f'<div class="vs-note">No saved answers for this case from {esc(", ".join(a["label"] for a in missing))} '
                '(it ran on a subset of the test cases).</div>') if missing else ""
        out += (f'<div class="vs-gh">{esc(g["title"])}</div><div class="vs-scroll"><table class="vs-t vs-cmp">{head}{body}'
                f'{foot}{cost}</table></div><div class="vs-note">{esc(g.get("note", ""))} Hover a cell for the full '
                f'reason.</div>{more}')
    return out or '<div class="vs-note">No saved inside-solvi answers for this case.</div>'


def audit_panel(res, sysm, st, set_key):
    from audit_view import audit_html
    from sandbox import serialize
    mod, _ = system(set_key)
    qs = list(sysm.questions.values()) if isinstance(getattr(sysm, "questions", None), dict) else mod.QUESTIONS
    out = serialize(res, mod.cat, st, qs, system=sysm)
    return audit_html(out, with_stats=False)


PRESS = ('<div class="vs-note">Press <b>Reorder the options</b> to see whether an answer depends on the order the '
         'options come in, or <b>Replay</b> to re-execute solvi\'s trace.</div>')


def show(set_key, case_id):
    """→ (input facts, side by side, what solvi did, inside solvi, audit, reorder/replay panel)."""
    set_key = set_key if set_key in bundle()["sets"] else next(iter(bundle()["sets"]))
    S, c = _case(set_key, case_id)
    try:
        mod, sysm = system(set_key)
        res, ans, ms, st = decide(set_key, c["id"])
        _LAST[c["id"]] = (res, sysm, st)
        cmp_ = compare_html(set_key, c["id"], res, ans, ms)
        sol = solvi_html(set_key, c["id"], res, sysm, st, ms)
        aud = audit_panel(res, sysm, st, set_key)
    except Exception as e:  # noqa: BLE001 — shown, not raised into the UI
        cmp_ = sol = f'<div class="vs-note vs-bad">solvi {esc(solvi_version())} failed on this case: {esc(type(e).__name__)}: {esc(e)}</div>'
        aud = ""
    return facts_html(set_key, c["id"]), cmp_, sol, inside_html(set_key, c["id"]), aud, PRESS


# ---------------------------------------------------------------------------------------------- reorder
def _seed(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def _shuffle_keys(x, rng):
    if isinstance(x, dict):
        ks = list(x)
        rng.shuffle(ks)
        return {k: _shuffle_keys(x[k], rng) for k in ks}
    if isinstance(x, list):
        return [_shuffle_keys(v, rng) for v in x]
    return x


def reordered_questions(questions, rng):
    """The questions with their options in another order (never the original one)."""
    out = []
    for q in questions:
        opts = list(q.answer.options)
        new = rng.sample(opts, len(opts))
        if new == opts:
            new = opts[::-1]
        out.append(dataclasses.replace(q, answer=dataclasses.replace(q.answer, options=new)))
    return out


def _flip_cell(x, y):
    """original → reordered answer of one question; → (html, changed?)"""
    if x is None or y is None:
        return f'<td class="vs-c">{esc(_fmt(x[0])) if x else "—"} → {esc(_fmt(y[0])) if y else "—"}</td>', False
    ch = not _same(x[0], y[0])
    conf = f' <span class="vs-conf">{y[1]:.2f}</span>' if len(y) > 1 and isinstance(y[1], (int, float)) else ""
    return (f'<td class="vs-c {"vs-wrong" if ch else ""}">{esc(_fmt(x[0]))} → <b>{esc(_fmt(y[0]))}</b>{conf}'
            f'{" " + _chip("changed", "bad") if ch else ""}</td>', ch)


def reorder(set_key, case_id):
    """Run solvi again with every question's options in another order and the input's keys shuffled (seeded by the case
    id), and show the saved answers of the LLMs to the reordered request next to it, where the run has them."""
    S, c = _case(set_key, case_id)
    qs_names = [q["name"] for q in S["questions"]]
    arms = [a for a in _arms("llm", set_key) if a["key"] in c["arms"]]
    try:
        mod, sysm = system(set_key)
        res0, ans0, _, _ = decide(set_key, c["id"])
        rng = random.Random(_seed(c["id"]))
        qs = reordered_questions(mod.QUESTIONS, rng)
        sys2 = _build(mod, S, qs)
        res1, ans1, _, _ = decide(set_key, c["id"], sysm=sys2, state=_shuffle_keys(c["state"], rng))
        rep = res1.trace.replay(sys2)
        same = all(_same(ans0[q], ans1[q]) for q in ans0)
        opts = " · ".join(f'<code>{esc(q.name)}</code> {esc(" / ".join(map(str, q.answer.options)))}' for q in qs)
    except Exception as e:  # noqa: BLE001
        return f'<div class="vs-note vs-bad">solvi failed: {esc(type(e).__name__)}: {esc(e)}</div>'
    head = ('<tr><th>question</th><th class="vs-live">solvi <span class="vs-sub">live</span></th>'
            + "".join(_head(a, ' <span class="vs-sub">saved</span>') for a in arms) + "</tr>")
    body, changed = "", {a["key"]: 0 for a in arms}
    for q in qs_names:
        h, _ = _flip_cell([ans0[q]], [ans1[q]])
        h = h.replace('<td class="vs-c', '<td class="vs-c vs-live', 1)
        row = f'<tr><td class="vs-q">{_qn(q)}</td>{h}'
        for a in arms:
            rec = c["arms"][a["key"]]
            if "order" not in rec:
                row += '<td class="vs-c vs-na">—</td>'
                continue
            h, ch = _flip_cell((rec.get("answers") or {}).get(q), (rec.get("order") or {}).get(q))
            changed[a["key"]] += ch
            row += h
        body += row + "</tr>"
    n = len(qs_names)
    def verdict_of(k):
        if "order" not in c["arms"][k]:
            return "not reordered in the run"
        return f"changed {changed[k]} of {n}" if changed[k] else f"the same {n} answers"
    foot = (f'<tr class="vs-foot"><td>this case</td><td class="vs-live">{f"the same {n} answers" if same else "CHANGED"}'
            '</td>' + "".join(f"<td>{verdict_of(a['key'])}</td>" for a in arms) + "</tr>")
    foot += ('<tr class="vs-foot"><td>whole set</td><td class="vs-live">0% changed (by construction)</td>' + "".join(
        (f'<td>{100 * a["sets"][set_key]["flip_order"]:.1f}% of {a["sets"][set_key].get("n_order", "?")} answers changed</td>'
         if a["sets"].get(set_key, {}).get("flip_order") is not None else "<td></td>") for a in arms) + "</tr>")
    verdict = (f'<b class="vs-okt">solvi: the same {n} answers</b>' if same else '<b class="vs-bad">solvi: answers changed</b>')
    return (f'<div class="vs-ord vs-live"><div>{verdict}, recomputed live by a System built with the options in a new '
            f'order and the input\'s JSON keys shuffled; replay of its trace: <b>{"ok" if rep["ok"] else "FAILED"}</b> '
            f'({rep["steps"]} steps). A rule returns an option by name, so its position cannot matter.</div>'
            f'<div class="vs-sub">New option order: {opts}.</div>'
            f'<div class="vs-scroll"><table class="vs-t vs-cmp">{head}{body}{foot}</table></div>'
            '<div class="vs-sub">LLMs: the saved answer to the same request with the options and JSON keys in another '
            'order. The run reordered a subsample of the test cases for each model; "—" means this case was not in '
            'it.</div></div>')


# ---------------------------------------------------------------------------------------------- replay
def replay(set_key, case_id):
    """Replay the last live decision's trace, then a copy with one computed value changed (hashes recomputed too)."""
    S, c = _case(set_key, case_id)
    try:
        if c["id"] not in _LAST:
            res, _, _, st = decide(set_key, c["id"])
            _LAST[c["id"]] = (res, system(set_key)[1], st)
        res, sysm, st = _LAST[c["id"]]
        t0 = time.perf_counter()
        rep = res.trace.replay(sysm)
        ms = (time.perf_counter() - t0) * 1000
        ok = rep["ok"]
        head = res.trace.records[-1].hash if res.trace.records else ""
        out = (f'<div class="vs-ord vs-live"><div class="vs-big {"" if ok else "vs-bad"}">'
               f'{"✓ replay ok" if ok else "✗ replay FAILED"}</div>'
               f'<div>{rep["steps"]} steps re-executed from the recorded input in {ms:.1f} ms, '
               f'{len(rep["mismatches"])} mismatches, catalog: {esc(rep.get("catalog", "?"))}. Each step\'s inputs, '
               f'value and hash are checked; the chain starts at the hash of the input '
               f'<code>{esc(str(res.trace.init_hash)[:16])}…</code> and ends at <code>{esc(str(head)[:16])}…</code>.</div>')
        for m in rep["mismatches"][:5]:
            out += f'<div class="vs-bad">step {esc(m[0])} {esc(m[1])}: {esc(m[2])}</div>'
        out += _tampered(res, sysm) + "</div>"
        return out
    except Exception as e:  # noqa: BLE001
        return f'<div class="vs-note vs-bad">replay failed: {esc(type(e).__name__)}: {esc(e)}</div>'


def _tampered(res, sysm):
    """Change one recorded value in a copy of the trace (the first computed number, else the first check), recompute
    every hash after it, and replay the copy."""
    try:
        from solvi.runtime import MISSING, vhash
    except ImportError:
        return ""
    t = copy.deepcopy(res.trace, {id(MISSING): MISSING})    # keep the "not computed" sentinel itself: it is hashed by identity
    rec = next((r for r in t.records if r.kind == "fn" and isinstance(r.value, (int, float))
                and not isinstance(r.value, bool) and not r.error), None)
    rec = rec or next((r for r in t.records if r.kind == "check" and isinstance(r.value, bool) and not r.error), None)
    if rec is None:
        return ""
    old = rec.value
    rec.value = (not old) if isinstance(old, bool) else round(old * 10 + 1, 2)
    prev = rec.prev
    for x in t.records[t.records.index(rec):]:
        x.prev = prev
        x.hash = vhash(x.body())
        prev = x.hash
    rep = t.replay(sysm)
    first = rep["mismatches"][0] if rep["mismatches"] else None
    where = f' at step {esc(first[0])} ({esc(first[1])}: {esc(first[2])})' if first else ""
    return (f'<div class="vs-sub">And a copy of the trace with <code>{esc(rec.name)}</code> changed from {esc(old)} to '
            f'{esc(rec.value)}, every later hash recomputed: replay {"still says ok" if rep["ok"] else "fails"}'
            f'{where}.</div>')


# ---------------------------------------------------------------------------------------------- summary and texts
def summary_html():
    s = bundle()["summary"]
    links = bundle()["links"]
    head = "<tr>" + "".join(f"<th>{esc(c)}</th>" for c in s["columns"]) + "</tr>"
    body = ""
    for r in s["rows"]:
        cells = "".join(f'<td class="{"vs-arm" if i == 0 else ("vs-num" if i < 8 else "vs-wide")}">'
                        + (f"<b>{esc(x)}</b>" if i == 0 and x == "solvi" else esc(x)) + "</td>" for i, x in enumerate(r))
        cls = ' class="vs-live-row"' if r[0] == "solvi" else ""
        body += f"<tr{cls}>{cells}</tr>"
    notes = "".join(f"<li>[{i}] {esc(n)}</li>" for i, n in enumerate(s["notes"], 1))
    return (f'<div class="vs-note">{esc(s["about"])}</div><div class="vs-scroll"><table class="vs-t vs-sum">{head}{body}'
            f'</table></div><ul class="vs-notes">{notes}</ul><div class="vs-note">Method, every table and the caveats: '
            f'<a href="{esc(links["docs"])}" target="_blank">solvi vs asking an LLM</a>. Data, written policies, runner '
            f'and every raw answer: <a href="{esc(links["code"])}" target="_blank">benchmarks/vs_llm on GitHub</a> '
            '(<code>bench.py score --check</code> recomputes these numbers from the saved answers, without an API '
            'key).</div>')


def intro(in_browser=True):
    where = "live in your browser" if in_browser else "live on this server"
    links = bundle()["links"]
    return (
        "**solvi vs asking an LLM.** We gave the same inputs and the same written policy to solvi and to LLMs, and "
        "saved every answer. The strong LLMs, Grok 4.7 and gpt-oss-120b, which reason before they answer, followed these "
        "short rules nearly perfectly: on refunds and 3-way invoice matching, solvi is **not** more accurate than they "
        "are. The differences are elsewhere. solvi decides in about a millisecond and costs nothing per decision, "
        "where an LLM takes seconds and costs money; reordering the options cannot change a solvi answer; every "
        "answer replays from its trace; and a hard check holds whatever else happens. A cheap LLM without reasoning, "
        "Qwen3-235B-2507, broke refund and payment limits, mostly with a confidence of 0.9 or more. Keep in mind that "
        "we wrote three of the benchmark's four sets ourselves (the gallery, refunds and 3-way match), together with "
        "the catalogs solvi runs; only the bank-message set has someone else's labels. The LLM answers below are the "
        f"saved answers of the published run, so this page calls no API; solvi runs {where}. "
        f"[Full results]({links['docs']}) · [data and code]({links['code']})")


CSS = """
.vs-case, .vs-solvi, .vs-cmp, .vs-ord { font-size: 13.5px; line-height: 1.45; }
.vs-h { font-size: 15px; margin-bottom: 4px; }
.vs-why { margin: 2px 0 8px; }
.vs-lab { font-size: 11.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; opacity: .7; margin: 10px 0 3px; }
.vs-gh { font-size: 14px; font-weight: 700; margin: 12px 0 4px; }
.vs-ticket { white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12.5px; background: rgba(100,116,139,.08); border-left: 3px solid rgba(79,70,229,.55); padding: 6px 10px; border-radius: 4px; }
mark.vs-inj { background: rgba(234,88,12,.25); color: inherit; border-radius: 3px; padding: 0 2px; }
.vs-scroll { overflow-x: auto; max-width: 100%; }
table.vs-t { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 2px 0; }
.vs-t th, .vs-t td { padding: 3px 7px; border-bottom: 1px solid var(--border-color-primary, #e2e8f0); text-align: left; vertical-align: top; }
.vs-t th { font-weight: 600; opacity: .85; white-space: nowrap; }
.vs-num { text-align: right !important; font-variant-numeric: tabular-nums; white-space: nowrap; }
.vs-mono, .vs-q { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; }
.vs-mono { white-space: nowrap; }
.vs-kv { margin: 3px 0; display: grid; grid-template-columns: 104px 1fr; gap: 6px; align-items: baseline; }
.vs-kv > span:first-child { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: .03em; opacity: .7; }
.vs-c { white-space: normal; min-width: 72px; } .vs-c b { white-space: nowrap; }
.vs-na { opacity: .5; text-align: center; }
table.vs-sum td, table.vs-sum th { white-space: nowrap; } table.vs-sum th { white-space: normal; min-width: 64px; vertical-align: bottom; }
.vs-sum td.vs-num { text-align: left !important; } .vs-live-row td { background: rgba(79,70,229,.07); }
ul.vs-notes { font-size: 12px; opacity: .85; margin: 6px 0; padding-left: 0; list-style: none; }
table.vs-sum td.vs-wide { white-space: normal; min-width: 130px; }
.vs-case .vs-t th { white-space: normal; } ul.vs-notes li { margin: 2px 0; }
.vs-right { background: rgba(22,163,74,.07); } .vs-wrong { background: rgba(220,38,38,.12); } .vs-person { background: rgba(234,88,12,.10); }
th.vs-live, td.vs-live { border-left: 2px solid rgba(79,70,229,.45); border-right: 2px solid rgba(79,70,229,.45); }
.vs-gold { font-weight: 600; }
.vs-conf { font-size: 11.5px; opacity: .7; font-variant-numeric: tabular-nums; }
.vs-sub { font-size: 11.5px; opacity: .75; white-space: normal; }
.vs-mk { font-weight: 700; margin-left: 2px; }
.vs-ok, .vs-okt { color: #15803d; } .vs-bad { color: #dc2626; }
.dark .vs-ok, .dark .vs-okt { color: #4ade80; } .dark .vs-bad { color: #f87171; }
.vs-chip { display: inline-block; font-size: 11.5px; padding: 0 7px; border-radius: 999px; border: 1px solid var(--border-color-primary, #cbd5e1); margin: 1px 2px; white-space: nowrap; font-weight: 400; }
.vs-chip.ok { background: rgba(22,163,74,.14); border-color: rgba(22,163,74,.5); }
.vs-chip.bad { background: rgba(220,38,38,.14); border-color: rgba(220,38,38,.55); }
.vs-chip.warn { background: rgba(234,88,12,.16); border-color: rgba(234,88,12,.55); }
.vs-foot td { font-size: 12px; opacity: .85; border-bottom: none; }
.vs-note { font-size: 12.5px; opacity: .8; margin: 4px 0 6px; }
.vs-top { font-size: 12.5px; margin-bottom: 6px; opacity: .85; }
.vs-ans { border: 1px solid var(--border-color-primary, #e2e8f0); border-radius: 8px; padding: 5px 10px; margin: 5px 0; }
.vs-whyline { font-size: 12px; opacity: .85; word-break: break-word; }
.vs-checks { margin-top: 2px; }
.vs-comp code, .vs-solvi code { font-family: ui-monospace, monospace; font-size: 11.5px; background: rgba(100,116,139,.12); padding: 0 4px; border-radius: 3px; margin: 1px 2px; display: inline-block; word-break: break-all; }
.vs-ord { border: 1px solid var(--border-color-primary, #e2e8f0); border-radius: 8px; padding: 6px 10px; margin: 6px 0; }
.vs-ord.vs-live { border-color: rgba(79,70,229,.5); }
.vs-flip td { background: rgba(220,38,38,.12); }
.vs-big { font-size: 16px; font-weight: 700; color: #15803d; } .dark .vs-big { color: #4ade80; }
.vs-big.vs-bad, .dark .vs-big.vs-bad { color: #dc2626; }
@media (max-width: 640px) { .vs-kv { grid-template-columns: 1fr; gap: 0; } }
"""
