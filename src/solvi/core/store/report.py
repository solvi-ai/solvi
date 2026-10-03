"""Human-readable reports for an auditor or a customer: of one decision (`res.report()`) and of a period of stored decisions
(`store.report()`), as Markdown, self-contained HTML (no external assets, every value escaped) or data.

A decision: each answer with what it rests on (given inputs, computed facts, quotes with their offsets — highlighted in the
source text —, model decisions, learned parts, checks, the rule), the safeguards that fired, the guarantee line (what the
model thresholds behind the answer promise, or that no model decided it), the models' fingerprints, the trace's hashes and
the replay status.

A period: counts by answer, status and safeguard per question, the escalation rate (abstentions: handed to a person), the
guarantee coverage of the answers a model took part in, changes of the catalog's and the models' fingerprints over the
period, and example decisions (their stored ids) for each answer, escalation and safeguard."""
from __future__ import annotations

import datetime
import html
import json
import re

from .audit import LABEL
from .audit import build

FORMATS = ("md", "html", "data")
MAX_TEXT = 20_000                   # longer source texts are shown as excerpts around the highlighted quotes
CONTEXT = 300                       # characters of context around a quote in an excerpt (HTML) ...
MD_CONTEXT = 40                     # ... and in Markdown


# ---------------------------------------------------------------- data
def _short(v, n=80):
    s = v if isinstance(v, str) else repr(v)
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def _answer_text(a):
    from ..primitives import fmt
    if a.status == "abstain" or a.answer is None:
        return "—"
    return fmt(a.answer, a.kind, a.extra)


def _guarantee(au):
    """(text, covered): covered is None when no model took part in the answer, else whether every model decision behind it
    has a calibrated threshold (act_guard / calibrate_for)."""
    if au.guarantee:
        return au.guarantee, not au.guarantee.startswith("none") and "none for some" not in au.guarantee
    c = au.counts
    if not any(c.get(k) for k in ("decided", "learned", "proposed")):
        if c.get("quoted_by_model"):
            return ("no model decided this answer: it follows from code; the models' quotes it reads were checked "
                    "literally at their offsets in the text"), None
        return "no model decided this answer: it follows from code over the given, computed and quoted facts", None
    return "none: a learned part or a model's output is behind this answer without a calibrated threshold", False


def _rests_on(au):
    """The audit of one answer as rows {"kind", "name", "value", "note"}."""
    rows = []
    for g in au.given:
        rows.append({"kind": "given", "name": g["name"], "value": _short(g["value"]), "note": ""})
    for c in au.computed:
        rows.append({"kind": "computed", "name": c["name"], "value": "—" if c.get("error") else _short(c["value"]),
                     "note": f"failed: {c['error']}" if c.get("error") else ""})
    for q in au.quoted:
        if q.get("error"):
            rows.append({"kind": "quoted", "name": q["name"], "value": "—", "note": f"rejected: {q['error']}"
                         + (f" [{q['model']}]" if q.get("model") else "")})
            continue
        how = {True: "literal", False: "derived from", None: "converted from"}[q["match"]]
        rows.append({"kind": "quoted", "name": q["name"], "value": _short(q["value"]),
                     "note": f"{q['source']}[{q['start']}:{q['end']}] {how} {q['text']!r}"
                     + (f"; confidence {q['confidence']:.2f}" if q["confidence"] < 1 else "")
                     + (f" [{q['model']}]" if q.get("model") else "")})
    for d in au.decided:
        p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((d.get("probs") or {}).items(), key=lambda t: -t[1])[:4])
        rows.append({"kind": "decided", "name": d["name"], "value": "—" if d.get("error") else _short(d["value"]),
                     "note": (f"rejected: {d['error']}; " if d.get("error") else "") + (f"({p}) " if p else "")
                     + (f"[{d['model']}]" if d.get("model") else "")})
    for x in au.learned:
        p = ", ".join(f"{k} {v:.2f}" for k, v in sorted((x.get("probs") or {}).items(), key=lambda t: -t[1])[:4])
        rows.append({"kind": "learned", "name": x["name"], "value": "—" if x.get("error") else _short(x["value"]),
                     "note": (f"({p}) " if p else "") + (f"[{x['model']}]" if x.get("model") else "")})
    for c in au.checks:
        rows.append({"kind": "check", "name": c["name"], "value": repr(c["value"]),
                     "note": ("hard" if c["hard"] else "hard or soft: not recorded" if c["hard"] is None else "soft")
                     + (", decided the answer" if c["decides"] else "")
                     + (f"; {c['error']}" if c.get("error") else "")})
    if au.rule:
        rows.append({"kind": "rule", "name": au.rule["name"], "value": "", "note": au.rule["provenance"]
                     + (f" [{au.rule['model']}]" if au.rule.get("model") else "")
                     + (f"; not computed: {au.rule['error']}" if au.rule.get("error") else "")})
    for e in au.evidence:
        rows.append({"kind": "span" if e.get("span") else "evidence", "name": "", "value": _short(e["text"]),
                     "note": f"{e['source']}[{e['start']}:{e['end']}] "
                     + (("in the text" if e["by_model"] else "verified") if e["verified"] else "NOT IN THE TEXT")})
    for c in au.constraints:
        rows.append({"kind": "constraint", "name": c["name"], "value": "", "note": "satisfied" if c["satisfied"] else "BROKEN"})
    for n, w in au.not_run:
        rows.append({"kind": "not run", "name": n, "value": "", "note": w})
    return rows


def _replay(res, system, mode):
    """The replay status: re-run the deterministic steps; model outputs verified from the record ("trusted", the default:
    no model is called) or re-run too ("full")."""
    if mode is False or mode is None:
        return {"status": "not run", "detail": "replay=False"}
    target = system if system is not None else getattr(res, "catalog", None)
    if target is None:
        return {"status": "not run", "detail": "no catalog: pass system= (or load the response with its catalog)"}
    try:
        rep = res.trace.replay(target, res.flow, trust_models=(mode != "full"))
    except Exception as e:  # noqa: BLE001
        return {"status": "failed", "detail": f"{type(e).__name__}: {e}"}
    how = "model outputs re-run" if mode == "full" else "model outputs verified from the record, not re-run"
    return {"status": "ok" if rep["ok"] else "mismatches", "steps": rep["steps"], "catalog": rep.get("catalog"),
            "changed_parts": rep.get("changed_parts", []), "mismatches": [list(m) for m in rep["mismatches"]],
            "models": [list(m) for m in rep["models"]],
            "detail": (f"{rep['steps']} steps replayed, {len(rep['mismatches'])} mismatch(es); {how}"
                       + (f"; catalog {rep['catalog']}" if rep.get("catalog") else ""))}


def decision(res, question=None, system=None, replay="trusted"):
    """The report of one response as data (a dict of plain values). question: one question or a list (default: all).
    system: the System that answered (default: the one recorded on the response) — for the replay and answer heads.
    replay: "trusted" (default: deterministic steps re-run, model outputs verified from the record — no model is called),
    "full" (models re-run too) or False."""
    system = system if system is not None else getattr(res, "_system", None)
    audit = res.audit() if question is None else None
    if audit is None:
        from .audit import build
        audit = build(res, question)
    tr = res.trace
    answers, spans = [], {}
    for q, au in audit.answers.items():
        a = res.results[q]
        g, covered = _guarantee(au)
        answers.append({
            "question": q, "text": getattr((getattr(system, "questions", None) or {}).get(q), "text", ""),
            "answer": _answer_text(a), "status": a.status, "confidence": round(float(a.confidence), 4),
            "provenance": a.provenance, "source": a.source, "why": a.why, "guard": a.guard,
            # the report is English (labels, headings, <html lang="en">), whatever System(lang=...) renders
            "support": au.support_line(lang="en"), "share_deterministic": au.share_deterministic, "rests_on": _rests_on(au),
            "guarantee": g, "guaranteed": covered,
            "safeguards": [{"kind": e["kind"], "label": LABEL.get(e["kind"], e["kind"]), "fact": e["fact"],
                            "detail": e["detail"]} for e in au.safeguards]})
        for x in au.quoted:
            if not x.get("error"):
                # a quote without an error was grounded: its offsets lie in the text. A value derived from it ("1.5
                # million" → 1500000.0, match False) or converted is still that text, not a mismatch
                spans.setdefault(x["source"], []).append({"start": x["start"], "end": x["end"], "label": f"{q}: {x['name']}",
                                                          "ok": True})
        for e in au.evidence:
            spans.setdefault(e["source"], []).append({"start": e["start"], "end": e["end"],
                                                      "label": f"{q}: {'span' if e.get('span') else 'evidence'}",
                                                      "ok": bool(e["verified"])})
    docs = []
    for src, ss in spans.items():
        text = tr.init.get(src)
        if not isinstance(text, str):
            continue
        uniq = {}
        for s in ss:
            if 0 <= s["start"] <= s["end"] <= len(text):
                k = (s["start"], s["end"])
                if k in uniq:
                    uniq[k]["label"] += "; " + s["label"]
                    uniq[k]["ok"] = uniq[k]["ok"] and s["ok"]
                else:
                    uniq[k] = dict(s)
        docs.append({"source": src, "length": len(text), "text": text, "spans": sorted(uniq.values(),
                                                                                       key=lambda s: (s["start"], s["end"]))})
    models = []
    cat = getattr(system, "catalog", None) or getattr(res, "catalog", None)
    for r in tr.records:
        if r.model is not None:
            models.append({"step": r.step, "name": r.name, "type": r.model.get("type"), "id": r.model.get("id"),
                           "fp": r.model.get("fp"), "used": True})
        for name, why in r.tried or ():               # model-backed producers that ran and were not used
            if name == r.producer or why.startswith("shadow") or cat is None or not hasattr(cat, "alternative"):
                continue
            try:
                alt = cat.alternative(r.name, name)
            except (KeyError, TypeError):
                continue
            if alt is not None and alt.model is not None:
                from ..provenance import model_info
                m = model_info(alt.model)
                models.append({"step": r.step, "name": f"{r.name} ({name}, not used: {why})", "type": m["type"],
                               "id": m["id"], "fp": m["fp"], "used": False})
    fp = getattr(tr, "fingerprint", None) or {}
    return {"kind": "decision", "stored_id": getattr(res, "stored_id", None), "answers": answers, "documents": docs,
            "models": models, "overall": _plain(res.overall), "ms": round(float(res.ms), 3),
            "trace": {"init_hash": tr.init_hash, "head": tr.records[-1].hash if tr.records else tr.init_hash,
                      "steps": len(tr.records), "catalog": fp.get("catalog"), "questions": fp.get("questions")},
            "replay": _replay(res, system, replay)}


def _plain(v):
    from ..schema import dumps
    return json.loads(dumps(v, ensure_ascii=False, default=repr))


def _when(t):
    return datetime.datetime.fromtimestamp(t).isoformat(sep=" ", timespec="seconds") if t is not None else "—"


def _shown(v):
    """A stored answer (JSON form) for people."""
    from ..schema import untag_floats
    v = untag_floats(v)                               # {"$float": "inf"} as stored → inf
    if v is None:
        return "—"
    if v == "<not stated>":
        return "not stated"
    return repr(tuple(v)) if isinstance(v, list) else repr(v)


def period(store, since=None, until=None, question=None, examples=3, system=None, **filters):
    """The report of a period of stored decisions as data. since / until: as TraceStorage.query (seconds, a datetime, a
    date or an ISO string; since <= time < until); question: only this question (and decisions that asked it); other
    `filters` go to query (status, safeguard, model, catalog). examples: stored ids shown per answer, escalation reason and
    safeguard. system: a System (or Catalog) to load the stored responses with (default: the store's)."""
    rows = store.query(question=question, since=since, until=until, **filters)
    from . import _when as _seconds
    t0, t1 = _seconds(since), _seconds(until)

    def within(t):
        return (t0 is None or t >= t0) and (t1 is None or t < t1)
    # what the period's rows no longer show: decisions erased (redacted) and corrections recorded in it — whatever
    # the other filters, since an erased record has nothing left to filter by
    erased = sum(1 for s in store.iter("ask", redacted=True) if s.data.get("redacted") and within(s.time))
    corrections = sum(1 for s in store.iter("teach", redacted=True) if within(s.time))
    cat = system if system is not None else store.catalog
    per_q, sg_total, changes = {}, {}, []
    last_cat, last_models, in_use = None, {}, {"catalogs": {}, "models": {}}
    for s in rows:
        d = s.data
        ref = {"id": s.id, "seq": s.seq, "time": _when(s.time)}
        c = d.get("catalog")
        if c:
            in_use["catalogs"][c] = in_use["catalogs"].get(c, 0) + 1
            if last_cat is not None and c != last_cat:
                changes.append({"what": "catalog", "from": last_cat, "to": c, **ref})
            last_cat = c
        for m in d.get("models") or ():
            key = f"{m.get('type')} {m.get('id')}"
            k2 = f"{key} #{str(m.get('fp'))[:8]}"
            in_use["models"][k2] = in_use["models"].get(k2, 0) + 1
            if key in last_models and last_models[key] != m.get("fp"):
                changes.append({"what": f"model {key}", "from": last_models[key], "to": m.get("fp"), **ref})
            last_models[key] = m.get("fp")
        from . import view
        results = (view(d) or {}).get("results") or {}
        audits = None
        for q, (ans, conf, status) in (d.get("answers") or {}).items():
            if question is not None and q != question:
                continue
            p = per_q.setdefault(q, {"asked": 0, "answers": {}, "status": {}, "guards": {}, "safeguards": {},
                                     "model_backed": 0, "guaranteed": 0, "deterministic": 0, "unknown": 0,
                                     "examples": {"answers": {}, "escalations": {}, "safeguards": {}}})
            p["asked"] += 1
            shown = "—" if status == "abstain" else _shown(ans)
            p["answers"][shown] = p["answers"].get(shown, 0) + 1
            p["status"][status] = p["status"].get(status, 0) + 1
            why = (results.get(q) or {}).get("why", "")
            ex = dict(ref, why=_short(why, 160), confidence=conf)
            if status == "abstain":
                g = (d.get("guards") or {}).get(q) or "abstained"
                p["guards"][g] = p["guards"].get(g, 0) + 1
                _add(p["examples"]["escalations"], g, ex, examples)
            else:
                _add(p["examples"]["answers"], shown, ex, examples)
            for kind, fact, qs in d.get("safeguards") or ():
                if q in qs:
                    p["safeguards"][kind] = p["safeguards"].get(kind, 0) + 1
                    _add(p["examples"]["safeguards"], kind, dict(ref, fact=fact, why=_short(why, 160)), examples)
            if status != "abstain":
                if audits is None:
                    audits = _audits(s, cat)
                au = audits.get(q) if audits else None
                if au is None:
                    p["unknown"] += 1
                else:
                    covered = _guarantee(au)[1]
                    if covered is None:
                        p["deterministic"] += 1
                    else:
                        p["model_backed"] += 1
                        p["guaranteed"] += bool(covered)
        for kind, fact, qs in d.get("safeguards") or ():
            if question is None or question in qs:
                sg_total[kind] = sg_total.get(kind, 0) + 1
    for p in per_q.values():
        ab = p["status"].get("abstain", 0)
        p["escalation_rate"] = ab / p["asked"] if p["asked"] else 0.0
        p["guarantee_coverage"] = p["guaranteed"] / p["model_backed"] if p["model_backed"] else None
    return {"kind": "period", "since": _when(rows[0].time) if rows else None, "until": _when(rows[-1].time) if rows else None,
            "filters": {k: str(v) for k, v in dict(filters, since=since, until=until, question=question).items() if v is not None},
            "decisions": len(rows), "erased": erased, "corrections": corrections, "questions": per_q,
            "safeguards": {k: {"label": LABEL.get(k, k), "count": v} for k, v in sorted(sg_total.items(), key=lambda t: -t[1])},
            "changes": changes, "in_use": in_use}


def _add(bucket, key, ex, n):
    xs = bucket.setdefault(key, [])
    if len(xs) < n:
        xs.append(ex)


def _audits(s, cat):
    try:                                              # a compact record is re-derived when the report has the System
        res = s.response(cat) if not s.compact else _rederived(s, cat)
        return build_audit(res).answers
    except Exception:  # noqa: BLE001
        return None


def _rederived(s, cat):
    from . import rederive
    res, _ = rederive(s.data, cat)
    if res is None:
        raise ValueError("not re-derivable")
    return res


def build_audit(res):
    return build(res)


# ---------------------------------------------------------------- rendering
def render(data, format="md"):
    """Report data (decision() / period()) → Markdown or HTML text; format="data" returns the data."""
    if format not in FORMATS:
        raise ValueError(f"format must be one of {', '.join(FORMATS)}")
    if format == "data":
        return data
    if data["kind"] == "decision":
        return _decision_md(data) if format == "md" else _page("Decision report", _decision_html(data))
    return _period_md(data) if format == "md" else _page("Decisions report", _period_html(data))


# --- Markdown
_MD_SPECIAL = "\\`*[]<>#|~"                          # these act anywhere in a line
_MD_START = re.compile(r"\s*(?:[-+=]|\d+(?=[.)]))")   # ... these only at its start: a list, a setext rule


def md(s):
    """Text escaped for Markdown, where Markdown (or HTML in it) would act: `\\ ` * [ ] < > # | ~` anywhere; "_" unless it
    is inside a word ("known_customer" stays); "!" before "["; a leading "-", "+", "=" or "1." / "1)" (a list item, a
    heading rule). Brackets after an escaped "]" are inert, so "0 mismatch(es)" and "1970-01-01" stay as written.
    Newlines are spaces (table cells and list items stay on one line)."""
    s = str(s).replace("\r", " ").replace("\n", " ")
    m = _MD_START.match(s)
    at = m.end() - (1 if m.group().strip()[:1] in "-+=" else 0) if m else -1      # the character to escape at the start
    out = []
    for i, ch in enumerate(s):
        inside = ch == "_" and 0 < i < len(s) - 1 and s[i - 1].isalnum() and s[i + 1].isalnum()
        if ch in _MD_SPECIAL or (ch == "_" and not inside) or i == at or (ch == "!" and s[i + 1:i + 2] == "["):
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _table(head, rows):
    out = ["| " + " | ".join(md(h) for h in head) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(md(c) for c in r) + " |" for r in rows]
    return out


def _excerpt(text, s, e, width):
    a, b = max(0, s - width), min(len(text), e + width)
    return (("…" if a > 0 else "") + text[a:s], text[s:e], text[e:b] + ("…" if b < len(text) else ""))


def _decision_md(d):
    L = ["# Decision report", ""]
    if d["stored_id"]:
        L.append(f"Stored id: {md(d['stored_id'])}  ")
    o = d["overall"]
    L.append(f"Overall confidence {o['confidence']:.2f}; {o['answered']} answered, {len(o['abstained'])} abstained"
             + ("; constraints violated" if not o["feasible"] else "") + ".")
    L += ["", "## Answers", ""]
    L += _table(["question", "answer", "status", "confidence", "why"],
                [[a["question"], a["answer"], a["status"], f"{a['confidence']:.2f}", a["why"]] for a in d["answers"]])
    for a in d["answers"]:
        L += ["", f"## {md(a['question'])} = {md(a['answer'])}", ""]
        if a["text"]:
            L += [f"*{md(a['text'])}*", ""]
        L.append(f"- **Status:** {md(a['status'])}, confidence {a['confidence']:.2f}"
                 + (f", by {md(a['source'])}" if a["source"] else "") + (f" ({md(a['provenance'])})" if a["provenance"] else ""))
        L.append(f"- **Why:** {md(a['why'])}")
        L.append(f"- **Support:** {md(a['support'])}")
        L.append(f"- **Guarantee:** {md(a['guarantee'])}")
        L.append("- **Safeguards:** " + ("none fired" if not a["safeguards"] else
                                         "; ".join(f"{md(e['label'])} — {md(e['fact'])}: {md(e['detail'])}" for e in a["safeguards"])))
        if a["rests_on"]:
            L += ["", "What it rests on:", ""]
            L += _table(["kind", "fact", "value", "note"], [[r["kind"], r["name"], r["value"], r["note"]] for r in a["rests_on"]])
    if d["documents"]:
        L += ["", "## Quotes in the source text", ""]
        for doc in d["documents"]:
            for sp in doc["spans"]:
                pre, mid, post = _excerpt(doc["text"], sp["start"], sp["end"], MD_CONTEXT)
                L.append(f"- {md(doc['source'])}\\[{sp['start']}:{sp['end']}\\] ({md(sp['label'])}"
                         + ("" if sp["ok"] else "; NOT the text at these offsets") + f"): {md(pre)}**{md(mid)}**{md(post)}")
    L += ["", "## Models", ""]
    if d["models"]:
        L += _table(["step", "fact", "model", "fingerprint"], [[str(m["step"]), m["name"], f"{m['type']} {m['id']}", m["fp"]]
                                                              for m in d["models"]])
    else:
        L.append("No model took part in this decision.")
    t, r = d["trace"], d["replay"]
    L += ["", "## Trace and replay", "",
          f"- Input hash: {md(t['init_hash'])}; last record hash: {md(t['head'])}; {t['steps']} records",
          f"- Catalog fingerprint: {md(t['catalog'] or 'not recorded')}; questions: {md(t['questions'] or 'not recorded')}",
          f"- Replay: **{md(r['status'])}** — {md(r['detail'])}"]
    for m in r.get("mismatches") or ():
        L.append(f"  - step {m[0]} {md(m[1])}: {md(m[2])}")
    return "\n".join(L) + "\n"


def _also(d):
    """What a period's count leaves out, said next to it: erased (redacted) decisions and corrections in the period."""
    e, c = d.get("erased", 0), d.get("corrections", 0)
    return f" ({e} erased and not shown; {c} correction(s) recorded)" if e or c else ""


def _period_md(d):
    L = ["# Decisions report", ""]
    L.append(f"{d['decisions']} stored decision(s)" + _also(d) + (f", {md(d['since'])} to {md(d['until'])}" if d["decisions"] else "")
             + (" — filters: " + ", ".join(f"{md(k)} = {md(v)}" for k, v in d["filters"].items()) if d["filters"] else "") + ".")
    for q, p in d["questions"].items():
        L += ["", f"## {md(q)}", "",
              f"- Asked {p['asked']} time(s); escalated (abstained) {p['status'].get('abstain', 0)} — "
              f"{p['escalation_rate']:.1%}"
              + (" (" + ", ".join(f"{md(LABEL.get(g, g))} {n}" for g, n in p["guards"].items()) + ")" if p["guards"] else ""),
              "- Guarantee coverage: " + (f"{p['guaranteed']}/{p['model_backed']} answers a model took part in "
                                          f"({p['guarantee_coverage']:.0%}) have a calibrated threshold" if p["model_backed"]
                                          else "no answer rested on a model")
              + f"; {p['deterministic']} answer(s) from code alone" + (f"; {p['unknown']} not loadable" if p["unknown"] else ""),
              "", "By answer:", ""]
        L += _table(["answer", "count", "examples"], [[a, str(n), ", ".join(x["id"] for x in p["examples"]["answers"].get(a, []))]
                                                     for a, n in sorted(p["answers"].items(), key=lambda t: -t[1])])
        L += ["", "By status: " + ", ".join(f"{md(k)} {n}" for k, n in p["status"].items())]
        if p["safeguards"]:
            L += ["", "Safeguards:", ""]
            L += _table(["safeguard", "count", "examples"],
                        [[LABEL.get(k, k), str(n), ", ".join(x["id"] for x in p["examples"]["safeguards"].get(k, []))]
                         for k, n in sorted(p["safeguards"].items(), key=lambda t: -t[1])])
        if p["examples"]["escalations"]:
            L += ["", "Escalations:", ""]
            for g, xs in p["examples"]["escalations"].items():
                for x in xs:
                    L.append(f"- {md(x['id'])} ({md(x['time'])}, {md(LABEL.get(g, g))}): {md(x['why'])}")
    L += ["", "## Safeguards over the period", ""]
    L += (_table(["safeguard", "events"], [[v["label"], str(v["count"])] for v in d["safeguards"].values()])
          if d["safeguards"] else ["None fired."])
    L += ["", "## Catalog and models", ""]
    for c, n in d["in_use"]["catalogs"].items():
        L.append(f"- catalog {md(c)}: {n} decision(s)")
    for m, n in d["in_use"]["models"].items():
        L.append(f"- model {md(m)}: {n} decision(s)")
    if d["changes"]:
        L += ["", "Changes:", ""]
        for c in d["changes"]:
            L.append(f"- {md(c['what'])} {md(str(c['from'])[:16])} → {md(str(c['to'])[:16])} from #{md(c['seq'])} "
                     f"({md(c['id'])}, {md(c['time'])})")
    else:
        L.append("- no change of the catalog or of a model over the period")
    return "\n".join(L) + "\n"


# --- HTML
def h(s):
    """Text escaped for HTML (quotes too)."""
    return html.escape(str(s), quote=True)


CSS = """
:root{--bg:#fff;--fg:#1d1d1f;--mut:#666;--line:#ddd;--mark:#fff1a8;--bad:#ffc9c9;--ok:#1a7f37;--err:#b42318;--card:#f7f7f8}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--fg:#e8e8ea;--mut:#9a9aa0;--line:#333;--mark:#6b5b00;--bad:#7a1f1f;
--ok:#4ac26b;--err:#ff7b72;--card:#1d1d20}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,
Roboto,sans-serif}main{max-width:1000px;margin:0 auto;padding:24px 16px}h1{font-size:1.6em;margin:0 0 .3em}
h2{font-size:1.2em;margin:1.6em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.2em}.mut{color:var(--mut)}
table{border-collapse:collapse;width:100%;margin:.5em 0;font-size:.93em;display:block;overflow-x:auto}
th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top;word-break:break-word}
th{background:var(--card)}.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;
margin:12px 0}pre.doc{white-space:pre-wrap;word-break:break-word;background:var(--card);border:1px solid var(--line);
border-radius:8px;padding:12px;max-height:480px;overflow:auto;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
mark{background:var(--mark);color:inherit;border-radius:2px}mark.bad{background:var(--bad)}code{font-family:ui-monospace,
monospace;font-size:.92em}.ok{color:var(--ok);font-weight:600}.err{color:var(--err);font-weight:600}
.gap{color:var(--mut);display:block;margin:.3em 0}ul{padding-left:1.2em}
"""


def _page(title, body):
    return (f'<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{h(title)}</title>'
            f"<style>{CSS}</style></head><body><main>\n{body}\n</main></body></html>\n")


def _html_table(head, rows):
    return ("<table><thead><tr>" + "".join(f"<th>{h(x)}</th>" for x in head) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{h(c)}</td>" for c in r) + "</tr>" for r in rows) + "</tbody></table>")


def highlight(text, spans, max_text=MAX_TEXT, context=CONTEXT):
    """A source text as escaped HTML with the spans ({"start", "end", "label", "ok"}) in <mark> (overlapping spans are
    split at every boundary); a text longer than `max_text` is cut to excerpts `context` characters around the spans."""
    cuts = {0, len(text)}
    for s in spans:
        cuts |= {s["start"], s["end"]}
    if len(text) > max_text:
        keep = []
        for s in spans:
            a, b = max(0, s["start"] - context), min(len(text), s["end"] + context)
            if keep and a <= keep[-1][1]:
                keep[-1][1] = max(keep[-1][1], b)
            else:
                keep.append([a, b])
        if not keep:
            keep = [[0, min(len(text), max_text)]]
        for a, b in keep:
            cuts |= {a, b}
    else:
        keep = [[0, len(text)]]
    pts = sorted(cuts)
    out, shown_to = [], 0
    for a, b in zip(pts, pts[1:]):
        if not any(ka <= a and b <= kb for ka, kb in keep):
            continue
        if a > shown_to and len(text) > max_text:
            out.append(f'<span class="gap">… [{shown_to}:{a}] not shown …</span>')
        on = [s for s in spans if s["start"] <= a and b <= s["end"] and s["start"] < s["end"]]
        seg = h(text[a:b])
        if on:
            cls = "" if all(s["ok"] for s in on) else ' class="bad"'
            label = "; ".join(f"{s['label']} [{s['start']}:{s['end']}]" for s in on)
            seg = f'<mark{cls} title="{h(label)}">{seg}</mark>'
        out.append(seg)
        shown_to = b
    if shown_to < len(text):
        out.append(f'<span class="gap">… [{shown_to}:{len(text)}] not shown …</span>')
    return "".join(out)


def _status(s):
    return f'<span class="{"ok" if s == "ok" else "err"}">{h(s)}</span>'


def _decision_html(d):
    o = d["overall"]
    B = ["<h1>Decision report</h1>",
         '<p class="mut">' + (f"Stored id <code>{h(d['stored_id'])}</code> · " if d["stored_id"] else "")
         + f"overall confidence {o['confidence']:.2f} · {o['answered']} answered, {len(o['abstained'])} abstained"
         + ("" if o["feasible"] else " · constraints violated") + "</p>",
         "<h2>Answers</h2>",
         _html_table(["question", "answer", "status", "confidence", "why"],
                     [[a["question"], a["answer"], a["status"], f"{a['confidence']:.2f}", a["why"]] for a in d["answers"]])]
    for a in d["answers"]:
        B.append(f'<div class="card"><h2>{h(a["question"])} = {h(a["answer"])}</h2>')
        if a["text"]:
            B.append(f'<p class="mut">{h(a["text"])}</p>')
        B.append("<ul>"
                 f"<li><b>Status:</b> {_status(a['status']) if a['status'] != 'forced' else h('forced')}, confidence "
                 f"{a['confidence']:.2f}" + (f", by <code>{h(a['source'])}</code>" if a["source"] else "")
                 + (f" ({h(a['provenance'])})" if a["provenance"] else "") + "</li>"
                 f"<li><b>Why:</b> {h(a['why'])}</li><li><b>Support:</b> {h(a['support'])}</li>"
                 f"<li><b>Guarantee:</b> {h(a['guarantee'])}</li>"
                 "<li><b>Safeguards:</b> " + ("none fired" if not a["safeguards"] else "<ul>" + "".join(
                     f"<li>{h(e['label'])} — <code>{h(e['fact'])}</code>: {h(e['detail'])}</li>" for e in a["safeguards"])
                     + "</ul>") + "</li></ul>")
        if a["rests_on"]:
            B.append("<p><b>What it rests on</b></p>")
            B.append(_html_table(["kind", "fact", "value", "note"],
                                 [[r["kind"], r["name"], r["value"], r["note"]] for r in a["rests_on"]]))
        B.append("</div>")
    if d["documents"]:
        B.append("<h2>Quotes in the source text</h2>")
        for doc in d["documents"]:
            B.append(f'<p><code>{h(doc["source"])}</code> <span class="mut">({doc["length"]} characters; '
                     f'highlighted: {len(doc["spans"])} quote(s), offsets on hover)</span></p>')
            B.append("<ul>" + "".join(f"<li><code>[{s['start']}:{s['end']}]</code> {h(s['label'])}: "
                                      f"“{h(doc['text'][s['start']:s['end']])}”"
                                      + ("" if s["ok"] else ' <span class="err">not the text at these offsets</span>')
                                      + "</li>" for s in doc["spans"]) + "</ul>")
            B.append(f'<pre class="doc">{highlight(doc["text"], doc["spans"])}</pre>')
    B.append("<h2>Models</h2>")
    B.append(_html_table(["step", "fact", "model", "fingerprint"],
                         [[m["step"], m["name"], f"{m['type']} {m['id']}", m["fp"]] for m in d["models"]])
             if d["models"] else "<p>No model took part in this decision.</p>")
    t, r = d["trace"], d["replay"]
    B.append("<h2>Trace and replay</h2><ul>"
             f"<li>Input hash <code>{h(t['init_hash'])}</code>; last record hash <code>{h(t['head'])}</code>; "
             f"{t['steps']} records</li>"
             f"<li>Catalog fingerprint <code>{h(t['catalog'] or 'not recorded')}</code>; questions "
             f"<code>{h(t['questions'] or 'not recorded')}</code></li>"
             f"<li>Replay: {_status(r['status'])} — {h(r['detail'])}"
             + ("<ul>" + "".join(f"<li>step {h(m[0])} <code>{h(m[1])}</code>: {h(m[2])}</li>" for m in r.get("mismatches") or ())
                + "</ul>" if r.get("mismatches") else "") + "</li></ul>")
    return "\n".join(B)


def _period_html(d):
    B = ["<h1>Decisions report</h1>",
         f'<p class="mut">{d["decisions"]} stored decision(s){_also(d)}'
         + (f", {h(d['since'])} to {h(d['until'])}" if d["decisions"] else "")
         + (" · filters: " + ", ".join(f"{h(k)} = {h(v)}" for k, v in d["filters"].items()) if d["filters"] else "") + "</p>"]
    for q, p in d["questions"].items():
        cov = (f"{p['guaranteed']}/{p['model_backed']} answers a model took part in ({p['guarantee_coverage']:.0%}) have a "
               "calibrated threshold" if p["model_backed"] else "no answer rested on a model")
        B.append(f'<div class="card"><h2>{h(q)}</h2><ul>'
                 f"<li>Asked {p['asked']} time(s); escalated (abstained) {p['status'].get('abstain', 0)} — "
                 f"<b>{p['escalation_rate']:.1%}</b>"
                 + (" (" + ", ".join(f"{h(LABEL.get(g, g))} {n}" for g, n in p["guards"].items()) + ")" if p["guards"] else "")
                 + f"</li><li>Guarantee coverage: {h(cov)}; {p['deterministic']} answer(s) from code alone"
                 + (f"; {p['unknown']} not loadable" if p["unknown"] else "") + "</li>"
                 "<li>By status: " + h(", ".join(f"{k} {n}" for k, n in p["status"].items())) + "</li></ul>")
        B.append(_html_table(["answer", "count", "examples"],
                             [[a, n, ", ".join(x["id"] for x in p["examples"]["answers"].get(a, []))]
                              for a, n in sorted(p["answers"].items(), key=lambda t: -t[1])]))
        if p["safeguards"]:
            B.append(_html_table(["safeguard", "count", "examples"],
                                 [[LABEL.get(k, k), n, ", ".join(x["id"] for x in p["examples"]["safeguards"].get(k, []))]
                                  for k, n in sorted(p["safeguards"].items(), key=lambda t: -t[1])]))
        if p["examples"]["escalations"]:
            B.append("<p><b>Escalations</b></p><ul>" + "".join(
                f"<li><code>{h(x['id'])}</code> <span class=\"mut\">{h(x['time'])}, {h(LABEL.get(g, g))}</span>: {h(x['why'])}</li>"
                for g, xs in p["examples"]["escalations"].items() for x in xs) + "</ul>")
        B.append("</div>")
    B.append("<h2>Safeguards over the period</h2>")
    B.append(_html_table(["safeguard", "events"], [[v["label"], v["count"]] for v in d["safeguards"].values()])
             if d["safeguards"] else "<p>None fired.</p>")
    B.append("<h2>Catalog and models</h2><ul>"
             + "".join(f"<li>catalog <code>{h(c)}</code>: {n} decision(s)</li>" for c, n in d["in_use"]["catalogs"].items())
             + "".join(f"<li>model <code>{h(m)}</code>: {n} decision(s)</li>" for m, n in d["in_use"]["models"].items())
             + "</ul>")
    B.append("<ul>" + "".join(f"<li>{h(c['what'])} <code>{h(str(c['from'])[:16])}</code> → <code>{h(str(c['to'])[:16])}</code>"
                              f" from #{h(c['seq'])} (<code>{h(c['id'])}</code>, {h(c['time'])})</li>" for c in d["changes"])
             + "</ul>" if d["changes"] else "<p>No change of the catalog or of a model over the period.</p>")
    return "\n".join(B)


__all__ = ["CSS", "decision", "highlight", "md", "period", "render"]
