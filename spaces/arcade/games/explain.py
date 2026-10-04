"""Shared helpers: turn a solvi Response into a readable "why" panel (rule inputs, facts computed this turn, hard checks)."""
from __future__ import annotations

import html

from solvi.core.runtime import MISSING


def _short(v, limit=90):
    s = repr(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def hard_checks(resp, catalog):
    """[(name, passed)] for every hard check that ran in this response."""
    out = []
    for r in resp.trace.records:
        p = catalog.parts.get(r.name)
        if p is not None and p.kind == "check" and p.hard and r.value is not MISSING:
            out.append((r.name, bool(r.value)))
    return out


def facts(resp):
    """[(name, value, kind)] for every fact computed in this response (rules excluded)."""
    return [(r.name, r.value, r.kind) for r in resp.trace.records if r.kind != "rule" and r.value is not MISSING]


def why_html(resp, catalog, questions, show_facts=True, title=None):
    """An HTML card: answers with status/why, hard checks, and (optionally) all facts computed this turn."""
    rows = []
    for q in questions:
        r = resp[q]
        badge = {"ok": "ok", "forced": "forced by a hard check", "abstain": "abstained"}[r.status]
        cls = {"ok": "sv-ok", "forced": "sv-forced", "abstain": "sv-abstain"}[r.status]
        rows.append(f"<div class='sv-ans'><b>{html.escape(q)}</b> = <code>{html.escape(str(r.answer))}</code> "
                    f"<span class='sv-badge {cls}'>{badge}</span> <span class='sv-dim'>confidence {r.confidence:.2f}</span>"
                    f"<div class='sv-why'>why: {html.escape(r.why)}</div></div>")
    hc = hard_checks(resp, catalog)
    if hc:
        items = " ".join(f"<span class='sv-badge {'sv-ok' if ok else 'sv-forced'}'>{html.escape(n)}: "
                         f"{'passed' if ok else 'FIRED'}</span>" for n, ok in hc)
        rows.append(f"<div class='sv-hc'>hard checks: {items}</div>")
    else:
        rows.append("<div class='sv-hc sv-dim'>hard checks: none in this flow</div>")
    if show_facts:
        fr = "".join(f"<tr><td>{html.escape(n)}</td><td class='sv-dim'>{k}</td><td><code>{html.escape(_short(v))}</code></td></tr>"
                     for n, v, k in facts(resp))
        rows.append(f"<details open><summary>facts computed this turn ({len(facts(resp))} steps, "
                    f"{resp.ms:.2f} ms)</summary><table class='sv-facts'>{fr}</table></details>")
    head = f"<div class='sv-title'>{html.escape(title)}</div>" if title else ""
    return f"<div class='sv-card'>{head}{''.join(rows)}</div>"
