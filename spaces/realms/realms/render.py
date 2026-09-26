"""HTML/SVG rendering for the app: the map, faction table, event log, sparklines, the solvi decision card, health panel."""
from __future__ import annotations

import html

from solvi.runtime import MISSING

from . import econ
from .world import H, TERRAIN, W

T = 24  # tile size in px
TERRAIN_COLOR = {"plains": "#b7d38a", "forest": "#5f9a52", "hills": "#b39b6b", "mountains": "#8a8580", "water": "#6aa5d8",
                 "desert": "#e7d49a"}
DEP_COLOR = {"food": "#fff176", "wood": "#2e7d32", "stone": "#9e9e9e", "gold": "#ffb300"}
UNIT_LETTER = {"warrior": "W", "archer": "A", "settler": "S", "worker": "K", "caravan": "C"}
KIND_COLOR = {"war": "#dc2626", "peace": "#16a34a", "capture": "#b45309", "fall": "#7f1d1d", "rebels": "#9333ea",
              "spawn": "#2563eb", "found": "#0d9488", "event": "#6b7280", "trade": "#ca8a04", "veto": "#db2777"}

PICK_JS = ("const e=event.target.closest('[data-k]');if(!e)return;"
           "const t=document.querySelector('#pick textarea, #pick input');if(!t)return;"
           "t.value=e.getAttribute('data-k');t.dispatchEvent(new Event('input',{bubbles:true}));")


def esc(x):
    return html.escape(str(x))


def map_svg(g, selected=None):
    parts = [f"<svg class='rm-map' viewBox='0 0 {W * T} {H * T}' onclick=\"{PICK_JS}\" "
             f"xmlns='http://www.w3.org/2000/svg' role='img' aria-label='map, turn {g.turn}'>"]
    drought = [(c, k) for k, c, _ in g.effects if k == "drought"]
    for i in range(W * H):
        x, y = (i % W) * T, (i // W) * T
        parts.append(f"<rect x='{x}' y='{y}' width='{T}' height='{T}' fill='{TERRAIN_COLOR[TERRAIN[g.terrain[i]]]}'/>")
        o = g.owner[i]
        if o >= 0 and o in g.cities:
            col = g.factions[g.cities[o]["fid"]]["color"]
            parts.append(f"<rect x='{x}' y='{y}' width='{T}' height='{T}' fill='{col}' fill-opacity='0.33'/>")
        if g.improved[i]:
            parts.append(f"<path d='M{x + 3} {y + T - 5}h{T - 6}' stroke='#5b4636' stroke-width='1.5' opacity='.6'/>")
        d = g.dep[i]
        if d:
            op = 0.35 + 0.13 * g.amt[i]
            parts.append(f"<circle cx='{x + 6}' cy='{y + 6}' r='3.6' fill='{DEP_COLOR[d]}' stroke='#333' "
                         f"stroke-width='.6' opacity='{op:.2f}'><title>{d} deposit, richness {g.amt[i]}/5</title></circle>")
    for c, _ in drought:
        cx, cy = (c % W) * T, (c // W) * T
        parts.append(f"<rect x='{cx - 4 * T}' y='{cy - 4 * T}' width='{9 * T}' height='{9 * T}' fill='#d97706' "
                     f"fill-opacity='.10' stroke='#d97706' stroke-dasharray='4 3' stroke-opacity='.6'/>")
    # territory borders
    for i in range(W * H):
        o = g.owner[i]
        if o < 0 or o not in g.cities:
            continue
        f = g.cities[o]["fid"]
        x, y = (i % W) * T, (i // W) * T
        col = g.factions[f]["color"]
        for dx, dy, seg in ((1, 0, f"M{x + T} {y}v{T}"), (-1, 0, f"M{x} {y}v{T}"), (0, 1, f"M{x} {y + T}h{T}"),
                            (0, -1, f"M{x} {y}h{T}")):
            nx, ny = i % W + dx, i // W + dy
            if 0 <= nx < W and 0 <= ny < H:
                j = ny * W + nx
                oj = g.owner[j]
                if oj >= 0 and oj in g.cities and g.cities[oj]["fid"] == f:
                    continue
            parts.append(f"<path d='{seg}' stroke='{col}' stroke-width='2'/>")
    stacks = {}
    for u in sorted(g.units.values(), key=lambda u: u["id"]):
        stacks.setdefault(u["pos"], []).append(u)
    for c in g.cities.values():
        x, y = (c["pos"] % W) * T, (c["pos"] // W) * T
        f = g.factions[c["fid"]]
        sel = selected == f"city:{c['id']}"
        cap = f["capital"] == c["id"]
        parts.append(f"<g data-k='city:{c['id']}' class='rm-click'><rect x='{x + 2}' y='{y + 2}' width='{T - 4}' "
                     f"height='{T - 4}' rx='4' fill='{f['color']}' stroke='{'#000' if sel else '#fff'}' "
                     f"stroke-width='{3 if sel else 1.5}'/><text x='{x + T / 2}' y='{y + T / 2 + 4}' font-size='11' "
                     f"text-anchor='middle' fill='#fff' font-weight='700'>{c['pop']}</text>"
                     + (f"<text x='{x + T - 3}' y='{y + 8}' font-size='9' text-anchor='middle'>★</text>" if cap else "")
                     + ("<rect x='{0}' y='{1}' width='{2}' height='{2}' fill='none' stroke='#dc2626' stroke-width='2' "
                        "stroke-dasharray='3 2' rx='5'/>".format(x, y, T) if c.get("siege") else "")
                     + f"<title>{esc(c['name'])} ({esc(f['name'])}, {f['pers']}) pop {c['pop']}, building "
                       f"{c['build'] or '—'}; click for its solvi decision</title></g>")
    for pos, us in stacks.items():
        x, y = (pos % W) * T, (pos // W) * T
        city_here = g.city_at(pos) is not None
        for k, u in enumerate(us[:3]):
            f = g.factions[u["fid"]]
            cx = x + T - 6 - 7 * k if city_here else x + 7 + 7 * k
            cy = y + T - 5 if city_here else y + T / 2 + 3
            sel = selected == f"unit:{u['id']}"
            parts.append(f"<g data-k='unit:{u['id']}' class='rm-click'><circle cx='{cx}' cy='{cy}' r='{6 if sel else 5}' "
                         f"fill='{f['color']}' stroke='{'#000' if sel else '#fff'}' stroke-width='{2 if sel else 1}'/>"
                         f"<text x='{cx}' y='{cy + 3}' font-size='7' text-anchor='middle' fill='#fff' font-weight='700'>"
                         f"{UNIT_LETTER[u['type']]}</text><title>{u['type']} #{u['id']} ({esc(f['name'])}), order "
                         f"{(u['goal'] or ['—'])[0]}{' (fortified)' if u['fort'] else ''}</title></g>")
    parts.append("</svg>")
    return "<div class='rm-mapwrap'>" + "".join(parts) + "</div>"


def factions_html(g):
    rows = []
    for s in g.summary():
        badge = " <span class='rm-badge'>learns in-game</span>" if s["pers"] == "adaptive" else ""
        if g.human == s["fid"]:
            badge += " <span class='rm-badge rm-you'>you</span>"
        wars = ", ".join(s["wars"]) or "—"
        rows.append(f"<tr><td><span class='rm-sw' style='background:{s['color']}'></span>{esc(s['name'])}</td>"
                    f"<td>{s['pers']}{badge}</td><td>{s['cities']}</td><td>{s['pop']}</td><td>{s['units']}</td>"
                    f"<td>{s['gold']} / {s['wood']} / {s['stone']}</td><td>{s['strength']}</td><td>{esc(wars)}</td></tr>")
    return ("<table class='rm-tab'><tr><th>faction</th><th>personality</th><th>cities</th><th>pop</th><th>units</th>"
            "<th>gold / wood / stone</th><th>strength</th><th>at war with</th></tr>" + "".join(rows) + "</table>")


def log_html(g, n=28):
    items = []
    for t, kind, text in list(g.log)[-n:][::-1]:
        items.append(f"<div class='rm-ev'><span class='rm-t'>t{t}</span> <span class='rm-k' "
                     f"style='color:{KIND_COLOR.get(kind, '#6b7280')}'>{kind}</span> {esc(text)}</div>")
    return "<div class='rm-log'>" + ("".join(items) or "<div class='sv-dim'>nothing yet</div>") + "</div>"


def spark(values, title, fmt="{:.2f}", color="#7c3aed", w=250, h=54, floor=None):
    vs = [v for v in values if v is not None]
    if not vs:
        return f"<div class='rm-spark'><div class='rm-st'>{esc(title)}</div><div class='sv-dim'>no data yet</div></div>"
    lo = min(vs) if floor is None else min(floor, min(vs))
    hi = max(vs)
    span = (hi - lo) or 1.0
    n = len(vs)
    pts = " ".join(f"{(k / max(1, n - 1)) * (w - 4) + 2:.1f},{h - 3 - (v - lo) / span * (h - 8):.1f}" for k, v in enumerate(vs))
    return (f"<div class='rm-spark'><div class='rm-st'>{esc(title)} <b>{fmt.format(vs[-1])}</b></div>"
            f"<svg viewBox='0 0 {w} {h}' width='100%' height='{h}' preserveAspectRatio='none'>"
            f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='1.6'/></svg>"
            f"<div class='rm-sr'>min {fmt.format(min(vs))} · max {fmt.format(max(vs))}</div></div>")


def health_html(recs, checks, perf):
    sp = [spark([r["dec_ms_med"] for r in recs], "decision ms (median)", "{:.2f}"),
          spark([r["dec_ms_p99"] for r in recs], "decision ms (p99)", "{:.2f}", "#db2777"),
          spark([r["turn_ms_med"] for r in recs], "turn ms (median)", "{:.0f}", "#0d9488"),
          spark([r["alive"] for r in recs], "factions alive", "{:.0f}", "#2563eb", floor=0),
          spark([r["pop"] for r in recs], "total population", "{:.0f}", "#16a34a", floor=0),
          spark([r["vetoes_per_100"] for r in recs], "hard-check vetoes /100 decisions", "{:.1f}", "#dc2626", floor=0),
          spark([r["state_bytes"] / 1024 for r in recs], "saved state KB", "{:.0f}", "#ca8a04", floor=0),
          spark([r["adaptive_share"] for r in recs], "adaptive score / others", "{:.2f}", "#9333ea", floor=0)]
    rows = []
    for iid, name, ok, detail in checks:
        mark = {True: "<span class='sv-badge sv-ok'>held</span>", False: "<span class='sv-badge sv-forced'>FAILED</span>",
                None: "<span class='sv-badge sv-abstain'>n/a yet</span>"}[ok]
        rows.append(f"<tr><td>{iid}</td><td>{esc(name)}</td><td>{mark}</td><td class='sv-dim'>{esc(detail)}</td></tr>")
    return (f"<div class='rm-perf'>{perf}</div><div class='rm-sparks'>{''.join(sp)}</div>"
            f"<table class='rm-tab rm-inv'>{''.join(rows)}</table>")


def _short(v, limit=110):
    s = repr(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def card_html(g, key):
    """The solvi decision card of a city, unit or stance: answer, why, the strategist's plan, hard checks, skipped steps."""
    if not key:
        return "<div class='sv-card sv-dim'>Click a city or a unit on the map to see how its faction's solvi system decided.</div>"
    item = g.last.get(key)
    head = _entity_line(g, key)
    if item is None:
        return (f"<div class='sv-card'>{head}<div class='sv-dim'>No decision recorded for it yet (cities decide when their "
                "queue empties, units every few turns). Play a turn and click again.</div></div>")
    res, q, title, pers, turn = item
    r = res[q]
    cat = g.systems[pers][0]
    badge = {"ok": ("ok", "sv-ok"), "forced": ("forced by a hard check", "sv-forced"),
             "abstain": ("abstained", "sv-abstain")}[r.status]
    out = [f"<div class='sv-card'>{head}<div class='sv-title'>{esc(title)} — turn {turn} ({pers} catalog)</div>",
           f"<div class='sv-ans'><b>{esc(q)}</b> = <code>{esc(r.answer)}</code> <span class='sv-badge {badge[1]}'>"
           f"{badge[0]}</span> <span class='sv-dim'>confidence {r.confidence:.2f} · {res.ms:.2f} ms</span>"
           f"<div class='sv-why'>why: {esc(r.why)}</div></div>"]
    if r.probs:
        top = sorted(r.probs.items(), key=lambda t: -t[1])[:5]
        out.append("<div class='sv-dim'>learned head (fit_fast + teach) probabilities: " +
                   ", ".join(f"{esc(k)} {v:.2f}" for k, v in top) + "</div>")
    hc = []
    for rec in res.trace.records:
        p = cat.parts.get(rec.name)
        if p is not None and p.kind == "check" and p.hard and rec.value is not MISSING:
            gov = ", ".join(f"{k}→{v}" for k, v in p.then.items())
            hc.append(f"<span class='sv-badge {'sv-ok' if rec.value else 'sv-forced'}' title='{esc(p.doc)}'>"
                      f"{esc(rec.name)}: {'passed' if rec.value else 'FIRED'} ({esc(gov)})</span>")
    out.append("<div class='sv-hc'>hard checks: " + (" ".join(hc) or "<span class='sv-dim'>none ran</span>") + "</div>")
    if res.trace.skipped:
        out.append("<div class='rm-skip'><b>early exit</b> — skipped at run time: " +
                   ", ".join(f"<code>{esc(n)}</code>" for n, _ in res.trace.skipped) +
                   f" <span class='sv-dim'>({esc(res.trace.skipped[0][1])})</span></div>")
    plan = []
    for k, s in enumerate(res.flow.steps, 1):
        p = s.part
        plan.append(f"<tr><td>{k}</td><td class='sv-dim'>{p.kind}{' HARD' if getattr(p, 'hard', False) else ''}</td>"
                    f"<td><code>{esc(p.name)}</code></td><td class='sv-dim'>{esc(', '.join(p.inputs))}</td>"
                    f"<td class='sv-dim'>{esc('; '.join(s.reasons))}</td></tr>")
    not_taken = sorted(k for k, v in res.flow.skipped.items() if v == "not needed for questions")
    out.append(f"<details open><summary>the strategist's plan: {len(res.flow.steps)} steps of {len(cat.parts) + len(cat.rules)} "
               f"catalog parts</summary><table class='sv-facts'>{''.join(plan)}</table>"
               f"<div class='sv-dim'>not taken (not needed for this question): {esc(', '.join(not_taken)) or '—'}</div></details>")
    facts = "".join(f"<tr><td>{esc(rec.name)}</td><td class='sv-dim'>{rec.kind}</td><td><code>{esc(_short(rec.value))}</code>"
                    f"</td></tr>" for rec in res.trace.records if rec.kind != "rule" and rec.value is not MISSING)
    inputs = "".join(f"<tr><td>{esc(k)}</td><td><code>{esc(_short(v))}</code></td></tr>" for k, v in res.trace.init.items())
    rep = res.trace.replay(cat)
    out.append(f"<details><summary>facts computed ({len(res.trace.records)} trace records)</summary>"
               f"<table class='sv-facts'>{facts}</table></details>"
               f"<details><summary>init_state the engine passed in</summary><table class='sv-facts'>{inputs}</table></details>"
               f"<div class='sv-hc'>trace replay: <span class='sv-badge {'sv-ok' if rep['ok'] else 'sv-forced'}'>"
               f"{'OK' if rep['ok'] else 'MISMATCH'}</span> <span class='sv-dim'>{rep['steps']} steps recomputed from the "
               f"recorded init_state, hash chain {res.trace.init_hash} → {res.trace.records[-1].hash if res.trace.records else '—'}"
               f"</span></div></div>")
    return "".join(out)


def _entity_line(g, key):
    kind, _, rest = key.partition(":")
    if kind == "city" and int(rest) in g.cities:
        c = g.cities[int(rest)]
        f = g.factions[c["fid"]]
        ys = econ.city_yields(g.worked(c)[0], c["buildings"])
        return (f"<div class='rm-ent'><span class='rm-sw' style='background:{f['color']}'></span><b>{esc(c['name'])}</b> "
                f"({esc(f['name'])}, {f['pers']}{', capital' if f['capital'] == c['id'] else ''}) · pop {c['pop']} · "
                f"food {c['food']}/{econ.growth_threshold(c['pop'])} · yields f{ys[0]} w{ys[1]} s{ys[2]} g{ys[3]} · "
                f"building <b>{c['build'] or '—'}</b> ({c['prog']}/{econ.COST[c['build']][0] if c['build'] else '—'}) · "
                f"{', '.join(c['buildings']) or 'no buildings'}</div>")
    if kind == "unit" and int(rest) in g.units:
        u = g.units[int(rest)]
        f = g.factions[u["fid"]]
        return (f"<div class='rm-ent'><span class='rm-sw' style='background:{f['color']}'></span><b>{u['type']} #{u['id']}</b>"
                f" ({esc(f['name'])}) at ({u['pos'] % W},{u['pos'] // W}) · order <b>{(u['goal'] or ['—'])[0]}</b>"
                f"{' · fortified' if u['fort'] else ''}</div>")
    if kind == "stance":
        a, b = rest.split(">")
        return f"<div class='rm-ent'>{esc(g.factions[int(a)]['name'])} about {esc(g.factions[int(b)]['name'])}</div>"
    return "<div class='rm-ent sv-dim'>(this entity no longer exists; its last decision is shown)</div>"
