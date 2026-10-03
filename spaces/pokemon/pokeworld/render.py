"""Pictures for the viewer: the player's world map as an SVG (what it knows at a decision), and a strip of the run's
decisions coloured by who made them. Positions: towns and routes at their rough place on the region's map, the places
inside them (buildings, cave floors, decks) around the outdoor place they are entered from."""
from __future__ import annotations

import html
import math
from collections import deque

from .world import place_map, place_type

OUTDOOR = {   # rough positions on the region's map (x east, y south), for drawing only
    "PALLET_TOWN": (3, 14), "ROUTE_1": (3, 12), "VIRIDIAN_CITY": (3, 10), "ROUTE_22": (1, 10),
    "ROUTE_23 (south)": (0.6, 8), "ROUTE_23 (north)": (0.6, 5.5), "INDIGO_PLATEAU": (0.6, 3), "ROUTE_2": (3, 7.6),
    "PEWTER_CITY": (3, 5), "ROUTE_3": (6, 5), "ROUTE_4 (west)": (8, 3.4), "ROUTE_4 (east)": (11, 3.4),
    "CERULEAN_CITY": (13, 3.4), "ROUTE_24": (13, 1.6), "ROUTE_25": (15.6, 0.8), "ROUTE_9": (15.6, 3.4),
    "ROUTE_10": (18.2, 5), "LAVENDER_TOWN": (18.2, 8), "ROUTE_5": (13, 5.6), "SAFFRON_CITY": (13, 8),
    "ROUTE_6": (13, 10.4), "VERMILION_CITY": (13, 12.6), "ROUTE_11": (16, 12.6), "ROUTE_12": (18.2, 11),
    "ROUTE_7 (west)": (10, 8), "ROUTE_7 (east)": (11.2, 8), "CELADON_CITY": (8, 8), "ROUTE_8 (west)": (15, 8),
    "ROUTE_8 (east)": (16.4, 8), "ROUTE_16": (5, 8), "ROUTE_17": (5, 12), "ROUTE_18": (6, 15.4),
    "FUCHSIA_CITY": (9, 15.6), "ROUTE_13": (17.4, 13.8), "ROUTE_14": (16.4, 15.2), "ROUTE_15": (12.6, 15.6),
    "ROUTE_19": (9, 17.6), "ROUTE_20": (6, 18.2), "CINNABAR_ISLAND": (3, 18.2), "ROUTE_21": (3, 16.2),
}
COLORS = {"s1": "#2563eb", "s2": "#d97706", "human": "#dc2626"}
SX, SY, OX, OY = 46, 36, 34, 26


def layout(world):
    """Every place of the world → (x, y) in grid units: outdoor places fixed, the rest around where they are entered."""
    pos = {p: xy for p, xy in OUTDOOR.items() if p in world.places}
    adj = {}
    for p, d in world.places.items():
        for e in d["exits"].values():
            adj.setdefault(p, set()).add(e["to"])
            adj.setdefault(e["to"], set()).add(p)
    parent, depth = {}, {p: 0 for p in pos}
    q = deque(sorted(pos))
    while q:
        s = q.popleft()
        for t in sorted(adj.get(s, ())):
            if t not in depth:
                depth[t], parent[t] = depth[s] + 1, s
                q.append(t)
    kids = {}
    for t, s in sorted(parent.items()):
        kids.setdefault(s, []).append(t)

    def place_kids(s):
        ks = kids.get(s, [])
        x0, y0 = pos[s]
        r = 0.75 if depth[s] == 0 else 0.55
        for i, t in enumerate(ks):
            a = (2 * math.pi * i / max(1, len(ks))) + (0.4 if depth[s] else -math.pi / 2)
            pos[t] = (x0 + r * math.cos(a), y0 + r * math.sin(a))
            place_kids(t)
    for s in sorted(OUTDOOR):
        if s in pos:
            place_kids(s)
    for p in world.places:                       # anything unreachable from the outdoors: a row at the bottom
        pos.setdefault(p, (19.5, 18 - 0.3 * len(pos) % 6))
    return pos


def _xy(p):
    return OX + SX * p[0], OY + SY * p[1]


def short(place):
    n = place_map(place).replace("_", " ").title().replace("Pokecenter", "Center")
    part = place[len(place_map(place)):]
    return n + part


def map_svg(world, pos, wmap, here=None, targets=(), trail=(), title=""):
    """The world map as the player knows it: places visited (filled), places known only as somewhere an exit led
    (hollow), confirmed exits (lines), unexplored exits (short stubs), the last moves (orange), where it is now and the
    goal's places (ringed)."""
    e = html.escape
    out = []
    known = set(wmap.states)
    seen_to = {x["to"] for x in wmap.edges.values() if x["to"] is not None}
    for (s, a), x in sorted(wmap.edges.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if s not in pos:
            continue
        x1, y1 = _xy(pos[s])
        if x["status"] == "confirmed" and x["to"] in pos and x["to"] != s:
            x2, y2 = _xy(pos[x["to"]])
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="#94a3b8" stroke-width="1.4"/>')
        elif not x["taken"]:
            ang = (sum(map(ord, str(a))) * 37 % 360) * math.pi / 180
            out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x1 + 9 * math.cos(ang):.1f}" y2="{y1 + 9 * math.sin(ang):.1f}" '
                       f'stroke="#cbd5e1" stroke-width="1.2" stroke-dasharray="2,2"/>')
    pts = [_xy(pos[p]) for p in trail if p in pos]
    if len(pts) > 1:
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out.append(f'<polyline points="{d}" fill="none" stroke="#f97316" stroke-width="3" stroke-opacity="0.75" '
                   f'stroke-linejoin="round"/>')
    for p in sorted(known | seen_to, key=str):
        if p not in pos:
            continue
        x, y = _xy(pos[p])
        outdoor = p in OUTDOOR
        r = 6 if outdoor else 3.6
        fill = "#2563eb" if p in known else "none"
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="#1e3a8a" stroke-width="1">'
                   f'<title>{e(p)} ({place_type(p)})</title></circle>')
        if outdoor and p in known and place_type(p) == "town" and p != here:
            out.append(f'<text x="{x + 8:.1f}" y="{y - 7:.1f}" font-size="11" fill="#1e293b">{e(short(p))}</text>')
    for t in targets:
        if t in pos:
            x, y = _xy(pos[t])
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="none" stroke="#16a34a" stroke-width="2.5">'
                       f'<title>goal: {e(t)}</title></circle>')
            if t in known:
                out.append(f'<text x="{x + 12:.1f}" y="{y + 14:.1f}" font-size="11" fill="#15803d">{e(short(t))}</text>')
    if here and here in pos:
        x, y = _xy(pos[here])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="#f97316" stroke="#7c2d12" stroke-width="2">'
                   f'<title>here: {e(here)}</title></circle>')
        out.append(f'<text x="{x + 11:.1f}" y="{y + 4:.1f}" font-size="12" font-weight="700" fill="#9a3412">'
                   f'{e(short(here))}</text>')
    pts = [_xy(pos[p]) for p in (known | seen_to | set(targets)) if p in pos] or [(OX, OY)]
    x0, x1 = min(x for x, _ in pts) - 60, max(x for x, _ in pts) + 140
    y0, y1 = min(y for _, y in pts) - 40, max(y for _, y in pts) + 40
    w, h = max(x1 - x0, 420), max(y1 - y0, 300)
    head = (f'<svg viewBox="{x0:.0f} {y0:.0f} {w:.0f} {h:.0f}" xmlns="http://www.w3.org/2000/svg" '
            f'style="width:100%;height:auto;max-height:640px;background:#f8fafc;border-radius:12px;'
            f'font-family:system-ui,sans-serif">')
    cap = (f'<div style="font-size:.85rem;color:#475569;margin:2px 0 6px">{e(title)}</div>' if title else "")
    legend = ('<div style="font-size:.8rem;color:#475569;margin-top:4px">● blue: visited · ○ hollow: known only as '
              'where an exit led · <span style="color:#f97316">●</span> the player now and the last moves · '
              '<span style="color:#16a34a">◯</span> the goal\'s places · dashed stub: an exit not taken yet</div>')
    return cap + head + "".join(out) + "</svg>" + legend


def strip_svg(moves, at=None):
    """The run's decisions, one bar each: blue System 1, amber System 2; the goals marked; `at` highlighted."""
    n = max(1, len(moves))
    w = max(2.0, min(8.0, 900 / n))
    out = [f'<svg viewBox="0 0 {int(w * n) + 20} 46" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:46px">']
    last = None
    for i, m in enumerate(moves):
        x = 10 + i * w
        if m["objective"] != last:
            out.append(f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="46" stroke="#64748b" stroke-width="0.6"/>')
            last = m["objective"]
        hgt = 30 if m["by"] == "s2" else 16
        out.append(f'<rect x="{x:.1f}" y="{38 - hgt}" width="{max(1.0, w - 0.6):.1f}" height="{hgt}" '
                   f'fill="{COLORS.get(m["by"], "#999")}"><title>#{m["n"]} {html.escape(m["here"])} → '
                   f'{html.escape(str(m["exit"]))} ({m["by"]})</title></rect>')
        if m["surprise"]:
            out.append(f'<text x="{x:.1f}" y="8" font-size="9" fill="#dc2626">!</text>')
    if at is not None and 0 <= at < len(moves):
        x = 10 + at * w
        out.append(f'<rect x="{x - 1:.1f}" y="2" width="{w + 2:.1f}" height="40" fill="none" stroke="#0f172a" stroke-width="1.5"/>')
    out.append("</svg>")
    return "".join(out)
