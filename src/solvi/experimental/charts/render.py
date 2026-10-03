"""The chart renderer: a VerifiedChart → SVG bytes, deterministic (no clock, no randomness, fixed number formatting), no
dependencies. Accessible: <title> and <desc> (the data as text), a <title> on every mark, direct value labels instead of
an axis (so the only numbers drawn are the verified ones), text at 12 px or more, colours at 3:1 or more against the
background and text at 4.5:1 or more. A small layout solver keeps text from overlapping: wrapped titles and labels,
vertical bars that turn horizontal when their labels do not fit, candidate positions for line labels, pushed-apart pie
labels with leader lines. Text width is estimated from a per-character table (no font files): the estimate is
deliberately wide."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from decimal import Decimal
from xml.sax.saxutils import escape

from .spec import VerifiedChart

W = 640
MARGIN = 24
FONT = "DejaVu Sans, Verdana, Arial, sans-serif"
MIN_FONT = 12
TITLE_FONT = 18
LABEL_FONT = 13
VALUE_FONT = 12
BG = "#ffffff"
INK = "#1a1a1a"
MUTED = "#4a4a4a"
AXIS = "#595959"
# categorical colours, each ≥ 3:1 against the white background (checked by contrast())
PALETTE = ["#1f5fa8", "#c2410c", "#15803d", "#7c3aed", "#be185d", "#0f766e", "#a16207", "#475569"]
NV = "n/v"                     # the mark of a proposed value that did not verify (its <title> says so)
_SCALE_SHORT = {"": "", "thousand": "k", "million": "m", "billion": "bn"}
_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}


# ------------------------------------------------------------------------------------------------------------- helpers
def _luminance(hex_color):
    h = hex_color.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b):
    """The WCAG contrast ratio of two colours (#rrggbb)."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def _cw(c):
    if c in "il.,:;|!'’ ":
        return 0.32
    if c in "fjrtI()[]-–—/":
        return 0.42
    if c in "mwMW@%":
        return 0.92
    if c.isupper():
        return 0.72
    if c.isdigit():
        return 0.64
    return 0.62


def text_width(s, size, bold=False):
    """An upper estimate of a string's width in px at a font size (sans-serif)."""
    return sum(_cw(c) for c in s) * size * (1.16 if bold else 1.0)


def wrap(s, width, size, bold=False, hard=False):
    """Greedy word wrap → lines; hard=True also breaks a word longer than the width."""
    words, lines, cur = s.split(), [], ""
    for w in words:
        if hard:
            while text_width(w, size, bold) > width and len(w) > 1:
                k = len(w)
                while k > 1 and text_width(w[:k] + "-", size, bold) > width:
                    k -= 1
                if cur:
                    lines.append(cur)
                    cur = ""
                lines.append(w[:k] + "-")
                w = w[k:]
        t = f"{cur} {w}" if cur else w
        if cur and text_width(t, size, bold) > width:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    return lines or [""]


def _n(x):
    """A coordinate: fixed two decimals, trailing zeros cut — the same bytes on every platform."""
    t = f"{x:.2f}".rstrip("0").rstrip(".")
    return "0" if t in ("-0", "") else t


def fmt_number(v: Decimal):
    """1234567.5 → "1,234,567.5" (the decimals the value was written with)."""
    t = format(v, "f")
    neg = t.startswith("-")
    t = t.lstrip("-")
    ip, _, fp = t.partition(".")
    ip = f"{int(ip):,}"
    return ("-" if neg else "") + ip + (f".{fp}" if fp else "")


def value_text(v, unit, scale):
    s = fmt_number(v) + _SCALE_SHORT[scale]
    if unit == "%":
        return s + "%"
    if unit == "pp":
        return s + " pp"
    if unit in _SYMBOL:
        return ("-" + _SYMBOL[unit] + s[1:]) if s.startswith("-") else _SYMBOL[unit] + s
    if unit == "RUB":
        return s + " ₽"
    return s


def unit_caption(unit, scale, label=""):
    if unit in ("%", "pp"):
        return {"%": "percent", "pp": "percentage points"}[unit]
    if not unit and not scale:
        return ""
    if label and not (unit in _SYMBOL or unit == "RUB"):
        unit = label
    return ", ".join(x for x in (unit, {"thousand": "thousands", "million": "millions", "billion": "billions"}.get(scale, "")) if x)


@dataclass
class Box:
    x0: float
    y0: float
    x1: float
    y1: float
    what: str

    def hits(self, o, pad=1.0):
        return self.x0 < o.x1 + pad and o.x0 < self.x1 + pad and self.y0 < o.y1 + pad and o.y0 < self.y1 + pad


@dataclass
class Drawing:
    """The SVG and what the layout solver placed: every text box (for the no-overlap and in-canvas checks), the fonts
    used and notes (a label it could not place: it is in the description)."""
    svg: str
    width: int
    height: float
    texts: list = field(default_factory=list)
    fonts: set = field(default_factory=set)
    notes: list = field(default_factory=list)


class _Canvas:
    def __init__(self):
        self.parts, self.texts, self.fonts = [], [], set()

    def text(self, x, y, s, size, anchor="start", bold=False, fill=INK, title=None):
        """y is the baseline; the box spans the ascent (0.8 em) and the descent (0.25 em)."""
        w = text_width(s, size, bold)
        x0 = x - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
        self.texts.append(Box(x0, y - 0.8 * size, x0 + w, y + 0.25 * size, s))
        self.fonts.add(size)
        attrs = f'x="{_n(x)}" y="{_n(y)}" font-size="{size}"'
        if anchor != "start":
            attrs += f' text-anchor="{anchor}"'
        if bold:
            attrs += ' font-weight="bold"'
        if fill != INK:
            attrs += f' fill="{fill}"'
        inner = escape(s) + (f"<title>{escape(title)}</title>" if title else "")
        self.parts.append(f"<text {attrs}>{inner}</text>")

    def free(self, box, others=None):
        return not any(box.hits(t) for t in (self.texts if others is None else others))

    def raw(self, s):
        self.parts.append(s)


def _header(cv, chart, y):
    for line in wrap(chart.title or "Chart", W - 2 * MARGIN, TITLE_FONT, bold=True):
        y += TITLE_FONT * 1.25
        cv.text(MARGIN, y, line, TITLE_FONT, bold=True)
    cap = unit_caption(chart.unit, chart.scale, chart.unit_label)
    if cap:
        y += LABEL_FONT * 1.4
        cv.text(MARGIN, y, cap[:1].upper() + cap[1:], LABEL_FONT, fill=MUTED)
    return y + 10


def _legend(cv, chart, y):
    if len(chart.series) < 2:
        return y
    x, row_h = MARGIN, LABEL_FONT * 1.6
    y += row_h
    for i, s in enumerate(chart.series):
        name = s.name or f"Series {i + 1}"
        w = 16 + text_width(name, LABEL_FONT) + 18
        if x + w > W - MARGIN and x > MARGIN:
            x, y = MARGIN, y + row_h
        cv.raw(f'<rect x="{_n(x)}" y="{_n(y - 11)}" width="12" height="12" fill="{PALETTE[i % len(PALETTE)]}"/>')
        cv.text(x + 16, y, name, LABEL_FONT)
        x += w
    return y + 8


# ------------------------------------------------------------------------------------------------------------- charts
def _bars_vertical(cv, chart, top):
    """→ the y where the chart ends, or None when the labels do not fit (the caller goes horizontal)."""
    n, k = len(chart.categories), len(chart.series)
    slot = (W - 2 * MARGIN) / n
    lane = slot * 0.8 / k
    bar_w = min(56.0, lane * 0.85)
    vals = [p.value for s in chart.series for p in s.points if p is not None]
    texts = {(si, ci): value_text(p.value, chart.unit, chart.scale) for si, s in enumerate(chart.series)
             for ci, p in enumerate(s.points) if p is not None}
    if any(text_width(t, VALUE_FONT) > lane - 2 for t in list(texts.values()) + [NV]):
        return None
    cat_lines = []
    for c in chart.categories:
        ls = wrap(c, slot - 6, LABEL_FONT)
        if len(ls) > 2 or any(text_width(x, LABEL_FONT) > slot - 6 for x in ls):
            return None
        cat_lines.append(ls)
    vmax, vmin = max(max(vals), 0), min(min(vals), 0)
    span = float(vmax - vmin) or 1.0
    ph = 240.0
    pad_top, pad_bot = VALUE_FONT * 1.6, (VALUE_FONT * 1.6 if vmin < 0 else 0)
    y_top = top + pad_top
    zero = y_top + ph * float(vmax) / span
    y_bot = y_top + ph
    for ci, c in enumerate(chart.categories):
        cx = MARGIN + slot * (ci + 0.5)
        for si, s in enumerate(chart.series):
            lx = cx - slot * 0.4 + lane * (si + 0.5)
            p = s.points[ci]
            color = PALETTE[si % len(PALETTE)]
            if p is None:
                cv.text(lx, zero - 4, NV, VALUE_FONT, anchor="middle", fill=MUTED,
                        title=f"{c}{' / ' + s.name if s.name else ''}: not verified in the source")
                continue
            v = float(p.value)
            h = ph * abs(v) / span
            y = zero - h if v >= 0 else zero
            t = texts[(si, ci)]
            cv.raw(f'<rect x="{_n(lx - bar_w / 2)}" y="{_n(y)}" width="{_n(bar_w)}" height="{_n(max(h, 0.5))}" '
                   f'fill="{color}"><title>{escape(c)}{escape(" / " + s.name) if s.name else ""}: {escape(t)}</title></rect>')
            ty = (y - 4) if v >= 0 else (y + h + VALUE_FONT + 2)
            cv.text(lx, ty, t, VALUE_FONT, anchor="middle")
    cv.raw(f'<line x1="{_n(MARGIN)}" y1="{_n(zero)}" x2="{_n(W - MARGIN)}" y2="{_n(zero)}" stroke="{AXIS}" stroke-width="1"/>')
    y = y_bot + pad_bot + 4
    base = y
    for ci, ls in enumerate(cat_lines):
        cx = MARGIN + slot * (ci + 0.5)
        for li, line in enumerate(ls):
            cv.text(cx, base + LABEL_FONT * (1.2 + 1.25 * li), line, LABEL_FONT, anchor="middle")
    return base + LABEL_FONT * (1.2 + 1.25 * (max(len(x) for x in cat_lines) - 1)) + 8


def _bars_horizontal(cv, chart, top):
    k = len(chart.series)
    texts = {(si, ci): value_text(p.value, chart.unit, chart.scale) for si, s in enumerate(chart.series)
             for ci, p in enumerate(s.points) if p is not None}
    vw = max([text_width(t, VALUE_FONT) for t in texts.values()] + [text_width(NV, VALUE_FONT)]) + 6
    col = min(max(text_width(c, LABEL_FONT) for c in chart.categories), (W - 2 * MARGIN) * 0.38)
    labels = [wrap(c, col, LABEL_FONT, hard=True) for c in chart.categories]
    col = max(text_width(x, LABEL_FONT) for ls in labels for x in ls)
    vals = [p.value for s in chart.series for p in s.points if p is not None]
    vmax, vmin = max(max(vals), 0), min(min(vals), 0)
    span = float(vmax - vmin) or 1.0
    x_left = MARGIN + col + 10 + (vw if vmin < 0 else 0)
    x_right = W - MARGIN - (vw if vmax > 0 else 0)
    pw = x_right - x_left
    zero = x_left + pw * float(-vmin) / span
    bar_h, gap = 18.0, 4.0
    y = top + 6
    for ci, c in enumerate(chart.categories):
        group_h = k * bar_h + (k - 1) * gap
        text_h = len(labels[ci]) * LABEL_FONT * 1.25
        row_h = max(group_h, text_h)
        gy = y + (row_h - group_h) / 2
        ty = y + (row_h - text_h) / 2
        for li, line in enumerate(labels[ci]):
            cv.text(MARGIN + col, ty + LABEL_FONT * (0.95 + 1.25 * li), line, LABEL_FONT, anchor="end")
        for si, s in enumerate(chart.series):
            p = s.points[ci]
            by = gy + si * (bar_h + gap)
            if p is None:
                cv.text(zero + 4, by + bar_h * 0.72, NV, VALUE_FONT, fill=MUTED,
                        title=f"{c}{' / ' + s.name if s.name else ''}: not verified in the source")
                continue
            v = float(p.value)
            w = pw * abs(v) / span
            x = zero if v >= 0 else zero - w
            t = texts[(si, ci)]
            cv.raw(f'<rect x="{_n(x)}" y="{_n(by)}" width="{_n(max(w, 0.5))}" height="{_n(bar_h)}" '
                   f'fill="{PALETTE[si % len(PALETTE)]}"><title>{escape(c)}{escape(" / " + s.name) if s.name else ""}: '
                   f'{escape(t)}</title></rect>')
            if v >= 0:
                cv.text(x + w + 4, by + bar_h * 0.72, t, VALUE_FONT)
            else:
                cv.text(x - 4, by + bar_h * 0.72, t, VALUE_FONT, anchor="end")
        y += row_h + 12
    cv.raw(f'<line x1="{_n(zero)}" y1="{_n(top)}" x2="{_n(zero)}" y2="{_n(y - 6)}" stroke="{AXIS}" stroke-width="1"/>')
    return y + 4


def _line(cv, chart, top, notes, ph=220.0):
    n = len(chart.categories)
    texts = {(si, ci): value_text(p.value, chart.unit, chart.scale) for si, s in enumerate(chart.series)
             for ci, p in enumerate(s.points) if p is not None}
    vw = max(text_width(t, VALUE_FONT) for t in texts.values())
    x0, x1 = MARGIN + vw / 2 + 10, W - MARGIN - vw / 2 - 10
    if n == 1:                                        # one point: in the middle
        x0 = x1 = (x0 + x1) / 2
    step = (x1 - x0) / (n - 1) if n > 1 else 0.0
    vals = [float(p.value) for s in chart.series for p in s.points if p is not None]
    vmin, vmax = min(vals), max(vals)
    if vmin == vmax:
        vmin, vmax = vmin - 1, vmax + 1
    pad = VALUE_FONT * 2.6
    y_bot = top + pad + ph

    def Y(v):
        return y_bot - ph * (v - vmin) / (vmax - vmin)

    cv.raw(f'<line x1="{_n(x0 - 10)}" y1="{_n(y_bot + pad - 4)}" x2="{_n(x1 + 10)}" y2="{_n(y_bot + pad - 4)}" '
           f'stroke="{AXIS}" stroke-width="1"/>')
    marks, placed, segs = [], [], []
    for si, s in enumerate(chart.series):
        color = PALETTE[si % len(PALETTE)]
        prev = None
        for ci, p in enumerate(s.points):
            cur = None if p is None else (x0 + step * ci, Y(float(p.value)))
            if prev is not None and cur is not None:
                segs.append((prev, cur))
            prev = cur
        run = []
        for ci, p in enumerate(s.points):
            if p is None:
                if len(run) > 1:
                    cv.raw(f'<polyline points="{" ".join(run)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
                run = []
                continue
            run.append(f"{_n(x0 + step * ci)},{_n(Y(float(p.value)))}")
        if len(run) > 1:
            cv.raw(f'<polyline points="{" ".join(run)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for si, s in enumerate(chart.series):
        color = PALETTE[si % len(PALETTE)]
        for ci, p in enumerate(s.points):
            x = x0 + step * ci
            if p is None:
                continue
            y = Y(float(p.value))
            marks.append(Box(x - 4, y - 4, x + 4, y + 4, "mark"))
            cv.raw(f'<circle cx="{_n(x)}" cy="{_n(y)}" r="4" fill="{color}"><title>{escape(chart.categories[ci])}'
                   f'{escape(" / " + s.name) if s.name else ""}: {escape(texts[(si, ci)])}</title></circle>')
    for ci in range(n):
        if any(s.points[ci] is None for s in chart.series):
            cv.text(x0 + step * ci, y_bot - 2, NV, VALUE_FONT, anchor="middle", fill=MUTED,
                    title=f"{chart.categories[ci]}: a proposed value not verified in the source")
    fixed = list(cv.texts)
    cands = [(0, -9, "middle"), (0, 9 + VALUE_FONT * 0.8, "middle"), (7, -7, "start"), (-7, -7, "end"),
             (7, 7 + VALUE_FONT * 0.8, "start"), (-7, 7 + VALUE_FONT * 0.8, "end"), (0, -23, "middle"),
             (0, 23 + VALUE_FONT * 0.8, "middle")]
    for si, s in enumerate(chart.series):
        for ci, p in enumerate(s.points):
            if p is None:
                continue
            x, y, t = x0 + step * ci, Y(float(p.value)), texts[(si, ci)]
            w = text_width(t, VALUE_FONT)
            for dx, dy, anchor in cands:
                lx = x + dx
                bx0 = lx - (w if anchor == "end" else w / 2 if anchor == "middle" else 0)
                box = Box(bx0, y + dy - 0.8 * VALUE_FONT, bx0 + w, y + dy + 0.25 * VALUE_FONT, t)
                if box.y0 < top or box.y1 > y_bot + pad - 2 or box.x0 < MARGIN or box.x1 > W - MARGIN:
                    continue
                if any(box.hits(b) for b in fixed + placed + marks) or any(_crosses(box, a, b) for a, b in segs):
                    continue
                placed.append(box)
                cv.text(lx, y + dy, t, VALUE_FONT, anchor=anchor)
                break
            else:
                return None                               # the caller retries taller, then gives up on this label
    # category labels: every k-th when they do not fit side by side
    base = y_bot + pad
    every = 1
    while every < n and max(text_width(c, LABEL_FONT) for c in chart.categories) > step * every - 6:
        every += 1
    for ci, c in enumerate(chart.categories):
        if ci % every and ci != n - 1:
            notes.append(f"category label {c!r} not drawn (no room); it is in the description")
            continue
        cv.text(x0 + step * ci, base + LABEL_FONT * 1.2, c, LABEL_FONT, anchor="middle")
    return base + LABEL_FONT * 1.2 + 10


def _crosses(box, a, b, pad=1.5):
    """Does the segment a–b pass through the box (sampled every 2 px)?"""
    (xa, ya), (xb, yb) = a, b
    k = max(1, int(math.hypot(xb - xa, yb - ya) / 2))
    for i in range(k + 1):
        x, y = xa + (xb - xa) * i / k, ya + (yb - ya) * i / k
        if box.x0 - pad <= x <= box.x1 + pad and box.y0 - pad <= y <= box.y1 + pad:
            return True
    return False


def _pie(cv, chart, top):
    s = chart.series[0]
    pts = [p for p in s.points if p is not None]
    total = sum(float(p.value) for p in pts)
    r = 110.0
    cx, cy = W / 2, top + r + 34
    a = 0.0
    labels = []
    for i, p in enumerate(pts):
        frac = float(p.value) / total
        a0, a1 = a, a + frac * 2 * math.pi
        a = a1
        color = PALETTE[i % len(PALETTE)]
        t = value_text(p.value, chart.unit, chart.scale)
        tip = f"<title>{escape(p.label)}: {escape(t)}</title>"
        if frac >= 0.999999:
            cv.raw(f'<circle cx="{_n(cx)}" cy="{_n(cy)}" r="{_n(r)}" fill="{color}">{tip}</circle>')
        else:
            xa, ya = cx + r * math.sin(a0), cy - r * math.cos(a0)
            xb, yb = cx + r * math.sin(a1), cy - r * math.cos(a1)
            large = 1 if a1 - a0 > math.pi else 0
            cv.raw(f'<path d="M{_n(cx)},{_n(cy)} L{_n(xa)},{_n(ya)} A{_n(r)},{_n(r)} 0 {large} 1 {_n(xb)},{_n(yb)} Z" '
                   f'fill="{color}" stroke="{BG}" stroke-width="1.5">{tip}</path>')
        mid = (a0 + a1) / 2
        labels.append((mid, f"{p.label}: {t}", color))
    side_w = W / 2 - r - 30 - MARGIN
    placed = {1: [], -1: []}
    for mid, text, color in labels:
        side = 1 if math.sin(mid) >= 0 else -1
        lines = wrap(text, side_w, LABEL_FONT, hard=True)
        h = len(lines) * LABEL_FONT * 1.25
        y = cy - (r + 14) * math.cos(mid) - h / 2
        placed[side].append([y, h, mid, lines])
    bottom = cy + r + 20
    for side, items in placed.items():
        items.sort(key=lambda it: it[0])
        last = top - 1e9
        for it in items:
            it[0] = max(it[0], last + 3, top)
            last = it[0] + it[1]
        for y, h, mid, lines in items:
            ex, ey = cx + (r + 8) * math.sin(mid), cy - (r + 8) * math.cos(mid)
            px, py = cx + r * math.sin(mid), cy - r * math.cos(mid)
            lx = cx + side * (r + 26)
            cv.raw(f'<polyline points="{_n(px)},{_n(py)} {_n(ex)},{_n(ey)} {_n(lx - side * 4)},{_n(y + h / 2)}" '
                   f'fill="none" stroke="{AXIS}" stroke-width="1"/>')
            for li, line in enumerate(lines):
                cv.text(lx, y + LABEL_FONT * (0.95 + 1.25 * li), line, LABEL_FONT, anchor="start" if side > 0 else "end")
            bottom = max(bottom, y + h + 6)
    return bottom


def _desc(chart):
    cap = unit_caption(chart.unit, chart.scale, chart.unit_label)
    parts = [f"{chart.kind.capitalize()} chart" + (f" in {cap}" if cap else "") + "."]
    for s in chart.series:
        vals = []
        for c, p in zip(chart.categories, s.points):
            vals.append(f"{c}: " + (value_text(p.value, chart.unit, chart.scale) if p is not None else "not verified"))
        parts.append((f"{s.name}: " if s.name else "") + "; ".join(vals) + ".")
    if chart.total is not None:
        parts.append(f"Total stated in the source: {value_text(chart.total.value, chart.unit, chart.scale)}.")
    parts.append("Every value is quoted from the source text.")
    return " ".join(parts)


def render_svg(chart: VerifiedChart, proposed=None) -> Drawing:
    """A VerifiedChart → Drawing (the SVG and its layout). proposed: how many values were proposed (the footer says how
    many of them verified). A chart the checker never produces — no categories, no verified value, a pie whose values
    do not add up to more than zero or hold a negative — raises ValueError (it cannot be drawn honestly)."""
    vals = [p.value for s in chart.series for p in s.points if p is not None]
    if not chart.categories or not vals:
        raise ValueError("render_svg: the chart has no categories or no verified value: nothing to draw")
    if chart.kind == "pie" and (sum(vals) <= 0 or any(v < 0 for v in vals)):
        raise ValueError("render_svg: a pie needs values that are not negative and add up to more than zero")
    notes = []
    uid = "c" + hashlib.sha256(chart.model_dump_json().encode()).hexdigest()[:10]
    for attempt in range(4):
        cv = _Canvas()
        y = _legend(cv, chart, _header(cv, chart, MARGIN - 6))
        if chart.kind == "pie":
            end = _pie(cv, chart, y + 8)
        elif chart.kind == "line":
            end = _line(cv, chart, y + 8, notes, ph=220.0 * (1.35 ** attempt))
            if end is None:
                if attempt < 3:
                    notes.clear()
                    continue
                cv = _Canvas()                           # no room for every label even when taller: draw bars
                y = _legend(cv, chart, _header(cv, chart, MARGIN - 6))
                notes.append("line labels did not fit: drawn as bars")
                end = _bars_vertical(cv, chart, y + 8) or _bars_horizontal(cv, chart, y + 8)
        else:
            end = _bars_vertical(cv, chart, y + 8)
            if end is None:
                cv = _Canvas()
                y = _legend(cv, chart, _header(cv, chart, MARGIN - 6))
                end = _bars_horizontal(cv, chart, y + 8)
        break
    n_ok = sum(p is not None for s in chart.series for p in s.points)
    foot = f"{n_ok} value{'s' if n_ok != 1 else ''} quoted from the source"
    if proposed is not None and proposed > n_ok:
        foot += f"; {proposed - n_ok} proposed, not verified — not drawn"
    if any(p is None for s in chart.series for p in s.points):
        foot += f" ({NV}: not verified)"
    lines = wrap(foot, W - 2 * MARGIN, MIN_FONT)
    for i, line in enumerate(lines):
        cv.text(MARGIN, end + 10 + MIN_FONT * (1 + 1.25 * i), line, MIN_FONT, fill=MUTED)
    H = math.ceil(end + 10 + MIN_FONT * (1 + 1.25 * (len(lines) - 1)) + MARGIN)
    title = escape(chart.title or "Chart")
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-labelledby="{uid}-t {uid}-d" font-family="{FONT}" fill="{INK}">'
           f'<title id="{uid}-t">{title}</title><desc id="{uid}-d">{escape(_desc(chart))}</desc>'
           f'<rect width="{W}" height="{H}" fill="{BG}"/>' + "".join(cv.parts) + "</svg>\n")
    return Drawing(svg, W, H, cv.texts, cv.fonts, notes)


__all__ = ["BG", "Box", "contrast", "Drawing", "INK", "MIN_FONT", "MUTED", "PALETTE", "render_svg"]
