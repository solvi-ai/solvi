"""The chart checker: every number of a ChartSpec against the source, then the chart type against the data. Deterministic;
what does not verify is dropped or changed with a reason, never repaired by a guess."""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from ..specialist import BLOCKED, CHANGED, DROPPED, WARNING, Checked, Issue
from ..textin import _CUR_AFTER, NUMBER_RE, _SCALES, ParseError, _digits
from .spec import SCALES, ChartSpec, VerifiedChart, VerifiedPoint, VerifiedSeries

MAX_SLICES = 8

_CURRENCIES = {"$": "USD", "usd": "USD", "us$": "USD", "dollar": "USD", "dollars": "USD",
               "€": "EUR", "eur": "EUR", "euro": "EUR", "euros": "EUR",
               "£": "GBP", "gbp": "GBP", "pound": "GBP", "pounds": "GBP",
               "₽": "RUB", "rub": "RUB", "р.": "RUB", "ruble": "RUB", "rubles": "RUB", "rouble": "RUB", "roubles": "RUB",
               "¥": "JPY", "jpy": "JPY", "yen": "JPY"}
_PERCENT = {"%", "percent", "per cent", "pct", "процент", "процентов", "процента"}
_POINTS = {"pp", "p.p.", "pp.", "percentage point", "percentage points", "п.п.", "п. п.", "пп", "п.п"}
_PP_AFTER = re.compile(r"\s?(?:pp\.?|p\.p\.|percentage[\s\-]points?|п\.\s?п\.?)(?!\w)", re.I)
# a sentence ends at . ! ? before a space, at a blank line, and at a line break before a capital, a digit or a bullet (a
# list, a table); a line break inside a hard-wrapped sentence is not an end
SENT_END = re.compile(r"[.!?](?=\s|$)|\n(?=[ \t]*(?:\n|[-*•\dA-ZА-ЯЁ]))")
_WORD = re.compile(r"[^\W\d_][\w\-]*", re.U)
_STOP_AFTER = re.compile(r"[.,;:!?()\[\]$€£₽¥\d]|\n[ \t]*\n")     # the words after a number end at the next number


def canon_unit(u):
    """A unit as written → its canonical form: "USD" / "EUR" / … for currencies, "%" for percent, "pp" for percentage
    points, otherwise the lower-cased word without a plural "s" ("tonnes" → "tonne"), "" for none."""
    t = re.sub(r"\s+", " ", str(u or "")).strip().lower()
    if not t:
        return ""
    if t in _CURRENCIES:
        return _CURRENCIES[t]
    if t.startswith("руб"):
        return "RUB"
    if t in _PERCENT:
        return "%"
    if t in _POINTS:
        return "pp"
    return _singular(t)


def _singular(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def is_measure(u):
    """A unit the source marks next to the number by a sign or code (so it is checked strictly)."""
    return u == "%" or u == "pp" or u in set(_CURRENCIES.values())


@dataclass(frozen=True)
class Reading:
    """A number as the source states it: its absolute value (scale applied), its unit and where it is."""
    start: int
    end: int
    as_written: str
    value: Decimal | None
    unit: str             # "%", "pp", a currency code, or "" (then `words` holds what follows the number)
    words: tuple
    ambiguous: str = ""   # why the number cannot be read without a guess


_NEXT_NUMBER = re.compile(r"\s?[-−]?\d")
_SIGNED_CUR = re.compile(r"(?<![\w.,/])[-−]\s?(?P<cur>[$€£₽¥])\s?$")


def _cur_after(source, e):
    """A currency right after a number — unless a digit follows it: in "2023 $120 million" the sign belongs to 120, and
    2023 is not an amount."""
    c = _CUR_AFTER.match(source, e)
    return None if c is not None and _NEXT_NUMBER.match(source, c.end()) else c


def _read(source, m, decimal=None):
    s, e = m.start(), m.end()
    cur = m.group("cur")
    unit, amb = "", ""
    neg = None if cur else _SIGNED_CUR.search(source, max(0, s - 4), s)     # "-$45 million": a negative amount
    if neg is not None and m.group("d") is not None and m.group("d")[:1] not in "-−+":
        cur, s = neg.group("cur"), neg.start()
    else:
        neg = None
    if m.group("d") is not None:
        d = m.group("d")
        after_cur = _cur_after(source, e)
        try:
            v = _digits(d, decimal)
        except ParseError as err:
            v, amb = None, str(err)
        if " " in d.replace(" ", " ").replace(" ", " ") and not cur and not m.group("s1") and not after_cur:
            amb = f"{d!r}: digits grouped by spaces without a currency may be two numbers"
        sc = m.group("s1")
        if sc and len(sc) == 1 and m.group("sp") and not cur:
            amb = f"{m.group()!r}: a one-letter scale apart from the number may be a unit (metres? millions?)"
    else:
        v, sc = Decimal(_num_word(m.group("w"))), m.group("s2")
        after_cur = _cur_after(source, e)
    if v is not None and sc:
        v = v * _SCALES[sc.lower()]
    if v is not None and neg is not None:
        v = -v
    if m.group("pct"):
        unit = "%"
    elif (pp := _PP_AFTER.match(source, e)):
        unit, e = "pp", pp.end()
    elif cur:
        unit = canon_unit(cur.strip())
    elif after_cur:
        unit = canon_unit(after_cur.group().strip())
        e = after_cur.end()
    tail = source[e:e + 48]
    stop = _STOP_AFTER.search(tail)
    words = tuple(_singular(w.lower()) for w in _WORD.findall(tail[:stop.start()] if stop else tail)[:4])
    return Reading(s, e, source[s:e], v, unit, words, amb)


def _num_word(w):
    from ..textin import _NUM_WORDS
    return _NUM_WORDS[w.lower()]


class SourceIndex:
    """Every number candidate of a source, read once."""

    def __init__(self, source, decimal=None):
        self.source = source
        self.matches = list(NUMBER_RE.finditer(source))
        self.readings = [_read(source, m, decimal) for m in self.matches]
        self.digit_forms = {re.sub(r"[^\d.,]", "", r.as_written).strip(".,") for r in self.readings}
        self.digit_forms |= {_plain_digits(r.as_written) for r in self.readings}

    def in_quote(self, qs, qe):
        """The readings whose digits lie inside [qs, qe)."""
        out = []
        for m, r in zip(self.matches, self.readings):
            g = "d" if m.group("d") is not None else "w"
            if qs <= m.start(g) and m.end(g) <= qe:
                out.append(r)
        return out

    def sentence(self, qs, qe):
        a = 0
        for m in SENT_END.finditer(self.source, 0, qs):
            a = m.end()
        m = SENT_END.search(self.source, qe)
        return self.source[a:m.end() if m else len(self.source)]


_POWERS = {Decimal(10) ** k for k in (-9, -6, -3, 3, 6, 9)}


def spec_scale_name(mult):
    return {10 ** 3: "thousand", 10 ** 6: "million", 10 ** 9: "billion"}.get(int(mult), "as is")


def _plain_digits(s):
    return re.sub(r"\D", "", s)


def _dp(v):
    e = v.as_tuple().exponent
    return max(0, -e) if isinstance(e, int) else 0


def _fmt(v):
    return format(v, "f")


def _tolerance(values):
    """How far a sum of rounded values may be from the whole: half a unit of the last shown digit, per value."""
    if not values:
        return Decimal(0)
    dp = max(_dp(v) for v in values)
    return Decimal(len(values)) * Decimal("0.5") * Decimal(10) ** -dp


class ChartChecker:
    """Checks a ChartSpec against its source. decimal: "." or "," when the source's locale says which one is the
    decimal separator (else "1.000" is ambiguous and dropped)."""

    def __init__(self, decimal=None, pie_tolerance=None):
        self.decimal = decimal
        self.pie_tolerance = pie_tolerance

    def __call__(self, spec: ChartSpec, source: str) -> Checked:
        idx = SourceIndex(source, self.decimal)
        issues, kept = [], []
        unit = canon_unit(spec.unit)
        mult = Decimal(SCALES[spec.scale])
        title = self._text_numbers(spec.title, idx, "title", issues)
        used = {}

        def verify(pt, path, u):
            """→ VerifiedPoint or None (an issue added)."""
            if pt.quote is None:
                issues.append(Issue(DROPPED, "no_quote", f"{pt.label!r} = {_fmt(pt.value)}{_unit_suffix(u, spec.scale)}: no quote in the source — "
                                    "a value without a quote is not drawn", path))
                return None
            q = pt.quote
            if q.start is not None:
                if source[q.start:q.start + len(q.text)] != q.text:
                    issues.append(Issue(DROPPED, "quote_outside", f"{pt.label!r}: the quote {q.text!r} is not in the "
                                        f"source at offset {q.start}", path))
                    return None
                places = [q.start]
            else:
                places = [m.start() for m in re.finditer(re.escape(q.text), source)]
                if not places:
                    issues.append(Issue(DROPPED, "quote_outside", f"{pt.label!r}: the quote {q.text!r} is not in the "
                                        "source", path))
                    return None
            first_reason = None
            for p in places:
                vp, why = self._verify_at(pt, idx, p, p + len(q.text), u, mult, used)
                if vp is not None:
                    used[(vp.start, vp.end)] = f"{path} ({pt.label!r})"
                    return vp
                first_reason = first_reason or why
            code, msg = first_reason
            issues.append(Issue(DROPPED, code, f"{pt.label!r} = {_fmt(pt.value)}{_unit_suffix(u, spec.scale)}: {msg}",
                                path))
            return None

        # the series: labels, numbers, units
        categories, series_out, first_unit = [], [], None
        for si, se in enumerate(spec.series):
            spath = f"series[{si}]"
            u = canon_unit(se.unit) if se.unit is not None else unit
            if first_unit is None:
                first_unit = u
            elif u != first_unit:
                issues.append(Issue(DROPPED, "unit_mismatch_series", f"series {se.name!r} is in {u or 'plain numbers'!r}, "
                                    f"the chart's axis in {first_unit or 'plain numbers'!r}: one chart, one unit", spath))
                continue
            name = self._text_numbers(se.name, idx, f"{spath}.name", issues)
            pts, seen = {}, set()
            for pi, pt in enumerate(se.points):
                path = f"{spath}.points[{pi}]"
                if not pt.label.strip():
                    issues.append(Issue(DROPPED, "no_label", f"a value {_fmt(pt.value)} without a label", path))
                    continue
                bad = self._label_numbers(pt.label, idx)
                if bad:
                    issues.append(Issue(DROPPED, "label_number", f"the label {pt.label!r} has a number not in the "
                                        f"source: {', '.join(bad)}", path))
                    continue
                if pt.label in seen:
                    issues.append(Issue(DROPPED, "duplicate_label", f"{pt.label!r} appears twice in the series", path))
                    continue
                seen.add(pt.label)
                if pt.label not in categories:
                    categories.append(pt.label)
                vp = verify(pt, path, u)
                if vp is not None:
                    pts[pt.label] = vp
                    kept.append(f"{(name + ' / ') if name else ''}{pt.label} = {_fmt(pt.value)}"
                                f"{_unit_suffix(u, spec.scale)} (source: {vp.as_written!r} at {vp.start})")
                    near = self._near(pt.label, idx.sentence(vp.start, vp.end))
                    if not near:
                        issues.append(Issue(WARNING, "label_not_near", f"{pt.label!r}: no word of the label in the "
                                            f"sentence of its value — the pairing is the proposer's", path))
            series_out.append((name, pts))

        total = None
        if spec.total is not None:
            total = verify(spec.total, "total", first_unit if first_unit is not None else unit)
            if total is not None:
                kept.append(f"total = {_fmt(total.value)}{_unit_suffix(first_unit or unit, spec.scale)} "
                            f"(source: {total.as_written!r} at {total.start})")

        n_ok = sum(len(p) for _, p in series_out)
        n_proposed = sum(len(se.points) for se in spec.series)
        if not n_ok:
            issues.append(Issue(BLOCKED, "nothing_verified", "no value verified in the source: nothing is drawn"))
            return Checked(None, issues, kept, {"proposed": n_proposed, "verified": 0})
        vseries = [VerifiedSeries(name=n, points=[p.get(c) for c in categories]) for n, p in series_out]
        kind = self._fit(spec.kind, vseries, categories, total, first_unit, issues)
        shown = next((se.unit for se in spec.series if se.unit), None) or spec.unit
        return Checked(VerifiedChart(kind=kind, title=title, unit=first_unit or "", unit_label=shown.strip(),
                                     scale=spec.scale,
                                     categories=categories, series=vseries, total=total), issues, kept,
                       {"proposed": n_proposed, "verified": n_ok})

    # --- one value at one place
    def _verify_at(self, pt, idx, qs, qe, u, mult, used):
        rs = idx.in_quote(qs, qe)
        if not rs:
            return None, ("no_number", f"the quote {idx.source[qs:qe]!r} holds no whole number")
        claimed = pt.value * mult
        best = None
        for r in rs:
            if r.ambiguous:
                best = best or ("ambiguous_number", f"the source's number is ambiguous — {r.ambiguous}")
                continue
            if r.value != claimed:
                ratio = r.value / claimed if claimed else None
                if ratio is not None and ratio in _POWERS:
                    best = best or ("scale_mismatch", f"the source states {r.as_written!r} = {_fmt(r.value)}; "
                                    f"{_fmt(pt.value)} in the chart's scale ({spec_scale_name(mult)}) is {_fmt(claimed)}")
                else:
                    best = best or ("value_mismatch", f"the quote states {r.as_written!r}, not {_fmt(pt.value)}")
                continue
            why = _unit_problem(r, u)
            if why:
                best = ("unit_mismatch", why)
                continue
            taken = next((k for k in used if k[0] < r.end and r.start < k[1]), None)    # the same number, or one that
            if taken is not None:                                                       # shares characters with it
                best = ("quote_reused", f"the number {r.as_written!r} at {r.start} is already drawn as {used[taken]}")
                continue
            return VerifiedPoint(label=pt.label, value=pt.value, start=r.start, end=r.end, as_written=r.as_written), None
        return None, best

    # --- words
    def _text_numbers(self, text, idx, path, issues):
        """A title / series name: a number in it that is not in the source is replaced by "[?]" (marked, not kept)."""
        bad = self._label_numbers(text, idx)
        if not bad:
            return text
        issues.append(Issue(CHANGED, "text_number", f"{text!r} has a number not in the source ({', '.join(bad)}): "
                            "shown as [?]", path))
        out = text
        for b in bad:
            out = out.replace(b, "[?]")
        return out

    @staticmethod
    def _label_numbers(text, idx):
        bad = []
        for m in NUMBER_RE.finditer(text or ""):
            g = m.group("d") if m.group("d") is not None else m.group()
            forms = {g.strip(".,"), _plain_digits(g)}
            if not forms & idx.digit_forms:
                bad.append(g)
        return bad

    @staticmethod
    def _near(label, sentence):
        s = sentence.lower()
        for w in re.findall(r"\w+", label.lower()):
            if len(w) < 3 and re.search(rf"(?<!\w){re.escape(w)}(?!\w)", s):
                return True
            stem = w[:5] if len(w) > 5 else w          # inflected forms: "Европа" / "Европе"
            if re.search(rf"(?<!\w){re.escape(stem)}", s):
                return True
        return False

    # --- the chart type against the data
    def _fit(self, kind, vseries, categories, total, unit, issues):
        if total is not None and len(vseries) == 1:
            parts = [p.value for p in vseries[0].points if p is not None]
            tol = _tolerance(parts + [total.value])
            gaps = sum(p is None for p in vseries[0].points)
            if abs(sum(parts) - total.value) > tol and kind != "pie":
                issues.append(Issue(WARNING, "total_mismatch", f"the values add up to {_fmt(sum(parts))}, the source "
                                    f"states a total of {_fmt(total.value)}" + (f" ({gaps} not verified)" if gaps else "")
                                    + ": not all parts are here, or they are not parts of it", "total"))
        if kind == "pie":
            why = self._pie_problem(vseries, unit, total)
            if why:
                issues.append(Issue(CHANGED, "pie_refused", f"not a pie: {why} — drawn as bars", "kind"))
                return "bar"
        if kind == "line":
            if len(categories) < 2 or not any(sum(p is not None for p in s.points) >= 2 for s in vseries):
                issues.append(Issue(CHANGED, "line_refused", "not a line: fewer than two verified points in a series — "
                                    "drawn as bars", "kind"))
                return "bar"
        return kind

    def _pie_problem(self, vseries, unit, total):
        if len(vseries) != 1:
            return "a pie shows one series"
        pts = vseries[0].points
        if any(p is None for p in pts):
            return "a slice did not verify, so the whole cannot be shown"
        if len(pts) > MAX_SLICES:
            return f"more than {MAX_SLICES} slices"
        vals = [p.value for p in pts]
        if any(v <= 0 for v in vals):
            return "a slice is zero or negative"
        tol = Decimal(str(self.pie_tolerance)) if self.pie_tolerance is not None else _tolerance(vals)
        s = sum(vals)
        if total is not None:
            if abs(s - total.value) > max(tol, _tolerance(vals + [total.value])):
                return f"the slices add up to {_fmt(s)}, the source's total is {_fmt(total.value)}"
            return ""
        if unit == "%":
            if abs(s - 100) > tol:
                return f"the shares add up to {_fmt(s)}%, not 100%"
            return ""
        return "the values are not shares of a whole (no total stated, not in %)"


def _unit_problem(r, u):
    """Why the reading's unit is not the chart's unit u, or ""."""
    src = r.unit
    if is_measure(u) or is_measure(src):
        if src != u:
            said = {"%": "percent", "pp": "percentage points"}.get(src, src) or "a plain number"
            want = {"%": "percent", "pp": "percentage points"}.get(u, u) or "a plain number"
            return f"the source gives {r.as_written!r} in {said}, the chart shows it in {want}"
        return ""
    if u and u not in r.words and not any(w.startswith(u[:5]) for w in r.words if len(u) > 5):
        after = " ".join(r.words) or "nothing"
        return f"the unit {u!r} is not next to {r.as_written!r} in the source (after it: {after!r})"
    return ""


def _unit_suffix(u, scale):
    sc = {"thousand": " thousand", "million": " million", "billion": " billion"}.get(scale, "")
    if u == "%":
        return "%"
    return f"{sc} {u}".rstrip() if u else sc
