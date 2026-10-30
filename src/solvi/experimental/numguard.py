"""Exact values through a side channel: the model writes pointers, code writes the numbers, a guard checks that every
number in the final text has a source (experimental).

A model that types numbers into free text makes some of them up. Here it writes pointers instead:

- `[k]` — the k-th numeral of the input text (1-based, in reading order; `annotate(source)` shows the numbering);
- `<calc>expr</calc>` / `<date>expr</date>` — a call computed exactly by code (`Decimal` arithmetic, calendar
  arithmetic); the calls are numbered in order of appearance, all kinds together, and `@k` (or `{@k}`) is the
  value of the k-th call;
- `<table>` (a cell, or sum / avg / count / max / min / argmax / argmin over a column or a row of a table in the
  source), `<cmp>` (comparisons and and / or / not, a yes/no), `<scale>` (thousand ↔ million, percent ↔ fraction,
  seconds ↔ hours ↔ days ↔ weeks, months ↔ years, km/h ↔ m/s) and `<count>` (how many items match, membership) —
  typed calls computed by code the same way (`table_calc`, `cmp_calc`, `scale_calc`, `count_calc` on their own).
  `MODES` lists every call kind with its slot grammar, result type and validator; `register_mode` adds one.

`fill` replaces the pointers with the exact values (an input numeral as the input writes it: "$1,234", "12.5%",
"4.2 million") and each call with its result; `check` reads every number left in a text and names its source — an
input numeral, a call's result, or an allowed constant — or marks it `unsourced`; `NumberGuard` runs both with one
configuration and a fail-closed mode (raise, or return a refusal) or a report mode; `NumberGuard.run_with_retry` asks
your model once more with the list of what failed (any callable: no LLM client here).

    from solvi.experimental.numguard import NumberGuard, annotate

    source = "Revenue was $1,250 million in 2023 and $1,100 million in 2022. Invoice dated 2024-03-01, net 45 days."
    prompt_text, _ = annotate(source)       # "Revenue was [1]$1,250 million in [2]2023 ..." for the prompt
    reply = "Revenue grew by <calc>[1] - [3]</calc> million (<calc>(@1) / [3] * 100</calc>%). Due <date>[5] + [6]D</date>."
    out = NumberGuard().run(reply, source)
    out.text            # 'Revenue grew by 150 million (13.63636364%). Due 2024-04-15.'
    out.verdict.ok      # True: every number has a source

What it gives is provenance, not correctness: a pointer at the wrong input numeral is filled and passes. Numbers
written as words ("three") and numbers inside identifiers ("A-10457", "COVID-19", "v2") are not checked (identifiers
can be checked against the input verbatim with `ids="check"`). Only integers 0–12 and 100 pass as constants by
default (`SMALL`); `MEASURED_CONSTANTS` is the wider set the measurements used (integers 0–100 and a few common
factors), opt-in, and `years()` or your own set or predicate extend either. `pi` (`PI`, 28 digits, an
approximation) is opt-in the same way: `constants=SMALL | {PI}`."""
from __future__ import annotations

import calendar
import datetime as dt
import re
import typing
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from functools import lru_cache
from types import MappingProxyType
from typing import Callable, Iterable, Literal

from . import warn_on_import

warn_on_import(__name__)

__all__ = ["Numeral", "Inserted", "CallResult", "Filled", "NumberCheck", "Verdict", "Outcome", "NumberGuard",
           "UnsourcedNumbers", "ProgramError", "numerals", "annotate", "calc", "date_calc", "fill", "check",
           "MEASURED_CONSTANTS", "SMALL", "years", "Table", "Cell", "parse_tables", "table_calc",
           "cmp_calc", "scale_calc", "count_calc", "PI", "Mode", "CallContext", "MODES", "register_mode",
           "unregister_mode"]

# ============================================================================================== constants
SMALL = frozenset(Decimal(i) for i in range(13)) | {Decimal(100)}
"""Integers 0–12 and 100: counts, months, a percent base. The default constants, for literals in calls and free
numbers alike."""
MEASURED_CONSTANTS = frozenset(Decimal(i) for i in range(101)) | frozenset(
    Decimal(x) for x in (365, 360, 52, 1000, 10 ** 6, 10 ** 9))
"""Integers 0–100, days in a year (365, 360), weeks (52), 1,000, a million, a billion: the set the measurements behind
the docs used, for calc literals and free constants alike. Opt-in (`constants=MEASURED_CONSTANTS`): an invented
number from 13 to 99 passes it."""


PI = Decimal("3.141592653589793238462643383")
"""π to 28 significant digits — an approximation (the error is below 3e-28), not an exact value. `pi` in a `<calc>`
expression (and as a value in the typed calls) stands for it. Not in `SMALL` or `MEASURED_CONSTANTS`: inside `fill` it
passes only when it is an allowed constant (`constants=SMALL | {PI}`) or with `literals="any"`."""


def years(lo: int = 1900, hi: int = 2100) -> frozenset:
    """The years lo..hi as constants: `constants=SMALL | years(2000, 2030)`."""
    return frozenset(Decimal(y) for y in range(lo, hi + 1))


# ============================================================================================== numerals
_MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(calendar.month_abbr) if m})
_MONTHS["sept"] = 9
_MON = "|".join(sorted(_MONTHS, key=len, reverse=True))
_SCALE_WORDS = {"thousand": 10 ** 3, "thousands": 10 ** 3, "million": 10 ** 6, "millions": 10 ** 6, "mln": 10 ** 6,
                "billion": 10 ** 9, "billions": 10 ** 9, "bln": 10 ** 9, "trillion": 10 ** 12, "trillions": 10 ** 12}
_SCALE_GLUED = {"k": 10 ** 3, "K": 10 ** 3, "mn": 10 ** 6, "bn": 10 ** 9, "tn": 10 ** 12}
_SW = "|".join(sorted(_SCALE_WORDS, key=len, reverse=True))
_SG = "|".join(sorted(_SCALE_GLUED, key=len, reverse=True))
_CORE = r"\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+"
_CUR = r"[$€£¥₽]"

_DATE_RE = re.compile(
    r"(?<![\w.\-/])(?:"
    r"(?P<iso>(?P<iy>\d{4})-(?P<im>\d{2})-(?P<id>\d{2}))"
    r"|(?P<dmy>(?P<d1>\d{1,2})(?P<sep>[./])(?P<m1>\d{1,2})(?P=sep)(?P<y1>\d{4}))"
    rf"|(?P<mdy>(?P<mn2>{_MON})\.?\s+(?P<d2>\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(?P<y2>\d{{4}}))?)"
    rf"|(?P<dmy2>(?P<d3>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<mn3>{_MON})\.?(?:,?\s+(?P<y3>\d{{4}}))?)"
    r")(?![\w\-/]|\.\d)", re.I)

_NUM_RE = re.compile(
    r"(?P<par>\(\s*(?P<pcur>" + _CUR + r")?\s*(?P<pcore>" + _CORE + r")\s*(?P<ppct>%)?\s*\)(?P<ppct2>\s?%)?)"
    r"|(?P<plain>(?P<sign>(?<![\w.,\d])[-−])?(?P<cur>" + _CUR + r")?\s?(?<![\w.])(?P<core>" + _CORE + r")"
    r"(?:(?P<pct>\s?%|\s?percent\b|\s?per\s+cent\b)"
    r"|(?P<ord>st|nd|rd|th)\b"
    rf"|(?P<sg>{_SG})\b"
    rf"|\s(?P<sw>{_SW})\b)?)", re.I)
_TOKEN_RE = re.compile(r"(?<![\w\-/.])\w(?:[\w\-/.]*\w)?")


@dataclass(frozen=True)
class Numeral:
    """A number as a text writes it.

    `number` is the written number with its sign ("(12.3)" and "-12.3" → -12.3; "4.2 million" → 4.2); `scale` the
    multiplier of a scale word or suffix (1,000 for "thousand" / "k"); `value` their product. `unit` is "%", a currency
    sign, an ordinal suffix or "". A date has `kind="date"` and `date` (a `datetime.date`, or `(month, day)` when the
    text gives no year). An identifier ("A-10457", "COVID-19") has `kind="id"` and is not a number."""
    start: int
    end: int
    surface: str
    kind: Literal["number", "date", "id"]
    number: Decimal | None = None
    scale: int = 1
    unit: str = ""
    date: object = None

    @property
    def value(self) -> Decimal | None:
        return None if self.number is None else self.number * self.scale

    def plain(self) -> str:
        """The surface as prose would write it: a negative in parentheses as "-12.3"."""
        m = re.fullmatch(r"\((.*)\)(\s?%)?", self.surface, re.S)
        if m and self.kind == "number":
            return "-" + m.group(1).strip() + (m.group(2) or "")
        return self.surface


def _dec(s: str) -> Decimal:
    s = s.replace(",", "")
    return Decimal("0" + s if s.startswith(".") else s)


def _date_of(m) -> object:
    g = m.groupdict()
    try:
        if g["iso"]:
            return dt.date(int(g["iy"]), int(g["im"]), int(g["id"]))
        if g["dmy"]:
            return dt.date(int(g["y1"]), int(g["m1"]), int(g["d1"]))
        mon = _MONTHS[(g["mn2"] or g["mn3"]).lower().rstrip(".")]
        day, year = int(g["d2"] or g["d3"]), g["y2"] or g["y3"]
        if year:
            return dt.date(int(year), mon, day)
        dt.date(2000, mon, day)                                  # validates the day (2000 is a leap year)
        return (mon, day)
    except ValueError:
        return None


def numerals(text: str) -> list[Numeral]:
    """Every number, date and identifier with digits in `text`, in order. Dates: ISO "2024-03-15", "15.03.2024" /
    "15/03/2024" (day first), "March 15, 2024", "15 March 2024", "March 15" (no year). Numbers: a sign or a negative
    in parentheses, a currency sign, comma thousands, a decimal point, then "%" / "percent", an ordinal suffix
    ("3rd"), a scale suffix ("12k", "3bn") or a scale word ("4.2 million"). A token where digits follow letters
    ("A10", "v2.1", "COVID-19", "1Z999AA1") is an identifier; a number followed by a unit ("5km") is a number."""
    out: list[Numeral] = []
    taken: list[tuple[int, int]] = []

    def free(a, b):
        return all(b <= x or a >= y for x, y in taken)

    for m in _DATE_RE.finditer(text):
        d = _date_of(m)
        if d is None:
            continue
        out.append(Numeral(m.start(), m.end(), m.group(), "date", date=d))
        taken.append((m.start(), m.end()))
    for m in _TOKEN_RE.finditer(text):
        tok = m.group()
        letter = re.search(r"[^\W\d_]", tok)
        if not letter or not re.search(r"\d", tok) or not free(m.start(), m.end()):
            continue
        if tok[0].isdigit() and not re.search(r"\d", tok[letter.start():]):
            continue                                             # "5km", "3rd", "12k", "1.2bn": a number with a unit
        out.append(Numeral(m.start(), m.end(), tok, "id"))
        taken.append((m.start(), m.end()))
    for m in _NUM_RE.finditer(text):
        if not free(m.start(), m.end()):
            continue
        g = m.groupdict()
        if g["par"]:
            n = -_dec(g["pcore"])
            unit = "%" if g["ppct"] or g["ppct2"] else (g["pcur"] or "")
            out.append(Numeral(m.start(), m.end(), m.group(), "number", n, 1, unit))
        else:
            n = _dec(g["core"])
            if g["sign"]:
                n = -n
            scale = 1
            unit = g["cur"] or ""
            if g["pct"]:
                unit = "%"
            elif g["ord"]:
                unit = g["ord"].lower()
            elif g["sg"]:
                scale = _SCALE_GLUED.get(g["sg"], _SCALE_GLUED.get(g["sg"].lower(), 1))
            elif g["sw"]:
                scale = _SCALE_WORDS[g["sw"].lower()]
            start = m.start("sign") if g["sign"] else (m.start("cur") if g["cur"] else m.start("core"))
            out.append(Numeral(start, m.end(), text[start:m.end()], "number", n, scale, unit))
        taken.append((m.start(), m.end()))
    out.sort(key=lambda x: x.start)
    return out


def annotate(text: str) -> tuple[str, list[Numeral]]:
    """The text with `[k]` before each numeral (numbers and dates, not identifiers), for the model's prompt, and the
    numerals in that order: `[1]` is `numerals[0]`."""
    nums = [n for n in numerals(text) if n.kind != "id"]
    parts, pos = [], 0
    for k, n in enumerate(nums, 1):
        parts.append(text[pos:n.start] + f"[{k}]" + n.surface)
        pos = n.end
    parts.append(text[pos:])
    return "".join(parts), nums


# ============================================================================================== exact executors
class ProgramError(ValueError):
    """A call that cannot be computed, or a pointer that does not resolve: the message says why, for a retry."""


_CTOK = re.compile(r"\s*(\d+(?:\.\d+)?|[-+*/()%^]|(?i:pi)(?!\w))")
_PI_RE = re.compile(r"(?<![\w.])pi(?!\w)", re.I)
_PREC = 50


def _fmt(d: Decimal, digits: int | None = 10, places: int | None = None, group: bool = False) -> str:
    """A Decimal as plain text: an integer in full, a fraction to `places` decimals or `digits` significant digits,
    trailing zeros trimmed (unless `places` is given)."""
    if not d.is_finite():
        raise ProgramError("the result is not a finite number")
    with localcontext() as ctx:
        ctx.prec = _PREC + 20
        return _fmt_in(d, digits, places, group)


def _fmt_in(d, digits, places, group):
    if places is not None:
        s = f"{d.quantize(Decimal(1).scaleb(-places)) if places > 0 else d.quantize(Decimal(1)):f}"
    elif d == d.to_integral_value():
        s = f"{d.quantize(Decimal(1)):f}"
    else:
        sig = digits or 28
        p = max(0, sig - d.adjusted() - 1)
        s = f"{round(d, p):f}"
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    if s.lstrip("-").strip("0.") == "":
        s = s.lstrip("-")
    if group:
        neg, s = s.startswith("-"), s.lstrip("-")
        ip, _, fp = s.partition(".")
        s = ("-" if neg else "") + f"{int(ip):,}" + ("." + fp if fp else "")
    return s


def calc(expr: str) -> Decimal:
    """Exact arithmetic on number literals: + - * / ^ (or **), parentheses, unary minus, a postfix % (divides by 100);
    `Decimal` with 50 digits. An integer power up to 1,000 is exact; a fractional one needs a positive base. `pi` is
    `PI` (28 digits, an approximation). Raises `ProgramError` with a readable reason."""
    s = (expr.replace("**", "^").replace("×", "*").replace("÷", "/").replace("−", "-"))
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    toks, pos = [], 0
    while pos < len(s):
        m = _CTOK.match(s, pos)
        if not m:
            if not s[pos:].strip():
                break
            raise ProgramError(f"calc allows only numbers, pi, + - * / ^ ( ) and a postfix %; it stopped at "
                               f"{s[pos:pos + 12]!r}")
        toks.append(m.group(1))
        pos = m.end()
    if not toks:
        raise ProgramError("the calc expression is empty")
    if len(toks) > 400:
        raise ProgramError("the calc expression is too long (over 400 tokens)")
    i = 0

    def peek():
        return toks[i] if i < len(toks) else None

    def eat():
        nonlocal i
        i += 1
        return toks[i - 1]

    def expr_():
        v = term()
        while peek() in ("+", "-"):
            v = v + term() if eat() == "+" else v - term()
        return v

    def term():
        v = unary()
        while peek() in ("*", "/"):
            if eat() == "*":
                v = v * unary()
            else:
                d = unary()
                if d == 0:
                    raise ProgramError("division by zero")
                v = v / d
        return v

    def unary():
        if peek() in ("-", "+"):
            return -unary() if eat() == "-" else unary()
        return power()

    def power():
        base = postfix()
        if peek() == "^":
            eat()
            ex = unary()                                         # right-associative, a sign allowed: 2^-1
            if ex == ex.to_integral_value() and abs(ex) <= 1000:
                if base == 0 and ex < 0:
                    raise ProgramError("division by zero (0 to a negative power)")
                return base ** int(ex)
            if base > 0:
                return (ex * base.ln()).exp()
            raise ProgramError("a fractional power needs a positive base")
        return base

    def postfix():
        v = primary()
        while peek() == "%":
            eat()
            v = v / 100
        return v

    def primary():
        t = peek()
        if t is None:
            raise ProgramError("the calc expression ends too early")
        if t == "(":
            eat()
            v = expr_()
            if peek() != ")":
                raise ProgramError("a '(' is not closed")
            eat()
            return v
        if re.fullmatch(r"\d+(?:\.\d+)?", t):
            eat()
            return Decimal(t)
        if t.lower() == "pi":
            eat()
            return PI
        raise ProgramError(f"unexpected {t!r} in the calc expression")

    with localcontext() as ctx:
        ctx.prec = _PREC
        try:
            v = expr_()
        except (InvalidOperation, OverflowError, ArithmeticError) as e:
            if isinstance(e, ProgramError):
                raise
            raise ProgramError(f"the calc expression cannot be computed ({type(e).__name__})") from None
        if i != len(toks):
            raise ProgramError(f"unexpected {toks[i]!r} after the end of the calc expression")
        return +v


_DTOK = re.compile(r"\s*(\d{4}-\d{2}-\d{2}|\d+\s?[DWMY]\b|[-+]|WEEKDAY|DAYS)", re.I)
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _add_months(d: dt.date, n: int) -> dt.date:
    y, m = divmod(d.month - 1 + n, 12)
    y += d.year
    return dt.date(y, m + 1, min(d.day, calendar.monthrange(y, m + 1)[1]))


def date_calc(expr: str) -> dt.date | int | str:
    """Calendar arithmetic: `DATE ((+|-) N(D|W|M|Y))*` → a date (a month added to Jan 31 gives the month's last day);
    `DATE - DATE` → days (an int; `DAYS` in front is allowed); `WEEKDAY DATE ...` → the weekday's name. Dates are ISO
    (`YYYY-MM-DD`) or `DD.MM.YYYY`. Raises `ProgramError`."""
    s = re.sub(r"\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b", lambda m: f"{m.group(3)}-{int(m.group(2)):02d}-{int(m.group(1)):02d}",
               expr.replace("−", "-"))
    toks, pos = [], 0
    while pos < len(s):
        m = _DTOK.match(s, pos)
        if not m:
            if not s[pos:].strip():
                break
            raise ProgramError("date allows dates, + / - N D|W|M|Y, a date minus a date, and WEEKDAY in front; "
                               f"it stopped at {s[pos:pos + 12]!r}")
        toks.append(m.group(1).replace(" ", ""))
        pos = m.end()
    head = toks[0].upper() if toks else ""
    wd = head == "WEEKDAY"
    if head in ("WEEKDAY", "DAYS"):
        toks = toks[1:]
    if not toks or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", toks[0]):
        raise ProgramError("a date expression starts with a date")
    try:
        cur = dt.date.fromisoformat(toks[0])
        i = 1
        while i < len(toks):
            op = toks[i]
            if op not in "+-" or i + 1 >= len(toks):
                raise ProgramError(f"expected + or - and an amount after {toks[i - 1]!r}")
            a = toks[i + 1]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", a):
                if op != "-" or wd or i + 2 != len(toks):
                    raise ProgramError("a date minus a date (days between) must be the whole expression")
                return (cur - dt.date.fromisoformat(a)).days
            if not re.fullmatch(r"\d+[DWMY]", a, re.I):
                raise ProgramError(f"{a!r} is not an amount such as 45D, 2W, 3M or 1Y")
            n, u = int(a[:-1]), a[-1].upper()
            n = n if op == "+" else -n
            if u in "DW":
                cur = cur + dt.timedelta(days=n * (7 if u == "W" else 1))
            else:
                cur = _add_months(cur, n if u == "M" else 12 * n)
            i += 2
    except (ValueError, OverflowError) as e:
        raise ProgramError(f"not a valid date ({e})") from None
    return _WEEKDAYS[cur.weekday()] if wd else cur


# ============================================================================================== tables
_CAPTION_RE = re.compile(r"\bin\s+(thousands|millions|billions|trillions)\b|(?<![\w'])(?:'000s?|000s|000's)(?!\w)",
                         re.I)
_SEP_ROW = re.compile(r"\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?")
_BLANK = frozenset({"", "-", "–", "—", "n/a", "na", "n.a.", "nm", "n.m."})
_CELL_REST = re.compile(r"[\s$€£¥₽%*†‡]*(?:[A-Z]{3})?[\s*†‡]*")


@dataclass(frozen=True)
class Cell:
    """A table cell a result came from: table `table<table>`, its 1-based data row and column (the header row and the
    row-label column are not counted), their labels and the cell as written."""
    table: int
    row: int
    col: int
    row_label: str
    col_label: str
    surface: str

    def to_dict(self) -> dict:
        return {"table": self.table, "row": self.row, "col": self.col, "row_label": self.row_label,
                "col_label": self.col_label, "surface": self.surface}


@dataclass
class Table:
    """A table for `<table>` and `<count>`: `rows` row-major, the header row first (the column labels); with
    `row_labels=True` the first column holds the row labels. `scale` multiplies the money cells under `norm` (1,000
    for a caption "in thousands"; percent cells are not scaled). Cells are text ("$1,250", "(3.2)", "12.5%", "—"),
    read like the input's numerals: a parenthesised number is negative, a dash or "n/a" is blank. Short rows are
    padded with blanks."""
    rows: list
    row_labels: bool = True
    scale: int = 1
    caption: str = ""

    def __post_init__(self):
        rows = [[str(c).strip() for c in r] for r in self.rows]
        if len(rows) < 2:
            raise ValueError("a table needs a header row and at least one data row")
        width = max(len(r) for r in rows)
        if width < (2 if self.row_labels else 1):
            raise ValueError("a table with row labels needs at least two columns")
        self.rows = [r + [""] * (width - len(r)) for r in rows]

    @property
    def columns(self) -> list[str]:
        """The column labels (without the row-label column's header)."""
        return self.rows[0][1:] if self.row_labels else list(self.rows[0])

    @property
    def labels(self) -> list[str]:
        """The row labels ([] without a row-label column)."""
        return [r[0] for r in self.rows[1:]] if self.row_labels else []

    @property
    def shape(self) -> tuple[int, int]:
        """(data rows, data columns)."""
        return len(self.rows) - 1, len(self.columns)

    def text(self, row: int, col: int) -> str:
        """The cell at 1-based data row and column, as written (column 0: the row label)."""
        return self.rows[row][col - (0 if self.row_labels else 1)]

    def describe(self, name: str = "table1") -> str:
        """The table's pointers for a prompt: its rows and columns with their numbers and labels."""
        nr, nc = self.shape
        cols = ", ".join(f"{j} {c!r}" for j, c in enumerate(self.columns, 1))
        rows = ", ".join(f"{i} {r!r}" for i, r in enumerate(self.labels, 1)) or f"1..{nr} (no labels)"
        sc = f"; values in {_SCALE_NAME[self.scale]}" if self.scale != 1 else ""
        return f"{name}: {nr} rows ({rows}); {nc} columns ({cols}){sc}"


def _is_pipe_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") or s.count("|") >= 2


def _split_pipe(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def parse_tables(text: str, row_labels: bool = True) -> list[Table]:
    """The pipe (markdown) tables of a text, in order: `table1`, `table2`, ... A run of lines that start with "|" or
    hold two "|" or more is a table; a `|---|---|` line is skipped; the first row is the header. A caption "in thousands"
    / "in millions" / "in billions" / "in trillions" (or "'000", "000s") in the last two non-blank lines of the three
    above the table or in its header row sets `Table.scale`."""
    lines = text.splitlines()
    out, i = [], 0
    while i < len(lines):
        if not _is_pipe_row(lines[i]):
            i += 1
            continue
        j = i
        while j < len(lines) and _is_pipe_row(lines[j]):
            j += 1
        rows = [_split_pipe(ln) for ln in lines[i:j] if not _SEP_ROW.fullmatch(ln.strip())]
        if len(rows) >= 2:
            above = [ln.strip() for ln in lines[max(0, i - 3):i] if ln.strip()][-2:]
            caption = " ".join(above + rows[0])
            m = _CAPTION_RE.search(caption)
            scale = (_SCALE_WORDS[m.group(1).lower()] if m.group(1) else 1000) if m else 1
            out.append(Table(rows, row_labels, scale, " ".join(above)))
        i = j
    return out


def _as_tables(tables) -> list[Table]:
    if tables is None:
        return []
    if isinstance(tables, str):
        return parse_tables(tables)
    if isinstance(tables, Table):
        return [tables]
    out = []
    for t in tables:
        out.append(t if isinstance(t, Table) else Table([list(r) for r in t]))
    return out


# ============================================================================================== typed values
_SCALE_NAME = {1: "units", 10 ** 3: "thousands", 10 ** 6: "millions", 10 ** 9: "billions", 10 ** 12: "trillions"}
_UNITS = frozenset("%$€£¥₽")


@dataclass(frozen=True)
class _V:
    """An operand of the typed tools: a number (the written number, its unit, its written scale — None when unknown —
    and a table's caption scale `base`), a date, a string or a boolean; the cells it came from."""
    kind: str
    value: object
    unit: str = ""
    scale: int | None = None
    base: int = 1
    cells: tuple = ()
    label: str = ""

    def norm(self) -> Decimal:
        return self.value * (self.scale or 1) * (1 if self.unit == "%" else self.base)

    def desc(self) -> str:
        return self.label or (repr(self.value) if self.kind == "str" else str(self.value))


def _cell_value(t: Table, ti: int, r: int, c: int) -> _V | None:
    """A cell as an operand: None when blank."""
    s = t.text(r, c)
    cell = Cell(ti, r, c, t.labels[r - 1] if t.row_labels else "", t.columns[c - 1] if c else t.rows[0][0], s)
    lab = f"table {ti} {cell.row_label or f'row {r}'!r}/{cell.col_label or f'column {c}'!r} ({s!r})"
    if s.casefold() in _BLANK:
        return None
    ns = [n for n in numerals(s) if n.kind != "id"]
    if len(ns) == 1 and _CELL_REST.fullmatch(s[:ns[0].start] + s[ns[0].end:]):
        n = ns[0]
        if n.kind == "date" and isinstance(n.date, dt.date):
            return _V("date", n.date, cells=(cell,), label=lab)
        if n.kind == "number":
            return _V("number", n.number, n.unit if n.unit in _UNITS else "", n.scale, t.scale, (cell,), lab)
    return _V("str", s, cells=(cell,), label=lab)


def _at(t: Table, ti: int, axis: str, idx: int, other: int) -> _V | None:
    """The cell of column `idx` in row `other` (axis "column") or of row `idx` in column `other` (axis "row")."""
    return _cell_value(t, ti, idx, other) if axis == "row" else _cell_value(t, ti, other, idx)


def _need_number(v: _V, what: str) -> _V:
    if v.kind != "number":
        raise ProgramError(f"{what}: {v.desc()} is a {v.kind}, not a number")
    return v


def _same_units(vs, what: str):
    units = {v.unit for v in vs if v.unit}
    if len(units) > 1:
        raise ProgramError(f"{what}: mixed units ({', '.join(sorted(units))})")


def _same_scales(vs, what: str, norm: bool):
    scales = {v.scale for v in vs if v.scale is not None}
    if len(scales) > 1 and not norm:
        raise ProgramError(f"{what}: mixed scales ({', '.join(_SCALE_NAME.get(s, str(s)) for s in sorted(scales))}); "
                           "normalise them (norm in <table>, or <scale>)")


def _str_key(s: str) -> str:
    return " ".join(s.split()).casefold()


_CMP_OPS = {"<", "<=", ">", ">=", "==", "!=", "=", "≤", "≥", "≠"}
_OP_CANON = {"=": "==", "≤": "<=", "≥": ">=", "≠": "!="}


def _compare(a: _V, op: str, b: _V, norm: bool = False) -> bool:
    op = _OP_CANON.get(op, op)
    what = f"{a.desc()} {op} {b.desc()}"
    if a.kind != b.kind:
        raise ProgramError(f"cannot compare {what}: a {a.kind} with a {b.kind}")
    if a.kind in ("str", "bool"):
        if op not in ("==", "!="):
            raise ProgramError(f"cannot compare {what}: {a.kind}s compare only with == and !=")
        x, y = (_str_key(a.value), _str_key(b.value)) if a.kind == "str" else (a.value, b.value)
    elif a.kind == "number":
        _same_units((a, b), f"cannot compare {what}")
        _same_scales((a, b), f"cannot compare {what}", norm)
        x, y = (a.norm(), b.norm()) if norm else (a.value, b.value)
    else:
        x, y = a.value, b.value
    return {"<": x < y, "<=": x <= y, ">": x > y, ">=": x >= y, "==": x == y, "!=": x != y}[op]


# ============================================================================================== the typed tools' parser
_PTOK = re.compile(r"""\s*(?:
 (?P<str>"[^"]*"|'[^']*'|“[^”]*”)
|(?P<in>\[\d+\])
|(?P<slot>\{\s*@\d+\s*\}|@\d+)
|(?P<date>\d{4}-\d{2}-\d{2})
|(?P<rule>\d+/\d+(?![\d.]))
|(?P<num>-?\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?%?|-?\d+(?:\.\d+)?%?|-?\.\d+%?)
|(?P<op>\.\.|<=|>=|==|!=|->|→|[=<>≤≥≠(),:%])
|(?P<word>(?:km/h|km/s|m/s)(?![\w/])|[^\W\d]\w*)
)""", re.X)


@dataclass
class _Env:
    """What a typed call can point at: the input's numerals, the earlier calls, the tables, the literal policy."""
    nums: list | None = None
    calls: list | None = None
    tables: list = field(default_factory=list)
    k: int = 0
    lit_ok: Callable = lambda d: True
    date_ok: Callable = lambda d: True


def _ptokens(expr: str, mode: str) -> list[tuple[str, str]]:
    s = expr.replace("−", "-")
    toks, pos = [], 0
    while pos < len(s):
        m = _PTOK.match(s, pos)
        if not m or m.end() == pos:
            if not s[pos:].strip():
                break
            raise ProgramError(f"{mode} cannot read {s[pos:pos + 12]!r}")
        kind = m.lastgroup
        toks.append((kind, m.group(kind)))
        pos = m.end()
    if not toks:
        raise ProgramError(f"the {mode} expression is empty")
    if len(toks) > 400:
        raise ProgramError(f"the {mode} expression is too long (over 400 tokens)")
    return toks


class _Parser:
    def __init__(self, expr: str, env: _Env, mode: str):
        self.toks, self.i, self.env, self.mode = _ptokens(expr, mode), 0, env, mode

    # ---- tokens
    def peek(self, ahead: int = 0):
        j = self.i + ahead
        return self.toks[j] if j < len(self.toks) else (None, None)

    def eat(self):
        if self.i >= len(self.toks):
            raise ProgramError(f"the {self.mode} expression ends too early")
        self.i += 1
        return self.toks[self.i - 1]

    def is_word(self, *words, ahead: int = 0) -> bool:
        k, t = self.peek(ahead)
        return k == "word" and t.lower() in words

    def is_op(self, *ops, ahead: int = 0) -> bool:
        k, t = self.peek(ahead)
        return k == "op" and t in ops

    def accept_word(self, *words) -> bool:
        if self.is_word(*words):
            self.i += 1
            return True
        return False

    def expect_op(self, op: str):
        k, t = self.peek()
        if k != "op" or t != op:
            raise ProgramError(f"{self.mode}: expected {op!r}, found {t!r}" if t else
                               f"{self.mode}: expected {op!r} at the end")
        self.i += 1

    def done(self):
        if self.i != len(self.toks):
            raise ProgramError(f"{self.mode}: unexpected {self.toks[self.i][1]!r} after the end of the expression")

    # ---- operands
    def operand(self) -> _V:
        k, t = self.eat()
        if k == "str":
            return _V("str", t[1:-1], label=t)
        if k == "num":
            unit = "%" if t.endswith("%") else ""
            d = _dec(t.rstrip("%").replace("-", "", 1))
            d = -d if t.startswith("-") else d
            if not self.env.lit_ok(abs(d)):
                raise ProgramError(f"the literal {t} is not allowed; point at the input ([k]) or an earlier call (@k)")
            return _V("number", d, unit, None, label=t)
        if k == "date":
            try:
                d = dt.date.fromisoformat(t)
            except ValueError:
                raise ProgramError(f"{t!r} is not a valid date") from None
            if not self.env.date_ok(d):
                raise ProgramError(f"the literal date {t} is not allowed; point at the input ([k]) or an earlier call")
            return _V("date", d, label=t)
        if k == "in":
            return self.input(int(t[1:-1]))
        if k == "slot":
            return self.slot(int(re.search(r"\d+", t).group()))
        if k == "word" and t.lower() in ("true", "false"):
            return _V("bool", t.lower() == "true", label=t.lower())
        if k == "word" and t.lower() == "pi":
            if not self.env.lit_ok(PI):
                raise ProgramError("pi is not an allowed constant here; pass constants=... | {PI}")
            return _V("number", PI, label="pi")
        raise ProgramError(f"{self.mode}: expected a value ([k], @k, a number, a date or a \"string\"), found {t!r}")

    def input(self, k: int) -> _V:
        nums = self.env.nums
        if nums is None:
            raise ProgramError(f"[{k}] points at the input: use the call inside fill")
        if not 1 <= k <= len(nums):
            raise ProgramError(f"input numeral [{k}] does not exist (the input has [1]..[{len(nums)}])")
        n = nums[k - 1]
        if n.kind == "date":
            if not isinstance(n.date, dt.date):
                raise ProgramError(f"[{k}] is {n.surface!r}, a date without a year")
            return _V("date", n.date, label=f"[{k}]")
        return _V("number", n.number, n.unit if n.unit in _UNITS else "", n.scale, label=f"[{k}] ({n.surface})")

    def slot(self, j: int) -> _V:
        calls = self.env.calls
        if calls is None:
            raise ProgramError(f"@{j} points at an earlier call: use the call inside fill")
        if not 1 <= j < self.env.k:
            raise ProgramError(f"@{j} is not an earlier call (calls are numbered in order, all kinds together; this "
                               f"one is @{self.env.k})")
        c = calls[j - 1]
        if c.error:
            raise ProgramError(f"@{j} failed ({c.error})")
        v, lab = c.value, f"@{j}"
        if isinstance(v, bool):
            return _V("bool", v, cells=c.cells, label=lab)
        if isinstance(v, (int, Decimal)):
            return _V("number", Decimal(v), cells=c.cells, label=f"{lab} ({c.shown})")
        if isinstance(v, dt.date):
            return _V("date", v, cells=c.cells, label=lab)
        return _V("str", str(v), cells=c.cells, label=f"{lab} ({v!r})")

    # ---- tables
    def table(self) -> tuple[int, Table]:
        ti = 1
        k, t = self.peek()
        if k == "word" and re.fullmatch(r"table\d+", t, re.I) and self.is_op(":", ahead=1):
            ti = int(t[5:])
            self.i += 2
        tables = self.env.tables
        if not tables:
            raise ProgramError(f"{self.mode} needs a table: the source has none (write it as a pipe table in the "
                               "source or pass tables=)")
        if not 1 <= ti <= len(tables):
            raise ProgramError(f"table {ti} does not exist (the tables are table1..table{len(tables)})")
        return ti, tables[ti - 1]

    def pointer(self, t: Table, ti: int, axis: str) -> int:
        """A row or column: a 1-based number, a "label", [k] (the label as the input writes it) or @k."""
        labels = t.labels if axis == "row" else t.columns
        n = len(labels) if axis == "column" else t.shape[0]
        k, s = self.eat()
        if k == "num" and re.fullmatch(r"\d+", s):
            idx = int(s)
            if not 1 <= idx <= n:
                raise ProgramError(f"table {ti} has no {axis} {idx} (it has {axis}s 1..{n})")
            return idx
        if k == "slot":
            v = self.slot(int(re.search(r"\d+", s).group()))
            if v.kind == "number" and v.value == v.value.to_integral_value():
                if not 1 <= v.value <= n:
                    raise ProgramError(f"table {ti} has no {axis} {v.value} (from {v.label}; it has {axis}s 1..{n})")
                return int(v.value)
            if v.kind != "str":
                raise ProgramError(f"{v.desc()} is not a {axis} number or a label")
            label = v.value
        elif k == "in":
            nums = self.env.nums
            idx = int(s[1:-1])
            if nums is None or not 1 <= idx <= len(nums):
                self.input(idx)                                    # raises the right error
            label = nums[idx - 1].surface
        elif k == "str":
            label = s[1:-1]
        else:
            raise ProgramError(f"expected a {axis} (a number, a \"label\", [k] or @k), found {s!r}")
        if axis == "row" and not t.row_labels:
            raise ProgramError(f"table {ti} has no row labels; give the row's number")
        hits = [i for i, lab in enumerate(labels, 1) if _str_key(lab) == _str_key(label)]
        corner = t.rows[0][0] if t.row_labels else ""
        if not hits and axis == "column" and corner and _str_key(corner) == _str_key(label):
            return 0                                               # the row-label column, by its header
        if not hits:
            shown = ", ".join(repr(x) for x in labels[:8]) + (", ..." if len(labels) > 8 else "")
            raise ProgramError(f"table {ti} has no {axis} labelled {label!r} (its {axis}s: {shown})")
        if len(hits) > 1:
            raise ProgramError(f"table {ti}: {axis}s {', '.join(map(str, hits))} are all labelled {label!r}; give the "
                               "number")
        return hits[0]

    def span(self, t: Table, ti: int, axis: str = "row") -> list[int]:
        """`a..b`: rows (or columns) a to b."""
        a = self.pointer(t, ti, axis)
        self.expect_op("..")
        b = self.pointer(t, ti, axis)
        if b < a:
            raise ProgramError(f"table {ti}: the {axis} range {a}..{b} is empty")
        return list(range(a, b + 1))

    def value_list(self) -> list[_V]:
        """`(a, b, ...)` or `column(...)`."""
        if self.is_op("("):
            self.eat()
            if self.is_op(")"):
                raise ProgramError(f"{self.mode}: an empty list")
            items = [self.operand()]
            while self.is_op(","):
                self.eat()
                items.append(self.operand())
            self.expect_op(")")
            return items
        k, t = self.peek()
        if self.is_word("column", "row") or (k == "word" and re.fullmatch(r"table\d+", t, re.I) and
                                             self.is_op(":", ahead=1)):
            return self.line()
        raise ProgramError(f"{self.mode}: expected a list (a, b, ...), column(...) or row(...), found {t!r}")

    def line(self) -> list[_V]:
        """`[tableN:] column(col[, a..b][, where cond])` or `row(r[, a..b][, where cond])`: the non-blank cells of a
        column or a row."""
        ti, t = self.table()
        w = self.peek()[1]
        if not self.accept_word("column", "row"):
            raise ProgramError(f"{self.mode}: expected column(...) or row(...) after table{ti}:")
        axis, idx, others, cond, _ = self.selection(t, ti, allow_norm=False, axis=w.lower())
        return [v for o in others if (v := _at(t, ti, axis, idx, o)) is not None and cond(o)]

    def selection(self, t: Table, ti: int, allow_norm: bool = True, axis: str | None = None):
        """`([column] col | row r[, a..b][, where cond][, norm])` → the axis ("column" or "row"), its index, the
        cross indices (the rows of a column, the columns of a row), a predicate on them, norm."""
        self.expect_op("(")
        if axis is None:
            axis = "row" if self.accept_word("row") else "column"
            if axis == "column":
                self.accept_word("column")
        idx = self.pointer(t, ti, axis)
        cross = "column" if axis == "row" else "row"
        others = list(range(1, t.shape[1 if axis == "row" else 0] + 1))
        cond, norm = (lambda o: True), False
        while self.is_op(","):
            self.eat()
            if self.accept_word("where"):
                cond = self.cond(t, ti, cross)
            elif allow_norm and self.accept_word("norm"):
                norm = True
            else:
                others = self.span(t, ti, cross)
        self.expect_op(")")
        return axis, idx, others, cond, norm

    # ---- conditions
    def pred(self) -> Callable[[_V], bool]:
        """`OP value`, `between a and b`, `[not] in list` on an operand still to come."""
        neg = False
        k, t = self.peek()
        if k == "op" and t in _CMP_OPS:
            self.eat()
            b = self.operand()
            return lambda a, t=t, b=b: _compare(a, t, b)
        if self.accept_word("between"):
            lo = self.operand()
            if not self.accept_word("and"):
                raise ProgramError(f"{self.mode}: between needs 'and': x between a and b")
            hi = self.operand()
            return lambda a: _compare(a, ">=", lo) and _compare(a, "<=", hi)
        if self.is_word("not") and self.is_word("in", ahead=1):
            self.i += 1
            neg = True
        if self.accept_word("in"):
            items = self.value_list()
            return lambda a: any(_compare(a, "==", x) for x in items) != neg
        raise ProgramError(f"{self.mode}: expected a comparison (< <= > >= == != between in), found {t!r}")

    def cond(self, t: Table, ti: int, over: str = "row") -> Callable[[int], bool]:
        """A row filter: column comparisons joined by and / or / not, in parentheses or not (`over="column"`: a
        column filter, row comparisons)."""
        by = "column" if over == "row" else "row"
        def or_():
            fs = [and_()]
            while self.accept_word("or"):
                fs.append(and_())
            return fs[0] if len(fs) == 1 else (lambda r: any(f(r) for f in fs))

        def and_():
            fs = [not_()]
            while self.accept_word("and"):
                fs.append(not_())
            return fs[0] if len(fs) == 1 else (lambda r: all(f(r) for f in fs))

        def not_():
            if self.accept_word("not"):
                f = not_()
                return lambda r: not f(r)
            if self.is_op("("):
                self.eat()
                f = or_()
                self.expect_op(")")
                return f
            line = self.pointer(t, ti, by)
            p = self.pred()

            def f(o, line=line, p=p):
                v = _at(t, ti, by, line, o)
                return v is not None and p(v)                     # a blank cell matches no condition
            return f
        return or_()

    def bool_expr(self) -> _V:
        """and / or / not over comparisons of operands (and boolean @k)."""
        cells: list = []

        def or_():
            v = and_()
            while self.accept_word("or"):
                w = and_()
                v = v or w
            return v

        def and_():
            v = not_()
            while self.accept_word("and"):
                w = not_()
                v = v and w
            return v

        def not_():
            if self.accept_word("not"):
                return not not_()
            if self.is_op("("):
                self.eat()
                v = or_()
                self.expect_op(")")
                return v
            a = self.operand()
            cells.extend(a.cells)
            k, t = self.peek()
            if (k == "op" and t in _CMP_OPS) or self.is_word("between", "in") or (
                    self.is_word("not") and self.is_word("in", ahead=1)):
                return self.pred()(a)
            if a.kind != "bool":
                raise ProgramError(f"cmp: {a.desc()} is a {a.kind}, not a yes/no; compare it")
            return a.value
        v = or_()
        return _V("bool", bool(v), cells=tuple(dict.fromkeys(cells)))


# ============================================================================================== the typed tools
_AGG = ("sum", "avg", "count", "max", "min", "argmax", "argmin")


def _table_run(p: _Parser) -> tuple[object, tuple]:
    ti, t = p.table()
    k, op = p.eat()
    op = op.lower() if k == "word" else op
    if op == "cell":
        p.expect_op("(")
        r = p.pointer(t, ti, "row")
        p.expect_op(",")
        c = p.pointer(t, ti, "column")
        norm = False
        if p.is_op(","):
            p.eat()
            if not p.accept_word("norm"):
                raise ProgramError("table: cell(row, column[, norm])")
            norm = True
        p.expect_op(")")
        p.done()
        v = _cell_value(t, ti, r, c)
        if v is None:
            raise ProgramError(f"table {ti} row {r} column {c} is blank ({t.text(r, c)!r})")
        if v.kind == "number":
            return (v.norm() if norm else v.value), v.cells
        if norm:
            raise ProgramError(f"{v.desc()} is not a number; norm applies to numbers")
        return v.value, v.cells
    if op not in _AGG:
        raise ProgramError(f"table allows cell, {', '.join(_AGG)}; found {op!r}")
    axis, idx, others, cond, norm = p.selection(t, ti)
    p.done()
    picked = [(o, _at(t, ti, axis, idx, o)) for o in others if cond(o)]
    if axis == "column":
        name = (t.columns[idx - 1] if idx else t.rows[0][0]) or idx
    else:
        name = (t.labels[idx - 1] if t.row_labels else "") or idx

    def label(o: int):                                             # argmax / argmin: the label across the line
        if axis == "column":
            return t.labels[o - 1] if t.row_labels else o
        return t.columns[o - 1] or o

    what = f"{op} over table {ti} {axis} {name!r}"
    if op == "count":
        vs = [v for _, v in picked if v is not None]
        return len(vs), tuple(c for v in vs for c in v.cells)
    if not picked:
        raise ProgramError(f"{what}: division by zero (no rows selected)" if op == "avg" else
                           f"{what}: no rows selected")
    vs = []
    for o, v in picked:
        if v is None:
            r, c = (o, idx) if axis == "column" else (idx, o)
            raise ProgramError(f"{what}: table {ti} row {r} column {c} is blank ({t.text(r, c)!r})")
        vs.append(_need_number(v, what))
    _same_units(vs, what)
    _same_scales(vs, what, norm)
    xs = [v.norm() if norm else v.value for v in vs]
    with localcontext() as ctx:
        ctx.prec = _PREC
        if op in ("sum", "avg"):
            total = sum(xs, Decimal(0))
            return (+total if op == "sum" else total / len(xs)), tuple(c for v in vs for c in v.cells)
    best = max(xs) if op in ("max", "argmax") else min(xs)
    tied = [(r, v) for (r, _), v, x in zip(picked, vs, xs) if x == best]
    cells = tuple(c for _, v in tied for c in v.cells)
    if op in ("max", "min"):
        return best, cells
    if len(tied) > 1:
        names = ", ".join(repr(x) if isinstance(x := label(o), str) else str(x) for o, _ in tied)
        raise ProgramError(f"{what}: a tie, {'rows' if axis == 'column' else 'columns'} {names} share the "
                           f"{op[3:]} {_fmt(best)}")
    return label(tied[0][0]), cells


def _cmp_run(p: _Parser) -> tuple[object, tuple]:
    v = p.bool_expr()
    p.done()
    return v.value, v.cells


def _count_run(p: _Parser) -> tuple[object, tuple]:
    k, t = p.peek()
    is_list = p.is_op("(") or p.is_word("column", "row") or (
        k == "word" and re.fullmatch(r"table\d+", t, re.I) and p.is_op(":", ahead=1))
    if is_list:
        items = p.value_list()
        if p.accept_word("if", "where"):
            f = p.pred()
            items = [x for x in items if f(x)]
        p.done()
        return len(items), tuple(c for x in items for c in x.cells)
    a = p.operand()
    neg = p.is_word("not") and p.is_word("in", ahead=1)
    if neg:
        p.eat()
    if not p.accept_word("in"):
        raise ProgramError("count: a list (a, b, ...) or column(...) with an optional 'if <condition>', or "
                           "'x in <list>'")
    items = p.value_list()
    p.done()
    hit = [x for x in items if _compare(a, "==", x)]
    return (not hit) if neg else bool(hit), tuple(dict.fromkeys(a.cells + tuple(c for x in hit for c in x.cells)))


_SCALE_UNITS = {**{w: ("scale", Decimal(v)) for w, v in _SCALE_WORDS.items()},
                "units": ("scale", Decimal(1)), "unit": ("scale", Decimal(1)), "ones": ("scale", Decimal(1)),
                "one": ("scale", Decimal(1)), "k": ("scale", Decimal(10 ** 3)), "mn": ("scale", Decimal(10 ** 6)),
                "bn": ("scale", Decimal(10 ** 9)), "tn": ("scale", Decimal(10 ** 12)),
                "percent": ("ratio", Decimal("0.01")), "pct": ("ratio", Decimal("0.01")),
                "%": ("ratio", Decimal("0.01")), "fraction": ("ratio", Decimal(1)), "ratio": ("ratio", Decimal(1)),
                "bp": ("ratio", Decimal("0.0001")), "bps": ("ratio", Decimal("0.0001")),
                **{w: ("time", Decimal(1)) for w in ("second", "seconds", "sec", "secs", "s")},
                **{w: ("time", Decimal(60)) for w in ("minute", "minutes", "min", "mins")},
                **{w: ("time", Decimal(3600)) for w in ("hour", "hours", "hr", "hrs", "h")},
                "day": ("time", Decimal(86400)), "days": ("time", Decimal(86400)),
                "week": ("time", Decimal(604800)), "weeks": ("time", Decimal(604800)),
                # speed in metres an hour, so that km/h ↔ m/s is one exact multiplication or division by 3.6
                "m/s": ("speed", Decimal(3600)), "mps": ("speed", Decimal(3600)),
                "km/h": ("speed", Decimal(1000)), "kmh": ("speed", Decimal(1000)), "kph": ("speed", Decimal(1000)),
                "km/s": ("speed", Decimal(3600000)), "mph": ("speed", Decimal("1609.344")),
                "month": ("months", Decimal(1)), "months": ("months", Decimal(1)),
                "quarter": ("months", Decimal(3)), "quarters": ("months", Decimal(3)),
                "year": ("months", Decimal(12)), "years": ("months", Decimal(12))}
_RULES = {"360": Decimal(360), "30/360": Decimal(360), "365": Decimal(365), "366": Decimal(366)}   # days a year
_DAY = Decimal(86400)                                                # seconds


def _scale_unit(p: _Parser) -> tuple[str, str, Decimal] | None:
    k, t = p.peek()
    if (k == "word" or (k == "op" and t == "%")) and t.lower() in _SCALE_UNITS:
        p.eat()
        return (t.lower(), *_SCALE_UNITS[t.lower()])
    return None


def _scale_run(p: _Parser) -> tuple[object, tuple]:
    k, t = p.peek()
    if k == "num" and t.endswith("%"):                           # "12.5% to fraction"
        p.toks[p.i] = ("num", t[:-1])
        p.toks.insert(p.i + 1, ("op", "%"))
    v = _need_number(p.operand(), "scale")
    src = _scale_unit(p)
    if src is None:
        if v.unit == "%":
            src = ("%", "ratio", Decimal("0.01"))
        elif v.scale not in (None, 1):
            name = _SCALE_NAME[v.scale].rstrip("s")
            src = (name, "scale", Decimal(v.scale))
        else:
            raise ProgramError(f"scale: give the unit {v.desc()} is in (thousand, percent, days, ...)")
    elif v.unit == "%" and src[1] != "ratio":
        raise ProgramError(f"scale: {v.desc()} is a percent, not {src[0]}")
    elif v.scale not in (None, 1) and src[1] == "scale" and src[2] != v.scale:
        raise ProgramError(f"scale: {v.desc()} is written in {_SCALE_NAME[v.scale]}, not {src[0]}")
    if not (p.accept_word("to", "in") or (p.is_op("->", "→") and p.eat())):
        raise ProgramError("scale: write VALUE [UNIT] to UNIT, e.g. [1] million to thousand")
    dst = _scale_unit(p)
    if dst is None:
        raise ProgramError(f"scale: unknown target unit {p.peek()[1]!r} (thousand, million, percent, fraction, bp, "
                           "seconds, minutes, hours, days, weeks, months, quarters, years, m/s, km/h, ...)")
    rule = None
    if p.accept_word("rule"):
        k, r = p.eat()
        if r not in _RULES:
            raise ProgramError(f"scale: rule {r!r} is not one of 360 (30/360), 365, 366")
        rule = _RULES[r]
    p.done()
    (sn, sg, sf), (dn, dg, df) = src, dst
    with localcontext() as ctx:
        ctx.prec = _PREC
        num, den = v.value * sf, df                                # one division at the end: 3.6, 1/3600 stay exact
        if sg != dg:
            pair = {sg, dg}
            if pair != {"time", "months"}:
                raise ProgramError(f"scale: {sn} and {dn} are different kinds of unit")
            if rule is None:
                raise ProgramError(f"scale: {sn} to {dn} has no exact ratio; give a rule (rule 360: a month is 30 "
                                   "days; rule 365: a year is 365 days) or use <date> on real dates")
            if sg == "months":
                num, den = num * rule * _DAY, den * 12
            else:
                num, den = num * 12, den * rule * _DAY
        elif rule is not None:
            raise ProgramError(f"scale: a rule applies only between time units and months/years, not {sn} to {dn}")
        return +(num / den), v.cells


_TOOLS = {"table": _table_run, "cmp": _cmp_run, "count": _count_run, "scale": _scale_run}


def _typed(mode: str, expr: str, env: _Env) -> tuple[object, tuple]:
    try:
        return _TOOLS[mode](_Parser(expr, env, mode))
    except (InvalidOperation, ArithmeticError) as e:
        if isinstance(e, ProgramError):
            raise
        raise ProgramError(f"the {mode} expression cannot be computed ({type(e).__name__})") from None


def table_calc(expr: str, tables, *, cells: bool = False):
    """A `<table>` call on its own. `tables`: a text with pipe tables, a `Table`, or a list of tables (each a `Table`
    or rows, the header first). `[tableN:] cell(row, column[, norm])`;
    `[tableN:] sum|avg|max|min|count|argmax|argmin([column] col[, a..b][, where <condition>][, norm])` over a column
    (a..b: rows) and `...(row r[, a..b][, where <condition>][, norm])` over a row (a..b: columns). A row or column is a
    1-based number or a "label"; a condition compares a column (for a row aggregate: another row, in each column)
    with a value (< <= > >= == != between in), joined by and / or / not. Values are the written numbers; `norm`
    multiplies by each cell's scale word and the table's caption scale. Returns the value (a Decimal, an int for
    count, a label for argmax / argmin — the row's over a column, the column's over a row — a date or a string for a
    cell), or `(value, cells)` with `cells=True`. Raises `ProgramError`."""
    v, cs = _typed("table", expr, _Env(tables=_as_tables(tables)))
    return (v, cs) if cells else v


def cmp_calc(expr: str) -> bool:
    """A `<cmp>` call on its own: numbers, ISO dates or "strings" compared with < <= > >= == != (strings only == and
    !=, case-insensitive), `x between a and b` (inclusive), `x [not] in (a, b, ...)`, joined by and / or / not.
    Raises `ProgramError` for a number against a date, two different units ("%" against "$") or scale words."""
    return bool(_typed("cmp", expr, _Env())[0])


def scale_calc(expr: str) -> Decimal:
    """A `<scale>` call on its own: `VALUE [UNIT] to UNIT [rule 360|365|366]`. Units: units, thousand, million,
    billion, trillion (k, mn, bn, tn); percent (%), fraction, bp; seconds (s, sec), minutes (min), hours (h, hr),
    days, weeks; months, quarters, years; m/s, km/h (kmh, kph), km/s, mph. Seconds to weeks against months, quarters
    and years need a rule (360: a month is 30 days; 365 or 366: a year). Exact `Decimal`, one division at the end
    (1 m/s is 3.6 km/h exactly; 1 km/h to m/s is 1000/3600 to 50 digits); raises `ProgramError`."""
    return typing.cast("Decimal", _typed("scale", expr, _Env())[0])


def count_calc(expr: str, tables=None):
    """A `<count>` call on its own: `(a, b, ...) [if <condition>]` or `[tableN:] column(col[, a..b][, where ...]) [if
    <condition>]` (or `row(r[, a..b][, where ...])`) → how many items (non-blank cells) match (an int); `x [not] in
    <list>` → a bool. Raises `ProgramError`."""
    return _typed("count", expr, _Env(tables=_as_tables(tables)))[0]


# ============================================================================================== the mode registry
@dataclass(frozen=True)
class CallContext:
    """What an extra mode's executor sees inside `fill`: the input's numerals (`[k]` is `inputs[k - 1]`), the earlier
    calls (`@j` is `calls[j - 1]`, a `CallResult`; check its `error`), the tables, this call's number `k`, and
    `literal_ok(Decimal)`, the literal policy of this `fill`."""
    inputs: list
    calls: list
    tables: list
    k: int
    literal_ok: Callable[[Decimal], bool]


@dataclass(frozen=True)
class Mode:
    """A call kind `<name>…</name>`: its slot grammar in short, the type of its result, `validator(value) -> bool`
    on a result, the executor and a one-line description. A built-in mode's `executor` is its standalone function
    (`calc`, `date_calc`, `table_calc`, `cmp_calc`, `scale_calc`, `count_calc`); an extra mode's is
    `executor(expr, ctx: CallContext) -> value`, raising `ProgramError` for a bad call, or None (the mode is
    described but has no executor: its calls fail)."""
    name: str
    slots: str
    result: str
    validator: Callable[[object], bool]
    executor: Callable | None
    description: str = ""
    builtin: bool = False


def _not_bool(*types):
    return lambda v: isinstance(v, types) and not isinstance(v, bool)


_MODES: dict[str, Mode] = {m.name: m for m in (
    Mode("calc", "an arithmetic expression over [k], @k, allowed constants and pi: + - * / ^ ( ) and a postfix %",
         "Decimal", _not_bool(Decimal), calc, "exact arithmetic (Decimal, 50 digits)", True),
    Mode("date", "DATE ± nD|nW|nM|nY, DATE - DATE, WEEKDAY DATE, DAYS DATE DATE over [k], @k and ISO dates",
         "date, int (days) or weekday name", lambda v: (isinstance(v, (dt.date, str)) or _not_bool(int)(v)),
         date_calc, "calendar arithmetic: offsets, days between, the weekday", True),
    Mode("table", "[tableN:] cell(row, col[, norm]) | sum|avg|count|max|min|argmax|argmin(col | row r[, a..b]"
         "[, where <condition>][, norm])", "Decimal, int (count), label (argmax / argmin), date or string (cell)",
         _not_bool(Decimal, int, str, dt.date), table_calc,
         "a table cell, or an aggregate over a column or a row with a range and a filter", True),
    Mode("cmp", "comparisons (< <= > >= == != between in) of values, joined by and / or / not", "bool",
         lambda v: isinstance(v, bool), cmp_calc, "a yes/no from comparisons of numbers, dates and strings", True),
    Mode("scale", "VALUE [UNIT] to UNIT [rule 360|365|366]", "Decimal", _not_bool(Decimal), scale_calc,
         "scale words, percent and bp, time units, speed (km/h, m/s)", True),
    Mode("count", "(a, b, ...) | column(...) | row(...) [if <condition>], or x [not] in <list>", "int or bool",
         lambda v: isinstance(v, (int, bool)), count_calc, "how many items match, or membership", True),
)}
MODES = MappingProxyType(_MODES)
"""The call kinds `fill` runs, by name (read-only; `register_mode` adds one): the single list of typed calls."""
_MODE_NAME = re.compile(r"[a-z][a-z0-9_]{0,31}")


def register_mode(name: str, executor: Callable | None = None, *, slots: str = "", result: str = "",
                  validator: Callable[[object], bool] | None = None, description: str = "",
                  replace: bool = False) -> Mode:
    """Add a call kind `<name>…</name>` that `fill` runs with `executor(expr, ctx)` (see `Mode`, `CallContext`); its
    calls share the `@k` numbering, a result failing `validator` (default: any) is an error, and `check` reports its
    numbers with the name as `how`. `executor=None` registers a described mode whose calls fail. A built-in mode
    cannot be replaced; an extra one only with `replace=True`."""
    if not _MODE_NAME.fullmatch(name):
        raise ValueError(f"a mode name is lowercase letters, digits and _, starting with a letter: {name!r}")
    old = _MODES.get(name)
    if old is not None and (old.builtin or not replace):
        raise ValueError(f"the mode {name!r} exists" + ("" if old.builtin else "; pass replace=True"))
    m = Mode(name, slots, result, validator or (lambda v: True), executor, description)
    _MODES[name] = m
    return m


def unregister_mode(name: str) -> None:
    """Remove an extra mode (a built-in one stays)."""
    m = _MODES.get(name)
    if m is None:
        raise KeyError(name)
    if m.builtin:
        raise ValueError(f"the built-in mode {name!r} cannot be removed")
    del _MODES[name]


@lru_cache(maxsize=16)
def _tag_re_for(names: tuple) -> re.Pattern:
    return re.compile(rf"<({'|'.join(names)})>(.*?)</\1>", re.S | re.I)


def _tag_re() -> re.Pattern:
    return _tag_re_for(tuple(sorted(_MODES, key=lambda n: (-len(n), n))))


# ============================================================================================== fill
_IN_RE = re.compile(r"\[(\d+)\]")
_SLOT_RE = re.compile(r"\{\s*@(\d+)\s*\}|(?<![\w@.])@(\d+)(?=[DWMYdwmy]\b|[^\w]|$)")
_LIT_RE = re.compile(r"(?<![\w.\[@])(\d+(?:\.\d+)?)(?=\s?[DWMY]\b|[^\w]|$)", re.I)


@dataclass(frozen=True)
class CallResult:
    """The k-th call: its kind, the expression as written, the value (a Decimal, a date, an int of days or a count,
    a weekday or a table label, a bool) and the text it was shown as; `error` instead of a value when it failed;
    `cells`, the table cells a `<table>` / `<count>` value (or a `<cmp>` / `<scale>` on one) came from."""
    k: int
    kind: str
    expr: str
    value: object = None
    shown: str = ""
    error: str = ""
    cells: tuple = ()


@dataclass(frozen=True)
class Inserted:
    """A span of the filled text written by code: from the input numeral `[index]` (`source="input"`) or from call
    `@index` (`source="tool"`)."""
    start: int
    end: int
    source: str
    index: int


@dataclass
class Filled:
    """`fill`'s result: the final text, the calls, the spans code wrote, and every error (an empty list = clean)."""
    text: str
    calls: list[CallResult]
    inserted: list[Inserted]
    errors: list[str]
    source_numerals: list[Numeral]
    tables: list = field(default_factory=list)


def _literal_ok(lit: str, policy, constants, inputs) -> bool:
    if policy == "any":
        return True
    v = Decimal(lit)
    if _is_const(v, constants):
        return True
    return policy == "input" and any(n.number is not None and abs(n.number) == v for n in inputs)


def _is_const(v: Decimal, constants) -> bool:
    if callable(constants):
        return bool(constants(v))
    return abs(v) in constants


def fill(text: str, source: str | list[Numeral], *, answer_marker: str | None = None,
         literals: Literal["constants", "input", "any"] = "constants", constants=SMALL,
         digits: int = 10, places: int | None = None, grouping: bool | None = None,
         date_format: str | None = None, show_calls: bool = True, tables=None,
         booleans: tuple[str, str] = ("yes", "no")) -> Filled:
    """Run the calls of a model's text and write in the exact values.

    `source` is the input text (or its `numerals`). Each `<calc>` / `<date>` call is computed in order (`[k]` is the
    k-th input numeral's written number — "4.2 million" gives 4.2, "12.5%" gives 12.5 — or its date; `@k` an earlier
    call's value) and replaced by its value where it stands (`show_calls=False` removes it: refer to it as `@k`). Then
    `[k]` and `@k` / `{@k}` in the prose are replaced: an input numeral as the input writes it (a currency sign or "%"
    the text repeats around the pointer is not doubled), a result with `digits` significant digits or `places`
    decimals, comma-grouped when `grouping` (None: when the input groups its thousands), a date as ISO or
    `date_format` (strftime). With `answer_marker` ("ANSWER:") only the text after its last occurrence is returned;
    the calls before it still run.

    A literal number inside a call must be a constant (`literals="constants"`), or also an input numeral written
    verbatim (`"input"`), or anything (`"any"`). A pointer out of range, an unknown or failed `@k`, a malformed call or
    a missing marker is an error in `Filled.errors`; its pointer is left as written.

    `<table>`, `<cmp>`, `<scale>`, `<count>` and modes added with `register_mode` share the numbering (see
    `table_calc`, `cmp_calc`, `scale_calc`, `count_calc`, `MODES`); `[k]` and `@k` work inside them, a literal number in a value position follows
    `literals` (a literal date needs `literals="any"`, or `"input"` and an input date). `tables`: the tables they
    read (a `Table`, a list of them or of rows; None: the pipe tables of `source`, `table1` first). A bool is shown as
    `booleans` ("yes" / "no")."""
    nums = [n for n in numerals(source) if n.kind != "id"] if isinstance(source, str) else list(source)
    tbls = _as_tables(source if tables is None and isinstance(source, str) else tables)
    in_dates = {n.date for n in nums if isinstance(n.date, dt.date)}
    if grouping is None:
        grouping = isinstance(source, str) and re.search(r"\d,\d{3}(?!\d)", source) is not None
    errors: list[str] = []
    calls: list[CallResult] = []

    def show(v) -> str:
        if isinstance(v, bool):
            return booleans[0] if v else booleans[1]
        if isinstance(v, Decimal):
            return _fmt(v, digits, places, grouping)
        if isinstance(v, dt.date):
            return v.strftime(date_format) if date_format else v.isoformat()
        return str(v)

    def in_ref(k: int, where: str):
        if not 1 <= k <= len(nums):
            raise ProgramError(f"{where}: input numeral [{k}] does not exist (the input has [1]..[{len(nums)}])")
        return nums[k - 1]

    tag_re = _tag_re()
    for k, m in enumerate(tag_re.finditer(text), 1):
        kind, expr = m.group(1).lower(), m.group(2).strip()
        lab = re.match(r"@(\d+)\s*=(?!=)\s*", expr)              # "<calc>@3 = [1] - [2]</calc>" names its own call
        if lab and int(lab.group(1)) == k:
            expr = expr[lab.end():]
        where = f"call @{k} <{kind}>{m.group(2).strip()}</{kind}>"
        if kind in _TOOLS:
            env = _Env(nums, calls, tbls, k, lambda d: _literal_ok(str(d), literals, constants, nums),
                       lambda d: literals == "any" or (literals == "input" and d in in_dates))
            try:
                v, cs = _typed(kind, expr, env)
                calls.append(CallResult(k, kind, expr, v, show(v), cells=cs))
            except ProgramError as e:
                errors.append(f"{where}: {e}")
                calls.append(CallResult(k, kind, expr, error=f"{where}: {e}"))
            continue
        mode = _MODES.get(kind)
        if mode is not None and not mode.builtin:
            try:
                if mode.executor is None:
                    raise ProgramError(f"the mode {kind!r} has no executor")
                ctx = CallContext(nums, list(calls), tbls, k,
                                  lambda d: _literal_ok(str(d), literals, constants, nums))
                try:
                    v = mode.executor(expr, ctx)
                except ProgramError:
                    raise
                except Exception as e:  # noqa: BLE001 - a crashing executor is a failed call, fail closed
                    raise ProgramError(f"the {kind} executor failed ({type(e).__name__}: {e})") from None
                if not mode.validator(v):
                    raise ProgramError(f"the {kind} executor returned a {type(v).__name__}, not "
                                       f"{mode.result or 'a valid result'}")
                calls.append(CallResult(k, kind, expr, v, show(v)))
            except ProgramError as e:
                errors.append(f"{where}: {e}")
                calls.append(CallResult(k, kind, expr, error=f"{where}: {e}"))
            continue
        try:
            rest = _IN_RE.sub(" ", _SLOT_RE.sub(" ", expr))
            if kind == "calc" and _PI_RE.search(rest) and not _literal_ok(str(PI), literals, constants, nums):
                raise ProgramError(f"{where}: pi is not an allowed constant; pass constants=... | {{PI}} or point at "
                                   "the input")
            for lit in _LIT_RE.findall(rest):
                if not _literal_ok(lit, literals, constants, nums):
                    raise ProgramError(f"{where}: the literal {lit} is not allowed; point at the input ([k]) or an "
                                       "earlier call (@k)")

            def rin(mm, kind=kind, where=where):
                n = in_ref(int(mm.group(1)), where)
                if kind == "date":
                    if isinstance(n.date, dt.date):
                        return n.date.isoformat()
                    if n.kind == "number" and n.number >= 0 and n.number == n.number.to_integral_value():
                        return str(int(n.number))                # an amount: "[5]D" with [5] = 45
                    raise ProgramError(f"{where}: [{mm.group(1)}] is {n.surface!r}, not a full date or a whole number")
                if n.kind == "date":
                    raise ProgramError(f"{where}: [{mm.group(1)}] is the date {n.surface!r}; use <date> for dates")
                return f"({n.number:f})" if n.number < 0 else f"{n.number:f}"

            def rslot(mm, kind=kind, where=where, k=k):
                j = int(mm.group(1) or mm.group(2))
                if not 1 <= j < k:
                    raise ProgramError(f"{where}: @{j} is not an earlier call (calls are numbered in order, calc and "
                                       f"date together; this one is @{k})")
                c = calls[j - 1]
                if c.error:
                    raise ProgramError(f"{where}: @{j} failed ({c.error})")
                v = c.value
                if isinstance(v, bool):
                    raise ProgramError(f"{where}: @{j} is {show(v)!r}, a yes/no, not a number")
                if kind == "date":
                    if isinstance(v, dt.date):
                        return v.isoformat()
                    if isinstance(v, (int, Decimal)) and v == int(v) and v >= 0:
                        return str(int(v))
                    raise ProgramError(f"{where}: @{j} is {show(v)!r}, not a date or a whole number")
                if isinstance(v, dt.date) or isinstance(v, str):
                    raise ProgramError(f"{where}: @{j} is {show(v)!r}, not a number; use <date> for dates")
                v = Decimal(v)
                # plain notation: str(Decimal) can write 1.2463E+6, which the expression grammar does not read
                return f"({v:f})" if v < 0 else f"{v:f}"

            body = _IN_RE.sub(rin, expr)
            body = _SLOT_RE.sub(rslot, body)
            try:
                v = calc(body) if kind == "calc" else date_calc(body)
            except ProgramError as e:
                raise ProgramError(f"{where}: {e}") from None
            calls.append(CallResult(k, kind, expr, v, show(v)))
        except ProgramError as e:
            errors.append(str(e))
            calls.append(CallResult(k, kind, expr, error=str(e)))

    # the text: inline call values, then pointers; spans written by code are recorded
    out, ins = [], []
    pos = 0
    length = 0
    events = sorted([(m.start(), m.end(), "tag", m) for m in tag_re.finditer(text)])
    tag_no = {m.start(): k for k, (_, _, _, m) in enumerate(events, 1)}

    def emit(s, src=None, idx=0):
        nonlocal length
        if src is not None and s:
            ins.append(Inserted(length, length + len(s), src, idx))
        out.append(s)
        length += len(s)

    def emit_prose(seg):
        p = 0
        refs = sorted([(m.start(), m.end(), "in", m) for m in _IN_RE.finditer(seg)]
                      + [(m.start(), m.end(), "slot", m) for m in _SLOT_RE.finditer(seg)])
        for a, b, kind, m in refs:
            if a < p:
                continue
            before = seg[p:a]
            if kind == "in":
                k = int(m.group(1))
                if not 1 <= k <= len(nums):
                    errors.append(f"the text points at input numeral [{k}], which does not exist (the input has "
                                  f"[1]..[{len(nums)}])")
                    emit(before + m.group())
                    p = b
                    continue
                n = nums[k - 1]
                s = n.plain()
                after = seg[b:b + 1]
                lead = re.match(_CUR, s)
                if lead and before.rstrip().endswith(lead.group()):     # "$[3]" with "$1,234" → "$1,234"
                    s = s[1:].lstrip()
                if n.unit == "%" and after == "%":
                    s = re.sub(r"\s?%$", "", s)
                emit(before)
                emit(s, "input", k)
            else:
                j = int(m.group(1) or m.group(2))
                if not 1 <= j <= len(calls) or calls[j - 1].error:
                    errors.append(f"the text uses @{j}, which " + ("failed" if 1 <= j <= len(calls) else
                                  f"does not exist (the calls are @1..@{len(calls)})"))
                    emit(before + m.group())
                    p = b
                    continue
                emit(before)
                emit(calls[j - 1].shown, "tool", j)
            p = b
        emit(seg[p:])

    for a, b, _, m in events:
        emit_prose(text[pos:a])
        k = tag_no[a]
        c = calls[k - 1]
        if show_calls:
            if c.error:
                emit(m.group())
            else:
                emit(c.shown, "tool", k)
        pos = b
    emit_prose(text[pos:])
    final = "".join(out)

    if answer_marker is not None:
        cut = final.rfind(answer_marker)
        if cut < 0:
            errors.append(f"no {answer_marker!r} line: the answer goes after it")
        else:
            off = cut + len(answer_marker)
            stripped = final[off:]
            lead = len(stripped) - len(stripped.lstrip())
            off += lead
            final = final[off:].rstrip()
            ins = [Inserted(i.start - off, i.end - off, i.source, i.index) for i in ins if i.start >= off]
    return Filled(final, calls, ins, errors, nums, tbls)


# ============================================================================================== check
Source = Literal["input", "tool", "constant", "unsourced"]


@dataclass(frozen=True)
class NumberCheck:
    """One number of the checked text and where it comes from.

    `source`: "input", "tool", "constant" or "unsourced"; `index`: the input numeral `[k]` or call `@k` (0 otherwise);
    `how`: "pointer" (written by code), "exact", "sign" (equal but for the sign), "scale_word" (equal once a scale word
    is applied: "4.2 million" against "4,200,000"), "rescaled x1000" / "rescaled /1000" (and 10^6, 10^9: digits off by
    a whole scale, only with `rescale=True`), "rounded" (only with `rounding=True`), "date_year" (the year of a source
    date), "verbatim" (an identifier found in the input), "table cell" (a cell of a table passed as `tables`), or ""
    when unsourced. A number from a `<table>`, `<cmp>`, `<scale>` or `<count>` call has the mode as `how` ("table",
    ...) where a calc or date call has "pointer" or "exact", and `cells`, the table cells it came from."""
    surface: str
    start: int
    end: int
    kind: str
    source: Source
    index: int = 0
    how: str = ""
    cells: tuple = ()

    def to_dict(self) -> dict:
        d = {"surface": self.surface, "start": self.start, "end": self.end, "kind": self.kind,
             "source": self.source, "index": self.index, "how": self.how}
        if self.cells:
            d["cells"] = [c.to_dict() for c in self.cells]
        return d


@dataclass
class Verdict:
    """Every number of a text with its source; `ok` when none is unsourced and the program had no error."""
    numbers: list[NumberCheck]
    errors: list[str] = field(default_factory=list)

    @property
    def unsourced(self) -> list[NumberCheck]:
        return [c for c in self.numbers if c.source == "unsourced"]

    @property
    def ok(self) -> bool:
        return not self.unsourced and not self.errors

    def counts(self) -> dict[str, int]:
        out = {"input": 0, "tool": 0, "constant": 0, "unsourced": 0}
        for c in self.numbers:
            out[c.source] += 1
        return out

    def feedback(self) -> str:
        """What to tell the model in a retry: each error and each unsourced number."""
        lines = list(self.errors)
        if self.unsourced:
            lines.append("These numbers have no source: " + ", ".join(repr(c.surface) for c in self.unsourced)
                         + ". Write [k] for the k-th number of the input and <calc>…</calc> / <date>…</date> for any "
                         "computed value (its result is @k); do not type numbers.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {"ok": self.ok, "counts": self.counts(), "errors": list(self.errors),
                "numbers": [c.to_dict() for c in self.numbers]}


_SCALES = (10 ** 3, 10 ** 6, 10 ** 9)


def _decimals(n: Numeral) -> int:
    m = re.search(r"\.(\d+)", n.surface)
    return len(m.group(1)) if m else 0


def _match_number(n: Numeral, pool, signed: bool, rescale: bool, rounding: bool):
    """(source, index, how) of the best pool entry for the number n; pool: [(source, index, written, value)]."""
    order = ("exact", "sign", "scale_word", "rescaled", "rounded")
    best = None
    for src, idx, written, value in pool:
        how = None
        if value == n.value:
            how = "exact" if written == n.number else "scale_word"
        elif not signed and value == -n.value:
            how = "sign" if written == -n.number else "scale_word"
        elif rescale and n.value != 0 and (not signed or (value < 0) == (n.value < 0)):
            a, b = abs(value), abs(n.value)
            how = next((f"rescaled x{sc}" if a * sc == b else f"rescaled /{sc}" for sc in _SCALES
                        if a * sc == b or b * sc == a), None)
        if how is None and rounding and _sig_digits(n.number) >= 2:
            q = Decimal(1).scaleb(-_decimals(n))
            for v in {abs(value), abs(written)}:
                with localcontext() as ctx:
                    ctx.prec = _PREC
                    if v.quantize(q) * n.scale == abs(n.value) and (not signed or (value < 0) == (n.value < 0)):
                        how = "rounded"
                        break
        if how is not None:
            rank = order.index(how.split(" ")[0])
            if best is None or rank < best[0]:
                best = (rank, src, idx, how)
            if rank == 0:
                break
    return None if best is None else best[1:]


def _sig_digits(d: Decimal) -> int:
    return len(str(abs(d)).replace(".", "").lstrip("0"))


def check(text: str, source: str | list[Numeral] = "", results: Iterable = (), *, inserted: Iterable[Inserted] = (),
          constants=SMALL, signed: bool = False, rescale: bool = False, rounding: bool = False,
          ids: Literal["skip", "check"] = "skip", errors: Iterable[str] = (), tables=None) -> Verdict:
    """Name the source of every number in `text`.

    A number is sourced when code wrote it (`inserted`, from `fill`), when it equals an input numeral (sign ignored
    unless `signed`; a scale word applied: "4.2 million" equals "4,200,000"), a result in `results` (`CallResult`s, or
    plain values: numbers, dates, or strings such as a native tool's output), or a constant (`constants`: a set of
    numbers or a predicate on a `Decimal`). A date is sourced when it equals an input date or a result date; a year
    when it is the year of one. With `rescale=True` a number off by a whole scale (8,220 against 8,220,000) is
    sourced as "rescaled"; with `rounding=True` a number equal to a source rounded to its own decimals (two significant
    digits at least) as "rounded". Identifiers are skipped, or with `ids="check"` must appear verbatim in the input.
    `tables`: tables given apart from `source` (a `Table`, a list, or a text with pipe tables); a number equal to one
    of their cells is an input with index 0 and `how="table cell"` (the tables inside a `source` text are already its
    numerals)."""
    src_text = source if isinstance(source, str) else ""
    nums = [n for n in numerals(source) if n.kind != "id"] if isinstance(source, str) else list(source)
    pool_num, pool_date = [], []
    for k, n in enumerate(nums, 1):
        if n.kind == "date":
            pool_date.append(("input", k, n.date))
        else:
            pool_num.append(("input", k, n.number, n.value))
    for t in _as_tables(tables):
        for row in t.rows[1:]:
            for n in numerals(" ; ".join(row[1:] if t.row_labels else row)):
                if n.kind == "date":
                    pool_date.append(("input", 0, n.date))
                elif n.kind == "number":
                    pool_num.append(("input", 0, n.number, n.value))
    modes: dict[int, tuple[str, tuple]] = {}                 # call @k of a typed tool: its mode and cells
    plain = 0
    for r in results:
        if not isinstance(r, CallResult):
            plain += 1
        idx = r.k if isinstance(r, CallResult) else plain       # a plain result: its place among the plain ones
        v = r.value if isinstance(r, CallResult) else r
        if isinstance(r, CallResult) and r.error:
            continue
        if isinstance(r, CallResult) and r.kind not in ("calc", "date"):
            modes[r.k] = (r.kind, r.cells)
        if isinstance(v, dt.date):
            pool_date.append(("tool", idx, v))
            continue
        vals = []
        if isinstance(v, (int, Decimal, float)) and not isinstance(v, bool):
            vals.append(Decimal(str(v)))
        if isinstance(r, CallResult) and r.shown:
            v = r.shown
        if isinstance(v, str):
            for n in numerals(v):
                if n.kind == "date":
                    pool_date.append(("tool", idx, n.date))
                elif n.kind == "number":
                    pool_num.append(("tool", idx, n.number, n.value))
        for d in vals:
            pool_num.append(("tool", idx, d, d))
    inserted = list(inserted)
    checks = []
    for n in numerals(text):
        if n.kind == "id":
            if ids == "check":
                ok = re.search(rf"(?<![\w]){re.escape(n.surface)}(?![\w])", src_text) is not None
                checks.append(NumberCheck(n.surface, n.start, n.end, "id", "input" if ok else "unsourced", 0,
                                          "verbatim" if ok else ""))
            continue
        by_code = next((i for i in inserted if i.start < n.end and n.start < i.end), None)
        if by_code is not None:
            mode, cells = modes.get(by_code.index, ("pointer", ())) if by_code.source == "tool" else ("pointer", ())
            checks.append(NumberCheck(n.surface, n.start, n.end, n.kind, by_code.source, by_code.index, mode, cells))
            continue
        if n.kind == "date":
            hit = None
            for src, idx, d in pool_date:
                if d == n.date or (isinstance(n.date, tuple) and isinstance(d, dt.date) and (d.month, d.day) == n.date):
                    hit = (src, idx, "exact")
                    break
            checks.append(_with_mode(NumberCheck(n.surface, n.start, n.end, "date", *(hit or ("unsourced", 0, ""))),
                                     modes))
            continue
        hit = _match_number(n, pool_num, signed, rescale, rounding)
        if hit is None and n.scale == 1 and n.number == n.number.to_integral_value():
            hit = next(((src, idx, "date_year") for src, idx, d in pool_date
                        if isinstance(d, dt.date) and d.year == abs(n.number)), None)
        if hit is None and (_is_const(n.number, constants) if n.scale == 1 else _is_const(n.value, constants)):
            hit = ("constant", 0, "exact")
        checks.append(_with_mode(NumberCheck(n.surface, n.start, n.end, "number", *(hit or ("unsourced", 0, ""))),
                                 modes))
    return Verdict(checks, list(errors))


def _with_mode(c: NumberCheck, modes: dict) -> NumberCheck:
    """A number matched exactly to a typed tool's result gets the mode as `how` and the result's cells; an input
    numeral with index 0 is a cell of a table given apart."""
    if c.source == "tool" and c.index in modes and c.how == "exact":
        mode, cells = modes[c.index]
        return NumberCheck(c.surface, c.start, c.end, c.kind, c.source, c.index, mode, cells)
    if c.source == "input" and c.index == 0 and c.how == "exact":
        return NumberCheck(c.surface, c.start, c.end, c.kind, c.source, 0, "table cell")
    return c


# ============================================================================================== the guard
class UnsourcedNumbers(ValueError):
    """Raised by a fail-closed guard (`on_fail="raise"`): `outcome` holds the verdict and the filled text."""

    def __init__(self, outcome: "Outcome"):
        self.outcome = outcome
        super().__init__(outcome.verdict.feedback() or "the text failed the number guard")


@dataclass
class Outcome:
    """A guarded answer: `text` (the filled text, or the refusal when refused), `ok`, `refused`, the `verdict`, the
    calls and, after `run_with_retry`, every attempt (`attempts`, this one last)."""
    text: str
    ok: bool
    refused: bool
    verdict: Verdict
    filled: Filled
    attempts: list["Outcome"] = field(default_factory=list)

    @property
    def calls(self) -> list[CallResult]:
        return self.filled.calls

    def feedback(self) -> str:
        return self.verdict.feedback()

    def to_dict(self) -> dict:
        return {"text": self.text, "ok": self.ok, "refused": self.refused, "verdict": self.verdict.to_dict(),
                "calls": [{"k": c.k, "kind": c.kind, "expr": c.expr, "shown": c.shown, "error": c.error}
                          for c in self.filled.calls], "attempts": len(self.attempts) or 1}


REFUSAL = "I cannot give this answer: some of its numbers have no source in the input."


@dataclass
class NumberGuard:
    """`fill` and `check` with one configuration and a mode for a failure.

    `on_fail`: "report" returns the outcome as it is (the verdict says what failed), "refuse" returns `refusal` as the
    text, "raise" raises `UnsourcedNumbers`. The other fields are `fill`'s and `check`'s options."""
    on_fail: Literal["report", "refuse", "raise"] = "refuse"
    refusal: str = REFUSAL
    constants: object = SMALL
    literals: Literal["constants", "input", "any"] = "constants"
    signed: bool = False
    rescale: bool = False
    rounding: bool = False
    ids: Literal["skip", "check"] = "skip"
    answer_marker: str | None = None
    digits: int = 10
    places: int | None = None
    grouping: bool | None = None
    date_format: str | None = None
    show_calls: bool = True
    booleans: tuple = ("yes", "no")

    def fill(self, text: str, source: str, tables=None) -> Filled:
        return fill(text, source, answer_marker=self.answer_marker, literals=self.literals, constants=self.constants,
                    digits=self.digits, places=self.places, grouping=self.grouping, date_format=self.date_format,
                    show_calls=self.show_calls, tables=tables, booleans=self.booleans)

    def check(self, text: str, source: str, results: Iterable = (), **kw) -> Verdict:
        """Check a text that was not filled (a model typing its numbers, a native tool caller's answer)."""
        return check(text, source, results, constants=self.constants, signed=self.signed, rescale=self.rescale,
                     rounding=self.rounding, ids=self.ids, **kw)

    def run(self, text: str, source: str, results: Iterable = (), tables=None) -> Outcome:
        """Fill the text, check every number left in it, apply `on_fail`. `results`: values from tools run outside
        (a native tool call's output) that also count as sources. `tables`: tables given apart from `source` (None:
        the pipe tables of `source`)."""
        f = self.fill(text, source, tables)
        v = check(f.text, f.source_numerals, list(f.calls) + list(results), inserted=f.inserted,
                  constants=self.constants, signed=self.signed, rescale=self.rescale, rounding=self.rounding,
                  ids=self.ids, errors=f.errors, tables=tables)
        if self.ids == "check":                                  # identifiers need the input text itself
            idv = check(f.text, source, (), constants=self.constants, ids="check")
            ids_by_span = {(c.start, c.end): c for c in idv.numbers if c.kind == "id"}
            v = Verdict([c for c in v.numbers if c.kind != "id"] + list(ids_by_span.values()), v.errors)
            v.numbers.sort(key=lambda c: c.start)
        return self._decide(Outcome(f.text, v.ok, False, v, f))

    def _decide(self, out: Outcome) -> Outcome:
        if out.ok or self.on_fail == "report":
            return out
        if self.on_fail == "raise":
            raise UnsourcedNumbers(out)
        out.text, out.refused = self.refusal, True
        return out

    def run_with_retry(self, generate: Callable[[str | None], str], source: str, retries: int = 1,
                       results: Iterable = (), tables=None) -> Outcome:
        """Ask, guard, and on a failure ask again with what failed. `generate(feedback)` returns the model's text:
        it is called with None first, then with `Verdict.feedback()` (the errors and the unsourced numbers) up to
        `retries` more times. The last attempt goes through `on_fail`; `Outcome.attempts` holds them all."""
        report = NumberGuard(**{**self.__dict__, "on_fail": "report"})
        attempts: list[Outcome] = []
        feedback = None
        for _ in range(retries + 1):
            out = report.run(generate(feedback), source, results, tables)
            attempts.append(out)
            if out.ok:
                break
            feedback = out.feedback()
        out = attempts[-1]
        out.attempts = attempts
        return self._decide(out)
