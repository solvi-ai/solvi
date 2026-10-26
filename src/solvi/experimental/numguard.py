"""Exact values through a side channel: the model writes pointers, code writes the numbers, a guard checks that every
number in the final text has a source (experimental).

A model that types numbers into free text makes some of them up. Here it writes pointers instead:

- `[k]` — the k-th numeral of the input text (1-based, in reading order; `annotate(source)` shows the numbering);
- `<calc>expr</calc>` / `<date>expr</date>` — a call computed exactly by code (`Decimal` arithmetic, calendar
  arithmetic); the calls are numbered in order of appearance, calc and date together, and `@k` (or `{@k}`) is the
  value of the k-th call.

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
can be checked against the input verbatim with `ids="check"`). Integers 0–100 and a few common factors pass as
constants by default (`DEFAULT_CONSTANTS`); pass `constants=SMALL` or your own set to tighten it."""
from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, localcontext
from typing import Callable, Iterable, Literal

from . import warn_on_import

warn_on_import(__name__)

__all__ = ["Numeral", "Inserted", "CallResult", "Filled", "NumberCheck", "Verdict", "Outcome", "NumberGuard",
           "UnsourcedNumbers", "ProgramError", "numerals", "annotate", "calc", "date_calc", "fill", "check",
           "DEFAULT_CONSTANTS", "SMALL", "years"]

# ============================================================================================== constants
SMALL = frozenset(Decimal(i) for i in range(13)) | {Decimal(100)}
"""Integers 0–12 and 100: counts, months, a percent base."""
DEFAULT_CONSTANTS = frozenset(Decimal(i) for i in range(101)) | frozenset(
    Decimal(x) for x in (365, 360, 52, 1000, 10 ** 6, 10 ** 9))
"""Integers 0–100, days in a year (365, 360), weeks (52), 1,000, a million, a billion: the set the measurements behind
the docs used, for calc literals and free constants alike."""


def years(lo: int = 1900, hi: int = 2100) -> frozenset:
    """The years lo..hi as constants: `constants=DEFAULT_CONSTANTS | years(2000, 2030)`."""
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


_CTOK = re.compile(r"\s*(\d+(?:\.\d+)?|[-+*/()%^])")
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
    `Decimal` with 50 digits. An integer power up to 1,000 is exact; a fractional one needs a positive base. Raises
    `ProgramError` with a readable reason."""
    s = (expr.replace("**", "^").replace("×", "*").replace("÷", "/").replace("−", "-"))
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)
    toks, pos = [], 0
    while pos < len(s):
        m = _CTOK.match(s, pos)
        if not m:
            if not s[pos:].strip():
                break
            raise ProgramError(f"calc allows only numbers, + - * / ^ ( ) and a postfix %; it stopped at {s[pos:pos + 12]!r}")
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


# ============================================================================================== fill
_TAG_RE = re.compile(r"<(calc|date)>(.*?)</\1>", re.S | re.I)
_IN_RE = re.compile(r"\[(\d+)\]")
_SLOT_RE = re.compile(r"\{\s*@(\d+)\s*\}|(?<![\w@.])@(\d+)(?=[DWMYdwmy]\b|[^\w]|$)")
_LIT_RE = re.compile(r"(?<![\w.\[@])(\d+(?:\.\d+)?)(?=\s?[DWMY]\b|[^\w]|$)", re.I)


@dataclass(frozen=True)
class CallResult:
    """The k-th call: its kind, the expression as written, the value (a Decimal, a date, an int of days or a weekday)
    and the text it was shown as; `error` instead of a value when it failed."""
    k: int
    kind: str
    expr: str
    value: object = None
    shown: str = ""
    error: str = ""


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
         literals: Literal["constants", "input", "any"] = "constants", constants=DEFAULT_CONSTANTS,
         digits: int = 10, places: int | None = None, grouping: bool | None = None,
         date_format: str | None = None, show_calls: bool = True) -> Filled:
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
    a missing marker is an error in `Filled.errors`; its pointer is left as written."""
    nums = [n for n in numerals(source) if n.kind != "id"] if isinstance(source, str) else list(source)
    if grouping is None:
        grouping = isinstance(source, str) and re.search(r"\d,\d{3}(?!\d)", source) is not None
    errors: list[str] = []
    calls: list[CallResult] = []

    def show(v) -> str:
        if isinstance(v, Decimal):
            return _fmt(v, digits, places, grouping)
        if isinstance(v, dt.date):
            return v.strftime(date_format) if date_format else v.isoformat()
        return str(v)

    def in_ref(k: int, where: str):
        if not 1 <= k <= len(nums):
            raise ProgramError(f"{where}: input numeral [{k}] does not exist (the input has [1]..[{len(nums)}])")
        return nums[k - 1]

    for k, m in enumerate(_TAG_RE.finditer(text), 1):
        kind, expr = m.group(1).lower(), m.group(2).strip()
        lab = re.match(r"@(\d+)\s*=(?!=)\s*", expr)              # "<calc>@3 = [1] - [2]</calc>" names its own call
        if lab and int(lab.group(1)) == k:
            expr = expr[lab.end():]
        where = f"call @{k} <{kind}>{m.group(2).strip()}</{kind}>"
        try:
            rest = _IN_RE.sub(" ", _SLOT_RE.sub(" ", expr))
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
                return f"({n.number})" if n.number < 0 else str(n.number)

            def rslot(mm, kind=kind, where=where, k=k):
                j = int(mm.group(1) or mm.group(2))
                if not 1 <= j < k:
                    raise ProgramError(f"{where}: @{j} is not an earlier call (calls are numbered in order, calc and "
                                       f"date together; this one is @{k})")
                c = calls[j - 1]
                if c.error:
                    raise ProgramError(f"{where}: @{j} failed ({c.error})")
                v = c.value
                if kind == "date":
                    if isinstance(v, dt.date):
                        return v.isoformat()
                    if isinstance(v, (int, Decimal)) and v == int(v) and v >= 0:
                        return str(int(v))
                    raise ProgramError(f"{where}: @{j} is {show(v)!r}, not a date or a whole number")
                if isinstance(v, dt.date) or isinstance(v, str):
                    raise ProgramError(f"{where}: @{j} is {show(v)!r}, not a number; use <date> for dates")
                v = Decimal(v)
                return f"({v})" if v < 0 else str(v)

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
    events = sorted([(m.start(), m.end(), "tag", m) for m in _TAG_RE.finditer(text)])
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
    return Filled(final, calls, ins, errors, nums)


# ============================================================================================== check
Source = Literal["input", "tool", "constant", "unsourced"]


@dataclass(frozen=True)
class NumberCheck:
    """One number of the checked text and where it comes from.

    `source`: "input", "tool", "constant" or "unsourced"; `index`: the input numeral `[k]` or call `@k` (0 otherwise);
    `how`: "pointer" (written by code), "exact", "sign" (equal but for the sign), "scale_word" (equal once a scale word
    is applied: "4.2 million" against "4,200,000"), "rescaled x1000" / "rescaled /1000" (and 10^6, 10^9: digits off by
    a whole scale, only with `rescale=True`), "rounded" (only with `rounding=True`), "date_year" (the year of a source
    date), "verbatim" (an identifier found in the input), or "" when unsourced."""
    surface: str
    start: int
    end: int
    kind: str
    source: Source
    index: int = 0
    how: str = ""

    def to_dict(self) -> dict:
        return {"surface": self.surface, "start": self.start, "end": self.end, "kind": self.kind,
                "source": self.source, "index": self.index, "how": self.how}


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
          constants=DEFAULT_CONSTANTS, signed: bool = False, rescale: bool = False, rounding: bool = False,
          ids: Literal["skip", "check"] = "skip", errors: Iterable[str] = ()) -> Verdict:
    """Name the source of every number in `text`.

    A number is sourced when code wrote it (`inserted`, from `fill`), when it equals an input numeral (sign ignored
    unless `signed`; a scale word applied: "4.2 million" equals "4,200,000"), a result in `results` (`CallResult`s, or
    plain values: numbers, dates, or strings such as a native tool's output), or a constant (`constants`: a set of
    numbers or a predicate on a `Decimal`). A date is sourced when it equals an input date or a result date; a year
    when it is the year of one. With `rescale=True` a number off by a whole scale (8,220 against 8,220,000) is
    sourced as "rescaled"; with `rounding=True` a number equal to a source rounded to its own decimals (two significant
    digits at least) as "rounded". Identifiers are skipped, or with `ids="check"` must appear verbatim in the input."""
    src_text = source if isinstance(source, str) else ""
    nums = [n for n in numerals(source) if n.kind != "id"] if isinstance(source, str) else list(source)
    pool_num, pool_date = [], []
    for k, n in enumerate(nums, 1):
        if n.kind == "date":
            pool_date.append(("input", k, n.date))
        else:
            pool_num.append(("input", k, n.number, n.value))
    plain = 0
    for r in results:
        if not isinstance(r, CallResult):
            plain += 1
        idx = r.k if isinstance(r, CallResult) else plain       # a plain result: its place among the plain ones
        v = r.value if isinstance(r, CallResult) else r
        if isinstance(r, CallResult) and r.error:
            continue
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
            checks.append(NumberCheck(n.surface, n.start, n.end, n.kind, by_code.source, by_code.index, "pointer"))
            continue
        if n.kind == "date":
            hit = None
            for src, idx, d in pool_date:
                if d == n.date or (isinstance(n.date, tuple) and isinstance(d, dt.date) and (d.month, d.day) == n.date):
                    hit = (src, idx, "exact")
                    break
            checks.append(NumberCheck(n.surface, n.start, n.end, "date", *(hit or ("unsourced", 0, ""))))
            continue
        hit = _match_number(n, pool_num, signed, rescale, rounding)
        if hit is None and n.scale == 1 and n.number == n.number.to_integral_value():
            hit = next(((src, idx, "date_year") for src, idx, d in pool_date
                        if isinstance(d, dt.date) and d.year == abs(n.number)), None)
        if hit is None and (_is_const(n.number, constants) if n.scale == 1 else _is_const(n.value, constants)):
            hit = ("constant", 0, "exact")
        checks.append(NumberCheck(n.surface, n.start, n.end, "number", *(hit or ("unsourced", 0, ""))))
    return Verdict(checks, list(errors))


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
    constants: object = DEFAULT_CONSTANTS
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

    def fill(self, text: str, source: str) -> Filled:
        return fill(text, source, answer_marker=self.answer_marker, literals=self.literals, constants=self.constants,
                    digits=self.digits, places=self.places, grouping=self.grouping, date_format=self.date_format,
                    show_calls=self.show_calls)

    def check(self, text: str, source: str, results: Iterable = (), **kw) -> Verdict:
        """Check a text that was not filled (a model typing its numbers, a native tool caller's answer)."""
        return check(text, source, results, constants=self.constants, signed=self.signed, rescale=self.rescale,
                     rounding=self.rounding, ids=self.ids, **kw)

    def run(self, text: str, source: str, results: Iterable = ()) -> Outcome:
        """Fill the text, check every number left in it, apply `on_fail`. `results`: values from tools run outside
        (a native tool call's output) that also count as sources."""
        f = self.fill(text, source)
        v = check(f.text, f.source_numerals, list(f.calls) + list(results), inserted=f.inserted,
                  constants=self.constants, signed=self.signed, rescale=self.rescale, rounding=self.rounding,
                  ids=self.ids, errors=f.errors)
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
                       results: Iterable = ()) -> Outcome:
        """Ask, guard, and on a failure ask again with what failed. `generate(feedback)` returns the model's text:
        it is called with None first, then with `Verdict.feedback()` (the errors and the unsourced numbers) up to
        `retries` more times. The last attempt goes through `on_fail`; `Outcome.attempts` holds them all."""
        report = NumberGuard(**{**self.__dict__, "on_fail": "report"})
        attempts: list[Outcome] = []
        feedback = None
        for _ in range(retries + 1):
            out = report.run(generate(feedback), source, results)
            attempts.append(out)
            if out.ok:
                break
            feedback = out.feedback()
        out = attempts[-1]
        out.attempts = attempts
        return self._decide(out)
