"""solvi.document — a decision over a document: a model reads one fact per field, code computes the answer.

    import solvi
    from pydantic import BaseModel, Field
    from solvi import Span
    from solvi.models import llm

    class Operands(BaseModel):
        a: Span[float] = Field(description="the first number the question names, exactly as written")
        b: Span[float] = Field(description="the second number, exactly as written")

    recipe = solvi.document(Operands, llm(URL, MODEL, api_key=KEY),
                            task="The text starts with a QUESTION, then a table and notes.",
                            compute=lambda f, s: f.a - f.b,                  # f: the facts, s: the rest of the input
                            checks={"b_not_zero": lambda f, s: f.b != 0},
                            storage="decisions.jsonl")
    recipe.calibrate([({"doc": d, "op": op}, gold) for d, op, gold in labelled], max_risk=0.10)
    res = recipe.ask({"doc": text, "op": "diff"})
    res["answer"].answer, res["answer"].status, res["answer"].why, res.stored_id
    recipe.verify(), recipe.replay_all()

The scheme it sets up (all of it existing low-level blocks, wired):

- one read per field: each field of the schema is one decision part (`DecideModel.decision`), its description the
  question (after `task=`), the text fact `text=` (default "doc") its input — one model call per field, the same text
  each call. A field with `json_schema_extra={"options": {option: description}}` asks with the plain option list and
  the descriptions in the question's text (a model that echoes "option: description" back is spared the chance).
- grounding: a `Span[...]` field must be found in the text (normalized view; a number as a number of its own, not
  inside a longer one — solvi.core.catalog.locate, cuts_number); not found, the field escalates and the answer
  abstains with the reason. The kept value is the text's own substring.
- numbers: a `Span[float]` (int, Decimal) is read from what was quoted, as the document writes it, into a fact
  `<field>_value` (the rule's `why` shows it next to the quote). One scale rule — a scale is never multiplied, the
  number is kept in the document's own unit ("1,250 million", "$1.2bn", "1.2M", "3k" → 1250, 1.2, 1.2, 3); the
  currency (a sign — "$", "US$", "R$", "HK$", "Rs", "€" ... — or a known ISO code: "EUR", "USD"; "EBT", "NET" are not
  currencies) dropped; a percentage as its number ("12.5%" → 12.5); a sign in any dash, before or after the currency,
  or trailing ("1,234-"); parentheses as an accounting negative ("(1,234)", "$(1,234) million", "EUR (1,234)",
  "(12%)", "(12)%" → −1234, −12) — only around a number with a separator, decimals, a currency or a percent ("(2019)",
  "(1)" do not read: a year, a footnote). Anything else — a range, two numbers, two scales or currencies, an
  exponent ("1e3"), "1_000", NaN or infinity, an ambiguous "2.500" — does not read, and the answer abstains with the
  reason (a CR / DR marker or brackets do not read either: the sign depends on the account). One rule keeps the
  quote honest: at every occurrence of its digits in the text (with at least the quote's currency and scale, any
  spaces between; full-width digits and marks read as their ASCII) the number token is extended over what is bound
  to the digits — a sign in any dash, minus or "&minus;" (any horizontal space after it), a currency, "(" with ")"
  closing right after (any white space inside), "[" "]", a CR / DR marker, a scale word or suffix, a percent. A sign,
  parentheses or marker the quote left out abstains naming it ("$1,234" in prose and "(1,234)" in a table do not
  read as +1234), so does a quote that reads a sign the text does not write there ("- 1,234" from a list bullet) and
  a quote whose digits do not stand in the text at all. A currency or a scale the quote left out is not a refusal: it
  is taken from the text into the field's unit ("$1.2" of "$1.2 billion" is in billion; so is a percent: "12" of
  "12%" is a percent, and of "-12%" it abstains for the sign), and occurrences that
  disagree ("$1,234" here, "€1,234" there) abstain as ambiguous. Not bound to the digits: a "- " bullet at a line's
  start, a dash right after a letter, a digit or "%" ("10-K", "2018-2019", "50%-60%"), a spaced dash after a word, a
  digit, "%" or a dash ("Revenue - 1,234", "1,234 – 1,300", "50% - 60%"), "(" that does not close right after the
  number ("1,234 (Note 5)"). A number in parentheses after another number on its line — up to ten characters of a
  unit between, one new line, a tab or a comma allowed; a table's "|" is a separator — is a comparative or a
  negative, the text does not say ("SEK 1,234 m (1,100)", "12.3 percent (11.8)", "2.1x (2.5)"), and the answer
  abstains. An occurrence with a percent where the quote has none does not count. Each look reads a bounded window,
  so a long text costs no more per occurrence.
- units: each number field's unit — what its quote writes, completed from the text — is `facts.units[field]`, a
  Unit(scale, currency, percent) (also the fact "units" in the record). With two or more number fields a hard check
  `same_units` compares the amounts (a percentage is not an amount and is not compared): `units="same"` (the
  default) abstains when their scales differ — "$1.2bn" and "$950 million", and a field without a scale next to one
  with a scale ("3k" and "3,000") — or their currencies differ ("$100" and "€100", and a bare "$" next to an
  explicit "C$"; a field without a currency conflicts with nothing); `units="lenient"` lets a field without a scale
  match any and a bare "$" any dollar; `units="any"` turns the check off.
- code computes: `compute=(facts, state) → value` is the question's rule; `facts` is the schema filled with the read
  values, `state` the input without the text. The answer is a number (`Answer.estimate()`) unless `answer=` says
  otherwise (`solvi.Answer.choice(["pay", "hold"])`, `Answer.yes_no()`); its confidence is the weakest fact's.
  A result that is not a finite number (NaN, infinity) abstains. Without `compute=` a schema of one field answers
  with that field's value: a number (estimate), or a `Literal` / `Enum` / `bool` field's option; a `Span[str]` or
  `Span[date]` field needs compute= and answer=.
- checks: `checks={name: (facts, state) → bool}` are hard checks on the answer: one that fails makes it abstain
  ("failed check"). The names `units`, `same_units`, `compute` and `<field>_value` are the recipe's own.
- confidence: an LLM made with `logprobs="auto"` (llm()'s default) is rebuilt to ask for one stated confidence per
  read (`logprobs=False, ask="confidence"`), with a UserWarning that says so; a gateway that mixes logprob and stated
  replies cannot then split a calibration. A model made with `logprobs=True` or `False`, and any other decider, is
  used as given. `recipe.model` is the model the reads go to (its usage counters are there).
- calibration: `calibrate(examples, max_risk=)` is System.guarantee on the answer (conformal risk control; max_error=
  for learn-then-test; groups=, correct=, weak= pass through): below the threshold the answer abstains. Fewer than
  100 examples: a GuaranteeWarning (the threshold rests on few examples and can move); a signal that does not
  separate right from wrong raises, as System.guarantee does. A number answer is judged by its label: a label
  written as a string states its precision ("33.33" takes 33.3333, "1,500" takes 1500.4: within half a unit of its
  last digit; a whole number below 100, "33", is exact); a label given as a number is matched up to float noise
  (math.isclose: the int 1 does not take 1.4).
  Pass correct=(result, label) → bool to judge otherwise.
- record: `storage=` (a path: JSONLStorage, `record="full"` so every decision replays) keeps every ask hash-chained;
  `verify()` checks the chain, `replay_all()` re-checks every stored decision against this system.
  `save_calibration(path)` / `document(..., calibration=path)` keep the calibrated threshold across processes
  (System.save_guarantee / load_guarantee: a file made for another catalog — another task=, a field's
  description, compute or a check — or for other models is refused; the models are compared as every model of the
  System fingerprints itself: an endpoint's name, settings and the question asked — a server that changes the
  weights behind an unversioned model name is not told apart).

The input of `ask` is a dict with the text under `text=` and anything else the rule and checks read; the system is
given the text and the rest as one fact named "state", made plain data first (numpy / pandas scalars as numbers,
tuples and sets as lists, dates and Decimals as they are; a numpy array or anything else the record cannot store is
refused), so a stored decision replays.
DocumentSystem holds a solvi.System (`recipe.system`) and does not subclass it (System is final): the low-level calls
(`recipe.system.ask`, `.guarantee`, `.report`) stay open."""
from __future__ import annotations

import inspect
import sys
import time
import math
import re
import typing
import warnings

from ..core.catalog import Answer, Catalog, Question

QUESTION = "answer"
STATE = "state"                     # the fact that carries the input without the text
MIN_CALIBRATION = 100               # below this, calibrate warns: the threshold rests on few examples
UNIT_MODES = ("same", "lenient", "any")

# every dash or minus sign a document writes (length-preserving, so offsets into the text stay as they are); "△" / "▲",
# the minus of Japanese and Chinese financial tables, stay themselves (never a separator) and read as a minus
_DASHES = str.maketrans({**{c: "-" for c in "−–—―‑‐‒﹣－⁻˗"},     # ... and full-width digits and marks
                         **{chr(0xFF10 + i): str(i) for i in range(10)}, "（": "(", "）": ")", "，": ",", "．": ".",
                         "［": "[", "］": "]", "＄": "$", "％": "%"})
_ISO = ("USD EUR GBP JPY CNY RMB RUB CHF CAD AUD NZD HKD SGD TWD INR BRL MXN ZAR SEK NOK DKK PLN CZK HUF KRW TRY ILS "
        "AED SAR").split()
_CUR_SYMBOL = {"U.S.$": "USD", "US$": "USD", "A$": "AUD", "AU$": "AUD", "C$": "CAD", "CA$": "CAD", "HK$": "HKD",
               "NZ$": "NZD", "S$": "SGD", "NT$": "TWD", "R$": "BRL", "Rs.": "INR", "Rs": "INR", "₹": "INR",
               "$": "$", "€": "EUR", "£": "GBP", "¥": "JPY", "₽": "RUB", "RMB": "CNY"}
_DOLLARS = {"USD", "AUD", "CAD", "HKD", "NZD", "SGD", "TWD", "BRL", "MXN"}     # what a bare "$" may stand for
_CUR_RX = (r"(?:(?<![A-Za-z])(?:U\.S\.|US ?|AU|A|CA|C|HK|NZ|S|NT|R)\$|[$€£¥₽₹]|\bRs\b\.?|\b(?:"
           + "|".join(_ISO) + r")\b)")
_CURRENCY = re.compile(_CUR_RX)
_PERCENT = re.compile(r"\s*(?:%|\bper\s?cent\b)", re.IGNORECASE)
_SCALES = {"thousand": "thousand", "thousands": "thousand", "k": "thousand", "million": "million",
           "millions": "million", "mn": "million", "m": "million", "billion": "billion", "billions": "billion",
           "bn": "billion", "b": "billion", "trillion": "trillion", "trillions": "trillion", "tn": "trillion"}
_SCALE_WORD = re.compile(r"\s*\b(thousands?|millions?|billions?|trillions?|mn|bn|tn)\b\.?", re.IGNORECASE)
_SCALE_SUFFIX = re.compile(r"(?<=\d)\s?(bn|mn|tn|k|m|b)\b", re.IGNORECASE)
_CORE = re.compile(r"[0-9][0-9,. ]*|\.[0-9]+")
_SP = "[    ]"                      # one space of a line (not a tab, not a new line)
_HS = r"[^\S\n]"                                   # any horizontal white space (tab, U+2007, U+3000 ...)


class Unit(typing.NamedTuple):
    """What a number field's quote writes besides the number: scale ("thousand", "million", "billion", "trillion" or
    None — never multiplied in), currency (an ISO code or None), percent (True for "12%")."""
    scale: str | None = None
    currency: str | None = None
    percent: bool = False


def _currency_code(tok):
    tok = re.sub(r"\s+", "", tok)
    return _CUR_SYMBOL.get(tok, tok)


def _same_currency(x, y):
    """Do two written currencies agree? A bare "$" is any dollar ("US$", "C$", "USD" ...)."""
    return x == y or (x == "$" and y in _DOLLARS) or (y == "$" and x in _DOLLARS)


def _read_quote(vtype, text):
    """A quoted number → (the value as vtype, its Unit). See the module docs, "numbers": the scale never multiplied,
    the currency dropped (an ISO code or a sign), a percentage as its number, a sign in any dash before or after the
    currency or trailing, parentheses as a negative around a number with a separator, decimals, a currency or a
    percent; the digits and separators as solvi.core.types.span_value reads them. ValueError when it is not one number."""
    import unicodedata

    from ..core.types import span_value
    if not isinstance(text, str):
        raise ValueError(f"not a quoted number: {text!r}")
    t = unicodedata.normalize("NFKC", text.replace("&minus;", "-")).translate(_DASHES).replace("△", "-") \
        .replace("▲", "-").strip()
    if re.search(r"\b(?:CR|Cr|cr|DR|Dr|dr)\b|\[|\]", t):
        raise ValueError("a CR / DR marker or brackets: the sign depends on the account, the quote does not say")
    if re.search(r"[0-9][eE][+-]?[0-9]|_", t):
        raise ValueError("an exponent or an underscore is not how a document writes a number")
    if re.search(r"%\s*[0-9]", t):
        raise ValueError("a percent sign before the number")
    curs = {_currency_code(m.group()) for m in _CURRENCY.finditer(t)}
    if len(curs) > 1 and not (len(curs) == 2 and _same_currency(*sorted(curs))):
        raise ValueError(f"two currencies ({', '.join(sorted(curs))})")
    curs = {max(curs, key=len)} if curs else curs
    t = _CURRENCY.sub(" ", t)
    scales = {_SCALES[m.group(1).lower()] for rx in (_SCALE_WORD, _SCALE_SUFFIX) for m in rx.finditer(t)}
    if len(scales) > 1:
        raise ValueError(f"two scales ({', '.join(sorted(scales))})")
    t = _SCALE_SUFFIX.sub(" ", _SCALE_WORD.sub(" ", t))
    pct = bool(_PERCENT.search(t))
    t = _PERCENT.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\(\s*", "(", re.sub(r"\s*\)", ")", t))
    neg = False
    if t.startswith("(") and t.endswith(")"):
        neg, t = True, t[1:-1].strip()
        if not curs and not pct and not re.search(r"[.,]", t):
            raise ValueError("a bare number in parentheses (a year, a footnote?) is not read as a negative")
    elif t.startswith("-"):
        neg, t = True, t[1:].strip()
    elif t.endswith("-"):
        neg, t = True, t[:-1].strip()
    elif t.startswith("+"):
        t = t[1:].strip()
    if not _CORE.fullmatch(t or "-"):
        raise ValueError("not one number")
    v = span_value(vtype, t)
    if isinstance(v, float) and not math.isfinite(v):
        raise ValueError("not a finite number")
    return (-v if neg else v), Unit(next(iter(scales), None), next(iter(curs), None), pct)


def _read_number(vtype, text):
    """A quoted number → its value (_read_quote without the unit)."""
    return _read_quote(vtype, text)[0]


_R_SCALE = re.compile(_SP + r"?(?:thousands?|millions?|billions?|trillions?|mn|bn|tn|k|m|b)\b\.?", re.IGNORECASE)
_R_PCT = re.compile(_SP + r"?(?:%|bps\b|basis points\b|percent\b|per" + _SP + r"?cent\b)", re.IGNORECASE)
_R_ISO = re.compile(_SP + r"?(?:" + "|".join(_ISO) + r")\b")
_R_MARK = re.compile(_HS + r"*(?:CR|Cr|cr|DR|Dr|dr)\b")
_R_CLOSE = re.compile(r"\s*\)")
_R_CLOSE_SQ = re.compile(r"\s*\]")
_L_CUR = re.compile(r"(?:" + _CUR_RX + r")" + _HS + r"*\Z")
_L_DASH = re.compile(r"(?:[-△▲]|&minus;)\s*\Z")
_L_OPEN = re.compile(r"\(\s*\Z")
_L_OPEN_SQ = re.compile(r"\[\s*\Z")
# a parenthesised number after another number on its logical line (one new line, a tab or a comma between allowed;
# a table's "|" or ";" is a separator): "SEK 1,234 m (1,100)", "12.3 percent (11.8)", "2.1x\n(2.5)"
_AFTER_NUMBER = re.compile(r"[0-9][^\n|;()\[\]0-9]{0,10}(?:\n[^\S\n]*)?\Z")
_NUM_CORE = re.compile(r"[0-9](?:[0-9,.]*[0-9])?(?:[ \u00a0\u2009\u202f][0-9]{3}(?![0-9]))*")
_GROUP_LEFT = re.compile(r"[0-9][ \u00a0\u2009\u202f]\Z")
_GROUP_RIGHT = re.compile(r"[ \u00a0\u2009\u202f][0-9]{3}(?![0-9])")
_WINDOW = 64


def _line_start(s, k):
    """Is s[k] the first non-blank character of its line?"""
    ls = s.rfind("\n", 0, k) + 1
    return s[ls:k].strip() == ""


def _token(s, a, b, quote=False):
    """The number token around the digits s[a:b]: what is bound to them — a sign (any dash or "&minus;", with any
    horizontal space after it), a currency, "(" with ")" closing right after (any white space inside), "[" with "]",
    a scale word or suffix, a percent or bps, a CR / DR marker, a trailing "-" — → ({kind: text} for kinds "sign",
    "paren", "mark", "currency", "scale", "percent"; start; end; ambiguous). Not bound: a "- " list bullet at a
    line's start, a dash right after a letter, a digit or "%" ("10-K", "2018-2019", "50%-60%"; a currency code before
    it is bound: "EUR-1,234"), a spaced dash after a word, a digit, "%" or a dash (a separator), "(" that does not close
    right after the number. quote=True: s is a quote alone (its start is not a line's start). ambiguous: a
    parenthesised number right after another number on its line ("1,234 m (1,100)") — a comparative or a negative.
    Every look to the left reads a window of 64 characters, so the cost does not grow with the text."""
    p, j, i = {}, b, a
    for kind, rx in (("scale", _R_SCALE), ("percent", _R_PCT), ("currency", _R_ISO), ("mark", _R_MARK)):
        m = rx.match(s, j)
        if m and kind not in p:
            p[kind], j = m.group().strip(), m.end()
    close, close_sq = _R_CLOSE.match(s, j), _R_CLOSE_SQ.match(s, j)
    while True:
        w0 = max(0, i - _WINDOW)
        left = s[w0:i]
        m = _L_CUR.search(left)
        if m and "currency" not in p:
            p["currency"], i = m.group().strip(), w0 + m.start()
            continue
        m = _L_DASH.search(left)
        if m and "sign" not in p:
            k = w0 + m.start()
            if s[k] in "△▲":                          # never a separator, never a bullet
                p["sign"], i = "-", k
                continue
            spaced = s[k] == "-" and k + 1 < i
            if spaced and "\n" in s[k:i]:             # "-" then the number on the next line: bound, as "( ... )"
                if not (s[k - 1:k].isalnum() or s[k - 1:k] == "%"):
                    p["sign"], i = "-", k
                    continue
                break
            prev = s[k - 1:k]
            bare_prev = s[max(0, k - _WINDOW):k].rstrip()[-1:]
            if not spaced:
                if (prev.isalnum() or prev == "%") and not _L_CUR.search(s[max(0, k - _WINDOW):k]):
                    break                              # "10-K", "2018-2019", "50%-60%"
            elif not quote and _line_start(s, k):
                break                                  # a "- " list bullet
            elif bare_prev.isalnum() or bare_prev in ("-", "%"):
                break                                  # "Revenue - 1,234", "1,234 – 1,300", "50% - 60%": a separator
            p["sign"], i = "-", k
            continue
        m = _L_OPEN.search(left)
        if m and "paren" not in p and close:
            p["paren"], i, j = "()", w0 + m.start(), close.end()
            continue
        m = _L_OPEN_SQ.search(left)
        if m and "mark" not in p and close_sq:
            p["mark"], i, j = "[]", w0 + m.start(), close_sq.end()
            continue
        break
    if "sign" not in p and "paren" not in p and s[j:j + 1] == "-" and not s[j + 1:j + 2].isalnum():
        p["sign"], j = "-", j + 1
    ambiguous = "paren" in p and not quote and bool(_AFTER_NUMBER.search(s[max(0, i - _WINDOW):i].replace("\t", " ")
                                                                         .replace(",", " ")))
    return p, i, j, ambiguous


_KIND_WORDS = {"sign": "sign", "paren": "parentheses (a negative)", "mark": "CR / DR marker or brackets",
               "currency": "currency", "scale": "scale", "percent": "percent"}
_SIGNED = ("sign", "paren", "mark")


def _ground(raw, doc):
    """The grounded quote against its number tokens in the text (_token) → (reason to abstain or None, {"scale",
    "currency"} the text writes and the quote does not). At every occurrence of the quote — a place its digits stand
    as a number of their own with at least the quote's currency and scale, whatever the spaces ("$1,234" occurs in
    "-$ 1,234"), with or without a percent as the quote — the token may carry no sign, parentheses or CR / DR marker
    the quote left out, and the quote no sign the text does not write there (a "- " bullet read as a minus); a
    currency or a scale the quote left out is taken from the text (several occurrences that disagree: ambiguous). A
    quote whose digits do not stand in the text at all abstains too."""
    from ..core.catalog import cuts_number
    if not isinstance(raw, str) or not isinstance(doc, str):
        return None, {}
    q = raw.replace("&minus;", "-").translate(_DASHES).strip()
    core = _NUM_CORE.search(q)
    if core is None:
        return None, {}
    qp, _, _, _ = _token(q, core.start(), core.end(), quote=True)
    n = doc.translate(_DASHES)
    digits, spans, k = core.group(), [], n.find(core.group())
    while k >= 0:                                      # every place the quote's digits stand as a number of their own
        e = k + len(digits)                            # (not a group of "1 234" either)
        if not cuts_number(n, k, e) and not (_GROUP_LEFT.search(n[max(0, k - 2):k]) and re.match(r"[0-9]{3}(?![0-9])",
                                                                                                digits)) \
                and not _GROUP_RIGHT.match(n, e):
            spans.append(k)
        k = n.find(digits, k + 1)
    signed_q = any(x in qp for x in _SIGNED)
    seen, unsigned, inherited = 0, None, {}
    found = []
    for k in spans:
        tp, a, b, ambiguous = _token(n, k, k + len(digits))
        if any(x in qp and x not in tp for x in ("currency", "scale")):
            continue                                   # not an occurrence of the quote: it writes less there
        found.append((tp, a, b, ambiguous))
    same = [o for o in found if ("percent" in o[0]) == ("percent" in qp)]
    if same:
        found = same                                   # another quantity (a percent next to the amount) is left out
    elif found and "percent" not in qp:
        inherited["percent"] = True                    # "12" of "12%": the field is in percent
    for tp, a, b, ambiguous in found:
        written = n[a:b].rstrip(".")
        if signed_q and not any(x in tp for x in _SIGNED):
            unsigned = unsigned or written             # the digits without the quote's sign: another occurrence
            continue
        seen += 1
        if ambiguous:
            return (f"the text writes {written!r} right after another number ({n[max(0, a - 16):b].strip()!r}): a "
                    "comparative or a negative, the text does not say"), {}
        lost = [x for x in _SIGNED if x in tp and x not in qp]   # a "-" quote of "(1,234)" keeps its sign
        if lost and (not signed_q or "mark" in lost):
            return (f"the text writes {written!r}, the quote {raw!r} left out its "
                    + " and ".join(_KIND_WORDS[x] for x in lost)), {}
        for x in ("currency", "scale"):
            if x in tp and x not in qp:
                v = _currency_code(tp[x]) if x == "currency" else _SCALES.get(tp[x].lower().rstrip("."), tp[x])
                if inherited.get(x, v) != v:
                    return (f"the quote {raw!r} stands in the text with different "
                            f"{'currencies' if x == 'currency' else 'scales'} ({inherited[x]}, {v}): which "
                            "one it means, the text does not say"), {}
                inherited[x] = v
    if signed_q and not seen and unsigned is not None:
        return (f"the quote {raw!r} reads a sign the text does not write there ({unsigned!r}: a list bullet or a "
                "separator)"), {}
    if not spans or not found:
        return f"the number of the quote {raw!r} does not stand in the text as a number of its own", {}
    return None, inherited


def _left_out(raw, doc):
    """_ground's reason alone (None: the quote reads as the text writes it)."""
    return _ground(raw, doc)[0]


def _plain(v, where="state"):
    """An input value → plain data the record stores and a replay reads back as the same: numpy / pandas scalars as
    Python numbers, tuples as lists, sets as sorted lists; dates and Decimals as they are. TypeError for anything else
    (a numpy array included)."""
    import datetime
    from decimal import Decimal
    if v is None or isinstance(v, (bool, str, Decimal)):
        return v
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        return float(v)
    if type(v).__module__.split(".")[0] in ("numpy", "pandas"):
        if getattr(v, "shape", ()) != () or not callable(getattr(v, "item", None)):
            raise TypeError(f"{where}: a {type(v).__name__} is not stored as JSON — pass a list (.tolist())")
        return _plain(v.item(), where)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if not isinstance(k, str):
                raise TypeError(f"{where}: the key {k!r} is not a string (the input is stored as JSON)")
            out[k] = _plain(x, f"{where}[{k!r}]")
        return out
    if isinstance(v, (list, tuple)):
        return [_plain(x, f"{where}[{i}]") for i, x in enumerate(v)]
    if isinstance(v, (set, frozenset)):
        try:
            items = sorted(v)
        except TypeError:
            raise TypeError(f"{where}: a set whose values do not sort; pass a list") from None
        return [_plain(x, where) for x in items]
    raise TypeError(f"{where}: a {type(v).__name__} is not stored as JSON — give numbers, strings, dates, Decimals, "
                    "and lists or dicts of them")


def _value_reader(fi):
    """A non-number field → a function: the raw value its part gave → the field's value. A span of a date (or another
    type): span_value; a span of str and a choice: as it is. A value that is not a string (solvi.Unknown for a
    Maybe[...] field) passes as it is."""
    from ..core.types import span_value
    t = fi.annotation
    if _span_source(fi) is None or t is str:
        return lambda raw: raw
    return lambda raw: span_value(t, raw) if isinstance(raw, str) else raw


def _is_number(fi):
    from decimal import Decimal
    return _span_source(fi) is not None and fi.annotation in (int, float, Decimal)


def _span_source(fi):
    from ..core.types import SpanOf
    return next((m.source for m in fi.metadata if isinstance(m, SpanOf)), None)


def _part(name, inputs, body, wrapped=None):
    """A catalog part named `name` reading the facts `inputs` (by name) and calling body({input: value}). `wrapped`:
    the person's function behind it — the catalog's fingerprint hashes its code (inspect.unwrap), so a changed compute=
    or check is a changed catalog."""
    def run(**kw):
        return body(kw)
    run.__name__ = run.__qualname__ = name
    run.__signature__ = inspect.Signature([inspect.Parameter(x, inspect.Parameter.POSITIONAL_OR_KEYWORD) for x in inputs])
    if wrapped is not None:
        run.__wrapped__ = wrapped
    return run


def _stated_confidence(model):
    """An LLM decider made with logprobs="auto" → the same server and settings with logprobs=False, ask="confidence"
    (one stated confidence per read); any other model as it is."""
    from ..core.deciders import DecideModel
    from ..core.deciders.llm import LLMScorer
    sc = getattr(model, "scorer", None)
    if type(sc) is not LLMScorer or sc.logprobs != "auto":
        return model
    new = LLMScorer(sc.url, sc.model, sc._key, timeout=sc.timeout, retries=sc.retries, backoff=sc.backoff,
                    response_format=sc.response_format, logprobs=False, ask="confidence", max_tokens=sc.max_tokens,
                    seed=sc.seed, headers=sc._headers, extra_body=sc.extra_body, workers=sc.workers, opener=sc.opener,
                    sleep=sc.sleep)
    m = DecideModel(new, meta=model.meta, model_id=new.model_id, backend="llm")
    m.deterministic = False
    warnings.warn(f"solvi.document: {sc.model_id} was made with logprobs=\"auto\"; the recipe reads with "
                  "logprobs=False, ask=\"confidence\" (one stated confidence per read) on a copy of it — its usage "
                  "counters are on recipe.model. Make the model with logprobs=True or False to keep your own", UserWarning,
                  stacklevel=3)
    return m


class DocumentSystem:
    """What solvi.document() returns (see the module docs). system: the solvi.System; catalog, schema, model (the one the
    reads go to), text (the text fact's name), fields (field names in order), checks (check names), calibration (the
    last calibration report, or None)."""

    def __init__(self, system, schema, model, text, fields, checks):
        self.system, self.schema, self.model, self.text = system, schema, model, text
        self.fields, self.checks = list(fields), list(checks)
        self.catalog = system.catalog
        self.question = QUESTION
        self.calibration = None

    def given(self, state):
        """The input of ask → the given facts: the text, and the rest as the fact "state"."""
        if hasattr(state, "model_dump"):
            state = state.model_dump()
        if not isinstance(state, dict):
            raise TypeError(f"the input is a dict with the text under {self.text!r}, not {type(state).__name__}")
        if self.text not in state:
            raise KeyError(f"the input has no {self.text!r} (the text the fields are read from); it has "
                           f"{sorted(state)}")
        return {self.text: state[self.text],
                STATE: _plain({k: v for k, v in state.items() if k != self.text})}

    def ask(self, state, **kw):
        """One input → a solvi.Response: `res["answer"]` (answer, status "ok" / "abstain", confidence, why),
        `res.stored_id` when storage= was given. Keyword options go to System.ask."""
        return self.system.ask(self.given(state), [QUESTION], **kw)

    def calibrate(self, examples, *, max_risk=None, max_error=None, correct=None, workers=None, progress=None, **kw):
        """A calibrated threshold on the answer from labelled examples [(input, correct answer)] (System.guarantee:
        max_risk= or max_error=, and groups=, weak=, delta=, method= ...) → the calibration report (also kept as
        `self.calibration`). Fewer than 100 examples: a GuaranteeWarning. A number is judged right at the label's own
        precision (see the module docs) unless correct=(result, label) → bool is given. workers > 1: the examples run
        in a thread pool (results in input order; one broken example counts as escalated). progress: (done, total)
        after every example; when workers > 1 and progress is None and there are 50 or more examples, a heartbeat
        prints to stderr every 10 s."""
        from ..core.guarantees.guarantee import GuaranteeWarning
        examples = [(self.given(st), y) for st, y in examples]
        n = len(examples)
        small = None
        if 0 < n < MIN_CALIBRATION:
            small = (f"calibrating on {n} examples: the threshold rests on them and can move with a few more "
                     f"({MIN_CALIBRATION} or more recommended)")
            warnings.warn(small, GuaranteeWarning, stacklevel=2)
        default = correct is None and self.system.questions[QUESTION].answer.kind == "estimate"
        if default:
            correct = _same_number
        if progress is None and workers is not None and workers > 1 and n >= 50:
            cell = {"t": time.time()}

            def progress(done, total, _cell=cell):          # the heartbeat: a counter + a time check
                now = time.time()
                if done == total or now - _cell["t"] >= 10:
                    print(f"\r[calibrate] {done}/{total} ({now - _cell['t']:.0f} s since the last beat)",
                          end="\n" if done == total else "", file=sys.stderr, flush=True)
                    _cell["t"] = now
        if workers is not None and (not isinstance(workers, int) or isinstance(workers, bool) or workers < 1):
            raise ValueError(f"workers must be a positive int, not {workers!r}")
        try:
            rep = self.system.guarantee(QUESTION, examples, max_risk=max_risk, max_error=max_error, correct=correct,
                                        workers=workers, progress=progress, warn_small=False, **kw)
        except ValueError as e:
            if default:
                raise ValueError(f"{e} — the answers were judged by the default number match (at the label's "
                                 "precision): if right answers count as wrong, pass correct=(result, label) → bool") from e
            raise
        if small:
            rep.setdefault("warnings", []).append(small)
        self.calibration = rep
        return rep

    def save_calibration(self, path):
        """Write the calibration in force to a JSON file (System.save_guarantee) → path."""
        return self.system.save_guarantee(QUESTION, path)

    def load_calibration(self, path, **kw):
        """Attach a calibration written by save_calibration (System.load_guarantee: refuses a file made for another
        catalog — other prompts, compute or checks — or for other models; strict=False attaches it anyway) → its
        report."""
        rep = self.system.load_guarantee(path, **kw)
        self.calibration = rep
        return rep

    @property
    def storage(self):
        return self.system.storage

    def verify(self, **kw):
        """Check the store's hash chain (TraceStorage.verify) → its report; needs storage=."""
        return self._store().verify(**kw)

    def replay_all(self, trust_models=False, **filters):
        """Replay every stored decision against this system (TraceStorage.replay_all) → the ones that fail; needs
        storage=."""
        return self._store().replay_all(self.system, trust_models=trust_models, **filters)

    def _store(self):
        if self.system.storage is None:
            raise ValueError("this system keeps no record: give document(..., storage=\"decisions.jsonl\")")
        return self.system.storage

    def __repr__(self):
        return (f"DocumentSystem({self.schema.__name__}: {', '.join(self.fields)}; checks {self.checks or 'none'}; "
                f"{'calibrated' if QUESTION in self.system.guards else 'not calibrated'})")


def _same_number(r, y):
    """The default judge of a number answer. A label written as a string ("33.33", "1,500", "(2.5)", read like a
    quote) states its precision: d decimals match an answer within 0.5·10^-d ("33.33" takes 33.3333; "1,500" takes
    1500.4) — except a whole number below 100 ("33", "7"), which is a count or a small amount: exact. A label given as
    a number has no written precision: equal up to float noise (math.isclose, 1e-9) — the int 1 does not take 1.4,
    the float 33.33 does not take 33.3333."""
    try:
        if r.answer is None or isinstance(y, bool) or isinstance(r.answer, bool):
            return False
        a = float(r.answer)
        if not math.isfinite(a):
            return False
        if not isinstance(y, str):
            g = float(y)
            return math.isfinite(g) and math.isclose(a, g, rel_tol=1e-9, abs_tol=1e-9)
        g = float(_read_number(float, y))
        core = re.sub(r"[^0-9.,]", "", y)
        m = re.search(r"[.,](\d+)$", core)
        d = len(m.group(1)) if m else 0
        if m and len(m.group(1)) == 3 and "," in core and "." not in core:
            d = 0                                      # "1,500": a thousands separator, not three decimals
        if d == 0 and abs(g) < 100:
            return math.isclose(a, g, rel_tol=1e-9, abs_tol=1e-9)
        return math.isfinite(g) and abs(a - g) <= 0.5 * 10 ** -d + 1e-9 * abs(g)
    except (TypeError, ValueError, OverflowError):
        return False


def document(schema, model, *, task=None, text="doc", compute=None, answer=None, checks=None, units="same",
             storage=None, calibration=None, record="full", workers=1, reads_with=None):
    """A decision over a document (see the module docs) → DocumentSystem.

    schema: a pydantic model class, one field per fact to read (`Span[float]`, `Span[date]`, `Span[str]`, a
    `Literal[...]` choice ...). model: a decider (`solvi.models.llm(url, name)`, `decider("solvi-base")`). task: text
    put before every field's description (what the document is). text: the name of the text fact in the input
    (default "doc"). compute: (facts, state) → the answer. answer: its answer type (default `Answer.estimate()`, a
    number). checks: {name: (facts, state) → bool}, hard checks. units: "same" (default: the answer abstains when the
    amounts write different scales — "$1.2bn" and "$950 million", and "3k" next to "3,000": a field without a scale
    next to one with a scale — or different currencies: "$100" and "€100", "$" and "C$"; a field without a currency
    conflicts with nothing; percentages are not compared), "lenient" (a field without a scale matches any, a bare "$"
    any dollar) or "any" (no check; compute reads `facts.units`). storage: a path (JSONLStorage with `record=`) or a
    TraceStorage. calibration: a file written by save_calibration, attached at once. workers: parallel steps inside
    one ask (System(workers=)). reads_with: state keys every field's read sees — a `<state>` block in the read's
    prompt between the task and the text; the grounding and the quote offsets stay on the text (a span is still read
    from the one text fact); the input must carry each named key, and a recipe with reads_with has its own catalog
    fingerprint (a calibration or a record made without it does not attach)."""
    from ..core.store import JSONLStorage
    from ..core.system import System
    if not (isinstance(schema, type) and isinstance(getattr(schema, "model_fields", None), dict)):
        raise TypeError(f"schema is a pydantic model class (one field per fact to read), not {schema!r}")
    if not schema.model_fields:
        raise ValueError(f"{schema.__name__} has no fields: nothing to read")
    if not isinstance(text, str) or not text:
        raise TypeError("text= is the name of the text fact in the input (default \"doc\")")
    fields = list(schema.model_fields)
    if units not in UNIT_MODES:
        raise ValueError(f'units must be "same", "lenient" or "any", not {units!r}')
    for bad in (text, STATE, QUESTION, "units", "same_units", "compute"):
        if bad in fields:
            raise ValueError(f"{schema.__name__}.{bad}: a field cannot be named {bad!r} (the "
                             + {text: "text fact", STATE: "input fact", QUESTION: "question", "units": "facts' units",
                                "same_units": "units check", "compute": "rule"}[bad] + ")")
    made = {f"{x}_value" for x in fields}
    if made & set(fields):
        x = sorted(made & set(fields))[0]
        raise ValueError(f"{schema.__name__}.{x}: a field cannot be named {x!r} — the recipe names the value it reads "
                         f"from field {x[:-6]!r} so")
    if text == STATE:
        raise ValueError(f"text= cannot be {STATE!r}: that fact carries the rest of the input")
    reads_with = tuple(reads_with or ())
    for k in reads_with:
        if not (isinstance(k, str) and k.isidentifier()):
            raise TypeError(f"reads_with={reads_with!r}: every state key is a Python identifier, not {k!r}")
        if k == text:
            raise ValueError(f"reads_with={reads_with!r}: {k!r} is the text fact itself")
    checks = dict(checks or {})
    for name, f in checks.items():
        if not isinstance(name, str) or not name.isidentifier() or name in fields or name in made \
                or name in (text, STATE, QUESTION, "compute", "units", "same_units"):
            raise ValueError(f"check name {name!r}: a Python identifier other than a field, <field>_value, compute, "
                             f"units, same_units, {text!r}, {STATE!r} or {QUESTION!r}")
        if not callable(f):
            raise TypeError(f"check {name!r} is a function (facts, state) → bool, not {type(f).__name__}")
    if compute is not None and not callable(compute):
        raise TypeError(f"compute= is a function (facts, state) → the answer, not {type(compute).__name__}")
    if compute is None and len(fields) != 1:
        raise ValueError(f"{schema.__name__} has {len(fields)} fields: give compute=(facts, state) → the answer "
                         "(without it, a schema of one field answers with that field's value)")
    from decimal import Decimal

    from ..core.types import Bins, Ordinal, RankOf, SpanOf, closed_values
    for name, fi in schema.model_fields.items():           # a plain float / str field is not read from the text
        if _span_source(fi) is None and fi.annotation is not bool and closed_values(fi.annotation) is None:
            t = _type_name(fi)
            raise TypeError(f"{schema.__name__}.{name}: {t} is not a fact read from the text — write Span[{t}] (a "
                            "value quoted from it), or a Literal[...] / Enum / bool for a choice")
    if compute is None and answer is None:
        fi = schema.model_fields[fields[0]]
        if _span_source(fi) is not None:
            if fi.annotation not in (int, float, Decimal):
                raise TypeError(f"{schema.__name__}.{fields[0]} is a {_type_name(fi)}: without compute= the answer "
                                "is a number or a choice — give compute=(facts, state) → the answer and its answer=")
        else:
            answer = Answer.from_type(fi.annotation)
    model = _stated_confidence(model)
    cat = Catalog()
    readers = {}
    for name, fi in schema.model_fields.items():
        src = _span_source(fi)
        if src is not None and src != text:
            raise ValueError(f"{schema.__name__}.{name} is a span of {src!r}, but the text is {text!r}: write "
                             f"Span[..., {text!r}] or document(..., text={src!r})")
        t = fi.annotation
        marks = [m for m in fi.metadata if isinstance(m, (Ordinal, SpanOf, RankOf, Bins))]
        if marks:                                      # pydantic moves Annotated metadata to the field: put solvi's back
            t = typing.Annotated[(t, *marks)]
        extra = fi.json_schema_extra if isinstance(fi.json_schema_extra, dict) else {}
        desc = fi.description or fi.title or name.replace("_", " ").capitalize() + "?"
        opts = extra.get("options") or ()
        if isinstance(opts, dict):                     # the plain option list; the descriptions go into the question
            desc = desc.rstrip() + " The options mean: " + "; ".join(f"{o} — {d}" if d else str(o) for o, d in opts.items()) + "."
            opts = () if closed_values(fi.annotation) is not None else list(opts)   # a Literal / Enum: from the type
        args = {k: extra[k] for k in ("min_confidence", "min_act", "use_act", "max_error", "other", "score_value",
                                      "kind", "evidence", "coverage", "unit") if k in extra}
        q = f"{task.rstrip()} {desc}" if task else desc
        cat.fn(model.decision(name, q, text, options=opts, type=t, state_keys=reads_with, **args))
        readers[name] = _value_reader(fi)

    from decimal import Decimal as _Dec
    numbers = [x for x, fi in schema.model_fields.items() if _is_number(fi)]
    for name in numbers:                                   # a number field's value: read, its sign checked in the text
        vt = schema.model_fields[name].annotation

        def value(kw, name=name, vt=vt):
            raw = kw[name]
            if not isinstance(raw, str):
                return raw
            try:
                v = _read_quote(vt, raw)[0]
            except ValueError as e:
                raise ValueError(f"{name}: {raw!r} does not read as {vt.__name__}: {e}") from None
            why = _left_out(raw, kw[text])
            if why is not None:
                raise ValueError(f"{name}: {why}")
            return v
        cat.fn(_part(f"{name}_value", [name, text], value))

    def finite(v):
        import numbers as _numbers
        if isinstance(v, (_numbers.Number, _Dec)) and not isinstance(v, bool):
            try:
                ok = math.isfinite(v)
            except (TypeError, ValueError, OverflowError):
                ok = True                              # a complex or a huge int: judged by the answer type
            if not ok:
                raise ValueError(f"the computed answer is {v!r}, not a finite number")
        return v

    def units_fact(kw):
        """{field: [scale, currency, percent]}: what each number's quote writes, and the currency or scale its text
        token writes where the quote left them out."""
        out = {}
        for x in numbers:
            raw = kw[x]
            try:
                u = _read_quote(schema.model_fields[x].annotation, raw)[1]
            except (ValueError, TypeError):            # unreadable: its value part says why
                u = Unit()
            why, got = _ground(raw, kw[text])
            if why is None:
                u = u._replace(scale=u.scale or got.get("scale"), currency=u.currency or got.get("currency"),
                               percent=u.percent or bool(got.get("percent")))
            out[x] = [u.scale, u.currency, u.percent]
        return out

    def units_of(kw):
        return {x: Unit(*v) for x, v in kw["units"].items()} if numbers else {}

    from ..core.slow.refine import Fail

    def same_units(us):
        us = {x: u for x, u in us.items() if not u.percent}   # a percentage is no amount: neither scale nor currency
        scales = {x: u.scale for x, u in us.items()}
        said = {x: v for x, v in scales.items() if v is not None}
        if len(set(said.values())) > 1 or (units == "same" and said and len(said) < len(scales)):
            shown = ", ".join(f"{x} in {v or 'no scale'}" for x, v in scales.items())
            raise ValueError(f"units differ: {shown} — the scales of the facts do not match (units=\"lenient\": a "
                             "fact without a scale matches any; units=\"any\": no check; compute reads facts.units)")
        curs = {x: u.currency for x, u in us.items() if u.currency is not None}
        vals = list(curs.values())
        agree = _same_currency if units == "lenient" else (lambda v, w: v == w)
        if any(not agree(v, w) for v in vals for w in vals):
            raise ValueError(f"units differ: {', '.join(f'{x} in {v}' for x, v in curs.items())} — the currencies "
                             'of the facts do not match (units="lenient": a bare "$" matches any dollar; units="any": '
                             "no check; compute reads facts.units)")

    def facts_of(kw):
        vals = {}
        for name in fields:
            if name in numbers:
                vals[name] = kw[f"{name}_value"]
                continue
            raw = kw[name]
            try:
                vals[name] = readers[name](raw)
            except ValueError as e:
                raise ValueError(f"{name}: {raw!r} does not read as {_type_name(schema.model_fields[name])}: {e}") from None
        f = schema.model_construct(**vals)
        object.__setattr__(f, "units", units_of(kw))
        return f

    if numbers:
        cat.fn(_part("units", list(numbers) + [text], units_fact))
    inputs = [f"{x}_value" for x in numbers] + fields + (["units"] if numbers else []) + [STATE]   # why: value, quote
    for name, f in checks.items():
        cat.check(_part(name, inputs, lambda kw, f=f: f(facts_of(kw), kw[STATE]), wrapped=f), hard=True)
    required = []
    if units in ("same", "lenient") and len(numbers) > 1:              # two number fields: their written units must agree

        def same(kw):
            try:
                same_units(units_of(kw))
            except ValueError as e:
                return Fail(str(e))
            return True
        cat.check(_part("same_units", ["units"], same), hard=True)
        required.append("same_units")
    if compute is None:
        only = fields[0]
        if only in numbers:
            rule = _part("compute", [f"{only}_value", only], lambda kw: finite(kw[f"{only}_value"]))
        else:
            rule = _part("compute", [only], lambda kw: readers[only](kw[only]))
    else:
        rule = _part("compute", inputs, lambda kw: finite(compute(facts_of(kw), kw[STATE])), wrapped=compute)
    cat.rule(QUESTION)(rule)
    if answer is None:
        answer = Answer.estimate()
    q = Question(QUESTION, task or f"{schema.__name__} → the computed answer", answer, requires=required + list(checks))
    if isinstance(storage, str) or hasattr(storage, "__fspath__"):
        storage = JSONLStorage(storage, record=record)
    s = System(cat, [q], storage=storage, workers=workers)
    out = DocumentSystem(s, schema, model, text, fields, checks)
    if calibration is not None:
        out.load_calibration(calibration)
    return out


def _type_name(fi):
    from ..core.types import type_name
    return type_name(fi.annotation)


__all__ = ["DocumentSystem", "document"]
