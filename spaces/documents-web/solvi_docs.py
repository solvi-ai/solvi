"""Glue between the page and solvi, running in Pyodide (in your browser).

The page extracts every field with the ONNX model first and passes {field: [value, start, end, score] or absent} here.
Each field becomes a catalog part (@cat.extract) that returns a Quote with exact character offsets, the use case's rule code
(plain Python with @cat.fn / @cat.check / @cat.rule and a QUESTIONS list) is added on top, and System(cat, QUESTIONS).ask()
decides. The trace is replayed after every run.

Helpers available to rule code: money, number, parse_date, days, hours, found, need, mentions, iban_ok."""
from __future__ import annotations

import copy
import json
import keyword
import math
import re
import traceback
from datetime import date, datetime, timedelta

from solvi import Answer, Catalog, Question, Quote, System
from solvi.runtime import MISSING, vhash

# ------------------------------------------------------------------------------------------------------------ helpers
NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
                "eighteen": 18, "twenty": 20, "twenty-one": 21, "twenty-four": 24, "thirty": 30, "thirty-six": 36,
                "forty": 40, "forty-five": 45, "forty-eight": 48, "sixty": 60, "seventy-two": 72, "ninety": 90,
                "hundred": 100, "one hundred twenty": 120, "one hundred eighty": 180, "a": 1, "an": 1}
MONTHS = {m: i + 1 for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])}
MONTHS.update({"mei": 5, "agu": 8, "okt": 10, "des": 12, "mär": 3, "mrz": 3, "dez": 12})
UNIT_DAYS = {"hour": 1 / 24, "day": 1, "business day": 7 / 5, "week": 7, "month": 30, "year": 365}


def found(value) -> bool:
    """Was the field found? (an absent field is the empty string)"""
    return bool(value) and bool(str(value).strip())


def need(value, what="the field"):
    """Return value, or raise (→ the question abstains) when the field is absent."""
    if not found(value):
        raise ValueError(f"{what} was not found in the document")
    return value


def mentions(text, *words) -> bool:
    """Does the text mention any of the words or phrases (case-insensitive, whole words)?"""
    low = str(text).lower()
    return any(re.search(r"(?<!\w)" + re.escape(w.lower()) + r"(?!\w)", low) for w in words)


def number(text) -> float:
    """First number in the text: "sixty (60) days" -> 60, "three months" -> 3, "1,250.00" -> 1250."""
    need(text, "the value")
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(text))
    if m:
        return float(m.group(0).replace(",", ""))
    low = str(text).lower()
    for w in sorted(NUMBER_WORDS, key=len, reverse=True):
        if len(w) > 2 and re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", low):
            return float(NUMBER_WORDS[w])
    raise ValueError(f"no number in {text!r}")


def money(text) -> float:
    """The last amount in the text: "RM 66.20" -> 66.2, "4 104,00 €" -> 4104, "$ 4,850.00" -> 4850, "€ 12.480,00" -> 12480, "108,900" -> 108900."""
    need(text, "the amount")
    s = re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}(?!\d))", "", str(text))   # "4 104,00 €": space as thousands separator
    toks = re.findall(r"\d[\d.,']*", s)
    if not toks:
        raise ValueError(f"no amount in {text!r}")
    t = toks[-1].rstrip(".,").replace("'", "")
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", t):                # 108,900 / 74.500: thousands separators only
        return float(re.sub(r"[.,]", "", t))
    if "," in t and "." in t:                                     # 1,234.50 / 1.234,50: the last separator is the decimal one
        k = max(t.rfind(","), t.rfind("."))
        return float(re.sub(r"[.,]", "", t[:k]) + "." + t[k + 1:])
    return float(t.replace(",", "."))


def parse_date(text, dayfirst=True) -> date:
    """Dates as written in documents: 2026-03-15, 24/02/2018 (day first unless dayfirst=False), March 3, 2021,
    3rd of March 2021, 03 Nov 2017, 15.10.2026."""
    need(text, "the date")
    s = str(text)
    if m := re.search(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s):
        y, mo, d = (int(x) for x in m.groups())
    elif m := re.search(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{2,4})", s):
        a, b, y = (int(x) for x in m.groups())
        d, mo = (a, b) if dayfirst else (b, a)
    elif (m := re.search(r"([A-Za-zä]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)) and m.group(1)[:3].lower() in MONTHS:
        mo, d, y = MONTHS[m.group(1)[:3].lower()], int(m.group(2)), int(m.group(3))
    elif (m := re.search(r"(\d{1,2})(?:st|nd|rd|th)?(?:\s+day)?(?:\s+of)?[\s\-/.]*([A-Za-zä]{3,9})\.?[\s\-/.,]*(\d{2,4})", s)) \
            and m.group(2)[:3].lower() in MONTHS:
        d, mo, y = int(m.group(1)), MONTHS[m.group(2)[:3].lower()], int(m.group(3))
    else:
        raise ValueError(f"cannot read a date from {text!r}")
    return date(y if y > 99 else 2000 + y, mo, d)


def _duration(text):
    low = str(need(text, "the period")).lower().replace("-", " ")
    unit_re = r"(?:calendar\s+)?(business\s+day|working\s+day|hour|day|week|month|year)s?"
    m = re.search(r"(\d+(?:\.\d+)?)\s*\)?\s*" + unit_re, low)
    if m:
        n, unit = float(m.group(1)), m.group(2)
    else:
        spaced = {w.replace("-", " "): v for w, v in NUMBER_WORDS.items()}
        words = "|".join(re.escape(w) for w in sorted(spaced, key=len, reverse=True))
        m = re.search(r"(?<!\w)(" + words + r")\s+" + unit_re, low)
        if not m:
            raise ValueError(f"no period (days, weeks, months, years, hours) in {text!r}")
        n, unit = float(spaced[m.group(1)]), m.group(2)
    unit = re.sub(r"\s+", " ", unit)
    unit = "business day" if unit in ("business day", "working day") else unit
    return n, unit


def days(text) -> float:
    """A period in days: "sixty (60) days" -> 60, "three (3) months" -> 90, "two weeks" -> 14, "72 hours" -> 3."""
    n, unit = _duration(text)
    return round(n * UNIT_DAYS[unit], 2)


def hours(text) -> float:
    """A period in hours: "within 48 hours" -> 48, "without undue delay and within five (5) business days" -> 168."""
    n, unit = _duration(text)
    return round(n * UNIT_DAYS[unit] * 24, 2)


def iban_ok(text) -> bool:
    """IBAN checksum (ISO 13616 mod-97) of the first IBAN-looking string in the text."""
    s = re.sub(r"[\s-]", "", str(need(text, "the IBAN"))).upper()
    m = re.search(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", s)
    if not m:
        raise ValueError(f"no IBAN in {text!r}")
    iban = m.group(0)
    digits = "".join(str(int(c, 36)) for c in iban[4:] + iban[:4])
    return int(digits) % 97 == 1


HELPERS = dict(found=found, need=need, mentions=mentions, number=number, money=money, parse_date=parse_date, days=days,
               hours=hours, iban_ok=iban_ok)


# ------------------------------------------------------------------------------------------------------------- running
_last = {}


def _extractor(name, desc, doc, hit):
    """A catalog part that returns the precomputed extraction as a Quote (offsets are Python string indices)."""
    if hit and hit.get("present"):
        s, e, sc = int(hit["start"]), int(hit["end"]), float(hit["score"])
        def f(doc):
            return Quote(doc[s:e], s, e, confidence=sc)
    else:
        sc = float(hit["score"]) if hit else 0.0
        def f(doc):
            return Quote("", 0, 0, confidence=1 - sc)
    f.__name__ = name
    f.__doc__ = desc
    return f


def _presence_rule(field):
    ns = {}
    exec(f"def {field}_found({field}):\n    return found({field})\n", {"found": found}, ns)
    return ns[f"{field}_found"]


def _r(v, n=160):
    if v is MISSING:
        return None
    s = repr(v)
    return s if len(s) <= n else s[:n - 3] + "..."


def run(payload_json):
    p = json.loads(payload_json)
    doc = p["doc"]
    try:
        today = date.fromisoformat(p.get("today") or date.today().isoformat())
    except ValueError:
        today = date.today()
    cat = Catalog()
    fields = [f for f in p["fields"] if f.get("name")]
    for f in fields:
        cat.extract(_extractor(f["name"], f.get("desc", ""), doc, f.get("hit")))
    ns = dict(HELPERS, cat=cat, Answer=Answer, Question=Question, Quote=Quote, date=date, datetime=datetime,
              timedelta=timedelta, re=re, math=math, QUESTIONS=[], INPUTS={})
    try:
        exec(compile(p.get("code") or "", "<rules>", "exec"), ns)
    except SyntaxError as e:
        return json.dumps({"error": f"SyntaxError in the rule code, line {e.lineno}: {e.msg}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"{type(e).__name__} in the rule code: {e}", "traceback": traceback.format_exc(limit=3)})
    questions = list(ns.get("QUESTIONS") or [])
    # every field no part reads gets a "found?" question, so it still goes through solvi (quote, trace, replay)
    used = {x for part in list(cat.parts.values()) + list(cat.rules.values()) for x in part.inputs}
    auto = []
    for f in fields:
        n = f["name"]
        if n not in used and f"{n}_found" not in cat.parts and f"{n}_found" not in {q.name for q in questions}:
            cat.rule(f"{n}_found")(_presence_rule(n))
            questions.append(Question(f"{n}_found", f"Is {n.replace('_', ' ')} in the document?", Answer.yes_no()))
            auto.append(f"{n}_found")
    if not questions:
        return json.dumps({"error": "No questions: define QUESTIONS in the rule code, or add a field."})
    init = {"doc": doc, "today": today}
    init.update(ns.get("INPUTS") or {})
    try:
        system = System(cat, questions)
        res = system.ask(init)
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=4)})
    rep = res.trace.replay(system)                            # with the System, replay also checks the stored answers
    _last.update(cat=cat, res=res, system=system)
    by = {r.name: r for r in res.trace.records}
    kinds = {n: part.kind for n, part in cat.parts.items()}
    answers = []
    for q in questions:
        r = res[q.name]
        facts = res.flow.per_question.get(q.name, [])
        cites = [f for f in facts if kinds.get(f) == "extract"]
        checks = [{"name": f, "hard": cat.parts[f].hard, "value": _r(by[f].value) if f in by else None,
                   "doc": cat.parts[f].doc} for f in facts if kinds.get(f) == "check"]
        answers.append({"name": q.name, "text": q.text, "answer": r.answer, "confidence": r.confidence, "status": r.status,
                        "why": r.why, "options": list(q.answer.options), "cites": cites, "checks": checks,
                        "auto": q.name in auto})
    records = [{"step": r.step, "kind": r.kind, "name": r.name, "value": _r(r.value), "quote": list(r.quote[:2]) if r.quote else None,
                "confidence": r.confidence, "error": r.error, "hash": r.hash, "prev": r.prev} for r in res.trace.records]
    flow = [{"i": i, "kind": s.part.kind, "name": s.part.name, "inputs": list(s.part.inputs), "reasons": list(s.reasons),
             "hard": bool(getattr(s.part, "hard", False)), "doc": s.part.doc}
            for i, s in enumerate(res.flow.steps, 1)]
    return json.dumps({
        "answers": answers, "records": records, "flow": flow, "flow_text": str(res.flow),
        "not_taken": sorted(res.flow.skipped.items()), "skipped_at_run": [list(x) for x in res.trace.skipped],
        "computed_state": res.state_text(), "init_hash": res.trace.init_hash,
        "replay": {"ok": rep["ok"], "steps": rep["steps"], "mismatches": [[str(x) for x in m] for m in rep["mismatches"]]},
        "ms": res.ms, "n_parts": len(cat.parts) + len(cat.rules), "auto": auto,
    }, default=str)


def _tampered(v):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v * 10 if v else 1
    if isinstance(v, date):
        return v + timedelta(days=365)
    if isinstance(v, str):
        return re.sub(r"\d", lambda m: str((int(m.group(0)) + 1) % 10), v) if re.search(r"\d", v) else v + " (edited)"
    return None


def tamper():
    """Simulate a careful forger: change one computed fact in a copy of the trace and re-hash that record, then replay."""
    if not _last:
        return json.dumps({"error": "run the use case first"})
    res, system = _last["res"], _last["system"]
    tr = copy.deepcopy(res.trace)
    target = None
    for kind in ("fn", "extract", "check"):
        for r in tr.records:
            if r.kind == kind and r.value is not MISSING and r.error is None and _tampered(r.value) is not None \
                    and _tampered(r.value) != r.value:
                target = r
                break
        if target:
            break
    if target is None:
        return json.dumps({"error": "nothing to tamper with in this trace"})
    old = target.value
    target.value = _tampered(old)
    target.hash = vhash(target.body())                        # re-hash the edited record so it looks self-consistent
    rep = tr.replay(system)
    return json.dumps({"name": target.name, "step": target.step, "from": _r(old), "to": _r(target.value),
                       "replay": {"ok": rep["ok"], "steps": rep["steps"],
                                  "mismatches": [[str(x) for x in m] for m in rep["mismatches"]]}}, default=str)


def check_code(code):
    """Syntax check for the editor (no execution)."""
    try:
        compile(code, "<rules>", "exec")
        return ""
    except SyntaxError as e:
        return f"line {e.lineno}: {e.msg}"


def valid_name(name):
    return name.isidentifier() and not keyword.iskeyword(name) and name not in ("doc", "today")
