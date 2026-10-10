"""PREREG X3: verified charts on real texts — `solvi.experimental.charts` on 50 paragraphs of US federal news releases
(benchmarks/datasets/charts_texts), against a hand reading of every number written before any proposer saw the texts.

    python benchmarks/charts_real_texts.py --dry                 # the 3 gallery texts of tests/test_charts.py
    python benchmarks/charts_real_texts.py --run --out out/x3    # the measured run on the 50 texts (one run)

For each text `chart(text)` with the rule-based proposer (the LLM proposer only as a report arm: --llm MODEL, through
the shared proxy), then:

  X3-1  every rendered value re-verified against the text by this script's own reader: each verified value is the
        number written at its place (the text there is what the checker says, and its quantity is the value × scale),
        and every number in the SVG's text (labels, values, title, description; the footer's count excluded) states a
        quantity the text states or is written in the text as is;
  X3-2  each drawn point against the hand reading (labels.json): the label carries the reading's keywords, the unit
        and scale give the reading's quantity; the sign is reported apart;
  X3-3  drawn or refused, each refusal with its reason (invalid proposal, no value proposed, or the commonest reason a
        proposed value was dropped);
  X3-4  solvi.testing.corrupt on the runs' recorded proposals — change_digit and swap of point values, drop of a point —
        and, reported beside it, on the proposals reduced to their verified points;
  X3-5  each run's record replays from itself (re-parsed, re-checked, re-rendered without calling the proposer:
        identical issues and bytes), with the rule proposer a second run gives identical bytes (a model proposer is
        not re-called: a remote LLM at temperature 0 does not repeat itself), and
        solvi.testing.conformance.check_specialist passes on the texts.

    python benchmarks/charts_real_texts.py --replay out/x3_llm/runs.jsonl    # X3-5's replay alone, from stored runs

Until 1.1 the second run of X3-5 used the rule proposer for every arm, so the LLM arm's "same bytes" compared a rule
chart with the model's (0/50); the replay of its records was 50/50.

Results are saved per text as they come (out/runs.jsonl; a rerun with the same --out resumes), then out/report.json and
out/report.md. Every read of texts.jsonl is logged (benchmarks/datasets/charts_texts/reads.log)."""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from decimal import Decimal
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "benchmarks" / "datasets" / "charts_texts"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.charts import ChartSpec, ChartSpecialist, LLMProposer, RuleProposer, canon_unit
    from solvi.experimental.charts.spec import SCALES
    from solvi.experimental.specialist import BLOCKED, DROPPED
from solvi.testing.conformance import ConformanceError, check_specialist  # noqa: E402
from solvi.testing.corrupt import change_digit, corrupt, drop, swap  # noqa: E402

VALUE = r"^/series/\d+/points/\d+/value$"
POINT = r"^/series/\d+/points/\d+$"
PROXY = "http://127.0.0.1:8765/x3/{arm}/v1"

# ------------------------------------------------------------------------------------------ units of the hand reading
# base unit -> {chart unit (canonical): chart units per base unit}; labels.json carries its own copy (used when present)
UNITS = {
    "%": {"%": 1}, "pp": {"pp": 1}, "USD": {"USD": 1}, "USD billion": {"USD": 10**9}, "USD trillion": {"USD": 10**12},
    "USD million": {"USD": 10**6}, "USD per barrel": {"USD": 1}, "USD per gallon": {"USD": 1}, "USD per person": {"USD": 1},
    "billion kWh": {"kilowatthour": 10**9, "kwh": 10**9, "bkwh": 1},
    "Bcf/d": {"bcf/d": 1, "bcf": 1, "cubic feet per day": 10**9, "cubic foot per day": 10**9, "cubic feet": 10**9,
              "cubic foot": 10**9},
    "b/d": {"b/d": 1, "barrel": 1, "barrels per day": 1, "barrel per day": 1},
    "million b/d": {"b/d": 10**6, "barrel": 10**6, "barrels per day": 10**6, "barrel per day": 10**6},
    "million barrels": {"barrel": 10**6}, "holes": {"hole": 1, "": 1}, "million feet": {"feet": 10**6, "foot": 10**6},
    "counties": {"county": 1, "countie": 1, "": 1}, "states": {"state": 1, "": 1},
    "storms": {"storm": 1, "tropical storm": 1, "": 1}, "inches": {"inch": 1, "inche": 1, "in": 1},
    "square miles": {"square mile": 1, "sq mi": 1}, "million square miles": {"square mile": 10**6, "sq mi": 10**6},
    "billion pounds": {"pound": 10**9}, "pounds per cow": {"pound": 1}, "cows": {"cow": 1, "": 1},
    "herds": {"herd": 1, "": 1}, "billion bushels": {"bushel": 10**9},
    "million people": {"people": 10**6, "person": 10**6, "": 10**6}, "million workers": {"worker": 10**6, "": 10**6},
    "billion acres": {"acre": 10**9}, "awards": {"award": 1, "": 1}, "count": {"": 1}, "platforms": {"platform": 1, "": 1},
    "tonnes": {"tonne": 1, "ton": 1}, "employees": {"employee": 1, "": 1}, "stores": {"store": 1, "": 1},
    "°F": {"°f": 1, "f": 1},
    "year": {}, "date": {}, "reference": {}, "rank": {}, "threshold": {}, "name": {}, "ratio": {}, "age": {},
}


# ------------------------------------------------------------------------------------------ the gallery (development)
def _gallery():
    sys.path.insert(0, str(ROOT / "tests"))
    try:
        from test_charts import PRESS, QUARTERS, WASTE
    finally:
        sys.path.pop(0)
    return {"press": PRESS, "quarters": QUARTERS, "waste": WASTE}


GALLERY = _gallery()


def _v(aw, value, unit, of, kw="", no="", occ=0):
    return {"as_written": aw, "occurrence": occ, "value": str(value), "unit": unit, "role": "value", "of": of,
            "label_keywords": kw, "label_excludes": no}


def _o(aw, unit, of, occ=0):
    return {"as_written": aw, "occurrence": occ, "value": None, "unit": unit, "role": unit, "of": of,
            "label_keywords": "", "label_excludes": ""}


def locate(text, numbers):
    """Each labelled number's place: its `occurrence`-th whole match of `as_written` (not inside a longer number) →
    the numbers with `start` / `end`; ValueError when it is not there."""
    out = []
    for n in numbers:
        aw, occ = n["as_written"], n.get("occurrence", 0)
        pre = r"(?<![\w.,])" if aw[0].isalnum() else ""
        post = r"(?![\w]|[.,]\d)" if aw[-1].isalnum() else ""
        hits = [m.start() for m in re.finditer(pre + re.escape(aw) + post, text)]
        if occ >= len(hits):
            raise ValueError(f"{aw!r}: occurrence {occ} not in the text ({len(hits)} found)")
        out.append({**n, "start": hits[occ], "end": hits[occ] + len(aw)})
    return out


GALLERY_LABELS = {
    "press": {"numbers": locate(GALLERY["press"], [
        _o("2025", "year", "year", 0), _o("14", "date", "14 October"), _o("2025", "year", "year", 1),
        _v("$4.2 billion", "4.2", "USD billion", "revenue, third quarter", "revenue"),
        _v("12%", "12", "%", "revenue growth from a year earlier", "up|growth|revenue", occ=0),
        _v("42%", "42", "%", "Europe's share of revenue", "Europe"),
        _v("35%", "35", "%", "North America's share", "North America"),
        _v("23%", "23", "%", "Asia-Pacific's share", "Asia"),
        _v("3 percentage points", "3", "pp", "change in operating margin", "margin"),
        _v("18%", "18", "%", "operating margin", "margin"),
        _v("15%", "15", "%", "headcount growth", "headcount"),
        _v("480", "480", "employees", "headcount", "headcount"),
        _v("12", "12", "stores", "new stores", "store", occ=0)]),
        "chart": {"kind": "pie", "unit": "%", "scale": "", "series": [{"name": "revenue by region", "points": [
            ["Europe", "42"], ["North America", "35"], ["Asia-Pacific", "23"]]}]}},
    "quarters": {"numbers": locate(GALLERY["quarters"], [
        _o("2025", "year", "fiscal 2025", 0), _o("Q1", "date", "quarter"), _o("2025", "year", "year", 1),
        _v("$1.2 million", "1.2", "USD million", "revenue, Q1 2025", "Q1"), _o("Q2", "date", "quarter"),
        _o("2025", "year", "year", 2), _v("$1.5 million", "1.5", "USD million", "revenue, Q2 2025", "Q2"),
        _o("Q3", "date", "quarter"), _o("2025", "year", "year", 3),
        _v("$1.9 million", "1.9", "USD million", "revenue, Q3 2025", "Q3"), _o("Q4", "date", "quarter"),
        _o("2025", "year", "year", 4), _v("$2.3 million", "2.3", "USD million", "revenue, Q4 2025", "Q4"),
        _v("$6.9 million", "6.9", "USD million", "total revenue for the year", "total|year")]),
        "chart": {"kind": "line", "unit": "USD", "scale": "million", "series": [{"name": "revenue", "points": [
            ["Q1 2025", "1.2"], ["Q2 2025", "1.5"], ["Q3 2025", "1.9"], ["Q4 2025", "2.3"]]}]}},
    "waste": {"numbers": locate(GALLERY["waste"], [
        _o("2025", "year", "year"),
        _v("1,200", "1200", "tonnes", "waste recycled", "recycl"),
        _v("800", "800", "tonnes", "waste composted", "compost"),
        _v("2,500", "2500", "tonnes", "waste sent to landfill", "landfill"),
        _v("4,500", "4500", "tonnes", "total waste", "total")]),
        "chart": {"kind": "pie", "unit": "tonne", "scale": "", "series": [{"name": "waste, 2025", "points": [
            ["recycled", "1200"], ["composted", "800"], ["landfill", "2500"]]}]}},
}


# ------------------------------------------------------------------------------------------ the logged loader
def load_texts(path, mode, log=None):
    """texts.jsonl → [record]; every read is appended to the read log (time, mode, process, file hash, command)."""
    path = Path(path)
    raw = path.read_bytes()
    log = Path(log) if log is not None else path.parent / "reads.log"
    with open(log, "a") as f:
        f.write(f"{_dt.datetime.now().isoformat(timespec='seconds')}\tmode={mode}\tpid={os.getpid()}\t"
                f"sha256={hashlib.sha256(raw).hexdigest()}\targv={' '.join(sys.argv)}\n")
    return [json.loads(line) for line in raw.decode().splitlines() if line.strip()]


# ------------------------------------------------------------------------------------------ X3-1: this script's reader
_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                                     "fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
               "hundred": 100})
_SCALE_WORDS = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_SCALE_SUFFIX = {"k": 10**3, "m": 10**6, "bn": 10**9, "tn": 10**12}
_NUM = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?")


def parse_quantity(s):
    """A number as written ("$4.2 billion", "1,200 tonnes", "42%", "eight", "$1.2m") → its magnitude (Decimal, the sign
    left out), or None. Independent of the library's reader: digits with thousands commas and a decimal point, a scale
    word after it or a scale letter attached to it, number words."""
    s = (s or "").strip()
    m = _NUM.search(s)
    if m is None:
        w = re.match(r"[^\w]*([A-Za-z]+)", s)
        if w and w.group(1).lower() in _WORDS:
            v = Decimal(_WORDS[w.group(1).lower()])
            rest = s[w.end():]
        else:
            return None
    else:
        v = Decimal(m.group(1).replace(",", "") + (m.group(2) or ""))
        rest = s[m.end():]
    suf = re.match(r"(bn|tn|k|m)(?![A-Za-z])", rest)
    if suf:
        return v * _SCALE_SUFFIX[suf.group(1)]
    word = re.match(r"\s+([A-Za-z]+)", rest)
    if word and word.group(1).lower() in _SCALE_WORDS:
        return v * _SCALE_WORDS[word.group(1).lower()]
    return v


def source_quantities(text):
    """The quantities a text states: every number with the scale word after it, and without it; and the digit forms
    (commas dropped) of every number."""
    qs, forms = set(), set()
    for m in re.finditer(r"(?<![\w.,])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?", text):
        digits = m.group(1).replace(",", "") + (m.group(2) or "")
        forms.add(digits)
        qs.add(Decimal(digits))
        q = parse_quantity(text[m.start():m.end() + 12])
        if q is not None:
            qs.add(q)
    for m in re.finditer(r"(?<![\w])([A-Za-z]+)(?![\w])", text):
        if m.group(1).lower() in _WORDS:
            q = parse_quantity(text[m.start():m.end() + 12])
            if q is not None:
                qs.add(q)
    return qs, forms


_TOKEN = re.compile(r"[-−]?[$€£₽¥]?(?<![\w.])\d[\d,]*(?:\.\d+)?(?:(?:bn|tn|k|m)(?![A-Za-z]))?%?")
_META = [re.compile(r"\d+ values? quoted from the source.*$"), re.compile(r"\bSeries \d+\b")]


def _svg_texts(svg):
    root = ET.fromstring(svg)
    out = []
    for el in root.iter():
        tag = el.tag.split("}")[-1]
        if tag in ("text", "title", "desc", "tspan") and el.text:
            out.append(el.text)
    return out


def svg_numbers(svg):
    """The numbers written in an SVG's text (labels, values, title, description; not coordinates), the library's own
    footer count and default series names left out."""
    toks = []
    for t in _svg_texts(svg):
        for rx in _META:
            t = rx.sub(" ", t)
        toks += [m.group().strip() for m in _TOKEN.finditer(t)]
    return toks


def _points(verified):
    for s in verified.series:
        for p in s.points:
            if p is not None:
                yield s, p


def reverify(run, text):
    """X3-1 for one run → {"values", "values_ok", "svg_numbers", "svg_numbers_ok", "problems"}."""
    res = {"values": 0, "values_ok": 0, "svg_numbers": 0, "svg_numbers_ok": 0, "problems": []}
    v = run.checked.verified if run.checked is not None else None
    if run.output is None or v is None:
        return res
    mult = Decimal(SCALES[v.scale])
    pts = [p for _, p in _points(v)] + ([v.total] if v.total is not None else [])
    for p in pts:
        res["values"] += 1
        at = text[p.start:p.end]
        q = parse_quantity(at)
        if at != p.as_written:
            res["problems"].append(f"{p.label!r}: the text at {p.start}–{p.end} is {at!r}, not {p.as_written!r}")
        elif q is None or q != abs(p.value) * mult:
            res["problems"].append(f"{p.label!r} = {p.value} × {v.scale or 1}: the text at {p.start} states {at!r} "
                                   f"({q})")
        else:
            res["values_ok"] += 1
    qs, forms = source_quantities(text)
    for tok in svg_numbers(run.output):
        res["svg_numbers"] += 1
        q = parse_quantity(tok)
        plain = re.sub(r"[^\d.]", "", tok).strip(".")
        if (q is not None and q in qs) or plain in forms:
            res["svg_numbers_ok"] += 1
        else:
            res["problems"].append(f"the chart shows {tok!r}: no such number in the text")
    return res


# ------------------------------------------------------------------------------------------ X3-2: the hand reading
def label_agrees(keywords, excludes, haystack):
    """keywords "a|b ; c": every group has an alternative in the haystack (case-insensitive); excludes "x|y": none."""
    h = (haystack or "").lower()
    for group in [g for g in (keywords or "").split(";") if g.strip()]:
        if not any(a.strip().lower() in h for a in group.split("|") if a.strip()):
            return False
    return not any(x.strip().lower() in h for x in (excludes or "").split("|") if x.strip())


def _sign(x):
    return -1 if Decimal(str(x)) < 0 else 1


def match_points(run, label, units=None):
    """X3-2 for one run → a row per drawn point: label, value, place, the labelled number it stands on, label_ok,
    unit_ok, sign_ok and a reason when it disagrees."""
    units = units or UNITS
    v = run.checked.verified if run.checked is not None else None
    if run.output is None or v is None:
        return []
    rows = []
    mult = Decimal(SCALES[v.scale])
    for s, p in _points(v):
        row = {"label": p.label, "series": s.name, "value": str(p.value), "as_written": p.as_written,
               "start": p.start, "end": p.end, "unit": v.unit, "scale": v.scale, "label_ok": False, "unit_ok": False,
               "sign_ok": False, "of": None, "reason": ""}
        hits = [n for n in label["numbers"] if n["start"] < p.end and p.start < n["end"]]
        vals = [n for n in hits if n["role"] == "value"]
        if not vals:
            row["reason"] = (f"not a value of the hand reading ({hits[0]['role']}: {hits[0]['of']})" if hits
                             else "no labelled number at this place")
            rows.append(row)
            continue
        n = vals[0]
        row["of"] = n["of"]
        hay = " ".join([p.label, s.name or "", v.title or ""])
        row["label_ok"] = label_agrees(n["label_keywords"], n["label_excludes"], hay)
        acc = {canon_unit(k) if k else "": Decimal(str(f)) for k, f in units.get(n["unit"], {}).items()}
        row["unit_ok"] = v.unit in acc and abs(p.value) * mult == abs(Decimal(n["value"])) * acc[v.unit]
        row["sign_ok"] = _sign(p.value) == _sign(n["value"])
        why = []
        if not row["label_ok"]:
            why.append(f"label {p.label!r} does not say {n['label_keywords']!r}"
                       + (f" (or says one of {n['label_excludes']!r})" if n["label_excludes"] else ""))
        if not row["unit_ok"]:
            why.append(f"drawn as {p.value} {v.scale} {v.unit or 'plain'}; read as {n['value']} {n['unit']}")
        row["reason"] = "; ".join(why)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------ X3-3: refusals
def refusal_reason(run):
    """None for a drawn chart; else the blocking reason (invalid_proposal, proposer_failed), "no_values_proposed", or
    the commonest reason the proposed values were dropped (ties: the first in the report)."""
    if run.output is not None:
        return None
    blocked = [i.code for i in run.issues if i.severity == BLOCKED]
    if blocked and blocked[0] != "nothing_verified":
        return blocked[0]
    proposal = run.proposal
    n = sum(len(se.points) for se in proposal.series) if isinstance(proposal, ChartSpec) else 0
    if n == 0:
        return "no_values_proposed"
    dropped = [i.code for i in run.issues if i.severity == DROPPED]
    if not dropped:
        return blocked[0] if blocked else "not_drawn"
    c = Counter(dropped)
    top = max(c.values())
    return next(code for code in dropped if c[code] == top)


def tally(runs):
    reasons = Counter(r for r in (refusal_reason(x) for x in runs) if r is not None)
    drawn = sum(x.output is not None for x in runs)
    return {"drawn": drawn, "refused": len(runs) - drawn, "reasons": dict(sorted(reasons.items()))}


# ------------------------------------------------------------------------------------------ X3-4: corruption
def _proposal_and_issues(x):
    if isinstance(x, dict):                                   # a stored record (run.to_dict())
        return x.get("proposal"), x.get("issues") or []
    p = x.proposal.model_dump(mode="json") if hasattr(x.proposal, "model_dump") else x.proposal
    return p, [{"severity": i.severity, "code": i.code, "path": i.path} for i in x.issues]


def verified_only(x):
    """A run (or its record) → its proposal with the points, series and total the check dropped taken out (None when
    nothing is left, or the proposal was not valid)."""
    p, issues = _proposal_and_issues(x)
    if not isinstance(p, dict) or "series" not in p:
        return None
    p = copy.deepcopy(p)
    gone_pts, gone_series, gone_total = set(), set(), False
    for i in issues:
        if i["severity"] != DROPPED:
            continue
        m = re.fullmatch(r"series\[(\d+)\]\.points\[(\d+)\]", i.get("path") or "")
        if m:
            gone_pts.add((int(m.group(1)), int(m.group(2))))
        elif re.fullmatch(r"series\[(\d+)\]", i.get("path") or ""):
            gone_series.add(int(re.findall(r"\d+", i["path"])[0]))
        elif i.get("path") == "total":
            gone_total = True
    series = []
    for si, se in enumerate(p["series"]):
        if si in gone_series:
            continue
        se["points"] = [pt for pj, pt in enumerate(se["points"]) if (si, pj) not in gone_pts]
        if se["points"]:
            series.append(se)
    if not series:
        return None
    p["series"] = series
    if gone_total:
        p["total"] = None
    return p


def corruption(runs, texts, n=300, seed=0):
    """X3-4: {"recorded" | "verified_only": {kind: {"n", "caught", "skipped", "rate", "rows"}}} — each kind run on its
    own with `n` mutations and the same seed. runs: {id: Run or record}; texts: {id: text}."""
    sp = ChartSpecialist()
    kinds = {"change_digit": change_digit(paths=VALUE), "swap": swap(paths=VALUE), "drop": drop(paths=POINT)}
    arms = {"recorded": [], "verified_only": []}
    for rid in sorted(runs):
        p, _ = _proposal_and_issues(runs[rid])
        try:
            ChartSpec.model_validate(p)
        except Exception:                                     # noqa: BLE001 — an invalid proposal has nothing to corrupt
            continue
        arms["recorded"].append((rid, texts[rid], p))
        vo = verified_only(runs[rid])
        if vo is not None:
            arms["verified_only"].append((rid, texts[rid], vo))
    out = {}
    for arm, recs in arms.items():
        out[arm] = {}
        for kind, mut in kinds.items():
            try:
                rep = corrupt(sp, recs, {kind: mut}, n=n, seed=seed)
            except ValueError as e:
                out[arm][kind] = {"n": 0, "caught": 0, "skipped": 0, "rate": None, "rows": [], "note": str(e)}
                continue
            out[arm][kind] = {"n": rep.n, "caught": rep.caught, "skipped": rep.skipped,
                              "rate": rep.rate if rep.n else None, "rows": rep.rows}
    return out


# ------------------------------------------------------------------------------------------ X3-5: replay
def _not_called(source, question=None):
    raise RuntimeError("replay called the proposer")


def replay_checks(records, texts, *, rerun=None):
    """{id: record} → per text: the record replays from itself after a JSON round trip — re-parsed, re-checked and
    re-rendered with a proposer that may not be called — to the recorded bytes ("same_bytes"); with `rerun` (a
    deterministic proposer: the rules) a fresh run gives the same bytes too ("rerun_same"). A model proposer is not
    re-run: its records are the outputs."""
    sp = ChartSpecialist(_not_called)
    out = {}
    for rid in sorted(records):
        rec = json.loads(json.dumps(records[rid], ensure_ascii=False))
        r = sp.replay(rec, texts[rid])
        row = {"replay_ok": bool(r), "problems": r.problems, "from_record": r.from_record,
               "same_bytes": bool(r) and r.from_record and r.output_sha256 == rec.get("output_sha256")}
        if rerun is not None:
            again = ChartSpecialist(rerun).run(texts[rid])
            sha = hashlib.sha256(again.output.encode()).hexdigest() if again.output is not None else None
            row["rerun_same"] = sha == rec.get("output_sha256")
        out[rid] = row
    return out


# ------------------------------------------------------------------------------------------ the run
def _heartbeat(state, msg):
    now = time.time()
    if now - state.get("t", 0) >= 30 or msg.startswith("done"):
        print(f"[{_dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)
        state["t"] = now


def run_all(texts, labels, out, *, proposer=None, n_corrupt=300, seed=0, arm="rules", units=None):
    """texts: {id: text}; labels: {id: label}; out: the output directory (resumes from out/runs.jsonl)."""
    out = Path(out)
    (out / "svg").mkdir(parents=True, exist_ok=True)
    runs_path = out / "runs.jsonl"
    done = {}
    if runs_path.exists():
        for line in runs_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done[r["id"]] = r
    sp = ChartSpecialist(proposer or RuleProposer())
    hb = {}
    with open(runs_path, "a") as f:
        for i, (tid, text) in enumerate(texts.items(), 1):
            if tid in done:
                continue
            try:
                run = sp.run(text)
                rec = {"id": tid, "arm": arm, "ok": run.ok, "record": run.to_dict(), "reverify": reverify(run, text),
                       "points": match_points(run, labels[tid], units), "refusal": refusal_reason(run),
                       "report": run.report()}
                if run.output is not None:
                    (out / "svg" / f"{tid}.svg").write_text(run.output)
            except Exception as e:                            # noqa: BLE001 — logged, the run goes on
                rec = {"id": tid, "arm": arm, "error": f"{type(e).__name__}: {e}"}
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            done[tid] = rec
            _heartbeat(hb, f"{i}/{len(texts)} {tid}: " + ("drawn" if rec.get("ok") else
                                                           f"refused ({rec.get('refusal') or rec.get('error')})"))
    _heartbeat(hb, f"done: {len(done)} texts; corruption and replay next")
    recs = {tid: done[tid] for tid in texts}
    good = {tid: r["record"] for tid, r in recs.items() if "record" in r}
    report = summarise(recs, texts, labels)
    report["x3_4"] = {arm_: {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in kinds.items()}
                      for arm_, kinds in corruption(good, texts, n=n_corrupt, seed=seed).items()}
    rp = replay_checks(good, texts, rerun=RuleProposer() if arm == "rules" else None)
    try:
        cs = check_specialist(ChartSpecialist(), list(texts.values()))
        cs_res = {"passed": True, **cs}
    except ConformanceError as e:
        cs_res = {"passed": False, "error": str(e)}
    report["x3_5"] = {"replayed_ok": sum(r["replay_ok"] for r in rp.values()),
                      "same_bytes": sum(r["same_bytes"] for r in rp.values()), "texts": len(texts),
                      "rerun_same": (sum(r["rerun_same"] for r in rp.values()) if arm == "rules" else None),
                      "failures": {k: r for k, r in rp.items() if not (r["replay_ok"] and r["same_bytes"]
                                                                      and r.get("rerun_same", True))},
                      "check_specialist": cs_res}
    report["config"] = {"arm": arm, "proposer": getattr(sp.proposer, "id", str(sp.proposer)), "n_corrupt": n_corrupt,
                        "seed": seed, "texts": len(texts)}
    report["verdict_inputs"] = bars(report)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False, default=str))
    (out / "report.md").write_text(markdown(report))
    return report


def summarise(recs, texts, labels):
    rows = [dict(r, id=tid) for tid, rec in recs.items() for r in rec.get("points", [])]
    rv = [rec["reverify"] for rec in recs.values() if "reverify" in rec]
    reasons = Counter(rec["refusal"] for rec in recs.values() if rec.get("refusal"))
    errors = {tid: rec["error"] for tid, rec in recs.items() if "error" in rec}
    agree = [r for r in rows if r["label_ok"] and r["unit_ok"]]
    return {
        "x3_1": {"texts_drawn": sum(1 for r in recs.values() if r.get("ok")),
                 "texts_all_reverified": sum(1 for rec in recs.values() if rec.get("ok") and not rec["reverify"]["problems"]),
                 "values": sum(r["values"] for r in rv), "values_ok": sum(r["values_ok"] for r in rv),
                 "svg_numbers": sum(r["svg_numbers"] for r in rv), "svg_numbers_ok": sum(r["svg_numbers_ok"] for r in rv),
                 "problems": {tid: rec["reverify"]["problems"] for tid, rec in recs.items()
                              if rec.get("reverify", {}).get("problems")}},
        "x3_2": {"points": len(rows), "agree": len(agree), "label_ok": sum(r["label_ok"] for r in rows),
                 "unit_ok": sum(r["unit_ok"] for r in rows), "sign_ok": sum(r["sign_ok"] for r in rows),
                 "rate": len(agree) / len(rows) if rows else None,
                 "disagreements": [r for r in rows if not (r["label_ok"] and r["unit_ok"])],
                 "sign_disagreements": [r for r in rows if not r["sign_ok"] and r["of"] is not None]},
        "x3_3": {"texts": len(texts), "drawn": sum(1 for r in recs.values() if r.get("ok")),
                 "refused": sum(1 for r in recs.values() if not r.get("ok")), "reasons": dict(sorted(reasons.items())),
                 "errors": errors,
                 "refused_texts": {tid: rec.get("refusal") or rec.get("error") for tid, rec in recs.items() if not rec.get("ok")}},
    }


def bars(rep):
    """The numbers each bar of PREREG X3 is read on (the verdict is the coordinator's)."""
    x1, x2, x3, x4, x5 = rep["x3_1"], rep["x3_2"], rep["x3_3"], rep["x3_4"]["recorded"], rep["x3_5"]
    return {
        "X3-1": {"bar": "0 numbers on the charts not in the text; every rendered value re-verified (texts drawn)",
                 "measured": f"{x1['texts_all_reverified']}/{x1['texts_drawn']} drawn texts clean; values "
                             f"{x1['values_ok']}/{x1['values']}; chart numbers {x1['svg_numbers_ok']}/{x1['svg_numbers']}",
                 "met": x1["values_ok"] == x1["values"] and x1["svg_numbers_ok"] == x1["svg_numbers"]},
        "X3-2": {"bar": ">= 95% of drawn points agree in label and unit", "measured": x2["rate"],
                 "met": x2["rate"] is not None and x2["rate"] >= 0.95},
        "X3-3": {"bar": "a chart drawn for >= 50% of the texts", "measured": x3["drawn"] / x3["texts"],
                 "met": x3["drawn"] / x3["texts"] >= 0.5},
        "X3-4": {"bar": "change_digit caught 100%; swap >= 95%; drop reported",
                 "measured": {k: (v["caught"], v["n"]) for k, v in x4.items()},
                 "met": (x4["change_digit"]["n"] > 0 and x4["change_digit"]["caught"] == x4["change_digit"]["n"]
                         and x4["swap"]["n"] > 0 and x4["swap"]["caught"] >= 0.95 * x4["swap"]["n"])},
        "X3-5": {"bar": "replay 50/50 identical bytes; check_specialist passes",
                 "measured": f"replayed {x5['replayed_ok']}/{x5['texts']}, same bytes {x5['same_bytes']}/{x5['texts']}"
                             + (f", a second rule run the same {x5['rerun_same']}/{x5['texts']}"
                                if x5.get("rerun_same") is not None else "")
                             + f", check_specialist {'passed' if x5['check_specialist']['passed'] else 'FAILED'}",
                 "met": x5["replayed_ok"] == x5["texts"] and x5["same_bytes"] == x5["texts"]
                 and x5.get("rerun_same") in (None, x5["texts"]) and x5["check_specialist"]["passed"]},
    }


def markdown(rep):
    c, b = rep["config"], rep["verdict_inputs"]
    lines = [f"# X3 — verified charts on real texts ({c['arm']}, {c['texts']} texts)", "",
             f"Proposer: `{c['proposer']}`; corruption: n = {c['n_corrupt']} per kind, seed {c['seed']}.", "",
             "| bar | measured | met |", "|---|---|---|"]
    for k, v in b.items():
        m = v["measured"]
        m = f"{m:.1%}" if isinstance(m, float) else m
        lines.append(f"| {k}: {v['bar']} | {m} | {'yes' if v['met'] else 'no'} |")
    x2, x3, x4 = rep["x3_2"], rep["x3_3"], rep["x3_4"]
    lines += ["", "## X3-2 points", "",
              f"{x2['points']} drawn points: label agrees {x2['label_ok']}, unit agrees {x2['unit_ok']}, both {x2['agree']}; "
              f"sign agrees {x2['sign_ok']} (reported apart).", ""]
    for r in x2["disagreements"]:
        lines.append(f"- {r['id']} {r['label']!r} = {r['value']} ({r['as_written']!r}): {r['reason']}")
    lines += ["", "## X3-3 refusals", "", f"drawn {x3['drawn']}/{x3['texts']}; reasons: {x3['reasons']}", ""]
    for tid, why in x3["refused_texts"].items():
        lines.append(f"- {tid}: {why}")
    lines += ["", "## X3-4 corruption", "", "| arm | change_digit | swap | drop |", "|---|---|---|---|"]
    for arm, kinds in x4.items():
        lines.append(f"| {arm} | " + " | ".join(f"{v['caught']}/{v['n']}" + (f" ({v['skipped']} skipped)" if v["skipped"] else "")
                                                for v in kinds.values()) + " |")
    if rep["x3_1"]["problems"]:
        lines += ["", "## X3-1 problems", ""] + [f"- {k}: {v}" for k, v in rep["x3_1"]["problems"].items()]
    return "\n".join(lines) + "\n"


def _check_labels(texts, labels):
    bad = [tid for tid, text in texts.items()
           if labels[tid].get("text_sha256") not in (None, hashlib.sha256(text.encode()).hexdigest())]
    if bad:
        raise SystemExit(f"labels.json does not belong to these texts: {bad}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry", action="store_true", help="the 3 gallery texts of tests/test_charts.py (development)")
    ap.add_argument("--run", action="store_true", help="the measured run on the 50 texts (after the coordinator's word)")
    ap.add_argument("--texts", default=str(DATA / "texts.jsonl"))
    ap.add_argument("--labels", default=str(DATA / "labels.json"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-corrupt", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--llm", default=None, help="report arm: an LLM proposer (model id) through the shared proxy")
    ap.add_argument("--replay", default=None, help="X3-5's replay alone, from a runs.jsonl a run wrote (no proposer)")
    a = ap.parse_args(argv)
    if a.replay:
        texts = {r["id"]: r["text"] for r in load_texts(a.texts, mode="replay")}
        recs = [json.loads(x) for x in Path(a.replay).read_text().splitlines() if x.strip()]
        good = {r["id"]: r["record"] for r in recs if "record" in r}
        rp = replay_checks(good, texts)
        rep = {"runs": len(recs), "records": len(good), "replayed_ok": sum(r["replay_ok"] for r in rp.values()),
               "from_record": sum(r["from_record"] for r in rp.values()),
               "same_bytes": sum(r["same_bytes"] for r in rp.values()),
               "failures": {k: r for k, r in rp.items() if not r["same_bytes"]}}
        print(json.dumps(rep, indent=1, ensure_ascii=False))
        return rep
    if a.dry == a.run:
        ap.error("say --dry (the gallery texts) or --run (the measured set), one of them")
    proposer, arm = None, "rules"
    if a.llm:
        arm = "llm-" + re.sub(r"[^\w.-]", "_", a.llm)
        proposer = LLMProposer(PROXY.format(arm=arm), a.llm, timeout=120.0)
    if a.dry:
        texts, labels, units = dict(GALLERY), GALLERY_LABELS, UNITS
        out = Path(a.out or ROOT / "out" / f"x3_dry_{arm}")
    else:
        lab = json.loads(Path(a.labels).read_text())
        texts = {r["id"]: r["text"] for r in load_texts(a.texts, mode=f"measured:{arm}")}
        labels, units = lab["texts"], lab.get("units") or UNITS
        _check_labels(texts, labels)
        out = Path(a.out or ROOT / "out" / f"x3_{arm}")
    rep = run_all(texts, labels, out, proposer=proposer, n_corrupt=a.n_corrupt, seed=a.seed, arm=arm, units=units)
    print((out / "report.md").read_text())
    return rep


if __name__ == "__main__":
    main()
