"""SROIE receipts (ICDAR 2019: OCR text + reference fields company / date / address / total).

Provides the document text, typed questions (in the format of the Laya baseline) with ground truth derived from the
reference fields, and field spans in the text for training an extractor.

Questions:
  total_band  - choice: total is "under 10" / "10 to 30" / "30 to 100" / "over 100" (extract + compare);
  weekend     - yes/no: bought on a Saturday or Sunday (extract the date + compute the weekday);
  year        - choice: 2016 / 2017 / 2018 / 2019 (extract + parse);
  quarter     - choice: Q1..Q4 (extract + parse);
  state       - choice: Selangor / Kuala Lumpur / Johor / other (from the address);
  sdn         - yes/no: the seller is an "Sdn Bhd" company (SDN in the name; OCR also gives "SDN BND")."""
from __future__ import annotations

import os
import difflib
import re
from datetime import date
from pathlib import Path

DATA = Path(os.environ.get("SOLVI_DATA", "data")) / "sroie"
FIELDS = {"company": "the name of the company that issued the receipt",
          "date": "the date of the purchase",
          "address": "the address of the shop",
          "total": "the total amount paid"}
BANDS = ["under 10", "10 to 30", "30 to 100", "over 100"]
YEARS = ["2016", "2017", "2018", "2019"]
QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
STATES = ["Selangor", "Kuala Lumpur", "Johor", "other"]
MONTHS = {m: i + 1 for i, m in enumerate(["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}


def load(split, limit=None):
    """Receipt texts. Uses {split}.jsonl (text and fields only, no images) if present, else {split}.parquet."""
    import json
    js = DATA / f"{split}.jsonl"
    if js.exists():
        out = [json.loads(line) for line in js.read_text().splitlines()]
        return out[:limit] if limit else out
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(f"{js} is missing; reading {split}.parquet instead needs pandas and pyarrow (not solvi "
                          "dependencies): pip install pandas pyarrow") from None
    d = pd.read_parquet(DATA / f"{split}.parquet", columns=["key", "entities", "words"])
    out = []
    for r in d.itertuples():
        text = "\n".join(str(w) for w in r.words)
        out.append({"key": r.key, "text": text, "gold": {k: str(r.entities[k]) for k in FIELDS}})
    return out[:limit] if limit else out


# ---------------------------------------------------------------- value parsing (shared by ground truth and predictions)
def parse_total(s):
    m = re.findall(r"\d[\d,]*\.\d{1,2}|\d+", str(s).replace(" ", ""))
    if not m:
        raise ValueError(f"no number: {s!r}")
    return float(m[-1].replace(",", ""))


def parse_date(s):
    s = str(s).strip().upper()
    m = re.search(r"(\d{1,2})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{2,4})", s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        y += 2000 if y < 100 else 0
        d, mo = (a, b) if b <= 12 else (b, a)          # day first; swap if the "month" is > 12
        return date(y, mo, d)
    m = re.search(r"(\d{1,2})\s*([A-Z]{3})[A-Z]*\s*,?\s*(\d{2,4})", s)
    if m and m.group(2) in MONTHS:
        y = int(m.group(3))
        y += 2000 if y < 100 else 0
        return date(y, MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"(\d{4})\s*[/\-.]\s*(\d{1,2})\s*[/\-.]\s*(\d{1,2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    raise ValueError(f"not a date: {s!r}")


def band(total):
    return BANDS[0] if total < 10 else BANDS[1] if total < 30 else BANDS[2] if total < 100 else BANDS[3]


def state_of(address):
    a = str(address).upper()
    if "SELANGOR" in a or "S'GOR" in a or "SHAH ALAM" in a or "PETALING JAYA" in a or "KLANG" in a:
        return "Selangor"
    if "KUALA LUMPUR" in a or "W.P" in a or "WILAYAH" in a:
        return "Kuala Lumpur"
    if "JOHOR" in a:
        return "Johor"
    return "other"


def is_sdn(company):
    return "SDN" in str(company).upper()


def labels(gold):
    out = {}
    try:
        t = parse_total(gold["total"])
        out["total_band"] = band(t)
    except ValueError:
        pass
    try:
        d = parse_date(gold["date"])
        out["weekend"] = "yes" if d.weekday() >= 5 else "no"
        if str(d.year) in YEARS:
            out["year"] = str(d.year)
        out["quarter"] = QUARTERS[(d.month - 1) // 3]
    except ValueError:
        pass
    out["state"] = state_of(gold["address"])
    out["sdn"] = "yes" if is_sdn(gold["company"]) else "no"
    return out


# ---------------------------------------------------------------- questions in the Laya baseline format
LAYA_Q = {
    "total_band": {"type": "choice", "instructions": "What is the total amount paid on this receipt?",
                   "criteria": {b: f"The total is {b} (in RM)." for b in BANDS}},
    "weekend": {"type": "noul", "instructions": "This purchase was made on a Saturday or a Sunday.",
                "criteria": {"false": "It was made on a weekday.", "true": "It was made on a weekend."}},
    "year": {"type": "choice", "instructions": "In which year was this purchase made?", "criteria": {y: f"In {y}." for y in YEARS}},
    "quarter": {"type": "choice", "instructions": "In which quarter of the year was this purchase made?",
                "criteria": {"Q1": "January to March.", "Q2": "April to June.", "Q3": "July to September.", "Q4": "October to December."}},
    "state": {"type": "choice", "instructions": "In which Malaysian state is the shop located?",
              "criteria": {s: (f"In {s}." if s != "other" else "In another state.") for s in STATES}},
    "sdn": {"type": "noul", "instructions": "The seller is a private limited company (Sdn Bhd).",
            "criteria": {"false": "It is not an Sdn Bhd company.", "true": "It is an Sdn Bhd company."}},
}


def laya_rows(docs):
    rows = []
    for d in docs:
        lab = labels(d["gold"])
        q = {k: v for k, v in LAYA_Q.items() if k in lab}
        gold = {}
        for k in q:
            a = lab[k]
            if q[k]["type"] == "noul":
                t = "true" if a == "yes" else "false"
                gold[k] = {"label": t, "probabilities": {"false": float(t == "false"), "true": float(t == "true")}, "noul": float(t == "true")}
            else:
                gold[k] = {"label": a, "probabilities": {c: float(c == a) for c in q[k]["criteria"]}}
        rows.append({"id": d["key"], "workflow": "receipts", "state": {"receipt_text": d["text"]}, "questions": q, "gold": gold,
                     "factors": {}})
    return rows


# ---------------------------------------------------------------- field spans in the text (for extractor training)
def _norm(s):
    return re.sub(r"\s+", " ", str(s).upper()).strip()


def _toks(s):
    return re.findall(r"[A-Z0-9]+", str(s).upper())


def token_f1(pred, gold):
    p, g = _toks(pred), _toks(gold)
    if not p or not g:
        return 0.0
    inter = sum(min(p.count(t), g.count(t)) for t in set(p))
    if inter == 0:
        return 0.0
    pr, rc = inter / len(p), inter / len(g)
    return 2 * pr * rc / (pr + rc)


def find_span(text, field, value):
    """Best span of a value in noisy OCR text: exact search, else the most similar window of lines. -> (start, end) | None."""
    if field == "total":
        v = str(value).strip()
        hits = [m.start() for m in re.finditer(re.escape(v), text)]
        if not hits:
            return None
        tot = [m.end() for m in re.finditer(r"TOTAL", text, flags=re.I)]
        after = [h for h in hits if any(0 < h - t < 60 for t in tot)]
        h = (after or hits)[-1]
        return h, h + len(v)
    if field == "date":
        v = str(value).strip()
        i = text.find(v)
        return (i, i + len(v)) if i >= 0 else None
    lines, pos, off = text.split("\n"), [], 0
    for ln in lines:
        pos.append(off)
        off += len(ln) + 1
    if field == "address":
        # in this SROIE version the OCR text often misses parts of the address ("JALAN DEDAP 13" is in the reference but
        # not in the text), so take the window of lines covering the most reference words with window precision >= 0.7
        gt = set(_toks(value))
        bestc = (0, None)
        for i in range(len(lines)):
            for j in range(i, min(len(lines), i + 8)):
                wt = _toks(" ".join(lines[i:j + 1]))
                if not wt:
                    continue
                inter = sum(1 for t in wt if t in gt)
                if inter / len(wt) >= 0.7 and inter > bestc[0]:
                    bestc = (inter, (pos[i], pos[j] + len(lines[j])))
        return bestc[1] if bestc[0] >= 2 else None
    target = _norm(value)
    best = (0.0, None)
    for i in range(len(lines)):
        for j in range(i, min(len(lines), i + 5)):
            cand = _norm(" ".join(lines[i:j + 1]))
            r = difflib.SequenceMatcher(None, cand, target).ratio()
            if r > best[0]:
                best = (r, (pos[i], pos[j] + len(lines[j])))
    return best[1] if best[0] >= 0.8 else None


def value_match(field, pred, gold):
    """Does an extracted value match the reference: total and date by value; company by string similarity >= 0.8;
    address by token F1 >= 0.5 (OCR text often misses parts of the address)."""
    try:
        if field == "total":
            return abs(parse_total(pred) - parse_total(gold)) < 0.005
        if field == "date":
            return parse_date(pred) == parse_date(gold)
    except ValueError:
        return False
    if field == "address":
        return token_f1(pred, gold) >= 0.5
    return difflib.SequenceMatcher(None, _norm(pred), _norm(gold)).ratio() >= 0.8
