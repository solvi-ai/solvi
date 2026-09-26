"""Kleister-NDA non-disclosure agreements (applicaai/kleister-nda; 254 train / 83 dev-0). Ground truth is normalized
values (ISO date, state, parties, term) without spans in the text, so matching is by value."""
from __future__ import annotations

import lzma
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("SOLVI_DATA", "data")) / "kleister_nda"
FIELDS = {"effective_date": "the date when the agreement becomes effective",
          "jurisdiction": "the state whose law governs the agreement",
          "party": "the name of a party to the agreement (a company or a person)",
          "term": "the length of time the agreement stays in force"}
NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
       "twelve": 12, "eighteen": 18, "twenty-four": 24, "thirty-six": 36, "sixty": 60}


def load(split="dev-0"):
    rows = lzma.open(DATA / split / "in.tsv.xz", "rt").read().splitlines()
    exp = (DATA / split / "expected.tsv").read_text().splitlines()
    out = []
    for r, e in zip(rows, exp):
        cols = r.split("\t")
        text = cols[2].replace("\\n", "\n")
        gold = {}
        for kv in e.split():
            k, v = kv.split("=", 1)
            gold.setdefault(k, []).append(v.replace("_", " "))
        out.append({"key": cols[0], "text": text, "gold": gold})
    return out


def _date(s):
    from dateutil import parser
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", s).replace("day of", "")
    try:
        return parser.parse(s, fuzzy=True, default=None).date().isoformat()
    except Exception:  # noqa: BLE001
        return None


def _norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace(",", " ")).split()


def _term(s):
    s = s.lower()
    m = re.search(r"(\d+|" + "|".join(NUM) + r")\s*(?:\(\d+\)\s*)?(year|month|day)s?", s)
    if not m:
        return None
    n = int(m.group(1)) if m.group(1).isdigit() else NUM[m.group(1)]
    return f"{n} {m.group(2)}s"


def match(field, pred, gold):
    """pred: extracted fragment ('' means "none"); gold: list of reference values (or None)."""
    if not gold:
        return pred == ""
    if not pred:
        return False
    if field == "effective_date":
        return _date(pred) == gold[0]
    if field == "jurisdiction":
        return gold[0].lower() in pred.lower()
    if field == "term":
        return _term(pred) == gold[0].lower()
    p = set(_norm(pred)) - {"inc", "llc", "corp", "corporation", "ltd", "the", "co", "company"}
    for g in gold:
        gs = set(_norm(g)) - {"inc", "llc", "corp", "corporation", "ltd", "the", "co", "company"}
        if gs and len(p & gs) / len(gs) >= 0.6 and len(p & gs) / max(1, len(p)) >= 0.5:
            return True
    return False
