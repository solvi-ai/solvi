"""CUAD v1 contracts (510 real contracts annotated by lawyers; test = the official test.json, 102 contracts; train = the
other 408). Fields are clause spans in the text; typed questions and their ground truth come from the annotations."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

DATA = Path(os.environ.get("SOLVI_DATA", "data")) / "cuad"
FIELDS = {"governing_law": ("Governing Law", "the clause that says which state's or country's law governs the contract"),
          "agreement_date": ("Agreement Date", "the date when the contract was signed or entered into"),
          "non_compete": ("Non-Compete", "a clause that restricts a party from competing or operating in a business, market or area"),
          "exclusivity": ("Exclusivity", "a clause granting exclusive rights or requiring exclusive dealing"),
          "cap_liability": ("Cap On Liability", "a clause that caps or limits the amount of liability of a party")}
STATES = ["Delaware", "New York", "California", "other"]
ERAS = ["before 2005", "2005 to 2012", "2013 to 2016", "2017 or later"]


def load(split):
    """split: train | test -> [{key, text, spans{field: (start, end) | None}, gold{field: text | None}}]."""
    test_titles = {d["title"] for d in json.load(open(DATA / "test.json"))["data"]}
    out = []
    for d in json.load(open(DATA / "CUADv1.json"))["data"]:
        if (d["title"] in test_titles) != (split == "test"):
            continue
        p = d["paragraphs"][0]
        text = p["context"]
        spans, gold = {}, {}
        for f, (cat, _) in FIELDS.items():
            qa = next((q for q in p["qas"] if f'"{cat}"' in q["question"]), None)
            ans = qa["answers"] if qa else []
            if ans:
                a = min(ans, key=lambda x: x["answer_start"])
                spans[f] = (a["answer_start"], a["answer_start"] + len(a["text"]))
                gold[f] = a["text"]
            else:
                spans[f], gold[f] = None, None
        out.append({"key": d["title"], "text": text, "spans": spans, "gold": gold})
    return out


def state_of(clause):
    c = str(clause or "")
    for s in STATES[:3]:
        if s.lower() in c.lower():
            return s
    return "other"


def era_of(date_text):
    ys = [int(y) for y in re.findall(r"(19\d\d|20\d\d)", str(date_text or ""))]
    if not ys:
        raise ValueError("no year")
    y = ys[0]
    return ERAS[0] if y < 2005 else ERAS[1] if y <= 2012 else ERAS[2] if y <= 2016 else ERAS[3]


def labels(gold):
    lab = {}
    if gold["governing_law"]:
        lab["law"] = state_of(gold["governing_law"])
    if gold["agreement_date"]:
        try:
            lab["era"] = era_of(gold["agreement_date"])
        except ValueError:
            pass
    for f in ("non_compete", "exclusivity", "cap_liability"):
        lab[f] = "yes" if gold[f] else "no"
    return lab


LAYA_Q = {
    "law": {"type": "choice", "instructions": "Which state's law governs this contract?",
            "criteria": {s: (f"The law of {s}." if s != "other" else "Another state's or country's law.") for s in STATES}},
    "era": {"type": "choice", "instructions": "When was this contract signed?", "criteria": {e: f"Signed {e}." for e in ERAS}},
    "non_compete": {"type": "noul", "instructions": "The contract contains a non-compete clause.",
                    "criteria": {"false": "No non-compete clause.", "true": "It has a non-compete clause."}},
    "exclusivity": {"type": "noul", "instructions": "The contract contains an exclusivity clause.",
                    "criteria": {"false": "No exclusivity clause.", "true": "It has an exclusivity clause."}},
    "cap_liability": {"type": "noul", "instructions": "The contract caps the liability of a party.",
                      "criteria": {"false": "No cap on liability.", "true": "Liability is capped."}},
}


def load_all(split):
    """All 41 CUAD clause categories -> ([{key, text, cats{category: (start, end) | None}}], {category: CUAD description})."""
    test_titles = {d["title"] for d in json.load(open(DATA / "test.json"))["data"]}
    out, details = [], {}
    for d in json.load(open(DATA / "CUADv1.json"))["data"]:
        if (d["title"] in test_titles) != (split == "test"):
            continue
        p = d["paragraphs"][0]
        cats = {}
        for q in p["qas"]:
            m = re.search(r'related to "([^"]+)".*?Details: (.*)$', q["question"], re.S)
            if not m:
                continue
            cat, det = m.group(1), m.group(2).strip()
            details[cat] = det
            if q["answers"]:
                a = min(q["answers"], key=lambda x: x["answer_start"])
                cats[cat] = (a["answer_start"], a["answer_start"] + len(a["text"]))
            else:
                cats[cat] = None
        out.append({"key": d["title"], "text": p["context"], "cats": cats})
    return out, details
