"""Abt-Buy pair facts as a solvi catalog: typed facts read from each offer by code (brand, model number with its quote,
the other codes, sizes, colours, price) and their comparison. One request is one pair:
{"a_name", "a_description", "a_price", "b_name", "b_description", "b_price"}.

This is the part of the solution written for these two catalogs (Abt puts the model number last in the name, Buy writes
it with dashes or spaces); solvi supplies the head over these facts, the threshold and the trace. No model here."""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402

from solvi import Answer, Catalog, Question, Quote  # noqa: E402

PREPARED = DATA / "abtbuy/prepared"


def load():
    """The two catalogs: {id: offer}."""
    return tuple({r["id"]: r for r in read_jsonl(PREPARED / f"{f}.jsonl")} for f in ("abt", "buy"))


def state(a, b):
    """The input of one pair, keys always in this order (a fixed key order: see the best practices)."""
    return {"a_name": a["name"], "a_description": a["description"], "a_price": a["price"],
            "b_name": b["name"], "b_description": b["description"], "b_price": b["price"]}


# ------------------------------------------------------------------------------------------------ reading one offer
WORD = re.compile(r"[a-z0-9][a-z0-9./+\-]*")
COLOURS = {"black", "white", "silver", "red", "blue", "pink", "green", "gray", "grey", "titanium", "brown", "purple",
           "orange", "yellow", "gold", "stainless", "bisque", "platinum", "charcoal"}
UNITS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:'|\"|-inch|inch|gb|mb|tb|mp|megapixel|w\b|watt|x\b|ghz|mhz|cu|btu|port|ft|channel|ch\b|-channel|disc)")


def squash(t):
    return re.sub(r"[^a-z0-9]", "", t.lower())


SPEC = re.compile(r"^\d+(?:\.\d+)?(?:p|i|mm|cm|in|gb|mb|tb|kb|hz|khz|mhz|ghz|w|watt|watts|v|mah|rpm|x|k|lb|lbs|ft|cf|btu|"
                  r"disc|point|channel|ch|port|piece|pack|bit|mbps|gbps|dpi|ppm|th|nd|rd|st|d)$|^\d+x\d+$|^(?:\d+[pi])+$")


def is_code(tok):
    """A model / part number: letters and digits together (pslx350h, kx-tg9342t, gr-4), or a long number (10020630).
    Not a spec: 1080p, 320gb, 18-55mm, 1920x1080."""
    s = squash(tok)
    if len(s) < 3 or SPEC.match(s):
        return False
    d = sum(c.isdigit() for c in s)
    return (0 < d < len(s)) or (d == len(s) and len(s) >= 5)


def code_spans(text):
    """[(squashed code, start, end)] in reading order."""
    return [(squash(m.group()), m.start(), m.end()) for m in WORD.finditer(text.lower()) if is_code(m.group())]


def stem(code):
    """A code up to its last digit: du1055ss → du1055, kxtg9342t → kxtg9342 (colour / region suffixes dropped)."""
    m = re.match(r".*\d", code)
    return m.group() if m else code


def words(text):
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1]


class Idf:
    """Inverse document frequency of name words over both catalogs (unlabelled): rare words weigh more."""

    def __init__(self, offers):
        df = Counter(w for o in offers for w in set(words(o["name"])))
        self.n, self.df = len(offers), df

    def __call__(self, w):
        return math.log((self.n + 1) / (self.df.get(w, 0) + 1))


def build(idf):
    """The catalog and the "match" question, which has no rule: a head fitted on labelled pairs answers it."""
    cat = Catalog()

    # --- typed facts per offer, read by code ---------------------------------------------------------------
    @cat.fn
    def a_brand(a_name) -> str:
        return a_name.split()[0] if a_name.split() else ""

    @cat.fn
    def b_brand(b_name) -> str:
        return b_name.split()[0] if b_name.split() else ""

    @cat.extract
    def a_model(a_name):
        "the model number of offer A: the last word of its name (Abt's convention), quoted"
        m = re.search(r"(\S+)\s*$", a_name)
        if not m or len(squash(m.group(1))) < 3:
            raise ValueError("no model code in the name")
        return Quote(squash(m.group(1)), m.start(1), m.end(1))

    @cat.fn
    def a_key(a_name) -> str:
        "the same value as a plain fact that is never missing ('' when the name has no usable last word)"
        m = re.search(r"(\S+)\s*$", a_name)
        return squash(m.group(1)) if m and len(squash(m.group(1))) >= 3 else ""

    @cat.fn
    def a_codes(a_name) -> list:
        return sorted({c for c, _, _ in code_spans(a_name)})

    @cat.fn
    def b_codes(b_name) -> list:
        toks = b_name.split()                           # "rm 705" → rm705: short letters, then digits
        joined = [x + y for x, y in zip(toks, toks[1:]) if x.isalpha() and len(x) <= 4 and y.isdigit() and len(y) >= 2]
        return sorted({c for c, _, _ in code_spans(b_name)} | set(joined))

    @cat.fn
    def b_desc_codes(b_description) -> list:
        return sorted({c for c, _, _ in code_spans(b_description)})

    # --- comparisons in code -------------------------------------------------------------------------------
    @cat.check
    def same_brand(a_brand, b_brand) -> bool:
        return bool(a_brand) and a_brand == b_brand

    @cat.check
    def model_equal(a_key, b_codes, b_desc_codes) -> bool:
        "A's model number is literally one of B's codes"
        return bool(a_key) and (a_key in b_codes or a_key in b_desc_codes)

    @cat.check
    def model_in_b(a_key, b_name, b_description) -> bool:
        "A's model number appears in B's text with separators removed (kx-tg 9342 t, ai sh hphone)"
        return len(a_key) >= 4 and (a_key in squash(b_name) or a_key in squash(b_description))

    @cat.check
    def b_code_in_a(b_codes, a_name) -> bool:
        "a code of B's name appears in A's name with separators removed"
        sa = squash(a_name)
        return any(len(c) >= 4 and c in sa for c in b_codes)

    @cat.check
    def stem_equal(a_key, b_codes, b_desc_codes) -> bool:
        "the model numbers agree up to their last digit (du1055ss / du1055xtss: a colour or region suffix differs)"
        sa = stem(a_key)
        return len(sa) >= 4 and any(stem(c) == sa for c in list(b_codes) + list(b_desc_codes))

    @cat.fn
    def model_sim(a_key, b_codes) -> float:
        "the best similarity between A's model number and a code in B's name (1.0 = the same string)"
        return round(max([SequenceMatcher(None, a_key, y).ratio() for y in b_codes] or [0.0]), 4) if a_key else 0.0

    @cat.check
    def digit_conflict(a_key, b_codes) -> bool:
        "B's closest code differs from A's model number in a digit (ln52a550 / ln52a650): another model of the line"
        best = max(b_codes, key=lambda y: SequenceMatcher(None, a_key, y).ratio(), default=None)
        if best is None or best == a_key or not a_key:
            return False
        da, db = re.sub(r"\D", "", a_key), re.sub(r"\D", "", best)
        return bool(da) and bool(db) and da != db and not (da.startswith(db) or db.startswith(da))

    @cat.fn
    def shared_codes(a_codes, b_codes, b_desc_codes) -> int:
        return len(set(a_codes) & (set(b_codes) | set(b_desc_codes)))

    @cat.check
    def b_has_code(b_codes) -> bool:
        return bool(b_codes)

    @cat.fn
    def name_overlap(a_name, b_name) -> float:
        "idf-weighted share of name words the two offers have in common"
        wa, wb = set(words(a_name)), set(words(b_name))
        if not wa or not wb:
            return 0.0
        both = sum(idf(w) for w in wa & wb)
        return round(both / max(1e-9, sum(idf(w) for w in wa | wb)), 4)

    @cat.fn
    def name_in_b(a_name, b_name, b_description) -> float:
        "share (idf-weighted) of A's name words found anywhere in B"
        wa, wb = set(words(a_name)), set(words(b_name + " " + b_description))
        return round(sum(idf(w) for w in wa & wb) / max(1e-9, sum(idf(w) for w in wa)), 4) if wa else 0.0

    @cat.fn
    def b_in_a(a_name, a_description, b_name) -> float:
        "share (idf-weighted) of B's name words found anywhere in A"
        wa, wb = set(words(a_name + " " + a_description)), set(words(b_name))
        return round(sum(idf(w) for w in wa & wb) / max(1e-9, sum(idf(w) for w in wb)), 4) if wb else 0.0

    @cat.fn
    def char_sim(a_name, b_name) -> float:
        return round(SequenceMatcher(None, squash(a_name), squash(b_name)).ratio(), 4)

    @cat.check
    def both_priced(a_price, b_price) -> bool:
        return bool(a_price) and bool(b_price)

    @cat.fn
    def price_gap(a_price, b_price) -> float:
        "relative price difference; 1.0 when a price is missing (both_priced says which)"
        if not a_price or not b_price:
            return 1.0
        x, y = float(a_price), float(b_price)
        return round(abs(x - y) / max(x, y, 1e-9), 4)

    @cat.check
    def colour_conflict(a_name, b_name, b_description) -> bool:
        ca, cb = set(words(a_name)) & COLOURS, set(words(b_name + " " + b_description)) & COLOURS
        return bool(ca) and bool(cb) and not (ca & cb)

    @cat.check
    def size_conflict(a_name, b_name) -> bool:
        "both names state sizes (50 ', 8gb, 7.2 megapixel) and share none"
        na, nb = set(UNITS.findall(a_name)), set(UNITS.findall(b_name))
        return bool(na) and bool(nb) and not (na & nb)

    q = Question("match", "Are the two offers the same product (the same model, not just the same kind)?",
                 Answer.yes_no())
    return cat, q
