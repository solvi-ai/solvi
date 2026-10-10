"""Alignment of two hypotheses (solvi.core.specialist.align): two readers' readings matched one-to-one by the
overlap of their spans — time intervals (speech), boxes (receipts), character offsets (two extractors of one text)."""
import json
import random
import re
from pathlib import Path

import pytest

from solvi.core.specialist import Agreed, Alignment, Reading, align, box_iou, interval_overlap

FIX = Path(__file__).parent / "fixtures" / "align"


def R(value, *span, confidence=None):
    return Reading(value, tuple(span), confidence)


def values(al):
    return {"agreed": [[x.a.value, x.b.value] for x in al.agreed],
            "conflicts": [[a.value, b.value] for a, b, _ in al.conflicts],
            "only_a": [r.value for r in al.only_a], "only_b": [r.value for r in al.only_b]}


def test_the_overlap_functions():
    assert interval_overlap((0, 2), (0, 2)) == 1.0
    assert interval_overlap((0, 1), (0, 4)) == 1.0                 # over the shorter span
    assert interval_overlap((0, 2), (1, 5)) == 0.5 and interval_overlap((0, 1), (1, 2)) == 0.0
    assert interval_overlap((1, 1), (0, 2)) == 0.0                 # an empty span overlaps nothing
    assert box_iou((0, 0, 2, 2), (0, 0, 2, 2)) == 1.0
    assert box_iou((0, 0, 2, 2), (1, 0, 3, 2)) == pytest.approx(1 / 3)
    assert box_iou((0, 0, 1, 1), (1, 0, 2, 1)) == 0.0              # touching boxes share no area


def test_intervals_agreed_apart_conflicting_and_filtered_by_confidence():
    full = align([R("hello", 0.0, 1.0)], [R("hello", 0.0, 1.0)])
    assert isinstance(full, Alignment) and len(full.agreed) == 1 and full.agreed[0].overlap == 1.0
    x = full.agreed[0]
    assert isinstance(x, Agreed) and x.value == "hello" and x.a.span == (0.0, 1.0) and x.b.span == (0.0, 1.0)
    low = align([R("a", 0.0, 1.0)], [R("a", 0.6, 2.0)])                            # 0.4 of the shorter span
    assert values(low) == {"agreed": [], "conflicts": [], "only_a": ["a"], "only_b": ["a"]}
    diff = align([R("1990", 1.0, 2.0)], [R("1999", 1.0, 2.0)])
    assert values(diff)["conflicts"] == [["1990", "1999"]] and diff.conflicts[0][2] == 1.0
    conf = align([R("x", 0, 1, confidence=0.2), R("y", 2, 3)], [R("x", 0, 1), R("y", 2, 3, confidence=0.9)],
                 min_a=0.5, min_b=0.95)
    assert values(conf) == {"agreed": [], "conflicts": [], "only_a": ["y"], "only_b": ["x"]}   # dropped before matching


def test_two_readings_competing_for_one_the_larger_overlap_wins():
    al = align([R("a1", 0.0, 1.0), R("a2", 0.8, 2.0)], [R("b", 0.5, 2.0)])
    assert values(al) == {"agreed": [], "conflicts": [["a2", "b"]], "only_a": ["a1"], "only_b": []}
    tie = align([R("v", 0.0, 1.0), R("v", 0.0, 1.0)], [R("v", 0.0, 1.0)])            # a tie: a's order decides
    assert len(tie.agreed) == 1 and tie.agreed[0].a is tie.agreed[0].a and len(tie.only_a) == 1
    assert tie.only_a[0] is not tie.agreed[0].a


def test_boxes():
    al = align([R("TOTAL", 0, 0, 10, 10), R("TAX", 20, 0, 30, 10)],
               [R("TOTAL", 1, 1, 10, 10), R("TAX", 30, 0, 40, 10)], overlap=box_iou)
    assert values(al) == {"agreed": [["TOTAL", "TOTAL"]], "conflicts": [], "only_a": ["TAX"], "only_b": ["TAX"]}


def test_character_offsets_of_two_extractors_over_one_text():
    text = "Refund 120 EUR for order 4471, charged twice."
    q = lambda s: R(s, text.index(s), text.index(s) + len(s))                      # noqa: E731
    al = align([q("120 EUR"), q("4471")], [q("120"), q("order 4471")])
    assert values(al)["agreed"] == [] and values(al)["conflicts"] == [["120 EUR", "120"], ["4471", "order 4471"]]
    same = align([q("120 EUR"), q("4471")], [q("120"), q("order 4471")],
                 same=lambda a, b: re.findall(r"\d+", a) == re.findall(r"\d+", b))
    assert [x.value for x in same.agreed] == ["120 EUR", "4471"]                     # the value is a's


def test_a_normaliser_as_same():
    norm = lambda v: v.strip().replace(",", "")                                    # noqa: E731
    al = align([R(" 1,400 ", 0, 1)], [R("1400", 0, 1)], same=lambda a, b: norm(a) == norm(b))
    assert len(al.agreed) == 1 and not al.conflicts
    assert values(align([R(" 1,400 ", 0, 1)], [R("1400", 0, 1)]))["conflicts"] == [[" 1,400 ", "1400"]]


def test_deterministic_whatever_the_order_of_the_inputs():
    rng = random.Random(0)
    a = [R(f"w{i}", i * 0.5, i * 0.5 + 0.7, confidence=0.9) for i in range(12)]
    b = [R(f"w{i}", i * 0.5 + 0.1, i * 0.5 + 0.6) for i in range(12)]
    first = align(a, b)
    key = lambda al: (sorted((x.a.value, x.b.value) for x in al.agreed),                # noqa: E731
                      sorted((p.value, q.value) for p, q, _ in al.conflicts))
    for _ in range(5):
        sa, sb = a[:], b[:]
        rng.shuffle(sa)
        rng.shuffle(sb)
        assert key(align(sa, sb)) == key(first)
    assert key(align(a, b)) == key(first) and len(first.agreed) == 12


def _fixture(name):
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_real_rows_of_the_speech_stand():
    d = _fixture("stt_fleurs_ru.json")
    a = [Reading(w["text"], (w["start"], w["end"]), w["p"]) for w in d["a"]]
    b = [Reading(w["text"], (w["start"], w["end"]), w["c"]) for w in d["b"]]
    norm = lambda v: re.sub(r"[^\w]", "", v.lower())                                # noqa: E731
    al = align(a, b, overlap=interval_overlap, same=lambda x, y: norm(x) == norm(y))
    assert values(al) == d["expected"]


def test_real_rows_of_the_receipts_stand():
    d = _fixture("ocr_cord_test_000.json")
    a = [Reading(w["text"], tuple(w["box"]), w["conf"]) for w in d["a"]]
    b = [Reading(w["text"], tuple(w["box"]), w["conf"]) for w in d["b"]]
    al = align(a, b, overlap=box_iou, min_b=0.1)
    assert values(al) == d["expected"]
    assert round(al.agreed[0].overlap, 3) == 0.638 and al.agreed[0].a.confidence == 1.0 and al.agreed[0].b.confidence == 0.83
