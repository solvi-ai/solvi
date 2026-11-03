"""Descriptive agreement counts on fixed, caller-normalized answers.

No fitting, weighting, significance test or independence/certification claim.
"""
from collections.abc import Mapping, Sequence
from itertools import combinations
from math import sqrt


_CLASSES = ("primary_correct", "primary_wrong")


def _validated(rows, checkers):
    if not isinstance(checkers, Sequence) or isinstance(checkers, (str, bytes)):
        raise ValueError("checkers must be an explicit sequence of unique names")
    try:
        names = tuple(checkers)
    except TypeError as error:
        raise ValueError("checkers must be an explicit sequence of names") from error
    if any(type(n) is not str or not n.strip() for n in names) or len(set(names)) != len(names):
        raise ValueError("checkers must have unique nonempty string names")
    try:
        source = iter(rows)
    except TypeError as error:
        raise ValueError("rows must be an iterable of answer mappings") from error
    prepared = []
    for row in source:
        if not isinstance(row, Mapping) or set(row) != {"gold", "primary", "checkers"}:
            raise ValueError("each row requires exactly gold, primary and checkers")
        answers = row["checkers"]
        if not isinstance(answers, Mapping) or any(n not in names for n in answers):
            raise ValueError("row checkers must be a mapping containing only declared names")
        if any(value is not None and type(value) is not str
               for value in (row["gold"], row["primary"], *answers.values())):
            raise ValueError("answers must be caller-normalized string tokens or None")
        prepared.append({"gold": row["gold"], "primary": row["primary"],
                         "checkers": {n: answers.get(n) for n in names}})
    return prepared, names


def _rate(rows, name):
    observed = [r for r in rows if r["checkers"][name] is not None]
    agree = sum(r["checkers"][name] == r["primary"] for r in observed)
    n = len(observed)
    return {"n_class": len(rows), "n_observed": n, "n_missing": len(rows) - n,
            "n_agree": agree, "n_disagree": n - agree, "p_agree": agree / n if n else None}


def _pair(rows, a, b):
    cells = dict(n00=0, n01=0, n10=0, n11=0)
    missing_a = missing_b = missing_both = 0
    for row in rows:
        av, bv = row["checkers"][a], row["checkers"][b]
        missing_a += av is None
        missing_b += bv is None
        missing_both += av is None and bv is None
        if av is not None and bv is not None:
            cells[f"n{int(av == row['primary'])}{int(bv == row['primary'])}"] += 1
    n00, n01, n10, n11 = (cells[k] for k in ("n00", "n01", "n10", "n11"))
    n = sum(cells.values())
    denom = (n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)
    reason = "no_pairs" if not n else "constant_agreement" if not denom else None
    return {"n_class": len(rows), "n_observed": n, "missing_a": missing_a,
            "missing_b": missing_b, "missing_both": missing_both, "cells": cells,
            "phi": (n11 * n00 - n10 * n01) / sqrt(denom) if denom else None,
            "phi_reason": reason}


def _same_wrong(rows, a, b):
    n = both = same = 0
    for row in rows:
        av = row["primary"] if a is None else row["checkers"][a]
        bv = row["checkers"][b]
        if av is None or bv is None:
            continue
        n += 1
        if av != row["gold"] and bv != row["gold"]:
            both += 1
            same += av == bv
    return {"n_observed": n, "n_both_wrong": both, "n_same_wrong": same,
            "p_same_wrong": same / both if both else None}


def agreement_diagnostics(rows, *, checkers):
    """Describe agreement signals, conditional on the primary answer's correctness.

    Each row has exactly ``gold``, ``primary`` and ``checkers`` (a name → answer
    mapping). Answers are exact caller-normalized string tokens; None or a missing
    checker key means missing. No parsing, tolerance or case folding is applied.
    Explicit ``checkers`` names must be unique. All rows are validated before any
    statistics are returned; inputs are not mutated.

    Rows missing gold or primary are excluded with explicit counts. Per-checker
    agreement rates use that checker's observed denominator. Pairwise 2×2 tables
    and phi use jointly observed checks in each primary-correct/wrong class.
    Undefined rates/phi are None (phi_reason explains empty/constant cases).
    Same-wrong probabilities divide matching wrong values by jointly wrong
    observations, both checker–checker and primary–checker.

    These are descriptive counts on the supplied rows, not learned weights,
    uncertainty estimates, tests of independence or a statistical certificate.
    """
    rows, names = _validated(rows, checkers)
    eligible = [r for r in rows if r["gold"] is not None and r["primary"] is not None]
    grouped = {"primary_correct": [], "primary_wrong": []}
    for row in eligible:
        grouped["primary_correct" if row["primary"] == row["gold"] else "primary_wrong"].append(row)
    result = {"comparator": "exact_normalized_string",
              "input": {"n_rows": len(rows), "n_eligible": len(eligible),
                        "missing_gold": sum(r["gold"] is None for r in rows),
                        "missing_primary": sum(r["primary"] is None for r in rows),
                        "missing_both": sum(r["gold"] is None and r["primary"] is None for r in rows)},
              "classes": {k: len(grouped[k]) for k in _CLASSES}, "checkers": {}, "pairs": []}
    for name in names:
        result["checkers"][name] = {k: _rate(grouped[k], name) for k in _CLASSES}
        result["checkers"][name]["same_wrong_with_primary"] = _same_wrong(eligible, None, name)
    for a, b in combinations(names, 2):
        pair = {"a": a, "b": b, **{k: _pair(grouped[k], a, b) for k in _CLASSES},
                "same_wrong": _same_wrong(eligible, a, b)}
        result["pairs"].append(pair)
    return result


__all__ = ["agreement_diagnostics"]
