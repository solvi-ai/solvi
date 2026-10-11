"""Guarded rules learned by counting on NoRA (Das et al., NeurIPS 2025), the no-ambiguity configuration — the held-out
part of the training file:

    python benchmarks/relations_rules_nora.py --data DIR     # DIR holds training_data_allNoAmb.csv (or SOLVI_NORA=DIR)

Data: only the training file is read, in place; the test files are not opened. Its rows are split by a hash of the
row index — sha256("<salt>:<index>") mod 100: < 70 fit, < 85 check, else held-out (the file is ordered by relation, so
never by position; --salt, default "nora"). A row is one story graph (story_edges, edge_types; a self-loop
is_* / no_* is a unary fact) and one query pair; its label is the set {query_relation} ∪ other_relationships.

Settings: `strict` — GuardedRules(majority=1.0, support=10); `default` — GuardedRules(majority=0.9, support=2); both with
residual votes, the negative stratum, the check part as `check=` with check_min=2 (two confirmations), the joint recount
after each round, `complements={"Not_living_in": "living_in"}`, at most 12 rounds. Control: the most frequent label
set of the fit part per entity kinds of the query pair (person / place / other, from the unary facts). Metrics on the
held-out part: exact set match over every pair (an empty answer is a refusal and counts as wrong), precision among the
answered (exact match among non-empty answers), coverage. One deterministic run.

Measured (2026-10, this machine, CPU, one run per split; exact / precision among answered / coverage):
  --salt nora   (fit 9,249, check 2,023, held-out 1,972): strict 0.328 / 0.681 / 0.482, 469 rules (7 rounds, 565 s);
                default 0.274 / 0.568 / 0.482, 720 rules (9 rounds, 2,889 s); control 0.335.
  --salt nora-b (fit 9,300, check 1,971, held-out 1,973): strict 0.485 / 0.743 / 0.652, 436 rules (7 rounds, 581 s);
                default 0.398 / 0.609 / 0.653, 838 rules (12 rounds, 3,451 s); control 0.341.
The «elsewhere» relation (Not_living_in, about a third of the rows) is refused by every setting without a declaration;
the strict setting varies with the split (whether the living_in chain rule survives).
The test files were scored once in a separate pre-registered run (strict: BL-na 0.734, D-na 0.477 at coverage 0.496,
OPEC-na 0.093; see docs/api/relations.md); this script does not open them.
--settings strict-closed: the strict setting with functional={"living_in": "person"} (the closed-world stratum; the same
rules). Measured (2026-10, one run per split): --salt nora 0.337 / 0.684 / 0.492, rows holding Not_living_in answered
exactly 17 of 590 (0 without), other rows answered wrongly 307 (303 without); --salt nora-b 0.674 / 0.798 / 0.844,
373 of 588, 301 (296 without). The stratum needs a known living_in for the subject, here always derived through the
living_in chain rule, which the strict setting keeps on nora-b and not on nora."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import os
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import GuardedRules, Story

csv.field_size_limit(sys.maxsize)
FILE = "training_data_allNoAmb.csv"
SETTINGS = {"strict": dict(majority=1.0, support=10), "default": dict(majority=0.9, support=2),
            "strict-closed": dict(majority=1.0, support=10, functional={"living_in": "person"})}


def part(i: int, salt: str = "nora") -> str:
    h = int(hashlib.sha256(f"{salt}:{i}".encode()).hexdigest(), 16) % 100
    return "fit" if h < 70 else ("check" if h < 85 else "held")


def read(path: Path, salt: str = "nora"):
    parts = {"fit": [], "check": [], "held": []}
    with open(path, encoding="utf-8", newline="") as f:
        for i, rec in enumerate(csv.DictReader(f)):
            edges = [tuple(e) for e in ast.literal_eval(rec["story_edges"])]
            st = Story.from_edges(edges, ast.literal_eval(rec["edge_types"]))
            y = frozenset([rec["query_relation"]] + list(ast.literal_eval(rec["other_relationships"])))
            parts[part(i, salt)].append((st, tuple(ast.literal_eval(rec["query_edge"])), y))
    return parts


def kinds(st, q):
    def k(e):
        us = {u for x, u in st.unary if x == e}
        return "person" if "is_person" in us else ("place" if "is_place" in us else "other")
    return (k(q[0]), k(q[1]))


def score(preds, held):
    n = len(held)
    right = sum(p == y for p, (_, _, y) in zip(preds, held))
    ans = [(p, y) for p, (_, _, y) in zip(preds, held) if p]
    return right / n, (sum(p == y for p, y in ans) / len(ans) if ans else None), len(ans) / n


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.environ.get("SOLVI_NORA"))
    ap.add_argument("--settings", default="strict,default", help="strict, default, strict-closed (strict + "
                    "functional={'living_in': 'person'}: the closed-world stratum)")
    ap.add_argument("--salt", default="nora", help="the split: sha256('<salt>:<row index>')")
    a = ap.parse_args(argv)
    if not a.data:
        ap.error("pass --data DIR (or set SOLVI_NORA): the directory holding " + FILE)
    parts = read(Path(a.data) / FILE, a.salt)
    fit, check, held = parts["fit"], parts["check"], parts["held"]
    print(f"split {a.salt!r}: fit {len(fit)}, check {len(check)}, held-out {len(held)} query pairs")
    by = {}
    for st, q, y in fit:
        by.setdefault(kinds(st, q), Counter())[y] += 1
    top = {k: c.most_common(1)[0][0] for k, c in by.items()}
    ctrl = score([top.get(kinds(st, q), frozenset()) for st, q, _ in held], held)
    print(f"control (label set per entity kinds): exact {ctrl[0]:.3f}, coverage {ctrl[2]:.3f}")
    comp = {"Not_living_in": "living_in"}
    for name in [s for s in a.settings.split(",") if s]:
        t0 = time.time()
        g = GuardedRules(**SETTINGS[name], complements=comp, check_min=2, rounds=12).fit(fit, check=check)
        preds = [g.predict(st, q) for st, q, _ in held]
        ex, pr, cov = score(preds, held)
        nl = [(p, y) for p, (_, _, y) in zip(preds, held) if "Not_living_in" in y]
        other = [(p, y) for p, (_, _, y) in zip(preds, held) if "Not_living_in" not in y]
        print(f"{name}: exact {ex:.3f}, precision {pr:.3f}, coverage {cov:.3f}; {len(g)} rules {g.by_template()}, "
              f"{g.rounds} rounds, {time.time() - t0:.0f} s; rows holding Not_living_in {len(nl)}, answered exactly "
              f"{sum(p == y for p, y in nl)}; other rows {len(other)}, answered wrongly "
              f"{sum(bool(p) and p != y for p, y in other)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
