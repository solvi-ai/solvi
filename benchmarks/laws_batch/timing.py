"""Diagnostic timing under PROTOCOL.md, not a general performance claim."""
import json
import statistics
import time
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


g = Grammar([Atom(f"b{i}", lambda s, i=i: bool(s & (1 << i))) for i in range(6)], k=3)
b = Buffer(g)
for i in range(2000):
    b.add(Transition(i % 64, "a", i % 2))
ls = LawSet(g, base=lambda s, a: 0)
conditions = list(g.conditions(g.by_name))
for i in range(40):
    ls.add(Candidate("a", conditions[i], i % 2))
cs = [Candidate("a", conditions[i % len(conditions)], i % 2) for i in range(200)]

def scalar():
    return [ls.score(c, b) for c in cs]


def batch():
    return ls.score_many(cs, b, stable=True)


result = {"rows": len(b), "laws": len(ls.laws), "candidates": len(cs), "pairs": []}
path = Path(__file__).with_name("timing.json")
assert result["laws"] == 40
expected = scalar()
assert batch() == expected
for i in range(7):
    cell = {}
    for name, fn in (("scalar", scalar), ("batch", batch))[::1 if i % 2 == 0 else -1]:
        start = time.perf_counter()
        actual = fn()
        cell[name] = time.perf_counter() - start
        cell[name + "_equal"] = actual == expected
    result["pairs"].append(cell)
    path.write_text(json.dumps(result, indent=2) + "\n")
result["median_paired_ratio"] = statistics.median(p["batch"] / p["scalar"] for p in result["pairs"])
path.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
