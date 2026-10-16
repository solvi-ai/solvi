"""Narrows / widens proved over the rule text (solvi.experimental.changes.prove_direction, the smt extra) on the
gallery's own rules: every top-level function of gallery/*/task.py with a number in a comparison is mutated (each such
threshold shifted by ±1, or ±10 % when it is not a whole number, and each < / <= / > / >= made strict or not), and
every (rule, mutant) pair is proved once per outcome — "accepts" (returns a value that is not a refusal) and each string
constant the rule returns. Reported: the share of pairs the prover decides, the reasons it does not, milliseconds per
proof, and two soundness checks — every witness input is run through both Python functions and must show what the proof
says, and random inputs (around the constants of the rule) must never contradict a decided proof.

    python benchmarks/changes_prove.py            # needs z3: pip install "solvi[smt]"; exits 1 on a soundness failure
"""
from __future__ import annotations

import ast
import collections
import random
import statistics
import sys
import time
import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.changes import is_refusal, prove_direction

ROOT = Path(__file__).resolve().parent.parent
FLIP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}


def functions(path):
    """(name, text) per top-level function of a task file; the text carries the module's literal constants."""
    tree = ast.parse(path.read_text())
    consts = []
    for st in tree.body:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            try:
                ast.literal_eval(st.value)
                consts.append(ast.unparse(st))
            except ValueError:
                pass
    for st in tree.body:
        if isinstance(st, ast.FunctionDef):
            st.decorator_list = []
            yield st.name, "\n".join(consts + [ast.unparse(st)])


def mutants(text):
    """The rule's mutants: each number in a comparison shifted both ways, each order comparison flipped."""
    tree = ast.parse(text)
    fn = tree.body[-1]
    sites = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Compare):
            for side in [node.left, *node.comparators]:
                if isinstance(side, ast.Constant) and type(side.value) in (int, float):
                    sites.append(("num", side))
            for i, op in enumerate(node.ops):
                if type(op) in FLIP:
                    sites.append(("op", (node, i)))
    out = []
    for kind, site in sites:
        if kind == "num":
            v = site.value
            for d in ((1, -1) if float(v).is_integer() else (0.1 * abs(v) or 0.1, -(0.1 * abs(v) or 0.1))):
                site.value = type(v)(v + d) if isinstance(v, int) else v + d
                out.append(ast.unparse(tree))
                site.value = v
        else:
            node, i = site
            op = node.ops[i]
            node.ops[i] = FLIP[type(op)]()
            out.append(ast.unparse(tree))
            node.ops[i] = op
    return out


def returned_strings(text):
    fn = ast.parse(text).body[-1]
    return sorted({n.value.value for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.Constant)
                   and isinstance(n.value.value, str)} |
                  {x.value for n in ast.walk(fn) if isinstance(n, ast.Return) and isinstance(n.value, ast.IfExp)
                   for x in ast.walk(n.value) if isinstance(x, ast.Constant) and isinstance(x.value, str)})


def compiled(text):
    ns = {}
    try:
        exec(compile(text, "<rule>", "exec"), ns)                              # the gallery's own code
    except Exception:                                                        # noqa: BLE001 — a rule that needs the rest of its module
        return None
    return ns[ast.parse(text).body[-1].name]


def accepted(fn, x, outcome):
    try:
        y = fn(**x)
    except Exception:                                                        # noqa: BLE001 — a helper it calls, a wrong type
        return None
    if outcome is not None:
        return y == outcome
    return not (y is None or y is False or (not isinstance(y, bool) and is_refusal(y)))


def samples(text, sorts, rng, n=300):
    nums = sorted({float(c.value) for c in ast.walk(ast.parse(text)) if isinstance(c, ast.Constant)
                   and type(c.value) in (int, float)} | {0.0})
    strs = sorted({c.value for c in ast.walk(ast.parse(text)) if isinstance(c, ast.Constant)
                   and isinstance(c.value, str)}) + ["zz-other"]
    pool = {"Real": [v + d for v in nums for d in (-1, -0.5, 0, 0.5, 1)], "Int": [int(v) + d for v in nums for d in (-1, 0, 1)],
            "String": strs, "Bool": [True, False]}
    for _ in range(n):
        yield {k: rng.choice(pool[s]) for k, s in sorts.items()}


def main():
    rng = random.Random(0)
    rules = [(p.parent.name, name, text) for p in sorted((ROOT / "gallery").glob("*/task.py"))
             for name, text in functions(p)]
    mutable = [(g, n, t, m) for g, n, t in rules for m in [mutants(t)] if m]
    counts, reasons, ms, failures = collections.Counter(), collections.Counter(), [], []
    witnessed = sampled = 0
    for g, name, text, ms_ in mutable:
        old = compiled(text)
        for mut in ms_:
            new = compiled(mut)
            for outcome in [None] + returned_strings(text):
                kw = {} if outcome is None else {"outcome": outcome}
                t0 = time.perf_counter()
                r = prove_direction(text, mut, **kw)
                ms.append((time.perf_counter() - t0) * 1000)
                counts[r["proved"]] += 1
                if r["proved"] == "unknown":
                    reasons[r["reason"].split(" (")[0].split(":")[0][:60]] += 1
                    continue
                if old is None or new is None:
                    continue
                for k, (want_new, want_old) in (("widens", (True, False)), ("narrows", (False, True))):
                    w = r.get(f"witness_{k}")
                    if w is not None:
                        got = (accepted(new, w, outcome), accepted(old, w, outcome))
                        if None not in got:
                            witnessed += 1
                            if got != (want_new, want_old):
                                failures.append((g, name, outcome, k, w, got))
                for x in samples(text, r["sorts"], rng):
                    a, b = accepted(old, x, outcome), accepted(new, x, outcome)
                    if a is None or b is None:
                        continue
                    sampled += 1
                    if (b and not a and r["proved"] not in ("widens", "both")) or \
                            (a and not b and r["proved"] not in ("narrows", "both")):
                        failures.append((g, name, outcome, r["proved"], x, (b, a)))
    total = sum(counts.values())
    decided = total - counts["unknown"]
    print(f"gallery: {len(rules)} rule functions, {len(mutable)} with a number in a comparison, "
          f"{sum(len(m) for *_, m in mutable)} mutants, {total} proofs (per outcome)")
    print(f"decided {decided} of {total} ({decided / max(total, 1):.0%}): " +
          ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    print("not decided, by reason: " + "; ".join(f"{k} ×{v}" for k, v in reasons.most_common(8)))
    if ms:
        ms.sort()
        print(f"ms per proof: median {statistics.median(ms):.1f}, p95 {ms[int(0.95 * (len(ms) - 1))]:.1f}, "
              f"max {ms[-1]:.1f}")
    print(f"soundness: {witnessed} witnesses run through both functions, {sampled} random inputs checked, "
          f"{len(failures)} contradictions")
    for f in failures[:10]:
        print("  CONTRADICTION", f)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
