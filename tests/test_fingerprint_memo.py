"""Nested code fingerprints are memoised within one catalog_fingerprint (and one top-level code_fingerprint) call: a
helper that many rules call is hashed once, not once per rule. The memo must give exactly the fingerprints the plain
recursion gives — including the path-dependent "recursive" marks of cycles and diamonds — and be faster on a catalog
whose rules share a deep helper graph."""
import importlib.util
import random
import time

import pytest

from solvi import Catalog
from solvi.core import provenance


def _module(tmp_path, name, helpers, rules, seed):
    """A module of `helpers` helper functions calling each other at random (cycles and diamonds included) and a
    build(cat) registering `rules` rules over them."""
    rng = random.Random(seed)
    lines = ["LIMIT = 3", ""]
    for i in range(helpers):
        calls = sorted(rng.sample(range(helpers), k=min(helpers, rng.randint(0, 3))))
        body = " + ".join(f"h{j}(x - 1)" for j in calls) or "0"
        lines += [f"def h{i}(x):", f"    if x <= {i % 4}:", "        return LIMIT",
                  f"    return {body}", ""]
    lines += ["def build(cat):"]
    for p in range(rules):
        uses = rng.sample(range(helpers), k=min(helpers, rng.randint(1, 3)))
        lines += [f"    @cat.rule('q{p}')", f"    def q{p}(x) -> int:",
                  "        return " + " + ".join(f"h{j}(x)" for j in uses) + f" + {p}"]
    lines += ["    return cat", ""]
    f = tmp_path / f"{name}.py"
    f.write_text("\n".join(lines))
    spec = importlib.util.spec_from_file_location(name, f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fresh_fp(build, memo):
    provenance._CODE = None                          # no per-function cache from an earlier call
    old = provenance._MEMO_ON
    provenance._MEMO_ON = memo
    try:
        cat = build(Catalog())
        t = time.perf_counter()
        out = provenance.catalog_fingerprint(cat)
        return out, time.perf_counter() - t
    finally:
        provenance._MEMO_ON = old


@pytest.mark.parametrize("seed", range(12))
def test_the_memo_gives_the_same_fingerprints_on_random_call_graphs(tmp_path, seed):
    mod = _module(tmp_path, f"fpgraph{seed}", helpers=3 + seed, rules=6 + seed, seed=seed)
    plain, _ = _fresh_fp(mod.build, memo=False)
    memo, _ = _fresh_fp(mod.build, memo=True)
    assert memo == plain


def test_the_memo_gives_the_same_code_fingerprints_for_each_helper(tmp_path):
    mod = _module(tmp_path, "fphelpers", helpers=14, rules=1, seed=7)
    got = {}
    for on in (False, True):
        provenance._CODE = None
        old, provenance._MEMO_ON = provenance._MEMO_ON, on
        try:
            got[on] = [provenance.code_fingerprint(getattr(mod, f"h{i}")) for i in range(14)]
        finally:
            provenance._MEMO_ON = old
    assert got[True] == got[False]


def test_the_memo_is_faster_on_shared_helpers(tmp_path):
    rng_lines = ["LIMIT = 3", ""]
    for i in range(30):                              # a chain with diamonds: h_i calls h_{i+1} and h_{i+2}
        calls = " + ".join(f"h{j}(x)" for j in range(i + 1, min(30, i + 3))) or "0"
        rng_lines += [f"def h{i}(x):", f"    y = x * {i} + LIMIT", f"    return y + {calls}", ""]
    rng_lines += ["def build(cat):"]
    for p in range(120):
        rng_lines += [f"    @cat.rule('q{p}')", f"    def q{p}(x) -> int:", f"        return h0(x) + h{p % 30}(x) + {p}"]
    rng_lines += ["    return cat", ""]
    f = tmp_path / "fpchain.py"
    f.write_text("\n".join(rng_lines))
    spec = importlib.util.spec_from_file_location("fpchain", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    plain, t_plain = _fresh_fp(mod.build, memo=False)
    memo, t_memo = _fresh_fp(mod.build, memo=True)
    assert memo == plain
    assert t_plain / t_memo > 3, (t_plain, t_memo)
