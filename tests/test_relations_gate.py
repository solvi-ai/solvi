"""The relation-algebra enumerator's fidelity gate and the compiled-code benchmark (run with -m slow).

1. benchmarks/relations_battery.py: a pre-registered battery of 28 random rules over three relations (two of them the
   same rule), with the source's own example seeds and graph families (32 random graphs of 6–12 nodes; held-out graphs
   of 32 and 64 nodes of three families). Bar: at least 22 of 28 found within 7 nodes, every find exact on the
   held-out graphs of the other families, peak memory of the process at most 3 GB. The rule file is read at test
   time from SOLVI_RULE_BATTERY (or the research checkout next to this repository); absent → skipped.
2. benchmarks/relations_compile.py: found rules executed as plain set code — closure, grandparent and sibling on a
   forest of 100,000 nodes pair for pair equal to hand-written set code and at most 2× its time; compiled programs
   against the interpreter on random worlds of 1,000 nodes (0 differing pairs); a compiled law module against
   LawSet.predict on 100,000 rows (0 differences, both times reported)."""
import importlib.util
import json
import os
import resource
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


def _rules_file():
    env = os.environ.get("SOLVI_RULE_BATTERY")
    if env:
        return Path(env) if Path(env).exists() else None
    for up in ROOT.parents:
        p = up / "kelly_nn" / "albert_lab" / "iter12_rules.json"
        if p.exists():
            return p
    return None


@pytest.mark.slow
def test_the_enumerator_passes_the_battery_gate():
    path = _rules_file()
    if path is None:
        pytest.skip("the rule battery file is not on this machine (set SOLVI_RULE_BATTERY)")
    rules = json.loads(path.read_text())["rules"]
    assert len(rules) == 28
    r = _load("relations_battery").battery(rules, max_size=7)
    found = [x for x in r["rules"] if x["found"]]
    print(json.dumps({k: v for k, v in r.items() if k != "rules"}), [(x["id"], x["program"]) for x in found])
    assert len(found) >= 22
    assert all(x["generalized"] for x in found)
    assert all(x["size"] <= 7 for x in found)
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    assert peak_gb <= 3.0 and r["peak_rss_gb"] <= 3.0


@pytest.mark.slow
def test_an_inexpressible_rule_is_never_falsely_accepted():
    r = _load("relations_battery").controls(max_size=7)
    k4 = next(x for x in r["rules"] if x["task"] == "4-clique")
    assert r["false_accepts"] == 0 and not k4["generalized"]
    print(json.dumps(r, ensure_ascii=False))


@pytest.mark.slow
def test_compiled_forest_rules_equal_hand_written_set_code_within_twice_its_time():
    r = _load("relations_compile").forest(n=100_000, seed=0)
    for rule in ("closure", "grand", "sibling"):
        x = r[rule]
        assert x["equal"] and x["pairs"] > 10_000, (rule, x)
        assert x["compiled_s"] <= 2.0 * x["hand_s"], (rule, x)


@pytest.mark.slow
def test_compiled_programs_agree_with_the_interpreter_on_large_worlds():
    t0 = time.time()
    r = _load("relations_compile").differential(worlds=100, n=1000, seed=0)
    assert r["differing_pairs"] == 0 and r["checks"] >= 100
    assert time.time() - t0 < 300                       # the dense interpreter's closures at 1,000 nodes dominate


@pytest.mark.slow
def test_a_compiled_law_module_agrees_with_the_interpreter_on_100k_rows():
    r = _load("relations_compile").laws(rows=100_000, seed=0)
    assert r["rows"] == 100_000 and r["differences"] == 0
    assert r["compiled_s"] > 0 and r["interpreter_s"] > 0
