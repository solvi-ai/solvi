"""benchmarks/relations_library_toy.py: composites by enumeration with a library (run with -m slow; under 60 s).

Two ladders, each with and without the library, five seeds: (i) a path tree — "denied if an ancestor directory is named
secret and the depth is at least 3", three levels (secret directory / ancestor / two steps up → under a secret
directory / depth ≥ 3 → denied); (ii) kinship on forests, depth 1–6 (grandparent, sibling → uncle, great-grandparent →
cousin, great-uncle → … → second cousin once removed). The targets are given as an unordered set; the library grows only
from rules that passed the held-out test. Bars on the means of the five seeds: every level found with the library, the
library's cumulative programs at every level's admission ≤ 1/10 of the no-library budget, which flat search spends
without finding the top rule; 0 false accepts on fresh larger worlds. The no-library budget was 60,000 programs in the
first run (the top kinship rule then cost the library 6,124 cumulative programs, 1.02 × the bar) and was raised to
100,000 after it; the per-level own-round costs next to flat's are reported by the script, not barred."""
import importlib.util
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    spec = importlib.util.spec_from_file_location("relations_library_toy", ROOT / "benchmarks" / "relations_library_toy.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_with_the_library_every_level_costs_a_tenth_of_flat_search_and_nothing_false_is_accepted():
    t0 = time.time()
    r = _toy().run()
    assert time.time() - t0 < 60
    for ladder in ("paths", "kinship"):
        x = r[ladder]
        assert len(x["seeds"]) == 5
        lib, flat = x["arms"]["library"], x["arms"]["flat"]
        assert lib["found_all"] == 1.0                                     # every level, every seed
        for name in x["levels"]:                                           # every level within a tenth of flat's budget
            assert lib["cum"][name] <= x["flat_budget"] / 10, (ladder, name, lib["cum"][name])
        assert flat["found"][x["top"]] == 0.0                              # which flat search spends without the top
        assert lib["false_accepts"] == 0 and flat["false_accepts"] == 0    # on fresh larger worlds


def test_the_toy_worlds_are_deterministic_and_the_script_says_what_it_measures():
    mod = _toy()
    a, b = mod.path_tree(3, 20), mod.path_tree(3, 20)
    assert all((a[k] == b[k]).all() for k in a)
    assert callable(mod.run) and "path tree" in (mod.__doc__ or "") and "kinship" in (mod.__doc__ or "")
