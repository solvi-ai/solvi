"""benchmarks/laws_toy.py: the law loop on two deterministic toys (run with -m slow; each well under 60 s).

1. A gridworld with written rules (walkable, blocked and pickup cells; chopping a tree; mining stone only with wood):
   the loop recovers the rules from at most 500 transitions, and a random proposer with the same held-out test is no
   worse than the enumerating one by more than 1 point on the mean of five seeds — the test does the work, the
   mismatch is a compact way to propose.
2. A stateless path policy (allowed / denied by path class, unknown to the agent): on every one of five seeds, probes
   planned from each mismatch leave fewer refusals over two sessions than random probes at the same budget, halve the
   refusals from the first session to the second, and never skip an allowed call."""
import importlib.util
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    spec = importlib.util.spec_from_file_location("laws_toy", ROOT / "benchmarks" / "laws_toy.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_the_gridworld_rules_are_recovered_and_random_proposals_tie():
    t0 = time.time()
    r = _toy().gridworld()
    assert time.time() - t0 < 60
    assert r["train_transitions"] <= 500 and len(r["seeds"]) == 5
    assert r["arms"]["written"]["accuracy"] == 1.0
    base, laws, rnd = (r["arms"][k]["accuracy"] for k in ("base", "laws", "random"))
    assert laws >= 0.97 and r["gap_closed"] >= 0.9 and laws > base
    assert rnd >= laws - 0.01                                     # the test does the work
    assert r["arms"]["laws"]["laws"] <= r["arms"]["random"]["laws"]      # the mismatch proposes compactly


@pytest.mark.slow
def test_the_policy_is_learned_with_fewer_refusals_from_planned_probes():
    t0 = time.time()
    r = _toy().policy()
    assert time.time() - t0 < 60
    assert len(r["seeds"]) == 5
    for run in r["runs"]:
        mm, rnd = run["arms"]["mismatch"], run["arms"]["random"]
        assert mm["probes"] == rnd["probes"] > 0                  # the same budget
        assert mm["refusals_2"] <= 0.5 * mm["refusals_1"] and mm["wrongly_skipped"] == 0
        assert mm["refusals"] < rnd["refusals"]


def test_the_toy_script_is_importable_and_says_what_it_measures():
    mod = _toy()
    assert callable(mod.gridworld) and callable(mod.policy)
    assert "written" in (mod.__doc__ or "")
