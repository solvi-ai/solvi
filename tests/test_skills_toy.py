"""benchmarks/skills_toy.py: learning in a deterministic gridworld described by code (run with -m slow; under 60 s).

A key, a door, a gem behind the door and an exit; fresh layouts every episode from a seed (positions, decoration, now and
then a boulder). Four arms on 20 episodes × 5 seeds: `skills` (plan on the exact model, compile verified plans, act
alone under the certificate), `plan-always` (System 2 at every goal: the ceiling of goals), `cases` (replay the action
sequence that worked from the same start state: no condition, no certificate) and `scrambled` (random action sequences
compiled without verification). Bars on the means of the five seeds."""
import importlib.util
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _toy():
    spec = importlib.util.spec_from_file_location("skills_toy", ROOT / "benchmarks" / "skills_toy.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_skills_grow_system_1_under_the_promise_and_the_controls_do_not():
    t0 = time.time()
    r = _toy().run()
    assert time.time() - t0 < 60
    assert len(r["seeds"]) == 5 and r["episodes"] == 20
    a = r["arms"]
    sk, pa, ca, sc = a["skills"], a["plan-always"], a["cases"], a["scrambled"]
    assert sk["s1_share_last5"] >= 0.8                              # System 1 grows
    assert sk["goals"] >= 0.9 * pa["goals"]                         # without losing goals
    assert sk["s1_failure_rate"] <= r["promise"]                    # the promise on System 1's decisions
    assert sk["alone_without_certificate"] == 0                     # every alone decision under a certificate
    assert pa["s1_share"] == 0.0                                    # the floor of System 1's share
    assert sc["goals"] <= 0.2 * pa["goals"]                         # compiling without the judge does not help
    assert ca["s1_share_last5"] < sk["s1_share_last5"]              # cases do not transfer, rules do
    assert sk["planner_calls_last5"] < sk["planner_calls_first5"]   # System 2 is called less


@pytest.mark.slow
def test_the_real_world_is_never_stepped_by_the_planner_in_the_toy():
    r = _toy().run(seeds=(0,), episodes=3, arms=("skills",))
    assert r["arms"]["skills"]["guard_violations"] == 0


def test_the_toy_world_is_deterministic_and_the_script_says_what_it_measures():
    mod = _toy()
    w1, w2 = mod.GridWorld(), mod.GridWorld()
    assert w1.reset(7) == w2.reset(7)
    for a in ("up", "right", "pick", "down", "down", "right"):
        assert w1.step(a) == w2.step(a)
    assert callable(mod.run) and "deterministic gridworld" in (mod.__doc__ or "")
