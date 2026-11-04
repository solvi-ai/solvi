"""Independent opt-in failure vetoes use actual Environment episodes and paired seeds."""
from types import SimpleNamespace

import pytest

from solvi.experimental.changes import Change, FreshEnvironments


def stage(base=(), candidate=(), **kwargs):
    class World:
        def reset(self, seed=None):
            self.seed = seed
            return {}

        def actions(self, state):
            return [False, True]

        def step(self, action):
            failed = self.seed in (candidate if action else base)
            return SimpleNamespace(state={}, done=True, info={"failed": failed, "progress": float(action)})

    return FreshEnvironments(World, lambda ch, ctx: lambda state, actions: ch is not None,
                             select=[1, 2], confirm=[3, 4], boot=20,
                             score=lambda row: row["progress"], **kwargs)


def run(s):
    return s.run(Change("person"), None)


@pytest.mark.parametrize("policy", ["zero_failures", "no_new_failures", "max_failure_rate"])
@pytest.mark.parametrize("failed_seed,phase", [(1, "select"), (3, "confirm")])
def test_rewarding_score_cannot_override_explicit_failure_policy(policy, failed_seed, phase):
    kw = {"max_failure_rate": 0.0} if policy == "max_failure_rate" else {}
    result = run(stage(candidate=[failed_seed], failure_policy=policy, **kw))
    assert result.decision == "refuse"
    evidence = result.measured[phase]["failure_policy"]
    assert evidence["candidate_failures"] == evidence["new_failures"] == 1
    assert evidence["candidate_failure_rate"] == 0.5
    assert evidence["refused"] and evidence["reason"] == result.reason
    assert ("confirm" in result.measured) == (phase == "confirm")


def test_default_stays_score_only_and_zero_failures_is_absolute():
    assert run(stage(candidate=[1, 3])).decision == "admit"
    assert run(stage(base=[1, 3], candidate=[1, 3], failure_policy="zero_failures")).decision == "refuse"


def test_no_new_failures_allows_old_failures_and_improvements():
    for candidate in ([1, 3], []):
        result = run(stage(base=[1, 3], candidate=candidate, failure_policy="no_new_failures"))
        assert result.decision == "admit"
        assert result.measured["confirm"]["failure_policy"]["new_failures"] == 0


def test_equal_failure_totals_do_not_hide_new_paired_failure():
    result = run(stage(base=[1], candidate=[2], failure_policy="no_new_failures"))
    assert result.decision == "refuse"
    assert result.measured["select"]["failed"] == [1, 1]
    assert result.measured["select"]["failure_policy"]["new_failures"] == 1


def test_rate_cap_inclusive_boundary_and_reuse_has_no_accumulated_state():
    s = stage(candidate=[1, 3], failure_policy="max_failure_rate", max_failure_rate=0.5)
    first, second = run(s), run(s)
    assert first.decision == second.decision == "admit"
    assert first.measured == second.measured
    assert first.measured["confirm"]["failure_policy"]["max_failure_rate"] == 0.5


@pytest.mark.parametrize("kw", [{"failure_policy": "unknown"}, {"failure_policy": "max_failure_rate"},
    {"max_failure_rate": 0.2}, {"failure_policy": "zero_failures", "max_failure_rate": 0},
    *({"failure_policy": "max_failure_rate", "max_failure_rate": cap}
      for cap in [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])])
def test_bad_policy_configuration_fails_before_episodes(kw):
    with pytest.raises(ValueError):
        stage(**kw)


def test_seed_hooks_and_replay_still_run_with_safe_policy():
    seen = []
    result = run(stage(failure_policy="zero_failures", on_episode_start=lambda policy, seed: seen.append(seed)))
    assert result.decision == "admit"
    assert seen == [1, 2, 1, 2, 1, 2, 3, 4, 3, 4]
    assert result.measured["deterministic"]


@pytest.mark.parametrize("phase", ["select", "confirm"])
def test_evidence_preserves_seed_paired_failure_sources(phase):
    result = run(stage(base=[1, 3], candidate=[1, 3], failure_policy="no_new_failures"))
    start = 1 if phase == "select" else 3
    assert result.measured[phase]["failure_policy"]["paired"] == [
        {"seed": start, "baseline_failed": True, "candidate_failed": True},
        {"seed": start + 1, "baseline_failed": False, "candidate_failed": False},
    ]


def test_empty_seed_set_refuses_before_constructing_agents():
    def unexpected(*args):
        raise AssertionError("must not construct an agent")
    for select, confirm in [([], [1]), ([1], [])]:
        s = FreshEnvironments(unexpected, unexpected, select=select, confirm=confirm,
                              failure_policy="zero_failures")
        assert "no selection or confirmation" in run(s).reason


def test_policy_errors_keep_priority_and_failure_evidence():
    s = stage(failure_policy="zero_failures")
    def agent(change, ctx):
        def crash(state, actions):
            raise RuntimeError("broken policy")
        return crash if change else lambda state, actions: False
    s.agent = agent
    s.score = lambda row: float(row["errors"])
    result = run(s)
    assert result.decision == "refuse" and "crashed" in result.reason
    assert result.measured["select"]["failure_policy"]["candidate_failures"] == 0
