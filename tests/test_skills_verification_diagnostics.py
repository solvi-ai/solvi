"""S285: preserve failed checks and warn about repeated exact-start disagreement without blaming a clone."""
import warnings

from test_skills import Corridor, at_goal
from solvi.experimental.skills import Plan, SkillLearner, plan


def wrong_prediction(model, state, goal, **kwargs):
    return Plan(('right',), predicted=({'pos': 999, 'colour': state['colour']},),
                start=model.key(state), goal=goal)


def learner(planner=wrong_prediction):
    env = Corridor()
    return env, SkillLearner(env, None, {'door': at_goal}, planner=planner, learn=False)


def test_propose_preserves_actual_failed_check_and_warns_once_after_three():
    env, learned = learner()
    before = learned.model.fingerprint()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for i in range(5):
            goal, action, record = learned.propose(env._state())
            assert goal == 'door' and action is None
            assert 'predicted one' in record['why']
            assert record['verification']['steps'] == 1
            assert bool(seen) is (i >= 2)
    assert len(seen) == 1
    assert 'possible causes' in str(seen[0].message)
    assert 'inaccurate plan predictions' in str(seen[0].message)
    assert len(learned.report()['plan_rejections']) == learned.totals['plans_rejected'] == 5
    assert learned.model.fingerprint() == before and not learned.skills.skills


def test_act_retains_failure_without_changing_fallback_or_compiling():
    env, learned = learner()
    action = learned.act(env._state())
    assert action in env.actions(env._state())
    assert learned.decisions[-1]['by'] == 'fallback'
    assert learned.plan_rejections[-1]['why'].endswith('other than the predicted one')
    assert learned.totals['plans_rejected'] == learned.totals['verify_calls'] == 1
    assert not learned.skills.skills


def test_goal_failures_do_not_warn_about_clone_determinism():
    def short(model, state, goal, **kwargs):
        return Plan(('right',), predicted=({'pos': 1, 'colour': state['colour']},),
                    start=model.key(state), goal=goal)
    env, learned = learner(short)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        for _ in range(4):
            assert learned.propose(env._state())[1] is None
    assert not seen
    assert all(row['why'] == 'the goal does not hold at the end' for row in learned.plan_rejections)


def test_successful_plan_does_not_expose_previous_rejection_as_current_failure():
    env, learned = learner()
    assert learned.propose(env._state())[1] is None
    learned.planner = plan
    goal, action, record = learned.propose(env._state())
    assert goal == 'door' and action == 'right' and 'verification' not in record
    assert len(learned.plan_rejections) == learned.totals['plans_rejected'] == 1
    assert learned._rejection_streak is None


def test_changed_start_resets_rejection_streak():
    env, learned = learner()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter('always')
        learned.propose(env.reset(0))
        learned.propose(env._state())
        learned.propose(env.reset(1))
        learned.propose(env._state())
    assert not seen and learned._rejection_streak[1] == 2


def test_verification_disabled_control_does_not_record_or_warn():
    env, learned = learner()
    learned.verify_plans = False
    with warnings.catch_warnings(record=True) as seen:
        assert learned.propose(env._state())[1] == 'right'
    assert not seen and not learned.plan_rejections
    assert learned.totals['verify_calls'] == learned.totals['plans_rejected'] == 0
