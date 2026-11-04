"""Zero-weight checks are absent evidence, including returned candidates/collisions."""
import copy

import pytest

from test_answer_policy import policy


@pytest.mark.parametrize('q0', [.15, .7, .95])
@pytest.mark.parametrize('state', [
    {'D': 'agree'}, {'D': 'fail'}, {'D': ('val', 'g')},
    {'D': ('val', 'g'), 'P': ('val', 'g')},
    {'D': ('val', 'g'), 'P': ('val', 'other')},
])
def test_zero_weight_evidence_preserves_prior_and_creates_no_candidate(q0, state):
    p = policy(weights=lambda obs: {k: 0.0 for k in obs})
    before = copy.deepcopy(state)
    original, candidates, unseen = p.posterior(q0, state)
    assert original == pytest.approx(q0)
    assert candidates == {}
    assert unseen == pytest.approx(1-q0)
    assert state == before


@pytest.mark.parametrize('disabled', ['agree', 'fail', ('val', 'g'), ('val', 'other')])
def test_disabled_check_cannot_change_active_candidate_or_collision(disabled):
    p = policy(weights=lambda obs: {k: 0.0 if k == 'D' else .5 for k in obs})
    active = {'P': ('val', 'g')}
    assert p.posterior(.7, {**active, 'D': disabled}) == p.posterior(.7, active)


@pytest.mark.parametrize('bad', [-1, float('nan'), float('inf'), True])
def test_invalid_weight_still_refuses(bad):
    p = policy(weights=lambda obs: {k: bad for k in obs})
    with pytest.raises(ValueError):
        p.posterior(.7, {'D': ('val', 'g')})


@pytest.mark.parametrize("disabled_first", [True, False])
def test_disabled_first_alias_is_not_selected_as_answer_source_or_reweighted(disabled_first):
    calls = []
    def weights(obs):
        calls.append(tuple(obs))
        return {k: 0.0 if k == 'D' else 1.0 for k in obs}
    p = policy(weights=weights, max_checks=2, units={'D': 2, 'P': 3}, max_units=5)
    state = dict([('D', ('val', 'g')), ('P', ('val', 'g'))][::1 if disabled_first else -1])
    answer = p.decide(.15, state, 1)
    assert answer.kind == 'answer_check' and answer.check == 'P'
    assert calls == [tuple(state), tuple(state)]  # decision + existing why report; none for source selection
    assert p.spent(state) == 5  # disabled evidence was still purchased
    assert state == {'D': ('val', 'g'), 'P': ('val', 'g')}


def test_all_disabled_candidates_cannot_be_answer_sources():
    p = policy(weights=lambda obs: {k: 0.0 for k in obs}, max_checks=2)
    answer = p.decide(.15, {'D': ('val', 'g'), 'P': ('val', 'g')}, 1)
    assert answer.kind in ('answer', 'abstain') and answer.check is None
