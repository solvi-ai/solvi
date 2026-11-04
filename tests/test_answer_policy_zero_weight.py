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
