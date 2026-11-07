"""Independent preferences review regressions and direct finite references."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from itertools import combinations

import pytest

from solvi.experimental.experience._common import BudgetStop, WorkBudget, digest
from solvi.experimental.experience.preferences import (
    Alternative, PreferenceSession, Requirement, check_regret_certificate,
)


def make(alternatives=None, weights=None, requirements=()):
    return PreferenceSession(
        alternatives or [Alternative('a', (1, 0, 0)), Alternative('b', (0, 1, 0))],
        weights or [(1, 0, 0), (0, 1, 0)], requirements=requirements,
        budget=WorkBudget(1000000),
    )


def restore_for(session, saved, **kwargs):
    if hasattr(session, 'authority'):
        kwargs['authority'] = session.authority
    return PreferenceSession.restore(saved, budget=WorkBudget(1000000), **kwargs)


def test_review_snapshot_cannot_remove_answer_and_rehash():
    s = make()
    s.answer('x', 'a', 'b', source='author')
    saved = deepcopy(s.snapshot())
    body = saved.get('body', saved)
    body['answers'] = []
    if 'digest' in saved:
        saved['digest'] = digest({k: v for k, v in saved.items() if k != 'digest'})
    with pytest.raises(ValueError):
        restore_for(s, saved)


def test_review_snapshot_cannot_rollback_retraction():
    s = make()
    s.answer('x', 'a', 'b', source='author')
    saved = s.snapshot()
    s.retract('x', source='author correction')
    with pytest.raises(ValueError):
        restore_for(s, saved)


def test_review_snapshot_cannot_rollback_revision():
    s = make()
    s.answer('x', 'a', 'b', source='author')
    saved = s.snapshot()
    s.answer('x', 'b', 'a', source='author correction')
    with pytest.raises(ValueError):
        restore_for(s, saved)


@pytest.mark.parametrize('field,value', [
    ('regret', True), ('surviving_weights', (False, True)),
    ('regrets', (('a', True), ('b', True))),
])
def test_review_certificate_rejects_bool_integer_aliases(field, value):
    s = make()
    cert = s.choose()
    assert cert.regret == 1
    assert not check_regret_certificate(s, replace(cert, **{field: value}), budget=s.budget)


def test_review_checker_charges_regret_pair_comparisons():
    s = make([Alternative(str(i), (i, 7 - i, 0)) for i in range(8)])
    cert = s.choose()
    # There are 8*8*2 regret comparisons in addition to table construction.
    with pytest.raises(BudgetStop):
        check_regret_certificate(s, cert, budget=WorkBudget(40))


def reference_regret(alternatives, weights):
    def utility(a, w):
        return sum(a.attributes[k] * w[k] for k in range(3))
    table = [(a.id, max(utility(b, w) - utility(a, w) for b in alternatives for w in weights))
             for a in alternatives]
    return min(table, key=lambda x: (x[1], x[0])), tuple(table)


@pytest.mark.parametrize('weights', [
    [(1, 0, 0), (0, 1, 0), (1, 1, 0)],
    [(-1, 2, 0), (2, -1, 0), (0, 0, 0)],
    [(1, 1, 1)],
])
def test_review_regret_and_question_against_direct_reference(weights):
    alternatives = [Alternative('z', (3, 0, 1)), Alternative('a', (0, 3, 1)),
                    Alternative('c', (2, 2, -1))]
    s = make(alternatives, weights)
    best, table = reference_regret(alternatives, weights)
    cert = s.choose()
    assert (cert.choice, cert.regret, cert.regrets) == (*best, table)
    reductions = []
    for a, b in combinations(alternatives, 2):
        branches = {}
        for w in weights:
            difference = sum((a.attributes[k] - b.attributes[k]) * w[k] for k in range(3))
            branches.setdefault((difference > 0) - (difference < 0), []).append(w)
        after = max(reference_regret(alternatives, branch)[0][1] for branch in branches.values())
        reductions.append((a.id, b.id, Fraction(best[1] - after)))
    maximum = max(r[2] for r in reductions)
    q = s.next_question()
    if maximum <= 0:
        assert q is None
    else:
        expected = next(r for r in reductions if r[2] == maximum)
        assert (q.left, q.right, q.reduction) == expected


@pytest.mark.parametrize('relation,expected', [('strict', (0,)), ('weak', (0, 2)), ('tie', (2,))])
def test_review_disjoint_tie_strict_and_weak_semantics(relation, expected):
    s = make(weights=[(1, 0, 0), (0, 1, 0), (1, 1, 0)])
    s.answer('x', 'a', 'b', source='author', relation=relation)
    assert s.choose().surviving_weights == expected


def test_review_conflict_sets_are_inclusion_minimal_and_hard_cannot_relax():
    reqs = [Requirement('hard', 2, 0, True), Requirement('left', 0, 1, False),
            Requirement('right', 1, 1, False), Requirement('impossible', 2, 1, False)]
    s = make(requirements=reqs)
    assert set(s.conflicts()['requirements']) == {('impossible',), ('left', 'right')}
    assert set(s.relaxations()) == {('left', 'impossible'), ('right', 'impossible')}
    assert all('hard' not in r for r in s.relaxations())
    assert s.requirements == tuple(reqs)


def test_review_out_of_family_answers_leave_no_numeric_decision():
    s = make()
    s.answer('tie', 'a', 'b', source='out-of-family-author', relation='tie')
    cert = s.choose()
    assert cert.status == 'conflict'
    assert cert.choice is None and cert.regret is None
    assert s.next_question() is None
    assert s.conflicts()['answers'] == (('tie',),)


def test_review_restore_pays_before_copying_untrusted_nested_payload(monkeypatch):
    import solvi.experimental.experience.preferences as module

    s = make()
    saved = s.snapshot()
    saved['weights'] = [list(range(1000))]
    def copy_forbidden(_value):
        raise AssertionError('untrusted portable copy reached before paid structural validation')
    monkeypatch.setattr(module, 'portable', copy_forbidden)
    kwargs = {'authority': s.authority} if hasattr(s, 'authority') else {}
    with pytest.raises((ValueError, BudgetStop)):
        PreferenceSession.restore(saved, budget=WorkBudget(10), **kwargs)
