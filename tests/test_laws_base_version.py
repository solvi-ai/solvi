"""S235: owner-declared base revisions affect exceptions, not absolute laws."""
import pickle

import pytest

from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition
from solvi.experimental.laws._compile import law_to_code, lawset_to_module


def discounted(state):
    return state['discount']


def base_one(state, action):
    return 11.81


def base_two(state, action):
    return 13.0


def grammar():
    return Grammar([Atom('discount', discounted, code="s['discount']")], k=1)


def exceptions():
    return LawSet(grammar(), base=base_one, lapse_on_base_change=True, base_version='tariff-1')


def candidate():
    return Candidate('bill', ('discount',), 8.03)


def test_new_revision_retires_active_and_suspended_exceptions_before_serving():
    ls = exceptions()
    active = ls.add(candidate())
    suspended = ls.add(Candidate('other', ('discount',), 9.0))
    ls.suspend(suspended)
    assert ls.predict({'discount': True}, 'bill') == 8.03
    before, cost = ls.version, dict(ls.cost)
    retired = ls.set_base(base_two, version='tariff-2')
    assert set(retired) == {active, suspended}
    assert not ls.laws and not ls.suspended
    assert all(law.status == 'retired' and law.base_version == 'tariff-1' for law in retired)
    assert all('base revision changed' in law.why for law in retired)
    assert ls.predict({'discount': True}, 'bill') == 13.0
    assert ls.version > before and ls.cost == cost  # no refund or reset of acceptance accounting
    with pytest.raises(ValueError, match='fresh admission'):
        ls.reactivate(suspended)
    fresh = ls.add(candidate())  # a separate, explicit hand admission
    assert fresh is not active and fresh.base_version == 'tariff-2'
    assert fresh.n == fresh.hits_after == fresh.support == 0


def test_absolute_laws_preserve_their_meaning_when_only_fallback_changes():
    ls = LawSet(grammar(), base=base_one)
    law = ls.add(candidate())
    before = ls.version
    ls.base = base_two
    assert ls.predict({'discount': True}, 'bill') == 8.03
    assert ls.predict({'discount': False}, 'bill') == 13.0
    assert law.status == 'active' and law.base_version is None
    assert ls.version > before
    assert 'def predict' in ls.to_module()


def test_invalid_revision_changes_are_atomic_and_tokens_cannot_be_recycled():
    ls = exceptions()
    law = ls.add(candidate())
    before = (ls.version, ls.base, ls.base_version, list(ls.laws))
    for operation in (lambda: ls.set_base(base_two), lambda: ls.set_base(base_two, version=''),
                      lambda: ls.set_base(base_two, version='tariff-1'),
                      lambda: ls.set_base(123, version='tariff-2')):
        with pytest.raises(ValueError):
            operation()
        assert (ls.version, ls.base, ls.base_version, ls.laws) == before
    with pytest.raises(ValueError, match='set_base'):
        ls.base = base_two
    with pytest.raises(ValueError, match='override'):
        ls.predict({'discount': True}, 'bill', base=base_two)
    assert ls.set_base(base_one, version='tariff-1') == []
    assert law.status == 'active'
    ls.set_base(base_two, version='tariff-2')
    with pytest.raises(ValueError, match='recycled'):
        ls.set_base(base_one, version='tariff-1')


def test_bound_exports_refuse_to_erase_dependencies_and_hypotheses_lapse():
    ls = exceptions()
    law = ls.add(candidate())
    hypothesis = ls.hypothesis(Candidate('bill', (), 7))
    for export in (ls.to_module, lambda: law.to_code(ls.grammar),
                   lambda: law_to_code(law, ls.grammar), lambda: lawset_to_module(ls),
                   lambda: ls.to_knowledge(object())):
        with pytest.raises(ValueError, match='base-revision'):
            export()
    ls.set_base(base_two, version='tariff-2')
    with pytest.raises(ValueError, match='older base'):
        hypothesis({'discount': True}, 'bill')


def test_pickle_keeps_binding_and_legacy_state_defaults_to_absolute_laws():
    ls = exceptions()
    ls.add(candidate())
    restored = pickle.loads(pickle.dumps(ls))
    assert restored.base_version == 'tariff-1' and restored.laws[0].base_version == 'tariff-1'
    restored.set_base(base_two, version='tariff-2')
    assert restored.predict({'discount': True}, 'bill') == 13.0
    old = LawSet(grammar(), base=base_one)
    old.add(candidate())
    state = old.__getstate__()
    state['base'] = state.pop('_base')
    for key in ('_lapse_on_base_change', '_base_version', '_base_versions_seen'):
        state.pop(key)
    legacy = LawSet.__new__(LawSet)
    legacy.__setstate__(state)
    assert legacy.lapse_on_base_change is False and legacy.base_version is None
    legacy.base = base_two
    assert legacy.predict({'discount': True}, 'bill') == 8.03


@pytest.mark.parametrize('commit', [False, True])
def test_probe_callback_cannot_move_base_inside_admission_or_resurrect_old_laws(commit):
    ls = exceptions()
    old = ls.add(Candidate('bill', (), 8.03))
    calls = []

    def ask(state, action):
        calls.append(True)
        ls.set_base(base_two, version='tariff-2')
        return 9.0

    with pytest.raises(ValueError, match='during admission'):
        ls.admit_by_probes([Candidate('bill', ('discount',), 9.0)], ask,
                          probes=[({'discount': True}, 'bill')], budget=10,
                          commit=commit, falsify=None)
    assert calls and ls.base_version == old.base_version == 'tariff-1'
    assert old.status == 'active' and ls.predict({'discount': True}, 'bill') == 8.03
    assert ls._base_operations == 0
    ls.set_base(base_two, version='tariff-2')
    assert old.status == 'retired' and ls.predict({'discount': True}, 'bill') == 13.0


BOX = {'price': 11.81}


def mutable_base(state, action):
    return BOX['price']


def test_explicit_version_refreshes_mutable_callable_buffer_cache(monkeypatch):
    g = grammar()
    ls = LawSet(g, base=mutable_base, lapse_on_base_change=True, base_version='1', memo=True)
    buffer = Buffer(g)
    buffer.add(Transition({'discount': True}, 'bill', 11.81, group='sample'))
    proposal = Candidate('bill', ('discount',), 11.81)
    assert ls.score(proposal, buffer) == (1, 1, 0)  # both the proposal and old base are right
    old_key = ls._base_key
    monkeypatch.setitem(BOX, 'price', 13.0)
    ls.set_base(mutable_base, version='2')
    assert old_key != ls._base_key
    assert ls.score(proposal, buffer) == (1, 1, 1)  # the same proposal now improves the changed base


@pytest.mark.parametrize('kwargs', [dict(lapse_on_base_change=True), dict(base_version='v'),
                                    dict(lapse_on_base_change=1),
                                    dict(lapse_on_base_change=True, base_version=' ')])
def test_binding_requires_an_explicit_valid_revision(kwargs):
    with pytest.raises(ValueError):
        LawSet(grammar(), **kwargs)
