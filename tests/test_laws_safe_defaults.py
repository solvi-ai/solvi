"""Default safety regressions S389/S15, finite counterexamples (not risk guarantees)."""
import itertools
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def test_default_support_checks_the_whole_matching_domain():
    g = Grammar([Atom("special", lambda s: s["special"])], k=1)
    b = Buffer(g)
    for i in range(40):
        s = {"special": i < 20}
        b.add(Transition(s, "do", "S" if s["special"] else "D", group=i))
    for kw, accepted in [({}, False), ({"support_on": "applied"}, True)]:
        ls = LawSet(g, min_precision=.95, retire_below=.9, **kw)
        ls.add(Candidate("do", ("special",), "S"))
        laws = ls.test([Candidate("do", (), "D")], b)
        assert bool(laws) is accepted


def admission(**kw):
    states = list(itertools.product((False, True), repeat=4))
    g = Grammar([Atom(f"x{i}", lambda s, i=i: s[i]) for i in range(4)], k=2)
    ls = LawSet(g, base=lambda s, a: "N")
    truth = lambda s, a: "E" if s[0] and not s[1] else "N"
    return ls.admit_by_probes([Candidate("do", ("x0",), "E")], truth,
                             probes=[(s, "do") for s in states], budget=8, **kw)


def test_default_falsifies_a_single_survivor_on_unused_probes():
    legacy = admission(falsify=None)
    default = admission()
    assert legacy.law is not None
    assert default.status == "undetermined" and default.law is None
    assert "falsified by probe" in default.why


def test_auto_preserves_explicit_noise_and_two_law_paths():
    for kw in ({"noise": .1}, {"max_new": 2}):
        auto = admission(**kw)
        old = admission(falsify=None, **kw)
        assert (auto.status, auto.why, len(auto.asked)) == (old.status, old.why, len(old.asked))


def test_all_predictions_include_equal_specificity_shadowed_laws_and_defaults():
    from solvi.experimental.laws import load_module
    g = Grammar([Atom('off', lambda s: s['off'], code="s['off']"),
                 Atom('last', lambda s: s['last'], code="s['last']")], k=2)
    for ties in ('earlier', 'newer'):
        ls = LawSet(g, ties=ties)
        default = ls.add(Candidate('dur', (), 0))
        first = ls.add(Candidate('dur', ('off',), 1))
        hidden = ls.add(Candidate('dur', ('last',), 4))
        state = dict(off=True, last=True)
        siblings = [first, hidden] if ties == 'earlier' else [hidden, first]
        assert ls.matching(state, 'dur') == siblings + [default]
        assert ls.maximal(state, 'dur') == siblings
        assert ls.predictions(state, 'dur') == [(law, law.effect) for law in siblings + [default]]
        assert ls.predict(state, 'dur') == siblings[0].effect
        assert ls.violations(state, 'dur', 1) == [hidden]
        assert ls.violations(state, 'unknown', 99) == []
        assert ls.predict(state, 'unknown') is None
        for suspended in (False, True):
            if suspended:
                ls.suspend(hidden)
                assert hidden not in ls.matching(state, 'dur')
            module = load_module(ls.to_module())
            for off, last in itertools.product((False, True), repeat=2):
                s = dict(off=off, last=last)
                for action in ('dur', 'unknown'):
                    assert module.matching(s, action) == [law.id for law in ls.matching(s, action)]
                    assert module.maximal(s, action) == [law.id for law in ls.maximal(s, action)]
                    for maximal in (False, True):
                        assert module.predictions(s, action, maximal=maximal) == [
                            (law.id, pred) for law, pred in ls.predictions(s, action, maximal=maximal)]
                    for observed in (0, 1, 4):
                        assert module.violations(s, action, observed) == [
                            law.id for law in ls.violations(s, action, observed)]
                    assert module.predict(s, action) == ls.predict(s, action)


def test_matching_checker_respects_exceptions_and_custom_effect_equality():
    from solvi.experimental.laws import load_module
    g = Grammar([Atom('a', lambda s: s['a'], code="s['a']"),
                 Atom('b', lambda s: s['b'], code="s['b']")], k=2)
    ls = LawSet(g, apply=lambda effect, state: effect + state['offset'], tol=.1)
    ls.add(Candidate('do', (), 0))
    ls.add(Candidate('do', ('a',), 1))
    exception = ls.add(Candidate('do', ('a', 'b'), 2))
    state = dict(a=True, b=True, offset=10)
    assert ls.maximal(state, 'do') == [exception]
    assert ls.violations(state, 'do', 12.05) == []
    assert ls.violations(state, 'do', 11) == [exception]
    module = load_module(ls.to_module())
    assert module.predictions(state, 'do', apply=ls.apply, maximal=True) == [(exception.id, 12)]
    assert module.violations(state, 'do', 12.05, apply=ls.apply, agrees=ls.agrees) == []
