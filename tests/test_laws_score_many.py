"""Finite vector parity and per-call view reuse; no persistent callback cache."""
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition


def fixture():
    grammar = Grammar([Atom("positive", lambda s: s > 0)], k=1)
    buffer = Buffer(grammar)
    for action in ("a", "b"):
        for i in range(8):
            buffer.add(Transition(i - 3, action, i % 2, group=i % 2))
    laws = LawSet(grammar, base=lambda s, a: 0)
    laws.add(Candidate("a", ("positive",), 1))
    candidates = [Candidate(a, c, e) for a in ("a", "b", "missing")
                  for c in ((), ("positive",)) for e in (0, 1)]
    return laws, buffer, candidates + candidates[:2]


@pytest.mark.parametrize("exclude", [None, 0, 1])
@pytest.mark.parametrize("support", ["matched", "applied"])
@pytest.mark.parametrize("ties", ["newer", "older"])
def test_vector_equals_scalar_and_builds_one_view_per_action(exclude, support, ties):
    laws, buffer, candidates = fixture()
    laws.support_on, laws.ties = support, ties
    expected = [laws.score(c, buffer, exclude) for c in candidates]
    original, calls = laws._view, []

    def counted(*args):
        calls.append(args[1])
        return original(*args)

    laws._view = counted
    assert laws.score_many(iter(candidates), buffer, exclude, stable=True) == expected
    assert calls == ["a", "b", "missing"]
    calls.clear()
    assert laws.score_many([], buffer, stable=True) == [] and calls == []
    buffer.add(Transition(9, "a", 1, group=2))
    expected = [laws.score(c, buffer, exclude) for c in candidates]
    calls.clear()
    assert laws.score_many(candidates, buffer, exclude, stable=True) == expected
    assert len(calls) == 3
    assert not laws.memo and laws._memo == {}


def test_default_preserves_iterator_mutation_boundary():
    laws, buffer, candidates = fixture()
    candidate = candidates[0]

    def stream():
        yield candidate
        laws.add(Candidate("a", (), 1))
        buffer.add(Transition(10, "a", 1))
        yield candidate

    # Two independent fixtures undergo the same mutations in the same order.
    expected = []
    for c in stream():
        expected.append(laws.score(c, buffer))
    laws, buffer, candidates = fixture()
    assert laws.score_many(stream(), buffer) == expected
