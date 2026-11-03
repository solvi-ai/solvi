"""Oracle attempts cost one probe even if the callback raises; no law commits on error."""
import itertools
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition, Unknown


class OracleFailed(RuntimeError):
    pass


def case(mode):
    grammar = Grammar([Atom("x", lambda s: bool(s[0]))], k=1)
    laws = LawSet(grammar, base=lambda s, a: False)
    candidates = [Candidate("do", ("x",), True)]
    pool = [((1, i), "do") for i in range(4)]
    kw = {"max_new": 2} if mode.startswith("pair") else {}
    if mode == "noisy":
        kw = {"noise": .1, "falsify": None}
    if mode.endswith("leftover"):
        buffer = Buffer(grammar)
        buffer.add(Transition((1, -1), "do", True))
        kw["buffer"] = buffer
    return laws, candidates, pool, kw


@pytest.mark.parametrize("mode,commit,fail_at", list(itertools.product(
    ["exact-split", "exact-leftover", "pair-split", "pair-leftover", "noisy"], [True, False], [1, 2])))
def test_raised_oracle_counts_the_actual_attempt_before_calling(mode, commit, fail_at):
    laws, candidates, pool, kw = case(mode)
    before = (list(laws.laws), laws.version, laws.cost["accepted"])
    calls = []

    def ask(state, action):
        calls.append(state)
        assert laws.cost["probes"] == len(calls)
        if len(calls) == fail_at:
            raise OracleFailed("sentinel")
        return Unknown  # keep every mode asking until the sentinel attempt

    with pytest.raises(OracleFailed, match="sentinel"):
        laws.admit_by_probes(candidates, ask, probes=pool, budget=3, commit=commit, **kw)
    assert laws.cost["probes"] == len(calls) == fail_at
    assert (laws.laws, laws.version, laws.cost["accepted"]) == before


@pytest.mark.parametrize("mode,unknown", list(itertools.product(
    ["exact-split", "exact-leftover", "pair-split", "pair-leftover", "noisy"], [True, False])))
def test_successful_or_unknown_attempts_are_charged_once(mode, unknown):
    laws, candidates, pool, kw = case(mode)
    calls = []

    def ask(state, action):
        calls.append(state)
        assert laws.cost["probes"] == len(calls)
        return Unknown if unknown else True

    result = laws.admit_by_probes(candidates, ask, probes=pool, budget=3, **kw)
    assert laws.cost["probes"] == len(calls) == len(result.asked) + len(result.unknown)
    assert 0 < len(calls) <= 3
    assert bool(result.unknown) is unknown
