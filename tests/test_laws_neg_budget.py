"""solvi.experimental.laws (experimental): `BudgetedProposer(inner, candidates, neg_budget=, mismatches=)` — a candidate
budget where candidates with a negated literal are charged to their own budget, so they never crowd out the positive
candidates of a later mismatch (the cause of the found loss on the second task (relational laws over generated graphs) of negations="refine")."""
from __future__ import annotations

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import BudgetedProposer, Candidate, Enumerate

from test_laws_negations import run_unless_r, word_grammar  # noqa: E402


class _Fixed:
    """An inner proposer that returns the next fixed list of candidates per call."""
    name = "fixed"

    def __init__(self, lists):
        self.lists, self.calls = [list(x) for x in lists], 0

    def propose(self, lawset, state, action, observed):
        self.calls += 1
        return self.lists.pop(0) if self.lists else []


class _LS:
    def __init__(self, g):
        self.grammar = g


def _c(*cond):
    return Candidate("turn", tuple(cond), "become", "fixed")


G = word_grammar("refine")
POS = [_c("d0"), _c("d1"), _c("d2")]
NEG = [_c("d0", "not:d1"), _c("not:nself"), _c("d2", "not:d3")]


def _mixed(i):
    """One mismatch's list: negated candidates ranked before the positives (the crowd-out order)."""
    return NEG + [_c(f"d{i % 4}", "nsubj")] + POS


def test_positive_candidates_of_every_mismatch_are_those_without_negations():
    """The positive stream under the separate budget equals the positive-only proposer's, mismatch by mismatch."""
    neg = BudgetedProposer(_Fixed([_mixed(i) for i in range(5)]), 10, neg_budget=100)
    pos = BudgetedProposer(_Fixed([[c for c in _mixed(i) if not any(x.startswith("not:") for x in c.condition)]
                                   for i in range(5)]), 10)
    for _ in range(5):
        a = [c for c in neg.propose(_LS(G), {}, "turn", "become") if not any(x.startswith("not:") for x in c.condition)]
        b = pos.propose(_LS(G), {}, "turn", "become")
        assert a == b
    assert neg.used == pos.used == 10 and neg.mismatches_seen == pos.mismatches_seen == 3
    assert neg.used_neg == 9                                  # 3 mismatches × 3 negated, all within their own budget


def test_shared_budget_reaches_fewer_mismatches():
    """The failure the option fixes: one shared budget is spent by the negated candidates two mismatches earlier."""
    shared = BudgetedProposer(_Fixed([_mixed(i) for i in range(5)]), 10, neg_budget="shared")
    out = [shared.propose(_LS(G), {}, "turn", "become") for _ in range(5)]
    assert [len(x) for x in out] == [7, 3, 0, 0, 0] and shared.mismatches_seen == 2
    assert shared.used + shared.used_neg == 10 and shared.used_neg == 6
    assert out[1] == NEG                                      # the second mismatch's positives were never offered
    sep = BudgetedProposer(_Fixed([_mixed(i) for i in range(5)]), 10)                      # default: separate, 10
    assert sep.neg_budget == 10
    assert [len(x) for x in (sep.propose(_LS(G), {}, "turn", "become") for _ in range(5))] == [7, 7, 5, 0, 0]


def test_neg_budget_zero_drops_negated_and_the_negated_budget_runs_out_alone():
    p = BudgetedProposer(_Fixed([_mixed(i) for i in range(3)]), 100, neg_budget=0)
    out = [c for _ in range(3) for c in p.propose(_LS(G), {}, "turn", "become")]
    assert len(out) == 12 and p.used_neg == 0
    q = BudgetedProposer(_Fixed([_mixed(i) for i in range(3)]), 100, neg_budget=4)
    out = [q.propose(_LS(G), {}, "turn", "become") for _ in range(3)]
    assert [sum(any(x.startswith("not:") for x in c.condition) for c in o) for o in out] == [3, 1, 0]
    assert q.used == 12 and q.used_neg == 4 and q.mismatches_seen == 3   # positives go on while their budget lasts


def test_plain_budget_without_negations_and_the_mismatch_cap():
    g = word_grammar(False)
    lists = [[_c("d0"), _c("d1"), _c("d2")] for _ in range(4)]
    p = BudgetedProposer(_Fixed(lists), 7, mismatches=2)
    out = [p.propose(_LS(g), {}, "turn", "become") for _ in range(4)]
    assert [len(x) for x in out] == [3, 3, 0, 0] and p.mismatches_seen == 2 and p.spent()
    p = BudgetedProposer(_Fixed(lists), 7)
    out = [p.propose(_LS(g), {}, "turn", "become") for _ in range(4)]
    assert [len(x) for x in out] == [3, 3, 1, 0] and p.used == 7 and p.used_neg == 0 and p.last == []


@pytest.mark.parametrize("bad", [-1, "half", 1.5])
def test_bad_neg_budget_is_refused(bad):
    with pytest.raises(ValueError):
        BudgetedProposer(_Fixed([]), 10, neg_budget=bad)
    with pytest.raises(ValueError):
        BudgetedProposer(_Fixed([]), -1)


def test_known_answer_lone_negated_law_learned_through_the_budgeted_proposer():
    """The library's "become unless R" world: under refine with Enumerate wrapped in a budget (large enough not to bind)
    the same laws are learned as without the wrapper — not:R → become, no wrong law."""
    import test_laws_negations as T
    ls0, wrong0 = run_unless_r("refine")
    orig = T.LawSet

    class Wrapped(orig):
        def __init__(self, *a, **kw):
            kw["proposer"] = BudgetedProposer(Enumerate(), 10**6)
            super().__init__(*a, **kw)
    T.LawSet = Wrapped
    try:
        ls1, wrong1 = run_unless_r("refine")
    finally:
        T.LawSet = orig
    assert [(lw.condition, lw.effect) for lw in ls1.laws] == [(lw.condition, lw.effect) for lw in ls0.laws]
    assert [lw.condition for lw in ls1.laws if lw.effect == "become"] == [("not:R",)] and wrong1 == []
    assert ls1.proposer.used_neg > 0 and ls1.proposer.used > 0
