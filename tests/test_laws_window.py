"""solvi.experimental.laws: a sliding window on the buffer, retirement over recent observations, a score memo and the
law set's own Proposals count. Synthetic data only."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.core.slow.proposals import Proposals, proposals_of
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition

NA = 6


def grammar(k=2):
    return Grammar([Atom(f"f{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(NA)], k=k)


def stream(rng, n, law=("f0", "f1"), eff="E"):
    out = []
    for j in range(n):
        s = tuple(int(rng.random() < 0.4) for _ in range(NA))
        a = rng.choice(("go", "do"))
        hit = all(s[int(x[1:])] for x in law)
        obs = eff if hit and rng.random() < 0.95 else ("N" if rng.random() < 0.8 else "E")
        out.append(Transition(s, a, obs, group=j // 7, id=f"r{j}"))
    return out


CANDS = [Candidate(a, c, e) for a in ("go", "do") for c in ((), ("f0",), ("f0", "f1"), ("f2", "f3"), ("f1",))
         for e in ("E", "N")]


@pytest.mark.parametrize("window", [1, 5, 37, 200])
def test_a_windowed_buffer_scores_as_a_fresh_buffer_of_its_last_rows(window):
    rng = random.Random(window)
    g = grammar()
    rows = stream(rng, 300)
    w = Buffer(g, window=window)
    for j, t in enumerate(rows):
        w.add(t)
        if j % 23 == 0 or j == len(rows) - 1:
            fresh = Buffer(g)
            fresh.extend(rows[max(0, j + 1 - window):j + 1])
            assert len(w) == len(fresh) == min(window, j + 1)
            assert w.groups() == fresh.groups()
            for a in ("go", "do"):
                assert [t.id for t in w.rows(a)] == [t.id for t in fresh.rows(a)]
            ls = LawSet(g, base=lambda s, a: "N")
            ls.add(Candidate("go", ("f0", "f1"), "E"))
            for c in CANDS:
                for ex in (None, j // 7, (j // 7) - 3):
                    assert ls.score(c, w, exclude=ex) == ls.score(c, fresh, exclude=ex)
                    assert ls.base_rate(w, c.action, c.effect, ex) == ls.base_rate(fresh, c.action, c.effect, ex)
    assert w.left == len(rows) - min(window, len(rows)) and w.added == len(rows)


def test_an_atom_added_after_rows_left_is_indexed_on_the_rows_kept():
    rng = random.Random(1)
    g = grammar()
    rows = stream(rng, 120)
    w = Buffer(g, window=50)
    w.extend(rows[:90])
    g.add(Atom("f0&f2", lambda s: s[0] == 1 and s[2] == 1, feature="combo"))
    w.extend(rows[90:])
    fresh = Buffer(g)
    fresh.extend(rows[-50:])
    for a in ("go", "do"):
        assert w.match(a, ("f0&f2",)) == fresh.match(a, ("f0&f2",))


def test_n_never_falls_and_a_candidate_counts_again_only_after_the_window_turned_over():
    g = grammar()
    rows = stream(random.Random(2), 400)
    b = Buffer(g, window=100)
    b.extend(rows[:150])
    ls = LawSet(g, base=lambda s, a: "N")
    ls.test(CANDS[:4], b, exclude=-1)
    assert b.tested == 4
    b.extend(rows[150:200])                                  # 50 rows: not a turnover
    ls.test(CANDS[:4], b, exclude=-1)
    assert b.tested == 4
    b.extend(rows[200:260])                                  # 110 rows since they were counted: other rows now
    ls.test(CANDS[:4], b, exclude=-1)
    assert b.tested == 8
    plain = Buffer(g)
    plain.extend(rows[:260])
    for _ in range(3):
        ls.test(CANDS[:4], plain, exclude=-1)
    assert plain.tested == 4                                 # without a window: distinct candidates, as before
    with pytest.raises(ValueError):
        Buffer(g, window=0)


def regime_change(recent):
    g = grammar()
    ls = LawSet(g, base=lambda s, a: "N", recent=recent)
    law = ls.add(Candidate("go", ("f0",), "E"))
    s = (1, 0, 0, 0, 0, 0)
    for _ in range(400):                                     # right 400 times
        ls.observe(Transition(s, "go", "E", group="w"))
    for j in range(60):                                      # then the world changes
        ls.observe(Transition(s, "go", "N", group="w"))
        if law.status == "retired":
            return j + 1
    return None


def test_retirement_over_recent_observations_follows_a_change_the_lifetime_count_hides():
    assert regime_change(None) is None                       # 400 / 460 = 0.87: still above 0.8
    assert regime_change(50) == 11                           # 39 / 50 < 0.8 after 11 wrong
    with pytest.raises(ValueError):
        LawSet(grammar(), recent=2)                          # below retire_min


def test_the_memo_gives_the_same_scores_and_laws_and_counts_its_hits():
    rows = stream(random.Random(3), 600)
    outs = []
    for memo in (False, True):
        g = grammar()
        b = Buffer(g, window=250)
        ls = LawSet(g, base=lambda s, a: "N", memo=memo)
        acc = []
        for t in rows:
            st = ls.step(t, b, remember=True)
            acc += [(law.condition, law.effect) for law in st.accepted]
            if memo:
                first = [ls.score(c, b, exclude=t.group) for c in CANDS]
                again = [ls.score(c, b, exclude=t.group) for c in CANDS]
                assert first == again
        outs.append((acc, [law.condition for law in ls.laws], ls.memo_hits))
    assert outs[0][:2] == outs[1][:2]
    assert outs[1][2] >= len(rows) * len(CANDS)              # the repeated scores were lookups


def test_the_law_set_carries_its_own_proposals_count():
    rows = stream(random.Random(4), 300)
    g = grammar()
    b = Buffer(g)
    ls = LawSet(g, base=lambda s, a: "N")
    ls.learn(rows, b, remember=True)
    p = ls.proposals
    assert isinstance(p, Proposals) and p == proposals_of(ls)
    assert p.proposed == p.dropped + p.refuted + p.admitted and p.admitted == ls.cost["accepted"]


def test_on_a_window_a_retired_law_waits_for_a_turnover_before_it_can_return():
    g = grammar()
    b = Buffer(g, window=40)
    s1 = (1, 0, 0, 0, 0, 0)
    b.extend(Transition(s1, "go", "E", group=j % 5, id=f"a{j}") for j in range(40))        # the old world: f0 → E
    ls = LawSet(g, base=lambda s, a: "N", recent=5, readmit="grown", sized=False)
    c = Candidate("go", ("f0",), "E")
    assert [law.condition for law in ls.test([c], b, exclude=-1)] == [("f0",)]
    law = ls.laws[0]
    for j in range(6):                                                                  # the world changed
        t = Transition(s1, "go", "N", group=j % 5, id=f"b{j}")
        ls.step(t, b, remember=True)
    assert law.status == "retired"
    assert ls.test([c], b, exclude=-1) == [] and ls.refused_readmit == 1               # stale rows: refused
    assert ls.why_not(c, b, exclude=-1)["failed"] == "readmit"
    b.extend(Transition(s1, "go", "E", group=j % 5, id=f"c{j}") for j in range(40))     # a full turnover later
    assert [law.condition for law in ls.test([c], b, exclude=-1)] == [("f0",)]
