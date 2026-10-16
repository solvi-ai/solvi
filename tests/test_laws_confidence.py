"""solvi.experimental.laws: per-law estimated confidence (posterior mean) and tail abstention (LawConfidence). Synthetic data only."""
from __future__ import annotations

import random
import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import (Abstained, Atom, Buffer, Grammar, LawConfidence, LawSet, Transition,
                                         fisher_p)

NA = 8          # f0..f6 plain atoms, f7 the proxy of the hidden cause (when the world has one)


def grammar():
    return Grammar([Atom(f"f{i}", (lambda s, i=i: s[i] == 1), feature=f"f{i}") for i in range(NA)], k=2)


def test_fisher_known_answers():
    assert fisher_p(3, 1, 1, 3) == pytest.approx(0.4857, abs=1e-4)
    assert fisher_p(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-5)
    assert fisher_p(5, 5, 5, 5) == pytest.approx(1.0)


def world(seed, n, tail, signal):
    """Law: f0 and f1 → E. Inside it a hidden cause makes the observation background (N 0.8 / E 0.1 / F 0.1) at rate
    `tail`; with signal, the cause sits where f7 holds (rate 3·tail there, tail/3 elsewhere; f7 true w.p. 0.25)."""
    rng = random.Random(seed)
    out = []
    for j in range(n):
        s = tuple(int(rng.random() < (0.25 if i == 7 else 0.4)) for i in range(NA))
        if s[0] and s[1]:
            q = (3 * tail if s[7] else tail / 3) if signal else tail
            hidden = rng.random() < q
            o = "E" if not hidden else rng.choices("NEF", (0.8, 0.1, 0.1))[0]
        else:
            o = rng.choices("NEF", (0.8, 0.1, 0.1))[0]
        out.append(Transition(s, "go", o, group=j // 20, id=f"r{j}"))
    return out


def learn(rows):
    g = grammar()
    ls, buf = LawSet(g, base=lambda s, a: "N"), Buffer(g)
    for t in rows:
        ls.step(t, buf, remember=True)
    return ls, buf


def law_of(ls):
    got = [law for law in ls.laws if set(law.condition) == {"f0", "f1"} and law.effect == "E"]
    assert got, [law.text() for law in ls.laws]
    return got[0]


def test_an_exact_law_is_exact_and_confident():
    ls, buf = learn(world(1, 4000, 0.0, False))
    lc = LawConfidence(ls, buf)
    st = lc.fit()[law_of(ls).id]
    assert st.exact and st.hits == st.n and st.confidence == pytest.approx((st.n + 1) / (st.n + 2))
    assert st.atom is None
    assert lc.predict((1, 1, 0, 0, 0, 0, 0, 0), "go", below=0.95) == "E"


def test_a_95_percent_law_is_not_exact_and_its_tail_is_found_where_a_signal_exists():
    ls, buf = learn(world(2, 6000, 0.08, True))
    lc = LawConfidence(ls, buf)
    st = lc.fit()[law_of(ls).id]
    assert not st.exact and st.hits < st.n
    assert st.atom == "f7" and st.when_true < st.confidence < st.when_false
    on, off = (1, 1, 0, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0, 0, 0)
    v, law, c_on = lc.of(on, "go")
    assert v == "E" and law is not None and c_on == st.when_true
    a = lc.predict(on, "go", below=0.9)
    assert isinstance(a, Abstained) and a != "E" and a.prediction == "E"
    assert lc.predict(off, "go", below=0.9) == "E"
    assert lc.predict((0, 0, 0, 0, 0, 0, 0, 0), "go", below=0.99) == "N"      # no law: the base, never abstained


def test_no_tail_split_without_a_signal():
    splits = 0
    for seed in range(10, 20):
        ls, buf = learn(world(seed, 4000, 0.08, False))
        st = LawConfidence(ls, buf).fit()[law_of(ls).id]
        assert not st.exact
        splits += st.atom is not None
    assert splits <= 1


def test_confidence_refits_when_the_buffer_grows():
    rows = world(3, 3000, 0.05, False)
    ls, buf = learn(rows[:2000])
    lc = LawConfidence(ls, buf)
    n0 = lc.fit()[law_of(ls).id].n
    for t in rows[2000:]:
        ls.step(t, buf, remember=True)
    lc.of((1, 1, 0, 0, 0, 0, 0, 0), "go")
    assert lc.stats[law_of(ls).id].n > n0
