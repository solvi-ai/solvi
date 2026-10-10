"""solvi.experimental.compile: a certificate on fresh examples sized by the hypothesis space — the count a search over
|H| candidates needs (N = max(20, ⌈(ln|H| + ln 1/δ) / ε⌉)), a skew guard that keeps a constant from passing a set whose
answers are nearly all the same, and their use as an extra gate of compile_spec's acceptance with the count in the
record."""
import math

import pytest

from solvi.experimental.compile import Certificate, Spec, compile_spec, fresh_examples_needed, recompile, skew_guard

from test_compile import GOOD, INPUTS, OFF_BY_ONE, POLICY, QS, TEST_OK, Writer


def _truth(x):
    if x["zone"] == "world" and x["weight"] > 30:
        return {"ship": "refused"}
    return {"ship": "free" if x["total"] >= 50 else "paid"}


def _fresh(n, seed=11, totals=range(0, 201)):
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        x = {"zone": rng.choice(["domestic", "world"]), "total": rng.choice(list(totals)), "weight": rng.randint(0, 50)}
        out.append((x, _truth(x)))
    return out


def test_the_count_grows_with_the_log_of_the_hypothesis_space_and_never_falls_below_the_floor():
    assert fresh_examples_needed(1) == math.ceil(math.log(10) / 0.05) == 47
    assert fresh_examples_needed(10 ** 6) == math.ceil((math.log(10 ** 6) + math.log(10)) / 0.05) == 323
    assert fresh_examples_needed(10 ** 6, eps=0.1, delta=0.05) == math.ceil((math.log(10 ** 6) + math.log(20)) / 0.1)
    assert fresh_examples_needed(1, eps=1.0, delta=0.5) == 20                      # the floor
    assert fresh_examples_needed(1, eps=1.0, delta=0.5, at_least=5) == 5
    assert fresh_examples_needed(10 ** 9) > fresh_examples_needed(10 ** 6) > fresh_examples_needed(10 ** 3)
    for bad in ({"n_hypotheses": 0}, {"n_hypotheses": 10, "eps": 0}, {"n_hypotheses": 10, "delta": 1.0},
                {"n_hypotheses": 10, "eps": 1.5}):
        with pytest.raises(ValueError):
            fresh_examples_needed(**bad)


def test_the_skew_guard_refuses_a_set_a_constant_could_pass():
    assert skew_guard([str(i) for i in range(10)]) is None                         # ten distinct answers
    assert skew_guard(["yes"] * 10 + ["no"] * 10) is None                          # ten off the most common one
    why = skew_guard(["no"] * 50 + ["yes"] * 3)
    assert why and "3" in why and "constant" in why
    assert skew_guard(["a"] * 30) is not None
    assert skew_guard([{"q": "a"}, {"q": "b"}] * 5, min_distinct=2) is None       # answers may be dicts
    assert skew_guard(["no"] * 5 + ["yes"] * 2, min_distinct=10, min_off_mode=2) is None


def test_a_certificate_states_its_count_from_the_hypothesis_space_or_a_given_n():
    ex = _fresh(5)
    assert Certificate(ex, n_hypotheses=10 ** 6).needed == 323
    assert Certificate(ex, n=60).needed == 60
    with pytest.raises(ValueError):
        Certificate(ex)                                    # neither |H| nor n
    with pytest.raises(ValueError):
        Certificate(ex, n_hypotheses=10, n=60)             # both


def test_compile_accepts_only_when_both_drafts_pass_the_sized_certificate_and_records_the_count():
    cert = Certificate(_fresh(400), n_hypotheses=10 ** 6)
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]), certificate=cert)
    assert c.accepted and c.reason == "accepted in round 1"
    r = c.record["certificate"]
    assert r["needed"] == 323 and r["checked"] == 400 and r["n_hypotheses"] == 10 ** 6 and r["passed"]
    assert r["eps"] == 0.05 and r["delta"] == 0.1 and r["skew"] is None and r["runs_agree"] is True
    assert r["failures"] == [] and r["distinct"] >= 2


def test_too_few_fresh_examples_is_not_an_acceptance():
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]),
                     certificate=Certificate(_fresh(50), n_hypotheses=10 ** 6))
    assert not c.accepted and "323 fresh examples" in c.reason and "50" in c.reason
    assert c.record["certificate"]["passed"] is False


def test_a_draft_wrong_on_one_fresh_example_is_not_accepted_and_the_failure_is_recorded():
    """Both drafts share the off-by-one at total = 50; the pool, the tests and the examples miss it, the fresh set
    holds it."""
    fresh = _fresh(330) + [({"zone": "domestic", "total": 50, "weight": 1}, {"ship": "free"})]
    w = Writer([[OFF_BY_ONE], [OFF_BY_ONE]], tests=[TEST_OK[1]])
    plain = compile_spec(Spec(POLICY), QS, INPUTS, w, rounds=1)
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[OFF_BY_ONE], [OFF_BY_ONE]], tests=[TEST_OK[1]]), rounds=1,
                     certificate=Certificate(fresh, n_hypotheses=10 ** 6))
    if plain.accepted:                              # the drawn pool did not catch it: the certificate must
        assert not c.accepted and "certificate" in c.reason
        f = c.record["certificate"]["failures"]
        assert f and f[0]["input"]["total"] == 50 and f[0]["expected"] == {"ship": "free"}
    else:
        assert not c.accepted


def test_a_skewed_fresh_set_is_not_a_certificate_even_when_every_answer_matches():
    fresh = [({"zone": "domestic", "total": t, "weight": 1}, {"ship": "free"}) for t in range(50, 201)] * 3
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]),
                     certificate=Certificate(fresh, n_hypotheses=10 ** 6))
    assert not c.accepted and "skew" in c.reason
    assert c.record["certificate"]["skew"] and c.record["certificate"]["failures"] == []


def test_fresh_examples_that_repeat_a_labelled_example_or_a_test_do_not_count():
    ex = [({"zone": "domestic", "total": 50, "weight": 1}, {"ship": "free"})]
    fresh = _fresh(330) + ex * 5 + [(TEST_OK[1]["input"], TEST_OK[1]["expect"])]
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]), examples=ex,
                     certificate=Certificate(fresh, n_hypotheses=10 ** 6))
    r = c.record["certificate"]
    assert c.accepted and r["not_fresh"] == 6 and r["checked"] == 330


def test_a_recompile_meets_the_same_certificate():
    V2 = POLICY.replace("50 or more", "80 or more")
    PATCH = 'def free_shipping(total):\n    return total >= 80\n\nPARTS = {"free_shipping": {"kind": "fn", "clauses": ["c3"]}}\n'
    c = compile_spec(Spec(POLICY), QS, INPUTS, Writer([[GOOD], [GOOD]]))
    old_truth = _fresh(400)                         # labelled by the old policy: the revised module must fail it
    c2 = recompile(c, c.spec.revise(V2), INPUTS, Writer([[PATCH], [PATCH]],
                                                        tests=[{**TEST_OK[1]}]),
                   certificate=Certificate(old_truth, n_hypotheses=10 ** 6))
    assert not c2.accepted and "certificate" in c2.reason
