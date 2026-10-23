"""How often a person was wrong (system report, corrections["people"]): a person's label of a stored decision ("human"
or "audit") against the outcome label of the same decision ("outcome"), with a Clopper–Pearson interval."""
import random

import pytest
from scipy.stats import binom

from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage
from solvi.core.store.sysreport import clopper_pearson


def _system(tmp_path):
    rng = random.Random(0)
    cat = Catalog()

    @cat.fn
    def score(x: float) -> float:
        return x
    s = System(cat, [Question("label", "label?", Answer.yes_no())], storage=JSONLStorage(tmp_path / "s.jsonl"))
    s.fit("label", [({"x": x}, x > 0.5) for x in (rng.random() for _ in range(80))], features=["score"])
    return s


def test_person_error_from_paired_labels(tmp_path):
    s = _system(tmp_path)
    res = [s.ask({"x": i / 20}) for i in range(20)]
    truth = [i / 20 > 0.5 for i in range(20)]
    # ana labels decisions 0–9 and is wrong on 2 and 7; bob labels 10–15 and is wrong on 12
    for i in range(10):
        s.teach("label", {"x": i / 20}, truth[i] != (i in (2, 7)), of=res[i].stored_id, by="ana")
    for i in range(10, 16):
        s.teach("label", {"x": i / 20}, truth[i] != (i == 12), of=res[i].stored_id, by="bob")
    # outcomes known for 0–5 and 10–13 (and 18, which no person labelled)
    for i in list(range(6)) + list(range(10, 14)) + [18]:
        s.outcome(res[i], truth[i], by="carrier")
    P = s.report().corrections["people"]
    assert (P["paired"], P["wrong"]) == (10, 2) and P["error"] == pytest.approx(0.2)
    assert P["by"]["ana"]["paired"] == 6 and P["by"]["ana"]["wrong"] == 1            # 7 has no outcome
    assert P["by"]["bob"]["paired"] == 4 and P["by"]["bob"]["wrong"] == 1
    assert P["interval"] == pytest.approx(list(clopper_pearson(2, 10)))
    assert P["by_question"]["label"]["paired"] == 10
    assert "People: wrong on 2 of 10" in str(s.report())


def test_no_pairs_no_line(tmp_path):
    s = _system(tmp_path)
    r = s.ask({"x": 0.9})
    s.teach("label", {"x": 0.9}, True, of=r.stored_id, by="ana")                     # no outcome
    P = s.report().corrections["people"]
    assert P["paired"] == 0 and P["error"] is None and P["interval"] == [0.0, 1.0]
    assert "People:" not in str(s.report())


@pytest.mark.parametrize("k,n", [(0, 10), (2, 10), (10, 10), (1, 50), (17, 40)])
def test_clopper_pearson_known_values(k, n):
    """Clopper & Pearson (1934): the bounds solve P(Bin(n, lo) >= k) = a/2 and P(Bin(n, hi) <= k) = a/2; for k = 0 the
    upper bound is 1 - (a/2)^(1/n) (0.3085 for n = 10 at 95%)."""
    lo, hi = clopper_pearson(k, n)
    if k == 0:
        assert lo == 0 and hi == pytest.approx(1 - 0.025 ** (1 / n))
    else:
        assert binom.sf(k - 1, n, lo) == pytest.approx(0.025, rel=1e-6)
    if k == n:
        assert hi == 1 and lo == pytest.approx(0.025 ** (1 / n))
    else:
        assert binom.cdf(k, n, hi) == pytest.approx(0.025, rel=1e-6)
    assert clopper_pearson(0, 10)[1] == pytest.approx(0.3085, abs=1e-4)


def test_a_person_label_after_the_outcome_is_not_judged(tmp_path):
    s = _system(tmp_path)
    res = [s.ask({"x": i / 20}) for i in range(6)]
    truth = [i / 20 > 0.5 for i in range(6)]
    for i in range(3):                                         # before the outcome: judged (0 wrong)
        s.teach("label", {"x": i / 20}, truth[i], of=res[i].stored_id, by="ana")
    for i in range(6):
        s.outcome(res[i], truth[i], by="carrier")
    for i in range(3, 6):                                      # after the outcome: a correction, not a judgement
        s.teach("label", {"x": i / 20}, not truth[i], of=res[i].stored_id, by="bob")
    P = s.report().corrections["people"]
    assert (P["paired"], P["wrong"], P["after_outcome"]) == (3, 0, 3) and "bob" not in P["by"]
    assert "before its first outcome" in P["rule"]
    assert P["by_question"] == s.report(question="label").corrections["people"]["by_question"]
