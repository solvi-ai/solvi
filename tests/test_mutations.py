"""solvi.testing.mutations: corrupt a specialist's recorded proposals and count how many corruptions its check catches —
on the charts specialist with the rule-based proposer over three texts of tests/test_charts.py."""
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from test_charts import PRESS, QUARTERS, SURVEY, WASTE  # noqa: E402

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.charts import ChartSpecialist, RuleProposer
from solvi.testing.mutations import change_digit, drop, leaves, mutate, swap  # noqa: E402

TEXTS = {"press": PRESS, "quarters": QUARTERS, "waste": WASTE}
VALUE = r"^/series/\d+/points/\d+/value$"
POINT = r"^/series/\d+/points/\d+$"


def setup(texts=TEXTS):
    sp = ChartSpecialist(RuleProposer())
    return sp, [(k, t, sp.run(t).proposal) for k, t in texts.items()]


def at(proposal, path):
    """The proposal's value at a JSON pointer (the proposal as recorded: its JSON form)."""
    return dict(leaves(proposal.model_dump(mode="json")))[path]


def test_a_changed_digit_in_a_points_value_is_caught_every_time():
    sp, recs = setup()
    rep = mutate(sp, recs, {"digit": change_digit(paths=VALUE)}, n=40, seed=1)
    assert rep.n == 40 and rep.caught == 40 and rep.rate == 1.0
    assert rep.per_kind == {"digit": {"n": 40, "caught": 40, "skipped": 0}}
    props = {k: p for k, _, p in recs}
    for r in rep.rows:
        assert re.match(VALUE, r["path"]) and r["kind"] == "digit" and r["caught"] is True
        old, new = str(at(props[r["id"]], r["path"])), str(r["value"])
        assert len(old) == len(new) and sum(a != b for a, b in zip(old, new)) == 1       # one digit, the quote kept


def test_values_swapped_between_two_points_are_caught_and_whole_points_swapped_are_not():
    sp, recs = setup()
    vals = mutate(sp, recs, {"swap": swap(paths=VALUE)}, n=30, seed=2)
    assert vals.n == 30 and vals.rate == 1.0          # each label now holds a number its quote does not state
    whole = mutate(sp, recs, {"swap": swap(paths=POINT)}, n=30, seed=2)
    assert whole.n == 30 and whole.caught == 0        # label, value and quote move together: the chart is as true


def test_a_dropped_point_is_judged_by_a_relation():
    sp, recs = setup()
    rep = mutate(sp, recs, {"drop": drop(paths=POINT)}, n=30, seed=3)
    assert rep.n == 30 and {r["criterion"] for r in rep.rows} == {"relation"}
    assert rep.rate == 1.0                            # a pie that no longer makes a whole; a total no longer added up
    bars, brecs = setup({"survey": SURVEY})           # bars with no total: nothing relates the points
    plain = mutate(bars, brecs, {"drop": drop(paths=POINT)}, n=10, seed=3)
    assert plain.n == 10 and plain.caught == 0


def test_draws_are_deterministic_and_skips_are_counted():
    sp, recs = setup()
    kinds = {"digit": change_digit(paths=VALUE), "total": change_digit(paths=r"^/total/value$")}
    a = mutate(sp, recs, kinds, n=30, seed=7)
    b = mutate(sp, list(reversed(recs)), kinds, n=30, seed=7)       # records are taken sorted by id
    assert a.rows == b.rows and a.per_kind == b.per_kind
    assert mutate(sp, recs, kinds, n=30, seed=8).rows != a.rows
    assert a.n == 30 and sum(k["n"] for k in a.per_kind.values()) == 30
    assert a.per_kind["total"]["skipped"] > 0 and a.skipped == a.per_kind["total"]["skipped"]   # the press text: no total
    assert all(r["id"] != "press" for r in a.rows if r["kind"] == "total")


def test_a_mutator_of_your_own_and_an_accepted_of_your_own():
    sp, recs = setup()

    def retitle(p, rng):
        p["title"] = "Something else"
        return p, "/title", "Something else"
    rep = mutate(sp, recs, {"title": retitle}, n=6, seed=0)
    assert rep.n == 6 and rep.caught == 0             # a title without numbers is not checked: the new one is accepted
    nothing = mutate(sp, recs, {"title": retitle}, n=6, seed=0, accepted=lambda run: {})
    assert nothing.caught == 6                        # nothing accepted: every mutation counts as caught
