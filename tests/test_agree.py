"""solvi.core.slow.agree: K generated candidates grouped by a user-given key, the largest group chosen, its share a plain fact —
failed candidates counted in K, `prefer` before the rest, ties to the earliest, nothing voted → the chosen fact missing;
the tally recorded in the trace and recomputed on replay."""
import pytest

from solvi import Answer, Catalog, Question, System
from solvi.core.slow.agree import agree, consensus
from solvi.core.slow.refine import Fail

ROWS = {"select 1": ("A",), "select one": ("A",), "select 2": ("B",), "select none": ()}


def rows_of(sql):
    if sql == "boom":
        raise RuntimeError("no such table")
    return ROWS[sql]


def test_the_largest_group_under_the_key_wins_and_its_share_counts_every_candidate():
    c = consensus(["select 2", "select 1", "select one"], key=rows_of)
    assert c["index"] == 1 and c["value"] == "select 1" and c["share"] == pytest.approx(2 / 3)
    assert c["groups"] == [[1, 2], [0]] and c["k"] == 3
    assert consensus(["select 1", "select one", "select 1"], key=rows_of)["share"] == 1.0


def test_a_failed_candidate_or_key_does_not_vote_but_lowers_the_share():
    c = consensus(["select 1", None, "boom"], key=rows_of)
    assert c["index"] == 0 and c["share"] == pytest.approx(1 / 3)
    why = {r["index"]: r.get("why") for r in c["candidates"]}
    assert why[1] == "no candidate (its generation failed)" and why[2] == "key failed: RuntimeError: no such table"


def test_a_tie_goes_to_the_group_that_starts_first():
    assert consensus(["select 2", "select 1"], key=rows_of)["index"] == 0
    assert consensus(["select 1", "select 2", "select 2", "select one"], key=rows_of)["groups"] == [[0, 3], [1, 2]]


def test_prefer_lets_only_the_preferred_candidates_vote_when_there_are_any():
    c = consensus(["select none", "select none", "select 1"], key=rows_of, prefer=lambda s: bool(rows_of(s)))
    assert c["index"] == 2 and c["share"] == pytest.approx(1 / 3)
    assert [r["voted"] for r in c["candidates"]] == [False, False, True]
    c = consensus(["select none", "select none"], key=rows_of, prefer=lambda s: bool(rows_of(s)))
    assert c["index"] == 0 and c["share"] == 1.0                   # none preferred: all with a key vote


def test_nothing_voted_gives_no_value_and_share_zero():
    c = consensus([None, "boom"], key=rows_of)
    assert c["index"] == -1 and c["value"] is None and c["share"] == 0.0
    with pytest.raises(ValueError, match="no candidates"):
        consensus([], key=rows_of)


def _system():
    cat = Catalog()

    @cat.fn
    def candidates(drafts):
        return drafts
    agree(cat, "sql", "candidates", key=lambda c, db: db[c] if c in db else None)

    @cat.check(hard=True, then={"ok": "no"})
    def all_agree(sql_agreement) -> bool:
        return sql_agreement == 1.0 or Fail(f"only {sql_agreement:.0%} of the candidates agree")

    @cat.rule("ok")
    def ok(sql, all_agree) -> bool:
        return True
    return System(cat, [Question("ok", "Can the query be returned?", Answer.yes_no(), requires=["all_agree"])])


def test_agree_in_a_catalog_gives_the_chosen_value_the_share_and_a_recorded_tally_that_replays():
    s = _system()
    db = {"q1": "A", "q2": "A", "q3": "B"}
    res = s.ask({"drafts": ["q1", "q2", "q2"], "db": db})
    assert res["ok"].answer == "yes" and res.values["sql"] == "q1" and res.values["sql_agreement"] == 1.0
    tally = res.values["sql_tally"]
    assert tally["groups"] == [[0, 1, 2]] and "value" not in tally
    assert res.trace.replay(s)["ok"]
    res = s.ask({"drafts": ["q1", "q3", None], "db": db})
    assert res["ok"].answer == "no" and res["ok"].why == "hard check all_agree is false: only 33% of the candidates agree"


def test_agree_with_nothing_chosen_leaves_the_share_and_the_tally_and_the_question_abstains_with_the_cause():
    s = _system()
    res = s.ask({"drafts": [None, "zz"], "db": {}}, early_exit=False)
    assert res.values["sql_agreement"] == 0.0 and "sql" not in res.values
    r = res["ok"]
    assert r.answer == "no"                                          # the hard check on the share decides first
    rec = next(x for x in res.trace.records if x.name == "sql")
    assert rec.error.startswith("ValueError: no candidate has a key (0: no candidate (its generation failed); 1: no key)")
