"""diff(facts=True): the stored decisions whose facts changed, also where every answer stayed the same."""
from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage
from solvi.core.store.diff import diff


def _system(scale, storage=None):
    cat = Catalog()

    @cat.fn
    def amount(raw: float) -> float:
        return raw * scale

    @cat.rule("big")
    def big(amount: float) -> str:
        return "yes" if amount > 100 else "no"
    return System(cat, [Question("big", "big?", Answer.choice(["yes", "no"]))], storage=storage)


def test_facts_changed_are_listed_even_where_the_answer_stayed(tmp_path):
    st = JSONLStorage(tmp_path / "d.jsonl")
    old = _system(1.0, st)
    for raw in (10, 60, 500):
        old.ask({"raw": raw})
    new = _system(2.0)                                  # 10 → 20 (no), 60 → 120 (no → yes), 500 → 1000 (yes)
    plain = diff(st, new)
    assert len(plain.changed) == 1 and plain.facts is None
    rep = diff(st, new, facts=True)
    assert len(rep.changed) == 1                        # answers: as without facts=
    assert len(rep.facts) == 3
    assert all([f["name"] for f in d["facts"]] == ["amount"] for d in rep.facts)
    first = rep.facts[0]["facts"][0]
    assert (first["old"], first["new"]) == ("10.0", "20.0") and "code" in first["why"]
    assert "3 decision(s) with a changed fact" in str(rep) and "amount 3" in str(rep)
    assert rep.to_dict()["facts"] == rep.facts
    same = diff(st, _system(1.0), facts=True)
    assert same.facts == [] and same.ok
