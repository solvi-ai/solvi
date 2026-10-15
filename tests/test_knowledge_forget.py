"""KnowledgeStore.forget(source, storage, decide): retract_source → the stored decisions that rested on what was taken
back → re-decided on the journal as it is now (redecide(upto="now")) → one "forget" journal entry naming what was taken
back and which decisions changed."""
import pytest

import solvi
from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage


def _setup(tmp_path):
    st = JSONLStorage(tmp_path / "d.jsonl")
    cat = Catalog()

    @cat.rule("dose")
    def dose(knowledge) -> str:
        return solvi.Knowledge.value(knowledge, "drugX", "max_dose", "unknown")

    @cat.rule("form")
    def form(knowledge) -> str:
        return solvi.Knowledge.value(knowledge, "drugY", "form", "unknown")
    s = System(cat, [Question("dose", "Dose?", Answer.choice(["10mg", "20mg", "40mg", "unknown"])),
                     Question("form", "Form?", Answer.choice(["pill", "drop", "unknown"]))], storage=st)
    km = solvi.Knowledge()
    return st, s, km


def test_forget_takes_back_a_source_redecides_and_journals_one_entry(tmp_path):
    st, s, km = _setup(tmp_path)
    leak = km.tell(("drugX", "max_dose", "40mg"), source="spec", by="patient-42")
    km.tell(("drugY", "form", "pill"), source="spec", by="pharmacy")
    a = s.ask({"knowledge": km.snapshot(about="drugX")})                 # rests on the source's fact
    b = s.ask({"knowledge": km.snapshot(about="drugY")})                 # does not
    km.tell(("drugX", "max_dose", "20mg"), source="spec", by="pharmacy")
    n0 = len(km.store.journal)
    out = km.store.forget("patient-42", st, s, by="dpo", why="erasure request")
    assert out["source"] == "patient-42" and out["retracted"][leak] == ("disputed", "retracted")   # the 20mg disputed it
    assert out["decisions"] == [a.stored_id] and b.stored_id not in out["decisions"]
    assert [(c["id"], c["old"]["dose"][0], c["new"]["dose"][0]) for c in out["changed"]] == [(a.stored_id, "40mg", "20mg")]
    new = km.store.journal[n0:]
    forget = [r for r in new if r["op"] == "forget"]
    assert len(forget) == 1 and forget[0]["source"] == "patient-42" and forget[0]["by"] == "dpo"
    assert forget[0]["retracted"] == [leak] and forget[0]["changed"] == [a.stored_id]
    assert km.store.status(leak) == "retracted"
    again = km.store.rebuild()                                              # the entry replays (a no-op on items)
    assert again.status(leak) == "retracted"


def test_forget_without_a_store_only_retracts_and_journals(tmp_path):
    _, _, km = _setup(tmp_path)
    leak = km.tell(("drugX", "max_dose", "40mg"), source="spec", by="patient-42")
    out = km.store.forget("patient-42")
    assert out["retracted"] == {leak: ("active", "retracted")} and out["decisions"] == [] and out["changed"] == []
    assert km.store.journal[-1]["op"] == "forget"
    with pytest.raises(ValueError, match="decide"):
        km.store.forget("someone", JSONLStorage(tmp_path / "x.jsonl"))
