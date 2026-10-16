"""run_context: a run id and a parent decision in every record across stores; chronology puts them in one order."""
import sys
from pathlib import Path

from solvi.core.knowledge.store import KnowledgeStore
from solvi.core.store import JSONLStorage, SQLiteStorage, chronology, current_run, run_context

sys.path.insert(0, str(Path(__file__).parent / "semantics"))
from semantics_helpers import load_schema, validate  # noqa: E402

from test_diff_incremental import build  # noqa: E402

STATE = {"text": "pay", "amount": 50, "customer": "ann"}


def test_records_carry_the_run_and_the_parent(tmp_path):
    decisions, other = JSONLStorage(tmp_path / "d.jsonl"), SQLiteStorage(tmp_path / "o.db")
    s = build()
    s.storage = decisions
    first = s.ask(STATE).stored_id                               # outside a run: nothing added
    assert current_run() is None
    with run_context("nightly-7") as rc:
        assert rc.id == "nightly-7" and current_run() == {"id": "nightly-7", "parent": None}
        d1 = s.ask(STATE).stored_id
        with run_context(parent=d1):                             # the run id is kept, the parent set
            r2 = s.ask({**STATE, "amount": 500})
            d2 = r2.stored_id
            other.save(r2)
            decisions.save_correction("approve", STATE, True)
            kb = KnowledgeStore(other)
            kb.add("fact", {"x": 1}, source="human", by="me")
            mem = KnowledgeStore()
            mem.add("fact", {"x": 2}, source="human", by="me")
    assert current_run() is None
    recs = {x.id: x for x in decisions.iter(None)}
    assert recs[first].run is None and "run" not in recs[first].data
    assert recs[d1].run == "nightly-7" and recs[d1].parent is None
    assert recs[d2].run == "nightly-7" and recs[d2].parent == d1
    assert all(x.run == "nightly-7" for x in other.iter(None))
    assert mem.journal[-1]["run"] == {"id": "nightly-7", "parent": d1}
    assert decisions.verify()["ok"] and other.verify()["ok"]
    schema = load_schema()
    plain = JSONLStorage(tmp_path / "p.jsonl")                  # a catalog without a model: its records validate whole
    with run_context("nightly-7", parent=d1):
        pid = plain.save(_plain_system().ask({"amount": 5}))
        plain.save_correction("big", {"amount": 5}, True)
    plain.redact(pid)
    for x in plain.iter(None, redacted=True):
        assert not validate(x.data, schema), x.kind
    ch = chronology({"decisions": decisions, "other": other}, run="nightly-7")
    assert [c["time"] for c in ch] == sorted(c["time"] for c in ch)
    assert {c["store"] for c in ch} == {"decisions", "other"} and all(c["run"] == "nightly-7" for c in ch)
    assert len(chronology(decisions, kb)) == len(list(decisions.iter(None))) + len(list(other.iter(None)))
    decisions.redact(d1)                                         # the run stays with a redacted record
    assert decisions.record(d1)["run"]["id"] == "nightly-7" and decisions.verify()["ok"]


def _plain_system():
    from solvi import Answer, Catalog, Question, System
    cat = Catalog()

    @cat.rule("big")
    def big(amount) -> bool:
        return amount > 10
    return System(cat, [Question("big", "Big?", Answer.yes_no())])


def test_a_new_run_id_when_none_is_given():
    with run_context() as a:
        with run_context() as b:
            assert b.id == a.id
    with run_context() as c:
        assert c.id != a.id and len(c.id) == 16
