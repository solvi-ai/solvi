"""A stored record lists the parts its flow did not plan grouped by reason, so a large catalog does not make every
record large; the record loads, replays and verifies as before, and the older per-part form still loads."""
import json

from solvi import Answer, Catalog, Question, System
from solvi.core.schema import load
from solvi.core.store import JSONLStorage


def _system(storage=None, extra=300):
    cat = Catalog()

    @cat.rule("approve")
    def approve(amount) -> bool:
        return amount < 100

    for k in range(extra):                            # unrelated parts, never asked
        ns = {}
        exec(f"def part_number_{k}(text):\n    return len(text) + {k}", ns)
        cat.fn(ns[f"part_number_{k}"])
    return System(cat, [Question("approve", "Approve?", Answer.yes_no())], storage=storage)


def test_skipped_parts_are_a_minor_share_of_the_record(tmp_path):
    p = tmp_path / "s.jsonl"
    s = _system(JSONLStorage(str(p)))
    res = s.ask({"amount": 5, "text": "x"})
    line = p.read_text().strip().splitlines()[-1]
    flow = json.loads(line)["response"]["flow"]
    assert "skipped" not in flow and flow["skipped_rest"]["n"] == 300
    share = len(json.dumps([flow.get("skipped_by"), flow["skipped_rest"]])) / len(line)
    assert share < 0.05, share
    st = JSONLStorage(str(p))
    back = st.get(res.stored_id, s)
    assert back.flow.skipped == res.flow.skipped
    assert st.verify()["ok"] and st.replay_all(s) == []


def test_the_per_part_form_still_loads():
    from solvi.core.runtime import Flow
    s = _system(extra=3)
    d = s.ask({"amount": 5, "text": "x"}, store=False).to_dict()
    assert "skipped" in d["flow"]                     # Response.to_dict() is unchanged
    f = load(Flow, d["flow"], s.catalog)
    assert f.skipped == {f"part_number_{k}": "not needed for questions" for k in range(3)}


def test_a_changed_catalog_gives_back_only_what_the_record_names(tmp_path):
    p = tmp_path / "s.jsonl"
    s = _system(JSONLStorage(str(p)), extra=20)
    res = s.ask({"amount": 5, "text": "x"})
    other = _system(extra=21)                         # one part more since
    back = JSONLStorage(str(p)).get(res.stored_id, other)
    assert back.flow.skipped == {}
    assert back.results["approve"].answer == res.results["approve"].answer
