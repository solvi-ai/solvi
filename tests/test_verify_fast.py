"""verify(deep=False): the hash chain only (hashes, links, sequence, head, anchor) — not each stored response against its
own summary, which is the slow part on big records."""
import json

from solvi import Answer, Catalog, Question, System
from solvi.core.store import JSONLStorage, SQLiteStorage
from solvi.core.store import record_hash


def _store(st):
    cat = Catalog()

    @cat.rule("team")
    def team(email) -> str:
        return "billing" if "charged" in email else "shipping"
    s = System(cat, [Question("team", "Which team?", Answer.choice(["billing", "shipping"]))], storage=st)
    for i in range(5):
        s.ask({"email": "charged twice" if i % 2 else "where is my parcel"})
    return s


def test_shallow_verify_checks_the_chain_and_skips_the_summaries(tmp_path):
    p = tmp_path / "d.jsonl"
    _store(JSONLStorage(p))
    lines = p.read_text().splitlines()
    last = json.loads(lines[-1])
    last["response"]["results"]["team"]["answer"] = "billing" if last["answers"]["team"][0] == "shipping" else "shipping"
    last["hash"] = record_hash(last)                     # a consistent hash over an inconsistent record
    last["id"] = last["hash"][:16]
    p.write_text("\n".join(lines[:-1] + [json.dumps(last)]) + "\n")
    st = JSONLStorage(p)
    deep, shallow = st.verify(), st.verify(deep=False)
    summary = "stored answers differ"
    assert any(summary in r for _, _, r in deep["problems"])
    assert not any(summary in r for _, _, r in shallow["problems"])
    assert any("stored head" in r for _, _, r in shallow["problems"])     # the chain checks still run (the head file)
    assert shallow["count"] == deep["count"] == 5
    lines = p.read_text().splitlines()
    first = json.loads(lines[0])
    first["answers"]["team"][0] = "edited"
    p.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n")
    assert not JSONLStorage(p).verify(deep=False)["ok"]   # an edit still breaks the hash


def test_sqlite_takes_deep_too(tmp_path):
    st = SQLiteStorage(tmp_path / "d.db")
    _store(st)
    assert st.verify(deep=False)["ok"] and st.verify()["ok"]
