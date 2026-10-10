"""solvi.core.specialist (stable since 1.1, graduated from solvi.experimental.specialist): a model proposes a typed spec,
code checks it against the source, code renders — and with a TraceStorage every run is one record of kind
"specialist" in the store's hash chain (the chain the decisions use): verify() covers it, replay re-checks and
re-renders from the stored record, replay_runs replays every stored run of a specialist. Without a storage the run's
own hash chain (Trace) and to_dict / replay work as before."""
import json
import re
import subprocess
import sys
import warnings

import pytest
from pydantic import BaseModel

from solvi import Answer, Catalog, Question, System
from solvi.core.specialist import (BLOCKED, CHANGED, DROPPED, SEVERITIES, WARNING, Checked, Issue, Replay, Run,
                                   Specialist, Trace, canonical, replay_runs, sha256)
from solvi.core.store import JSONLStorage, SQLiteStorage

TEXT = "Sales by region: north 120 units, south 80 units, east 45 units."
OTHER = "Staff by site: berlin 30 people, paris 12 people."


class Item(BaseModel):
    label: str
    value: int
    quote: str


class Items(BaseModel):
    items: list[Item]


class Numbers(Specialist):
    """A toy specialist: each value must be quoted from the source, with the number in its quote."""
    name = "numbers"
    version = "1"
    spec_type = Items

    def check(self, spec, source):
        kept, issues = [], []
        for i, it in enumerate(spec.items):
            if it.quote in source and str(it.value) in re.findall(r"\d+", it.quote):
                kept.append(it)
            else:
                issues.append(Issue(DROPPED, "quote_outside", f"{it.label}: {it.quote!r} is not in the source", f"items[{i}]"))
        if not kept:
            return Checked(None, issues + [Issue(BLOCKED, "nothing_verified", "nothing verified")])
        return Checked(Items(items=kept), issues, [f"{x.label} = {x.value}" for x in kept], {"n": len(kept)})

    def render(self, checked):
        return "\n".join(f"{x.label}: {x.value}" for x in checked.verified.items)


class Reported(Numbers):
    name = "reported"

    def fingerprints(self):
        return {"proposer": "rules-1", "renderer": "plain-1"}


def rules(source, question):
    """A rule-based proposer: "<word> <number> <unit>"."""
    return {"items": [{"label": m.group(1), "value": int(m.group(2)), "quote": m.group(0)}
                      for m in re.finditer(r"(\w+) (\d+) \w+", source)]}


def made_up(source, question):
    return {"items": [{"label": "west", "value": 99, "quote": "west 99 units"},
                      {"label": "north", "value": 120, "quote": "north 120 units"}]}


# ------------------------------------------------------------------------------------------------ the contract
def test_without_a_storage_a_run_keeps_its_own_chain_and_replays_from_its_record():
    r = Numbers(rules).run(TEXT, "sales")
    assert r.ok and r.output == "north: 120\nsouth: 80\neast: 45" and r.stored_id is None
    rec = json.loads(json.dumps(r.to_dict()))
    assert set(rec) == {"specialist", "version", "question", "proposal", "issues", "output_sha256", "trace"}
    assert not Trace(rec["trace"]).verify() and Numbers().replay(rec, TEXT).ok and Numbers().replay(r).ok
    assert Numbers().replay(rec, TEXT + " ").problems == ["the source is not the one the run read (its sha256 differs)"]
    bad = json.loads(json.dumps(rec))
    bad["trace"][1]["data"]["spec"]["items"][0]["value"] = 121
    assert any("its hash does not match" in p for p in Numbers().replay(bad, TEXT).problems)
    assert not Numbers().replay(rec).ok and Numbers().replay(r.to_dict(with_source=True)).ok
    assert isinstance(Numbers().replay(rec, TEXT), Replay) and SEVERITIES == (DROPPED, CHANGED, WARNING, BLOCKED)
    assert canonical({"b": 1, "a": "é"}) == '{"a":"é","b":1}'.encode() and sha256("x") == sha256(b"x")


def test_dropped_values_blocked_proposals_and_a_failing_proposer():
    r = Numbers(made_up).run(TEXT)
    assert r.ok and [i.code for i in r.dropped] == ["quote_outside"] and "kept: north = 120" in r.report()
    blocked = Numbers().run(TEXT, proposal="not json")
    assert not blocked.ok and [i.code for i in blocked.issues] == ["invalid_proposal"]

    def boom(source, question):
        raise RuntimeError("model offline")
    failed = Numbers(boom).run(TEXT)
    assert not failed.ok and "model offline" in failed.report() and Numbers().replay(failed.to_dict(), TEXT).ok


# ------------------------------------------------------------------------------------------------ the core store
def test_a_run_is_one_record_of_kind_specialist_in_the_stores_chain(tmp_path):
    store = JSONLStorage(str(tmp_path / "s.jsonl"))
    sp = Numbers(rules, storage=store)
    r = sp.run(TEXT, "sales")
    assert r.stored_id is not None and len(list(store.iter("specialist"))) == 1
    d = store.record(r.stored_id)
    assert d["kind"] == "specialist" and d["specialist"] == "numbers" and d["version"] == "1"
    assert d["question"] == "sales" and d["source_sha256"] == sha256(TEXT.encode()) and d["source_chars"] == len(TEXT)
    assert d["proposer"] == "rules" and d["proposal"] == rules(TEXT, None)
    assert d["issues"] == [] and d["kept"] == ["north = 120", "south = 80", "east = 45"] and d["meta"] == {"n": 3}
    assert d["verified"] == rules(TEXT, None) and d["renderer"] == "numbers 1"
    assert d["output_sha256"] == sha256(r.output.encode()) and d["output_bytes"] == len(r.output.encode())
    assert d["trace_head"] == r.trace.head and "fingerprints" not in d and "source" not in d
    assert set(r.to_dict()) == set(Numbers(rules).run(TEXT).to_dict())     # to_dict keeps its shape
    assert store.verify()["ok"]
    rep = Reported(rules, storage=store).run(TEXT)
    assert store.record(rep.stored_id)["fingerprints"] == {"proposer": "rules-1", "renderer": "plain-1"}


def test_verify_catches_an_edited_specialist_record(tmp_path):
    path = tmp_path / "s.jsonl"
    store = JSONLStorage(str(path))
    Numbers(rules, storage=store).run(TEXT)
    Numbers(rules, storage=store).run(OTHER)
    store.close()
    lines = path.read_text().splitlines()
    rec = json.loads(lines[0])
    rec["output_sha256"] = "0" * 64
    lines[0] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n")
    got = JSONLStorage(str(path)).verify()
    assert not got["ok"] and any("hash does not match" in p[2] for p in got["problems"])


def test_replay_from_the_store_by_id_or_record_gives_the_same_issues_and_bytes(tmp_path):
    store = SQLiteStorage(tmp_path / "s.db")
    sp = Numbers(made_up, storage=store)
    r = sp.run(TEXT)
    assert sp.replay(r.stored_id, TEXT).ok                              # by id, from the specialist's own store
    assert Numbers().replay(r.stored_id, TEXT, storage=store).ok        # another instance, the store given
    d = store.record(r.stored_id)
    assert Numbers().replay(d, TEXT).ok
    assert Numbers().replay(d, OTHER).problems == ["the source is not the one the run read (its sha256 differs)"]
    assert not Numbers().replay(d).ok                                   # the store keeps no source: pass it
    edited = {**d, "issues": []}
    assert any("edited after it was stored" in p for p in Numbers().replay(edited, TEXT).problems)

    class Stricter(Numbers):                                            # the check changed since the run
        def check(self, spec, source):
            out = super().check(spec, source)
            out.issues.append(Issue(WARNING, "new_rule", "a rule added later"))
            return out
    assert "the check gives other issues now" in Stricter().replay(d, TEXT).problems
    assert any("recorded by" in p for p in Reported().replay(d, TEXT).problems)
    with pytest.raises(ValueError, match="no storage"):
        Numbers().replay(r.stored_id, TEXT)
    blocked = sp.run(TEXT, proposal="not json")
    assert store.record(blocked.stored_id)["proposal_error"] and sp.replay(blocked.stored_id, TEXT).ok


def test_two_specialists_in_one_store_and_replay_runs(tmp_path):
    store = JSONLStorage(str(tmp_path / "s.jsonl"))
    a, b = Numbers(rules, storage=store), Reported(rules, storage=store)
    for t in (TEXT, OTHER):
        a.run(t)
        b.run(t)
    a.run(TEXT, proposal="not json")
    assert len(list(store.iter("specialist"))) == 5 and store.verify()["ok"]
    assert replay_runs(store, Numbers(), [TEXT, OTHER]) == []
    assert replay_runs(store, Reported(), {sha256(t.encode()): t for t in (TEXT, OTHER)}) == []
    missing = replay_runs(store, Numbers(), [TEXT])
    assert len(missing) == 1 and "no source" in missing[0]["problems"][0] and set(missing[0]) == {"id", "seq", "problems"}


def test_decisions_and_specialist_runs_share_one_chain_and_the_report_counts_the_runs(tmp_path):
    from solvi.core.store.sysreport import system_report
    store = JSONLStorage(str(tmp_path / "s.jsonl"))
    cat = Catalog()

    @cat.rule("big")
    def decide_big(x: int) -> bool:
        return x > 10
    s = System(cat, [Question("big", "big?", Answer.yes_no())], storage=store)
    s.ask({"x": 3})
    Numbers(rules, storage=store).run(TEXT)
    s.ask({"x": 30})
    assert [r.kind for r in store.iter(None)] == ["ask", "specialist", "ask"] and store.verify()["ok"]
    assert len(store.replay_all(s)) == 0
    rep = system_report(store)
    assert rep.period["decisions"] == 2 and rep.period["specialists"] == {"numbers": 1}
    assert "Specialist runs: numbers 1." in str(rep)


def test_a_storage_given_as_a_path(tmp_path):
    sp = Numbers(rules, storage=str(tmp_path / "s.jsonl"))
    r = sp.run(TEXT)
    assert sp.storage.record(r.stored_id)["kind"] == "specialist"


# ------------------------------------------------------------------------------------------------ the move
def test_the_old_path_re_exports_the_same_objects_with_the_experimental_warning():
    sys.modules.pop("solvi.experimental.specialist", None)
    from solvi.core import ExperimentalWarning, specialist
    with pytest.warns(ExperimentalWarning, match="graduated in 1.1.*solvi.core.specialist"):
        from solvi.experimental import specialist as old
    for name in specialist.__all__:
        assert getattr(old, name) is getattr(specialist, name), name
    from solvi.experimental import STATUS
    assert "specialist" not in STATUS and "charts" in STATUS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.charts import ChartSpecialist
    assert issubclass(ChartSpecialist, specialist.Specialist)
    assert {"Run", "Replay", "Checked", "Issue", "align", "Reading", "Agreed", "Alignment", "interval_overlap",
            "box_iou", "replay_runs", "canonical", "sha256"} <= set(specialist.__all__)
    assert isinstance(Run, type)


def test_the_moved_module_keeps_its_fingerprint_name_and_imports_nothing_experimental():
    from solvi.core.provenance import fp_module
    assert fp_module("solvi.core.specialist", "Specialist") == "solvi.specialist"
    code = ("import sys, warnings\nwarnings.simplefilter('error')\nimport solvi.core.specialist\n"
            "assert not [m for m in sys.modules if m.startswith('solvi.experimental.')]\n")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr[-2000:]


def test_the_charts_specialist_stores_its_runs(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solvi.experimental.charts import ChartSpecialist, RuleProposer
    store = JSONLStorage(str(tmp_path / "c.jsonl"))
    text = "Revenue by region: Europe 42%, North America 35%, Asia-Pacific 23%."
    sp = ChartSpecialist(RuleProposer(), storage=store)
    r = sp.run(text, "revenue by region")
    assert r.ok and store.record(r.stored_id)["specialist"] == "charts" and sp.replay(r.stored_id, text).ok
    assert replay_runs(store, ChartSpecialist(), [text]) == []
