"""The system report (System.report, solvi.core.store.sysreport, `solvi report --overview`): read from the store alone, its numbers
equal the ones recomputed from the raw records — who answered, the promise against the labels, the dispatcher's final
answers and cost, drift — on an empty store, a store with corrections, a dispatcher's store, a period."""
import json
from collections import Counter

import pytest
from test_dispatch import _hard_slice

from solvi import Answer, Catalog, Decision, Question, System
from solvi.core.store import JSONLStorage, SQLiteStorage
from solvi.cli import main
from solvi.core.dispatch import AskPath, Dispatcher
from solvi.core.guarantees.guarantee import calibrate
from solvi.core.store import open_storage
from solvi.core.store.sysreport import SystemReport, system_report


def guarded(storage):
    """A rule answers the team; a guarantee on a trust score holds back the "hello" e-mails (risk ≤ 0.1)."""
    cat = Catalog()

    @cat.fn
    def trust(email) -> float:
        return 0.1 if "hello" in email else 0.9

    @cat.rule("team")
    def team(email):
        return "billing" if "charged" in email or "hello" in email else "shipping"
    s = System(cat, [Question("team", "Which team?", Answer.choice(["billing", "shipping"]))], storage=storage)
    s.guarantee("team", promise=calibrate([0.1] * 20 + [0.9] * 20, [False] * 20 + [True] * 20, max_risk=0.1,
                                          signal="trust"), signal="trust")
    return s


def raw(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def test_an_empty_store_reports_nothing_happened(tmp_path):
    rep = system_report(JSONLStorage(tmp_path / "empty.jsonl"))
    d = rep.to_dict()
    assert isinstance(rep, SystemReport) and d["period"]["decisions"] == 0 and d["system1"] == {} and d["dispatch"] == {}
    assert d["corrections"]["total"] == 0 and d["cost"]["system1"]["decisions"] == 0
    assert "0 System 1 decision(s)" in str(rep)


def test_a_system_without_storage_says_where_the_report_reads_from():
    with pytest.raises(ValueError, match="storage"):
        guarded(None).report()


def test_who_answered_the_promise_and_the_labels_equal_the_raw_records(tmp_path):
    path = tmp_path / "d.jsonl"
    s = guarded(JSONLStorage(path))
    emails = ["charged twice", "hello", "a parcel is late", "charged again", "hello there", "parcel lost",
              "charged for a parcel", "where is my parcel"]
    ids = [s.ask({"email": e}).stored_id for e in emails]
    store = s.storage
    store.save_correction("team", {"email": "charged twice"}, "billing", label_source="human", of=ids[0])
    store.save_correction("team", {"email": "a parcel is late"}, "billing", label_source="outcome", of=ids[2])  # wrong
    store.save_correction("team", {"email": "hello"}, "shipping", label_source="human", of=ids[1])        # abstained
    store.save_correction("team", {"email": "parcel lost"}, "shipping")                       # no of=: by its input
    store.save_correction("team", {"email": "charged again"}, "billing", label_source="verified", of=ids[3])
    store.save_correction("team", {"email": "never asked"}, "billing")                        # labels nothing
    rep = s.report(drift_window=None)
    p = rep.to_dict()["system1"]["team"]

    recs = raw(path)
    asks = [r for r in recs if r["kind"] == "ask"]
    status = Counter(r["answers"]["team"][2] for r in asks)
    assert (p["asked"], p["alone"], p["abstained"], p["forced"]) == (len(asks), status["ok"], status["abstain"], 0)
    assert p["answered_by"] == {"rule: team": status["ok"]}
    assert p["abstained_by"] == {"low_confidence": status["abstain"]}
    g = [r["response"]["results"]["team"]["extra"]["guarantee"] for r in asks]
    pr = p["promises"][0]
    assert len(p["promises"]) == 1 and pr["fingerprint"] == g[0]["fingerprint"]
    assert (pr["decisions"], pr["alone"], pr["held_back"]) == (len(g), sum(x["answered"] for x in g),
                                                               sum(not x["answered"] for x in g))
    assert (pr["measure"], pr["level"], pr["method"]) == ("risk", 0.1, "crc")
    # labels: human + outcome + human + by-input = 4 measured; the verified one counted apart; one names no decision
    L = p["labels"]
    assert (L["labelled"], L["alone"], L["wrong_alone"]) == (4, 3, 1)
    assert L["by_source"] == {"human": 3, "outcome": 1} and L["verified_not_measured"] == 1
    assert L["error"] == pytest.approx(1 / 3) and L["risk"] == pytest.approx(1 / 4)
    assert pr["labels"]["verdict"].startswith("above the promised level")         # 1 of 4 against 0.1: not significant
    assert rep.corrections["by_source"] == {"human": 4, "outcome": 1, "verified": 1}
    assert rep.corrections["total"] == 6 and rep.corrections["labelled_decisions"] == 5
    assert rep.corrections["unmatched"] == 1
    assert rep.cost["system1"]["decisions"] == len(asks)
    assert rep.cost["system1"]["ms"] == pytest.approx(sum(r["response"]["ms"] for r in asks))
    text = str(rep)
    assert "who answered        rule: team 6" in text and "promised risk ≤ 0.1" in text
    assert "1 verified label(s) not measured" in text


def test_a_promise_broken_on_many_labels_is_said_to_be_broken(tmp_path):
    s = guarded(JSONLStorage(tmp_path / "d.jsonl"))
    for i in range(40):
        r = s.ask({"email": f"parcel {i}"})                        # "shipping", answered alone
        s.storage.save_correction("team", {"email": f"parcel {i}"}, "billing" if i % 2 else "shipping", of=r.stored_id)
    pr = s.report(drift_window=None).to_dict()["system1"]["team"]["promises"][0]
    assert pr["labels"]["wrong_alone"] == 20 and pr["labels"]["verdict"].startswith("above the promise (p =")
    assert pr["labels"]["p_value"] < 1e-6


def test_the_dispatchers_final_answers_cost_and_calibrated_promise_equal_the_raw_records(tmp_path):
    s1, s2, data = _hard_slice(guess_right=0.6, slow_right=0.95)
    path = tmp_path / "d.jsonl"
    store = JSONLStorage(path)
    d = Dispatcher(s1, AskPath(s2), price=(1.0, 2.0), storage=store)
    first = d.ask(data[350][0])                                     # before calibrate: no policy
    rep_cal = d.calibrate(data[:300], max_risk=0.05)
    for st, y in data[300:360]:
        r = d.ask(st)
        store.save_correction("label", st, y, of=r.stored_id)
    # the policy record keeps the chain and the replay; only the decision made before calibrate is under another config
    assert [x[1] for x in d.replay_all()] == [first.stored_id] and store.verify()["ok"]
    p = system_report(open_storage(str(path)), drift_window=None).to_dict()["dispatch"]["label"]

    recs = raw(path)
    ds = [r for r in recs if r["kind"] == "dispatch"]
    teach = {r["of"]: r["answer"] for r in recs if r["kind"] == "teach"}
    assert [r["kind"] for r in recs].count("policy") == 1
    assert p["decisions"] == len(ds) == 61
    assert p["by"] == {k: sum(r["by"] == k for r in ds) for k in ("s1", "s2", "human")}
    assert p["actions"] == dict(Counter(r["action"] for r in ds))
    assert p["slices"] == dict(Counter(r["slice"] for r in ds if r["slice"]))
    assert p["uncalibrated"] == 1 and first.stored_id == ds[0]["id"]
    pol = p["policies"][0]
    assert pol["decisions"] == 60 and pol["level"] == 0.05 and pol["measure"] == "risk"
    assert {k: v["answer"] for k, v in pol["slices"].items()} == {k: v["answer"] for k, v in rep_cal["slices"].items()}
    alone = [r for r in ds if r["by"] in ("s1", "s2") and r["id"] in teach]
    wrong = sum(r["answer"] != teach[r["id"]] for r in alone)
    assert (pol["labels"]["alone"], pol["labels"]["wrong_alone"]) == (len(alone), wrong)
    for who in ("s1", "s2"):
        rows = [r for r in alone if r["by"] == who]
        assert p["labels"][f"by_{who}"]["wrong_alone"] == sum(r["answer"] != teach[r["id"]] for r in rows)
    rep = system_report(open_storage(str(path)), drift_window=None)
    c = rep.cost["dispatch"]["slow_path"]
    assert c["calls"] == sum(r["cost"]["s2"]["calls"] for r in ds)
    assert c["input_tokens"] == sum(r["cost"]["s2"]["input_tokens"] for r in ds)
    assert c["usd"] == pytest.approx(sum(r["cost"]["s2"]["usd"] for r in ds))
    assert c["decisions"] == sum(r["s2"] is not None for r in ds)
    text = str(rep)
    assert "dispatcher, 61 decision(s)" in text and "1 decision(s) under no calibrated policy" in text


def test_drift_is_found_after_the_stream_changes_and_not_run_when_asked_not_to(tmp_path):
    cat = Catalog()

    @cat.rule("ok")
    def ok(x):
        p = 0.99 if x < 150 else 0.6
        return Decision("yes", {"yes": p, "no": 1 - p})
    s = System(cat, [Question("ok", "?", Answer.yes_no(), min_confidence=0.8)], storage=JSONLStorage(tmp_path / "d.jsonl"))
    for i in range(300):
        s.ask({"x": i})
    dr = s.report(drift_window=50).system1["ok"]["drift"]
    assert dr["tested"] and dr["flagged"] and dr["at"] > 150 and dr["signals"] and dr["id"]
    assert s.report(drift_window=None).system1["ok"]["drift"] == {"tested": False, "why": "not run (drift_window=None)"}
    short = s.report(drift_window=250).system1["ok"]["drift"]
    assert not short["tested"] and "fewer than" in short["why"]


def test_a_period_counts_only_its_decisions_and_labels_them_with_later_corrections(tmp_path):
    t = [1_000.0]
    store = SQLiteStorage(tmp_path / "d.db", clock=lambda: t[0])
    s = guarded(store)
    ids = []
    for i in range(6):
        t[0] = 1_000.0 + 100 * i
        ids.append(s.ask({"email": f"parcel {i}"}).stored_id)
    t[0] = 5_000.0
    for i in ids:
        store.save_correction("team", {"email": "?"}, "shipping", of=i)
    rep = s.report(since=1_100.0, until=1_400.0, drift_window=None)
    p = rep.system1["team"]
    assert rep.period["decisions"] == 3 and p["asked"] == 3
    assert p["labels"]["labelled"] == 3 and rep.corrections["in_period"] == 0 and rep.corrections["unmatched"] == 3


def test_solvi_report_overview_prints_the_text_or_the_data(tmp_path, capsys):
    path = str(tmp_path / "d.jsonl")
    s = guarded(JSONLStorage(path))
    for e in ["charged", "hello", "parcel"]:
        s.ask({"email": e})
    assert main(["report", path, "--overview", "--no-drift"]) == 0
    assert "System report" in capsys.readouterr().out
    assert main(["report", path, "--overview", "--json", "--drift-window", "50"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["kind"] == "system" and d["system1"]["team"]["asked"] == 3
    with pytest.raises(SystemExit) as e:
        main(["report", path, "--overview", "--html", str(tmp_path / "x.html")])
    assert e.value.code == 2
