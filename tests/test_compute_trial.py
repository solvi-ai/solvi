import json
import pytest

from benchmarks.compute_tasks import generate, ARMS, FAMILIES
from benchmarks.compute_trial import trial


def _truth(fixture):
    values=dict(fixture["initial_inputs"])
    snapshots=[]
    for update in fixture["updates"]:
        values.update(update["changes"])
        for node in fixture["spec"]["nodes"]:
            x,y=(values[k] for k in node["deps"])
            kind=node["kind"]
            value={"blend":lambda:x+y*y,"add":lambda:x+y,"sub":lambda:x-y,
                   "mul":lambda:x*y,"max":lambda:max(x,y),"min":lambda:min(x,y),
                   "lt":lambda:x<y,"eq":lambda:x==y}[kind]()
            values[node["name"]]=value
        snapshots.append({k:values[k] for k in fixture["spec"]["outputs"]})
    return snapshots


@pytest.mark.parametrize("family",FAMILIES)
def test_fixture_repeat_revision_typed_and_distinct(family):
    fixture=generate(17,family)
    assert fixture==generate(17,family)
    assert len(fixture["updates"])==12
    assert fixture["updates"][4]["changes"]==fixture["updates"][9]["changes"]=={}
    assert fixture["updates"][6]["revision"]=={"kind":"add","revision":"r2"}
    assert all(type(v) is int and 0<=v<=3 for v in fixture["initial_inputs"].values())
    assert len({json.dumps(generate(17,f)["spec"]) for f in FAMILIES})==3


@pytest.mark.parametrize("arm",ARMS)
def test_all_arms_conserve_small_global_budget_and_denominator(arm):
    result=trial(17,"reconcile",400,arm)
    assert len(result["epochs"])==12
    assert result["ledger"]["spent"]<=400
    assert sum(result["ledger"]["counts"].values())==result["ledger"]["spent"]
    assert all(a["spent"]<=b["spent"] for a,b in zip(result["charges"],result["charges"][1:]))
    assert result["ledger"]["counts"]["host_construction"]>0
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("arm",("DIRECT","DIRTY","FIFO","CHEAP","NO_LEARN"))
def test_known_answers_revision_and_pruned_noise(family,arm):
    result=trial(17,family,100000,arm)
    assert [e["outputs"] for e in result["epochs"]]==_truth(result)
    assert all(e["status"]=="complete" for e in result["epochs"])
    # No producer for the irrelevant e*f branch: no event can have its isolated phase count.
    # Every complete DIRECT epoch has exactly five relevant callbacks.
    if arm=="DIRECT":
        assert len(result["callbackevents"])==60
    assert all(e["accepted"] for e in result["callbackevents"])


def test_dirty_repeat_reuses_and_revision_invalidates():
    result=trial(17,"reconcile",100000,"DIRTY")
    events=result["callbackevents"]
    assert not [e for e in events if e["phase"] in ("epoch:4","epoch:9")]
    assert any(e["kind"]=="add" and e["phase"]=="epoch:6" for e in events)


def test_unpaid_source_read_never_exposes_stale_cache():
    result=trial(17,"reconcile",0,"DIRTY")
    assert all(e["outputs"]=={} for e in result["epochs"])
    for limit in range(100,250,10):
        result=trial(17,"reconcile",limit,"DIRTY")
        gold=_truth(result)
        for epoch,expected in zip(result["epochs"],gold):
            assert all(type(v) is type(expected[k]) and v==expected[k] for k,v in epoch["outputs"].items())


@pytest.mark.parametrize("arm",("DISPATCH","NO_CACHE"))
def test_interrupted_acquisition_keeps_global_spend(arm):
    result=trial(17,"reconcile",400,arm)
    assert result["acquisition"]["status"]=="budget"
    assert result["acquisition"]["spent"]>0
    setup=result["ledger"]["counts"]["host_construction"]
    assert setup+result["acquisition"]["spent"]<=result["ledger"]["spent"]<=400
    assert sum(row["amount"] for row in result["charges"])==result["ledger"]["spent"]


@pytest.mark.parametrize("family",FAMILIES)
@pytest.mark.parametrize("arm",("DISPATCH","NO_CACHE"))
def test_learning_uses_actual_two_trace_sessions_and_full_global_budget(family,arm):
    result=trial(17,family,100000,arm)
    assert result["acquisition"]["status"]=="acquired"
    assert result["acquisition"]["spent"]>0
    training=[e for e in result["callbackevents"] if e["phase"]=="training"]
    assert [e["args"] for e in training]==[[0,1],[1,2]]
    assert result["ledger"]["counts"]["operation_trace"]>0
    assert [e["outputs"] for e in result["epochs"]]==_truth(result)
    # A learned callable may cost more; acquisition never forces it into the policy.
    assert not [e for e in result["callbackevents"] if e["opname"].startswith("learned:")]


def test_dirty_hot_outputs_have_no_redundant_topological_pass():
    row = trial(17, "invoice", 100000, "DIRTY")
    assert row["epochs"][4]["spent"] == row["epochs"][3]["spent"]
    assert row["epochs"][9]["spent"] == row["epochs"][8]["spent"]


def test_no_cache_keeps_one_preprocessed_session():
    enabled = trial(17, "invoice", 100000, "DISPATCH")
    disabled = trial(17, "invoice", 100000, "NO_CACHE")
    assert enabled["ledger"]["counts"]["compute_setup"] == disabled["ledger"]["counts"]["compute_setup"]
    assert len([e for e in disabled["callbackevents"] if e["phase"] == "epoch:4"]) == 5
