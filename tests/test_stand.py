"""The task stand (benchmarks/tasks): every task's scorer, run as a reader runs it, on a tiny fixture written here — so a
change to a scorer or to the stand's common code cannot break the published numbers silently. No download, no model, no
LLM call. Not run by default: `pytest -m stand`."""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.stand
TASKS = Path(__file__).resolve().parents[1] / "benchmarks" / "tasks"


def jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def score(data, task, *args):
    """Run <task>/score.py on the fixture data → its JSON output."""
    env = {**os.environ, "STAND_DATA": str(data)}
    out = subprocess.run([sys.executable, str(TASKS / task / "score.py"), *map(str, args)], env=env,
                         capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_ragtruth_scorer_counts_f1_and_the_answered_alone_part(tmp_path):
    d = tmp_path / "data"
    jsonl(d / "ragtruth/prepared/eval.jsonl", [{"id": "1", "task_type": "QA", "hallucinated": True},
                                               {"id": "2", "task_type": "QA", "hallucinated": False},
                                               {"id": "3", "task_type": "Summary", "hallucinated": True}])
    pred = jsonl(tmp_path / "p.jsonl", [{"id": "1", "hallucinated": True}, {"id": "2", "hallucinated": True, "escalate": True},
                                        {"id": "3", "hallucinated": False}])
    r = score(d, "ragtruth", pred)
    assert r["all"]["precision"] == 0.5 and r["all"]["recall"] == 0.5 and r["all"]["f1"] == 0.5
    assert r["answered_alone"]["all"]["n"] == 2


def test_cuad_scorer_counts_found_quotes_false_claims_and_verbatim(tmp_path):
    d = tmp_path / "data"
    text = "This Agreement is governed by the laws of the State of Delaware. Either party may terminate it."
    (d / "cuad/prepared/contracts").mkdir(parents=True)
    (d / "cuad/prepared/contracts/c1.txt").write_text(text, encoding="utf-8")
    gold = [{"id": "c1__Governing Law", "contract": "c1", "category": "Governing Law", "is_impossible": False,
             "answers": [{"text": "governed by the laws of the State of Delaware", "start": 18}]},
            {"id": "c1__Non-Compete", "contract": "c1", "category": "Non-Compete", "is_impossible": True, "answers": []}]
    jsonl(d / "cuad/prepared/eval.jsonl", gold)
    pred = jsonl(tmp_path / "p.jsonl", [
        {"id": "c1__Governing Law", "present": True, "quotes": ["governed by the laws of the State of Delaware"]},
        {"id": "c1__Non-Compete", "present": True, "quotes": ["not a passage of this contract"], "escalate": True}])
    r = score(d, "cuad", pred)
    assert r["found_of_present"] == "1 of 1" and r["false_claims_where_absent"] == "1 of 1"
    assert r["quotes_verbatim"] == "1 of 2" and r["escalated"] == 1 and r["wrong_among_answered_alone"] == 0.0


def test_banking77_scorer_splits_the_stream_at_the_shift_and_reads_the_flag(tmp_path):
    d = tmp_path / "data"
    stream = [{"n": 0, "intent": "a", "phase": "before", "known": True}, {"n": 1, "intent": "b", "phase": "before", "known": True},
              {"n": 1000, "intent": "x", "phase": "after", "known": False}, {"n": 1001, "intent": "a", "phase": "after", "known": True}]
    jsonl(d / "banking77/prepared/stream.jsonl", stream)
    pred = jsonl(tmp_path / "p.jsonl", [{"n": 0, "intent": "a", "escalate": False}, {"n": 1, "intent": "a", "escalate": True},
                                        {"n": 1000, "intent": "a", "escalate": False}, {"n": 1001, "intent": "a", "escalate": False},
                                        {"drift_flag_at": 1001}])
    r = score(d, "banking77", pred)
    assert r["before"]["answered_alone"] == 0.5 and r["before"]["error_among_answered"] == 0.0
    assert r["after"]["error_among_answered"] == 0.5 and not r["promise_kept_after"]
    assert r["drift"] == "1 requests after the shift"


def test_credit_scorer_compares_with_the_reference_policy_and_costs_the_new_applications(tmp_path):
    sys.path.insert(0, str(TASKS / "credit"))
    try:
        from reference import decide
    finally:
        sys.path.remove(str(TASKS / "credit"))
    base = {"checking_account": "below 0 DM", "duration_months": 30, "credit_amount_dm": 9000, "savings": "below 100 DM",
            "employed_since": "less than 1 year", "credit_history": "existing credits paid back duly till now",
            "other_installment_plans": "none", "property": "unknown or none", "other_debtors": "none", "purpose": "education"}
    hist = [{"id": "h1", **base, "outcome": "bad"}]
    new = [{"id": "n1", **base, "outcome": "bad"}, {"id": "n2", **base, "checking_account": "no checking account",
                                                   "duration_months": 6, "credit_amount_dm": 500, "outcome": "good"}]
    d = tmp_path / "data"
    jsonl(d / "credit/prepared/applications_history.jsonl", hist)
    jsonl(d / "credit/prepared/applications_new.jsonl", new)
    rows = [{"id": a["id"], "version": v, "decision": decide(a, v)[0]} for a in hist + new for v in (1, 2)]
    rows[-1] = {**rows[-1], "decision": "refuse"}                    # one decision that is not the policy's
    r = score(d, "credit", jsonl(tmp_path / "p.jsonl", rows))
    assert r["v1"]["equal_to_reference"] == "3 of 3" and r["v2"]["equal_to_reference"] == "2 of 3"
    assert r["cost_approve_all"] == 5 and r["cost_refuse_all"] == 1


def test_bird_scorer_runs_both_queries_and_compares_their_rows(tmp_path):
    d = tmp_path / "data"
    db = d / "bird/db/shop.sqlite"
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.executescript("create table t (a int); insert into t values (1), (2), (3);")
    con.commit()
    con.close()
    qs = [{"id": f"q{i}", "db_path": "db/shop.sqlite", "sql": "select count(*) from t", "difficulty": "simple"} for i in range(4)]
    jsonl(d / "bird/prepared/questions.jsonl", qs)
    (d / "bird/prepared/eval_150_ids.json").write_text(json.dumps([q["id"] for q in qs]))
    pred = jsonl(tmp_path / "p.jsonl", [{"id": "q0", "sql": "select 3"}, {"id": "q1", "sql": "select 4"},
                                        {"id": "q2", "sql": "select nothing from nowhere"}, {"id": "q3", "sql": None}])
    r = score(d, "bird", pred)
    assert (r["right"], r["wrong_result"], r["sql_error"], r["no_query"]) == (1, 1, 1, 1)
    assert r["wrong_among_answered"] == round(2 / 3, 3)


def test_abtbuy_scorer_counts_f1_conflicts_and_errors_answered_alone(tmp_path):
    d = tmp_path / "data"
    jsonl(d / "abtbuy/prepared/pairs_eval_test.jsonl", [{"abt": "a1", "buy": "b1", "match": True},
                                                       {"abt": "a1", "buy": "b2", "match": False},
                                                       {"abt": "a2", "buy": "b2", "match": True}])
    pred = jsonl(tmp_path / "p.jsonl", [{"abt": "a1", "buy": "b1", "match": True}, {"abt": "a1", "buy": "b2", "match": True},
                                        {"abt": "a2", "buy": "b2", "match": False, "escalate": True}])
    r = score(d, "abtbuy", pred)
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["conflicts"] == 1
    assert r["escalated"] == 1 and r["errors_answered_alone"] == 0.5


def test_naturalplan_scorer_asks_the_benchmarks_evaluators(tmp_path):
    d = tmp_path / "data"
    repo = d / "naturalplan/repo"
    repo.mkdir(parents=True)
    # stand-ins with the evaluators' interface: a plan is right when its text equals the golden plan
    (repo / "evaluate_calendar_scheduling.py").write_text("def _parse_response(t):\n    return (t.strip(), 0)\n")
    (repo / "evaluate_meeting_planning.py").write_text(
        "def process_constraints(c):\n    return c\n"
        "def parse_text_plan(t):\n    return t.strip()\n"
        "def validator_from_text(p, cons, start, t0, dist):\n    return p.strip()\n")
    (repo / "evaluate_trip_planning.py").write_text(
        "def parse_response(t):\n    return t.strip()\n"
        "def compute_example_score(cities, durations, plan):\n    return float(plan == cities)\n")
    rows = {"calendar_scheduling": {"id": "c", "golden_plan": "Monday, 9:00 - 9:30"},
            "meeting_planning": {"id": "m", "golden_plan": "meet Ann", "constraints": [["A", "9:00AM"]], "dist_matrix": {}},
            "trip_planning": {"id": "t", "golden_plan": "", "cities": "Oslo**Rome", "durations": "2**3"}}
    for task, row in rows.items():
        jsonl(d / f"naturalplan/prepared/{task}_eval.jsonl", [row])
    pred = jsonl(tmp_path / "p.jsonl", [{"id": "c", "text": "Monday, 9:00 - 9:30"}, {"id": "m", "text": "meet Bob"},
                                        {"id": "t", "text": "Oslo**Rome", "escalate": True}])
    r = score(d, "naturalplan", pred)
    assert r["calendar_scheduling"]["right"] == 1 and r["meeting_planning"]["right"] == 0
    assert r["trip_planning"]["answered"] == 0


def test_nab_scorer_finds_windows_merges_alerts_and_skips_the_warm_up(tmp_path):
    d = tmp_path / "data"
    ts = [f"2026-01-01 00:{i // 60:02d}:{i % 60:02d}" for i in range(200)]
    csv = d / "nab/repo/data/real/s.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("timestamp,value\n" + "".join(f"{t},{i}\n" for i, t in enumerate(ts)))
    jsonl(d / "nab/prepared/series.jsonl", [{"series": "real/s.csv", "split": "eval", "windows": [[ts[100], ts[110]]]}])
    # an alert in the warm-up (ignored), one in the window, two close together outside it (one false event)
    pred = jsonl(tmp_path / "p.jsonl", [{"series": "real/s.csv", "alerts": [ts[5], ts[105], ts[150], ts[151]]}])
    r = score(d, "nab", pred)
    assert (r["windows"], r["found"], r["events"], r["false_alarms"]) == (1, 1, 2, 1)


def test_taubench_scorer_counts_solved_tasks_and_changes_not_in_gold(tmp_path):
    d = tmp_path / "data"
    (d / "taubench/prepared").mkdir(parents=True)
    (d / "taubench/prepared/eval_30_ids.json").write_text(json.dumps(["retail_test_1", "retail_test_2"]))
    row = {"steps": 4, "writes": [["cancel_pending_order", {"order_id": "#1"}]], "gold_writes": 1, "writes_refused_by_env": 0}
    pred = jsonl(tmp_path / "p.jsonl", [{"id": "retail_test_1", "reward": 1.0, "writes_not_in_gold": 0, **row},
                                        {"id": "retail_test_2", "reward": 0.0, "writes_not_in_gold": 1, **row}])
    r = score(d, "taubench", pred)
    assert (r["solved"], r["changes_not_in_gold"], r["tasks_with_a_change_not_in_gold"], r["missing"]) == (1, 1, 1, 0)


@pytest.mark.parametrize("script", sorted(p.relative_to(TASKS).as_posix() for p in TASKS.glob("*/*.py")))
def test_every_stand_script_compiles(script):
    compile((TASKS / script).read_text(encoding="utf-8"), script, "exec")


# --- stand.py and replies.py: the published numbers, results.json and the packed replies

def stand_module():
    sys.path.insert(0, str(TASKS))
    import stand
    return stand


def test_stand_check_finds_every_quoted_passage_of_the_docs_with_the_numbers_of_results_json():
    out = subprocess.run([sys.executable, str(TASKS / "stand.py"), "check"], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "every published number matches results.json" in out.stdout


def test_stand_check_fails_when_results_json_and_a_doc_disagree_on_one_number(tmp_path, monkeypatch, capsys):
    stand = stand_module()
    res = json.loads(stand.RESULTS.read_text())
    res["numbers"]["cuad/solution:accuracy"] = 0.898                     # the docs say 0.899
    changed = tmp_path / "results.json"
    changed.write_text(json.dumps(res))
    monkeypatch.setattr(stand, "RESULTS", changed)
    with pytest.raises(SystemExit) as e:
        stand.check(argparse.Namespace(measured=None, require=""))
    assert e.value.code == 1
    out = capsys.readouterr().out
    assert "DIFFERS" in out and "0.898" in out and "README.md" in out


def test_stand_check_compares_a_run_and_does_not_compare_a_skipped_task_or_a_timing(tmp_path, monkeypatch, capsys):
    stand = stand_module()
    res = {"numbers": {"cuad/solution:accuracy": 0.899, "abtbuy/baseline:f1": 0.872, "nab/solution:ms_per_decision": 2.65},
           "quoted": []}
    (tmp_path / "results.json").write_text(json.dumps(res))
    monkeypatch.setattr(stand, "RESULTS", tmp_path / "results.json")
    run = {"cuad/solution": {"accuracy": 0.899}, "nab/solution": {"ms_per_decision": 9.0}, "_misses": 0,
           "skipped": {"abtbuy/baseline": "no replies"}}
    (tmp_path / "stand.json").write_text(json.dumps(run))
    stand.check(argparse.Namespace(measured=str(tmp_path / "stand.json"), require="cuad"))
    assert "1 of 3 numbers" in capsys.readouterr().out
    run["cuad/solution"]["accuracy"] = 0.9
    (tmp_path / "stand.json").write_text(json.dumps(run))
    with pytest.raises(SystemExit):
        stand.check(argparse.Namespace(measured=str(tmp_path / "stand.json"), require=""))
    with pytest.raises(SystemExit):                                       # a required task may not be skipped
        run["cuad/solution"]["accuracy"] = 0.899
        (tmp_path / "stand.json").write_text(json.dumps(run))
        stand.check(argparse.Namespace(measured=str(tmp_path / "stand.json"), require="abtbuy"))


def test_stand_formats_numbers_as_the_docs_print_them_rounding_half_up():
    stand = stand_module()
    assert stand.number(0.625, ".2f") == "0.63" and stand.number(0.676098, ".1%") == "67.6%"
    assert stand.number(2000, ",") == "2,000" and stand.number(1.0, ".0%") == "100%"
    assert stand.fill("F1 {a:x|.3f}; {b:y} of {b:y/of}", {"a:x": 0.86, "b:y": 3, "b:y/of": 4}) == "F1 0.860; 3 of 4"


def test_stand_measure_reads_the_final_json_counts_of_and_the_lines_of_a_step():
    stand = stand_module()
    printed = ("head over 17 facts, fitted on 5743 dev-train pairs\n"
               '{\n "head": {"f1": 0.931, "conflicts": 1},\n "verbatim": "263 of 263",\n "ok": true\n}\n')
    got = stand.measure("abtbuy/solution", printed)
    assert got == {"head.f1": 0.931, "head.conflicts": 1, "verbatim": 263, "verbatim/of": 263, "dev_pairs_fitted": 5743}


def test_replies_pack_keeps_what_solvi_reads_and_drops_the_generation_id_provider_and_billing():
    sys.path.insert(0, str(TASKS))
    import replies
    r = {"id": "gen-123", "provider": "X", "created": 1, "model": "m", "system_fingerprint": "fp",
         "choices": [{"index": 0, "finish_reason": "stop", "native_finish_reason": "stop",
                      "message": {"role": "assistant", "content": "{}", "reasoning": "r", "reasoning_details": [{"x": 1}]}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.1, "is_byok": False, "cost_details": {"a": 1},
                   "completion_tokens_details": {"reasoning_tokens": 1, "image_tokens": 0}}}
    assert replies.slim_raw(r) == {
        "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "{}", "reasoning": "r"},
                                   "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "cost": 0.1, "completion_tokens_details": {"reasoning_tokens": 1}}}
    assert replies.SECRET.search("key sk-or-v1-abc") and replies.SECRET.search("write to someone@gmail.com")
    assert not replies.SECRET.search("mia.garcia2723@example.com") and not replies.SECRET.search("ethan.g@email.com")
