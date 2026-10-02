"""τ-bench retail: did the agent leave the database as the task's gold actions do, and what did it change on the way.

    python taubench/score.py runs/<arm>.jsonl

A row per task, as harness.run_task returns it ("reward" 1.0 = the database ended as the gold actions leave it)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402


def score(pred_file):
    want = json.loads((DATA / "taubench/prepared/eval_30_ids.json").read_text())
    rows = {r["id"]: r for r in read_jsonl(pred_file)}
    got = [rows[i] for i in want if i in rows]
    n = len(want)
    return {"tasks": n, "missing": n - len(got), "solved": sum(r["reward"] == 1.0 for r in got),
            "solve_rate": round(sum(r["reward"] == 1.0 for r in got) / n, 3),
            "tasks_with_a_change_not_in_gold": sum(r["writes_not_in_gold"] > 0 for r in got),
            "changes_not_in_gold": sum(r["writes_not_in_gold"] for r in got),
            "changes_made": sum(len(r["writes"]) for r in got), "changes_in_gold": sum(r["gold_writes"] for r in got),
            "changes_refused_by_env": sum(r["writes_refused_by_env"] for r in got),
            "mean_steps": round(sum(r["steps"] for r in got) / max(len(got), 1), 1),
            "agent_errors": sum("agent_error" in r for r in got)}


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=1))
