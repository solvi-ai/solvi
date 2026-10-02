"""Credit: are the decisions the policy's, and what do they cost.

    python credit/score.py runs/<arm>.jsonl

A prediction: {"id", "version": 1 or 2, "decision": "approve" | "refuse" | "review"} for every application and version."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DATA, read_jsonl  # noqa: E402
from reference import decide  # noqa: E402

D = DATA / "credit/prepared"


def score(pred_file):
    apps = {f: read_jsonl(D / f"applications_{f}.jsonl") for f in ("history", "new")}
    pred = {(r["id"], int(r["version"])): r["decision"] for r in read_jsonl(pred_file)}
    out = {}
    for v in (1, 2):
        allapps = apps["history"] + apps["new"]
        same = sum(pred.get((a["id"], v)) == decide(a, v)[0] for a in allapps)
        new = [(a, pred.get((a["id"], v))) for a in apps["new"]]
        out[f"v{v}"] = {"equal_to_reference": f"{same} of {len(allapps)}",
                        "cost_new": 5 * sum(d == "approve" and a["outcome"] == "bad" for a, d in new)
                        + sum(d == "refuse" and a["outcome"] == "good" for a, d in new),
                        "approved": sum(d == "approve" for _, d in new), "refused": sum(d == "refuse" for _, d in new),
                        "review": sum(d == "review" for _, d in new)}
    bad = sum(a["outcome"] == "bad" for a in apps["new"])
    out["cost_approve_all"], out["cost_refuse_all"] = 5 * bad, len(apps["new"]) - bad
    out["history_changed_v1_to_v2"] = sum(pred.get((a["id"], 1)) != pred.get((a["id"], 2)) for a in apps["history"])
    return out


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=1))
