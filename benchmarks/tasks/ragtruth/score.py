"""RAGTruth: does a response contain something its sources do not support?

    python ragtruth/score.py ragtruth/runs/<run>.jsonl [split=eval]

A prediction per response: {"id", "hallucinated": bool, "escalate": bool (optional: handed to a person)}. Precision,
recall, F1 and accuracy over all responses and per kind of task (QA, Summary, Data2txt); a missing prediction counts as
"clean". When some responses were escalated, "answered_alone" gives the same numbers on the rest, with their share."""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402


def prf(rows):
    """[(gold, said)] → precision, recall, F1, accuracy of "hallucinated"."""
    tp = sum(g and p for g, p in rows)
    fp = sum(p and not g for g, p in rows)
    fn = sum(g and not p for g, p in rows)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"n": len(rows), "precision": round(pr, 3), "recall": round(rc, 3), "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3),
            "accuracy": round(sum(g == p for g, p in rows) / max(len(rows), 1), 3)}


def score(pred_file, split="eval"):
    gold = {r["id"]: r for r in read_jsonl(DATA / f"ragtruth/prepared/{split}.jsonl")}
    pred = {str(r["id"]): r for r in read_jsonl(pred_file)}
    groups, answered = defaultdict(list), defaultdict(list)
    for i, g in gold.items():
        p = pred.get(i)
        said = bool(p and p.get("hallucinated"))
        for k in ("all", g["task_type"]):
            groups[k].append((g["hallucinated"], said))
            if p and not p.get("escalate"):
                answered[k].append((g["hallucinated"], said))
    out = {k: prf(v) for k, v in groups.items()}
    out["missing"] = sum(i not in pred for i in gold)
    if len(answered["all"]) != len(groups["all"]):
        out["answered_alone"] = {k: {**prf(v), "share": round(len(v) / len(groups[k]), 3)} for k, v in answered.items()}
    return out


if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:]), indent=1))
