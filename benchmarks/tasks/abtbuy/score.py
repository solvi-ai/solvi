"""Abt-Buy: is a pair of offers the same product?   python abtbuy/score.py runs/<run>.jsonl
A prediction per eval pair: {"abt", "buy", "match": bool, "escalate": bool (optional: handed to a person)}.
"conflicts": offers predicted to match more than one offer of the other catalog (a product has one counterpart).
With escalations: the error among the pairs answered alone, and the same split by the predicted answer."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402


def score(pred_file):
    gold = {(r["abt"], r["buy"]): r["match"] for r in read_jsonl(DATA / "abtbuy/prepared/pairs_eval_test.jsonl")}
    pred = {(r["abt"], r["buy"]): r for r in read_jsonl(pred_file)}
    said = {k: bool(pred.get(k, {}).get("match")) for k in gold}
    tp = sum(gold[k] and said[k] for k in gold)
    fp = sum(said[k] and not gold[k] for k in gold)
    fn = sum(gold[k] and not said[k] for k in gold)
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    a, b = Counter(k[0] for k in gold if said[k]), Counter(k[1] for k in gold if said[k])
    out = {"pairs": len(gold), "matches": sum(gold.values()), "precision": round(pr, 3), "recall": round(rc, 3),
           "f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3), "missing": sum(k not in pred for k in gold),
           "conflicts": sum(v > 1 for v in a.values()) + sum(v > 1 for v in b.values())}
    esc = {k for k in gold if pred.get(k, {}).get("escalate")}
    if esc:
        alone = [k for k in gold if k not in esc]
        yes = [k for k in alone if said[k]]
        out.update(escalated=len(esc), answered_alone=round(len(alone) / len(gold), 4),
                   errors_answered_alone=round(sum(gold[k] != said[k] for k in alone) / max(len(alone), 1), 4),
                   matches_answered_alone=len(yes),
                   wrong_among_matches_answered_alone=round(sum(not gold[k] for k in yes) / max(len(yes), 1), 4))
    return out


if __name__ == "__main__":
    print(json.dumps(score(sys.argv[1]), indent=1))
