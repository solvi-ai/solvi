"""CUAD: is a kind of clause in the contract, and where.

    python cuad/score.py cuad/runs/<run>.jsonl [split=eval]

A prediction per (contract, kind of clause): {"id", "present": bool, "quotes": [text copied from the contract],
"escalate": bool (optional)}. "found": a quote overlaps a gold span (word Jaccard >= 0.5, or one contains the other).
"verbatim": the quote is in the contract's text. An item is right when the clause is absent and none is claimed, or
present and found. Accuracy, found and false claims count every item, escalated or not."""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402

D = DATA / "cuad/prepared"
norm = lambda t: re.sub(r"\s+", " ", t).strip().lower()
words = lambda t: set(re.findall(r"\w+", t.lower()))


def overlaps(quote, gold):
    a, b = words(quote), words(gold)
    if not a or not b:
        return False
    return len(a & b) / len(a | b) >= 0.5 or norm(quote) in norm(gold) or norm(gold) in norm(quote)


def score(pred_file, split="eval"):
    gold = read_jsonl(D / f"{split}.jsonl")
    pred = {r["id"]: r for r in read_jsonl(pred_file)}
    texts = {}
    c = dict.fromkeys(("right", "tp", "fp", "fn", "found", "quotes", "verbatim", "missing", "escalated", "wrong_alone"), 0)
    for g in gold:
        p = pred.get(g["id"])
        c["missing"] += p is None
        p = p or {}
        present, has = bool(p.get("present")) and bool(p.get("quotes")), not g["is_impossible"]
        text = texts.setdefault(g["contract"], norm((D / "contracts" / f"{g['contract']}.txt").read_text(encoding="utf-8")))
        qs = [q for q in p.get("quotes") or [] if isinstance(q, str) and q.strip()] if present else []
        c["quotes"] += len(qs)
        c["verbatim"] += sum(norm(q) in text for q in qs)
        found = has and any(overlaps(q, a["text"]) for q in qs for a in g["answers"])
        ok = found if has else not present
        c["right"] += ok
        c["found"] += found
        c["tp"] += present and has
        c["fp"] += present and not has
        c["fn"] += has and not present
        if p.get("escalate"):
            c["escalated"] += 1
        else:
            c["wrong_alone"] += not ok
    n = len(gold)
    pr, rc = c["tp"] / max(c["tp"] + c["fp"], 1), c["tp"] / max(c["tp"] + c["fn"], 1)
    has_n = c["tp"] + c["fn"]
    return {"n": n, "accuracy": round(c["right"] / n, 3), "presence_precision": round(pr, 3), "presence_recall": round(rc, 3),
            "presence_f1": round(2 * pr * rc / max(pr + rc, 1e-9), 3), "found_of_present": f"{c['found']} of {has_n}",
            "false_claims_where_absent": f"{c['fp']} of {n - has_n}", "quotes_verbatim": f"{c['verbatim']} of {c['quotes']}",
            "missing": c["missing"], "escalated": c["escalated"],
            "wrong_among_answered_alone": round(c["wrong_alone"] / max(n - c["escalated"], 1), 3)}


if __name__ == "__main__":
    print(json.dumps(score(*sys.argv[1:]), indent=1))
