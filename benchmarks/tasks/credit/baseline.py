"""Credit baseline, no solvi: the policy as plain functions (reference.py), a CSV log of the decisions, replay and diff
written by hand. Prints what plain code gives for the eight checks of POLICY.md, and what it does not.

    python credit/baseline.py [--out runs/baseline.jsonl]"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import read_jsonl, write_jsonl  # noqa: E402
from reference import decide  # noqa: E402
from score import D, score  # noqa: E402


def main(out):
    hist, new = read_jsonl(D / "applications_history.jsonl"), read_jsonl(D / "applications_new.jsonl")
    write_jsonl(out, [{"id": a["id"], "version": v, "decision": decide(a, v)[0]} for a in hist + new for v in (1, 2)])
    log = Path(out).with_suffix(".log.csv")                                  # 2. the record
    with open(log, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "version", "input", "points", "total", "hard", "decision"])
        for a in hist:
            d, total, p, hard = decide(a, 1)
            w.writerow([a["id"], 1, json.dumps({k: v for k, v in a.items() if k != "outcome"}), json.dumps(p), total, hard or "", d])
    with open(log) as f:
        rows = list(csv.DictReader(f))
    replayed = sum(decide(json.loads(r["input"]), int(r["version"]))[0] == r["decision"] for r in rows)       # 3. replay
    causes = {}                                                              # 4. diff, with the rules whose points moved
    for r in rows:
        a = json.loads(r["input"])
        d2, _, p2, _ = decide(a, 2)
        p1 = json.loads(r["points"])
        if d2 != r["decision"]:
            k = "+".join(sorted(k for k in set(p1) | set(p2) if p1.get(k, 0) != p2.get(k, 0))) or "threshold"
            causes[k] = causes.get(k, 0) + 1
    rows[7]["input"] = rows[7]["input"].replace('"credit_amount_dm": ', '"credit_amount_dm": 1')              # 5. tamper
    seen = decide(json.loads(rows[7]["input"]), 1)[0] != rows[7]["decision"]
    checks = {
        "1 correctness": score(out),
        "2 record": "a CSV row per decision with the version, the input and the points",
        "3 replay": f"{replayed} of {len(rows)}",
        "4 diff": {"changed": sum(causes.values()), "by_cause": causes},
        "5 tamper": ("noticed only because the decision flipped" if seen else "not detected") + ": a CSV has no integrity check",
        "6 moved rules": "fail: a replay under the new functions gives another decision; nothing tells a rule change from damage",
        "7 sensitive field": "fail: nothing inspects what a rule reads",
    }
    print(json.dumps(checks, indent=1))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default=str(Path(__file__).parent / "runs" / "baseline.jsonl"))
    main(p.parse_args().out)
