"""Abt-Buy with solvi: is a pair of offers from two shops the same product?

The offers are read by code into typed facts (pairfacts.py: brand, model number, other codes, sizes, colours, price,
and how A's and B's compare), and solvi does the rest:

1. `System.fit(..., select=False)` fits an answer head over every fact on the labelled dev-train pairs (a closed-form
   ridge head, seconds); every answer then comes with the head's probability and a trace of the facts it read.
2. `System.guarantee(max_risk=0.01)` calibrates a threshold on the dev-valid pairs: a pair is answered alone only when
   the head is sure enough that P(answered alone and wrong) ≤ 1% of all pairs, for pairs like these; the others are
   handed to a person (still with the head's answer).
3. `solvi.sets.decide_set` makes the answers consistent: an offer has at most one counterpart in the other catalog, so
   the most probable combination of the pairs' answers with at most one match per offer is kept.

No model reads the text and no LLM is called. Everything is fitted and calibrated on dev; eval is only scored.

    python abtbuy/solution.py                             # runs/solution.jsonl (+ runs/solution_head.jsonl: no step 3)
    python abtbuy/solution.py --repair runs/baseline.jsonl  # step 3 on another solver's answers (no probabilities:
                                                          # its matches are ranked by the head's probability)
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.llm import read_jsonl, write_jsonl  # noqa: E402
from pairfacts import PREPARED, Idf, build, load, state  # noqa: E402
from score import score  # noqa: E402

from solvi import System  # noqa: E402
from solvi.sets import AtMostOne, Item, decide_set  # noqa: E402

ABT, BUY = load()
ONE_COUNTERPART = [AtMostOne("abt"), AtMostOne("buy")]


def examples(pairs):
    return [(state(ABT[p["abt"]], BUY[p["buy"]]), "yes" if p["match"] else "no") for p in pairs]


def one_counterpart(rows):
    """[{"abt", "buy", "p": P(match), ...}] → the same rows with "match" decided under at most one match per offer."""
    items = [Item((r["abt"], r["buy"]), {"yes": r["p"], "no": 1 - r["p"]}, keys={"abt": r["abt"], "buy": r["buy"]})
             for r in rows]
    out = decide_set(items, ONE_COUNTERPART)
    print(f"one counterpart: {len(out.changed)} answers changed, feasible {out.feasible}, exact {out.exact}")
    return [{**r, "match": out[(r["abt"], r["buy"])].answer == "yes"} for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-risk", type=float, default=0.01, help="the promise: P(answered alone and wrong) of all pairs")
    p.add_argument("--repair", help="an answer file of another solver to make one-to-one with the head's probabilities")
    p.add_argument("--out", default=str(HERE / "runs" / "solution.jsonl"))
    a = p.parse_args()

    cat, question = build(Idf(list(ABT.values()) + list(BUY.values())))
    system = System(cat, [question])
    train, valid = (read_jsonl(PREPARED / f"pairs_dev_{n}.jsonl") for n in ("train", "valid"))
    head = system.fit("match", examples(train), select=False)
    print(f"head over {len(head.features)} facts, fitted on {len(train)} dev-train pairs")
    rep = system.guarantee("match", examples(valid), max_risk=a.max_risk)
    print(f"guarantee on {rep['n']} dev-valid pairs: threshold {rep['threshold']:.4f}, answered alone "
          f"{rep['answered']:.1%}; {rep['promise']}")

    rows = []
    for x in read_jsonl(PREPARED / "pairs_eval_test.jsonl"):
        r = system.ask(state(ABT[x["abt"]], BUY[x["buy"]]), store=False)["match"]
        py = float(r.probs["yes"])
        rows.append({"abt": x["abt"], "buy": x["buy"], "p": round(py, 6), "match": py >= 0.5, "escalate": r.status != "ok"})
    out = Path(a.out)
    report = {"head": score(write_jsonl(out.with_name(out.stem + "_head.jsonl"), rows)),
              "head + one counterpart": score(write_jsonl(out, one_counterpart(rows)))}
    if a.repair:
        head_p = {(r["abt"], r["buy"]): r["p"] for r in rows}
        theirs = read_jsonl(a.repair)
        ranked = [{**r, "p": 0.5 + head_p[(r["abt"], r["buy"])] / 2 if r["match"] else 0.0} for r in theirs]
        fixed = Path(a.repair).with_name(Path(a.repair).stem + "_one.jsonl")
        report[f"{Path(a.repair).name} as given"] = score(a.repair)
        report[f"{Path(a.repair).name} + one counterpart"] = score(write_jsonl(fixed, one_counterpart(ranked)))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
