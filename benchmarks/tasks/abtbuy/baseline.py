"""Abt-Buy baseline, no solvi: an LLM is asked about each pair on its own.

    python abtbuy/baseline.py [--cache DIR] [--offline]      # needs OPENROUTER_API_KEY unless every answer is cached"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, DEFAULT, chat_many, json_in, options, parse, read_jsonl, write_jsonl  # noqa: E402
from score import score  # noqa: E402

PROMPT = """Two product offers from two online shops. Are they the same product (the same model, not just the same kind)?

Offer A: {a}
Offer B: {b}

Answer with JSON only: {{"match": true or false}}"""


def text(r):
    return " | ".join(x for x in (r["name"], r["description"], f"price {r['price']}" if r["price"] else "") if x)[:1500]


def main():
    p = options(__doc__)
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--out", default=str(Path(__file__).parent / "runs" / "baseline.jsonl"))
    a = parse(p)
    d = DATA / "abtbuy/prepared"
    abt, buy = ({r["id"]: r for r in read_jsonl(d / f"{f}.jsonl")} for f in ("abt", "buy"))
    pairs = read_jsonl(d / "pairs_eval_test.jsonl")
    answers = chat_many(a.model, [PROMPT.format(a=text(abt[x["abt"]]), b=text(buy[x["buy"]])) for x in pairs],
                        tag="abtbuy/baseline", max_tokens=800)
    rows = []
    for x, ans in zip(pairs, answers):
        j = json_in(ans) or {}
        rows.append({"abt": x["abt"], "buy": x["buy"], "match": bool(j.get("match")), "parsed": bool(j)})
    print(json.dumps(score(write_jsonl(a.out, rows)), indent=1))


if __name__ == "__main__":
    main()
