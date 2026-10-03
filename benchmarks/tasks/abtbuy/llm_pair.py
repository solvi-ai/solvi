"""Abt-Buy, the comparison arm: the baseline's LLM asked about each raw pair through `solvi.core.deciders.llm` (a typed yes/no
decision with a confidence) instead of a plain call, and the same "one counterpart" step (`solvi.core.sets.decide_set`) on its
answers. Not the solution (solution.py is): it shows what solvi's LLM decider gives on a task where code reads better.

With reasoning asked for, solvi.core.deciders.llm sends the reply contract in the prompt and no response_format; --response-format
json_schema has the server enforce the format instead (the 0.7 default). A reply cut off at max_tokens escalates with no
value and counts as "no" here. The runs measured in the README used max_tokens=400; solvi.core.deciders.llm's default with reasoning is
2,048, which leaves room for the thinking.

    python common/proxy.py --cache cache &                  # the stand's caching endpoint (OPENROUTER_API_KEY)
    python abtbuy/llm_pair.py [--llm http://127.0.0.1:8765/abtbuy/v1] [--max-tokens 400]
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common.llm import DEFAULT, read_jsonl, write_jsonl  # noqa: E402
from pairfacts import PREPARED, load  # noqa: E402
from score import score  # noqa: E402
from solution import one_counterpart  # noqa: E402

from solvi.core.deciders.llm import llm  # noqa: E402

TASK = "Are offer A and offer B the same product (the same model, not just the same kind)?"
ABT, BUY = load()


def offer(r):
    return {k: v for k, v in (("name", r["name"]), ("description", r["description"][:400]), ("price", r["price"])) if v}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", default="http://127.0.0.1:8765/abtbuy/v1")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--max-tokens", type=int, default=400)
    p.add_argument("--response-format", default="auto", help='"auto" (with reasoning: the contract in the prompt only) '
                   'or "json_schema" (the server enforces the reply format, and some providers then skip the thinking)')
    p.add_argument("--out", default=str(HERE / "runs" / "llm_pair.jsonl"))
    a = p.parse_args()
    model = llm(a.llm, a.model, api_key="proxy", workers=8, extra_body={"reasoning": {"effort": "low"}}, logprobs=False,
                ask="confidence", max_tokens=a.max_tokens, response_format=a.response_format)
    part = model.decision("match", TASK, "pair", bool)
    pairs = read_jsonl(PREPARED / "pairs_eval_test.jsonl")
    ds = part.decide([{"offer_a": offer(ABT[x["abt"]]), "offer_b": offer(BUY[x["buy"]])} for x in pairs])
    print("escalated:", dict(Counter(str(d.escalate)[:70] for d in ds if d.escalate)))
    rows = []
    for x, d in zip(pairs, ds):
        py = float(d.probs.get("yes", 0.0)) if d.value is not None and d.probs else 0.0
        rows.append({"abt": x["abt"], "buy": x["buy"], "p": round(py, 6), "match": d.value is True})
    out = Path(a.out)
    print(json.dumps({"llm": score(write_jsonl(out, rows)),
                      "llm + one counterpart": score(write_jsonl(out.with_name(out.stem + "_one.jsonl"),
                                                                 one_counterpart(rows)))}, indent=1))


if __name__ == "__main__":
    main()
