"""NATURAL PLAN without solvi: the benchmark's own 5-shot prompt goes to an LLM (medium reasoning), its answer is the plan.

    python naturalplan/baseline.py [--split eval] [--model openai/gpt-oss-120b] [--cache DIR]"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.llm import DEFAULT, chat_many, options, parse, read_jsonl, write_jsonl  # noqa: E402
from score import D, TASKS, score  # noqa: E402

FORMAT = ("Answer with the solution only, written exactly in the format of the SOLUTIONs in the examples: the same sentences, "
          "plain text lines, no tables, no headings, no commentary before or after.")


def main():
    p = options(__doc__)
    p.add_argument("--split", default="eval")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--out", default=None)
    a = parse(p)
    rows_out = []
    for task in TASKS:
        rows = read_jsonl(D / f"prepared/{task}_{a.split}.jsonl")
        prompts = [[{"role": "system", "content": FORMAT}, {"role": "user", "content": r["prompt_5shot"]}] for r in rows]
        answers = chat_many(a.model, prompts, tag=f"naturalplan/baseline/{task}", max_tokens=12000, reasoning="medium",
                            deadline=420, retries=2)
        rows_out += [{"id": r["id"], "text": t} for r, t in zip(rows, answers)]
    out = write_jsonl(a.out or Path(__file__).parent / "runs" / f"baseline_{a.split}.jsonl", rows_out)
    print(json.dumps(score(out, a.split), indent=1))


if __name__ == "__main__":
    main()
