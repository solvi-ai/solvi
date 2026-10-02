"""RAGTruth baseline, no solvi: an LLM judge reads the sources, the writer's task and the response, and says whether
anything in the response is unsupported (with the unsupported parts, verbatim).

    python ragtruth/baseline.py [--split eval] [--cache DIR] [--offline]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, DEFAULT, chat_many, json_in, options, parse, read_jsonl, write_jsonl  # noqa: E402

PROMPT = """You check whether a response is supported by its source material.

SOURCE MATERIAL:
{context}

TASK GIVEN TO THE WRITER:
{query}

RESPONSE:
{output}

Does the response contain any statement that contradicts the source material or that the source material does not support?
Answer with JSON only: {{"hallucinated": true or false, "quotes": [the unsupported parts of the response, verbatim]}}"""


def main():
    p = options(__doc__)
    p.add_argument("--split", default="eval")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--out", default=None)
    a = parse(p)
    rows = read_jsonl(DATA / f"ragtruth/prepared/{a.split}.jsonl")
    answers = chat_many(a.model, [PROMPT.format(**r) for r in rows], tag="ragtruth/baseline", max_tokens=1500)
    out = []
    for r, ans in zip(rows, answers):
        j = json_in(ans) or {}
        out.append({"id": r["id"], "hallucinated": bool(j.get("hallucinated")), "quotes": j.get("quotes", []), "parsed": bool(j)})
    path = write_jsonl(a.out or Path(__file__).parent / "runs" / f"baseline_{a.split}.jsonl", out)
    from score import score
    print(json.dumps(score(path, a.split), indent=1))


if __name__ == "__main__":
    main()
