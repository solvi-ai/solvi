"""CUAD baseline, no solvi: an LLM reads the whole contract and returns, for eight kinds of clause per request, the
passages a lawyer should review (an empty list when the contract has none).

    python cuad/baseline.py [--split eval] [--cache DIR] [--offline]
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, DEFAULT, chat_many, json_in, options, parse, read_jsonl, write_jsonl  # noqa: E402

D = DATA / "cuad/prepared"
PROMPT = """CONTRACT:
{contract}

END OF CONTRACT.

For each kind of clause below, copy from the contract the passages that a lawyer should review for it — word for word, each
passage as short as it can be while still containing the clause. If the contract has no such clause, give an empty list.

{kinds}

Answer with JSON only: {{"<kind>": ["<passage>", ...], ...}} with every kind above as a key."""


def main():
    p = options(__doc__)
    p.add_argument("--split", default="eval")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--out", default=None)
    a = parse(p)
    rows = read_jsonl(D / f"{a.split}.jsonl")
    jobs = []
    for c in sorted({r["contract"] for r in rows}):
        text = (D / "contracts" / f"{c}.txt").read_text(encoding="utf-8")
        items = [r for r in rows if r["contract"] == c]
        for i in range(0, len(items), 8):
            part = items[i:i + 8]
            kinds = "\n".join(f'- "{r["category"]}": {r["question"].split("Details:")[-1].strip()}' for r in part)
            jobs.append((part, PROMPT.format(contract=text, kinds=kinds)))
    answers = chat_many(a.model, [q for _, q in jobs], tag="cuad/baseline", max_tokens=6000, workers=6)
    out = []
    for (part, _), ans in zip(jobs, answers):
        j = json_in(ans)
        j = j if isinstance(j, dict) else {}
        for r in part:
            q = j.get(r["category"]) or []
            q = [x for x in (q if isinstance(q, list) else [q]) if isinstance(x, str) and x.strip()]
            out.append({"id": r["id"], "present": bool(q), "quotes": q, "parsed": r["category"] in j})
    path = write_jsonl(a.out or Path(__file__).parent / "runs" / f"baseline_{a.split}.jsonl", out)
    from score import score
    print(json.dumps(score(path, a.split), indent=1))


if __name__ == "__main__":
    main()
