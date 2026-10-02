"""BIRD baseline, no solvi: the schema (with three sample rows per table) and the question go to an LLM; the query it
writes is the answer, one try.

    python bird/baseline.py [--ids eval_150_ids.json] [--cache DIR] [--offline]"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DEFAULT, chat_many, options, parse, write_jsonl  # noqa: E402
from score import questions, schema, score  # noqa: E402

PROMPT = """You write SQLite queries.

DATABASE SCHEMA (with three sample rows per table):
{schema}

QUESTION: {question}
HINT: {evidence}

Write one SQLite query that answers the question. Return only the query in a ```sql block."""


def sql_in(text):
    """The query in a model's answer: the first ```sql block, else the whole text; None when empty."""
    m = re.search(r"```(?:sql|sqlite)?\s*(.*?)```", text or "", re.S | re.I)
    s = (m.group(1) if m else (text or "")).strip()
    return s or None


def main():
    p = options(__doc__)
    p.add_argument("--ids", default="eval_150_ids.json")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--out", default=str(Path(__file__).parent / "runs" / "baseline.jsonl"))
    a = parse(p)
    qs, sch = questions(a.ids), {}
    prompts = [PROMPT.format(schema=sch.setdefault(q["db_path"], schema(q["db_path"], 3)), question=q["question"],
                             evidence=q["evidence"]) for q in qs]
    answers = chat_many(a.model, prompts, tag="bird/baseline", max_tokens=3000)
    out = write_jsonl(a.out, [{"id": q["id"], "sql": sql_in(t)} for q, t in zip(qs, answers)])
    print(json.dumps(score(out, a.ids), indent=1))


if __name__ == "__main__":
    main()
