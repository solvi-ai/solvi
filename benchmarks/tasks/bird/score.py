"""BIRD mini-dev: a question into a SQLite query.   python bird/score.py runs/<run>.jsonl [--ids eval_150_ids.json]
A prediction: {"id", "sql": str or None, "escalate": bool (optional: no query given, handed to a person)}.
Right = the query's rows equal the gold query's rows as sets (the benchmark's execution accuracy)."""
import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402

BIRD = DATA / "bird"
MAX_ROWS, MAX_SECONDS = 200_000, 30


def run_sql(db_path, sql):
    """→ (set of rows, None) or (None, the error). Read-only, time- and row-limited."""
    con = sqlite3.connect(f"file:{BIRD / db_path}?mode=ro", uri=True)
    t0 = time.time()
    con.set_progress_handler(lambda: time.time() - t0 > MAX_SECONDS, 100_000)
    try:
        rows = con.execute(sql).fetchmany(MAX_ROWS + 1)
        if len(rows) > MAX_ROWS:
            return None, f"more than {MAX_ROWS} rows"
        return set(rows), None
    except Exception as e:                                   # noqa: BLE001 - any failure of the query is its error
        return None, str(e)[:200]
    finally:
        con.close()


def schema(db_path, sample_rows=0):
    """The CREATE statements of a database (and, if asked, a few rows of each table)."""
    con = sqlite3.connect(f"file:{BIRD / db_path}?mode=ro", uri=True)
    out = []
    for name, sql in con.execute("select name, sql from sqlite_master where type='table' and name not like 'sqlite_%'"):
        out.append(sql.strip() + ";")
        if sample_rows:
            cur = con.execute(f'select * from "{name}" limit {int(sample_rows)}')
            out.append("/* " + ", ".join(d[0] for d in cur.description) + "\n" + "\n".join(str(r)[:300] for r in cur.fetchall()) + " */")
    con.close()
    return "\n".join(out)


def questions(ids="eval_150_ids.json"):
    want = json.loads((BIRD / "prepared" / ids).read_text())
    by = {q["id"]: q for q in read_jsonl(BIRD / "prepared/questions.jsonl")}
    return [by[i] for i in want]


def score(pred_file, ids="eval_150_ids.json"):
    qs = questions(ids)
    pred = {r["id"]: r for r in read_jsonl(pred_file)}
    res = {"n": len(qs), "right": 0, "wrong_result": 0, "sql_error": 0, "no_query": 0, "by_difficulty": {}}
    for q in qs:
        p = pred.get(q["id"]) or {}
        gold, _ = run_sql(q["db_path"], q["sql"])
        if not p.get("sql") or p.get("escalate"):
            kind = "no_query"
        else:
            got, err = run_sql(q["db_path"], p["sql"])
            kind = "sql_error" if err else "right" if got == gold else "wrong_result"
        res[kind] += 1
        d = res["by_difficulty"].setdefault(q["difficulty"], [0, 0])
        d[0] += kind == "right"
        d[1] += 1
    answered = res["n"] - res["no_query"]
    res["accuracy"] = round(res["right"] / len(qs), 3)
    res["answered"] = round(answered / len(qs), 3)
    res["wrong_among_answered"] = round((res["wrong_result"] + res["sql_error"]) / max(answered, 1), 3)
    res["by_difficulty"] = {k: f"{a} of {b}" for k, (a, b) in sorted(res["by_difficulty"].items())}
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("pred")
    p.add_argument("--ids", default="eval_150_ids.json")
    a = p.parse_args()
    print(json.dumps(score(a.pred, a.ids), indent=1))
