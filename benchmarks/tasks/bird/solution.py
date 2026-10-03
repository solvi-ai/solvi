"""BIRD mini-dev with solvi: a question about a database into a SQLite query — and the decision whether to return it
without a person.

solvi does not make the query writer better; it decides which queries can go out alone, and records why.

1. `solvi.core.slow.generate`: the writer (the baseline's model and prompt) proposes three queries — the baseline's request at
   temperature 0 and two samples at 0.8 (`Generator.part(..., k=3)`).
2. `solvi.core.slow.agree`: each query is run read-only; queries that return the same rows are one group, the first query of the
   largest group is chosen, and its share of the three (`agreement`: 1/3, 2/3, 1) is a fact.
3. Hard checks: the chosen query runs, and returns rows that are not all NULL. A failed check says why (`Fail`), and
   `solvi.core.slow.refine` asks the writer once more with those reasons (two rounds at most).
4. The question `correct` ("can this query be returned without a person?") is answered by a head fitted on the 100 dev
   questions over the agreement and the query's shape (`System.fit`), behind `System.guarantee`: a query goes out alone
   only when P(right) clears a threshold set on dev (out of fold) for at most 30% wrong among the answered — an
   empirical target, not a promise; the others are handed to a person.

Every round is a stored decision with the candidates, each execution's digest and the checks; `Refinement.replay`
re-checks it without calling the model. The writer's calls go to an OpenAI-compatible server (the stand's proxy).

    python common/proxy.py --cache cache &                   # the stand's caching endpoint (OPENROUTER_API_KEY)
    python bird/solution.py [--llm http://127.0.0.1:8765/bird/v1]
→ runs/solution.jsonl (the head behind the guarantee) and runs/solution_unanimous.jsonl (the plain rule "return the
query only when all three candidates agree", for comparison).
"""
import argparse
import hashlib
import json
import re
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from baseline import PROMPT, sql_in  # noqa: E402
from common.llm import DEFAULT, read_jsonl, write_jsonl  # noqa: E402
from score import questions, run_sql, schema, score  # noqa: E402

from solvi import Answer, Catalog, Fail, Question, System  # noqa: E402
from solvi.core.slow.agree import agree  # noqa: E402
from solvi.core.slow.generate import generator  # noqa: E402
from solvi.core.slow.refine import refine  # noqa: E402

SHAPE = ["agreement", "n_cols", "has_subquery", "is_aggregate", "computes_ratio"]
FEEDBACK = """

EARLIER ATTEMPTS at this question were checked and rejected. Do not repeat them; fix what the checks found.
{attempts}
Write one corrected SQLite query. Return only the query in a ```sql block."""
RUNS = HERE / "runs"
_lock = threading.Lock()
_schemas, _executions = {}, {}


# ---------------------------------------------------------------- the database side (no solvi)
def execute(db_path, sql):
    """What the checks need of a query's result, never the table: {"ok", "error", "n_rows", "n_cols", "digest" (of the row
    set: equal digests = equal answers), "preview" (5 rows), "all_null"}. Cached on disk by (database, query)."""
    if not sql:
        return {"ok": False, "error": "no query", "n_rows": 0, "n_cols": 0, "digest": None, "preview": [], "all_null": False}
    key = hashlib.sha256(f"{db_path}\n{sql}".encode()).hexdigest()
    with _lock:
        if key in _executions:
            return _executions[key]
    rows, err = run_sql(db_path, sql)
    if err:
        val = {"ok": False, "error": err, "n_rows": 0, "n_cols": 0, "digest": None, "preview": [], "all_null": False}
    else:
        h = hashlib.sha256()
        for r in sorted(repr(x) for x in rows):
            h.update(r.encode() + b"\n")
        some = sorted(rows, key=repr)[:5]
        preview = json.loads(json.dumps([list(r) for r in some], default=str))
        val = {"ok": True, "error": None, "n_rows": len(rows), "n_cols": len(some[0]) if some else 0,
               "digest": h.hexdigest()[:16], "all_null": bool(rows) and all(all(v is None for v in r) for r in rows),
               "preview": [[(v[:80] + "…") if isinstance(v, str) and len(v) > 80 else v for v in r] for r in preview]}
    with _lock:
        if key not in _executions:
            _executions[key] = val
            with open(RUNS / "exec_cache.jsonl", "a") as f:
                f.write(json.dumps({"key": key, "val": val}, ensure_ascii=False) + "\n")
    return val


def db_schema(db_path):
    with _lock:
        if db_path not in _schemas:
            _schemas[db_path] = schema(db_path, 3)
        return _schemas[db_path]


def _short(sql):
    return " ".join((sql or "").split())


# ---------------------------------------------------------------- the solvi side
def build(writer):
    cat = Catalog()

    def prompt(question, evidence, db_path, feedback):
        """The baseline's prompt; the reasons of earlier rounds appended."""
        p = PROMPT.format(schema=db_schema(db_path), question=question, evidence=evidence)
        if feedback:
            p += FEEDBACK.format(attempts="\n".join(f"{i + 1}. {t}" for i, t in enumerate(feedback)))
        return p

    def digest(sql, db_path):
        e = execute(db_path, sql)
        return e["digest"] if e["ok"] else None

    def has_rows(sql, db_path):
        e = execute(db_path, sql)
        return e["n_rows"] > 0 and not e["all_null"]

    cat.fn(writer.part("candidates", prompt, k=3, temperature=0.8, parse=sql_in))
    agree(cat, "sql", "candidates", key=digest, prefer=has_rows, share="agreement")

    @cat.check(hard=True, then={"correct": "no"})
    def query_runs(sql_tally, candidates, db_path) -> bool:
        if sql_tally["index"] >= 0:
            return True
        fails = [f"`{_short(c)}` failed: {execute(db_path, c)['error']}" for c in candidates if c][:2]
        return Fail(*(fails or ["no query was written"]))

    @cat.fn
    def execution(sql, db_path):
        return execute(db_path, sql)

    @cat.check(hard=True, then={"correct": "no"})
    def returns_rows(sql, execution) -> bool:
        e = execution
        if e["n_rows"] > 0 and not e["all_null"]:
            return True
        shown = (f"`{_short(sql)}` returned {e['n_rows']} row(s) of {e['n_cols']} column(s), first rows "
                 f"{json.dumps(e['preview'][:3], ensure_ascii=False)}")
        return Fail(shown + " — an empty or all-NULL result: a filter value, a join or a column is probably wrong; "
                    "compare the literals with the sample rows of the schema.")

    @cat.fn
    def n_cols(execution) -> int:
        return execution["n_cols"]

    @cat.check
    def has_subquery(sql):
        return len(re.findall(r"\bselect\b", (sql or "").lower())) > 1

    @cat.check
    def is_aggregate(sql):
        return bool(re.search(r"\b(count|sum|avg|min|max)\s*\(", (sql or "").lower()))

    @cat.check
    def computes_ratio(sql):
        return "/" in (sql or "")

    q = Question("correct", "Is the chosen query right, so that it can be returned without a person?", Answer.yes_no(),
                 requires=["query_runs", "returns_rows"], uses=SHAPE)
    return System(cat, [q])


def solve(system, q):
    """Up to two rounds (the second with the failed checks' reasons) → the round that counts: the accepted one, else
    the first."""
    run = refine(system, {"question": q["question"], "evidence": q["evidence"], "db_path": q["db_path"]}, "correct",
                 rounds=2, accept="checks", feedback_into="feedback")
    assert run.replay(system)["ok"], f"{q['id']}: the refinement does not replay"
    rd = next((r for r in run.rounds if r.accepted), run.rounds[0])
    v = rd.response.values
    t = v.get("sql_tally") or {"index": -1}
    return {"id": q["id"], "sql": v["candidates"][t["index"]] if t["index"] >= 0 else None, "accepted": rd.accepted,
            "agreement": v.get("agreement"), "init": rd.response.trace.init, "result": rd.response["correct"]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", default="http://127.0.0.1:8765/bird/v1")
    p.add_argument("--model", default=DEFAULT)
    p.add_argument("--max-error", type=float, default=0.30, help="the dev target: wrong among the queries returned alone")
    p.add_argument("--out", default=str(RUNS / "solution.jsonl"))
    a = p.parse_args()
    RUNS.mkdir(exist_ok=True)
    if (RUNS / "exec_cache.jsonl").exists():
        _executions.update((r["key"], r["val"]) for r in read_jsonl(RUNS / "exec_cache.jsonl"))
    writer = generator(a.llm, a.model, api_key="proxy", max_tokens=3000, timeout=300,
                       extra_body={"reasoning": {"effort": "low"}})
    system = build(writer)

    dev = questions("dev_100_ids.json")
    rounds = [solve(system, q) for q in dev]
    right = [r["sql"] is not None and execute(q["db_path"], r["sql"])["digest"] == execute(q["db_path"], q["sql"])["digest"]
             for q, r in zip(dev, rounds)]
    examples = [(r["init"], "yes" if ok else "no") for r, ok in zip(rounds, right)]
    system.fit("correct", examples, features=SHAPE)
    rep = system.guarantee("correct", examples, answer="yes", folds=5, max_error=a.max_error, method="empirical")
    print(f"dev: {sum(right)} of {len(dev)} right; threshold {rep['threshold']:.4f}, answered alone {rep['answered']:.1%}, "
          f"wrong among them {rep['error']:.1%}; {rep['promise']}")

    final, unanimous = [], []
    for q in questions("eval_150_ids.json"):
        r = solve(system, q)
        alone = r["accepted"] and r["result"].status == "ok" and r["result"].answer == "yes"
        final.append({"id": q["id"], "sql": r["sql"] if alone else None, "escalate": not alone,
                      "agreement": r["agreement"], "why": r["result"].why})
        both = r["accepted"] and r["agreement"] == 1
        unanimous.append({"id": q["id"], "sql": r["sql"] if both else None, "escalate": not both})
    out = Path(a.out)
    print(json.dumps({"solution": score(write_jsonl(out, final)),
                      "unanimous": score(write_jsonl(out.with_name(out.stem + "_unanimous.jsonl"), unanimous))},
                     indent=1))


if __name__ == "__main__":
    main()
