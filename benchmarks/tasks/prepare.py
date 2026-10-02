"""Builds the fixed dev / eval files of the nine tasks from the downloads of fetch.sh in $STAND_DATA/<name>/.

    uv run --no-project --with pyarrow python benchmarks/tasks/prepare.py [name ...]

Every sample is drawn with SEED, so a rerun gives the same files (data/manifest.json lists their counts and sha256, to
compare with the table in the README). `dev` is for fitting and calibration, `eval` is held out: a solution reads eval
only to be scored. Output: $STAND_DATA/<name>/prepared/ and $STAND_DATA/manifest.json."""
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

SEED = 20261002
D = Path(os.environ.get("STAND_DATA") or Path(__file__).resolve().parent / "data")
MANIFEST = {}


def out(name, file, rows):
    p = D / name / "prepared" / file
    p.parent.mkdir(parents=True, exist_ok=True)
    if file.endswith(".jsonl"):
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    else:
        p.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    MANIFEST.setdefault(name, {})[file] = {"rows": len(rows), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()[:16]}


def stratified(rows, key, n, rng):
    """n rows, as equal a number per stratum as the strata allow; strata and rows in a fixed order."""
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    for g in groups.values():
        rng.shuffle(g)
    picked, names = [], sorted(groups, key=str)
    while len(picked) < n and any(groups[k] for k in names):
        for k in names:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return picked


# 1. a tool agent under a company policy
def taubench():
    env = D / "taubench/repo/tau_bench/envs/retail"

    class Obj(dict):
        def __init__(self, **kw):
            super().__init__(kw)

    for split in ("test", "dev", "train"):
        ns = {"Task": Obj, "Action": Obj}
        exec((env / f"tasks_{split}.py").read_text().replace("from tau_bench.types import Task, Action", ""), ns)
        tasks = next(v for k, v in ns.items() if k.startswith("TASKS") and isinstance(v, list))
        rows = [{"id": f"retail_{split}_{i}", "user_id": t["user_id"], "instruction": t["instruction"],
                 "actions": [dict(a) for a in t["actions"]], "outputs": t.get("outputs", [])} for i, t in enumerate(tasks)]
        out("taubench", f"tasks_{split}.jsonl", rows)
        if split == "test":
            rng = random.Random(SEED)
            writes = lambda r: sum(a["name"] not in READS for a in r["actions"])          # noqa: E731
            pick = stratified(rows, lambda r: min(writes(r), 3), 30, rng)
            out("taubench", "eval_30_ids.json", sorted(r["id"] for r in pick))
    out("taubench", "policy.json", {"wiki": (env / "wiki.md").read_text(), "tools": sorted(p.stem for p in (env / "tools").glob("*.py") if p.stem != "__init__")})


READS = {"find_user_id_by_name_zip", "find_user_id_by_email", "get_order_details", "get_product_details",
         "get_user_details", "list_all_product_types", "calculate", "think"}


# 2. fields of long contracts, with the quote
def cuad():
    held_out = {c["title"] for c in json.loads((D / "cuad/test.json").read_text())["data"]}

    def rows_of(file, n, rng):
        data = json.loads((D / "cuad" / file).read_text())["data"]
        if file != "test.json":                                   # dev: the full set minus the test contracts
            data = [c for c in data if c["title"] not in held_out]
        data = rng.sample(sorted(data, key=lambda c: c["title"]), n)
        rows, docs = [], {}
        for c in data:
            (p,) = c["paragraphs"]
            docs[c["title"]] = p["context"]
            for q in p["qas"]:
                cat = q["id"].split("__")[-1]
                rows.append({"id": q["id"], "contract": c["title"], "category": cat, "question": q["question"],
                             "is_impossible": bool(q.get("is_impossible", not q["answers"])),
                             "answers": [{"text": a["text"], "start": a["answer_start"]} for a in q["answers"]]})
        return rows, docs

    rng = random.Random(SEED)
    for split, file in (("eval", "test.json"), ("dev", "CUADv1.json")):
        rows, docs = rows_of(file, 25, rng)
        out("cuad", f"{split}.jsonl", rows)
        for title, text in docs.items():
            p = D / "cuad/prepared/contracts" / f"{title}.txt"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        MANIFEST["cuad"][f"{split}_contracts"] = {"rows": len(docs), "chars_median": sorted(map(len, docs.values()))[len(docs) // 2],
                                                 "chars_max": max(map(len, docs.values()))}


# 3. is a RAG answer grounded in its sources
def ragtruth():
    import pyarrow.parquet as pq
    rng = random.Random(SEED)
    for split, file in (("eval", "test"), ("dev", "train")):
        rows = [r for r in pq.read_table(D / f"ragtruth/{file}.parquet").to_pylist() if r["quality"] == "good"]
        for r in rows:
            spans = json.loads(r["hallucination_labels"]) if isinstance(r["hallucination_labels"], str) else r["hallucination_labels"]
            r["spans"] = [{"start": s["start"], "end": s["end"], "text": s["text"], "type": s.get("label_type")} for s in spans]
            r["hallucinated"] = bool(spans)
        pick = stratified(rows, lambda r: (r["task_type"], r["hallucinated"]), 600, rng)
        out("ragtruth", f"{split}.jsonl", [{"id": str(r["id"]), "task_type": r["task_type"], "model": r["model"], "query": r["query"],
                                           "context": r["context"], "output": r["output"], "hallucinated": r["hallucinated"],
                                           "spans": r["spans"]} for r in pick])


# 4. a stream of requests under a promised error rate, with a shift
def banking77():
    src = D / "banking77/repo/banking_data"
    train = list(csv.DictReader(open(src / "train.csv", encoding="utf-8")))
    test = list(csv.DictReader(open(src / "test.csv", encoding="utf-8")))
    cats = sorted({r["category"] for r in train})
    rng = random.Random(SEED)
    unseen = set(rng.sample(cats, 20))                       # intents that appear only after the shift
    known = [c for c in cats if c not in unseen]
    row = lambda r, i, pre: {"id": f"{pre}{i}", "text": r["text"], "intent": r["category"]}          # noqa: E731
    tr = [row(r, i, "tr") for i, r in enumerate(train) if r["category"] in known]
    rng.shuffle(tr)
    cut = len(tr) * 4 // 5
    out("banking77", "fit.jsonl", tr[:cut])                  # known intents only
    out("banking77", "calib.jsonl", tr[cut:])
    te = [row(r, i, "te") for i, r in enumerate(test)]
    rng.shuffle(te)
    before = [r for r in te if r["intent"] in known][:1000]
    used = {r["id"] for r in before}
    after = rng.sample([r for r in te if r["id"] not in used], 1000)   # what is left of the test set, evenly mixed
    stream = [{**r, "phase": "before", "known": True} for r in before] + \
             [{**r, "phase": "after", "known": r["intent"] in known} for r in after]
    out("banking77", "stream.jsonl", [{"n": i, **r} for i, r in enumerate(stream)])
    out("banking77", "test_all.jsonl", [row(r, i, "te") for i, r in enumerate(test)])
    out("banking77", "intents.json", {"known": known, "unseen": sorted(unseen)})


# 5. decisions by rules, audited when the rules change
GERMAN = [
    ("checking_account", {"A11": "below 0 DM", "A12": "0 to 200 DM", "A13": "200 DM or more, or salary assigned for a year", "A14": "no checking account"}),
    ("duration_months", int),
    ("credit_history", {"A30": "no credits taken, or all paid back duly", "A31": "all credits at this bank paid back duly",
                        "A32": "existing credits paid back duly till now", "A33": "delay in paying off in the past",
                        "A34": "critical account, or other credits elsewhere"}),
    ("purpose", {"A40": "car (new)", "A41": "car (used)", "A42": "furniture/equipment", "A43": "radio/television",
                 "A44": "domestic appliances", "A45": "repairs", "A46": "education", "A47": "vacation", "A48": "retraining",
                 "A49": "business", "A410": "others"}),
    ("credit_amount_dm", int),
    ("savings", {"A61": "below 100 DM", "A62": "100 to 500 DM", "A63": "500 to 1000 DM", "A64": "1000 DM or more",
                 "A65": "unknown or no savings account"}),
    ("employed_since", {"A71": "unemployed", "A72": "less than 1 year", "A73": "1 to 4 years", "A74": "4 to 7 years", "A75": "7 years or more"}),
    ("installment_rate_pct_of_income", int),
    ("personal_status_and_sex", {"A91": "male, divorced/separated", "A92": "female, divorced/separated/married", "A93": "male, single",
                                 "A94": "male, married/widowed", "A95": "female, single"}),
    ("other_debtors", {"A101": "none", "A102": "co-applicant", "A103": "guarantor"}),
    ("residence_since_years", int),
    ("property", {"A121": "real estate", "A122": "building society savings or life insurance", "A123": "car or other", "A124": "unknown or none"}),
    ("age", int),
    ("other_installment_plans", {"A141": "bank", "A142": "stores", "A143": "none"}),
    ("housing", {"A151": "rent", "A152": "own", "A153": "for free"}),
    ("existing_credits_at_bank", int),
    ("job", {"A171": "unemployed or unskilled non-resident", "A172": "unskilled resident", "A173": "skilled employee or official",
             "A174": "management, self-employed or highly qualified"}),
    ("people_liable_for", int),
    ("telephone", {"A191": "none", "A192": "yes, registered"}),
    ("foreign_worker", {"A201": "yes", "A202": "no"}),
]


def credit():
    rows = []
    for i, line in enumerate((D / "credit/german/german.data").read_text().splitlines()):
        v = line.split()
        r = {"id": f"app{i:04d}"}
        for (name, kind), x in zip(GERMAN, v):
            r[name] = int(x) if kind is int else kind[x]
        r["outcome"] = {"1": "good", "2": "bad"}[v[20]]
        rows.append(r)
    random.Random(SEED).shuffle(rows)
    out("credit", "applications_history.jsonl", rows[:600])       # decided under the old rules; outcomes known
    out("credit", "applications_new.jsonl", rows[600:])           # the stream the changed rules meet
    out("credit", "about.json", {"cost": {"approve_bad": 5, "refuse_good": 1},
                                 "sensitive": ["personal_status_and_sex", "foreign_worker", "age"],
                                 "note": "outcome is known only for the applications the original bank approved"})


# 6. a question into a SQL query
def bird():
    import sqlite3
    root = next((D / "bird").rglob("mini_dev_sqlite.json"), None)
    if root is None:
        print("bird: run fetch.sh first (minidev.zip unpacked)")
        return
    dbs = {p.stem: p for p in (D / "bird").rglob("*.sqlite")}
    rows, seen = [], set()
    for q in json.loads(root.read_text()):
        if q["question_id"] in seen:                         # the source repeats two questions word for word
            continue
        seen.add(q["question_id"])
        r = {"id": f"bird{q['question_id']}", "db": q["db_id"], "question": q["question"], "evidence": q.get("evidence", ""),
             "sql": q["SQL"], "difficulty": q["difficulty"], "db_path": str(dbs[q["db_id"]].relative_to(D / "bird"))}
        con = sqlite3.connect(f"file:{dbs[q['db_id']]}?mode=ro", uri=True)
        t0 = time.time()
        con.set_progress_handler(lambda: time.time() - t0 > 30, 100000)           # a gold query that runs over 30 s
        try:
            got = con.execute(q["SQL"]).fetchall()
            r["gold_rows"], r["gold_ok"] = len(got), True
        except Exception as e:                                                    # noqa: BLE001
            r["gold_rows"], r["gold_ok"], r["gold_error"] = None, False, str(e)[:120]
        con.close()
        rows.append(r)
    out("bird", "questions.jsonl", rows)
    ok = [r for r in rows if r["gold_ok"]]
    rng = random.Random(SEED)
    ev = stratified(ok, lambda r: r["difficulty"], 150, rng)
    ids = {r["id"] for r in ev}
    out("bird", "eval_150_ids.json", sorted(ids))
    out("bird", "dev_100_ids.json", sorted(r["id"] for r in stratified([r for r in ok if r["id"] not in ids], lambda r: r["difficulty"], 100, rng)))
    MANIFEST["bird"]["gold"] = {"ok": len(ok), "failed": len(rows) - len(ok), "difficulty": dict(Counter(r["difficulty"] for r in rows))}


# 7. the same product in two catalogs
def abtbuy():
    src = D / "abtbuy"
    for t, name in (("tableA", "abt"), ("tableB", "buy")):
        out("abtbuy", f"{name}.jsonl", [{"id": f"{name}{r['id']}", "name": r["name"], "description": r["description"], "price": r["price"]}
                                        for r in csv.DictReader(open(src / f"{t}.csv", encoding="utf-8"))])
    gold = set()
    for split in ("train", "valid", "test"):
        rows = [{"abt": f"abt{r['ltable_id']}", "buy": f"buy{r['rtable_id']}", "match": r["label"] == "1"}
                for r in csv.DictReader(open(src / f"{split}.csv", encoding="utf-8"))]
        out("abtbuy", f"pairs_{'dev' if split != 'test' else 'eval'}_{split}.jsonl", rows)
        gold |= {(r["abt"], r["buy"]) for r in rows if r["match"]}
    out("abtbuy", "all_known_matches.jsonl", [{"abt": a, "buy": b} for a, b in sorted(gold)])


# 8. a plan under constraints
def naturalplan():
    rng = random.Random(SEED)
    keys = {"calendar_scheduling": lambda r: (r["num_people"], r["num_days"]), "meeting_planning": lambda r: r["num_people"],
            "trip_planning": lambda r: r["num_cities"]}
    for task, key in keys.items():
        data = json.loads((D / f"naturalplan/repo/data/{task}.json").read_text())
        rows = [{"id": k, **v} for k, v in data.items()]       # pred_5shot_pro: Gemini 1.5 Pro's answers, a free reference
        ev = stratified(rows, key, 100, rng)
        ids = {r["id"] for r in ev}
        out("naturalplan", f"{task}_eval.jsonl", ev)
        out("naturalplan", f"{task}_dev.jsonl", stratified([r for r in rows if r["id"] not in ids], key, 50, rng))


# 9. alerts on a numeric series
def nab():
    windows = json.loads((D / "nab/repo/labels/combined_windows.json").read_text())
    rows, seen = [], Counter()
    for name, w in sorted(windows.items()):
        n = sum(1 for _ in open(D / "nab/repo/data" / name)) - 1
        group = name.split("/")[0]
        seen[group] += 1                                     # dev: the artificial series and every fourth real one
        rows.append({"series": name, "group": group, "points": n, "windows": w,
                     "split": "dev" if name.startswith("artificial") or seen[group] % 4 == 1 else "eval"})
    out("nab", "series.jsonl", rows)
    MANIFEST["nab"]["split"] = dict(Counter(r["split"] for r in rows))
    MANIFEST["nab"]["anomaly_windows_eval"] = sum(len(r["windows"]) for r in rows if r["split"] == "eval")


ALL = {"taubench": taubench, "cuad": cuad, "ragtruth": ragtruth, "banking77": banking77, "credit": credit, "bird": bird,
       "abtbuy": abtbuy, "naturalplan": naturalplan, "nab": nab}

if __name__ == "__main__":
    mf = D / "manifest.json"
    MANIFEST.update(json.loads(mf.read_text()) if mf.exists() else {})
    for name in sys.argv[1:] or ALL:
        MANIFEST.pop(name, None)
        ALL[name]()
        print(name, json.dumps(MANIFEST.get(name), ensure_ascii=False)[:600])
    mf.write_text(json.dumps(MANIFEST, ensure_ascii=False, indent=1), encoding="utf-8")
