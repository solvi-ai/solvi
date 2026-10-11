"""Counterfactual explanations (solvi.experimental.counterfactual) on the credit policy v2: for every application v2
refuses or sends to review, the smallest change of its declared fields that makes it "approve" — checked against the
policy itself (validity), against a brute-force scan of the integer domain (minimality), for coverage and for
determinism. A pre-registered measurement; no LLM, no network.

    benchmarks/tasks/fetch.sh credit && python benchmarks/tasks/prepare.py      # the German Credit applications, once
    python benchmarks/counterfactual_credit.py --dry                             # 20 applications by hash (development)
    python benchmarks/counterfactual_credit.py                                   # every refused / review application

The catalog is the hand-written v2 of benchmarks/changes_credit.py (imported: `build(2)`, its fields `DECLARED`, its
`app_input`); the applications are read where that script reads them (`STAND_DATA`/credit/prepared). Every v2 decision
of the 1,000 applications is stored (SQLiteStorage, `<out>/decisions.db`) and each explanation is searched on the
response loaded back from the store (`store.get(id, system)`), over the declared fields except `id`. Domains: a number's
is the data's (min, max) over the 1,000 applications, a text field's is the set of its values in the data. The script
asserts that neither the catalog nor the searched fields read a sensitive field of about.json.

Per application: `search(res, "decision", target="approve", max_changes=1)` and `max_changes=2` (default max_evals);
  validity    — every explanation returned is applied to the application and decided again with `ask`: "approve";
  minimality  — for credit_amount_dm and duration_months, in each direction: the nearest integer of the domain at
                which the decision is "approve" (every integer tried, outward from the current value) against the
                single-field change found, within 1 unit; a case is a (field, direction) where such a value exists;
  coverage    — a one- or two-field change is found (≥ 80%);
  determinism — the searches are run twice, in two separate pools of freshly spawned processes (each its own hash
                seed): identical `to_dict()`s; `max_evals` never reached; evaluations and seconds per search reported.
An application without an explanation gets a diagnosis: the points it has, the points approve needs, and for each field
the largest drop of points a change of that field alone reaches (every value of its domain tried); the policy's points
are a sum of one rule per field, so when the two largest drops do not reach the need no pair of fields can.

Out (`--out`, default out/counterfactual_credit[_dry]/ at the repository's root): decisions.db, decisions.jsonl, pass1.jsonl,
pass2.jsonl (one line per application, appended as it is done: `--resume` continues), report.json, report.md. A line
every 60 s at least while it runs. Exit status 0 when every bar holds, 1 otherwise, 2 without the data."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import statistics
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
warnings.simplefilter("ignore")

import changes_credit as cc  # noqa: E402

from solvi.core.store import SQLiteStorage  # noqa: E402
from solvi.experimental.counterfactual import search  # noqa: E402

QUESTION, TARGET = "decision", "approve"
NUMERIC = ("credit_amount_dm", "duration_months")             # the minimality fields
SENSITIVE_DEFAULT = ("personal_status_and_sex", "foreign_worker", "age")
OVER = [f for f in cc.DECLARED if f != "id"]
BARS = {"validity": "validity = 100% of the explanations returned", "minimality": "minimality within 1 unit ≥ 95% of the cases",
        "coverage": "coverage (one or two fields) ≥ 80%", "determinism": "two runs identical; max_evals never reached"}
DRY_N, BEST_N = 20, 10


# ---------------------------------------------------------------- the stand
def build_v2(sensitive=()):
    """The hand-written credit catalog v2 of benchmarks/changes_credit.py, as that script builds it."""
    return cc.build(2, sensitive=set(sensitive))


def given_inputs(system):
    """The facts the catalog reads that no part of it sets: what it takes from the application."""
    cat = system.catalog
    made = set(cat.parts) | set(cat.rules)
    return {x for p in list(cat.parts.values()) + list(cat.rules.values()) for x in p.inputs} - made


def assert_no_sensitive(system, over, sensitive):
    read = given_inputs(system) & set(sensitive)
    assert not read, f"the catalog reads sensitive fields: {sorted(read)}"
    searched = set(over) & set(sensitive)
    assert not searched, f"sensitive fields among the searched ones: {sorted(searched)}"


def by_hash(ids):
    return sorted(ids, key=lambda i: hashlib.sha256(str(i).encode()).hexdigest())


def population(decisions):
    """{id: decision} → the ids v2 refuses or sends to review, in hash order."""
    return by_hash([i for i, d in decisions.items() if d in ("refuse", "review")])


def domains_of(apps):
    """A number's domain: the data's (min, max); a text field's: its values in the data."""
    out = {}
    for f in OVER:
        vals = [a[f] for a in apps]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in vals):
            out[f] = (min(vals), max(vals))
        else:
            out[f] = sorted(set(vals))
    return out


# ---------------------------------------------------------------- checks independent of the search
def decide(system, inp, early_exit=None):
    kw = {} if early_exit is None else {"early_exit": early_exit}
    r = system.ask(inp, store=False, **kw)[QUESTION]
    return r.answer, r.status


def approves(system, inp):
    a, s = decide(system, inp)
    return a == TARGET and s != "abstain"


def nearest_crossing(ok, x0, lo, hi, sign, evals=None):
    """The nearest integer x of [lo, hi] beyond x0 in direction `sign` with ok(x), every integer tried in order; None
    when there is none."""
    x = x0 + sign
    while lo <= x <= hi:
        if evals is not None:
            evals[0] += 1
        if ok(x):
            return x
        x += sign
    return None


def brute_crossings(system, inp, field, domain, evals=None):
    lo, hi = domain
    return {s: nearest_crossing(lambda x: approves(system, {**inp, field: x}), inp[field], lo, hi, s, evals)
            for s in (-1, 1)}


def apply_changes(inp, cf):
    """The application with an explanation's changes made (a number at its boundary, which ≤ / ≥ include)."""
    out = dict(inp)
    for c in cf["changes"]:
        if c["op"] not in ("=", "≤", "≥"):
            raise ValueError(f"a strict bound ({c['fact']} {c['op']} {c['to']}) names no value to apply")
        out[c["fact"]] = c["to"]
    return out


def validity(system, inp, cf):
    """→ (valid, answer, status) of the application decided again with the change applied."""
    try:
        a, s = decide(system, apply_changes(inp, cf))
    except ValueError as e:
        return False, None, str(e)
    return a == TARGET and s != "abstain", a, s


DIRS = {"≤": -1, "≥": 1}


def minimality_cases(found, brute):
    """found: the explanations (to_dict) of the one-change search; brute: {field: {-1|1: crossing or None}} → one case
    per (field, direction) where the brute force finds a crossing, ok when a single change of that field in that
    direction is found within 1 unit of it."""
    single = {}
    for cf in found:
        if len(cf["changes"]) == 1 and cf["changes"][0]["op"] in DIRS:
            c = cf["changes"][0]
            single.setdefault((c["fact"], DIRS[c["op"]]), c["to"])
    out = []
    for f, by in brute.items():
        for d, x in by.items():
            if x is None:
                continue
            got = single.get((f, d))
            out.append({"field": f, "dir": d, "brute": x, "found": got, "ok": got is not None and abs(got - x) <= 1})
    return out


def _approve_at_most():
    return max(p for p in range(-30, 31) if cc.decision_v2(p) == TARGET)


def diagnose(system, inp, domains):
    """Why no explanation: the points now, what approve needs, and the largest drop of points each field reaches alone
    (every value of its domain tried)."""
    def pts(x):
        r = system.ask(x, store=False, early_exit=False)
        return r.values.get("points")
    p0 = pts(inp)
    top = _approve_at_most()
    reach = {}
    for f in OVER:
        d = domains[f]
        vals = range(d[0], d[1] + 1) if isinstance(d, tuple) else d
        best = 0
        for v in vals:
            if v == inp[f]:
                continue
            best = max(best, p0 - pts({**inp, f: v}))
        reach[f] = best
    two = sum(sorted(reach.values(), reverse=True)[:2])
    need = p0 - top
    ans, status = decide(system, inp)
    why = (f"points {p0}, approve needs ≤ {top}: a drop of {need}; the two largest single-field drops give {two} — "
           + ("no pair of fields reaches it (the points are a sum of one rule per field)" if two < need else
              "a pair could reach it: the search missed it"))
    if status == "forced":
        why = "the hard rule too_large decides; " + why
    return {"points": p0, "approve_at_most": top, "needed": need, "reach": reach, "best_two": two,
            "status_now": status, "why": why}


# ---------------------------------------------------------------- one application
def _search(res, system, domains, mc):
    t = time.perf_counter()
    cf = search(res, QUESTION, max_changes=mc, over=OVER, target=TARGET, domains=domains, system=system)
    d = cf.to_dict()
    d["seconds"] = time.perf_counter() - t
    d["best"] = str(cf.best) if cf.best else None
    d["str"] = str(cf)
    return d


def explain_one(system, res, inp, domains, brute=True):
    """Both searches on one decision; with brute: validity, the brute-force minimality cases and, when nothing is
    found, the diagnosis."""
    rec = {"id": inp.get("id"), "answer": res.results[QUESTION].answer, "status": res.results[QUESTION].status,
           "mc1": _search(res, system, domains, 1), "mc2": _search(res, system, domains, 2)}
    if not brute:
        return rec
    rec["valid"] = [{"mc": mc, "text": c["text"], **dict(zip(("ok", "answer", "status"), validity(system, inp, c)))}
                    for mc in (1, 2) for c in rec[f"mc{mc}"]["found"]]
    n = [0]
    t = time.perf_counter()
    bf = {f: brute_crossings(system, inp, f, domains[f], n) for f in NUMERIC}
    rec["brute"] = {f: {str(k): v for k, v in d.items()} for f, d in bf.items()}
    rec["brute_evals"], rec["brute_seconds"] = n[0], time.perf_counter() - t
    rec["minimality"] = minimality_cases(rec["mc1"]["found"], bf)
    if not rec["mc2"]["found"]:
        rec["diagnosis"] = diagnose(system, inp, domains)
    return rec


# ---------------------------------------------------------------- the worker pool
_W = {}


def _init(db, inputs, domains, sensitive):
    warnings.simplefilter("ignore")
    _W["system"] = build_v2(sensitive)
    _W["store"] = SQLiteStorage(db)
    _W["inputs"], _W["domains"] = inputs, domains


def _work(job):
    app_id, sid, brute = job
    s = _W["system"]
    res = _W["store"].get(sid, s)
    try:
        rec = explain_one(s, res, _W["inputs"][app_id], _W["domains"], brute)
    except Exception as e:                                          # noqa: BLE001 — logged, the run goes on
        rec = {"id": app_id, "error": f"{type(e).__name__}: {e}"}
    rec["id"] = app_id
    return rec


def _done(path):
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["id"]] = r
    return out


def run_pass(name, path, ids, sids, args, inputs, domains, sensitive, brute):
    done = _done(path) if args.resume else {}
    if not args.resume and path.exists():
        path.unlink()
    todo = [i for i in ids if i not in done]
    print(f"[{name}] {len(done)} done, {len(todo)} to do, {args.workers} worker(s)", flush=True)
    if todo:
        ctx = mp.get_context("spawn")                              # fresh interpreters: each its own hash seed
        t0, last = time.time(), time.time()
        with ctx.Pool(args.workers, _init, (str(args.out / "decisions.db"), inputs, domains, sorted(sensitive))) as pool, \
                path.open("a", encoding="utf-8") as fh:
            it = pool.imap(_work, [(i, sids[i], brute) for i in todo])
            n = 0
            while n < len(todo):
                try:
                    rec = it.next(timeout=30)
                except mp.TimeoutError:
                    rec = None
                if rec is not None:
                    fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    fh.flush()
                    done[rec["id"]] = rec
                    n += 1
                if time.time() - last >= 60 or n == len(todo):
                    el = time.time() - t0
                    eta = el / n * (len(todo) - n) if n else float("nan")
                    print(f"[{name}] {n}/{len(todo)}  {el:.0f} s  ETA {eta:.0f} s", flush=True)
                    last = time.time()
    return done


# ---------------------------------------------------------------- the report
def _stats(xs):
    xs = sorted(xs)
    if not xs:
        return {}
    return {"n": len(xs), "median": statistics.median(xs), "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))],
            "max": xs[-1], "mean": statistics.fmean(xs)}


def _key(d):
    keep = ("found", "answer", "status", "searched", "not_searched", "evals", "exhausted", "inconclusive")
    return json.dumps({k: d.get(k) for k in keep}, sort_keys=True, ensure_ascii=False, default=str)


def report(ids, p1, p2, decisions, meta):
    errors = [i for i in ids if "error" in p1.get(i, {}) or "error" in p2.get(i, {}) or i not in p1 or i not in p2]
    ok_ids = [i for i in ids if i not in errors]
    valid = [v for i in ok_ids for v in p1[i]["valid"]]
    n_valid = sum(v["ok"] for v in valid)
    cases = [dict(c, id=i) for i in ok_ids for c in p1[i]["minimality"]]
    n_min = sum(c["ok"] for c in cases)
    cov2 = [i for i in ok_ids if p1[i]["mc2"]["found"]]
    cov1 = [i for i in ok_ids if p1[i]["mc1"]["found"]]
    same = [i for i in ok_ids if all(_key(p1[i][m]) == _key(p2[i][m]) for m in ("mc1", "mc2"))]
    exhausted = [i for i in ok_ids for m in ("mc1", "mc2") if p1[i][m]["exhausted"] or p2[i][m]["exhausted"]]
    n = len(ok_ids)
    bars = {
        "validity": {"bar": BARS["validity"], "valid": n_valid, "explanations": len(valid),
                 "rate": n_valid / len(valid) if valid else None,
                 "pass": bool(valid) and n_valid == len(valid)},
        "minimality": {"bar": BARS["minimality"], "ok": n_min, "cases": len(cases), "rate": n_min / len(cases) if cases else None,
                 "pass": bool(cases) and n_min / len(cases) >= 0.95,
                 "by_field": {f: f"{sum(c['ok'] for c in cases if c['field'] == f)} of "
                                 f"{sum(c['field'] == f for c in cases)}" for f in NUMERIC},
                 "misses": [c for c in cases if not c["ok"]][:50]},
        "coverage": {"bar": BARS["coverage"], "covered": len(cov2), "population": len(ids), "rate": len(cov2) / len(ids) if ids
                 else None, "one_field_only": len(cov1), "pass": bool(ids) and len(cov2) / len(ids) >= 0.80},
        "determinism": {"bar": BARS["determinism"], "identical": len(same), "compared": n, "exhausted": len(exhausted),
                 "pass": n > 0 and len(same) == n and not exhausted and not errors,
                 "differ": [i for i in ok_ids if i not in same][:20]},
    }
    cost = {f"mc{m}": {"evals": _stats([p1[i][f"mc{m}"]["evals"] for i in ok_ids]),
                       "seconds": _stats([p1[i][f"mc{m}"]["seconds"] for i in ok_ids]),
                       "seconds_run2": _stats([p2[i][f"mc{m}"]["seconds"] for i in ok_ids]),
                       "over_2s": sum(p1[i][f"mc{m}"]["seconds"] > 2 for i in ok_ids)} for m in (1, 2)}
    best_fields = Counter("+".join(sorted(c["fact"] for c in p1[i]["mc2"]["found"][0]["changes"])) for i in cov2)
    all_fields = Counter(c["fact"] for i in ok_ids for cf in p1[i]["mc2"]["found"] for c in cf["changes"])
    none = [{"id": i, "answer": p1[i]["answer"], "status": p1[i]["status"], **p1[i].get("diagnosis", {}),
             "search": p1[i]["mc2"]["str"]} for i in ok_ids if not p1[i]["mc2"]["found"]]
    best10 = [{"id": i, "answer": p1[i]["answer"], "best": p1[i]["mc2"]["best"]} for i in ids[:BEST_N] if i in ok_ids]
    out = {"meta": meta, "bars": bars, "verdict": "pass" if all(b["pass"] for b in bars.values()) else "fail",
           "cost_per_search": cost,
           "brute_force": {"evals": _stats([p1[i]["brute_evals"] for i in ok_ids]),
                           "seconds": _stats([p1[i]["brute_seconds"] for i in ok_ids])},
           "decisions_v2": dict(Counter(decisions.values())),
           "population_by_answer": dict(Counter(p1[i]["answer"] for i in ok_ids)),
           "fields_of_the_best": dict(best_fields.most_common()), "fields_of_all_found": dict(all_fields.most_common()),
           "best_10_by_hash": best10, "no_explanation": none, "errors": [{"id": i, "error": (p1.get(i) or p2.get(i) or
                                                                                                {}).get("error")}
                                                                            for i in errors]}
    return out


def to_md(r):
    m, b = r["meta"], r["bars"]
    L = [f"# Counterfactual credit decisions — explanations on the credit policy v2 "
         f"({'dry' if m['dry'] else 'full'} run)", "",
         f"Verdict of the bars: **{r['verdict']}**. Population: {b['coverage']['population']} applications v2 refuses or "
         f"sends to review (of {m['applications']}; v2 decisions {r['decisions_v2']}).", "",
         "| id | bar | measured | pass |", "|---|---|---|---|",
         f"| validity | {b['validity']['bar']} | {b['validity']['valid']} of {b['validity']['explanations']} | {b['validity']['pass']} |",
         f"| minimality | {b['minimality']['bar']} | {b['minimality']['ok']} of {b['minimality']['cases']} ({b['minimality']['by_field']}) | "
         f"{b['minimality']['pass']} |",
         f"| coverage | {b['coverage']['bar']} | {b['coverage']['covered']} of {b['coverage']['population']} (one field: "
         f"{b['coverage']['one_field_only']}) | {b['coverage']['pass']} |",
         f"| determinism | {b['determinism']['bar']} | {b['determinism']['identical']} of {b['determinism']['compared']} identical; "
         f"{b['determinism']['exhausted']} reached max_evals | {b['determinism']['pass']} |", "",
         "## Cost per search", "", "| search | evaluations median / p95 / max | seconds median / p95 / max | > 2 s |",
         "|---|---|---|---|"]
    for k, c in r["cost_per_search"].items():
        e, s = c["evals"], c["seconds"]
        if e:
            L.append(f"| {k} | {e['median']:g} / {e['p95']:g} / {e['max']:g} | {s['median']:.3f} / {s['p95']:.3f} / "
                     f"{s['max']:.3f} | {c['over_2s']} |")
    L += ["", "## Fields changed", "", "best explanation: " + json.dumps(r["fields_of_the_best"], ensure_ascii=False),
          "", "all explanations found: " + json.dumps(r["fields_of_all_found"], ensure_ascii=False), "",
          "## cf.best for 10 applications by hash", ""]
    L += [f"- {x['id']} ({x['answer']}): {x['best']}" for x in r["best_10_by_hash"]]
    L += ["", f"## Without an explanation ({len(r['no_explanation'])})", ""]
    L += [f"- {x['id']} ({x['answer']}, {x['status']}): {x.get('why')}" for x in r["no_explanation"]]
    if r["bars"]["minimality"]["misses"]:
        L += ["", "## Minimality misses", ""] + [f"- {json.dumps(c)}" for c in r["bars"]["minimality"]["misses"]]
    if r["errors"]:
        L += ["", "## Errors", ""] + [f"- {e['id']}: {e['error']}" for e in r["errors"]]
    L += ["", "## Run", "", "```", json.dumps(m, indent=1, ensure_ascii=False, default=str), "```", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--dry", action="store_true", help=f"{DRY_N} applications of the population by hash")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", help="keep decisions.db and the passes' finished lines")
    args = p.parse_args(argv)
    args.out = args.out or HERE.parent / "out" / ("counterfactual_credit_dry" if args.dry else "counterfactual_credit")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        apps = cc.read_jsonl(cc.PREPARED / "applications_history.jsonl") + \
            cc.read_jsonl(cc.PREPARED / "applications_new.jsonl")
        sensitive = set(json.loads((cc.PREPARED / "about.json").read_text())["sensitive"])
    except OSError:
        print(f"the German Credit applications are not in {cc.PREPARED}: benchmarks/tasks/fetch.sh credit && python "
              "benchmarks/tasks/prepare.py (or set STAND_DATA). Nothing was run.")
        return 2
    t0 = time.time()
    v2 = build_v2(sensitive)
    assert_no_sensitive(v2, OVER, sensitive)
    inputs = {a["id"]: cc.app_input(a) for a in apps}
    assert not any(set(x) & sensitive for x in inputs.values()), "an input carries a sensitive field"
    domains = domains_of(apps)

    db, index = args.out / "decisions.db", args.out / "decisions.jsonl"
    if args.resume and db.exists() and index.exists():
        rows = [json.loads(x) for x in index.read_text().splitlines() if x.strip()]
    else:
        for f in args.out.glob("decisions.db*"):
            f.unlink()
        store = SQLiteStorage(db)
        v2.storage, store.catalog = store, v2
        rows = []
        for a in apps:
            res = v2.ask(inputs[a["id"]])
            rows.append({"id": a["id"], "stored_id": res.stored_id, "answer": res[QUESTION].answer,
                         "status": res[QUESTION].status})
        v2.storage = None
        index.write_text("".join(json.dumps(r) + "\n" for r in rows))
        assert store.verify()["ok"], "the decisions store does not verify"
        store.close()
    decisions = {r["id"]: r["answer"] for r in rows}
    sids = {r["id"]: r["stored_id"] for r in rows}
    ids = population(decisions)
    if args.dry:
        ids = ids[:DRY_N]
    print(f"{len(apps)} applications, v2 {dict(Counter(decisions.values()))}; population {len(ids)}"
          + (" (dry)" if args.dry else ""), flush=True)
    jdomains = {k: list(v) for k, v in domains.items()}
    p1 = run_pass("pass 1", args.out / "pass1.jsonl", ids, sids, args, inputs, domains, sensitive, True)
    p2 = run_pass("pass 2", args.out / "pass2.jsonl", ids, sids, args, inputs, domains, sensitive, False)
    import solvi
    meta = {"dry": args.dry, "applications": len(apps), "population": len(ids), "workers": args.workers,
            "catalog": "benchmarks/changes_credit.py build(2)", "catalog_fp": v2.fingerprint()["catalog"],
            "data": str(cc.PREPARED), "over": OVER, "domains": jdomains, "sensitive": sorted(sensitive),
            "max_evals": 5000, "solvi": getattr(solvi, "__version__", "?"), "seconds": round(time.time() - t0, 1),
            "python": sys.version.split()[0]}
    r = report(ids, p1, p2, decisions, meta)
    (args.out / "report.json").write_text(json.dumps(r, indent=1, ensure_ascii=False, default=str))
    (args.out / "report.md").write_text(to_md(r))
    print(to_md(r))
    return 0 if r["verdict"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
