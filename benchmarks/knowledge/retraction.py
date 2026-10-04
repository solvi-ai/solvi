"""Retraction exactness of the knowledge store: the store after a retraction has the fingerprint of the store rebuilt
from its journal without the retracted items — on a synthetic store of 10,000 items, 1,000 random retractions.

    uv run python benchmarks/knowledge/retraction.py                       # 10,000 items, 1,000 retractions (~15 min)
    uv run python benchmarks/knowledge/retraction.py --items 2000 --retractions 100     # a quick run
    → benchmarks/knowledge/results/retraction.json (rows: retraction.jsonl next to it; a rerun resumes)

The synthetic store (no model, no network, seed 20261103): label facts of 20 questions, base facts "is" with 15% reused
subjects (so values contradict and disputes, refutations and confirmations happen), derived items of every kind with
1–3 premises up to depth 5 (10% with a second justification), confirmations from other sources (5%), promotions of
behaviour-changing items (70% of the proposed), a person resolving open disputes, clock ticks with reconfirm_after on 10%
of the items, and drift flags on a question's scope that later end. Sources: person, outcome, spec (a "verified" item
needs a stored System 2 decision behind it; the research prototype this follows also drew "verified" sources).

Each retraction is applied to the live store, which is then compared with `rebuild(skip=every item retracted so far)`:
"equal" when the fingerprints match. Also reported: the journal's chain after each retraction, `verify()` at the end,
the size of the connected component the retraction touched against the time it took (retraction cost grows with it)."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from solvi.core.knowledge import KnowledgeStore  # noqa: E402

SEED = 20261103
QUESTIONS = [f"q{i:02d}" for i in range(20)]
CLASSES = ["c0", "c1", "c2", "c3", "c4"]
SOURCES = ["person"] * 45 + ["outcome"] * 33 + ["spec"] * 22
OUT = Path(__file__).resolve().parent / "results"


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def build(n_items, seed):
    """The synthetic store → (store, stats)."""
    rng = random.Random(seed)
    ks = KnowledgeStore()
    depth, keys, hyp, present = {}, [], [], []
    stats = {k: 0 for k in ("labels", "base", "derived", "second", "confirm", "promote", "resolve", "tick", "flag",
                            "end_flag", "refused")}
    flags, writes, t0, last = [], 0, time.time(), time.time()
    while len(ks.items) < n_items:
        with ks.batch():
            for _ in range(50):
                if len(ks.items) >= n_items:
                    break
                writes += 1
                u = rng.random()
                if writes % 20 == 0:
                    ks.tick(rng.randint(1, 5))
                    stats["tick"] += 1
                if writes % 400 == 0:
                    flags.append(ks.flag(scope={"q": rng.choice(QUESTIONS)}, why="synthetic drift"))
                    stats["flag"] += 1
                if writes % 400 == 200 and flags:
                    ks.end_flag(flags.pop(0), why="re-confirmed")
                    stats["end_flag"] += 1
                if u < 0.05 and present:                              # a confirmation from another source
                    it = ks.items[rng.choice(present)]
                    ks.add(it.kind, it.body, it.scope, source=rng.choice(SOURCES), by="c",
                           derived_from=it.asserts[0]["just"])
                    stats["confirm"] += 1
                    continue
                if u < 0.08 and hyp:                                  # a promotion (70%)
                    i = hyp.pop(rng.randrange(len(hyp)))
                    if rng.random() < 0.7:
                        ks.promote(i, measured={"synthetic": True})
                        stats["promote"] += 1
                    continue
                if u < 0.10 and ks.questions:                         # a person resolves an open dispute (50%)
                    q = ks.questions[rng.randrange(len(ks.questions))]
                    if rng.random() < 0.5:
                        ks.resolve(tuple(q["key"]), ks.items[rng.choice(q["items"])].value, by="person")
                        stats["resolve"] += 1
                    continue
                ra = 30 if rng.random() < 0.10 else None
                k = rng.random()
                n = len(ks.items)
                if k < 0.20:                                          # a label
                    q = rng.choice(QUESTIONS)
                    same = [x for x in keys if x[0] == "label" and x[1] == q]
                    ex = rng.choice(same)[2] if same and rng.random() < 0.15 else f"{q}:x{n}"
                    iid = ks.add("fact", {"s": ex, "r": "label", "o": rng.choice(CLASSES)}, {"q": q},
                                 source=rng.choice(["person"] * 7 + ["outcome"] * 3), by="p", reconfirm_after=ra)
                    keys.append(("label", q, ex))
                    stats["labels"] += 1
                elif k < 0.65 or len(present) < 20:                   # a base fact
                    world = [x for x in keys if x[0] == "fact"]
                    if world and rng.random() < 0.15:
                        s, o = rng.choice(world)[2], (rng.randint(0, 1) if rng.random() < 0.5 else 0)
                    else:
                        s, o = f"s{n}", 0
                    iid = ks.add("fact", {"s": s, "r": "is", "o": o}, {"q": "world"}, source=rng.choice(SOURCES), by="b",
                                 reconfirm_after=ra)
                    keys.append(("fact", "world", s))
                    stats["base"] += 1
                else:                                                 # a derived item
                    cand = [i for i in present if depth.get(i, 0) <= 4]
                    if not cand:
                        continue
                    prem = rng.sample(cand, min(len(cand), rng.randint(1, 3)))
                    kind = rng.choice(["rule", "skill", "action", "fact"])
                    if kind == "fact":
                        world = [x for x in keys if x[0] == "fact"]
                        if world and rng.random() < 0.15:
                            body = {"s": rng.choice(world)[2], "r": "is", "o": rng.randint(0, 1)}
                        else:
                            body = {"s": f"d{n}", "r": "is", "o": 0}
                        keys.append(("fact", "world", body["s"]))
                        scope = {"q": "world"}
                    else:
                        body, scope = {"name": f"{kind}{n}", "uses": sorted(prem)}, {"q": rng.choice(QUESTIONS)}
                    iid = ks.add(kind, body, scope, source=rng.choice(SOURCES), by="d", derived_from=prem)
                    if iid is None:
                        stats["refused"] += 1
                        continue
                    depth[iid] = 1 + max(depth.get(p, 0) for p in prem)
                    stats["derived"] += 1
                    if kind != "fact":
                        hyp.append(iid)
                    if rng.random() < 0.10:                            # a second justification
                        prem2 = rng.sample(cand, min(len(cand), rng.randint(1, 3)))
                        if ks.add(kind, body, scope, source=rng.choice(SOURCES), by="d2", derived_from=prem2):
                            stats["second"] += 1
                if iid is None:
                    stats["refused"] += 1
        present = sorted(ks.state)
        if time.time() - last > 30:
            log(f"build: {len(ks.items)} items, journal {len(ks.journal)}, {time.time() - t0:.0f} s")
            last = time.time()
    return ks, stats


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sx = sum((a - mx) ** 2 for a in rx) ** 0.5
    sy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (sx * sy) if sx and sy else None


def median(v):
    v = sorted(v)
    return (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2 if v else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=10000)
    ap.add_argument("--retractions", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tag", default="retraction")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    rows_path = OUT / f"{a.tag}.jsonl"
    done = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    t0 = time.time()
    ks, stats = build(a.items, a.seed)
    log(f"built {len(ks.items)} items, journal {len(ks.journal)}, {ks.counts()}, {time.time() - t0:.0f} s; "
        f"verify {ks.verify()}")
    rng = random.Random(a.seed + 1)
    retracted = []
    for row in done:                                   # resume: the same draws, re-applied
        x = rng.choice(sorted(ks.state))
        if x != row["id"]:
            raise RuntimeError("resume: the draws differ from the saved rows; delete the rows file to start again")
        ks.retract(x, by="test", why="random retraction")
        retracted.append(x)
    last = time.time()
    with open(rows_path, "a") as fh:
        for k in range(len(done), a.retractions):
            x = rng.choice(sorted(ks.state))
            comp = len(ks.component([x]))
            t1 = time.time()
            changed = ks.retract(x, by="test", why="random retraction")
            t_retract = time.time() - t1
            retracted.append(x)
            t1 = time.time()
            equal = ks.rebuild(skip=set(retracted)).fingerprint() == ks.fingerprint()
            row = {"k": k, "id": x, "kind": ks.items[x].kind, "component": comp, "changed": len(changed),
                   "equal": equal, "t_retract_s": round(t_retract, 4), "t_rebuild_s": round(time.time() - t1, 4)}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            done.append(row)
            if time.time() - last > 30:
                log(f"retraction {k + 1}/{a.retractions}: equal {sum(r['equal'] for r in done)}/{len(done)}, "
                    f"component {comp}, retract {t_retract:.2f} s")
                last = time.time()
    comp = [r["component"] for r in done]
    tr = [r["t_retract_s"] for r in done]
    cut = median(comp)
    summary = {"items": len(ks.items), "journal": len(ks.journal), "build": stats, "counts_end": ks.counts(),
               "retractions": len(done), "equal": sum(r["equal"] for r in done), "verify_end": ks.verify(),
               "component_median": cut, "component_max": max(comp) if comp else None,
               "t_retract_median_s": median(tr), "t_rebuild_median_s": median([r["t_rebuild_s"] for r in done]),
               "t_retract_median_small_half_s": median([t for c, t in zip(comp, tr) if c < cut]),
               "t_retract_median_large_half_s": median([t for c, t in zip(comp, tr) if c >= cut]),
               "spearman_component_time": spearman(comp, tr) if len(done) > 2 else None,
               "seconds": round(time.time() - t0, 1)}
    (OUT / f"{a.tag}.json").write_text(json.dumps(summary, indent=1))
    log(f"equal {summary['equal']} / {summary['retractions']}, verify {summary['verify_end']}, "
        f"median retraction {summary['t_retract_median_s']} s (component median {cut})")
    return 0 if summary["equal"] == summary["retractions"] and summary["verify_end"] else 1


if __name__ == "__main__":
    sys.exit(main())
