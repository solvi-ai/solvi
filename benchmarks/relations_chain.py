"""Forward chaining over a learned composition table, against the table's own derivation, on CLUTRR stories:

    python benchmarks/relations_chain.py --data DIR     # DIR holds the clean split's train.csv (or SOLVI_CLUTRR=DIR)

Data: CLUTRR (Sinha et al., 2019), the clean split `gen_train23_test2to10` (stories of 2–3 relations for training);
only its training file is read, in place; the test file is never opened. The training stories are split by a hash of
the id: sha256(id) % 5 == 0 is the held-out fifth (the stories chained over), the rest fits the table. The input is
the benchmark's relation graph (story edges and their relation names), not the story text.

The table: CompositionTable(majority=0.9, support=2) with residual fill, fitted on every shortest path between the
query's two people in the fit stories (at most 33 per story), labelled with the query's relation. Per held-out story:
a fresh in-memory KnowledgeStore, the table written as rule items (`to_knowledge`), each story edge a fact
(`pair_fact`, source "outcome"), `forward_chain` to its fixed point; the store's reading of the query pair is
compared with `derive_paths` over the shortest paths between the two people:

  equal          the table answers r and the store holds r on the pair (active, or a hypothesis: it rests only on a
                 disputed pair, listed apart);
  refusals agree the table refuses because the bracketings disagree and the pair is disputed, or the table cannot
                 derive the pair and the store holds nothing on it;
  differences    anything else — each listed with its paths (the store chains over the whole story graph, the table
                 over the shortest paths only, so a difference can come from a longer path).

Also printed: the fixed-point depth (rounds) per story, the time of the fit and of the chains, peak memory, and, for
reference, the held-out labels against both readings (wrong among answered). One deterministic run.

Measured (2026-10, this machine, CPU, one deterministic run): the table fitted on 7,258 stories (7,258 chains, 111
cells) in 0.03 s; of the 1,816 held-out stories (840 of 2 relations, 976 of 3) 1,675 equal (0.922; none of them a
hypothesis), 11 disputed where the bracketings disagree (0.006), 130 with nothing derived where the table cannot derive
(0.072), 0 differences; the fixed point took 0 rounds for 72 stories, 1 for 826 and 2 for 918; chaining 2.56 s in all,
1.4 ms a story; peak memory 0.06 GB; labels for reference: 4 wrong of 1,675 answered, the same for both readings."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import resource
import sys
import time
import warnings
from collections import Counter, defaultdict, deque
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.relations import CompositionTable, forward_chain, pair_fact

from solvi.core.knowledge import KnowledgeStore

RULES = {"domain": "kinship"}


def held_out(row_id: str) -> bool:
    return int(hashlib.sha256(row_id.encode()).hexdigest(), 16) % 5 == 0


def read_train(data: Path):
    """Rows of the training file: id, task length k, edges, edge types, query, label."""
    path = data / "train.csv"
    if path.name != "train.csv" or not path.exists():
        sys.exit(f"no train.csv in {data}")
    with open(path, encoding="utf-8", newline="") as f:
        for rec in csv.DictReader(f):
            yield {"id": rec["id"], "k": int(rec["task_name"].split(".")[-1]),
                   "edges": [tuple(e) for e in ast.literal_eval(rec["story_edges"])],
                   "types": list(ast.literal_eval(rec["edge_types"])),
                   "q": tuple(ast.literal_eval(rec["query_edge"])), "label": rec["target_text"]}


def paths(row, cap=32):
    """Every shortest directed path from the query's head to its tail, as tuples of relation names (≤ cap + 1)."""
    adj = defaultdict(list)
    for (i, j), t in zip(row["edges"], row["types"]):
        adj[i].append((j, t))
    s, g = row["q"]
    dist, preds, dq = {s: 0}, defaultdict(list), deque([s])
    while dq:
        u = dq.popleft()
        for v, t in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                dq.append(v)
                preds[v].append((u, t))
            elif dist[v] == dist[u] + 1:
                preds[v].append((u, t))
    if g not in dist:
        return []
    out = []

    def back(v, acc):
        if v == s:
            out.append(tuple(acc[::-1]))
            return
        for u, t in preds[v]:
            back(u, acc + [t])
            if len(out) > cap:
                return
    back(g, [])
    return out


def store_reading(ks, x, y, scope):
    vals = {}
    for rec in ks.find(kind="fact", s=[x, y], scope=scope):
        if rec["status"] in ("active", "hypothesis", "disputed"):
            vals[rec["body"]["o"]] = rec["status"]
    if not vals:
        return ("absent", None, None)
    if len(vals) == 1:
        (r, st), = vals.items()
        return ("answered", r, st)
    return ("disputed", tuple(sorted(vals)), "disputed")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("SOLVI_CLUTRR"), help="the split's directory (train.csv)")
    ap.add_argument("--limit", type=int, default=None, help="chain over the first N held-out stories only (a smoke)")
    ap.add_argument("--json", default=None, help="write the counts and the differences here")
    a = ap.parse_args(argv)
    if not a.data:
        sys.exit("--data DIR (or SOLVI_CLUTRR) is the directory of the split's train.csv")
    rows = list(read_train(Path(a.data).expanduser()))
    fit = [r for r in rows if not held_out(r["id"])]
    dev = [r for r in rows if held_out(r["id"])]
    if a.limit:
        dev = dev[:a.limit]
    t0 = time.perf_counter()
    examples = [(p, r["label"]) for r in fit for p in paths(r)]
    table = CompositionTable(majority=0.9, support=2).fit(examples)
    fit_s = time.perf_counter() - t0
    print(f"fit: {len(fit)} stories, {len(examples)} chains, {len(table)} cells, {fit_s:.2f} s; held out: {len(dev)}",
          flush=True)

    cats, depth, by_k = Counter(), Counter(), defaultdict(Counter)
    wrong = Counter()
    diffs, chain_s, last = [], 0.0, time.time()
    for n, r in enumerate(dev):
        ks = KnowledgeStore()
        table.to_knowledge(ks, RULES)
        sc = {"story": r["id"]}
        for (i, j), t in zip(r["edges"], r["types"]):
            ks.add("fact", pair_fact(i, j, t), sc, source="outcome", by="story")
        t1 = time.perf_counter()
        rep = forward_chain(ks, scope=sc, rule_scope=RULES)
        chain_s += time.perf_counter() - t1
        depth[rep.rounds] += 1
        h, tl = r["q"]
        got = store_reading(ks, h, tl, sc)
        ps = paths(r)
        d = table.derive_paths(ps)
        if d.reason == "derived" and got[0] == "answered" and got[1] == d.relation:
            cat = "equal" if got[2] == "active" else "equal (hypothesis)"
        elif d.reason == "disagree" and got[0] == "disputed":
            cat = "refusals agree: disagree = disputed"
        elif d.reason == "underivable" and got[0] == "absent":
            cat = "refusals agree: underivable = absent"
        else:
            cat = "difference"
            diffs.append({"id": r["id"], "k": r["k"], "table": [d.reason, d.relation, dict(d.candidates)],
                          "store": [got[0], got[1], got[2]], "paths": [list(p) for p in ps],
                          "edges": len(r["edges"]), "rounds": rep.rounds})
        cats[cat] += 1
        by_k[r["k"]][cat] += 1
        if d.answered:
            wrong["table answered"] += 1
            wrong["table wrong"] += d.relation != r["label"]
        if got[0] == "answered":
            wrong["store answered"] += 1
            wrong["store wrong"] += got[1] != r["label"]
        if not ks.verify():
            sys.exit(f"store failed verify() on story {r['id']}")
        if time.time() - last > 30:
            print(f"  {n + 1}/{len(dev)} stories, {chain_s:.1f} s chaining", flush=True)
            last = time.time()
    total = sum(cats.values())
    print(f"\nstories: {total}")
    for c in ("equal", "equal (hypothesis)", "refusals agree: disagree = disputed",
              "refusals agree: underivable = absent", "difference"):
        print(f"  {c:40s} {cats[c]:6d}  ({cats[c] / total:.3f})")
    print("by length k:", {k: dict(v) for k, v in sorted(by_k.items())})
    print("fixed-point depth (rounds → stories):", dict(sorted(depth.items())))
    print(f"labels (reference): table wrong {wrong['table wrong']} of {wrong['table answered']} answered; "
          f"store wrong {wrong['store wrong']} of {wrong['store answered']} answered")
    print(f"time: fit {fit_s:.2f} s; chaining {chain_s:.2f} s in all, {1000 * chain_s / max(1, total):.1f} ms per "
          f"story; peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
    for x in diffs:
        print("difference:", json.dumps(x))
    if a.json:
        Path(a.json).write_text(json.dumps({"counts": dict(cats), "by_k": {k: dict(v) for k, v in by_k.items()},
                                            "depth": dict(depth), "labels": dict(wrong), "fit_s": fit_s,
                                            "chain_s": chain_s, "cells": len(table), "chains": len(examples),
                                            "differences": diffs}, indent=1))


if __name__ == "__main__":
    main()
