"""What a signature (solvi.core.store.signature) adds over the hash chain: its syndrome code, and the octonion experiment next to it.

    uv run python benchmarks/trace_signature.py            # ~30 seconds
    uv run python benchmarks/trace_signature.py --json out.json

The attack: one stored record is edited, and every hash after it and the stored head are recomputed (an attacker with
write access to the store). Kept elsewhere before the attack: the head (count, last hash) and a signature. The hash chain
with that anchor (TraceStorage.verify(anchor=head)) says the store was rewritten, not which record. Compared:
- "syndrome" (solvi.core.store.signature): S0 = Σ h_i, S1 = Σ (i+1) h_i mod a 256-bit prime — 64 bytes; restores the whole hash;
- "octonion" (benchmarks/octonion_signature.py, out of the package since 0.8): the positional octonion product —
  32 float64 = 256 bytes; restores 28 bytes of the hash;
- (not a code) all record hashes kept elsewhere: 32 bytes per record, locates any number of changes.

Also: two and three changes (a code must refuse to locate, never name a wrong record), and time per store size — of the
code alone over 40-byte items ("ms"), and through a real store of stored decisions ("store_ms": store.signature() and
store.verify(signature=), which read and hash every record and check each stored response; a SQLite and a JSON-lines
store of a small three-question System's decisions)."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import os
import sys
import time

from solvi.core.store import signature as syndrome

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import octonion_signature as octonion  # noqa: E402

ALGS = ("syndrome", "octonion")
CODES = {"syndrome": (syndrome.sign, syndrome.check), "octonion": (octonion.sign, octonion.check)}


def sign(items, alg):
    return CODES[alg][0](items)


def check(items, sig):
    return CODES[sig["alg"]][1](items, sig)


def trial(rng, n, changes):
    items = [rng.randbytes(40) for _ in range(n)]
    sigs = {a: sign(items, a) for a in ALGS}
    ks = rng.sample(range(n), changes)
    orig = hashlib.sha256(items[ks[0]]).digest()
    for k in ks:
        items[k] = rng.randbytes(40)
    out = {}
    for a in ALGS:
        r = check(items, sigs[a])
        out[f"{a}_detected"] = not r["ok"]
        out[f"{a}_located_right"] = changes == 1 and r["index"] == ks[0] and orig.hex().startswith(r["digest"])
        out[f"{a}_located_wrong"] = r["index"] is not None and (changes > 1 or r["index"] != ks[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    rng = random.Random(0)
    out = {"rates": {}, "ms": {}}
    for changes, trials in ((1, 2000), (2, 1000), (3, 500)):
        rows = [trial(rng, rng.randint(max(2, changes), 500), changes) for _ in range(trials)]
        out["rates"][f"{changes} change(s), {trials} stores of 2..500 records"] = {
            k: sum(r[k] for r in rows) / trials for k in rows[0]}
    for n in (100, 1000, 10000, 50000):
        items = [rng.randbytes(40) for _ in range(n)]
        row = {}
        for alg in ALGS:
            t = time.perf_counter()
            sig = sign(items, alg)
            row[f"{alg}_sign"] = round((time.perf_counter() - t) * 1e3, 1)
            edited = list(items)
            edited[n // 3] = b"edited"
            t = time.perf_counter()
            assert check(edited, sig)["index"] == n // 3
            row[f"{alg}_locate"] = round((time.perf_counter() - t) * 1e3, 1)
        out["ms"][n] = row
    out["store_ms"] = store_times((1000, 10000))
    out["bytes"] = {"syndrome": 64, "octonion": 32 * 8, "hash chain head": "count + 32",
                    "all record hashes": "32 per record"}
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)


def store_times(sizes, reps=3):
    """store.signature() and store.verify(signature=) on stores of real decisions → {backend: {n: {sign, verify}}} in ms
    (the best of `reps`: the machine's other load is not the code's time)."""
    import tempfile

    from solvi import Answer, Catalog, Question, System
    from solvi.core.store import open_storage
    cat = Catalog()

    @cat.fn
    def total(amount: float, tax: float) -> float:
        return amount * (1 + tax)

    @cat.check
    def large(total: float) -> bool:
        return total > 1000

    @cat.rule("approve")
    def approve(total, large):
        return "review" if large else "approve"

    @cat.rule("tier")
    def tier(total):
        return "high" if total > 500 else "low"

    @cat.rule("flag")
    def flag(large):
        return large

    qs = [Question("approve", "?", Answer.choice(["approve", "review"])), Question("tier", "?", Answer.choice(["low", "high"])),
          Question("flag", "?", Answer.yes_no())]
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for ext in ("db", "jsonl"):
            out[ext] = {}
            for n in sizes:
                store = open_storage(os.path.join(d, f"s{n}.{ext}"))
                s = System(cat, qs, storage=store)
                for i in range(n):
                    s.ask({"amount": 10.0 * i, "tax": 0.2})
                best = {"sign": float("inf"), "verify": float("inf")}
                for _ in range(reps):
                    t = time.perf_counter()
                    sig = store.signature()
                    best["sign"] = min(best["sign"], (time.perf_counter() - t) * 1e3)
                    t = time.perf_counter()
                    assert store.verify(signature=sig)["ok"]
                    best["verify"] = min(best["verify"], (time.perf_counter() - t) * 1e3)
                out[ext][n] = {k: round(v, 1) for k, v in best.items()}
    return out


if __name__ == "__main__":
    main()
