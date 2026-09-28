"""What the algebraic signature (solvi.signature) adds over the hash chain, and what a classical code does on the same job.

    uv run python benchmarks/trace_signature.py            # ~1 minute
    uv run python benchmarks/trace_signature.py --json out.json

The attack: one stored record is edited, and every hash after it and the stored head are recomputed (an attacker with
write access to the store). Kept elsewhere before the attack: the head (count, last hash) and a signature. Compared:
- hash chain + anchor (TraceStorage.verify(anchor=head)): does it say WHICH record changed?
- solvi.signature (positional octonion code, 32 float64 = 256 bytes): the changed record's position and content hash;
- syndrome code (classical, not shipped): S0 = sum h_i, S1 = sum (i+1) h_i mod a 256-bit prime (64 bytes). One change
  at k by d: S0' − S0 = d, S1' − S1 = (k+1) d → k and h_k = h_k' − d exactly;
- all record hashes kept elsewhere (32 bytes per record): locates any number of changes, grows with the store.

Also: two changes (both codes should refuse to locate, never name a wrong record), and time per store size."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

from solvi.signature import BYTES, check, sign

P = 2**256 - 189                     # the largest prime below 2^256


def syn_sign(ds):
    s0 = s1 = 0
    for i, d in enumerate(ds):
        h = int.from_bytes(d, "big")
        s0, s1 = (s0 + h) % P, (s1 + (i + 1) * h) % P
    return s0, s1


def syn_locate(ds, sig):
    """→ ("ok" | index | None, restored hash int)."""
    c0, c1 = syn_sign(ds)
    d0, d1 = (c0 - sig[0]) % P, (c1 - sig[1]) % P
    if d0 == 0 and d1 == 0:
        return "ok", None
    if d0 == 0:
        return None, None
    k = d1 * pow(d0, -1, P) % P - 1
    if not 0 <= k < len(ds):
        return None, None
    return k, (int.from_bytes(ds[k], "big") - d0) % P


def trial(rng, n, changes):
    items = [rng.randbytes(40) for _ in range(n)]
    ds = [hashlib.sha256(x).digest() for x in items]
    osig, ssig = sign(items), syn_sign(ds)
    ks = rng.sample(range(n), changes)
    orig = {k: ds[k] for k in ks}
    for k in ks:
        items[k] = rng.randbytes(40)
        ds[k] = hashlib.sha256(items[k]).digest()
    t = time.perf_counter()
    r = check(items, osig)
    t_oct = time.perf_counter() - t
    t = time.perf_counter()
    sk, sh = syn_locate(ds, ssig)
    t_syn = time.perf_counter() - t
    one = ks[0]
    return {"oct_detect": not r["ok"], "oct_right": changes == 1 and r["index"] == one
            and r["digest"] == orig[one][:BYTES].hex(), "oct_wrong": r["index"] is not None and changes > 1,
            "syn_detect": sk != "ok", "syn_right": changes == 1 and sk == one and sh == int.from_bytes(orig[one], "big"),
            "syn_wrong": sk not in ("ok", None) and changes > 1, "t_oct": t_oct, "t_syn": t_syn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    a = ap.parse_args()
    rng = random.Random(0)
    out = {"rates": {}, "time": {}}
    for changes, trials in ((1, 2000), (2, 1000), (3, 500)):
        rows = [trial(rng, rng.randint(max(2, changes), 500), changes) for _ in range(trials)]
        rate = {k: sum(r[k] for r in rows) / trials for k in rows[0] if not k.startswith("t_")}
        out["rates"][f"{changes} change(s), {trials} stores of 2..500 records"] = rate
    for n in (100, 1000, 10000, 50000):
        items = [rng.randbytes(40) for _ in range(n)]
        ds = [hashlib.sha256(x).digest() for x in items]
        t = time.perf_counter()
        sig = sign(items)
        t_sign = time.perf_counter() - t
        items[n // 3] = b"edited"
        t = time.perf_counter()
        assert check(items, sig)["index"] == n // 3
        t_loc = time.perf_counter() - t
        t = time.perf_counter()
        syn_sign(ds)
        t_syn = time.perf_counter() - t
        out["time"][n] = {"oct_sign_ms": round(t_sign * 1e3, 1), "oct_locate_ms": round(t_loc * 1e3, 1),
                          "syn_sign_ms": round(t_syn * 1e3, 1)}
    out["bytes"] = {"hash chain head": "count + 32", "octonion signature": 32 * 8, "syndrome code": 64,
                    "all record hashes": "32 per record"}
    print(json.dumps(out, indent=2))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
