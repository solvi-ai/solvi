"""solvi.signature: a signature of a trace / a store names the one changed record and restores its content hash; its
limits are pinned too. Up to 0.7 every test ran for two codes, "syndrome" (the default) and "octonion"; the octonion code
is an experiment in benchmarks/octonion_signature.py since 0.8 (its self-check runs at the end of this file).

Claims, written down before the numbers below were measured (the source: our trace-signature experiment on solver
traces — 7218 derivation trees, 34 356 substitutions of 7 kinds; the positional octonion code, 32 numbers: every kind
detected in 100%, 0 false alarms on the 7218 clean trees; one root located the substituted step and restored the
original record in 100% of value / same-value / operation substitutions; 3.0 ms median per case, the signature 0.5 ms):

  S1  one changed item in a random trace (1..300 items; a random position; random content) is located in 100% of 600
      cases, and its original content hash (28 bytes) restored in 100%; a candidate list finds the original in 100%;
  S2  unchanged traces: 0 false alarms in 600;
  S3  two changed items: detected in 100% of 300, never located at a wrong position (NotLocatable every time);
  S4  a reorder (two different items swapped), a deletion and an insertion in the middle: detected, not located;
      a cut-off tail: detected as missing items; items appended after signing: not covered (the signed prefix verifies);
  S5  a store rewritten after an edit with every hash and the stored head recomputed (the anchor only says "rewritten"):
      verify(signature=...) names the edited record's seq and restores its content hash; a backup copy is matched;
  S6  a trace whose record value was changed and every hash of the chain recomputed: locate names the record (without
      the catalog code that replay needs), repair finds the original value among candidate values;
  S7  speed (a laptop CPU, numpy): sign 1000 items < 100 ms, locate one change in 1000 items < 150 ms (these two bounds
      were set after a first timing of a prototype, not before it).

Measured (this machine, 2026-09-28): S1 600/600 located, 600/600 restored, 600/600 matched; S2 0/600; S3 300/300
detected, 0 located; S4, S5, S6 as claimed; S7 sign 1000 items 14–18 ms, locate 15 ms. benchmarks/trace_signature.py
at a larger scale: 2000/2000 single changes located and restored, 1500/1500 multi-changes detected, 0 wrongly located;
10 000 items: sign 0.19 s, locate 0.16 s. A classical syndrome code (two sums mod a 256-bit prime, 64 bytes) does the
same on these flat lists — 2000/2000, 0 wrong — about 60x faster: on a flat list of records the octonions add nothing
over it; what they add over the hash chain, any single-error-locating code adds. So the syndrome code became the
default, and S1–S7 hold for it too (the whole 32-byte hash restored instead of 28 bytes; S7: sign / locate 1000 items
1.3 / 1.4 ms, ~10x faster than the octonion code with the hashing of the items included).
"""
import hashlib
import json
import os
import random
import time

import pytest
from test_storage import STATES, Clock, _edit_answer, _jsonl_lines, _jsonl_write, _rehash, _sql_bodies, build, make_store

from solvi import System
from solvi.signature import ALGS, NotLocatable, check, extend, load, locate, record_digest, repair, sign

alg = pytest.mark.parametrize("alg", ALGS)


def _restored(alg, digest):
    """What the signature restores of a content hash: all of it."""
    return digest.hex()


def _rand_items(rng, n):
    return [rng.randbytes(rng.randint(1, 60)) for _ in range(n)]


# --- S1, S2: one change located and restored; no false alarms
@alg
def test_single_change_located_and_restored(alg):
    rng = random.Random(1)
    located = restored = matched = alarms = 0
    for _ in range(600):
        its = _rand_items(rng, rng.randint(1, 300))
        sig = load(json.dumps(sign(its, alg)))                              # kept as JSON elsewhere
        alarms += locate(its, sig) is not None
        k = rng.randrange(len(its))
        orig, its[k] = its[k], rng.randbytes(rng.randint(1, 60))
        r = repair(its, sig, candidates=[rng.randbytes(8), orig, rng.randbytes(8)])
        located += r["index"] == k
        restored += r["digest"] == _restored(alg, hashlib.sha256(orig).digest())
        matched += r["match"] == orig
    assert (located, restored, matched, alarms) == (600, 600, 600, 0)


@alg
def test_unchanged_and_empty(alg):
    assert locate([], sign([], alg)) is None
    its = [b"a", b"b", b"c"]
    assert check(its, sign(its, alg))["ok"] and repair(its, sign(its, alg)) is None
    assert sign(its, alg)["alg"] == alg and check(its, sign(its, alg))["alg"] == alg


# --- S3, S4: honest limits
@alg
def test_two_changes_detected_not_located(alg):
    rng = random.Random(2)
    detected = wrong = 0
    for _ in range(300):
        its = _rand_items(rng, rng.randint(2, 300))
        sig = sign(its, alg)
        for k in rng.sample(range(len(its)), 2):
            its[k] = rng.randbytes(20)
        r = check(its, sig)
        detected += not r["ok"]
        wrong += r["index"] is not None
    assert (detected, wrong) == (300, 0)


@alg
def test_reorder_delete_insert_detected_not_located(alg):
    rng = random.Random(3)
    base = _rand_items(rng, 50)
    sig = sign(base, alg)
    swapped = list(base)
    swapped[10], swapped[30] = swapped[30], swapped[10]
    deleted = base[:20] + base[21:]
    inserted = base[:20] + [b"new"] + base[20:]
    for its in (swapped, inserted):
        with pytest.raises(NotLocatable):
            locate(its, sig)
    r = check(deleted, sig)                                  # one item fewer than signed
    assert not r["ok"] and r["index"] is None and "missing" in r["reason"]
    r = check(base[:-5], sig)                                # a cut-off tail
    assert not r["ok"] and "missing" in r["reason"]
    with pytest.raises(NotLocatable):
        locate(base[:20] + base[21:] + [b"filler"], sig)    # a deletion hidden by an append: still not one change


@alg
def test_appended_items_not_covered_and_extend(alg):
    rng = random.Random(4)
    its = _rand_items(rng, 40)
    sig = sign(its, alg)
    more = its + _rand_items(rng, 7)
    assert locate(more, sig) is None                         # the signed prefix is unchanged
    ext, full = extend(sig, more[40:]), sign(more, alg)
    assert ext["count"] == 47 and ext["alg"] == alg
    assert ext["root"] == full["root"]
    assert extend(sign([], alg), more)["root"] == full["root"]
    more[42] = b"edited after signing"
    assert locate(more, sig) is None                         # ... and what came after it is not covered


def test_bad_signature_refused():
    for bad in ({"alg": "other", "count": 1, "root": [0.0] * 32}, {"alg": "syndrome", "count": 1, "root": [0.0] * 32},
                {"alg": "octonion", "count": 1, "root": ["a", "b"]}):
        with pytest.raises(ValueError):
            check([b"a"], bad)
    with pytest.raises(ValueError):
        sign([b"a"], "other")
    assert sign([b"a"])["alg"] == "syndrome" and len(sign([b"a"])["root"]) == 2      # the default: 64 bytes
    assert load(json.dumps(sign([b"a"])))["count"] == 1


# --- S5: a store rewritten after an edit, the stored head included
@alg
@pytest.mark.parametrize("kind", ["jsonl", "sqlite"])
def test_store_rewrite_located_by_signature(alg, kind, tmp_path):
    store = make_store(kind, tmp_path, Clock())
    cat, qs = build()
    s = System(cat, qs, storage=store)
    for st in STATES:
        s.ask(st)
    anchor, sig = store.head(), store.signature(alg)
    assert sig["count"] == 4 and store.verify(signature=sig)["ok"]
    recs = _jsonl_lines(store) if kind == "jsonl" else _sql_bodies(store)
    backup = json.loads(json.dumps(recs[1]))
    _edit_answer(recs[1])
    _rehash(recs, 1)                                         # every hash after the edit recomputed ...
    if kind == "jsonl":
        _jsonl_write(store, recs)
        with open(store.head_path, "w") as fh:               # ... and the stored head
            json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
    else:
        for t in ("records", "answers", "safeguards", "models"):
            store.db.execute(f"DELETE FROM {t}")
        for d in recs:
            store._insert(d)
        store.db.execute("UPDATE meta SET value = ? WHERE key = 'head'",
                         (json.dumps({"count": len(recs), "hash": recs[-1]["hash"]}),))
    assert store.verify()["ok"]                              # consistent inside
    va = store.verify(anchor=anchor)
    assert not va["ok"] and va["problems"][-1][0] == 3       # the anchor: "rewritten", at its own position only
    v = store.verify(signature=sig, candidates=[recs[0], backup])
    assert not v["ok"] and [p[0] for p in v["problems"]] == [1]
    assert v["signature"]["digest"] == _restored(alg, record_digest(backup)) and v["signature"]["match"] is backup


# --- S6: a trace changed consistently (value and every hash recomputed)
@alg
def test_trace_record_located_and_value_restored(alg):
    from solvi.runtime import vhash
    cat, qs = build()
    res = System(cat, qs).ask(STATES[0])
    sig = res.signature(alg)
    assert sig["count"] == 1 + len(res.trace.records) and locate(res, sig) is None
    i = next(k for k, r in enumerate(res.trace.records) if r.name == "risk")
    old = res.trace.records[i].value
    res.trace.records[i].value = "high"
    prev = res.trace.records[i - 1].hash if i else res.trace.init_hash
    for r in res.trace.records[i:]:                          # the chain recomputed: the trace's own hashes agree
        r.prev = prev
        r.hash = vhash(r.body())
        prev = r.hash
    assert locate(res, sig) == i + 1                         # 0 is the input
    assert repair(res, sig, candidates=["high", "medium", old])["match"] == old
    res.trace.init["amount"] = 49                            # a second change: detected, not located
    res.trace.init_hash = vhash(res.trace.init)
    with pytest.raises(NotLocatable):
        locate(res, sig)


# --- the CLI
@alg
def test_cli_verify_signature(alg, tmp_path, capsys):
    from solvi.cli import main
    store = make_store("jsonl", tmp_path, Clock())
    cat, qs = build()
    s = System(cat, qs, storage=store)
    for st in STATES:
        s.ask(st)
    out = tmp_path / "sig.json"
    assert main(["verify", store.path, "--sign", str(out)]) == 0 and load(str(out))["alg"] == alg
    assert main(["verify", store.path, "--signature", str(out)]) == 0
    assert "signature verified" in capsys.readouterr().out
    recs = _jsonl_lines(store)
    _edit_answer(recs[2])
    _rehash(recs, 2)
    _jsonl_write(store, recs)
    with open(store.head_path, "w") as fh:
        json.dump({"count": len(recs), "hash": recs[-1]["hash"]}, fh)
    assert main(["verify", store.path, "--signature", str(out)]) == 1
    text = capsys.readouterr().out
    assert "#2 " in text and "original content hash of #2" in text
    assert main(["verify", store.path, "--signature", str(out), "--sign", str(tmp_path / "no.json")]) == 1
    assert not os.path.exists(tmp_path / "no.json")          # a store that does not verify is never signed


# --- S7: speed
@alg
def test_speed(alg):
    rng = random.Random(5)
    its = _rand_items(rng, 1000)
    t = time.perf_counter()
    sig = sign(its, alg)
    t_sign = time.perf_counter() - t
    its[500] = b"x"
    t = time.perf_counter()
    assert locate(its, sig) == 500
    t_loc = time.perf_counter() - t
    assert t_sign < 0.1 * 5 and t_loc < 0.15 * 5             # the claim with 5x headroom for a loaded CI machine


# --- the octonion code: out of the package since 0.8
def test_the_octonion_code_is_refused_with_a_pointer_to_the_benchmark_that_holds_it(tmp_path):
    assert ALGS == ("syndrome",)
    for call in (lambda: sign([b"a"], "octonion"), lambda: check([b"a"], {"alg": "octonion", "count": 1, "root": [0.0] * 32}),
                 lambda: load({"alg": "octonion", "count": 1, "root": [0.0] * 32})):
        with pytest.raises(ValueError, match="benchmarks/octonion_signature.py"):
            call()


def test_the_octonion_experiment_in_benchmarks_still_locates_one_change():
    import importlib.util
    import pathlib
    path = pathlib.Path(__file__).parent.parent / "benchmarks" / "octonion_signature.py"
    spec = importlib.util.spec_from_file_location("octonion_signature", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._self_check()
