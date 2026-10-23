"""Stores without overlapping `valid` behave as before the effective-interval change, and the three shapes the change
admits (A-B-A, a derived item over premises with `valid`, S231 on a plain key) differ exactly as listed. Random
journals over every write (facts and rules, chains of derived_from, supersedes, inferred, consulted, retract,
only_from, retract_source, retest, flag / end_flag, resolve, tick, rollback, forget) give the same journal bytes, fingerprint, report, counts and
snapshots as the store before the change (with `valid`: disjoint intervals, one period per content) (golden values in knowledge_unchanged.json, written by it), and journals it
wrote load, verify and replay.

Regenerate the golden file only with the store from before the change on the path:
    PYTHONPATH=<old src> python tests/test_knowledge_unchanged.py"""
import json
import random
import shutil
from pathlib import Path

from solvi.core.knowledge import KnowledgeStore
from solvi.core.runtime import vhash

GOLDEN = Path(__file__).with_name("knowledge_unchanged.json")
FIXTURES = Path(__file__).parent / "fixtures" / "knowledge_1_3"
OLD_JOURNALS = ("plain-3", "plain-7", "plain-11", "disjoint-2")
N_PLAIN, N_DISJOINT = 40, 10
WRITERS = ("ann", "bob", "crm", "lab")


def _journal(seed, disjoint=False, storage=None):
    """A random journal (a store built by random writes); disjoint: facts carry `valid` on disjoint intervals only."""
    rng = random.Random(seed)
    ks = KnowledgeStore(storage)
    ids = []
    keys = [(f"s{k}", f"r{k % 2}") for k in range(4)]
    slots = {}                                          # disjoint: key → the next free since
    for _ in range(rng.randint(15, 45)):
        op = rng.random()
        try:
            if op < 0.45 or not ids:
                s, r = rng.choice(keys)
                kind = "rule" if (not disjoint and rng.random() < 0.15) else "fact"
                kw = {"source": rng.choice(("person", "outcome", "spec")), "by": rng.choice(WRITERS)}
                if disjoint:
                    since = slots.get((s, r), 0)
                    until = since + rng.randint(1, 5)
                    slots[(s, r)] = until + rng.randint(0, 2)
                    kw["valid"] = (since, until)
                elif ids and rng.random() < 0.3:
                    prem = rng.sample(ids, min(len(ids), rng.randint(1, 2)))
                    kw["derived_from"] = prem
                    kw["inferred"] = rng.random() < 0.4
                    if rng.random() < 0.2:
                        kw["consulted"] = [rng.choice(ids)]
                    if kind == "fact" and kw["source"] == "outcome" and not kw["inferred"]:
                        kw["source"] = "spec"
                elif ids and rng.random() < 0.15:
                    kw["supersedes"] = rng.choice(ids)
                o = since if disjoint else rng.randint(0, 2)   # disjoint: one period per content (one told for
                i = ks.add(kind, {"s": s, "r": r, "o": o}, None, **kw)     # two periods now holds in both: A-B-A)
                if i is not None and i not in ids:
                    ids.append(i)
            elif op < 0.55:
                ks.retract(rng.choice(ids), by="rev", why="wrong")
            elif op < 0.62 and not disjoint:
                i = rng.choice(ids)
                ws = sorted({a["by"] for a in ks.items[i].asserts})
                ks.retract(i, only_from=rng.choice(ws))
            elif op < 0.66 and not disjoint:
                ks.retract_source(rng.choice(WRITERS), by="audit")
            elif op < 0.74:
                pend = [i for i in ids if ks.pending_retest(i)]
                if pend:
                    ks.retest(rng.choice(pend), ok=rng.random() < 0.7, by="qa")
            elif op < 0.79:
                f = ks.flag(ids=[rng.choice(ids)], why="drift") if rng.random() < 0.5 else \
                    ks.flag(key=list(ks.items[rng.choice(ids)].key or ["", "s0", "r0"]), why="drift")
                if rng.random() < 0.5:
                    ks.end_flag(f, by="qa")
            elif op < 0.85:
                d = ks.disputes()
                if d:
                    q = rng.choice(d)
                    ks.resolve(q["key"], ks.items[rng.choice(q["items"])].value, by="person")
            elif op < 0.9:
                ks.tick(rng.randint(1, 3000))
            elif op < 0.95 and not disjoint:
                ks.rollback(rng.choice(ids), by="sup", why="undo")
            elif not disjoint:
                ks.forget(rng.choice(WRITERS), by="dpo")
        except (KeyError, ValueError):
            pass
    return ks


def _observe(ks):
    keys = sorted({it.key for it in ks.items.values() if it.key is not None})
    snaps = {json.dumps(k): ks.snapshot(about=k[1]) for k in keys}
    return {"journal": vhash(ks.journal), "fp": ks.fingerprint(), "report": vhash(ks.report()),
            "counts": ks.counts(), "snapshots": vhash(snaps), "statuses": {i: ks.status(i) for i in sorted(ks.items)}}


def _all():
    out = {}
    for seed in range(N_PLAIN):
        out[f"plain-{seed}"] = _observe(_journal(seed))
    for seed in range(N_DISJOINT):
        out[f"disjoint-{seed}"] = _observe(_journal(1000 + seed, disjoint=True))
    return out


def test_stores_without_overlapping_valid_are_byte_identical_to_the_store_before_the_change():
    golden = json.loads(GOLDEN.read_text())
    now = _all()
    assert sorted(now) == sorted(golden)
    diff = [k for k in sorted(golden) if now[k] != golden[k]]
    assert diff == [], diff


def test_journals_written_before_the_change_load_verify_and_replay(tmp_path):
    golden = json.loads(GOLDEN.read_text())
    for name in OLD_JOURNALS:
        path = tmp_path / f"{name}.jsonl"
        shutil.copy(FIXTURES / f"{name}.jsonl", path)
        shutil.copy(FIXTURES / f"{name}.jsonl.head", f"{path}.head")
        ks = KnowledgeStore(str(path))
        assert ks.verify() and ks.fingerprint() == golden[name]["fp"]
        assert ks.rebuild().fingerprint() == golden[name]["fp"]
        assert {i: ks.status(i) for i in sorted(ks.items)} == golden[name]["statuses"]


def test_the_three_admitted_shapes_differ_from_the_store_before_the_change_exactly_as_listed():
    """C1 is "identical or explained": the random journals above never produce the three shapes the change admits, so
    each is written here with the store-before-the-change's values (in the comments) and its one expected difference."""
    # A-B-A: one value told for two periods. Before: the row carried only the latest `valid` [9, 14] (the value was
    # not a fact at 2); now the row also carries "periods". Statuses, the fingerprint and the read at 7 are the same.
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "s0", "r": "r0", "o": 12}, source="outcome", by="crm", valid=(0, 5))
    b = ks.add("fact", {"s": "s0", "r": "r0", "o": 24}, source="outcome", by="crm", valid=(5, 9))
    ks.add("fact", {"s": "s0", "r": "r0", "o": 12}, source="outcome", by="crm", valid=(9, 14))
    rows = {r["id"]: r for r in ks.snapshot()["facts"]}
    assert ks.status(a) == ks.status(b) == "active" and ks.fingerprint().startswith("860b12da2aef")
    assert rows[a]["valid"] == [9, 14] and rows[a]["periods"] == [[0, 5], [9, 14]]
    assert rows[b] == {"id": b, "kind": "fact", "body": {"s": "s0", "r": "r0", "o": 24}, "source": "outcome",
                       "valid": [5, 9]}
    assert [r["body"]["o"] for r in ks.snapshot(about="s0", valid_at=7)["facts"]] == [24]
    # A derived item over premises with `valid`. Before: its "assert" entry had valid None and its row no "valid";
    # now the entry and the row carry the intersection [4, 10]. Its status and the fingerprint are the same.
    ks = KnowledgeStore()
    p = ks.add("fact", {"s": "s0", "r": "r0", "o": 1}, source="outcome", by="crm", valid=(0, 10))
    q = ks.add("fact", {"s": "s1", "r": "r1", "o": 2}, source="outcome", by="lab", valid=(4, 20))
    d = ks.add("fact", {"s": "s2", "r": "r0", "o": 3}, source="spec", by="rule", derived_from=[p, q])
    assert [r.get("valid") for r in ks.journal if r.get("item") == d] == [[4, 10]]
    assert ks.status(d) == "active" and ks.fingerprint().startswith("51b4397b3af2")
    assert {r["id"]: r.get("valid") for r in ks.snapshot()["facts"]}[d] == [4, 10]
    # S231 on a plain key: an item a rule only re-infers follows its premise's passed re-test. Before: d stayed a
    # hypothesis "pending retest" (fingerprint b12854ad1810…); now it is active, and the fingerprint changes.
    ks = KnowledgeStore()
    a = ks.add("fact", {"s": "s0", "r": "r0", "o": 1}, source="outcome", by="crm")
    d = ks.add("fact", {"s": "s1", "r": "r1", "o": 1}, source="spec", by="rule", derived_from=[a], inferred=True)
    ks.retract(a, by="crm")
    ks.add("fact", {"s": "s0", "r": "r0", "o": 1}, source="outcome", by="sync")
    ks.retest(a, ok=True)
    assert ks.status(d) == "active" and not ks.pending_retest(d)
    assert ks.fingerprint().startswith("ea8fb3057797") and not ks.fingerprint().startswith("b12854ad1810")
    assert sorted(r["id"] for r in ks.snapshot()["facts"]) == sorted([a, d]) and ks.snapshot()["hints"] == []


def _write_golden():
    GOLDEN.write_text(json.dumps(_all(), indent=1, sort_keys=True) + "\n")
    FIXTURES.mkdir(exist_ok=True)
    for name in OLD_JOURNALS:                          # the same writes on a file journal
        kind, seed = name.split("-")
        path = FIXTURES / f"{name}.jsonl"
        if path.exists():
            path.unlink()
        _journal(int(seed) + (1000 if kind == "disjoint" else 0), kind == "disjoint", storage=str(path))
    print("wrote", GOLDEN, FIXTURES)


if __name__ == "__main__":
    _write_golden()
