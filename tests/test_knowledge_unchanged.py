"""Stores without overlapping `valid` behave as before the effective-interval change: random journals over every write
(facts and rules, chains of derived_from, supersedes, inferred, consulted, retract, only_from, retract_source, retest,
flag / end_flag, resolve, tick, rollback, forget) give the same journal bytes, fingerprint, report, counts and
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
