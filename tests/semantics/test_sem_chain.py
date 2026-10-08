"""The hash chain: any edit of any field of any stored record is caught by verify(); what is not damage (the order keys
are written in) is not reported; a rewrite of the whole chain is caught only against an anchor or a signature kept
elsewhere. docs/trace_spec.md §3–§4."""
import copy
import json
import sqlite3

import pytest
from semantics_helpers import (SEEDS, MemoryStorage, fill, leaf_paths, mutated, read_lines, rechain, rng_for,
                               shuffled_keys, system, write_lines)

from solvi.core.store import JSONLStorage, SQLiteStorage, record_hash


def _filled(seed, record="full"):
    st = MemoryStorage(system(), record=record)
    fill(st, system(), rng_for(seed), n=3)
    assert st.verify()["ok"]
    return st


@pytest.mark.parametrize("record", ["full", "compact"])
@pytest.mark.parametrize("seed", range(3))
def test_every_field_of_every_record_edited_alone_fails_verify(seed, record):
    """Exhaustive over the leaves: a changed scalar, or an emptied / filled container, anywhere in any record."""
    st = _filled(seed, record)
    rng = rng_for(seed)
    n = 0
    for i, rec in enumerate(list(st.rows)):
        for path in leaf_paths(rec):
            st.rows[i] = mutated(rec, path, rng)
            out = st.verify()
            assert not out["ok"], (seed, i, path)
            n += 1
        st.rows[i] = rec
    assert st.verify()["ok"] and n > 100


@pytest.mark.parametrize("seed", SEEDS)
def test_a_field_added_or_removed_fails_verify(seed):
    st = _filled(seed)
    rng = rng_for(seed)
    i = rng.randrange(len(st.rows))
    rec = st.rows[i]
    st.rows[i] = {**rec, rng.choice(["note", "extra", "answers2"]): "x"}
    assert not st.verify()["ok"]
    k = rng.choice(sorted(k for k in rec if k not in ("id",)))
    st.rows[i] = {x: v for x, v in rec.items() if x != k}
    assert not st.verify()["ok"], k


@pytest.mark.parametrize("seed", SEEDS)
def test_deleting_inserting_reordering_or_cutting_records_fails_verify(seed):
    rng = rng_for(seed)
    st = _filled(seed)
    rows = list(st.rows)
    i = rng.randrange(len(rows) - 1)
    for edited in (rows[:i] + rows[i + 1:],                          # delete
                   rows[:i] + [rows[i + 1], rows[i]] + rows[i + 2:],  # swap neighbours
                   rows[:i + 1] + [copy.deepcopy(rows[i])] + rows[i + 1:],   # duplicate
                   rows[:-1]):                                     # cut the tail (the stored head still says more)
        st.rows = edited
        assert not st.verify()["ok"]
    st.rows = rows
    assert st.verify()["ok"]


@pytest.mark.parametrize("seed", SEEDS)
def test_the_order_keys_are_written_in_is_not_damage(seed, tmp_path):
    s = system()
    st = JSONLStorage(tmp_path / "s.jsonl", system=s)
    fill(st, s, rng_for(seed), n=3)
    rng = rng_for(seed)
    recs = [shuffled_keys(r, rng) for r in read_lines(st.path)]
    write_lines(st.path, recs)
    again = JSONLStorage(st.path, system=s)
    assert again.verify()["ok"] and again.replay_all(s) == []


@pytest.mark.parametrize("seed", SEEDS)
def test_a_rewrite_of_the_whole_chain_is_caught_only_by_an_anchor_or_a_signature(seed, tmp_path):
    """The limit of a chain: edit a record and recompute every hash and the stored head — verify() alone says ok; a
    head (anchor) or a signature taken before and kept elsewhere catches it, and the signature names the record."""
    s = system()
    rng = rng_for(seed)
    st = JSONLStorage(tmp_path / "s.jsonl", system=s)
    fill(st, s, rng, n=3)
    anchor, sig = st.head(), st.signature()
    recs = read_lines(st.path)
    last = max(k for k, r in enumerate(recs) if r["kind"] == "redaction")
    i = rng.choice(range(last + 1, len(recs)))          # before it, the rewrite moves the erased record's hash, which the
    recs[i]["time"] += 1.0                              # redaction record names: two signed items change (detected only)
    write_lines(st.path, recs, rechain(recs))
    st = JSONLStorage(st.path, system=s)
    assert st.verify()["ok"]
    assert not st.verify(anchor=anchor)["ok"]
    out = st.verify(signature=sig)
    assert not out["ok"] and out["signature"]["index"] == i


@pytest.mark.parametrize("seed", SEEDS)
def test_a_record_cannot_claim_an_older_format(seed):
    st = _filled(seed)
    rng = rng_for(seed)
    i = rng.randrange(1, len(st.rows))
    rec = dict(st.rows[i], v=1)
    rec["hash"] = record_hash(rec)
    st.rows[i] = rec
    assert any("format 1 after a record of format 2" in p[2] for p in st.verify()["problems"])


@pytest.mark.parametrize("seed", range(4))
def test_sqlite_body_and_index_edits_fail_verify(seed, tmp_path):
    s = system()
    rng = rng_for(seed)
    st = SQLiteStorage(tmp_path / "s.db", system=s)
    fill(st, s, rng, n=3)
    st.close()
    con = sqlite3.connect(tmp_path / "s.db")
    seq, body = rng.choice(con.execute("SELECT seq, body FROM records").fetchall())
    d = json.loads(body)
    d["time"] = d["time"] + 1
    con.execute("UPDATE records SET body=? WHERE seq=?", (json.dumps(d), seq))
    con.commit()
    st = SQLiteStorage(tmp_path / "s.db", system=s)
    assert not st.verify()["ok"]
    st.close()
    con.execute("UPDATE records SET body=? WHERE seq=?", (body, seq))
    con.execute("UPDATE answers SET answer='\"forged\"' WHERE seq=(SELECT MIN(seq) FROM answers)")
    con.commit()
    con.close()
    st = SQLiteStorage(tmp_path / "s.db", system=s)
    out = st.verify()
    st.close()
    assert not out["ok"] and any("answers index differs" in p[2] for p in out["problems"])
