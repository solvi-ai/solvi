"""TraceStorage.provenance_index(): quarantine / where_is over the records that can match give the store's answers."""
import random
import warnings

from solvi.core.store import JSONLStorage, SQLiteStorage

from test_diff_incremental import _states, build

FACTS = ["text", "amount", "customer", "words", "risk", "size", "known_customer", "nothing"]


def _fill(store, seed, n=30):
    rng = random.Random(seed)
    s = build()
    s.storage = store
    store.catalog = s
    ids = []
    for st in _states(seed, n):
        r = s.ask(st, rng.choice([None, ["approve"], ["big"]]))
        ids.append(r.stored_id)
        if rng.random() < 0.2:
            store.save_correction("approve", st, rng.random() < 0.5)
    return s, ids


def _check(store, index):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for f in FACTS:
            assert index.quarantine(f) == store.quarantine(f)
            assert index.where_is(f) == store.where_is(f)
        for v in ["ann", "blocked", 50, 500, "pay", "nobody"]:
            for f in ("customer", "amount", "text"):
                assert index.quarantine(f, v) == store.quarantine(f, v)
                assert index.where_is(f, v) == store.where_is(f, v)


def test_index_answers_as_the_store(tmp_path):
    for k, make in enumerate([lambda: JSONLStorage(tmp_path / "a.jsonl"), lambda: SQLiteStorage(tmp_path / "a.db"),
                              lambda: JSONLStorage(tmp_path / "c.jsonl", record="sample:3")]):
        store = make()
        s, ids = _fill(store, k)
        index = store.provenance_index()
        _check(store, index)
        assert len(index.reads["customer"]) < len(ids)          # only the decisions that read it are candidates
        store.redact(ids[0])                                    # a redacted record is passed over, as the store does
        for st in _states(k + 10, 5):                           # later records are read on the next query
            s.ask(st)
        _check(store, index)
