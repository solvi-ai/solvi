from solvi.core.knowledge import KnowledgeStore


def add(s, name="p", value=1, **kw):
    return s.add("fact", {"s": name, "r": "v", "o": value}, source="outcome", **kw)


def test_repeated_reads_compute_once_and_public_mutation_is_detached(monkeypatch):
    s = KnowledgeStore()
    p = add(s, valid=(0, 10))
    d = add(s, "d", derived_from=[p], valid=(0, 20))
    calls = []
    old = s._compute_periods

    def count(*a, **k):
        calls.append(a[0])
        return old(*a, **k)

    monkeypatch.setattr(s, "_compute_periods", count)
    first = s.periods(d)
    assert first == [[0, 10]]
    n = len(calls)
    first[0][1] = 999
    first.append([999, None])
    for _ in range(30):
        assert s.periods(d) == [[0, 10]]
        assert s.usable(d, valid_at=5) and not s.usable(d, valid_at=15)
    assert len(calls) == n
    assert all(isinstance(iv, tuple) for iv in s._period_cache[d][1])
    assert s.periods("missing") is None and "missing" not in s._period_cache


def test_premise_writes_closed_versions_batch_tick_flags_and_rebuild():
    s = KnowledgeStore()
    p = add(s, valid=(0, None))
    d = add(s, "d", derived_from=[p])
    assert s.periods(d) == [[0, None]]
    cache = s._period_cache
    epoch = s._period_epoch
    add(s, value=2, valid=(10, None))
    assert s._period_cache is cache and s._period_epoch > epoch
    assert s.periods(d) == [[0, 10]] and not s.usable(d, valid_at=15)
    with s.batch():
        add(s, value=3, valid=(5, 8))
        assert s.periods(d) == [[0, 5], [8, 10]]
    s.flag(ids=[p])
    assert not s.usable(d, valid_at=1)
    assert s.verify() and s.rebuild().periods(d) == s.periods(d)
    assert s.branch().periods(d) == s.periods(d)
    s.retract(p)
    assert not s.usable(d, valid_at=1)
    assert s.verify()


def test_seen_context_never_populates_global_cache():
    s = KnowledgeStore()
    p = add(s, valid=(0, 10))
    d = add(s, "d", derived_from=[p])
    s._periods(d, {p}, {})  # seen-dependent computation must never be promoted globally
    assert d not in s._period_cache and p not in s._period_cache
    assert s.periods(d) == [[0, 10]]
    before = dict(s._period_cache)
    assert s._periods(p, set(), {p: [[20, 30]]}) == [[20, 30]]
    assert s._period_cache == before


def test_expiry_tick_and_missing_periods_preserved():
    s = KnowledgeStore()
    p = add(s, reconfirm_after=1)
    assert s.periods(p) is None and s.usable(p, valid_at=10)
    s.tick(2)
    assert not s.usable(p, valid_at=10)
    assert s.verify() and s.rebuild().periods(p) is None


def test_warm_root_cache_preserves_cyclic_shared_snapshot_memo():
    import warnings

    class Reference(KnowledgeStore):
        def _periods(self, iid, seen, memo=None, **kw):
            return self._compute_periods(iid, seen, {} if kw.get("_isolated") else memo)

    def make(cls):
        s = cls()
        p = add(s, "P", valid=(0, 1))
        a = s.add("fact", {"s": "A", "r": "v", "o": 1}, source="spec", valid=(0, 10), derived_from=[p], inferred=True)
        b = s.add("fact", {"s": "B", "r": "v", "o": 1}, source="spec", valid=(5, 15), derived_from=[a], inferred=True)
        s.add("fact", {"s": "A", "r": "v", "o": 1}, source="spec", valid=(0, 10), derived_from=[b], inferred=True)
        return s, a, b

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast, a, b = make(KnowledgeStore)
        reference, ra, rb = make(Reference)
    assert fast.periods(a) == reference.periods(ra)
    assert fast.periods(b) == reference.periods(rb)
    assert fast.snapshot() == reference.snapshot()
    local = {}
    fast._periods(a, set(), local)
    assert a in local  # even initially empty caller memo must be populated
    assert fast.rebuild().snapshot() == reference.rebuild().snapshot()
