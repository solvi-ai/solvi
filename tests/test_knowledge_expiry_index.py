"""solvi.core.knowledge.store: the expiry index behind `tick` — a tick re-resolves only the items whose expiry can flip
between the old and the new clock (not every item with reconfirm_after), and the store it gives is the one a full
re-resolution gives, on random histories."""
import random

from solvi.core.knowledge import KnowledgeStore
from solvi.core.knowledge import store as store_mod


def fact(s, o, r="r"):
    return {"s": s, "r": r, "o": o}


class FullTick(KnowledgeStore):
    """The behaviour before the index: a tick re-resolves every item with reconfirm_after."""

    def _flush(self):
        if self._tick_dirty or self.clock != self._resolved_clock:
            self._touched |= {i for i, it in self.items.items() if it.reconfirm_after}
        super()._flush()


def test_a_tick_re_resolves_only_the_items_whose_expiry_can_flip(monkeypatch):
    ks = KnowledgeStore()
    with ks.batch():
        ids = [ks.add("fact", fact(f"x{i}", i), source="person", reconfirm_after=1000 + (i % 7)) for i in range(2000)]
    seen = []
    real = store_mod._resolve
    monkeypatch.setattr(store_mod, "_resolve", lambda items, comp, *a, **k: seen.append(len(comp)) or real(
        items, comp, *a, **k))
    ks.tick(1)
    assert sum(seen) == 0                                  # nothing can expire yet: nothing is re-resolved
    ks.tick(999)                                           # clock 1000: deadlines 1000..1006 are near
    assert 0 < sum(seen) < 2000
    assert all(ks.usable(i) for i in ids)
    ks.tick(1)                                             # clock 1001: the items with reconfirm_after 1000 expire
    assert sum(ks.status(i) == "expired" for i in ids) == sum(1 for i in range(2000) if i % 7 == 0)


def _history(rng, n_ops):
    ops, ids_hint = [], 0
    for _ in range(n_ops):
        u = rng.random()
        if u < 0.45:
            ops.append(("add", rng.randrange(6), rng.randrange(3), rng.choice(["person", "outcome", "spec"]),
                        rng.choice([None, None, 1, 2, 3, 5, 8, 2.5]), rng.random() < 0.3, rng.random() < 0.15))
        elif u < 0.75:
            ops.append(("tick", rng.choice([1, 1, 2, 3, 5, 0])))
        elif u < 0.83:
            ops.append(("retract",))
        elif u < 0.88:
            ops.append(("flag",))
        elif u < 0.91:
            ops.append(("end_flag",))
        elif u < 0.95:
            ops.append(("batch_ticks", rng.randrange(1, 4)))
        else:
            ops.append(("retire",))
        ids_hint += 1
    return ops


def _run(cls, ops, seed):
    rng = random.Random(seed)
    ks, ids, flags, trail = cls(), [], [], []
    for op in ops:
        if op[0] == "add":
            _, s, o, src, ra, derived, batched = op
            kw = {"reconfirm_after": ra} if ra is not None else {}
            if derived and ids:
                kw["derived_from"] = [rng.choice(ids)]
            if batched:
                with ks.batch():
                    i = ks.add("fact", fact(f"s{s}", o), source=src, by="w", **kw)
                    ks.tick(1)
            else:
                i = ks.add("fact", fact(f"s{s}", o), source=src, by="w", **kw)
            if i is not None:
                ids.append(i)
        elif op[0] == "tick":
            ks.tick(op[1])
        elif op[0] == "batch_ticks":
            with ks.batch():
                for _ in range(op[1]):
                    ks.tick(2)
        elif op[0] == "retract" and ids:
            ks.retract(rng.choice(ids), by="p")
        elif op[0] == "retire" and ids:
            ks.retire(rng.choice(ids))
        elif op[0] == "flag" and ids:
            flags.append(ks.reconfirm(rng.choice(ids)))
        elif op[0] == "end_flag" and flags:
            ks.end_flag(flags.pop(rng.randrange(len(flags))))
        trail.append((ks.fingerprint(), {i: ks.status(i) for i in sorted(ks.items)},
                      {i: ks.usable(i) for i in sorted(ks.items)}))
    return ks, trail


def test_the_index_gives_the_state_a_full_re_resolution_gives_on_random_histories():
    expired = 0
    for seed in range(60):
        ops = _history(random.Random(seed), 120)
        new, t_new = _run(KnowledgeStore, ops, seed)
        old, t_old = _run(FullTick, ops, seed)
        assert t_new == t_old, seed                        # after every write, not only at the end
        assert new.state == old.state and new.verify()
        assert new.rebuild().state == new.state
        expired += sum(st == "expired" for st in t_new[-1][1].values())
    assert expired > 0                                     # the histories do exercise expiry
