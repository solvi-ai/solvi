"""S186: a recycled numeric buffer id must never share a previous buffer's scores."""
import builtins
import gc
import pickle
import weakref
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from solvi.experimental.laws import Atom, Buffer, Candidate, Grammar, LawSet, Transition
    from solvi.experimental.laws import _core


def active(state):
    return state["x"]


def buffer(grammar, label):
    b = Buffer(grammar)
    b.add(Transition({"x": True}, "do", label, group=0))
    return b


def test_distinct_buffers_with_colliding_numeric_ids_do_not_share_memo(monkeypatch):
    g = Grammar([Atom("x", active)], k=1)
    ls = LawSet(g, memo=True)
    a, b = buffer(g, "a"), buffer(g, "b")
    c = Candidate("do", ("x",), "a")
    # Deterministic allocator seam: two buffer lifetimes can have this same numeric id and version.
    monkeypatch.setattr(_core, "id", lambda obj: 123 if isinstance(obj, Buffer) else builtins.id(obj), raising=False)
    assert a.version == b.version
    assert ls.score(c, a) == (1, 1, 1)
    assert ls.score(c, b) == (1, 0, 0)
    assert ls.memo_hits == 0
    assert ls.score(c, b) == (1, 0, 0) and ls.memo_hits == 1


def test_memo_does_not_keep_the_old_buffer_alive():
    g = Grammar([Atom("x", active)], k=1)
    ls = LawSet(g, memo=True)
    b = buffer(g, "a")
    ls.score(Candidate("do", ("x",), "a"), b)
    old = weakref.ref(b)
    del b
    gc.collect()
    assert old() is None
    assert ls._memo_buffer() is None


def test_version_changes_and_pickle_clear_the_identity_reference():
    g = Grammar([Atom("x", active)], k=1)
    ls = LawSet(g, memo=True)
    b, c = buffer(g, "a"), Candidate("do", ("x",), "a")
    assert ls.score(c, b) == (1, 1, 1)
    assert ls.score(c, b) == (1, 1, 1) and ls.memo_hits == 1
    b.add(Transition({"x": True}, "do", "b", group=1))
    assert ls.score(c, b) == (2, 1, 1) and ls.memo_hits == 1
    restored = pickle.loads(pickle.dumps(ls))
    assert restored._memo == {} and restored._memo_at is None and restored._memo_buffer is None
    assert restored.score(c, b) == (2, 1, 1)
