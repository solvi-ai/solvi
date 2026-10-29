"""Dispatcher(secret=): the supervision and audit draws keyed by HMAC-SHA256 with a secret (or SOLVI_AUDIT_SECRET), so
a caller who can vary the inputs cannot predict which decisions get checked. The secret enters no record, fingerprint
or repr; replay takes the recorded draw and needs the secret only to recompute it. Without a secret: as before, bit
for bit, with a one-time PredictableDrawWarning when something is sampled."""
import hashlib
import json
import warnings

import pytest
from test_dispatch import _hard_slice

import solvi.core.dispatch as dispatch
from solvi.core.dispatch import AskPath, Dispatcher, PredictableDrawWarning
from solvi.core.store import JSONLStorage

SECRET = "s3cr3t-Kx9-never-in-a-record"


@pytest.fixture(autouse=True)
def _no_env_secret(monkeypatch):
    monkeypatch.delenv(dispatch.SECRET_ENV, raising=False)


def _quiet(*a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PredictableDrawWarning)
        return Dispatcher(*a, **kw)


def _draws(d, data):
    out = [d.ask(st) for st, _ in data]
    return [r.draw for r in out], [r.audit for r in out], out


def test_keyed_draws_differ_from_the_seed_only_ones_and_repeat_for_the_same_secret(monkeypatch):
    s1, s2, data = _hard_slice(n=60)
    plain = _draws(_quiet(s1, AskPath(s2), supervise=0.3, audit=0.3), data)
    keyed = _draws(Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, secret=SECRET), data)
    again = _draws(Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, secret=SECRET.encode()), data)
    other = _draws(Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, secret="another"), data)
    monkeypatch.setenv(dispatch.SECRET_ENV, SECRET)
    env = _draws(Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3), data)
    assert keyed[:2] == again[:2] == env[:2]                         # str, bytes and the environment: one key
    assert all(a != b for a, b in zip(keyed[0], plain[0])) and keyed[1] != plain[1]
    assert keyed[0] != other[0]
    assert all(r.keyed for r in keyed[2]) and not any(r.keyed for r in plain[2])
    with pytest.raises(TypeError, match="secret"):
        Dispatcher(s1, AskPath(s2), secret="")


def test_without_a_secret_the_draws_and_records_are_as_before():
    s1, s2, data = _hard_slice(n=20)
    d = _quiet(s1, AskPath(s2), supervise=0.3, audit=0.3, seed=5)
    for st, _ in data:
        r = d.ask(st)
        h = r.s1.trace.init_hash
        assert r.draw == int(hashlib.sha256(f"5|{h}|{r.n}".encode()).hexdigest()[:13], 16) / 16 ** 13
        assert r.audit == (int(hashlib.sha256(f"audit|5|{h}|{r.n}".encode()).hexdigest()[:13], 16) / 16 ** 13 < 0.3)
        assert "draw_keyed" not in r.to_dict() and not r.keyed
    assert d.config() == Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, seed=5, secret=SECRET).config()


def test_the_secret_leaks_into_no_record_fingerprint_or_repr(tmp_path):
    s1, s2, data = _hard_slice(n=40)
    store = JSONLStorage(tmp_path / "keyed.jsonl")
    d = Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, secret=SECRET, storage=store)
    out = [d.ask(st) for st, _ in data]
    texts = [repr(d), repr(vars(d)), d.config(), str(d.summary()), open(store.path).read()]
    texts += [r.to_json() + repr(r) for r in out]
    for t in texts:
        assert SECRET not in t and SECRET.encode().hex() not in t
    assert all(json.loads(r.to_json())["draw_keyed"] is True for r in out)


def test_replay_needs_no_secret_and_a_wrong_secret_is_caught(tmp_path):
    s1, s2, data = _hard_slice(n=40)
    store = JSONLStorage(tmp_path / "keyed.jsonl")
    d = Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, secret=SECRET, storage=store)
    for st, _ in data:
        d.ask(st)
    assert d.replay_all() == []
    blind = _quiet(s1, AskPath(s2), supervise=0.3, audit=0.3, storage=store)
    assert blind.stored() and blind.replay_all() == []                # the recorded draw is taken
    wrong = Dispatcher(s1, AskPath(s2), supervise=0.3, audit=0.3, secret="not it", storage=store)
    bad = wrong.replay_all()
    assert bad and all(w == "draw" for _, _, ms in bad for w, _ in ms)
    with pytest.raises(ValueError, match="secret"):
        blind.draw("x", 1, keyed=True)


def test_the_predictable_draw_warning_is_given_once_and_only_when_something_is_sampled(monkeypatch):
    s1, s2, _ = _hard_slice(n=5)
    monkeypatch.setattr(dispatch, "_predictable_warned", False)
    with warnings.catch_warnings(record=True) as got:
        warnings.simplefilter("always")
        Dispatcher(s1, AskPath(s2))                                   # nothing sampled: silent
        Dispatcher(s1, AskPath(s2), supervise=1.0)                    # every answer checked: nothing to predict
        Dispatcher(s1, AskPath(s2), audit=0.1, secret=SECRET)         # keyed: silent
        Dispatcher(s1, AskPath(s2), supervise=0.1)
        Dispatcher(s1, AskPath(s2), audit=0.1)                        # once per process
    assert [w.category for w in got] == [PredictableDrawWarning]
