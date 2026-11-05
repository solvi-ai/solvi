"""Exact pre-optimization oracle, including ordered stored JSON and legacy records."""
import copy
import hashlib

import pytest

from solvi.core import store


def reference(rec):
    if int(rec.get('v') or 1) < 2:
        return {k: v for k, v in rec.items() if k not in ('id', 'hash')}
    mark = rec.get('redacted') if isinstance(rec.get('redacted'), dict) else {}
    body = {k: rec[k] for k in store.KEPT if k in rec}
    body['#answers'] = mark['answers'] if 'answers' in mark else store._digest(rec.get('answers'))
    body['#content'] = mark['content'] if 'content' in mark else store._digest(
        {k: v for k, v in rec.items() if k not in store.KEPT and k not in ('answers', 'id', 'hash', 'redacted')})
    return body


@pytest.mark.parametrize('version', [1, 2])
@pytest.mark.parametrize('kind', ['ask', 'teach', 'dispatch', 'process', 'redaction'])
@pytest.mark.parametrize('redacted', [None, {}, {'answers': 'a'}, {'content': 'c'}, {'answers': 'a', 'content': 'c'}])
def test_exact_body_hash_and_nonmutation(version, kind, redacted):
    rec = {'v': version, 'kind': kind, 'seq': 4, 'time': 1., 'prev': 'previous',
           'answers': {'q': [1, .9, 'ok']}, 'meta': {'unicode': 'метка', 2: 'two'},
           'signature': 'signed', 'record': 'compact', 'compact': {'trace': []},
           'legacy_nonfinite': float('inf'), 'redacted': redacted, 'id': 'ignored', 'hash': 'ignored'}
    before = store._cj(rec)
    expected = reference(rec)
    assert store.record_body(rec) == expected
    assert store._cj(store.record_body(rec)) == store._cj(expected)
    assert store.record_hash(rec) == hashlib.sha256(store._cj(expected).encode()).hexdigest()
    assert store._cj(rec) == before


def test_frozen_clock_jsonl_bytes_and_dynamic_kept(tmp_path, monkeypatch):
    bodies = [{'v': 2, 'kind': k, 'answers': {'q': i}, 'meta': {'x': i}}
              for i, k in enumerate(['note', 'teach', 'process'])]
    candidate = store.record_body
    outputs = []
    for name, fn in [('old', reference), ('new', candidate)]:
        monkeypatch.setattr(store, 'record_body', fn)
        path = tmp_path / name
        s = store.JSONLStorage(path, clock=lambda: 42.)
        for body in bodies:
            s._append(copy.deepcopy(body))
        assert s.verify()['ok'], str(s.verify())
        outputs.append(path.read_bytes())
    assert outputs[0] == outputs[1]
    monkeypatch.setattr(store, 'KEPT', store.KEPT + ('signature',))
    assert candidate(bodies[0]) == reference(bodies[0])


def test_invalid_key_errors_remain():
    rec = {'v': 2, 'meta': {object(): 'invalid'}}
    for fn in (reference, store.record_body):
        with pytest.raises(TypeError):
            fn(rec)
