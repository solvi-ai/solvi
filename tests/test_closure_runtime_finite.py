"""Operation-count witness: cached parsing does not eliminate full-prefix integrity reads."""
from solvi.core.store import JSONLStorage


def test_s267_s276_unchanged_incremental_reader_still_validates_entire_prefix(tmp_path, monkeypatch):
    store = JSONLStorage(tmp_path / 'prefix.jsonl')
    for i in range(8):
        store._append({'v': 2, 'kind': 'human', 'payload': 'x' * 1024, 'i': i})
    rows = list(store.iter(None))
    last_seq = rows[-1].seq
    size = store.path and __import__('os').path.getsize(store.path)
    reads = []
    validate = store._validate

    class Counted:
        def __init__(self, fh):
            self.fh = fh
            self.bytes = 0

        def read(self, n=-1):
            data = self.fh.read(n)
            self.bytes += len(data)
            return data

        def seek(self, *args):
            return self.fh.seek(*args)

    def measured(fh):
        wrapped = Counted(fh)
        result = validate(wrapped)
        reads.append(wrapped.bytes)
        return result

    monkeypatch.setattr(store, '_validate', measured)
    for _ in range(3):
        assert list(store.iter(None, after=last_seq)) == []
    assert reads == [size, size, size]
