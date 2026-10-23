"""KnowledgeStore.forget(storage=) takes a path as System(storage=) does, and opens it before anything is written; the
Knowledge facade has forget()."""
import pytest

import solvi


def _km():
    km = solvi.Knowledge(writers={"scraper": "outcome"})
    km.tell({"s": "a", "r": "b", "o": 1}, source="outcome", by="scraper")
    return km


def test_forget_with_a_storage_path(tmp_path):
    km = _km()
    out = km.store.forget("scraper", storage=str(tmp_path / "decisions.jsonl"), decide=lambda s: {"q": 1})
    assert out["source"] == "scraper" and list(out["retracted"].values()) == [("active", "retracted")]
    assert out["changed"] == [] and out["same"] == []


def test_a_storage_that_cannot_be_opened_leaves_the_journal_as_it_was():
    km = _km()
    n = len(km.store)
    with pytest.raises(TypeError):
        km.store.forget("scraper", storage=12345, decide=lambda s: {"q": 1})
    assert len(km.store) == n
    assert km.store.find(status="active")


def test_the_knowledge_facade_forgets_a_source(tmp_path):
    km = _km()
    out = km.forget("scraper", why="exposed", storage=tmp_path / "d.jsonl", decide=lambda s: {"q": 1})
    assert list(out["retracted"].values()) == [("active", "retracted")]
    assert not km.store.find(status="active")
