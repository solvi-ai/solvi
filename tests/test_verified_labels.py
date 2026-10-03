"""The "verified" label source: a System 2 answer that passed the checks and a guarantee is stored as a label
only with the decision it came from, is read by System.guarantee when asked to, and is refused by the channels the
measurement did not admit (a head, a memory of corrections, the learning loop)."""
import random

import pytest

from solvi import Answer, Catalog, JSONLStorage, Question, System
from solvi.storage import TRUSTED_SOURCES, VERIFIED, UntrustedLabel, check_source


def _draw(rng, n, noise=0.15):
    out = []
    for _ in range(n):
        x = rng.random()
        out.append(({"x": x}, (x + rng.gauss(0, noise)) > 0.5))
    return out


def _system(storage=None):
    cat = Catalog()

    @cat.fn
    def score(x: float) -> float:
        return x
    return System(cat, [Question("label", "label?", Answer.yes_no())], storage=storage)


def _guarded(tmp_path, seed=0):
    """A head with a guarantee over a JSONL store: what System 2 is here (any guarded System will do)."""
    rng = random.Random(seed)
    store = JSONLStorage(tmp_path / "s.jsonl")
    s = _system(store)
    s.fit("label", _draw(rng, 300), features=["score"])
    s.guarantee("label", _draw(rng, 300), max_risk=0.05)
    return s, store, rng


def _alone(s, x):
    res = s.ask({"x": x})
    return res, res["label"]


def test_check_source_takes_verified_only_where_a_channel_accepts_it_and_says_why_otherwise():
    for src in TRUSTED_SOURCES:
        assert check_source(src) == src
    assert check_source(VERIFIED, accept=(VERIFIED,)) == VERIFIED
    with pytest.raises(UntrustedLabel, match="memory of corrections fed verified"):
        check_source(VERIFIED, channel="memory")
    with pytest.raises(UntrustedLabel, match="not taken here"):
        check_source(VERIFIED)
    with pytest.raises(UntrustedLabel, match="not trusted"):
        check_source("model", accept=(VERIFIED,))


def test_a_verified_label_is_stored_with_the_decision_it_is_and_teaches_no_head(tmp_path):
    s, store, _ = _guarded(tmp_path)
    res, r = _alone(s, 0.97)
    assert r.status == "ok" and r.extra["guarantee"]["answered"]
    head = s.heads["label"]
    fp = head.fingerprint()
    assert s.teach("label", {"x": 0.97}, r.answer, label_source="verified", by="s2", of=res.stored_id) is None
    assert head.fingerprint() == fp                    # stored only: the head did not learn from it
    c = store.corrections()[-1]
    assert (c["source"], c["of"], c["by"]) == ("verified", res.stored_id, "s2") and store.verify()["ok"]


def test_a_verified_label_needs_a_stored_decision_that_answered_alone_under_a_guarantee_with_that_answer(tmp_path):
    s, store, _ = _guarded(tmp_path)
    res, r = _alone(s, 0.97)
    other = "no" if r.answer == "yes" else "yes"
    with pytest.raises(UntrustedLabel, match="names the System 2 decision"):
        s.teach("label", {"x": 0.97}, r.answer, label_source="verified")
    with pytest.raises(UntrustedLabel, match="not a stored decision"):
        s.teach("label", {"x": 0.97}, r.answer, label_source="verified", of="nope")
    with pytest.raises(UntrustedLabel, match="that decision's own answer"):
        s.teach("label", {"x": 0.97}, other, label_source="verified", of=res.stored_id)
    unsure, u = _alone(s, 0.5)                          # abstains under the guarantee: not verified
    assert u.status != "ok"
    with pytest.raises(UntrustedLabel, match="did not answer 'label' alone"):
        s.teach("label", {"x": 0.5}, "yes", label_source="verified", of=unsure.stored_id)
    s.guarantee("label", False)                         # no guarantee: an answer alone is still not verified
    plain, p = _alone(s, 0.97)
    with pytest.raises(UntrustedLabel, match="without a guarantee"):
        s.teach("label", {"x": 0.97}, p.answer, label_source="verified", of=plain.stored_id)
    assert store.corrections() == []                    # nothing refused was stored


def test_teach_verified_without_storage_raises_instead_of_doing_nothing():
    s = _system()
    s.fit("label", _draw(random.Random(0), 100), features=["score"])
    with pytest.raises(ValueError, match="no storage"):
        s.teach("label", {"x": 0.9}, True, label_source="verified", of="x")


def test_guarantee_reads_the_stored_verified_labels_only_when_its_sources_name_them(tmp_path):
    s, store, rng = _guarded(tmp_path)
    for x in (0.9, 0.95, 0.99, 0.05, 0.02):
        res, r = _alone(s, x)
        s.teach("label", {"x": x}, r.answer, label_source="verified", of=res.stored_id)
    s.teach("label", {"x": 0.3}, False, by="ann")
    base = _draw(rng, 200)
    rep = s.guarantee("label", base, max_risk=0.05, corrections=store)
    assert rep["labels"]["corrections"] == {"human": 1} and rep["labels"]["skipped"] == 5 and rep["n"] == 201
    rep = s.guarantee("label", base, max_risk=0.05, corrections=True, sources=TRUSTED_SOURCES + (VERIFIED,))
    assert rep["labels"]["corrections"] == {"human": 1, "verified": 5} and rep["n"] == 206
    assert len(rep["labels"]["ids"]) == 6 and rep["labels"]["examples"] == 200
    with pytest.raises(UntrustedLabel, match="not label sources"):
        s.guarantee("label", base, max_risk=0.05, corrections=store, sources=("model",))
    with pytest.raises(ValueError, match="give corrections="):
        s.guarantee("label", base, max_risk=0.05, sources=("verified",))


def test_the_memory_and_the_learning_loop_refuse_verified_labels_with_the_measured_reason(tmp_path):
    from solvi.learning import Learning
    from solvi.memory import CorrectionMemory
    with pytest.raises(UntrustedLabel, match="broke its system's promise|memory of corrections fed verified"):
        CorrectionMemory.add(object.__new__(CorrectionMemory), "x", "yes", source="verified")
    s, store, _ = _guarded(tmp_path)
    res, r = _alone(s, 0.97)
    s.teach("label", {"x": 0.97}, r.answer, label_source="verified", of=res.stored_id)
    loop = object.__new__(Learning)
    loop.storage, loop.parts, loop.holdout, loop.calibration = store, {"label": None}, 0.3, 0.2
    got = loop.labels()
    assert got["labels"] == [] and "learning loop's ladder" in got["rejected"][0][1]
