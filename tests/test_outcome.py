"""System.outcome: what really happened after a decision, stored as an "outcome" label of it (the same source check as
every label) that nothing learns from by itself; an explicit recalibration reads it (System.guarantee(...,
corrections=True)). Recalibration on the fly lives in the experimental solvi.oncalib, which warns on import."""
import importlib
import random
import sys

import pytest

from solvi import Answer, Catalog, JSONLStorage, Question, System


def _draw(rng, n, noise=0.15):
    out = []
    for _ in range(n):
        x = rng.random()
        out.append(({"x": x}, (x + rng.gauss(0, noise)) > 0.5))
    return out


def _guarded(tmp_path, record="full"):
    rng = random.Random(0)
    cat = Catalog()

    @cat.fn
    def score(x: float) -> float:
        return x
    store = JSONLStorage(tmp_path / "s.jsonl", record=record)
    s = System(cat, [Question("label", "label?", Answer.yes_no())], storage=store)
    s.fit("label", _draw(rng, 300), features=["score"])
    s.guarantee("label", _draw(rng, 300), max_risk=0.05)
    return s, store, rng


@pytest.mark.parametrize("record", ["full", "compact"])
def test_an_outcome_is_stored_as_a_label_of_its_decision_and_teaches_nothing_by_itself(tmp_path, record):
    s, store, _ = _guarded(tmp_path, record)
    res = s.ask({"x": 0.8})
    head = s.heads["label"]
    fp, threshold = head.fingerprint(), s.guards["label"].promise.report["threshold"]
    cid = s.outcome(res, False, note="the parcel was lost", by="carrier")
    assert head.fingerprint() == fp and s.guards["label"].promise.report["threshold"] == threshold
    c = store.corrections()[-1]
    assert c["id"] == cid and (c["source"], c["of"], c["by"], c["answer"]) == ("outcome", res.stored_id, "carrier", "no")
    assert c["init"] == {"x": 0.8} and c["note"] == "the parcel was lost" and store.verify()["ok"]
    assert s.outcome(res.stored_id, "yes")                                  # by its stored id too
    rep = s.report()
    assert rep.corrections["by_source"] == {"outcome": 2} and rep.corrections["labelled_decisions"] == 1


def test_outcome_refuses_what_it_cannot_store(tmp_path):
    s, store, _ = _guarded(tmp_path)
    res = s.ask({"x": 0.3})
    with pytest.raises(ValueError, match="not one of|options"):
        s.outcome(res, "maybe")
    with pytest.raises(KeyError):
        s.outcome(res, True, question="nope")
    with pytest.raises(KeyError):
        s.outcome("no-such-id", True)
    unstored = s.ask({"x": 0.3}, store=False)
    with pytest.raises(ValueError, match="stored id"):
        s.outcome(unstored, True)
    s.storage = None
    with pytest.raises(ValueError, match="storage"):
        s.outcome(res, True)
    assert store.corrections() == []


def test_explicit_recalibration_reads_the_outcomes(tmp_path):
    s, store, rng = _guarded(tmp_path)
    for st, y in _draw(rng, 40):
        s.outcome(s.ask(st), y)
    rep = s.guarantee("label", _draw(rng, 100), max_risk=0.05, corrections=True)
    assert rep["labels"]["corrections"] == {"outcome": 40} and rep["n"] == 140


def test_oncalib_is_experimental_and_recalibrates_every_n_outcomes_and_after_a_drift_flag(tmp_path):
    from solvi.core import ExperimentalWarning
    sys.modules.pop("solvi.oncalib", None)
    with pytest.warns(ExperimentalWarning, match="does not keep a guarantee's promise"):
        oncalib = importlib.import_module("solvi.oncalib")
    assert "RISK" in oncalib.__doc__
    s, store, rng = _guarded(tmp_path)
    live = oncalib.OnTheFly(s, "label", every=20, window=30, min_labels=10, max_risk=0.10)
    for st, y in _draw(rng, 45):
        live.outcome(s.ask(st), y)
    assert len(live.history) == 2 and len(live.history[-1]["labels"]) == 30 and live.new == 5
    assert live.drifted() is None                                           # no labels since the flag yet
    for st, y in _draw(rng, 12):
        live.outcome(s.ask(st), y)
    assert live.recalibrate()["n"] == 12
    with pytest.raises(ValueError, match="no examples="):
        oncalib.OnTheFly(s, "label", examples=[])
