"""Dispatcher(audit=rate): a seeded random share of the decisions is marked for labelling in its record; a label of a
marked decision is stored with label_source="audit" (refused for an unmarked one); corrections(label_source=) filters;
the system report measures the promise on the audited sample. Default: off, records and config unchanged."""
import json

import pytest
from test_dispatch import _hard_slice

from solvi.core.dispatch import AskPath, Dispatcher
from solvi.core.sources import UntrustedLabel
from solvi.core.store import JSONLStorage, open_storage
from solvi.core.store.sysreport import system_report


def _run(tmp_path, audit, n=1200):
    s1, s2, data = _hard_slice(guess_right=0.6, slow_right=0.95, n=n)
    store = JSONLStorage(tmp_path / f"d{audit}.jsonl")
    d = Dispatcher(s1, AskPath(s2), storage=store, audit=audit)
    out = [(d.ask(st), y) for st, y in data]
    return d, store, out


def test_off_by_default_leaves_records_and_config_as_before(tmp_path):
    s1, s2, data = _hard_slice(n=20)
    a = Dispatcher(s1, AskPath(s2))
    b = Dispatcher(s1, AskPath(s2), audit=0.0)
    assert a.config() == b.config() and a.audit == 0.0
    r = a.ask(data[0][0])
    assert r.audit is False and "audit" not in r.to_dict()
    with pytest.raises(ValueError, match="audit"):
        Dispatcher(s1, AskPath(s2), audit=1.5)
    assert Dispatcher(s1, AskPath(s2), audit=0.1).config() != a.config()


def test_a_seeded_share_is_marked_and_only_a_marked_decision_takes_an_audit_label(tmp_path):
    d, store, out = _run(tmp_path, 0.2, n=600)
    marked = [(r, y) for r, y in out if r.audit]
    assert abs(len(marked) / 600 - 0.2) < 0.05
    again = Dispatcher(d.system, d.slow, audit=0.2)
    assert [again.ask(r.s1.trace.init).audit for r, _ in out[:50]] == [r.audit for r, _ in out[:50]]
    rec = [json.loads(x) for x in open(store.path) if x.strip()]
    assert sum(1 for x in rec if x.get("kind") == "dispatch" and x.get("audit")) == len(marked)
    r0, y0 = marked[0]
    store.save_correction("label", r0.s1.trace.init, y0, label_source="audit", of=r0.stored_id)
    plain = next(r for r, _ in out if not r.audit)
    with pytest.raises(UntrustedLabel, match="audit"):
        store.save_correction("label", plain.s1.trace.init, "yes", label_source="audit", of=plain.stored_id)
    assert [c["source"] for c in store.corrections(label_source="audit")] == ["audit"]
    assert store.corrections(label_source="human") == []


def test_the_audited_sample_measures_the_error_the_biased_corrections_overstate(tmp_path):
    d, store, out = _run(tmp_path, 0.25)
    alone = [(r, y) for r, y in out if r.by in ("s1", "s2")]
    true_err = sum(r.answer != y for r, y in alone) / len(alone)
    for r, y in out:
        if r.audit:
            store.save_correction("label", r.s1.trace.init, y, label_source="audit", of=r.stored_id)
        elif r.by in ("s1", "s2") and r.answer != y:          # a person corrects what looked wrong: biased
            store.save_correction("label", r.s1.trace.init, y, of=r.stored_id)
    p = system_report(open_storage(str(store.path)), drift_window=None).to_dict()["dispatch"]["label"]
    au = p["audit"]
    assert au["marked"] == sum(r.audit for r, _ in out)
    assert au["labels"]["labelled"] == au["marked"]
    assert abs(au["labels"]["error"] - true_err) < 0.05
    assert p["labels"]["all"]["error"] > 3 * true_err      # the corrections made where it looked wrong overstate it
    text = str(system_report(open_storage(str(store.path)), drift_window=None))
    assert "audited sample" in text
