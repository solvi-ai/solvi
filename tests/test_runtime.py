"""Execution and independent replay: a clean trace passes; tampering (naive, or with recomputed hashes) is caught and located."""
import copy
import random

import pytest

from solvi import System
from examples_loader import load
from solvi.runtime import vhash

I = load("03_invoices")


@pytest.fixture(scope="module")
def trace():
    init, _ = I.make(random.Random(11), 11)
    return System(I.cat, [q for q in I.QUESTIONS if q.name != "risk"]).ask(init).trace


def _victim(t):
    return next(r for r in t.records if isinstance(r.value, (int, float)) and not isinstance(r.value, bool) and r.error is None)


def test_clean_replay(trace):
    rep = trace.replay(I.cat)
    assert rep["ok"] and rep["mismatches"] == []


def test_naive_tamper_found_and_located(trace):
    t = copy.deepcopy(trace)
    r = _victim(t)
    r.value = r.value + 1
    rep = t.replay(I.cat)
    assert not rep["ok"]
    assert rep["mismatches"][0][0] == r.step


def test_consistent_tamper_found_and_located(trace):
    t = copy.deepcopy(trace)
    r = _victim(t)
    r.value = r.value + 1
    prev = r.prev
    for x in t.records[r.step - 1:]:          # the attacker recomputed the hashes of the record and of the whole chain after it
        x.prev = prev
        x.hash = vhash(x.body())
        prev = x.hash
    rep = t.replay(I.cat)
    assert not rep["ok"]
    assert rep["mismatches"][0][0] == r.step


def test_quotes_point_into_document():
    init, _ = I.make(random.Random(12), 12)
    r = System(I.cat, [q for q in I.QUESTIONS if q.name == "approve"]).ask(init)
    for rec in r.trace.records:
        if rec.kind == "extract" and rec.error is None:
            s, e, src = rec.quote
            assert 0 <= s <= e <= len(init[src])
