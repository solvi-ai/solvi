"""S3 of the solvi.document plan: parallel calibration (workers > 1: the examples run in a thread pool, results in
input order, one broken example counts as escalated), the heartbeat, and the two calibration warnings (fewer than
100 examples; more than 20 % of the examples escalate). Acceptance: the same thresholds as sequential on a fixed
fake; 300 fake cases calibrate in < 5 s with workers=8."""
import io
import json
import re
import sys
import time
import warnings

import pytest

import solvi
from solvi import Answer
from solvi.core.catalog import Catalog, Question
from solvi.core.guarantees.guarantee import GuaranteeWarning
from solvi.core.store import JSONLStorage
from solvi.core.system import System, _rule_why

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_document_facts import part  # noqa: E402
from test_document_recipe import Server, case_reply, doc_of, model_of, ops_system  # noqa: E402


def labelled(n=30):
    """Deterministic labelled examples: one document per case, gold a-b (revenue $1250+i, cost (1,250))."""
    cases = []
    for i in range(n):
        d = doc_of(i, a=f"${1_250 + i}")
        cases.append(({"doc": d, "op": "diff"}, 2500.0 + i))
    return cases


def gold(r, y):
    return abs((r.answer or 0) - y) <= 1.0


def test_parallel_same_thresholds_as_sequential():
    """Acceptance: the same thresholds as sequential on a fixed fake."""
    s1 = ops_system(Server(case_reply()))
    rep1 = s1.calibrate(labelled(), max_error=0.10, workers=1, correct=gold)
    s8 = ops_system(Server(case_reply()))
    rep8 = s8.calibrate(labelled(), max_error=0.10, workers=8, correct=gold)
    for k in ("threshold", "n", "answered", "error", "risk", "method", "separation"):
        assert rep1[k] == rep8[k], k


def test_parallel_order_stable_with_failing_example():
    """One example whose ask raises counts as escalated: calibration completes, the report says so, and the signals
    stay in input order (the report's numbers match a sequential run on the same examples)."""
    rows = labelled()
    rows[7] = ({"doc": doc_of(7), "op": "boom"}, 2500.7)      # the compute raises on op="boom"
    s1 = ops_system(Server(case_reply()))
    rep1 = s1.calibrate(rows, max_error=0.10, workers=1, correct=gold)
    s8 = ops_system(Server(case_reply()))
    rep8 = s8.calibrate(rows, max_error=0.10, workers=8, correct=gold)
    assert rep1["threshold"] == rep8["threshold"] and rep1["n"] == rep8["n"] == len(rows)
    assert any("escalate" in w for w in rep8.get("warnings", []))


def test_300_cases_under_5s_with_8_workers():
    """Acceptance: 300 fake cases calibrate in < 5 s with workers=8."""
    s = ops_system(Server(case_reply()))
    t0 = time.time()
    s.calibrate(labelled(300), max_error=0.10, workers=8, correct=gold)
    assert time.time() - t0 < 5.0


def test_escalation_warning_over_20_percent():
    """More than 20 % of the examples escalate (abstain): a GuaranteeWarning names the share."""
    rows = labelled(30)
    # half the cases unknown to the reader → abstain (case_reply returns "$9,999" not in the text for unknown_a)
    from test_document_recipe import case_reply as cr
    s = ops_system(Server(cr(unknown_a=frozenset(range(0, 30, 2)))))   # 15 of 30 abstain
    with pytest.warns(GuaranteeWarning, match="escalate"):
        s.calibrate(rows, max_error=0.10, workers=4, correct=gold)


def test_system_guarantee_warns_on_few_examples():
    """The guarantee-level warning for a small calibration (the recipe passes warn_small=False: it has its own)."""
    rows = labelled(60)                                            # model confidence signals, all right
    gold_fn = lambda r, y: abs((r.answer or 0) - y) <= 1.0
    s = ops_system(Server(case_reply()))
    with pytest.warns(GuaranteeWarning, match="60 examples"):
        rows_sys = [(s.given(st), y) for st, y in rows]
        s.system.guarantee("answer", rows_sys, max_error=0.10, correct=gold_fn)
    s2 = ops_system(Server(case_reply()))
    import warnings as w
    with w.catch_warnings():
        w.simplefilter("error", GuaranteeWarning)
        rows_sys2 = [(s2.given(st), y) for st, y in rows]
        s2.system.guarantee("answer", rows_sys2, max_error=0.10, correct=gold_fn, warn_small=False)   # opt-out works


def test_progress_callback_counts_every_example():
    """progress receives (done, total) for every example, in order, ending at (total, total)."""
    seen = []
    s = ops_system(Server(case_reply()))
    s.calibrate(labelled(12), max_error=0.10, workers=4, correct=gold,
                progress=lambda done, total: seen.append((done, total)))
    assert seen == [(i + 1, 12) for i in range(12)]
