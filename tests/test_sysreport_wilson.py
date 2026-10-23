"""The system report's Wilson 95 % interval on every observed error and risk: known answers (scipy's
`binomtest(...).proportion_ci(method="wilson")` and the closed form), additive JSON keys, text that only gains the
interval."""
import math
import re

import pytest
from test_sysreport import guarded

from solvi.core.store import JSONLStorage
from solvi.core.store.sysreport import _labels_block, _labels_line, wilson

# (k, n) → (low, high) from scipy.stats.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
KNOWN = {(3, 40): (0.025836025774588198, 0.19864233524310543),
         (0, 25): (0.0, 0.1331922509390485),
         (25, 25): (0.8668077490609516, 1.0),
         (1, 3): (0.06149194472039626, 0.7923403991979523),
         (1, 4): (0.04558726080970055, 0.6993581574175981),
         (20, 40): (0.35199526933465386, 0.6480047306653461)}


@pytest.mark.parametrize("kn", sorted(KNOWN))
def test_wilson_equals_the_recorded_scipy_values(kn):
    assert wilson(*kn) == pytest.approx(KNOWN[kn], abs=1e-12)


def test_wilson_equals_scipy_live_and_the_closed_form():
    from scipy.stats import binomtest
    z = 1.959963984540054
    for n in (1, 2, 7, 25, 50, 100, 333):
        for k in sorted({0, 1, n // 3, n // 2, n - 1, n}):
            ci = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
            assert wilson(k, n) == pytest.approx((ci.low, ci.high), abs=1e-12)
            p = k / n
            c = (p + z * z / (2 * n)) / (1 + z * z / n)
            h = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
            assert wilson(k, n) == pytest.approx((max(0.0, c - h), min(1.0, c + h)), abs=1e-12)


def test_wilson_at_another_level_and_with_no_rows():
    from scipy.stats import binomtest
    ci = binomtest(3, 40).proportion_ci(confidence_level=0.9, method="wilson")
    assert wilson(3, 40, 0.9) == pytest.approx((ci.low, ci.high), abs=1e-12)
    assert wilson(0, 0) is None


def test_the_block_gains_two_keys_and_keeps_every_other_one():
    rows = [(True, True)] * 3 + [(True, False)] * 37 + [(False, None)] * 10
    b = _labels_block(rows, 0.05, "error")
    assert b["error_wilson95"] == pytest.approx(KNOWN[(3, 40)], abs=1e-12)
    assert b["risk_wilson95"] == pytest.approx(wilson(3, 50), abs=1e-12)
    old = {k: v for k, v in b.items() if not k.endswith("_wilson95")}
    assert set(b) - set(old) == {"error_wilson95", "risk_wilson95"}
    assert old["error"] == 3 / 40 and old["risk"] == 3 / 50 and old["wrong_alone"] == 3 and old["labelled"] == 50
    empty = _labels_block([])
    assert empty["error_wilson95"] is None and empty["risk_wilson95"] is None


def test_the_line_only_appends_the_interval():
    b = _labels_block([(True, True)] * 3 + [(True, False)] * 37)
    line = _labels_line(b)
    assert "error among them 7.50% [2.6%, 19.9%] (Wilson 95 %)" in line
    stripped = re.sub(r" \[[^\]]*\] \(Wilson 95 %\)", "", line)
    assert stripped == "40 labelled; answered alone and labelled 40, wrong 3 — error among them 7.50%, alone and " \
                       "wrong 7.50% of the labelled"


def test_the_system_report_carries_the_interval_in_json_and_text(tmp_path):
    s = guarded(JSONLStorage(tmp_path / "d.jsonl"))
    for i in range(40):
        r = s.ask({"email": f"parcel {i}"})                        # "shipping", answered alone
        s.storage.save_correction("team", {"email": f"parcel {i}"}, "billing" if i < 3 else "shipping", of=r.stored_id)
    rep = s.report(drift_window=None)
    p = rep.to_dict()["system1"]["team"]
    for L in (p["labels"], p["promises"][0]["labels"]):
        assert (L["alone"], L["wrong_alone"]) == (40, 3)
        assert L["error_wilson95"] == pytest.approx(KNOWN[(3, 40)], abs=1e-12)
        assert L["risk_wilson95"] == pytest.approx(KNOWN[(3, 40)], abs=1e-12)
    assert "error among them 7.50% [2.6%, 19.9%] (Wilson 95 %)" in str(rep)
