"""Bars for the benchmark scripts whose published numbers no other test asserts (run with -m slow; under a minute
each): drift_simulation.py, journal_size.py, ask_overhead.py. The published numbers and the bars are in
benchmarks/baseline.json; speed is held as ratios between settings measured in one run."""
import importlib.util
import json
import time
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "benchmarks" / "baseline.json").read_text())


def _script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "benchmarks" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_drift_monitor_flags_no_unchanged_stream_beyond_its_bar():
    bar = BASE["drift_simulation"]["bars"]
    t0 = time.time()
    out = _script("drift_simulation").stationary(0, parts=33)          # one stream per case: 36 streams
    assert time.time() - t0 < 90
    assert out["streams"] == bar["stationary_streams"] and out["flagged"] <= bar["stationary_max_flagged"]


@pytest.mark.slow
def test_drift_monitor_flags_a_change_within_its_published_delay():
    bar = BASE["drift_simulation"]["bars"]
    w = bar["delay_window"]
    out = _script("drift_simulation").delays(windows=(w,), streams=bar["delay_streams"])
    a, m = out[f"answered window={w}"], out[f"mix window={w}"]
    assert a["median"] is not None and a["median"] <= bar["answered_median_max"]
    assert m["median"] is not None and m["median"] <= bar["mix_median_max"]
    for r in (a, m):
        assert r["missed"] <= bar["missed_max"] and r["false"] <= bar["false_max"]


@pytest.mark.slow
def test_journal_size_holds_its_published_bytes_and_every_store_verifies_and_replays(tmp_path):
    base = BASE["journal_size"]
    pub, bar = base["published"], base["bars"]
    tol = bar["tolerance"]
    rows_path = tmp_path / "rows.json"
    assert _script("journal_size").main(["--only", "game", "--json", str(rows_path)]) == 0
    rows = {r["mode"]: r for r in json.loads(rows_path.read_text())}
    assert all(r["verify_ok"] and r["replay_failed"] <= bar["replay_failed_max"] for r in rows.values())
    full, comp = rows["full"]["bytes_per_decision"], rows["compact"]["bytes_per_decision"]
    assert abs(full / pub["game_full_bytes"] - 1) <= tol and abs(comp / pub["game_compact_bytes"] - 1) <= tol
    assert full / comp >= bar["min_full_over_compact"]


@pytest.mark.slow
def test_ask_overhead_ratios_hold(tmp_path):
    bar = BASE["ask_overhead"]["bars"]
    path = tmp_path / "overhead.json"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _script("ask_overhead").main(["--quick", "--json", str(path)]) == 0
    got = json.loads(path.read_text())
    for work in ("gallery", "decider"):
        med = {r["setting"]: r["median_ms"] for r in got[work]}
        assert med["0.7 default"] / med["0.5.0-style"] <= bar["default_over_050_max"], (work, med)
        assert med["+ store jsonl"] / med["0.7 default"] <= bar["store_jsonl_over_default_max"], (work, med)
        if "+ guarantee" in med:
            assert med["+ guarantee"] / med["0.7 default"] <= bar["guarantee_over_default_max"], (work, med)
