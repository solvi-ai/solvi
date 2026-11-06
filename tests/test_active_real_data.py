"""Offline data provenance, duplicate protection and transductive quantization checks."""

from dataclasses import replace
from decimal import Decimal, localcontext
import hashlib
import io
from pathlib import Path
import sys
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import active_real_data as data  # noqa: E402


def banknote(rows):
    return data._parse("banknote", ("\n".join(rows) + "\n").encode(), ("a", "b", "c", "d"))


def many():
    return banknote([f"{i},0,0,0,{i % 2}" for i in range(24)] + ["01.00,0,-0,0,0"])


def small(ds, seed, split):
    return data._prepare(ds, seed, split, holdout_size=5, pool_size=9, test_size=5, initial_size=2)


def test_exact_decimal_and_missingness_independent_of_target():
    with localcontext() as ctx:
        ctx.prec = 4
        assert data._numeric("+000123456789.0000") == "123456789"
        assert data._numeric("0.12345678900") == "0.123456789"
    ds = banknote(["0,-0.0,1.00,01,0", "-0E10,0,1.000,1,1", "?,1,1,1,1",
                   "1,,1,1,0", "inf,1,1,1,1", "nan,1,1,1,0"])
    assert ds.rows[0].features == ds.rows[1].features == ("0", "0", "1", "1")
    assert ds.rawcounts == dict(raw_rows=6, complete_rows=2, missing_feature_rows=4,
                                unique_groups=1, duplicate_groups=1, duplicate_rows=1)
    assert [r.row_id for r in ds.rows] == [1, 2]
    assert [type(r.label) for r in ds.rows] == [bool, bool]


@pytest.mark.parametrize("line", ["?,0,0,0,?", "0,0,0,0,2", "broken,1,1,1,1", "1,0,1"])
def test_invalid_labels_features_and_width_fail_hard(line):
    with pytest.raises(ValueError):
        banknote([line])


def test_wine_header_threshold_and_red_only_selection(monkeypatch, tmp_path):
    buf = io.BytesIO()
    header = ";".join([*data._WINE, "quality"])
    rows = [header, ";".join(["0"] * 11 + ["5"]), ";".join(["1"] * 11 + ["6"])]
    red = "\n".join(rows).encode()
    with ZipFile(buf, "w") as archive:
        archive.writestr("winequality-red.csv", red)
        archive.writestr("winequality-white.csv", b"must never be parsed")
    payload = buf.getvalue()
    (tmp_path / "wine_red.zip").write_bytes(payload)
    source = replace(data.SOURCES["wine_red"], sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setitem(data.SOURCES, "wine_red", source)
    ds = data.load_dataset("wine_red", tmp_path)
    assert tuple(r.label for r in ds.rows) == (False, True)
    assert ds.source_sha == hashlib.sha256(red).hexdigest()
    with pytest.raises(ValueError, match="header"):
        data._parse("wine_red", red.replace(b"quality", b"other"), data._WINE)
    with pytest.raises(ValueError, match="quality"):
        data._parse("wine_red", red.replace(b";6", b";11"), data._WINE)
    (tmp_path / "wine_red.zip").write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="SHA-256"):
        data.load_dataset("wine_red", tmp_path)


def test_fixed_holdout_excludes_all_dev_and_all_final_pools_even_with_conflicting_duplicates():
    ds = many()
    first_final = small(ds, 6901, "final")
    holdout = set(first_final["group_ids"]["test"])
    assert len(holdout) == 5
    for seed, split in [(5901, "dev"), (5902, "dev"), (6901, "final"), (6902, "final")]:
        part = small(ds, seed, split)
        assert set(part["group_ids"]["holdout"]) == holdout
        assert set(part["group_ids"]["pool"]).isdisjoint(holdout)
        test = set(part["group_ids"]["test"])
        assert test == holdout if split == "final" else test.isdisjoint(holdout)
        assert set(part["group_ids"]["pool"]).isdisjoint(test)
        assert 25 not in [r["id"] for r in part["pool"] + part["test"]]
        assert part["initial_ids"] == [r["id"] for r in part["pool"][:2]]
    assert first_final["rawcounts"]["conflicting_label_groups"] == 1
    assert first_final["rawcounts"]["unused_groups"] == 10
    assert first_final == small(ds, 6901, "final")


def test_splits_bins_and_costs_are_independent_of_labels():
    ds = many()
    a = small(ds, 31, "dev")
    b = small(replace(ds, rows=tuple(replace(r, label=not r.label) for r in ds.rows)), 31, "dev")
    for key in ("group_ids", "thresholds", "initial_ids", "preprocess_costs"):
        assert a[key] == b[key]
    for role in ("pool", "test"):
        assert [(r["id"], r["inputs"]) for r in a[role]] == [(r["id"], r["inputs"]) for r in b[role]]
    assert a["preprocess_costs"] == dict(preprocess_sort=9 * 5 * 4, preprocess_fields=14 * 4,
                                        preprocess_compare=3 * 14 * 4)
    with pytest.raises(ValueError, match="insufficient"):
        data._prepare(ds, 1, "dev")


def test_edges_depend_only_on_pool_even_if_test_values_shift(monkeypatch):
    ds = many()
    # Fix raw-group ordering in this metamorphic test so changing a test X
    # cannot reassign global hash roles and thereby indirectly change the pool.
    monkeypatch.setattr(data, "_group_id", lambda name, x: str(int(Decimal(x[0]))).zfill(3))
    a = small(ds, 8, "final")
    test_ids = {r["id"] for r in a["test"]}
    rows = tuple(replace(r, features=(r.features[0], "999", *r.features[2:])) if r.row_id in test_ids else r
                 for r in ds.rows if r.row_id != 25)
    b = small(replace(ds, rows=rows), 8, "final")
    assert b["thresholds"] == a["thresholds"]
    assert b["pool"] == a["pool"]
    assert b["thresholds"]["f1"] == ["0"] * 7
    assert all(r["inputs"]["f1"] == 7 for r in b["test"])
    assert all(r["inputs"]["f1"] == 7 for r in a["pool"])


def test_fetch_seals_archive_before_cache_write(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong upstream bytes"))
    with pytest.raises(ValueError, match="upstream"):
        data.fetch_sources(tmp_path)
    assert not list(tmp_path.iterdir())
