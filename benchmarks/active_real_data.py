"""Trusted host parser and label-independent splits for three archived UCI tasks.

Acquisition code must receive only allowed features and its own paid labels.
The returned pool/test records contain evaluator-only labels. Parsing, grouping
and archive access are host preparation. Fit and transform units are declared in
preprocess_costs for the caller to debit before actor work.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import random
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from zipfile import ZipFile


@dataclass(frozen=True)
class Source:
    uci_path: str
    member: str
    sha256: str
    citation: str
    doi: str

    @property
    def url(self):
        return f"https://archive.ics.uci.edu/static/public/{self.uci_path}.zip"


SOURCES = {
    "banknote": Source("267/banknote+authentication", "data_banknote_authentication.txt",
                       "1e2acd9a2085fadf3d8145c12d3d22af853320d52294a6590c2eaf75fdc05227",
                       "V. Lohweg (2012), Banknote Authentication, UCI", "10.24432/C55P57"),
    "spambase": Source("94/spambase", "spambase.data",
                       "813ac1df8effac70463c09c9c4b11e8803eefcab54771af66150852bcdcd1636",
                       "M. Hopkins, E. Reeber, G. Forman, J. Suermondt (1999), Spambase, UCI", "10.24432/C53G6X"),
    "wine_red": Source("186/wine+quality", "winequality-red.csv",
                       "3ed56667f4b828242bd732d7d1dd7f2861e54432239d7fa63877014cbb0304d4",
                       "P. Cortez, A. Cerdeira, F. Almeida, T. Matos, J. Reis (2009), Wine Quality, UCI",
                       "10.24432/C56S3T"),
}
_WINE = ("fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
         "total sulfur dioxide", "density", "pH", "sulphates", "alcohol")


@dataclass(frozen=True)
class Row:
    row_id: int
    features: tuple[str, ...]
    label: bool


@dataclass(frozen=True)
class Dataset:
    name: str
    feature_names: tuple[str, ...]
    rows: tuple[Row, ...]
    source_sha: str  # Original selected member; archive hash is independently pinned.
    rawcounts: dict


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _atomic(path, data):
    with NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temp = Path(stream.name)
    try:
        with temp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _numeric(token):
    text = token.strip()
    if text in ("?", ""):
        return None
    try:
        number = Decimal(text)
    except InvalidOperation as exc:
        raise ValueError(f"invalid numeric feature {text!r}") from exc
    if not number.is_finite():
        return None
    if not number:
        return "0"
    # normalize() rounds long inputs in the ambient Decimal context. Remove
    # only mathematically redundant trailing zeros from the exact coefficient.
    sign, digits, exp = number.as_tuple()
    end = len(digits)
    while digits[end - 1] == 0:
        end -= 1
    return str(Decimal((sign, digits[:end], exp + len(digits) - end)))


def _target(name, token):
    try:
        value = Decimal(token.strip())
    except InvalidOperation as exc:
        raise ValueError(f"invalid {name} target") from exc
    if not value.is_finite() or value != value.to_integral_value():
        raise ValueError(f"invalid {name} target")
    if name == "wine_red":
        if not 0 <= value <= 10:
            raise ValueError("wine quality outside 0..10")
        return value >= 6
    if value not in (0, 1):
        raise ValueError("binary target outside 0/1")
    return value == 1


def _parse(name, data, fields):
    rows = csv.reader(io.StringIO(data.decode("ascii")), delimiter=";" if name == "wine_red" else ",", strict=True)
    if name == "wine_red" and next(rows, None) != [*fields, "quality"]:
        raise ValueError("unexpected wine header")
    complete = []
    raw = missing = 0
    frequencies = Counter()
    for columns in rows:
        if not columns:  # Empty physical line, not a UCI observation.
            continue
        raw += 1
        if len(columns) != len(fields) + 1:
            raise ValueError(f"{name} row {raw}: unexpected field count")
        label = _target(name, columns[-1])
        features = tuple(_numeric(token) for token in columns[:-1])
        if None in features:
            missing += 1
            continue
        complete.append(Row(raw, features, label))
        frequencies[features] += 1
    counts = dict(raw_rows=raw, complete_rows=len(complete), missing_feature_rows=missing,
                  unique_groups=len(frequencies), duplicate_groups=sum(n > 1 for n in frequencies.values()),
                  duplicate_rows=len(complete) - len(frequencies))
    return Dataset(name, tuple(fields), tuple(complete), _sha(data), counts)


def load_dataset(name, cache_dir):
    """Verify the pinned archive, parse labels strictly, retain original 1-based data-row IDs."""
    source = SOURCES[name]
    data = (Path(cache_dir) / f"{name}.zip").read_bytes()
    if _sha(data) != source.sha256:
        raise ValueError(f"{name}: archive SHA-256 mismatch")
    with ZipFile(io.BytesIO(data)) as archive:
        if name == "spambase":
            names_file = archive.read("spambase.names").decode("ascii")
            definitions = (line.partition(":") for line in names_file.splitlines())
            fields = tuple(field.strip() for field, _, kind in definitions if kind.strip() == "continuous.")
            if len(fields) != 57 or len(set(fields)) != 57:
                raise ValueError("unexpected Spambase schema")
        elif name == "wine_red":
            fields = _WINE
        else:
            fields = ("variance", "skewness", "curtosis", "entropy")
        return _parse(name, archive.read(source.member), fields)


def fetch_sources(cache_dir):
    """Fetch missing pinned official archives; publish schema and X-only diagnostics."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, source in SOURCES.items():
        path = cache / f"{name}.zip"
        if not path.exists():
            with urlopen(source.url, timeout=60) as response:
                data = response.read()
            if _sha(data) != source.sha256:
                raise ValueError(f"{name}: unexpected upstream archive SHA-256")
            _atomic(path, data)
        ds = load_dataset(name, cache)
        if ds.rawcounts["unique_groups"] < 1280:
            raise ValueError(f"{name}: need 1280 complete distinct feature groups")
        manifest[name] = dict(name=name, url=source.url, archive_sha256=source.sha256, member=source.member,
                              member_sha256=ds.source_sha, archive_mtime_utc=datetime.fromtimestamp(
                                  path.stat().st_mtime, UTC).isoformat(), citation=source.citation,
                              doi=source.doi, license="CC BY 4.0",
                              license_url="https://creativecommons.org/licenses/by/4.0/",
                              source_card="https://archive.ics.uci.edu/dataset/" + source.uci_path,
                              features=list(ds.feature_names), rawcounts=ds.rawcounts)
    _atomic(cache / "DATA_MANIFEST.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode())
    return manifest


def _group_id(name, features):
    key = json.dumps(features, ensure_ascii=True, separators=(",", ":"))
    return _sha(("closed:" + name + key).encode())


def _prepare(ds, seed, split, *, holdout_size=512, pool_size=256, test_size=512, initial_size=16):
    if split not in ("dev", "final"):
        raise ValueError("expected dev or final split")
    if not (0 < initial_size <= pool_size and holdout_size == test_size > 0):
        raise ValueError("invalid dataset role sizes")
    representatives = {}
    for row in ds.rows:
        representatives.setdefault(row.features, row)
    if len(representatives) < holdout_size + pool_size + test_size:
        raise ValueError("insufficient complete distinct feature groups")
    by_id = sorted(((_group_id(ds.name, x), row) for x, row in representatives.items()), key=lambda pair: pair[0])
    closed = by_id[:holdout_size]
    open_groups = by_id[holdout_size:]
    random.Random(seed).shuffle(open_groups)
    pool = open_groups[:pool_size]
    test = closed if split == "final" else open_groups[pool_size:pool_size + test_size]

    names = [f"f{i}" for i in range(len(ds.feature_names))]
    edges = []
    for i in range(len(names)):
        values = sorted(Decimal(row.features[i]) for _, row in pool)
        edges.append([values[(pool_size * j) // 8] for j in range(1, 8)])

    def encode(groups):
        return [dict(id=row.row_id, inputs={k: bisect_right(th, Decimal(x)) for k, th, x in
                                             zip(names, edges, row.features, strict=True)}, label=row.label)
                for _, row in groups]

    # Only host-side aggregate QC; never choose a group member by its label.
    permitted = {row.features for _, row in open_groups} if split == "dev" else representatives.keys()
    conflicts = {row.features for row in ds.rows if row.features in permitted
                 and row.label != representatives[row.features].label}
    rawcounts = dict(ds.rawcounts, unused_groups=len(by_id) - len(pool) - len(test),
                     conflicting_label_groups=len(conflicts),
                     conflict_scope="nonholdout" if split == "dev" else "all_groups")
    encoded_pool, encoded_test = encode(pool), encode(test)
    encoded_groups = {tuple(row["inputs"].values()) for row in encoded_pool}
    collision_count = sum(tuple(row["inputs"].values()) in encoded_groups for row in encoded_test)
    return dict(
        name=ds.name, seed=seed, split=split, source_sha=ds.source_sha, rawcounts=rawcounts,
        feature_names=list(ds.feature_names),
        thresholds={k: list(map(str, column)) for k, column in zip(names, edges, strict=True)},
        group_ids=dict(pool=[gid for gid, _ in pool], test=[gid for gid, _ in test],
                       holdout=[gid for gid, _ in closed]),
        initial_ids=[row.row_id for _, row in pool[:initial_size]],
        pool=encoded_pool, test=encoded_test, quantized_test_rows_matching_pool=collision_count,
        preprocess_costs=dict(
            preprocess_sort=pool_size * (pool_size.bit_length() + 1) * len(names),
            preprocess_fields=(pool_size + test_size) * len(names),
            preprocess_compare=3 * (pool_size + test_size) * len(names),
        ),
        preprocessing_excludes="host archive/parse/group/split, metadata and disk I/O",
    )


def prepare_dataset(name, cache_dir, seed, split):
    """Return trusted host records and reproducible cost categories; fetch_sources runs separately."""
    return _prepare(load_dataset(name, cache_dir), seed, split)
