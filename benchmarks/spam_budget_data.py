"""Explicit exposure roles; trusted host labels never enter selection."""
from bisect import bisect_right
from decimal import Decimal
import json
import random
from pathlib import Path
from active_real_data import load_dataset, _group_id

def _prepare(ds, seed, split, roles, *, holdout_size=512, pool_size=256, test_size=512, initial_size=16):
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
    holdout = roles["holdout"]
    previous = roles["previous_used"]
    if not isinstance(holdout, list) or len(holdout) != 512 or len(set(holdout)) != 512:
        raise ValueError("holdout must contain exactly 512 distinct group IDs")
    known = {gid for gid, _ in by_id}
    if not set(holdout) <= known or not isinstance(previous, list) or not set(previous) <= known:
        raise ValueError("foreign exposure group")
    if ds.name == "spambase" and set(holdout) & set(previous):
        raise ValueError("Spambase holdout was previously exposed")
    closed = [(gid, row) for gid, row in by_id if gid in set(holdout)]
    open_groups = [(gid, row) for gid, row in by_id if gid not in set(holdout)]
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


def prepare(name, cache, seed, split, roles_path):
    roles = json.loads(Path(roles_path).read_text())["datasets"][name]
    return _prepare(load_dataset(name, cache), seed, split, roles)

prepare_dataset = prepare
