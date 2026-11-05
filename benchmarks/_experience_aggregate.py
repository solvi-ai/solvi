"""Pre-registered seed-paired analysis; never executes search or alters outcomes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

SEEDS = tuple(range(1101, 1113))
FAMILIES = ("resource", "workflow")
ARMS = ("SOLVER", "FORMULA_CACHE", "FIXED", "RANDOM", "ACTIVE", "FULL", "NO_CALL")
LIMITS = (16384, 65536, 262144)
PRICES = (1, 8, 32)
SPEC = "xy-int-v2-composite-teacher-prefix"
TCRIT = 1.7958848187
KEY_FIELDS = ("seed", "family", "arm", "limit", "observation_price", "stratum")


def _hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _key(row):
    return tuple(row[k] for k in KEY_FIELDS)


def _validate(records):
    seen = set()
    records = list(records)
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("trial must be an object")
        if any(type(row.get(name)) is not int for name in ("seed", "limit", "observation_price")):
            raise ValueError("grid coordinates must be strict integers")
        key = _key(row)
        if key in seen:
            raise ValueError(f"duplicate trial: {key}")
        seen.add(key)
        if (
            row.get("spec") != SPEC
            or row["seed"] not in SEEDS
            or row["family"] not in FAMILIES
            or row["arm"] not in ARMS
        ):
            raise ValueError("unknown spec/seed/family/arm")
        if (
            row["limit"] not in LIMITS
            or row["observation_price"] not in PRICES
            or row["stratum"] not in ("dev", "id", "encoding", "ood")
        ):
            raise ValueError("unknown grid coordinate")
        for name in ("denominator", "success", "applications", "weighted_total"):
            if type(row.get(name)) is not int or row[name] < 0:
                raise ValueError(f"invalid {name}")
        if (
            row["denominator"] != (64 if row["stratum"] == "dev" else 128)
            or max(row["success"], row["applications"]) > row["denominator"]
        ):
            raise ValueError("invalid denominator/count")
        raw, weighted = row["raw_vector"], row["weighted_vector"]
        if any(type(v) is not int or v < 0 for v in [*raw.values(), *weighted.values()]):
            raise ValueError("invalid cost vector")
        price = row["observation_price"]
        prices = {
            "observation": 8 * price,
            "observe_measure": 8 * price,
            "observe_diagnose": 4 * price,
            "observe_noise": price,
            "action": 64,
        }
        expected = {k: v * prices.get(k, 8) for k, v in raw.items()}
        if expected != weighted or sum(weighted.values()) != row["weighted_total"]:
            raise ValueError("inconsistent raw/weighted ledger")
        if type(row.get("accounting_scale")) is not int or row["accounting_scale"] != 8:
            raise ValueError("exact eighth-unit accounting required")
        exact = row.get("exact_accounting_total", {})
        if (
            set(exact) != {"numerator", "denominator"}
            or any(type(v) is not int for v in exact.values())
            or exact != {"numerator": row["weighted_total"], "denominator": 8}
        ):
            raise ValueError("inconsistent exact accounting total")
        if row["weighted_total"] > row["limit"] * 8:
            raise ValueError("weighted cost exceeds eighth-unit budget")
        if "hard_violations" in row and (
            type(row["hard_violations"]) is not int or not 0 <= row["hard_violations"] <= row["denominator"]
        ):
            raise ValueError("invalid hard violations")
        if "rows" in row:
            episodes = row["rows"]
            if not isinstance(episodes, list) or any(not isinstance(r, dict) for r in episodes):
                raise ValueError("episode rows must be a list of objects")
            task_ids = [r.get("task_id") for r in episodes]
            if any(not isinstance(task_id, str) or not task_id for task_id in task_ids) or len(set(task_ids)) != len(
                task_ids
            ):
                raise ValueError("missing or duplicate episode task IDs")
            if any(type(r.get("success")) is not bool or type(r.get("applied")) is not bool for r in episodes):
                raise ValueError("episode flags must be boolean")
            flags = [r["hard_violation"] for r in episodes if "hard_violation" in r]
            if flags and (
                len(flags) != len(episodes)
                or any(type(v) is not bool for v in flags)
                or sum(flags) != row.get("hard_violations")
            ):
                raise ValueError("episode hard violations disagree")
            if (
                len(episodes) != row["denominator"]
                or sum(bool(r["success"]) for r in episodes) != row["success"]
                or sum(bool(r["applied"]) for r in episodes) != row["applications"]
            ):
                raise ValueError("episode counts disagree")
    return records


def _compact_trial(row):
    """Retain analysis statistics after full validation; raw evidence remains on disk."""
    fields = (
        *KEY_FIELDS,
        "spec",
        "denominator",
        "success",
        "applications",
        "weighted_total",
        "accounting_scale",
        "exact_accounting_total",
        "raw_vector",
        "weighted_vector",
        "hard_violations",
        "false_applications",
        "phases",
        "training_cost",
        "wall_seconds",
        "peak_rss_kib",
    )
    compact = {name: row[name] for name in fields if name in row}
    for name in ("learning", "adaptation_learning"):
        if name in row:
            learning = row[name]
            compact[name] = {key: learning[key] for key in ("status", "reason", "record", "cost") if key in learning}
            if "candidate" in learning:
                candidate = learning["candidate"]
                compact[name]["candidate"] = {
                    key: candidate[key]
                    for key in ("program", "scope", "mechanism", "schema", "grammar_fingerprint")
                    if key in candidate
                }
    compact["source_result_sha256"] = _hash(row)
    return compact


def collect(rawpaths):
    """Validate one full JSONL trial at a time, then retain only compact statistics.

    source_result_sha256 binds each compact row to its complete canonical result.
    dev_raw_hash/final_raw_hash bind the sorted analysis records, not file bytes;
    the CLI's *_raw_files_sha256 separately bind the original physical JSONL files.
    """
    records, wire_keys, semantic_keys = [], set(), set()
    for path in rawpaths:
        with Path(path).open() as stream:
            for line in stream:
                item = json.loads(line)
                if item.get("kind") == "episode":
                    continue
                if item.get("kind") == "trial":
                    if item["key"] in wire_keys:
                        raise ValueError("duplicate raw trial key")
                    wire_keys.add(item["key"])
                    result = item["result"]
                else:
                    result = item
                _validate([result])
                key = _key(result)
                if key in semantic_keys:
                    raise ValueError(f"duplicate trial: {key}")
                semantic_keys.add(key)
                records.append(_compact_trial(result))
                del result, item
    return records


def paired_interval(values):
    """One-sided 95% bounds using exactly 12 independent seed means."""
    values = list(values)
    if len(values) != 12 or not all(math.isfinite(x) for x in values):
        raise ValueError("exactly 12 finite seed means required")
    mean = statistics.mean(values)
    margin = TCRIT * statistics.stdev(values) / math.sqrt(12)
    return dict(mean=mean, lower=mean - margin, upper=mean + margin, n_seeds=12)


def _arm(records, family, arm, stratum, denominator):
    chosen = [
        r
        for r in records
        if r["family"] == family
        and r["arm"] == arm
        and r["stratum"] == stratum
        and r["limit"] == 65536
        and r["observation_price"] == 8
    ]
    by_seed = {r["seed"]: r for r in chosen}
    if set(by_seed) != set(SEEDS) or any(r["denominator"] != denominator for r in chosen):
        raise ValueError(f"incomplete 12-seed primary arm: {family}/{arm}/{stratum}")
    return [by_seed[s] for s in SEEDS]


def choose_comparators(dev_trials):
    records = _validate(dev_trials)
    if any(r["stratum"] != "dev" for r in records):
        raise ValueError("comparator selection accepts dev only")
    selections, table = {}, []
    for family in FAMILIES:
        scores = []
        for arm in ARMS[:5]:
            rows = _arm(records, family, arm, "dev", 64)
            item = dict(
                family=family,
                arm=arm,
                success=statistics.mean(r["success"] / 64 for r in rows),
                cost=statistics.mean(r["weighted_total"] for r in rows),
            )
            scores.append(item)
            table.append(item)
        rank = lambda r: (-r["success"], r["cost"], r["arm"])
        selections[family] = dict(
            strong=min(scores, key=rank)["arm"],
            info_best_fixed_random=min((r for r in scores if r["arm"] in ("FIXED", "RANDOM")), key=rank)["arm"],
        )
    source = Path(__file__)
    expected = set(itertools.product(SEEDS, FAMILIES, ARMS, LIMITS, PRICES, ("dev",)))
    dev_complete = {_key(r) for r in records} == expected and all(r["denominator"] == 64 for r in records)
    return dict(
        dev_complete_grid=dev_complete,
        expected_dev_trials=1512,
        actual_dev_trials=len(records),
        spec=SPEC,
        seeds=list(SEEDS),
        primary_limit=65536,
        primary_price=8,
        dev_raw_hash=_hash(sorted(records, key=_key)),
        dev_raw_hash_scope="canonical_sorted_analysis_records; physical files bound separately",
        source_code_hashes={source.name: hashlib.sha256(source.read_bytes()).hexdigest()},
        comparators=selections,
        selection_table=table,
    )


def _compare(treatment, control):
    success = paired_interval(
        a["success"] / a["denominator"] - b["success"] / b["denominator"] for a, b in zip(treatment, control)
    )
    saving = (
        None
        if any(b["weighted_total"] == 0 for b in control)
        else paired_interval(
            (b["weighted_total"] - a["weighted_total"]) / b["weighted_total"] for a, b in zip(treatment, control)
        )
    )
    eligible = (
        min(statistics.mean(r["success"] / r["denominator"] for r in rows) for rows in (treatment, control)) >= 0.90
    )
    passed = bool(
        saving is not None
        and saving["mean"] >= 0.10
        and saving["lower"] > 0
        and success["mean"] >= -0.01
        and success["lower"] > -0.01
        and eligible
    )
    return dict(
        success_difference=success, relative_cost_saving=saving, both_success_at_least_90pct=eligible, passed=passed
    )


def _learned(row):
    learning = row.get("learning", {})
    ast = learning.get("candidate", {}).get("program", {}).get("ast", [])
    operators, names = [], set()

    def walk(node):
        if not isinstance(node, (list, tuple)) or not node:
            return
        if node[0] == "input" and len(node) == 2:
            names.add(node[1])
        elif node[0] in ("sub", "max", "add"):
            operators.append(node[0])
        for child in node[1:]:
            walk(child)

    walk(ast)
    return learning.get("record", {}).get("accepted") is True and len(operators) >= 2 and names == {"x", "y"}


def finalaggregate(final_trials, lock):
    records = _validate(final_trials)
    if (
        lock.get("spec") != SPEC
        or lock.get("seeds") != list(SEEDS)
        or lock.get("primary_limit") != 65536
        or lock.get("primary_price") != 8
    ):
        raise ValueError("incompatible comparator lock")
    if any(r["stratum"] == "dev" for r in records):
        raise ValueError("dev data in final analysis")
    expected = set(itertools.product(SEEDS, FAMILIES, ARMS, LIMITS, PRICES, ("id", "encoding", "ood")))
    missing = expected - {_key(r) for r in records}
    complete = not missing and all(r["denominator"] == 128 for r in records)
    audited = all("hard_violations" in r for r in records)
    violations = sum(r.get("hard_violations", 0) for r in records)
    families = {}
    for family in FAMILIES:
        selection = lock["comparators"][family]
        if selection["strong"] not in ARMS[:5] or selection["info_best_fixed_random"] not in ("FIXED", "RANDOM"):
            raise ValueError("invalid comparator arm")
        arms = {
            arm: _arm(records, family, arm, "id", 128)
            for arm in set(("ACTIVE", "FULL", "NO_CALL", selection["strong"], selection["info_best_fixed_random"]))
        }
        info = _compare(arms["ACTIVE"], arms[selection["info_best_fixed_random"]])
        info["passed"] = info["success_difference"]["mean"] >= 0.05 and info["success_difference"]["lower"] > 0
        operation = _compare(arms["FULL"], arms["NO_CALL"])
        operation["application_fraction"] = statistics.mean(r["applications"] / 128 for r in arms["FULL"])
        operation["learned_composite_all_seeds"] = all(_learned(r) for r in arms["FULL"])
        operation["passed"] &= operation["application_fraction"] >= 0.20 and operation["learned_composite_all_seeds"]
        overall = _compare(arms["FULL"], arms[selection["strong"]])
        for result in (info, operation, overall):
            result["passed"] &= audited and violations == 0
            result["verdict"] = "PASS" if result["passed"] else "NOT_CONFIRMED"
        families[family] = dict(H_info=info, H_operation=operation, overall=overall)
    # Descriptive grid preserves prefix/suffix and announced/hidden OOD groups.
    groups = {}
    for row in records:
        stratum = row["stratum"]
        if stratum == "ood":
            stratum += "_announced" if row["seed"] <= 1106 else "_hidden"
        for phase, counts in [("all", row), *row.get("phases", {}).items()]:
            if not counts["denominator"]:
                continue
            key = (row["family"], row["arm"], row["limit"], row["observation_price"], stratum, phase)
            groups.setdefault(key, []).append((row, counts))
    tables = [
        dict(
            zip(("family", "arm", "limit", "observation_price", "stratum", "phase"), key),
            n_seeds=len(rows),
            success_mean=statistics.mean(c["success"] / c["denominator"] for _, c in rows),
            total_cost_mean=statistics.mean(r["weighted_total"] for r, _ in rows) if key[-1] == "all" else None,
        )
        for key, rows in sorted(groups.items())
    ]
    passed = (
        complete
        and lock.get("dev_complete_grid") is True
        and audited
        and violations == 0
        and all(item["passed"] for family in families.values() for item in family.values())
    )
    return dict(
        dev_complete_grid=lock.get("dev_complete_grid") is True,
        spec=SPEC,
        scientific_verdict="PASS" if passed else "NOT_CONFIRMED",
        complete_grid=complete,
        expected_trials=4536,
        actual_trials=len(records),
        missing_trials=len(missing),
        safety_status="audited" if audited else "safety_not_audited",
        hard_violations=violations if audited else None,
        families=families,
        tables=tables,
        comparator_lock=lock,
        final_raw_hash=_hash(sorted(records, key=_key)),
        final_raw_hash_scope="canonical_sorted_analysis_records; physical files bound separately",
    )


aggregate = finalaggregate


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_hashes(paths):
    return {str(Path(path).resolve()): _file_hash(path) for path in paths}


def _write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _markdown(result):
    lines = [
        "# Анализ предзарегистрированного эксперимента",
        "",
        f"Научный вердикт: **{result['scientific_verdict']}**.",
        f"Полнота dev: {result['dev_complete_grid']}; final: {result['actual_trials']}/{result['expected_trials']}; безопасность: {result['safety_status']}.",
        "",
        "Интервалы: односторонние 95%, парные средние 12 seeds; t(0.95,11)=1.7958848187. Неопределённая экономия запрещает PASS.",
        "",
        "## Основные сравнения ID",
        "",
        "| Семейство | Гипотеза | Δ success | Нижняя граница | Экономия | Нижняя граница | Вердикт |",
        "|---|---|---:|---:|---:|---:|---|",
    ]

    def fmt(value):
        return "не определено" if value is None else f"{value:.6f}"

    operation_notes = []
    for family, tests in result["families"].items():
        for name, test in tests.items():
            success, saving = test["success_difference"], test["relative_cost_saving"]
            lines.append(
                f"| {family} | {name} | {fmt(success['mean'])} | {fmt(success['lower'])} | {fmt(saving['mean'] if saving else None)} | {fmt(saving['lower'] if saving else None)} | {test['verdict']} |"
            )
        operation = tests["H_operation"]
        operation_notes.append(
            f"{family}: доля вызовов FULL {operation['application_fraction']:.6f}; составная операция допущена во всех seeds: {operation['learned_composite_all_seeds']}."
        )
    lines.extend(["", *operation_notes, ""])
    lines.extend(
        [
            "",
            "## Все предзаданные страты, бюджеты и цены",
            "",
            "Стоимость — точные восьмые accounting unit, включая обучение. Для prefix/suffix стоимость отдельно не определена, поэтому не приписывается из полной стоимости. OOD announced и hidden показаны отдельно; ID-интервалы к ним не применяются.",
            "",
            "| Семейство | Arm | Бюджет | Цена | Страта | Фаза | Seeds | Success | Полная стоимость /8 |",
            "|---|---|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for row in result["tables"]:
        cells = [str(row[k]) for k in ("family", "arm", "limit", "observation_price", "stratum", "phase", "n_seeds")]
        cells.extend(
            [fmt(row["success_mean"]), fmt(row["total_cost_mean"] / 8 if row["total_cost_mean"] is not None else None)]
        )
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "NOT_CONFIRMED означает, что совместный предзарегистрированный критерий не выполнен или данные неполны; отсутствие значимости не доказывает нулевой эффект.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev", type=Path, nargs="+", help="dev raw JSONL files only")
    mode.add_argument("--final", type=Path, nargs="+", help="final raw JSONL files only")
    parser.add_argument("--lock", type=Path, help="previously frozen comparator_lock.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, help="bind protocol bytes in dev lock")
    parser.add_argument(
        "--bind-file", type=Path, action="append", default=[], help="additional source/freeze binding, repeatable"
    )
    args = parser.parse_args(argv)
    if args.dev:
        if args.lock:
            parser.error("--lock is only valid with --final")
        lock = choose_comparators(collect(args.dev))
        lock["dev_raw_files_sha256"] = _file_hashes(args.dev)
        bound = [
            Path(__file__),
            Path(__file__).with_name("experience_transfer.py"),
            Path(__file__).with_name("_experience_scorer.py"),
            *args.bind_file,
        ]
        if args.protocol:
            bound.append(args.protocol)
        lock["bound_files_sha256"] = _file_hashes(bound)
        args.output.mkdir(parents=True, exist_ok=True)
        destination = args.output / "comparator_lock.json"
        if destination.exists() and json.loads(destination.read_text()) != lock:
            raise ValueError("refusing to overwrite a different comparator lock")
        _write_json(destination, lock)
    else:
        if not args.lock or args.protocol or args.bind_file:
            parser.error("--final requires --lock; protocol/source bindings must be fixed with --dev")
        lock = json.loads(args.lock.read_text())
        if lock.get("dev_complete_grid") is not True or lock.get("actual_dev_trials") != 1512:
            raise ValueError("final CLI requires a complete 1512-trial dev lock")
        if lock.get("source_code_hashes", {}).get(Path(__file__).name) != _file_hash(__file__):
            raise ValueError("aggregation source changed since comparator lock")
        if not lock.get("bound_files_sha256") or not lock.get("dev_raw_files_sha256"):
            raise ValueError("lock lacks source/raw file bindings")
        for path, expected in {**lock["bound_files_sha256"], **lock["dev_raw_files_sha256"]}.items():
            if _file_hash(path) != expected:
                raise ValueError(f"bound source/protocol/dev raw changed: {path}")
        result = finalaggregate(collect(args.final), lock)
        result["final_raw_files_sha256"] = _file_hashes(args.final)
        args.output.mkdir(parents=True, exist_ok=True)
        _write_json(args.output / "analysis.json", result)
        (args.output / "analysis.md").write_text(_markdown(result))


if __name__ == "__main__":
    main()
