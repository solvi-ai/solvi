"""Offline, selective prediction evaluation; no semantic-validity claim is made."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import beta

ARMS = ("direct", "plain", "solvi")
GRID = (0., .5, .7, .8, .9, .95, .99, 1.)
DELTA = .05
MAX_ERROR = .05


def upper_error(errors, n, alpha):
    """One-sided exact binomial upper bound; undefined with no answers."""
    if n == 0:
        return None
    if errors == n:
        return 1.0
    return float(beta.ppf(1 - alpha, errors + 1, n - errors))


def load_rows(path):
    raw = Path(path).read_bytes()
    rows = []
    ids = set()
    split_values = {s: {k: set() for k in ("group", "id", "texthash")} for s in ("calibration", "eval")}
    for line_no, line in enumerate(raw.splitlines(), 1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("row must be an object")
            for key in ("id", "group"):
                if not isinstance(row.get(key), str) or not row[key]:
                    raise ValueError(f"{key} must be a nonempty string")
            if row["id"] in ids:
                raise ValueError("duplicate id")
            ids.add(row["id"])
            split = row.get("split")
            if split not in split_values or row.get("gold") not in ("approve", "reject"):
                raise ValueError("invalid split or gold")
            if "variants" in row and (not isinstance(row["variants"], dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in row["variants"].items())):
                raise ValueError("variants must map strings to strings")
            for key in ("group", "id", "texthash"):
                if key in row:
                    if not isinstance(row[key], str) or not row[key]:
                        raise ValueError(f"{key} must be a nonempty string")
                    split_values[split][key].add(row[key])
            arms = row.get("arms")
            if not isinstance(arms, dict) or set(arms) != set(ARMS):
                raise ValueError("arms must contain exactly direct, plain, solvi")
            for name in ARMS:
                arm = arms[name]
                if not isinstance(arm, dict) or "answer" not in arm or (arm["answer"] is not None and not isinstance(arm["answer"], str)):
                    raise ValueError(f"invalid {name} answer")
                for key in ("confidence", "latency_s", "cost_usd"):
                    value = arm.get(key)
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                        raise ValueError(f"invalid {name} {key}")
                    if key == "confidence" and value > 1:
                        raise ValueError(f"invalid {name} confidence")
            rows.append(row)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"line {line_no}: {exc}") from exc
    if not rows or any(not any(r["split"] == s for r in rows) for s in split_values):
        raise ValueError("both calibration and eval splits must be nonempty")
    for key in ("group", "id", "texthash"):
        if split_values["calibration"][key] & split_values["eval"][key]:
            raise ValueError(f"cross-split {key} overlap")
    return rows, hashlib.sha256(raw).hexdigest()


def answered(row, arm, threshold):
    a = row["arms"][arm]
    return threshold is not None and a["answer"] not in (None, "") and a["confidence"] >= threshold


def metrics(rows, arm, threshold, alpha):
    selected = [r for r in rows if answered(r, arm, threshold)]
    n, count = len(rows), len(selected)
    errors = sum(r["arms"][arm]["answer"] != r["gold"] for r in selected)
    latencies = [r["arms"][arm]["latency_s"] for r in rows]
    return {"n": n, "answered": count, "errors": errors, "coverage": count / n if n else None,
            "conditional_error": errors / count if count else None,
            "joint_risk": errors / n if n else None, "upper_error": upper_error(errors, count, alpha),
            "total_cost_usd": sum(r["arms"][arm]["cost_usd"] for r in rows),
            "latency_s": {f"p{p}": float(np.percentile(latencies, p)) if n else None for p in (50, 90, 95, 99)}}


def select_threshold(rows, arm):
    candidates = []
    for threshold in GRID:
        m = metrics(rows, arm, threshold, DELTA / (len(GRID) * len(ARMS)))
        if m["upper_error"] is not None and m["upper_error"] <= MAX_ERROR:
            candidates.append((m["answered"], threshold))
    return max(candidates, key=lambda c: (c[0], -c[1]))[1] if candidates else None


def evaluate(rows, digest, bootstrap_samples=10000, seed=141):
    calibration = [r for r in rows if r["split"] == "calibration"]
    evaluation = [r for r in rows if r["split"] == "eval"]
    thresholds = {a: select_threshold(calibration, a) for a in ARMS}
    results = {}
    for arm in ARMS:
        threshold = thresholds[arm]
        m = metrics(evaluation, arm, threshold, DELTA / len(ARMS))
        m["error_gate_passed"] = m["upper_error"] is not None and m["upper_error"] <= MAX_ERROR
        m["threshold"] = threshold
        m["ungated_secondary"] = metrics(evaluation, arm, 0., DELTA / len(ARMS))
        raw = m["ungated_secondary"]
        raw["accuracy_on_answered"] = (1 - raw["conditional_error"]) if raw["conditional_error"] is not None else None
        raw["accuracy_all_attempts"] = (raw["answered"] - raw["errors"]) / raw["n"] if raw["n"] else None
        raw["certified"] = False
        m["calibration"] = metrics(calibration, arm, threshold, DELTA / (len(GRID) * len(ARMS)))
        m["groups"] = {g: metrics([r for r in evaluation if r["group"] == g], arm, threshold, DELTA / len(ARMS)) for g in sorted({r["group"] for r in evaluation})}
        m["variant_slices"] = {}
        for field in sorted({key for r in evaluation for key in r.get("variants", {})}):
            m["variant_slices"][field] = {}
            for value in sorted({r["variants"][field] for r in evaluation if field in r.get("variants", {})}):
                part = [r for r in evaluation if r.get("variants", {}).get(field) == value]
                m["variant_slices"][field][value] = {
                    "interpretation": "descriptive_only",
                    "selected": metrics(part, arm, threshold, DELTA / len(ARMS)),
                    "ungated_secondary": metrics(part, arm, 0., DELTA / len(ARMS))}
        results[arm] = m
    groups = sorted({r["group"] for r in evaluation})
    grouped = [[r for r in evaluation if r["group"] == g] for g in groups]
    comparisons = {}
    rng = np.random.default_rng(seed)
    for baseline in ("direct", "plain"):
        if not (results["solvi"]["error_gate_passed"] and results[baseline]["error_gate_passed"]):
            comparisons[baseline] = {"available": False, "reason": "both arms must pass the eval error gate"}
            continue
        sums = np.array([[sum(answered(r, "solvi", thresholds["solvi"]) - answered(r, baseline, thresholds[baseline]) for r in group), len(group)] for group in grouped], dtype=float)
        indices = rng.integers(0, len(groups), size=(bootstrap_samples, len(groups)))
        draws = sums[indices].sum(axis=1)
        differences = draws[:, 0] / draws[:, 1]
        comparisons[baseline] = {"available": True, "interpretation": "exploratory", "coverage_difference": results["solvi"]["coverage"] - results[baseline]["coverage"],
                                 "lower_one_sided_bound": float(np.quantile(differences, DELTA / 2)),
                                 "bonferroni_alpha": DELTA / 2, "bootstrap_samples": bootstrap_samples}
    repeated_groups = any(len(g) > 1 for g in grouped)
    # The input format does not establish independent sampling, even when group IDs are unique.
    reasons = []
    if len(evaluation) < 1000:
        reasons.append("eval has fewer than 1000 documents")
    reasons.append("independent sampling is unverified by the input schema")
    if repeated_groups:
        reasons.append("multiple documents share an eval group")
    if not results["solvi"]["error_gate_passed"]:
        reasons.append("solvi error gate failed or is unavailable")
    if results["solvi"]["coverage"] is None or results["solvi"]["coverage"] < .5:
        reasons.append("solvi coverage is below 0.5 or unavailable")
    for baseline, comparison in comparisons.items():
        if not comparison["available"] or comparison["lower_one_sided_bound"] <= 0:
            reasons.append(f"paired coverage lower bound versus {baseline} is unavailable or not positive")
    return {"primary_acceptance": {"passed": not reasons, "reasons": reasons}, "manifest": {"input_sha256": digest, "threshold_grid": list(GRID), "delta": DELTA, "max_error": MAX_ERROR,
                         "calibration_alpha_per_test": DELTA / (len(GRID) * len(ARMS)), "eval_alpha_per_arm": DELTA / len(ARMS),
                         "bootstrap_seed": seed, "bootstrap_unit": "group", "eval_groups": len(groups),
                         "within_group_non_independence_possible": repeated_groups,
                         "inferential_validity": False, "independence_assumption": "unverified",
                         "semantic_claim": "unsupported: exact string matching only"},
            "arms": results, "paired_coverage_vs_baseline": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, digest = load_rows(args.rows)
    report = evaluate(rows, digest)
    Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
