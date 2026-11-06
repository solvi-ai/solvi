"""Recompute pre-registered costs and correctness gates from retained trials."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import hashlib
from pathlib import Path
import sys

# Standalone CLI must select this lane before importing either solvi or a
# benchmark which imports it; an editable baseline install must not win.
_LANE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_LANE / "src"))

from experience_v2 import FAMILIES, canonical, digest, roles  # noqa: E402 - local lane must precede editable installs
from experience_v2_run import portable_measurements, trial_id, trial_plan  # noqa: E402 - same source pin


def tenth_reduction(old, new):
    return type(old) is int and type(new) is int and old > 0 and new >= 0 and new * 10 <= old * 9


def validate_s1(result):
    errors = []
    if [cell["cell"] for cell in result.get("cells", [])] != ["A1", "A2", "A3", "B1", "B2", "B3"]:
        return ["invalid S1 cells/order"]
    for version in ("old", "new"):
        previous = 0
        for cell in result["cells"]:
            item = cell["pair"][version]
            if (
                any(type(item[key]) is not int or item[key] < 0 for key in ("start", "end", "spent", "remaining"))
                or item["start"] != previous
                or item["end"] - item["start"] != item["spent"]
                or item["spent"] > 100000
                or item["end"] > 200000
                or item["remaining"] != min(100000 - item["spent"], 200000 - item["end"])
            ):
                errors.append(f"{version}/{cell['cell']}: invalid work interval")
            previous = item["end"]
        ledger = result.get("ledgers", {}).get(version, {})
        counts = ledger.get("counts", {})
        if (
            not counts
            or any(type(v) is not int or v < 0 for v in counts.values())
            or type(ledger.get("spent")) is not int
            or ledger["spent"] != previous
            or sum(counts.values()) != previous
        ):
            errors.append(f"{version}: invalid final ledger")
    return errors


def normalized_ast(value, wire):
    if value is None:
        return None
    tree = value["ast"] if isinstance(value, dict) else value

    def visit(node):
        if node[0] == "input" and wire:
            return ["input", int(node[1][1:])]
        if node[0] == "input":
            names = (
                ("stock", "reserved", "damaged")
                if node[1] in ("stock", "reserved", "damaged")
                else (
                    ("requested", "available", "expired")
                    if node[1] in ("requested", "available", "expired")
                    else ("a", "b", "c", "d")
                )
            )
            return ["input", names.index(node[1])]
        if node[0] == "const":
            return list(node)
        return [node[0], *[visit(child) for child in node[1:]]]

    return visit(tree)


def summarize(out, phase, manifest):
    from experience_v2_audit import audit_trial, evaluate, truth

    out = Path(out)
    expected = {trial_id(spec): spec for spec in trial_plan(phase)}
    records, previous, errors = {}, "", []
    config_hash = digest({"phase": phase, "manifest": digest(manifest)})
    for line in (out / "trials.jsonl").read_text().splitlines():
        record = json.loads(line)
        checksum = record.pop("hash")
        if (
            digest(record) != checksum
            or record["previous"] != previous
            or record["config"] != config_hash
            or record["trial"] in records
        ):
            raise ValueError("invalid or duplicate trial log")
        previous = checksum
        identity = record["trial"]
        if identity not in expected or record["payload"]["spec"] != expected[identity]:
            raise ValueError("unexpected trial specification")
        records[identity] = record
    missing = sorted(set(expected) - set(records))
    if missing:
        errors.append(f"missing {len(missing)} mandatory trials")
    costs = {"old": {"s1": 0, "s2": 0}, "new": {"s1": 0, "s2": 0}}
    exhaustions = {"old": Counter(), "new": Counter()}
    operational, pairs, audits, planner = {}, defaultdict(dict), {}, {}
    totals = defaultdict(lambda: Counter())
    useful = defaultdict(lambda: Counter())
    census = Counter()
    for identity, record in records.items():
        spec = expected[identity]
        result = record["payload"].get("result", {})
        census[record["status"]] += 1
        if record["status"] in ("censored", "interrupted", "error") or not result:
            errors.append(f"{identity}: health {record['status']}")
            continue
        if spec["kind"] == "s1":
            validation = validate_s1(result)
            errors.extend(f"{identity}: {error}" for error in validation)
            if validation:
                continue
            planner[(spec["seed"], spec["profile"], spec["wire"])] = result
            for cell in result["cells"]:
                pair = cell["pair"]
                for version in ("old", "new"):
                    item = pair[version]
                    if not spec["wire"]:
                        costs[version]["s1"] += item["spent"]
                        exhaustions[version]["s1"] += int(item["status"] == "budget")
                    reference = item.get("reference")
                    if reference is None:
                        errors.append(f"{identity}/{cell['cell']}: missing reference")
                    elif item["status"] != "budget" and any(
                        item[key] != reference[key] for key in ("status", "name", "cost")
                    ):
                        errors.append(f"{identity}/{cell['cell']}/{version}: reference mismatch")
                if pair["old"]["status"] != "budget" and pair["new"]["status"] == "budget":
                    errors.append(f"{identity}/{cell['cell']}: new exhaustion")
        elif spec["kind"] == "operational":
            if any(canonical(result.get(key)) != canonical(spec[key]) for key in ("seed", "family", "wire", "arm")):
                errors.append(f"{identity}: result metadata differs from assigned trial")
                continue
            raw_path = out / identity / "operational" / "result.json"
            if not raw_path.exists():
                errors.append(f"{identity}: missing trial-local raw result")
                continue
            raw = json.loads(raw_path.read_text())
            if any(canonical(portable_measurements(value)) != canonical(result.get(key)) for key, value in raw.items()):
                errors.append(f"{identity}: terminal result differs from trial-local raw data")
                continue
            key = (spec["seed"], spec["family"], spec["wire"], spec["arm"])
            operational[key] = result
            audit = audit_trial(out / identity / "operational", result,
                                expected_engine_sha256=manifest["source"]["src/solvi/experimental/experience/programs.py"])
            audits[identity] = audit
            errors.extend(f"{identity}: {error}" for error in audit["errors"])
            if not spec["wire"]:
                totals[spec["arm"]]["spent"] += result["spent"]
                for group in ("R", "H"):
                    totals[spec["arm"]]["correct_" + group] += audit["correct_" + group]
                    if spec["arm"] == "NEW_CHECKED":
                        useful[spec["family"]][group] += audit["useful_" + group]
                if spec["arm"] in ("OLD_EMPIRICAL", "NEW_EMPIRICAL"):
                    version = spec["arm"].split("_")[0].lower()
                    search = result.get("search")
                    if search is None:
                        errors.append(f"{identity}: missing paid search prefix")
                    else:
                        if (
                            type(search["spent"]) is not int
                            or not 0 <= search["spent"] <= 100000
                            or type(result["spent"]) is not int
                            or result["spent"] < search["spent"]
                            or sum(result["counts"].values()) != result["spent"]
                        ):
                            errors.append(f"{identity}: invalid search/total ledger")
                            continue
                        costs[version]["s2"] += search["spent"]
                        exhaustions[version]["s2"] += int(search["reason"] == "budget_exhausted")
                        pairs[(spec["seed"], spec["family"])][version] = search
    for key, pair in pairs.items():
        if set(pair) != {"old", "new"}:
            errors.append(f"{key}: incomplete S2 pair")
            continue
        old, new = pair["old"], pair["new"]
        if old["reason"] != "budget_exhausted":
            if any(canonical(old[k]) != canonical(new[k]) for k in ("ast", "tried", "reason")):
                errors.append(f"{key}: S2 first-match mismatch")
        elif new["reason"] == "found":
            seed, family = key
            names = FAMILIES[family][0]
            grammar = dict(
                inputs=list(names),
                constants=[0, 1],
                operators=list(FAMILIES[family][2]),
                max_depth=4,
                max_nodes=7,
                max_abs=32,
            )
            try:
                for row in roles(seed, family)["proposal"]:
                    actual = evaluate(new["ast"]["ast"], dict(zip(names, row)), grammar, [0])
                    if canonical(actual) != canonical(truth(family, row)):
                        raise ValueError("proposal mismatch")
            except (ValueError, TypeError, KeyError, IndexError) as error:
                errors.append(f"{key}: newly completed search fails independent proposal check: {error}")
    wire_errors = []
    for (seed, profile, wire), result in planner.items():
        if wire:
            continue
        changed = planner.get((seed, profile, True))
        if changed is None:
            wire_errors.append(f"{seed}/{profile}: missing planner wire audit")
            continue
        for left_cell, right_cell in zip(result["cells"], changed["cells"]):
            for version in ("old", "new"):
                left, right = left_cell["pair"][version], right_cell["pair"][version]
                if "budget" in (left["status"], right["status"]):
                    continue
                name = right["name"]
                decoded = None if name is None else "p" + str(int(name[1:]))
                if (left["status"], left["name"], left["cost"]) != (right["status"], decoded, right["cost"]):
                    wire_errors.append(f"{seed}/{profile}/{left_cell['cell']}/{version}: planner wire changed")
    for (seed, family, wire, arm), result in operational.items():
        if wire:
            continue
        changed = operational.get((seed, family, True, arm))
        if changed is None:
            wire_errors.append(f"{seed}/{family}/{arm}: missing wire trial")
            continue
        if result["status"] == "budget" or changed["status"] == "budget":
            continue  # Explicitly retain budget sensitivity rather than call it invariance.
        if result["status"] != changed["status"] or [r["status"] for r in result["transfers"]] != [
            r["status"] for r in changed["transfers"]
        ]:
            wire_errors.append(f"{seed}/{family}/{arm}: semantic status changed")
        left, right = result.get("search"), changed.get("search")
        if (
            left
            and right
            and canonical(normalized_ast(left["ast"], False)) != canonical(normalized_ast(right["ast"], True))
        ):
            wire_errors.append(f"{seed}/{family}/{arm}: first AST changed")
    errors.extend(wire_errors)

    def gain(stage=None):
        old = costs["old"][stage] if stage else sum(costs["old"].values())
        new = costs["new"][stage] if stage else sum(costs["new"].values())
        return None if not old else 1 - new / old

    cost_gain = gain()
    no_extra_exhaustion = all(exhaustions["new"][stage] <= exhaustions["old"][stage] for stage in ("s1", "s2"))
    denominator = (1 if phase == "smoke" else 12) * 32
    passed_families = [
        family for family, counts in useful.items() if (counts["R"] + counts["H"]) * 10 >= denominator * 9
    ]
    return dict(
        phase=phase,
        manifest_hash=digest(manifest),
        scorer_provenance=dict(
            lane=str(_LANE), python_executable=sys.executable,
            solvi_origin=str(Path(sys.modules["solvi"].__file__).resolve()),
            source_hashes={str(path.relative_to(_LANE)): hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [Path(__file__).resolve(), _LANE / "benchmarks/experience_v2_audit.py",
                                        _LANE / "benchmarks/experience_v2.py", _LANE / "benchmarks/experience_v2_run.py",
                                        _LANE / "src/solvi/experimental/experience/programs.py"]},
            actor_engine_sha256=manifest.get("source", {}).get("src/solvi/experimental/experience/programs.py"),
        ),
        expected=len(expected),
        observed=len(records),
        census=dict(census),
        errors=errors,
        costs=costs,
        gains={stage: gain(stage) for stage in ("s1", "s2")},
        combined_gain=cost_gain,
        exhaustions={k: dict(v) for k, v in exhaustions.items()},
        total_strategy_costs={k: dict(v) for k, v in totals.items()},
        useful={k: dict(v) for k, v in useful.items()},
        useful_denominator_per_family=denominator,
        Hcost="INDETERMINATE"
        if errors
        else (
            "PASS"
            if tenth_reduction(sum(costs["old"].values()), sum(costs["new"].values())) and no_extra_exhaustion
            else "FAIL"
        ),
        Huseful="INDETERMINATE" if errors else ("PASS" if len(passed_families) >= 2 else "FAIL"),
        wire_errors=wire_errors,
        audits=audits,
        note="Hsafety/Hrecovery are separate engineering audits; component outcomes do not evaluate curriculum scheduling.",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--phase", choices=("smoke", "development", "final"), required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    summary = summarize(args.out, args.phase, json.loads(Path(args.manifest).read_text()))
    Path(args.out, "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "audits"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
