"""Paid empirical pool acquisition on pinned external data; no domain certificate.

The host owns gold and splits. Selectors receive models and feature-only options.
Label prices are simulated work units, not measured annotation fees.
"""
from __future__ import annotations

import random
import time

from active_real_data import prepare_dataset
from active_real_models import fit_forest
from solvi.experimental.experience._common import BudgetStop, WorkBudget, canonical, digest, portable
from solvi.experimental.experience.observations import ObservationOption, select_observation_bounded

ARMS = ("RANDOM", "UNCERTAINTY", "QBC_FULL", "QBC_CAPPED", "FULL_LABEL")
PRICES = (128, 8192)
LIMIT = 4000000


class TraceBudget(WorkBudget):
    """Run-length encoded actual debit sequence; no inferred or refunded costs."""

    def __init__(self, limit):
        super().__init__(limit)
        self.trace = []

    def charge(self, kind, units=1):
        before = self.spent
        try:
            super().charge(kind, units)
        finally:
            if self.spent != before:
                if self.trace and self.trace[-1][:2] == [kind, units]:
                    self.trace[-1][2] += 1
                else:
                    self.trace.append([kind, units, 1])


class PredictionCache:
    """Paid pure forecasts; cache is recreated for each immutable forest."""

    def __init__(self, model, fit_index, call_log):
        self.model, self.fit_index, self.call_log = model, fit_index, call_log
        self.cache = {}
        self.last_inputs, self.last_key = None, None

    def leaf(self, tree_index, inputs, budget):
        new_key = inputs is not self.last_inputs
        if not new_key:
            budget.charge("real_predict_key")
            key = self.last_key
        else:
            budget.charge("real_predict_key", len(inputs) + 1)
            key = canonical(inputs)
            self.last_inputs, self.last_key = inputs, key
        budget.charge("real_predict_lookup")
        marker = (tree_index, key)
        hit = marker in self.cache
        if hit:
            leaf = self.cache[marker]
        else:
            leaf = self.model.trees[tree_index].leaf(inputs, budget)
            budget.charge("real_predict_store")
            self.cache[marker] = leaf
        self.call_log.append(dict(fit=self.fit_index, tree=tree_index, point_digest=digest(inputs), hit=hit, new_key=new_key))
        return leaf

    def predict(self, tree_index, inputs, budget):
        n, pos = self.leaf(tree_index, inputs, budget)
        return 2 * pos > n


def full_committee_choice(options, cache, budget):
    """Uncapped full scan with paid metadata, forecasts and comparisons."""
    scored = []
    budget.charge("observation_plan")
    for _ in range(3):
        budget.charge("observation_plan")
    names = set()
    for option in options:
        budget.charge("observation_plan")
        if type(option) is not ObservationOption or option.name in names:
            raise ValueError("invalid observation metadata")
        names.add(option.name)
    for option in options:
        outputs = []
        point = option.inputs
        for tree in range(3):
            budget.charge("observation_plan")
            output = cache.predict(tree, point, budget)
            budget.charge("observation_output")
            outputs.append(output)
        pairs = 0
        for index, left in enumerate(outputs):
            for right in outputs[index + 1:]:
                budget.charge("observation_compare")
                pairs += left != right
        scored.append((option, pairs))
    budget.charge("observation_plan", len(scored))
    best, count = None, 0
    for option, pairs in scored:
        if pairs and option.cost <= budget.remaining and (
                best is None or pairs * best.cost > count * option.cost):
            best, count = option, pairs
    return None if best is None else best.name


def uncertain_choice(options, cache, budget):
    """Leaf probability uncertainty, not hard-vote entropy of the committee."""
    best, numerator, denominator = None, 0, 1
    for option in options:
        budget.charge("real_uncertainty")
        n, pos = cache.leaf(0, option.inputs, budget)
        distance, total = abs(2 * pos - n), n + 2
        budget.charge("real_uncertainty", 3)
        if best is None or distance * denominator < numerator * total:
            best, numerator, denominator = option.name, distance, total
    return best


def real_trial(dataset, cache_dir, seed, split, arm, price, *, cap=1024, emit=None, prepared=None):
    """Run one isolated paid learner; gold is used only by host oracle/evaluator."""
    if arm not in ARMS or price not in PRICES or cap not in (1024, 4096):
        raise ValueError("outside frozen experiment configuration")
    data = prepare_dataset(dataset, cache_dir, seed, split) if prepared is None else prepared
    if data["name"] != dataset or data["seed"] != seed or data["split"] != split:
        raise ValueError("foreign prepared data")
    if len(data["pool"]) != 256 or len(data["test"]) != 512:
        raise ValueError("incomplete benchmark partition")
    started = time.monotonic()
    budget = TraceBudget(LIMIT)
    pool = {str(row["id"]): portable(row["inputs"]) for row in data["pool"]}
    # This dictionary belongs solely to the host oracle; never passed to a model.
    labels = {str(row["id"]): row["label"] for row in data["pool"]}
    pool_order = list(pool)
    receipts, acquired, fits, decisions, predict_calls = [], [], [], [], []
    model, cache, option_by_id = None, None, None
    reason = "label_quota"
    sequence = f"{dataset}:{seed}:{arm}:{price}:{cap}"

    def event(kind, payload):
        if emit is not None:
            emit(kind, portable(payload))

    def read(rid):
        if rid not in pool or any(record["id"] == rid for record in receipts):
            raise ValueError("foreign or repeated label request")
        budget.charge("real_label", price)
        request = f"{sequence}:label:{len(receipts)}"
        event("label_intent", dict(request=request, id=rid, ledger=budget.snapshot()))
        receipt = dict(request=request, id=rid, expected=labels[rid], source=data["source_sha"])
        event("label_receipt", receipt)
        receipts.append(receipt)
        acquired.append(dict(id=rid, inputs=pool[rid], label=receipt["expected"]))

    def train(logical_round):
        nonlocal model, cache
        index = len(fits)
        fit_seed = seed * 100000 + logical_round * 100
        before = budget.snapshot()
        event("fit_intent", dict(index=index, ids=[row["id"] for row in acquired], ledger=before))
        proposed = fit_forest(acquired, budget, seed=fit_seed)
        record = dict(index=index, logical_round=logical_round, acquired_ids=[row["id"] for row in acquired], before=before,
                      after=budget.snapshot(), model=proposed.to_dict())
        event("fit_commit", record)
        fits.append(record)
        model, cache = proposed, PredictionCache(proposed, index, predict_calls)

    try:
        for kind, units in data["preprocess_costs"].items():
            budget.charge(kind, units)
        budget.charge("real_pool_setup", sum(len(point) + 1 for point in pool.values()))
        if arm == "FULL_LABEL":
            for rid in pool_order:
                read(rid)
            train(0)
        else:
            for rid in pool_order[:16]:
                read(rid)
            if arm != "RANDOM":
                train(0)
                budget.charge("real_options", len(pool) * (len(next(iter(pool.values()))) + 1))
                option_by_id = {rid: ObservationOption(rid, point, price) for rid, point in pool.items()}
            while len(receipts) < 48:
                for _ in range(min(4, 48 - len(receipts))):
                    used = {row["id"] for row in receipts}
                    budget.charge("real_pool_filter", len(pool))
                    remaining = [rid for rid in pool_order if rid not in used]
                    options = ()
                    if option_by_id is not None:
                        budget.charge("real_option_route", len(remaining))
                        options = tuple(option_by_id[rid] for rid in remaining)
                    start = budget.snapshot()
                    calls_start = len(predict_calls)
                    choice, policy_reason, pairs = None, "random", 0
                    if arm == "UNCERTAINTY":
                        choice = uncertain_choice(options, cache, budget)
                        policy_reason = "leaf_uncertainty"
                    elif arm == "QBC_FULL":
                        choice = full_committee_choice(options, cache, budget)
                        policy_reason = "full_scan"
                    elif arm == "QBC_CAPPED":
                        selected = select_observation_bounded(
                            (0, 1, 2), options, budget, predict=cache.predict,
                            max_work=cap, reserve=price, result_type=bool,
                        )
                        choice, policy_reason, pairs = selected.name, selected.reason, selected.pairs_separated
                    fallback = choice is None
                    if fallback:
                        budget.charge("real_choice", len(remaining))
                        choice = random.Random(seed + len(receipts)).choice(remaining)
                    if choice not in remaining:
                        raise RuntimeError("policy chose an acquired or unknown row")
                    decision = dict(query=len(receipts), fit=len(fits) - 1, id=choice, fallback=fallback,
                                    reason=policy_reason, pairs=pairs, before=start, after=budget.snapshot(),
                                    calls=[calls_start, len(predict_calls)])
                    decisions.append(decision)
                    event("observation_choice", decision)
                    read(choice)
                if arm != "RANDOM":
                    train((len(receipts) - 16) // 4)
            if arm == "RANDOM":
                train(8)
    except BudgetStop:
        reason = "budget"
    frozen = dict(reason=reason, model=None if model is None else model.to_dict(), ledger=budget.snapshot(),
                  labels=len(receipts), fit_count=len(fits))
    event("frozen", frozen)
    # No test input or label has entered the learner. No feedback/refit follows.
    predictions = []
    for row in data["test"]:
        rid = str(row["id"])
        if model is None:
            predictions.append(dict(id=rid, expected=row["label"], outcome="abstain"))
            continue
        try:
            value = model.predict(row["inputs"], budget)
        except BudgetStop:
            predictions.append(dict(id=rid, expected=row["label"], outcome="budget"))
        else:
            predictions.append(dict(id=rid, value=value, expected=row["label"],
                                    outcome="correct" if type(value) is bool and value == row["label"] else "wrong"))
    return dict(tier="real", dataset=dataset, seed=seed, split=split, arm=arm, price=price, cap=cap,
                data_sha=data["source_sha"], partition_digest=digest(data["group_ids"]),
                thresholds=data["thresholds"], rawcounts=data["rawcounts"],
                quantized_collisions=data["quantized_test_rows_matching_pool"],
                receipts=receipts, fits=fits, decisions=decisions, predict_calls=predict_calls,
                frozen=frozen, predictions=predictions, budget=budget.snapshot(), trace=budget.trace,
                wall_seconds=f"{time.monotonic() - started:.9f}")


__all__ = ["ARMS", "PRICES", "LIMIT", "real_trial", "TraceBudget", "PredictionCache"]
