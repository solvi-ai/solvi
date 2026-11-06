"""Independent acquisition/debit replay; CART reference reused from prior scorer.

Does not call the actor, public planner, fit_forest or model.predict. Host data
preparation is shared and is separately checked by the role/exposure audit.
"""

from __future__ import annotations
from fractions import Fraction
from active_real_score import Replay, same, require
from spam_budget_data import prepare
from solvi.experimental.experience._common import canonical, digest

ARMS = ("RANDOM48", "RANDOM40", "RANDOM32", "MIXED40", "MIXED32", "FULL_LABEL")


class BudgetReplay(Replay):
    def __init__(self, row, data):
        super().__init__(row, data)
        self.plan_complete = False

    def run(self):
        result = super().run()
        require((self.row["plan"] is not None) == self.plan_complete, "unexplained plan record")
        return result

    def label(self, rid):
        self.trace.pay("real_route")
        self.trace.pay("real_label", self.price)
        n = len(self.ids)
        expected = dict(
            request=f"{self.row['dataset']}:{self.seed}:{self.row['split']}:{self.arm}:{self.price}:label:{n}",
            id=rid,
            expected=self.pool[rid]["label"],
            source=self.data["source_sha"],
        )
        same(self.row["receipts"][n], expected, "paid receipt identity/target/source")
        require(rid not in self.ids, "repeated purchase")
        self.ids.append(rid)

    def stratified(self, pilot, candidates):
        t = self.trace
        p = self.width
        n = len(pilot)
        q = len(candidates)
        k = 8
        t.pay("stratified_validate")
        t.pay("stratified_validate", n + q)
        t.pay("stratified_decode", sum(len(canonical(self.pool[i]["inputs"])) for i in pilot + candidates))
        t.pay("stratified_validate", (n + q) * (p + 1))
        t.pay("stratified_histogram", n * p + 8 * p)
        t.pay("stratified_score", p * 16)
        scores = []
        for index, feature in enumerate(self.names):
            values = sorted(self.pool[i]["inputs"][feature] for i in pilot)
            threshold = values[n // 2]
            left = [i for i in pilot if self.pool[i]["inputs"][feature] < threshold]
            right = [i for i in pilot if self.pool[i]["inputs"][feature] >= threshold]
            if min(len(left), len(right)) < 4:
                continue
            gap = abs(
                Fraction(sum(self.pool[i]["label"] for i in left) + 1, len(left) + 2)
                - Fraction(sum(self.pool[i]["label"] for i in right) + 1, len(right) + 2)
            )
            if gap:
                scores.append((gap, index, threshold))
        t.pay("stratified_select", 4 * p + 4 * q + 4 * k)
        scores = sorted(scores, key=lambda v: (-v[0], v[1]))[:2]
        if not scores or len({self.pool[i]["label"] for i in pilot}) == 1:
            return dict(names=candidates[:k], features=[], thresholds=[], reason="random_prefix")
        chosen = []
        codes = {
            i: sum(
                (self.pool[i]["inputs"][self.names[index]] >= threshold) << bit
                for bit, (_, index, threshold) in enumerate(scores)
            )
            for i in candidates
        }
        # Reference scans ranks in each cell; actor builds queues/cursors.
        buckets = [[i for i in candidates if codes[i] == code] for code in range(2 ** len(scores))]
        for rank in range(q):
            for bucket in buckets:
                if rank < len(bucket):
                    chosen.append(bucket[rank])
                if len(chosen) == k:
                    break
            if len(chosen) == k:
                break
        return dict(
            names=chosen,
            features=[self.names[i] for _, i, _ in scores],
            thresholds=[th for _, _, th in scores],
            reason="stratified",
        )

    def acquisition(self):
        t = self.trace
        order = list(self.pool)
        p = self.width
        for kind, units in self.data["preprocess_costs"].items():
            t.pay(kind, units)
        t.pay("real_pool_setup", (p + 1) * 256)
        t.pay("real_order", 256)
        quota = 256 if self.arm == "FULL_LABEL" else int(self.arm[-2:])
        pilot = quota - 8 if self.arm.startswith("MIXED") else quota
        for rid in order[:pilot]:
            self.label(rid)
        if self.arm.startswith("MIXED"):
            t.pay("real_examples", (p + 1) * pilot)
            t.pay("real_options", (p + 1) * 64)
            saved = self.row["plan"]
            before = t.spent
            if saved is not None:
                t.snapshot(saved["before"])
            result = self.stratified(self.ids.copy(), order[pilot : pilot + 64])
            require(saved is not None, "missing completed plan")
            same(
                {key: saved[key] for key in ("names", "features", "thresholds", "reason")},
                result,
                "independent stratum plan",
            )
            require(
                saved.keys() == {"names", "features", "thresholds", "reason", "before", "after"}, "unexpected plan keys"
            )
            t.snapshot(saved["after"])
            require(t.spent >= before, "nonmonotonic plan ledger")
            self.plan_complete = True
            t.pay("real_candidate_index", 64)
            self.fallbacks = 8 if result["reason"] == "random_prefix" else 0
            for rid in result["names"]:
                self.label(rid)
        else:
            require(self.row["plan"] is None, "unexpected plan on random/all-label control")
        logical = 0 if self.arm == "FULL_LABEL" else 8
        if self.row["fits"]:
            require(self.row["fits"][0]["seed"] == self.seed * 100000 + logical * 100, "fit seed")
        self.train(logical)


def audit_budget(row, cache_dir, roles_path, *, prepared=None):
    fields = {
        "tier",
        "dataset",
        "seed",
        "split",
        "arm",
        "price",
        "quota",
        "data_sha",
        "partition_digest",
        "thresholds",
        "rawcounts",
        "quantized_collisions",
        "receipts",
        "fits",
        "plan",
        "frozen",
        "predictions",
        "budget",
        "trace",
        "wall_seconds",
    }
    try:
        require(type(row) is dict and row.keys() == fields, "unexpected trial schema")
        require(
            row["tier"] == "budget_real"
            and row["dataset"] in ("banknote", "spambase", "wine_red")
            and type(row["seed"]) is int
            and row["split"] in ("dev", "final")
            and row["arm"] in ARMS
            and type(row["price"]) is int
            and row["price"] in (128, 8192),
            "invalid trial identity",
        )
        quota = 256 if row["arm"] == "FULL_LABEL" else int(row["arm"][-2:])
        require(type(row["quota"]) is int and row["quota"] == quota, "invalid quota")
        require(
            type(row["wall_seconds"]) is str
            and row["wall_seconds"].isascii()
            and row["wall_seconds"].replace(".", "", 1).isdigit(),
            "invalid elapsed time",
        )
        data = (
            prepare(row["dataset"], cache_dir, row["seed"], row["split"], roles_path) if prepared is None else prepared
        )
        same(
            [
                row[k]
                for k in (
                    "dataset",
                    "seed",
                    "split",
                    "data_sha",
                    "partition_digest",
                    "thresholds",
                    "rawcounts",
                    "quantized_collisions",
                )
            ],
            [
                data["name"],
                data["seed"],
                data["split"],
                data["source_sha"],
                digest(data["group_ids"]),
                data["thresholds"],
                data["rawcounts"],
                data["quantized_test_rows_matching_pool"],
            ],
            "source/split/transform identity",
        )
        pool, test, hold = (set(data["group_ids"][key]) for key in ("pool", "test", "holdout"))
        require(
            len(pool) == 256
            and len(test) == len(hold) == 512
            and not pool & (test | hold)
            and (test == hold if row["split"] == "final" else not test & hold),
            "overlapping roles",
        )
        require(
            len(data["pool"]) == 256
            and len(data["test"]) == 512
            and len({str(r["id"]) for r in data["pool"] + data["test"]}) == 768,
            "row denominator",
        )
        # Reuse the independently implemented exact CART reference, not the actor.
        adapter = dict(row, decisions=[], predict_calls=[])
        return BudgetReplay(adapter, data).run()
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("invalid trial structure: " + str(error)) from error
