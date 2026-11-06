"""Independent packed-policy, purchase and debit replay; no actor/planner calls.

The exact CART reference is shared with the earlier independent scorer. Data
parsing/quantization is shared with the host and audited separately by roles.
"""

from fractions import Fraction as F
from active_real_score import Replay, same, require
from spam_budget_score import audit_budget
from solvi.experimental.experience._common import digest

ARMS = ("RANDOM48", "RANDOM40", "FULL_LABEL", "LEGACY40", "MEDIAN40", "CUT40", "ADAPT40")
ACTIVE = ("MEDIAN40", "CUT40", "ADAPT40")


def preparation(counts, legacy=False):
    extra = (
        ("real_examples", "real_options", "real_candidate_index")
        if legacy
        else ("real_frame_route", "real_pilot_labels", "real_candidate_index")
    )
    return sum(v for k, v in counts.items() if k.startswith("stratified_" if legacy else "packed_") or k in extra)


class PackedReplay(Replay):
    def __init__(self, row, data, fold):
        super().__init__(dict(row, decisions=[], predict_calls=[]), data)
        self.fold = fold
        self.started = False
        self.initialized = False
        self.steps = 0

    def label(self, rid):
        self.trace.pay("real_route")
        self.trace.pay("real_label", self.price)
        n = len(self.ids)
        require(rid not in self.ids, "duplicate paid label")
        same(
            self.row["receipts"][n],
            dict(
                request=f"{self.row['dataset']}:{self.seed}:{self.row['split']}:{self.arm}:{self.price}:label:{n}",
                id=rid,
                expected=self.pool[rid]["label"],
                source=self.data["source_sha"],
            ),
            "source-bound paid receipt",
        )
        self.ids.append(rid)

    def initialize(self, pilot, candidates):
        t = self.trace
        p = self.width
        n = len(pilot)
        q = len(candidates)
        N = n + q
        mode = "median" if self.arm == "MEDIAN40" else "all"
        t.pay("real_frame_route", N)
        t.pay("packed_frame_meta")
        t.pay("packed_frame_meta", 2 * N)
        t.pay("packed_frame_schema", p)
        t.pay("packed_frame_fields", 2 * N * p)
        t.pay("real_pilot_labels", n)
        t.pay("packed_validate")
        t.pay("packed_pilot", n)
        t.pay("packed_hist", n * p + 8 * p)
        t.pay("packed_cuts", (16 if mode == "median" else 98) * p)
        ranked = []
        for index, feature in enumerate(self.names):
            values = sorted(self.pool[i]["inputs"][feature] for i in pilot)
            cuts = (values[n // 2],) if mode == "median" else range(1, 8)
            best = None
            for cut in cuts:
                left = [i for i in pilot if self.pool[i]["inputs"][feature] < cut]
                right = [i for i in pilot if self.pool[i]["inputs"][feature] >= cut]
                if min(len(left), len(right)) < 4:
                    continue
                gap = abs(
                    F(sum(self.pool[i]["label"] for i in left) + 1, len(left) + 2)
                    - F(sum(self.pool[i]["label"] for i in right) + 1, len(right) + 2)
                )
                if gap and (best is None or gap > best[0]):
                    best = (gap, index, cut)
            if best is not None:
                ranked.append(best)
        t.pay("packed_top2", 4 * p)
        selected = (
            sorted(ranked, key=lambda v: (-v[0], v[1]))[:2] if len({self.pool[i]["label"] for i in pilot}) > 1 else []
        )
        k = len(selected)
        t.pay("packed_cells", 2 * N * k if k else N)
        t.pay("packed_queues", N)

        def code(i):
            return sum(
                (self.pool[i]["inputs"][self.names[f]] >= cut) << bit for bit, (_, f, cut) in enumerate(selected)
            )

        queues = [[i for i in candidates if code(i) == c] for c in range(2**k)]
        counts = [sum(code(i) == c for i in pilot) for c in range(2**k)]
        positives = [sum(self.pool[i]["label"] for i in pilot if code(i) == c) for c in range(2**k)]
        features = [self.names[f] for _, f, _ in selected]
        thresholds = [cut for _, _, cut in selected]
        t.pay("real_candidate_index", q)
        return queues, counts, positives, features, thresholds

    def acquisition(self):
        r = self.row
        t = self.trace
        order = list(self.pool)
        p = self.width
        for kind, units in self.data["preprocess_costs"].items():
            t.pay(kind, units)
        t.pay("real_pool_setup", (p + 1) * 256)
        t.pay("real_order", 256)
        quota = 256 if self.arm == "FULL_LABEL" else int(self.arm[-2:])
        pilot = 32 if self.arm in ACTIVE else quota
        for rid in order[:pilot]:
            self.label(rid)
        if self.arm in ACTIVE:
            self.started = True
            c = r["controller"]
            require(type(c) is dict, "missing controller intent")
            strategy = "adaptive" if self.arm == "ADAPT40" else "round_robin"
            mode = "median" if self.arm == "MEDIAN40" else "all"
            same([c[k] for k in ("strategy", "cut_mode", "max_steps")], [strategy, mode, 8], "controller config")
            t.snapshot(c["init_before"])
            queues, counts, pos, features, thresholds = self.initialize(self.ids.copy(), order[32:96])
            self.initialized = True
            reason = "stratified" if features else "random_prefix"
            source = f"{r['dataset']}:{self.data['source_sha']}:{self.fold}:{r['split']}:{self.seed}"
            same(
                [c[k] for k in ("source", "features", "thresholds", "reason")],
                [source, features, thresholds, reason],
                "controller initialized state",
            )
            t.snapshot(c["init_after"])
            positions = [0] * len(queues)
            cursor = 0
            self.fallbacks = 8 if not features else 0
            for index in range(8):
                step = c["steps"][index] if index < len(c["steps"]) else None
                if step is not None:
                    t.snapshot(step["before"])
                t.pay("packed_choose", 8 * len(queues) + 1)
                if strategy == "round_robin":
                    cell = next(
                        c
                        for c in [(cursor + i) % len(queues) for i in range(len(queues))]
                        if positions[c] < len(queues[c])
                    )
                else:
                    eligible = [c for c in range(len(queues)) if positions[c] < len(queues[c])]

                    def priority(c):
                        return F(
                            (len(queues[c]) - positions[c]) * (pos[c] + 1) * (counts[c] - pos[c] + 1),
                            (counts[c] + 2) ** 2 * (counts[c] + 1),
                        )

                    cell = max(eligible, key=lambda c: (priority(c), -c))
                rid = queues[cell][positions[cell]]
                if step is not None:
                    same([step["index"], step["id"]], [index, rid], "chosen row/sequence")
                    t.snapshot(step["after_choose"])
                self.label(rid)
                t.pay("packed_observe", 6)
                counts[cell] += 1
                pos[cell] += self.pool[rid]["label"]
                positions[cell] += 1
                cursor = (cell + 1) % len(queues)
                require(step is not None, "missing completed observation step")
                same(step["observed"], self.pool[rid]["label"], "feedback actual paid label")
                t.snapshot(step["after_observe"])
                require(
                    step.keys() == {"index", "before", "id", "after_choose", "observed", "after_observe"},
                    "unknown step fields",
                )
                self.steps += 1
            same(
                c["summary"],
                dict(received=8, features=features, thresholds=thresholds, reason=reason),
                "controller summary",
            )
        else:
            require(r["controller"] is None, "unexpected active controller on control")
        logical = 0 if self.arm == "FULL_LABEL" else 8
        if r["fits"]:
            require(r["fits"][0]["seed"] == self.seed * 100000 + 100 * logical, "fit seed mismatch")
        self.train(logical)

    def run(self):
        result = super().run()
        c = self.row["controller"]
        if not self.started:
            require(c is None, "unexplained controller")
        else:
            require(len(c["steps"]) == self.steps, "unexplained controller steps")
            expected = {"strategy", "cut_mode", "max_steps", "steps", "init_before"}
            if self.initialized:
                expected |= {"init_after", "source", "features", "thresholds", "reason"}
            if self.steps == 8:
                expected.add("summary")
            require(c.keys() == expected, "partial/unknown controller fields")
        result["preparation_units"] = preparation(self.trace.counts)
        return result


def audit_packed(payload, data, roles_path=None):
    del roles_path  # Prepared roles are bound and replayed by the frozen host.
    try:
        require(
            type(payload) is dict
            and payload.keys() == {"tier", "dataset", "seed", "split", "fold", "arm", "price", "record"},
            "outer record schema",
        )
        require(
            payload["tier"] == "packed_real"
            and payload["arm"] in ARMS
            and payload["dataset"] in ("banknote", "spambase", "wine_red")
            and type(payload["seed"]) is int
            and type(payload["price"]) is int
            and payload["price"] in (128, 8192),
            "outer identity",
        )
        same(
            [payload[k] for k in ("dataset", "seed", "split", "fold")],
            [data[k] for k in ("name", "seed", "split", "fold")],
            "host data identity",
        )
        pool, test, hold = (set(data["group_ids"][k]) for k in ("pool", "test", "holdout"))
        reserved = set(data["reserved_ids"])
        require(
            len(pool) == 256
            and len(test) == len(hold) == 512
            and hold <= reserved
            and not pool & (test | reserved)
            and (test == hold if payload["split"] == "final" else not test & reserved),
            "reserved/fold separation",
        )
        require(
            len(data["pool"]) == 256
            and len(data["test"]) == 512
            and len({str(r["id"]) for r in data["pool"] + data["test"]}) == 768,
            "complete assignments",
        )
        r = payload["record"]
        if payload["arm"] == "LEGACY40":
            same(
                [r[k] for k in ("arm", "dataset", "seed", "split", "price")],
                ["MIXED40", payload["dataset"], payload["seed"], payload["split"], payload["price"]],
                "legacy delegation identity",
            )
            result = audit_budget(r, None, None, prepared=data)
            result["preparation_units"] = preparation(r["budget"]["counts"], True)
            return result
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
            "controller",
            "frozen",
            "predictions",
            "budget",
            "trace",
            "wall_seconds",
        }
        require(type(r) is dict and r.keys() == fields and r["tier"] == "packed_native", "native schema")
        for key in ("dataset", "seed", "split", "arm", "price"):
            same(r[key], payload[key], "native identity " + key)
        quota = 256 if payload["arm"] == "FULL_LABEL" else int(payload["arm"][-2:])
        require(type(r["quota"]) is int and r["quota"] == quota, "quota")
        same(
            [r[k] for k in ("data_sha", "partition_digest", "thresholds", "rawcounts", "quantized_collisions")],
            [
                data["source_sha"],
                digest(data["group_ids"]),
                data["thresholds"],
                data["rawcounts"],
                data["quantized_test_rows_matching_pool"],
            ],
            "source/preprocessing pins",
        )
        require(
            type(r["wall_seconds"]) is str
            and r["wall_seconds"].isascii()
            and r["wall_seconds"].replace(".", "", 1).isdigit(),
            "elapsed format",
        )
        return PackedReplay(r, data, payload["fold"]).run()
    except (IndexError, KeyError, TypeError, StopIteration) as error:
        raise ValueError("invalid packed record: " + str(error)) from error
