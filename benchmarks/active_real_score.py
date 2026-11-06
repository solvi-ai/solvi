"""Replay a single finite classification record against host data and a paid reference.

No learner, policy, oracle, public fit or prediction method is called. A failed
debit consumes neither a trace entry nor a model/receipt commit.
"""

from __future__ import annotations

from collections import Counter
from fractions import Fraction
import random

from active_real_data import SOURCES, prepare_dataset
from active_real_models import Forest
from solvi.experimental.experience._common import WorkBudget, canonical, digest

LIMIT = 4_000_000
ARMS = ("RANDOM", "UNCERTAINTY", "QBC_FULL", "QBC_CAPPED", "FULL_LABEL")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def same(a, b, description):
    require(canonical(a) == canonical(b), description)


class NoFunds(Exception):
    pass


class WindowClosed(Exception):
    pass


class Trace:
    def __init__(self, payments):
        self.rows = payments
        self.i = self.repeat = self.spent = self.work = 0
        self.counts = Counter()
        self.window = None
        last = None
        total = 0
        for run in payments:
            require(type(run) is list and len(run) == 3, "malformed trace")
            k, u, n = run
            require(
                type(k) is str and type(u) is int and type(n) is int and min(u, n) > 0 and (k, u) != last,
                "noncanonical debit run",
            )
            last = k, u
            total += u * n
            require(total <= LIMIT, "overlong trace")

    def pay(self, k, u=1):
        self.work += 1
        if self.window is not None and self.spent + u > self.window:
            raise WindowClosed
        if self.spent + u > LIMIT:
            raise NoFunds
        require(self.i < len(self.rows), "missing debit: " + k)
        kind, units, n = self.rows[self.i]
        require((kind, units) == (k, u), f"debit {self.i}: expected {k}/{u}, got {kind}/{units}")
        self.repeat += 1
        if self.repeat == n:
            self.i += 1
            self.repeat = 0
        self.counts[k] += u
        self.spent += u

    def snapshot(self, stored):
        saved = WorkBudget.restore(stored)
        require(saved.limit == LIMIT and saved.spent == self.spent and saved.counts == self.counts, "ledger prefix")


class Replay:
    def __init__(self, row, data):
        self.row = row
        self.data = data
        self.trace = Trace(row["trace"])
        self.arm = row["arm"]
        self.price = row["price"]
        self.seed = row["seed"]
        self.pool = {str(r["id"]): r for r in data["pool"]}
        self.width = len(data["feature_names"])
        self.names = tuple(f"f{i}" for i in range(self.width))
        self.point_digests = {i: digest(r["inputs"]) for i, r in self.pool.items()}
        self.ids = []
        self.fit_count = 0
        self.decision_count = 0
        self.call_count = 0
        self.model = None
        self.cache = {}
        self.fallbacks = self.partial = 0

    def label(self, rid):
        self.trace.pay("real_label", self.price)
        n = len(self.ids)
        record = self.row["receipts"][n]
        key = f"{self.row['dataset']}:{self.seed}:{self.arm}:{self.price}:{self.row['cap']}:label:{n}"
        same(
            record,
            dict(request=key, id=rid, expected=self.pool[rid]["label"], source=self.data["source_sha"]),
            "label receipt or training role",
        )
        self.ids.append(rid)

    def node(self, bag, depth):
        t = self.trace
        n = len(bag)
        t.pay("tree_fit_node")
        t.pay("tree_fit_labels", n)
        pos = sum(r["label"] for r in bag)
        node = dict(n=n, pos=pos)
        if depth == 4 or n < 4 or pos in (0, n):
            return node
        t.pay("tree_fit_hist", n * self.width)
        hist = {f: [[0, 0] for _ in range(8)] for f in self.names}
        for r in bag:
            for f in self.names:
                hist[f][r["inputs"][f]][r["label"]] += 1
        t.pay("tree_fit_candidates", 98 * self.width)
        # Gini gain is proportional to (n * positives_left - pos * size_left)^2 /
        # (size_left * size_right). Compare these fractions with integer products.
        best_num, best_den = 0, 1
        best = None
        for f in self.names:
            left_size = lp = 0
            for cut, (neg, p) in enumerate(hist[f][:-1], 1):
                left_size += neg + p
                lp += p
                if left_size < 2 or n - left_size < 2:
                    continue
                num = (n * lp - pos * left_size) ** 2
                den = left_size * (n - left_size)
                if num * best_den > best_num * den:
                    best_num, best_den = num, den
                    best = (f, cut)
        if best:
            f, cut = best
            t.pay("tree_fit_route", n)
            left = [r for r in bag if r["inputs"][f] < cut]
            right = [r for r in bag if r["inputs"][f] >= cut]
            node.update(feature=f, threshold=cut, left=self.node(left, depth + 1), right=self.node(right, depth + 1))
        return node

    def train(self, logical_round):
        t = self.trace
        n = len(self.ids)
        seed = self.seed * 100000 + 100 * logical_round
        # Partial fits have no record. Reconstruct their debits up to the first refusal.
        fits = self.row["fits"]
        record = fits[self.fit_count] if self.fit_count < len(fits) else None
        if record is not None:
            t.snapshot(record["before"])
        t.pay("tree_fit_validate", n * self.width)
        trees = []
        for j in range(3):
            t.pay("forest_bootstrap", n)
            rng = random.Random(seed + j)
            ids = [self.ids[rng.randrange(n)] for _ in range(n)]
            trees.append(dict(bootstrap=ids, root=self.node([self.pool[i] for i in ids], 0)))
        require(record is not None, "missing completed fit")
        same(
            [record["index"], record["logical_round"], record["acquired_ids"]],
            [self.fit_count, logical_round, self.ids],
            "fit identity/acquired prefix",
        )
        snap = dict(v=1, features=list(self.names), seed=seed, max_depth=4, min_leaf=2, trees=trees)
        snap["fingerprint"] = digest(snap)
        same(snap, record["model"], "bootstrap or tree differs from independent optimal split")
        Forest.from_dict(record["model"])
        t.snapshot(record["after"])
        self.model = snap
        self.cache = {}
        self.fit_count += 1

    def leaf(self, j, x):
        t = self.trace
        node = self.model["trees"][j]["root"]
        while True:
            t.pay("tree_predict_node")
            if "feature" not in node:
                return node["n"], node["pos"]
            t.pay("tree_predict_input")
            node = node["left"] if x[node["feature"]] < node["threshold"] else node["right"]

    def cached(self, j, rid):
        # Each row visit constructs one fresh inputs dict. Its three tree calls share it.
        t = self.trace
        new_key = j == 0
        point = self.pool[rid]["inputs"]
        pd = self.point_digests[rid]
        t.pay("real_predict_key", self.width + 1 if new_key else 1)
        t.pay("real_predict_lookup")
        key = j, pd
        hit = key in self.cache
        if not hit:
            leaf = self.leaf(j, point)
            t.pay("real_predict_store")
            self.cache[key] = leaf
        call = self.row["predict_calls"][self.call_count]
        require(
            type(call) is dict
            and call.keys() == {"fit", "tree", "point_digest", "hit", "new_key"}
            and type(call["fit"]) is int
            and call["fit"] == self.fit_count - 1
            and type(call["tree"]) is int
            and call["tree"] == j
            and call["point_digest"] == pd
            and call["hit"] is hit
            and call["new_key"] is new_key,
            "cache call/provenance",
        )
        self.call_count += 1
        return self.cache[key]

    def choose(self, remaining):
        t = self.trace
        n = len(remaining)
        chosen = None
        pairs = 0
        reason = "random"
        if self.arm == "RANDOM":
            return None, reason, 0
        if self.arm == "UNCERTAINTY":
            best = None
            for rid in remaining:
                t.pay("real_uncertainty")
                size, pos = self.cached(0, rid)
                t.pay("real_uncertainty", 3)
                distance = Fraction(abs(2 * pos - size), size + 2)
                if best is None or distance < best:
                    chosen, best = rid, distance
            return chosen, "leaf_uncertainty", 0
        capped = self.arm == "QBC_CAPPED"
        if capped:
            t.window = min(t.spent + self.row["cap"], LIMIT - self.price)
        reason = "selected" if capped else "full_scan"
        interrupted = False
        try:
            for _ in range(4):
                t.pay("observation_plan")
            for _ in remaining:
                t.pay("observation_plan")
            if capped:
                t.pay("observation_compare", n)
                t.pay("observation_cache", n)
            for rid in remaining:
                if capped:
                    t.pay("observation_plan")
                    if pairs == 2:
                        break
                    t.pay("observation_cache")
                votes = 0
                for j in range(3):
                    t.pay("observation_plan")
                    if capped:
                        t.pay("observation_output")
                    size, pos = self.cached(j, rid)
                    votes += 2 * pos > size
                    t.pay("observation_compare" if capped else "observation_output")
                for _ in range(1 if capped else 3):
                    t.pay("observation_compare")
                if votes * (3 - votes) > pairs:
                    chosen, pairs = rid, votes * (3 - votes)
            if not capped:
                t.pay("observation_plan", n)
        except WindowClosed:
            interrupted = True
        finally:
            t.window = None
        if LIMIT - t.spent < self.price:
            chosen = None
        if capped:
            if not chosen:
                reason = "budget" if interrupted or pairs else "no_disagreement"
            elif interrupted:
                reason = "selected_partial"
        return chosen, reason, pairs if capped and chosen else 0

    def acquisition(self):
        t = self.trace
        for k, u in self.data["preprocess_costs"].items():
            t.pay(k, u)
        t.pay("real_pool_setup", len(self.pool) * (self.width + 1))
        for rid in list(self.pool)[: 256 if self.arm == "FULL_LABEL" else 16]:
            self.label(rid)
        if self.arm == "FULL_LABEL":
            self.train(0)
            return
        if self.arm != "RANDOM":
            self.train(0)
            t.pay("real_options", 256 * (self.width + 1))
        while len(self.ids) < 48:
            for _ in range(4):
                t.pay("real_pool_filter", 256)
                remaining = [i for i in self.pool if i not in self.ids]
                if self.arm != "RANDOM":
                    t.pay("real_option_route", len(remaining))
                records = self.row["decisions"]
                d = records[self.decision_count] if self.decision_count < len(records) else None
                if d is not None:
                    t.snapshot(d["before"])
                start = self.call_count
                rid, reason, pairs = self.choose(remaining)
                fallback = rid is None
                if fallback:
                    t.pay("real_choice", len(remaining))
                    rid = random.Random(self.seed + len(self.ids)).choice(remaining)
                require(d is not None, "missing decision")
                same(
                    [d[k] for k in ("query", "fit", "id", "reason", "pairs", "fallback", "calls")],
                    [len(self.ids), self.fit_count - 1, rid, reason, pairs, fallback, [start, self.call_count]],
                    "policy choice",
                )
                t.snapshot(d["after"])
                self.decision_count += 1
                self.fallbacks += int(fallback)
                self.partial += int(reason == "selected_partial")
                self.label(rid)
            if self.arm != "RANDOM":
                self.train((len(self.ids) - 16) // 4)
        if self.arm == "RANDOM":
            self.train(8)

    def run(self):
        r = self.row
        t = self.trace
        reason = "label_quota"
        try:
            self.acquisition()
        except NoFunds:
            reason = "budget"
        f = r["frozen"]
        same(
            [f[k] for k in ("reason", "model", "labels", "fit_count")],
            [reason, self.model, len(self.ids), self.fit_count],
            "frozen state/last committed fit",
        )
        t.snapshot(f["ledger"])
        require(
            [len(r[k]) for k in ("receipts", "fits", "decisions", "predict_calls")]
            == [len(self.ids), self.fit_count, self.decision_count, self.call_count],
            "unexplained records",
        )
        correct = wrong = abstain = budgeted = 0
        p = tp = tn = 0
        require(len(r["predictions"]) == 512, "missing assignments")
        for src, record in zip(self.data["test"], r["predictions"], strict=True):
            expected = src["label"]
            p += expected
            derived = dict(id=str(src["id"]), expected=expected)
            if self.model is None:
                derived["outcome"] = "abstain"
                abstain += 1
            else:
                try:
                    t.pay("forest_vote", 3)
                    votes = [self.leaf(j, src["inputs"]) for j in range(3)]
                except NoFunds:
                    derived["outcome"] = "budget"
                    budgeted += 1
                else:
                    value = sum(2 * pos > n for n, pos in votes) >= 2
                    good = value is expected
                    derived.update(value=value, outcome="correct" if good else "wrong")
                    correct += good
                    wrong += not good
                    tp += good and value
                    tn += good and not value
            same(record, derived, "held-out prediction")
        require(t.i == len(t.rows) and t.repeat == 0, "extra trace tail")
        t.snapshot(r["budget"])
        ba = (Fraction(tp, p) + Fraction(tn, 512 - p)) / 2 if 0 < p < 512 else None

        def pair(x):
            return None if x is None else [x.numerator, x.denominator]

        return dict(
            accuracy=pair(Fraction(correct, 512)),
            ba=pair(ba),
            balanced_accuracy=pair(ba),
            assigned=512,
            correct=correct,
            wrong=wrong,
            abstain=abstain,
            budget=budgeted,
            units=t.spent,
            labels=len(self.ids),
            fits=self.fit_count,
            fallbacks=self.fallbacks,
            selection_partial=self.partial,
            auditorwork=t.work,
            positives=p,
            negatives=512 - p,
            tp=tp,
            tn=tn,
        )


def audit_real(row, cache_dir):
    """Check recorded work, inputs, all models and all decisions. Never mutates the raw record."""
    try:
        fields = {
            "tier",
            "dataset",
            "seed",
            "split",
            "arm",
            "price",
            "cap",
            "data_sha",
            "partition_digest",
            "thresholds",
            "rawcounts",
            "quantized_collisions",
            "receipts",
            "fits",
            "decisions",
            "predict_calls",
            "frozen",
            "predictions",
            "budget",
            "trace",
            "wall_seconds",
        }
        require(type(row) is dict and row.keys() == fields, "unexpected trial schema")
        require(
            type(row["wall_seconds"]) is str
            and row["wall_seconds"].isascii()
            and row["wall_seconds"].replace(".", "", 1).isdigit(),
            "invalid recorded time",
        )
        require(
            row["tier"] == "real"
            and row["dataset"] in SOURCES
            and row["arm"] in ARMS
            and type(row["seed"]) is int
            and row["split"] in ("dev", "final")
            and type(row["price"]) is int
            and row["price"] in (128, 8192)
            and type(row["cap"]) is int
            and row["cap"] in (1024, 4096),
            "invalid trial identity",
        )
        data = prepare_dataset(row["dataset"], cache_dir, row["seed"], row["split"])
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
            "data pins or partition",
        )
        pool, test, hold = (set(data["group_ids"][k]) for k in ("pool", "test", "holdout"))
        require(
            len(pool) == 256
            and len(test) == len(hold) == 512
            and not pool.intersection(test | hold)
            and (test == hold if row["split"] == "final" else not test.intersection(hold)),
            "overlapping group roles",
        )
        require(
            len(data["pool"]) == 256
            and len(data["test"]) == 512
            and len({str(r["id"]) for r in data["pool"] + data["test"]}) == 768,
            "row-role identity",
        )
        return Replay(row, data).run()
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError("invalid trial structure: " + str(error)) from error
