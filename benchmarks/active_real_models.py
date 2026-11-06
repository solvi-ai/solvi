"""Paid deterministic binary CART baseline for the finite, eight-bin benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
import random

from solvi.experimental.experience._common import digest


def _int_between(value, low, high):
    return type(value) is int and low <= value <= high


def _parameters(seed, max_depth, min_leaf):
    if type(seed) is not int or not _int_between(max_depth, 0, 4) or not _int_between(min_leaf, 2, 256):
        raise ValueError("expected an integer seed, max_depth in 0..4 and min_leaf in 2..256")


@dataclass(frozen=True)
class Node:
    n: int
    pos: int
    feature: str | None = None
    threshold: int | None = None
    left: Node | None = None
    right: Node | None = None

    def to_dict(self):
        value = dict(n=self.n, pos=self.pos)
        if self.feature is not None:
            value.update(
                feature=self.feature, threshold=self.threshold, left=self.left.to_dict(), right=self.right.to_dict()
            )
        return value

    @classmethod
    def _restore(cls, value, features, remaining, min_leaf):
        if type(value) is not dict or not _int_between(value.get("n"), 1, 256):
            raise ValueError("invalid node size")
        n, pos = value["n"], value.get("pos")
        if not _int_between(pos, 0, n):
            raise ValueError("invalid node class count")
        if value.keys() == {"n", "pos"}:
            return cls(n, pos)
        if (
            value.keys() != {"n", "pos", "feature", "threshold", "left", "right"}
            or remaining == 0
            or value["feature"] not in features
            or not _int_between(value["threshold"], 1, 7)
        ):
            raise ValueError("invalid internal node")
        left = cls._restore(value["left"], features, remaining - 1, min_leaf)
        right = cls._restore(value["right"], features, remaining - 1, min_leaf)
        if (
            left.n + right.n != n
            or left.pos + right.pos != pos
            or min(left.n, right.n) < min_leaf
            or left.pos * right.n == right.pos * left.n
        ):
            raise ValueError("invalid child populations or split without gain")
        return cls(n, pos, value["feature"], value["threshold"], left, right)


@dataclass(frozen=True)
class Tree:
    root: Node
    bootstrap: tuple[str, ...]

    def leaf(self, inputs, budget):
        node = self.root
        while True:
            budget.charge("tree_predict_node")
            if type(inputs) is not dict or "label" in inputs:
                raise ValueError("predict expects unlabeled feature inputs")
            if node.feature is None:
                return node.n, node.pos
            budget.charge("tree_predict_input")
            value = inputs.get(node.feature)
            if not _int_between(value, 0, 7):
                raise ValueError("invalid or missing quantized feature")
            node = node.left if value < node.threshold else node.right

    def evaluate(self, inputs, budget, grammar=None):
        n, pos = self.leaf(inputs, budget)
        return 2 * pos > n

    def to_dict(self):
        return dict(bootstrap=list(self.bootstrap), root=self.root.to_dict())


@dataclass(frozen=True)
class Forest:
    features: tuple[str, ...]
    seed: int
    max_depth: int
    min_leaf: int
    trees: tuple[Tree, ...]
    fingerprint: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "fingerprint", digest(self._body()))

    def _body(self):
        return dict(
            v=1,
            features=list(self.features),
            seed=self.seed,
            max_depth=self.max_depth,
            min_leaf=self.min_leaf,
            trees=[tree.to_dict() for tree in self.trees],
        )

    def to_dict(self):
        return dict(**self._body(), fingerprint=self.fingerprint)

    @classmethod
    def from_dict(cls, value):
        if (
            type(value) is not dict
            or value.keys() != {"v", "features", "seed", "max_depth", "min_leaf", "trees", "fingerprint"}
            or not _int_between(value["v"], 1, 1)
        ):
            raise ValueError("invalid forest snapshot")
        _parameters(value["seed"], value["max_depth"], value["min_leaf"])
        names = value["features"]
        if (
            type(names) is not list
            or not 1 <= len(names) <= 128
            or names != [f"f{i}" for i in range(len(names))]
            or type(value["trees"]) is not list
            or len(value["trees"]) != 3
        ):
            raise ValueError("invalid forest shape")
        trees = []
        for tree in value["trees"]:
            if type(tree) is not dict or tree.keys() != {"bootstrap", "root"}:
                raise ValueError("invalid tree snapshot")
            ids = tree["bootstrap"]
            if type(ids) is not list or not 1 <= len(ids) <= 256 or any(type(i) is not str or not i for i in ids):
                raise ValueError("invalid bootstrap")
            root = Node._restore(tree["root"], names, value["max_depth"], value["min_leaf"])
            if root.n != len(ids) or (trees and root.n != trees[0].root.n):
                raise ValueError("bootstrap size differs from training population")
            trees.append(Tree(root, tuple(ids)))
        model = cls(tuple(names), value["seed"], value["max_depth"], value["min_leaf"], tuple(trees))
        if value["fingerprint"] != model.fingerprint:
            raise ValueError("forest fingerprint mismatch")
        return model

    def predict(self, inputs, budget):
        budget.charge("forest_vote", len(self.trees))
        return sum(tree.evaluate(inputs, budget) for tree in self.trees) >= 2


def _split(indices, xs, ys, pos, width, min_leaf, budget):
    n = len(indices)
    budget.charge("tree_fit_hist", n * width)
    hist = [[0] * 8 for _ in range(width)]
    positives = [[0] * 8 for _ in range(width)]
    for i in indices:
        for f, bin_value in enumerate(xs[i]):
            hist[f][bin_value] += 1
            positives[f][bin_value] += ys[i]
    # Declared integer-work proxy for seven candidate cuts per column.
    budget.charge("tree_fit_candidates", 14 * 7 * width)
    best_num, best_den = pos * pos + (n - pos) * (n - pos), n
    best = None
    for f in range(width):
        nl = lp = 0
        for threshold in range(1, 8):
            nl += hist[f][threshold - 1]
            lp += positives[f][threshold - 1]
            nr = n - nl
            if min(nl, nr) < min_leaf:
                continue
            rp = pos - lp
            num = (lp * lp + (nl - lp) ** 2) * nr + (rp * rp + (nr - rp) ** 2) * nl
            den = nl * nr
            if num * best_den > best_num * den:
                best_num, best_den = num, den
                best = f, threshold
    return best


def _fit_node(indices, xs, ys, features, depth, min_leaf, budget):
    n = len(indices)
    budget.charge("tree_fit_node")
    budget.charge("tree_fit_labels", n)
    pos = sum(ys[i] for i in indices)
    if depth == 0 or n < 2 * min_leaf or pos in (0, n):
        return Node(n, pos)
    split = _split(indices, xs, ys, pos, len(features), min_leaf, budget)
    if split is None:
        return Node(n, pos)
    f, threshold = split
    budget.charge("tree_fit_route", n)
    left, right = [], []
    for i in indices:
        (left if xs[i][f] < threshold else right).append(i)
    return Node(
        n,
        pos,
        features[f],
        threshold,
        _fit_node(left, xs, ys, features, depth - 1, min_leaf, budget),
        _fit_node(right, xs, ys, features, depth - 1, min_leaf, budget),
    )


def fit_forest(rows, budget, *, seed, max_depth=4, min_leaf=2):
    """Fit three bootstrapped trees to explicitly acquired labels on one caller ledger."""
    _parameters(seed, max_depth, min_leaf)
    if type(rows) not in (tuple, list) or not 1 <= len(rows) <= 256 or type(rows[0]) is not dict:
        raise ValueError("expected 1..256 training rows")
    first = rows[0].get("inputs")
    if type(first) is not dict or not 1 <= len(first) <= 128:
        raise ValueError("expected 1..128 quantized features")
    n, width = len(rows), len(first)
    budget.charge("tree_fit_validate", n * width)
    features = tuple(f"f{i}" for i in range(width))
    names = set(features)
    xs, ys, ids = [], [], []
    seen = set()
    for row in rows:
        if type(row) is not dict or row.keys() != {"id", "inputs", "label"}:
            raise ValueError("invalid training row shape")
        inputs, label, rid = row["inputs"], row["label"], row["id"]
        if (
            type(label) is not bool
            or type(rid) is not str
            or not rid
            or rid in seen
            or type(inputs) is not dict
            or inputs.keys() != names
        ):
            raise ValueError("invalid training schema, label or row identity")
        values = tuple(inputs[f] for f in features)
        if not all(_int_between(v, 0, 7) for v in values):
            raise ValueError("expected strict integer bins in 0..7")
        xs.append(values)
        ys.append(label)
        ids.append(rid)
        seen.add(rid)
    trees = []
    for tree_index in range(3):
        rng = random.Random(seed + tree_index)
        budget.charge("forest_bootstrap", n)
        sample = [rng.randrange(n) for _ in range(n)]
        root = _fit_node(sample, xs, ys, features, max_depth, min_leaf, budget)
        trees.append(Tree(root, tuple(ids[i] for i in sample)))
    return Forest(features, seed, max_depth, min_leaf, tuple(trees))
