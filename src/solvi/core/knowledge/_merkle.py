"""Canonical Merkle treap: O(N) nodes, expected O(log N) update hashes.

Priority is a domain-separated SHA-256 of the key, so tree shape depends only
on the key set. The internal state is not a mutable public data API.
"""
from dataclasses import dataclass
import hashlib
import json
from ..runtime import _canon


def digest(domain, value):
    encoded = json.dumps([domain, _canon(value)], sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


EMPTY = digest('solvi.km2.empty', None)


@dataclass(frozen=True)
class Node:
    key: str
    value: str
    priority: str
    left: object
    right: object
    hash: str


def node(key, value, priority, left=None, right=None):
    return Node(key, value, priority, left, right,
                digest('solvi.km2.node', [key, value, left.hash if left else EMPTY,
                                        right.hash if right else EMPTY]))


def merge(left, right):
    if left is None:
        return right
    if right is None:
        return left
    if (left.priority, left.key) < (right.priority, right.key):
        return node(left.key, left.value, left.priority, left.left, merge(left.right, right))
    return node(right.key, right.value, right.priority, merge(left, right.left), right.right)


def put(root, key, value):
    """Set a full leaf digest, or delete with None. Caller skips unchanged leaves."""
    if root is None:
        return None if value is None else node(key, value, digest('solvi.km2.priority', key))
    if key == root.key:
        return merge(root.left, root.right) if value is None else node(key, value, root.priority, root.left, root.right)
    if key < root.key:
        left = put(root.left, key, value)
        if left and (left.priority, left.key) < (root.priority, root.key):
            return node(left.key, left.value, left.priority, left.left,
                        node(root.key, root.value, root.priority, left.right, root.right))
        return node(root.key, root.value, root.priority, left, root.right)
    right = put(root.right, key, value)
    if right and (right.priority, right.key) < (root.priority, root.key):
        return node(right.key, right.value, right.priority,
                    node(root.key, root.value, root.priority, root.left, right.left), right.right)
    return node(root.key, root.value, root.priority, root.left, right)


__all__ = []  # Internal fingerprint implementation; no public extension points.
