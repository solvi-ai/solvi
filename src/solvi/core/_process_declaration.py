"""Core-owned JSON data contract for optional sequential process participation.

No scheduler or experimental dependencies live here. Process owns graph compilation.
"""
import json
import math
import re

__all__ = []

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
STATUSES = ("ok", "abstain", "forced", "human")
BYS = ("s1", "s2", "human")


def frozen(value):
    def check(v, depth=0):
        if depth > 32:
            raise ValueError("finite JSON nesting limit is 32")
        if type(v) in (str, int, bool, type(None)):
            return
        if type(v) is float and math.isfinite(v):
            return
        if type(v) is dict and all(type(k) is str for k in v):
            children = v.values()
        elif type(v) is list:
            children = v
        else:
            raise ValueError("only finite JSON with string object keys is supported")
        for child in children:
            check(child, depth + 1)
    check(value)
    return json.loads(json.dumps(value, allow_nan=False))


def _num(v, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ValueError(f"{where}: a number is needed, not {v!r}")
    return v


def check_predicate(p, where="when"):
    """Validate a predicate (see the module docs); None is "always". → p. ValueError names the bad place."""
    if p is None:
        return p
    if not isinstance(p, dict) or not p:
        raise ValueError(f"{where}: a predicate is a non-empty dict, not {p!r}")
    keys = set(p)
    if "answer" in keys:
        ops = keys - {"answer"}
        if len(ops) != 1 or not ops <= {"eq", "in", "lt", "gt"}:
            raise ValueError(f"{where}: {{'answer': fact}} takes exactly one of eq, in, lt, gt; got {sorted(keys)}")
        if not isinstance(p["answer"], str) or not p["answer"]:
            raise ValueError(f"{where}: 'answer' names a fact (a non-empty string), not {p['answer']!r}")
        op = ops.pop()
        if op == "in" and not isinstance(p["in"], list):
            raise ValueError(f"{where}: 'in' takes a list, not {p['in']!r}")
        if op in ("lt", "gt"):
            _num(p[op], f"{where}.{op}")
        return p
    if len(keys) != 1:
        raise ValueError(f"{where}: one form per predicate (combine with and / or), got {sorted(keys)}")
    (k, v), = p.items()
    if k == "status" and v not in STATUSES:
        raise ValueError(f"{where}: status is one of {', '.join(STATUSES)}, not {v!r}")
    elif k == "by" and v not in BYS:
        raise ValueError(f"{where}: by is one of {', '.join(BYS)}, not {v!r}")
    elif k == "confidence_below":
        if not 0.0 <= _num(v, f"{where}.confidence_below") <= 1.0:
            raise ValueError(f"{where}: confidence_below is in [0, 1], not {v!r}")
    elif k in ("budget_over", "drift"):
        if not isinstance(v, bool):
            raise ValueError(f"{where}: {k} takes true or false, not {v!r}")
    elif k in ("and", "or"):
        if not isinstance(v, list) or not v:
            raise ValueError(f"{where}: {k} takes a non-empty list of predicates")
        for i, q in enumerate(v):
            if q is None:
                raise ValueError(f"{where}.{k}[{i}]: null is not a predicate here")
            check_predicate(q, f"{where}.{k}[{i}]")
    elif k == "not":
        if v is None:
            raise ValueError(f"{where}.not: null is not a predicate here")
        check_predicate(v, f"{where}.not")
    elif k not in ("status", "by"):
        raise ValueError(f"{where}: unknown predicate {k!r} (answer, status, by, confidence_below, budget_over, "
                               "drift, and, or, not)")
    return p


def _participation(declaration):
    """Validate and detach the deliberately bounded System participation format."""
    d = frozen(declaration)
    keys = {"after", "when", "max_iter", "reads", "writes", "question", "end"}
    if not isinstance(d, dict) or set(d) != keys:
        raise ValueError("participation: expected after/when/max_iter/reads/writes/question/end")
    after, limit = d["after"], d["max_iter"]
    if after is not None and (not isinstance(after, str) or not _NAME.fullmatch(after)):
        raise ValueError("participation after: expected a node name or None")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("participation max_iter: expected a positive integer")
    if after is None and (d["when"] is not None or limit is not None):
        raise ValueError("participation entry cannot have when or max_iter")
    if type(d["end"]) is not bool:
        raise ValueError("participation end: expected bool")
    if d["question"] is not None and not isinstance(d["question"], str):
        raise ValueError("participation question: expected a string or None")
    check_predicate(d["when"])
    if not isinstance(d["reads"], list) or not all(isinstance(r, str) and r for r in d["reads"]):
        raise ValueError("participation reads: expected a list of fact names")
    writes = d["writes"]
    if not isinstance(writes, dict) or not all(isinstance(k, str) and k for k in writes):
        raise ValueError("participation writes: expected {fact: result selector}")
    for selector in writes.values():
        if not isinstance(selector, str) or not (selector in ("answer", "confidence", "by", "status") or
                selector.startswith("answer:") and len(selector) > 7):
            raise ValueError("participation writes: invalid result selector")
    return d

