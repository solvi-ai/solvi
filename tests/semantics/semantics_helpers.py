"""Shared pieces of the semantics tests (docs/trace_spec.md): a small system, a store filled with every kind of record,
JSON-path walking and mutation, a consistent rewrite of a whole chain, and a small JSON Schema validator (the
draft 2020-12 keywords record.schema.json uses; jsonschema, when installed, is used as well).

The tests are property-style without a new dependency: each property runs over seeded random cases (random.Random(seed)),
so a failure names its seed and reproduces."""
from __future__ import annotations

import copy
import json
import os
import random
import re

from solvi import Answer, Catalog, Question, System
from solvi.core.store import GENESIS, TraceStorage, record_hash

SEEDS = range(12)                                     # the seeds every property runs over


# --- a small system: a hard check with `then`, a computed fact, two rules
def system(big=1000):
    cat = Catalog()

    @cat.check(hard=True, then={"pay": "no"})
    def cap(amount, limit) -> bool:
        return amount <= limit

    @cat.fn
    def share(amount, limit) -> float:
        return round(amount / limit, 6)

    @cat.rule("pay")
    def pay(share):
        return "yes" if share < 0.5 else "review"

    @cat.rule("size")
    def size(amount):
        return "big" if amount >= big else "small"

    return System(cat, [Question("pay", "Pay?", Answer.choice(["yes", "review", "no"]), requires=["cap"]),
                        Question("size", "How big?", Answer.choice(["big", "small"]))])


OPTIONS = {"pay": ["yes", "review", "no"], "size": ["big", "small"]}
STATUSES = ["ok", "forced", "abstain"]


def random_input(rng):
    limit = rng.choice([1, 50, 100, 1000, 10 ** 6])
    amount = rng.choice([0, 1, rng.randint(1, 2 * limit), rng.uniform(0, 3 * limit), 10 ** 7])
    state = {"amount": amount, "limit": limit}
    if rng.random() < 0.3:
        state["note"] = rng.choice(["", "ünïcode ✓", "a\nb", "x" * 40])         # an extra given fact nobody reads
    return state


# --- a store of your own with its head kept apart from the records (a cut-off tail is then visible), in memory
class MemoryStorage(TraceStorage):
    def __init__(self, system=None, record="full"):
        super().__init__(system, record=record)
        self.rows = []
        self.stored_head = {"count": 0, "hash": GENESIS}

    def _append(self, body):
        rec = dict(json.loads(json.dumps(body)), seq=len(self.rows), time=float(self.clock()),
                   prev=self.rows[-1]["hash"] if self.rows else GENESIS)
        rec["hash"] = record_hash(rec)
        rec["id"] = rec["hash"][:16]
        self.rows.append(rec)
        self.stored_head = {"count": len(self.rows), "hash": rec["hash"]}
        return rec

    def _raw(self, snap=None):
        return iter(list(enumerate(self.rows)))

    def _find(self, id):
        return next((r for r in self.rows if r["id"] == id), None)

    def head(self):
        return dict(self.stored_head)

    def _rewrite(self, rec):
        self.rows[rec["seq"]] = rec

    def _backend_problems(self, rows, snap=None):
        from solvi.core.store import _head_problems
        return _head_problems(self.stored_head, rows)


def fill(store, sys_, rng, n=4):
    """Every kind of record a store holds: decisions (with and without meta), a correction, a redaction (keeping the
    answers or not, of a decision or of a correction) → the ids saved, in order."""
    ids = []
    for i in range(n):
        res = sys_.ask(random_input(rng))
        ids.append(store.save(res, meta={"ticket": i} if rng.random() < 0.5 else None))
    ids.append(store.save_correction("pay", random_input(rng), rng.choice(OPTIONS["pay"]), by="ann",
                                     label_source=rng.choice(["human", "outcome"]), note="checked"))
    victim = rng.choice(ids)
    ids.append(store.redact(victim, by="dpo", note="erasure request", keep_answers=rng.random() < 0.5))
    for _ in range(rng.randint(1, 3)):
        ids.append(store.save(sys_.ask(random_input(rng))))
    return ids


# --- JSONL files
def read_lines(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_lines(path, recs, head=None):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    if head is not None:
        with open(str(path) + ".head", "w") as fh:
            json.dump(head, fh)


def rechain(recs):
    """Recompute every record's hash, id and link, in order, and the redaction records' names of what they erased — the
    rewrite of a whole chain by someone who can write the store → its head."""
    prev, moved = GENESIS, {}
    for r in recs:
        if r.get("kind") == "redaction" and r.get("of_hash") in moved:
            r["of_hash"] = moved[r["of_hash"]]
            r["of"] = r["of_hash"][:16]
        old = r.get("hash")
        r["prev"] = prev
        r["hash"] = record_hash(r)
        r["id"] = r["hash"][:16]
        moved[old] = r["hash"]
        prev = r["hash"]
    return {"count": len(recs), "hash": prev}


# --- JSON paths
def leaf_paths(obj, path=()):
    """Every path to a scalar or an empty container of a JSON value."""
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            yield from leaf_paths(v, path + (k,))
    elif isinstance(obj, list) and obj:
        for i, v in enumerate(obj):
            yield from leaf_paths(v, path + (i,))
    else:
        yield path


def get_at(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def set_at(obj, path, value):
    for p in path[:-1]:
        obj = obj[p]
    obj[path[-1]] = value


def other_value(v, rng):
    """A JSON value surely different from v (in its canonical JSON too)."""
    if isinstance(v, bool):
        return not v
    if isinstance(v, int):
        return v + rng.choice([1, -1, 1000])
    if isinstance(v, float):
        return v + rng.choice([0.5, -0.25, 1e-3]) if abs(v) < 1e12 else v / 2
    if isinstance(v, str):
        return v + rng.choice(["x", " ", "0"]) if v else "x"
    if v is None:
        return rng.choice([0, "", False])
    if isinstance(v, list):
        return ["x"]
    if isinstance(v, dict):
        return {"x": 1}
    raise TypeError(v)


def mutated(rec, path, rng):
    out = copy.deepcopy(rec)
    set_at(out, path, other_value(get_at(rec, path), rng))
    return out


def shuffled_keys(obj, rng):
    """The same JSON value with every dict's keys in another order."""
    if isinstance(obj, dict):
        items = list(obj.items())
        rng.shuffle(items)
        return {k: shuffled_keys(v, rng) for k, v in items}
    if isinstance(obj, list):
        return [shuffled_keys(v, rng) for v in obj]
    return obj


def rng_for(seed):
    return random.Random(f"solvi-semantics-{seed}")


# --- a small JSON Schema validator (the keywords record.schema.json uses)
SCHEMA_PATH = os.path.join(os.path.dirname(__import__("solvi.core.store", fromlist=["x"]).__file__), "record.schema.json")


def load_schema():
    with open(SCHEMA_PATH, encoding="utf-8") as fh:
        return json.load(fh)


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is_type(v, t):
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def _eq(a, b):
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True) or (
        isinstance(a, (int, float)) and isinstance(b, (int, float)) and a == b)


def validate(value, schema, root=None, where="$"):
    """→ [error text]; empty when `value` is valid."""
    root = root if root is not None else schema
    if schema is True:
        return []
    if schema is False:
        return [f"{where}: not allowed"]
    errs = []
    if "$ref" in schema:
        ref = schema["$ref"]
        assert ref.startswith("#/$defs/"), ref
        errs += validate(value, root["$defs"][ref[8:]], root, where)
    if "type" in schema:
        ts = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_is_type(value, t) for t in ts):
            return errs + [f"{where}: {type(value).__name__} is not {ts}"]
    if "const" in schema and not _eq(value, schema["const"]):
        errs.append(f"{where}: {value!r} is not {schema['const']!r}")
    if "enum" in schema and not any(_eq(value, e) for e in schema["enum"]):
        errs.append(f"{where}: {value!r} not in {schema['enum']}")
    if isinstance(value, str):
        if "pattern" in schema and not re.search(schema["pattern"], value):
            errs.append(f"{where}: {value!r} does not match {schema['pattern']}")
        if len(value) < schema.get("minLength", 0):
            errs.append(f"{where}: too short")
    if _is_type(value, "number") and "minimum" in schema and value < schema["minimum"]:
        errs.append(f"{where}: {value} < {schema['minimum']}")
    if isinstance(value, dict):
        for k in schema.get("required", ()):
            if k not in value:
                errs.append(f"{where}: missing {k!r}")
        props = schema.get("properties", {})
        for k, v in value.items():
            if k in props:
                errs += validate(v, props[k], root, f"{where}.{k}")
            elif "additionalProperties" in schema:
                errs += validate(v, schema["additionalProperties"], root, f"{where}.{k}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            errs.append(f"{where}: {len(value)} items")
        pre = schema.get("prefixItems", [])
        for i, v in enumerate(value):
            if i < len(pre):
                errs += validate(v, pre[i], root, f"{where}[{i}]")
            elif "items" in schema:
                errs += validate(v, schema["items"], root, f"{where}[{i}]")
    for s in schema.get("allOf", ()):
        errs += validate(value, s, root, where)
    if "anyOf" in schema and not any(not validate(value, s, root, where) for s in schema["anyOf"]):
        errs.append(f"{where}: matches none of anyOf")
    if "oneOf" in schema:
        hits = [i for i, s in enumerate(schema["oneOf"]) if not validate(value, s, root, where)]
        if len(hits) != 1:
            detail = "" if hits else " — " + " | ".join(
                "; ".join(validate(value, s, root, where)[:2]) for s in schema["oneOf"])
            errs.append(f"{where}: matches {len(hits)} of oneOf {hits}{detail}")
    if "not" in schema and not validate(value, schema["not"], root, where):
        errs.append(f"{where}: matches `not`")
    if "if" in schema and not validate(value, schema["if"], root, where):
        errs += validate(value, schema.get("then", True), root, where)
    return errs
