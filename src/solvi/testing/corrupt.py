"""Corruption tests for specialists: corrupt a specialist's recorded proposals and count how many corruptions its check
catches (solvi.core.specialist: a model proposes, code checks against the source, code renders).

    from solvi.testing.corrupt import change_digit, corrupt, drop, swap

    records = [(rid, source, run.proposal) for rid, source, run in stored_runs]     # the proposals as recorded
    rep = corrupt(specialist, records, {"digit": change_digit(paths=r"^/series/\\d+/points/\\d+/value$"),
                                       "drop": drop(paths=r"^/series/\\d+/points/\\d+$")}, n=300, seed=0)
    rep.n, rep.caught, rep.rate        # applied mutations, caught ones, caught / n
    rep.per_kind                       # {kind: {"n", "caught", "skipped"}}
    rep.rows                           # [{"id", "kind", "path", "value", "caught", "criterion"}]

A proposal is corrupted as if the proposer had misread its source — one digit, an element left out, two elements
exchanged — and the specialist runs again on the same source with the corrupted proposal (`specialist.run(source,
proposal=mutated)`). The source is not touched: the check has what it had before to notice the corruption.

The draws are deterministic: records sorted by id, kinds in the dict's order, `random.Random(seed)`; each draw picks a
record and a kind and applies the kind's mutator to a copy of the record's proposal (its JSON form). A mutator that
finds no target in that proposal returns None: the draw is skipped and counted (`per_kind[kind]["skipped"]`), and draws
go on until `n` mutations were applied (at most 20 × n draws).

A mutator is `(proposal_dict, rng) → (mutated_dict, path, value) | None`: `path` is the JSON pointer (RFC 6901, e.g.
"/series/0/points/2/value") of what it changed, `value` what is there now. A mutator may carry its own judgement as an
attribute `caught(run, base, path, value) → bool` (base: the run on the unmutated proposal) and a `criterion` name.

What "caught" means:
  - always: the run has no output (the check blocked it);
  - "accepted" (change_digit, swap, and a mutator without its own judgement): the mutated `value` is not accepted at
    `path` — `accepted(run) → {path: value}` is what the check verified; by default the leaves of
    `run.checked.verified.model_dump(mode="json")` as JSON pointers, which suits a verified spec laid out like the
    proposal (pass `accepted=` otherwise). A value that is a dict or a list is accepted when every one of its leaves
    the verified spec has at the same place is equal and at least one is there. Numbers are compared by value
    ("4.20" is 4.2);
  - "relation" (drop): an omission leaves nothing at its own place to refuse — it is visible only through a relation
    the check holds (a total that no longer adds up, shares that no longer make a whole), as the receipts measurement found
    (27 of 76 omissions caught). So a drop is caught when the run reports an issue (code and severity, counted) that
    the unmutated run did not. (Comparing the output bytes would count every omission: the output shows one element
    fewer.)

The rate is that of the corruptions these mutators make, on these records: a check that holds relations between values
catches omissions, one that only quotes each value does not (see the changelog, 1.1, "the specialist contract graduates", for the
specialists measured this way)."""
from __future__ import annotations

import copy
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation


# ------------------------------------------------------------------------------------------------ JSON pointers
def _esc(k):
    return str(k).replace("~", "~0").replace("/", "~1")


def _unesc(k):
    return k.replace("~1", "/").replace("~0", "~")


def leaves(tree, prefix=""):
    """A JSON tree → [(JSON pointer, leaf value)] in document order; an empty dict or list, and None, are leaves."""
    out = []
    if isinstance(tree, dict) and tree:
        for k, v in tree.items():
            out += leaves(v, f"{prefix}/{_esc(k)}")
    elif isinstance(tree, list) and tree:
        for i, v in enumerate(tree):
            out += leaves(v, f"{prefix}/{i}")
    else:
        out.append((prefix, tree))
    return out


def nodes(tree, prefix=""):
    """Every node of a JSON tree, the root excluded → [(JSON pointer, value)] in document order."""
    out = []
    items = tree.items() if isinstance(tree, dict) else enumerate(tree) if isinstance(tree, list) else ()
    for k, v in items:
        p = f"{prefix}/{_esc(k)}"
        out.append((p, v))
        out += nodes(v, p)
    return out


def _tokens(path):
    return [_unesc(t) for t in path.split("/")[1:]]


def _parent_and_key(tree, path):
    toks = _tokens(path)
    node = tree
    for t in toks[:-1]:
        node = node[int(t)] if isinstance(node, list) else node[t]
    last = toks[-1]
    return node, (int(last) if isinstance(node, list) else last)


def _set(tree, path, value):
    node, key = _parent_and_key(tree, path)
    node[key] = value


def _plain(proposal):
    """A proposal as a JSON tree (a pydantic model → its JSON form; a dict or a list through JSON, a copy)."""
    if hasattr(proposal, "model_dump"):
        return proposal.model_dump(mode="json")
    return json.loads(json.dumps(proposal, ensure_ascii=False, default=str))


# ------------------------------------------------------------------------------------------------ built-in mutators
def _has_digit(v):
    return not isinstance(v, bool) and isinstance(v, (str, int, float)) and any(c.isdigit() for c in str(v))


def change_digit(paths=".*"):
    """One digit of a string or number leaf whose JSON pointer matches `paths` (a regex) becomes another digit; the leaf
    keeps its type (a number stays a number). Caught: the mutated value is not accepted at its path."""
    rx = re.compile(paths)

    def mutate_digit(p, rng):
        targets = [(path, v) for path, v in leaves(p) if rx.search(path) and _has_digit(v)]
        if not targets:
            return None
        path, v = rng.choice(targets)
        s = str(v)
        pos = [i for i, c in enumerate(s) if c.isdigit()]
        i = rng.choice(pos)
        new = s[:i] + rng.choice([d for d in "0123456789" if d != s[i]]) + s[i + 1:]
        if isinstance(v, int):
            value = int(new)
        elif isinstance(v, float):
            value = float(new)
        else:
            value = new
        _set(p, path, value)
        return p, path, value
    mutate_digit.criterion = "accepted"
    return mutate_digit


def drop(paths=".*"):
    """An element of a list, its JSON pointer matching `paths` (e.g. r"^/series/\\d+/points/\\d+$"), is removed. Caught:
    the run reports an issue the unmutated run did not — a relation broke (see the module docs: an omission has no
    place of its own to be refused at)."""
    rx = re.compile(paths)

    def mutate_drop(p, rng):
        targets = [path for path, _ in nodes(p) if rx.search(path) and isinstance(_parent_and_key(p, path)[0], list)]
        if not targets:
            return None
        path = rng.choice(targets)
        node, key = _parent_and_key(p, path)
        value = node.pop(key)
        return p, path, value
    mutate_drop.criterion = "relation"
    mutate_drop.caught = _new_issue
    return mutate_drop


def swap(paths=".*"):
    """Two nodes whose JSON pointers match `paths` and differ in one list index only — two elements of one list
    (r"^/series/\\d+/points/\\d+$") or the same field of two of its elements (r"^/series/\\d+/points/\\d+/value$") —
    exchange their values (only two that differ). Caught: the value now at the first of them is not accepted there."""
    rx = re.compile(paths)

    def mutate_swap(p, rng):
        found = [(path, _tokens(path), v) for path, v in nodes(p) if rx.search(path)]
        pairs = []
        for i, (pa, ta, va) in enumerate(found):
            for pb, tb, vb in found[i + 1:]:
                if len(ta) != len(tb) or va == vb:
                    continue
                diff = [k for k, (x, y) in enumerate(zip(ta, tb)) if x != y]
                if len(diff) == 1 and ta[diff[0]].isdigit() and tb[diff[0]].isdigit():
                    pairs.append((pa, pb))
        if not pairs:
            return None
        a, b = rng.choice(pairs)
        na, ka = _parent_and_key(p, a)
        nb, kb = _parent_and_key(p, b)
        na[ka], nb[kb] = nb[kb], na[ka]
        return p, a, copy.deepcopy(na[ka])
    mutate_swap.criterion = "accepted"
    return mutate_swap


# ------------------------------------------------------------------------------------------------ judging
def default_accepted(run):
    """What a run's check verified → {JSON pointer: value}: the leaves of `run.checked.verified` in JSON form ({} when
    nothing verified)."""
    v = run.checked.verified if run.checked is not None else None
    if v is None:
        return {}
    tree = v.model_dump(mode="json") if hasattr(v, "model_dump") else _plain(v)
    return dict(leaves(tree))


def _same(a, b):
    if a == b:
        return True
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return False
    try:
        return Decimal(str(a)) == Decimal(str(b))
    except (InvalidOperation, ValueError):
        return False


def _held(acc, path, value):
    """Is `value` accepted at `path`? A leaf: equal there. A dict / list: every one of its leaves present at the same
    place is equal, and at least one is present."""
    if not isinstance(value, (dict, list)) or not value:
        return path in acc and _same(acc[path], value)
    seen = 0
    for sub, v in leaves(value, path):
        if sub in acc:
            if not _same(acc[sub], v):
                return False
            seen += 1
    return seen > 0


def _new_issue(run, base, path, value):
    """Does the run report an issue (severity and code, counted) the unmutated run did not?"""
    now = Counter((i.severity, i.code) for i in run.checked.issues)
    before = Counter((i.severity, i.code) for i in (base.checked.issues if base is not None else ()))
    return bool(now - before)


@dataclass
class CorruptionReport:
    """What a corruption test found: n applied mutations, how many were caught, the rate, per kind {"n", "caught",
    "skipped"}, and one row per mutation {"id", "kind", "path", "value", "caught", "criterion"}."""
    n: int = 0
    caught: int = 0
    per_kind: dict = field(default_factory=dict)
    rows: list = field(default_factory=list)

    @property
    def rate(self):
        return self.caught / self.n if self.n else 0.0

    @property
    def skipped(self):
        return sum(k["skipped"] for k in self.per_kind.values())

    def to_dict(self):
        return {"n": self.n, "caught": self.caught, "rate": self.rate, "skipped": self.skipped,
                "per_kind": self.per_kind, "rows": self.rows}

    def __str__(self):
        lines = [f"corruptions: {self.caught} of {self.n} caught ({self.rate:.1%}), {self.skipped} draw(s) skipped"]
        for k, s in self.per_kind.items():
            lines.append(f"  {k}: {s['caught']} of {s['n']} caught" + (f", {s['skipped']} skipped" if s["skipped"] else ""))
        return "\n".join(lines)


def corrupt(specialist, records, corruptions, *, n=300, seed=0, accepted=None):
    """Corrupt recorded proposals and run the specialist on each corruption → CorruptionReport (see the module docs).

    specialist: a solvi.core.specialist.Specialist; records: (id, source, proposal) — the proposal as recorded
    (`Run.proposal`, or a stored record's "proposal"); corruptions: {kind: mutator} (a mutator: see the module docs); n: mutations to apply; seed: the
    draws' seed; accepted: run → {JSON pointer: value} of what the check verified (default: default_accepted)."""
    recs = sorted(((str(i), s, _plain(p)) for i, s, p in records), key=lambda r: r[0])
    if not recs or not corruptions:
        raise ValueError("corrupt needs records and corruptions")
    accepted = accepted or default_accepted
    kinds = list(corruptions)
    rng = random.Random(seed)
    rep = CorruptionReport(per_kind={k: {"n": 0, "caught": 0, "skipped": 0} for k in kinds})
    base = {}
    draws = 0
    while rep.n < n and draws < 20 * n:
        draws += 1
        rid, source, proposal = rng.choice(recs)
        kind = rng.choice(kinds)
        m = corruptions[kind]
        got = m(copy.deepcopy(proposal), rng)
        if got is None:
            rep.per_kind[kind]["skipped"] += 1
            continue
        mutated, path, value = got
        run = specialist.run(source, proposal=mutated)
        judge = getattr(m, "caught", None)
        criterion = getattr(m, "criterion", None) or ("own" if judge is not None else "accepted")
        if run.output is None:
            caught = True
        elif judge is not None:
            if rid not in base:
                base[rid] = specialist.run(source, proposal=copy.deepcopy(proposal))
            caught = bool(judge(run, base[rid], path, value))
        else:
            caught = not _held(accepted(run), path, value)
        rep.n += 1
        rep.caught += caught
        rep.per_kind[kind]["n"] += 1
        rep.per_kind[kind]["caught"] += caught
        rep.rows.append({"id": rid, "kind": kind, "path": path, "value": value, "caught": caught, "criterion": criterion})
    return rep


__all__ = ["change_digit", "default_accepted", "drop", "leaves", "corrupt", "CorruptionReport", "nodes", "swap"]
