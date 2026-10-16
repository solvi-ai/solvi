"""Regression cases from what a System has decided: `cases_from_store` turns recorded decisions into cases (their
inputs and the answers given), `boundary` makes inputs at the edges of what the cases show — per field the values seen,
then empty / single / long text and its Unicode forms, 0, -1, one past the smallest and largest number, a huge
positive and negative one, either side of a rule's threshold, None and another type where the field has shown them, an
empty, one-element or long list (of distinct values too), many dict keys, one level into dicts and lists — and, given
the System, pins its current answers on them as the expected ones. Both give case dicts in the cases.json format (see
solvi.testing), so `solvi test` and `run_case` run them."""
from __future__ import annotations

import copy
import json
import unicodedata
import warnings

from .honesty import _plain

LONG = 1000             # a "long" text or list has this many characters / elements
BIG = 2 ** 63           # a number past a 64-bit integer


def _key(x):
    return json.dumps(x, sort_keys=True, default=str)


def _outcome(answers, status):
    return _key([answers, status])


def cases_from_store(storage, n=None):
    """Regression cases from the decisions a store holds → [case dict]: per decision its recorded input as "state",
    the answers it gave as "expected" (an abstention as null), each question's status as "status" and the questions
    it asked as "ask"; the name is "stored <id>". The same input recorded twice is one case (the first). With n: at most
    n cases — one per distinct outcome (answers and statuses) first, in store order, then the rest in store order.
    Full and compact records alike (both keep the input and the answers); redacted ones are passed over. The state is
    the input as it was given to ask: a task whose cases go through prepare(state) should not prepare it again."""
    out, seen = [], set()
    for s in storage.iter():
        state = s.input
        if not isinstance(state, dict):
            continue
        k = _key(state)
        if k in seen:
            continue
        seen.add(k)
        rec = s.data.get("answers") or {}
        answers = s.answers
        status = {q: a[2] for q, a in rec.items() if len(a) > 2}
        expected = {q: None if status.get(q) == "abstain" else a for q, a in answers.items()}
        out.append({"name": f"stored {s.id}", "state": state, "expected": expected, "status": status,
                    "ask": list(answers)})
    if n is None or len(out) <= n:
        return out
    first, rest, outs = [], [], set()
    for c in out:
        o = _outcome(c["expected"], c["status"])
        (rest if o in outs else first).append(c)
        outs.add(o)
    keep = {id(c) for c in (first + rest)[:n]}
    return [c for c in out if id(c) in keep]


# --- boundary inputs
def _kind(v):
    return type(v).__name__


class _Profile:
    """What the cases show about one field: the values seen (by kind), the first value of each type, numbers' range,
    texts' and lists' lengths, and — one level deep — a profile per key of a dict value and of a list's elements."""

    def __init__(self):
        self.values = []
        self.kinds = {}                 # type name → the first value of that type seen (None included)
        self.nums = []
        self.keys = {}
        self.elem = None

    def add(self, v, depth=0):
        if len(self.values) < 8 and v not in self.values:
            self.values.append(v)
        self.kinds.setdefault(_kind(v), v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            self.nums.append(v)
        if depth < 1 and isinstance(v, dict):
            for k, x in v.items():
                self.keys.setdefault(k, _Profile()).add(x, depth + 1)
        if depth < 1 and isinstance(v, list):
            for x in v:
                self.elem = self.elem or _Profile()
                self.elem.add(x, depth + 1)


_ACCENT = str.maketrans("aeiouncAEIOUNC", "áéíóúñçÁÉÍÓÚÑÇ")


def _text(s):
    base = s or "abc"
    acc = base.translate(_ACCENT)
    if acc == base:
        acc = base + "é"
    acc = unicodedata.normalize("NFC", acc)
    return [("empty text", ""), ("one character", base[:1]), ("long text", (base * (LONG // len(base) + 1))[:LONG]),
            ("non-ASCII text", base + " é Привет 😀"), ("padded text", f"  {base}  "), ("upper case", base.upper()),
            # the same accents decomposed (a letter + a combining mark) and a compatibility ligature with a special
            # case mapping: an input that compares, casefolds or normalises text in one form only answers differently
            ("decomposed accents (NFD)", unicodedata.normalize("NFD", acc)),
            ("compatibility ligature", base + "\ufb01\u00df")]


def _number(v, p):
    out = [("zero", 0 if isinstance(v, int) else 0.0), ("minus one", -1 if isinstance(v, int) else -1.0)]
    if p is not None and p.nums:
        lo, hi = min(p.nums), max(p.nums)
        step = 1 if all(isinstance(x, int) for x in p.nums) else max(abs(hi - lo) * 0.01, 1e-9)
        out += [("one past the largest seen", hi + step), ("one below the smallest seen", lo - step),
                ("the largest seen", hi), ("the smallest seen", lo)]
    out += [("one above", v + 1), ("one below", v - 1)]
    out.append(("huge", BIG if isinstance(v, int) else 1e300))
    out.append(("huge negative", -BIG if isinstance(v, int) else -1e300))
    return out


def _thresholds(v, cs):
    """One below, at and one above each rule constant the field is compared with → [(what, value)]: ±1 for an int
    field and a whole constant, else the nearest floats on either side."""
    import math
    out = []
    for c in cs:
        if isinstance(v, int) and float(c).is_integer():
            c = int(c)
            below, above = c - 1, c + 1
        else:
            c = float(c)
            below, above = math.nextafter(c, -math.inf), math.nextafter(c, math.inf)
        out += [(f"below a rule threshold ({c!r})", below), (f"at a rule threshold ({c!r})", c),
                (f"above a rule threshold ({c!r})", above)]
    return out


def _variants(v, p, depth=0):
    """Boundary replacements of one field's value → [(what, value)]: by its type, then None when the field has shown
    None and a value of each other type the field has shown (a number recorded once as a string, say)."""
    out = _by_type(v, p, depth)
    if p is not None:
        if v is not None and "NoneType" in p.kinds:
            out.append(("none", None))
        for k, x in p.kinds.items():
            if k not in (_kind(v), "NoneType"):
                out.append(("another type seen", _as(x, v)))
    return out


def _as(example, v):
    """v in the type of `example` when it converts cleanly (7 → "7", "7" → 7, 2.0 → 2), else the example itself."""
    try:
        if isinstance(example, str) and isinstance(v, (int, float)) and not isinstance(v, bool):
            return str(v)
        if isinstance(example, (int, float)) and not isinstance(example, bool) and isinstance(v, (str, int, float)) \
                and not isinstance(v, bool):
            x = type(example)(v)
            return x if x == float(v) else example
    except (TypeError, ValueError, OverflowError):
        pass
    return example


def _by_type(v, p, depth=0):
    if isinstance(v, bool):
        return [("flipped", not v)]
    if isinstance(v, (int, float)):
        return _number(v, p)
    if isinstance(v, str):
        return _text(v) + [("another value seen", x) for x in (p.values if p else ()) if isinstance(x, str) and x != v]
    if isinstance(v, (set, frozenset)):
        out = [("empty set", type(v)())]
        many = _distinct(sorted(v, key=repr)[0] if v else None)
        if many is not None:
            out.append(("long set of distinct values", type(v)(many)))
        return out
    if isinstance(v, list):
        out = [("empty list", []), ("one element", v[:1])] if v else []
        if v:
            out.append(("long list", [v[i % len(v)] for i in range(LONG)]))
            many = _distinct(v[0])
            if many is not None:
                out.append(("long list of distinct values", many))
            if depth < 1:
                for what, x in _variants(v[0], p.elem if p else None, depth + 1):
                    out.append((f"first element: {what}", [x] + v[1:]))
        return out
    if isinstance(v, dict) and depth < 1:
        out = [("empty dict", {})]
        if v:
            k0, x0 = next(iter(v.items()))
            keys = _distinct(k0)
            if keys is not None:
                extra = [k for k in keys if k not in v][:max(LONG - len(v), 0)]
                out.append(("many keys", {**v, **{k: copy.deepcopy(x0) for k in extra}}))
        for k, x in v.items():
            for what, y in _variants(x, (p.keys.get(k) if p else None), depth + 1):
                out.append((f"{k}: {what}", {**v, k: y}))
        return out
    return []


def _distinct(example):
    """LONG distinct values of the example's type (texts or ints), or None for another type: a long list, set or set of
    dict keys whose elements are all different (a repeated element does not reach code that dedupes or indexes)."""
    if isinstance(example, str):
        return [f"{example}{i}" for i in range(LONG)]
    if isinstance(example, int) and not isinstance(example, bool):
        return [example + i for i in range(LONG)]
    return None


def _rule_thresholds(catalog):
    """{field: [numbers]}: the constants the catalog's parts compare their inputs with — `amount >= 100`,
    `0.5 < score <= LIMIT` (a number the module or the closure holds) — read from each part's source, inputs named as
    the function's arguments. A comparison of an expression (`x * 2 > 100`) or of an item (`row["x"] > 1`) is not
    read; nor are the functions a part calls."""
    import ast
    import inspect
    import textwrap
    out = {}
    parts = list(getattr(catalog, "parts", {}).values()) + list(getattr(catalog, "rules", {}).values())
    funcs = []
    for p in parts:
        funcs += [a.func for a in p.alternatives] if getattr(p, "alternatives", None) else [p.func]
    for f in funcs:
        try:
            f = inspect.unwrap(f)
            if not inspect.isfunction(f):
                continue
            tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        except (OSError, TypeError, SyntaxError, IndentationError, ValueError):
            continue
        args = set(inspect.signature(f).parameters)
        env = dict(f.__globals__)
        for name, cell in zip(f.__code__.co_freevars, f.__closure__ or ()):
            try:
                env[name] = cell.cell_contents
            except ValueError:
                pass

        def number(node):
            v = node.value if isinstance(node, ast.Constant) else env.get(node.id) if isinstance(node, ast.Name) \
                else -number(node.operand) if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) \
                and number(node.operand) is not None else None
            return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            ops = [node.left] + node.comparators
            for a, b in zip(ops, ops[1:]):
                for x, y in ((a, b), (b, a)):
                    if isinstance(x, ast.Name) and x.id in args and number(y) is not None:
                        cs = out.setdefault(x.id, [])
                        if number(y) not in cs:
                            cs.append(number(y))
    return out


def _store_kw(system):
    """store=False for System.ask (an object with a plain ask(state) gets nothing): a boundary input is not a decision."""
    import inspect
    try:
        return {"store": False} if "store" in inspect.signature(system.ask).parameters else {}
    except (TypeError, ValueError):
        return {}


def _answers(res):
    expected = {q: None if r.status == "abstain" else _plain(r.answer) for q, r in res.results.items()}
    return expected, {q: r.status for q, r in res.results.items()}


def _shape(state):
    return tuple(sorted((k, _kind(v)) for k, v in state.items()))


def _by_shape(states):
    """The states in the order their changes are taken: the first state of each distinct shape (field → type), then
    the shortest and the longest of the rest, then the rest in the order given — so with many cases of one shape the
    changes do not all come from the first few of them."""
    seen, first, rest = set(), [], []
    for st in states:
        sh = _shape(st)
        (rest if sh in seen else first).append(st)
        seen.add(sh)
    if len(rest) > 2:
        size = [len(_key(st)) for st in rest]
        lo = min(range(len(rest)), key=lambda i: (size[i], i))
        hi = max(range(len(rest)), key=lambda i: (size[i], -i))
        ends = [lo] if lo == hi else [lo, hi]
        rest = [rest[i] for i in ends] + [st for i, st in enumerate(rest) if i not in ends]
    return first + rest


def boundary(cases, n=50, system=None, thresholds=None):
    """Inputs at the edges of what the cases show → up to n new case dicts (name, state, note "boundary: <field>:
    <what>"): each case's state with one field changed — to a value just past the range the cases show, 0, -1, a huge
    number and a huge negative one; one below, at and one above each constant a rule compares the field with
    (`amount >= 100` → 99, 100, 101; see thresholds); empty, one-character, long, non-ASCII, padded text, its accents
    decomposed (NFD) and a compatibility ligature; None where the field has shown None, a value of another type the
    field has shown (the value itself converted when it converts: 7 → "7"); an empty, one-element or long list, a long
    list or set of distinct values, a dict with many keys; one level into dicts and lists.
    Deduplicated, without the cases' own states, taken in turn from each kind of change and, within a kind, from the
    first case of each input shape (field → type) first, then the shortest and the longest case, then the rest; the
    same for the same cases (no randomness).
    system: the current answers are pinned — "expected" and "status" from asking each input (never stored), so the
    cases fail when a later change moves an answer at an edge. An input whose ask raises is left out (fuzz reports
    those), so is one whose answers differ when asked again or whose trace does not replay (nondeterministic); a
    UserWarning says how many. Without a system the cases have no expectations: fill them in before running.
    thresholds: {field: [numbers]} to cut at; None (default) — the constants the system's catalog compares each field
    with, read from the parts' source where a top-level input is compared with a number or a numeric constant (none
    without a system); False — none."""
    states = _by_shape([c["state"] for c in cases if isinstance(c.get("state"), dict)])
    if thresholds is None:
        thresholds = _rule_thresholds(getattr(system, "catalog", None)) if system is not None else {}
    thresholds = thresholds or {}
    prof = {}
    for st in states:
        for k, v in st.items():
            prof.setdefault(k, _Profile()).add(v)
    seen = {_key(st) for st in states}
    by_kind = {}
    for i, st in enumerate(states):
        for k, v in st.items():
            cut = _thresholds(v, thresholds.get(k, ())) if isinstance(v, (int, float)) and not isinstance(v, bool) \
                else []
            for what, x in _variants(v, prof.get(k)) + cut:
                new = {**copy.deepcopy(st), k: x}
                key = _key(new)
                if key in seen:
                    continue
                seen.add(key)
                kind = what.split(": ")[-1].split(" (")[0]
                by_kind.setdefault(kind, []).append({"name": f"boundary {len(seen)}", "state": new,
                                                     "note": f"boundary: {k}: {what}"})
    queues = list(by_kind.values())                   # in turn from each kind of change
    out = []
    while queues and len(out) < (n if system is None else 4 * n):
        for q in list(queues):
            if q:
                out.append(q.pop(0))
            else:
                queues.remove(q)
    for i, c in enumerate(out):
        c["name"] = f"boundary {i + 1}"
    if system is None:
        return out[:n]
    kept, crashed, unstable = [], 0, 0
    for c in out:
        if len(kept) >= n:
            break
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = system.ask(copy.deepcopy(c["state"]), **_store_kw(system))
                b = system.ask(copy.deepcopy(c["state"]), **_store_kw(system))
                rep = a.trace.replay(system)
        except Exception:  # noqa: BLE001
            crashed += 1
            continue
        if _answers(a) != _answers(b) or not rep["ok"]:
            unstable += 1
            continue
        c["expected"], c["status"] = _answers(a)
        kept.append(c)
    for i, c in enumerate(kept):
        c["name"] = f"boundary {i + 1}"
    if crashed or unstable:
        warnings.warn(f"boundary: {crashed} input(s) left out because ask raised (run fuzz for them), {unstable} because "
                      "the answers were nondeterministic (another answer when asked again, or a trace that does not "
                      "replay)", UserWarning, stacklevel=2)
    return kept


__all__ = ["boundary", "cases_from_store"]
