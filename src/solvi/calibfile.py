"""Calibrations as files, and `solvi calibrate`.

`act_guard`, `calibrate_for` and `conformal` set a decision's thresholds in memory. A calibration file keeps them, so a
catalog calibrated once (on a few hundred labelled examples of your stream) loads the same thresholds every time it starts:

    info = part.act_guard(examples, risk=0.10)
    part.save_calibration("team.calib.json")

    # in the catalog module, after making the part and before registering it (groups add the part's inputs):
    part = model.decision("team", "Which team?", "email", TEAMS)
    part.load_calibration("team.calib.json")
    part.question(cat)

A file holds the escalation thresholds (`escalate_below`, `act_threshold`, per group; for a combination the shared
threshold, its `scale` and — on the rank scale — each member's sorted calibration signals, at most 1024 per member),
the guarantee record the trace shows, the conformal set, and what they were fitted for: the question (task, options,
kind) and the fingerprint of the model behind it — the checkpoint and this question's adaptation for a DecisionPart, every member for a Cascade / Vote /
Route. `load_calibration` refuses a file made for another question or another model (a threshold on one model's
confidence says nothing about another's); `strict=False` loads it anyway. After loading, the part's fingerprint is the
one it had right after calibrating, so stored decisions replay against it.

Thresholds per group by fact names load as they are; by a function, pass the same function again:
`part.load_calibration(path, groups=my_grouping)` (its code is fingerprinted and must match).

    solvi calibrate myapp.decisions:system team labels.csv --risk 0.1 [--groups domain,task] [--method crc|ltt]
                    [--conformal 0.9] [--out team.calib.json] [--json]

LABELS: a CSV or JSON-lines file, one labelled example per row: `label` (the correct answer; a multi-label answer is a
JSON list, or "a|b" in a CSV; "<not stated>" for solvi.Unknown) plus the input — the facts the part reads as columns
(`email`), or one `text` / `input` column, or else the other columns as a state. Group columns (--groups) are read as
facts. Exit status: 0 — something is answered alone; 1 — the calibration escalates everything (no threshold holds);
2 — usage errors."""
from __future__ import annotations

import csv
import json
import math
import sys

FORMAT = "solvi calibration v1"
_recalibrating = None           # `solvi calibrate` loads the catalog with calibration files skipped (listed here)


# --------------------------------------------------------------------------------------------------- JSON with inf
def _enc(v):
    if isinstance(v, float) and not math.isfinite(v):
        return {"$float": repr(v)}
    if isinstance(v, dict):
        return {k: _enc(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_enc(x) for x in v]
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):     # numpy scalars
        return _enc(v.item())
    return v


def _dec(v):
    if isinstance(v, dict):
        if set(v) == {"$float"}:
            return float(v["$float"])
        return {k: _dec(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_dec(x) for x in v]
    return v


def _norm(v):
    """JSON data as it reads back from the file (tuples as lists, other objects as their str)."""
    return _dec(json.loads(json.dumps(_enc(v), default=str)))


# --------------------------------------------------------------------------------------------------- what is saved
def _kind_of(part):
    from .decide import DecisionPart
    if isinstance(part, DecisionPart):
        return "DecisionPart"
    from .multi import _Combination
    if isinstance(part, _Combination):
        return type(part).__name__
    raise TypeError(f"a calibration file is made from a DecisionPart or a Cascade / Vote / Route, not {type(part).__name__}")


def base_fingerprint(part):
    """What a calibration is fitted to: the part's fingerprint without its thresholds (the checkpoint, the question, this
    question's adaptation, and how the part computes its signal — option_order="average" with its permutations, long=
    with top_k and rerank; for a combination every member)."""
    from .provenance import digest
    if _kind_of(part) == "DecisionPart":
        a = part.adaptation
        signal = {k: v for k, v in (("option_order", None if part.option_order != "average" else
                                     (part.option_order, part.permutations)),
                                    ("long", part.long_key()),
                                    ("lora", None if part.lora is None else part.lora.hash))
                  if v is not None}
        if signal:                                  # a part with the default signal keeps the fingerprint it had
            return digest("DecisionPart", part.model.weights_fingerprint(), part.spec.describe(), a.params() if a else None,
                          signal)
        return digest("DecisionPart", part.model.weights_fingerprint(), part.spec.describe(), a.params() if a else None)
    return digest(type(part).__name__, part._describe(), [m.fingerprint() for m in part.members], {})


def _models(part):
    """{model id: weights fingerprint} of the models behind a part (for the message when they differ)."""
    if _kind_of(part) == "DecisionPart":
        return {str(part.model_id): part.model.weights_fingerprint()}
    return {str(lf.part.model_id): lf.part.model.weights_fingerprint() for lf in part.leaves()}


def _groups_record(groups):
    if groups is None:
        return None
    by = groups["by"]
    return {"by": by.describe(), "label": by.label(), "signal": groups.get("signal"),
            "nodes": [[list(k), {"threshold": v["threshold"], "n": v["n"]}] for k, v in groups["nodes"].items()]}


def record(part):
    """A part's calibration as JSON-ready data (see the module docs)."""
    kind = _kind_of(part)
    out = {"format": FORMAT, "kind": kind, "name": part.__name__, "question": _norm(part.spec.describe()),
           "fingerprint": base_fingerprint(part), "models": _models(part)}
    if kind == "DecisionPart":
        out.update(escalate_below=part.escalate_below, act_threshold=part.act_threshold)
    else:
        out["threshold"] = part.threshold
        if part.scale is not None:
            out["scale"] = part.scale
        if part.scale == "rank":                    # each member's sorted calibration signals (at most multi.MAX_RANKS)
            out["ranks"] = [[float(x) for x in r] for r in part.ranks]
    out.update(guarantee=part.guarantee, conformal=part.conformal_set, groups=_groups_record(part.groups))
    if kind == "DecisionPart" and part.lora is not None:     # the adapter it was calibrated with, in a file beside it
        out["lora"] = {"hash": part.lora.hash}
    from . import __version__
    out["solvi"] = __version__
    return _norm(out)


def lora_path(path):
    """The adapter file written next to a calibration file: team.calib.json → team.calib.lora.safetensors."""
    base = str(path)[:-5] if str(path).endswith(".json") else str(path)
    return base + ".lora.safetensors"


def save(part, path):
    """Write a part's calibration to a JSON file (and its LoRA adapter, if it has one, next to it) → path."""
    import os
    rec = record(part)
    if "lora" in rec:
        from .lora import save as save_lora
        save_lora(part, lora_path(path))
        rec["lora"]["file"] = os.path.basename(lora_path(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_enc(rec), fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")
    return path


def _group_by(rec, groups):
    from .decide import GroupBy
    by = rec["by"]
    if isinstance(by, list):
        g = GroupBy(groups if groups is not None else by)
    else:
        if groups is None:
            raise ValueError(f"the calibration has thresholds per group by a function ({rec['label']}): pass it again, "
                             "load_calibration(path, groups=that_function)")
        g = GroupBy(groups)
    if _enc(g.describe()) != by:
        raise ValueError(f"groups= {g.label()} is not the grouping the calibration was made with ({rec['label']}): "
                         f"{g.describe()} != {by}")
    return g


def load(part, path, groups=None, strict=True):
    """Apply a calibration file to a part (see the module docs) → the part. strict: refuse a file made for another
    question or another model."""
    if _recalibrating is not None:                  # solvi calibrate: the part is calibrated afresh
        _recalibrating.append(str(path))
        return part
    with open(path, encoding="utf-8") as fh:
        rec = _dec(json.load(fh))
    if rec.get("format") != FORMAT:
        raise ValueError(f"{path}: not a solvi calibration file (format {rec.get('format')!r}, expected {FORMAT!r})")
    kind = _kind_of(part)
    if rec.get("kind") != kind:
        raise ValueError(f"{path}: a calibration of a {rec.get('kind')}, not of a {kind}")
    lo = rec.get("lora")
    if kind == "DecisionPart" and lo and (part.lora is None or part.lora.hash != lo.get("hash")):
        import os                                   # calibrated with an adapter: load it first (from beside the file)
        from .lora import load as load_lora
        f = os.path.join(os.path.dirname(os.path.abspath(str(path))), lo.get("file") or os.path.basename(lora_path(path)))
        load_lora(part, f, strict, expect=lo.get("hash"))
    if strict:
        if rec.get("question") != _norm(part.spec.describe()):
            raise ValueError(f"{path}: made for another question ({rec.get('name')!r}: {rec.get('question')}); this part "
                             f"asks {part.spec.describe()}")
        fp = base_fingerprint(part)
        if rec.get("fingerprint") != fp:
            mine = _models(part)
            theirs = rec.get("models") or {}
            if theirs != mine:
                raise ValueError(f"{path}: calibrated on another model ({', '.join(f'{k} #{v}' for k, v in theirs.items())}); "
                                 f"this part runs {', '.join(f'{k} #{v}' for k, v in mine.items())} — calibrate again "
                                 "(solvi calibrate) with this model")
            raise ValueError(f"{path}: calibrated for fingerprint #{rec.get('fingerprint')}, this part is #{fp}: the "
                             "question's adaptation (fit / teach / adapt), its LoRA adapter or how the part computes its "
                             "signal (option_order=\"average\", permutations, long, top_k, rerank) differs — load the "
                             "adaptations it was calibrated with (model.load_adaptations), make the part as it was, or "
                             "calibrate again")
    grp = None
    if rec.get("groups") is not None:
        g = rec["groups"]
        grp = {"by": _group_by(g, groups), "nodes": {tuple(k): dict(v) for k, v in g["nodes"]}}
        if kind == "DecisionPart":
            grp["signal"] = g.get("signal")
    if kind == "DecisionPart":
        part._set_threshold("act", rec.get("act_threshold"), rec.get("guarantee"), grp)   # sets groups, inputs
        part.escalate_below, part.act_threshold = rec.get("escalate_below"), rec.get("act_threshold")   # both, as saved
        part.guarantee = rec.get("guarantee")
        part.conformal_set = rec.get("conformal")
    else:
        part.threshold, part.guarantee, part.groups = rec.get("threshold"), rec.get("guarantee"), grp
        scale = rec.get("scale", "raw")             # a file from before 0.7 has no scale: raw, as it was made
        if scale == "rank":
            import numpy as np
            ranks = rec.get("ranks")
            if not isinstance(ranks, list) or len(ranks) != len(part.leaves()):
                raise ValueError(f"{path}: a rank-scale calibration needs the calibration signals of each of the "
                                 f"{len(part.leaves())} models (\"ranks\")")
            part.scale, part.ranks = "rank", [np.asarray(r, float) for r in ranks]
        elif scale == "raw":
            part.scale, part.ranks = "raw", None
        else:
            raise ValueError(f"{path}: unknown scale {scale!r} (rank or raw)")
        part.conformal_set = rec.get("conformal")
        part._setup()
    return part


# --------------------------------------------------------------------------------------------------- the command
def _cell(v):
    """A CSV cell: JSON when it looks like a list / object, else the text."""
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except ValueError:
            return v
    return v


def read_rows(path):
    """A CSV or JSON-lines file (or "-": JSON lines from stdin) → [dict]."""
    fh = sys.stdin if path == "-" else open(path, encoding="utf-8", newline="")
    try:
        if path.endswith(".csv"):
            return [{k: _cell(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        rows = []
        for n, line in enumerate(fh, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f"{path}:{n}: not JSON: {e}") from None
        return rows
    finally:
        if fh is not sys.stdin:
            fh.close()


def label_of(part, y):
    """A label as written in a file → an answer the part can take."""
    from .core import NOT_STATED_KEY, Unknown
    if y == NOT_STATED_KEY:
        return Unknown
    sp = part.spec
    if isinstance(y, str):
        s = y.strip()
        if sp.kind == "noul" and s.lower() in ("true", "false", "yes", "no", "1", "0"):
            return s.lower() in ("true", "yes", "1")
        if sp.multi:
            return [x.strip() for x in s.split("|") if x.strip()]
        if sp.kind == "number":
            try:
                return float(s)
            except ValueError:
                return s
        return s
    return y


def examples_of(part, rows, group_cols=()):
    """Rows (dicts with "label") → [(Facts, label)] for act_guard / calibrate_for / conformal."""
    from .decide import Facts
    facts = list(part.facts) if hasattr(part, "facts") else []
    facts = [f for f in facts if f not in group_cols]
    out = []
    for i, row in enumerate(rows, 1):
        if "label" not in row:
            raise ValueError(f"row {i}: no 'label' column")
        missing = [c for c in group_cols if c not in row]
        if missing:
            raise ValueError(f"row {i}: no group column {missing}")
        vals = {c: row[c] for c in group_cols}
        if all(f in row for f in facts):
            vals.update({f: row[f] for f in facts})
        elif len(facts) == 1:
            if "text" in row or "input" in row:
                vals[facts[0]] = row["text"] if "text" in row else row["input"]
            else:
                vals[facts[0]] = {k: v for k, v in row.items() if k != "label" and k not in group_cols}
        else:
            raise ValueError(f"row {i}: the part reads {facts}; give each as a column")
        out.append((Facts(vals), label_of(part, row["label"])))
    return out


def find_part(system, name):
    """The decision behind a question (its rule) or a catalog part by name → a DecisionPart / combination, or None."""
    from .decide import decision_of
    cat = system.catalog if hasattr(system, "catalog") else system
    d = decision_of(cat, name) if name in cat.rules else None
    if d is None and name in cat.parts:
        p = cat.parts[name]
        d = getattr(p.func, "__solvi_decision__", None) if p.alternatives is None else None
    return d


def _pct(x):
    return f"{100 * x:.1f}%"


def cmd_calibrate(a):
    """`solvi calibrate` (see solvi.cli) → exit status."""
    global _recalibrating
    from .cli import _dump, _fail, load_object
    _recalibrating = []
    try:                                           # calibration files the catalog loads are skipped: a file made for
        obj = load_object(a.system)                # another model must not stop it from being calibrated again
    finally:
        skipped, _recalibrating = _recalibrating, None
    if not hasattr(obj, "catalog") and not (hasattr(obj, "parts") and hasattr(obj, "rules")):
        _fail(f"calibrate {a.system}: not a solvi System or Catalog")
    part = find_part(obj, a.part)
    if part is None:
        _fail(f"calibrate: {a.part!r} is not a question answered by a model decision, nor a decision part")
    try:
        _kind_of(part)
    except TypeError as e:
        _fail(f"calibrate: {e}")
    groups = [g.strip() for g in a.groups.split(",") if g.strip()] if a.groups else None
    try:
        rows = read_rows(a.labels)
        if a.limit:
            rows = rows[:a.limit]
        examples = examples_of(part, rows, groups or ())
    except FileNotFoundError:
        _fail(f"no such file: {a.labels}")
    except ValueError as e:
        _fail(f"{a.labels}: {e}")
    if not examples:
        _fail(f"{a.labels}: no labelled examples")
    try:
        if a.method == "ltt":
            if groups:
                _fail("calibrate --method ltt: no thresholds per group (use --method crc)")
            if not hasattr(part, "calibrate_for"):
                _fail("calibrate --method ltt: a combination calibrates with act_guard (--method crc)")
            info = part.calibrate_for(examples, error=a.risk, method="ltt", delta=a.delta)
        else:
            info = part.act_guard(examples, risk=a.risk, groups=groups, min_group=a.min_group,
                                  delta=None if a.delta < 0 else a.delta)
        if a.conformal:
            info["conformal"] = part.conformal(examples, coverage=a.conformal)
    except ValueError as e:
        _fail(f"calibrate: {e}")
    out = a.out or f"{a.part}.calib.json"
    save(part, out)
    info["saved"] = out
    if skipped:
        info["not_applied"] = skipped
    thr = info["threshold"]
    ok = thr is not None and math.isfinite(thr)
    if a.json:
        _dump(_enc({k: ({"/".join(p) or "(rest)": v for p, v in val.items()} if k == "groups" else val)
                    for k, val in info.items()}))
    else:
        print(f"{a.part}: {len(examples)} labelled examples, method {a.method}, {info.get('signal', 'shared')} signal")
        if a.method == "ltt":
            print(f"  answered alone      {_pct(info['coverage'])}")
            print(f"  error among them    {_pct(info['error'])}   (target ≤ {_pct(a.risk)})")
        else:
            print(f"  answered alone      {_pct(info['answered'])}")
            print(f"  error among them    {_pct(info['error'])}")
            print(f"  risk                {_pct(info['risk'])}   (answered alone and wrong, of all; target ≤ {_pct(a.risk)})")
            if "must_escalate_at_least" in info:
                print(f"  model wrong on      {_pct(info['base_error'])} of the examples → must escalate at least "
                      f"{_pct(info['must_escalate_at_least'])}")
        print(f"  threshold           {thr:.4g}" if ok else "  threshold           inf — everything escalates")
        print(f"  guarantee           {info['guarantee']}")
        for w in info.get("warnings", []):
            print(f"  warning             {w}")
        if info.get("groups"):
            print("  per group:")
            print(f"    {'group':28s} {'n':>5s} {'threshold':>10s} {'answered':>9s} {'error':>7s} {'risk':>7s}  pooled")
            for p, g in sorted(info["groups"].items(), key=lambda kv: (len(kv[0]), kv[0])):
                from .decide import group_name
                t = g["threshold"]
                pooled = ", ".join(group_name(x) for x in g["pooled"]) if g["pooled"] else ""
                print(f"    {group_name(p)[:28]:28s} {g['n']:5d} {t if math.isfinite(t) else float('inf'):10.4g} "
                      f"{_pct(g['answered']):>9s} {_pct(g['error']):>7s} {_pct(g['risk']):>7s}  {pooled}")
        if a.conformal:
            c = info["conformal"]
            print(f"  conformal sets      {_pct(c['coverage'])} coverage, {c['mean_size']:.2f} candidates on average")
        for f in skipped:
            print(f"(the catalog's calibration {f} was not applied while calibrating)")
        print(f"wrote {out} — load it in the catalog: part.load_calibration({out!r})")
    return 0 if ok else 1


def add_parser(sub):
    c = sub.add_parser("calibrate", help="calibrate a model decision's escalation on labelled examples (act_guard) and "
                                         "save the thresholds to a file the catalog loads")
    c.add_argument("system", help="module:attr or file.py:attr — a System (or a function returning one), or a Catalog")
    c.add_argument("part", help="the question answered by a model decision (or the decision part's name)")
    c.add_argument("labels", help="labelled examples: a .csv, or JSON lines (.jsonl; '-' reads stdin) with a 'label' "
                                  "column and the input (the part's facts, a 'text' column, or a state)")
    c.add_argument("--risk", type=float, default=0.10, help="crc: P(answered alone and wrong) ≤ risk; ltt: the error "
                                                             "among the answered ≤ risk (default 0.10)")
    c.add_argument("--method", default="crc", choices=["crc", "ltt"],
                   help="crc: act_guard, conformal risk control (default); ltt: calibrate_for(method='ltt')")
    c.add_argument("--groups", help="a threshold per group: fact / column names, top first (domain,task)")
    c.add_argument("--min-group", type=int, default=100, help="examples a group needs for its own threshold (default 100)")
    c.add_argument("--delta", type=float, default=0.10, help="the bound's failure probability (default 0.10; with groups, "
                                                              "negative: conformal risk control per group)")
    c.add_argument("--conformal", type=float, metavar="COVERAGE", help="also fit conformal answer sets (e.g. 0.9)")
    c.add_argument("--limit", type=int, help="use at most this many rows")
    c.add_argument("--out", help="the calibration file to write (default PART.calib.json)")
    c.add_argument("--json", action="store_true", help="print JSON")
    return c
