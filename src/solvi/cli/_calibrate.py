"""`solvi calibrate`: calibrate a model decision's escalation on labelled examples and write the calibration file the
catalog loads (the file format: solvi.core.calibfile). Moved out of solvi.core.calibfile in 1.0 (which still resolves these
names), so that the file format imports no decider.

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

from ..core import calibfile
from ..core.calibfile import _enc, _kind_of, save
from .._command import dump as _dump, fail as _fail, load_object


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
    from ..core.catalog import NOT_STATED_KEY, Unknown
    if y == NOT_STATED_KEY:
        return Unknown
    sp = part.spec
    if isinstance(y, str):
        s = y.strip()
        if sp.kind == "noul" and s.lower() in ("true", "false", "yes", "no", "1", "0"):
            return s.lower() in ("true", "yes", "1")
        if sp.multi:
            return [_option(sp, x.strip()) for x in s.split("|") if x.strip()]
        if sp.kind == "number":
            try:
                return float(s)
            except ValueError:
                return s
        return _option(sp, s)
    return y


def _option(sp, s):
    """A text cell → the option it names: itself, or the one option that is not text and reads the same (a CSV file has
    only text: "3" is the level 3 of Scale[1, 2, 3, 4, 5])."""
    if s in sp.options:
        return s
    same = [o for o in sp.options if not isinstance(o, str) and str(o) == s]
    return same[0] if len(same) == 1 else s


def examples_of(part, rows, group_cols=()):
    """Rows (dicts with "label") → [(Facts, label)] for act_guard / calibrate_for / conformal."""
    from ..core.deciders import Facts
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
    from ..core.deciders import decision_of
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
    calibfile._recalibrating = []
    try:                                           # calibration files the catalog loads are skipped: a file made for
        obj = load_object(a.system)                # another model must not stop it from being calibrated again
    finally:
        skipped, calibfile._recalibrating = calibfile._recalibrating, None
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
            info = part.calibrate_for(examples, max_error=a.risk, method="ltt", delta=a.delta)
        else:
            info = part.act_guard(examples, max_risk=a.risk, groups=groups, min_group=a.min_group,
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
    if info.get("groups"):                         # per group: "threshold" is the rest of the stream's (inf when every
        ok = any(math.isfinite(g["threshold"]) for g in info["groups"].values())   # group has its own), not the verdict
    else:
        ok = thr is not None and math.isfinite(thr)
    if a.json:
        _dump(_enc({k: ({"/".join(p) or "(rest)": v for p, v in val.items()} if k == "groups" else val)
                    for k, val in info.items()}))
    else:
        print(f"{a.part}: {len(examples)} labelled examples, method {a.method}, {info.get('signal', 'shared')} signal")
        if a.method == "ltt":
            print(f"  answered alone      {_pct(info['answered'])}")
            print(f"  error among them    {_pct(info['error'])}   (target ≤ {_pct(a.risk)})")
        else:
            print(f"  answered alone      {_pct(info['answered'])}")
            print(f"  error among them    {_pct(info['error'])}")
            print(f"  risk                {_pct(info['risk'])}   (answered alone and wrong, of all; target ≤ {_pct(a.risk)})")
            if "must_escalate_at_least" in info:
                print(f"  model wrong on      {_pct(info['base_error'])} of the examples → must escalate at least "
                      f"{_pct(info['must_escalate_at_least'])}")
        if not ok:
            print("  threshold           inf — everything escalates")
        elif info.get("groups"):
            print("  threshold           per group (below)")
        else:
            print(f"  threshold           {thr:.4g}")
        print(f"  guarantee           {info['guarantee']}")
        for w in info.get("warnings", []):
            print(f"  warning             {w}")
        if info.get("groups"):
            print("  per group:")
            print(f"    {'group':28s} {'n':>5s} {'threshold':>10s} {'answered':>9s} {'error':>7s} {'risk':>7s}  pooled")
            for p, g in sorted(info["groups"].items(), key=lambda kv: (len(kv[0]), kv[0])):
                from ..core.calibration import group_name
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


__all__ = ["add_parser", "cmd_calibrate", "examples_of", "find_part", "label_of", "read_rows"]
