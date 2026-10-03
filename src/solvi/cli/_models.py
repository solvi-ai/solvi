"""`solvi models` — the command (list, pull, check); the library part is solvi.models."""
from __future__ import annotations

import statistics
import time

from .._command import dump as _dump, fail as _fail
from ..models import PUBLISHED, PULL_PATTERNS, ModelError, cache_dir, cached, declaration, load, pull, resolve


# --------------------------------------------------------------------------------------------------- check
def _question(rows, task=None, options=None, kind=None):
    """The question of a labelled set: --task / --options / --kind, or the rows' own "task" / "options"; options default
    to the labels seen (sorted)."""
    task = task or next((r["task"] for r in rows if r.get("task")), None)
    if not task:
        raise ModelError("the examples need a question: --task \"Which team should handle this?\" (or a 'task' field)")
    if options is None:
        options = next((r["options"] for r in rows if r.get("options")), None)
    if options is None:
        labels = set()
        for r in rows:
            y = r["label"]
            labels.update(y if isinstance(y, list) else [y])
        options = sorted(map(str, labels))
    return task, options, kind


def _input(row):
    if "text" in row:
        return row["text"]
    if "input" in row:
        return row["input"]
    return {k: v for k, v in row.items() if k not in ("label", "task", "options")}


def measure(model, rows, task, options, kind=None):
    """Decide every labelled row one at a time → {"n", "accuracy", "escalated", "answered_alone", "accuracy_alone",
    "first_ms", "ms_p50", "ms_p95", "ms_mean"}."""
    part = model.decision("check", task, "input", options, kind=kind)
    ms, right, alone, alone_right = [], 0, 0, 0
    first = None
    for i, r in enumerate(rows):
        x = _input(r)
        t0 = time.perf_counter()
        d = part.decide(x)
        dt = (time.perf_counter() - t0) * 1000
        if i == 0:
            first = dt
        else:
            ms.append(dt)
        from ._calibrate import label_of
        y = label_of(part, r["label"])
        try:
            ok = part.spec.label(d.value) == part.spec.label(y)
        except ValueError:
            ok = False
        right += ok
        if d.escalate is None:
            alone += 1
            alone_right += ok
    n = len(rows)
    q = sorted(ms or [first])
    return {"n": n, "accuracy": right / n, "escalated": 1 - alone / n, "answered_alone": alone / n,
            "accuracy_alone": alone_right / alone if alone else None, "first_ms": first,
            "ms_p50": statistics.median(q), "ms_p95": q[min(len(q) - 1, int(0.95 * len(q)))], "ms_mean": statistics.fmean(q)}


def _read_examples(path, limit):
    from ._calibrate import read_rows
    try:
        rows = read_rows(path)
    except FileNotFoundError:
        raise ModelError(f"no such file: {path}") from None
    except ValueError as e:
        raise ModelError(str(e)) from None
    rows = [r for r in rows if "label" in r]
    if not rows:
        raise ModelError(f"{path}: no labelled rows (a 'label' column)")
    return rows[:limit] if limit else rows


# --------------------------------------------------------------------------------------------------- the command
def _mb(n):
    return f"{n / 1e6:.0f} MB"


def cmd_list(a):
    got = {c["id"]: c for c in cached()}
    rows = [{"id": i, "published": True, "about": about, "cached": i in got, **({"path": got[i]["path"],
             "bytes": got[i]["bytes"]} if i in got else {})} for i, about in PUBLISHED.items()]
    rows += [{"id": c["id"], "published": False, "cached": True, "path": c["path"], "bytes": c["bytes"]}
             for c in got.values() if c["id"] not in PUBLISHED]
    if a.json:
        _dump({"cache": str(cache_dir()), "models": rows})
        return 0
    for r in rows:
        where = f"cached, {_mb(r['bytes'])}  {r['path']}" if r["cached"] else f"not downloaded (solvi models pull {r['id']})"
        print(f"{r['id']:28s} {r.get('about', 'a decider in the local cache'):52s} {where}")
    print(f"cache: {cache_dir()}")
    return 0


def cmd_pull(a):
    try:
        path = pull(a.id, a.backend, a.revision)
    except ModelError as e:
        _fail(str(e))
    print(path)
    return 0


def cmd_check(a):
    out = {"model": a.model}
    try:
        k, where = resolve(a.model)
        rows = _read_examples(a.examples, a.limit) if a.examples else None
        opts = [o.strip() for o in a.options.split(",")] if a.options else None
        q = _question(rows, a.task, opts, a.kind) if rows else None
    except ModelError as e:
        _fail(str(e))
    out["kind"] = k
    if k in ("folder", "hub"):
        try:
            meta, caps = declaration(where)
        except ModelError as e:
            _fail(str(e))
        out.update(path=where, format=meta.get("format"), declared=caps)
    try:
        t0 = time.perf_counter()
        m = load(a.model, a.backend, a.api_key)
        out["load_ms"] = (time.perf_counter() - t0) * 1000
    except ModelError as e:
        _fail(str(e))
    except Exception as e:  # noqa: BLE001  — a missing runtime, a broken checkpoint: say so, keep what we know
        out["error"] = f"cannot load: {type(e).__name__}: {e}"
        _report(out, a.json)
        return 1
    fp = getattr(m, "weights_fingerprint", None)      # a module:attr decider may be any object with decision(...)
    out.update(id=getattr(m, "model_id", type(m).__name__), backend=getattr(m, "backend", None),
               fingerprint=fp() if callable(fp) else None, capabilities=getattr(m, "caps", None))
    status = 0
    if rows:
        try:
            out["examples"] = {"file": a.examples, "task": q[0], "options": q[1], **measure(m, rows, *q)}
        except Exception as e:  # noqa: BLE001
            out["error"] = f"cannot run: {type(e).__name__}: {e}"
            _report(out, a.json)
            return 1
        if a.min_accuracy is not None and out["examples"]["accuracy"] < a.min_accuracy:
            out["below_min_accuracy"] = a.min_accuracy
            status = 1
    if a.json:
        _dump(out)
    else:
        _report(out, False)
    return status


def _report(out, as_json):
    if as_json:
        _dump(out)
        return
    print(f"model        {out.get('id', out['model'])}  ({out['kind']})")
    if out.get("path"):
        print(f"path         {out['path']}")
    caps = out.get("capabilities") or out.get("declared")
    if out.get("format") is not None or caps:
        print(f"format       {out.get('format') or (caps or {}).get('format', '?')}")
    if caps:
        act = caps.get("act")
        print(f"kinds        {', '.join(caps.get('modes', []))}")
        print(f"act head     {'yes' if act else 'no'}; questions per pass: {caps.get('max_questions') or 1}; "
              f"state: {caps.get('state_format', '?')}")
        extra = [n for n in ("unknown", "pointer") if caps.get(n)]
        if extra:
            print(f"also         {', '.join(extra)}")
    if out.get("fingerprint"):
        print(f"fingerprint  #{out['fingerprint']}  (backend {out.get('backend')}, loaded in {out['load_ms']:.0f} ms)")
    if out.get("error"):
        print(out["error"])
    ex = out.get("examples")
    if ex:
        alone = "—" if ex["accuracy_alone"] is None else f"{100 * ex['accuracy_alone']:.1f}%"
        print(f"examples     {ex['n']} from {ex['file']}  ({ex['task']!r}: {', '.join(map(str, ex['options']))})")
        print(f"accuracy     {100 * ex['accuracy']:.1f}%; answered alone {100 * ex['answered_alone']:.1f}% "
              f"(accuracy {alone}), escalated {100 * ex['escalated']:.1f}%")
        print(f"latency      first {ex['first_ms']:.1f} ms; then p50 {ex['ms_p50']:.1f} ms, p95 {ex['ms_p95']:.1f} ms, "
              f"mean {ex['ms_mean']:.1f} ms per decision")
        if out.get("below_min_accuracy") is not None:
            print(f"below --min-accuracy {out['below_min_accuracy']:g}")


def add_parser(sub):
    m = sub.add_parser("models", help="the published deciders and the cached ones; pull one; check any decider")
    ms = m.add_subparsers(dest="action")
    ls = ms.add_parser("list", help="solvi-ai deciders and every decider in the local Hugging Face cache (default)")
    ls.add_argument("--json", action="store_true", help="print JSON")
    p = ms.add_parser("pull", help="download a decider into the local cache (huggingface_hub)")
    p.add_argument("id", help="a Hugging Face id, e.g. solvi-ai/solvi-base")
    p.add_argument("--backend", default="onnx", choices=list(PULL_PATTERNS),
                   help="which weights: onnx (default; for solvi[onnx]), torch (safetensors) or all")
    p.add_argument("--revision", help="a branch, tag or commit (default main)")
    c = ms.add_parser("check", help="load a decider and print its capabilities, fingerprint, latency and accuracy")
    c.add_argument("model", help="a checkpoint folder, a cached Hugging Face id, systemone:URL#model, llm:URL#model or "
                   "module:attr")
    c.add_argument("--examples", help="labelled examples (.jsonl / .csv: 'label' plus 'text' / 'input' or a state)")
    c.add_argument("--task", help="the question the examples answer (else each row's 'task')")
    c.add_argument("--options", help="its options, comma-separated (default: the labels seen)")
    c.add_argument("--kind", choices=["choice", "multi", "score", "noul"], help="the question kind (default choice)")
    c.add_argument("--limit", type=int, default=200, help="at most this many examples (default 200; 0: all)")
    c.add_argument("--min-accuracy", type=float, help="exit 1 below this accuracy (e.g. 0.8)")
    c.add_argument("--backend", default="auto", choices=["auto", "onnx", "torch"], help="for a checkpoint folder / id")
    c.add_argument("--api-key", help="for systemone: / llm: (default $SOLVI_SYSTEMONE_API_KEY / $SOLVI_LLM_API_KEY)")
    c.add_argument("--json", action="store_true", help="print JSON")
    return m


def cmd_models(a):
    """`solvi models` (see solvi.cli) → exit status."""
    action = a.action or "list"
    if action == "list":
        if not hasattr(a, "json"):
            a.json = False
        return cmd_list(a)
    return {"pull": cmd_pull, "check": cmd_check}[action](a)


__all__ = ["add_parser", "cmd_models", "measure"]
