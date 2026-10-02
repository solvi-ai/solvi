"""NATURAL PLAN: a plan under constraints, scored by the benchmark's own evaluators.

    python naturalplan/score.py runs/<arm>.jsonl [--split eval]     (gemini: the answers stored in the dataset)

A prediction: {"id", "text": the plan in the benchmark's format, "escalate": bool (optional: no plan given)}."""
import contextlib
import importlib.util
import io
import json
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.llm import DATA, read_jsonl  # noqa: E402

D = DATA / "naturalplan"
TASKS = ("calendar_scheduling", "meeting_planning", "trip_planning")


def _evaluator(task):
    for name in ("absl", "absl.app", "absl.flags"):                     # the scripts only need absl to parse flags
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["absl.flags"].DEFINE_string = lambda *a, **k: None
    sys.modules["absl.app"].run = lambda *a, **k: None
    sys.modules["absl.app"].UsageError = Exception
    sys.modules["absl"].app, sys.modules["absl"].flags = sys.modules["absl.app"], sys.modules["absl.flags"]
    spec = importlib.util.spec_from_file_location(f"np_{task}", D / f"repo/evaluate_{task}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def clean(text):
    """Typography a model adds must not cost a point: narrow spaces, long dashes."""
    for a, b in ((" ", " "), (" ", " "), (" ", " "), ("–", "-"), ("—", "-"), ("‑", "-"),
                 ("’", "'")):
        text = text.replace(a, b)
    return text


def right(task, ev, row, text):
    try:
        with contextlib.redirect_stdout(io.StringIO()):                 # the evaluators print every parse error
            return _right(task, ev, row, clean(text))
    except Exception:                                                   # noqa: BLE001 - an unparsable plan is a wrong plan
        return False


def _right(task, ev, row, text):
    if task == "calendar_scheduling":
        return ev._parse_response(text) == ev._parse_response(row["golden_plan"]) and ev._parse_response(text)[0] != ""
    if task == "meeting_planning":
        start, t0 = row["constraints"][0]
        cons = ev.process_constraints(row["constraints"][1:])
        got = ev.validator_from_text(ev.parse_text_plan(text), cons, start, t0, row["dist_matrix"])
        return got == ev.validator_from_text(row["golden_plan"], cons, start, t0, row["dist_matrix"])
    return ev.compute_example_score(row["cities"], row["durations"], ev.parse_response(text)) == 1.0


def score(pred_file=None, split="eval", field=None):
    """field="pred_5shot_pro" scores the answers stored in the dataset (Gemini 1.5 Pro) instead of a prediction file."""
    pred = {r["id"]: r for r in read_jsonl(pred_file)} if pred_file else {}
    out = {}
    for task in TASKS:
        ev, rows = _evaluator(task), read_jsonl(D / f"prepared/{task}_{split}.jsonl")
        ok = answered = 0
        for r in rows:
            p = {"text": r[field]} if field else pred.get(r["id"]) or {}
            if p.get("text") and not p.get("escalate"):
                answered += 1
                ok += right(task, ev, r, p["text"])
        out[task] = {"n": len(rows), "right": ok, "accuracy": round(ok / len(rows), 3), "answered": answered,
                     "wrong_among_answered": round((answered - ok) / max(answered, 1), 3)}
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    split = a[a.index("--split") + 1] if "--split" in a else "eval"
    print(json.dumps(score(field="pred_5shot_pro", split=split) if a and a[0] == "gemini" else score(a[0], split), indent=1))
