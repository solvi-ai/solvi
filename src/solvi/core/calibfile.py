"""Calibrations as files (the `solvi calibrate` command: solvi.cli._calibrate).

`act_guard`, `calibrate_for` and `conformal` set a decision's thresholds in memory. A calibration file keeps them, so a
catalog calibrated once (on a few hundred labelled examples of your stream) loads the same thresholds every time it starts:

    info = part.act_guard(examples, max_risk=0.10)
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

`solvi calibrate` (solvi.cli._calibrate) calibrates a model decision on labelled examples and writes the same file."""
from __future__ import annotations

import json
import math

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
# The file's format is here; what a part puts in it and takes from it is the part's own (DecisionPart and Combination:
# _calibration_kind, _calibration_base, _calibration_models, _calibration_thresholds, _calibration_adapter,
# _apply_calibration), so this module imports no decider (1.0).
def _kind_of(part):
    kind = getattr(part, "_calibration_kind", None)
    if kind is None:
        raise TypeError(f"a calibration file is made from a DecisionPart or a Cascade / Vote / Route, not "
                        f"{type(part).__name__}")
    return kind


def base_fingerprint(part):
    """What a calibration is fitted to: the part's fingerprint without its thresholds (the checkpoint, the question, this
    question's adaptation, and how the part computes its signal — option_order="average" with its permutations, long=
    with top_k and rerank; for a combination every member)."""
    _kind_of(part)
    return part._calibration_base()


def _models(part):
    """{model id: weights fingerprint} of the models behind a part (for the message when they differ)."""
    return part._calibration_models()


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
    out.update(part._calibration_thresholds())
    out.update(guarantee=part.guarantee, conformal=part.conformal_set, groups=_groups_record(part.groups))
    ad = part._calibration_adapter()
    if ad is not None:                              # the adapter it was calibrated with, in a file beside it
        out[ad.kind] = {"hash": ad.fingerprint()}
    from .. import __version__
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
    if "lora" in rec:                               # the part's adapter writes itself (the Adapter protocol)
        part._calibration_adapter().save(lora_path(path))
        rec["lora"]["file"] = os.path.basename(lora_path(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_enc(rec), fh, ensure_ascii=False, indent=1, allow_nan=False)
        fh.write("\n")
    return path


def _group_by(rec, groups):
    from .calibration import GroupBy
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
    if lo and callable(getattr(part, "_load_adapter", None)):
        ad = part._calibration_adapter()
        if ad is None or ad.hash != lo.get("hash"):
            import os                               # calibrated with an adapter: load it first (from beside the file)
            f = os.path.join(os.path.dirname(os.path.abspath(str(path))), lo.get("file") or os.path.basename(lora_path(path)))
            part._load_adapter("lora", f, strict, expect=lo.get("hash"))
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
    part._apply_calibration(rec, grp, path)
    return part


# The command's helpers lived here until 1.0; they moved to solvi.cli._calibrate (a command imports the deciders, a file format
# does not). The old names still resolve here, looked up on first use.
_MOVED_TO_CALIBRATE = ("add_parser", "cmd_calibrate", "examples_of", "find_part", "label_of", "read_rows")


def __getattr__(name):
    if name in _MOVED_TO_CALIBRATE:                 # until 1.1, with a SolviDeprecationWarning
        import importlib

        from .._deprecate import _warn_from_caller, moved_message
        _warn_from_caller(moved_message(f"solvi.calibfile.{name}", f"solvi.cli._calibrate.{name}"), skip=1)
        return getattr(importlib.import_module("solvi.cli._calibrate"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["base_fingerprint", "FORMAT", "load", "lora_path", "record", "save"]
