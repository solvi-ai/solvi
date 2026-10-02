"""A checkpoint's declared capabilities (solvi_decide.json) with their defaults per format. (Part of solvi.decide, which
re-exports every name.)"""
from __future__ import annotations



from .kinds import ACT_FEATURES, ACT_FEATURES_V3, LEGACY_FORMAT, MARKERS, TYPED2_FORMAT, TYPED_FORMAT, V3_MARKERS
from .state import SERIALIZATIONS


def _multi_question(v):
    """→ {"layout", "max_questions", "max_len", "window"} or None (one question per pass)."""
    if v is None or v is False or v == 0:
        return None
    if v is True:
        v = {}
    elif isinstance(v, int) and not isinstance(v, bool):
        v = {"max_questions": v}
    if not isinstance(v, dict) or not v.get("enabled", True):
        return None
    mq = {"layout": str(v.get("layout", "block")), "max_questions": int(v.get("max_questions", 6)),
          "max_len": int(v.get("max_len", 1024)), "window": int(v.get("window", 64))}
    if mq["layout"] not in ("block", "concat"):
        raise ValueError(f"unknown multi-question layout {mq['layout']!r} (block, concat)")
    return mq if mq["max_questions"] > 1 else None


_DEFAULTS = {
    # the first, text-only deciders: choose-one / multi-label, text input, two head columns, one question per pass, no act head
    LEGACY_FORMAT: {"modes": ["single", "multi"], "columns": {"single": 0, "multi": 1}, "noul_labels": ["yes", "no"],
                    "state_serialization": ["text"], "act": None},
    # the first typed checkpoints: every kind natively, "paths" / "tree" / "json" states, three head columns (the third: act, at the mode token)
    TYPED_FORMAT: {"modes": ["single", "multi", "score", "noul"],
                   "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "act": 2}, "noul_labels": ["true", "false"],
                   "state_serialization": ["paths", "tree", "json"], "act": {}},
    # answer primitives: + rank, number (bins as ordered options), span; "not stated" (column 3 at the mode marker); a pointer
    # (columns 4 / 5 over the input's tokens, full layout only) for span answers and evidence quotes
    TYPED2_FORMAT: {"modes": ["single", "multi", "score", "noul", "rank", "number", "span"],
                    "columns": {"single": 0, "multi": 1, "score": 0, "noul": 0, "rank": 0, "number": 0, "act": 2,
                                "unknown": 3, "span_start": 4, "span_end": 5},
                    "noul_labels": ["true", "false"], "state_serialization": ["paths", "tree", "json"], "act": {},
                    "unknown": {"column": 3}, "pointer": {"start": 4, "end": 5}},
}


def _v3(meta):
    """Is this a checkpoint of the answer-primitives contract (`subformat` 'l14g typed v2', or format 'l14g typed v2'
    / 'solvi_decide v3')? Only such checkpoints get the new capability fields (older ones hash as before)."""
    fmt, sub = str(meta.get("format", "")), str(meta.get("subformat", ""))
    return fmt.startswith(("l14g", "solvi_decide v3")) or sub.startswith("l14g")


def _unknown_caps(v, columns):
    """"not stated": {"column", "label", "joint" (kinds whose options compete with it in one softmax), "multi" ("sigmoid"),
    "span" ("null_span"), "threshold"} or None."""
    if v is None or v is False:
        return None
    v = {} if v is True else dict(v)
    return {"column": int(v.get("column", columns.get("unknown", 3))), "label": str(v.get("label", "not stated")),
            "joint": [str(k) for k in v.get("joint", ["single", "score", "noul", "rank", "number"])],
            "multi": str(v.get("multi", "sigmoid")), "span": str(v.get("span", "null_span")),
            "threshold": float(v.get("threshold", 0.5))}


def _pointer_caps(v, columns):
    """The pointer: {"start", "end" (columns), "layouts", "max_span_tokens", "null", "evidence": {"threshold",
    "max_spans"}} or None."""
    if v is None or v is False:
        return None
    v = {} if v is True else dict(v)
    ev = dict(v.get("evidence") or {})
    return {"start": int(v.get("start", columns.get("span_start", 4))), "end": int(v.get("end", columns.get("span_end", 5))),
            "layouts": [str(x) for x in v.get("layouts", ["full"])], "max_span_tokens": int(v.get("max_span_tokens", 40)),
            "null": str(v.get("null", "mode")),
            "evidence": {"threshold": float(ev.get("threshold", 0.15)), "max_spans": int(ev.get("max_spans", 3))}}


def _max_len_long(v, max_len=512):
    """The long-input length a checkpoint declares (or a caller forces): an int above its max_len."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ValueError(f"max_len_long must be a number of tokens, not {v!r}") from None
    if isinstance(v, bool) or n <= int(max_len or 512):
        raise ValueError(f"max_len_long ({v!r}) must be larger than max_len ({max_len}): it is the length the checkpoint "
                         "reads whole with long=\"full\"")
    return n


def capabilities(meta, multi_question=None, act=None):
    """What a checkpoint can do, from its solvi_decide.json (see docs/decide_format.md): the fields it declares over the
    defaults of its format ('l14b_decider v1': the text-only deciders; 'l14f typed v1': the first typed ones; 'solvi_decide v2': the legacy defaults,
    everything else declared; a bare 'solvi_decide v3' too — the answer-primitives defaults come with the l14g format
    or subformat only). multi_question / act: overrides (experiments), part of the fingerprint."""
    meta = meta or {}
    fmt = str(meta.get("format", ""))
    v3 = _v3(meta)
    base = (_DEFAULTS[TYPED2_FORMAT] if v3 and "l14g" in fmt + str(meta.get("subformat", "")) else
            _DEFAULTS[TYPED_FORMAT] if fmt.startswith("l14f") else _DEFAULTS[LEGACY_FORMAT])
    version = 3 if v3 else 2 if fmt.startswith(("l14f", "solvi_decide")) else 1     # other formats (stand-ins) hash as before
    modes = [str(m) for m in (meta.get("modes") or base["modes"])]
    markers = {**MARKERS, **(V3_MARKERS if v3 else {}), **(meta.get("markers") or {})}
    columns = {**base["columns"], **(meta.get("columns") or {})}
    ser = [str(x) for x in (meta.get("state_serialization") or base["state_serialization"])]
    known = [x for x in ser if x in SERIALIZATIONS]
    if version >= 2 and not known and ser != ["text"]:
        raise ValueError(f"the checkpoint reads states as {ser}; this solvi writes {SERIALIZATIONS}")
    a = meta.get("act", base["act"] if "act_head" not in meta else ({} if meta["act_head"] else None))
    if a is False:
        a = None
    if act is not None:
        a = (a if a is not None else {}) if act else None
    if a is not None:
        a = dict(a)
        a.setdefault("column", columns.get("act", 2))
        a.setdefault("temperature", float((meta.get("temperature") or {}).get("act", 1.0))
                     if isinstance(meta.get("temperature"), dict) else 1.0)
        cal = a.get("calibrator")
        if cal is not None:
            bad = [f for f in cal.get("features", []) if f not in (ACT_FEATURES_V3 if v3 else ACT_FEATURES)]
            if bad or len(cal.get("features", [])) != len(cal.get("weights", [])):
                raise ValueError(f"act calibrator: unknown features {bad} or features / weights of different lengths "
                                 f"(known: {ACT_FEATURES})")
    mq = _multi_question(meta.get("multi_question") if multi_question is None else multi_question)
    caps = {"format": fmt, "version": version, "modes": modes, "markers": markers, "columns": columns,
            "noul_labels": [str(x) for x in (meta.get("noul_labels") or base["noul_labels"])],
            "serialization": ser, "state_format": known[0] if known else "paths", "act": a, "multi_question": mq,
            "max_questions": mq["max_questions"] if mq else 0}
    if meta.get("max_len_long") is not None:      # trained to read long inputs whole (long="full"); absent: hashes as before
        caps["max_len_long"] = _max_len_long(meta["max_len_long"], meta.get("max_len", 512))
    if v3:                                        # the answer-primitives contract (older formats: the dict above, as before)
        raw_mq = meta.get("multi_question") if multi_question is None else multi_question
        if mq is not None:
            mq["pointer"] = bool(raw_mq.get("pointer", False)) if isinstance(raw_mq, dict) else False
        caps["subformat"] = str(meta.get("subformat", ""))
        caps["unknown"] = _unknown_caps(meta.get("unknown", base.get("unknown")), columns)
        caps["pointer"] = _pointer_caps(meta.get("pointer", base.get("pointer")), columns)
        num = meta.get("number") if isinstance(meta.get("number"), dict) else {}
        caps["number"] = {"interval": float(num.get("interval", 0.8))}
    return caps


__all__ = ["capabilities"]
