"""Question kinds, wire markers and the options each kind takes. (Part of solvi.decide, which re-exports every name.)"""
from __future__ import annotations

import functools


from .. import _deprecate



OPT, ONE, MANY = "[unused0]", "[unused1]", "[unused2]"


MARKERS = {"option": OPT, "single": ONE, "multi": MANY, "score": "[unused3]", "noul": "[unused4]"}


OTHER_NAMES = ("other", "none", "none of the above", "none of these", "other / none", "nothing")


DEFAULT_T = {"l14b_decider v1": 1.45}          # temperature fitted on the training pool's validation split


LEGACY_FORMAT, TYPED_FORMAT, FORMAT = "l14b_decider v1", "l14f typed v1", "solvi_decide v2"


TYPED2_FORMAT = "l14g typed v2"                 # answer primitives: rank, number, span, "not stated", evidence (subformat of solvi_decide v2)


ACT_FEATURES = ("confidence", "margin", "entropy", "act_logit", "n_options", "kind=choice", "kind=multi", "kind=score",
                "kind=noul")


ACT_FEATURES_V3 = ACT_FEATURES + ("p_unknown", "kind=rank", "kind=number", "kind=span")


KINDS = ("choice", "multi", "score", "noul")


KINDS_V3 = KINDS + ("rank", "number", "span")


V3_MARKERS = {"rank": "[unused5]", "number": "[unused6]", "span": "[unused7]"}


WIRE = {"choice": "single", "multi": "multi", "score": "score", "noul": "noul", "rank": "rank", "number": "number",
        "span": "span"}                           # question kind → mode on the wire


_KIND = {"choice": "choice", "single": "choice", "one": "choice", "multi": "multi", "many": "multi", "score": "score",
         "ordinal": "score", "scale": "score", "noul": "noul", "yes_no": "noul", "yesno": "noul", "bool": "noul",
         "rank": "rank", "ranking": "rank", "number": "number", "estimate": "number", "span": "span"}


NULL_SOURCE = "text"                            # the source of a pointer quote before it is bound to its fact


def _spec_names(fn):
    """`not_stated=` is the name; `unknown=` (0.7) still works with a SolviDeprecationWarning. The internal spec keeps
    `unknown` (stored adaptations and calibration files carry that key)."""
    where = fn.__qualname__

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if "not_stated" in kw:
            if "unknown" in kw:
                raise TypeError(f"{where}() got both unknown= and not_stated=: unknown= is the old name of not_stated=")
            kw["unknown"] = kw.pop("not_stated")
        elif "unknown" in kw:
            _deprecate.renamed(f"{where}(unknown=)", "not_stated=", stacklevel=3)
        return fn(*args, **kw)
    return wrapper


_OPTION_KINDS = {"score_value": ("score",), "k": ("rank",), "bins": ("number",), "unit": ("number",),
                 "coverage": ("number",), "other": ("choice", "multi"), "min_margin": ("choice", "score", "noul", "rank")}


def _unused_options(name, k_, kind, multi, long, has_act, given):
    """An option that does nothing for this question raises (it used to be accepted, and some still changed the part's
    fingerprint, so a calibration file stopped loading)."""
    for opt, kinds in _OPTION_KINDS.items():
        if given.get(opt) is not None and k_ not in kinds:
            raise ValueError(f"{name}: {opt}= is for {' / '.join(kinds)} questions; this one is {k_} — drop it")
    if kind is not None and multi and k_ != "multi":
        raise ValueError(f"{name}: kind={kind!r} and multi=True contradict each other — drop one")
    for opt in ("top_k", "rerank"):
        if given.get(opt) is not None and long is None:
            raise ValueError(f'{name}: {opt}= selects sections of a long text: it needs long="retrieve" or "full"')
    for opt in ("min_act", "max_error"):
        if given.get(opt) is not None and not has_act:
            raise ValueError(f"{name}: {opt}= sets the act threshold, and this checkpoint has no act head — use "
                             "min_confidence= (a calibrated confidence) instead")


def _kind(kind, multi=False):
    if kind is None:
        return "multi" if multi else "choice"
    k = _KIND.get(str(kind).lower())
    if k is None:
        raise ValueError(f"kind must be one of {KINDS_V3}, not {kind!r}")
    return k


__all__ = ["KINDS"]
