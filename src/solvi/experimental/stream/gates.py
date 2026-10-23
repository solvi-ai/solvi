"""The arithmetic of the adapter loop's gates (experimental; the LoRA learning loop, stage 1): the label split, the
pre-freeze gates on the held-out labels, the memorisation probe and the improvement signal. Plain functions over
counts, no model, no torch; `stream.policy.DefaultPolicy` calls them.

    from solvi.experimental.stream.gates import split, heldout_gate, probe

    split("intent|3f2a…")                        # "train" | "calib" | "holdout", the same in every run, forever
    heldout_gate(41, 3, 30, 4, 100)               # {"ok", "n", "gain", "why"}: right − wrong alone, candidate vs served
    probe(alone_seen=80, wrong_seen=1, alone_novel=60, wrong_novel=30)   # {"ok": False, ...}: a memoriser

The split is copied from the change pipeline's update gates (`changes._updates.split_of`), not imported (that module
is private), with its own salt and the loop's shares (70 / 15 / 15 train / calib / holdout): a content key's split is a
hash of the key alone, so an input seen labelled twice is in the same split both times and a key never crosses. The
held-out gate is the same score as the change pipeline's ((answered alone and right − answered alone and wrong) / n);
the honesty gate compares `solvi.testing.honesty` numbers with its `compare`. Unmeasured: no stand has run on them."""
from __future__ import annotations

import hashlib

SPLITS = ("train", "calib", "holdout")
SALT = "solvi-stream-loop:"


def _share(name, v, *, closed=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not (0 <= v <= 1 if closed else 0 <= v < 1):
        raise ValueError(f"{name} must be a share in [0, 1{']' if closed else ')'}, not {v!r}")
    return float(v)


def split(key, *, calib: float = 0.15, holdout: float = 0.15) -> str:
    """A label's split by a hash of its content key → "holdout", "calib" or "train". A pure function of the key and the
    shares: the same key always lands in the same split. Shares outside [0, 1), or summing to 1 or more (no label
    left to train on), are a ValueError."""
    calib, holdout = _share("calib", calib), _share("holdout", holdout)
    if calib + holdout >= 1:
        raise ValueError(f"calib ({calib}) + holdout ({holdout}) leave no label to train on: their sum must be below 1")
    h = int(hashlib.sha256((SALT + str(key)).encode()).hexdigest()[:8], 16) / 2 ** 32
    return "holdout" if h < holdout else "calib" if h < holdout + calib else "train"


def heldout_gate(right, wrong, served_right, served_wrong, n, *, min_gain: float = 0.01,
                 min_holdout: int = 100) -> dict:
    """The candidate against the served version on the same n held-out labels: score = (answered alone and right −
    answered alone and wrong) / n; ok when n ≥ min_holdout and the candidate's score ≥ the served one's + min_gain.
    → {"ok", "n", "gain", "min_gain", "candidate", "served", "why"}."""
    if n < min_holdout:
        return {"ok": False, "n": n, "gain": None, "min_gain": min_gain, "candidate": None, "served": None,
                "why": f"{n} held-out label(s) < {min_holdout}: an improvement cannot be shown yet"}
    k, s = (right - wrong) / n, (served_right - served_wrong) / n
    gain = k - s
    return {"ok": gain >= min_gain, "n": n, "gain": gain, "min_gain": min_gain, "candidate": k, "served": s,
            "why": f"right − wrong answered alone on {n} held-out label(s): {s:+.3f} → {k:+.3f} (gain {gain:+.3f}, "
                   f"needed {min_gain:+.3f})"}


def honesty_gate(before: dict, after: dict, tolerance: float = 0.02) -> dict:
    """The honesty numbers (`solvi.testing.honesty.metrics`' gated keys) of the served version (before) and the
    candidate (after) on the held-out labels: ok when none got worse by more than `tolerance`. Empty `before`: nothing
    to compare, ok. → {"ok", "tolerance", "regressions", "why"}."""
    from ...testing.honesty import compare
    regs = compare(dict(after or {}), dict(before or {}), tolerance) if before else []
    return {"ok": not regs, "tolerance": tolerance, "regressions": regs,
            "why": "no honesty number got worse" if not regs else "; ".join(regs)}


def paired_gain_low(better: int, worse: int, confidence: float = 0.95):
    """The paired held-out comparison of label mode: of the held-out labels where the candidate and the served
    version score differently (an item scores +1 answered alone and right, −1 alone and wrong, 0 escalated), `better`
    favour the candidate and `worse` the served one. → the Wilson lower bound of 2p − 1, p = better / (better +
    worse) (> 0: the candidate is better on more of them than chance allows); None with no such label."""
    from ...core.store.sysreport import wilson
    m = int(better) + int(worse)
    if m == 0:
        return None
    lo, _ = wilson(int(better), m, confidence)
    return 2 * lo - 1


def probe(alone_seen: int, wrong_seen: int, alone_novel: int, wrong_novel: int, *, gap: float = 0.10,
          min_n: int = 30) -> dict:
    """The memorisation probe on post-freeze samples: the candidate's error among its alone answers on inputs whose
    key it trained on (seen) and on inputs it never saw labelled (novel). Fails when err_novel − err_seen > gap.
    Measured only when both sides have ≥ min_n alone answers; unmeasured it does not fail (the comparison on novel
    samples alone still decides promotion). → {"ok", "measured", "err_seen", "err_novel", "gap", "why"}."""
    es = wrong_seen / alone_seen if alone_seen else None
    en = wrong_novel / alone_novel if alone_novel else None
    if alone_seen < min_n or alone_novel < min_n:
        return {"ok": True, "measured": False, "err_seen": es, "err_novel": en, "gap": gap,
                "why": f"not measured yet ({alone_seen} seen / {alone_novel} novel alone answer(s), {min_n} each "
                       "needed)"}
    d = en - es
    return {"ok": d <= gap, "measured": True, "err_seen": es, "err_novel": en, "gap": gap,
            "why": f"error alone on novel − on seen: {en:.3f} − {es:.3f} = {d:+.3f} (allowed {gap:+.3f})"}


def incumbent_short(outcomes, alpha: float, target_share=None) -> dict:
    """The improvement signal: is the served version measurably short on its latest held-out labels? outcomes: [(alone,
    wrong)] of the served version, one per held-out label. Short when it answers nothing alone on them, or its error
    among the alone answers is above `alpha`, or (target_share given) its share alone is below target_share. With no
    held-out label there is no evidence that it is strong: short. → {"short", "n", "alone", "error", "share", "why"}."""
    rows = list(outcomes)
    n = len(rows)
    alone = sum(1 for a, _ in rows if a)
    wrong = sum(1 for a, w in rows if a and w)
    err = wrong / alone if alone else None
    share = alone / n if n else None
    if n == 0:
        short, why = True, "no held-out label of the served version yet"
    elif alone == 0:
        short, why = True, f"answers none of {n} held-out label(s) alone"
    elif err > alpha:
        short, why = True, f"error alone {err:.3f} > {alpha:g} on {n} held-out label(s)"
    elif target_share is not None and share < target_share:
        short, why = True, f"share alone {share:.3f} < target {target_share:g} on {n} held-out label(s)"
    else:
        short, why = False, (f"error alone {err:.3f} ≤ {alpha:g}, share alone {share:.3f} on {n} held-out label(s)"
                             + ("" if target_share is None else f" (target {target_share:g})"))
    return {"short": short, "n": n, "alone": alone, "error": err, "share": share, "why": why}


__all__ = ["SALT", "SPLITS", "heldout_gate", "honesty_gate", "incumbent_short", "paired_gain_low", "probe", "split"]
