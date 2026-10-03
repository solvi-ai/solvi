"""What decides alone: the act features, thresholds, promises (act_guard) and groups; inputs as Facts. (Part of
solvi.decide, which re-exports every name.)"""
from __future__ import annotations

import math

import numpy as np

from ..core.catalog import Quote, Unknown
from .kinds import KINDS, KINDS_V3
from .state import jsonable
from ..core.calibration import GroupBy, group_name   # noqa: F401 — re-exported (defined there since 1.0)


class Facts(dict):
    """An example input given as facts by name (`Facts(email=..., tier=...)`): each part reads its own facts, a route's
    predicates and a grouping (act_guard(groups=...)) read theirs. Any other input is the one input every part reads (a
    text or a state)."""


def group_record(g, path, node, info):
    """A decision's guarantee under thresholds per group: the part's promise, the input's group, the group whose
    threshold applied (its own, or a parent's when the group had too few examples), that threshold and its examples."""
    pooled = tuple(node) != tuple(path)
    out = dict(g, group=list(path), applied=list(node), threshold=info["threshold"], n=info["n"])
    out["promise"] = (f"{g['promise']}; here: group {group_name(node)} (threshold {info['threshold']:.4g}, n = "
                      f"{info['n']})" + (f", pooled: {group_name(path)} had fewer than {g['min_group']} examples"
                                         if pooled else ""))
    return out


def confidence_source(d):
    """Where a decision's probabilities came from when its model says so (an LLM decider: `extra["llm"]["probabilities"]`
    — "logprobs" from the answer's tokens, "stated" / "confidence" from the numbers the model wrote), else None."""
    info = d.extra.get("llm") if isinstance(getattr(d, "extra", None), dict) else None
    return info.get("probabilities") if isinstance(info, dict) else None


def one_source(ds, who="the calibration examples"):
    """The one source of the decisions' probabilities (see confidence_source; None when no decision names one). Raises
    ValueError when log-probabilities and written numbers are mixed: they are two scales (a token probability near 1
    against a stated 0.85-0.95), and one threshold over both answers alone by which server happened to reply."""
    from collections import Counter
    n = Counter(x for x in map(confidence_source, ds) if x is not None)
    if "logprobs" in n and len(n) > 1:
        raise ValueError(f"the probabilities of {who} come from two sources: {n['logprobs']} from log-probabilities, "
                         f"{sum(n.values()) - n['logprobs']} from the numbers the model wrote (extra['llm']"
                         "['probabilities']) — one threshold over both is not one signal. Make the model with "
                         "logprobs=False (or True, for a server that always returns them), or pin the provider "
                         "(extra_body={'provider': {...}}), and calibrate again")
    return "logprobs" if "logprobs" in n else ("stated" if n else None)


SEPARATION_MIN = 10        # right and wrong calibration examples each, before act_guard judges whether its signal separates


def guard_promise(risk, error, answered=True):
    """act_guard's promise in words, with the error among the answers given alone on the calibration examples."""
    among = (f"; among the answers given alone the error was {error:.1%} on the calibration examples, and it is not "
             "bounded (calibrate_for(max_error=..., method='ltt') bounds it)") if answered else "; nothing is answered alone"
    return (f"of all inputs like the calibration examples, answered or escalated, at most {risk:g} are answered alone "
            f"and wrong" + among)


def no_separation(sig, ok, name, error, base, who=""):
    """A warning when the signal does not tell right answers from wrong ones on the calibration examples (one-sided
    Mann-Whitney test of its AUROC against chance at the 5% level: solvi.core.calibration.separation), else None — tested
    only with at least SEPARATION_MIN right and as many wrong examples (fewer cannot tell). The promise still holds —
    by escalating, not by choosing: what is answered alone is wrong about as often as everything."""
    from ..core.calibration import separation
    right = int(sum(1 for o in ok if o))
    if min(right, len(ok) - right) < SEPARATION_MIN:
        return None
    auc, z = separation(sig, ok)
    if z is None or z >= 1.645:
        return None
    head = (f"{who}the {name} signal does not separate right from wrong answers on the calibration examples (AUROC "
            f"{auc:.2f}, not above chance at the 5% level)")
    if error is None:
        return head + ": a threshold on it escalates right and wrong answers alike"
    return (head + f": the answers given alone were wrong {error:.1%} of the time against {base:.1%} for all of them — "
            "the promise holds by escalating, not by choosing what to answer")


def _group_promise(risk, delta, n_groups):
    if delta is None:
        return (f"P(answered alone and wrong) ≤ {risk:g} within each group, for inputs like the calibration examples "
                "(conformal risk control per group)")
    return (f"P(answered alone and wrong) ≤ {risk:g} within every group at once ({n_groups} groups), with probability "
            f"≥ {1 - delta:g}, for inputs like the calibration examples")


def _group_info(nodes, owner, paths, auto, wrong):
    """The per-group report of act_guard(groups=...): each node's threshold, examples, answered share, error and risk on
    them, and the groups pooled into it."""
    info = {}
    for node, v in nodes.items():
        ix = [i for i, o in enumerate(owner) if o == node]
        a, w = auto[ix], wrong[ix]
        info[node] = {"threshold": v["threshold"], "n": v["n"], "answered": float(a.mean()) if ix else 0.0,
                      "error": float(w[a].mean()) if a.any() else 0.0, "risk": float((w * a).mean()) if ix else 0.0,
                      "pooled": sorted({paths[i] for i in ix if paths[i] != node})}
    return info


def _group_guard(score, wrong, paths, risk, min_group, delta):
    """Thresholds per group for one signal → (nodes {path: {"threshold", "n"}}, report per node, answered alone [n])."""
    from ..core.calibration import _signal_losses, certify_groups
    nodes, owner = certify_groups(_signal_losses(score, wrong), paths, risk, min_group, delta)
    nodes = {k: {"threshold": v["threshold"], "n": v["n"]} for k, v in nodes.items()}
    auto = np.array([score[i] >= nodes[owner[i]]["threshold"] for i in range(len(score))], bool)
    return nodes, _group_info(nodes, owner, [tuple(p) for p in paths], auto, np.asarray(wrong, float)), auto


def _shown(v):
    """A decision value as it reads: a quote by its text, a multi-label answer as a tuple."""
    return v.value if isinstance(v, Quote) else v


def _vkey(v):
    """What must be equal for two answers to be the same (a quote by its text: offsets move when a sentence is removed;
    a multi-label answer by its set)."""
    if isinstance(v, Quote):
        return ("quote", v.value)
    if isinstance(v, tuple):
        return ("set", frozenset(v))
    return ("value", v if v is Unknown else jsonable(v))


def act_features(sp, d, act_logit):
    """The features of the act calibrator (ACT_FEATURES) for a calibrated decision: confidence (the decision's calibrated
    confidence: the top probability; multi-label: the least certain option's max(p, 1 − p)), margin (top − second
    probability; multi-label: the smallest |2p − 1|), entropy (of the probabilities, nats; multi-label: the mean binary
    entropy), act_logit (raw), n_options, kind=<kind> (1 / 0)."""
    if sp.multi:
        p = np.clip(np.array([d.probs[o] for o in sp.real], float), 1e-12, 1 - 1e-12)
        margin = float(np.min(np.abs(2 * p - 1)))
        ent = float(np.mean(-(p * np.log(p) + (1 - p) * np.log(1 - p))))
    elif d.probs:
        p = np.clip(np.sort(np.array(list(d.probs.values()), float))[::-1], 1e-12, 1)
        margin = float(p[0] - p[1]) if len(p) > 1 else 1.0
        ent = float(-(p * np.log(p)).sum())
    else:                                          # a span without "not stated": only its confidence
        margin, ent = 1.0, 0.0
    x = {"confidence": float(d.conf), "margin": margin, "entropy": ent, "act_logit": float(act_logit),
         "n_options": float(len(sp.real))}
    x.update({f"kind={k}": float(sp.kind == k) for k in KINDS})
    # the l14g features: always present (0 when the question does not allow "not stated"), so an l14g act calibrator can
    # score every kind — a plain yes / no or choice question on an l14g checkpoint as well
    x["p_unknown"] = float((d.probs or {}).get(Unknown, 0.0))
    x.update({f"kind={k}": float(sp.kind == k) for k in KINDS_V3[len(KINDS):]})
    return x


def _threshold(sig, ok, error):
    """The lowest threshold t (among the observed values) such that the cases with signal ≥ t are wrong at most `error` of
    the time — ties are kept together, so the error holds for everything the threshold lets through; inf if none."""
    sig, ok = np.asarray(sig, float), np.asarray(ok, float)
    best = math.inf
    for t in sorted(set(sig.tolist()), reverse=True):
        m = sig >= t
        if 1 - ok[m].mean() <= error + 1e-12:
            best = t
        else:
            break
    return best


def decision_of(catalog, question):
    """The DecisionPart behind a question's answer: its rule is a decision, or a rule that only passes a decided fact on
    (e.g. `def team(route): return route`). → the part or None."""
    rule = catalog.rules.get(question)
    if rule is None or rule.func is None:
        return None
    d = getattr(rule.func, "__solvi_decision__", None)
    if d is None and len(rule.inputs) == 1:
        part = catalog.parts.get(rule.inputs[0])
        if part is not None and part.alternatives is None:
            d = getattr(part.func, "__solvi_decision__", None)
    return d


__all__ = ["act_features", "decision_of", "Facts", "group_name", "group_record", "GroupBy", "guard_promise", "no_separation", "one_source"]
